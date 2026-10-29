"""Experimental EP=1, ETP>1 capture with fixed-size local-token sampling.

Capture follows AlltoAll.combine_preprocess (including ETP reduce-scatter),
before expert contributions are summed by unpermute. No extra collective,
CPU route-count copy, or runtime-sized reservation is used.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import yaml
from dmi.api.v1 import HookOutput, OutputSizingMode, RecordType, TransportType

from .source_sampling import _GraphedSegmentRanges, round_robin
from .specs import HookPhase, MegatronHookSpec, MegatronOutputSpec, ShardPolicy

NAME = "moe_tp_complete_output"
FLAG = "moe-tp-complete-output"
TOKEN_INDICES = NAME + "_token_indices"
EXPERT_COUNTS = NAME + "_expert_counts"


@dataclass(frozen=True)
class TokenSampling:
    shards: int = 8
    count: int = 1
    offset: int = 0

    def __post_init__(self):
        for name in ("shards", "count", "offset"):
            if type(getattr(self, name)) is not int:
                raise TypeError(f"token_sampling.{name} must be an integer")
        if not 1 <= self.count <= self.shards:
            raise ValueError("token_sampling requires 1 <= count <= shards")

    def select(self, batch):
        return tuple(sorted(round_robin(int(batch) - 1, self.shards, self.count, self.offset)))


def read_token_sampling(path):
    """Absent/null token_sampling means full capture; no legacy policy is reused."""
    if not path:
        return None
    with open(path, encoding="utf-8") as stream:
        document = yaml.safe_load(stream)
    if not isinstance(document, dict) or not isinstance(document.get("hooks"), dict):
        raise ValueError("Expected a hooks mapping for TP-complete capture")
    if NAME not in document["hooks"]:
        raise ValueError(f"Hook config must explicitly contain {NAME}; legacy source_sampling is not reused")
    settings = document["hooks"][NAME]
    if not isinstance(settings, dict) or set(settings) - {"token_sampling"}:
        raise ValueError(f"{NAME} only accepts token_sampling")
    policy = settings.get("token_sampling")
    if policy is None:
        return None
    if not isinstance(policy, dict):
        raise ValueError("token_sampling must be a mapping")
    return TokenSampling(**policy)


def validate_config(args, config, tp, ep, etp, cp, sampling):
    if ep != 1 or etp <= 1 or cp != 1:
        raise ValueError(f"{FLAG} requires EP=1, ETP>1, CP=1")
    if (getattr(config, "moe_token_dispatcher_type", None) != "alltoall"
            or getattr(config, "moe_permute_fusion", False)
            or getattr(config, "moe_expert_capacity_factor", None) is not None
            or getattr(config, "moe_token_dropping", False)
            or getattr(config, "moe_pad_expert_input_to_capacity", False)
            or getattr(config, "moe_router_padding_for_quantization", False)
            or getattr(config, "moe_pad_experts_for_cuda_graph_inference", False)):
        raise ValueError(f"{FLAG} requires dropless, unpadded, non-fused AlltoAll")
    if getattr(config, "moe_latent_size", None) or getattr(args, "sft", False):
        raise ValueError(f"{FLAG} requires fixed-sequence training without MoE latent projections")
    if getattr(config, "recompute_granularity", None):
        raise ValueError(f"{FLAG} experiment does not support recomputation")
    graph = getattr(config, "cuda_graph_impl", "none")
    scopes = {getattr(scope, "name", scope) for scope in (getattr(config, "cuda_graph_scope", ()) or ())}
    if graph != "none" and (graph != "transformer_engine" or not scopes
                             or scopes - {"attn", "moe_router", "moe_preprocess"}):
        raise ValueError(f"{FLAG} supports eager or TE attn/moe_router/moe_preprocess scopes only")
    if tp > 1 and not config.sequence_parallel:
        raise ValueError(f"{FLAG} requires sequence parallelism when dense TP>1")
    seq, batch = int(args.seq_length), int(args.micro_batch_size)
    divisor = tp if config.sequence_parallel else 1
    if seq <= 0 or batch <= 0 or seq % divisor:
        raise ValueError("Sequence length must be positive and divisible by dense TP with SP")
    tokens = seq // divisor * batch
    if sampling and tokens % sampling.shards:
        raise ValueError("Local source token count must be divisible by token_sampling.shards")
    return tokens


class TPCompleteCapture:
    def __init__(self, tokens, topk, experts, hidden, dtype, sampling=None):
        self.tokens, self.topk, self.experts = tokens, topk, experts
        self.hidden, self.dtype, self.sampling = hidden, dtype, sampling
        self.rows = tokens * topk
        self.selected_rows = self.rows if sampling is None else self.rows // sampling.shards * sampling.count
        self.hook = None
        self.selected = self.starts = self.ends = self.selected_counts = None
        self._batch = None
        self._shards = None
        self._segment_ranges = _GraphedSegmentRanges()
        # One native launch computes all segment ranges and selected counts from
        # the routing map (resolved on first use, after the extension loads);
        # the eager ops below remain the fallback.
        self._native_ranges = False

    def __call__(self, values, routing_map, inverse_map):
        if (values.shape != (self.rows, self.hidden) or values.dtype != self.dtype
                or not values.is_contiguous()):
            raise ValueError("TP-complete output shape/dtype changed from the fixed experiment contract")
        if (routing_map.shape != (self.tokens, self.experts) or routing_map.dtype != torch.bool
                or routing_map.device != values.device):
            raise ValueError("TP-complete routing map does not match the source token layout")
        if (inverse_map.shape != (self.rows,) or inverse_map.dtype != torch.int64
                or inverse_map.device != values.device or not inverse_map.is_contiguous()):
            raise ValueError("TP-complete inverse map does not match the fixed routed rows")
        if self.sampling is None:
            return HookOutput(values), HookOutput(inverse_map), HookOutput(routing_map.sum(0, dtype=torch.int64))

        ctx = self.hook._hook_runtime.adaptor.current_context
        if ctx is None:
            raise RuntimeError("TP-complete sampling requires a current execution context")
        policy = self.sampling
        # Each contiguous local-token shard contains exactly tokens/shards tokens,
        # each with topk routes. The distribution among experts may change, but
        # total selected rows and reservation bytes cannot change.
        if self._native_ranges is False:
            self._native_ranges = getattr(getattr(torch.ops, "ring", None), "mask_shard_segment_ranges", None)
        if self._native_ranges is not None and values.is_cuda:
            if self.starts is None:
                self.starts = torch.empty(self.experts * policy.count, dtype=torch.int64, device=values.device)
                self.ends = torch.empty_like(self.starts)
                self.selected_counts = torch.empty(self.experts, dtype=torch.int64, device=values.device)
            if values.device != self.starts.device:
                raise ValueError("TP-complete capture cannot change device")
            if self._batch != ctx.global_batch_id:
                self._shards = list(policy.select(ctx.global_batch_id))
                self._batch = ctx.global_batch_id
            self._native_ranges(routing_map if routing_map.is_contiguous() else routing_map.contiguous(),
                                self._shards, policy.shards, self.starts, self.ends, self.selected_counts)
            ranges = (self.starts, self.ends)
            return HookOutput(values, ranges), HookOutput(inverse_map, ranges), HookOutput(self.selected_counts)
        counts = routing_map.reshape(policy.shards, self.tokens // policy.shards, self.experts).sum(1, dtype=torch.int64)
        if self.selected is None:
            self.selected = torch.empty(policy.count, dtype=torch.int64, device=values.device)
            self.starts = torch.empty(self.experts * policy.count, dtype=torch.int64, device=values.device)
            self.ends = torch.empty_like(self.starts)
        if values.device != self.selected.device:
            raise ValueError("TP-complete capture cannot change device")
        if self._batch != ctx.global_batch_id:
            selected_cpu = torch.tensor(policy.select(ctx.global_batch_id), dtype=torch.int64)
            if self.selected.is_cuda:
                selected_cpu = selected_cpu.pin_memory()
            self.selected.copy_(selected_cpu, non_blocking=self.selected.is_cuda)
            self._batch = ctx.global_batch_id
        self._segment_ranges(counts.T, self.selected, self.starts, self.ends)
        selected_counts = counts.index_select(0, self.selected).sum(0)
        ranges = (self.starts, self.ends)
        return HookOutput(values, ranges), HookOutput(inverse_map, ranges), HookOutput(selected_counts)


def install_hooks(model, *, tokens, sampling, capture_layers=None):
    from ..startup import _make_hook, _module_layer_selected
    roots = model if isinstance(model, list) else [model]
    for root in roots:
        for layer in root.modules():
            if type(layer).__name__ != "MoELayer" or not _module_layer_selected(layer, capture_layers):
                continue
            dispatcher = layer.token_dispatcher
            if (type(dispatcher).__name__ != "MoEAlltoAllTokenDispatcher"
                    or dispatcher.ep_size != 1 or dispatcher.tp_size <= 1):
                raise ValueError(f"{FLAG} requires an EP=1, ETP>1 AlltoAll dispatcher")
            if getattr(layer, "dmi_moe_tp_complete_output", None) is not None:
                raise ValueError("TP-complete hook is already installed")
            config = layer.config
            capture = TPCompleteCapture(tokens, config.moe_router_topk, config.num_moe_experts,
                                        config.hidden_size, config.params_dtype, sampling)
            transport = TransportType.SEGMENTED_PACK if sampling else TransportType.IDENTITY
            outputs = [
                MegatronOutputSpec(name, shape, dtype, output_shape=out_shape,
                    transport_type=transport, segment_ranges_from_preprocess=sampling is not None,
                    sizing_mode=OutputSizingMode.KNOWN_BEFORE_EXECUTION)
                for name, shape, out_shape, dtype in (
                    (NAME, (capture.rows, capture.hidden), (capture.selected_rows, capture.hidden), capture.dtype),
                    (TOKEN_INDICES, (capture.rows,), (capture.selected_rows,), torch.int64),
                )
            ]
            outputs.append(MegatronOutputSpec(EXPERT_COUNTS, (capture.experts,), torch.int64))
            hook = _make_hook(MegatronHookSpec(
                NAME, int(layer.layer_number) - 1, outputs, preprocess=capture,
                shard_policy=ShardPolicy.GLOBAL_RANK_SHARDED, enabled_by=frozenset({FLAG}),
                need_token_range=False, record_type=RecordType.PER_EXECUTION), hook_phase=HookPhase.FWD)
            layer.dmi_moe_tp_complete_output = hook
            capture.hook = hook


def capture_metadata(tokens, sampling):
    return {
        "format_version": 1,
        "capture_point": "alltoall_after_etp_reduce_scatter_before_unpermute",
        "etp_composition": "already_reduced",
        "layout": "source_local_expert_major_stable_token_order",
        "source_local_tokens": tokens,
        "token_index_layout": "flattened_sequence_batch_on_source_rank",
        "token_indices_output": TOKEN_INDICES,
        "expert_counts_output": EXPERT_COUNTS,
        "token_sampling": asdict(sampling) if sampling else None,
        "sampling_domain": "each_source_rank_independently",
        "iteration_field": "global_batch_id",
        "iteration_origin": 1,
    }
