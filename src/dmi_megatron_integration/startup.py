"""Startup wiring for enabling DMI inside Megatron-LM runs."""

from __future__ import annotations

import atexit
import os
import sys
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from functools import partial
from typing import Any, Mapping

import torch

from dmi.api.v1 import (
    HookPointV1,
    MonitoringEngine,
    OutputStorage,
    RecordType,
    TransportType,
)

from .topology.ep_topology_manifest import (
    MegatronEPTopologyFragment,
    MoELayerFragment,
    assemble_ep_topology_manifest,
    write_ep_topology_manifest,
)
from .adapter import (
    MegatronAdaptor,
    MegatronHookBinding,
    MegatronRouterWeightBinding,
    MegatronTrainingContext,
)
from .hooks.selection import parse_hook_selection
from .hooks.megatron_loss_summary import (
    per_sample_loss_from_token_loss,
    per_segment_loss_from_token_loss,
)
from .hooks.megatron_qk_weights import qk_weight_from_fused_qkv
from .hooks.megatron_vocab_logits import (
    vocab_logits_by_sample,
    vocab_logits_topk_by_sample,
)
from .metadata_context import (
    DMIMetadataFieldSpec,
    dataset_id_field_spec,
    segment_metadata_field_spec,
    valid_count_field_spec,
)
from .schedule_runtime import (
    build_megatron_schedule_runtime,
    set_active_megatron_schedule_runtime,
)
from .hooks.specs import (
    DPEmissionPolicy,
    DimSpec,
    HookLayerPlacement,
    HookInputLayout,
    HookPhase,
    MegatronDistributedInfo,
    MegatronMetadataField,
    MegatronHookSpec,
    MegatronOutputSpec,
    ShardPolicy,
)
from .records.format import MegatronRecordFormat, required_record_metadata_fields


def _print_warning(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


_TRUE_STRINGS = {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class MegatronDMIConfig:
    enabled: bool = False
    exact_resume: bool = False
    hook_selection: str = "router-summary"
    recompute_hook: str | None = None
    no_recompute_hook: str | None = None
    dataset_provenance_mode: str = "auto"
    model_id: str | None = None
    db_host: str = ""
    db_port: int = 9000
    db_database: str = "default"
    clickhouse_table: str = "dmi_training_tensors"
    ch_parallelism: int = 10
    ring_payload_mb: int = 4096
    ring_pinned_mb: int = 4096
    ring_task_entries: int = 65536
    recurring_d2h_windows_enabled: bool = False
    d2h_window_minimum_record_probe_retry_interval_occurrences: int = 4
    d2h_window_timing_revalidation_retry_interval_occurrences: int = 4
    d2h_window_capacity_flush_fallback_threshold: int = 3
    d2h_window_capacity_flush_count_reset_interval_periods: int = 32
    d2h_window_debug: bool = False
    drain_flush_payload_ratio: float = 0.0
    drain_flush_task_ratio: float = 0.0
    drain_flush_byte_threshold: int = 0
    drain_flush_entry_threshold: int = 0
    drain_flush_timeout_us: int = 0
    flush_every_n_train_iters: int = 0
    vocab_logits_top_k: int | None = None
    topology_manifest_path: str | None = None


class MegatronDMIHandle:
    """Owns DMI objects installed for one Megatron process."""

    def __init__(
        self,
        *,
        config: MegatronDMIConfig,
        model_id: str,
        engine: MonitoringEngine,
        schedule_runtime: Any,
        adaptor: MegatronAdaptor,
        current_phase_tensor: torch.Tensor,
        grad_norm_hook: HookPointV1 | None = None,
        router_weight_bindings: tuple[MegatronRouterWeightBinding, ...] = (),
        qk_weight_bindings: tuple[tuple[HookPointV1, torch.nn.Parameter], ...] = (),
    ) -> None:
        self.config = config
        self.model_id = model_id
        self.engine = engine
        self.schedule_runtime = schedule_runtime
        self.adaptor = adaptor
        self.current_phase_tensor = current_phase_tensor
        self.grad_norm_hook = grad_norm_hook
        self.router_weight_bindings = tuple(router_weight_bindings)
        self.qk_weight_bindings = tuple(qk_weight_bindings)
        self.closed = False

    def _emit_iteration_values(
        self,
        *,
        global_batch_id: int,
        values: tuple[tuple[HookPointV1, torch.Tensor], ...],
        allow_zero: bool = False,
    ) -> None:
        global_batch_id = int(global_batch_id)
        minimum = 0 if allow_zero else 1
        if global_batch_id < minimum:
            raise ValueError(
                f"DMI iteration ID must be >= {minimum}, got {global_batch_id}"
            )
        if not values:
            return
        self.adaptor.set_current_iteration(
            MegatronTrainingContext(
                global_batch_id=global_batch_id,
                microbatch_id=-1,
                valid_counts=(),
                dataset_ids=(),
                attempt_id=int(self.schedule_runtime.current_attempt_id),
                direction="iter",
                phase="train",
                dp_rank=-1,
                shard_rank=-1,
                token_start=0,
            )
        )
        try:
            with torch.no_grad():
                for hook, tensor in values:
                    hook(tensor)
        finally:
            self.adaptor.clear_current_iteration()

    def emit_grad_norm(self, tensor: torch.Tensor, *, training_iteration_id: int) -> None:
        hook = self.grad_norm_hook
        if hook is None:
            return
        self._emit_iteration_values(
            global_batch_id=training_iteration_id,
            values=((hook, tensor),),
        )

    def emit_router_weights(self, *, model_state_iteration_id: int, allow_zero: bool = False) -> None:
        self._emit_iteration_values(
            global_batch_id=model_state_iteration_id,
            values=tuple((binding.hook, binding.parameter) for binding in self.router_weight_bindings),
            allow_zero=allow_zero,
        )

    def emit_initial_router_weights(self, *, model_state_iteration_id: int) -> None:
        self.emit_router_weights(
            model_state_iteration_id=model_state_iteration_id,
            allow_zero=True,
        )

    def emit_qk_weights(self, *, model_state_iteration_id: int, allow_zero: bool = False) -> None:
        # Pass live QKV parameters; each hook preprocesses its own Q/K output.
        self._emit_iteration_values(
            global_batch_id=model_state_iteration_id,
            values=self.qk_weight_bindings,
            allow_zero=allow_zero,
        )

    def emit_initial_qk_weights(self, *, model_state_iteration_id: int) -> None:
        self.emit_qk_weights(
            model_state_iteration_id=model_state_iteration_id,
            allow_zero=True,
        )

    def flush_and_wait(self, timeout_s: float = 600.0) -> None:
        if self.closed:
            raise RuntimeError("cannot flush a closed Megatron DMI handle")
        self.engine.flush_and_wait(timeout_s)

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        failures: list[tuple[str, BaseException]] = []
        try:
            self.schedule_runtime.seal_current_phase()
        except Exception as exc:
            failures.append(("phase boundary", exc))
        finally:
            set_active_megatron_schedule_runtime(None)

        try:
            self.engine.close()
        except Exception as exc:
            failures.append(("monitoring engine", exc))
        if failures:
            detail = "; ".join(f"{stage}: {exc}" for stage, exc in failures)
            raise RuntimeError(f"Megatron DMI close failed: {detail}") from failures[0][1]


def _env_bool(environ: Mapping[str, str], name: str) -> bool | None:
    value = environ.get(name)
    if value is None:
        return None
    return value.strip().lower() in _TRUE_STRINGS


def _env_value(
    args: Any,
    attr: str,
    environ: Mapping[str, str],
    env_name: str,
    default: Any,
    cast: Any = None,
) -> Any:
    value = getattr(args, attr, None) if args is not None else None
    if value is None:
        value = environ.get(env_name)
    if value is None:
        return default
    return cast(value) if cast is not None else value


def resolve_megatron_dmi_config(
    args: Any | None = None,
    *,
    explicit: MegatronDMIConfig | None = None,
    environ: Mapping[str, str] | None = None,
) -> MegatronDMIConfig:
    """Resolve DMI config with explicit > CLI > environment > default precedence."""

    environ = os.environ if environ is None else environ
    cli_enabled = getattr(args, "dmi_enable", None) if args is not None else None
    env_enabled = _env_bool(environ, "DMI_ENABLE")
    cli_windows = getattr(args, "dmi_recurring_d2h_windows", None)
    cli_window_debug = getattr(args, "dmi_d2h_window_debug", None)
    cfg = MegatronDMIConfig(
        enabled=bool(cli_enabled if cli_enabled is not None else (env_enabled or False)),
        exact_resume=bool(getattr(args, "dmi_exact_resume", False)),
        hook_selection=str(
            _env_value(args, "dmi_hook_selection", environ, "DMI_HOOK_SELECTION", "router-summary")
        ),
        recompute_hook=_env_value(
            args,
            "dmi_recompute_hook",
            environ,
            "DMI_RECOMPUTE_HOOK",
            None,
        ),
        no_recompute_hook=_env_value(
            args,
            "dmi_no_recompute_hook",
            environ,
            "DMI_NO_RECOMPUTE_HOOK",
            None,
        ),
        dataset_provenance_mode=str(
            _env_value(
                args,
                "dmi_dataset_provenance_mode",
                environ,
                "DMI_DATASET_PROVENANCE_MODE",
                "auto",
            )
        ),
        model_id=_env_value(args, "dmi_model_id", environ, "DMI_MODEL_ID", None),
        db_host=str(_env_value(args, "dmi_db_host", environ, "DMI_DB_HOST", "")),
        db_port=int(_env_value(args, "dmi_db_port", environ, "DMI_DB_PORT", 9000, int)),
        db_database=str(_env_value(args, "dmi_db_database", environ, "DMI_DB_DATABASE", "default")),
        clickhouse_table=str(
            _env_value(
                args,
                "dmi_clickhouse_table",
                environ,
                "DMI_CLICKHOUSE_TABLE",
                "dmi_training_tensors",
            )
        ),
        ch_parallelism=int(
            _env_value(args, "dmi_ch_parallelism", environ, "DMI_CH_PARALLELISM", 10, int)
        ),
        ring_payload_mb=int(
            _env_value(args, "dmi_ring_payload_mb", environ, "DMI_RING_PAYLOAD_MB", 4096, int)
        ),
        ring_pinned_mb=int(
            _env_value(args, "dmi_ring_pinned_mb", environ, "DMI_RING_PINNED_MB", 4096, int)
        ),
        ring_task_entries=int(
            _env_value(args, "dmi_ring_task_entries", environ, "DMI_RING_TASK_ENTRIES", 65536, int)
        ),
        recurring_d2h_windows_enabled=bool(
            cli_windows if cli_windows is not None
            else (_env_bool(environ, "DMI_RECURRING_D2H_WINDOWS") or False)
        ),
        d2h_window_minimum_record_probe_retry_interval_occurrences=_env_value(
            args, "dmi_d2h_window_minimum_record_probe_retry_interval_occurrences", environ,
            "DMI_D2H_WINDOW_MINIMUM_RECORD_PROBE_RETRY_INTERVAL_OCCURRENCES", 4, int,
        ),
        d2h_window_timing_revalidation_retry_interval_occurrences=_env_value(
            args, "dmi_d2h_window_timing_revalidation_retry_interval_occurrences", environ,
            "DMI_D2H_WINDOW_TIMING_REVALIDATION_RETRY_INTERVAL_OCCURRENCES", 4, int,
        ),
        d2h_window_capacity_flush_fallback_threshold=_env_value(
            args, "dmi_d2h_window_capacity_flush_fallback_threshold", environ,
            "DMI_D2H_WINDOW_CAPACITY_FLUSH_FALLBACK_THRESHOLD", 3, int,
        ),
        d2h_window_capacity_flush_count_reset_interval_periods=_env_value(
            args, "dmi_d2h_window_capacity_flush_count_reset_interval_periods", environ,
            "DMI_D2H_WINDOW_CAPACITY_FLUSH_COUNT_RESET_INTERVAL_PERIODS", 32, int,
        ),
        d2h_window_debug=bool(
            cli_window_debug if cli_window_debug is not None
            else (_env_bool(environ, "DMI_D2H_WINDOW_DEBUG") or False)
        ),
        drain_flush_payload_ratio=float(
            _env_value(
                args,
                "dmi_drain_flush_payload_ratio",
                environ,
                "DMI_DRAIN_FLUSH_PAYLOAD_RATIO",
                0.0,
                float,
            )
        ),
        drain_flush_task_ratio=float(
            _env_value(
                args,
                "dmi_drain_flush_task_ratio",
                environ,
                "DMI_DRAIN_FLUSH_TASK_RATIO",
                0.0,
                float,
            )
        ),
        drain_flush_byte_threshold=int(
            _env_value(
                args,
                "dmi_drain_flush_byte_threshold",
                environ,
                "DMI_DRAIN_FLUSH_BYTE_THRESHOLD",
                0,
                int,
            )
        ),
        drain_flush_entry_threshold=int(
            _env_value(
                args,
                "dmi_drain_flush_entry_threshold",
                environ,
                "DMI_DRAIN_FLUSH_ENTRY_THRESHOLD",
                0,
                int,
            )
        ),
        drain_flush_timeout_us=int(
            _env_value(
                args,
                "dmi_drain_flush_timeout_us",
                environ,
                "DMI_DRAIN_FLUSH_TIMEOUT_US",
                0,
                int,
            )
        ),
        flush_every_n_train_iters=int(
            _env_value(
                args,
                "dmi_flush_every_n_train_iters",
                environ,
                "DMI_FLUSH_EVERY_N_TRAIN_ITERS",
                0,
                int,
            )
        ),
        vocab_logits_top_k=_env_value(
            args,
            "dmi_vocab_logits_top_k",
            environ,
            "DMI_VOCAB_LOGITS_TOP_K",
            None,
            int,
        ),
        topology_manifest_path=_env_value(
            args,
            "dmi_topology_manifest_path",
            environ,
            "DMI_TOPOLOGY_MANIFEST_PATH",
            None,
        ),
    )
    return explicit if explicit is not None else cfg


def _dist_ready(dist_module: Any) -> bool:
    return bool(hasattr(dist_module, "is_available") and dist_module.is_available()) and bool(
        hasattr(dist_module, "is_initialized") and dist_module.is_initialized()
    )


def resolve_model_id(
    cfg: MegatronDMIConfig,
    *,
    dist_module: Any | None = None,
    environ: Mapping[str, str] | None = None,
    printer: Any | None = None,
) -> str:
    """Resolve or generate a process-consistent model_id."""

    if cfg.model_id:
        return str(cfg.model_id)

    environ = os.environ if environ is None else environ
    dist = torch.distributed if dist_module is None else dist_module
    rank = int(dist.get_rank()) if _dist_ready(dist) else 0
    value = None
    if rank == 0:
        slurm_id = environ.get("SLURM_JOB_ID")
        if slurm_id:
            value = f"dmi-megatron-{slurm_id}"
        else:
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            value = f"dmi-megatron-{stamp}"
        message = f"[DMI] --dmi-model-id was not provided; generated model_id={value}"
        if printer is not None:
            printer(message)
        else:
            print(message, flush=True)

    if _dist_ready(dist):
        obj = [value]
        dist.broadcast_object_list(obj, src=0)
        value = obj[0]
    if value is None:
        raise RuntimeError("Failed to resolve DMI model_id")
    return str(value)


def _parallel_rank(parallel_state: Any, getter_name: str) -> int:
    getter = getattr(parallel_state, getter_name, None)
    if getter is None:
        return 0
    value = getter()
    return 0 if value is None else int(value)


def _parallel_world(parallel_state: Any, getter_name: str) -> int:
    getter = getattr(parallel_state, getter_name, None)
    if getter is None:
        return 1
    value = getter()
    return 1 if value is None else int(value)


@dataclass(frozen=True)
class MegatronRankContext:
    global_rank: int
    tp_rank: int
    tp_world_size: int
    pp_rank: int
    pp_world_size: int
    dp_rank: int
    dp_world_size: int
    cp_rank: int
    cp_world_size: int
    ep_rank: int
    ep_world_size: int
    vp_rank: int | None
    num_layers: int


@dataclass(frozen=True)
class _MetadataRequirementReport:
    global_rank: int
    dense_dp_rank: int
    tp_rank: int
    tp_world_size: int
    pp_rank: int
    pp_world_size: int
    input_layout: HookInputLayout
    max_batch_size: int
    segment_capacity: int | None
    gpu_fields: tuple[MegatronMetadataField, ...]
    cpu_record_fields: tuple[MegatronMetadataField, ...]
    has_per_sample_records: bool


@dataclass(frozen=True)
class _ResolvedMetadataRequirements:
    wire_fields: tuple[MegatronMetadataField, ...]
    local_gpu_fields: frozenset[MegatronMetadataField]
    has_per_sample_records: bool


def _build_rank_context(
    args: Any,
    model_config: Any,
    parallel_state: Any,
    *,
    global_rank: int,
) -> MegatronRankContext:
    vp_rank_getter = getattr(parallel_state, "get_virtual_pipeline_model_parallel_rank", None)
    vp_rank = vp_rank_getter() if vp_rank_getter is not None else None
    num_layers = getattr(model_config, "num_layers", None)
    if num_layers is None:
        num_layers = getattr(args, "num_layers", None)
    if num_layers is None:
        num_layers = 0
    return MegatronRankContext(
        global_rank=int(global_rank),
        tp_rank=_parallel_rank(parallel_state, "get_tensor_model_parallel_rank"),
        tp_world_size=_parallel_world(parallel_state, "get_tensor_model_parallel_world_size"),
        pp_rank=_parallel_rank(parallel_state, "get_pipeline_model_parallel_rank"),
        pp_world_size=_parallel_world(parallel_state, "get_pipeline_model_parallel_world_size"),
        dp_rank=_parallel_rank(parallel_state, "get_data_parallel_rank"),
        dp_world_size=_parallel_world(parallel_state, "get_data_parallel_world_size"),
        cp_rank=_parallel_rank(parallel_state, "get_context_parallel_rank"),
        cp_world_size=_parallel_world(parallel_state, "get_context_parallel_world_size"),
        ep_rank=_parallel_rank(parallel_state, "get_expert_model_parallel_rank"),
        ep_world_size=_parallel_world(parallel_state, "get_expert_model_parallel_world_size"),
        vp_rank=None if vp_rank is None else int(vp_rank),
        num_layers=int(num_layers),
    )


def _resolve_layer_selector(selector: int, num_layers: int) -> int:
    resolved = int(selector) if int(selector) >= 0 else int(num_layers) + int(selector)
    if resolved < 0 or resolved >= int(num_layers):
        raise ValueError(
            f"DMI layer selector {selector} resolves to {resolved}, outside [0, {num_layers})"
        )
    return resolved


def _layer_placement_allows(spec: MegatronHookSpec, rank_ctx: MegatronRankContext) -> bool:
    placement = spec.layer_placement
    selector = spec.layer_selector
    if placement == HookLayerPlacement.EVERY_LAYER:
        if selector is not None:
            raise ValueError("EVERY_LAYER hooks must not set layer_selector")
        return True
    if placement == HookLayerPlacement.LAYER_SET:
        if not selector:
            raise ValueError("LAYER_SET hooks require a non-empty layer_selector")
        selected = {
            _resolve_layer_selector(layer_no, rank_ctx.num_layers)
            for layer_no in selector
        }
        return int(spec.layer_no) in selected
    if placement == HookLayerPlacement.NO_LAYER_FIRST_PP:
        if selector is not None:
            raise ValueError("NO_LAYER_FIRST_PP hooks must not set layer_selector")
        return int(rank_ctx.pp_rank) == 0
    if placement == HookLayerPlacement.NO_LAYER_LAST_PP:
        if selector is not None:
            raise ValueError("NO_LAYER_LAST_PP hooks must not set layer_selector")
        return int(rank_ctx.pp_rank) == int(rank_ctx.pp_world_size) - 1
    raise ValueError(f"Unsupported DMI hook layer placement: {placement!r}")


def _shard_policy_allows(spec: MegatronHookSpec, rank_ctx: MegatronRankContext) -> bool:
    policy = spec.shard_policy
    if policy == ShardPolicy.REPLICATED:
        return rank_ctx.tp_rank == 0 and rank_ctx.ep_rank == 0 and rank_ctx.cp_rank == 0
    if policy == ShardPolicy.GLOBAL_RANK_SHARDED:
        return True
    if policy in (ShardPolicy.TP_SHARDED, ShardPolicy.TP_SEQUENCE_SHARDED):
        return rank_ctx.ep_rank == 0 and rank_ctx.cp_rank == 0
    if policy == ShardPolicy.EP_SHARDED:
        return rank_ctx.tp_rank == 0 and rank_ctx.cp_rank == 0
    if policy == ShardPolicy.CP_SHARDED:
        return rank_ctx.tp_rank == 0 and rank_ctx.ep_rank == 0
    if policy == ShardPolicy.DP_SHARDED:
        return rank_ctx.tp_rank == 0 and rank_ctx.ep_rank == 0 and rank_ctx.cp_rank == 0
    raise ValueError(f"Unsupported DMI shard policy: {policy!r}")


def _spec_active_on_rank(spec: MegatronHookSpec, rank_ctx: MegatronRankContext) -> bool:
    if spec.dp_emission == DPEmissionPolicy.DP_RANK_0 and rank_ctx.dp_rank != 0:
        return False
    return _layer_placement_allows(spec, rank_ctx) and _shard_policy_allows(spec, rank_ctx)


def _validate_hook_contract(hook: HookPointV1) -> None:
    spec = _megatron_hook_spec(hook)
    if spec.shard_policy is ShardPolicy.TP_SEQUENCE_SHARDED:
        if spec.record_type is not RecordType.PER_SAMPLE or len(spec.outputs) != 1:
            raise ValueError("TP_SEQUENCE_SHARDED requires one PER_SAMPLE output")
        output = spec.outputs[0]
        if (
            tuple(output.input_shape[:2]) != (DimSpec.SEQ, DimSpec.BATCH)
            or output.transport_type is not TransportType.SEQ_PREFIX_PACK
        ):
            raise ValueError(
                "TP_SEQUENCE_SHARDED requires [SEQ, BATCH, ...] "
                "SEQ_PREFIX_PACK input"
            )
    if spec.record_type == RecordType.PER_SAMPLE:
        if hook.hook_phase not in (HookPhase.FWD, HookPhase.BWD):
            raise ValueError("PER_SAMPLE hooks must use FWD or BWD phase")
        if spec.dp_emission != DPEmissionPolicy.ALL_DP_RANKS:
            raise ValueError("PER_SAMPLE hooks must emit on all data-parallel ranks")
        return
    if spec.record_type == RecordType.PER_EXECUTION:
        if hook.hook_phase not in (HookPhase.FWD, HookPhase.BWD):
            raise ValueError("PER_EXECUTION hooks must use FWD or BWD phase")
        if spec.dp_emission != DPEmissionPolicy.ALL_DP_RANKS:
            raise ValueError("PER_EXECUTION hooks must emit on all data-parallel ranks")
        if any(output.transport_type != TransportType.IDENTITY for output in spec.outputs):
            raise ValueError("PER_EXECUTION hooks initially require IDENTITY transport")
        if spec.binding_metadata_fields:
            raise ValueError("PER_EXECUTION hooks must not require bound metadata")
        return
    if spec.record_type != RecordType.PER_ITERATION:
        raise ValueError(f"Unsupported DMI record type: {spec.record_type!r}")
    if hook.hook_phase is not HookPhase.ITERATION:
        raise ValueError("PER_ITERATION hooks must use ITERATION phase")
    if spec.dp_emission != DPEmissionPolicy.DP_RANK_0:
        raise ValueError("PER_ITERATION hooks must use DP_RANK_0 emission")
    if spec.shard_policy == ShardPolicy.DP_SHARDED:
        raise NotImplementedError("DP-sharded PER_ITERATION records are not supported")
    if any(output.transport_type != TransportType.IDENTITY for output in spec.outputs):
        raise ValueError("PER_ITERATION hooks initially require IDENTITY transport")
    if spec.binding_metadata_fields:
        raise ValueError("PER_ITERATION hooks must not require bound metadata")


def _make_hook(
    spec: MegatronHookSpec,
    *,
    hook_phase: HookPhase,
    suppress_recompute: bool = True,
) -> HookPointV1:
    """Attach Megatron policy to a public hook until adapter binding resolves it."""

    hook = HookPointV1()
    hook._dmi_megatron_spec = spec
    hook.hook_phase = hook_phase
    hook.suppress_recompute = bool(suppress_recompute)
    return hook


def _megatron_hook_spec(hook: HookPointV1) -> MegatronHookSpec:
    spec = getattr(hook, "_dmi_megatron_spec", None)
    if not isinstance(spec, MegatronHookSpec):
        raise TypeError("DMI HookPointV1 is missing MegatronHookSpec policy")
    return spec


def _data_parallel_world_size(args: Any, parallel_state: Any) -> int:
    getter = getattr(parallel_state, "get_data_parallel_world_size", None)
    if getter is not None:
        try:
            return max(1, int(getter()))
        except Exception:
            pass
    return max(1, int(getattr(args, "data_parallel_size", 1) or 1))


def _num_scopes(parallel_state: Any) -> int:
    getter = getattr(parallel_state, "get_virtual_pipeline_model_parallel_world_size", None)
    vp_world = getter() if getter is not None else None
    return 1 if vp_world is None else int(vp_world)


def _max_num_microbatches(args: Any, dp_world_size: int) -> int:
    global_batch = int(getattr(args, "global_batch_size"))
    micro_batch = int(getattr(args, "micro_batch_size"))
    denom = max(1, micro_batch * int(dp_world_size))
    return max(1, global_batch // denom)


def _num_experts(model_config: Any) -> int:
    value = getattr(model_config, "num_moe_experts", None)
    if value is None:
        raise ValueError("DMI router-summary requires model_config.num_moe_experts")
    return int(value)


def _hidden_size(model_config: Any) -> int:
    value = getattr(model_config, "hidden_size", None)
    if value is None:
        raise ValueError("DMI hidden-states hook requires model_config.hidden_size")
    return int(value)


def _seq_length(args: Any) -> int:
    value = getattr(args, "seq_length", None)
    if value is None:
        raise ValueError("DMI sequence-shaped hooks require args.seq_length")
    return int(value)


def _router_logits_dtype(model_config: Any) -> torch.dtype:
    configured = getattr(model_config, "moe_router_dtype", None)
    if configured == "fp32":
        return torch.float32
    if configured == "fp64":
        return torch.float64
    if configured is not None:
        raise ValueError(f"Unsupported Megatron moe_router_dtype: {configured!r}")
    params_dtype = getattr(model_config, "params_dtype", None)
    if not isinstance(params_dtype, torch.dtype):
        raise ValueError(
            "DMI router-logits requires model_config.params_dtype when "
            "moe_router_dtype is unset"
        )
    return params_dtype


def _vocab_logits_dtype(model_config: Any) -> torch.dtype:
    dtype = getattr(model_config, "params_dtype", None)
    if not isinstance(dtype, torch.dtype):
        raise ValueError("DMI vocab-logits requires model_config.params_dtype")
    return dtype


def _padded_vocab_size(args: Any) -> int:
    value = getattr(args, "padded_vocab_size", None)
    if value is None:
        value = getattr(args, "vocab_size", None)
    if value is None or int(value) <= 0:
        raise ValueError("DMI vocab-logits requires a positive padded vocabulary size")
    return int(value)


def _selected_hooks(selection: str) -> set[str]:
    return parse_hook_selection(selection)


def _parse_hook_policy_list(value: str | None, *, option: str) -> tuple[str, ...]:
    if value is None:
        return ()
    parts = tuple(part.strip() for part in str(value).split(","))
    if not parts or any(not part for part in parts):
        raise ValueError(f"{option} contains an empty hook selection name")
    if len(parts) != len(set(parts)):
        raise ValueError(f"{option} contains a duplicate hook selection name")
    return parts


def _apply_recompute_hook_policy(
    hooks: list[MegatronHookBinding],
    *,
    selected_names: set[str],
    recompute_names_raw: str | None,
    no_recompute_names_raw: str | None,
) -> None:
    recompute_names = _parse_hook_policy_list(
        recompute_names_raw,
        option="--dmi-recompute-hook",
    )
    no_recompute_names = _parse_hook_policy_list(
        no_recompute_names_raw,
        option="--dmi-no-recompute-hook",
    )
    overlap = set(recompute_names) & set(no_recompute_names)
    if overlap:
        raise ValueError(
            "DMI recompute policy names occur in both lists: "
            f"{sorted(overlap)}"
        )
    requested = set(recompute_names) | set(no_recompute_names)
    missing_selection = requested - selected_names
    if missing_selection:
        raise ValueError(
            "DMI recompute policy cannot enable unselected hooks: "
            f"{sorted(missing_selection)}"
        )

    resolved: dict[int, bool] = {}
    resolved_names: set[str] = set()
    for binding in hooks:
        hook = binding.hook
        spec = _megatron_hook_spec(hook)
        matched_recompute = set(spec.enabled_by) & set(recompute_names)
        matched_no_recompute = set(spec.enabled_by) & set(no_recompute_names)
        if matched_recompute and matched_no_recompute:
            raise ValueError(
                f"DMI hook {spec.name!r} receives conflicting recompute policies"
            )
        if not matched_recompute and not matched_no_recompute:
            continue
        if spec.record_type is RecordType.PER_ITERATION:
            raise ValueError(
                f"DMI recompute policy does not apply to PER_ITERATION hook {spec.name!r}"
            )
        suppress = bool(matched_no_recompute)
        previous = resolved.get(id(hook))
        if previous is not None and previous != suppress:
            raise ValueError(f"Conflicting DMI recompute policy for hook {spec.name!r}")
        resolved[id(hook)] = suppress
        resolved_names.update(matched_recompute)
        resolved_names.update(matched_no_recompute)

    unresolved = requested - resolved_names
    if unresolved:
        raise ValueError(
            "DMI recompute policy names resolve to no selected HookPointV1: "
            f"{sorted(unresolved)}"
        )
    for binding in hooks:
        if id(binding.hook) in resolved:
            binding.hook.suppress_recompute = resolved[id(binding.hook)]


def _parse_explicit_dataset_provenance_modes(value: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for raw_entry in value.split(","):
        entry = raw_entry.strip()
        if not entry or entry.count("=") != 1:
            raise ValueError(
                "--dmi-dataset-provenance-mode requires phase=mode entries"
            )
        phase, mode = (part.strip().lower() for part in entry.split("=", 1))
        if phase not in {"train", "valid", "test"}:
            raise ValueError(f"Unknown DMI dataset provenance phase: {phase!r}")
        if phase in result:
            raise ValueError(f"Duplicate DMI dataset provenance phase: {phase!r}")
        if mode not in {"constant-zero", "dynamic"}:
            raise ValueError(f"Unknown DMI dataset provenance mode: {mode!r}")
        result[phase] = mode
    missing = {"train", "valid", "test"} - set(result)
    if missing:
        raise ValueError(
            "Explicit DMI dataset provenance mode is missing phases: "
            f"{sorted(missing)}"
        )
    return result


def _resolve_dataset_provenance_modes(
    args: Any,
    dataset_provider: Any | None,
    *,
    configured_mode: str,
    has_per_sample_hooks: bool,
) -> dict[str, str]:
    if not has_per_sample_hooks:
        return {phase: "constant-zero" for phase in ("train", "valid", "test")}
    normalized = str(configured_mode).strip().lower()
    if normalized != "auto":
        return _parse_explicit_dataset_provenance_modes(normalized)
    if not bool(getattr(dataset_provider, "dmi_standard_dataset_provider", False)):
        raise ValueError(
            "DMI dataset provenance auto mode requires Megatron's standard dataset "
            "provider; custom providers must configure every phase explicitly"
        )

    from megatron.training.utils import get_blend_and_blend_per_split

    if bool(getattr(args, "mock_data", False)):
        counts = {phase: 1 for phase in ("train", "valid", "test")}
    else:
        blend, blend_per_split = get_blend_and_blend_per_split(args)
        if blend is not None:
            source_count = len(blend[0])
            counts = {phase: source_count for phase in ("train", "valid", "test")}
        elif blend_per_split is not None:
            counts = {
                phase: 0 if item is None else len(item[0])
                for phase, item in zip(("train", "valid", "test"), blend_per_split)
            }
        else:
            counts = {phase: 1 for phase in ("train", "valid", "test")}

    modes = {
        phase: "dynamic" if source_count > 1 else "constant-zero"
        for phase, source_count in counts.items()
    }
    if bool(getattr(args, "multiple_validation_sets", False)):
        # Each validation invocation reads one direct source dataset.  The
        # invocation index is supplied by the validation loop without a wire field.
        modes["valid"] = "constant-zero"
    return modes


def _install_loss_summary_hook(
    model: Any,
    *,
    input_layout: HookInputLayout,
    segment_capacity: int | None,
) -> None:
    if input_layout is HookInputLayout.PACKED_SEGMENTED:
        if segment_capacity is None or int(segment_capacity) <= 0:
            raise ValueError("Packed loss-summary requires a positive segment capacity")
        outputs = [
            MegatronOutputSpec(
                name="lm_per_sample_loss",
                input_shape=[int(segment_capacity), 1],
                output_shape=[int(segment_capacity), 1],
                dtype=torch.float32,
                transport_type=TransportType.PREFIX_STRIP,
                storage=OutputStorage.SCALAR_FLOAT,
            ),
            MegatronOutputSpec(
                name="lm_per_sample_loss_token_count",
                input_shape=[int(segment_capacity), 1],
                output_shape=[int(segment_capacity), 1],
                dtype=torch.int64,
                transport_type=TransportType.PREFIX_STRIP,
                storage=OutputStorage.SCALAR_INT,
            ),
        ]
        preprocess = per_segment_loss_from_token_loss
        preprocess_metadata_fields = frozenset(
            {MegatronMetadataField.SEGMENT_METADATA}
        )
        supported_layouts = frozenset({HookInputLayout.PACKED_SEGMENTED})
    else:
        outputs = [
            MegatronOutputSpec(
                name="lm_per_sample_loss",
                input_shape=[DimSpec.BATCH, 1],
                output_shape=[DimSpec.BATCH, 1],
                dtype=torch.float32,
                transport_type=TransportType.IDENTITY,
                storage=OutputStorage.SCALAR_FLOAT,
            ),
            MegatronOutputSpec(
                name="lm_per_sample_loss_token_count",
                input_shape=[DimSpec.BATCH, 1],
                output_shape=[DimSpec.BATCH, 1],
                dtype=torch.int64,
                transport_type=TransportType.IDENTITY,
                storage=OutputStorage.SCALAR_INT,
            ),
        ]
        preprocess = per_sample_loss_from_token_loss
        preprocess_metadata_fields = frozenset()
        supported_layouts = frozenset({HookInputLayout.SEQ_BATCH})

    roots = model if isinstance(model, list) else [model]
    for root in roots:
        existing = getattr(root, "dmi_lm_per_sample_loss", None)
        if existing is not None:
            if not isinstance(existing, HookPointV1):
                raise TypeError("model.dmi_lm_per_sample_loss exists but is not HookPointV1")
            continue
        root.add_module(
            "dmi_lm_per_sample_loss",
            _make_hook(
                MegatronHookSpec(
                    name="lm_per_sample_loss",
                    layer_no=-1,
                    outputs=outputs,
                    preprocess=preprocess,
                    preprocess_metadata_fields=preprocess_metadata_fields,
                    shard_policy=ShardPolicy.REPLICATED,
                    layer_placement=HookLayerPlacement.NO_LAYER_LAST_PP,
                    enabled_by=frozenset({"loss-summary"}),
                    need_token_range=False,
                    supported_layouts=supported_layouts,
                ),
                hook_phase=HookPhase.FWD,
                suppress_recompute=False,
            ),
        )


def _install_router_summary_hooks(model: Any) -> None:
    roots = model if isinstance(model, list) else [model]
    for root in roots:
        for module in root.modules():
            if module.__class__.__name__ != "TopKRouter":
                continue
            existing = module.dmi_router_probs_mean
            if existing is not None:
                if not isinstance(existing, HookPointV1):
                    raise TypeError("router.dmi_router_probs_mean exists but is not HookPointV1")
                continue
            layer_number = getattr(module, "layer_number", None)
            layer_no = -1 if layer_number is None else int(layer_number) - 1
            module.dmi_router_probs_mean = _make_hook(
                MegatronHookSpec(
                    name="router_probs_mean",
                    layer_no=layer_no,
                    outputs=[
                        MegatronOutputSpec(
                            name="router_probs_mean",
                            input_shape=[DimSpec.BATCH, DimSpec.NUM_EXPERTS],
                            output_shape=[DimSpec.BATCH, DimSpec.NUM_EXPERTS],
                            dtype=torch.float32,
                            transport_type=TransportType.IDENTITY,
                        )
                    ],
                    preprocess=module._dmi_router_probs_mean_from_logits,
                    preprocess_metadata_fields=frozenset(
                        {MegatronMetadataField.VALID_COUNT}
                    ),
                    shard_policy=ShardPolicy.REPLICATED,
                    enabled_by=frozenset({"router-summary"}),
                ),
                hook_phase=HookPhase.FWD,
            )
            module.dmi_router_probs_mean.valid_count_fwd = torch.empty(0, dtype=torch.int64)
            module.dmi_router_probs_mean.valid_count_bwd = torch.empty(0, dtype=torch.int64)


def _install_router_logits_hooks(model: Any, *, dtype: torch.dtype) -> None:
    roots = model if isinstance(model, list) else [model]
    for root in roots:
        for module in root.modules():
            if module.__class__.__name__ != "TopKRouter":
                continue
            existing = module.dmi_router_logits
            if existing is not None:
                if not isinstance(existing, HookPointV1):
                    raise TypeError("router.dmi_router_logits exists but is not HookPointV1")
                continue
            layer_number = getattr(module, "layer_number", None)
            layer_no = -1 if layer_number is None else int(layer_number) - 1
            module.dmi_router_logits = _make_hook(
                MegatronHookSpec(
                    name="router_logits",
                    layer_no=layer_no,
                    outputs=[
                        MegatronOutputSpec(
                            name="router_logits",
                            input_shape=[
                                DimSpec.BATCH,
                                DimSpec.SEQ,
                                DimSpec.NUM_EXPERTS,
                            ],
                            output_shape=[
                                DimSpec.BATCH,
                                DimSpec.SEQ,
                                DimSpec.NUM_EXPERTS,
                            ],
                            dtype=dtype,
                            transport_type=TransportType.IDENTITY,
                        )
                    ],
                    preprocess=module._dmi_router_logits_by_sample,
                    shard_policy=ShardPolicy.REPLICATED,
                    enabled_by=frozenset({"router-logits"}),
                    need_token_range=False,
                ),
                hook_phase=HookPhase.FWD,
            )


def _install_router_topk_hooks(model: Any, *, dtype: torch.dtype) -> None:
    roots = model if isinstance(model, list) else [model]
    for root in roots:
        for module in root.modules():
            if module.__class__.__name__ != "TopKRouter":
                continue
            existing = module.dmi_router_topk
            if existing is not None:
                if not isinstance(existing, HookPointV1):
                    raise TypeError("router.dmi_router_topk exists but is not HookPointV1")
                continue
            layer_number = getattr(module, "layer_number", None)
            layer_no = -1 if layer_number is None else int(layer_number) - 1
            top_k = int(module.topk)
            module.dmi_router_topk = _make_hook(
                MegatronHookSpec(
                    name="router_topk",
                    layer_no=layer_no,
                    outputs=[
                        MegatronOutputSpec(
                            name="router_topk_expert_ids",
                            input_shape=[DimSpec.BATCH, DimSpec.SEQ, top_k],
                            output_shape=[DimSpec.BATCH, DimSpec.SEQ, top_k],
                            dtype=torch.int64,
                            transport_type=TransportType.IDENTITY,
                        ),
                        MegatronOutputSpec(
                            name="router_topk_weights",
                            input_shape=[DimSpec.BATCH, DimSpec.SEQ, top_k],
                            output_shape=[DimSpec.BATCH, DimSpec.SEQ, top_k],
                            dtype=dtype,
                            transport_type=TransportType.IDENTITY,
                        ),
                    ],
                    preprocess=module._dmi_router_topk_from_routing,
                    shard_policy=ShardPolicy.GLOBAL_RANK_SHARDED,
                    enabled_by=frozenset({"router-topk"}),
                    need_token_range=True,
                ),
                hook_phase=HookPhase.FWD,
            )


def _install_moe_inverse_map_hooks(model: Any) -> None:
    roots = model if isinstance(model, list) else [model]
    for root in roots:
        for module in root.modules():
            if module.__class__.__name__ != "MoELayer":
                continue
            dispatcher = module.token_dispatcher
            if dispatcher.__class__.__name__ != "MoEAlltoAllTokenDispatcher":
                raise NotImplementedError(
                    "DMI moe-inverse-map requires Megatron's AlltoAll token dispatcher"
                )
            existing = getattr(module, "dmi_moe_inverse_map", None)
            if existing is None:
                layer_number = getattr(module, "layer_number", None)
                layer_no = -1 if layer_number is None else int(layer_number) - 1
                module.add_module(
                    "dmi_moe_inverse_map",
                    _make_hook(
                        MegatronHookSpec(
                            name="moe_inverse_map",
                            layer_no=layer_no,
                            outputs=[
                                MegatronOutputSpec(
                                    name="moe_inverse_map",
                                    # Eager IDENTITY emission obtains shape and bytes from the
                                    # runtime tensor.  ACTUAL_TOKEN_PACKED is descriptive here;
                                    # it neither sizes nor validates the eager payload.  This
                                    # first-milestone hook must remain outside CUDA Graph replay.
                                    input_shape=[DimSpec.ACTUAL_TOKEN_PACKED],
                                    output_shape=[DimSpec.ACTUAL_TOKEN_PACKED],
                                    dtype=torch.int64,
                                    transport_type=TransportType.IDENTITY,
                                )
                            ],
                            shard_policy=ShardPolicy.GLOBAL_RANK_SHARDED,
                            enabled_by=frozenset({"moe-inverse-map"}),
                            need_token_range=False,
                            record_type=RecordType.PER_EXECUTION,
                        ),
                        hook_phase=HookPhase.FWD,
                    ),
                )
                existing = module.dmi_moe_inverse_map
            elif not isinstance(existing, HookPointV1):
                raise TypeError("layer.dmi_moe_inverse_map exists but is not HookPointV1")
            dispatcher.dmi_moe_inverse_map = existing


def _install_moe_packed_weighted_output_hooks(model: Any) -> None:
    roots = model if isinstance(model, list) else [model]
    for root in roots:
        for module in root.modules():
            if module.__class__.__name__ != "MoELayer":
                continue
            existing = module.dmi_moe_packed_weighted_output
            if existing is not None:
                if not isinstance(existing, HookPointV1):
                    raise TypeError(
                        "layer.dmi_moe_packed_weighted_output exists but is not HookPointV1"
                    )
                continue
            layer_number = getattr(module, "layer_number", None)
            layer_no = -1 if layer_number is None else int(layer_number) - 1
            module.dmi_moe_packed_weighted_output = _make_hook(
                MegatronHookSpec(
                    name="moe_packed_weighted_output",
                    layer_no=layer_no,
                    outputs=[
                        MegatronOutputSpec(
                            name="moe_packed_weighted_output",
                            # Eager IDENTITY emission obtains shape and bytes from the
                            # runtime tensor.  ACTUAL_TOKEN_PACKED and HIDDEN are descriptive
                            # annotations, not eager sizing or validation inputs.  CUDA Graph
                            # support requires a separate fixed-capacity or dynamic-plan design.
                            input_shape=[DimSpec.ACTUAL_TOKEN_PACKED, DimSpec.HIDDEN],
                            output_shape=[DimSpec.ACTUAL_TOKEN_PACKED, DimSpec.HIDDEN],
                            dtype=module.config.params_dtype,
                            transport_type=TransportType.IDENTITY,
                        )
                    ],
                    shard_policy=ShardPolicy.GLOBAL_RANK_SHARDED,
                    enabled_by=frozenset({"moe-packed-weighted-output"}),
                    need_token_range=False,
                    record_type=RecordType.PER_EXECUTION,
                ),
                hook_phase=HookPhase.FWD,
            )


def _install_vocab_logits_hooks(
    model: Any,
    *,
    dtype: torch.dtype,
    rank: int,
) -> ShardPolicy | None:
    roots = model if isinstance(model, list) else [model]
    effective_policy: ShardPolicy | None = None
    for root in roots:
        root_policy = (
            ShardPolicy.TP_SHARDED
            if bool(getattr(root, "parallel_output"))
            else ShardPolicy.REPLICATED
        )
        if effective_policy is not None and root_policy is not effective_policy:
            raise ValueError("DMI vocab-logits requires one fixed parallel_output layout")
        effective_policy = root_policy
        if not bool(getattr(root, "post_process", False)):
            continue
        existing = root.dmi_vocab_logits
        if existing is not None:
            if not isinstance(existing, HookPointV1):
                raise TypeError("model.dmi_vocab_logits exists but is not HookPointV1")
            if _megatron_hook_spec(existing).shard_policy is not root_policy:
                raise ValueError(
                    "existing DMI vocab-logits hook does not match GPTModel.parallel_output"
                )
            continue
        root.dmi_vocab_logits = _make_hook(
            MegatronHookSpec(
                name="vocab_logits",
                layer_no=-1,
                outputs=[
                    MegatronOutputSpec(
                        name="vocab_logits",
                        input_shape=[DimSpec.BATCH, DimSpec.SEQ, DimSpec.VOCAB],
                        output_shape=[DimSpec.BATCH, DimSpec.SEQ, DimSpec.VOCAB],
                        dtype=dtype,
                        transport_type=TransportType.IDENTITY,
                    )
                ],
                preprocess=vocab_logits_by_sample,
                shard_policy=root_policy,
                layer_placement=HookLayerPlacement.NO_LAYER_LAST_PP,
                enabled_by=frozenset({"vocab-logits"}),
                need_token_range=False,
                supported_layouts=frozenset(
                    {HookInputLayout.SEQ_BATCH, HookInputLayout.PACKED_SEGMENTED}
                ),
            ),
            hook_phase=HookPhase.FWD,
        )
    if effective_policy is ShardPolicy.REPLICATED and rank == 0:
        message = (
            "[DMI] WARNING: vocab-logits uses replicated layout because "
            "GPTModel.parallel_output=False"
        )
        _print_warning(message)
    return effective_policy


def _install_vocab_logits_topk_hooks(
    model: Any,
    *,
    dtype: torch.dtype,
    top_k: int,
) -> None:
    roots = model if isinstance(model, list) else [model]
    for root in roots:
        if not bool(getattr(root, "post_process", False)):
            continue
        existing = root.dmi_vocab_logits_topk
        if existing is not None:
            if not isinstance(existing, HookPointV1):
                raise TypeError("model.dmi_vocab_logits_topk exists but is not HookPointV1")
            continue

        def preprocess(logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            return vocab_logits_topk_by_sample(logits, k=top_k)

        root.dmi_vocab_logits_topk = _make_hook(
            MegatronHookSpec(
                name="vocab_logits_topk",
                layer_no=-1,
                outputs=[
                    MegatronOutputSpec(
                        name="vocab_logits_topk_values",
                        input_shape=[DimSpec.BATCH, DimSpec.SEQ, top_k],
                        output_shape=[DimSpec.BATCH, DimSpec.SEQ, top_k],
                        dtype=dtype,
                        transport_type=TransportType.IDENTITY,
                    ),
                    MegatronOutputSpec(
                        name="vocab_logits_topk_indices",
                        input_shape=[DimSpec.BATCH, DimSpec.SEQ, top_k],
                        output_shape=[DimSpec.BATCH, DimSpec.SEQ, top_k],
                        dtype=torch.int32,
                        transport_type=TransportType.IDENTITY,
                    ),
                ],
                preprocess=preprocess,
                shard_policy=ShardPolicy.REPLICATED,
                layer_placement=HookLayerPlacement.NO_LAYER_LAST_PP,
                enabled_by=frozenset({"vocab-logits-topk"}),
                need_token_range=False,
                supported_layouts=frozenset(
                    {HookInputLayout.SEQ_BATCH, HookInputLayout.PACKED_SEGMENTED}
                ),
            ),
            hook_phase=HookPhase.FWD,
        )


def _install_router_entropy_hooks(model: Any) -> None:
    roots = model if isinstance(model, list) else [model]
    for root in roots:
        for module in root.modules():
            if module.__class__.__name__ != "TopKRouter":
                continue
            existing = module.dmi_router_token_entropy_mean
            if existing is not None:
                if not isinstance(existing, HookPointV1):
                    raise TypeError(
                        "router.dmi_router_token_entropy_mean exists but is not HookPointV1"
                    )
                continue
            layer_number = getattr(module, "layer_number", None)
            layer_no = -1 if layer_number is None else int(layer_number) - 1
            module.dmi_router_token_entropy_mean = _make_hook(
                MegatronHookSpec(
                    name="router_token_entropy_mean",
                    layer_no=layer_no,
                    outputs=[
                        MegatronOutputSpec(
                            name="router_token_entropy_mean",
                            input_shape=[DimSpec.BATCH, 1],
                            output_shape=[DimSpec.BATCH, 1],
                            dtype=torch.float32,
                            transport_type=TransportType.IDENTITY,
                            storage=OutputStorage.SCALAR_FLOAT,
                        )
                    ],
                    preprocess=module._dmi_router_token_entropy_mean_from_logits,
                    preprocess_metadata_fields=frozenset(
                        {MegatronMetadataField.VALID_COUNT}
                    ),
                    shard_policy=ShardPolicy.REPLICATED,
                    enabled_by=frozenset({"router-entropy"}),
                    need_token_range=True,
                ),
                hook_phase=HookPhase.FWD,
            )
            module.dmi_router_token_entropy_mean.valid_count_fwd = torch.empty(0, dtype=torch.int64)
            module.dmi_router_token_entropy_mean.valid_count_bwd = torch.empty(0, dtype=torch.int64)


def _install_expert_count_hooks(model: Any) -> None:
    roots = model if isinstance(model, list) else [model]
    for root in roots:
        for module in root.modules():
            if module.__class__.__name__ != "TopKRouter":
                continue
            layer_number = getattr(module, "layer_number", None)
            layer_no = -1 if layer_number is None else int(layer_number) - 1
            if module.dmi_pre_drop_token_count is None:
                module.dmi_pre_drop_token_count = _make_hook(
                    MegatronHookSpec(
                        name="pre_drop_token_count",
                        layer_no=layer_no,
                        outputs=[
                            MegatronOutputSpec(
                                name="pre_drop_token_count",
                                input_shape=[DimSpec.BATCH, DimSpec.NUM_EXPERTS],
                                output_shape=[DimSpec.BATCH, DimSpec.NUM_EXPERTS],
                                dtype=torch.int64,
                                transport_type=TransportType.IDENTITY,
                            )
                        ],
                        preprocess=module._dmi_expert_token_count_from_routing_map,
                        preprocess_metadata_fields=frozenset(
                            {MegatronMetadataField.VALID_COUNT}
                        ),
                        shard_policy=ShardPolicy.REPLICATED,
                        enabled_by=frozenset({"expert-counts"}),
                        need_token_range=True,
                    ),
                    hook_phase=HookPhase.FWD,
                )
                module.dmi_pre_drop_token_count.valid_count_fwd = torch.empty(0, dtype=torch.int64)
                module.dmi_pre_drop_token_count.valid_count_bwd = torch.empty(0, dtype=torch.int64)
            elif not isinstance(module.dmi_pre_drop_token_count, HookPointV1):
                raise TypeError("router.dmi_pre_drop_token_count exists but is not HookPointV1")

            if module.dmi_post_drop_token_count is None:
                module.dmi_post_drop_token_count = _make_hook(
                    MegatronHookSpec(
                        name="post_drop_token_count",
                        layer_no=layer_no,
                        outputs=[
                            MegatronOutputSpec(
                                name="post_drop_token_count",
                                input_shape=[DimSpec.BATCH, DimSpec.NUM_EXPERTS],
                                output_shape=[DimSpec.BATCH, DimSpec.NUM_EXPERTS],
                                dtype=torch.int64,
                                transport_type=TransportType.IDENTITY,
                            )
                        ],
                        preprocess=module._dmi_expert_token_count_from_routing_map,
                        preprocess_metadata_fields=frozenset(
                            {MegatronMetadataField.VALID_COUNT}
                        ),
                        shard_policy=ShardPolicy.REPLICATED,
                        enabled_by=frozenset({"expert-counts"}),
                        need_token_range=True,
                    ),
                    hook_phase=HookPhase.FWD,
                )
                module.dmi_post_drop_token_count.valid_count_fwd = torch.empty(0, dtype=torch.int64)
                module.dmi_post_drop_token_count.valid_count_bwd = torch.empty(0, dtype=torch.int64)
            elif not isinstance(module.dmi_post_drop_token_count, HookPointV1):
                raise TypeError("router.dmi_post_drop_token_count exists but is not HookPointV1")


def _install_hidden_state_hooks(model: Any) -> None:
    roots = model if isinstance(model, list) else [model]
    for root in roots:
        for module in root.modules():
            if module.__class__.__name__ != "TransformerLayer":
                continue
            existing = module.dmi_hidden_states
            if existing is not None:
                if not isinstance(existing, HookPointV1):
                    raise TypeError("layer.dmi_hidden_states exists but is not HookPointV1")
                continue
            module.dmi_hidden_states = _make_hook(
                MegatronHookSpec(
                    name="hidden_states",
                    layer_no=int(module.layer_number) - 1,
                    outputs=[
                        MegatronOutputSpec(
                            name="hidden_states",
                            input_shape=[DimSpec.SEQ, DimSpec.BATCH, DimSpec.HIDDEN],
                            output_shape=[DimSpec.ACTUAL_TOKEN_PACKED, DimSpec.HIDDEN],
                            dtype=module.config.params_dtype,
                            transport_type=TransportType.SEQ_PREFIX_PACK,
                        )
                    ],
                    preprocess=None,
                    shard_policy=ShardPolicy.TP_SEQUENCE_SHARDED,
                    enabled_by=frozenset({"hidden-states"}),
                ),
                hook_phase=HookPhase.FWD,
            )


def _install_resid_final_hooks(model: Any) -> int:
    roots = model if isinstance(model, list) else [model]
    blocks = []
    for root in roots:
        for module in root.modules():
            if module.__class__.__name__ != "TransformerBlock":
                continue
            blocks.append(module)

    final_blocks = [block for block in blocks if block.final_layernorm is not None]
    # A non-last pipeline stage has no final norm, but recompute policy names are
    # parsed on every rank. Give that stage one placement-disabled HookPoint so
    # the globally selected name still resolves locally. It is never invoked:
    # TransformerBlock.forward calls dmi_resid_final only with final_layernorm.
    targets = final_blocks if final_blocks else blocks[:1]
    for module in targets:
        existing = module.dmi_resid_final
        if existing is not None:
            if not isinstance(existing, HookPointV1):
                raise TypeError("block.dmi_resid_final exists but is not HookPointV1")
            continue
        module.dmi_resid_final = _make_hook(
            MegatronHookSpec(
                name="resid_final",
                layer_no=-1,
                outputs=[
                    MegatronOutputSpec(
                        name="hook_resid_final",
                        input_shape=[DimSpec.SEQ, DimSpec.BATCH, DimSpec.HIDDEN],
                        output_shape=[DimSpec.ACTUAL_TOKEN_PACKED, DimSpec.HIDDEN],
                        dtype=module.config.params_dtype,
                        transport_type=TransportType.SEQ_PREFIX_PACK,
                    )
                ],
                preprocess=None,
                shard_policy=ShardPolicy.TP_SEQUENCE_SHARDED,
                layer_placement=HookLayerPlacement.NO_LAYER_LAST_PP,
                enabled_by=frozenset({"resid_final"}),
            ),
            hook_phase=HookPhase.FWD,
        )
    return len(final_blocks)


def _make_grad_norm_hook() -> HookPointV1:
    hook = _make_hook(
        MegatronHookSpec(
            name="grad_norm",
            layer_no=-1,
            outputs=[
                MegatronOutputSpec(
                    name="grad_norm",
                    input_shape=[1],
                    output_shape=[1],
                    dtype=torch.float32,
                    transport_type=TransportType.IDENTITY,
                    storage=OutputStorage.SCALAR_FLOAT,
                )
            ],
            shard_policy=ShardPolicy.REPLICATED,
            layer_placement=HookLayerPlacement.NO_LAYER_LAST_PP,
            enabled_by=frozenset({"grad-norm"}),
            need_token_range=False,
            record_type=RecordType.PER_ITERATION,
            dp_emission=DPEmissionPolicy.DP_RANK_0,
        ),
        hook_phase=HookPhase.ITERATION,
    )
    _validate_hook_contract(hook)
    return hook


def _make_attempt_status_hook() -> HookPointV1:
    hook = _make_hook(
        MegatronHookSpec(
            name="iteration_attempt_status",
            layer_no=-1,
            outputs=[
                MegatronOutputSpec(
                    name="iteration_attempt_status",
                    input_shape=[1],
                    output_shape=[1],
                    dtype=torch.int64,
                    transport_type=TransportType.IDENTITY,
                    storage=OutputStorage.SCALAR_INT,
                )
            ],
            shard_policy=ShardPolicy.REPLICATED,
            layer_placement=HookLayerPlacement.NO_LAYER_LAST_PP,
            enabled_by=frozenset(),
            need_token_range=False,
            record_type=RecordType.PER_ITERATION,
            dp_emission=DPEmissionPolicy.DP_RANK_0,
        ),
        hook_phase=HookPhase.ITERATION,
    )
    _validate_hook_contract(hook)
    return hook


def _router_weight_bindings(
    model: Any,
    *,
    rank_ctx: MegatronRankContext,
    num_experts: int,
    hidden_size: int,
) -> tuple[tuple[MegatronHookBinding, ...], tuple[MegatronRouterWeightBinding, ...]]:
    discovered: list[tuple[int, torch.nn.Parameter]] = []
    seen_modules: set[int] = set()
    for root in _model_roots(model):
        for module in root.modules():
            if module.__class__.__name__ != "TopKRouter" or id(module) in seen_modules:
                continue
            seen_modules.add(id(module))
            layer_number = getattr(module, "layer_number", None)
            if layer_number is None:
                raise ValueError("DMI router-weight collection requires a global layer number")
            layer_no = int(layer_number) - 1
            if layer_no < 0:
                raise ValueError(f"Invalid DMI router global layer number: {layer_number}")
            weight = getattr(module, "weight", None)
            if not isinstance(weight, torch.nn.Parameter):
                raise TypeError(f"TopKRouter layer {layer_no} weight is not a Parameter")
            if not weight.is_cuda:
                raise RuntimeError(f"TopKRouter layer {layer_no} weight must be CUDA-resident")
            expected_shape = (int(num_experts), int(hidden_size))
            if tuple(int(x) for x in weight.shape) != expected_shape:
                raise ValueError(
                    f"TopKRouter layer {layer_no} weight shape {tuple(weight.shape)} "
                    f"does not match complete router shape {expected_shape}"
                )
            if any(
                bool(getattr(weight, name, False))
                for name in ("tensor_model_parallel", "expert_model_parallel", "context_parallel")
            ):
                raise NotImplementedError(
                    f"TopKRouter layer {layer_no} weight is model-parallel sharded"
                )
            discovered.append((layer_no, weight))

    layer_nos = [layer_no for layer_no, _ in discovered]
    if len(layer_nos) != len(set(layer_nos)):
        raise ValueError(f"Duplicate local TopKRouter layer numbers: {sorted(layer_nos)}")

    hook_bindings: list[MegatronHookBinding] = []
    parameter_bindings: list[MegatronRouterWeightBinding] = []
    for layer_no, weight in sorted(discovered):
        hook = _make_hook(
            MegatronHookSpec(
                name="router_projection_weight",
                layer_no=layer_no,
                outputs=[
                    MegatronOutputSpec(
                        name="router_projection_weight",
                        input_shape=[int(num_experts), int(hidden_size)],
                        output_shape=[int(num_experts), int(hidden_size)],
                        dtype=weight.dtype,
                        transport_type=TransportType.IDENTITY,
                        storage=OutputStorage.TENSOR,
                    )
                ],
                shard_policy=ShardPolicy.REPLICATED,
                layer_placement=HookLayerPlacement.EVERY_LAYER,
                enabled_by=frozenset({"router-weights"}),
                need_token_range=False,
                record_type=RecordType.PER_ITERATION,
                dp_emission=DPEmissionPolicy.DP_RANK_0,
            ),
            hook_phase=HookPhase.ITERATION,
        )
        _validate_hook_contract(hook)
        if not _spec_active_on_rank(_megatron_hook_spec(hook), rank_ctx):
            hook.enabled = False
            continue
        hook_bindings.append(
            MegatronHookBinding(
                hook=hook,
                record_dp_rank=-1,
                record_shard_rank=0,
            )
        )
        parameter_bindings.append(
            MegatronRouterWeightBinding(hook=hook, parameter=weight)
        )
    return tuple(hook_bindings), tuple(parameter_bindings)


def _qk_weight_bindings(
    model: Any,
    *,
    rank_ctx: MegatronRankContext,
    selected_hooks: set[str],
) -> tuple[
    tuple[MegatronHookBinding, ...],
    tuple[tuple[HookPointV1, torch.nn.Parameter], ...],
]:
    """Bind standard Megatron/OLMoE self-attention Q/K weights, never V."""
    hook_bindings: list[MegatronHookBinding] = []
    parameter_bindings: list[tuple[HookPointV1, torch.nn.Parameter]] = []
    seen_modules: set[int] = set()
    seen_layers: set[int] = set()
    for root in _model_roots(model):
        for module in root.modules():
            # Include subclasses such as OlmoeSelfAttention, but not the nested
            # DotProductAttention, which also has attention_type="self".
            if not any(cls.__name__ == "SelfAttention" for cls in type(module).__mro__):
                continue
            if id(module) in seen_modules:
                continue
            seen_modules.add(id(module))
            layer_number = getattr(module, "layer_number", None)
            if layer_number is None or int(layer_number) <= 0:
                raise ValueError("DMI Q/K weights require a positive global layer number")
            layer_no = int(layer_number) - 1
            if layer_no in seen_layers:
                raise ValueError(f"Duplicate local self-attention layer number: {layer_number}")
            seen_layers.add(layer_no)
            config = module.config
            if bool(getattr(config, "attention_output_gate", False)):
                raise NotImplementedError("DMI Q/K weights do not support gated attention")
            if int(config.num_query_groups) < rank_ctx.tp_world_size:
                raise NotImplementedError("DMI Q/K weights require TP size <= number of KV heads")
            if getattr(config, "fp8", None) or getattr(config, "fp4", None):
                raise NotImplementedError("DMI Q/K weights require non-quantized QKV parameters")
            linear = getattr(module, "linear_qkv", None)
            weight = getattr(linear, "weight", None)
            if not isinstance(weight, torch.nn.Parameter):
                raise TypeError(f"Self-attention layer {layer_no} requires a fused QKV Parameter")
            if not weight.is_cuda:
                raise RuntimeError(f"Self-attention layer {layer_no} QKV weight must be CUDA-resident")
            groups = int(module.num_query_groups_per_partition)
            heads = int(module.num_attention_heads_per_partition)
            head_dim = int(module.hidden_size_per_attention_head)
            if min(groups, heads, head_dim) <= 0 or heads % groups:
                raise ValueError(f"Invalid local QKV head layout at layer {layer_no}")
            query_rows = (heads // groups) * head_dim
            hidden_size = int(config.hidden_size)
            expected = (groups * (query_rows + 2 * head_dim), hidden_size)
            if tuple(weight.shape) != expected or not weight.is_contiguous():
                raise ValueError(
                    f"Self-attention layer {layer_no} requires contiguous grouped QKV "
                    f"shape {expected}, got {tuple(weight.shape)}"
                )
            for projection, name, width in (
                ("q", "query_projection_weight", query_rows),
                ("k", "key_projection_weight", head_dim),
            ):
                selection = f"{projection}-weights"
                if selection not in selected_hooks:
                    continue
                shape = [groups * width, hidden_size]
                hook = _make_hook(
                    MegatronHookSpec(
                        name=name,
                        layer_no=layer_no,
                        outputs=[MegatronOutputSpec(
                            name=name,
                            input_shape=shape,
                            output_shape=shape,
                            dtype=weight.dtype,
                            transport_type=TransportType.IDENTITY,
                            storage=OutputStorage.TENSOR,
                        )],
                        preprocess=partial(
                            qk_weight_from_fused_qkv,
                            num_query_groups=groups,
                            query_rows_per_group=query_rows,
                            head_dim=head_dim,
                            projection=projection,
                        ),
                        shard_policy=ShardPolicy.TP_SHARDED,
                        layer_placement=HookLayerPlacement.EVERY_LAYER,
                        enabled_by=frozenset({selection}),
                        need_token_range=False,
                        record_type=RecordType.PER_ITERATION,
                        dp_emission=DPEmissionPolicy.DP_RANK_0,
                    ),
                    hook_phase=HookPhase.ITERATION,
                )
                _validate_hook_contract(hook)
                if not _spec_active_on_rank(_megatron_hook_spec(hook), rank_ctx):
                    hook.enabled = False
                    continue
                hook_bindings.append(MegatronHookBinding(
                    hook=hook, record_dp_rank=-1, record_shard_rank=rank_ctx.tp_rank,
                ))
                parameter_bindings.append((hook, weight))
    return tuple(hook_bindings), tuple(parameter_bindings)


def _model_roots(model: Any) -> list[torch.nn.Module]:
    if isinstance(model, torch.nn.Module):
        return [model]
    return list(model)


def _model_scope_id(root: torch.nn.Module, root_index: int, root_count: int) -> int:
    vp_stage = getattr(root, "vp_stage", None)
    return int(vp_stage if vp_stage is not None else root_index if root_count > 1 else 0)


def _collect_selected_hooks(
    model: Any,
    hook_selection: str,
) -> list[MegatronHookBinding]:
    selected = {name.strip() for name in str(hook_selection).split(",")}
    if "" in selected:
        raise ValueError(f"Invalid empty DMI hook selection entry: {hook_selection!r}")
    hooks: list[MegatronHookBinding] = []
    roots = _model_roots(model)
    for root_index, root in enumerate(roots):
        scope_id = _model_scope_id(root, root_index, len(roots))
        for module in root.modules():
            if not isinstance(module, HookPointV1):
                continue
            spec = _megatron_hook_spec(module)
            if spec.enabled_by and not (selected & set(spec.enabled_by)):
                continue
            _validate_hook_contract(module)
            hooks.append(MegatronHookBinding(hook=module, scope_id=scope_id))
    return hooks


def _resolve_tp_sequence_shard_policies(
    hooks: list[MegatronHookBinding],
    *,
    sequence_parallel_enabled: bool,
    sequence_length: int,
    tp_world_size: int,
    cp_world_size: int,
    rank: int,
) -> None:
    sequence_hooks = [
        binding.hook
        for binding in hooks
        if _megatron_hook_spec(binding.hook).shard_policy
        is ShardPolicy.TP_SEQUENCE_SHARDED
    ]
    if not sequence_hooks:
        return
    if cp_world_size != 1:
        raise NotImplementedError(
            "DMI TP-sequence-sharded capture requires context-parallel size 1; "
            f"got {cp_world_size}"
        )
    if sequence_parallel_enabled or sequence_length % tp_world_size == 0:
        return

    names = sorted({_megatron_hook_spec(hook).name for hook in sequence_hooks})
    for hook in sequence_hooks:
        hook._dmi_megatron_spec = replace(
            _megatron_hook_spec(hook),
            shard_policy=ShardPolicy.REPLICATED,
        )
    if rank == 0:
        message = (
            f"[DMI] WARNING: {','.join(names)} use replicated layout because "
            f"sequence length {sequence_length} is not divisible by "
            f"tensor-model-parallel size {tp_world_size} while sequence "
            "parallelism is disabled"
        )
        _print_warning(message)


def _active_hooks_for_rank(
    hooks: list[MegatronHookBinding],
    rank_ctx: MegatronRankContext,
) -> list[MegatronHookBinding]:
    active: list[MegatronHookBinding] = []
    for binding in hooks:
        hook = binding.hook
        spec = _megatron_hook_spec(hook)
        if _spec_active_on_rank(spec, rank_ctx):
            record_dp_rank = binding.record_dp_rank
            shard_rank = 0
            if spec.record_type == RecordType.PER_EXECUTION:
                record_dp_rank = -1
                shard_rank = rank_ctx.global_rank
            elif spec.shard_policy == ShardPolicy.GLOBAL_RANK_SHARDED:
                shard_rank = rank_ctx.global_rank
            elif spec.shard_policy in (
                ShardPolicy.TP_SHARDED,
                ShardPolicy.TP_SEQUENCE_SHARDED,
            ):
                shard_rank = rank_ctx.tp_rank
            elif spec.shard_policy == ShardPolicy.EP_SHARDED:
                shard_rank = rank_ctx.ep_rank
            elif spec.shard_policy == ShardPolicy.DP_SHARDED:
                shard_rank = rank_ctx.dp_rank
            elif spec.shard_policy == ShardPolicy.CP_SHARDED:
                shard_rank = rank_ctx.cp_rank
            active.append(
                replace(
                    binding,
                    record_dp_rank=record_dp_rank,
                    record_shard_rank=int(shard_rank),
                )
            )
        else:
            hook.enabled = False
    return active


def _hooks_for_input_layout(
    hooks: list[MegatronHookBinding],
    active_layout: HookInputLayout,
    *,
    rank: int,
) -> list[MegatronHookBinding]:
    active: list[MegatronHookBinding] = []
    for binding in hooks:
        spec = _megatron_hook_spec(binding.hook)
        if active_layout in spec.supported_layouts:
            active.append(binding)
            continue
        binding.hook.enabled = False
        if rank == 0:
            message = (
                f"[DMI] disabling hook {spec.name!r}: active input layout "
                f"{active_layout.value!r} is not in "
                f"{sorted(item.value for item in spec.supported_layouts)!r}"
            )
            _print_warning(message)
    return active


def _ordered_metadata_fields(
    fields: set[MegatronMetadataField] | frozenset[MegatronMetadataField],
) -> tuple[MegatronMetadataField, ...]:
    return tuple(field for field in MegatronMetadataField if field in fields)


def _local_metadata_requirement_report(
    hooks: list[MegatronHookBinding],
    *,
    rank_ctx: MegatronRankContext,
    input_layout: HookInputLayout,
    max_batch_size: int,
    segment_capacity: int | None,
) -> _MetadataRequirementReport:
    gpu_fields: set[MegatronMetadataField] = set()
    cpu_record_fields: set[MegatronMetadataField] = set()
    has_per_sample_records = False
    for binding in hooks:
        spec = _megatron_hook_spec(binding.hook)
        gpu_fields.update(spec.binding_metadata_fields)
        if spec.record_type is RecordType.PER_SAMPLE:
            has_per_sample_records = True
        for output in spec.outputs:
            cpu_record_fields.update(
                required_record_metadata_fields(
                    record_type=spec.record_type,
                    need_token_range=spec.need_token_range,
                    transport_type=output.transport_type,
                    dynamic_dataset_provenance=False,
                )
            )
    return _MetadataRequirementReport(
        global_rank=int(rank_ctx.global_rank),
        dense_dp_rank=int(rank_ctx.dp_rank),
        tp_rank=int(rank_ctx.tp_rank),
        tp_world_size=int(rank_ctx.tp_world_size),
        pp_rank=int(rank_ctx.pp_rank),
        pp_world_size=int(rank_ctx.pp_world_size),
        input_layout=input_layout,
        max_batch_size=int(max_batch_size),
        segment_capacity=(
            None if segment_capacity is None else int(segment_capacity)
        ),
        gpu_fields=_ordered_metadata_fields(gpu_fields),
        cpu_record_fields=_ordered_metadata_fields(cpu_record_fields),
        has_per_sample_records=bool(has_per_sample_records),
    )


def _gather_metadata_requirement_reports(
    local_report: _MetadataRequirementReport,
    *,
    dist_module: Any,
) -> tuple[_MetadataRequirementReport, ...]:
    if not _dist_ready(dist_module):
        return (local_report,)
    gathered: list[Any] = [None] * int(dist_module.get_world_size())
    dist_module.all_gather_object(gathered, local_report)
    if any(not isinstance(item, _MetadataRequirementReport) for item in gathered):
        raise TypeError(
            "WORLD all_gather_object returned an invalid DMI metadata requirement report"
        )
    reports = tuple(sorted(gathered, key=lambda item: int(item.global_rank)))
    ranks = tuple(int(item.global_rank) for item in reports)
    if len(ranks) != len(set(ranks)):
        raise ValueError("DMI metadata requirement reports contain duplicate global ranks")
    return reports


def _resolve_metadata_requirements(
    reports: tuple[_MetadataRequirementReport, ...],
    *,
    local_report: _MetadataRequirementReport,
    dataset_provenance_modes: Mapping[str, str],
) -> _ResolvedMetadataRequirements:
    domain = tuple(
        report
        for report in reports
        if int(report.dense_dp_rank) == int(local_report.dense_dp_rank)
    )
    if not domain:
        raise RuntimeError("DMI metadata requirement exchange omitted the local DP domain")
    if not any(
        int(report.global_rank) == int(local_report.global_rank) for report in domain
    ):
        raise RuntimeError("DMI metadata requirement exchange omitted the local rank")

    for report in domain:
        if (
            int(report.tp_world_size) != int(local_report.tp_world_size)
            or int(report.pp_world_size) != int(local_report.pp_world_size)
            or report.input_layout is not local_report.input_layout
            or int(report.max_batch_size) != int(local_report.max_batch_size)
            or report.segment_capacity != local_report.segment_capacity
        ):
            raise ValueError(
                "DMI metadata broadcast-domain members disagree on layout or capacity"
            )

    wire_fields: set[MegatronMetadataField] = set()
    for report in domain:
        wire_fields.update(report.gpu_fields)
        wire_fields.update(report.cpu_record_fields)
    has_per_sample_records = any(
        report.has_per_sample_records for report in domain
    )
    if has_per_sample_records and any(
        mode == "dynamic" for mode in dataset_provenance_modes.values()
    ):
        wire_fields.add(MegatronMetadataField.DATASET_ID)

    local_gpu_fields = frozenset(local_report.gpu_fields)
    if not local_gpu_fields.issubset(wire_fields):
        raise RuntimeError("DMI local GPU metadata requirements are absent from wire schema")
    return _ResolvedMetadataRequirements(
        wire_fields=_ordered_metadata_fields(wire_fields),
        local_gpu_fields=local_gpu_fields,
        has_per_sample_records=has_per_sample_records,
    )


def _metadata_field_specs_from_requirements(
    requirements: _ResolvedMetadataRequirements,
    *,
    segment_capacity: int | None,
) -> tuple[DMIMetadataFieldSpec, ...]:
    specs: list[DMIMetadataFieldSpec] = []
    for field in requirements.wire_fields:
        gpu_visible = field in requirements.local_gpu_fields
        if field is MegatronMetadataField.VALID_COUNT:
            specs.append(
                valid_count_field_spec(
                    segment_capacity if segment_capacity is not None else DimSpec.BATCH,
                    gpu_visible=gpu_visible,
                )
            )
        elif field is MegatronMetadataField.SEGMENT_METADATA:
            if segment_capacity is None or int(segment_capacity) <= 0:
                raise ValueError(
                    "DMI segment metadata requires a positive packed segment capacity"
                )
            specs.append(
                replace(
                    segment_metadata_field_spec(int(segment_capacity)),
                    gpu_visible=gpu_visible,
                )
            )
        elif field is MegatronMetadataField.DATASET_ID:
            if gpu_visible:
                raise ValueError("DMI dataset_id metadata must remain CPU-only")
            specs.append(
                dataset_id_field_spec(
                    segment_capacity if segment_capacity is not None else DimSpec.BATCH
                )
            )
        else:
            raise ValueError(f"Unsupported DMI metadata field: {field!r}")
    return tuple(specs)


def _build_engine(
    cfg: MegatronDMIConfig,
    model_id: str,
    record_format: MegatronRecordFormat,
    rank: int = 0,
) -> tuple[MonitoringEngine, Any | None]:
    from dmi.api.v1 import (
        ClickHouseClientConfig,
        DMXHostEngine,
        RecurringD2HWindowConfig,
        RingConfig,
        StageConfig,
    )

    del rank
    ring_cfg = RingConfig()
    ring_cfg.payload_ring_bytes = int(cfg.ring_payload_mb) * 1024 * 1024
    ring_cfg.pinned_staging_bytes = int(cfg.ring_pinned_mb) * 1024 * 1024
    ring_cfg.task_ring_entries = int(cfg.ring_task_entries)
    ring_cfg.drain_flush_payload_ratio = float(cfg.drain_flush_payload_ratio)
    ring_cfg.drain_flush_task_ratio = float(cfg.drain_flush_task_ratio)
    ring_cfg.drain_flush_byte_threshold = int(cfg.drain_flush_byte_threshold)
    ring_cfg.drain_flush_entry_threshold = int(cfg.drain_flush_entry_threshold)
    ring_cfg.drain_flush_timeout_us = int(cfg.drain_flush_timeout_us)
    window_cfg = RecurringD2HWindowConfig()
    window_cfg.enabled = cfg.recurring_d2h_windows_enabled
    window_cfg.minimum_record_probe_retry_interval_occurrences = (
        cfg.d2h_window_minimum_record_probe_retry_interval_occurrences
    )
    window_cfg.timing_revalidation_retry_interval_occurrences = (
        cfg.d2h_window_timing_revalidation_retry_interval_occurrences
    )
    window_cfg.capacity_flush_fallback_threshold = cfg.d2h_window_capacity_flush_fallback_threshold
    window_cfg.capacity_flush_count_reset_interval_periods = (
        cfg.d2h_window_capacity_flush_count_reset_interval_periods
    )
    window_cfg.debug_enabled = cfg.d2h_window_debug
    ring_cfg.recurring_d2h_windows = window_cfg

    host_engine = None
    if cfg.db_host:
        ch_cfg = ClickHouseClientConfig()
        ch_cfg.host = cfg.db_host
        ch_cfg.port = int(cfg.db_port)
        ch_cfg.database = cfg.db_database
        ch_cfg.create_database_if_missing = True
        if cfg.exact_resume:
            ch_cfg.client_settings = {"async_insert": False}
        host_engine = DMXHostEngine(
            StageConfig.clickhouse_records(
                ch_cfg,
                record_format.schema,
                parallelism=int(cfg.ch_parallelism),
                name="clickhouse_training_records",
            )
        )

    return (
        MonitoringEngine(
            config=None,
            model_id=model_id,
            host_engine=host_engine,
            ring_config=ring_cfg,
            record_mode_v1=True,
        ),
        host_engine,
    )


def _device_for_setup(device: torch.device | str | int | None = None) -> torch.device | str | int:
    if device is not None:
        return device
    if torch.cuda.is_available():
        return torch.cuda.current_device()
    return "cpu"


def _process_group_global_ranks(dist_module: Any, group: Any, name: str) -> tuple[int, ...]:
    ranks = tuple(int(rank) for rank in dist_module.get_process_group_ranks(group))
    if not ranks:
        raise ValueError(f"Megatron {name} process group must not be empty")
    return ranks


def _build_ep_topology_fragment(
    *,
    model_id: str,
    global_rank: int,
    model: Any,
    model_config: Any,
    parallel_state: Any,
    dist_module: Any,
) -> MegatronEPTopologyFragment:
    roots = _model_roots(model)
    moe_layers: list[MoELayerFragment] = []
    local_expert_orders: set[tuple[int, ...]] = set()
    for root_index, root in enumerate(roots):
        scope_id = _model_scope_id(root, root_index, len(roots))
        for module in root.modules():
            if module.__class__.__name__ != "MoELayer":
                continue
            layer_number = getattr(module, "layer_number", None)
            if layer_number is None or int(layer_number) <= 0:
                raise ValueError("DMI topology manifest requires a positive global MoE layer number")
            moe_layers.append(MoELayerFragment(int(layer_number) - 1, scope_id))
            local_expert_indices = getattr(module, "local_expert_indices", None)
            if local_expert_indices is None:
                raise ValueError("DMI topology manifest requires MoELayer.local_expert_indices")
            local_expert_orders.add(tuple(int(expert) for expert in local_expert_indices))
    if len(local_expert_orders) > 1:
        raise ValueError("Local MoE layers disagree on local expert order")

    tp_group = parallel_state.get_tensor_model_parallel_group()
    etp_group = parallel_state.get_expert_tensor_parallel_group(check_initialized=False)
    if etp_group is None:
        etp_group = tp_group

    return MegatronEPTopologyFragment(
        model_id=str(model_id),
        global_rank=int(global_rank),
        tp_group=_process_group_global_ranks(dist_module, tp_group, "TP"),
        pp_group=_process_group_global_ranks(
            dist_module, parallel_state.get_pipeline_model_parallel_group(), "PP"
        ),
        dp_group=_process_group_global_ranks(
            dist_module, parallel_state.get_data_parallel_group(), "DP"
        ),
        cp_group=_process_group_global_ranks(
            dist_module, parallel_state.get_context_parallel_group(), "CP"
        ),
        ep_group=_process_group_global_ranks(
            dist_module, parallel_state.get_expert_model_parallel_group(), "EP"
        ),
        etp_group=_process_group_global_ranks(dist_module, etp_group, "ETP"),
        dispatch_group=_process_group_global_ranks(
            dist_module,
            parallel_state.get_expert_tensor_and_model_parallel_group(),
            "ETP-by-EP",
        ),
        expert_dp_group=_process_group_global_ranks(
            dist_module, parallel_state.get_expert_data_parallel_group(), "expert-DP"
        ),
        moe_layers=tuple(sorted(set(moe_layers))),
        local_expert_order=(
            next(iter(local_expert_orders)) if local_expert_orders else None
        ),
        sequence_parallel=bool(model_config.sequence_parallel),
        top_k=int(model_config.moe_router_topk),
        dispatcher_type=str(model_config.moe_token_dispatcher_type),
        permutation_mode=("fused" if bool(model_config.moe_permute_fusion) else "non_fused"),
        etp_composition="matching_row_sum",
        dropless=(
            model_config.moe_expert_capacity_factor is None
            and not bool(model_config.moe_token_dropping)
        ),
        padded=bool(model_config.moe_pad_expert_input_to_capacity),
    )


def _freeze_ep_topology_manifest(
    *,
    path: str,
    model_id: str,
    global_rank: int,
    model: Any,
    model_config: Any,
    parallel_state: Any,
    dist_module: Any,
) -> None:
    if not _dist_ready(dist_module):
        raise RuntimeError("DMI topology manifest requires initialized torch.distributed")
    fragment = _build_ep_topology_fragment(
        model_id=model_id,
        global_rank=global_rank,
        model=model,
        model_config=model_config,
        parallel_state=parallel_state,
        dist_module=dist_module,
    )
    gathered: list[Any] = [None] * int(dist_module.get_world_size())
    dist_module.all_gather_object(gathered, fragment)
    if any(not isinstance(item, MegatronEPTopologyFragment) for item in gathered):
        raise TypeError("WORLD all_gather_object returned an invalid DMI topology fragment")
    manifest = assemble_ep_topology_manifest(gathered)
    if int(global_rank) == 0:
        write_ep_topology_manifest(path, manifest)


def setup_megatron_dmi(
    model: Any,
    *,
    args: Any | None = None,
    model_config: Any | None = None,
    explicit_config: MegatronDMIConfig | None = None,
    environ: Mapping[str, str] | None = None,
    parallel_state_module: Any | None = None,
    dist_module: Any | None = None,
    unwrap_fn: Any | None = None,
    printer: Any | None = None,
    engine_factory: Any | None = None,
    runtime_factory: Any | None = None,
    adaptor_cls: Any = MegatronAdaptor,
    device: torch.device | str | int | None = None,
    dataset_provider: Any | None = None,
) -> MegatronDMIHandle | None:
    """Enable DMI for Megatron when configured, otherwise return ``None``."""

    cfg = resolve_megatron_dmi_config(args, explicit=explicit_config, environ=environ)
    if not cfg.enabled:
        return None
    cfg = replace(
        cfg,
        recurring_d2h_windows_enabled=(
            cfg.recurring_d2h_windows_enabled
            and int(getattr(args, "pipeline_model_parallel_size", 1)) > 1
        ),
    )
    if int(cfg.flush_every_n_train_iters) < 0:
        raise ValueError("DMI iteration flush interval must be nonnegative")

    if parallel_state_module is None:
        from megatron.core import parallel_state as parallel_state_module
    if unwrap_fn is None:
        from megatron.training.utils import unwrap_model as unwrap_fn
    if model_config is None:
        from megatron.core.utils import get_model_config

        model_config = get_model_config(model[0] if isinstance(model, list) else model)
    dist_for_rank = torch.distributed if dist_module is None else dist_module
    rank = int(dist_for_rank.get_rank()) if _dist_ready(dist_for_rank) else 0
    if cfg.recurring_d2h_windows_enabled and bool(
        getattr(model_config, "batch_p2p_sync", False)
    ):
        message = (
            "[DMI] WARNING: recurring D2H windows require Megatron "
            "batch_p2p_sync=False; disabling recurring D2H windows"
        )
        if rank == 0:
            _print_warning(message)
        cfg = replace(cfg, recurring_d2h_windows_enabled=False)

    model_id = resolve_model_id(cfg, dist_module=dist_module, environ=environ, printer=printer)
    dp_world = _data_parallel_world_size(args, parallel_state_module)
    max_num_microbatches = _max_num_microbatches(args, dp_world)
    max_batch_size = int(getattr(args, "micro_batch_size"))
    scopes = _num_scopes(parallel_state_module)
    selected_hooks = _selected_hooks(cfg.hook_selection)
    tp_world = _parallel_world(
        parallel_state_module, "get_tensor_model_parallel_world_size"
    )
    cp_world = _parallel_world(
        parallel_state_module, "get_context_parallel_world_size"
    )
    requires_ep_topology_manifest = bool(
        {"moe-inverse-map", "moe-packed-weighted-output"} & selected_hooks
    )
    if requires_ep_topology_manifest and not cfg.topology_manifest_path:
        raise ValueError(
            "DMI MoE reconstruction hooks require DMI_TOPOLOGY_MANIFEST_PATH"
        )
    vocab_hook_names = {"vocab-logits", "vocab-logits-topk"}
    selected_vocab_hooks = selected_hooks & vocab_hook_names
    padded_vocab_size = None
    if selected_vocab_hooks:
        if "vocab-logits-topk" in selected_hooks and tp_world != 1:
            raise NotImplementedError(
                "DMI vocab-logits-topk requires tensor-model-parallel size 1; "
                f"got {tp_world}"
            )
        if cp_world != 1:
            raise NotImplementedError(
                "DMI vocab-logits requires context-parallel size 1; "
                f"got {cp_world}"
            )
        padded_vocab_size = _padded_vocab_size(args)
    sequence_parallel_enabled = bool(
        getattr(
            model_config,
            "sequence_parallel",
            getattr(args, "sequence_parallel", False),
        )
    )
    top_k = cfg.vocab_logits_top_k
    if "vocab-logits-topk" in selected_hooks:
        if top_k is None:
            raise ValueError(
                "DMI vocab-logits-topk requires --dmi-vocab-logits-top-k"
            )
        top_k = int(top_k)
        if top_k < 1 or padded_vocab_size is None or top_k > padded_vocab_size:
            raise ValueError(
                "DMI vocabulary-logit top-K must satisfy "
                f"1 <= K <= padded_vocab_size ({padded_vocab_size}); got {top_k}"
            )
    elif top_k is not None:
        raise ValueError(
            "DMI vocabulary-logit top-K was configured without selecting vocab-logits-topk"
        )
    active_input_layout = (
        HookInputLayout.PACKED_SEGMENTED
        if bool(getattr(args, "sft", False))
        else HookInputLayout.SEQ_BATCH
    )
    segment_capacity = None
    if active_input_layout is HookInputLayout.PACKED_SEGMENTED:
        row_capacity = getattr(
            args, "dmi_packed_max_conversations_per_row", None
        )
        if row_capacity is None or int(row_capacity) <= 0:
            raise RuntimeError(
                "DMI packed SFT requires a positive "
                "--dmi-packed-max-conversations-per-row"
            )
        segment_capacity = max_batch_size * int(row_capacity)
    if "loss-summary" in selected_hooks and int(
        getattr(args, "context_parallel_size", 1)
    ) != 1:
        raise NotImplementedError(
            "DMI exact loss-summary materialization currently requires context parallel size 1"
        )
    weight_selections = {"router-weights", "q-weights", "k-weights"} & selected_hooks
    if weight_selections:
        weight_names = ",".join(sorted(weight_selections))
        weight_dp_world = int(
            parallel_state_module.get_data_parallel_world_size(
                with_context_parallel=False
            )
        )
        if weight_dp_world != 1:
            raise NotImplementedError(
                f"DMI {weight_names} requires data-parallel world size exactly 1; "
                f"got {weight_dp_world}"
            )
        if bool(getattr(args, "reuse_grad_buf_for_mxfp8_param_ag", False)) and bool(
            getattr(args, "overlap_param_gather", False)
        ):
            raise NotImplementedError(
                f"DMI {weight_names} does not support "
                "reuse_grad_buf_for_mxfp8_param_ag + overlap_param_gather"
            )
    dims: dict[Any, int] = {DimSpec.BATCH: max_batch_size}
    if {
        "router-logits",
        "router-summary",
        "router-entropy",
        "expert-counts",
        "router-weights",
    } & selected_hooks:
        dims[DimSpec.NUM_EXPERTS] = _num_experts(model_config)
    if {
        "hidden-states",
        "resid_final",
        "router-weights",
        "moe-packed-weighted-output",
    } & selected_hooks:
        dims[DimSpec.HIDDEN] = _hidden_size(model_config)
    if {"router-logits", "router-topk", "hidden-states", "resid_final"} & selected_hooks:
        dims[DimSpec.SEQ] = _seq_length(args)
    if selected_vocab_hooks:
        dims[DimSpec.SEQ] = _seq_length(args)
    if "vocab-logits" in selected_hooks:
        dims[DimSpec.VOCAB] = int(padded_vocab_size)
    setup_device = _device_for_setup(device)

    engine = None
    host_engine = None
    runtime = None
    try:
        unwrapped = unwrap_fn(model)
        vocab_logits_policy = None
        if "vocab-logits" in selected_hooks:
            vocab_logits_policy = _install_vocab_logits_hooks(
                unwrapped,
                dtype=_vocab_logits_dtype(model_config),
                rank=rank,
            )
            if vocab_logits_policy is ShardPolicy.TP_SHARDED:
                assert padded_vocab_size is not None
                if int(padded_vocab_size) % tp_world != 0:
                    raise ValueError(
                        "DMI TP-sharded vocab-logits requires padded vocabulary "
                        "size divisible by tensor-model-parallel size: "
                        f"{padded_vocab_size} % {tp_world} != 0"
                    )
                dims[DimSpec.VOCAB] = int(padded_vocab_size) // tp_world
        if "vocab-logits-topk" in selected_hooks:
            _install_vocab_logits_topk_hooks(
                unwrapped,
                dtype=_vocab_logits_dtype(model_config),
                top_k=int(top_k),
            )
        if "router-logits" in selected_hooks:
            _install_router_logits_hooks(
                unwrapped,
                dtype=_router_logits_dtype(model_config),
            )
        if "router-topk" in selected_hooks:
            _install_router_topk_hooks(
                unwrapped,
                dtype=_router_logits_dtype(model_config),
            )
        if "moe-inverse-map" in selected_hooks:
            _install_moe_inverse_map_hooks(unwrapped)
        if "moe-packed-weighted-output" in selected_hooks:
            _install_moe_packed_weighted_output_hooks(unwrapped)
        if "router-summary" in selected_hooks:
            _install_router_summary_hooks(unwrapped)
        if "router-entropy" in selected_hooks:
            _install_router_entropy_hooks(unwrapped)
        if "expert-counts" in selected_hooks:
            _install_expert_count_hooks(unwrapped)
        if "loss-summary" in selected_hooks:
            _install_loss_summary_hook(
                unwrapped,
                input_layout=active_input_layout,
                segment_capacity=segment_capacity,
            )
        if "hidden-states" in selected_hooks:
            _install_hidden_state_hooks(unwrapped)
        resid_final_hook_count = 0
        if "resid_final" in selected_hooks:
            resid_final_hook_count = _install_resid_final_hooks(unwrapped)

        rank_ctx = _build_rank_context(
            args,
            model_config,
            parallel_state_module,
            global_rank=rank,
        )
        world_size = (
            int(dist_for_rank.get_world_size())
            if _dist_ready(dist_for_rank)
            else int(getattr(args, "world_size", 1))
        )
        distributed_info = MegatronDistributedInfo(
            global_rank=rank_ctx.global_rank,
            world_size=world_size,
            tp_rank=rank_ctx.tp_rank,
            tp_world_size=rank_ctx.tp_world_size,
            pp_rank=rank_ctx.pp_rank,
            pp_world_size=rank_ctx.pp_world_size,
            dp_rank=rank_ctx.dp_rank,
            dp_world_size=rank_ctx.dp_world_size,
            ep_rank=rank_ctx.ep_rank,
            ep_world_size=rank_ctx.ep_world_size,
            cp_rank=rank_ctx.cp_rank,
            cp_world_size=rank_ctx.cp_world_size,
            sequence_parallel_enabled=sequence_parallel_enabled,
        )
        if (
            "resid_final" in selected_hooks
            and rank_ctx.pp_rank == rank_ctx.pp_world_size - 1
            and resid_final_hook_count == 0
        ):
            raise RuntimeError(
                "DMI resid_final requires a TransformerBlock with final_layernorm "
                "on the last pipeline stage"
            )
        selected_model_hooks = _collect_selected_hooks(unwrapped, cfg.hook_selection)
        has_selected_tp_sequence_hook = any(
            _megatron_hook_spec(binding.hook).shard_policy
            is ShardPolicy.TP_SEQUENCE_SHARDED
            for binding in selected_model_hooks
        )
        if has_selected_tp_sequence_hook:
            dims[DimSpec.SEQ] = _seq_length(args)
            _resolve_tp_sequence_shard_policies(
                selected_model_hooks,
                sequence_parallel_enabled=sequence_parallel_enabled,
                sequence_length=int(dims[DimSpec.SEQ]),
                tp_world_size=tp_world,
                cp_world_size=cp_world,
                rank=rank,
            )
        _apply_recompute_hook_policy(
            selected_model_hooks,
            selected_names=selected_hooks,
            recompute_names_raw=cfg.recompute_hook,
            no_recompute_names_raw=cfg.no_recompute_hook,
        )
        active_model_hooks = _active_hooks_for_rank(selected_model_hooks, rank_ctx)
        active_model_hooks = _hooks_for_input_layout(
            active_model_hooks,
            active_input_layout,
            rank=rank,
        )
        local_metadata_report = _local_metadata_requirement_report(
            active_model_hooks,
            rank_ctx=rank_ctx,
            input_layout=active_input_layout,
            max_batch_size=max_batch_size,
            segment_capacity=segment_capacity,
        )
        metadata_reports = _gather_metadata_requirement_reports(
            local_metadata_report,
            dist_module=dist_for_rank,
        )
        domain_has_per_sample_records = any(
            report.has_per_sample_records
            for report in metadata_reports
            if int(report.dense_dp_rank) == int(rank_ctx.dp_rank)
        )
        dataset_provenance_modes = _resolve_dataset_provenance_modes(
            args,
            dataset_provider,
            configured_mode=cfg.dataset_provenance_mode,
            has_per_sample_hooks=domain_has_per_sample_records,
        )

        iteration_hooks: list[MegatronHookBinding] = []
        attempt_status_hook = _make_attempt_status_hook()
        if _spec_active_on_rank(_megatron_hook_spec(attempt_status_hook), rank_ctx):
            iteration_hooks.append(
                MegatronHookBinding(
                    hook=attempt_status_hook,
                    record_dp_rank=-1,
                    record_shard_rank=-1,
                )
            )
        else:
            attempt_status_hook.enabled = False
            attempt_status_hook = None
        grad_norm_hook = None
        if "grad-norm" in selected_hooks:
            candidate = _make_grad_norm_hook()
            if _spec_active_on_rank(_megatron_hook_spec(candidate), rank_ctx):
                grad_norm_hook = candidate
                iteration_hooks.append(
                    MegatronHookBinding(
                        hook=candidate,
                        record_dp_rank=-1,
                        record_shard_rank=-1,
                    )
                )

        router_weight_bindings: tuple[MegatronRouterWeightBinding, ...] = ()
        if "router-weights" in selected_hooks:
            router_hooks, router_weight_bindings = _router_weight_bindings(
                unwrapped,
                rank_ctx=rank_ctx,
                num_experts=int(dims[DimSpec.NUM_EXPERTS]),
                hidden_size=int(dims[DimSpec.HIDDEN]),
            )
            iteration_hooks.extend(router_hooks)

        qk_weight_bindings: tuple[tuple[HookPointV1, torch.nn.Parameter], ...] = ()
        if {"q-weights", "k-weights"} & selected_hooks:
            qk_hooks, qk_weight_bindings = _qk_weight_bindings(
                unwrapped, rank_ctx=rank_ctx, selected_hooks=selected_hooks,
            )
            iteration_hooks.extend(qk_hooks)

        for binding in (*active_model_hooks, *iteration_hooks):
            binding.hook.megatron_distributed_info = distributed_info
        has_active_tp_sequence_hook = any(
            _megatron_hook_spec(binding.hook).shard_policy
            is ShardPolicy.TP_SEQUENCE_SHARDED
            for binding in active_model_hooks
        )

        metadata_requirements = _resolve_metadata_requirements(
            metadata_reports,
            local_report=local_metadata_report,
            dataset_provenance_modes=dataset_provenance_modes,
        )
        field_specs = _metadata_field_specs_from_requirements(
            metadata_requirements,
            segment_capacity=segment_capacity,
        )

        record_format = MegatronRecordFormat(cfg.clickhouse_table)
        engine, host_engine = (
            engine_factory(cfg, model_id, record_format, rank)
            if engine_factory is not None
            else _build_engine(cfg, model_id, record_format, rank)
        )
        record_runtime = engine.create_record_runtime(record_format)
        runtime_builder = build_megatron_schedule_runtime if runtime_factory is None else runtime_factory
        runtime = runtime_builder(
            max_num_microbatches=max_num_microbatches,
            max_batch_size=max_batch_size,
            num_scopes=scopes,
            device=setup_device,
            parallel_state_module=parallel_state_module,
            dist_module=dist_module,
            tensor_model_parallel_size=int(getattr(args, "tensor_model_parallel_size", 1)),
            pipeline_model_parallel_size=int(getattr(args, "pipeline_model_parallel_size", 1)),
            data_parallel_size=int(getattr(args, "data_parallel_size", 1)),
            context_parallel_size=int(getattr(args, "context_parallel_size", 1)),
            expert_model_parallel_size=int(getattr(args, "expert_model_parallel_size", 1)),
            rank_order=(
                "tp-cp-ep-pp-dp"
                if bool(getattr(args, "use_tp_pp_dp_mapping", False))
                else "tp-cp-ep-dp-pp"
            ),
            field_specs=field_specs,
            dims=dims,
            megatron_distributed_info=distributed_info,
            tp_sequence_sharded_enabled=has_active_tp_sequence_hook,
            host_engine=host_engine,
        )
        runtime.configure_d2h_windows(
            enabled=cfg.recurring_d2h_windows_enabled, debug=cfg.d2h_window_debug,
        )
        flush_interval = int(cfg.flush_every_n_train_iters)
        if flush_interval == 0:
            runtime.configure_iteration_flush(0)
        else:
            if not callable(getattr(engine, "flush_and_wait", None)):
                raise TypeError("DMI engine does not provide flush_and_wait")
            if _dist_ready(dist_for_rank):
                barrier = getattr(dist_for_rank, "barrier", None)
                if not callable(barrier):
                    raise TypeError("DMI distributed iteration flushing requires barrier()")
            else:
                barrier = lambda: None

            def log_iteration_flush(completed_iteration: int, elapsed_s: float) -> None:
                if rank != 0:
                    return
                message = (
                    "[DMI] iteration-boundary flush "
                    f"iteration={int(completed_iteration)} "
                    f"elapsed_s={float(elapsed_s):.6f}"
                )
                if printer is None:
                    print(message, flush=True)
                else:
                    printer(message)

            runtime.configure_iteration_flush(
                flush_interval,
                flush_callback=engine.flush_and_wait,
                barrier_callback=barrier,
                logger=log_iteration_flush,
            )
        set_active_megatron_schedule_runtime(runtime)
        runtime.configure_dataset_provenance(dataset_provenance_modes)

        current_phase_tensor = torch.empty((), dtype=torch.int32, device=setup_device)
        current_phase_tensor.fill_(int(HookPhase.FWD.value))
        runtime.set_current_phase_tensor(current_phase_tensor)

        adaptor = adaptor_cls(
            engine,
            record_runtime,
            model_id,
            dims=dims,
        )
        adaptor.attach_hooks(
            model_hooks=active_model_hooks,
            iteration_hooks=iteration_hooks,
            current_phase_tensor=current_phase_tensor,
            metadata_context=runtime.propagator.context,
            activation_recompute_enabled=(
                getattr(args, "recompute_granularity", None) is not None
            ),
            active_input_layout=active_input_layout,
        )
        runtime.adaptor = adaptor
        runtime.dp_rank = _parallel_rank(parallel_state_module, "get_data_parallel_rank")
        runtime.set_attempt_status_hook(attempt_status_hook, device=setup_device)
        handle = MegatronDMIHandle(
            config=replace(cfg, model_id=model_id),
            model_id=model_id,
            engine=engine,
            schedule_runtime=runtime,
            adaptor=adaptor,
            current_phase_tensor=current_phase_tensor,
            grad_norm_hook=grad_norm_hook,
            router_weight_bindings=router_weight_bindings,
            qk_weight_bindings=qk_weight_bindings,
        )
        if requires_ep_topology_manifest:
            assert cfg.topology_manifest_path is not None
            _freeze_ep_topology_manifest(
                path=cfg.topology_manifest_path,
                model_id=model_id,
                global_rank=rank,
                model=unwrapped,
                model_config=model_config,
                parallel_state=parallel_state_module,
                dist_module=dist_for_rank,
            )
        setattr(
            args,
            "dmi_required_metadata_fields",
            tuple(field.value for field in metadata_requirements.wire_fields),
        )
        atexit.register(handle.close)
        return handle
    except Exception:
        set_active_megatron_schedule_runtime(None)
        if engine is not None:
            close = getattr(engine, "close", None)
            if close is not None:
                close()
        raise


__all__ = [
    "MegatronDMIConfig",
    "MegatronDMIHandle",
    "resolve_megatron_dmi_config",
    "resolve_model_id",
    "setup_megatron_dmi",
]
