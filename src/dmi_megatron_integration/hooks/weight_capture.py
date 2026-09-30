"""Pre-update model-weight fragments, independent of forward invocation count.

All ranges are bytes. A source maps persistent local storage to the TP-local
parameter; a projection maps that parameter into a complete logical matrix.
Only layout metadata is exchanged between ranks, never model-weight payloads.
"""
from dataclasses import dataclass
from math import prod
from typing import Callable

import torch

WEIGHT_NAMES = frozenset(('query_projection_weight', 'key_projection_weight',
                          'router_projection_weight'))


def intersect_maps(storage, projection):
    """Compose (local, parameter, length) and (parameter, output, length)."""
    result = []
    for local, parameter, length in storage:
        for start, output, count in projection:
            lo, hi = max(parameter, start), min(parameter + length, start + count)
            if lo < hi:
                result.append((local + lo - parameter, output + lo - start, hi - lo))
    return result


def qk_projection_ranges(*, heads, groups, head_dim, hidden, tp_rank, tp_size,
                         element_size, projection):
    """Map a contiguous TP slice of grouped QKV to global Q or K bytes.

    This also covers TP > KV heads: physical TP boundaries may cut a Q/K
    group, and some TP ranks contain no K at all.
    """
    if projection not in ('q', 'k') or heads % groups:
        raise ValueError('Invalid Q/K projection layout')
    qrows = heads // groups * head_dim
    group_rows = qrows + 2 * head_dim
    total_rows = groups * group_rows
    if total_rows % tp_size:
        raise ValueError('Fused QKV rows must be divisible by TP size')
    row_bytes = hidden * element_size
    local_rows = total_rows // tp_size
    begin, end = tp_rank * local_rows, (tp_rank + 1) * local_rows
    width, offset = (qrows, 0) if projection == 'q' else (head_dim, qrows)
    ranges = []
    for group in range(groups):
        start = group * group_rows + offset
        lo, hi = max(begin, start), min(end, start + width)
        if lo < hi:
            ranges.append(((lo - begin) * row_bytes,
                           (group * width + lo - start) * row_bytes,
                           (hi - lo) * row_bytes))
    return (groups * width, hidden), (local_rows, hidden), ranges


@dataclass
class WeightSource:
    get: Callable[[], torch.Tensor]
    shape: tuple
    dtype: torch.dtype
    ranges: list
    kind: str

    def read_bytes(self):
        tensor = self.get().detach()
        if tensor.dtype != self.dtype or not tensor.is_contiguous():
            raise ValueError('Model-weight storage dtype/layout changed after binding')
        return tensor.reshape(-1).view(torch.uint8)


def _matrix_local_ranges(shape, local_shape, offsets, element_size):
    """DTensor rectangles -> flat byte spans, including uneven column shards."""
    if len(shape) != 2:
        raise ValueError('Weight source must be a matrix')
    rows, cols = local_shape
    if cols == shape[1]:
        return [(0, offsets[0] * cols * element_size, rows * cols * element_size)]
    return [(r * cols * element_size,
             ((r + offsets[0]) * shape[1] + offsets[1]) * element_size,
             cols * element_size) for r in range(rows) if cols]


def resolve_weight_source(owner, parameter_name, *, model_roots, expected_shape):
    """Resolve retained compute-weight storage, never optimizer master weights."""
    parameter = getattr(owner, parameter_name)
    if not isinstance(parameter, torch.nn.Parameter):
        raise TypeError('Weight source must be a model Parameter')
    expected_shape = tuple(expected_shape)
    seen = set()
    for root in model_roots:
        for wrapper in root.modules():
            buffer = getattr(wrapper, 'param_and_grad_buffer', None)
            if buffer is None or id(buffer) in seen:
                continue
            seen.add(id(buffer))
            mapping = getattr(buffer, 'param_to_param_group', {})
            # Megatron FSDP exposes optimizer DTensors between forwards. Its
            # orig_param links back to the compute parameter indexed by wbuf.
            compute_parameter = getattr(parameter, 'orig_param', parameter)
            if compute_parameter not in mapping:
                continue
            group = buffer.parameter_groups[mapping[compute_parameter]]
            wbuf = getattr(group, 'hfsdp_helper_wbuf', None)
            if wbuf is None:
                wbuf = group.model_weight_buffer
            if wbuf is None:
                continue
            item = wbuf.param_idx[compute_parameter]
            if int(wbuf.item_index_map[item].size) != prod(expected_shape):
                raise ValueError('FSDP parameter extent does not match model layout')
            start, end = wbuf.locate_item_in_global_item(item)
            size = torch.empty((), dtype=wbuf.dtype).element_size()
            return WeightSource(lambda b=wbuf, i=item: b.get_item(i), expected_shape,
                                wbuf.dtype, [(0, start * size, (end - start) * size)],
                                'megatron_fsdp')

    if getattr(parameter, '__fsdp_param__', False):
        raise ValueError('Megatron FSDP compute-weight buffer could not be resolved')
    from torch.distributed.tensor import DTensor
    if isinstance(parameter, DTensor):
        from torch.distributed.tensor._utils import compute_local_shape_and_global_offset
        shape, offsets = compute_local_shape_and_global_offset(
            parameter.shape, parameter.device_mesh, parameter.placements)
        if tuple(parameter.shape) != expected_shape:
            raise ValueError('DTensor logical shape does not match TP-local parameter')
        ranges = _matrix_local_ranges(expected_shape, shape, offsets, parameter.element_size())
        return WeightSource(lambda: getattr(owner, parameter_name).to_local(),
                            expected_shape, parameter.dtype, ranges, 'torch_fsdp2')
    if tuple(parameter.shape) != expected_shape or not parameter.is_contiguous():
        raise ValueError(f'Weight shape/layout {tuple(parameter.shape)} != {expected_shape}')
    return WeightSource(lambda: getattr(owner, parameter_name), expected_shape,
                        parameter.dtype, [(0, 0, parameter.numel() * parameter.element_size())],
                        'replicated')


@dataclass
class WeightCapture:
    layer_no: int
    act_name: str
    source: WeightSource
    shape: tuple
    available: list
    producer_rank: int
    assigned: list | None = None
    hook: object = None

    def report(self):
        return dict(layer_no=self.layer_no, act_name=self.act_name, shape=list(self.shape),
                    dtype=str(self.source.dtype).removeprefix('torch.'),
                    producer_rank=self.producer_rank, available=self.available,
                    source_kind=self.source.kind)

    def pack(self):
        if self.assigned is None:
            raise RuntimeError('Weight capture has no rank assignment')
        source = self.source.read_bytes()
        parts = []
        for start, _, size in self.assigned:
            if start < 0 or size < 0 or start + size > source.numel():
                raise ValueError(
                    f'Weight fragment exceeds retained model storage: {self.act_name} '
                    f'layer={self.layer_no} rank={self.producer_rank} '
                    f'source={self.source.kind} range=({start}, {size}) '
                    f'available_bytes={source.numel()}')
            parts.append(source[start:start + size])
        # Nonempty captures currently make two GPU copies before D2H:
        # torch.cat copies assigned fragments into a temporary packed tensor,
        # then the DMI producer copies that tensor into the GPU ring on this
        # same stream before the optimizer may mutate the weights. Rank
        # deduplication reduces the bytes but does not remove the packing copy.
        return torch.cat(parts) if parts else source.new_empty((0,))


def assign_weight_fragments(reports):
    """Partition overlapping replicas, retain distinct shards, validate coverage."""
    grouped = {}
    for report in reports:
        grouped.setdefault((report['layer_no'], report['act_name']), []).append(report)
    layouts = []
    for (layer, name), members in sorted(grouped.items()):
        shapes = {(tuple(m['shape']), m['dtype']) for m in members}
        if len(shapes) != 1 or len({m['producer_rank'] for m in members}) != len(members):
            raise ValueError('Inconsistent weight shape/dtype or duplicate rank declaration')
        shape, dtype_name = shapes.pop()
        dtype = getattr(torch, dtype_name)
        total = prod(shape) * torch.empty((), dtype=dtype).element_size()
        cuts = {0, total}
        for member in members:
            previous_end = -1
            for local, start, size in sorted(member['available'], key=lambda r: r[1]):
                if local < 0 or start < 0 or size < 0 or start + size > total or start < previous_end:
                    raise ValueError('Invalid/overlapping local weight ranges')
                previous_end = start + size
                cuts.update((start, start + size))
        assignments = {m['producer_rank']: [] for m in members}
        cuts = sorted(cuts)
        for lo, hi in zip(cuts, cuts[1:]):
            owners = []
            for member in members:
                for local, start, size in member['available']:
                    if start <= lo and start + size >= hi:
                        owners.append((member['producer_rank'], local + lo - start))
            owners.sort()
            if not owners:
                raise ValueError(f'Missing weight coverage for {name} [{lo}, {hi})')
            for index, (rank, local) in enumerate(owners):
                a = (hi - lo) * index // len(owners)
                b = (hi - lo) * (index + 1) // len(owners)
                if a < b:
                    assignments[rank].append((local + a, lo + a, b - a))
        for member in members:
            rank = member['producer_rank']
            layouts.append(dict(layer_no=layer, act_name=name, producer_rank=rank,
                                shape=list(shape), dtype=dtype_name,
                                fragments=assignments[rank], source_kind=member['source_kind']))
    return layouts


def discover_weight_captures(model, rank_ctx, selected_hooks, *, capture_layers=None):
    roots = [model] if isinstance(model, torch.nn.Module) else list(model)
    captures, seen = [], set()
    for root in roots:
        for module in root.modules():
            if id(module) in seen:
                continue
            seen.add(id(module))
            router = module.__class__.__name__ == 'TopKRouter' and 'router-weights' in selected_hooks
            attention = any(c.__name__ == 'SelfAttention' for c in type(module).__mro__)
            if not router and not (attention and {'q-weights', 'k-weights'} & selected_hooks):
                continue
            layer = int(module.layer_number) - 1
            if layer < 0:
                raise ValueError('Weight hook requires a global layer number')
            if capture_layers is not None and layer not in capture_layers:
                continue
            config = module.config
            if getattr(config, 'fp8', None) or getattr(config, 'fp4', None):
                raise NotImplementedError('Weight capture requires non-quantized model weights')
            if router:
                if any(bool(getattr(module.weight, flag, False)) for flag in
                       ('tensor_model_parallel', 'expert_model_parallel', 'context_parallel')):
                    raise ValueError('Router parameter is not replicated in the declared model layout')
                shape = (int(config.num_moe_experts), int(config.hidden_size))
                owner, projections = module, [('router_projection_weight', shape, None)]
            else:
                if getattr(config, 'attention_output_gate', False):
                    raise NotImplementedError('Weight capture does not support gated attention')
                owner = module.linear_qkv
                groups = int(config.num_query_groups)
                heads = int(config.num_attention_heads)
                head_dim, hidden = int(module.hidden_size_per_attention_head), int(config.hidden_size)
                if int(getattr(owner.weight, 'partition_stride', 1)) != 1:
                    raise NotImplementedError('QKV capture requires contiguous TP row partitions')
                projections = []
                for short, name in [('q', 'query_projection_weight'), ('k', 'key_projection_weight')]:
                    if f'{short}-weights' not in selected_hooks:
                        continue
                    complete, shape, ranges = qk_projection_ranges(
                        heads=heads, groups=groups, head_dim=head_dim, hidden=hidden,
                        tp_rank=rank_ctx.tp_rank, tp_size=rank_ctx.tp_world_size,
                        element_size=1, projection=short)
                    projections.append((name, complete, ranges))
            source = resolve_weight_source(owner, 'weight', model_roots=roots, expected_shape=shape)
            if source.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
                raise NotImplementedError('Weight capture requires floating model weights')
            element_size = torch.empty((), dtype=source.dtype).element_size()
            for name, complete, projection in projections:
                if projection is None:
                    projection = [(0, 0, prod(shape) * element_size)]
                else:
                    # Derive byte offsets from compute storage, not the possibly
                    # FP32 optimizer DTensor currently installed on the module.
                    projection = [tuple(v * element_size for v in span) for span in projection]
                captures.append(WeightCapture(layer, name, source, complete,
                                              intersect_maps(source.ranges, projection),
                                              rank_ctx.global_rank))
    keys = [(c.layer_no, c.act_name) for c in captures]
    if len(set(keys)) != len(keys):
        raise ValueError('Duplicate logical weight on one producer rank')
    return captures
