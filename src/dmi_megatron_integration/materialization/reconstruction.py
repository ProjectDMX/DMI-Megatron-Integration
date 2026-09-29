"""Storage-side reconstruction transforms; numerical analyses remain user functions."""
import json
from collections import defaultdict
from dataclasses import asdict
from ..signals.definition import Row


def _manifest(rows):
    from ..topology.ep_topology_manifest import FrozenMegatronEPTopologyManifest
    values = {row['manifest_json'] for row in rows}
    if len(values) != 1:
        raise ValueError('Reconstruction requires one consistent topology manifest')
    return FrozenMegatronEPTopologyManifest.from_dict(json.loads(values.pop()))


def reconstruct_expert_outputs(payload_rows, topology_rows):
    """Return one row collection: source-sample [seq, top_k, hidden] outputs."""
    import torch
    from .ep_clickhouse_reconstruction import reconstruct_moe_clickhouse_rows
    from ..records.schema import TRAINING_ROW_COORDINATE_COLUMN_NAMES
    manifest = _manifest(topology_rows)
    grouped = defaultdict(list)
    seen = {}
    for row in payload_rows:
        identity = tuple(row[name] for name in TRAINING_ROW_COORDINATE_COLUMN_NAMES)
        if identity in seen:
            if not torch.equal(seen[identity], row['value']):
                raise ValueError('Duplicate EP payload identity has different values')
            continue
        seen[identity] = row['value']
        grouped[row['act_name']].append((identity, row['value']))
    output = []
    for invocation in reconstruct_moe_clickhouse_rows(manifest, grouped):
        for domain in invocation.source_domains:
            samples = defaultdict(list)
            for i, coordinate in enumerate(domain.token_coordinates):
                samples[(coordinate.dataset_id, coordinate.sample_index)].append(i)
            for (dataset, sample), indices in samples.items():
                output.append(Row(**asdict(invocation.key), dp_rank=domain.dense_dp_rank,
                                  dataset_id=dataset, sample_index=sample,
                                  token_indices=[domain.token_coordinates[i].token_index for i in indices],
                                  expert_ids=domain.selected_expert_ids[indices],
                                  weighted_outputs=domain.weighted_outputs[indices]))
    return (output,)


def regroup_ep_by_expert(rows):
    """Regroup without discarding identity; pairs must intersect token_ids for CKA."""
    import torch
    grouped = defaultdict(lambda: {'outputs': [], 'token_ids': []})
    for row in rows:
        for token, token_index in enumerate(row['token_indices']):
            identity = tuple(row[name] for name in ('model_id','phase','global_batch_id','attempt_id','microbatch_id','layer_no','direction','dp_rank','dataset_id','sample_index')) + (token_index,)
            for slot, expert in enumerate(row['expert_ids'][token].tolist()):
                grouped[expert]['outputs'].append(row['weighted_outputs'][token, slot])
                grouped[expert]['token_ids'].append(identity)
    return {expert: dict(outputs=torch.stack(data['outputs']), token_ids=tuple(data['token_ids'])) for expert, data in grouped.items()}


# The output name is part of the identity, so one invocation can merge several
# hook outputs without combining their values. Physical ranks are not outputs.
ROW_KEYS = (
    'model_id', 'phase', 'global_batch_id', 'attempt_id', 'microbatch_id',
    'layer_no', 'direction', 'dp_rank', 'dataset_id', 'sample_index',
    'invocation_id', 'act_name',
)
INTERVAL_KEYS = ROW_KEYS + ('token_start', 'token_end')


def _same_value(left, right):
    import torch
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        return left.dtype == right.dtype and torch.equal(left, right)
    return type(left) is type(right) and left == right


def _group_rows(rows, keys, shard_key):
    grouped = defaultdict(dict)
    for row in rows:
        key = tuple(row[name] for name in keys)
        shard = tuple(row[name] for name in shard_key)
        previous = grouped[key].get(shard)
        if previous is not None and (
            not _same_value(previous['value'], row['value'])
            or previous['producer_rank'] != row['producer_rank']
            or previous['shard_rank'] != row['shard_rank']
        ):
            raise ValueError('Conflicting duplicate shard')
        grouped[key][shard] = row
    return grouped


def _token_intervals(chunks):
    intervals = sorted(chunks)
    for i, (start, end) in enumerate(intervals):
        if end <= start or (i and start != intervals[i - 1][1]):
            raise ValueError('Token intervals overlap or contain gaps')
    return intervals


def _compatible_tensors(values, *, equal_shape=False):
    import torch
    if any(not isinstance(value, torch.Tensor) or value.ndim != 2 for value in values):
        raise ValueError('Expected two-dimensional tensor shards')
    if any(value.dtype != values[0].dtype for value in values):
        raise ValueError('Shard dtypes disagree')
    if equal_shape and any(value.shape != values[0].shape for value in values):
        raise ValueError('Shard shapes disagree')


def _run_topology(topology_rows, model_id):
    # Rows produced by capture carry model_id; historical manifest-only inputs
    # and existing CPU callers can omit it.
    return [row for row in topology_rows if row.get('model_id', model_id) == model_id]


def _tp_groups(topology_rows):
    if topology_rows and 'manifest_json' in topology_rows[0]:
        manifest = _manifest(topology_rows)
        if any(len(group) != 1 for group in manifest.cp_groups):
            raise NotImplementedError('Reconstruction currently requires CP=1')
        return manifest.tp_groups
    groups, sizes, producers = defaultdict(dict), {}, {}
    for row in topology_rows:
        if row['cp_rank'] != 0:
            raise NotImplementedError('Reconstruction currently requires CP=1')
        key = (row['pp_rank'], row['dp_rank'], row['cp_rank'])
        local, size, rank = row['tp_rank'], row['tp_world_size'], row['producer_rank']
        if size <= 0 or not 0 <= local < size:
            raise ValueError('Invalid TP topology')
        if key in sizes and sizes[key] != size:
            raise ValueError('Inconsistent TP group size')
        if local in groups[key] and groups[key][local] != rank:
            raise ValueError('Conflicting TP rank mapping')
        if rank in producers and producers[rank] != (key, local):
            raise ValueError('Producer belongs to multiple TP positions')
        producers[rank] = (key, local)
        sizes[key] = size
        groups[key][local] = rank
    result = []
    for key, ranks in groups.items():
        if set(ranks) != set(range(sizes[key])):
            raise ValueError('Incomplete capture topology')
        result.append(tuple(ranks[i] for i in range(sizes[key])))
    return tuple(result)


def _vocab_widths(topology_rows):
    widths = {}
    for row in topology_rows:
        if 'vocab_partition_size' not in row:
            raise ValueError('Vocabulary reconstruction needs vocab_partition_size metadata')
        rank, width = row['producer_rank'], row['vocab_partition_size']
        if width < 0 or (rank in widths and widths[rank] != width):
            raise ValueError('Conflicting or invalid vocabulary partition size')
        widths[rank] = width
    return widths


def _ordered_tp_rows(shards, topology_rows, *, vocabulary=False):
    by_rank = {key[0]: row for key, row in shards.items()}
    groups = [group for group in _tp_groups(topology_rows) if next(iter(by_rank)) in group]
    if len(groups) != 1:
        raise ValueError('Producer must belong to exactly one TP group')
    group = groups[0]
    expected = group
    if vocabulary and any('vocab_partition_size' in row for row in topology_rows):
        widths = _vocab_widths(topology_rows)
        expected = tuple(rank for rank in group if widths[rank] > 0)
        # Either TP shards on every rank, or full replicated output on TP rank 0.
        if expected not in (group, group[:1]):
            raise ValueError('Invalid vocabulary capture placement')
    elif vocabulary and len(by_rank) == 1 and next(iter(by_rank.values()))['shard_rank'] == -1:
        expected = group[:1]  # Legacy explicit replicated marker.
    if set(expected) != set(by_rank):
        raise ValueError('Shards do not cover exactly one complete TP group or recorded vocabulary placement')
    return [by_rank[rank] for rank in expected]


def merge_token_shards(payload_rows):
    """Merge [token, feature] intervals: hidden states, MoE inputs, router logits."""
    import torch
    result = []
    for key, chunks in _group_rows(payload_rows, ROW_KEYS, ('token_start', 'token_end')).items():
        intervals = _token_intervals(chunks)
        values = [chunks[interval]['value'] for interval in intervals]
        _compatible_tensors(values)
        for (start, end), value in zip(intervals, values):
            if value.shape[0] != end - start or value.shape[1] != values[0].shape[1]:
                raise ValueError('Sequence shard shape disagrees with token interval or feature width')
        result.append(Row(zip(ROW_KEYS, key), token_start=intervals[0][0],
                          token_end=intervals[-1][1], value=torch.cat(values, dim=0)))
    return (result,)


def merge_vocab_shards(payload_rows, topology_rows):
    """Merge [token, vocabulary] partitions in TP order; retain captured padding."""
    import torch
    result = []
    for key, shards in _group_rows(payload_rows, INTERVAL_KEYS, ('producer_rank',)).items():
        topology = _run_topology(topology_rows, key[0])
        rows = _ordered_tp_rows(shards, topology, vocabulary=True)
        values = [row['value'] for row in rows]
        _compatible_tensors(values)
        if any(value.shape[0] != values[0].shape[0] for value in values):
            raise ValueError('Vocabulary shards disagree on sequence extent')
        if any('vocab_partition_size' in row for row in topology):
            widths = _vocab_widths(topology)
            if any(row['value'].shape[1] != widths[row['producer_rank']] for row in rows):
                raise ValueError('Vocabulary payload disagrees with captured partition size')
        result.append(Row(zip(INTERVAL_KEYS, key), value=torch.cat(values, dim=1)))
    return (result,)


def merge_projection_shards(payload_rows, topology_rows):
    """Merge Q/K [projection output, hidden input] matrices in TP order."""
    import torch
    result = []
    for key, shards in _group_rows(payload_rows, INTERVAL_KEYS, ('producer_rank',)).items():
        rows = _ordered_tp_rows(shards, _run_topology(topology_rows, key[0]))
        values = [row['value'] for row in rows]
        _compatible_tensors(values, equal_shape=True)
        if any(row['shard_rank'] != tp for tp, row in enumerate(rows)):
            raise ValueError('Projection shard rank disagrees with TP topology')
        result.append(Row(zip(INTERVAL_KEYS, key), value=torch.cat(values, dim=0)))
    return (result,)


def merge_routing_shards(payload_rows, topology_rows):
    """Merge selected expert IDs or weights [token, top_k], cropping padding.

    Unlike SEQ_PREFIX_PACK hooks, these IDENTITY captures have a full-sample
    token interval on every TP rank and retain the local sequence padding.
    """
    import torch
    result = []
    for key, shards in _group_rows(payload_rows, INTERVAL_KEYS, ('producer_rank',)).items():
        rows = _ordered_tp_rows(shards, _run_topology(topology_rows, key[0]))
        values = [row['value'] for row in rows]
        _compatible_tensors(values, equal_shape=True)
        start, end = key[-2:]
        value = torch.cat(values, dim=0)
        if end <= start or end - start > value.shape[0]:
            raise ValueError('Routing token range disagrees with payload')
        result.append(Row(zip(INTERVAL_KEYS, key), value=value[:end - start]))
    return (result,)


def _merge_token_reductions(payload_rows, *, mean):
    import torch
    result = []
    for key, chunks in _group_rows(payload_rows, ROW_KEYS, ('token_start', 'token_end')).items():
        intervals = _token_intervals(chunks)
        values = [chunks[interval]['value'] for interval in intervals]
        template = values[0]
        if isinstance(template, torch.Tensor):
            if any(not isinstance(v, torch.Tensor) or v.shape != template.shape or v.dtype != template.dtype for v in values):
                raise ValueError('Reduction shards disagree on shape or dtype')
        elif any(type(v) is not type(template) for v in values):
            raise ValueError('Reduction shards disagree on value type')
        if mean:
            total = sum(end - start for start, end in intervals)
            value = sum(v * ((end - start) / total) for v, (start, end) in zip(values, intervals))
        else:
            value = sum(values)
        result.append(Row(zip(ROW_KEYS, key), token_start=intervals[0][0],
                          token_end=intervals[-1][1], value=value))
    return (result,)


def merge_token_means(payload_rows):
    """Weight per-shard means by token_end-token_start, separately per sample."""
    return _merge_token_reductions(payload_rows, mean=True)


def sum_token_counts(payload_rows):
    """Sum disjoint token-interval counts, separately per sample and output."""
    return _merge_token_reductions(payload_rows, mean=False)


def merge_vocab_topk(value_rows, index_rows, topology_rows):
    """Merge local top-K into global top-K values and global vocabulary IDs.

    The captured K determines output K. Each pair of inputs must select one
    matching hook-output pair. Ties among captured candidates prefer lower IDs.
    """
    import torch
    keys = tuple(name for name in INTERVAL_KEYS if name != 'act_name')
    if len({row['act_name'] for row in value_rows}) > 1 or len({row['act_name'] for row in index_rows}) > 1:
        raise ValueError('Top-K inputs must each select one hook output')
    values = _group_rows(value_rows, keys, ('producer_rank',))
    indices = _group_rows(index_rows, keys, ('producer_rank',))
    if values.keys() != indices.keys():
        raise ValueError('Top-K value/index sample identities disagree')
    result = []
    for key, shards in values.items():
        topology = _run_topology(topology_rows, key[0])
        widths = _vocab_widths(topology)
        rows = _ordered_tp_rows(shards, topology, vocabulary=True)
        index_shards = indices[key]
        if index_shards.keys() != shards.keys():
            raise ValueError('Top-K value/index producers disagree')
        _compatible_tensors([row['value'] for row in rows], equal_shape=True)
        k = rows[0]['value'].shape[-1]
        if k <= 0:
            raise ValueError('Top-K payload requires K > 0')
        all_values, all_ids, offset = [], [], 0
        for row in rows:
            index = index_shards[(row['producer_rank'],)]
            ids, scores = index['value'], row['value']
            width = widths[row['producer_rank']]
            if not isinstance(ids, torch.Tensor) or ids.shape != scores.shape or ids.dtype not in (torch.int32, torch.int64):
                raise ValueError('Top-K indices must be integer tensors matching value shapes')
            if k > width or (ids < 0).any() or (ids >= width).any():
                raise ValueError('Top-K local vocabulary index out of bounds')
            if row['shard_rank'] != index['shard_rank']:
                raise ValueError('Top-K value/index shard identities disagree')
            sorted_ids = ids.sort(dim=-1).values
            if (sorted_ids[..., 1:] == sorted_ids[..., :-1]).any():
                raise ValueError('Top-K local indices must be unique per token')
            all_values.append(scores)
            all_ids.append(ids.to(torch.int64) + offset)
            offset += width
        scores, ids = torch.cat(all_values, dim=-1), torch.cat(all_ids, dim=-1)
        order = ids.argsort(dim=-1, stable=True)
        scores, ids = scores.gather(-1, order), ids.gather(-1, order)
        order = scores.argsort(dim=-1, descending=True, stable=True)[..., :k]
        result.append(Row(zip(keys, key), values=scores.gather(-1, order), indices=ids.gather(-1, order)))
    return (result,)


def merge_weight_shards(payload_rows, topology_rows):
    """Merge physical/logical rank fragments into full Q/K/router matrices.

    Layout is published once per producer in capture_topology.weight_layout_json.
    Repeated delivery is deduplicated; different attempts are never combined.
    """
    import torch
    from math import prod
    from ..hooks.weight_capture import WEIGHT_NAMES
    layouts = {}
    for row in topology_rows:
        document = json.loads(row.get('weight_layout_json') or '[]')
        for layout in document:
            if int(layout['producer_rank']) != int(row['producer_rank']):
                raise ValueError('Weight layout producer disagrees with topology')
            key = (row['model_id'], layout['layer_no'], layout['act_name'])
            ranks = layouts.setdefault(key, {})
            rank = layout['producer_rank']
            if rank in ranks and ranks[rank] != layout:
                raise ValueError('Conflicting weight layout metadata')
            ranks[rank] = layout
    keys = ('model_id', 'phase', 'global_batch_id', 'attempt_id', 'layer_no', 'act_name')
    grouped = {}
    for row in payload_rows:
        if row['act_name'] not in WEIGHT_NAMES:
            raise ValueError('Not a parameter-weight output')
        if row.get('direction', 'iter') != 'iter' or row.get('microbatch_id', -1) != -1:
            raise ValueError('Weight merge requires per-iteration records')
        key = tuple(row[k] for k in keys)
        ranks = grouped.setdefault(key, {})
        rank = row['producer_rank']
        old = ranks.get(rank)
        if old is not None:
            if (old.get('invocation_id') != row.get('invocation_id')
                    or not _same_value(old['value'], row['value'])):
                raise ValueError('Conflicting duplicate weight producer record')
        ranks[rank] = row
    result = []
    for key, ranks in sorted(grouped.items()):
        specs = layouts.get((key[0], key[4], key[5]), {})
        if not specs or set(ranks) != set(specs):
            raise ValueError('Missing or unexpected weight producer ranks')
        formats = {(tuple(s['shape']), s['dtype']) for s in specs.values()}
        if len(formats) != 1:
            raise ValueError('Weight layout shape/dtype disagreement')
        shape, dtype_name = formats.pop()
        dtype = getattr(torch, dtype_name, None)
        if dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            raise ValueError('Invalid model-weight dtype')
        total = prod(shape) * torch.empty((), dtype=dtype).element_size()
        pieces = []
        for rank, row in ranks.items():
            value = row['value']
            if value.device.type != 'cpu' or value.dtype != torch.uint8 or value.ndim != 1:
                raise ValueError('Weight fragment must be a flat CPU byte tensor')
            packed = 0
            for source, output, length in specs[rank]['fragments']:
                if min(source, output, length) < 0 or output + length > total:
                    raise ValueError('Invalid weight fragment bounds')
                if length:
                    pieces.append((output, length, value[packed:packed + length]))
                packed += length
            if packed != value.numel():
                raise ValueError('Weight payload length differs from layout')
        cursor = 0
        for offset, length, value in sorted(pieces, key=lambda p: p[0]):
            if offset != cursor or value.numel() != length:
                raise ValueError('Weight fragment coverage has gaps or overlaps')
            cursor += length
        if cursor != total:
            raise ValueError('Incomplete weight matrix')
        merged = torch.empty(total, dtype=torch.uint8)
        for offset, length, value in pieces:
            merged[offset:offset + length].copy_(value)
        result.append(Row(zip(keys, key), value=merged.view(dtype).reshape(shape)))
    return (result,)
