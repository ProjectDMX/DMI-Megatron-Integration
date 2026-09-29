"""CPU oracles and on-demand ClickHouse execution for shared merge transforms."""
import itertools

import pytest
import torch

from dmi_megatron_integration.hooks.megatron_qk_weights import qk_weight_from_fused_qkv
from dmi_megatron_integration.hooks.megatron_router_summary import (
    router_probs_mean_from_logits, router_token_entropy_mean_from_logits,
    expert_token_count_from_routing_map,
)
from dmi_megatron_integration.hooks.megatron_vocab_logits import vocab_logits_topk_by_sample
from dmi_megatron_integration.materialization import reconstruction as merge
from dmi_megatron_integration.signals import Row
from tests.test_signals import storage


def topology():
    return [Row(model_id='r', producer_rank=pp*4+dp*2+tp, tp_rank=tp,
                tp_world_size=2, pp_rank=pp, dp_rank=dp, cp_rank=0,
                vocab_partition_size=12)
            for pp, dp, tp in itertools.product(range(2), repeat=3)][::-1]


def payload(rank, name, value, **overrides):
    row = Row(model_id='r', phase='valid', global_batch_id=3, attempt_id=0,
              microbatch_id=0, layer_no=rank//4, direction='fwd', dp_rank=rank%4//2,
              dataset_id=0, sample_index=0, invocation_id=0, act_name=name,
              token_start=0, token_end=5, producer_rank=rank, shard_rank=rank%2, value=value)
    row.update(overrides)
    return row


def test_token_shards_keep_outputs_pp_dp_sample_attempt_invocation_separate():
    rows, expected = [], {}
    for pp, dp, sample, attempt, invocation, name in itertools.product(
        range(2), range(2), range(2), range(2), range(2),
        ['hidden_states', 'hook_resid_final', 'moe_input', 'router_logits'],
    ):
        key = (pp, dp, sample, attempt, invocation, name)
        full = torch.arange(20).reshape(5, 4) + len(expected)*100
        expected[key] = full
        for tp, (start, end) in enumerate(((0, 3), (3, 5))):
            rows.append(payload(pp*4+dp*2+tp, name, full[start:end],
                                token_start=start, token_end=end, sample_index=sample,
                                attempt_id=attempt, invocation_id=invocation))
    result, = merge.merge_token_shards(rows[::-1] + [rows[0]])
    assert len(result) == len(expected)
    for row in result:
        key = (row.layer_no, row.dp_rank, row.sample_index, row.attempt_id, row.invocation_id, row.act_name)
        torch.testing.assert_close(row.value, expected[key])
        assert (row.token_start, row.token_end) == (0, 5)
        assert 'producer_rank' not in row and 'shard_rank' not in row


def test_token_shards_validate_duplicates_intervals_and_shapes():
    first = payload(0, 'hidden_states', torch.ones(3, 2), token_end=3)
    with pytest.raises(ValueError, match='Conflicting'):
        merge.merge_token_shards([first, Row(first, value=first.value+1)])
    with pytest.raises(ValueError, match='gaps'):
        merge.merge_token_shards([first, Row(first, token_start=4, token_end=5, value=torch.ones(1, 2))])
    with pytest.raises(ValueError, match='shape'):
        merge.merge_token_shards([first, Row(first, token_start=3, token_end=5, value=torch.ones(2, 3))])
    # A single complete record (replicated capture or TP=1) needs no concatenation.
    result, = merge.merge_token_shards([first])
    torch.testing.assert_close(result[0].value, first.value)


def test_projection_merging_matches_full_gqa_q_and_k_on_each_pp_stage():
    rows, expected = [], {}
    for pp in range(2):
        fused = torch.arange(4*8*8, dtype=torch.float32).reshape(32, 8) + pp*1000
        for projection, name in [('q', 'query_projection_weight'), ('k', 'key_projection_weight')]:
            expected[(pp, name)] = qk_weight_from_fused_qkv(
                fused, num_query_groups=4, query_rows_per_group=4, head_dim=2, projection=projection)
            for tp in range(2):
                local = qk_weight_from_fused_qkv(fused[tp*16:(tp+1)*16],
                    num_query_groups=2, query_rows_per_group=4, head_dim=2, projection=projection)
                rows.append(payload(pp*4+tp, name, local, dp_rank=-1,
                    sample_index=-1, dataset_id=-1, direction='iter', token_end=1))
    result, = merge.merge_projection_shards(rows[::-1]+[rows[0]], topology())
    assert len(result) == 4
    for row in result:
        torch.testing.assert_close(row.value, expected[(row.layer_no, row.act_name)])
    with pytest.raises(ValueError, match='complete TP'):
        merge.merge_projection_shards(rows[1:], topology())
    with pytest.raises(ValueError, match='Conflicting'):
        merge.merge_projection_shards(rows+[Row(rows[0], value=rows[0].value+1)], topology())


def test_token_means_and_counts_match_full_computation_with_unequal_intervals():
    torch.manual_seed(11)
    logits = torch.randn(6, 2, 4)
    valid = torch.tensor([5, 2])  # Sample 1 emits no row on TP rank 1.
    routing = torch.nn.functional.one_hot(logits.argmax(-1), 4).bool()
    oracle = {
        'router_probs_mean': router_probs_mean_from_logits(logits, valid, 'softmax'),
        'router_token_entropy_mean': router_token_entropy_mean_from_logits(logits, valid, 'softmax'),
        'pre_drop_token_count': expert_token_count_from_routing_map(routing.reshape(12, 4), valid, seq_length=6, batch_size=2),
    }
    rows = []
    for tp in range(2):
        counts = (valid-tp*3).clamp(0, 3)
        local = logits[tp*3:(tp+1)*3]
        values = {
            'router_probs_mean': router_probs_mean_from_logits(local, counts, 'softmax'),
            'router_token_entropy_mean': router_token_entropy_mean_from_logits(local, counts, 'softmax'),
            'pre_drop_token_count': expert_token_count_from_routing_map(
                routing[tp*3:(tp+1)*3].reshape(6, 4), counts, seq_length=3, batch_size=2),
        }
        for name, value in values.items():
            for sample in range(2):
                if counts[sample] == 0:
                    continue
                v = value[sample].item() if name == 'router_token_entropy_mean' else value[sample]
                rows.append(payload(tp, name, v, sample_index=sample,
                    token_start=tp*3, token_end=tp*3+int(counts[sample])))
    means = [row for row in rows if row.act_name != 'pre_drop_token_count']
    counts = [row for row in rows if row.act_name == 'pre_drop_token_count']
    counts += [Row(row, act_name='post_drop_token_count') for row in counts]
    oracle['post_drop_token_count'] = oracle['pre_drop_token_count']
    result = merge.merge_token_means(means[::-1]+[means[0]])[0] + merge.sum_token_counts(counts)[0]
    assert len(result) == 8
    for row in result:
        expected = oracle[row.act_name][row.sample_index]
        if row.act_name == 'router_token_entropy_mean':
            assert row.value == pytest.approx(expected.item())
        else:
            torch.testing.assert_close(row.value, expected)
        assert row.token_end == int(valid[row.sample_index])


def test_vocab_topk_and_raw_match_full_vocab_on_two_dp_replicas():
    torch.manual_seed(12)
    values, indices, raw, expected, full_by_dp = [], [], [], {}, {}
    for dp in range(2):
        full = torch.randn(5, 2, 24)
        full_by_dp[dp], expected[dp] = full, full.topk(3, dim=-1)
        for tp in range(2):
            shard = full[..., tp*12:(tp+1)*12]
            v, ids = vocab_logits_topk_by_sample(shard, k=3)
            for sample in range(2):
                kwargs = dict(sample_index=sample, token_end=1)  # need_token_range=False
                rank = 4+dp*2+tp
                values.append(payload(rank, 'vocab_logits_topk_values', v[sample], **kwargs))
                indices.append(payload(rank, 'vocab_logits_topk_indices', ids[sample], **kwargs))
                raw.append(payload(rank, 'vocab_logits', shard[:, sample], **kwargs))
    result, = merge.merge_vocab_topk(values[::-1], indices[::-1], topology())
    assert len(result) == 4
    for row in result:
        torch.testing.assert_close(row['values'], expected[row.dp_rank].values[:, row.sample_index])
        torch.testing.assert_close(row.indices, expected[row.dp_rank].indices[:, row.sample_index])
    for row in merge.merge_vocab_shards(raw, topology())[0]:
        torch.testing.assert_close(row.value, full_by_dp[row.dp_rank][:, row.sample_index])
    with pytest.raises(ValueError, match='complete TP'):
        merge.merge_vocab_topk(values[1:], indices, topology())
    with pytest.raises(ValueError, match='vocab_partition_size'):
        merge.merge_vocab_topk(values, indices, [Row({k:v for k,v in r.items() if k!='vocab_partition_size'}) for r in topology()])


def test_vocab_replicated_capture_with_real_shard_rank_zero_and_topk_ties():
    top = [Row(row, vocab_partition_size=24 if row.tp_rank==0 else 0) for row in topology()]
    v = payload(4, 'vocab_logits_topk_values', torch.tensor([[2., 2.]]), shard_rank=0)
    ids = Row(v, act_name='vocab_logits_topk_indices', value=torch.tensor([[7, 3]], dtype=torch.int32))
    result, = merge.merge_vocab_topk([v], [ids], top)
    assert result[0].indices.tolist() == [[3, 7]]
    raw = Row(v, act_name='vocab_logits', value=torch.ones(2, 24))
    torch.testing.assert_close(merge.merge_vocab_shards([raw], top)[0][0].value, raw.value)
    with pytest.raises(ValueError, match='out of bounds'):
        merge.merge_vocab_topk([v], [Row(ids, value=torch.tensor([[24, 1]]))], top)
    with pytest.raises(ValueError, match='unique'):
        merge.merge_vocab_topk([v], [Row(ids, value=torch.tensor([[3, 3]]))], top)


def test_routing_ids_and_weights_merge_separately_and_crop_padding():
    rows = []
    for dp, tp in itertools.product(range(2), repeat=2):
        ids = torch.arange(tp*6, tp*6+6).reshape(3, 2)
        rank = dp*2+tp
        rows += [payload(rank, 'router_topk_expert_ids', ids, shard_rank=rank),
                 payload(rank, 'router_topk_weights', ids.float()/10, shard_rank=rank)]
    result, = merge.merge_routing_shards(rows[::-1], topology())
    assert len(result) == 4
    for row in result:
        expected = torch.arange(10).reshape(5, 2)
        if row.act_name == 'router_topk_weights':
            expected = expected.float()/10
        torch.testing.assert_close(row.value, expected)


def identity(rows):
    return (rows,)


@pytest.mark.parametrize('name', ['merge_token_shards', 'merge_vocab_shards', 'merge_projection_shards',
    'merge_routing_shards', 'merge_token_means', 'sum_token_counts', 'merge_vocab_topk'])
def test_shared_merge_as_on_demand_signal_in_clickhouse(storage, name):
    from dmi_megatron_integration.signals import parse_config, SignalWorker
    from dmi_megatron_integration.signals.definition import Output
    from dmi_megatron_integration.signals.events import Event
    if name == 'merge_vocab_topk':
        rows = [payload(tp, 'vocab_logits_topk_values', torch.tensor([[float(tp), float(tp+2)]])) for tp in range(2)]
        ids = [Row(row, act_name='vocab_logits_topk_indices', value=torch.tensor([[1, 3]], dtype=torch.int32)) for row in rows]
        arguments = [rows, ids, topology()]
    elif name == 'merge_vocab_shards':
        arguments = [[payload(tp, 'vocab_logits', torch.full((5, 12), float(tp))) for tp in range(2)], topology()]
    elif name == 'merge_projection_shards':
        arguments = [[payload(tp, 'query_projection_weight', torch.full((2, 4), float(tp)), dp_rank=-1) for tp in range(2)], topology()]
    elif name == 'merge_routing_shards':
        arguments = [[payload(tp, 'router_topk_expert_ids', torch.tensor([[0, 1], [2, 3], [1, 2]])) for tp in range(2)], topology()]
    elif name in ('merge_token_means', 'sum_token_counts'):
        # Include scalar entropy and tensor means in the CPU test above; DB path
        # uses scalar means here and tensor integer counts for the count helper.
        arguments = [[payload(tp, 'router_token_entropy_mean' if name=='merge_token_means' else 'pre_drop_token_count',
            float(tp) if name=='merge_token_means' else torch.tensor([tp+1, 2]),
            token_start=tp*3, token_end=min(5, (tp+1)*3)) for tp in range(2)]]
    else:
        arguments = [[payload(tp, 'router_logits', torch.full((min(3, 5-tp*3), 4), float(tp)),
            token_start=tp*3, token_end=min(5, (tp+1)*3)) for tp in range(2)]]
    def columns(row):
        return tuple((k, 'Tensor' if isinstance(v, torch.Tensor) else 'String' if isinstance(v, str)
            else 'Float64' if isinstance(v, float) else 'Int64') for k, v in row.items())
    inputs = []
    for i, argument in enumerate(arguments):
        table = f'input_{i}'
        storage.write(Output('table', table, columns(argument[0])), argument)
        inputs.append({'from': {'table': table}})
    expected, = getattr(merge, name)(*arguments)
    schema = [{'name': key, 'type': dtype} for key, dtype in columns(expected[0])]
    producer = dict(name='reconstruct', trigger={'event':'on_demand'}, inputs=inputs,
        transform=f'builtin:{name}', output=[{'signal':'reconstructed', 'columns':schema}])
    root = dict(name='consume', trigger={'event':'iteration_end'}, inputs=[{'from':{'signal':'reconstructed'}}],
        transform='tests.test_signal_shared_reconstruction:identity', output=[{'table':'result', 'columns':schema}])
    actual, = SignalWorker(parse_config({'signals':[root, producer]}), storage).execute('consume', Event('r','iteration_end','valid',5,3))
    assert len(actual) == len(expected)
    for key, value in expected[0].items():
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(actual[0][key], value)
        else:
            assert actual[0][key] == value
    assert storage.client.execute('EXISTS TABLE '+storage.table('reconstructed')) == [(0,)]
