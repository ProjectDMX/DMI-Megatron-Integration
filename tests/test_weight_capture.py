"""Weight ownership/reconstruction and retained-storage contracts (no training)."""
import json
from types import SimpleNamespace
import pytest
import torch
from torch import nn
from dmi_megatron_integration.hooks.weight_capture import (
    WeightSource, WeightCapture, qk_projection_ranges, intersect_maps,
    assign_weight_fragments, resolve_weight_source,
)
from dmi_megatron_integration.materialization.reconstruction import merge_weight_shards
from dmi_megatron_integration.signals.config import load_config


def make_case(tp=2, fsdp=3, replicas=2, dtype=torch.bfloat16):
    heads, groups, dim, hidden = 8, 2, 2, 7
    fused = torch.arange(groups * (heads // groups + 2) * dim * hidden).reshape(-1, hidden).to(dtype)
    router = torch.arange(5 * hidden).reshape(5, hidden).to(dtype)
    q = fused.view(groups, -1, hidden)[:, :heads // groups * dim].reshape(-1, hidden)
    k = fused.view(groups, -1, hidden)[:, heads // groups * dim:heads // groups * dim + dim].reshape(-1, hidden)
    captures = []
    for t in range(tp):
        local = fused.chunk(tp)[t].contiguous()
        for d in range(fsdp):
            for replica in range(replicas):
                rank = (t * fsdp + d) * replicas + replica
                for short, name, full in [('q', 'query_projection_weight', local),
                                          ('k', 'key_projection_weight', local),
                                          ('router', 'router_projection_weight', router)]:
                    if short == 'router':
                        shape, param_shape = router.shape, router.shape
                        mapping = [(0, 0, router.numel() * router.element_size())]
                    else:
                        shape, param_shape, mapping = qk_projection_ranges(
                            heads=heads, groups=groups, head_dim=dim, hidden=hidden,
                            tp_rank=t, tp_size=tp, element_size=full.element_size(), projection=short)
                    start, end = full.numel() * d // fsdp, full.numel() * (d + 1) // fsdp
                    owned = full.flatten()[start:end].clone()
                    storage = [(0, start * full.element_size(), owned.numel() * full.element_size())]
                    source = WeightSource(lambda v=owned: v, param_shape, dtype, storage, 'test_fsdp')
                    captures.append(WeightCapture(0, name, source, shape,
                                                  intersect_maps(storage, mapping), rank))
    return captures, {'query_projection_weight': q, 'key_projection_weight': k,
                      'router_projection_weight': router}


def materialize(captures, *, attempt=1, iteration=3):
    layouts = assign_weight_fragments([c.report() for c in captures])
    index = {(r['producer_rank'], r['act_name']): r for r in layouts}
    payload, topology = [], []
    for c in captures:
        layout = index[c.producer_rank, c.act_name]
        c.assigned = layout['fragments']
        payload.append(dict(model_id='run', phase='train', global_batch_id=iteration,
                            attempt_id=attempt, layer_no=c.layer_no, act_name=c.act_name,
                            producer_rank=c.producer_rank, invocation_id=0, value=c.pack()))
        topology.append(dict(model_id='run', producer_rank=c.producer_rank,
                             weight_layout_json=json.dumps([layout])))
    return payload, topology


@pytest.mark.parametrize('tp,fsdp,replicas', [(1,1,1), (1,1,5), (2,3,2), (8,3,2), (4,1,3)])
@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_all_weight_types_merge_all_ranks_exactly(tp, fsdp, replicas, dtype):
    captures, expected = make_case(tp, fsdp, replicas, dtype)
    rows, topology = materialize(captures)
    result, = merge_weight_shards(rows[::-1] + [rows[0]], topology)
    assert len(result) == 3
    assert sum(r['value'].numel() for r in rows) == sum(v.numel()*v.element_size() for v in expected.values())
    for row in result:
        torch.testing.assert_close(row.value, expected[row.act_name], rtol=0, atol=0)
        assert 'producer_rank' not in row and 'dp_rank' not in row
    for c in captures:
        c.source.get().add_(10)
    for row in merge_weight_shards(rows, topology)[0]:
        torch.testing.assert_close(row.value, expected[row.act_name], rtol=0, atol=0)


def test_empty_k_contributions_are_required_and_attempts_do_not_mix():
    captures, _ = make_case(tp=8, fsdp=1, replicas=1)
    rows, topology = materialize(captures)
    empty = next(r for r in rows if r['act_name']=='key_projection_weight' and not r['value'].numel())
    with pytest.raises(ValueError, match='producer ranks'):
        merge_weight_shards([r for r in rows if r is not empty], topology)
    rows2, _ = materialize(captures, attempt=2)
    assert len(merge_weight_shards(rows+rows2, topology)[0]) == 6
    with pytest.raises(ValueError, match='producer ranks'):
        merge_weight_shards(rows[:-1]+rows2[-1:], topology)


def test_reject_corruption_missing_metadata_and_duplicate_invocations():
    captures, _ = make_case()
    rows, topology = materialize(captures)
    with pytest.raises(ValueError, match='producer ranks'):
        merge_weight_shards(rows, [])
    for bad in [dict(rows[0], value=rows[0]['value']+1), dict(rows[0], invocation_id=1)]:
        with pytest.raises(ValueError, match='Conflicting'):
            merge_weight_shards(rows+[bad], topology)
    altered = json.loads(topology[0]['weight_layout_json'])
    altered[0]['fragments'][0][1] += 1
    with pytest.raises(ValueError, match='coverage|bounds'):
        merge_weight_shards(rows, [dict(topology[0], weight_layout_json=json.dumps(altered))]+topology[1:])


@pytest.mark.parametrize('hybrid', [False, True])
def test_megatron_fsdp_retained_compute_shard_not_freed_parameter_or_master(hybrid):
    owner = nn.Linear(7, 6, bias=False)
    parameter = owner.weight
    retained = torch.arange(11, dtype=torch.float32)
    fake = SimpleNamespace(dtype=torch.float32, param_idx={parameter:0},
                           item_index_map={0:SimpleNamespace(size=42)},
                           get_item=lambda _: retained,
                           locate_item_in_global_item=lambda _: (5,16))
    group = SimpleNamespace(model_weight_buffer=None if hybrid else fake,
                            hfsdp_helper_wbuf=fake if hybrid else None,
                            main_weight_buffer=object())
    wrapper = nn.Module()
    wrapper.module = owner
    wrapper.param_and_grad_buffer = SimpleNamespace(param_to_param_group={parameter:0}, parameter_groups=[group])
    source = resolve_weight_source(owner, 'weight', model_roots=[wrapper], expected_shape=(6,7))
    parameter.data = torch.empty(0)
    torch.testing.assert_close(source.read_bytes(), retained.view(torch.uint8))
    assert source.ranges == [(0,20,44)]
    retained.add_(1)
    torch.testing.assert_close(source.read_bytes(), retained.view(torch.uint8))


@pytest.mark.parametrize('hybrid', [False, True])
def test_megatron_fsdp_optimizer_proxy_resolves_compute_dtype_and_qk_offsets(hybrid):
    from tests.test_megatron_qk_weights import _attention, _rank
    from dmi_megatron_integration.hooks.weight_capture import discover_weight_captures
    attention, q, k = _attention()
    original = attention.linear_qkv.weight
    original.data = original.data.to(torch.bfloat16)
    retained = original.detach().flatten().clone()
    proxy = nn.Parameter(original.float().clone())
    proxy.orig_param = original
    proxy.__fsdp_param__ = True
    attention.linear_qkv.weight = proxy
    buffer = SimpleNamespace(dtype=torch.bfloat16, param_idx={original: 0},
        item_index_map={0: SimpleNamespace(size=retained.numel())},
        get_item=lambda _: retained,
        locate_item_in_global_item=lambda _: (0, retained.numel()))
    wrapper = nn.Module()
    wrapper.module = attention
    wrapper.param_and_grad_buffer = SimpleNamespace(param_to_param_group={original: 0},
        parameter_groups=[SimpleNamespace(model_weight_buffer=None if hybrid else buffer,
            hfsdp_helper_wbuf=buffer if hybrid else None)])
    captures = discover_weight_captures([wrapper], _rank(), {'q-weights', 'k-weights'})
    # Neither the released gathered parameter nor the optimizer proxy is read.
    original.data = torch.empty(0, dtype=torch.bfloat16)
    proxy.data.zero_()
    rows, topology = materialize(captures)
    expected = {'query_projection_weight': q, 'key_projection_weight': k}
    for row in merge_weight_shards(rows, topology)[0]:
        assert row.value.dtype == torch.bfloat16
        torch.testing.assert_close(row.value, expected[row.act_name].to(torch.bfloat16))


def test_non_fsdp_refreshes_storage_and_exact_byte_replica_dedup():
    owner = nn.Linear(3, 5, bias=False)
    source = resolve_weight_source(owner, 'weight', model_roots=[owner], expected_shape=(5,3))
    captures = [WeightCapture(0,'router_projection_weight',source,(5,3),source.ranges,i) for i in range(7)]
    rows, topology = materialize(captures)
    assert sum(r['value'].numel() for r in rows) == 60
    assert any(r['value'].numel() % 4 for r in rows)
    torch.testing.assert_close(merge_weight_shards(rows,topology)[0][0].value, owner.weight)
    owner.weight.data = torch.full((5,3),9.)
    rows, topology = materialize(captures)
    torch.testing.assert_close(merge_weight_shards(rows,topology)[0][0].value, owner.weight)


def test_example_resolves_one_shared_weight_signal():
    from pathlib import Path
    registry = load_config(Path(__file__).resolve().parents[1]/'examples/signals/reconstruction.yaml')
    signal = registry.signals['weight_matrices']
    assert signal.transform is merge_weight_shards
    assert signal.outputs[0].name == 'complete_weight_matrices'
    assert set(signal.inputs[0]['where']['act_name']['in']) == {
        'query_projection_weight', 'key_projection_weight', 'router_projection_weight'}

from tests.test_signals import storage


def identity(rows):
    return (rows,)


def test_weight_signal_clickhouse_virtual_output(storage):
    from pathlib import Path
    import yaml
    from dmi_megatron_integration.signals import parse_config, SignalWorker
    from dmi_megatron_integration.signals.definition import Output
    from dmi_megatron_integration.signals.events import Event
    captures, expected = make_case(tp=8, fsdp=3, replicas=2)
    rows, topology = materialize(captures)
    def columns(row):
        return tuple((key, 'Tensor' if isinstance(value,torch.Tensor) else
                      'String' if isinstance(value,str) else 'Int64') for key,value in row.items())
    storage.write(Output('table','dmi_training_tensors',columns(rows[0])), rows)
    storage.write(Output('table','dmi_training_tensors_capture_topology',columns(topology[0])),topology)
    document = yaml.safe_load((Path(__file__).resolve().parents[1]/'examples/signals/reconstruction.yaml').read_text())
    helper = next(s for s in document['signals'] if s['name']=='weight_matrices')
    root = dict(name='analyze',trigger={'event':'iteration_end'},
                inputs=[{'from':{'signal':'complete_weight_matrices'}}],
                transform='tests.test_weight_capture:identity',
                output=[dict(table='analysis',columns=helper['output'][0]['columns'])])
    result, = SignalWorker(parse_config({'signals':[helper,root]}), storage).execute(
        'analyze', Event('run','iteration_end','train',3,3,attempt_id=1))
    assert len(result)==3
    for row in result:
        torch.testing.assert_close(row.value,expected[row.act_name],rtol=0,atol=0)
    assert storage.client.execute('EXISTS TABLE '+storage.table('complete_weight_matrices')) == [(0,)]


def _distributed_source_worker(rank, path):
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import distribute_tensor, Shard
    dist.init_process_group('gloo',init_method='file://'+path,rank=rank,world_size=2)
    try:
        mesh=init_device_mesh('cpu',(2,))
        full=torch.arange(35.).reshape(5,7)
        for dim in (0,1):
            owner=nn.Linear(7,5,bias=False)
            owner.weight=nn.Parameter(distribute_tensor(full,mesh,[Shard(dim)]))
            source=resolve_weight_source(owner,'weight',model_roots=[owner],expected_shape=(5,7))
            capture=WeightCapture(0,'router_projection_weight',source,(5,7),source.ranges,rank)
            reports=[None]*2
            dist.all_gather_object(reports,capture.report())
            layout=next(r for r in assign_weight_fragments(reports) if r['producer_rank']==rank)
            capture.assigned=layout['fragments']
            row=dict(model_id='run',phase='train',global_batch_id=3,attempt_id=0,layer_no=0,
                     act_name=capture.act_name,producer_rank=rank,value=capture.pack())
            rows,topology=[None]*2,[None]*2
            dist.all_gather_object(rows,row)
            dist.all_gather_object(topology,dict(model_id='run',producer_rank=rank,weight_layout_json=json.dumps([layout])))
            torch.testing.assert_close(merge_weight_shards(rows,topology)[0][0].value,full)
        # Exercise the real Megatron-FSDP flattened bucket/index implementation,
        # including its padding and released temporary parameter storage.
        from megatron.core.distributed.fsdp.src.megatron_fsdp.param_and_grad_buffer import DataParallelBuffer
        from megatron.core.distributed.fsdp.src.megatron_fsdp.distributed_data_parallel_config import DistributedDataParallelConfig
        for rows_count in (5, 513):
            full=torch.arange(rows_count*7.).reshape(rows_count,7)
            owner=nn.Linear(7,rows_count,bias=False)
            parameter=owner.weight
            buffer=DataParallelBuffer(DistributedDataParallelConfig(data_parallel_sharding_strategy='optim_grads_params'),
                [parameter],True,0,device=torch.device('cpu'),data_parallel_group=dist.group.WORLD)
            buffer.init_data(torch.full((buffer.data_size,),-1234.,dtype=torch.float32))
            buffer.set_item(0,full)
            wrapper=nn.Module()
            wrapper.module=owner
            wrapper.param_and_grad_buffer=SimpleNamespace(param_to_param_group={parameter:0},
                parameter_groups=[SimpleNamespace(model_weight_buffer=buffer,hfsdp_helper_wbuf=None)])
            source=resolve_weight_source(owner,'weight',model_roots=[wrapper],expected_shape=full.shape)
            parameter.data=torch.empty(0)
            capture=WeightCapture(0,'router_projection_weight',source,full.shape,source.ranges,rank)
            reports=[None]*2
            dist.all_gather_object(reports,capture.report())
            layout=next(r for r in assign_weight_fragments(reports) if r['producer_rank']==rank)
            capture.assigned=layout['fragments']
            rows,topology=[None]*2,[None]*2
            dist.all_gather_object(rows,dict(model_id='run',phase='train',global_batch_id=3,attempt_id=0,
                layer_no=0,act_name=capture.act_name,producer_rank=rank,value=capture.pack()))
            dist.all_gather_object(topology,dict(model_id='run',producer_rank=rank,weight_layout_json=json.dumps([layout])))
            torch.testing.assert_close(merge_weight_shards(rows,topology)[0][0].value,full)
    finally:
        dist.destroy_process_group()


def test_real_dtensor_shards_reconstruct_with_two_gloo_ranks(tmp_path):
    import torch.multiprocessing as mp
    mp.spawn(_distributed_source_worker,args=(str(tmp_path/'rendezvous'),),nprocs=2,join=True)


def test_initial_weight_snapshot_does_not_count_toward_iteration_readiness():
    from dmi.api.v1 import HookOutput, ProducerPlanEntry, TransportSpec, RecordType
    from dmi_megatron_integration.records.format import MegatronRecordFormat
    from dmi_megatron_integration.records.metadata import MegatronRecordMetadata
    from dataclasses import replace
    fmt=MegatronRecordFormat('raw')
    fmt.count_records=True
    metadata=MegatronRecordMetadata('run','query_projection_weight','iter','train',3,-1,-1,0,0,0,attempt_id=-1)
    entry=ProducerPlanEntry.from_output(output_id=1,
        output_spec=TransportSpec('query_projection_weight',record_type=RecordType.PER_ITERATION),
        output=HookOutput(torch.empty(0,dtype=torch.uint8)))
    assert len(fmt.encode(metadata,entry).rows)==1
    assert fmt.take_expected_count('train',3,-1)==0
    assert len(fmt.encode(replace(metadata,attempt_id=0),entry).rows)==1
    assert fmt.take_expected_count('train',3,0)==1
    with pytest.raises(ValueError,match='attempt_id'):
        fmt.encode(replace(metadata,act_name='grad_norm'),entry)
