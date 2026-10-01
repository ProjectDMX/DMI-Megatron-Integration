"""Opt-in native static-weight capture test; no training/optimizer workload."""
import os
import uuid
from types import SimpleNamespace
import pytest
import torch
from torch import nn

pytestmark = pytest.mark.skipif(os.environ.get('DMI_TEST_WEIGHT_GPU') != '1',
                               reason='Set DMI_TEST_WEIGHT_GPU=1 for native CUDA/ClickHouse test')


class SelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config=config
        self.layer_number=1
        self.hidden_size_per_attention_head=2
        self.linear_qkv=nn.Linear(7,24 if config.attention_output_gate else 16,bias=False)


class TopKRouter(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.config=config
        self.layer_number=1
        self.weight=nn.Parameter(torch.arange(28.).reshape(4,7))


@pytest.mark.parametrize('attention_output_gate', [False, True])
def test_native_weight_snapshot_before_mutation(tmp_path, attention_output_gate):
    from clickhouse_driver import Client
    from dmi_megatron_integration.startup import MegatronDMIConfig, setup_megatron_dmi
    from dmi_megatron_integration.signals.storage import ClickHouseStorage
    from dmi_megatron_integration.signals.events import Event
    from dmi_megatron_integration.materialization.reconstruction import merge_weight_shards
    from tests.test_megatron_startup import FakeDist, FakeParallelState
    client=Client('localhost')
    database='dmi_weight_gpu_test_'+uuid.uuid4().hex
    config=SimpleNamespace(num_layers=1,hidden_size=7,num_moe_experts=4,
                           num_attention_heads=4,num_query_groups=2,
                           attention_output_gate=attention_output_gate)
    root=nn.Module()
    root.attention=SelfAttention(config)
    root.router=TopKRouter(config)
    root.cuda()
    with torch.no_grad():
        weight = root.attention.linear_qkv.weight
        weight.copy_(torch.arange(weight.numel(),device='cuda').reshape_as(weight))
    fused=root.attention.linear_qkv.weight.detach().cpu().reshape(2,-1,7)
    kstart = 8 if attention_output_gate else 4
    expected=dict(query_projection_weight=fused[:,:4].reshape(8,7),
                  key_projection_weight=fused[:,kstart:kstart+2].reshape(4,7),
                  router_projection_weight=root.router.weight.detach().cpu())
    handle=None
    try:
        handle=setup_megatron_dmi([root],args=SimpleNamespace(global_batch_size=2,micro_batch_size=1),
            model_config=config,explicit_config=MegatronDMIConfig(enabled=True,model_id='run',
                hook_selection='q-weights,k-weights,router-weights',db_host='localhost',db_database=database,
                clickhouse_table='raw',ch_parallelism=1,ring_payload_mb=1,ring_pinned_mb=1,
                ring_task_entries=128,drain_flush_entry_threshold=1),
            parallel_state_module=FakeParallelState(),dist_module=FakeDist(initialized=False),
            unwrap_fn=lambda x:x,device='cuda')
        handle.emit_initial_qk_weights(model_state_iteration_id=0)
        handle.emit_initial_router_weights(model_state_iteration_id=0)
        handle.emit_qk_weights(model_state_iteration_id=1)
        handle.emit_router_weights(model_state_iteration_id=1)
        with torch.no_grad():
            for parameter in root.parameters():
                parameter.zero_()
        handle.emit_qk_weights(model_state_iteration_id=2)
        handle.emit_router_weights(model_state_iteration_id=2)
        # Native publication of a valid zero-byte rank contribution.
        from dmi.api.v1 import HookPointV1, HookSpecV1, ProducerPlanEntry, TransportSpec, RecordType
        from dmi_megatron_integration.records.metadata import MegatronRecordMetadata
        empty=torch.empty(0,dtype=torch.uint8,device='cuda')
        spec=TransportSpec('key_projection_weight',record_type=RecordType.PER_ITERATION)
        metadata=MegatronRecordMetadata('run','key_projection_weight','iter','train',3,-1,-1,0,0,0)
        runtime=handle.adaptor.record_runtime
        def prepare_empty(**kwargs):
            entry=ProducerPlanEntry.from_output(output_id=kwargs['output_id'],
                output_spec=kwargs['output_spec'],output=kwargs['output'])
            return runtime.emit_output(entry,metadata,kwargs['output'])
        empty_hook=HookPointV1(HookSpecV1('empty_weight',(spec,)))
        runtime.bind_hook(empty_hook,hook_runtime=SimpleNamespace(
            should_emit=lambda hook:True, prepare_output=prepare_empty))
        empty_hook(empty)
        handle.flush_and_wait(30)
        storage=ClickHouseStorage(client,database=database,base_table='raw',num_layers=1)
        topology=storage.read({'from':{'table':'raw_capture_topology'}},Event('run','iteration_end','train',2,2),{})
        zero_rows=storage.read({'from':{'table':'raw'},'where':{'global_batch_id':3}},
                               Event('run','iteration_end','train',3,3),{})
        assert len(zero_rows)==1 and zero_rows[0].value.shape==(0,)
        for step, attempt in [(0,-1),(1,0),(2,0)]:
            rows=storage.read({'from':{'table':'raw'},'where':{'global_batch_id':step,'attempt_id':attempt}},
                              Event('run','iteration_end','train',step,step),{})
            merged,=merge_weight_shards(rows,topology)
            assert len(merged)==3
            for row in merged:
                target=expected[row.act_name] if step<2 else torch.zeros_like(expected[row.act_name])
                torch.testing.assert_close(row.value,target,rtol=0,atol=0)
    finally:
        if handle is not None:
            handle.close()
        client.execute('DROP DATABASE IF EXISTS '+database+' SYNC')


def _fsdp2_static_worker(rank, rendezvous):
    import json
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import fully_shard
    from dmi_megatron_integration.hooks.weight_capture import (
        resolve_weight_source, WeightCapture, assign_weight_fragments)
    from dmi_megatron_integration.materialization.reconstruction import merge_weight_shards
    torch.cuda.set_device(rank)
    dist.init_process_group('nccl',init_method='file://'+rendezvous,rank=rank,world_size=2)
    try:
        mesh=init_device_mesh('cuda',(2,))
        owner=nn.Linear(7,5,bias=False,device='cuda',dtype=torch.bfloat16)
        reference=torch.arange(35.,device='cuda',dtype=torch.bfloat16).reshape(5,7)
        with torch.no_grad():
            owner.weight.copy_(reference)
        fully_shard(owner,mesh=mesh,reshard_after_forward=True)
        source=resolve_weight_source(owner,'weight',model_roots=[owner],expected_shape=(5,7))
        # No forward/backward: exercise actual gather/reshard storage lifetime.
        owner.unshard()
        owner.reshard()
        capture=WeightCapture(0,'router_projection_weight',source,(5,7),source.ranges,rank)
        reports=[None]*2
        dist.all_gather_object(reports,capture.report())
        layout=next(r for r in assign_weight_fragments(reports) if r['producer_rank']==rank)
        capture.assigned=layout['fragments']
        snapshot=capture.pack()
        with torch.no_grad():
            owner.weight.to_local().zero_()
        payload=dict(model_id='run',phase='train',global_batch_id=1,attempt_id=0,layer_no=0,
                     act_name=capture.act_name,producer_rank=rank,value=snapshot.cpu())
        rows,topology=[None]*2,[None]*2
        dist.all_gather_object(rows,payload)
        dist.all_gather_object(topology,dict(model_id='run',producer_rank=rank,weight_layout_json=json.dumps([layout])))
        torch.testing.assert_close(merge_weight_shards(rows,topology)[0][0].value,reference.cpu(),rtol=0,atol=0)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count()<2,reason='Two visible GPUs required')
def test_real_fsdp2_retained_shards_after_reshard(tmp_path):
    import torch.multiprocessing as mp
    mp.spawn(_fsdp2_static_worker,args=(str(tmp_path/'fsdp2-rendezvous'),),nprocs=2,join=True)
