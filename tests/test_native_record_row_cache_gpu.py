"""Dynamic Megatron rows through eager producers and real CUDA-graph replay."""
import json
import pytest
import torch
from dmi.api.v1 import DropConfig, MonitoringConfig, MonitoringEngine, RingConfig, StepReservation, TransportType
from dmi_megatron_integration.adapter import MegatronAdaptor, MegatronHookBinding, MegatronTrainingContext
from dmi_megatron_integration.hooks.specs import DimSpec, HookPhase, MegatronHookSpec, MegatronOutputSpec, ShardPolicy
from dmi_megatron_integration.metadata_context import DMIMetadataContext
from dmi_megatron_integration.records.format import MegatronRecordFormat
from tests.test_megatron_adapter import _make_hook, _tp_sequence_distributed_info


@pytest.mark.skipif(torch.cuda.device_count()<2, reason='Two CUDA devices required')
@pytest.mark.parametrize('tp_rank',[0,1])
@pytest.mark.parametrize('graph_mode',[False,True],ids=['eager','cuda_graph'])
def test_native_rows_reuse_layout_but_refresh_payload_and_identity(tmp_path,tp_rank,graph_mode):
    with torch.cuda.device(tp_rank):
        cfg=RingConfig();cfg.payload_ring_bytes=cfg.pinned_staging_bytes=1024*1024
        cfg.task_ring_entries=64
        engine=MonitoringEngine(config=MonitoringConfig(storage_backend='drop',
            drop=DropConfig(str(tmp_path),rank=tp_rank)),ring_config=cfg,record_mode_v1=True)
        try:
            runtime=engine.create_record_runtime(MegatronRecordFormat('cache_gpu',producer_rank=tp_rank))
            info=_tp_sequence_distributed_info(tp_rank=tp_rank,sequence_parallel_enabled=True)
            metadata=DMIMetadataContext(max_num_microbatches=1,max_batch_size=2,num_scopes=1,
                dims={DimSpec.SEQ:8},device=torch.device('cuda',tp_rank),
                megatron_distributed_info=info,tp_sequence_sharded_enabled=True)
            hook=_make_hook(MegatronHookSpec(name='prefix_probe',layer_no=2,
                shard_policy=ShardPolicy.TP_SEQUENCE_SHARDED,outputs=(MegatronOutputSpec(
                    name='prefix_probe',input_shape=(DimSpec.SEQ,DimSpec.BATCH,4),
                    output_shape=(DimSpec.ACTUAL_TOKEN_PACKED,4),dtype=torch.float32,
                    transport_type=TransportType.SEQ_PREFIX_PACK),)))
            hook.megatron_distributed_info=info
            adaptor=MegatronAdaptor(engine,runtime,'cache_gpu',dims={DimSpec.SEQ:8,DimSpec.BATCH:2})
            adaptor.attach_hooks(model_hooks=(MegatronHookBinding(hook=hook,record_shard_rank=tp_rank),),
                                 iteration_hooks=(),metadata_context=metadata)
            base=torch.arange(64,dtype=torch.float32).reshape(8,2,4)
            local=base[tp_rank*4:(tp_rank+1)*4].cuda()
            graph=None;plan=None;offset=0;expected=[]
            for iteration,counts in enumerate(((3,2),(3,2),(1,4),(7,2),(0,0),(8,8),(8,8))):
                value=base+iteration*100
                local.copy_(value[tp_rank*4:(tp_rank+1)*4])
                metadata.begin_iteration(1);metadata.ingest_microbatch(0,{'valid_count':counts})
                metadata.enter_scope('fwd',0,0)
                packing=metadata.prepared_packing(0)
                ctx=MegatronTrainingContext(global_batch_id=iteration,microbatch_id=0,
                    valid_counts=counts,dataset_ids=(10,20),shard_rank=tp_rank,attempt_id=iteration,
                    _packing=packing)
                adaptor.set_current_event(ctx)
                if graph_mode and iteration==1:
                    adaptor.begin_capture_plan(warmup_enabled=True,capture_direction=HookPhase.FWD)
                    graph=torch.cuda.CUDAGraph();stream=torch.cuda.Stream(device=tp_rank)
                    stream.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.graph(graph,stream=stream):
                        metadata.enter_scope('fwd',0,0);hook(local)
                    torch.cuda.current_stream().wait_stream(stream)
                    plan=adaptor.finish_capture_plan()
                if graph is None:hook(local)
                else:
                    assert adaptor.prepare_replay(plan,plan_direction=HookPhase.FWD) is StepReservation.RESERVED
                    graph.replay()
                local_counts=packing.counts_for(tp_rank*4,4).counts
                pieces=[]
                for sample,count in enumerate(local_counts):
                    if count:
                        pieces.append(value[tp_rank*4:tp_rank*4+count,sample].contiguous().view(torch.uint8).flatten())
                        expected.append((iteration,sample,count,ctx.dataset_ids[sample]))
                torch.cuda.synchronize();engine.flush_and_wait()
                if pieces:
                    reference=torch.cat(pieces)
                    actual=runtime._transport._ring_payload[offset:offset+reference.numel()].cpu()
                    assert torch.equal(actual,reference)
                    offset+=reference.numel()
                adaptor.clear_current_event();metadata.end_iteration()
        finally:engine.close()
    events=map(json.loads,(tmp_path/f'rank_{tp_rank:05}/events.jsonl').read_text().splitlines())
    rows=[r for e in events if e['type']=='record' for r in e['rows'] if r.get('act_name')=='prefix_probe']
    assert len(rows)==len(expected)
    for row,(iteration,sample,count,dataset) in zip(rows,expected):
        assert (row['global_batch_id'],row['attempt_id'],row['sample_index'])==(iteration,iteration,sample)
        assert row['shape']==[count,4]
        assert (row['producer_rank'],row['shard_rank'])==(tp_rank,tp_rank)
        assert (row['token_start'],row['token_end'],row['dataset_id'])==(tp_rank*4,tp_rank*4+count,dataset)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_native_rows_dynamic_eager_tensor_shapes_and_dtypes(tmp_path):
    from dmi.hooks.producer_plan import _align_up
    cfg=RingConfig();cfg.payload_ring_bytes=cfg.pinned_staging_bytes=1024*1024
    engine=MonitoringEngine(config=MonitoringConfig(storage_backend='drop',
        drop=DropConfig(str(tmp_path))),ring_config=cfg,record_mode_v1=True)
    expected=[];offset=0
    try:
        runtime=engine.create_record_runtime(MegatronRecordFormat('dynamic_gpu'))
        hook=_make_hook(MegatronHookSpec(name='dynamic_probe',layer_no=0,
            outputs=(MegatronOutputSpec(name='dynamic_probe',
                input_shape=(DimSpec.BATCH,4),dtype=torch.float32),)))
        adaptor=MegatronAdaptor(engine,runtime,'dynamic_gpu',dims={DimSpec.BATCH:2})
        adaptor.attach_hooks(model_hooks=(MegatronHookBinding(hook=hook),),iteration_hooks=())
        for iteration,(n,dtype) in enumerate(((2,torch.float32),(5,torch.float64),
                                            (0,torch.float16),(2,torch.float32),(2,torch.float32))):
            x=(torch.arange(n*4,device='cuda',dtype=dtype)+iteration*100).reshape(n,4)
            ctx=MegatronTrainingContext(global_batch_id=iteration,microbatch_id=0,
                valid_counts=(1,)*n,dataset_ids=tuple(range(n)),attempt_id=iteration)
            adaptor.set_current_event(ctx);hook(x)
            torch.cuda.synchronize();engine.flush_and_wait()
            reference=x.cpu().view(torch.uint8).flatten()
            assert torch.equal(runtime._transport._ring_payload[offset:offset+reference.numel()].cpu(),reference)
            offset+=_align_up(x.numel()*x.element_size())
            for sample in range(n):expected.append((iteration,sample,str(dtype),x.element_size()*4))
            adaptor.clear_current_event()
    finally:engine.close()
    events=map(json.loads,(tmp_path/'rank_00000/events.jsonl').read_text().splitlines())
    rows=[r for e in events if e['type']=='record' for r in e['rows'] if r.get('act_name')=='dynamic_probe']
    assert len(rows)==len(expected)
    for row,(iteration,sample,dtype,nbytes) in zip(rows,expected):
        assert (row['global_batch_id'],row['attempt_id'],row['sample_index'])==(iteration,iteration,sample)
        assert row['shape']==[4] and row['dtype']==dtype
