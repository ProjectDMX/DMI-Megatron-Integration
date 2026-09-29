"""CPU tests for scheduled-payload counts and lifecycle metadata publication."""
from types import SimpleNamespace
from dataclasses import replace
import pytest
import torch
from dmi.api.v1 import HookOutput, ProducerPlanEntry, TransportSpec, TransportType, OutputStorage, RecordType
from dmi_megatron_integration.records.format import MegatronRecordFormat
from dmi_megatron_integration.records.metadata import MegatronRecordMetadata
from dmi_megatron_integration.schedule_runtime import MegatronScheduleRuntime
from tests.test_megatron_schedule_runtime import FakePropagator


def record(fmt,phase,batch,attempt,counts=(2,0,3),name='hidden'):
    metadata=MegatronRecordMetadata('r',name,'fwd',phase,batch,0,0,0,0,0,valid_counts=counts,attempt_id=attempt)
    entry=ProducerPlanEntry.from_output(output_id=65536,output_spec=TransportSpec(name, transport_type=TransportType.SEQ_PREFIX_PACK,feature_bytes=8,output_shape=(-1,2)),output=HookOutput(torch.zeros(3,3,2)))
    return fmt.encode(metadata,entry)


def runtime_with_records():
    rows=[]
    runtime=MegatronScheduleRuntime(FakePropagator(),host_engine=SimpleNamespace(submit_record=lambda layout,row,types,**kwargs:rows.append((layout,row,types))))
    runtime.record_format=MegatronRecordFormat('raw',producer_rank=5)
    runtime.producer_rank=5
    runtime.adaptor=SimpleNamespace(model_id='r',begin_attempt=lambda **kw:None,end_attempt=lambda **kw:None,clear_current_event=lambda:None)
    return runtime,rows


def test_count_is_post_split_scheduled_rows_including_scalars_not_status():
    fmt=MegatronRecordFormat('raw',producer_rank=5)
    record(fmt,'train',0,0)  # Initial/resume snapshot excluded.
    fmt.count_records=True
    descriptor=record(fmt,'train',1,0)
    assert len(descriptor.rows)==2
    assert descriptor.rows[0][-2]==5
    meta=MegatronRecordMetadata('r','grad_norm','iter','train',1,-1,-1,-1,-1,0)
    spec=TransportSpec('grad_norm',storage=OutputStorage.SCALAR_FLOAT,record_type=RecordType.PER_ITERATION)
    entry=ProducerPlanEntry.from_output(output_id=65537,output_spec=spec,output=HookOutput(torch.zeros(1)))
    fmt.encode(meta,entry)
    fmt.encode(replace(meta,act_name='iteration_attempt_status'),entry)
    assert fmt.take_expected_count('train',1,0)==3
    assert fmt.take_expected_count('train',0,0)==0
    assert fmt.take_expected_count('train',1,0)==0


def test_training_retry_skipped_update_and_phase_total():
    runtime,rows=runtime_with_records()
    runtime.enter_phase('train',training_iteration_id_start=1)
    runtime.begin_logical_iteration(1)
    runtime.begin_attempt(0)
    record(runtime.record_format,'train',1,0)
    runtime.finish_attempt(0)
    runtime.begin_attempt(1)
    record(runtime.record_format,'train',1,1,counts=(1,0,0))
    runtime.finish_attempt(1,weights_updated=False)
    runtime.finish_logical_iteration()
    runtime.seal_current_phase()
    attempts=[row for layout,row,_ in rows if layout=='iteration_metadata']
    assert [(r[6],r[7],r[8],r[9]) for r in attempts]==[(0,0,0,2),(1,1,0,1)]
    assert all(r[5]==5 for r in attempts)
    end=[row for layout,row,_ in rows if layout=='phase_metadata' and row[5]=='exit'][0]
    assert end[6:]==(1,2,3)


def test_validation_each_batch_and_phase_metadata_include_zero_producer():
    runtime,rows=runtime_with_records()
    runtime.enter_phase('valid',training_iteration_id_start=600000,global_batch_id_start=101,eval_index=4)
    runtime.begin_iteration(1,forward_only=True)
    record(runtime.record_format,'valid',101,0)
    runtime.end_iteration()
    runtime.begin_iteration(1,forward_only=True)
    runtime.end_iteration()
    runtime.seal_current_phase()
    attempts=[row for layout,row,_ in rows if layout=='iteration_metadata']
    assert [(r[2],r[3],r[4],r[6],r[7],r[8],r[9]) for r in attempts]==[(101,600000,4,0,1,0,2),(102,600000,4,0,1,0,0)]
    end=[row for layout,row,_ in rows if layout=='phase_metadata' and row[5]=='exit'][0]
    assert end[6:]==(101,103,2)


def test_graph_capture_does_not_count_dry_runs_and_restores_live_counting():
    runtime,rows=runtime_with_records()
    runtime.begin_logical_iteration(1)
    runtime.begin_attempt(0)
    runtime.begin_full_iteration_capture()
    record(runtime.record_format,'train',1,0)
    runtime.finish_full_iteration_capture()
    assert runtime.record_format.count_records
    record(runtime.record_format,'train',1,0)
    runtime.finish_attempt(1,weights_updated=True)
    meta=[row for layout,row,_ in rows if layout=='iteration_metadata'][0]
    assert meta[8:]==(1,2)


def test_incomplete_attempt_cannot_publish_phase_complete_marker():
    runtime,rows=runtime_with_records()
    runtime.enter_phase('train',training_iteration_id_start=1)
    runtime.begin_logical_iteration(1)
    runtime.begin_attempt(0)
    with pytest.raises(RuntimeError,match='unfinished attempt'):
        runtime.seal_current_phase()
    assert not any(layout=='phase_metadata' and row[5]=='exit' for layout,row,_ in rows)
