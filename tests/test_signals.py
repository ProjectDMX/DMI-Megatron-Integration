"""Signal contract tests; optional real ClickHouse tests run on CPU only."""
from copy import deepcopy
from types import SimpleNamespace
import json
import os
import subprocess
import sys
import uuid

import pytest

from dmi_megatron_integration.signals import parse_config, SignalWorker, Row
from dmi_megatron_integration.signals.definition import Output
from dmi_megatron_integration.signals.events import Event, EventPoller, readiness_query, matches
from dmi_megatron_integration.signals.storage import ClickHouseStorage

CALLS = []

def identity(rows):
    CALLS.append('identity')
    return (rows,)

def selected(rows):
    CALLS.append('selected')
    return ([{'id':row.id, 'score':row.value * 2} for row in rows],)

def merged(left, right):
    CALLS.append(('merged', len(left), len(right)))
    return ([{'id':row.id,'score':row.score + right[0].offset} for row in left], [{'n':len(left)}])


def declaration(name='root', event='validation_end', source=None, transform='identity', outputs=None):
    return dict(name=name, trigger={'event':event}, inputs=[{'from':source or {'table':'data'}, 'select':['*']}], transform=f'tests.test_signals:{transform}', output=outputs or [{'table':'result', 'columns':[{'name':'id','type':'Int64'},{'name':'value','type':'Float64'}]}])


def test_forward_reference_and_distinct_namespaces():
    producer=declaration('producer','on_demand',outputs=[{'signal':'same','columns':[{'name':'id','type':'Int64'},{'name':'value','type':'Float64'}]}])
    root=declaration(source={'table':'same'})
    root['inputs'][0]['joins']=[{'type':'inner','from':{'signal':'same','as':'v'},'on':[{'left':'same.id','op':'=','right':'v.id'}]}]
    registry=parse_config({'signals':[root,producer]})
    assert registry.outputs['same'][0].name=='producer'


@pytest.mark.parametrize('join', [False, True])
@pytest.mark.parametrize('event', ['iteration_end','validation_end','phase_end'])
def test_reject_non_on_demand_sources_at_parse(join,event):
    producer=declaration('producer',event,outputs=[{'signal':'v','columns':[{'name':'id','type':'Int64'}]}])
    root=declaration(source={'signal':'v'} if not join else None)
    if join:
        root['inputs'][0]['joins']=[{'type':'inner','from':{'signal':'v'},'on':[{'left':'data.id','op':'=','right':'v.id'}]}]
    with pytest.raises(ValueError,match=f'root.*v.*producer.*{event}'):
        parse_config({'signals':[root,producer]})


@pytest.mark.parametrize('source', [{'hook':'router_logits'}, {'table':'a','signal':'b'}, {'signal':'missing'}])
def test_invalid_sources(source):
    with pytest.raises(ValueError):
        parse_config({'signals':[declaration(source=source)]})


def test_cycles_and_root_cache_rejected():
    a=declaration('a','on_demand',{'signal':'v'},outputs=[{'signal':'v','columns':[{'name':'id','type':'Int64'}]}])
    with pytest.raises(ValueError,match='cycle'):
        parse_config({'signals':[a]})
    a=declaration();a['cache']={'mode':'once'}
    with pytest.raises(ValueError,match='on_demand'):
        parse_config({'signals':[a]})


def test_inclusive_trigger_and_on_demand_is_not_root():
    event=Event('run','validation_end','valid',600000)
    assert matches({'event':'validation_end','training_iteration_min':600000},event)
    assert not matches({'event':'validation_end','training_iteration_min':600001},event)
    assert not matches({'event':'on_demand'},event)
    registry=parse_config({'signals':[declaration('d','on_demand')]})
    with pytest.raises(ValueError,match='cannot be a root'):
        SignalWorker(registry,None).execute('d',event)


def test_config_import_has_no_training_or_gpu_initialization():
    code='import sys; import dmi_megatron_integration.signals; assert "torch" not in sys.modules; assert "megatron" not in sys.modules'
    subprocess.run([sys.executable,'-c',code],check=True)


@pytest.fixture
def storage():
    if os.environ.get('DMI_TEST_CLICKHOUSE')!='1':
        pytest.skip('Set DMI_TEST_CLICKHOUSE=1 for CPU ClickHouse integration tests')
    from clickhouse_driver import Client
    client=Client('localhost')
    database='dmi_signals_test_'+uuid.uuid4().hex
    client.execute('CREATE DATABASE '+database)
    s=ClickHouseStorage(client,database=database,base_table='raw',num_layers=4)
    try:
        yield s
    finally:
        client.execute('DROP DATABASE '+database+' SYNC')
        client.disconnect()


def seed(storage,name,columns,rows):
    storage.write(Output('table',name,tuple(columns)),rows)


def test_real_queries_virtual_joins_multiple_arguments_outputs_and_reuse(storage):
    CALLS.clear()
    seed(storage,'data',[('id','Int64'),('value','Float64')],[{'id':1,'value':2.0},{'id':2,'value':3.0}])
    seed(storage,'offsets',[('offset','Float64')],[{'offset':10.0}])
    producer=declaration('p','on_demand',transform='selected',outputs=[{'signal':'data','columns':[{'name':'id','type':'Int64'},{'name':'score','type':'Float64'}]}])
    root=declaration(transform='merged',outputs=[{'table':'result','columns':[{'name':'id','type':'Int64'},{'name':'score','type':'Float64'}]},{'signal':'summary','columns':[{'name':'n','type':'Int64'}]}])
    root['inputs']=[{'from':{'table':'data','as':'a'},'joins':[{'type':'inner','from':{'signal':'data','as':'b'},'on':[{'left':'a.id','op':'=','right':'b.id'}]}],'select':[{'column':'a.id','as':'id'},{'column':'b.score','as':'score'}]}, {'from':{'table':'offsets'},'select':['*']}]
    worker=SignalWorker(parse_config({'signals':[root,producer]}),storage)
    results=worker.execute('root',Event('run','validation_end','valid',10))
    assert sorted(results[0],key=lambda row:row.id)==[{'id':1,'score':14.0},{'id':2,'score':16.0}]
    assert results[1]==[{'n':2}]
    assert CALLS==['selected',('merged',2,1)]
    assert storage.client.execute('SELECT count() FROM '+storage.table('result'))==[(2,)]
    assert storage.client.execute('EXISTS TABLE '+storage.table('summary'))==[(0,)]


def test_once_cache_survives_worker_restart_and_empty_results(storage):
    CALLS.clear()
    seed(storage,'data',[('id','Int64'),('value','Float64')],[])
    producer=declaration('p','on_demand',outputs=[{'signal':'v','columns':[{'name':'id','type':'Int64'},{'name':'value','type':'Float64'}]}]);producer['cache']={'mode':'once'}
    root=declaration(source={'signal':'v'})
    registry=parse_config({'signals':[root,producer]})
    e=Event('run','validation_end','valid',600000)
    SignalWorker(registry,storage).execute('root',e)
    seed(storage,'data',[('id','Int64'),('value','Float64')],[{'id':1,'value':9.0}])
    assert SignalWorker(registry,storage).execute('root',Event('run','validation_end','valid',700000))==( [], )
    assert CALLS==['identity','identity','identity']  # Producer once, root twice.
    assert [c[0] for c in storage.schema('v_cache_once')]==['run_id','signal_name','id','value']
    assert SignalWorker(registry,storage).execute('root',Event('other-run','validation_end','valid',700000))[0][0].value==9.0


def test_tensor_roundtrip_and_layers_cutoff(storage):
    import torch
    seed(storage,'history',[('run_id','String'),('layer_no','Int32'),('training_iteration_id','Int64'),('x','Tensor')],
         [dict(run_id='r',layer_no=3,training_iteration_id=i,x=torch.tensor([[i]],dtype=torch.bfloat16)) for i in (5,6,7)])
    rows=storage.read({'from':{'table':'history'},'select':['*'],'scope':{'layers':[-1],'training_iteration':{'through':6}}},Event('r','validation_end','valid',8),{})
    assert sorted(r.training_iteration_id for r in rows)==[5,6]
    assert rows[0].x.dtype is torch.bfloat16
    assert sorted(r.x.item() for r in rows)==[5,6]


def capture_tables(storage):
    from dmi_megatron_integration.records.schema import TRAINING_ROW_COORDINATE_COLUMN_NAMES
    strings={'model_id','act_name','direction','phase'}
    coords=[(n,'String' if n in strings else 'Int64') for n in TRAINING_ROW_COORDINATE_COLUMN_NAMES]+[('producer_rank','Int32')]
    for suffix,values in [('',[('dtype','String'),('shape','Array(Int64)'),('bytes','String')]),('_scalar_float',[('value','Float64')]),('_scalar_int',[('value','Int64')])]:
        storage.create('raw'+suffix,coords+values)
    storage.create('raw_iteration_metadata',[('model_id','String'),('phase','String'),('global_batch_id','Int64'),('training_iteration_id','Int64'),('eval_index','Int32'),('producer_rank','Int32'),('attempt_id','Int32'),('status','Int32'),('weights_updated','Int32'),('expected_tensor_count','Int64')])
    storage.create('raw_phase_metadata',[('model_id','String'),('phase','String'),('training_iteration_id','Int64'),('eval_index','Int32'),('producer_rank','Int32'),('boundary_type','String'),('global_batch_id_start','Int64'),('global_batch_id_end','Int64'),('expected_tensor_count','Int64')])
    return coords


def payload(storage,coords,rank,attempt,batch=1,phase='train',invocation=0):
    row={n:0 for n,_ in coords}
    row.update(model_id='r',act_name='x',phase=phase,direction='fwd',global_batch_id=batch,producer_rank=rank,attempt_id=attempt,invocation_id=invocation,dtype='torch.float32',shape=[1],bytes=b'\x00\x00\x00\x00')
    storage._insert('raw',[row],coords+[('dtype','String'),('shape','Array(Int64)'),('bytes','String')])


def iteration_meta(storage,rank,attempt,count,status=1,batch=1,phase='train'):
    columns=storage.schema('raw_iteration_metadata')
    values=('r',phase,batch,600000,0,rank,attempt,status,0,count)
    storage._insert('raw_iteration_metadata',[dict(zip([n for n,_ in columns],values))],columns)


def test_iteration_readiness_counts_only_advancing_attempt_and_deduplicates_delivery(storage):
    coords=capture_tables(storage)
    e=Event('r','iteration_end','train',600000,1)
    poll=EventPoller(storage,run_id='r',expected_ranks=[0,1])
    iteration_meta(storage,0,0,1,status=0);payload(storage,coords,0,0)
    iteration_meta(storage,0,1,1);iteration_meta(storage,1,1,1)
    payload(storage,coords,0,1);payload(storage,coords,0,1)
    assert poll.ready(e) is None
    payload(storage,coords,1,1)
    assert poll.ready(e).attempt_id==1
    iteration_meta(storage,0,1,1)  # Duplicate metadata is harmless.
    assert poll.ready(e) is not None
    payload(storage,coords,1,1,invocation=1)  # Excess count is not ready.
    assert poll.ready(e) is None


def test_missing_zero_rank_metadata_and_disagreement(storage):
    capture_tables(storage)
    poll=EventPoller(storage,run_id='r',expected_ranks=[0,1])
    e=Event('r','iteration_end','valid',600000,1)
    iteration_meta(storage,0,0,0,phase='valid')
    assert poll.ready(e) is None
    iteration_meta(storage,1,0,0,phase='valid')
    assert poll.ready(e) is not None
    iteration_meta(storage,0,0,1,phase='valid')
    assert poll.ready(e) is None


def test_phase_waits_for_failed_attempts_and_late_earlier_batches(storage):
    coords=capture_tables(storage)
    columns=storage.schema('raw_phase_metadata')
    for rank in (0,1):
        row=dict(zip([n for n,_ in columns],('r','valid',600000,0,rank,'exit',1,3,3)))
        storage._insert('raw_phase_metadata',[row,row],columns)
        payload(storage,coords,rank,0,batch=2,phase='valid')
        payload(storage,coords,rank,1,batch=1,phase='valid')
    poll=EventPoller(storage,run_id='r',expected_ranks=[0,1]);e=Event('r','validation_end','valid',600000)
    assert poll.ready(e) is None
    payload(storage,coords,0,0,batch=1,phase='valid')
    assert poll.ready(e) is None
    payload(storage,coords,1,0,batch=1,phase='valid')
    assert poll.ready(e).end==3


def test_capture_history_cutoff_uses_training_iteration_not_validation_batch(storage):
    coords=capture_tables(storage)
    for batch, training in [(101,5),(102,6),(103,7)]:
        payload(storage,coords,0,0,batch=batch,phase='valid')
        columns=storage.schema('raw_iteration_metadata')
        values=('r','valid',batch,training,0,0,0,1,0,1)
        storage._insert('raw_iteration_metadata',[dict(zip([n for n,_ in columns],values))],columns)
    result=storage.read({'from':{'table':'raw'},'select':['*'],'scope':{'phase':'valid','training_iteration':{'through':6}}},Event('r','validation_end','valid',9),{})
    assert sorted(r.global_batch_id for r in result)==[101,102]


def test_multiple_virtual_uses_compute_producer_once(storage):
    CALLS.clear()
    seed(storage,'data',[('id','Int64'),('value','Float64')],[{'id':1,'value':2.0}])
    producer=declaration('p','on_demand',outputs=[{'signal':'v','columns':[{'name':'id','type':'Int64'},{'name':'value','type':'Float64'}]}])
    root=declaration(source={'signal':'v','as':'a'})
    root['inputs'][0].update(joins=[{'type':'inner','from':{'signal':'v','as':'b'},'on':[{'left':'a.id','op':'=','right':'b.id'}]}],select=[{'column':'a.id','as':'id'},{'column':'b.value','as':'value'}])
    SignalWorker(parse_config({'signals':[root,producer]}),storage).execute('root',Event('r','validation_end','valid',1))
    assert CALLS==['identity','identity']


def test_production_metadata_schema_matches_clickhouse_readiness(storage):
    from dmi_megatron_integration.records.schema import build_training_schema
    schema=build_training_schema('raw')
    for layout in schema.layouts:
        columns=[]
        for col in layout.columns:
            kind=col.type.value
            if kind=='tensor':
                columns.extend([(col.dtype_column,'String'),(col.shape_column,'Array(Int64)'),(col.bytes_column,'String')])
            else:
                columns.append((col.name,{'string':'String','int64':'Int64','int32':'Int32','float64':'Float64'}[kind]))
        storage.create(layout.table,columns)
    poller=EventPoller(storage,run_id='r',expected_ranks=[0])
    assert poller.ready(Event('r','iteration_end','train',1,1)) is None


def test_cache_keeps_full_output_before_consumer_selection(storage):
    CALLS.clear()
    seed(storage,'data',[('id','Int64'),('value','Float64')],[{'id':1,'value':2.0},{'id':2,'value':3.0}])
    producer=declaration('p','on_demand',outputs=[{'signal':'v','columns':[{'name':'id','type':'Int64'},{'name':'value','type':'Float64'}]}]);producer['cache']={'mode':'once'}
    one=declaration('one',source={'signal':'v'});one['inputs'][0]['where']={'id':1}
    two=declaration('two',source={'signal':'v'});two['inputs'][0]['where']={'id':2}
    registry=parse_config({'signals':[one,two,producer]})
    event=Event('r','validation_end','valid',1)
    assert SignalWorker(registry,storage).execute('one',event)[0][0].id==1
    assert SignalWorker(registry,storage).execute('two',event)[0][0].id==2
    assert CALLS==['identity','identity','identity']


def test_native_sink_metadata_to_worker_readiness(storage):
    import torch
    from dmi.api.v1 import ClickHouseClientConfig, DMXHostEngine, StageConfig
    from dmi_megatron_integration.records.schema import build_training_schema
    schema=build_training_schema('raw')
    config=ClickHouseClientConfig();config.host='localhost';config.database=storage.database
    engine=DMXHostEngine(StageConfig.clickhouse_records(config,schema,parallelism=2))
    try:
        engine.start()
        assert engine.wait_until_ready(10.0)
        engine.submit_record('iteration_metadata',('r','train',1,1,0,0,0,1,1,1),('string','string','int64','int64','int32','int32','int32','int32','int32','int64'),nbytes=56)
        assert engine.flush_and_wait(10.0)
        poller=EventPoller(storage,run_id='r',expected_ranks=[0])
        event=Event('r','iteration_end','train',1,1)
        assert poller.ready(event) is None
        cells=('r','x','fwd','train',1,0,0,0,0,0,0,1,0,0,0,0,torch.ones(1))
        types=tuple(c.type.value for c in schema.layout('tensor').columns)
        engine.submit_record('tensor',cells,types,nbytes=4)
        assert engine.flush_and_wait(10.0)
        assert poller.ready(event) is not None
    finally:
        assert engine.stop(True,10.0)


def test_worker_polls_readiness_before_root_and_does_not_relaunch(storage):
    CALLS.clear()
    coords=capture_tables(storage)
    seed(storage,'data',[('id','Int64'),('value','Float64')],[{'id':1,'value':2.0}])
    root=declaration(event='iteration_end')
    root['trigger']['phase']='train'
    worker=SignalWorker(parse_config({'signals':[root]}),storage)
    poller=EventPoller(storage,run_id='r',expected_ranks=[0])
    iteration_meta(storage,0,0,1)
    assert worker.poll_once(poller)==[]
    assert CALLS==[]
    payload(storage,coords,0,0)
    assert len(worker.poll_once(poller))==1
    assert CALLS==['identity']
    assert worker.poll_once(poller)==[]


def test_select_logical_tensor_column_and_alias(storage):
    import torch
    seed(storage,'t',[('x','Tensor')],[{'x':torch.arange(3)}])
    rows=storage.read({'from':{'table':'t'},'select':[{'column':'x','as':'hidden'}]},Event('r','validation_end','valid',1),{})
    assert set(rows[0])=={'hidden'}
    torch.testing.assert_close(rows[0].hidden,torch.arange(3))
