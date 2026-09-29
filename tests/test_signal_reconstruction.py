"""Information-preserving EP reconstruction and matched-token CKA, entirely on CPU."""
import itertools
import json
import torch
import pytest
from dmi_megatron_integration.signals.definition import Row
from dmi_megatron_integration.materialization.reconstruction import reconstruct_expert_outputs, regroup_ep_by_expert, merge_vocab_shards, merge_token_shards
from dmi_megatron_integration.topology.ep_topology_manifest import FrozenMegatronEPTopologyManifest, MoELayerPlacement


def fixture():
    torch.manual_seed(28)
    pairs=list(itertools.combinations(range(4),2))
    ids=torch.tensor(pairs*10)
    manifest=FrozenMegatronEPTopologyManifest(model_id='r',tp_groups=((0,),(1,)),pp_groups=((0,),(1,)),dp_groups=((0,1),),cp_groups=((0,),(1,)),ep_groups=((0,1),),etp_groups=((0,),(1,)),dispatch_groups=((0,1),),expert_dp_groups=((0,),(1,)),layer_placements=(MoELayerPlacement(0,0,0),),local_expert_order_by_ep_rank=((0,1),(2,3)),sequence_parallel=False,top_k=2,dispatcher_type='alltoall',permutation_mode='non_fused',etp_composition='matching_row_sum',dropless=True,padded=False)
    truth=[torch.randn(60,2,8,dtype=torch.float64) for _ in range(2)]
    weights=[torch.softmax(torch.randn(60,2,dtype=torch.float64),dim=1) for _ in range(2)]
    def row(rank,name,value,execution=False):
        return Row(model_id='r',act_name=name,direction='fwd',phase='train',global_batch_id=1,attempt_id=0,invocation_id=0,dp_rank=-1 if execution else rank,microbatch_id=0,sample_index=-1 if execution else 0,layer_no=0,shard_rank=rank,producer_rank=rank,token_start=-1 if execution else 0,token_end=-1 if execution else 60,dataset_id=-1 if execution else 0,value=value)
    rows=[];weight_rows=[]
    for rank in range(2):
        rows.append(row(rank,'router_topk_expert_ids',ids))
        weight_rows.append(row(rank,'router_topk_weights',weights[rank]))
        rows.append(row(rank,'moe_inverse_map',torch.argsort(ids.flatten(),stable=True)//2,True))
        packed=[]
        for expert in (rank*2,rank*2+1):
            for src in range(2):
                for token,slot in (ids==expert).nonzero().tolist():
                    packed.append(truth[src][token,slot]*weights[src][token,slot])
        rows.append(row(rank,'moe_packed_weighted_output',torch.stack(packed),True))
    return rows,weight_rows,[Row(manifest_json=json.dumps(manifest.to_dict()))],truth,weights,pairs


def cka(x,y):
    x=x-x.mean(0);y=y-y.mean(0)
    return ((x.T@y).square().sum()/((x.T@x).square().sum().sqrt()*(y.T@y).square().sum().sqrt())).item()


def test_three_input_signal_reconstruction_plus_separate_weights_matches_cka_oracle():
    rows,weight_rows,topology,truth,weights,pairs=fixture()
    result,=reconstruct_expert_outputs(rows,topology)
    assert len(result)==2
    recovered=[]
    for row in result:
        matched=[w for w in weight_rows if w.dp_rank==row.dp_rank]
        assert len(matched)==1
        unweighted=row.weighted_outputs/matched[0].value.unsqueeze(-1)
        torch.testing.assert_close(unweighted,truth[row.dp_rank],rtol=1e-12,atol=1e-12)
        recovered.append(unweighted)
    ids=rows[0].value
    for a,b in pairs:
        mask=(ids[:,0]==a)&(ids[:,1]==b)
        actual=cka(torch.cat([x[mask,0] for x in recovered]),torch.cat([x[mask,1] for x in recovered]))
        expected=cka(torch.cat([x[mask,0] for x in truth]),torch.cat([x[mask,1] for x in truth]))
        assert actual==pytest.approx(expected,abs=1e-12)
    experts=regroup_ep_by_expert(result)
    assert set(experts)==set(range(4))
    assert len(set(experts[0]['token_ids']) & set(experts[1]['token_ids']))==20
    # Duplicate storage delivery is removed; conflicts are rejected.
    duplicate,=reconstruct_expert_outputs(rows+[rows[0]],topology)
    torch.testing.assert_close(duplicate[0].weighted_outputs,result[0].weighted_outputs)
    with pytest.raises(ValueError,match='different values'):
        reconstruct_expert_outputs(rows+[Row(rows[0],value=rows[0].value+1)],topology)


def test_sequence_reconstruction_preserves_intervals_and_rejects_gaps():
    rows,_,_,_,_,_=fixture()
    first=Row(rows[0],token_start=0,token_end=2,value=torch.ones(2,4))
    second=Row(first,token_start=2,token_end=3,value=2*torch.ones(1,4))
    merged,=merge_token_shards([second,first])
    assert merged[0].value.shape==(3,4)
    assert merged[0].value[-1,0]==2
    with pytest.raises(ValueError,match='gaps'):
        merge_token_shards([first,Row(second,token_start=3,token_end=4)])


def test_vocab_and_router_weight_reconstruction_with_capture_topology():
    from dmi_megatron_integration.materialization.reconstruction import merge_routing_shards
    rows,_,_,_,_,_=fixture()
    topology=[Row(producer_rank=r,tp_rank=r,tp_world_size=2,pp_rank=0,dp_rank=0,cp_rank=0) for r in range(2)]
    shards=[Row(rows[0],producer_rank=r,shard_rank=r,dp_rank=0,token_start=0,token_end=2,value=torch.full((2,3),float(r))) for r in range(2)]
    vocab,=merge_vocab_shards(list(reversed(shards)),topology)
    torch.testing.assert_close(vocab[0].value,torch.tensor([[0.,0.,0.,1.,1.,1.]]*2))
    weights,=merge_routing_shards([Row(row,token_end=3) for row in shards],topology)
    assert (weights[0].token_start,weights[0].token_end)==(0,3)
    torch.testing.assert_close(weights[0].value,torch.tensor([[0.,0.,0.],[0.,0.,0.],[1.,1.,1.]]))
    with pytest.raises(ValueError,match='complete TP'):
        merge_vocab_shards(shards[:1],topology)


def expert_similarity(reconstructed,weight_rows):
    """Application transform: same-token CKA after removing routing weights."""
    lookup={(r.dp_rank,r.sample_index):r for r in weight_rows}
    unweighted=[]
    for row in reconstructed:
        w=lookup[(row.dp_rank,row.sample_index)]
        assert list(range(w.token_start,w.token_end))==row.token_indices
        unweighted.append(Row(row,weighted_outputs=row.weighted_outputs/w.value.unsqueeze(-1)))
    experts=regroup_ep_by_expert(unweighted)
    output=[]
    for a,b in itertools.combinations(sorted(experts),2):
        left={key:value for key,value in zip(experts[a]['token_ids'],experts[a]['outputs'])}
        right={key:value for key,value in zip(experts[b]['token_ids'],experts[b]['outputs'])}
        shared=sorted(left.keys() & right.keys())
        output.append(dict(expert_a=a,expert_b=b,score=cka(torch.stack([left[k] for k in shared]),torch.stack([right[k] for k in shared]))))
    return (output,)


from tests.test_signals import storage  # CPU ClickHouse fixture, opt-in via environment.


def test_yaml_worker_ep_reconstruction_to_cka_in_clickhouse(storage):
    from dmi_megatron_integration.signals import parse_config,SignalWorker
    from dmi_megatron_integration.signals.definition import Output
    from dmi_megatron_integration.signals.events import Event
    rows,weights,topology,truth,_,pairs=fixture()
    columns=[(key,'String' if isinstance(value,str) else 'Tensor' if isinstance(value,torch.Tensor) else 'Int64') for key,value in rows[0].items()]
    storage.write(Output('table','captures',tuple(columns)),rows+weights)
    storage.write(Output('table','topology',(('manifest_json','String'),)),topology)
    identity_columns=[{'name':name,'type':dtype} for name,dtype in [
        ('model_id','String'),('phase','String'),('global_batch_id','Int64'),('attempt_id','Int32'),('microbatch_id','Int32'),('layer_no','Int32'),('direction','String'),('dp_rank','Int32'),('dataset_id','Int32'),('sample_index','Int32'),('token_indices','Array(Int64)')]]
    ep=dict(name='ep',trigger={'event':'on_demand'},inputs=[{'from':{'table':'captures'},'where':{'act_name':{'in':['router_topk_expert_ids','moe_inverse_map','moe_packed_weighted_output']}},'scope':{'event':'current'}},{'from':{'table':'topology'}}],transform='builtin:reconstruct_expert_outputs',output=[{'signal':'ep_rows','columns':identity_columns+[{'name':'expert_ids','type':'Tensor'},{'name':'weighted_outputs','type':'Tensor'}]}])
    weight_signal=dict(name='weights',trigger={'event':'on_demand'},inputs=[{'from':{'table':'captures'},'where':{'act_name':'router_topk_weights'},'scope':{'event':'current'}},{'from':{'table':'topology'}}],transform='builtin:merge_routing_shards',output=[{'signal':'aligned_weights','columns':[c for c in identity_columns if c['name'] != 'token_indices'] + [{'name':'invocation_id','type':'Int64'},{'name':'act_name','type':'String'},{'name':'token_start','type':'Int64'},{'name':'token_end','type':'Int64'},{'name':'value','type':'Tensor'}]}])
    root=dict(name='cka',trigger={'event':'iteration_end','phase':'train'},inputs=[{'from':{'signal':'ep_rows'}},{'from':{'signal':'aligned_weights'}}],transform='tests.test_signal_reconstruction:expert_similarity',output=[{'table':'cka','columns':[{'name':'expert_a','type':'Int64'},{'name':'expert_b','type':'Int64'},{'name':'score','type':'Float64'}]}])
    result=SignalWorker(parse_config({'signals':[root,ep,weight_signal]}),storage).execute('cka',Event('r','iteration_end','train',1,1))
    ids=rows[0].value
    for row in result[0]:
        mask=(ids[:,0]==row.expert_a)&(ids[:,1]==row.expert_b)
        expected=cka(torch.cat([x[mask,0] for x in truth]),torch.cat([x[mask,1] for x in truth]))
        assert row.score==pytest.approx(expected,abs=1e-12)
    assert storage.client.execute('SELECT count() FROM '+storage.table('cka'))==[(6,)]
    assert storage.client.execute('EXISTS TABLE '+storage.table('ep_rows'))==[(0,)]
