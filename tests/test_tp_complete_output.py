"""CPU contract tests; these do not claim NCCL/GPU execution coverage."""
import ast
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch

from dmi.api.v1 import OutputSizingMode, RecordType, TransportType
from dmi_megatron_integration.hooks.tp_complete_output import (
    NAME, FLAG, TPCompleteCapture, TokenSampling, capture_metadata,
    install_hooks, read_token_sampling, validate_config,
)


BASE = Path(__file__).resolve().parents[1]


def function(path, name, namespace):
    tree = ast.parse(path.read_text())
    node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)
    node.decorator_list = []
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), str(path), 'exec'), namespace)
    return namespace[name]


def config(**kwargs):
    result = dict(sequence_parallel=True, moe_token_dispatcher_type='alltoall',
        moe_router_topk=2, num_moe_experts=6, hidden_size=3, params_dtype=torch.float32,
        cuda_graph_impl='transformer_engine', cuda_graph_scope=['attn', 'moe_router', 'moe_preprocess'])
    result.update(kwargs)
    return NS(**result)


def routes(tokens, experts, topk, seed, skew=False):
    generator = torch.Generator().manual_seed(seed)
    ids = torch.rand(tokens, experts, generator=generator).topk(topk, dim=1).indices
    if skew:
        ids[:] = torch.arange(topk)
    routing = torch.zeros(tokens, experts, dtype=torch.bool).scatter_(1, ids, True)
    pairs = [(e, t) for e in range(experts) for t in range(tokens) if routing[t, e]]
    inverse = torch.tensor([t for e, t in pairs], dtype=torch.int64)
    values = torch.tensor([[seed * 1000 + t * 10 + e, t, e] for e, t in pairs], dtype=torch.float32)
    return routing, inverse, values, pairs


def materialize(output):
    if not output.producer_meta:
        return output.tensor
    starts, ends = (x.tolist() for x in output.producer_meta)
    return torch.cat([output.tensor[s:e] for s, e in zip(starts, ends)])


@pytest.mark.parametrize('etp', [2, 4])
@pytest.mark.parametrize('sampling', [None, TokenSampling(8, 1), TokenSampling(8, 3, 7), TokenSampling(8, 8)])
@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='requires a CUDA device'))])
def test_fixed_capture_reconstructs_routes_across_sources_and_changing_routing(etp, sampling, device):
    # Separate capture instances for each DP/ETP source; local selection does not
    # drop whole ranks. Every source emits exactly the declared number of rows.
    for dp in range(2):
        for rank in range(etp):
            capture = TPCompleteCapture(16, 2, 6, 3, torch.float32, sampling)
            ctx = NS(global_batch_id=1, microbatch_id=0, attempt_id=0)
            capture.hook = NS(_hook_runtime=NS(adaptor=NS(current_context=ctx)))
            for batch, micro, attempt, skew in [(1, 0, 0, False), (1, 1, 0, True),
                                               (2, 0, 0, False), (9, 0, 1, True)]:
                ctx.global_batch_id, ctx.microbatch_id, ctx.attempt_id = batch, micro, attempt
                routing, inverse, values, pairs = routes(16, 6, 2, 100*dp + 10*rank + batch, skew)
                routing, inverse, values = (x.to(device) for x in (routing, inverse, values))
                outputs = capture(values, routing, inverse)
                packed, tokens, counts = (materialize(output).cpu() for output in outputs)
                assert packed.shape == (capture.selected_rows, 3)
                assert tokens.shape == (capture.selected_rows,)
                assert counts.sum().item() == capture.selected_rows
                experts = torch.repeat_interleave(torch.arange(6), counts)
                reconstructed = list(zip(experts.tolist(), tokens.tolist()))
                selected = None if sampling is None else set(sampling.select(batch))
                wanted = [(e, t) for e, t in pairs if selected is None or t // (16//sampling.shards) in selected]
                assert reconstructed == wanted
                torch.testing.assert_close(packed, torch.tensor(
                    [[(100*dp + 10*rank + batch)*1000 + t*10+e, t, e] for e,t in wanted], dtype=torch.float32))
                assert outputs[0].tensor is values  # direct packing, no payload gather/copy
                assert outputs[1].tensor is inverse
            if device == 'cuda' and sampling:
                assert capture._segment_ranges.graph is not None


@pytest.mark.parametrize('etp', [2, 4])
def test_actual_combine_preprocess_keeps_source_expert_order(etp):
    td = BASE / 'third_party/megatron-lm/megatron/core/transformer/moe/token_dispatcher.py'
    mu = td.with_name('moe_utils.py')
    from typing import Optional, Tuple
    ns = dict(torch=torch, Optional=Optional, Tuple=Tuple)
    sort_chunks = function(mu, 'sort_chunks_by_idxs', ns)
    sources = [routes(16, 6, 2, r+1, r == 1) for r in range(etp)]
    counts = torch.stack([r[0].sum(0) for r in sources])
    all_rows = [(e, src, t) for e in range(6) for src in range(etp)
                for t in range(16) if sources[src][0][t,e]]
    expert_major = torch.tensor([[src*100 + t*10 + e] for e,src,t in all_rows], dtype=torch.float32)
    source_major = torch.cat([torch.tensor([[src*100+t*10+e] for e,t in source[3]], dtype=torch.float32)
                             for src, source in enumerate(sources)])
    calls = []
    for rank in range(etp):
        def reduce_scatter(value, group, input_split_sizes):
            calls.append(rank)
            assert input_split_sizes == [32]*etp
            torch.testing.assert_close(value, source_major / etp)
            # Simulate equal ETP partials, reduced and scattered by source.
            return (value*etp).split(input_split_sizes)[rank]
        ns.update(sort_chunks_by_idxs=sort_chunks, reduce_scatter_to_sequence_parallel_region=reduce_scatter)
        tree = ast.parse(td.read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'MoEAlltoAllTokenDispatcher')
        node = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'combine_preprocess')
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(td), 'exec'), ns)
        dispatcher = NS(num_local_experts=6, drop_and_pad=False, config=NS(moe_permute_fusion=False),
            num_global_tokens_per_local_expert=counts, tp_size=etp, tp_group=object(),
            output_splits_tp=torch.tensor([32]*etp), probs=torch.ones(1),
            restore_output_by_local_experts=torch.arange(6*etp).reshape(6,etp).T.ravel())
        result = ns['combine_preprocess'](dispatcher, expert_major/etp)
        torch.testing.assert_close(result, source_major.split(32)[rank])
    assert calls == list(range(etp))


def test_hook_location_preserves_old_hook_and_training_result():
    path = BASE / 'third_party/megatron-lm/megatron/core/transformer/moe/moe_layer.py'
    fn = function(path, 'routed_experts_compute', dict(torch=torch, apply_module=lambda x:x))
    events = []
    partial = torch.randn(4, 3, requires_grad=True)
    reduced = partial * 2
    def combine(value):
        assert value is partial
        events.append('etp_reduce_scatter')
        return reduced
    routing, inverse = object(), object()
    dispatcher = NS(dispatch_postprocess=lambda h,p:(h, None, p), combine_preprocess=combine,
                    routing_map=routing, reversed_local_input_permutation_mapping=inverse)
    def old(value):
        assert value is partial
        events.append('old')
    def new(value, rm, inv):
        assert value is reduced and rm is routing and inv is inverse
        events.append('new')
    layer = NS(token_dispatcher=dispatcher, experts=lambda *a:(partial,None),
               dmi_moe_packed_weighted_output=old, dmi_moe_tp_complete_output=new)
    output, bias = fn(layer, None, None)
    assert output is reduced and bias is None
    assert events == ['old', 'etp_reduce_scatter', 'new']
    output.sum().backward()
    torch.testing.assert_close(partial.grad, torch.full_like(partial, 2))


@pytest.mark.parametrize('overrides,topology', [
    ({}, (2,2,2,1)), ({}, (2,1,1,1)), ({}, (2,1,2,2)),
    ({'moe_token_dispatcher_type':'flex'}, (2,1,2,1)),
    ({'moe_permute_fusion':True}, (2,1,2,1)),
    ({'moe_expert_capacity_factor':1.0}, (2,1,2,1)),
    ({'moe_router_padding_for_quantization':True}, (2,1,2,1)),
    ({'sequence_parallel':False}, (2,1,2,1)),
    ({'cuda_graph_scope':[]}, (2,1,2,1)),
    ({'cuda_graph_scope':['moe']}, (2,1,2,1)),
    ({'recompute_granularity':'full'}, (2,1,2,1)),
])
def test_unsupported_layouts_fail_early(overrides, topology):
    with pytest.raises(ValueError):
        validate_config(NS(seq_length=32,micro_batch_size=1), config(**overrides), *topology, TokenSampling())


def test_config_and_fixed_shape_validation(tmp_path):
    args = NS(seq_length=4096, micro_batch_size=1)
    assert validate_config(args, config(), 2,1,2,1,TokenSampling()) == 2048
    with pytest.raises(ValueError, match='divisible'):
        validate_config(NS(seq_length=30,micro_batch_size=1),config(),2,1,2,1,TokenSampling())
    path=tmp_path/'hooks.yaml'
    path.write_text('hooks:\n  moe_tp_complete_output:\n    token_sampling: {shards: 8, count: 1, offset: 0}\n')
    assert read_token_sampling(path) == TokenSampling()
    path.write_text('hooks:\n  moe_tp_complete_output:\n    source_sampling: {}\n')
    with pytest.raises(ValueError):
        read_token_sampling(path)
    path.write_text('hooks:\n  moe_packed_weighted_output:\n    source_sampling: {}\n')
    with pytest.raises(ValueError,match='explicitly contain'):
        read_token_sampling(path)
    cap=TPCompleteCapture(16,2,6,3,torch.float32)
    routing, inverse, values, _=routes(16,6,2,1)
    with pytest.raises(ValueError,match='shape/dtype'):
        cap(values[:-1],routing,inverse)


@pytest.mark.parametrize('sampling',[None, TokenSampling()])
def test_installer_known_size_reservations_and_reconstruction_metadata(sampling,monkeypatch):
    from dmi_megatron_integration.startup import _validate_hook_contract
    from dmi_megatron_integration.adapter import MegatronAdaptor, _ProducerSemantics
    import dmi.hooks.producer_plan as pp
    # Only native alignment is mocked: no GPU/native engine is present on login.
    monkeypatch.setattr(pp, '_payload_alignment', lambda:256)
    layer=type('MoELayer',(torch.nn.Module,),{})()
    layer.config=config()
    layer.layer_number=1
    layer.token_dispatcher=type('MoEAlltoAllTokenDispatcher',(),dict(ep_size=1,tp_size=2))()
    install_hooks(layer,tokens=16,sampling=sampling)
    hook=layer.dmi_moe_tp_complete_output
    _validate_hook_contract(hook)
    spec=hook._dmi_megatron_spec
    cap=spec.preprocess
    cap.hook=NS(_hook_runtime=NS(adaptor=NS(current_context=NS(global_batch_id=1))))
    routing,inverse,values,_=routes(16,6,2,1)
    outputs=cap(values,routing,inverse)
    adapter=object.__new__(MegatronAdaptor)
    adapter._event_packing=None
    for i,(declared,output) in enumerate(zip(spec.outputs,outputs)):
        assert declared.sizing_mode is OutputSizingMode.KNOWN_BEFORE_EXECUTION
        assert declared.eager_nbytes is None
        physical=declared.resolve({},record_type=RecordType.PER_EXECUTION)
        semantic=_ProducerSemantics(i,declared.name,0,RecordType.PER_EXECUTION,
            declared.transport_type,-1,0,False,True,'fwd')
        entry,reserved=adapter._build_eager_entry(semantic,physical,output,None,NS())
        actual=materialize(output)
        assert entry.output_shape == tuple(actual.shape)
        assert reserved == pp._align_up(actual.numel()*actual.element_size())
        assert entry.reservation_upper_bytes == actual.numel()*actual.element_size()
    metadata=capture_metadata(16,sampling)
    assert metadata['etp_composition']=='already_reduced'
    assert metadata['sampling_domain']=='each_source_rank_independently'
    assert metadata['token_indices_output']==spec.outputs[1].name


def test_fixed_segmented_emission_keeps_actual_shape_and_bytes(monkeypatch):
    from dmi_megatron_integration.adapter import MegatronTrainingContext, _ProducerSemantics
    from dmi_megatron_integration.records.format import MegatronRecordFormat
    from dmi_megatron_integration.hooks.specs import MegatronOutputSpec
    from tests.test_megatron_adapter import FakeEngine, _make_adaptor
    import dmi.hooks.producer_plan as pp
    monkeypatch.setattr(pp, '_payload_alignment', lambda:256)
    engine = FakeEngine()
    adapter = _make_adaptor(engine, 'fixed-tp-test')
    ctx = MegatronTrainingContext(model_id='fixed-tp-test', global_batch_id=1,
                                  microbatch_id=0, valid_counts=())
    adapter.current_context = ctx
    capture = TPCompleteCapture(16,2,6,3,torch.float32,TokenSampling())
    capture.hook = NS(_hook_runtime=NS(adaptor=adapter))
    routing,inverse,values,_ = routes(16,6,2,1)
    output = capture(values,routing,inverse)[0]
    declared = MegatronOutputSpec(NAME,(32,3),torch.float32,output_shape=(4,3),
        transport_type=TransportType.SEGMENTED_PACK,segment_ranges_from_preprocess=True)
    physical = declared.resolve({},record_type=RecordType.PER_EXECUTION)
    semantic = _ProducerSemantics(1<<16,NAME,0,RecordType.PER_EXECUTION,
        TransportType.SEGMENTED_PACK,-1,0,False,True,'fwd')
    adapter.emit_immediate_output(semantics=semantic,output_spec=physical,output=output)
    entry, metadata = engine.record_runtime.emit_calls[-1][:2]
    assert entry.output_shape == (4,3)
    assert entry.reservation_upper_bytes == 48
    assert len(MegatronRecordFormat('test').encode(metadata,entry).rows)==1
