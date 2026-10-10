"""Eager preparation must preserve live metadata without changing graph paths."""
import pytest
import torch
from dmi.api.v1 import HookOutput, ProducerPlanEntry, ProducerPlanBuilder, TransportSpec, TransportType
from dmi.hooks.producer_plan import _align_up
from dmi_megatron_integration.adapter import MegatronTrainingContext
from dmi_megatron_integration.records.format import MegatronRecordFormat
from tests.test_megatron_adapter import (
    FakeEngine, TinyModel, _make_adaptor, _make_hook, MegatronHookSpec,
    MegatronOutputSpec, MegatronHookBinding, DimSpec, HookPhase,
)


def attached():
    engine = FakeEngine()
    adaptor = _make_adaptor(engine, 'test', dims={DimSpec.BATCH: 2, DimSpec.NUM_EXPERTS: 4})
    model = TinyModel()
    adaptor.attach_model(model, hook_selection='router-summary')
    semantic = adaptor._configured_hook(model.router0).output_semantics[0]
    return engine, adaptor, model, semantic


def test_eager_metadata_uses_live_context_without_rederiving_requirements(monkeypatch):
    _, adaptor, _, semantic = attached()
    fmt = MegatronRecordFormat('test')
    for i, counts in enumerate(((1, 2), (2, 1), (0, 0), (4, 3))):
        ctx = MegatronTrainingContext(global_batch_id=i, microbatch_id=i % 2,
              valid_counts=counts, dataset_ids=(() if i == 0 else (i, i+1)),
              attempt_id=i, phase=('train' if i < 2 else 'valid'), model_id=f'run{i}')
        expected = adaptor._record_metadata(ctx, semantic)
        def forbidden(**kw):
            raise AssertionError('reconstructed requirements on eager path')
        with monkeypatch.context() as m:
            m.setattr('dmi_megatron_integration.adapter.required_record_metadata_fields', forbidden)
            actual = adaptor._record_metadata(ctx, semantic, eager=True)
        assert actual == expected
        entry = ProducerPlanEntry.from_output(output_id=semantic.output_id,
            output_spec=TransportSpec('x'), output=HookOutput(torch.ones(2, 4)))
        assert fmt.encode(actual, entry) == fmt.encode(expected, entry)


def test_eager_identity_refreshes_shapes_and_does_not_call_old_factory(monkeypatch):
    engine, adaptor, model, _ = attached()
    def forbidden(*a, **kw):
        raise AssertionError('old factory called on eager path')
    monkeypatch.setattr(ProducerPlanEntry, 'from_output', forbidden)
    monkeypatch.setattr(ProducerPlanBuilder, '_transport_args', forbidden)
    for n, dtype in [(2, torch.float32), (5, torch.float64), (0, torch.float16), (2, torch.float32)]:
        adaptor.set_current_event(MegatronTrainingContext(global_batch_id=n, microbatch_id=0,
                                                         valid_counts=(1,) * n))
        x = torch.empty(4, n, dtype=dtype).t()
        model.router0(x)
        entry, _, output = engine.record_runtime.emit_calls[-1]
        assert entry.input_shape == x.shape and entry.dtype == dtype
        assert output.tensor is x
        assert engine.record_runtime.reservation_calls[-1][0] == _align_up(x.numel()*x.element_size())


@pytest.mark.parametrize('full', [False, True])
def test_capture_and_replay_never_call_eager_factory_or_metadata(monkeypatch, full):
    _, adaptor, model, _ = attached()
    def forbidden(*a, **kw):
        raise AssertionError('eager preparation used for a graph')
    monkeypatch.setattr(ProducerPlanEntry, '_from_eager_output', forbidden)
    old = adaptor._record_metadata
    def metadata(*a, **kw):
        assert not kw.get('eager', False)
        return old(*a, **kw)
    monkeypatch.setattr(adaptor, '_record_metadata', metadata)
    ctx = MegatronTrainingContext(global_batch_id=1, microbatch_id=0, valid_counts=(1, 1))
    adaptor.set_current_event(ctx)
    adaptor.begin_capture_plan(warmup_enabled=False, capture_event_context=full,
                               capture_direction=None if full else HookPhase.FWD)
    model.router0(torch.ones(2, 4))
    plan = adaptor.finish_capture_plan()
    if not full:
        adaptor.prepare_replay(plan, ctx, plan_direction='fwd')


def test_packed_wrapper_capture_uses_original_path_and_eager_uses_fresh_metadata(monkeypatch):
    engine = FakeEngine();adaptor = _make_adaptor(engine, 'test', dims={DimSpec.SEQ: 4, DimSpec.BATCH: 2})
    hook = _make_hook(MegatronHookSpec(name='packed', layer_no=0, outputs=[
        MegatronOutputSpec(name='packed', input_shape=[DimSpec.SEQ, DimSpec.BATCH, 3],
            output_shape=[-1, 3], dtype=torch.float32, transport_type=TransportType.SEQ_PREFIX_PACK)],
        preprocess=None))
    hook.valid_count_fwd = torch.tensor([1, 2]);hook.valid_count_prefix_fwd = torch.tensor([0, 1, 3])
    adaptor.attach_hooks(model_hooks=(MegatronHookBinding(hook=hook),), iteration_hooks=())
    x = torch.arange(24.).reshape(4,2,3)
    fmt = MegatronRecordFormat('packed')
    for i, counts in enumerate(((1,2), (2,1), (0,0), (4,4))):
        hook.valid_count_fwd = torch.tensor(counts)
        hook.valid_count_prefix_fwd = torch.tensor([0, counts[0], sum(counts)])
        ctx = MegatronTrainingContext(global_batch_id=i, microbatch_id=0, valid_counts=counts)
        adaptor.set_current_event(ctx)
        enriched = hook.spec.preprocess(x)
        assert enriched.tensor is x and enriched.producer_meta[0] is hook.valid_count_fwd
        hook(x)
        entry, metadata, _ = engine.record_runtime.emit_calls[-1]
        assert metadata.valid_counts == counts
        assert engine.record_runtime.reservation_calls[-1][0] == _align_up(sum(counts)*12)
        assert len(fmt.encode(metadata, entry).rows) == sum(c > 0 for c in counts)
    def forbidden(*a, **kw):raise AssertionError('eager wrapper during capture')
    monkeypatch.setattr(adaptor, '_enrich_eager_output', forbidden)
    adaptor.begin_capture_plan(warmup_enabled=True, capture_direction=HookPhase.FWD)
    hook(x)
    adaptor.finish_capture_plan()


def test_eager_explicit_segment_metadata_keeps_original_checks_and_ranges():
    _, adaptor, _, _ = attached()
    spec = TransportSpec('segmented', transport_type=TransportType.SEGMENTED_PACK,
                         feature_bytes=4, output_shape=(-1, 1))
    x = torch.arange(12.)
    for starts, ends in [([0, 4], [1, 6]), ([2, 8], [5, 12])]:
        original = HookOutput(x, (torch.tensor(starts), torch.tensor(ends)))
        result = adaptor._enrich_eager_output(None, spec, original)
        assert result.tensor is x
        assert result.producer_meta[0] is original.producer_meta[0]
        assert result.producer_meta[1] is original.producer_meta[1]
    with pytest.raises(ValueError, match='int64'):
        adaptor._enrich_eager_output(None, spec,
            HookOutput(x, (torch.tensor([0.]), torch.tensor([2.]))))
