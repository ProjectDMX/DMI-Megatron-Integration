"""Phase checks precede preprocessing and shared CUDA graph selection."""
from types import SimpleNamespace
from dataclasses import replace
from unittest.mock import Mock
import pytest
import torch
from dmi_megatron_integration.hooks.selection import resolve_phase_hook_selections, parse_hook_selection
from dmi_megatron_integration.hooks.specs import DimSpec
from dmi_megatron_integration import schedule_runtime as schedule
from tests.test_megatron_adapter import FakeEngine, TinyModel, _make_adaptor


def test_phase_options_inherit_replace_disable_and_normalize():
    assert resolve_phase_hook_selections('router-summary', valid='hidden-states', test='none') == {
        'train': {'router-summary'}, 'valid': {'hidden-states'}, 'test': set()}
    assert len(set(resolve_phase_hook_selections('router-topk', valid='router-topk-weights,router-topk-expert-ids').values())) == 1
    assert parse_hook_selection('none') == set()
    with pytest.raises(ValueError): parse_hook_selection('none,hidden-states')
    with pytest.raises(ValueError): resolve_phase_hook_selections('hidden-states', test='typo')
    assert resolve_phase_hook_selections('custom', additional_names={'custom'})['train'] == {'custom'}


def test_phase_cli_resolves_into_config():
    from argparse import ArgumentParser
    from megatron.training.arguments import _add_dmi_args
    from dmi_megatron_integration.startup import resolve_megatron_dmi_config
    parser = _add_dmi_args(ArgumentParser())
    cfg = resolve_megatron_dmi_config(parser.parse_args([
        '--dmi-hook-selection', 'router-logits',
        '--dmi-train-hook-selection', 'q-weights',
        '--dmi-valid-hook-selection', 'hidden-states',
        '--dmi-test-hook-selection', 'none',
    ]), environ={})
    assert (cfg.hook_selection, cfg.train_hook_selection, cfg.valid_hook_selection,
            cfg.test_hook_selection) == ('router-logits', 'q-weights', 'hidden-states', 'none')


def make_hook():
    engine = FakeEngine()
    model = TinyModel()
    adaptor = _make_adaptor(engine, 'r', dims={DimSpec.BATCH: 1, DimSpec.NUM_EXPERTS: 2})
    adaptor.attach_model(model, hook_selection='router-summary')
    return engine, adaptor, model.router0


@pytest.mark.parametrize('phase', ['train', 'valid', 'test'])
def test_disabled_hook_skips_preprocess_and_record_work(phase):
    engine, adaptor, hook = make_hook()
    hook.megatron_enabled_phases = frozenset({'valid'})
    hook.spec = replace(hook.spec, preprocess=Mock(side_effect=AssertionError('preprocessing called')))
    adaptor.current_context = SimpleNamespace(phase=phase, direction='fwd')
    if phase == 'valid':
        with pytest.raises(AssertionError, match='preprocessing called'): hook(torch.ones(1, 2))
    else:
        hook(torch.ones(1, 2))
    assert not engine.record_runtime.emit_calls
    assert not engine.record_runtime.reservation_calls
    assert not engine.record_runtime.dispatch_calls


def test_local_capture_uses_train_policy_despite_stale_eval_context():
    engine, adaptor, hook = make_hook()
    hook.megatron_enabled_phases = frozenset({'valid'})
    adaptor.current_context = SimpleNamespace(phase='valid', direction='fwd')
    adaptor.begin_capture_plan(warmup_enabled=True, capture_direction='fwd')
    hook(torch.ones(1, 2))
    assert adaptor.finish_capture_plan().entries == ()
    assert not engine.record_runtime.dispatch_calls


def test_full_iteration_capture_uses_live_phase():
    _, adaptor, hook = make_hook()
    hook.megatron_enabled_phases = frozenset({'valid'})
    adaptor.begin_capture_plan(warmup_enabled=True, capture_event_context=True)
    for phase in ('train', 'valid', 'test'):
        adaptor.current_context = SimpleNamespace(phase=phase, direction='fwd')
        assert adaptor.hook_runtime.should_emit(hook) == (phase == 'valid')
    adaptor.abort_capture_plan()


def test_disabled_output_is_absent_from_graph_plan():
    engine, adaptor, hook = make_hook()
    hook.megatron_output_phases = {'router_probs_mean': {'valid'}}
    adaptor.begin_capture_plan(warmup_enabled=True, capture_direction='fwd')
    hook(torch.ones(1, 2))
    assert adaptor.finish_capture_plan().entries == ()
    assert not engine.record_runtime.dispatch_calls


def test_validation_only_weights_are_not_packed_for_training():
    from dmi_megatron_integration.startup import MegatronDMIHandle
    handle = object.__new__(MegatronDMIHandle)
    captures = [SimpleNamespace(
        act_name=name, hook=SimpleNamespace(megatron_enabled_phases={'valid'}),
        pack=Mock(side_effect=AssertionError('weight packing')),
    ) for name in ('query_projection_weight', 'key_projection_weight', 'router_projection_weight')]
    handle.weight_captures = captures
    values = []
    handle._emit_iteration_values = lambda **kwargs: values.extend(kwargs['values'])
    handle.emit_qk_weights(model_state_iteration_id=1)
    handle.emit_router_weights(model_state_iteration_id=1)
    handle.emit_initial_qk_weights(model_state_iteration_id=0)
    handle.emit_initial_router_weights(model_state_iteration_id=0)
    assert not values
    for capture in captures:
        capture.pack.assert_not_called()


@pytest.mark.parametrize('rank', [0, 1])
def test_fallback_phase_and_one_time_warning(monkeypatch, capsys, rank):
    runtime = SimpleNamespace(phase_hook_selections_differ=True, phase='train', producer_rank=rank, _eager_phase_warning_printed=False)
    monkeypatch.setattr(schedule, '_active_runtime', runtime)
    assert not schedule.dmi_local_graph_evaluation_eager(warn=True)
    runtime.phase = 'valid'
    assert schedule.dmi_local_graph_evaluation_eager()
    assert not capsys.readouterr().err
    assert schedule.dmi_local_graph_evaluation_eager(warn=True)
    runtime.phase = 'test'
    assert schedule.dmi_local_graph_evaluation_eager(warn=True)
    assert capsys.readouterr().err.count('DMI: phase-specific') == (1 if rank == 0 else 0)
    runtime.phase_hook_selections_differ = False
    assert not schedule.dmi_local_graph_evaluation_eager(warn=True)


@pytest.mark.parametrize('wrapped', [False, True])
def test_local_manager_bypass_preserves_runners(monkeypatch, wrapped):
    from megatron.core.transformer import cuda_graphs as graphs
    from megatron.core.transformer.module import MegatronModule
    from megatron.core.transformer.transformer_config import TransformerConfig
    class Module(MegatronModule):
        def forward(self, x): return x + 2
    module = Module(TransformerConfig(num_layers=1, hidden_size=4, num_attention_heads=1))
    module.eval()
    manager = SimpleNamespace(func=(lambda x: x + 3) if wrapped else None,
                              get_cudagraph_runner=Mock(side_effect=AssertionError('runner selection')))
    monkeypatch.setattr(graphs, 'dmi_local_graph_evaluation_eager', lambda **kw: True)
    monkeypatch.setattr(graphs, 'dmi_prepare_local_forward_replay', Mock(side_effect=AssertionError('replay plan')))
    assert graphs.CudaGraphManager.__call__(manager, module, (4,), {}) == (7 if wrapped else 6)
    manager.get_cudagraph_runner.assert_not_called()
    monkeypatch.setattr(graphs._CudagraphGlobalRecord, 'create_cudagraphs', Mock(side_effect=AssertionError('capture')))
    assert graphs.create_cudagraphs() is None
