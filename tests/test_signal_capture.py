"""Signal dependency planning and its CPU-only training startup boundary."""
from argparse import ArgumentParser, Namespace
import json
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

from dmi_megatron_integration.signals.capture import (
    compile_capture_plan, configure_signal_capture_args, merge_capture_selections,
)


def declaration(name='root', *, observation='router_logits', phase='train', helper=False):
    return {
        'name': name,
        'trigger': {'event': 'on_demand'} if helper else {'event': 'iteration_end', 'phase': phase},
        'inputs': [{'from': {'table': 'dmi_training_tensors'}, 'select': ['*'],
                    'where': {'act_name': observation}, 'scope': {'event': 'current'}}],
        # Intentionally unavailable: capture planning must not import user code.
        'transform': 'unavailable_analysis_module:analyze',
        'output': [{'signal' if helper else 'table': name + '_out',
                    'columns': [{'name': 'value', 'type': 'Float64'}]}],
    }


def write_config(tmp_path, signals):
    path = tmp_path / 'monitoring.yaml'
    path.write_text(yaml.safe_dump({'signals': signals}))
    return str(path)


def test_recursive_helpers_union_phases_and_ignore_unused_helpers():
    leaf = declaration('leaf', observation={'in': ['router_logits', 'hidden_states']}, helper=True)
    middle = declaration('middle', helper=True)
    middle['inputs'] = [{'from': {'signal': 'leaf_out'}}]
    train = declaration('training')
    train['inputs'] = [{'from': {'signal': 'middle_out'}}]
    valid = declaration('validation', phase='valid')
    valid['inputs'] = [{'from': {'table': 'external'}, 'joins': [
        {'type': 'cross', 'from': {'signal': 'middle_out'}}]}]
    unused = declaration('unused', observation='vocab_logits', helper=True)
    plan = compile_capture_plan({'signals': [train, valid, middle, unused, leaf]})
    assert plan == {'train': {'router-logits', 'hidden-states'},
                    'valid': {'router-logits', 'hidden-states'}, 'test': set()}


def test_history_and_cadence_do_not_remove_required_capture():
    root = declaration()
    root['trigger'] = {'event': 'validation_end', 'every_n_iterations': 100}
    root['inputs'][0]['scope'] = {'training_iteration': {'through': 600000}, 'layers': [-1]}
    plan = compile_capture_plan({'signals': [root]})
    assert all(hooks == {'router-logits'} for hooks in plan.values())
    root['inputs'][0]['scope']['phase'] = 'train'
    assert compile_capture_plan({'signals': [root]}) == {
        'train': {'router-logits'}, 'valid': set(), 'test': set()}


def test_parent_filters_are_not_pushed_through_helper_function():
    root = declaration()
    root['inputs'] = [{'from': {'signal': 'helper_out'}, 'scope': {'phase': 'train'}}]
    helper = declaration('helper', helper=True)
    helper['inputs'][0].pop('scope')
    assert all(compile_capture_plan({'signals': [root, helper]}).values())


def test_aliases_scalar_tables_and_automatic_metadata():
    root = declaration(observation={'eq': 'router_logits'})
    root['inputs'][0]['from']['as'] = 'r'
    root['inputs'][0]['joins'] = [
        {'type': 'cross', 'from': {'table': 'dmi_training_tensors_scalar_int', 'as': 's'}}]
    root['inputs'][0]['where'] = {
        'r.act_name': {'eq': 'router_logits'},
        's.act_name': {'in': ['pre_drop_token_count', 'iteration_attempt_status']},
    }
    root['inputs'].append({'from': {'table': 'dmi_training_tensors_iteration_metadata'}})
    assert compile_capture_plan({'signals': [root]})['train'] == {'router-logits', 'expert-counts'}


@pytest.mark.parametrize('observation', [None, {'ne': 'router_logits'}, {'in': []}, 'typo'])
def test_reject_unresolved_observations(observation):
    with pytest.raises(ValueError, match='act_name|nonempty|registered capture hook'):
        compile_capture_plan({'signals': [declaration(observation=observation)]})


def test_missing_dependency_and_cycle_rejected_before_training():
    root = declaration()
    root['inputs'] = [{'from': {'signal': 'helper_out'}}]
    with pytest.raises(ValueError):
        compile_capture_plan({'signals': [root]})
    helper = declaration('helper', helper=True)
    helper['inputs'] = [{'from': {'signal': 'helper_out'}}]
    with pytest.raises(ValueError, match='cycle'):
        compile_capture_plan({'signals': [root, helper]})


def test_validation_event_is_valid_only_and_conflicting_phase_fails():
    root = declaration()
    root['trigger'] = {'event': 'validation_end'}
    assert compile_capture_plan({'signals': [root]}) == {
        'train': set(), 'valid': {'router-logits'}, 'test': set()}
    root['trigger']['phase'] = 'train'
    with pytest.raises(ValueError, match='conflicts'):
        compile_capture_plan({'signals': [root]})


def test_manual_selection_union_preserves_custom_hooks_and_aliases():
    plan = compile_capture_plan({'signals': [declaration()]})
    merged = merge_capture_selections(plan, 'router-topk', valid='custom-hook', test='none')
    assert set(merged['train'].split(',')) == {
        'router-logits', 'router-topk-expert-ids', 'router-topk-weights'}
    assert merged['valid'] == 'custom-hook'
    assert merged['test'] == 'none'


def test_cli_enables_capture_before_startup_without_default_extra_hooks(tmp_path, monkeypatch):
    from megatron.training.arguments import parse_args
    from dmi_megatron_integration.startup import resolve_megatron_dmi_config
    path = write_config(tmp_path, [declaration(observation='grad_norm')])
    monkeypatch.setattr(sys, 'argv', ['pretrain_gpt.py', '--dmi-signal-config', path])
    args = parse_args()
    assert args.dmi_enable is True
    assert args.dmi_train_hook_selection == 'grad-norm'
    assert args.dmi_valid_hook_selection == args.dmi_test_hook_selection == 'none'
    cfg = resolve_megatron_dmi_config(args, environ={})
    assert cfg.enabled and cfg.signal_config_path == path
    assert cfg.train_hook_selection == 'grad-norm'
    assert cfg.valid_hook_selection == cfg.test_hook_selection == 'none'


def test_env_and_cli_precedence_with_custom_base_table(tmp_path):
    from dmi_megatron_integration.startup import resolve_megatron_dmi_config
    root = declaration()
    root['inputs'][0]['from']['table'] = 'custom_raw'
    path = write_config(tmp_path, [root])
    env = {'DMI_SIGNAL_CONFIG': path, 'DMI_CLICKHOUSE_TABLE': 'custom_raw',
           'DMI_HOOK_SELECTION': 'hidden-states'}
    args = configure_signal_capture_args(Namespace(dmi_hook_selection='grad-norm'), environ=env)
    assert args.dmi_enable
    assert args.dmi_train_hook_selection == 'grad-norm,router-logits'
    assert args.dmi_valid_hook_selection == 'grad-norm'
    cfg = resolve_megatron_dmi_config(environ=env)
    assert cfg.enabled
    assert cfg.train_hook_selection == 'hidden-states,router-logits'
    assert cfg.valid_hook_selection == 'hidden-states'


def test_without_yaml_manual_path_is_unchanged():
    from megatron.training.arguments import _add_dmi_args
    from dmi_megatron_integration.startup import resolve_megatron_dmi_config
    args = _add_dmi_args(ArgumentParser()).parse_args([])
    before = vars(args).copy()
    assert configure_signal_capture_args(args, environ={}) is args
    assert vars(args) == before
    cfg = resolve_megatron_dmi_config(args, environ={})
    assert not cfg.enabled and cfg.hook_selection == 'router-summary'


def test_preview_needs_no_cuda_megatron_or_analysis_module(tmp_path):
    path = write_config(tmp_path, [declaration()])
    result = subprocess.run([sys.executable, '-c', '''
import sys
from dmi_megatron_integration.signals.capture import main
main()
assert 'torch' not in sys.modules
assert 'megatron' not in sys.modules
assert 'unavailable_analysis_module' not in sys.modules
''', path], text=True, capture_output=True, check=True)
    assert json.loads(result.stdout) == {'train': ['router-logits'], 'valid': [], 'test': []}


def test_checked_in_monitoring_example():
    from dmi_megatron_integration.signals.capture import load_capture_plan
    path = Path(__file__).parents[1] / 'examples/signals/monitoring.yaml'
    assert load_capture_plan(path) == {'train': {'grad-norm'}, 'valid': set(), 'test': set()}


def test_yaml_startup_installs_requested_hooks_and_phase_guards(tmp_path):
    from types import SimpleNamespace
    from dmi_megatron_integration.startup import setup_megatron_dmi
    from tests.test_megatron_startup import (
        TinyGPTModel, FakeDist, FakeParallelState, FakeAdaptor, _fake_engine_factory,
        DMIMetadataContext, LocalMetadataPropagator, MegatronScheduleRuntime,
        CONSTANT_PROVENANCE, set_active_megatron_schedule_runtime,
    )
    path = write_config(tmp_path, [declaration(observation='lm_per_token_loss', phase='valid')])
    args = Namespace(global_batch_size=4, micro_batch_size=2, seq_length=16,
                     dmi_signal_config=path, dmi_model_id='test',
                     dmi_dataset_provenance_mode=CONSTANT_PROVENANCE)
    configure_signal_capture_args(args, environ={})
    model = TinyGPTModel()

    def runtime_factory(**kwargs):
        context = DMIMetadataContext(
            max_num_microbatches=kwargs['max_num_microbatches'],
            max_batch_size=kwargs['max_batch_size'], num_scopes=kwargs['num_scopes'],
            field_specs=kwargs['field_specs'], device='cpu',
        )
        return MegatronScheduleRuntime(LocalMetadataPropagator(context), host_engine=kwargs['host_engine'])

    FakeAdaptor.instances.clear()
    handle = None
    try:
        handle = setup_megatron_dmi(
            [model], args=args, model_config=SimpleNamespace(), environ={},
            parallel_state_module=FakeParallelState(), dist_module=FakeDist(initialized=False),
            unwrap_fn=lambda x: x, engine_factory=_fake_engine_factory,
            runtime_factory=runtime_factory, adaptor_cls=FakeAdaptor, device='cpu',
        )
        assert handle is not None
        assert model.dmi_lm_per_token_loss.megatron_enabled_phases == {'valid'}
        assert not hasattr(model, 'dmi_lm_per_sample_loss')
        assert len(FakeAdaptor.instances[0].attach_calls[0]['model_hooks']) == 1
    finally:
        if handle is not None:
            handle.close()
        set_active_megatron_schedule_runtime(None)
        FakeAdaptor.instances.clear()
