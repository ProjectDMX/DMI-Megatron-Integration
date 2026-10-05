"""The loss call site must honor the same phase overrides as hook installation."""
from types import SimpleNamespace
import pytest
import pretrain_gpt


@pytest.mark.parametrize('phase', ['train', 'valid', 'test'])
@pytest.mark.parametrize('common,override,expected', [
    ('none', 'loss-summary', True),
    ('loss-summary', 'none', False),
    ('loss-summary', 'router-logits', False),
    ('loss-summary', None, True),
    ('none', None, False),
])
def test_loss_call_site_phase_selection(monkeypatch, phase, common, override, expected):
    args = SimpleNamespace(dmi_hook_selection=common)
    setattr(args, f'dmi_{phase}_hook_selection', override)
    monkeypatch.setattr(pretrain_gpt, 'dmi_current_phase', lambda **kw: phase)
    assert pretrain_gpt._dmi_hook_selected(args, 'loss-summary') == expected


def test_validation_only_loss_does_not_enable_training_or_test(monkeypatch):
    args = SimpleNamespace(dmi_hook_selection='none', dmi_train_hook_selection='none',
                           dmi_valid_hook_selection='resid_final,router-logits,loss-summary',
                           dmi_test_hook_selection='none')
    for phase in ('train', 'valid', 'test'):
        monkeypatch.setattr(pretrain_gpt, 'dmi_current_phase', lambda **kw: phase)
        assert pretrain_gpt._dmi_hook_selected(args, 'loss-summary') == (phase == 'valid')


def test_phase_override_takes_precedence_over_environment(monkeypatch):
    monkeypatch.setenv('DMI_HOOK_SELECTION', 'loss-summary')
    monkeypatch.setattr(pretrain_gpt, 'dmi_current_phase', lambda **kw: 'valid')
    assert not pretrain_gpt._dmi_hook_selected(SimpleNamespace(dmi_valid_hook_selection='none'), 'loss-summary')
