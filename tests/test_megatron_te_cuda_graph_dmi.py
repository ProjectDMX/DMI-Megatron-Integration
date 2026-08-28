from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest
import torch

from megatron.core.transformer import cuda_graphs as cuda_graphs_module
from megatron.core.transformer import module as module_under_test
from megatron.core.transformer.module import GraphableMegatronModule
from megatron.core.transformer.transformer_layer import TransformerLayer


class FakeGraphable:
    def __init__(self, *, graphs=()):
        self.cuda_graphs = list(graphs)
        self.cuda_graph_manual_hooks = []
        self.current_microbatch = 0
        self.capture_calls = []

    def _should_call_local_cudagraph(self, *args, **kwargs):
        del args, kwargs
        return False

    def _should_call_te_cudagraph(self, *args, **kwargs):
        del args, kwargs
        return True

    def _te_cuda_graph_capture(self, *args, **kwargs):
        self.capture_calls.append((args, kwargs))
        return (args[0] + 1,)

    def _get_te_cuda_graph_replay_args(self, *args, **kwargs):
        return tuple(args), dict(kwargs)


class CompatiblePlan:
    def __init__(self, signature):
        self.signature = signature
        self.compatibility_checks = []

    def assert_compatible(self, other):
        self.compatibility_checks.append(other)
        if self.signature != other.signature:
            raise ValueError("incompatible plan")


def test_te_actual_capture_records_one_plan_and_warmup_does_not(monkeypatch):
    layer = FakeGraphable()
    calls = []
    plan = CompatiblePlan("p0")
    monkeypatch.setattr(module_under_test, "dmi_is_te_capture_session_active", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(
        module_under_test,
        "dmi_begin_te_forward_capture",
        lambda: calls.append("begin"),
    )
    monkeypatch.setattr(
        module_under_test,
        "dmi_finish_te_forward_capture",
        lambda: calls.append("finish") or plan,
    )

    warmup = GraphableMegatronModule.__call__(layer, torch.tensor([1.0]))
    assert torch.equal(warmup[0], torch.tensor([2.0]))
    assert calls == []
    assert not hasattr(layer, "_dmi_te_captured_forward_plans")

    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    captured = GraphableMegatronModule.__call__(layer, torch.tensor([3.0]))
    assert torch.equal(captured[0], torch.tensor([4.0]))
    assert calls == ["begin", "finish"]
    assert layer._dmi_te_captured_forward_plans == [plan]


def test_te_actual_capture_aborts_plan_on_failure(monkeypatch):
    layer = FakeGraphable()
    layer._te_cuda_graph_capture = lambda *args, **kwargs: (_ for _ in ()).throw(
        RuntimeError("capture failed")
    )
    calls = []
    monkeypatch.setattr(module_under_test, "dmi_is_te_capture_session_active", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    monkeypatch.setattr(
        module_under_test,
        "dmi_begin_te_forward_capture",
        lambda: calls.append("begin"),
    )
    monkeypatch.setattr(
        module_under_test,
        "dmi_abort_te_forward_capture",
        lambda: calls.append("abort"),
    )

    with pytest.raises(RuntimeError, match="capture failed"):
        GraphableMegatronModule.__call__(layer, torch.tensor([1.0]))
    assert calls == ["begin", "abort"]


def test_te_replay_prepares_matching_plan_before_graph(monkeypatch):
    replay_calls = []

    def graph(x, **kwargs):
        replay_calls.append((x.clone(), kwargs))
        return (x + 2,)

    layer = FakeGraphable(graphs=(graph,))
    plan = CompatiblePlan("p0")
    layer._dmi_te_forward_plans = (plan,)
    monkeypatch.setattr(module_under_test, "dmi_is_enabled", lambda: True)
    monkeypatch.setattr(
        module_under_test,
        "dmi_prepare_te_forward_replay",
        lambda actual: replay_calls.append(("prepare", actual)) or False,
    )

    result = GraphableMegatronModule._te_cuda_graph_replay(layer, torch.tensor([4.0]))

    assert replay_calls[0] == ("prepare", plan)
    assert torch.equal(replay_calls[1][0], torch.tensor([4.0]))
    assert torch.equal(result[0], torch.tensor([6.0]))
    assert layer.capture_calls == []


def test_te_replay_capacity_fallback_executes_only_graph_covered_function(monkeypatch):
    graph_calls = []
    layer = FakeGraphable(graphs=(lambda x: graph_calls.append(x) or (x + 2,),))
    layer._dmi_te_forward_plans = (CompatiblePlan("p0"),)
    monkeypatch.setattr(module_under_test, "dmi_is_enabled", lambda: True)
    monkeypatch.setattr(module_under_test, "dmi_prepare_te_forward_replay", lambda plan: True)

    result = GraphableMegatronModule._te_cuda_graph_replay(layer, torch.tensor([4.0]))

    assert torch.equal(result[0], torch.tensor([5.0]))
    assert len(layer.capture_calls) == 1
    assert graph_calls == []


def test_te_replay_fails_on_missing_or_misaligned_plans(monkeypatch):
    layer = FakeGraphable(graphs=(lambda x: (x,),))
    monkeypatch.setattr(module_under_test, "dmi_is_enabled", lambda: True)

    with pytest.raises(RuntimeError, match="missing forward plans"):
        GraphableMegatronModule._te_cuda_graph_replay(layer, torch.tensor([1.0]))

    layer._dmi_te_forward_plans = ()
    with pytest.raises(RuntimeError, match="plan/graph count mismatch"):
        GraphableMegatronModule._te_cuda_graph_replay(layer, torch.tensor([1.0]))


def test_te_plan_mapping_validates_count_compatibility_and_cleanup():
    layer = SimpleNamespace(
        layer_number=3,
        cuda_graphs=[object(), object()],
        _dmi_te_captured_forward_plans=[CompatiblePlan("same"), CompatiblePlan("same")],
    )

    cuda_graphs_module._dmi_finalize_te_forward_plans(layer)

    assert len(layer._dmi_te_forward_plans) == 2
    assert not hasattr(layer, "_dmi_te_captured_forward_plans")
    cuda_graphs_module._dmi_clear_te_forward_plans(layer)
    assert not hasattr(layer, "_dmi_te_forward_plans")

    layer.cuda_graphs = [object()]
    layer._dmi_te_captured_forward_plans = []
    with pytest.raises(RuntimeError, match="plan/graph count mismatch"):
        cuda_graphs_module._dmi_finalize_te_forward_plans(layer)

    layer.cuda_graphs = [object(), object()]
    layer._dmi_te_captured_forward_plans = [CompatiblePlan("a"), CompatiblePlan("b")]
    with pytest.raises(ValueError, match="incompatible plan"):
        cuda_graphs_module._dmi_finalize_te_forward_plans(layer)


def test_hidden_state_hook_is_owned_by_common_attention_entry():
    attention_source = inspect.getsource(TransformerLayer._forward_attention)
    forward_source = inspect.getsource(TransformerLayer.forward)

    assert "self.dmi_hidden_states(hidden_states)" in attention_source
    assert "self.dmi_hidden_states" not in forward_source
