"""Focused contracts for the per-iteration Q/K weight hooks (no GPU required)."""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from dmi.api.v1 import RecordType, StepReservation, TransportType
from dmi_megatron_integration.hooks.megatron_qk_weights import qk_weight_from_fused_qkv
from dmi_megatron_integration.hooks.specs import DPEmissionPolicy, HookPhase, ShardPolicy
from dmi_megatron_integration.startup import (
    MegatronDMIConfig,
    MegatronRankContext,
    _megatron_hook_spec,
    setup_megatron_dmi,
)
from tests.test_megatron_startup import (
    FakeAdaptor,
    FakeDist,
    FakeEngine,
    FakeParallelState,
    TopKRouter,
)


class FakeCudaParameter(nn.Parameter):
    @property
    def is_cuda(self):
        # Discovery guard only; all arithmetic in this unit test remains CPU.
        return True


class SelfAttention(nn.Module):
    pass


class OlmoeSelfAttention(SelfAttention):
    pass


def _attention(*, heads=8, groups=2, head_dim=2, hidden=7, layer=1, tp=1,
               attention_output_gate=False):
    module = OlmoeSelfAttention()
    module.attention_type = "self"
    module.layer_number = layer
    module.config = SimpleNamespace(num_query_groups=groups * tp, hidden_size=hidden, num_attention_heads=heads * tp)
    module.config.attention_output_gate = attention_output_gate
    module.num_query_groups_per_partition = groups
    module.num_attention_heads_per_partition = heads
    module.hidden_size_per_attention_head = head_dim
    module.linear_qkv = nn.Module()
    q = torch.arange(heads * head_dim * hidden).reshape(heads * head_dim, hidden).float()
    k = 1000 + torch.arange(groups * head_dim * hidden).reshape(groups * head_dim, hidden).float()
    v = torch.full_like(k, -5000)
    parts = [q.reshape(groups, -1, hidden)]
    if attention_output_gate:
        parts.append(torch.full_like(parts[0], -9000))
    packed = torch.cat((*parts,
        k.reshape(groups, head_dim, hidden),
        v.reshape(groups, head_dim, hidden),
    ), dim=1).reshape(-1, hidden)
    module.linear_qkv.weight = FakeCudaParameter(packed)
    module.core_attention = nn.Module()
    module.core_attention.attention_type = "self"
    module.core_attention.layer_number = layer
    return module, q, k


def _rank(**overrides):
    values = dict(global_rank=0, tp_rank=0, tp_world_size=1, pp_rank=0,
                  pp_world_size=1, dp_rank=0, dp_world_size=1, cp_rank=0,
                  cp_world_size=1, ep_rank=0, ep_world_size=1, vp_rank=None,
                  num_layers=4)
    values.update(overrides)
    return MegatronRankContext(**values)


def _capture_hook_outputs(hook, on_output, *, should_emit=True):
    """Exercise the real hook/preprocess path, replacing only physical transfer."""
    def prepare_output(**kwargs):
        on_output(kwargs["output"].tensor)
        return StepReservation.OVERSIZED  # no GPU producer in these CPU tests

    hook.spec = _megatron_hook_spec(hook).resolve({})
    hook._bind_record_runtime(
        output_ids=(0,), ring_payload=torch.empty(0, dtype=torch.uint8),
        hook_runtime=SimpleNamespace(
            should_emit=lambda _hook: should_emit, prepare_output=prepare_output,
        ),
        gate_tensor=None, gate_value=0,
    )


@pytest.mark.parametrize("heads,groups", [(4,4), (8,2), (8,1)])
def test_mha_gqa_mqa_projection_reference(heads, groups):
    model, q, k = _attention(heads=heads, groups=groups)
    for projection, expected in [("q", q), ("k", k)]:
        actual = qk_weight_from_fused_qkv(model.linear_qkv.weight,
            num_query_groups=groups, query_rows_per_group=heads//groups*2,
            head_dim=2, projection=projection)
        torch.testing.assert_close(actual, expected)
        assert actual.untyped_storage().data_ptr() != model.linear_qkv.weight.untyped_storage().data_ptr()


@pytest.mark.parametrize("selection", ["q-weights", "k-weights", "q-weights,k-weights"])
@pytest.mark.parametrize("restriction", ["param_gather"])
def test_setup_rejects_unsafe_weight_access_before_engine(selection, restriction):
    model, _q, _k = _attention()
    calls = []
    args = SimpleNamespace(global_batch_size=4, micro_batch_size=2)
    if restriction == "param_gather":
        args.reuse_grad_buf_for_mxfp8_param_ag = True
        args.overlap_param_gather = True
    with pytest.raises(NotImplementedError):
        setup_megatron_dmi(
            [model], args=args, model_config=model.config,
            explicit_config=MegatronDMIConfig(enabled=True, hook_selection=selection, model_id="qk-test"),
            parallel_state_module=FakeParallelState(dp_world=2 if restriction == "dp" else 1),
            dist_module=FakeDist(initialized=False), unwrap_fn=lambda x: x,
            engine_factory=lambda *args: calls.append(True), adaptor_cls=FakeAdaptor, device="cpu",
        )
    assert calls == []


def test_setup_emits_initial_and_restored_states_with_fresh_values():
    model, q, k = _attention()

    class RecordingAdaptor(FakeAdaptor):
        def set_current_iteration(self, context):
            self.context = context

        def clear_current_iteration(self):
            self.context = None

    handle = setup_megatron_dmi(
        [model], args=SimpleNamespace(global_batch_size=4, micro_batch_size=2),
        model_config=model.config,
        explicit_config=MegatronDMIConfig(enabled=True, hook_selection="q-weights,k-weights", model_id="qk-test"),
        parallel_state_module=FakeParallelState(), dist_module=FakeDist(initialized=False),
        unwrap_fn=lambda x: x, engine_factory=lambda *args: (FakeEngine(), None),
        adaptor_cls=RecordingAdaptor, device="cpu",
    )
    observed = []
    try:
        assert len(handle.weight_captures) == 2
        hook_inputs = []
        for capture in handle.weight_captures:
            hook = capture.hook
            hook.register_forward_pre_hook(
                lambda _hook, args: hook_inputs.append(args[0])
            )
            _capture_hook_outputs(
                hook,
                lambda value, name=hook.spec.name: observed.append((
                    name, handle.adaptor.context, value,
                )),
            )
        handle.emit_initial_qk_weights(model_state_iteration_id=0)
        handle.emit_initial_qk_weights(model_state_iteration_id=600000)
        with torch.no_grad():
            model.linear_qkv.weight.add_(5)
        handle.emit_qk_weights(model_state_iteration_id=600001)
        assert len(observed) == 6
        assert len(hook_inputs) == 6
        assert all(value.dtype == torch.uint8 for value in hook_inputs)
        for idx, (name, context, value) in enumerate(observed):
            assert context.global_batch_id == (0, 600000, 600001)[idx // 2]
            assert context.microbatch_id == -1 and context.dataset_ids == ()
            assert context.direction == "iter" and context.phase == "train"
            expected = q if name == "query_projection_weight" else k
            assert context.attempt_id == (-1 if idx < 4 else 0)
            torch.testing.assert_close(value.view(expected.dtype).reshape(expected.shape), expected + (5 if idx >= 4 else 0))
        assert handle.adaptor.context is None
        with pytest.raises(ValueError, match="must be >= 1"):
            handle.emit_qk_weights(model_state_iteration_id=0)
    finally:
        handle.close()


def test_layer_stride_filters_weight_bindings_before_iteration_capture():
    model = nn.Module()
    model.layers = nn.ModuleList(_attention(layer=layer + 1)[0] for layer in range(3))
    for layer, module in enumerate(model.layers):
        module.router = TopKRouter(layer_number=layer + 1)
        module.router.weight = FakeCudaParameter(torch.zeros(4, 7))
        module.router.config = SimpleNamespace(num_moe_experts=4, hidden_size=7)
    handle = setup_megatron_dmi(
        [model], args=SimpleNamespace(global_batch_size=4, micro_batch_size=2),
        model_config=SimpleNamespace(num_layers=3, hidden_size=7, num_moe_experts=4),
        explicit_config=MegatronDMIConfig(
            enabled=True, layer_stride=2,
            hook_selection="q-weights,k-weights,router-weights,grad-norm", model_id="stride-test",
        ),
        parallel_state_module=FakeParallelState(), dist_module=FakeDist(initialized=False),
        unwrap_fn=lambda x: x, engine_factory=lambda *args: (FakeEngine(), None),
        adaptor_cls=FakeAdaptor, device="cpu",
    )
    try:
        assert [c.layer_no for c in handle.weight_captures if c.act_name != "router_projection_weight"] == [0, 0, 2, 2]
        assert [c.layer_no for c in handle.weight_captures if c.act_name == "router_projection_weight"] == [0, 2]
        bindings = handle.adaptor.attach_calls[0]["iteration_hooks"]
        unlayered = [binding for binding in bindings if _megatron_hook_spec(binding.hook).layer_no == -1]
        assert len(unlayered) == 2  # Attempt status and gradient norm.
        assert all(binding.record_shard_rank == -1 for binding in unlayered)
        assert handle.grad_norm_hook is not None
    finally:
        handle.close()


def test_training_calls_qk_only_at_router_weight_boundaries():
    path = Path(__file__).resolve().parents[1] / "third_party/megatron-lm/megatron/training/training.py"
    tree = ast.parse(path.read_text())
    parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
    for name, router_name in (
        ("emit_qk_weights", "emit_router_weights"),
        ("emit_initial_qk_weights", "emit_initial_router_weights"),
    ):
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Attribute) and node.func.attr == name]
        assert len(calls) == 1
        call = calls[0]
        block = parents[parents[call]]
        assert isinstance(block, ast.If)
        assert router_name in ast.unparse(block)
        if name == "emit_qk_weights":
            assert ast.unparse(block.test) == "dmi_handle is not None"
            optimizer_calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
                               and ast.unparse(n.func) == "optimizer.step"]
            assert any(call.lineno < n.lineno < call.lineno + 10 for n in optimizer_calls)
            assert ast.unparse(call.keywords[0].value) == "int(iteration) + 1"
        else:
            assert ast.unparse(call.keywords[0].value) == "int(iteration)"
