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
    _qk_weight_bindings,
    setup_megatron_dmi,
)
from tests.test_megatron_startup import (
    FakeAdaptor,
    FakeDist,
    FakeEngine,
    FakeParallelState,
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


def _attention(*, heads=8, groups=2, head_dim=2, hidden=7, layer=1, tp=1):
    module = OlmoeSelfAttention()
    module.attention_type = "self"
    module.layer_number = layer
    module.config = SimpleNamespace(num_query_groups=groups * tp, hidden_size=hidden)
    module.num_query_groups_per_partition = groups
    module.num_attention_heads_per_partition = heads
    module.hidden_size_per_attention_head = head_dim
    module.linear_qkv = nn.Module()
    q = torch.arange(heads * head_dim * hidden).reshape(heads * head_dim, hidden).float()
    k = 1000 + torch.arange(groups * head_dim * hidden).reshape(groups * head_dim, hidden).float()
    v = torch.full_like(k, -5000)
    packed = torch.cat((
        q.reshape(groups, -1, hidden),
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


@pytest.mark.parametrize("heads,groups", [(4, 4), (8, 2), (8, 1)])
@pytest.mark.parametrize("selected", [{"q-weights"}, {"k-weights"}, {"q-weights", "k-weights"}])
def test_mha_gqa_mqa_extract_exact_qk_and_refresh(heads, groups, selected):
    model, q, k = _attention(heads=heads, groups=groups)
    hooks, bindings = _qk_weight_bindings(model, rank_ctx=_rank(), selected_hooks=selected)
    assert len(hooks) == len(bindings) == len(selected)
    expected = {"query_projection_weight": q, "key_projection_weight": k}
    old = []
    captured = []
    for hook, parameter in bindings:
        _capture_hook_outputs(hook, captured.append)
        assert hook.spec.preprocess.func is qk_weight_from_fused_qkv
        assert parameter is model.linear_qkv.weight
        hook(parameter)
        value = captured[-1]
        torch.testing.assert_close(value, expected[hook.spec.name])
        assert value.is_contiguous() and not value.requires_grad
        assert value.device == parameter.device
        assert value.numel() < parameter.numel()  # the runtime never receives full QKV
        assert value.untyped_storage().data_ptr() != parameter.untyped_storage().data_ptr()
        old.append(value)
    with torch.no_grad():
        model.linear_qkv.weight.add_(10)
    for (hook, parameter), before in zip(bindings, old):
        torch.testing.assert_close(before, expected[hook.spec.name])
        hook(parameter)
        torch.testing.assert_close(captured[-1], expected[hook.spec.name] + 10)


@pytest.mark.parametrize("enabled,should_emit", [(False, True), (True, False)])
def test_ineligible_hook_does_not_preprocess(enabled, should_emit):
    model, _q, _k = _attention()
    _hooks, ((hook, parameter),) = _qk_weight_bindings(
        model, rank_ctx=_rank(), selected_hooks={"q-weights"},
    )
    captured = []
    _capture_hook_outputs(hook, captured.append, should_emit=should_emit)
    hook.enabled = enabled
    # An invalid input would fail during extraction, so this also verifies the
    # callback is behind the hook/runtime eligibility checks, not in the caller.
    hook(parameter[:0])
    assert captured == []


@pytest.mark.parametrize("tp_rank", [0, 1])
@pytest.mark.parametrize("pp_rank", [0, 1])
def test_tp_shards_keep_global_pp_layer_and_iteration_contract(tp_rank, pp_rank):
    model, _q, _k = _attention(layer=pp_rank + 1, tp=2)
    hooks, bindings = _qk_weight_bindings(
        [model, model],  # shared references must not double-emit
        rank_ctx=_rank(tp_rank=tp_rank, tp_world_size=2, pp_rank=pp_rank, pp_world_size=2),
        selected_hooks={"q-weights", "k-weights"},
    )
    assert len(hooks) == len(bindings) == 2
    for hook_binding in hooks:
        spec = _megatron_hook_spec(hook_binding.hook)
        assert spec.layer_no == pp_rank
        assert spec.name in {"query_projection_weight", "key_projection_weight"}
        assert spec.shard_policy is ShardPolicy.TP_SHARDED
        assert spec.record_type is RecordType.PER_ITERATION
        assert spec.dp_emission is DPEmissionPolicy.DP_RANK_0
        assert hook_binding.hook.hook_phase is HookPhase.ITERATION
        assert spec.outputs[0].transport_type is TransportType.IDENTITY
        assert not spec.need_token_range
        assert hook_binding.record_dp_rank == -1
        assert hook_binding.record_shard_rank == tp_rank


@pytest.mark.parametrize("rank", [_rank(dp_rank=1), _rank(cp_rank=1), _rank(ep_rank=1)])
def test_replicated_non_tp_coordinates_do_not_duplicate_weights(rank):
    model, _q, _k = _attention()
    assert _qk_weight_bindings(model, rank_ctx=rank, selected_hooks={"q-weights"}) == ((), ())


@pytest.mark.parametrize("case,error,match", [
    ("cpu", RuntimeError, "CUDA-resident"),
    ("shape", ValueError, "grouped QKV shape"),
    ("layer", ValueError, "global layer number"),
    ("duplicate", ValueError, "Duplicate local"),
    ("gate", NotImplementedError, "gated attention"),
    ("tp_kv", NotImplementedError, "number of KV heads"),
    ("fp8", NotImplementedError, "non-quantized"),
    ("missing", TypeError, "fused QKV Parameter"),
])
def test_unsupported_weight_layouts_fail_explicitly(case, error, match):
    model, _q, _k = _attention()
    rank = _rank()
    if case == "cpu":
        model.linear_qkv.weight = nn.Parameter(model.linear_qkv.weight.detach())
    elif case == "shape":
        model.linear_qkv.weight = FakeCudaParameter(torch.zeros(7, 7))
    elif case == "layer":
        model.layer_number = 0
    elif case == "duplicate":
        other, _q, _k = _attention()
        model = [model, other]
    elif case == "gate":
        model.config.attention_output_gate = True
    elif case == "tp_kv":
        rank = _rank(tp_world_size=4)
    elif case == "fp8":
        model.config.fp8 = "hybrid"
    elif case == "missing":
        del model.linear_qkv
    with pytest.raises(error, match=match):
        _qk_weight_bindings(model, rank_ctx=rank, selected_hooks={"q-weights", "k-weights"})


@pytest.mark.parametrize("selection", ["q-weights", "k-weights", "q-weights,k-weights"])
@pytest.mark.parametrize("restriction", ["dp", "param_gather"])
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
        assert len(handle.qk_weight_bindings) == 2
        assert handle.router_weight_bindings == ()
        hook_inputs = []
        for hook, parameter in handle.qk_weight_bindings:
            assert parameter is model.linear_qkv.weight
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
        assert all(value is model.linear_qkv.weight for value in hook_inputs)
        for idx, (name, context, value) in enumerate(observed):
            assert context.global_batch_id == (0, 600000, 600001)[idx // 2]
            assert context.microbatch_id == -1 and context.dataset_ids == ()
            assert context.direction == "iter" and context.phase == "train"
            expected = q if name == "query_projection_weight" else k
            torch.testing.assert_close(value, expected + (5 if idx >= 4 else 0))
        assert handle.adaptor.context is None
        with pytest.raises(ValueError, match="must be >= 1"):
            handle.emit_qk_weights(model_state_iteration_id=0)
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
            assert ast.unparse(block.test) == "dmi_handle is not None and update_successful"
            assert ast.unparse(call.keywords[0].value) == "int(iteration) + 1"
        else:
            assert ast.unparse(call.keywords[0].value) == "int(iteration)"
