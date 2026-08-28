from __future__ import annotations

from dataclasses import replace

import pytest
import torch
from torch import nn

from dmi_megatron_integration.adapter import (
    InvocationIdAllocator,
    MegatronAdaptor,
    MegatronFullIterationPlan,
    MegatronHookBinding,
    MegatronTrainingContext,
)
from dmi_megatron_integration.hooks.specs import (
    DPEmissionPolicy,
    DimSpec,
    HookInputLayout,
    HookPhase,
    HookRuntimeMode,
    MegatronHookSpec,
    MegatronMetadataField,
    MegatronOutputSpec,
)
from dmi_megatron_integration.metadata_context import (
    DMIMetadataContext,
    segment_metadata_field_spec,
    valid_count_field_spec,
)
from dmi.api.v1 import (
    HookPointV1,
    HookSpecV1,
    OutputStorage,
    RingCapacities,
    RecordType,
    StepReservation,
    TransportSpec,
    TransportType,
    align_up,
)


FIRST_RECORD_OUTPUT_ID = 1 << 16


def _allocate_invocation(
    allocator: InvocationIdAllocator,
    **overrides,
) -> int:
    values = {
        "model_id": "train-run",
        "phase": "train",
        "global_batch_id": 7,
        "attempt_id": 0,
        "dp_rank": 0,
        "microbatch_id": 1,
        "direction": "fwd",
        "output_id": FIRST_RECORD_OUTPUT_ID,
        "layer_no": 3,
        "shard_rank": 0,
        "token_start": 0,
        "record_type": RecordType.PER_SAMPLE,
    }
    values.update(overrides)
    return allocator.allocate(**values)


class FakeRecordRuntime:
    """Adapter-test double for the public RecordRuntime boundary."""

    def __init__(self, replay_result: StepReservation = StepReservation.RESERVED) -> None:
        self.replay_result = StepReservation(replay_result)
        self.emit_calls = []
        self.replay_calls = []
        self.published_replay_metadata = []
        self.reservation_calls = []
        self.bind_calls = []
        self.dispatch_calls = []
        self._next_output_id = FIRST_RECORD_OUTPUT_ID
        self._output_ids_by_name = {}

    def bind_hook(
        self,
        hook,
        *,
        hook_runtime,
        gate_tensor=None,
        gate_value=0,
    ) -> None:
        output_ids = []
        for output in hook.spec.outputs:
            output_id = self._output_ids_by_name.get(output.name)
            if output_id is None:
                output_id = self._next_output_id
                self._next_output_id += 1
                self._output_ids_by_name[output.name] = output_id
            output_ids.append(output_id)
        hook._output_ids = tuple(output_ids)
        hook._ring_payload = torch.empty(16, dtype=torch.uint8)
        hook._hook_runtime = hook_runtime
        hook._gate_tensor = gate_tensor
        hook._gate_value = int(gate_value)
        hook._dispatch = lambda spec, output: self.dispatch_calls.append(
            (spec, output)
        )
        self.bind_calls.append((hook, tuple(output_ids), gate_tensor, int(gate_value)))

    def emit_output(self, entry, metadata, output):
        self.emit_calls.append((entry, metadata, output))
        self.reservation_calls.append((entry.aligned_reservation_bytes, 1))
        # Adapter tests do not launch the public native producer. OVERSIZED is
        # the public signal that the runtime completed the output directly.
        return StepReservation.OVERSIZED

    def prepare_replay(self, plan, metadata):
        metadata = tuple(metadata)
        self.replay_calls.append((plan, metadata))
        self.reservation_calls.append(
            (plan.total_reservation_bytes, plan.task_count)
        )
        if self.replay_result is not StepReservation.OVERSIZED:
            self.published_replay_metadata.append(metadata)
        return self.replay_result


class FakeEngine:
    def __init__(
        self,
        replay_result: StepReservation = StepReservation.RESERVED,
    ) -> None:
        self.record_runtime = FakeRecordRuntime(replay_result)
        self.capture_enabled_calls = []
        self.payload_cap = 1 << 30
        self.staging_cap = 1 << 30
        self.task_cap = 1 << 20

    def set_capture_enabled(self, enabled: bool) -> None:
        self.capture_enabled_calls.append(bool(enabled))

    def ring_capacities(self) -> RingCapacities:
        return RingCapacities(
            payload_bytes=self.payload_cap,
            staging_bytes=self.staging_cap,
            task_entries=self.task_cap,
        )


def _make_adaptor(
    engine: FakeEngine,
    model_id: str,
    *,
    dims=None,
) -> MegatronAdaptor:
    return MegatronAdaptor(
        engine,
        engine.record_runtime,
        model_id,
        dims=dims,
    )


def _make_hook(
    policy: MegatronHookSpec,
    *,
    hook_phase: HookPhase = HookPhase.FWD,
    suppress_recompute: bool = True,
) -> HookPointV1:
    hook = HookPointV1()
    hook._dmi_megatron_spec = policy
    hook.hook_phase = hook_phase
    hook.suppress_recompute = bool(suppress_recompute)
    return hook


class TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.router0 = _make_hook(
            MegatronHookSpec(
                name="router_summary",
                layer_no=0,
                outputs=[
                    MegatronOutputSpec(
                        name="router_probs_mean",
                        input_shape=[DimSpec.BATCH, DimSpec.NUM_EXPERTS],
                        dtype=torch.float32,
                    )
                ],
                preprocess=lambda x: x,
                enabled_by=frozenset({"router-summary"}),
            )
        )
        self.router1 = _make_hook(
            MegatronHookSpec(
                name="router_summary",
                layer_no=1,
                outputs=[
                    MegatronOutputSpec(
                        name="router_entropy",
                        input_shape=[DimSpec.BATCH],
                        dtype=torch.float32,
                    )
                ],
                preprocess=lambda x: x,
                enabled_by=frozenset({"other"}),
            )
        )


class TinyMultiHookModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.a = _make_hook(
            MegatronHookSpec(
                name="summary_a",
                layer_no=0,
                outputs=[
                    MegatronOutputSpec(
                        name="summary_a",
                        input_shape=[DimSpec.BATCH, DimSpec.NUM_EXPERTS],
                        dtype=torch.float32,
                    )
                ],
                preprocess=lambda x: x,
                enabled_by=frozenset({"full"}),
            )
        )
        self.b = _make_hook(
            MegatronHookSpec(
                name="summary_b",
                layer_no=0,
                outputs=[
                    MegatronOutputSpec(
                        name="summary_b",
                        input_shape=[DimSpec.BATCH],
                        dtype=torch.float32,
                    )
                ],
                preprocess=lambda x: x,
                enabled_by=frozenset({"full"}),
            )
        )


class TinyNoRecomputeSuppressModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.loss = _make_hook(
            MegatronHookSpec(
                name="loss_summary",
                layer_no=-1,
                outputs=[
                    MegatronOutputSpec(
                        name="loss_summary",
                        input_shape=[DimSpec.BATCH, 1],
                        dtype=torch.float32,
                    )
                ],
                preprocess=lambda x: x,
                enabled_by=frozenset({"loss-summary"}),
                need_token_range=False,
            ),
            suppress_recompute=False,
        )


class TinyPackedIdentityModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.raw = _make_hook(
            MegatronHookSpec(
                name="raw",
                layer_no=-1,
                outputs=[
                    MegatronOutputSpec(
                        name="raw",
                        input_shape=[DimSpec.BATCH, DimSpec.SEQ, 4],
                        dtype=torch.float32,
                    )
                ],
                enabled_by=frozenset({"raw"}),
                need_token_range=False,
                supported_layouts=frozenset(
                    {HookInputLayout.SEQ_BATCH, HookInputLayout.PACKED_SEGMENTED}
                ),
            )
        )


class TinyValidCountBindingModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.router0 = _make_hook(
            MegatronHookSpec(
                name="valid_count_binding",
                layer_no=0,
                outputs=[
                    MegatronOutputSpec(
                        name="valid_count_binding",
                        input_shape=[DimSpec.BATCH, DimSpec.NUM_EXPERTS],
                        dtype=torch.float32,
                    )
                ],
                preprocess_metadata_fields=frozenset(
                    {MegatronMetadataField.VALID_COUNT}
                ),
                enabled_by=frozenset({"binding"}),
                need_token_range=False,
            )
        )


def test_attach_model_selects_policy_and_binds_public_record_ids():
    engine = FakeEngine()
    model = TinyModel()
    current_phase = torch.tensor(HookPhase.FWD.value, dtype=torch.int32)
    adaptor = _make_adaptor(
        engine,
        "train-run",
        dims={DimSpec.BATCH: 2, DimSpec.NUM_EXPERTS: 4},
    )

    adaptor.attach_model(
        model,
        hook_selection="router-summary",
        current_phase_tensor=current_phase,
    )

    assert len(adaptor.configured_hooks) == 1
    assert len(engine.record_runtime.bind_calls) == 1
    assert model.router0._output_ids == (FIRST_RECORD_OUTPUT_ID,)
    assert isinstance(model.router0._ring_payload, torch.Tensor)
    assert model.router0._hook_runtime is adaptor.hook_runtime
    assert model.router0._gate_tensor is current_phase
    assert model.router0.hook_phase is HookPhase.FWD
    assert model.router1._output_ids == ()


def test_attach_model_respects_hook_recompute_suppression_policy():
    engine = FakeEngine()
    model = TinyNoRecomputeSuppressModel()
    current_phase = torch.tensor(HookPhase.FWD.value, dtype=torch.int32)
    adaptor = _make_adaptor(
        engine,
        "train-run",
        dims={DimSpec.BATCH: 2},
    )

    adaptor.attach_model(
        model,
        hook_selection="loss-summary",
        current_phase_tensor=current_phase,
    )

    assert len(adaptor.configured_hooks) == 1
    assert model.loss._gate_tensor is None


def test_packed_identity_hook_does_not_bind_unused_segment_metadata():
    engine = FakeEngine()
    model = TinyPackedIdentityModel()
    metadata = DMIMetadataContext(
        max_num_microbatches=1,
        max_batch_size=2,
        num_scopes=1,
        field_specs=(),
        device="cpu",
    )
    adaptor = _make_adaptor(
        engine,
        "train-run",
        dims={DimSpec.BATCH: 2, DimSpec.SEQ: 8},
    )

    adaptor.attach_model(
        model,
        hook_selection="raw",
        metadata_context=metadata,
        active_input_layout=HookInputLayout.PACKED_SEGMENTED,
    )

    assert not hasattr(model.raw, "sample_start_ptr_fwd")
    assert not hasattr(model.raw, "sample_end_ptr_fwd")


def test_iteration_hook_emits_one_unsplit_record_with_semantic_coordinates():
    engine = FakeEngine()
    adaptor = _make_adaptor(engine, "train-run", dims={})
    hook = _make_hook(
        MegatronHookSpec(
            name="grad_norm",
            layer_no=-1,
            outputs=[
                MegatronOutputSpec(
                    name="grad_norm",
                    input_shape=[1],
                    dtype=torch.float32,
                    storage=OutputStorage.SCALAR_FLOAT,
                )
            ],
            need_token_range=False,
            record_type=RecordType.PER_ITERATION,
            dp_emission=DPEmissionPolicy.DP_RANK_0,
        ),
        hook_phase=HookPhase.ITERATION,
    )
    adaptor.attach_hooks(
        model_hooks=(),
        iteration_hooks=(
            MegatronHookBinding(
                hook=hook,
                record_dp_rank=-1,
                record_shard_rank=-1,
            ),
        ),
    )
    adaptor.set_current_iteration(
        MegatronTrainingContext(
            global_batch_id=7,
            microbatch_id=-1,
            valid_counts=(),
            direction="iter",
            phase="train",
            dp_rank=-1,
            shard_rank=-1,
        )
    )

    hook(torch.tensor([3.5], dtype=torch.float32))

    entry, metadata, output = engine.record_runtime.emit_calls[0]
    assert metadata.direction == "iter"
    assert metadata.global_batch_id == 7
    assert metadata.dp_rank == -1
    assert metadata.microbatch_id == -1
    assert metadata.layer_no == -1
    assert metadata.shard_rank == -1
    assert metadata.attempt_id == 0
    assert metadata.invocation_id == 0
    assert metadata.valid_counts == ()
    assert metadata.dataset_ids == ()
    assert entry.output_shape == (1,)
    assert entry.storage is OutputStorage.SCALAR_FLOAT
    assert entry.record_type is RecordType.PER_ITERATION
    assert torch.equal(output.tensor, torch.tensor([3.5]))


def test_iteration_hook_requires_explicit_iteration_context():
    engine = FakeEngine()
    adaptor = _make_adaptor(engine, "train-run", dims={})
    hook = _make_hook(
        MegatronHookSpec(
            name="iteration_tensor",
            layer_no=3,
            outputs=[
                MegatronOutputSpec(
                    name="iteration_tensor",
                    input_shape=[2],
                    dtype=torch.float32,
                )
            ],
            need_token_range=False,
            record_type=RecordType.PER_ITERATION,
            dp_emission=DPEmissionPolicy.DP_RANK_0,
        ),
        hook_phase=HookPhase.ITERATION,
    )
    adaptor.attach_hooks(
        model_hooks=(),
        iteration_hooks=(
            MegatronHookBinding(hook=hook, record_dp_rank=-1, record_shard_rank=0),
        ),
    )

    with pytest.raises(RuntimeError, match="explicit context"):
        hook(torch.ones(2))


def test_execution_hook_emits_one_unsplit_record_without_source_coordinates():
    engine = FakeEngine()
    adaptor = _make_adaptor(
        engine,
        "train-run",
        dims={DimSpec.HIDDEN: 3},
    )
    hook = _make_hook(
        MegatronHookSpec(
            name="moe_inverse_map",
            layer_no=2,
            outputs=[
                MegatronOutputSpec(
                    name="moe_inverse_map",
                    input_shape=[DimSpec.ACTUAL_TOKEN_PACKED],
                    dtype=torch.int64,
                    transport_type=TransportType.IDENTITY,
                )
            ],
            need_token_range=False,
            record_type=RecordType.PER_EXECUTION,
        ),
        hook_phase=HookPhase.FWD,
    )
    adaptor.attach_hooks(
        model_hooks=(
            MegatronHookBinding(
                hook=hook,
                record_dp_rank=-1,
                record_shard_rank=4,
            ),
        ),
        iteration_hooks=(),
    )
    adaptor.set_current_event(
        MegatronTrainingContext(
            global_batch_id=7,
            microbatch_id=3,
            valid_counts=[5, 2],
            direction="fwd",
            dp_rank=1,
            shard_rank=4,
            token_start=11,
        )
    )

    payload = torch.tensor([2, 0, 3, 1], dtype=torch.int64)
    hook(payload)

    expected_bytes = align_up(payload.numel() * payload.element_size(), 16)
    assert engine.record_runtime.reservation_calls == [(expected_bytes, 1)]
    entry, metadata, output = engine.record_runtime.emit_calls[0]
    assert metadata.direction == "fwd"
    assert metadata.global_batch_id == 7
    assert metadata.dp_rank == -1
    assert metadata.microbatch_id == 3
    assert metadata.layer_no == 2
    assert metadata.shard_rank == 4
    assert metadata.token_start == -1
    assert metadata.valid_counts == ()
    assert metadata.dataset_ids == ()
    assert entry.input_shape == (4,)
    assert entry.output_shape == (4,)
    assert entry.dtype is torch.int64
    assert entry.transport_type is TransportType.IDENTITY
    assert entry.record_type is RecordType.PER_EXECUTION
    assert output.tensor is payload


def test_execution_hook_requires_explicit_physical_rank_coordinates():
    adaptor = _make_adaptor(FakeEngine(), "train-run", dims={})
    hook = _make_hook(
        MegatronHookSpec(
            name="moe_inverse_map",
            layer_no=2,
            outputs=[
                MegatronOutputSpec(
                    name="moe_inverse_map",
                    input_shape=[DimSpec.ACTUAL_TOKEN_PACKED],
                    dtype=torch.int64,
                    transport_type=TransportType.IDENTITY,
                )
            ],
            need_token_range=False,
            record_type=RecordType.PER_EXECUTION,
        ),
        hook_phase=HookPhase.FWD,
    )

    with pytest.raises(ValueError, match="explicit physical rank coordinates"):
        adaptor.attach_hooks(
            model_hooks=(MegatronHookBinding(hook=hook, record_shard_rank=4),),
            iteration_hooks=(),
        )


def test_hook_spec_v1_has_no_direction_contract():
    spec = HookSpecV1(
        name="summary",
        outputs=(
            TransportSpec(
                name="summary",
                output_shape=(2,),
            ),
        ),
    )

    assert not hasattr(spec, "direction")
    assert not hasattr(spec, "active_for_direction")


@pytest.mark.parametrize(
    ("transport_type", "input_shape", "output_shape", "expected_output_shape"),
    [
        (
            TransportType.IDENTITY,
            (DimSpec.BATCH, DimSpec.SEQ, 2),
            (DimSpec.BATCH, DimSpec.SEQ, 2),
            None,
        ),
        (
            TransportType.PREFIX_STRIP,
            (DimSpec.BATCH, DimSpec.SEQ, 2),
            (DimSpec.BATCH, DimSpec.SEQ, 2),
            None,
        ),
        (
            TransportType.CHUNKED,
            (DimSpec.BATCH, DimSpec.SEQ, 2),
            (DimSpec.BATCH, DimSpec.SEQ, 2),
            None,
        ),
        (
            TransportType.SEQ_PREFIX_PACK,
            (DimSpec.SEQ, DimSpec.BATCH, 2),
            (DimSpec.ACTUAL_TOKEN_PACKED, 2),
            (-1, 2),
        ),
        (
            TransportType.SEGMENTED_PACK,
            (DimSpec.BATCH, 2),
            (DimSpec.ACTUAL_TOKEN_PACKED, 2),
            (-1, 2),
        ),
    ],
)
def test_resolved_output_shape_matches_presplit_runtime_ownership(
    transport_type,
    input_shape,
    output_shape,
    expected_output_shape,
):
    spec = MegatronOutputSpec(
        name="shape_contract",
        input_shape=input_shape,
        output_shape=output_shape,
        dtype=torch.float32,
        transport_type=transport_type,
        row_bytes=8 if transport_type is TransportType.PREFIX_STRIP else None,
    )

    resolved = spec.resolve(
        {DimSpec.BATCH: 1, DimSpec.SEQ: 16},
        record_type=RecordType.PER_SAMPLE,
    )

    assert resolved.output_shape == expected_output_shape


def test_prepare_immediate_output_pushes_one_metadata_row_for_current_event():
    engine = FakeEngine()
    model = TinyModel()
    adaptor = _make_adaptor(
        engine,
        "train-run",
        dims={DimSpec.BATCH: 2, DimSpec.NUM_EXPERTS: 4},
    )
    adaptor.attach_model(model, hook_selection="router-summary")
    adaptor.set_current_event(
        MegatronTrainingContext(
            global_batch_id=7,
            microbatch_id=3,
            valid_counts=[5, 2],
            direction="fwd",
            dp_rank=1,
            shard_rank=0,
            token_start=0,
        )
    )

    model.router0(torch.ones(2, 4))

    expected_bytes = align_up(
        2 * 4 * torch.empty((), dtype=torch.float32).element_size(), 16
    )
    assert engine.record_runtime.reservation_calls == [(expected_bytes, 1)]
    assert len(engine.record_runtime.emit_calls) == 1
    entry, metadata, _output = engine.record_runtime.emit_calls[0]
    assert entry.output_id == FIRST_RECORD_OUTPUT_ID
    assert entry.output_shape == (2, 4)
    assert entry.dtype is torch.float32
    assert entry.storage is OutputStorage.TENSOR
    assert entry.transport_type is TransportType.IDENTITY
    assert entry.record_type is RecordType.PER_SAMPLE
    assert metadata.direction == "fwd"
    assert metadata.phase == "train"
    assert metadata.global_batch_id == 7
    assert metadata.dp_rank == 1
    assert metadata.microbatch_id == 3
    assert metadata.layer_no == 0
    assert metadata.shard_rank == 0
    assert metadata.token_start == 0
    assert metadata.attempt_id == 0
    assert metadata.invocation_id == 0
    assert metadata.valid_counts == (5, 2)
    assert metadata.dataset_ids == ()
    assert metadata.model_id == "train-run"


def test_eager_phase_filter_queues_only_eligible_hook():
    engine = FakeEngine()
    model = TinyMultiHookModel()
    model.b.hook_phase = HookPhase.BWD
    adaptor = _make_adaptor(
        engine,
        "train-run",
        dims={DimSpec.BATCH: 2, DimSpec.NUM_EXPERTS: 4},
    )
    adaptor.attach_model(model, hook_selection="full")
    adaptor.set_current_event(
        MegatronTrainingContext(
            global_batch_id=0,
            microbatch_id=0,
            valid_counts=[6, 4],
            direction="bwd",
        )
    )

    model.b(torch.ones(2))
    model.a(torch.ones(2, 4))

    assert len(engine.record_runtime.emit_calls) == 1
    entry, metadata, _output = engine.record_runtime.emit_calls[0]
    assert entry.output_id == FIRST_RECORD_OUTPUT_ID + 1
    assert metadata.direction == "bwd"
    assert metadata.phase == "train"
    assert engine.record_runtime.reservation_calls == [(16, 1)]


def test_eager_non_suppressing_hook_emits_in_backward_event():
    engine = FakeEngine()
    model = TinyNoRecomputeSuppressModel()
    adaptor = _make_adaptor(engine, "train-run", dims={DimSpec.BATCH: 2})
    adaptor.attach_model(model, hook_selection="loss-summary")
    adaptor.set_current_event(
        MegatronTrainingContext(
            global_batch_id=3,
            microbatch_id=1,
            valid_counts=[1, 1],
            direction="bwd",
        )
    )

    model.loss(torch.ones(2, 1))

    assert len(engine.record_runtime.emit_calls) == 1
    entry, metadata, _output = engine.record_runtime.emit_calls[0]
    assert entry.output_id == FIRST_RECORD_OUTPUT_ID
    assert metadata.direction == "fwd"


def test_prepare_immediate_output_raises_without_active_event():
    engine = FakeEngine()
    model = TinyModel()
    adaptor = _make_adaptor(
        engine,
        "train-run",
        dims={DimSpec.BATCH: 2, DimSpec.NUM_EXPERTS: 4},
    )
    adaptor.attach_model(model, hook_selection="router-summary")

    with pytest.raises(RuntimeError, match="active schedule event"):
        model.router0(torch.ones(2, 4))

    assert engine.record_runtime.reservation_calls == []
    assert engine.record_runtime.emit_calls == []


def test_prepare_immediate_output_accepts_public_cpu_direct_completion():
    engine = FakeEngine()
    model = TinyModel()
    adaptor = _make_adaptor(
        engine,
        "train-run",
        dims={DimSpec.BATCH: 2, DimSpec.NUM_EXPERTS: 4},
    )
    adaptor.attach_model(model, hook_selection="router-summary")
    adaptor.set_current_event(
        MegatronTrainingContext(
            global_batch_id=0,
            microbatch_id=0,
            valid_counts=[6, 4],
        )
    )

    model.router0(torch.ones(2, 4))

    assert len(engine.record_runtime.emit_calls) == 1
    assert engine.record_runtime.reservation_calls == [(32, 1)]


def test_eager_retained_recompute_uses_semantic_fwd_direction_and_invocation_ids():
    engine = FakeEngine()
    model = TinyNoRecomputeSuppressModel()
    adaptor = _make_adaptor(engine, "train-run", dims={DimSpec.BATCH: 2})
    adaptor.attach_model(
        model,
        hook_selection="loss-summary",
        activation_recompute_enabled=True,
    )
    adaptor.begin_attempt(phase="train", global_batch_id=5, attempt_id=0)

    for schedule_direction in ("fwd", "bwd"):
        adaptor.set_current_event(
            MegatronTrainingContext(
                global_batch_id=5,
                microbatch_id=1,
                valid_counts=(1, 1),
                attempt_id=0,
                direction=schedule_direction,
                phase="train",
            )
        )
        model.loss(torch.ones(2, 1))

    calls = engine.record_runtime.emit_calls
    assert len(calls) == 2
    assert [metadata.direction for _entry, metadata, _output in calls] == [
        "fwd",
        "fwd",
    ]
    assert [metadata.attempt_id for _entry, metadata, _output in calls] == [0, 0]
    assert [metadata.invocation_id for _entry, metadata, _output in calls] == [0, 1]
    adaptor.end_attempt(attempt_id=0)


def test_capture_warmup_records_plan_without_metadata_push():
    engine = FakeEngine()
    model = TinyModel()
    adaptor = _make_adaptor(
        engine,
        "train-run",
        dims={DimSpec.BATCH: 2, DimSpec.NUM_EXPERTS: 4},
    )
    adaptor.attach_model(model, hook_selection="router-summary")

    adaptor.begin_capture_plan(warmup_enabled=True)
    model.router0(torch.ones(2, 4))
    plan = adaptor.finish_capture_plan()

    assert adaptor.hook_runtime.mode == HookRuntimeMode.EAGER_IMMEDIATE
    assert engine.capture_enabled_calls == [False, True]
    assert plan.task_count == 1
    assert plan.entries[0].output_id == FIRST_RECORD_OUTPUT_ID
    assert adaptor._plan_semantics(plan)[0].act_name == "router_probs_mean"
    assert plan.entries[0].input_shape == (2, 4)
    assert plan.entries[0].output_shape == (2, 4)
    assert engine.record_runtime.reservation_calls == []
    assert engine.record_runtime.emit_calls == []


def test_capture_without_warmup_records_and_pushes_metadata():
    engine = FakeEngine()
    model = TinyModel()
    adaptor = _make_adaptor(
        engine,
        "train-run",
        dims={DimSpec.BATCH: 2, DimSpec.NUM_EXPERTS: 4},
    )
    adaptor.attach_model(model, hook_selection="router-summary")
    adaptor.set_current_event(
        MegatronTrainingContext(
            global_batch_id=11,
            microbatch_id=2,
            valid_counts=[5, 3],
            direction="fwd",
        )
    )

    adaptor.begin_capture_plan(
        warmup_enabled=False,
        capture_direction=HookPhase.FWD,
    )
    model.router0(torch.ones(2, 4))
    plan = adaptor.finish_capture_plan()

    expected_bytes = align_up(
        2 * 4 * torch.empty((), dtype=torch.float32).element_size(), 16
    )
    assert engine.capture_enabled_calls == [True, True]
    assert plan.task_count == 1
    assert engine.record_runtime.reservation_calls == [(expected_bytes, 1)]
    assert len(engine.record_runtime.emit_calls) == 1
    entry, metadata, _output = engine.record_runtime.emit_calls[0]
    assert entry.output_id == FIRST_RECORD_OUTPUT_ID
    assert metadata.direction == "fwd"
    assert metadata.phase == "train"
    assert metadata.global_batch_id == 11


def test_capture_plan_follows_actual_hook_call_order():
    engine = FakeEngine()
    model = TinyMultiHookModel()
    adaptor = _make_adaptor(
        engine,
        "train-run",
        dims={DimSpec.BATCH: 2, DimSpec.NUM_EXPERTS: 4},
    )
    adaptor.attach_model(model, hook_selection="full")

    adaptor.begin_capture_plan(warmup_enabled=True)
    model.b(torch.ones(2))
    model.a(torch.ones(2, 4))
    plan = adaptor.finish_capture_plan()

    assert [item.act_name for item in adaptor._plan_semantics(plan)] == [
        "summary_b",
        "summary_a",
    ]
    assert [entry.output_id for entry in plan.entries] == [
        FIRST_RECORD_OUTPUT_ID + 1,
        FIRST_RECORD_OUTPUT_ID,
    ]


def test_local_capture_keeps_only_hooks_eligible_for_capture_direction():
    engine = FakeEngine()
    model = TinyMultiHookModel()
    model.b.hook_phase = HookPhase.BWD
    adaptor = _make_adaptor(
        engine,
        "train-run",
        dims={DimSpec.BATCH: 2, DimSpec.NUM_EXPERTS: 4},
    )
    adaptor.attach_model(model, hook_selection="full")

    adaptor.begin_capture_plan(
        warmup_enabled=True,
        capture_direction=HookPhase.FWD,
    )
    model.b(torch.ones(2))
    model.a(torch.ones(2, 4))
    fwd_plan = adaptor.finish_capture_plan()

    adaptor.begin_capture_plan(
        warmup_enabled=True,
        capture_direction=HookPhase.BWD,
    )
    model.a(torch.ones(2, 4))
    model.b(torch.ones(2))
    bwd_plan = adaptor.finish_capture_plan()

    assert [item.act_name for item in adaptor._plan_semantics(fwd_plan)] == [
        "summary_a"
    ]
    assert [item.act_name for item in adaptor._plan_semantics(bwd_plan)] == [
        "summary_b"
    ]


def test_local_capture_keeps_non_suppressing_hook_in_backward_plan():
    engine = FakeEngine()
    model = TinyNoRecomputeSuppressModel()
    adaptor = _make_adaptor(engine, "train-run", dims={DimSpec.BATCH: 2})
    adaptor.attach_model(model, hook_selection="loss-summary")

    adaptor.begin_capture_plan(
        warmup_enabled=True,
        capture_direction=HookPhase.BWD,
    )
    model.loss(torch.ones(2, 1))
    plan = adaptor.finish_capture_plan()

    assert plan.task_count == 1
    assert adaptor._plan_semantics(plan)[0].suppress_recompute is False


def test_full_iteration_capture_records_event_coordinates():
    engine = FakeEngine()
    model = TinyModel()
    model.router0.hook_phase = HookPhase.BWD
    adaptor = _make_adaptor(
        engine,
        "train-run",
        dims={DimSpec.BATCH: 2, DimSpec.NUM_EXPERTS: 4},
    )
    adaptor.attach_model(model, hook_selection="router-summary")
    adaptor.set_current_event(
        MegatronTrainingContext(
            global_batch_id=3,
            microbatch_id=1,
            valid_counts=[5, 3],
            direction="bwd",
            scope_id=2,
            dp_rank=4,
            shard_rank=7,
            token_start=9,
        )
    )

    adaptor.begin_capture_plan(warmup_enabled=True, capture_event_context=True)
    model.router0(torch.ones(2, 4))
    plan = adaptor.finish_capture_plan()

    assert isinstance(plan, MegatronFullIterationPlan)
    assert plan.output_count == 1
    event = plan.entries[0].event
    assert event.direction == "bwd"
    assert event.microbatch_id == 1
    assert event.scope_id == 2
    assert event.dp_rank == 4
    assert event.shard_rank == 7
    assert event.token_start == 9
    assert not hasattr(event, "valid_counts")


def test_full_iteration_retained_recompute_allocates_in_captured_producer_order():
    engine = FakeEngine()
    model = TinyNoRecomputeSuppressModel()
    adaptor = _make_adaptor(engine, "train-run", dims={DimSpec.BATCH: 2})
    adaptor.attach_model(
        model,
        hook_selection="loss-summary",
        activation_recompute_enabled=True,
    )
    adaptor.begin_capture_plan(warmup_enabled=True, capture_event_context=True)
    for schedule_direction in ("fwd", "bwd"):
        adaptor.set_current_event(
            MegatronTrainingContext(
                global_batch_id=5,
                microbatch_id=0,
                valid_counts=(1, 1),
                attempt_id=0,
                direction=schedule_direction,
                phase="train",
            )
        )
        model.loss(torch.ones(2, 1))
    plan = adaptor.finish_capture_plan()
    assert isinstance(plan, MegatronFullIterationPlan)
    assert len(plan.entries) == 2
    assert [item.record_direction for item in plan._semantics] == ["fwd", "fwd"]
    assert [entry.event.direction for entry in plan.entries] == ["fwd", "bwd"]

    adaptor.begin_attempt(phase="train", global_batch_id=5, attempt_id=0)
    contexts = [
        entry.event.to_context(
            global_batch_id=5,
            valid_counts=(1, 1),
            attempt_id=0,
        )
        for entry in plan.entries
    ]
    assert adaptor.prepare_full_iteration_replay(
        plan,
        contexts,
    ) is StepReservation.RESERVED
    _replay_plan, metadata = engine.record_runtime.replay_calls[0]
    assert [item.direction for item in metadata] == ["fwd", "fwd"]
    assert [item.invocation_id for item in metadata] == [0, 1]
    adaptor.end_attempt(attempt_id=0)


def test_full_iteration_replay_pushes_per_entry_current_contexts():
    engine = FakeEngine()
    model = TinyMultiHookModel()
    model.b.hook_phase = HookPhase.BWD
    adaptor = _make_adaptor(
        engine,
        "train-run",
        dims={DimSpec.BATCH: 2, DimSpec.NUM_EXPERTS: 4},
    )
    adaptor.attach_hooks(
        model_hooks=[
            MegatronHookBinding(hook=model.a, record_shard_rank=5),
            MegatronHookBinding(hook=model.b, record_shard_rank=7),
        ],
        iteration_hooks=[],
    )

    adaptor.set_current_event(
        MegatronTrainingContext(global_batch_id=0, microbatch_id=0, valid_counts=[1, 1])
    )
    adaptor.begin_capture_plan(warmup_enabled=True, capture_event_context=True)
    model.a(torch.ones(2, 4))
    adaptor.set_current_event(
        MegatronTrainingContext(
            global_batch_id=0,
            microbatch_id=1,
            valid_counts=[1, 1],
            direction="bwd",
            scope_id=3,
        )
    )
    model.b(torch.ones(2))
    plan = adaptor.finish_capture_plan()

    contexts = [
        plan.entries[0].event.to_context(global_batch_id=9, valid_counts=[6, 5]),
        plan.entries[1].event.to_context(global_batch_id=9, valid_counts=[4, 3]),
    ]
    decision = adaptor.prepare_full_iteration_replay(plan, contexts)

    assert decision is StepReservation.RESERVED
    assert engine.record_runtime.reservation_calls == [
        (plan.total_aligned_nbytes, plan.output_count)
    ]
    replay_plan, metadata = engine.record_runtime.replay_calls[0]
    assert [entry.output_id for entry in replay_plan.entries] == [
        FIRST_RECORD_OUTPUT_ID,
        FIRST_RECORD_OUTPUT_ID + 1,
    ]
    assert [item.direction for item in metadata] == ["fwd", "bwd"]
    assert [item.phase for item in metadata] == ["train", "train"]
    assert [item.global_batch_id for item in metadata] == [9, 9]
    assert [item.microbatch_id for item in metadata] == [0, 1]
    assert [item.shard_rank for item in metadata] == [5, 7]
    assert [item.attempt_id for item in metadata] == [0, 0]
    assert [item.invocation_id for item in metadata] == [0, 0]
    assert [item.valid_counts for item in metadata] == [(6, 5), (4, 3)]
    assert [item.dataset_ids for item in metadata] == [(), ()]
    assert [entry.output_shape for entry in replay_plan.entries] == [(2, 4), (2,)]
    assert [entry.record_type for entry in replay_plan.entries] == [
        RecordType.PER_SAMPLE,
        RecordType.PER_SAMPLE,
    ]
    assert [item.model_id for item in metadata] == ["train-run", "train-run"]


def test_full_iteration_replay_fallback_does_not_publish_metadata():
    engine = FakeEngine(replay_result=StepReservation.OVERSIZED)
    model = TinyModel()
    adaptor = _make_adaptor(
        engine,
        "train-run",
        dims={DimSpec.BATCH: 2, DimSpec.NUM_EXPERTS: 4},
    )
    adaptor.attach_model(model, hook_selection="router-summary")
    adaptor.set_current_event(
        MegatronTrainingContext(global_batch_id=0, microbatch_id=0, valid_counts=[1, 1])
    )
    adaptor.begin_capture_plan(warmup_enabled=True, capture_event_context=True)
    model.router0(torch.ones(2, 4))
    plan = adaptor.finish_capture_plan()

    decision = adaptor.prepare_full_iteration_replay(
        plan,
        [plan.entries[0].event.to_context(global_batch_id=1, valid_counts=[7, 2])],
    )

    assert decision is StepReservation.OVERSIZED
    assert engine.record_runtime.reservation_calls == [
        (plan.total_aligned_nbytes, plan.output_count)
    ]
    assert engine.record_runtime.published_replay_metadata == []


def test_prepare_replay_reserves_and_pushes_plan_metadata_in_plan_order():
    engine = FakeEngine()
    model = TinyMultiHookModel()
    model.a.hook_phase = HookPhase.BWD
    model.b.hook_phase = HookPhase.BWD
    adaptor = _make_adaptor(
        engine,
        "train-run",
        dims={DimSpec.BATCH: 2, DimSpec.NUM_EXPERTS: 4},
    )
    adaptor.attach_hooks(
        model_hooks=[
            MegatronHookBinding(hook=model.a, record_shard_rank=5),
            MegatronHookBinding(hook=model.b, record_shard_rank=7),
        ],
        iteration_hooks=[],
    )

    adaptor.begin_capture_plan(
        warmup_enabled=True,
        capture_direction=HookPhase.BWD,
    )
    model.b(torch.ones(2))
    model.a(torch.ones(2, 4))
    plan = adaptor.finish_capture_plan()

    ctx = MegatronTrainingContext(
        global_batch_id=9,
        microbatch_id=4,
        valid_counts=[6, 1],
        direction="bwd",
        dp_rank=3,
        shard_rank=2,
        token_start=8,
    )
    decision = adaptor.prepare_replay(
        plan,
        ctx,
        plan_direction=HookPhase.BWD,
    )

    assert decision is StepReservation.RESERVED
    assert engine.record_runtime.reservation_calls == [
        (plan.total_reservation_bytes, plan.task_count)
    ]
    replay_plan, metadata = engine.record_runtime.replay_calls[0]
    assert [entry.output_id for entry in replay_plan.entries] == [
        FIRST_RECORD_OUTPUT_ID + 1,
        FIRST_RECORD_OUTPUT_ID,
    ]
    assert [item.direction for item in metadata] == ["bwd", "bwd"]
    assert [item.phase for item in metadata] == ["train", "train"]
    assert [item.global_batch_id for item in metadata] == [9, 9]
    assert [item.dp_rank for item in metadata] == [3, 3]
    assert [item.microbatch_id for item in metadata] == [4, 4]
    assert [item.layer_no for item in metadata] == [0, 0]
    assert [item.shard_rank for item in metadata] == [7, 5]
    assert [item.token_start for item in metadata] == [8, 8]
    assert [item.attempt_id for item in metadata] == [0, 0]
    assert [item.invocation_id for item in metadata] == [0, 0]
    assert [item.valid_counts for item in metadata] == [(6, 1), (6, 1)]
    assert [item.dataset_ids for item in metadata] == [(), ()]
    assert [entry.output_shape for entry in replay_plan.entries] == [(2,), (2, 4)]
    assert [entry.transport_type for entry in replay_plan.entries] == [
        TransportType.IDENTITY,
        TransportType.IDENTITY,
    ]
    assert [entry.dtype for entry in replay_plan.entries] == [
        torch.float32,
        torch.float32,
    ]
    assert [entry.storage for entry in replay_plan.entries] == [
        OutputStorage.TENSOR,
        OutputStorage.TENSOR,
    ]
    assert [entry.record_type for entry in replay_plan.entries] == [
        RecordType.PER_SAMPLE,
        RecordType.PER_SAMPLE,
    ]
    assert [item.model_id for item in metadata] == ["train-run", "train-run"]


def test_forward_plan_replayed_in_backward_queues_only_non_suppressing_entries():
    engine = FakeEngine()
    gated = TinyModel()
    always = TinyNoRecomputeSuppressModel()
    adaptor = _make_adaptor(
        engine,
        "train-run",
        dims={DimSpec.BATCH: 2, DimSpec.NUM_EXPERTS: 4},
    )
    adaptor.attach_model(
        [gated, always],
        hook_selection="router-summary,loss-summary",
    )

    adaptor.begin_capture_plan(
        warmup_enabled=True,
        capture_direction=HookPhase.FWD,
    )
    gated.router0(torch.ones(2, 4))
    always.loss(torch.ones(2, 1))
    plan = adaptor.finish_capture_plan()

    decision = adaptor.prepare_replay(
        plan,
        MegatronTrainingContext(
            global_batch_id=5,
            microbatch_id=2,
            valid_counts=[1, 1],
            direction="bwd",
        ),
        plan_direction=HookPhase.FWD,
    )

    assert decision is StepReservation.RESERVED
    assert [item.suppress_recompute for item in adaptor._plan_semantics(plan)] == [
        True,
        False,
    ]
    assert engine.record_runtime.reservation_calls == [(16, 1)]
    replay_plan, metadata = engine.record_runtime.replay_calls[0]
    assert [entry.output_id for entry in replay_plan.entries] == [
        FIRST_RECORD_OUTPUT_ID + 1
    ]
    assert [item.direction for item in metadata] == ["fwd"]


def test_forward_plan_replayed_in_backward_is_noop_when_all_entries_are_gated():
    engine = FakeEngine()
    model = TinyModel()
    adaptor = _make_adaptor(
        engine,
        "train-run",
        dims={DimSpec.BATCH: 2, DimSpec.NUM_EXPERTS: 4},
    )
    adaptor.attach_model(model, hook_selection="router-summary")
    adaptor.begin_capture_plan(
        warmup_enabled=True,
        capture_direction=HookPhase.FWD,
    )
    model.router0(torch.ones(2, 4))
    plan = adaptor.finish_capture_plan()

    decision = adaptor.prepare_replay(
        plan,
        MegatronTrainingContext(
            global_batch_id=5,
            microbatch_id=2,
            valid_counts=[1, 1],
            direction="bwd",
        ),
        plan_direction=HookPhase.FWD,
    )

    assert decision is StepReservation.SKIPPED
    assert engine.record_runtime.reservation_calls == []
    assert engine.record_runtime.replay_calls == []


def test_prepare_replay_uses_current_context_when_ctx_is_omitted():
    engine = FakeEngine()
    model = TinyModel()
    adaptor = _make_adaptor(
        engine,
        "train-run",
        dims={DimSpec.BATCH: 2, DimSpec.NUM_EXPERTS: 4},
    )
    adaptor.attach_model(model, hook_selection="router-summary")
    adaptor.begin_capture_plan(
        warmup_enabled=True,
        capture_direction=HookPhase.FWD,
    )
    model.router0(torch.ones(2, 4))
    plan = adaptor.finish_capture_plan()

    adaptor.set_current_event(
        MegatronTrainingContext(
            global_batch_id=2,
            microbatch_id=1,
            valid_counts=[5, 3],
        )
    )

    assert (
        adaptor.prepare_replay(plan, plan_direction=HookPhase.FWD)
        is StepReservation.RESERVED
    )
    replay_plan, metadata = engine.record_runtime.replay_calls[0]
    assert [entry.output_id for entry in replay_plan.entries] == [
        FIRST_RECORD_OUTPUT_ID
    ]
    assert [item.global_batch_id for item in metadata] == [2]


def test_prepare_replay_raises_without_active_context():
    engine = FakeEngine()
    model = TinyModel()
    adaptor = _make_adaptor(
        engine,
        "train-run",
        dims={DimSpec.BATCH: 2, DimSpec.NUM_EXPERTS: 4},
    )
    adaptor.attach_model(model, hook_selection="router-summary")
    adaptor.begin_capture_plan(
        warmup_enabled=True,
        capture_direction=HookPhase.FWD,
    )
    model.router0(torch.ones(2, 4))
    plan = adaptor.finish_capture_plan()

    with pytest.raises(RuntimeError, match="active schedule event"):
        adaptor.prepare_replay(plan, plan_direction=HookPhase.FWD)

    assert engine.record_runtime.reservation_calls == []
    assert engine.record_runtime.replay_calls == []


def test_prepare_replay_signature_mismatch_hard_fails_before_reservation():
    engine = FakeEngine()
    model = TinyMultiHookModel()
    adaptor = _make_adaptor(
        engine,
        "train-run",
        dims={DimSpec.BATCH: 2, DimSpec.NUM_EXPERTS: 4},
    )
    adaptor.attach_model(model, hook_selection="full")

    adaptor.begin_capture_plan(
        warmup_enabled=True,
        capture_direction=HookPhase.FWD,
    )
    model.a(torch.ones(2, 4))
    expected_plan = adaptor.finish_capture_plan()

    adaptor.begin_capture_plan(
        warmup_enabled=True,
        capture_direction=HookPhase.FWD,
    )
    model.b(torch.ones(2))
    actual_plan = adaptor.finish_capture_plan()

    with pytest.raises(ValueError, match="producer plan mismatch"):
        adaptor.prepare_replay(
            actual_plan,
            MegatronTrainingContext(global_batch_id=0, microbatch_id=0, valid_counts=[1, 1]),
            plan_direction=HookPhase.FWD,
            expected_plan=expected_plan,
        )

    assert engine.record_runtime.reservation_calls == []
    assert engine.record_runtime.replay_calls == []


def test_prepare_replay_rejects_runtime_sized_plan_before_reservation():
    engine = FakeEngine()
    model = TinyModel()
    adaptor = _make_adaptor(
        engine,
        "train-run",
        dims={DimSpec.BATCH: 2, DimSpec.NUM_EXPERTS: 4},
    )
    adaptor.attach_model(model, hook_selection="router-summary")
    adaptor.begin_capture_plan(
        warmup_enabled=True,
        capture_direction=HookPhase.FWD,
    )
    model.router0(torch.ones(2, 4))
    plan = adaptor.finish_capture_plan()
    dynamic_entry = replace(
        plan.entries[0],
        transport_type=TransportType.PREFIX_STRIP,
        transport_args=(16,),
    )
    dynamic_plan = replace(plan, entries=(dynamic_entry,))

    with pytest.raises(NotImplementedError, match="does not yet support"):
        adaptor.prepare_replay(
            dynamic_plan,
            MegatronTrainingContext(global_batch_id=0, microbatch_id=0, valid_counts=[1, 1]),
            plan_direction=HookPhase.FWD,
        )

    assert engine.record_runtime.reservation_calls == []
    assert engine.record_runtime.replay_calls == []


def test_prepare_replay_capacity_only_does_not_reserve_or_publish_records():
    engine = FakeEngine()
    model = TinyModel()
    adaptor = _make_adaptor(
        engine,
        "train-run",
        dims={DimSpec.BATCH: 2, DimSpec.NUM_EXPERTS: 4},
    )
    adaptor.attach_model(model, hook_selection="router-summary")
    adaptor.begin_capture_plan(
        warmup_enabled=True,
        capture_direction=HookPhase.FWD,
    )
    model.router0(torch.ones(2, 4))
    plan = adaptor.finish_capture_plan()

    assert (
        adaptor.prepare_replay_capacity_only(
            plan,
            plan_direction=HookPhase.FWD,
            live_direction=HookPhase.FWD,
        )
        is StepReservation.RESERVED
    )
    assert engine.record_runtime.reservation_calls == []
    assert engine.record_runtime.replay_calls == []

    engine.payload_cap = plan.total_reservation_bytes - 1
    assert (
        adaptor.prepare_replay_capacity_only(
            plan,
            plan_direction=HookPhase.FWD,
            live_direction=HookPhase.FWD,
        )
        is StepReservation.OVERSIZED
    )
    assert engine.record_runtime.reservation_calls == []
    assert engine.record_runtime.replay_calls == []


def test_prepare_replay_oversized_returns_without_record_publication():
    engine = FakeEngine(replay_result=StepReservation.OVERSIZED)
    model = TinyModel()
    adaptor = _make_adaptor(
        engine,
        "train-run",
        dims={DimSpec.BATCH: 2, DimSpec.NUM_EXPERTS: 4},
    )
    adaptor.attach_model(model, hook_selection="router-summary")
    adaptor.begin_capture_plan(
        warmup_enabled=True,
        capture_direction=HookPhase.FWD,
    )
    model.router0(torch.ones(2, 4))
    plan = adaptor.finish_capture_plan()

    decision = adaptor.prepare_replay(
        plan,
        MegatronTrainingContext(global_batch_id=0, microbatch_id=0, valid_counts=[1, 1]),
        plan_direction=HookPhase.FWD,
    )

    assert decision is StepReservation.OVERSIZED
    assert engine.record_runtime.reservation_calls == [
        (plan.total_reservation_bytes, plan.task_count)
    ]
    assert engine.record_runtime.published_replay_metadata == []


def test_local_graph_fallback_does_not_consume_retained_invocation_id():
    engine = FakeEngine()
    model = TinyNoRecomputeSuppressModel()
    adaptor = _make_adaptor(engine, "train-run", dims={DimSpec.BATCH: 2})
    adaptor.attach_model(
        model,
        hook_selection="loss-summary",
        activation_recompute_enabled=True,
    )
    adaptor.begin_capture_plan(
        warmup_enabled=True,
        capture_direction=HookPhase.FWD,
    )
    model.loss(torch.ones(2, 1))
    plan = adaptor.finish_capture_plan()
    engine.payload_cap = plan.total_reservation_bytes - 1
    adaptor.begin_attempt(phase="train", global_batch_id=5, attempt_id=0)
    ctx = MegatronTrainingContext(
        global_batch_id=5,
        microbatch_id=1,
        valid_counts=(1, 1),
        attempt_id=0,
        direction="bwd",
        phase="train",
    )

    assert adaptor.prepare_replay(
        plan,
        ctx,
        plan_direction=HookPhase.FWD,
    ) is StepReservation.OVERSIZED
    assert engine.record_runtime.replay_calls == []

    engine.payload_cap = 1 << 30
    assert adaptor.prepare_replay(
        plan,
        ctx,
        plan_direction=HookPhase.FWD,
    ) is StepReservation.RESERVED
    assert len(engine.record_runtime.replay_calls) == 1
    _replay_plan, metadata = engine.record_runtime.replay_calls[0]
    assert [item.direction for item in metadata] == ["fwd"]
    assert [item.invocation_id for item in metadata] == [0]
    adaptor.end_attempt(attempt_id=0)


def test_prepare_replay_empty_plan_is_noop():
    engine = FakeEngine()
    adaptor = _make_adaptor(engine, "train-run")
    adaptor.begin_capture_plan(warmup_enabled=True)
    plan = adaptor.finish_capture_plan()

    decision = adaptor.prepare_replay(
        plan,
        MegatronTrainingContext(global_batch_id=0, microbatch_id=0, valid_counts=[]),
        plan_direction=HookPhase.FWD,
    )

    assert decision is StepReservation.SKIPPED
    assert engine.record_runtime.reservation_calls == []
    assert engine.record_runtime.replay_calls == []


def test_capture_empty_plan_is_valid_but_missing_plan_errors():
    engine = FakeEngine()
    adaptor = _make_adaptor(engine, "train-run")

    adaptor.begin_capture_plan(warmup_enabled=True)
    plan = adaptor.finish_capture_plan()
    assert plan.task_count == 0
    assert plan.entries == ()

    with pytest.raises(RuntimeError, match="capture plan is not active"):
        adaptor.finish_capture_plan()


def test_capture_abort_restores_eager_mode_and_null_mode():
    engine = FakeEngine()
    adaptor = _make_adaptor(engine, "train-run")

    adaptor.begin_capture_plan(warmup_enabled=True)
    assert adaptor.hook_runtime.mode == HookRuntimeMode.CAPTURE_RECORD
    assert engine.capture_enabled_calls == [False]

    adaptor.abort_capture_plan()

    assert adaptor.hook_runtime.mode == HookRuntimeMode.EAGER_IMMEDIATE
    assert engine.capture_enabled_calls == [False, True]


def test_te_capture_session_suppresses_warmup_and_records_each_forward_without_null_toggle():
    engine = FakeEngine()
    model = TinyModel()
    adaptor = _make_adaptor(
        engine,
        "train-run",
        dims={DimSpec.BATCH: 2, DimSpec.NUM_EXPERTS: 4},
    )
    adaptor.attach_model(model, hook_selection="router-summary")

    adaptor.begin_te_capture_session()
    assert adaptor.te_capture_session_active is True
    model.router0(torch.ones(2, 4))

    adaptor.begin_te_forward_capture()
    model.router0(torch.ones(2, 4))
    plan = adaptor.finish_te_forward_capture()

    model.router0(torch.ones(2, 4))
    assert adaptor.te_capture_session_active is True
    assert adaptor.hook_runtime.mode is HookRuntimeMode.EAGER_IMMEDIATE
    assert plan.task_count == 1
    assert adaptor._plan_semantics(plan)[0].act_name == "router_probs_mean"
    assert engine.capture_enabled_calls == [False]
    assert engine.record_runtime.reservation_calls == []
    assert engine.record_runtime.emit_calls == []

    adaptor.finish_te_capture_session()
    assert adaptor.te_capture_session_active is False
    assert engine.capture_enabled_calls == [False, True]


def test_te_forward_capture_requires_outer_session_and_rejects_nesting():
    adaptor = _make_adaptor(FakeEngine(), "train-run")

    with pytest.raises(RuntimeError, match="active outer session"):
        adaptor.begin_te_forward_capture()

    adaptor.begin_te_capture_session()
    with pytest.raises(RuntimeError, match="already active"):
        adaptor.begin_te_capture_session()
    adaptor.begin_te_forward_capture()
    with pytest.raises(RuntimeError, match="already active"):
        adaptor.begin_te_forward_capture()
    adaptor.abort_te_forward_capture()
    adaptor.finish_te_capture_session()


def test_te_capture_session_cleans_dangling_forward_plan_and_restores_null_mode():
    engine = FakeEngine()
    adaptor = _make_adaptor(engine, "train-run")

    adaptor.begin_te_capture_session()
    adaptor.begin_te_forward_capture()
    with pytest.raises(RuntimeError, match="ended with an active forward plan"):
        adaptor.finish_te_capture_session()

    assert adaptor.te_capture_session_active is False
    assert adaptor.hook_runtime.mode is HookRuntimeMode.EAGER_IMMEDIATE
    assert engine.capture_enabled_calls == [False, True]


@pytest.mark.parametrize(
    ("phase", "direction", "opposite"),
    (
        (HookPhase.FWD, "fwd", "bwd"),
        (HookPhase.BWD, "bwd", "fwd"),
    ),
)
def test_transport_metadata_binding_is_phase_local_and_does_not_leak_between_hooks(
    phase,
    direction,
    opposite,
):
    engine = FakeEngine()
    record_only = _make_hook(
        MegatronHookSpec(
            name="record_only",
            layer_no=0,
            outputs=[
                MegatronOutputSpec(
                    name="record_only",
                    input_shape=[DimSpec.BATCH, 4],
                    dtype=torch.float32,
                )
            ],
            need_token_range=True,
        ),
        hook_phase=phase,
    )
    count_independent = _make_hook(
        MegatronHookSpec(
            name="count_independent",
            layer_no=0,
            outputs=[
                MegatronOutputSpec(
                    name="count_independent",
                    input_shape=[DimSpec.BATCH, 4],
                    dtype=torch.float32,
                )
            ],
            need_token_range=False,
        ),
        hook_phase=phase,
    )
    transport_bound = _make_hook(
        MegatronHookSpec(
            name="seq_prefix_pack",
            layer_no=0,
            outputs=[
                MegatronOutputSpec(
                    name="seq_prefix_pack",
                    input_shape=[DimSpec.SEQ, DimSpec.BATCH, 4],
                    output_shape=[DimSpec.ACTUAL_TOKEN_PACKED, 4],
                    dtype=torch.float32,
                    transport_type=TransportType.SEQ_PREFIX_PACK,
                )
            ],
            need_token_range=False,
        ),
        hook_phase=phase,
    )
    ctx = DMIMetadataContext(
        max_num_microbatches=2,
        max_batch_size=2,
        num_scopes=1,
        device="cpu",
    )
    ctx.begin_iteration(2)
    ctx.ingest_microbatch(0, {"valid_count": [5, 2]})
    ctx.enter_scope("fwd", 0, 0)
    ctx.ingest_microbatch(1, {"valid_count": [7, 3]})
    ctx.enter_scope("bwd", 0, 1)

    adaptor = _make_adaptor(
        engine,
        "train-run",
        dims={DimSpec.BATCH: 2, DimSpec.SEQ: 8},
    )

    adaptor.attach_hooks(
        model_hooks=[
            MegatronHookBinding(record_only),
            MegatronHookBinding(count_independent),
            MegatronHookBinding(transport_bound),
        ],
        iteration_hooks=[],
        metadata_context=ctx,
    )

    bound = getattr(transport_bound, f"valid_count_{direction}")
    expected = ctx.current("valid_count", direction, 0)
    assert bound.data_ptr() == expected.data_ptr()
    assert torch.equal(
        bound,
        torch.tensor([5, 2] if direction == "fwd" else [7, 3]),
    )
    assert not hasattr(transport_bound, f"valid_count_{opposite}")
    assert not hasattr(record_only, "valid_count_fwd")
    assert not hasattr(record_only, "valid_count_bwd")
    assert not hasattr(count_independent, "valid_count_fwd")
    assert not hasattr(count_independent, "valid_count_bwd")

    event = MegatronTrainingContext(
        global_batch_id=7,
        microbatch_id=0,
        valid_counts=(5, 2),
        direction=direction,
    )

    def record_metadata(hook):
        semantics = adaptor._producer_semantics(
            hook=hook,
            output_id=hook._output_ids[0],
            output_spec=hook.spec.outputs[0],
        )
        return adaptor._record_metadata(event, semantics)

    assert record_metadata(record_only).valid_counts == (5, 2)
    assert record_metadata(transport_bound).valid_counts == (5, 2)
    assert record_metadata(count_independent).valid_counts == ()


def test_prefix_strip_preprocess_and_seq_pack_transport_bind_independent_metadata():
    engine = FakeEngine()
    policy = MegatronHookSpec(
        name="mixed_metadata",
        layer_no=0,
        outputs=[
            MegatronOutputSpec(
                name="prefix_output",
                input_shape=[DimSpec.BATCH, 4],
                output_shape=[DimSpec.BATCH, 4],
                dtype=torch.float32,
                transport_type=TransportType.PREFIX_STRIP,
            ),
            MegatronOutputSpec(
                name="packed_output",
                input_shape=[DimSpec.SEQ, DimSpec.BATCH, 4],
                output_shape=[DimSpec.ACTUAL_TOKEN_PACKED, 4],
                dtype=torch.float32,
                transport_type=TransportType.SEQ_PREFIX_PACK,
            ),
        ],
        preprocess_metadata_fields=frozenset(
            {MegatronMetadataField.SEGMENT_METADATA}
        ),
        need_token_range=False,
    )
    hook = _make_hook(policy)
    ctx = DMIMetadataContext(
        max_num_microbatches=1,
        max_batch_size=2,
        num_scopes=1,
        field_specs=(
            valid_count_field_spec(),
            segment_metadata_field_spec(2),
        ),
        device="cpu",
    )
    ctx.begin_iteration(1)
    ctx.ingest_microbatch(
        0,
        {
            "valid_count": [3, 2],
            "segment_metadata": [0, 3, 3, 5],
        },
    )
    ctx.enter_scope("fwd", 0, 0)
    adaptor = _make_adaptor(
        engine,
        "train-run",
        dims={DimSpec.BATCH: 2, DimSpec.SEQ: 8},
    )

    adaptor.attach_hooks(
        model_hooks=[MegatronHookBinding(hook)],
        iteration_hooks=[],
        metadata_context=ctx,
    )

    assert policy.binding_metadata_fields == frozenset(
        {
            MegatronMetadataField.VALID_COUNT,
            MegatronMetadataField.SEGMENT_METADATA,
        }
    )
    assert hook.valid_count_fwd.data_ptr() == ctx.current(
        "valid_count", "fwd", 0
    ).data_ptr()
    segments = ctx.current("segment_metadata", "fwd", 0)
    assert hook.sample_start_ptr_fwd.data_ptr() == segments[:2].data_ptr()
    assert hook.sample_end_ptr_fwd.data_ptr() == segments[2:].data_ptr()
    assert torch.equal(hook.sample_start_ptr_fwd, torch.tensor([0, 3]))
    assert torch.equal(hook.sample_end_ptr_fwd, torch.tensor([3, 5]))
    assert not hasattr(hook, "valid_count_bwd")
    assert not hasattr(hook, "sample_start_ptr_bwd")
    assert not hasattr(hook, "sample_end_ptr_bwd")


def test_attach_model_uses_vp_stage_as_context_scope_for_model_chunks():
    engine = FakeEngine()
    model0 = TinyValidCountBindingModel()
    model1 = TinyValidCountBindingModel()
    model0.vp_stage = 0
    model1.vp_stage = 1
    ctx = DMIMetadataContext(
        max_num_microbatches=1,
        max_batch_size=2,
        num_scopes=2,
        device="cpu",
    )
    ctx.begin_iteration(1)

    adaptor = _make_adaptor(
        engine,
        "train-run",
        dims={DimSpec.BATCH: 2, DimSpec.NUM_EXPERTS: 4},
    )

    adaptor.attach_model(
        [model0, model1],
        hook_selection="binding",
        metadata_context=ctx,
    )

    assert model0.router0.valid_count_fwd.data_ptr() == ctx.current(
        "valid_count", "fwd", 0
    ).data_ptr()
    assert model1.router0.valid_count_fwd.data_ptr() == ctx.current(
        "valid_count", "fwd", 1
    ).data_ptr()


def test_invocation_allocator_sequences_each_producer_key_and_resets_per_attempt():
    allocator = InvocationIdAllocator()
    allocator.begin_attempt(
        model_id="train-run",
        phase="train",
        global_batch_id=7,
        attempt_id=0,
    )

    assert _allocate_invocation(allocator) == 0
    assert _allocate_invocation(allocator) == 1
    assert _allocate_invocation(
        allocator,
        output_id=FIRST_RECORD_OUTPUT_ID + 1,
    ) == 0
    assert _allocate_invocation(allocator, microbatch_id=2) == 0
    allocator.end_attempt(attempt_id=0)

    allocator.begin_attempt(
        model_id="train-run",
        phase="train",
        global_batch_id=7,
        attempt_id=1,
    )
    assert _allocate_invocation(allocator, attempt_id=1) == 0
    allocator.end_attempt(attempt_id=1)


def test_invocation_allocator_rejects_lifecycle_and_outer_coordinate_mismatches():
    allocator = InvocationIdAllocator()
    with pytest.raises(RuntimeError, match="does not match the active attempt"):
        _allocate_invocation(allocator)

    allocator.begin_attempt(
        model_id="train-run",
        phase="train",
        global_batch_id=7,
        attempt_id=0,
    )
    with pytest.raises(RuntimeError, match="already active"):
        allocator.begin_attempt(
            model_id="train-run",
            phase="train",
            global_batch_id=7,
            attempt_id=0,
        )
    with pytest.raises(RuntimeError, match="does not match the active attempt"):
        _allocate_invocation(allocator, global_batch_id=8)
    with pytest.raises(RuntimeError, match="ended the wrong attempt"):
        allocator.end_attempt(attempt_id=1)
    allocator.end_attempt(attempt_id=0)


def test_invocation_allocator_disabled_path_returns_zero_without_building_a_key():
    class ExplodingModelId:
        def __str__(self):
            raise AssertionError("disabled invocation path accessed model_id")

    adaptor = _make_adaptor(
        FakeEngine(),
        "train-run",
        dims={DimSpec.BATCH: 2, DimSpec.NUM_EXPERTS: 4},
    )
    model = TinyModel()
    adaptor.attach_model(model, hook_selection="router-summary")
    assert adaptor.invocation_allocator is None
    ctx = MegatronTrainingContext(
        global_batch_id=7,
        microbatch_id=0,
        valid_counts=(1, 1),
    )
    semantic = adaptor._producer_semantics(
        hook=model.router0,
        output_id=model.router0._output_ids[0],
        output_spec=model.router0.spec.outputs[0],
    )

    invocation_id = adaptor._allocate_invocation_id(
        ctx,
        semantic=semantic,
        dp_rank=0,
        shard_rank=0,
        token_start=0,
        model_id=ExplodingModelId(),
    )

    assert invocation_id == 0


def test_attach_constructs_allocator_only_for_retained_recompute_hooks():
    suppressing = _make_adaptor(
        FakeEngine(),
        "train-run",
        dims={DimSpec.BATCH: 2, DimSpec.NUM_EXPERTS: 4},
    )
    suppressing.attach_model(
        TinyModel(),
        hook_selection="router-summary",
        activation_recompute_enabled=True,
    )
    assert suppressing.invocation_allocator is None

    retained = _make_adaptor(
        FakeEngine(),
        "train-run",
        dims={DimSpec.BATCH: 2},
    )
    retained.attach_model(
        TinyNoRecomputeSuppressModel(),
        hook_selection="loss-summary",
        activation_recompute_enabled=True,
    )
    assert isinstance(retained.invocation_allocator, InvocationIdAllocator)

    recompute_disabled = _make_adaptor(
        FakeEngine(),
        "train-run",
        dims={DimSpec.BATCH: 2},
    )
    recompute_disabled.attach_model(
        TinyNoRecomputeSuppressModel(),
        hook_selection="loss-summary",
        activation_recompute_enabled=False,
    )
    assert recompute_disabled.invocation_allocator is None
