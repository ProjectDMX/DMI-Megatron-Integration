from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from dmi.api.v1 import (
    OutputStorage,
    ProducerPlan,
    ProducerPlanEntry,
    RecordType,
    StepReservation,
    TransportType,
)

from dmi_megatron_integration.schedule_runtime import (
    MegatronScheduleRuntime,
    build_megatron_schedule_runtime,
    dmi_begin_iteration,
    dmi_end_iteration,
    dmi_enter_current_scope,
    dmi_force_eager_unit,
    dmi_guard_schedule_supported,
    dmi_is_enabled,
    dmi_finish_full_iteration_replay,
    dmi_prepare_full_iteration_replay,
    dmi_prepare_local_backward_replay,
    dmi_prepare_local_forward_boundary,
    dmi_prepare_local_forward_replay,
    dmi_record_current_microbatch_metadata,
    dmi_set_current_event,
    dmi_take_local_backward_token,
    set_active_megatron_schedule_runtime,
    _rank_groups_for_token,
)
from dmi_megatron_integration.adapter import (
    MegatronEventCoordinates,
    MegatronFullIterationPlan,
    _ProducerSemantics,
)
from dmi_megatron_integration.hooks.specs import HookPhase
from dmi_megatron_integration.metadata_context import (
    DMIMetadataContext,
    LocalMetadataPropagator,
    PerDPCPUMetadataPropagator,
    dataset_id_field_spec,
    valid_count_field_spec,
)


class FakePropagator:
    def __init__(self) -> None:
        self.is_metadata_source_rank = True
        self.calls = []
        self.context = SimpleNamespace(field_specs={"valid_count": object()})
        self.context.set_active_fields = self.set_active_fields
        self.context.active_field_names = frozenset({"valid_count"})

    def set_active_fields(self, names):
        self.active_fields = tuple(names)
        self.context.active_field_names = frozenset(names)

    def begin_iteration(self, active_num_microbatches):
        self.calls.append(("begin", active_num_microbatches))

    def ingest_microbatch(self, microbatch_id, fields, cpu_fields=None):
        self.calls.append(("ingest", microbatch_id, dict(fields), cpu_fields))

    def enter_scope(self, direction, scope_id, microbatch_id):
        self.calls.append(("enter", direction, scope_id, microbatch_id))

    def end_iteration(self):
        self.calls.append(("end",))


class FakeContext:
    def __init__(self, counts):
        self.counts = counts
        self.field_specs = {"valid_count": object()}
        self.active_field_names = frozenset({"valid_count"})

    def set_active_fields(self, names):
        self.active_fields = tuple(names)
        self.active_field_names = frozenset(names)

    def source(self, name, microbatch_id):
        assert name == "valid_count"
        return torch.tensor(self.counts[microbatch_id])

    def source_cpu(self, name, microbatch_id):
        assert name == "valid_count"
        return tuple(self.counts[microbatch_id])


class FakePropagatorWithContext(FakePropagator):
    def __init__(self, counts) -> None:
        super().__init__()
        self.context = FakeContext(counts)


class FakeAdaptor:
    def __init__(self) -> None:
        self.current_events = []
        self.current_context = None
        self.clear_calls = 0
        self.model_id = "model-a"
        self.hook_runtime = SimpleNamespace(capture_event_context=False)
        self.attempt_calls = []
        self.current_iteration_context = None
        self.iteration_contexts = []

    def begin_attempt(self, *, phase, global_batch_id, attempt_id):
        self.attempt_calls.append(("begin", phase, global_batch_id, attempt_id))

    def end_attempt(self, *, attempt_id):
        self.attempt_calls.append(("end", attempt_id))

    def set_current_iteration(self, ctx):
        self.current_iteration_context = ctx
        self.iteration_contexts.append(ctx)

    def clear_current_iteration(self):
        self.current_iteration_context = None

    def set_current_event(self, ctx):
        self.current_context = ctx
        self.current_events.append(ctx)

    def clear_current_event(self):
        self.current_context = None
        self.clear_calls += 1

    @staticmethod
    def normalize_direction(direction):
        normalized = str(direction).strip().lower()
        if normalized in {"fwd", "forward"}:
            return HookPhase.FWD
        if normalized in {"bwd", "backward"}:
            return HookPhase.BWD
        raise ValueError(direction)


class FakeHostEngine:
    def __init__(self) -> None:
        self.boundaries = []
        self.submissions = []

    def submit_record(self, layout, values, cell_types, *, nbytes):
        row = tuple(values)
        self.boundaries.append(row)
        self.submissions.append(
            (str(layout), row, tuple(cell_types), int(nbytes))
        )


class FakeReplayAdaptor(FakeAdaptor):
    def __init__(self, decisions=None) -> None:
        super().__init__()
        self.decisions = list(decisions or [])
        self.capacity_calls = []
        self.replay_calls = []
        self.full_replay_calls = []
        self.current_context = SimpleNamespace(direction="fwd")

    def prepare_replay_capacity_only(self, plan, *, plan_direction, live_direction):
        self.capacity_calls.append((plan, plan_direction, live_direction))
        if self.decisions:
            return self.decisions.pop(0)
        return StepReservation.RESERVED

    def prepare_replay(self, plan, *, plan_direction, live_direction):
        self.replay_calls.append((plan, plan_direction, live_direction))
        return StepReservation.RESERVED

    def prepare_full_iteration_replay(self, plan, contexts):
        self.full_replay_calls.append((plan, contexts))
        if self.decisions:
            return self.decisions.pop(0)
        return StepReservation.RESERVED


class FakeWork:
    def __init__(self) -> None:
        self.wait_count = 0

    def wait(self):
        self.wait_count += 1


class FakeBroadcastDist:
    def __init__(self, payloads=None) -> None:
        self.payloads = [torch.tensor(item, dtype=torch.int64) for item in (payloads or [])]
        self.broadcast_calls = []

    def broadcast(self, tensor, src, group=None, async_op=False):
        self.broadcast_calls.append((tensor, src, group, async_op))
        if async_op and self.payloads:
            tensor.copy_(self.payloads.pop(0))
        return FakeWork()


class FakeRunner:
    def __init__(self, *, grad_enabled=True, fwd_plan="fwd-plan", bwd_plan="bwd-plan"):
        self.grad_enabled = grad_enabled
        self._dmi_fwd_plan = fwd_plan
        self._dmi_bwd_plan = bwd_plan


def _full_iteration_plan(microbatch_ids=(0, 1)):
    producers = []
    events = []
    semantics = []
    for index, microbatch_id in enumerate(microbatch_ids):
        producer = ProducerPlanEntry(
            output_id=100 + index,
            input_shape=(2,),
            output_shape=(2,),
            dtype=torch.float32,
            transport_type=TransportType.IDENTITY,
            transport_args=(),
            storage=OutputStorage.TENSOR,
            record_type=RecordType.PER_SAMPLE,
            reservation_upper_bytes=8,
        )
        event = MegatronEventCoordinates(
            direction="fwd" if index == 0 else "bwd",
            microbatch_id=int(microbatch_id),
            scope_id=index,
            dp_rank=2,
            shard_rank=3,
            token_start=0,
        )
        producers.append(producer)
        events.append(event)
        semantics.append(
            _ProducerSemantics(
                output_id=producer.output_id,
                act_name="router_probs_mean",
                layer_no=index,
                record_type=RecordType.PER_SAMPLE,
                transport_type=producer.transport_type,
                record_dp_rank=None,
                record_shard_rank=None,
                need_token_range=True,
                suppress_recompute=True,
                record_direction=event.direction,
            )
        )
    return MegatronFullIterationPlan.from_plan_and_events(
        ProducerPlan(tuple(producers)),
        events,
        semantics,
    )


class FakeConfig:
    overlap_moe_expert_parallel_comm = False
    hybrid_context_parallel = False
    context_parallel_size = 1
    cuda_graph_scope = ()


class FakeParallelState:
    def __init__(
        self,
        *,
        tp_world_size=1,
        pp_world_size=1,
        vp_world_size=None,
        tp_rank=0,
        pp_rank=0,
    ):
        self.tp_world_size = tp_world_size
        self.pp_world_size = pp_world_size
        self.vp_world_size = vp_world_size
        self.tp_rank = tp_rank
        self.pp_rank = pp_rank
        self.created_groups = []

    def get_virtual_pipeline_model_parallel_world_size(self):
        return self.vp_world_size

    def get_tensor_model_parallel_world_size(self):
        return self.tp_world_size

    def get_pipeline_model_parallel_world_size(self):
        return self.pp_world_size

    def get_tensor_model_parallel_rank(self):
        return self.tp_rank

    def get_pipeline_model_parallel_rank(self):
        return self.pp_rank

    def create_group(self, ranks, backend=None, group_desc=None):
        self.created_groups.append((tuple(ranks), backend, group_desc))
        return (tuple(ranks), backend, group_desc)


class FakeDist:
    def __init__(self, initialized=True, rank=0, world_size=1):
        self.initialized = initialized
        self.rank = rank
        self.world_size = world_size

    def is_initialized(self):
        return self.initialized

    def get_rank(self):
        return self.rank

    def get_world_size(self):
        return self.world_size


def teardown_function():
    set_active_megatron_schedule_runtime(None)


def test_schedule_runtime_records_once_and_enters_current_scope():
    propagator = FakePropagator()
    runtime = MegatronScheduleRuntime(propagator)

    runtime.begin_iteration(4)
    runtime.set_event("fwd", 2, 1)
    runtime.record_current_microbatch_metadata(torch.tensor([7, 3]))
    runtime.record_current_microbatch_metadata(torch.tensor([9, 9]))
    runtime.enter_current_scope()
    runtime.end_iteration()

    assert propagator.calls[0] == ("begin", 4)
    assert propagator.calls[1][0:2] == ("ingest", 2)
    assert torch.equal(propagator.calls[1][2]["valid_count"], torch.tensor([7, 3]))
    assert propagator.calls[1][3] is None
    assert propagator.calls[2] == ("enter", "fwd", 1, 2)
    assert propagator.calls[3] == ("end",)


def test_train_schedule_rerun_keeps_outer_logical_global_batch_id():
    runtime = MegatronScheduleRuntime(FakePropagator())
    runtime.enter_phase(
        "train",
        training_iteration_id_start=10,
        global_batch_id_start=10,
    )
    runtime.begin_logical_iteration(10)
    runtime.begin_attempt(0)
    runtime.begin_iteration(1)
    runtime.end_iteration()
    assert runtime.global_batch_id == 10
    runtime.finish_attempt(0)
    runtime.begin_attempt(1)
    runtime.begin_iteration(1)
    runtime.end_iteration()
    assert runtime.global_batch_id == 10
    runtime.finish_attempt(1)
    runtime.finish_logical_iteration()

    runtime.begin_logical_iteration(11)
    runtime.begin_attempt(0)
    runtime.begin_iteration(1)
    runtime.end_iteration()
    runtime.finish_attempt(1)
    runtime.finish_logical_iteration()
    assert runtime.global_batch_id == 11


def test_iteration_boundary_flush_runs_after_configured_accepted_iterations():
    runtime = MegatronScheduleRuntime(FakePropagator())
    calls = []
    runtime.configure_iteration_flush(
        2,
        flush_callback=lambda: calls.append("flush"),
        barrier_callback=lambda: calls.append("barrier"),
        logger=lambda iteration, elapsed: calls.append((iteration, elapsed)),
    )

    for iteration in range(1, 5):
        runtime.begin_logical_iteration(iteration)
        runtime.begin_attempt(0)
        runtime.finish_attempt(1)
        runtime.finish_logical_iteration()

    assert calls[0:2] == ["flush", "barrier"]
    assert calls[2][0] == 2
    assert calls[3:5] == ["flush", "barrier"]
    assert calls[5][0] == 4
    assert calls[2][1] >= 0.0
    assert calls[5][1] >= 0.0


def test_iteration_boundary_flush_runs_once_after_rerun_is_accepted():
    runtime = MegatronScheduleRuntime(FakePropagator())
    calls = []
    runtime.configure_iteration_flush(
        1,
        flush_callback=lambda: calls.append("flush"),
        barrier_callback=lambda: calls.append("barrier"),
    )
    runtime.begin_logical_iteration(1)
    runtime.begin_attempt(0)
    runtime.finish_attempt(0)
    runtime.begin_attempt(1)
    runtime.finish_attempt(1)
    runtime.finish_logical_iteration()

    assert calls == ["flush", "barrier"]


def test_iteration_boundary_flush_disabled_and_validation_contracts():
    runtime = MegatronScheduleRuntime(FakePropagator())
    runtime.configure_iteration_flush(0)
    with pytest.raises(ValueError, match="nonnegative"):
        runtime.configure_iteration_flush(-1)
    with pytest.raises(ValueError, match="must not configure callbacks"):
        runtime.configure_iteration_flush(0, flush_callback=lambda: None)
    with pytest.raises(TypeError, match="flush callback"):
        runtime.configure_iteration_flush(1, barrier_callback=lambda: None)
    with pytest.raises(TypeError, match="barrier callback"):
        runtime.configure_iteration_flush(1, flush_callback=lambda: None)


def test_iteration_boundary_flush_failure_propagates_after_closing_iteration():
    runtime = MegatronScheduleRuntime(FakePropagator())

    def fail_flush():
        raise RuntimeError("flush failed")

    runtime.configure_iteration_flush(
        1,
        flush_callback=fail_flush,
        barrier_callback=lambda: None,
    )
    runtime.begin_logical_iteration(1)
    runtime.begin_attempt(0)
    runtime.finish_attempt(1)
    with pytest.raises(RuntimeError, match="flush failed"):
        runtime.finish_logical_iteration()
    assert runtime._logical_training_iteration_id is None


def test_attempt_lifecycle_emits_one_status_row_per_attempt_with_stable_iteration():
    runtime = MegatronScheduleRuntime(FakePropagator())
    adaptor = FakeAdaptor()
    runtime.adaptor = adaptor
    emitted = []

    def status_hook(tensor):
        ctx = adaptor.current_iteration_context
        assert ctx is not None
        emitted.append((int(tensor.item()), ctx))

    runtime.set_attempt_status_hook(status_hook, device="cpu")
    runtime.enter_phase(
        "train",
        training_iteration_id_start=12,
        global_batch_id_start=12,
    )
    runtime.begin_logical_iteration(12)
    runtime.begin_attempt(0)
    runtime.finish_attempt(0)
    runtime.begin_attempt(1)
    runtime.finish_attempt(1)
    runtime.finish_logical_iteration()

    assert [value for value, _ctx in emitted] == [0, 1]
    assert [ctx.global_batch_id for _value, ctx in emitted] == [12, 12]
    assert [ctx.attempt_id for _value, ctx in emitted] == [0, 1]
    assert all(ctx.direction == "iter" for _value, ctx in emitted)
    assert all(ctx.dataset_ids == () for _value, ctx in emitted)
    assert adaptor.attempt_calls == [
        ("begin", "train", 12, 0),
        ("end", 0),
        ("begin", "train", 12, 1),
        ("end", 1),
    ]


def test_attempt_lifecycle_rejects_missing_or_multiple_terminal_outcomes():
    runtime = MegatronScheduleRuntime(FakePropagator())
    runtime.begin_logical_iteration(1)
    runtime.begin_attempt(0)
    runtime.finish_attempt(0)
    with pytest.raises(RuntimeError, match="one accepted or controlled-abort"):
        runtime.finish_logical_iteration()

    runtime.begin_attempt(1)
    runtime.finish_attempt(1)
    runtime.begin_attempt(2)
    with pytest.raises(RuntimeError, match="more than one accepted"):
        runtime.finish_attempt(1)


def test_schedule_runtime_facade_is_noop_without_active_runtime():
    assert dmi_is_enabled() is False

    dmi_begin_iteration(2)
    dmi_set_current_event("fwd", 0)
    dmi_record_current_microbatch_metadata(torch.tensor([1]), torch.tensor([1]))
    dmi_enter_current_scope()
    dmi_end_iteration()


def test_schedule_runtime_facade_forwards_to_active_runtime():
    propagator = FakePropagator()
    set_active_megatron_schedule_runtime(MegatronScheduleRuntime(propagator))

    dmi_begin_iteration(1)
    dmi_set_current_event("bwd", 0, 3)
    dmi_enter_current_scope()
    dmi_end_iteration()

    assert propagator.calls == [
        ("begin", 1),
        ("enter", "bwd", 3, 0),
        ("end",),
    ]


def test_schedule_runtime_updates_current_phase_tensor():
    runtime = MegatronScheduleRuntime(FakePropagator())
    current_phase = torch.zeros((), dtype=torch.int32)
    runtime.set_current_phase_tensor(current_phase)

    runtime.begin_iteration(2)
    runtime.set_event("fwd", 0, 0)
    assert int(current_phase.item()) == HookPhase.FWD.value
    runtime.set_event("bwd", 0, 0)
    assert int(current_phase.item()) == HookPhase.BWD.value
    runtime.set_event("forward", 1, 0)
    assert int(current_phase.item()) == HookPhase.FWD.value


def test_schedule_runtime_rejects_bad_phase_tensor_and_direction():
    runtime = MegatronScheduleRuntime(FakePropagator())
    with pytest.raises(TypeError, match="torch.int32"):
        runtime.set_current_phase_tensor(torch.zeros((), dtype=torch.int64))
    with pytest.raises(ValueError, match="scalar"):
        runtime.set_current_phase_tensor(torch.zeros(2, dtype=torch.int32))

    runtime.set_current_phase_tensor(torch.zeros((), dtype=torch.int32))
    runtime.begin_iteration(1)
    with pytest.raises(ValueError, match="Unsupported DMI schedule direction"):
        runtime.set_event("sideways", 0, 0)


def test_schedule_runtime_sets_backward_event_context():
    propagator = FakePropagatorWithContext({0: [5, 2]})
    runtime = MegatronScheduleRuntime(propagator)
    runtime.adaptor = FakeAdaptor()
    runtime.global_batch_id = 9
    runtime.dp_rank = 1
    runtime.shard_rank = 3

    runtime.begin_iteration(1)
    runtime.set_event("bwd", 0, 2)
    runtime.enter_current_scope()

    assert propagator.calls == [
        ("begin", 1),
        ("enter", "bwd", 2, 0),
    ]
    assert len(runtime.adaptor.current_events) == 1
    ctx = runtime.adaptor.current_events[0]
    assert ctx.direction == "bwd"
    assert ctx.global_batch_id == 9
    assert ctx.microbatch_id == 0
    assert ctx.valid_counts == (5, 2)
    assert ctx.dp_rank == 1
    assert ctx.shard_rank == 3


def test_schedule_runtime_writes_eval_boundaries_only_for_eval_phases():
    host = FakeHostEngine()
    adaptor = FakeAdaptor()
    runtime = MegatronScheduleRuntime(
        FakePropagatorWithContext({0: [5, 2]}),
        host_engine=host,
    )
    runtime.adaptor = adaptor

    runtime.enter_phase(
        "train",
        training_iteration_id_start=1,
        training_iteration_id_end=1,
        global_batch_id_start=1,
    )
    runtime.begin_logical_iteration(1)
    runtime.begin_attempt(0)
    runtime.begin_iteration(1)
    runtime.end_iteration()
    runtime.finish_attempt(1)
    runtime.finish_logical_iteration()

    runtime.enter_phase(
        "valid",
        training_iteration_id_start=1,
        training_iteration_id_end=2,
        global_batch_id_start=1,
    )
    runtime.begin_iteration(1, forward_only=True)
    runtime.end_iteration()
    runtime.seal_current_phase()

    assert host.boundaries == [
        ("model-a", 1, "valid", 0, "entry", 1),
        ("model-a", 1, "valid", 0, "exit", 2),
    ]
    assert host.submissions == [
        (
            "eval_phase_boundary",
            ("model-a", 1, "valid", 0, "entry", 1),
            ("string", "int64", "string", "int32", "string", "int64"),
            8,
        ),
        (
            "eval_phase_boundary",
            ("model-a", 1, "valid", 0, "exit", 2),
            ("string", "int64", "string", "int32", "string", "int64"),
            8,
        ),
    ]


def test_schedule_runtime_does_not_filter_event_context_by_direction():
    propagator = FakePropagatorWithContext({0: [5, 2]})
    runtime = MegatronScheduleRuntime(propagator)
    runtime.adaptor = FakeAdaptor()

    runtime.begin_iteration(1)
    runtime.set_event("bwd", 0, 0)
    runtime.enter_current_scope()

    assert len(runtime.adaptor.current_events) == 1
    assert runtime.adaptor.current_events[0].direction == "bwd"


def test_local_forward_boundary_checks_fwd_and_bwd_independently():
    runtime = MegatronScheduleRuntime(FakePropagator())
    runtime.adaptor = FakeReplayAdaptor()
    set_active_megatron_schedule_runtime(runtime)
    runner = FakeRunner()

    assert dmi_prepare_local_forward_boundary(runner) is False

    assert runtime.adaptor.capacity_calls == [
        ("fwd-plan", HookPhase.FWD, HookPhase.FWD),
        ("bwd-plan", HookPhase.BWD, HookPhase.BWD),
    ]
    assert runner._dmi_next_bwd_decision is StepReservation.RESERVED


def test_local_forward_boundary_fallbacks_without_summing_units():
    runtime = MegatronScheduleRuntime(FakePropagator())
    runtime.adaptor = FakeReplayAdaptor(
        [
            StepReservation.RESERVED,
            StepReservation.OVERSIZED,
        ]
    )
    set_active_megatron_schedule_runtime(runtime)
    runner = FakeRunner()

    assert dmi_prepare_local_forward_boundary(runner) is True

    assert runtime.adaptor.capacity_calls == [
        ("fwd-plan", HookPhase.FWD, HookPhase.FWD),
        ("bwd-plan", HookPhase.BWD, HookPhase.BWD),
    ]
    assert runner._dmi_next_bwd_decision is None


def test_local_forward_boundary_skips_bwd_for_forward_only_runner():
    runtime = MegatronScheduleRuntime(FakePropagator())
    runtime.adaptor = FakeReplayAdaptor()
    set_active_megatron_schedule_runtime(runtime)
    runner = FakeRunner(grad_enabled=False)

    assert dmi_prepare_local_forward_boundary(runner) is False

    assert runtime.adaptor.capacity_calls == [
        ("fwd-plan", HookPhase.FWD, HookPhase.FWD)
    ]
    assert runner._dmi_next_bwd_decision is None


def test_local_forward_and_backward_replay_prepare_plan_metadata():
    runtime = MegatronScheduleRuntime(FakePropagator())
    runtime.adaptor = FakeReplayAdaptor()
    set_active_megatron_schedule_runtime(runtime)
    runner = FakeRunner()
    runner._dmi_next_bwd_decision = StepReservation.RESERVED

    dmi_prepare_local_forward_replay(runner)
    token = dmi_take_local_backward_token(runner)
    runtime.adaptor.current_context = SimpleNamespace(direction="bwd")
    dmi_prepare_local_backward_replay(runner, token)

    assert runtime.adaptor.replay_calls == [
        ("fwd-plan", HookPhase.FWD, HookPhase.FWD),
        ("bwd-plan", HookPhase.BWD, HookPhase.BWD),
    ]
    assert token is StepReservation.RESERVED
    assert runner._dmi_next_bwd_decision is None


def test_local_backward_replay_requires_token_for_training_runner():
    runtime = MegatronScheduleRuntime(FakePropagator())
    runtime.adaptor = FakeReplayAdaptor()
    set_active_megatron_schedule_runtime(runtime)

    with pytest.raises(RuntimeError, match="backward replay is missing replay token"):
        dmi_prepare_local_backward_replay(FakeRunner(), None)

    assert runtime.adaptor.replay_calls == []


def test_force_eager_unit_preserves_public_runtime_state_and_propagates_errors():
    runtime = MegatronScheduleRuntime(FakePropagator())
    runtime.adaptor = FakeReplayAdaptor()
    set_active_megatron_schedule_runtime(runtime)

    with dmi_force_eager_unit():
        assert runtime.adaptor.current_context.direction == "fwd"

    with pytest.raises(RuntimeError, match="boom"):
        with dmi_force_eager_unit():
            raise RuntimeError("boom")
    assert runtime.adaptor.current_context.direction == "fwd"


def test_full_iteration_capture_end_iteration_clears_without_logical_commit():
    context = DMIMetadataContext(
        max_num_microbatches=1,
        max_batch_size=2,
        num_scopes=1,
        device="cpu",
    )
    runtime = MegatronScheduleRuntime(LocalMetadataPropagator(context))
    runtime.global_batch_id = 7
    runtime.execution_order_id = 4
    runtime.adaptor = FakeAdaptor()
    runtime.adaptor.hook_runtime.capture_event_context = True

    runtime.begin_full_iteration_capture([[16, 16]])
    runtime.begin_iteration(1)
    runtime.set_event("fwd", 0, 0)
    runtime.record_current_microbatch_metadata(None, None, None)
    runtime.enter_current_scope()
    assert runtime.active is True
    runtime.end_iteration()
    runtime.finish_full_iteration_capture()

    assert runtime.global_batch_id == 7
    assert runtime.execution_order_id == 4
    assert runtime.active is False
    assert runtime.current_event is None
    assert runtime.adaptor.clear_calls == 1
    assert context.source_cpu("valid_count", 0) == (16, 16)


def test_full_iteration_replay_uses_current_counts_and_advances_global_batch():
    context = DMIMetadataContext(
        max_num_microbatches=2,
        max_batch_size=2,
        num_scopes=1,
        device="cpu",
    )
    runtime = MegatronScheduleRuntime(LocalMetadataPropagator(context))
    runtime.adaptor = FakeReplayAdaptor()
    runtime.global_batch_id = 11
    set_active_megatron_schedule_runtime(runtime)
    plan = _full_iteration_plan((0, 1))

    fallback = dmi_prepare_full_iteration_replay(plan, [[6, 5], [4, 3]])
    dmi_finish_full_iteration_replay()

    assert fallback is False
    assert runtime.global_batch_id == 12
    assert len(runtime.adaptor.full_replay_calls) == 1
    _plan, contexts = runtime.adaptor.full_replay_calls[0]
    assert contexts[0].global_batch_id == 11
    assert contexts[0].microbatch_id == 0
    assert contexts[0].valid_counts == (6, 5)
    assert contexts[0].direction == "fwd"
    assert contexts[1].global_batch_id == 11
    assert contexts[1].microbatch_id == 1
    assert contexts[1].valid_counts == (4, 3)
    assert contexts[1].direction == "bwd"
    assert context.source_cpu("valid_count", 0) == (6, 5)
    assert context.source_cpu("valid_count", 1) == (4, 3)
    assert torch.equal(context.source("valid_count", 0), torch.tensor([6, 5]))
    assert torch.equal(context.source("valid_count", 1), torch.tensor([4, 3]))


def test_full_iteration_replay_receives_counts_on_non_source_pp_rank():
    context = DMIMetadataContext(
        max_num_microbatches=2,
        max_batch_size=2,
        num_scopes=1,
        device="cpu",
    )
    dist = FakeBroadcastDist(payloads=([6, 5], [4, 3]))
    propagator = PerDPCPUMetadataPropagator(
        context,
        rank=1,
        pp_source_rank=0,
        tp_source_rank=1,
        pp_cpu_ranks=[0, 1],
        tp_cpu_ranks=[1],
        pp_cpu_group="pp_cpu",
        tp_cpu_group=None,
        dist_module=dist,
    )
    runtime = MegatronScheduleRuntime(propagator)
    runtime.adaptor = FakeReplayAdaptor()
    set_active_megatron_schedule_runtime(runtime)

    fallback = dmi_prepare_full_iteration_replay(_full_iteration_plan((0, 1)), [None, None])
    dmi_finish_full_iteration_replay()

    assert fallback is False
    assert [call[1:] for call in dist.broadcast_calls] == [
        (0, "pp_cpu", True),
        (0, "pp_cpu", True),
    ]
    _plan, contexts = runtime.adaptor.full_replay_calls[0]
    assert contexts[0].valid_counts == (6, 5)
    assert contexts[1].valid_counts == (4, 3)
    assert context.source_cpu("valid_count", 0) == (6, 5)
    assert context.source_cpu("valid_count", 1) == (4, 3)


def test_non_source_rank_receives_dynamic_dataset_ids_without_local_batch_metadata():
    context = DMIMetadataContext(
        max_num_microbatches=1,
        max_batch_size=2,
        num_scopes=1,
        field_specs=(valid_count_field_spec(), dataset_id_field_spec()),
        device="cpu",
    )
    dist = FakeBroadcastDist(payloads=([6, 5, 4, 9],))
    propagator = PerDPCPUMetadataPropagator(
        context,
        rank=1,
        pp_source_rank=0,
        tp_source_rank=1,
        pp_cpu_ranks=[0, 1],
        tp_cpu_ranks=[1],
        pp_cpu_group="pp_cpu",
        tp_cpu_group=None,
        dist_module=dist,
    )
    runtime = MegatronScheduleRuntime(propagator)
    runtime.configure_dataset_provenance(
        {"train": "dynamic", "valid": "constant-zero", "test": "constant-zero"}
    )

    runtime.begin_iteration(1)
    runtime.set_event("fwd", 0, 0)
    runtime.record_current_microbatch_metadata(None, None, None)
    runtime.enter_current_scope()

    assert len(dist.broadcast_calls) == 1
    packet = dist.broadcast_calls[0][0]
    assert packet.dtype == torch.int64
    assert packet.numel() == 4
    assert context.source_cpu("valid_count", 0) == (6, 5)
    assert context.source_cpu("dataset_id", 0) == (4, 9)


def test_full_iteration_replay_fallback_does_not_advance_global_batch():
    context = DMIMetadataContext(
        max_num_microbatches=1,
        max_batch_size=2,
        num_scopes=1,
        device="cpu",
    )
    runtime = MegatronScheduleRuntime(LocalMetadataPropagator(context))
    runtime.adaptor = FakeReplayAdaptor([StepReservation.OVERSIZED])
    runtime.global_batch_id = 5
    set_active_megatron_schedule_runtime(runtime)

    fallback = dmi_prepare_full_iteration_replay(_full_iteration_plan((0,)), [[8, 1]])

    assert fallback is True
    assert runtime.global_batch_id == 5
    assert len(runtime.adaptor.full_replay_calls) == 1

    runtime.begin_iteration(1)
    runtime.set_event("fwd", 0, 0)
    runtime.record_current_microbatch_metadata(None, None, None)
    runtime.enter_current_scope()
    runtime.end_iteration()

    assert runtime.global_batch_id == 6


def test_full_iteration_replay_requires_current_valid_counts():
    context = DMIMetadataContext(
        max_num_microbatches=1,
        max_batch_size=2,
        num_scopes=1,
        device="cpu",
    )
    runtime = MegatronScheduleRuntime(LocalMetadataPropagator(context))
    runtime.adaptor = FakeReplayAdaptor()
    set_active_megatron_schedule_runtime(runtime)

    with pytest.raises(RuntimeError, match="missing valid counts"):
        dmi_prepare_full_iteration_replay(_full_iteration_plan((0,)), [None])

    assert runtime.adaptor.full_replay_calls == []


def test_schedule_runtime_clears_adaptor_event_at_iteration_end():
    propagator = FakePropagatorWithContext({0: [5, 2]})
    runtime = MegatronScheduleRuntime(propagator)
    runtime.adaptor = FakeAdaptor()

    runtime.begin_iteration(1)
    runtime.set_event("fwd", 0, 0)
    runtime.enter_current_scope()
    runtime.end_iteration()

    assert runtime.adaptor.clear_calls == 1


def test_schedule_guard_rejects_unsupported_modes_only_when_active():
    config = FakeConfig()
    config.overlap_moe_expert_parallel_comm = True

    dmi_guard_schedule_supported(config, forward_only=False)

    set_active_megatron_schedule_runtime(MegatronScheduleRuntime(FakePropagator()))
    with pytest.raises(NotImplementedError, match="MoE-overlap"):
        dmi_guard_schedule_supported(config, forward_only=False)

    config.overlap_moe_expert_parallel_comm = False
    config.hybrid_context_parallel = True
    with pytest.raises(NotImplementedError, match="hybrid context"):
        dmi_guard_schedule_supported(config, forward_only=False)

    config.hybrid_context_parallel = False
    config.context_parallel_size = 2
    with pytest.raises(NotImplementedError, match="context parallel"):
        dmi_guard_schedule_supported(config, forward_only=False)

    config.context_parallel_size = 1
    config.expert_model_parallel_size = 2
    dmi_guard_schedule_supported(config, forward_only=False)
    config.cuda_graph_scope = ("full_iteration",)
    dmi_guard_schedule_supported(config, forward_only=False)


def test_build_megatron_schedule_runtime_uses_local_for_pp1():
    runtime = build_megatron_schedule_runtime(
        max_num_microbatches=2,
        max_batch_size=3,
        parallel_state_module=FakeParallelState(tp_world_size=1, pp_world_size=1),
        dist_module=FakeDist(initialized=True, rank=0, world_size=1),
        device="cpu",
    )

    assert isinstance(runtime.propagator, LocalMetadataPropagator)
    assert runtime.propagator.context.num_scopes == 1


def test_rank_groups_for_token_reshape_model_parallel_as_pp_by_tp():
    for order in ("tp-cp-ep-dp-pp", "tp-cp-ep-pp-dp"):
        mp_groups = _rank_groups_for_token(
            token="tp-pp",
            tensor_model_parallel_size=2,
            pipeline_model_parallel_size=2,
            data_parallel_size=2,
            context_parallel_size=1,
            expert_model_parallel_size=1,
            rank_order=order,
        )
        tp_groups = {
            tuple(group)
            for group in _rank_groups_for_token(
                token="tp",
                tensor_model_parallel_size=2,
                pipeline_model_parallel_size=2,
                data_parallel_size=2,
                context_parallel_size=1,
                expert_model_parallel_size=1,
                rank_order=order,
            )
        }
        pp_groups = {
            tuple(group)
            for group in _rank_groups_for_token(
                token="pp",
                tensor_model_parallel_size=2,
                pipeline_model_parallel_size=2,
                data_parallel_size=2,
                context_parallel_size=1,
                expert_model_parallel_size=1,
                rank_order=order,
            )
        }
        for mp_group in mp_groups:
            matrix = [tuple(mp_group[i * 2 : (i + 1) * 2]) for i in range(2)]
            assert all(row in tp_groups for row in matrix)
            assert tuple(row[0] for row in matrix) in pp_groups
            assert tuple(row[1] for row in matrix) in pp_groups


@pytest.mark.parametrize("expert_model_parallel_size", (1, 2))
def test_build_megatron_schedule_runtime_uses_dmi_cpu_metadata_groups(
    expert_model_parallel_size,
):
    parallel_state = FakeParallelState(
        tp_world_size=2,
        pp_world_size=2,
        vp_world_size=4,
        tp_rank=1,
        pp_rank=0,
    )
    runtime = build_megatron_schedule_runtime(
        max_num_microbatches=2,
        max_batch_size=3,
        parallel_state_module=parallel_state,
        dist_module=FakeDist(initialized=True, rank=1, world_size=4),
        device="cpu",
        tensor_model_parallel_size=2,
        pipeline_model_parallel_size=2,
        data_parallel_size=1,
        expert_model_parallel_size=expert_model_parallel_size,
        rank_order="tp-cp-ep-dp-pp",
    )

    assert isinstance(runtime.propagator, PerDPCPUMetadataPropagator)
    assert runtime.propagator.rank == 1
    assert runtime.propagator.pp_source_rank == 0
    assert runtime.propagator.tp_source_rank == 0
    assert runtime.propagator.pp_cpu_ranks == (0, 2)
    assert runtime.propagator.tp_cpu_ranks == (0, 1)
    assert runtime.propagator.pp_cpu_group is None
    assert runtime.propagator.tp_cpu_group == ((0, 1), "gloo", "DMI_TP_METADATA_GLOO_0_0")
    assert runtime.propagator.context.num_scopes == 4
    assert parallel_state.created_groups == [
        ((0, 2), "gloo", "DMI_PP_METADATA_GLOO_0"),
        ((0, 1), "gloo", "DMI_TP_METADATA_GLOO_0_0"),
        ((2, 3), "gloo", "DMI_TP_METADATA_GLOO_0_1"),
    ]
