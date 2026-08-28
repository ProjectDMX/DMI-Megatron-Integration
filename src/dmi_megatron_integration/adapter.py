"""Megatron-owned hook/runtime binding for public DMI records."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import torch

from dmi.api.v1 import (
    HookOutput,
    HookPointV1,
    HookSpecV1,
    OutputStorage,
    ProducerPlan,
    ProducerPlanBuilder,
    ProducerPlanEntry,
    RecordRuntime,
    RecordType,
    StepReservation,
    TransportSpec,
    TransportType,
)

from .hooks.specs import (
    DimSpec,
    HookInputLayout,
    HookPhase,
    HookRuntimeMode,
    MegatronHookSpec,
    MegatronMetadataField,
)
from .metadata_context import DMIMetadataContext
from .records.format import required_record_metadata_fields
from .records.metadata import MegatronRecordMetadata


@dataclass(frozen=True)
class MegatronTrainingContext:
    """Metadata shared by every output for one Megatron schedule event."""

    global_batch_id: int
    microbatch_id: int
    valid_counts: Sequence[int]
    dataset_ids: Sequence[int] = ()
    attempt_id: int = 0
    direction: str = "fwd"
    phase: str = "train"
    scope_id: int = 0
    dp_rank: int = 0
    shard_rank: int = 0
    token_start: int = 0
    model_id: str | None = None


@dataclass(frozen=True)
class MegatronEventCoordinates:
    """Replay-stable coordinates captured at one Megatron hook emission."""

    microbatch_id: int
    direction: str
    phase: str = "train"
    scope_id: int = 0
    dp_rank: int = 0
    shard_rank: int = 0
    token_start: int = 0
    model_id: str | None = None

    @classmethod
    def from_context(cls, ctx: MegatronTrainingContext) -> "MegatronEventCoordinates":
        return cls(
            microbatch_id=int(ctx.microbatch_id),
            direction=str(ctx.direction),
            phase=str(ctx.phase),
            scope_id=int(ctx.scope_id),
            dp_rank=int(ctx.dp_rank),
            shard_rank=int(ctx.shard_rank),
            token_start=int(ctx.token_start),
            model_id=ctx.model_id,
        )

    def to_context(
        self,
        *,
        global_batch_id: int,
        valid_counts: Sequence[int],
        dataset_ids: Sequence[int] = (),
        attempt_id: int = 0,
    ) -> MegatronTrainingContext:
        return MegatronTrainingContext(
            global_batch_id=int(global_batch_id),
            microbatch_id=int(self.microbatch_id),
            valid_counts=tuple(int(value) for value in valid_counts),
            dataset_ids=tuple(int(value) for value in dataset_ids),
            attempt_id=int(attempt_id),
            direction=str(self.direction),
            phase=str(self.phase),
            scope_id=int(self.scope_id),
            dp_rank=int(self.dp_rank),
            shard_rank=int(self.shard_rank),
            token_start=int(self.token_start),
            model_id=self.model_id,
        )

    def signature_tuple(self) -> tuple[object, ...]:
        return (
            self.direction,
            self.phase,
            self.microbatch_id,
            self.scope_id,
            self.dp_rank,
            self.shard_rank,
            self.token_start,
            self.model_id,
        )


@dataclass(frozen=True)
class _ProducerSemantics:
    output_id: int
    act_name: str
    layer_no: int
    record_type: RecordType
    transport_type: TransportType
    record_dp_rank: int | None
    record_shard_rank: int | None
    need_token_range: bool
    suppress_recompute: bool
    record_direction: str

    @property
    def signature(self) -> tuple[object, ...]:
        return (
            self.output_id,
            self.act_name,
            self.layer_no,
            self.record_type,
            self.transport_type,
            self.record_dp_rank,
            self.record_shard_rank,
            self.need_token_range,
            self.suppress_recompute,
            self.record_direction,
        )


@dataclass(frozen=True)
class MegatronFullIterationPlanEntry:
    producer: ProducerPlanEntry
    event: MegatronEventCoordinates


@dataclass(frozen=True)
class MegatronFullIterationPlan:
    """Ordered full-iteration physical plan plus Megatron event coordinates."""

    entries: tuple[MegatronFullIterationPlanEntry, ...]
    _semantics: tuple[_ProducerSemantics, ...] = field(
        default=(), repr=False, compare=False
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "entries", tuple(self.entries))
        object.__setattr__(self, "_semantics", tuple(self._semantics))
        if len(self.entries) != len(self._semantics):
            raise ValueError(
                "full-iteration plan entry and semantic counts must match"
            )

    @classmethod
    def from_plan_and_events(
        cls,
        plan: ProducerPlan,
        events: Sequence[MegatronEventCoordinates],
        semantics: Sequence[_ProducerSemantics],
    ) -> "MegatronFullIterationPlan":
        if len(plan.entries) != len(events) or len(plan.entries) != len(semantics):
            raise ValueError(
                "full-iteration plan, event, and semantic counts must match"
            )
        return cls(
            tuple(
                MegatronFullIterationPlanEntry(producer=entry, event=event)
                for entry, event in zip(plan.entries, events)
            ),
            tuple(semantics),
        )

    @property
    def output_count(self) -> int:
        return len(self.entries)

    @property
    def total_aligned_nbytes(self) -> int:
        return self.producer_plan.total_reservation_bytes

    @property
    def producer_plan(self) -> ProducerPlan:
        return ProducerPlan(tuple(item.producer for item in self.entries))

    @property
    def signature(self) -> tuple[tuple[object, ...], ...]:
        physical = self.producer_plan.signature
        semantic = tuple(item.signature for item in self._semantics)
        events = tuple(item.event.signature_tuple() for item in self.entries)
        return tuple((*p, *s, *event) for p, s, event in zip(physical, semantic, events))

    def assert_compatible(self, other: "MegatronFullIterationPlan") -> None:
        if self.signature != other.signature:
            raise ValueError("Megatron full-iteration DMI plan signature mismatch")


@dataclass(frozen=True)
class _ConfiguredHook:
    hook: HookPointV1
    policy: MegatronHookSpec
    physical_spec: HookSpecV1
    scope_id: int
    record_dp_rank: int | None
    record_shard_rank: int | None


@dataclass(frozen=True)
class MegatronHookBinding:
    hook: HookPointV1
    scope_id: int = 0
    record_dp_rank: int | None = None
    record_shard_rank: int | None = None


@dataclass(frozen=True)
class MegatronRouterWeightBinding:
    hook: HookPointV1
    parameter: torch.nn.Parameter


class InvocationIdAllocator:
    """Rank-local invocation sequence for retained recomputation outputs."""

    def __init__(self) -> None:
        self._active: tuple[str, str, int, int] | None = None
        self._next_by_key: dict[tuple[object, ...], int] = {}

    def begin_attempt(
        self,
        *,
        model_id: str,
        phase: str,
        global_batch_id: int,
        attempt_id: int,
    ) -> None:
        if self._active is not None:
            raise RuntimeError("DMI invocation allocator attempt is already active")
        self._active = (
            str(model_id),
            str(phase),
            int(global_batch_id),
            int(attempt_id),
        )
        self._next_by_key.clear()

    def allocate(
        self,
        *,
        model_id: str,
        phase: str,
        global_batch_id: int,
        attempt_id: int,
        dp_rank: int,
        microbatch_id: int,
        direction: str,
        output_id: int,
        layer_no: int,
        shard_rank: int,
        token_start: int,
        record_type: RecordType,
    ) -> int:
        prefix = (str(model_id), str(phase), int(global_batch_id), int(attempt_id))
        if self._active is None or prefix != self._active:
            raise RuntimeError(
                "DMI invocation allocation does not match the active attempt: "
                f"active={self._active!r}, row={prefix!r}"
            )
        key = (
            *prefix,
            int(dp_rank),
            int(microbatch_id),
            str(direction),
            int(output_id),
            int(layer_no),
            int(shard_rank),
            int(token_start),
            int(record_type),
        )
        invocation_id = self._next_by_key.get(key, 0)
        self._next_by_key[key] = invocation_id + 1
        return invocation_id

    def end_attempt(self, *, attempt_id: int) -> None:
        if self._active is None:
            raise RuntimeError("DMI invocation allocator attempt is not active")
        if int(attempt_id) != self._active[3]:
            raise RuntimeError(
                "DMI invocation allocator ended the wrong attempt: "
                f"active={self._active[3]}, requested={int(attempt_id)}"
            )
        self._active = None
        self._next_by_key.clear()


class MegatronHookRuntime:
    """Public HookPointV1 runtime for eager, capture, and replay association."""

    def __init__(self, adaptor: "MegatronAdaptor") -> None:
        self.adaptor = adaptor
        self.mode = HookRuntimeMode.EAGER_IMMEDIATE
        self.warmup_enabled = False
        self.capture_event_context = False
        self.capture_direction: HookPhase | None = None
        self._capture_builder: ProducerPlanBuilder | None = None
        self._capture_semantics: list[_ProducerSemantics] | None = None
        self._capture_events: list[MegatronEventCoordinates] | None = None
        self._te_capture_session_active = False

    def begin_capture_plan(
        self,
        *,
        warmup_enabled: bool,
        capture_event_context: bool = False,
        capture_direction: HookPhase | str | None = None,
    ) -> None:
        if capture_event_context and capture_direction is not None:
            raise ValueError(
                "full-iteration DMI capture derives direction from schedule events"
            )
        if not warmup_enabled and not capture_event_context and capture_direction is None:
            raise ValueError(
                "non-warmup local DMI capture requires an explicit direction"
            )
        self.mode = HookRuntimeMode.CAPTURE_RECORD
        self.warmup_enabled = bool(warmup_enabled)
        self.capture_event_context = bool(capture_event_context)
        self.capture_direction = (
            None
            if capture_direction is None
            else self.adaptor.normalize_direction(capture_direction)
        )
        self._capture_builder = ProducerPlanBuilder()
        self._capture_semantics = []
        self._capture_events = [] if capture_event_context else None
        self._set_null_mode(self.warmup_enabled)

    def finish_capture_plan(self) -> ProducerPlan | MegatronFullIterationPlan:
        return self._finish_capture_plan(set_null_mode=True)

    def _finish_capture_plan(
        self, *, set_null_mode: bool
    ) -> ProducerPlan | MegatronFullIterationPlan:
        builder = self._capture_builder
        semantics = self._capture_semantics
        if builder is None or semantics is None:
            raise RuntimeError("DMI capture plan is not active")
        plan = builder.build()
        events = self._capture_events
        self._capture_builder = None
        self._capture_semantics = None
        self._capture_events = None
        self.warmup_enabled = False
        self.capture_event_context = False
        self.capture_direction = None
        self.mode = HookRuntimeMode.EAGER_IMMEDIATE
        if set_null_mode:
            self._set_null_mode(False)
        self.adaptor._remember_plan(plan, semantics)
        if events is not None:
            return MegatronFullIterationPlan.from_plan_and_events(
                plan, events, semantics
            )
        return plan

    def abort_capture_plan(self) -> None:
        self._abort_capture_plan(set_null_mode=True)

    def _abort_capture_plan(self, *, set_null_mode: bool) -> None:
        self._capture_builder = None
        self._capture_semantics = None
        self._capture_events = None
        self.warmup_enabled = False
        self.capture_event_context = False
        self.capture_direction = None
        self.mode = HookRuntimeMode.EAGER_IMMEDIATE
        if set_null_mode:
            self._set_null_mode(False)

    @property
    def te_capture_session_active(self) -> bool:
        return self._te_capture_session_active

    def begin_te_capture_session(self) -> None:
        if self._te_capture_session_active:
            raise RuntimeError("Megatron DMI TE capture session is already active")
        if self._capture_builder is not None or self.mode is not HookRuntimeMode.EAGER_IMMEDIATE:
            raise RuntimeError("Megatron DMI TE capture session requires an idle hook runtime")
        self._set_null_mode(True)
        self._te_capture_session_active = True

    def finish_te_capture_session(self) -> None:
        if not self._te_capture_session_active:
            raise RuntimeError("Megatron DMI TE capture session is not active")
        dangling_capture = self._capture_builder is not None
        if dangling_capture:
            self._abort_capture_plan(set_null_mode=False)
        try:
            self._set_null_mode(False)
        finally:
            self._te_capture_session_active = False
        if dangling_capture:
            raise RuntimeError(
                "Megatron DMI TE capture session ended with an active forward plan"
            )

    def begin_te_forward_capture(self) -> None:
        if not self._te_capture_session_active:
            raise RuntimeError(
                "Megatron DMI TE forward capture requires an active outer session"
            )
        if self._capture_builder is not None or self.mode is not HookRuntimeMode.EAGER_IMMEDIATE:
            raise RuntimeError("Megatron DMI TE forward capture is already active")
        self.mode = HookRuntimeMode.CAPTURE_RECORD
        self.warmup_enabled = True
        self.capture_direction = HookPhase.FWD
        self._capture_builder = ProducerPlanBuilder()
        self._capture_semantics = []
        self._capture_events = None

    def finish_te_forward_capture(self) -> ProducerPlan:
        if not self._te_capture_session_active:
            raise RuntimeError(
                "Megatron DMI TE forward capture requires an active outer session"
            )
        plan = self._finish_capture_plan(set_null_mode=False)
        if not isinstance(plan, ProducerPlan):
            raise TypeError("Megatron DMI TE capture produced a non-producer plan")
        return plan

    def abort_te_forward_capture(self) -> None:
        if not self._te_capture_session_active:
            raise RuntimeError(
                "Megatron DMI TE forward capture requires an active outer session"
            )
        self._abort_capture_plan(set_null_mode=False)

    def _set_null_mode(self, enabled: bool) -> None:
        self.adaptor.engine.set_capture_enabled(not bool(enabled))

    def should_emit(self, hook: HookPointV1) -> bool:
        if self._te_capture_session_active and self._capture_builder is None:
            return False
        if not self.adaptor._hook_suppress_recompute(hook):
            return True
        direction: HookPhase | str | None = None
        if self.mode is HookRuntimeMode.CAPTURE_RECORD:
            if self.capture_direction is not None:
                direction = self.capture_direction
            elif not self.capture_event_context:
                return True
        if direction is None:
            ctx = self.adaptor.current_context
            if ctx is None:
                raise RuntimeError(
                    "Megatron DMI hook eligibility requires an active schedule event"
                )
            direction = ctx.direction
        return self.adaptor.normalize_direction(direction) is self.adaptor._hook_phase(hook)

    def prepare_output(
        self,
        *,
        hook: HookPointV1,
        output_index: int,
        output_id: int,
        output_spec: TransportSpec,
        output: HookOutput,
    ) -> StepReservation | None:
        semantics = self.adaptor._producer_semantics(
            hook=hook,
            output_id=output_id,
            output_spec=output_spec,
        )
        builder = self._capture_builder
        if builder is None:
            return self.adaptor.emit_immediate_output(
                semantics=semantics,
                output_spec=output_spec,
                output=output,
            )
        entry = builder.record_output(
            output_id=int(output_id),
            output_spec=output_spec,
            output=output,
        )
        assert self._capture_semantics is not None
        self._capture_semantics.append(semantics)
        if self._capture_events is not None:
            ctx = self.adaptor.current_context
            if ctx is None:
                raise RuntimeError(
                    "Megatron full-iteration capture emitted without a schedule event"
                )
            self._capture_events.append(MegatronEventCoordinates.from_context(ctx))
        if self.warmup_enabled:
            return None
        return self.adaptor.emit_prepared_output(entry, semantics, output)


class MegatronIterationHookRuntime:
    """Immediate-only runtime for coordinator-owned iteration hooks."""

    mode = HookRuntimeMode.EAGER_IMMEDIATE
    warmup_enabled = False

    def __init__(self, adaptor: "MegatronAdaptor") -> None:
        self.adaptor = adaptor

    def should_emit(self, hook: HookPointV1) -> bool:
        configured = self.adaptor._configured_hook(hook)
        if self.adaptor._hook_phase(hook) is not HookPhase.ITERATION:
            raise RuntimeError("Iteration runtime received a non-ITERATION hook")
        if configured.policy.record_type is not RecordType.PER_ITERATION:
            raise RuntimeError("Iteration runtime received a non-PER_ITERATION hook")
        return True

    def prepare_output(
        self,
        *,
        hook: HookPointV1,
        output_index: int,
        output_id: int,
        output_spec: TransportSpec,
        output: HookOutput,
    ) -> StepReservation:
        del output_index
        semantics = self.adaptor._producer_semantics(
            hook=hook,
            output_id=output_id,
            output_spec=output_spec,
        )
        return self.adaptor.emit_iteration_output(semantics, output_spec, output)


class MegatronAdaptor:
    """Bind Megatron policy and schedule metadata to public DMI records."""

    def __init__(
        self,
        engine: Any,
        record_runtime: RecordRuntime[MegatronRecordMetadata],
        model_id: str,
        *,
        dims: Mapping[str | DimSpec, int] | None = None,
    ) -> None:
        self.engine = engine
        self.record_runtime = record_runtime
        self.model_id = str(model_id)
        self.dims: dict[str | DimSpec, int] = dict(dims or {})
        self.configured_hooks: list[_ConfiguredHook] = []
        self._configured_by_hook: dict[int, _ConfiguredHook] = {}
        self._plan_semantics_by_id: dict[int, tuple[_ProducerSemantics, ...]] = {}
        self.current_context: MegatronTrainingContext | None = None
        self.current_iteration_context: MegatronTrainingContext | None = None
        self.hook_runtime = MegatronHookRuntime(self)
        self.iteration_hook_runtime = MegatronIterationHookRuntime(self)
        self.invocation_allocator: InvocationIdAllocator | None = None

    def begin_capture_plan(
        self,
        *,
        warmup_enabled: bool = True,
        capture_event_context: bool = False,
        capture_direction: HookPhase | str | None = None,
    ) -> None:
        self.hook_runtime.begin_capture_plan(
            warmup_enabled=warmup_enabled,
            capture_event_context=capture_event_context,
            capture_direction=capture_direction,
        )

    def finish_capture_plan(self) -> ProducerPlan | MegatronFullIterationPlan:
        return self.hook_runtime.finish_capture_plan()

    def abort_capture_plan(self) -> None:
        self.hook_runtime.abort_capture_plan()

    @property
    def te_capture_session_active(self) -> bool:
        return self.hook_runtime.te_capture_session_active

    def begin_te_capture_session(self) -> None:
        self.hook_runtime.begin_te_capture_session()

    def finish_te_capture_session(self) -> None:
        self.hook_runtime.finish_te_capture_session()

    def begin_te_forward_capture(self) -> None:
        self.hook_runtime.begin_te_forward_capture()

    def finish_te_forward_capture(self) -> ProducerPlan:
        return self.hook_runtime.finish_te_forward_capture()

    def abort_te_forward_capture(self) -> None:
        self.hook_runtime.abort_te_forward_capture()

    def attach_model(
        self,
        model: torch.nn.Module | Sequence[torch.nn.Module],
        *,
        hook_selection: str = "full",
        dims: Mapping[str | DimSpec, int] | None = None,
        active_hook_ids: frozenset[int] | None = None,
        current_phase_tensor: torch.Tensor | None = None,
        metadata_context: DMIMetadataContext | None = None,
        activation_recompute_enabled: bool = False,
        active_input_layout: HookInputLayout = HookInputLayout.SEQ_BATCH,
    ) -> None:
        selected = self._collect_enabled_hooks(
            model, hook_selection, active_hook_ids=active_hook_ids
        )
        self.attach_hooks(
            model_hooks=tuple(
                MegatronHookBinding(hook=hook, scope_id=scope_id)
                for hook, scope_id in selected
            ),
            iteration_hooks=(),
            dims=dims,
            current_phase_tensor=current_phase_tensor,
            metadata_context=metadata_context,
            activation_recompute_enabled=activation_recompute_enabled,
            active_input_layout=active_input_layout,
        )

    def attach_hooks(
        self,
        *,
        model_hooks: Sequence[MegatronHookBinding],
        iteration_hooks: Sequence[MegatronHookBinding],
        dims: Mapping[str | DimSpec, int] | None = None,
        current_phase_tensor: torch.Tensor | None = None,
        metadata_context: DMIMetadataContext | None = None,
        activation_recompute_enabled: bool = False,
        active_input_layout: HookInputLayout = HookInputLayout.SEQ_BATCH,
    ) -> None:
        if dims is not None:
            self.dims = dict(dims)
        bindings = (*model_hooks, *iteration_hooks)
        iteration_ids = {id(item.hook) for item in iteration_hooks}
        if len({id(item.hook) for item in bindings}) != len(bindings):
            raise ValueError("DMI hook bindings contain a duplicate HookPointV1")
        self.configured_hooks = []
        self._configured_by_hook = {}
        for binding in bindings:
            hook = binding.hook
            policy = self._hook_policy(hook)
            is_iteration = id(hook) in iteration_ids
            self._validate_binding(
                binding,
                policy,
                is_iteration=is_iteration,
                active_input_layout=active_input_layout,
            )
            if metadata_context is not None and not is_iteration:
                phase = self._hook_phase(hook)
                for metadata_field in policy.binding_metadata_fields:
                    if metadata_field is MegatronMetadataField.VALID_COUNT:
                        self._bind_valid_count_context(
                            hook,
                            metadata_context,
                            binding.scope_id,
                            phase,
                        )
                    elif metadata_field is MegatronMetadataField.SEGMENT_METADATA:
                        self._bind_segment_context(
                            hook,
                            metadata_context,
                            binding.scope_id,
                            phase,
                        )
                    else:
                        raise ValueError(
                            "unsupported hook binding metadata field: "
                            f"{metadata_field.value}"
                        )
            physical = policy.resolve(self.dims)
            physical = HookSpecV1(
                name=physical.name,
                outputs=physical.outputs,
                preprocess=self._record_preprocess(hook, physical),
                enabled_by=physical.enabled_by,
            )
            hook.spec = physical
            configured = _ConfiguredHook(
                hook=hook,
                policy=policy,
                physical_spec=physical,
                scope_id=int(binding.scope_id),
                record_dp_rank=binding.record_dp_rank,
                record_shard_rank=binding.record_shard_rank,
            )
            self.configured_hooks.append(configured)
            self._configured_by_hook[id(hook)] = configured
            runtime = self.iteration_hook_runtime if is_iteration else self.hook_runtime
            gate_tensor = (
                current_phase_tensor
                if not is_iteration and self._hook_suppress_recompute(hook)
                else None
            )
            self.record_runtime.bind_hook(
                hook,
                hook_runtime=runtime,
                gate_tensor=gate_tensor,
                gate_value=int(self._hook_phase(hook).value),
            )
        retain_recompute = any(
            item.policy.record_type in (RecordType.PER_SAMPLE, RecordType.PER_EXECUTION)
            and not self._hook_suppress_recompute(item.hook)
            for item in self.configured_hooks
        )
        self.invocation_allocator = (
            InvocationIdAllocator()
            if bool(activation_recompute_enabled) and retain_recompute
            else None
        )

    def _validate_binding(
        self,
        binding: MegatronHookBinding,
        policy: MegatronHookSpec,
        *,
        is_iteration: bool,
        active_input_layout: HookInputLayout,
    ) -> None:
        hook = binding.hook
        if is_iteration:
            if policy.record_type is not RecordType.PER_ITERATION:
                raise ValueError("Iteration hook binding must use PER_ITERATION")
            if self._hook_phase(hook) is not HookPhase.ITERATION:
                raise ValueError("Iteration hook binding must use HookPhase.ITERATION")
            if binding.record_dp_rank is None or binding.record_shard_rank is None:
                raise ValueError("Iteration hook binding requires semantic rank coordinates")
            if policy.binding_metadata_fields:
                raise ValueError("Iteration hooks must not require binding metadata")
            return
        if policy.record_type not in (RecordType.PER_SAMPLE, RecordType.PER_EXECUTION):
            raise ValueError("Model hook binding must use PER_SAMPLE or PER_EXECUTION")
        if self._hook_phase(hook) not in (HookPhase.FWD, HookPhase.BWD):
            raise ValueError("Model hook binding must use FWD or BWD phase")
        if policy.record_type is RecordType.PER_SAMPLE:
            if binding.record_dp_rank is not None:
                raise ValueError("PER_SAMPLE hooks use the schedule-owned DP rank")
        elif binding.record_dp_rank is None or binding.record_shard_rank is None:
            raise ValueError(
                "PER_EXECUTION hooks require explicit physical rank coordinates"
            )
        if active_input_layout not in policy.supported_layouts:
            raise ValueError(
                f"DMI hook {policy.name!r} does not support input layout "
                f"{active_input_layout.value!r}"
            )

    def begin_attempt(self, *, phase: str, global_batch_id: int, attempt_id: int) -> None:
        if self.invocation_allocator is not None:
            self.invocation_allocator.begin_attempt(
                model_id=self.model_id,
                phase=phase,
                global_batch_id=global_batch_id,
                attempt_id=attempt_id,
            )

    def end_attempt(self, *, attempt_id: int) -> None:
        if self.invocation_allocator is not None:
            self.invocation_allocator.end_attempt(attempt_id=attempt_id)

    def set_current_event(self, ctx: MegatronTrainingContext) -> None:
        self.current_context = ctx

    def clear_current_event(self) -> None:
        self.current_context = None

    def set_current_iteration(self, ctx: MegatronTrainingContext) -> None:
        if self.current_iteration_context is not None:
            raise RuntimeError("Megatron DMI iteration context is already active")
        if ctx.direction != "iter" or ctx.phase != "train":
            raise ValueError("Megatron DMI iteration context must be train/iter")
        if ctx.global_batch_id < 0 or ctx.valid_counts or ctx.dataset_ids:
            raise ValueError("invalid Megatron DMI iteration context")
        self.current_iteration_context = ctx

    def clear_current_iteration(self) -> None:
        self.current_iteration_context = None

    @staticmethod
    def normalize_direction(direction: HookPhase | str) -> HookPhase:
        if isinstance(direction, HookPhase):
            return direction
        normalized = str(direction).strip().lower()
        if normalized in {"fwd", "forward"}:
            return HookPhase.FWD
        if normalized in {"bwd", "backward"}:
            return HookPhase.BWD
        raise ValueError(f"Unsupported Megatron DMI direction: {direction!r}")

    def emit_immediate_output(
        self,
        *,
        semantics: _ProducerSemantics,
        output_spec: TransportSpec,
        output: HookOutput,
    ) -> StepReservation:
        entry = ProducerPlanBuilder().record_output(
            output_id=semantics.output_id,
            output_spec=output_spec,
            output=output,
        )
        return self.emit_prepared_output(entry, semantics, output)

    def emit_prepared_output(
        self,
        entry: ProducerPlanEntry,
        semantics: _ProducerSemantics,
        output: HookOutput,
    ) -> StepReservation:
        ctx = self.current_context
        if ctx is None:
            raise RuntimeError("Megatron DMI hook emitted without a schedule event")
        metadata = self._record_metadata(ctx, semantics)
        return self.record_runtime.emit_output(entry, metadata, output)

    def emit_iteration_output(
        self,
        semantics: _ProducerSemantics,
        output_spec: TransportSpec,
        output: HookOutput,
    ) -> StepReservation:
        ctx = self.current_iteration_context
        if ctx is None:
            raise RuntimeError("Megatron DMI iteration hook requires an explicit context")
        if output_spec.transport_type is not TransportType.IDENTITY:
            raise ValueError("PER_ITERATION supports IDENTITY transport only")
        if output_spec.storage in (OutputStorage.SCALAR_FLOAT, OutputStorage.SCALAR_INT):
            if output.tensor.numel() != 1:
                raise ValueError("PER_ITERATION scalar output must contain one value")
        entry = ProducerPlanBuilder().record_output(
            output_id=semantics.output_id,
            output_spec=output_spec,
            output=output,
        )
        return self.record_runtime.emit_output(
            entry, self._record_metadata(ctx, semantics), output
        )

    def prepare_replay_capacity_only(
        self,
        plan: ProducerPlan,
        *,
        plan_direction: HookPhase | str,
        live_direction: HookPhase | str,
        expected_plan: ProducerPlan | None = None,
    ) -> StepReservation:
        entries, _ = self._select_replay_entries(
            plan,
            plan_direction=plan_direction,
            live_direction=live_direction,
            expected_plan=expected_plan,
        )
        if not entries:
            return StepReservation.SKIPPED
        selected = ProducerPlan(entries)
        capacities = self.engine.ring_capacities()
        if (
            selected.total_reservation_bytes > capacities.effective_bytes
            or selected.task_count > capacities.task_entries
        ):
            return StepReservation.OVERSIZED
        return StepReservation.RESERVED

    def prepare_replay(
        self,
        plan: ProducerPlan,
        ctx: MegatronTrainingContext | None = None,
        *,
        plan_direction: HookPhase | str,
        live_direction: HookPhase | str | None = None,
        expected_plan: ProducerPlan | None = None,
    ) -> StepReservation:
        replay_ctx = self.current_context if ctx is None else ctx
        if replay_ctx is None:
            raise RuntimeError("Megatron DMI replay requires an active schedule event")
        entries, semantics = self._select_replay_entries(
            plan,
            plan_direction=plan_direction,
            live_direction=(
                replay_ctx.direction if live_direction is None else live_direction
            ),
            expected_plan=expected_plan,
        )
        if not entries:
            return StepReservation.SKIPPED
        selected = ProducerPlan(entries)
        if self._plan_is_oversized(selected):
            return StepReservation.OVERSIZED
        metadata = tuple(self._record_metadata(replay_ctx, item) for item in semantics)
        return self.record_runtime.prepare_replay(selected, metadata)

    def prepare_full_iteration_replay(
        self,
        plan: MegatronFullIterationPlan,
        contexts: Sequence[MegatronTrainingContext],
        *,
        expected_plan: MegatronFullIterationPlan | None = None,
    ) -> StepReservation:
        if len(contexts) != plan.output_count:
            raise ValueError("Megatron full-iteration replay context count mismatch")
        if expected_plan is not None:
            expected_plan.assert_compatible(plan)
        self._validate_replay_plan(plan.producer_plan)
        self._validate_full_iteration_contexts(plan, contexts)
        if not plan.entries:
            return StepReservation.SKIPPED
        if self._plan_is_oversized(plan.producer_plan):
            return StepReservation.OVERSIZED
        metadata = tuple(
            self._record_metadata(ctx, semantic)
            for ctx, semantic in zip(contexts, plan._semantics)
        )
        return self.record_runtime.prepare_replay(plan.producer_plan, metadata)

    def _plan_is_oversized(self, plan: ProducerPlan) -> bool:
        capacities = self.engine.ring_capacities()
        return bool(
            plan.total_reservation_bytes > capacities.effective_bytes
            or plan.task_count > capacities.task_entries
        )

    def _select_replay_entries(
        self,
        plan: ProducerPlan,
        *,
        plan_direction: HookPhase | str,
        live_direction: HookPhase | str,
        expected_plan: ProducerPlan | None,
    ) -> tuple[tuple[ProducerPlanEntry, ...], tuple[_ProducerSemantics, ...]]:
        self._validate_replay_plan(plan, expected_plan=expected_plan)
        semantics = self._plan_semantics(plan)
        planned = self.normalize_direction(plan_direction)
        live = self.normalize_direction(live_direction)
        if planned is live:
            return plan.entries, semantics
        if planned is HookPhase.FWD and live is HookPhase.BWD:
            selected = tuple(
                (entry, semantic)
                for entry, semantic in zip(plan.entries, semantics)
                if not semantic.suppress_recompute
            )
            return (
                tuple(item[0] for item in selected),
                tuple(item[1] for item in selected),
            )
        raise RuntimeError(
            "Megatron DMI local CUDA graph direction mismatch: "
            f"captured={planned.name.lower()} live={live.name.lower()}"
        )

    def _validate_replay_plan(
        self,
        plan: ProducerPlan,
        *,
        expected_plan: ProducerPlan | None = None,
    ) -> None:
        if not isinstance(plan, ProducerPlan):
            raise TypeError("Megatron replay requires a public ProducerPlan")
        if expected_plan is not None:
            expected_plan.assert_compatible(plan)
            if tuple(item.signature for item in self._plan_semantics(expected_plan)) != tuple(
                item.signature for item in self._plan_semantics(plan)
            ):
                raise ValueError("Megatron producer plan semantic signature mismatch")
        for entry in plan.entries:
            if entry.transport_type in (TransportType.PREFIX_STRIP, TransportType.CHUNKED):
                raise NotImplementedError(
                    "Megatron DMI replay preparation does not yet support "
                    f"{entry.transport_type.value} transport"
                )

    @classmethod
    def _validate_full_iteration_contexts(
        cls,
        plan: MegatronFullIterationPlan,
        contexts: Sequence[MegatronTrainingContext],
    ) -> None:
        for index, (item, ctx) in enumerate(zip(plan.entries, contexts)):
            if int(item.event.microbatch_id) != int(ctx.microbatch_id):
                raise ValueError(
                    f"Megatron full-iteration microbatch mismatch at entry {index}"
                )
            if cls.normalize_direction(item.event.direction) is not cls.normalize_direction(
                ctx.direction
            ):
                raise ValueError(
                    f"Megatron full-iteration direction mismatch at entry {index}"
                )
            if item.producer.transport_type in (
                TransportType.SEQ_PREFIX_PACK,
                TransportType.SEGMENTED_PACK,
            ) and not ctx.valid_counts:
                raise ValueError(
                    "Megatron full-iteration packed replay requires valid counts "
                    f"at entry {index}"
                )

    def _record_metadata(
        self,
        ctx: MegatronTrainingContext,
        semantic: _ProducerSemantics,
    ) -> MegatronRecordMetadata:
        required_fields = required_record_metadata_fields(
            record_type=semantic.record_type,
            need_token_range=semantic.need_token_range,
            transport_type=semantic.transport_type,
            dynamic_dataset_provenance=bool(ctx.dataset_ids),
        )
        if semantic.record_type is RecordType.PER_SAMPLE:
            dp_rank = int(ctx.dp_rank)
            valid_counts = (
                tuple(int(value) for value in ctx.valid_counts)
                if MegatronMetadataField.VALID_COUNT in required_fields
                else ()
            )
            dataset_ids = (
                tuple(int(value) for value in ctx.dataset_ids)
                if MegatronMetadataField.DATASET_ID in required_fields
                else ()
            )
            token_start = int(ctx.token_start) if semantic.need_token_range else 0
        else:
            if semantic.record_dp_rank is None:
                raise RuntimeError("non-sample record is missing its semantic DP rank")
            dp_rank = int(semantic.record_dp_rank)
            valid_counts = ()
            dataset_ids = ()
            token_start = -1 if semantic.record_type is RecordType.PER_EXECUTION else 0
        shard_rank = (
            int(ctx.shard_rank)
            if semantic.record_shard_rank is None
            else int(semantic.record_shard_rank)
        )
        model_id = self.model_id if ctx.model_id is None else str(ctx.model_id)
        invocation_id = self._allocate_invocation_id(
            ctx,
            semantic=semantic,
            dp_rank=dp_rank,
            shard_rank=shard_rank,
            token_start=token_start,
            model_id=model_id,
        )
        return MegatronRecordMetadata(
            model_id=model_id,
            act_name=semantic.act_name,
            direction=semantic.record_direction,
            phase=str(ctx.phase),
            global_batch_id=int(ctx.global_batch_id),
            dp_rank=dp_rank,
            microbatch_id=int(ctx.microbatch_id),
            layer_no=int(semantic.layer_no),
            shard_rank=shard_rank,
            token_start=token_start,
            valid_counts=valid_counts,
            dataset_ids=dataset_ids,
            attempt_id=int(ctx.attempt_id),
            invocation_id=invocation_id,
        )

    def _allocate_invocation_id(
        self,
        ctx: MegatronTrainingContext,
        *,
        semantic: _ProducerSemantics,
        dp_rank: int,
        shard_rank: int,
        token_start: int,
        model_id: str,
    ) -> int:
        allocator = self.invocation_allocator
        if (
            allocator is None
            or semantic.suppress_recompute
            or ctx.phase != "train"
        ):
            return 0
        return allocator.allocate(
            model_id=model_id,
            phase=str(ctx.phase),
            global_batch_id=int(ctx.global_batch_id),
            attempt_id=int(ctx.attempt_id),
            dp_rank=dp_rank,
            microbatch_id=int(ctx.microbatch_id),
            direction=semantic.record_direction,
            output_id=semantic.output_id,
            layer_no=semantic.layer_no,
            shard_rank=shard_rank,
            token_start=token_start,
            record_type=semantic.record_type,
        )

    def _producer_semantics(
        self,
        *,
        hook: HookPointV1,
        output_id: int,
        output_spec: TransportSpec,
    ) -> _ProducerSemantics:
        configured = self._configured_hook(hook)
        hook_phase = self._hook_phase(hook)
        return _ProducerSemantics(
            output_id=int(output_id),
            act_name=str(output_spec.name),
            layer_no=int(configured.policy.layer_no),
            record_type=configured.policy.record_type,
            transport_type=output_spec.transport_type,
            record_dp_rank=configured.record_dp_rank,
            record_shard_rank=configured.record_shard_rank,
            need_token_range=bool(configured.policy.need_token_range),
            suppress_recompute=self._hook_suppress_recompute(hook),
            record_direction=(
                "iter"
                if hook_phase is HookPhase.ITERATION
                else hook_phase.name.lower()
            ),
        )

    def _remember_plan(
        self,
        plan: ProducerPlan,
        semantics: Sequence[_ProducerSemantics],
    ) -> None:
        if len(plan.entries) != len(semantics):
            raise ValueError("producer plan semantic count mismatch")
        self._plan_semantics_by_id[id(plan)] = tuple(semantics)

    def _plan_semantics(self, plan: ProducerPlan) -> tuple[_ProducerSemantics, ...]:
        semantics = self._plan_semantics_by_id.get(id(plan))
        if semantics is None:
            # Subsets are constructed only immediately before a public runtime
            # call; persistent replay plans must originate from this adaptor.
            raise RuntimeError("Megatron producer plan is not owned by this adaptor")
        return semantics

    def _configured_hook(self, hook: HookPointV1) -> _ConfiguredHook:
        configured = self._configured_by_hook.get(id(hook))
        if configured is None:
            raise RuntimeError("Megatron DMI hook was invoked without registration")
        return configured

    @staticmethod
    def _hook_policy(hook: HookPointV1) -> MegatronHookSpec:
        policy = getattr(hook, "_dmi_megatron_spec", None)
        if not isinstance(policy, MegatronHookSpec):
            raise TypeError("Megatron HookPointV1 is missing MegatronHookSpec policy")
        return policy

    @staticmethod
    def _hook_phase(hook: HookPointV1) -> HookPhase:
        phase = getattr(hook, "hook_phase", None)
        if not isinstance(phase, HookPhase):
            raise TypeError("Megatron HookPointV1 is missing HookPhase")
        return phase

    @staticmethod
    def _hook_suppress_recompute(hook: HookPointV1) -> bool:
        value = getattr(hook, "suppress_recompute", None)
        if not isinstance(value, bool):
            raise TypeError("Megatron HookPointV1 is missing suppress_recompute")
        return value

    def _collect_enabled_hooks(
        self,
        model: torch.nn.Module | Sequence[torch.nn.Module],
        hook_selection: str,
        *,
        active_hook_ids: frozenset[int] | None,
    ) -> list[tuple[HookPointV1, int]]:
        selected = {name.strip() for name in str(hook_selection).split(",")}
        if "" in selected:
            raise ValueError(f"Invalid empty DMI hook selection: {hook_selection!r}")
        roots = self._model_roots(model)
        hooks: list[tuple[HookPointV1, int]] = []
        for root_index, root in enumerate(roots):
            vp_stage = getattr(root, "vp_stage", None)
            scope_id = int(
                vp_stage
                if vp_stage is not None
                else root_index if len(roots) > 1 else 0
            )
            for module in root.modules():
                if not isinstance(module, HookPointV1):
                    continue
                policy = self._hook_policy(module)
                if policy.enabled_by and not (selected & set(policy.enabled_by)):
                    continue
                if active_hook_ids is not None and id(module) not in active_hook_ids:
                    module.enabled = False
                    continue
                hooks.append((module, scope_id))
        return hooks

    @staticmethod
    def _model_roots(
        model: torch.nn.Module | Sequence[torch.nn.Module],
    ) -> list[torch.nn.Module]:
        return [model] if isinstance(model, torch.nn.Module) else list(model)

    @staticmethod
    def _bind_valid_count_context(
        hook: HookPointV1,
        metadata_context: DMIMetadataContext,
        scope_id: int,
        phase: HookPhase,
    ) -> None:
        if phase not in (HookPhase.FWD, HookPhase.BWD):
            raise ValueError("valid-count metadata requires an FWD or BWD hook phase")
        direction = phase.name.lower()
        setattr(
            hook,
            f"valid_count_{direction}",
            metadata_context.current("valid_count", direction, scope_id),
        )

    @staticmethod
    def _bind_segment_context(
        hook: HookPointV1,
        metadata_context: DMIMetadataContext,
        scope_id: int,
        phase: HookPhase,
    ) -> None:
        if phase not in (HookPhase.FWD, HookPhase.BWD):
            raise ValueError("segment metadata requires an FWD or BWD hook phase")
        direction = phase.name.lower()
        segments = metadata_context.current("segment_metadata", direction, scope_id)
        if segments.numel() == 0 or segments.numel() % 2 != 0:
            raise ValueError("DMI segment_metadata requires equal non-empty halves")
        capacity = segments.numel() // 2
        setattr(hook, f"sample_start_ptr_{direction}", segments[:capacity])
        setattr(hook, f"sample_end_ptr_{direction}", segments[capacity:])

    def _record_preprocess(
        self,
        hook: HookPointV1,
        physical: HookSpecV1,
    ) -> Any:
        original = physical.preprocess
        if not any(
            item.transport_type in (
                TransportType.SEQ_PREFIX_PACK,
                TransportType.SEGMENTED_PACK,
            )
            for item in physical.outputs
        ):
            return original

        def preprocess(*inputs: Any) -> Any:
            raw = (
                original(*inputs)
                if original is not None
                else inputs[0] if len(physical.outputs) == 1 and len(inputs) == 1 else inputs
            )
            values = [raw] if len(physical.outputs) == 1 else list(raw)
            if len(values) != len(physical.outputs):
                raise ValueError("Megatron hook preprocessing output count mismatch")
            enriched = [
                self._enrich_output(hook, spec, value)
                for spec, value in zip(physical.outputs, values)
            ]
            return enriched[0] if len(enriched) == 1 else tuple(enriched)

        return preprocess

    def _enrich_output(
        self,
        hook: HookPointV1,
        spec: TransportSpec,
        value: Any,
    ) -> Any:
        if isinstance(value, HookOutput):
            output = value
        elif isinstance(value, torch.Tensor):
            output = HookOutput(value)
        elif isinstance(value, tuple) and value and isinstance(value[0], torch.Tensor):
            output = HookOutput(value[0], tuple(value[1:]))
        else:
            raise TypeError("Megatron hook output must begin with a Tensor")
        if spec.transport_type is TransportType.SEQ_PREFIX_PACK:
            valid = self._current_valid_count(hook)
            prefix = getattr(hook, "_dmi_valid_prefix_sum", None)
            needed = int(valid.numel()) + 1
            if (
                prefix is None
                or prefix.device != valid.device
                or prefix.dtype != valid.dtype
                or prefix.numel() != needed
            ):
                prefix = torch.empty(needed, dtype=valid.dtype, device=valid.device)
                hook._dmi_valid_prefix_sum = prefix
            prefix[0].zero_()
            torch.cumsum(valid, dim=0, out=prefix[1:])
            return HookOutput(output.tensor, (valid, prefix))
        if spec.transport_type is TransportType.SEGMENTED_PACK:
            starts, ends = self._current_segment_ranges(hook)
            return HookOutput(output.tensor, (starts, ends))
        return output

    def _current_valid_count(self, hook: HookPointV1) -> torch.Tensor:
        phase = self._hook_phase(hook)
        value = (
            getattr(hook, "valid_count_fwd", None)
            if phase is HookPhase.FWD
            else getattr(hook, "valid_count_bwd", None)
        )
        if not isinstance(value, torch.Tensor) or value.dtype is not torch.int64:
            raise TypeError("SEQ_PREFIX_PACK requires int64 valid-count context")
        return value

    def _current_segment_ranges(
        self, hook: HookPointV1
    ) -> tuple[torch.Tensor, torch.Tensor]:
        suffix = "fwd" if self._hook_phase(hook) is HookPhase.FWD else "bwd"
        starts = getattr(hook, f"sample_start_ptr_{suffix}", None)
        ends = getattr(hook, f"sample_end_ptr_{suffix}", None)
        if not isinstance(starts, torch.Tensor) or not isinstance(ends, torch.Tensor):
            raise TypeError("SEGMENTED_PACK requires segment-range context")
        return starts, ends


__all__ = [
    "InvocationIdAllocator",
    "MegatronAdaptor",
    "MegatronEventCoordinates",
    "MegatronFullIterationPlan",
    "MegatronFullIterationPlanEntry",
    "MegatronHookBinding",
    "MegatronHookRuntime",
    "MegatronIterationHookRuntime",
    "MegatronRouterWeightBinding",
    "MegatronTrainingContext",
]
