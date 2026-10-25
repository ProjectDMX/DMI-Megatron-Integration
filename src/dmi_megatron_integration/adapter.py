"""Megatron-owned hook/runtime binding for public DMI records."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from math import prod
from typing import Any, Callable, Mapping, Sequence
import weakref

import torch
from dmi.hooks import point as _hook_point
from dmi.hooks.producer_plan import _align_up, _dtype_element_size, _payload_alignment
from dmi.hooks.record import _as_hook_output, _normalize_outputs

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

from .hooks.selection import hook_enabled_in_phase
from .hooks.specs import (
    DimSpec,
    HookInputLayout,
    HookPhase,
    HookRuntimeMode,
    MegatronDistributedInfo,
    MegatronHookSpec,
    MegatronMetadataField,
    ShardPolicy,
)
from .metadata_context import (
    DMIMetadataContext, _PreparedPacking, _normalize_int_tuple, _prepare_valid_counts,
)
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
    _packing: _PreparedPacking | None = field(default=None, repr=False, compare=False)

    @classmethod
    def _trusted(cls, **fields: Any) -> "MegatronTrainingContext":
        """Schedule-runtime constructor for already normalized fields."""
        ctx = object.__new__(cls)
        for name in cls.__dataclass_fields__:
            object.__setattr__(ctx, name, fields.get(name, cls.__dataclass_fields__[name].default))
        return ctx

    def __post_init__(self) -> None:
        counts = _normalize_int_tuple(self.valid_counts)
        object.__setattr__(self, "valid_counts", counts)
        object.__setattr__(self, "dataset_ids", _normalize_int_tuple(self.dataset_ids))
        for name in (
            "global_batch_id", "microbatch_id", "attempt_id", "scope_id", "dp_rank",
            "shard_rank", "token_start",
        ):
            object.__setattr__(self, name, int(getattr(self, name)))
        object.__setattr__(self, "phase", str(self.phase))
        if self.model_id is not None:
            object.__setattr__(self, "model_id", str(self.model_id))
        if self._packing is None:
            object.__setattr__(self, "_packing", _PreparedPacking(_prepare_valid_counts(counts)))
        elif self._packing.global_counts.counts != counts:
            raise ValueError("schedule valid counts disagree with prepared CPU packing metadata")


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
            valid_counts=valid_counts,
            dataset_ids=dataset_ids,
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
    tp_sequence_start: int | None = None
    tp_sequence_length: int | None = None
    packed_size_fn: Callable[[int, int], int] | None = field(
        default=None, repr=False, compare=False,
    )
    eager_nbytes: Callable[[], int] | None = field(default=None, repr=False, compare=False)
    eager_transport_args: tuple[int, ...] | None = field(default=None, repr=False, compare=False)
    requires_valid_count: bool | None = field(default=None, repr=False, compare=False)

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
            self.tp_sequence_start,
            self.tp_sequence_length,
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
    _producer_plan: ProducerPlan = field(init=False, repr=False, compare=False)
    _sizing: _CapturedSizing = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "entries", tuple(self.entries))
        object.__setattr__(self, "_semantics", tuple(self._semantics))
        if len(self.entries) != len(self._semantics):
            raise ValueError(
                "full-iteration plan entry and semantic counts must match"
            )
        plan = ProducerPlan(tuple(item.producer for item in self.entries))
        object.__setattr__(self, "_producer_plan", plan)
        object.__setattr__(self, "_sizing", _CapturedSizing(
            plan.entries, self._semantics, tuple(item.event.microbatch_id for item in self.entries),
        ))

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
        return self._producer_plan

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
    tp_sequence_start: int | None
    tp_sequence_length: int | None
    output_semantics: tuple[_ProducerSemantics, ...] = ()


def _make_packed_size_fn(feature_bytes: int, *, local: bool) -> Callable[[int, int], int]:
    alignment = _payload_alignment()
    if local:
        def size_fn(token_count: int, local_token_count: int) -> int:
            return _align_up(local_token_count * feature_bytes, alignment)
    else:
        def size_fn(token_count: int, local_token_count: int) -> int:
            return _align_up(token_count * feature_bytes, alignment)
    return size_fn


class _CapturedSizing:
    """Graph-owned sizing function; live counts exist only during resolve()."""

    def __init__(self, entries, semantics, microbatches=None):
        self.entries = tuple(entries)
        self.semantics = tuple(semantics)
        groups = tuple(0 for _ in entries) if microbatches is None else tuple(microbatches)
        self.groups = groups
        self._representatives = tuple(dict.fromkeys(groups))
        alignment = _payload_alignment()
        coefficients = {key: [0, 0] for key in self._representatives}
        constant = 0
        eligible = True
        local_intervals = set()
        for entry, semantic, group in zip(entries, semantics, groups):
            if entry.transport_type is TransportType.IDENTITY:
                value = prod(entry.input_shape) * entry.element_size
                constant += value
                eligible &= value % alignment == 0
            elif entry.transport_type is TransportType.SEQ_PREFIX_PACK:
                value = int(entry.transport_args[0])
                local = semantic.tp_sequence_start is not None
                coefficients[group][int(local)] += value
                eligible &= value % alignment == 0
                if local:
                    local_intervals.add((semantic.tp_sequence_start, semantic.tp_sequence_length))
            else:
                eligible = False
        pairs = tuple(tuple(pair) for pair in coefficients.values())
        eligible &= len(set(pairs)) <= 1 and len(local_intervals) <= 1
        self._local_interval = next(iter(local_intervals), None)
        self._parameters = None
        self._resolved = None
        if eligible:
            a, c = pairs[0] if pairs else (0, 0)
            self.mode = "simplified"
            self.coefficients = (a, c, constant)

            def size_fn(token_count, local_token_count):
                return a * token_count + c * local_token_count + constant
        else:
            self.mode = "recursive"
            self.coefficients = None

            def child(index):
                def leaf(token_count, local_token_count):
                    value = MegatronAdaptor._resolved_output_bytes(
                        self.entries[index], self.semantics[index], self._parameters[index],
                    )
                    self._resolved.append(value)
                    return value
                return leaf

            children = tuple(child(index) for index in range(len(entries)))

            def size_fn(token_count, local_token_count):
                return sum(leaf(token_count, local_token_count) for leaf in children)
        self.size_fn = size_fn

    def resolve(self, packings):
        if len(packings) != len(self.entries):
            raise ValueError("graph sizing context count mismatch")
        if self.mode == "recursive":
            self._parameters = tuple(packings)
            self._resolved = []
            try:
                total = self.size_fn(0, 0)
                return total, tuple(self._resolved)
            finally:
                self._parameters = None
                self._resolved = None
        representatives = {}
        for group, packing in zip(self.groups, packings):
            representatives.setdefault(group, packing)
        global_total = 0
        local_total = 0
        for packing in representatives.values():
            if packing is not None:
                global_total += packing.global_counts.token_count
                if self._local_interval is not None:
                    start, length = self._local_interval
                    key = (start, length)
                    if key not in packing.local_counts:
                        packing.local_counts[key] = _prepare_valid_counts(
                            packing.global_counts.counts, sequence_start=start, sequence_length=length,
                        )
                    local_total += packing.local_counts[key].token_count
        total = self.size_fn(global_total, local_total)
        sizes = tuple(MegatronAdaptor._resolved_output_bytes(entry, semantic, packing)
                      for entry, semantic, packing in zip(self.entries, self.semantics, packings))
        return total, sizes


@dataclass
class _CapturedPlanState:
    owner: Any
    semantics: tuple[_ProducerSemantics, ...]
    sizing: _CapturedSizing
    selected: dict[tuple[HookPhase, HookPhase], ProducerPlan] = field(default_factory=dict)


@dataclass(frozen=True)
class _StaticStep:
    """One planned hook firing: its own preprocessing and (output index, slot) pairs."""

    hook: HookPointV1
    preprocess: Callable[..., Any] | None
    output_count: int
    slots: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class _StaticLaunch:
    """Native launcher whose slots were bound once at plan confirmation."""

    steps: tuple[_StaticStep, ...]
    launcher: Any
    fire: Callable[[int, torch.Tensor], None]
    set_counts: Callable[[int, int, int], None]
    packed_slots: tuple[tuple[int, HookPointV1], ...]


class _GraphedPreprocess:
    """Replay a graph-safe, fixed-shape hook preprocess as one CUDA graph.

    Inputs are copied into captured buffers; outputs are the captured buffers,
    consumed by producers already ordered on the same stream. Any other input
    shape or dtype runs the original function eagerly.
    """

    def __init__(self, fn: Callable[..., Any]) -> None:
        self.fn = fn
        self.graph = None
        self.inputs: tuple[torch.Tensor, ...] = ()
        self.outputs: Any = None

    def __call__(self, *inputs: Any) -> Any:
        if self.graph is None:
            if not inputs or not all(isinstance(x, torch.Tensor) and x.is_cuda for x in inputs):
                return self.fn(*inputs)
            self._capture(inputs)
        captured = self.inputs
        if len(inputs) != len(captured) or any(
            x.shape != c.shape or x.dtype != c.dtype for x, c in zip(inputs, captured)
        ):
            return self.fn(*inputs)
        for target, value in zip(captured, inputs):
            target.copy_(value)
        self.graph.replay()
        return self.outputs

    def _capture(self, inputs: tuple[torch.Tensor, ...]) -> None:
        self.inputs = tuple(x.detach().clone() for x in inputs)
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            self.fn(*self.inputs)
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        # Thread-local capture: MIND's drain thread keeps using its own stream.
        with torch.cuda.graph(graph, capture_error_mode="thread_local"):
            self.outputs = self.fn(*self.inputs)
        self.graph = graph


@dataclass(frozen=True)
class _StaticBatchPlan:
    """Confirmed eager output order for one (phase, direction, scope) event."""

    plan: ProducerPlan
    hook_ids: tuple[int, ...]
    output_ids: tuple[int, ...]
    launch: _StaticLaunch | None = None
    replay_cache: dict = field(default_factory=dict, compare=False)


_MISSING = object()


def _native_extension():
    try:
        from dmi.transport.native import _load_extension
        return _load_extension()
    except Exception:  # no native backend: the per-hook path remains
        return None


def _MONITORING_DEBUG() -> bool:
    return bool(_hook_point._MONITORING_DEBUG)




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
        key = self._key(
            model_id=model_id, phase=phase, global_batch_id=global_batch_id,
            attempt_id=attempt_id, dp_rank=dp_rank, microbatch_id=microbatch_id,
            direction=direction, output_id=output_id, layer_no=layer_no,
            shard_rank=shard_rank, token_start=token_start, record_type=record_type,
        )
        invocation_id = self._next_by_key.get(key, 0)
        self._next_by_key[key] = invocation_id + 1
        return invocation_id

    def is_first(self, **allocation: Any) -> bool:
        """Whether ``allocate(**allocation)`` would return 0, without allocating."""
        return self._key(**allocation) not in self._next_by_key

    def _key(
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
    ) -> tuple[object, ...]:
        prefix = (str(model_id), str(phase), int(global_batch_id), int(attempt_id))
        if self._active is None or prefix != self._active:
            raise RuntimeError(
                "DMI invocation allocation does not match the active attempt: "
                f"active={self._active!r}, row={prefix!r}"
            )
        return (
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
        # Opt-in eager static batch: one reservation per non-train microbatch
        # once two consecutive microbatches emit the same static outputs.
        self.eager_static_batch_enabled = False
        self._static_plans: dict[tuple[str, str, int], _StaticBatchPlan] = {}
        self._static_candidates: dict[tuple[str, str, int], tuple] = {}
        self._static_ineligible: set[tuple[str, str, int]] = set()
        self._static_recording: list | None = None
        self._static_recording_key: tuple[str, str, int] | None = None
        self._static_armed: _StaticBatchPlan | None = None
        self._static_cursor = 0
        self._static_step = 0
        self._static_bound_counts = None
        # HookPointV1 fast-path entry; set only while a static batch is armed.
        self.fast_forward = None

    def reset_static_batches(self) -> None:
        self._static_plans.clear()
        self._static_candidates.clear()
        self._static_ineligible.clear()
        self._static_recording = None
        self._static_recording_key = None
        self._static_armed = None
        self._static_cursor = 0
        self._static_step = 0
        self._static_bound_counts = None
        self.fast_forward = None

    def begin_static_microbatch(self, ctx: MegatronTrainingContext) -> None:
        """Reserve a confirmed static plan before this microbatch's forward.

        Unconfirmed events record their eager outputs instead. Train phases,
        capture sessions, and oversized or skipped batches use the per-hook path.
        """

        self.finish_static_microbatch()
        if (
            ctx.phase == "train"
            or self.mode is not HookRuntimeMode.EAGER_IMMEDIATE
            or self._capture_builder is not None
            or self._te_capture_session_active
        ):
            return
        key = (ctx.phase, ctx.direction, ctx.scope_id)
        static = self._static_plans.get(key)
        if static is None:
            if key not in self._static_ineligible:
                self._static_recording = []
                self._static_recording_key = key
            return
        reservation = self._submit_static_batch(static, ctx)
        if reservation in (StepReservation.OVERSIZED, StepReservation.SKIPPED):
            return
        self._static_armed = static
        self._static_cursor = 0
        self._static_step = 0
        if static.launch is not None and not _MONITORING_DEBUG():
            # Graph-like eager path: hooks only run their own preprocessing
            # and launch producers into the slots reserved above.
            # Rebind counts only when this microbatch's count tensors changed
            # (value-cached direct counts repeat across microbatches).
            context = self.adaptor._metadata_context
            counts = None if context is None else context._direct_microbatches.get(
                self.adaptor._direct_packed_microbatch)
            if counts is None or counts is not self._static_bound_counts:
                set_counts = static.launch.set_counts
                for slot, hook in static.launch.packed_slots:
                    valid, prefix = self.adaptor._packed_meta(hook)
                    set_counts(slot, valid.data_ptr(), prefix.data_ptr())
                self._static_bound_counts = counts
            self.fast_forward = self._fast_forward

    def finish_static_microbatch(self) -> None:
        armed = self._static_armed
        if armed is not None:
            self._static_armed = None
            self.fast_forward = None
            if self._static_cursor != len(armed.output_ids):
                raise RuntimeError(
                    "DMI eager static batch reserved "
                    f"{len(armed.output_ids)} outputs but {self._static_cursor} fired"
                )
            return
        recording = self._static_recording
        if recording is None:
            return
        key = self._static_recording_key
        self._static_recording = None
        self._static_recording_key = None
        signature = tuple(recording)
        if not signature or any(item is None for item in signature):
            self._static_candidates.pop(key, None)
            self._mark_static_ineligible(key, "non-static output")
            return
        candidate = self._static_candidates.pop(key, None)
        if candidate is None:
            self._static_candidates[key] = signature
            return
        if candidate != signature:
            self._mark_static_ineligible(key, "outputs changed between microbatches")
            return
        hooks, output_indices, entries, semantics = zip(*signature)
        plan = ProducerPlan(tuple(entries))
        self.adaptor._remember_plan(plan, semantics)
        launch = self._build_static_launch(hooks, output_indices, entries)
        self._static_plans[key] = _StaticBatchPlan(
            plan, tuple(id(hook) for hook in hooks), tuple(item.output_id for item in semantics),
            launch=launch,
        )
        print(f"DMI_EAGER_STATIC_BATCH_PLAN phase={key[0]} direction={key[1]} "
              f"scope={key[2]} outputs={len(entries)} fast_launch={launch is not None}",
              flush=True)

    def _mark_static_ineligible(self, key: tuple[str, str, int], reason: str) -> None:
        self._static_ineligible.add(key)
        print(f"DMI_EAGER_STATIC_BATCH_INELIGIBLE phase={key[0]} direction={key[1]} "
              f"scope={key[2]} reason={reason}", flush=True)

    def _record_static_output(
        self,
        hook: HookPointV1,
        output_index: int,
        semantics: _ProducerSemantics,
        output_spec: TransportSpec,
        output: HookOutput,
    ) -> None:
        if semantics.eager_nbytes is not None or output_spec.transport_type not in (
            TransportType.IDENTITY, TransportType.SEQ_PREFIX_PACK,
        ):
            self._static_recording.append(None)
            return
        entry, _ = ProducerPlanEntry._from_eager_output(
            output_id=semantics.output_id,
            output_spec=output_spec,
            output=output,
            transport_args=(semantics.eager_transport_args
                            if semantics.eager_transport_args is not None
                            else ProducerPlanBuilder._transport_args(output_spec)),
        )
        self._static_recording.append((hook, int(output_index), entry, semantics))

    def _consume_static_output(
        self, armed: _StaticBatchPlan, hook: HookPointV1, output_id: int, output: HookOutput,
    ) -> StepReservation:
        # Descriptors are already queued, so a mismatch cannot fall back.
        index = self._static_cursor
        if index >= len(armed.output_ids):
            raise RuntimeError("DMI eager static batch received an unplanned output")
        entry = armed.plan.entries[index]
        tensor = output.tensor
        if (
            armed.output_ids[index] != output_id
            or armed.hook_ids[index] != id(hook)
            or tensor.shape != entry.input_shape
            or tensor.dtype != entry.dtype
        ):
            raise RuntimeError(
                f"DMI eager static batch output {index} does not match the reserved plan"
            )
        self._static_cursor = index + 1
        return StepReservation.RESERVED

    def _build_static_launch(self, hooks, output_indices, entries) -> "_StaticLaunch | None":
        """Bind native producer slots per planned output; None keeps the per-hook path."""
        engine = getattr(getattr(self.adaptor.record_runtime, "_transport", None), "_ring_engine", None)
        launcher_type = getattr(_native_extension(), "StaticLauncher", None)
        if engine is None or launcher_type is None:
            return None
        launcher = launcher_type(engine)
        steps, packed_slots = [], []
        for hook, output_index, entry in zip(hooks, output_indices, entries):
            info = self.adaptor._fast_hook_info.get(id(hook))
            packed = entry.transport_type is TransportType.SEQ_PREFIX_PACK
            if (info is None or not info[2] or hook._ring_payload is None
                    or "_dispatch" in vars(hook)
                    or type(hook)._dispatch is not HookPointV1._dispatch
                    or (packed and id(hook) not in self.adaptor._eager_packed_info)):
                # Custom dispatch, a real sequence-shard narrow, or unbound
                # packing metadata keeps the per-hook path.
                return None
            gate = hook._gate_tensor
            slot = launcher.add(
                packed, list(entry.input_shape), entry.dtype, hook._ring_payload.device.index,
                int(entry.input_shape[1]) if packed else 0,
                int(entry.transport_args[0]) if packed else 0,
                0 if gate is None else gate.data_ptr(), int(hook._gate_value),
            )
            if packed:
                packed_slots.append((slot, hook))
            if steps and steps[-1][0] is hook and output_index > steps[-1][2][-1][0]:
                steps[-1][2].append((output_index, slot))
            else:
                preprocess = info[0]
                if getattr(preprocess, "_dmi_graph_safe", False):
                    preprocess = _GraphedPreprocess(preprocess)
                steps.append((hook, (preprocess, info[1]), [(output_index, slot)]))
        return _StaticLaunch(
            steps=tuple(_StaticStep(hook, pre[0], pre[1], tuple(pairs)) for hook, pre, pairs in steps),
            launcher=launcher,
            fire=launcher.launch,
            set_counts=launcher.set_counts,
            packed_slots=tuple(packed_slots),
        )

    def _fast_forward(self, hook: HookPointV1, inputs: tuple) -> bool:
        index = self._static_step
        steps = self._static_armed.launch.steps
        if index >= len(steps) or steps[index].hook is not hook:
            # Not the next planned firing: the normal path decides (no-op
            # for non-emitting hooks, an error for a genuine plan mismatch).
            return False
        step = steps[index]
        if step.preprocess is None:
            values = inputs
            if len(values) != step.output_count:
                raise ValueError(
                    f"HookPointV1 expected {step.output_count} outputs, got {len(values)}"
                )
        else:
            values = _normalize_outputs(step.preprocess(*inputs), step.output_count)
        fire = self._static_armed.launch.fire
        for output_index, slot in step.slots:
            value = values[output_index]
            # The native slot checks device, dtype and shape before launching.
            fire(slot, value if isinstance(value, torch.Tensor) else _as_hook_output(value).tensor)
        self._static_cursor += len(step.slots)
        self._static_step = index + 1
        return True

    def _submit_static_batch(
        self, static: _StaticBatchPlan, ctx: MegatronTrainingContext,
    ) -> StepReservation:
        """One reservation for the whole microbatch; cached like a graph replay.

        The first microbatch of each packing key uses the public replay path
        (encoding, sizing, capacity check); later ones reuse its reservation
        items and descriptor rows, refreshing only the batch/microbatch ids.
        """
        adaptor = self.adaptor
        runtime = adaptor.record_runtime
        transport = getattr(runtime, "_transport", None)
        submit = getattr(runtime, "_combined_submit", None)
        if (
            submit is None or _MONITORING_DEBUG()
            or getattr(runtime, "_native_encode", None) is None
            or ctx.phase == "train"
        ):
            return adaptor.prepare_replay(static.plan, ctx, plan_direction=ctx.direction)
        if transport.null_offload:
            return StepReservation.SKIPPED
        key = (ctx.model_id, ctx.phase, ctx.direction, ctx.attempt_id, ctx.dp_rank,
               ctx.shard_rank, ctx.token_start, ctx.valid_counts, ctx.dataset_ids)
        cached = static.replay_cache.get(key, _MISSING)
        launcher = None if static.launch is None else static.launch.launcher
        if cached is _MISSING:
            cached = static.replay_cache[key] = adaptor._static_replay_template(
                static.plan, ctx, launcher)
            return cached[0] if cached[1] is None else cached[1]
        result, _, items, templates, total_rows, template_id = cached
        if result is StepReservation.OVERSIZED:
            return result
        batch, microbatch = ctx.global_batch_id, ctx.microbatch_id
        record_format = runtime._format
        if record_format.count_records and ctx.attempt_id >= 0 and total_rows:
            counts = record_format._expected_records
            count_key = (ctx.phase, batch, ctx.attempt_id)
            counts[count_key] = counts.get(count_key, 0) + total_rows
        if template_id is not None:
            # Native template: one call patches the ids, reserves and pushes.
            return StepReservation(launcher.submit(template_id, batch, microbatch))
        descriptors = tuple(
            (layout, (*head, batch, dp_rank, microbatch, *tail), rows, output_id)
            for layout, head, dp_rank, tail, rows, output_id in templates
        )
        return StepReservation(submit(items, descriptors, submit_oversized=False))

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

    def execution_phase(self) -> str:
        ctx = self.adaptor.current_context
        if self.mode is HookRuntimeMode.CAPTURE_RECORD and not self.capture_event_context:
            return "train"
        return ctx.phase if ctx else "train"

    def should_emit(self, hook: HookPointV1) -> bool:
        if not hook_enabled_in_phase(hook, self.execution_phase()):
            return False
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
        output_phases = getattr(hook, "megatron_output_phases", None)
        if output_phases is not None:
            name = self.adaptor._configured_hook(hook).policy.outputs[output_index].name
            if self.execution_phase() not in output_phases[name]:
                # No reservation/metadata/producer: same no-dispatch return as
                # the existing CPU-direct path, before recording graph entries.
                return StepReservation.OVERSIZED
        armed = self._static_armed
        if armed is not None:
            return self._consume_static_output(armed, hook, int(output_id), output)
        semantics = self.adaptor._configured_hook(hook).output_semantics[output_index]
        builder = self._capture_builder
        if builder is None:
            reservation = self.adaptor.emit_immediate_output(
                semantics=semantics,
                output_spec=output_spec,
                output=output,
            )
            if self._static_recording is not None:
                self._record_static_output(hook, output_index, semantics, output_spec, output)
            return reservation
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
        ctx = self.adaptor.current_iteration_context
        return hook_enabled_in_phase(hook, ctx.phase if ctx else "train")

    def prepare_output(
        self,
        *,
        hook: HookPointV1,
        output_index: int,
        output_id: int,
        output_spec: TransportSpec,
        output: HookOutput,
    ) -> StepReservation:
        semantics = self.adaptor._configured_hook(hook).output_semantics[output_index]
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
        self._plan_states_by_id: dict[int, _CapturedPlanState] = {}
        self.current_context: MegatronTrainingContext | None = None
        self._event_packing: _PreparedPacking | None = None
        self.current_iteration_context: MegatronTrainingContext | None = None
        self.hook_runtime = MegatronHookRuntime(self)
        self.iteration_hook_runtime = MegatronIterationHookRuntime(self)
        self.invocation_allocator: InvocationIdAllocator | None = None
        self._metadata_context: DMIMetadataContext | None = None
        # Per hook: (raw preprocess, output count, launchable without the
        # sequence-shard narrow), recorded when its wrapper is built.
        self._fast_hook_info: dict[int, tuple] = {}
        # Per packed hook: [valid attr, prefix attr, source field, source
        # prefix field, last validated valid tensor, last validated prefix].
        self._eager_packed_info: dict[int, list] = {}
        self._direct_packed_consumers: tuple[HookPointV1, ...] = ()
        self._direct_packed_phase_ok: dict[str, bool] = {}
        self.eager_direct_packed_enabled = False
        self._direct_packed_microbatch: int | None = None
        # Graph-replay reservation templates (see _prepare_cached_replay), the
        # native launcher that submits them, and the invocation allocations
        # recorded while one template is built.
        self._replay_templates: dict[tuple, tuple] = {}
        self._emit_templates: dict[tuple, tuple] = {}
        self._eager_entries: dict[tuple, tuple] = {}
        self._replay_launcher: Any = _MISSING
        self._invocation_log: list | None = None

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
        self._plan_states_by_id.clear()
        self._replay_templates.clear()
        self._emit_templates.clear()
        self._eager_entries.clear()
        self.hook_runtime.reset_static_batches()
        self._metadata_context = metadata_context
        self._fast_hook_info = {}
        self._eager_packed_info = {}
        self._direct_packed_phase_ok = {}
        self._direct_packed_microbatch = None
        direct_consumers = []
        for binding in bindings:
            hook = binding.hook
            policy = self._hook_policy(hook)
            tp_sequence_start, tp_sequence_length = self._tp_sequence_interval(
                hook, policy
            )
            is_iteration = id(hook) in iteration_ids
            self._validate_binding(
                binding,
                policy,
                is_iteration=is_iteration,
                active_input_layout=active_input_layout,
            )
            if not is_iteration and any(
                output.transport_type is TransportType.SEQ_PREFIX_PACK for output in policy.outputs
            ):
                direction = self._hook_phase(hook).name.lower()
                field_name = ("tp_seq_sharded_valid_count"
                              if policy.shard_policy is ShardPolicy.TP_SEQUENCE_SHARDED
                              else "valid_count")
                self._eager_packed_info[id(hook)] = [
                    f"valid_count_{direction}", f"valid_count_prefix_{direction}",
                    field_name, field_name + "_prefix", None, None,
                ]
            if MegatronMetadataField.VALID_COUNT in policy.preprocess_metadata_fields:
                # Reads the current valid-count buffers itself, not via producers.
                direct_consumers.append(hook)
            if metadata_context is not None and not is_iteration:
                phase = self._hook_phase(hook)
                for metadata_field in policy.binding_metadata_fields:
                    if metadata_field is MegatronMetadataField.VALID_COUNT:
                        self._bind_valid_count_context(
                            hook,
                            metadata_context,
                            binding.scope_id,
                            phase,
                            policy,
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
            hook_dims = self.dims
            if tp_sequence_length is not None:
                hook_dims = dict(self.dims)
                hook_dims[DimSpec.SEQ] = tp_sequence_length
                hook_dims[DimSpec.SEQ.value] = tp_sequence_length
            physical = policy.resolve(hook_dims)
            physical = HookSpecV1(
                name=physical.name,
                outputs=physical.outputs,
                preprocess=self._record_preprocess(
                    hook,
                    physical,
                    policy=policy,
                    tp_sequence_start=tp_sequence_start,
                    tp_sequence_length=tp_sequence_length,
                ),
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
                tp_sequence_start=tp_sequence_start,
                tp_sequence_length=tp_sequence_length,
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
            for output in policy.outputs:
                _dtype_element_size(output.dtype)
            configured = replace(configured, output_semantics=tuple(
                self._producer_semantics(
                    hook=hook, output_id=output_id, output_spec=output_spec,
                )
                for output_id, output_spec in zip(hook._output_ids, physical.outputs)
            ))
            self.configured_hooks[-1] = configured
            self._configured_by_hook[id(hook)] = configured
        self._direct_packed_consumers = tuple(direct_consumers)
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

    def _tp_sequence_interval(
        self,
        hook: HookPointV1,
        policy: MegatronHookSpec,
    ) -> tuple[int | None, int | None]:
        if policy.shard_policy is not ShardPolicy.TP_SEQUENCE_SHARDED:
            return None, None
        if policy.record_type is not RecordType.PER_SAMPLE or len(policy.outputs) != 1:
            raise ValueError(
                "TP_SEQUENCE_SHARDED requires one PER_SAMPLE output"
            )
        output = policy.outputs[0]
        # A reduced batch-first output still describes this source interval.
        if not (
            (
                output.transport_type is TransportType.SEQ_PREFIX_PACK
                and tuple(output.input_shape[:2]) == (DimSpec.SEQ, DimSpec.BATCH)
            )
            or (
                output.transport_type is TransportType.IDENTITY
                and tuple(output.input_shape[:1]) == (DimSpec.BATCH,)
            )
        ):
            raise ValueError(
                "TP_SEQUENCE_SHARDED requires [SEQ, BATCH, ...] "
                "SEQ_PREFIX_PACK input or [BATCH, ...] IDENTITY input"
            )
        distributed_info = getattr(hook, "megatron_distributed_info", None)
        if not isinstance(distributed_info, MegatronDistributedInfo):
            raise TypeError(
                "TP_SEQUENCE_SHARDED hook requires MegatronDistributedInfo"
            )
        global_sequence_length = self.dims.get(
            DimSpec.SEQ, self.dims.get(DimSpec.SEQ.value)
        )
        if global_sequence_length is None:
            raise KeyError("TP_SEQUENCE_SHARDED requires DimSpec.SEQ")
        tp_world_size = int(distributed_info.tp_world_size)
        tp_rank = int(distributed_info.tp_rank)
        if tp_world_size <= 0 or not 0 <= tp_rank < tp_world_size:
            raise ValueError("invalid TP rank or world size in MegatronDistributedInfo")
        if int(global_sequence_length) % tp_world_size != 0:
            raise ValueError(
                "DMI TP sequence sharding requires sequence length divisible by "
                f"tensor-model-parallel size: {global_sequence_length} % "
                f"{tp_world_size} != 0"
            )
        local_length = int(global_sequence_length) // tp_world_size
        return tp_rank * local_length, local_length

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

    def configure_eager_direct_packed_metadata(self, enabled: bool) -> None:
        self.eager_direct_packed_enabled = bool(enabled)
        self._direct_packed_phase_ok = {}

    def eager_direct_packed_allowed(self, phase: str) -> bool:
        """Whether this eager scope may skip the current packed-count copies.

        Only non-train phases outside capture, and only when no enabled hook
        reads the current valid-count buffers directly.
        """
        if not self.eager_direct_packed_enabled or phase == "train" or self._metadata_context is None:
            return False
        runtime = self.hook_runtime
        if (runtime.mode is not HookRuntimeMode.EAGER_IMMEDIATE
                or runtime._capture_builder is not None or runtime.te_capture_session_active):
            return False
        allowed = self._direct_packed_phase_ok.get(phase)
        if allowed is None:
            allowed = self._direct_packed_phase_ok[phase] = not any(
                hook.enabled and hook_enabled_in_phase(hook, phase)
                for hook in self._direct_packed_consumers
            )
        return allowed

    def set_direct_packed_microbatch(self, microbatch_id: int | None) -> None:
        self._direct_packed_microbatch = microbatch_id

    def configure_eager_static_batch(self, enabled: bool) -> None:
        self.hook_runtime.reset_static_batches()
        self.hook_runtime.eager_static_batch_enabled = bool(enabled)

    def set_current_event(self, ctx: MegatronTrainingContext) -> None:
        self.current_context = ctx
        self._event_packing = ctx._packing

    def clear_current_event(self) -> None:
        self.current_context = None
        self._event_packing = None

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
        # Only prepare_output's non-capture branch calls this method. Capture
        # builders/replay retain from_output(), emit_prepared_output(), and
        # the original metadata path.
        ctx = self.current_context
        tensor = output.tensor
        if ctx is not None and isinstance(tensor, torch.Tensor):
            # The entry and reservation bytes depend only on these inputs; a
            # repeat skips rebuilding them (first use runs every check below).
            nbytes = None if semantics.eager_nbytes is None else int(semantics.eager_nbytes())
            key = (id(semantics), id(output_spec), tensor.shape, tensor.dtype, nbytes,
                   ctx.valid_counts)
            cached = self._eager_entries.get(key)
            if cached is None or cached[0] is not semantics or cached[1] is not output_spec:
                entry, reservation_bytes = self._build_eager_entry(
                    semantics, output_spec, output, nbytes, ctx)
                if len(self._eager_entries) >= 4096:
                    self._eager_entries.clear()
                self._eager_entries[key] = cached = (semantics, output_spec, entry, reservation_bytes)
            _, _, entry, reservation_bytes = cached
            reservation = self._emit_cached(entry, semantics, ctx, output, reservation_bytes,
                                            key=key)
            if reservation is not None:
                return reservation
            return self.record_runtime._emit_prepared_output(
                entry, self._record_metadata(ctx, semantics, eager=True), output,
                reservation_bytes=reservation_bytes,
            )
        entry, input_bytes = ProducerPlanEntry._from_eager_output(
            output_id=semantics.output_id,
            output_spec=output_spec,
            output=output,
            transport_args=(semantics.eager_transport_args
                            if semantics.eager_transport_args is not None
                            else ProducerPlanBuilder._transport_args(output_spec)),
        )
        if semantics.eager_nbytes is not None:
            nbytes = int(semantics.eager_nbytes())
            if entry.transport_type is not TransportType.SEGMENTED_PACK:
                raise ValueError("Explicit eager byte sizing requires segmented transport")
            row_bytes = entry.transport_args[0]
            if nbytes < 0 or nbytes % row_bytes or nbytes > input_bytes:
                raise ValueError("Invalid sampled output byte count")
            entry = replace(entry, output_shape=(nbytes // row_bytes, *entry.input_shape[1:]),
                            reservation_upper_bytes=nbytes)
        ctx = self.current_context
        if ctx is None:
            raise RuntimeError("Megatron DMI hook emitted without a schedule event")
        reservation_bytes = (_align_up(input_bytes)
                             if entry.transport_type is TransportType.IDENTITY
                             else self._resolved_output_bytes(entry, semantics, self._event_packing))
        return self.record_runtime._emit_prepared_output(
            entry, self._record_metadata(ctx, semantics, eager=True), output,
            reservation_bytes=reservation_bytes,
        )

    def _build_eager_entry(
        self,
        semantics: _ProducerSemantics,
        output_spec: TransportSpec,
        output: HookOutput,
        nbytes: int | None,
        ctx: MegatronTrainingContext,
    ) -> tuple[ProducerPlanEntry, int]:
        """The eager entry and reservation bytes, with the general path's checks."""
        entry, input_bytes = ProducerPlanEntry._from_eager_output(
            output_id=semantics.output_id,
            output_spec=output_spec,
            output=output,
            transport_args=(semantics.eager_transport_args
                            if semantics.eager_transport_args is not None
                            else ProducerPlanBuilder._transport_args(output_spec)),
        )
        if nbytes is not None:
            if entry.transport_type is not TransportType.SEGMENTED_PACK:
                raise ValueError("Explicit eager byte sizing requires segmented transport")
            row_bytes = entry.transport_args[0]
            if nbytes < 0 or nbytes % row_bytes or nbytes > input_bytes:
                raise ValueError("Invalid sampled output byte count")
            entry = replace(entry, output_shape=(nbytes // row_bytes, *entry.input_shape[1:]),
                            reservation_upper_bytes=nbytes)
        reservation_bytes = (_align_up(input_bytes)
                             if entry.transport_type is TransportType.IDENTITY
                             else self._resolved_output_bytes(entry, semantics, self._event_packing))
        return entry, reservation_bytes

    def _emit_cached(
        self,
        entry: ProducerPlanEntry,
        semantics: _ProducerSemantics,
        ctx: MegatronTrainingContext,
        output: HookOutput,
        reservation_bytes: int,
        *,
        key: tuple,
    ) -> StepReservation | None:
        """Eager emits that differ only in batch/microbatch ids reuse one template.

        The key holds the full eager entry (shapes, dtype, sampled rows, bound)
        and its reservation bytes, so any size change misses. A hit submits the
        template first and allocates invocation ids and record counts only when
        it reserved; an oversized hit has changed nothing and runs the general
        path (descriptor push plus CPU-direct payload). A miss runs the general
        path's steps and keeps a template only when every row has the expected
        id cells and invocation id 0. Returns None for the general path.
        """
        runtime = self.record_runtime
        transport = getattr(runtime, "_transport", None)
        if (
            _MONITORING_DEBUG() or transport is None or transport.null_offload
            or getattr(runtime, "_combined_submit", None) is None
            or getattr(runtime, "_native_encode", None) is None
        ):
            return None
        launcher = self._replay_template_launcher()
        if launcher is None:
            return None
        # ``key`` (hook, output spec, tensor shape/dtype, sampled bytes, valid
        # counts) determines the entry and reservation bytes.
        key = (key, ctx.model_id, ctx.phase, ctx.direction, ctx.attempt_id, ctx.dp_rank,
               ctx.shard_rank, ctx.token_start, ctx.dataset_ids)
        batch, microbatch = ctx.global_batch_id, ctx.microbatch_id
        cached = self._emit_templates.get(key)
        if cached is not None and cached[0] is semantics:
            _, template_id, counted_rows, allocations = cached
            calls = [dict(allocation, global_batch_id=int(batch), microbatch_id=int(microbatch))
                     for allocation in allocations]
            allocator = self.invocation_allocator
            if calls and not all(allocator.is_first(**call) for call in calls):
                return None
            reservation = StepReservation(launcher.submit(template_id, batch, microbatch))
            if reservation is StepReservation.OVERSIZED:
                return None
            for call in calls:
                allocator.allocate(**call)
            record_format = runtime._format
            if record_format.count_records and ctx.attempt_id >= 0 and counted_rows:
                counts = record_format._expected_records
                count_key = (ctx.phase, batch, ctx.attempt_id)
                counts[count_key] = counts.get(count_key, 0) + counted_rows
            return reservation
        self._invocation_log = []
        try:
            metadata = self._record_metadata(ctx, semantics, eager=True)
        finally:
            log, self._invocation_log = self._invocation_log, None
        descriptor = runtime._encode_submission(metadata, entry)
        items = runtime._reservation_items((entry,), (reservation_bytes,))
        _, common, rows, _ = descriptor
        if (all(invocation_id == 0 for _, invocation_id in log) and rows
                and common[4] == batch and common[6] == microbatch):
            template_id = launcher.add_template(
                items, (descriptor,), transport._compiled_record_schema, 4, 6)
            counted_rows = (len(rows) if metadata.act_name != "iteration_attempt_status" else 0)
            if len(self._emit_templates) >= 4096:
                self._emit_templates.clear()
            self._emit_templates[key] = (semantics, template_id, counted_rows,
                                         tuple(allocation for allocation, _ in log))
        reservation = StepReservation(runtime._combined_submit(
            items, (descriptor,), submit_oversized=True))
        if reservation is StepReservation.OVERSIZED:
            transport.submit_record_cpu_direct(output, entry)
        return reservation

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
        return self.record_runtime._emit_prepared_output(
            entry, metadata, output,
            reservation_bytes=self._resolved_output_bytes(entry, semantics, self._event_packing),
        )

    def emit_iteration_output(
        self,
        semantics: _ProducerSemantics,
        output_spec: TransportSpec,
        output: HookOutput,
    ) -> StepReservation:
        self._require_no_static_batch()
        ctx = self.current_iteration_context
        if ctx is None:
            raise RuntimeError("Megatron DMI iteration hook requires an explicit context")
        if output_spec.transport_type is not TransportType.IDENTITY:
            raise ValueError("PER_ITERATION supports IDENTITY transport only")
        if output_spec.storage in (OutputStorage.SCALAR_FLOAT, OutputStorage.SCALAR_INT):
            if output.tensor.numel() != 1:
                raise ValueError("PER_ITERATION scalar output must contain one value")
        entry = ProducerPlanEntry.from_output(
            output_id=semantics.output_id,
            output_spec=output_spec,
            output=output,
        )
        return self.record_runtime._emit_prepared_output(
            entry, self._record_metadata(ctx, semantics), output,
            reservation_bytes=_align_up(output.tensor.numel() * output.tensor.element_size()),
        )

    @staticmethod
    def _resolved_output_bytes(
        entry: ProducerPlanEntry, semantic: _ProducerSemantics,
        packing: _PreparedPacking | None,
    ) -> int:
        if entry.transport_type is TransportType.IDENTITY:
            return _align_up(prod(entry.input_shape) * entry.element_size)
        if semantic.eager_nbytes is not None:
            return _align_up(entry.reservation_upper_bytes)
        if packing is None or not packing.global_counts.counts:
            raise ValueError("packed outputs require current CPU packing metadata")
        counts = packing.counts_for(semantic.tp_sequence_start, semantic.tp_sequence_length)
        if entry.transport_type is TransportType.SEQ_PREFIX_PACK:
            if len(entry.input_shape) < 2 or len(counts.counts) != entry.input_shape[1]:
                raise ValueError("CPU valid-count length does not match packed input batch")
            if any(value > entry.input_shape[0] for value in counts.counts):
                raise ValueError("CPU valid count exceeds packed input sequence length")
            if prod(entry.input_shape[2:]) * entry.element_size != entry.transport_args[0]:
                raise ValueError("packed feature bytes do not match current tensor metadata")
            if semantic.packed_size_fn is None:
                raise RuntimeError("packed output has no bound sizing function")
            return semantic.packed_size_fn(packing.global_counts.token_count, counts.token_count)
        if entry.transport_type is TransportType.SEGMENTED_PACK:
            return _align_up(counts.token_count * entry.transport_args[0])
        if entry.transport_type is TransportType.PREFIX_STRIP:
            return _align_up(sum(value > 0 for value in counts.counts) * entry.transport_args[0])
        raise NotImplementedError(f"no CPU sizing contract for {entry.transport_type.value}")

    def prepare_replay_capacity_only(
        self,
        plan: ProducerPlan,
        ctx: MegatronTrainingContext | None = None,
        *,
        plan_direction: HookPhase | str,
        live_direction: HookPhase | str,
        expected_plan: ProducerPlan | None = None,
    ) -> StepReservation:
        selected = self._select_replay_plan(
            plan,
            plan_direction=plan_direction,
            live_direction=live_direction,
            expected_plan=expected_plan,
        )
        if not selected.entries:
            return StepReservation.SKIPPED
        # A backward preflight at its paired forward boundary uses that same
        # microbatch's counts, not the last backward event from another batch.
        packing = self._event_packing if ctx is None else self._prepare_packing(ctx)
        total, _ = self._plan_state(selected).sizing.resolve(
            (packing,) * len(selected.entries),
        )
        if self._plan_is_oversized(selected, total):
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
        self._require_no_static_batch()
        replay_ctx = self.current_context if ctx is None else ctx
        if replay_ctx is None:
            raise RuntimeError("Megatron DMI replay requires an active schedule event")
        live = replay_ctx.direction if live_direction is None else live_direction
        if ctx is None and expected_plan is None:
            reservation = self._prepare_cached_replay(plan, replay_ctx, plan_direction, live)
            if reservation is not None:
                return reservation
        selected = self._select_replay_plan(
            plan,
            plan_direction=plan_direction,
            live_direction=live,
            expected_plan=expected_plan,
        )
        if not selected.entries:
            return StepReservation.SKIPPED
        state = self._plan_state(selected)
        packing = self._event_packing if ctx is None else self._prepare_packing(replay_ctx)
        total, sizes = state.sizing.resolve((packing,) * len(selected.entries))
        if self._plan_is_oversized(selected, total):
            return StepReservation.OVERSIZED
        metadata = tuple(self._record_metadata(replay_ctx, item) for item in state.semantics)
        return self.record_runtime.prepare_replay(selected, metadata, reservation_bytes=sizes)

    def _prepare_cached_replay(
        self,
        plan: ProducerPlan,
        ctx: MegatronTrainingContext,
        plan_direction: HookPhase | str,
        live_direction: HookPhase | str,
    ) -> StepReservation | None:
        """Reuse one graph replay's reservation for replays that differ only in ids.

        As in _submit_static_batch, the first replay of each key runs the
        general path (selection, sizing, capacity check, encoding) and keeps its
        reservation items and descriptors as a native template; later replays
        patch only the batch and microbatch ids. A template is kept and reused
        only while every record gets invocation id 0, so recomputation replays
        (ids >= 1) keep the general path. Returns None for the general path.
        """
        runtime = self.record_runtime
        transport = getattr(runtime, "_transport", None)
        if (
            _MONITORING_DEBUG() or transport is None or transport.null_offload
            or getattr(runtime, "_combined_submit", None) is None
            or getattr(runtime, "_native_encode", None) is None
        ):
            return None
        launcher = self._replay_template_launcher()
        if launcher is None:
            return None
        key = (id(plan), self.normalize_direction(plan_direction),
               self.normalize_direction(live_direction), ctx.model_id, ctx.phase,
               ctx.attempt_id, ctx.dp_rank, ctx.shard_rank, ctx.token_start,
               ctx.valid_counts, ctx.dataset_ids)
        cached = self._replay_templates.get(key)
        if cached is not None and cached[0] is plan:
            _, result, template_id, counted_rows, allocations = cached
            if result is StepReservation.OVERSIZED:
                return result
            batch, microbatch = ctx.global_batch_id, ctx.microbatch_id
            if allocations:
                allocator = self.invocation_allocator
                calls = [dict(allocation, global_batch_id=int(batch), microbatch_id=int(microbatch))
                         for allocation in allocations]
                if not all(allocator.is_first(**call) for call in calls):
                    return None
                for call in calls:
                    allocator.allocate(**call)
            record_format = runtime._format
            if record_format.count_records and ctx.attempt_id >= 0 and counted_rows:
                counts = record_format._expected_records
                count_key = (ctx.phase, batch, ctx.attempt_id)
                counts[count_key] = counts.get(count_key, 0) + counted_rows
            return StepReservation(launcher.submit(template_id, batch, microbatch))
        # Miss: the general path's steps in its order (selection, sizing,
        # capacity check, metadata, encoding, reservation items, submit). The
        # template is registered before the submit, and only when every
        # descriptor has rows with the expected id cells and every invocation
        # id is 0; otherwise this replay is simply not cached.
        selected = self._select_replay_plan(
            plan, plan_direction=plan_direction, live_direction=live_direction,
            expected_plan=None,
        )
        if not selected.entries:
            return None
        state = self._plan_state(selected)
        total, sizes = state.sizing.resolve((ctx._packing,) * len(selected.entries))
        if self._plan_is_oversized(selected, total):
            self._remember_replay_template(key, (plan, StepReservation.OVERSIZED, None, 0, ()))
            return StepReservation.OVERSIZED
        self._invocation_log = []
        try:
            metadata = tuple(self._record_metadata(ctx, item) for item in state.semantics)
        finally:
            log, self._invocation_log = self._invocation_log, None
        descriptors = tuple(runtime._encode_submission(item, entry)
                            for item, entry in zip(metadata, selected.entries))
        items = runtime._reservation_items(selected.entries, sizes)
        batch, microbatch = ctx.global_batch_id, ctx.microbatch_id
        if all(invocation_id == 0 for _, invocation_id in log) and all(
            rows and common[4] == batch and common[6] == microbatch
            for _, common, rows, _ in descriptors
        ):
            template_id = launcher.add_template(
                items, descriptors, transport._compiled_record_schema, 4, 6)
            counted_rows = sum(
                len(rows) for (_, _, rows, _), item in zip(descriptors, metadata)
                if item.act_name != "iteration_attempt_status")
            self._remember_replay_template(
                key, (plan, StepReservation.RESERVED, template_id, counted_rows,
                      tuple(allocation for allocation, _ in log)))
        return StepReservation(runtime._combined_submit(items, descriptors, submit_oversized=False))

    def _remember_replay_template(self, key: tuple, value: tuple) -> None:
        if len(self._replay_templates) >= 4096:
            self._replay_templates.clear()
        self._replay_templates[key] = value

    def _replay_template_launcher(self) -> Any:
        if self._replay_launcher is _MISSING:
            engine = getattr(getattr(self.record_runtime, "_transport", None), "_ring_engine", None)
            launcher_type = getattr(_native_extension(), "StaticLauncher", None)
            self._replay_launcher = (
                None if engine is None or launcher_type is None else launcher_type(engine))
        return self._replay_launcher

    def prepare_full_iteration_replay(
        self,
        plan: MegatronFullIterationPlan,
        contexts: Sequence[MegatronTrainingContext],
        *,
        expected_plan: MegatronFullIterationPlan | None = None,
    ) -> StepReservation:
        self._require_no_static_batch()
        if len(contexts) != plan.output_count:
            raise ValueError("Megatron full-iteration replay context count mismatch")
        if expected_plan is not None:
            expected_plan.assert_compatible(plan)
        self._validate_replay_plan(plan.producer_plan)
        self._validate_full_iteration_contexts(plan, contexts)
        if not plan.entries:
            return StepReservation.SKIPPED
        prepared = {}
        packings = []
        for ctx in contexts:
            key = (ctx.model_id, ctx.phase, ctx.global_batch_id, ctx.attempt_id, ctx.microbatch_id)
            packing = prepared.get(key)
            if packing is None:
                packing = prepared[key] = self._prepare_packing(ctx)
            elif packing.global_counts.counts != tuple(ctx.valid_counts):
                raise ValueError("full-iteration contexts disagree on one microbatch's valid counts")
            packings.append(packing)
        total, sizes = plan._sizing.resolve(packings)
        if self._plan_is_oversized(plan.producer_plan, total):
            return StepReservation.OVERSIZED
        metadata = tuple(
            self._record_metadata(ctx, semantic)
            for ctx, semantic in zip(contexts, plan._semantics)
        )
        return self.record_runtime.prepare_replay(
            plan.producer_plan, metadata, reservation_bytes=sizes,
        )

    def _static_replay_template(self, plan: ProducerPlan, ctx: MegatronTrainingContext,
                                launcher: Any = None):
        """Reserve one static microbatch through the public path and keep its parts.

        Returns (result, early_result, items, templates, total_rows, template_id).
        Templates hold each descriptor with the batch and microbatch ids
        factored out; ``template_id`` names the same template registered in
        the native launcher, when one is given.
        """
        runtime = self.record_runtime
        state = self._plan_state(plan)
        total, sizes = state.sizing.resolve((ctx._packing,) * len(plan.entries))
        if self._plan_is_oversized(plan, total):
            return (StepReservation.OVERSIZED, StepReservation.OVERSIZED, None, None, 0, None)
        metadata = tuple(self._record_metadata(ctx, item) for item in state.semantics)
        descriptors = tuple(runtime._encode_submission(item, entry)
                            for item, entry in zip(metadata, plan.entries))
        items = runtime._reservation_items(plan.entries, sizes)
        result = StepReservation(runtime._combined_submit(
            items, descriptors, submit_oversized=False))
        templates, total_rows = [], 0
        for item, descriptor in zip(metadata, descriptors):
            layout, common, rows, output_id = descriptor
            if common[4] != ctx.global_batch_id or common[6] != ctx.microbatch_id:
                raise RuntimeError("unexpected Megatron record coordinate layout")
            templates.append((layout, tuple(common[:4]), common[5], tuple(common[7:]),
                              rows, output_id))
            total_rows += len(rows)
        template_id = None if launcher is None else launcher.add_template(
            items, descriptors, runtime._transport._compiled_record_schema, 4, 6)
        return (result, None, items, tuple(templates), total_rows, template_id)

    def _require_no_static_batch(self) -> None:
        # Another reservation would interleave with the queued static batch.
        if self.hook_runtime._static_armed is not None:
            raise RuntimeError("DMI reservation requested inside an eager static batch")

    def _plan_is_oversized(self, plan: ProducerPlan, total_bytes: int) -> bool:
        capacities = self.engine.ring_capacities()
        return bool(
            total_bytes > capacities.effective_bytes
            or plan.task_count > capacities.task_entries
        )

    @staticmethod
    def _prepare_packing(ctx: MegatronTrainingContext) -> _PreparedPacking:
        return ctx._packing

    def _select_replay_plan(
        self,
        plan: ProducerPlan,
        *,
        plan_direction: HookPhase | str,
        live_direction: HookPhase | str,
        expected_plan: ProducerPlan | None,
    ) -> ProducerPlan:
        self._validate_replay_plan(plan, expected_plan=expected_plan)
        semantics = self._plan_semantics(plan)
        planned = self.normalize_direction(plan_direction)
        live = self.normalize_direction(live_direction)
        if planned is live:
            return plan
        if planned is HookPhase.FWD and live is HookPhase.BWD:
            state = self._plan_state(plan)
            key = (planned, live)
            if key in state.selected:
                return state.selected[key]
            selected = tuple(
                (entry, semantic)
                for entry, semantic in zip(plan.entries, semantics)
                if not semantic.suppress_recompute
            )
            subset = ProducerPlan(tuple(item[0] for item in selected))
            self._remember_plan(subset, tuple(item[1] for item in selected))
            state.selected[key] = subset
            return subset
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
        *,
        eager: bool = False,
    ) -> MegatronRecordMetadata:
        if semantic.requires_valid_count is not None:
            needs_counts = semantic.requires_valid_count
            needs_datasets = semantic.record_type is RecordType.PER_SAMPLE and bool(ctx.dataset_ids)
        else:
            required_fields = required_record_metadata_fields(
                record_type=semantic.record_type,
                need_token_range=semantic.need_token_range,
                transport_type=semantic.transport_type,
                dynamic_dataset_provenance=bool(ctx.dataset_ids),
            )
            needs_counts = MegatronMetadataField.VALID_COUNT in required_fields
            needs_datasets = MegatronMetadataField.DATASET_ID in required_fields
        if semantic.record_type is RecordType.PER_SAMPLE:
            dp_rank = ctx.dp_rank
            valid_counts = (
                ctx._packing.counts_for(
                    semantic.tp_sequence_start, semantic.tp_sequence_length,
                ).counts
                if needs_counts
                else ()
            )
            dataset_ids = (
                ctx.dataset_ids
                if needs_datasets
                else ()
            )
            token_start = ctx.token_start if semantic.need_token_range else 0
            if semantic.tp_sequence_start is not None:
                if semantic.need_token_range:
                    token_start += semantic.tp_sequence_start
        else:
            if semantic.record_dp_rank is None:
                raise RuntimeError("non-sample record is missing its semantic DP rank")
            dp_rank = int(semantic.record_dp_rank)
            valid_counts = ()
            dataset_ids = ()
            token_start = -1 if semantic.record_type is RecordType.PER_EXECUTION else 0
        shard_rank = (
            ctx.shard_rank
            if semantic.record_shard_rank is None
            else int(semantic.record_shard_rank)
        )
        model_id = self.model_id if ctx.model_id is None else ctx.model_id
        invocation_id = self._allocate_invocation_id(
            ctx,
            semantic=semantic,
            dp_rank=dp_rank,
            shard_rank=shard_rank,
            token_start=token_start,
            model_id=model_id,
        )
        return MegatronRecordMetadata._from_normalized(
            model_id=model_id,
            act_name=semantic.act_name,
            direction=semantic.record_direction,
            phase=ctx.phase,
            global_batch_id=ctx.global_batch_id,
            dp_rank=dp_rank,
            microbatch_id=ctx.microbatch_id,
            layer_no=int(semantic.layer_no),
            shard_rank=shard_rank,
            token_start=token_start,
            valid_counts=valid_counts,
            dataset_ids=dataset_ids,
            attempt_id=ctx.attempt_id,
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
        allocation = dict(
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
        invocation_id = allocator.allocate(**allocation)
        if self._invocation_log is not None:
            self._invocation_log.append((allocation, invocation_id))
        return invocation_id

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
            tp_sequence_start=configured.tp_sequence_start,
            tp_sequence_length=configured.tp_sequence_length,
            packed_size_fn=(
                _make_packed_size_fn(
                    int(output_spec.feature_bytes), local=configured.tp_sequence_start is not None,
                ) if output_spec.transport_type is TransportType.SEQ_PREFIX_PACK else None
            ),
            eager_nbytes=next(output.eager_nbytes for output in configured.policy.outputs
                              if output.name == output_spec.name),
            eager_transport_args=ProducerPlanBuilder._transport_args(output_spec),
            requires_valid_count=(MegatronMetadataField.VALID_COUNT in
                required_record_metadata_fields(
                    record_type=configured.policy.record_type,
                    need_token_range=bool(configured.policy.need_token_range),
                    transport_type=output_spec.transport_type,
                    dynamic_dataset_provenance=False,
                )),
        )

    def _remember_plan(
        self,
        plan: ProducerPlan,
        semantics: Sequence[_ProducerSemantics],
    ) -> None:
        if len(plan.entries) != len(semantics):
            raise ValueError("producer plan semantic count mismatch")
        key = id(plan)
        adaptor_ref = weakref.ref(self)

        def released(plan_ref):
            adaptor = adaptor_ref()
            if adaptor is not None:
                state = adaptor._plan_states_by_id.get(key)
                if state is not None and state.owner is plan_ref:
                    del adaptor._plan_states_by_id[key]

        self._plan_states_by_id[key] = _CapturedPlanState(
            weakref.ref(plan, released), tuple(semantics), _CapturedSizing(plan.entries, semantics),
        )

    def _plan_state(self, plan: ProducerPlan) -> _CapturedPlanState:
        state = self._plan_states_by_id.get(id(plan))
        if state is None or state.owner() is not plan:
            raise RuntimeError("Megatron producer plan is not owned by this adaptor")
        return state

    def _plan_semantics(self, plan: ProducerPlan) -> tuple[_ProducerSemantics, ...]:
        return self._plan_state(plan).semantics

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
        policy: MegatronHookSpec,
    ) -> None:
        if phase not in (HookPhase.FWD, HookPhase.BWD):
            raise ValueError("valid-count metadata requires an FWD or BWD hook phase")
        direction = phase.name.lower()
        setattr(
            hook,
            f"valid_count_{direction}",
            metadata_context.current(
                (
                    "tp_seq_sharded_valid_count"
                    if policy.shard_policy is ShardPolicy.TP_SEQUENCE_SHARDED
                    else "valid_count"
                ),
                direction,
                scope_id,
            ),
        )
        if any(output.transport_type is TransportType.SEQ_PREFIX_PACK for output in policy.outputs):
            prefix_name = metadata_context.register_valid_count_prefix(
                tp_local=policy.shard_policy is ShardPolicy.TP_SEQUENCE_SHARDED,
            )
            setattr(hook, f"valid_count_prefix_{direction}",
                    metadata_context.current(prefix_name, direction, scope_id))

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
        *,
        policy: MegatronHookSpec,
        tp_sequence_start: int | None,
        tp_sequence_length: int | None,
    ) -> Any:
        original = physical.preprocess
        sequence_sharded = policy.shard_policy is ShardPolicy.TP_SEQUENCE_SHARDED
        if not any(
            item.transport_type in (
                TransportType.SEQ_PREFIX_PACK,
                TransportType.SEGMENTED_PACK,
            )
            for item in physical.outputs
        ) and not sequence_sharded:
            self._fast_hook_info[id(hook)] = (original, len(physical.outputs), True)
            return original

        sequence_parallel_enabled = False
        global_sequence_length = None
        if sequence_sharded:
            distributed_info = getattr(hook, "megatron_distributed_info", None)
            if not isinstance(distributed_info, MegatronDistributedInfo):
                raise TypeError(
                    "TP_SEQUENCE_SHARDED hook requires MegatronDistributedInfo"
                )
            sequence_parallel_enabled = bool(
                distributed_info.sequence_parallel_enabled
            )
            global_sequence_length = self.dims.get(
                DimSpec.SEQ, self.dims.get(DimSpec.SEQ.value)
            )
            assert tp_sequence_start is not None
            assert tp_sequence_length is not None
        # TP=1: the local interval is the whole sequence, so narrow is a no-op.
        full_sequence = (
            sequence_sharded and int(tp_sequence_start) == 0
            and global_sequence_length is not None
            and int(tp_sequence_length) == int(global_sequence_length)
        )
        # The static fast launch checks the planned input shape, which covers
        # the sequence-length check; only a real narrow needs the wrapper.
        self._fast_hook_info[id(hook)] = (
            original, len(physical.outputs),
            not sequence_sharded or sequence_parallel_enabled or full_sequence,
        )

        def preprocess(*inputs: Any) -> Any:
            adjusted_inputs = inputs
            if sequence_sharded:
                if not inputs or not isinstance(inputs[0], torch.Tensor):
                    raise TypeError(
                        "TP_SEQUENCE_SHARDED preprocessing requires a Tensor input"
                    )
                input_tensor = inputs[0]
                expected_length = (
                    int(tp_sequence_length)
                    if sequence_parallel_enabled
                    else int(global_sequence_length)
                )
                if input_tensor.dim() < 2 or int(input_tensor.shape[0]) != expected_length:
                    raise ValueError(
                        "TP_SEQUENCE_SHARDED input has the wrong sequence dimension: "
                        f"expected {expected_length}, got "
                        f"{tuple(int(value) for value in input_tensor.shape)}"
                    )
                if not sequence_parallel_enabled and not full_sequence:
                    input_tensor = input_tensor.narrow(
                        0, int(tp_sequence_start), int(tp_sequence_length)
                    )
                adjusted_inputs = (input_tensor, *inputs[1:])
            raw = (
                original(*adjusted_inputs)
                if original is not None
                else adjusted_inputs[0]
                if len(physical.outputs) == 1 and len(adjusted_inputs) == 1
                else adjusted_inputs
            )
            values = [raw] if len(physical.outputs) == 1 else list(raw)
            if len(values) != len(physical.outputs):
                raise ValueError("Megatron hook preprocessing output count mismatch")
            enrich = (self._enrich_eager_output
                      if self.hook_runtime.mode is HookRuntimeMode.EAGER_IMMEDIATE
                      and not self.hook_runtime.te_capture_session_active
                      else self._enrich_output)
            enriched = [
                enrich(hook, spec, value)
                for spec, value in zip(physical.outputs, values)
            ]
            return enriched[0] if len(enriched) == 1 else tuple(enriched)

        return preprocess

    def _packed_meta(self, hook: HookPointV1) -> tuple[torch.Tensor, torch.Tensor]:
        """(valid_count, prefix) device tensors for an eager packed producer."""
        info = self._eager_packed_info[id(hook)]
        microbatch = self._direct_packed_microbatch
        if microbatch is not None:
            context = self._metadata_context
            if microbatch in context._direct_microbatches:
                return (context.direct_count(info[2], microbatch),
                        context.direct_count(info[3], microbatch))
            return (context.source_view(info[2], microbatch),
                    context.source_view(info[3], microbatch))
        valid = getattr(hook, info[0], None)
        prefix = getattr(hook, info[1], None)
        if valid is not info[4] or prefix is not info[5]:
            # Validate once per bound tensor object, not per call.
            if not isinstance(valid, torch.Tensor) or valid.dtype is not torch.int64:
                raise TypeError("SEQ_PREFIX_PACK requires int64 valid-count context")
            if prefix is None:
                raise RuntimeError(
                    "SEQ_PREFIX_PACK requires a metadata-owned valid-count prefix"
                )
            if not isinstance(prefix, torch.Tensor):
                raise TypeError("HookOutput producer metadata must contain only tensors")
            info[4], info[5] = valid, prefix
        return valid, prefix

    def _enrich_eager_output(
        self, hook: HookPointV1, spec: TransportSpec, value: Any,
    ) -> Any:
        # A bare tensor has no producer metadata to validate/preserve. Build
        # its final wrapper once. Tuple/custom metadata follows the old path.
        if isinstance(value, torch.Tensor) and spec.transport_type is TransportType.SEQ_PREFIX_PACK:
            if id(hook) in self._eager_packed_info:
                return HookOutput._trusted(value, self._packed_meta(hook))
            valid = self._current_valid_count(hook)
            direction = self._hook_phase(hook).name.lower()
            prefix = getattr(hook, f"valid_count_prefix_{direction}", None)
            if prefix is None:
                raise RuntimeError("SEQ_PREFIX_PACK requires a metadata-owned valid-count prefix")
            return HookOutput(value, (valid, prefix))
        return self._enrich_output(hook, spec, value)

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
            direction = self._hook_phase(hook).name.lower()
            prefix = getattr(hook, f"valid_count_prefix_{direction}", None)
            if prefix is None:
                raise RuntimeError("SEQ_PREFIX_PACK requires a metadata-owned valid-count prefix")
            return HookOutput(output.tensor, (valid, prefix))
        if spec.transport_type is TransportType.SEGMENTED_PACK:
            if output.producer_meta:
                if len(output.producer_meta) != 2:
                    raise ValueError("SEGMENTED_PACK requires explicit starts and ends")
                starts, ends = output.producer_meta
                if (starts.dtype != torch.int64 or ends.dtype != torch.int64
                        or starts.ndim != 1 or ends.shape != starts.shape or starts.numel() == 0
                        or starts.device != output.tensor.device or ends.device != output.tensor.device
                        or not starts.is_contiguous() or not ends.is_contiguous()):
                    raise ValueError("SEGMENTED_PACK ranges must be matching contiguous int64 vectors")
            else:
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
