"""Runtime facade for Megatron schedule-side DMI metadata events."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import time
from typing import Any, Callable

import torch

from .metadata_context import (
    DMIMetadataContext,
    DMIMetadataFieldSpec,
    DMIMetadataPropagator,
    LocalMetadataPropagator,
    PerDPCPUMetadataPropagator,
)
from .adapter import (
    MegatronFullIterationPlan,
    MegatronTrainingContext,
)
from dmi.api.v1 import ProducerPlan, StepReservation
from .hooks.specs import HookPhase
from .records.format import (
    EVALUATION_BOUNDARY_CELL_TYPES,
    EVALUATION_BOUNDARY_LAYOUT_NAME,
    EVALUATION_BOUNDARY_NBYTES,
    evaluation_boundary_row,
)


_SCHEDULE_RESUME_STATE_VERSION = 1


@dataclass(frozen=True)
class DMIScheduleEvent:
    direction: str
    microbatch_id: int
    scope_id: int


@dataclass(frozen=True)
class DMICPUMetadataGroups:
    pp_source_rank: int
    tp_source_rank: int
    pp_cpu_ranks: tuple[int, ...]
    tp_cpu_ranks: tuple[int, ...]
    pp_cpu_group: Any | None
    tp_cpu_group: Any | None


class MegatronScheduleRuntime:
    """Stateful bridge between Megatron schedules and DMI metadata buffers."""

    def __init__(
        self,
        propagator: DMIMetadataPropagator,
        *,
        host_engine: Any | None = None,
    ) -> None:
        self.propagator = propagator
        self.host_engine = host_engine
        self.current_event: DMIScheduleEvent | None = None
        self.active = False
        self._ingested_microbatches: set[int] = set()
        self.adaptor: Any | None = None
        self.global_batch_id = 0
        self.phase = "train"
        self.execution_order_id = 1
        self._phase_open = False
        self._phase_training_iteration_id = 1
        self._phase_eval_index = 0
        self.dp_rank = 0
        self.shard_rank = 0
        self.token_start = 0
        self.current_phase_tensor: torch.Tensor | None = None
        self._last_phase: HookPhase | None = None
        self._full_iteration_capture_active = False
        self._full_iteration_capture_valid_counts: list[list[int] | None] | None = None
        self._full_iteration_metadata_preloaded = False
        self._logical_training_iteration_id: int | None = None
        self._active_attempt_id: int | None = None
        self._next_attempt_id = 0
        self._attempt_statuses: dict[int, int] = {}
        self._flush_every_n_train_iters = 0
        self._iteration_flush_callback: Callable[[], None] | None = None
        self._iteration_flush_barrier: Callable[[], None] | None = None
        self._iteration_flush_logger: Callable[[int, float], None] | None = None
        self.attempt_status_hook: Any | None = None
        self._attempt_status_tensor: torch.Tensor | None = None
        self.dataset_provenance_modes: dict[str, str] = {
            "train": "constant-zero",
            "valid": "constant-zero",
            "test": "constant-zero",
        }
        self._dataset_id_override: int | None = None
        self._next_phase_global_batch_id = {
            "valid": 1,
            "test": 1,
        }
        self._next_eval_index = {
            "valid": 0,
            "test": 0,
        }

    @property
    def current_attempt_id(self) -> int:
        return 0 if self._active_attempt_id is None else int(self._active_attempt_id)

    def configure_iteration_flush(
        self,
        interval: int,
        *,
        flush_callback: Callable[[], None] | None = None,
        barrier_callback: Callable[[], None] | None = None,
        logger: Callable[[int, float], None] | None = None,
    ) -> None:
        """Configure a durable flush after every N accepted train iterations."""

        interval = int(interval)
        if interval < 0:
            raise ValueError("DMI iteration flush interval must be nonnegative")
        if interval == 0:
            if flush_callback is not None or barrier_callback is not None or logger is not None:
                raise ValueError("DMI disabled iteration flushing must not configure callbacks")
        else:
            if not callable(flush_callback):
                raise TypeError("DMI iteration flushing requires a flush callback")
            if not callable(barrier_callback):
                raise TypeError("DMI iteration flushing requires a barrier callback")
            if logger is not None and not callable(logger):
                raise TypeError("DMI iteration flush logger must be callable")
        self._flush_every_n_train_iters = interval
        self._iteration_flush_callback = flush_callback
        self._iteration_flush_barrier = barrier_callback
        self._iteration_flush_logger = logger

    def configure_dataset_provenance(self, modes: dict[str, str]) -> None:
        expected = {"train", "valid", "test"}
        if set(modes) != expected:
            raise ValueError(
                "DMI dataset provenance modes must define train, valid, and test"
            )
        invalid = {
            phase: mode
            for phase, mode in modes.items()
            if mode not in {"constant-zero", "dynamic"}
        }
        if invalid:
            raise ValueError(f"Invalid DMI dataset provenance modes: {invalid}")
        if "dataset_id" not in self.propagator.context.field_specs and any(
            mode == "dynamic" for mode in modes.values()
        ):
            raise RuntimeError(
                "DMI dynamic dataset provenance requires a dataset_id metadata field"
            )
        self.dataset_provenance_modes = dict(modes)

    def exact_resume_state_dict(self, *, checkpoint_iteration: int) -> dict[str, object]:
        checkpoint_iteration = int(checkpoint_iteration)
        if checkpoint_iteration < 0:
            raise ValueError("DMI checkpoint iteration must be nonnegative")
        if self.active or self.current_event is not None:
            raise RuntimeError("DMI checkpoint requires an inactive schedule")
        if self._logical_training_iteration_id is not None:
            raise RuntimeError("DMI checkpoint requires a closed logical iteration")
        if self._active_attempt_id is not None:
            raise RuntimeError("DMI checkpoint requires no active attempt")
        if self._full_iteration_capture_active:
            raise RuntimeError("DMI checkpoint requires no active full-iteration capture")
        if self._full_iteration_metadata_preloaded:
            raise RuntimeError("DMI checkpoint found preloaded full-iteration metadata")
        if self.phase != "train" or not self._phase_open:
            raise RuntimeError("the first DMI exact-resume path checkpoints train phase only")
        accepted = sum(status == 1 for status in self._attempt_statuses.values())
        aborted = sum(status == -1 for status in self._attempt_statuses.values())
        normal_boundary = self.global_batch_id == checkpoint_iteration and accepted == 1
        rerun_boundary = self.global_batch_id == checkpoint_iteration + 1 and aborted == 1
        if not normal_boundary and not rerun_boundary:
            raise RuntimeError(
                "DMI capture iteration disagrees with the checkpoint boundary: "
                f"capture={self.global_batch_id}, checkpoint={checkpoint_iteration}, "
                f"accepted={accepted}, aborted={aborted}"
            )
        return {
            "schema_version": _SCHEDULE_RESUME_STATE_VERSION,
            "next_training_global_batch_id": checkpoint_iteration + 1,
            "next_validation_global_batch_id": int(
                self._next_phase_global_batch_id["valid"]
            ),
            "next_test_global_batch_id": int(
                self._next_phase_global_batch_id["test"]
            ),
            "next_validation_eval_index": int(self._next_eval_index["valid"]),
            "next_test_eval_index": int(self._next_eval_index["test"]),
        }

    def load_exact_resume_state_dict(
        self,
        state: dict[str, object],
        *,
        checkpoint_iteration: int,
    ) -> None:
        expected = {
            "schema_version",
            "next_training_global_batch_id",
            "next_validation_global_batch_id",
            "next_test_global_batch_id",
            "next_validation_eval_index",
            "next_test_eval_index",
        }
        if set(state) != expected:
            missing = sorted(expected - set(state))
            extra = sorted(set(state) - expected)
            raise ValueError(
                f"DMI capture-state fields mismatch: missing={missing}, extra={extra}"
            )
        if int(state["schema_version"]) != _SCHEDULE_RESUME_STATE_VERSION:
            raise ValueError(
                "unsupported DMI capture-state version: "
                f"{state['schema_version']!r}"
            )
        checkpoint_iteration = int(checkpoint_iteration)
        next_training = int(state["next_training_global_batch_id"])
        if next_training != checkpoint_iteration + 1:
            raise ValueError(
                "DMI capture state does not continue after the checkpoint: "
                f"next={next_training}, checkpoint={checkpoint_iteration}"
            )
        phase_ids = {
            "valid": int(state["next_validation_global_batch_id"]),
            "test": int(state["next_test_global_batch_id"]),
        }
        eval_indices = {
            "valid": int(state["next_validation_eval_index"]),
            "test": int(state["next_test_eval_index"]),
        }
        if any(value < 1 for value in phase_ids.values()):
            raise ValueError("DMI phase-local global batch IDs must be positive")
        if any(value < 0 for value in eval_indices.values()):
            raise ValueError("DMI evaluation indices must be nonnegative")
        self.global_batch_id = checkpoint_iteration
        self.phase = "train"
        self._phase_open = False
        self._phase_training_iteration_id = next_training
        self._phase_eval_index = 0
        self._next_phase_global_batch_id = phase_ids
        self._next_eval_index = eval_indices
        self._logical_training_iteration_id = None
        self._active_attempt_id = None
        self._next_attempt_id = 0
        self._attempt_statuses.clear()

    def set_attempt_status_hook(
        self,
        hook: Any | None,
        *,
        device: torch.device | str | int,
    ) -> None:
        self.attempt_status_hook = hook
        self._attempt_status_tensor = (
            None
            if hook is None
            else torch.empty((1,), dtype=torch.int64, device=device)
        )

    def set_dataset_id_override(self, dataset_id: int | None) -> None:
        if dataset_id is None:
            self._dataset_id_override = None
            return
        dataset_id = int(dataset_id)
        if dataset_id < 0:
            raise ValueError("DMI dataset ID override must be nonnegative")
        if self.dataset_provenance_modes[self.phase] != "constant-zero":
            raise RuntimeError(
                "DMI dataset ID override is only valid for a constant-source invocation"
            )
        self._dataset_id_override = dataset_id

    def begin_logical_iteration(self, global_batch_id: int) -> None:
        if self.phase != "train":
            raise RuntimeError("DMI logical training iteration requires train phase")
        if self._logical_training_iteration_id is not None:
            raise RuntimeError("DMI logical training iteration is already active")
        global_batch_id = int(global_batch_id)
        if global_batch_id < 1:
            raise ValueError("DMI training global_batch_id must be positive")
        self._logical_training_iteration_id = global_batch_id
        self.global_batch_id = global_batch_id
        self._active_attempt_id = None
        self._next_attempt_id = 0
        self._attempt_statuses.clear()

    def begin_attempt(self, attempt_id: int) -> None:
        if self._logical_training_iteration_id is None:
            raise RuntimeError("DMI attempt requires an active logical training iteration")
        if self._active_attempt_id is not None:
            raise RuntimeError("DMI training attempt is already active")
        attempt_id = int(attempt_id)
        if attempt_id != self._next_attempt_id:
            raise ValueError(
                f"DMI attempt ID must be {self._next_attempt_id}, got {attempt_id}"
            )
        self._active_attempt_id = attempt_id
        if self.adaptor is not None:
            self.adaptor.begin_attempt(
                phase="train",
                global_batch_id=int(self.global_batch_id),
                attempt_id=attempt_id,
            )

    def finish_attempt(self, status: int) -> None:
        attempt_id = self._active_attempt_id
        if attempt_id is None:
            raise RuntimeError("DMI training attempt is not active")
        status = int(status)
        if status not in {-1, 0, 1}:
            raise ValueError(f"Invalid DMI attempt status: {status}")
        if attempt_id in self._attempt_statuses:
            raise RuntimeError(f"DMI attempt {attempt_id} already has a status")
        if status == 1 and 1 in self._attempt_statuses.values():
            raise RuntimeError("DMI logical iteration has more than one accepted attempt")
        self._emit_attempt_status(attempt_id=attempt_id, status=status)
        if self.adaptor is not None:
            self.adaptor.end_attempt(attempt_id=attempt_id)
        self._attempt_statuses[attempt_id] = status
        self._active_attempt_id = None
        self._next_attempt_id = attempt_id + 1

    def _emit_attempt_status(self, *, attempt_id: int, status: int) -> None:
        hook = self.attempt_status_hook
        tensor = self._attempt_status_tensor
        adaptor = self.adaptor
        if hook is None:
            return
        if tensor is None or adaptor is None:
            raise RuntimeError("DMI attempt status hook is not fully configured")
        tensor.fill_(int(status))
        adaptor.set_current_iteration(
            MegatronTrainingContext(
                global_batch_id=int(self.global_batch_id),
                microbatch_id=-1,
                valid_counts=(),
                dataset_ids=(),
                attempt_id=int(attempt_id),
                direction="iter",
                phase="train",
                dp_rank=-1,
                shard_rank=-1,
                token_start=0,
            )
        )
        try:
            with torch.no_grad():
                hook(tensor)
        finally:
            adaptor.clear_current_iteration()

    def finish_logical_iteration(self) -> None:
        if self._logical_training_iteration_id is None:
            raise RuntimeError("DMI logical training iteration is not active")
        if self.active:
            raise RuntimeError("DMI cannot finish a logical iteration during schedule execution")
        if self._active_attempt_id is not None:
            raise RuntimeError("DMI cannot finish a logical iteration with an active attempt")
        accepted = sum(status == 1 for status in self._attempt_statuses.values())
        aborted = sum(status == -1 for status in self._attempt_statuses.values())
        if accepted != 1 and aborted != 1:
            raise RuntimeError(
                "DMI logical iteration must finish with one accepted or controlled-abort attempt"
            )
        completed_iteration = int(self._logical_training_iteration_id)
        self._logical_training_iteration_id = None
        interval = int(self._flush_every_n_train_iters)
        if accepted == 1 and interval > 0 and completed_iteration % interval == 0:
            flush_callback = self._iteration_flush_callback
            barrier_callback = self._iteration_flush_barrier
            if flush_callback is None or barrier_callback is None:
                raise RuntimeError("DMI iteration flush callbacks are not configured")
            start = time.monotonic()
            flush_callback()
            barrier_callback()
            logger = self._iteration_flush_logger
            if logger is not None:
                logger(completed_iteration, time.monotonic() - start)

    def begin_iteration(self, active_num_microbatches: int, *, forward_only: bool = False) -> None:
        del forward_only
        if (
            self.phase == "train"
            and self._phase_open
            and self._logical_training_iteration_id is None
        ):
            raise RuntimeError(
                "DMI training schedule execution requires an active logical iteration"
            )
        if (
            self.phase == "train"
            and self._logical_training_iteration_id is not None
            and self._active_attempt_id is None
        ):
            raise RuntimeError("DMI training schedule execution requires an active attempt")
        self._configure_active_metadata_fields()
        metadata_preloaded = self._full_iteration_metadata_preloaded
        if metadata_preloaded:
            self.propagator.context.begin_iteration(active_num_microbatches, clear_buffers=False)
            self._full_iteration_metadata_preloaded = False
        else:
            self.propagator.begin_iteration(active_num_microbatches)
        self.current_event = None
        self.active = True
        if not metadata_preloaded:
            self._ingested_microbatches.clear()
        self._last_phase = None

    def enter_phase(
        self,
        phase: str,
        *,
        training_iteration_id_start: int,
        training_iteration_id_end: int | None = None,
        global_batch_id_start: int = 1,
        eval_index: int = 0,
    ) -> None:
        phase = str(phase)
        if phase not in {"train", "valid", "test"}:
            raise ValueError(f"Unsupported DMI training phase: {phase!r}")
        training_start = int(training_iteration_id_start)
        if training_iteration_id_end is not None and int(training_iteration_id_end) < training_start:
            raise ValueError("DMI phase ids must be 1-based and monotonic")
        global_start = int(global_batch_id_start)
        eval_index = int(eval_index)
        if training_start < 1 or global_start < 1 or eval_index < 0:
            raise ValueError("DMI phase ids must be 1-based and monotonic")

        if self._phase_open:
            if self.phase == phase and self._phase_eval_index == eval_index:
                return
            self.seal_current_phase()

        self.phase = phase
        self._dataset_id_override = None
        self.global_batch_id = global_start
        self._phase_training_iteration_id = training_start
        self._phase_eval_index = eval_index
        self._submit_eval_boundary("entry")
        self._phase_open = True

    def seal_current_phase(self) -> None:
        if not self._phase_open:
            return
        self._submit_eval_boundary("exit")
        if self.phase in self._next_phase_global_batch_id:
            self._next_phase_global_batch_id[self.phase] = int(self.global_batch_id)
            self._next_eval_index[self.phase] = int(self._phase_eval_index) + 1
        self._phase_open = False

    def begin_full_iteration_capture(
        self,
        valid_counts_by_microbatch: list[list[int] | None] | None = None,
        dataset_ids_by_microbatch: list[list[int] | None] | None = None,
    ) -> None:
        if self._full_iteration_capture_active:
            raise RuntimeError("Megatron DMI full-iteration capture is already active")
        self._full_iteration_capture_active = True
        self._full_iteration_capture_valid_counts = valid_counts_by_microbatch
        if valid_counts_by_microbatch is not None or dataset_ids_by_microbatch is not None:
            self.load_full_iteration_metadata(
                valid_counts_by_microbatch or [None] * len(dataset_ids_by_microbatch or ()),
                dataset_ids_by_microbatch=dataset_ids_by_microbatch,
            )

    def finish_full_iteration_capture(self) -> None:
        if not self._full_iteration_capture_active:
            return
        self._full_iteration_capture_active = False
        self._full_iteration_capture_valid_counts = None

    def abort_full_iteration_capture(self) -> None:
        self.finish_full_iteration_capture()

    def full_iteration_contexts(
        self,
        plan: MegatronFullIterationPlan,
        valid_counts_by_microbatch: list[list[int] | None],
    ) -> list[MegatronTrainingContext]:
        contexts: list[MegatronTrainingContext] = []
        for item in plan.entries:
            microbatch_id = int(item.event.microbatch_id)
            if microbatch_id < 0 or microbatch_id >= len(valid_counts_by_microbatch):
                raise IndexError(
                    "DMI full-iteration replay missing valid-count slot for "
                    f"microbatch {microbatch_id}"
                )
            valid_counts = self._valid_counts_for_microbatch(microbatch_id)
            contexts.append(
                MegatronTrainingContext(
                    global_batch_id=int(self.global_batch_id),
                    microbatch_id=int(item.event.microbatch_id),
                    valid_counts=tuple(int(x) for x in valid_counts),
                    dataset_ids=self._dataset_ids_for_microbatch(microbatch_id),
                    attempt_id=self.current_attempt_id,
                    direction=str(item.event.direction),
                    phase=str(self.phase),
                    scope_id=int(item.event.scope_id),
                    dp_rank=int(item.event.dp_rank),
                    shard_rank=int(item.event.shard_rank),
                    token_start=int(item.event.token_start),
                    model_id=item.event.model_id,
                )
            )
        return contexts

    def load_full_iteration_metadata(
        self,
        valid_counts_by_microbatch: list[list[int] | None],
        *,
        dataset_ids_by_microbatch: list[list[int] | None] | None = None,
    ) -> None:
        if dataset_ids_by_microbatch is not None and (
            len(dataset_ids_by_microbatch) != len(valid_counts_by_microbatch)
        ):
            raise ValueError(
                "DMI full-iteration valid_count and dataset_id metadata lengths differ"
            )
        self._configure_active_metadata_fields()
        self.propagator.begin_iteration(len(valid_counts_by_microbatch))
        active_fields = self.propagator.context.active_field_names
        if not active_fields:
            return
        for microbatch_id, valid_counts in enumerate(valid_counts_by_microbatch):
            fields: dict[str, Any] = {}
            cpu_fields: dict[str, Any] = {}
            if "valid_count" in active_fields:
                if valid_counts is None and self._requires_local_full_iteration_counts():
                    raise RuntimeError(
                        "DMI full-iteration replay missing valid counts for "
                        f"microbatch {microbatch_id}"
                    )
                if valid_counts is not None:
                    fields["valid_count"] = valid_counts
                    cpu_fields["valid_count"] = valid_counts
            if "dataset_id" in active_fields:
                dataset_ids = (
                    None
                    if dataset_ids_by_microbatch is None
                    else dataset_ids_by_microbatch[microbatch_id]
                )
                if dataset_ids is None and self._requires_local_full_iteration_counts():
                    raise RuntimeError(
                        "DMI full-iteration replay missing dataset IDs for "
                        f"microbatch {microbatch_id}"
                    )
                if dataset_ids is not None:
                    cpu_fields["dataset_id"] = dataset_ids
            self.propagator.ingest_microbatch(
                microbatch_id,
                fields,
                cpu_fields=cpu_fields or None,
            )
            self._ingested_microbatches.add(int(microbatch_id))
        for microbatch_id in range(len(valid_counts_by_microbatch)):
            self.propagator.wait_microbatch(microbatch_id)
            self._ingested_microbatches.add(int(microbatch_id))
        self._full_iteration_metadata_preloaded = True

    def _requires_local_full_iteration_counts(self) -> bool:
        return bool(self.propagator.is_metadata_source_rank)

    def _has_metadata_field(self, name: str) -> bool:
        return name in self.propagator.context.field_specs

    def _configure_active_metadata_fields(self) -> None:
        names = [
            name
            for name in self.propagator.context.field_specs
            if name != "dataset_id"
            or self.dataset_provenance_modes[self.phase] == "dynamic"
        ]
        self.propagator.context.set_active_fields(names)

    def _valid_counts_for_microbatch(self, microbatch_id: int) -> tuple[int, ...]:
        if not self._has_metadata_field("valid_count"):
            return ()
        return self.propagator.context.source_cpu("valid_count", microbatch_id)

    def _dataset_ids_for_microbatch(self, microbatch_id: int) -> tuple[int, ...]:
        if self.dataset_provenance_modes[self.phase] == "constant-zero":
            if self._dataset_id_override in (None, 0):
                return ()
            valid_counts = self._valid_counts_for_microbatch(microbatch_id)
            batch_size = len(valid_counts) or self.propagator.context.max_batch_size
            return tuple(int(self._dataset_id_override) for _ in range(batch_size))
        if not self._has_metadata_field("dataset_id"):
            raise RuntimeError("DMI dynamic provenance is missing dataset_id metadata")
        return self.propagator.context.source_cpu("dataset_id", microbatch_id)

    def finish_full_iteration_replay(self) -> None:
        self.propagator.end_iteration()
        self._clear_transient_schedule_state()
        if self.phase != "train" or self._logical_training_iteration_id is None:
            self._commit_phase_batch()

    def _clear_transient_schedule_state(self) -> None:
        if self.adaptor is not None:
            self.adaptor.clear_current_event()
        self.current_event = None
        self.active = False
        self._ingested_microbatches.clear()
        self._full_iteration_metadata_preloaded = False
        self._last_phase = None

    def _commit_phase_batch(self) -> None:
        self.global_batch_id += 1
        self.execution_order_id += 1

    def set_event(self, direction: str, microbatch_id: int, scope_id: int = 0) -> None:
        if not self.active:
            return
        self._update_current_phase(str(direction))
        self.current_event = DMIScheduleEvent(
            direction=str(direction),
            microbatch_id=int(microbatch_id),
            scope_id=int(scope_id),
        )

    def record_current_microbatch_metadata(
        self,
        valid_count: torch.Tensor | None,
        valid_count_cpu: torch.Tensor | list[int] | tuple[int, ...] | int | None = None,
        dataset_id_cpu: torch.Tensor | list[int] | tuple[int, ...] | int | None = None,
        segment_metadata: torch.Tensor | None = None,
        segment_metadata_cpu: (
            torch.Tensor | list[int] | tuple[int, ...] | int | None
        ) = None,
    ) -> None:
        if not self.active or self.current_event is None:
            return
        active_fields = self.propagator.context.active_field_names
        if not active_fields:
            return
        microbatch_id = self.current_event.microbatch_id
        if microbatch_id in self._ingested_microbatches:
            return
        fields: dict[str, Any] = {}
        cpu_fields: dict[str, Any] = {}
        if self.propagator.is_metadata_source_rank:
            if "segment_metadata" in active_fields:
                if segment_metadata is None:
                    raise RuntimeError(
                        "DMI packed segmentation metadata is required for this microbatch"
                    )
                fields["segment_metadata"] = segment_metadata
                if segment_metadata_cpu is not None:
                    cpu_fields["segment_metadata"] = segment_metadata_cpu
            if "valid_count" in active_fields:
                if valid_count is None:
                    raise RuntimeError(
                        "DMI valid_count metadata is required for this microbatch"
                    )
                fields["valid_count"] = valid_count
                if valid_count_cpu is not None:
                    cpu_fields["valid_count"] = valid_count_cpu
            if "dataset_id" in active_fields:
                if dataset_id_cpu is None:
                    raise RuntimeError(
                        "DMI dynamic dataset provenance requires per-sample dataset_id"
                    )
                cpu_fields["dataset_id"] = dataset_id_cpu
                dataset_numel = int(torch.as_tensor(dataset_id_cpu).numel())
                valid_source = (
                    valid_count_cpu if valid_count_cpu is not None else valid_count
                )
                if valid_source is not None:
                    valid_numel = int(torch.as_tensor(valid_source).numel())
                    if dataset_numel != valid_numel:
                        raise RuntimeError(
                            "DMI dataset_id and valid_count batch lengths do not match"
                        )
                elif dataset_numel != self.propagator.context.max_batch_size:
                    raise RuntimeError(
                        "DMI dataset_id batch length does not match micro_batch_size"
                    )
        self.propagator.ingest_microbatch(
            microbatch_id,
            fields,
            cpu_fields=cpu_fields or None,
        )
        self._ingested_microbatches.add(microbatch_id)

    def enter_current_scope(self) -> None:
        if not self.active or self.current_event is None:
            return
        event = self.current_event
        self.propagator.enter_scope(event.direction, event.scope_id, event.microbatch_id)
        self._set_adaptor_event_if_needed(event)

    def end_iteration(self) -> None:
        if not self.active:
            return
        self.propagator.end_iteration()
        self._clear_transient_schedule_state()
        if not self._full_iteration_capture_active and (
            self.phase != "train" or self._logical_training_iteration_id is None
        ):
            self._commit_phase_batch()

    def set_current_phase_tensor(self, tensor: torch.Tensor | None) -> None:
        if tensor is not None:
            if tensor.dtype != torch.int32:
                raise TypeError("current_phase_tensor must use torch.int32")
            if tensor.numel() != 1:
                raise ValueError("current_phase_tensor must be scalar")
        self.current_phase_tensor = tensor
        self._last_phase = None

    def _update_current_phase(self, direction: str) -> None:
        phase = self._phase_for_direction(direction)
        if self.current_phase_tensor is None or phase is self._last_phase:
            return
        with torch.no_grad():
            self.current_phase_tensor.fill_(int(phase.value))
        self._last_phase = phase

    @staticmethod
    def _phase_for_direction(direction: str) -> HookPhase:
        normalized = direction.strip().lower()
        if normalized in {"fwd", "forward"}:
            return HookPhase.FWD
        if normalized in {"bwd", "backward"}:
            return HookPhase.BWD
        raise ValueError(f"Unsupported DMI schedule direction: {direction!r}")

    def _set_adaptor_event_if_needed(self, event: DMIScheduleEvent) -> None:
        adaptor = self.adaptor
        if adaptor is None:
            return
        if adaptor.hook_runtime.capture_event_context:
            valid_counts: tuple[int, ...] = ()
            if (
                self._has_metadata_field("valid_count")
                and self._full_iteration_capture_valid_counts is not None
            ):
                microbatch_id = int(event.microbatch_id)
                if (
                    microbatch_id < 0
                    or microbatch_id >= len(self._full_iteration_capture_valid_counts)
                ):
                    raise IndexError(
                        "DMI full-iteration capture missing valid-count slot for "
                        f"microbatch {microbatch_id}"
                    )
                raw_valid_counts = self._full_iteration_capture_valid_counts[microbatch_id]
                if raw_valid_counts is None:
                    raise RuntimeError(
                        "DMI full-iteration capture missing valid counts for "
                        f"microbatch {microbatch_id}"
                    )
                valid_counts = tuple(int(x) for x in raw_valid_counts)
            adaptor.set_current_event(
                MegatronTrainingContext(
                    global_batch_id=int(self.global_batch_id),
                    microbatch_id=int(event.microbatch_id),
                    valid_counts=valid_counts,
                    dataset_ids=self._dataset_ids_for_microbatch(event.microbatch_id),
                    attempt_id=self.current_attempt_id,
                    direction=str(event.direction),
                    phase=str(self.phase),
                    scope_id=int(event.scope_id),
                    dp_rank=int(self.dp_rank),
                    shard_rank=int(self.shard_rank),
                    token_start=int(self.token_start),
                )
            )
            return
        ctx = MegatronTrainingContext(
            global_batch_id=int(self.global_batch_id),
            microbatch_id=int(event.microbatch_id),
            valid_counts=self._valid_counts_for_microbatch(event.microbatch_id),
            dataset_ids=self._dataset_ids_for_microbatch(event.microbatch_id),
            attempt_id=self.current_attempt_id,
            direction=str(event.direction),
            phase=str(self.phase),
            dp_rank=int(self.dp_rank),
            shard_rank=int(self.shard_rank),
            token_start=int(self.token_start),
        )
        adaptor.set_current_event(ctx)

    def _submit_eval_boundary(self, boundary_type: str) -> None:
        if self.phase == "train":
            return
        host_engine = self.host_engine
        if host_engine is None:
            return
        if self.adaptor is None:
            raise RuntimeError("DMI evaluation boundary requires an active adaptor")
        host_engine.submit_record(
            EVALUATION_BOUNDARY_LAYOUT_NAME,
            evaluation_boundary_row(
                model_id=str(self.adaptor.model_id),
                training_iteration_id=int(self._phase_training_iteration_id),
                phase=str(self.phase),
                eval_index=int(self._phase_eval_index),
                boundary_type=str(boundary_type),
                next_global_batch_id=int(self.global_batch_id),
            ),
            EVALUATION_BOUNDARY_CELL_TYPES,
            nbytes=EVALUATION_BOUNDARY_NBYTES,
        )


_active_runtime: MegatronScheduleRuntime | None = None


def set_active_megatron_schedule_runtime(runtime: MegatronScheduleRuntime | None) -> None:
    global _active_runtime
    _active_runtime = runtime


def get_active_megatron_schedule_runtime() -> MegatronScheduleRuntime | None:
    return _active_runtime


def _prefix_product(values: list[int], init: int = 1) -> list[int]:
    result = [init]
    for value in values:
        init *= int(value)
        result.append(init)
    return result


def _decompose(index: int, shape: list[int], stride: list[int] | None = None) -> list[int]:
    if stride is None:
        stride = _prefix_product(shape)
    return [(index // dim_stride) % dim_size for dim_size, dim_stride in zip(shape, stride)]


def _generate_masked_rank_groups(
    *,
    world_size: int,
    parallel_size: list[int],
    mask: list[bool],
) -> list[list[int]]:
    masked_shape = [size for size, masked in zip(parallel_size, mask) if masked]
    unmasked_shape = [size for size, masked in zip(parallel_size, mask) if not masked]
    global_stride = _prefix_product(parallel_size)
    masked_stride = [stride for stride, masked in zip(global_stride, mask) if masked]
    unmasked_stride = [stride for stride, masked in zip(global_stride, mask) if not masked]

    group_size = _prefix_product(masked_shape)[-1]
    num_groups = world_size // group_size
    rank_groups: list[list[int]] = []
    for group_index in range(num_groups):
        group_index_parts = _decompose(group_index, unmasked_shape)
        ranks: list[int] = []
        for rank_in_group in range(group_size):
            rank_index_parts = _decompose(rank_in_group, masked_shape)
            rank = sum(a * b for a, b in zip(rank_index_parts, masked_stride))
            rank += sum(a * b for a, b in zip(group_index_parts, unmasked_stride))
            ranks.append(int(rank))
        rank_groups.append(ranks)
    return rank_groups


def _rank_groups_for_token(
    *,
    token: str,
    tensor_model_parallel_size: int,
    pipeline_model_parallel_size: int,
    data_parallel_size: int,
    context_parallel_size: int,
    expert_model_parallel_size: int,
    rank_order: str,
) -> list[list[int]]:
    name_to_size = {
        "tp": int(tensor_model_parallel_size),
        "pp": int(pipeline_model_parallel_size),
        "dp": int(data_parallel_size),
        "cp": int(context_parallel_size),
        "ep": int(expert_model_parallel_size),
    }
    normalized_order = str(rank_order).lower()
    for name, size in name_to_size.items():
        if name not in normalized_order and size != 1:
            raise RuntimeError(
                f"DMI metadata rank order {rank_order!r} omits non-singleton {name}={size}"
            )
        if name not in normalized_order:
            normalized_order = f"{normalized_order}-{name}"

    ordered_tokens = normalized_order.split("-")
    ordered_size = [name_to_size[name] for name in ordered_tokens]
    mask = [False] * len(ordered_tokens)
    for name in token.split("-"):
        mask[ordered_tokens.index(name)] = True

    world_size = 1
    for size in name_to_size.values():
        world_size *= int(size)
    return _generate_masked_rank_groups(
        world_size=world_size,
        parallel_size=ordered_size,
        mask=mask,
    )


def _build_dmi_cpu_metadata_groups(
    *,
    parallel_state_module: Any,
    dist: Any,
    tensor_model_parallel_size: int,
    pipeline_model_parallel_size: int,
    data_parallel_size: int,
    context_parallel_size: int,
    expert_model_parallel_size: int,
    rank_order: str,
) -> DMICPUMetadataGroups:
    if int(context_parallel_size) != 1:
        raise NotImplementedError("DMI CPU metadata propagation does not support CP yet")

    rank = int(dist.get_rank())
    tp_size = int(tensor_model_parallel_size)
    pp_size = int(pipeline_model_parallel_size)
    mp_groups = _rank_groups_for_token(
        token="tp-pp",
        tensor_model_parallel_size=tp_size,
        pipeline_model_parallel_size=pp_size,
        data_parallel_size=int(data_parallel_size),
        context_parallel_size=int(context_parallel_size),
        # Sample metadata follows Megatron's dense decoder topology.
        # EP is represented through dense-DP source ranks, not as an
        # independent metadata-group dimension.
        expert_model_parallel_size=1,
        rank_order=str(rank_order),
    )

    current_pp_cpu_ranks: tuple[int, ...] | None = None
    current_tp_cpu_ranks: tuple[int, ...] | None = None
    current_pp_cpu_group = None
    current_tp_cpu_group = None
    current_pp_source_rank: int | None = None
    current_tp_source_rank: int | None = None

    for mp_index, mp_ranks_raw in enumerate(mp_groups):
        mp_ranks = tuple(int(item) for item in mp_ranks_raw)
        if len(mp_ranks) != tp_size * pp_size:
            raise RuntimeError(
                "DMI metadata expected model-parallel group size "
                f"{tp_size * pp_size}, got {len(mp_ranks)}"
            )
        mp_matrix = tuple(
            tuple(mp_ranks[pp_idx * tp_size : (pp_idx + 1) * tp_size])
            for pp_idx in range(pp_size)
        )
        for pp_idx, row in enumerate(mp_matrix):
            if len(row) != tp_size:
                raise RuntimeError("DMI metadata failed to reshape model-parallel ranks")

        pp_cpu_ranks = tuple(row[0] for row in mp_matrix)
        pp_cpu_group = None
        if len(pp_cpu_ranks) > 1:
            pp_cpu_group = parallel_state_module.create_group(
                list(pp_cpu_ranks),
                backend="gloo",
                group_desc=f"DMI_PP_METADATA_GLOO_{mp_index}",
            )
        if rank in pp_cpu_ranks:
            current_pp_cpu_ranks = pp_cpu_ranks
            current_pp_cpu_group = pp_cpu_group
            current_pp_source_rank = int(pp_cpu_ranks[0])

        for pp_idx, tp_cpu_ranks_raw in enumerate(mp_matrix):
            tp_cpu_ranks = tuple(int(item) for item in tp_cpu_ranks_raw)
            tp_cpu_group = None
            if len(tp_cpu_ranks) > 1:
                tp_cpu_group = parallel_state_module.create_group(
                    list(tp_cpu_ranks),
                    backend="gloo",
                    group_desc=f"DMI_TP_METADATA_GLOO_{mp_index}_{pp_idx}",
                )
            if rank in tp_cpu_ranks:
                current_tp_cpu_ranks = tp_cpu_ranks
                current_tp_cpu_group = tp_cpu_group
                current_tp_source_rank = int(tp_cpu_ranks[0])
                if current_pp_source_rank is None:
                    current_pp_source_rank = int(pp_cpu_ranks[0])
                if current_pp_cpu_ranks is None:
                    current_pp_cpu_ranks = pp_cpu_ranks

    if current_pp_cpu_ranks is None or current_tp_cpu_ranks is None:
        raise RuntimeError(f"DMI metadata could not find rank {rank} in generated CPU groups")
    if current_pp_source_rank is None or current_tp_source_rank is None:
        raise RuntimeError(f"DMI metadata could not find source ranks for rank {rank}")

    tp_rank = int(parallel_state_module.get_tensor_model_parallel_rank())
    pp_rank = int(parallel_state_module.get_pipeline_model_parallel_rank())
    if rank != int(current_tp_cpu_ranks[tp_rank]):
        raise RuntimeError(
            "DMI metadata rank derivation disagrees with Megatron TP rank: "
            f"rank={rank}, tp_rank={tp_rank}, tp_cpu_ranks={current_tp_cpu_ranks}"
        )
    expected_tp_source = int(current_tp_cpu_ranks[0])
    if current_tp_source_rank != expected_tp_source:
        raise RuntimeError("DMI metadata TP source rank mismatch")
    if pp_rank < 0 or pp_rank >= len(current_pp_cpu_ranks):
        raise RuntimeError(
            "DMI metadata PP rank outside derived PP CPU ranks: "
            f"pp_rank={pp_rank}, pp_cpu_ranks={current_pp_cpu_ranks}"
        )

    return DMICPUMetadataGroups(
        pp_source_rank=int(current_pp_source_rank),
        tp_source_rank=int(current_tp_source_rank),
        pp_cpu_ranks=current_pp_cpu_ranks,
        tp_cpu_ranks=current_tp_cpu_ranks,
        pp_cpu_group=current_pp_cpu_group,
        tp_cpu_group=current_tp_cpu_group,
    )


def build_megatron_schedule_runtime(
    *,
    max_num_microbatches: int,
    max_batch_size: int,
    num_scopes: int | None = None,
    device: torch.device | str | int | None = None,
    parallel_state_module: Any | None = None,
    dist_module: Any | None = None,
    tensor_model_parallel_size: int | None = None,
    pipeline_model_parallel_size: int | None = None,
    data_parallel_size: int | None = None,
    context_parallel_size: int = 1,
    expert_model_parallel_size: int = 1,
    rank_order: str = "tp-cp-ep-dp-pp",
    field_specs: tuple[DMIMetadataFieldSpec, ...] | list[DMIMetadataFieldSpec] | None = None,
    host_engine: Any | None = None,
) -> MegatronScheduleRuntime:
    """Build a Megatron schedule runtime from Megatron parallel-state APIs."""

    if parallel_state_module is None:
        from megatron.core import parallel_state as parallel_state_module
    dist = dist_module if dist_module is not None else torch.distributed

    if num_scopes is None:
        vp_world = parallel_state_module.get_virtual_pipeline_model_parallel_world_size()
        num_scopes = 1 if vp_world is None else int(vp_world)

    context = DMIMetadataContext(
        max_num_microbatches=max_num_microbatches,
        max_batch_size=max_batch_size,
        num_scopes=num_scopes,
        field_specs=field_specs,
        device=device,
    )

    tp_world = int(
        tensor_model_parallel_size
        if tensor_model_parallel_size is not None
        else parallel_state_module.get_tensor_model_parallel_world_size()
    )
    pp_world = int(
        pipeline_model_parallel_size
        if pipeline_model_parallel_size is not None
        else parallel_state_module.get_pipeline_model_parallel_world_size()
    )
    dist_ready = hasattr(dist, "is_initialized") and dist.is_initialized()
    if (tp_world == 1 and pp_world == 1) or not dist_ready:
        return MegatronScheduleRuntime(
            LocalMetadataPropagator(context),
            host_engine=host_engine,
        )

    world_size = int(dist.get_world_size()) if hasattr(dist, "get_world_size") else None
    if data_parallel_size is None:
        if world_size is None:
            raise RuntimeError("DMI metadata group builder requires data_parallel_size")
        model_parallel_size = tp_world * pp_world
        if world_size % model_parallel_size != 0:
            raise RuntimeError(
                "DMI metadata group builder cannot infer data_parallel_size: "
                f"world_size={world_size}, model_parallel_size={model_parallel_size}"
            )
        data_parallel_size = world_size // model_parallel_size

    groups = _build_dmi_cpu_metadata_groups(
        parallel_state_module=parallel_state_module,
        dist=dist,
        tensor_model_parallel_size=tp_world,
        pipeline_model_parallel_size=pp_world,
        data_parallel_size=int(data_parallel_size),
        context_parallel_size=int(context_parallel_size),
        expert_model_parallel_size=int(expert_model_parallel_size),
        rank_order=str(rank_order),
    )
    rank = int(dist.get_rank())
    propagator = PerDPCPUMetadataPropagator(
        context,
        rank=rank,
        pp_source_rank=groups.pp_source_rank,
        tp_source_rank=groups.tp_source_rank,
        pp_cpu_ranks=groups.pp_cpu_ranks,
        tp_cpu_ranks=groups.tp_cpu_ranks,
        pp_cpu_group=groups.pp_cpu_group,
        tp_cpu_group=groups.tp_cpu_group,
        dist_module=dist,
    )
    return MegatronScheduleRuntime(propagator, host_engine=host_engine)


def dmi_is_enabled() -> bool:
    return _active_runtime is not None


def dmi_guard_schedule_supported(config: Any, forward_only: bool) -> None:
    if not dmi_is_enabled():
        return
    del forward_only
    if getattr(config, "overlap_moe_expert_parallel_comm", False):
        raise NotImplementedError("DMI does not support combined MoE-overlap schedules yet")
    if getattr(config, "hybrid_context_parallel", False):
        raise NotImplementedError("DMI does not support hybrid context parallel schedules yet")
    if int(getattr(config, "context_parallel_size", 1)) > 1:
        raise NotImplementedError("DMI does not support context parallel schedules yet")


def dmi_begin_iteration(active_num_microbatches: int, *, forward_only: bool = False) -> None:
    if _active_runtime is not None:
        _active_runtime.begin_iteration(active_num_microbatches, forward_only=forward_only)


def dmi_begin_logical_iteration(global_batch_id: int) -> None:
    if _active_runtime is not None:
        _active_runtime.begin_logical_iteration(global_batch_id)


def dmi_finish_logical_iteration() -> None:
    if _active_runtime is not None:
        _active_runtime.finish_logical_iteration()


def dmi_begin_attempt(attempt_id: int) -> None:
    if _active_runtime is not None:
        _active_runtime.begin_attempt(attempt_id)


def dmi_finish_attempt(status: int) -> None:
    if _active_runtime is not None:
        _active_runtime.finish_attempt(status)


def dmi_enter_phase(
    phase: str,
    *,
    training_iteration_id_start: int,
    training_iteration_id_end: int | None = None,
    global_batch_id_start: int = 1,
    eval_index: int = 0,
) -> None:
    if _active_runtime is not None:
        _active_runtime.enter_phase(
            phase,
            training_iteration_id_start=training_iteration_id_start,
            training_iteration_id_end=training_iteration_id_end,
            global_batch_id_start=global_batch_id_start,
            eval_index=eval_index,
        )


def dmi_seal_current_phase() -> None:
    if _active_runtime is not None:
        _active_runtime.seal_current_phase()


def dmi_set_dataset_id_override(dataset_id: int | None) -> None:
    if _active_runtime is not None:
        _active_runtime.set_dataset_id_override(dataset_id)


def dmi_set_current_event(direction: str, microbatch_id: int, scope_id: int = 0) -> None:
    if _active_runtime is not None:
        _active_runtime.set_event(direction, microbatch_id, scope_id)


def dmi_record_current_microbatch_metadata(
    valid_count: torch.Tensor | None,
    valid_count_cpu: torch.Tensor | list[int] | tuple[int, ...] | int | None = None,
    dataset_id_cpu: torch.Tensor | list[int] | tuple[int, ...] | int | None = None,
    segment_metadata: torch.Tensor | None = None,
    segment_metadata_cpu: (
        torch.Tensor | list[int] | tuple[int, ...] | int | None
    ) = None,
) -> None:
    if _active_runtime is not None:
        _active_runtime.record_current_microbatch_metadata(
            valid_count,
            valid_count_cpu,
            dataset_id_cpu,
            segment_metadata,
            segment_metadata_cpu,
        )


def dmi_enter_current_scope() -> None:
    if _active_runtime is not None:
        _active_runtime.enter_current_scope()


def dmi_end_iteration() -> None:
    if _active_runtime is not None:
        _active_runtime.end_iteration()


def dmi_begin_cuda_graph_capture(
    *,
    warmup_enabled: bool = True,
    full_iteration: bool = False,
    valid_counts_by_microbatch: list[list[int] | None] | None = None,
    dataset_ids_by_microbatch: list[list[int] | None] | None = None,
    capture_direction: HookPhase | str | None = None,
) -> None:
    if _active_runtime is None or _active_runtime.adaptor is None:
        return
    if full_iteration:
        _active_runtime.begin_full_iteration_capture(
            valid_counts_by_microbatch,
            dataset_ids_by_microbatch,
        )
    _active_runtime.adaptor.begin_capture_plan(
        warmup_enabled=warmup_enabled,
        capture_event_context=full_iteration,
        capture_direction=capture_direction,
    )


def dmi_finish_cuda_graph_capture():
    if _active_runtime is None or _active_runtime.adaptor is None:
        return None
    try:
        return _active_runtime.adaptor.finish_capture_plan()
    finally:
        _active_runtime.finish_full_iteration_capture()


def dmi_abort_cuda_graph_capture() -> None:
    if _active_runtime is None or _active_runtime.adaptor is None:
        return
    try:
        _active_runtime.adaptor.abort_capture_plan()
    finally:
        _active_runtime.abort_full_iteration_capture()


def dmi_begin_te_capture_session() -> bool:
    """Enter the outer TE capture session before TE starts any CUDA capture."""

    if _active_runtime is None or _active_runtime.adaptor is None:
        return False
    _active_runtime.adaptor.begin_te_capture_session()
    return True


def dmi_finish_te_capture_session() -> None:
    if _active_runtime is None or _active_runtime.adaptor is None:
        raise RuntimeError("Megatron DMI TE capture session lost its active adaptor")
    _active_runtime.adaptor.finish_te_capture_session()


def dmi_is_te_capture_session_active() -> bool:
    return bool(
        _active_runtime is not None
        and _active_runtime.adaptor is not None
        and _active_runtime.adaptor.te_capture_session_active
    )


def dmi_begin_te_forward_capture() -> None:
    if _active_runtime is None or _active_runtime.adaptor is None:
        raise RuntimeError("Megatron DMI TE forward capture requires an active adaptor")
    _active_runtime.adaptor.begin_te_forward_capture()


def dmi_finish_te_forward_capture() -> ProducerPlan:
    if _active_runtime is None or _active_runtime.adaptor is None:
        raise RuntimeError("Megatron DMI TE forward capture requires an active adaptor")
    return _active_runtime.adaptor.finish_te_forward_capture()


def dmi_abort_te_forward_capture() -> None:
    if _active_runtime is None or _active_runtime.adaptor is None:
        raise RuntimeError("Megatron DMI TE forward capture requires an active adaptor")
    _active_runtime.adaptor.abort_te_forward_capture()


def dmi_prepare_te_forward_replay(plan: Any) -> bool:
    """Prepare one TE forward graph and return whether its covered unit must run eagerly."""

    if _active_runtime is None or _active_runtime.adaptor is None:
        return False
    if not isinstance(plan, ProducerPlan):
        raise TypeError("Megatron DMI TE replay requires a ProducerPlan")
    decision = _active_runtime.adaptor.prepare_replay(
        plan,
        plan_direction=HookPhase.FWD,
        live_direction=_dmi_require_live_direction(),
    )
    return decision is StepReservation.OVERSIZED


def _dmi_require_local_plan(runner: Any, name: str):
    if not hasattr(runner, name):
        raise RuntimeError(f"Megatron DMI local CUDA graph runner is missing {name}")
    return getattr(runner, name)


def _dmi_require_live_direction() -> HookPhase:
    if _active_runtime is None or _active_runtime.adaptor is None:
        raise RuntimeError("Megatron DMI local replay requires an active adaptor")
    ctx = _active_runtime.adaptor.current_context
    if ctx is None:
        raise RuntimeError("Megatron DMI local replay requires an active schedule event")
    return _active_runtime.adaptor.normalize_direction(ctx.direction)


def dmi_prepare_local_forward_boundary(runner: Any) -> bool:
    """Return True when the whole local fwd/bwd pair must fall back to eager."""

    if _active_runtime is None or _active_runtime.adaptor is None:
        return False

    live_direction = _dmi_require_live_direction()
    fwd_plan = _dmi_require_local_plan(runner, "_dmi_fwd_plan")
    fwd_decision = _active_runtime.adaptor.prepare_replay_capacity_only(
        fwd_plan,
        plan_direction=HookPhase.FWD,
        live_direction=live_direction,
    )
    if fwd_decision is StepReservation.OVERSIZED:
        runner._dmi_next_bwd_decision = None
        return True

    bwd_decision = None
    if getattr(runner, "grad_enabled", False):
        bwd_plan = _dmi_require_local_plan(runner, "_dmi_bwd_plan")
        bwd_decision = _active_runtime.adaptor.prepare_replay_capacity_only(
            bwd_plan,
            plan_direction=HookPhase.BWD,
            live_direction=HookPhase.BWD,
        )
        if bwd_decision is StepReservation.OVERSIZED:
            runner._dmi_next_bwd_decision = None
            return True

    runner._dmi_next_bwd_decision = bwd_decision
    return False


def dmi_prepare_local_forward_replay(runner: Any) -> None:
    if _active_runtime is None or _active_runtime.adaptor is None:
        return
    _active_runtime.adaptor.prepare_replay(
        _dmi_require_local_plan(runner, "_dmi_fwd_plan"),
        plan_direction=HookPhase.FWD,
        live_direction=_dmi_require_live_direction(),
    )


def dmi_take_local_backward_token(runner: Any):
    token = getattr(runner, "_dmi_next_bwd_decision", None)
    runner._dmi_next_bwd_decision = None
    return token


def dmi_prepare_local_backward_replay(runner: Any, token: Any) -> None:
    if _active_runtime is None or _active_runtime.adaptor is None:
        return
    if getattr(runner, "grad_enabled", False) and token is None:
        raise RuntimeError("Megatron DMI local backward replay is missing replay token")
    _active_runtime.adaptor.prepare_replay(
        _dmi_require_local_plan(runner, "_dmi_bwd_plan"),
        plan_direction=HookPhase.BWD,
        live_direction=_dmi_require_live_direction(),
    )


def dmi_prepare_full_iteration_replay(
    plan: Any,
    valid_counts_by_microbatch: list[list[int] | None],
    dataset_ids_by_microbatch: list[list[int] | None] | None = None,
) -> bool:
    """Return True when the full-iteration unit must fall back to eager."""

    if _active_runtime is None or _active_runtime.adaptor is None:
        return False
    if not isinstance(plan, MegatronFullIterationPlan):
        raise TypeError("Megatron DMI full-iteration replay requires MegatronFullIterationPlan")
    _active_runtime.load_full_iteration_metadata(
        valid_counts_by_microbatch,
        dataset_ids_by_microbatch=dataset_ids_by_microbatch,
    )
    contexts = _active_runtime.full_iteration_contexts(plan, valid_counts_by_microbatch)
    decision = _active_runtime.adaptor.prepare_full_iteration_replay(plan, contexts)
    return decision is StepReservation.OVERSIZED


def dmi_finish_full_iteration_replay() -> None:
    if _active_runtime is not None:
        _active_runtime.finish_full_iteration_replay()


def dmi_current_phase(default: str = "validation") -> str:
    if _active_runtime is None:
        return str(default)
    return str(_active_runtime.phase)


@contextmanager
def dmi_force_eager_unit():
    """Retain the fork fallback boundary for the public record runtime.

    Public ``RecordRuntime.emit_output`` owns eager reservation and CPU-direct
    fallback per hook, so there is no transport-wide force-eager state to set.
    """

    yield


__all__ = [
    "DMIScheduleEvent",
    "MegatronScheduleRuntime",
    "build_megatron_schedule_runtime",
    "dmi_begin_iteration",
    "dmi_begin_attempt",
    "dmi_begin_logical_iteration",
    "dmi_begin_cuda_graph_capture",
    "dmi_abort_cuda_graph_capture",
    "dmi_abort_te_forward_capture",
    "dmi_begin_te_capture_session",
    "dmi_begin_te_forward_capture",
    "dmi_end_iteration",
    "dmi_enter_current_scope",
    "dmi_enter_phase",
    "dmi_finish_cuda_graph_capture",
    "dmi_finish_te_capture_session",
    "dmi_finish_te_forward_capture",
    "dmi_finish_attempt",
    "dmi_finish_logical_iteration",
    "dmi_current_phase",
    "dmi_force_eager_unit",
    "dmi_finish_full_iteration_replay",
    "dmi_guard_schedule_supported",
    "dmi_is_enabled",
    "dmi_is_te_capture_session_active",
    "dmi_prepare_full_iteration_replay",
    "dmi_prepare_local_backward_replay",
    "dmi_prepare_local_forward_boundary",
    "dmi_prepare_local_forward_replay",
    "dmi_prepare_te_forward_replay",
    "dmi_record_current_microbatch_metadata",
    "dmi_seal_current_phase",
    "dmi_set_dataset_id_override",
    "dmi_set_current_event",
    "dmi_take_local_backward_token",
    "get_active_megatron_schedule_runtime",
    "set_active_megatron_schedule_runtime",
]
