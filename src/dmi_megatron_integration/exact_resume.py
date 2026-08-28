"""Exact checkpoint coordination for the dynamic Megatron SFT path."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import random
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable, Mapping

import numpy as np
import torch

from .dynamic_mixture import get_active_dynamic_train_dataset


_RESUME_STATE_VERSION = 2
_DETERMINISTIC_ENVIRONMENT = {
    "CUDA_DEVICE_MAX_CONNECTIONS": "1",
    "NVTE_ALLOW_NONDETERMINISTIC_ALGO": "0",
    "NCCL_ALGO": "Ring",
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
}
_EXECUTION_CONTRACT = {
    "torch_jit_profiling_executor": False,
    "torch_jit_profiling_mode": False,
}
_active_manager: "DMIExactResumeManager | None" = None


def configure_dmi_exact_execution(args: Any) -> None:
    if not bool(getattr(args, "dmi_exact_resume", False)):
        return
    torch._C._jit_set_profiling_executor(False)
    torch._C._jit_set_profiling_mode(False)
    args._dmi_exact_execution_contract = copy.deepcopy(_EXECUTION_CONTRACT)


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _distributed() -> bool:
    return torch.distributed.is_available() and torch.distributed.is_initialized()


def _rank() -> int:
    return torch.distributed.get_rank() if _distributed() else 0


def _world_size() -> int:
    return torch.distributed.get_world_size() if _distributed() else 1


def _all_gather_object(value: object) -> list[object]:
    if not _distributed():
        return [value]
    values: list[object] = [None] * _world_size()
    torch.distributed.all_gather_object(values, value)
    return values


def _broadcast_from_rank_zero(value: object) -> object:
    if not _distributed():
        return value
    values = [value if _rank() == 0 else None]
    torch.distributed.broadcast_object_list(values, src=0)
    return values[0]


def _raise_collective_failures(stage: str, local_error: BaseException | None) -> None:
    local = None
    if local_error is not None:
        local = {
            "rank": _rank(),
            "type": type(local_error).__name__,
            "message": str(local_error),
        }
    failures = [item for item in _all_gather_object(local) if item is not None]
    if failures:
        raise RuntimeError(f"DMI exact checkpoint {stage} failed: {failures}")


def _run_collective_local_stage(stage: str, operation: Callable[[], Any]) -> Any:
    result = None
    error = None
    try:
        result = operation()
    except BaseException as exc:
        error = exc
    _raise_collective_failures(stage, error)
    return result


def _post_json(url: str, path: str, body: Mapping[str, object], timeout_s: float) -> dict:
    request = urllib.request.Request(
        f"{url.rstrip('/')}{path}",
        data=_canonical_bytes(dict(body)),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=float(timeout_s) + 5.0) as response:
            payload = json.loads(response.read().decode("utf-8"))
            if response.status != 200:
                raise RuntimeError(
                    f"controller returned HTTP {response.status}: {payload!r}"
                )
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"controller checkpoint request returned HTTP {exc.code}: {detail}"
        ) from exc
    if not isinstance(payload, dict):
        raise TypeError("controller checkpoint response must be a JSON object")
    return payload


def _validate_lineage(lineage: object, *, checkpoint_iteration: int) -> list[dict]:
    if not isinstance(lineage, list) or not lineage:
        raise ValueError("DMI resume lineage must be a nonempty list")
    normalized: list[dict] = []
    expected_start = 1
    run_ids: set[str] = set()
    model_ids: set[str] = set()
    for raw in lineage:
        if not isinstance(raw, Mapping):
            raise TypeError("DMI resume lineage entries must be mappings")
        expected = {
            "run_id",
            "model_id",
            "valid_start_iteration",
            "valid_end_iteration",
        }
        if set(raw) != expected:
            raise ValueError("DMI resume lineage entry fields mismatch")
        entry = {
            "run_id": str(raw["run_id"]),
            "model_id": str(raw["model_id"]),
            "valid_start_iteration": int(raw["valid_start_iteration"]),
            "valid_end_iteration": int(raw["valid_end_iteration"]),
        }
        if not entry["run_id"] or not entry["model_id"]:
            raise ValueError("DMI resume lineage identities must be nonempty")
        if entry["run_id"] in run_ids or entry["model_id"] in model_ids:
            raise ValueError("DMI resume lineage identities must be unique")
        if entry["valid_start_iteration"] != expected_start:
            raise ValueError("DMI resume lineage ranges are not contiguous")
        if entry["valid_end_iteration"] < entry["valid_start_iteration"]:
            raise ValueError("DMI resume lineage contains an empty range")
        run_ids.add(entry["run_id"])
        model_ids.add(entry["model_id"])
        expected_start = entry["valid_end_iteration"] + 1
        normalized.append(entry)
    if normalized[-1]["valid_end_iteration"] != int(checkpoint_iteration):
        raise ValueError("DMI resume lineage does not end at the checkpoint")
    return normalized


def validate_loaded_resume_state(
    args: Any,
    state: object,
    *,
    checkpoint_iteration: int,
    release: bool,
) -> dict:
    if not isinstance(state, Mapping):
        raise TypeError("DMI exact checkpoint is missing a resume-state mapping")
    expected = {
        "schema_version",
        "checkpoint_iteration",
        "consumed_train_samples",
        "consumed_valid_samples",
        "execution_contract",
        "segment_lineage",
        "database_identity",
        "dataset_contract",
        "dynamic_dataset_state",
        "dynamic_dataset_state_sha256",
        "controller_state",
        "controller_state_sha256",
        "capture_state",
        "durable_flush_state",
    }
    if set(state) != expected:
        missing = sorted(expected - set(state))
        extra = sorted(set(state) - expected)
        raise ValueError(
            f"DMI resume-state fields mismatch: missing={missing}, extra={extra}"
        )
    if int(state["schema_version"]) != _RESUME_STATE_VERSION:
        raise ValueError(
            f"unsupported DMI resume-state version: {state['schema_version']!r}"
        )
    if state["execution_contract"] != _EXECUTION_CONTRACT:
        raise ValueError("DMI checkpoint execution contract mismatch")
    if getattr(args, "_dmi_exact_execution_contract", None) != _EXECUTION_CONTRACT:
        raise RuntimeError("DMI exact execution policy was not configured before load")
    checkpoint_iteration = int(checkpoint_iteration)
    if int(state["checkpoint_iteration"]) != checkpoint_iteration:
        raise ValueError("DMI and Megatron checkpoint iterations disagree")
    if release:
        raise RuntimeError("DMI exact resume rejects release checkpoints")
    if bool(getattr(args, "finetune", False)):
        raise RuntimeError("DMI exact resume rejects --finetune")
    if bool(getattr(args, "no_load_optim", False)):
        raise RuntimeError("DMI exact resume requires optimizer and scheduler state")
    if bool(getattr(args, "no_load_rng", False)):
        raise RuntimeError("DMI exact resume requires RNG state")
    if getattr(args, "pretrained_checkpoint", None) is not None:
        raise RuntimeError("DMI exact resume requires a native --load checkpoint")
    global_batch_size = int(getattr(args, "global_batch_size"))
    expected_consumed = checkpoint_iteration * global_batch_size
    if int(state["consumed_train_samples"]) != expected_consumed:
        raise ValueError(
            "DMI checkpoint consumed_train_samples is incompatible with fixed GBS"
        )
    _validate_lineage(
        state["segment_lineage"],
        checkpoint_iteration=checkpoint_iteration,
    )
    dynamic_state = state["dynamic_dataset_state"]
    controller_state = state["controller_state"]
    if _sha256(dynamic_state) != str(state["dynamic_dataset_state_sha256"]):
        raise ValueError("DMI dynamic-dataset checkpoint hash mismatch")
    if _sha256(controller_state) != str(state["controller_state_sha256"]):
        raise ValueError("DMI controller checkpoint hash mismatch")
    durable = state["durable_flush_state"]
    if not isinstance(durable, Mapping) or set(durable) != {
        "durable_through_iteration"
    }:
        raise ValueError("DMI durable-flush state is invalid")
    if int(durable["durable_through_iteration"]) != checkpoint_iteration:
        raise ValueError("DMI checkpoint was not durable through its iteration")
    return dict(state)


def install_loaded_resume_state(
    args: Any,
    state: object,
    *,
    checkpoint_iteration: int,
    release: bool,
) -> dict:
    validated = validate_loaded_resume_state(
        args,
        state,
        checkpoint_iteration=checkpoint_iteration,
        release=release,
    )
    args._dmi_loaded_resume_state = validated
    args._dmi_dynamic_dataset_resume_state = validated["dynamic_dataset_state"]
    return validated


def capture_dmi_exact_loaded_rng_state(args: Any) -> None:
    """Save checkpoint RNG before resume repeats DataLoader iterator setup.

    The checkpoint state already includes the original iterator's pre-training
    ``_base_seed`` draw. Reconstructing the iterator after checkpoint load
    repeats that pre-snapshot operation and advances the global Torch CPU RNG.
    """

    if not bool(getattr(args, "dmi_exact_resume", False)):
        return
    if getattr(args, "_dmi_loaded_resume_state", None) is None:
        return
    if hasattr(args, "_dmi_exact_loaded_rng_state"):
        raise RuntimeError("DMI exact-resume RNG state was captured more than once")

    from megatron.core import tensor_parallel

    tracker = tensor_parallel.get_cuda_rng_tracker()
    tracker_states = {
        name: tensor_parallel.convert_cuda_rng_state(
            value,
            to_graphable=False,
        ).clone()
        for name, value in tracker.get_states().items()
    }
    args._dmi_exact_loaded_rng_state = {
        "random_rng_state": random.getstate(),
        "np_rng_state": copy.deepcopy(np.random.get_state()),
        "torch_rng_state": torch.get_rng_state().clone(),
        "cuda_rng_state": torch.cuda.get_rng_state().clone(),
        "rng_tracker_states": tracker_states,
    }


def restore_dmi_exact_loaded_rng_state(args: Any) -> None:
    """Cancel the resume-only iterator draw before resumed training.

    Exact-resume setup enforces ``num_workers=0``, so the newly generated
    ``_base_seed`` is unused and is not persistent training state. With
    workers, the iterator would cache that random result for later worker
    seeding. Rewinding its source RNG while retaining the cached seed would
    create an inconsistent state that never occurs in uninterrupted execution.
    """

    if not bool(getattr(args, "dmi_exact_resume", False)):
        return
    if getattr(args, "_dmi_loaded_resume_state", None) is None:
        return
    state = getattr(args, "_dmi_exact_loaded_rng_state", None)
    if state is None:
        raise RuntimeError(
            "DMI exact resume lost Megatron's restored RNG state during setup"
        )

    from megatron.core import tensor_parallel

    random.setstate(state["random_rng_state"])
    np.random.set_state(state["np_rng_state"])
    torch.set_rng_state(state["torch_rng_state"])
    torch.cuda.set_rng_state(state["cuda_rng_state"])

    tracker = tensor_parallel.get_cuda_rng_tracker()
    graph_safe_rng = tensor_parallel.is_graph_safe_cuda_rng_tracker(tracker)
    tracker_states = {
        name: tensor_parallel.convert_cuda_rng_state(
            value,
            to_graphable=graph_safe_rng,
        )
        for name, value in state["rng_tracker_states"].items()
    }
    tracker.set_states(tracker_states)
    delattr(args, "_dmi_exact_loaded_rng_state")


@dataclass
class DMIExactResumeManager:
    args: Any
    handle: Any
    printer: Callable[[str], None]
    loaded_state: dict | None

    def __post_init__(self) -> None:
        self.timeout_s = float(
            getattr(self.args, "dmi_exact_checkpoint_timeout_s", 600.0)
        )
        if self.timeout_s <= 0.0:
            raise ValueError("DMI exact checkpoint timeout must be positive")
        self.run_id = str(getattr(self.args, "dmi_dynamic_mixture_run_id", "") or "")
        self.model_id = str(self.handle.model_id)
        self.controller_url = str(
            getattr(self.args, "dmi_dynamic_mixture_control_url", "") or ""
        )
        self._last_exact_checkpoint_iteration = self.parent_checkpoint_iteration
        self._last_exact_checkpoint_path = (
            None if self.loaded_state is None else getattr(self.args, "load", None)
        )
        self._validate_runtime_contract()
        if self.loaded_state is not None:
            self._restore_capture_state()

    @property
    def parent_checkpoint_iteration(self) -> int:
        if self.loaded_state is None:
            return 0
        return int(self.loaded_state["checkpoint_iteration"])

    def _validate_runtime_contract(self) -> None:
        if not bool(getattr(self.args, "dmi_enable", False)):
            raise RuntimeError("DMI exact resume requires --dmi-enable")
        if not bool(getattr(self.args, "deterministic_mode", False)):
            raise RuntimeError("DMI exact resume requires --deterministic-mode")
        for name, expected in _DETERMINISTIC_ENVIRONMENT.items():
            actual = os.environ.get(name)
            if actual != expected:
                raise RuntimeError(
                    f"DMI exact resume requires {name}={expected}, got {actual!r}"
                )
        if (
            getattr(self.args, "_dmi_exact_execution_contract", None)
            != _EXECUTION_CONTRACT
        ):
            raise RuntimeError("DMI exact execution policy is not active")
        if not self.run_id or not self.model_id or not self.controller_url:
            raise RuntimeError(
                "DMI exact resume requires dynamic-mixture run_id, model_id, and controller URL"
            )
        if str(getattr(self.args, "dataloader_type", "")) != "single":
            raise RuntimeError("DMI exact resume requires --dataloader-type single")
        if int(getattr(self.args, "data_parallel_size", 0)) != 1:
            raise RuntimeError("the first DMI exact-resume path requires DP=1")
        # Exact resume rewinds the RNG draw used to construct DataLoader
        # _base_seed. This is valid only when the seed is unused. With workers,
        # the iterator caches that result and later seeds workers from it, so
        # rewinding the source RNG would leave cached and source RNG states
        # inconsistent.
        if int(getattr(self.args, "num_workers", -1)) != 0:
            raise RuntimeError("DMI exact resume requires --num-workers 0")
        if getattr(self.args, "rampup_batch_size", None) is not None:
            raise RuntimeError("DMI exact resume does not support batch-size ramp-up")
        if bool(getattr(self.args, "async_save", False)):
            raise RuntimeError("DMI exact resume requires synchronous checkpoints")
        if not str(getattr(self.args, "dmi_db_host", "") or ""):
            raise RuntimeError("DMI exact resume requires a ClickHouse sink")
        if self.loaded_state is None:
            return
        lineage = _validate_lineage(
            self.loaded_state["segment_lineage"],
            checkpoint_iteration=self.parent_checkpoint_iteration,
        )
        if self.run_id in {entry["run_id"] for entry in lineage}:
            raise RuntimeError("child DMI run_id already appears in parent lineage")
        if self.model_id in {entry["model_id"] for entry in lineage}:
            raise RuntimeError("child DMI model_id already appears in parent lineage")
        runtime_database = self._database_identity()
        if runtime_database != self.loaded_state["database_identity"]:
            raise RuntimeError(
                "DMI resume database identity mismatch: "
                f"checkpoint={self.loaded_state['database_identity']}, "
                f"runtime={runtime_database}"
            )
        if int(getattr(self.args, "iteration", -1)) != self.parent_checkpoint_iteration:
            raise RuntimeError("Megatron did not load the DMI parent checkpoint iteration")
        if int(getattr(self.args, "consumed_train_samples", -1)) != int(
            self.loaded_state["consumed_train_samples"]
        ):
            raise RuntimeError("Megatron and DMI consumed-train coordinates disagree")

    def _database_identity(self) -> dict[str, object]:
        return {
            "raw_database": str(self.handle.config.db_database),
            "raw_table": str(self.handle.config.clickhouse_table),
            "processed_database": getattr(
                self.args, "dmi_exact_processed_database", None
            ),
            "control_database": getattr(
                self.args, "dmi_exact_control_database", None
            ),
        }

    def _restore_capture_state(self) -> None:
        capture = self.loaded_state["capture_state"]
        if not isinstance(capture, dict):
            raise TypeError("DMI checkpoint capture state must be a mapping")
        self.handle.schedule_runtime.load_exact_resume_state_dict(
            capture,
            checkpoint_iteration=self.parent_checkpoint_iteration,
        )

    def _remaining(self, deadline: float, stage: str) -> float:
        value = deadline - time.monotonic()
        if value <= 0.0:
            raise TimeoutError(f"DMI exact checkpoint timed out before {stage}")
        return value

    def _snapshot_dataset(self) -> tuple[dict, str]:
        local_state = None
        local_error = None
        try:
            dataset = get_active_dynamic_train_dataset()
            local_state = None if dataset is None else dataset.state_dict()
        except BaseException as exc:
            local_error = f"{type(exc).__name__}: {exc}"
        local_payload = {
            "rank": _rank(),
            "state": local_state,
            "sha256": None if local_state is None else _sha256(local_state),
            "error": local_error,
        }
        gathered = _all_gather_object(local_payload)
        failures = [item for item in gathered if item["error"] is not None]
        if failures:
            raise RuntimeError(
                f"dynamic-dataset checkpoint snapshot failed: {failures}"
            )
        owners = [item for item in gathered if item["state"] is not None]
        if not owners:
            raise RuntimeError("no rank owns an active dynamic training dataset")
        hashes = {str(item["sha256"]) for item in owners}
        if len(hashes) != 1:
            raise RuntimeError(
                f"dynamic-dataset checkpoint states disagree across ranks: {gathered}"
            )
        canonical = owners[0]["state"]
        canonical_hash = owners[0]["sha256"]
        if not isinstance(canonical, dict):
            raise TypeError("canonical dynamic-dataset state must be a mapping")
        return canonical, str(canonical_hash)

    def validate_restored_dataset(self) -> None:
        if self.loaded_state is None:
            return
        state, state_hash = self._snapshot_dataset()
        expected_hash = str(self.loaded_state["dynamic_dataset_state_sha256"])
        if state_hash != expected_hash:
            raise RuntimeError(
                "restored dynamic-dataset state differs from its checkpoint: "
                f"runtime={state_hash}, checkpoint={expected_hash}"
            )
        self._validate_dataset_position(
            state,
            checkpoint_iteration=self.parent_checkpoint_iteration,
        )
        self.printer(
            "[DMI] restored canonical dynamic-dataset state "
            f"sha256={state_hash}"
        )

    def _validate_dataset_position(
        self,
        dataset_state: Mapping[str, object],
        *,
        checkpoint_iteration: int,
    ) -> None:
        configuration = dataset_state["configuration"]
        if not isinstance(configuration, Mapping):
            raise TypeError("dynamic-dataset configuration must be a mapping")
        if int(configuration["global_batch_size"]) != int(
            self.args.global_batch_size
        ):
            raise RuntimeError("dynamic-dataset and runtime global batch sizes disagree")
        samples_per_window = int(configuration["samples_per_window"])
        window_id = int(dataset_state["current_window_id"])
        window_start = (window_id - 1) * samples_per_window
        window_end = window_id * samples_per_window
        next_index = int(checkpoint_iteration) * int(self.args.global_batch_size)
        if not (window_start <= next_index <= window_end):
            raise RuntimeError(
                "restored selection window does not contain or immediately precede "
                f"the next absolute sample: window=[{window_start}, {window_end}), "
                f"next={next_index}"
            )

    def _snapshot_controller(
        self,
        *,
        checkpoint_iteration: int,
        installed_window_id: int,
        deadline: float,
    ) -> tuple[dict, str]:
        envelope = None
        if _rank() == 0:
            try:
                response = _post_json(
                    self.controller_url,
                    "/v1/checkpoint",
                    {
                        "checkpoint_iteration": int(checkpoint_iteration),
                        "installed_window_id": int(installed_window_id),
                        "timeout_s": self._remaining(
                            deadline, "controller checkpoint snapshot"
                        ),
                    },
                    self._remaining(deadline, "controller checkpoint request"),
                )
                state = response["state"]
                reported_hash = str(response["sha256"])
                if _sha256(state) != reported_hash:
                    raise RuntimeError("controller checkpoint response hash mismatch")
                envelope = {
                    "ok": True,
                    "state": state,
                    "sha256": reported_hash,
                }
            except BaseException as exc:
                envelope = {
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
        envelope = _broadcast_from_rank_zero(envelope)
        if not isinstance(envelope, dict) or not envelope.get("ok"):
            raise RuntimeError(
                "DMI controller snapshot failed: "
                f"{None if not isinstance(envelope, dict) else envelope.get('error')}"
            )
        state = envelope["state"]
        if not isinstance(state, dict):
            raise TypeError("controller checkpoint state must be a mapping")
        return state, str(envelope["sha256"])

    @staticmethod
    def _validate_dataset_controller_contract(
        dataset_state: Mapping[str, object],
        controller_state: Mapping[str, object],
    ) -> None:
        installed_window_id = int(dataset_state["current_window_id"])
        if int(controller_state["installed_window_id"]) != installed_window_id:
            raise RuntimeError("dataset and controller installed-window IDs disagree")
        dataset_weights = tuple(float(value) for value in dataset_state["weights"])
        temporal = controller_state["controller"]
        if not isinstance(temporal, Mapping):
            raise TypeError("controller temporal state must be a mapping")
        controller_weights = tuple(
            float(value) for value in temporal["previous_weights"]
        )
        store = controller_state["decision_store"]
        if not isinstance(store, Mapping):
            raise TypeError("controller decision-store state must be a mapping")
        pending = store["pending_decision"]
        if pending is None:
            if controller_weights != dataset_weights:
                raise RuntimeError(
                    "controller weights do not match the installed dataset window"
                )
            return
        if not isinstance(pending, Mapping):
            raise TypeError("pending controller decision must be a mapping")
        if int(pending["source_window_id"]) != installed_window_id:
            raise RuntimeError("pending decision source window is not installed")
        if int(pending["effective_window_id"]) != installed_window_id + 1:
            raise RuntimeError("pending decision does not target the immediate next window")
        pending_weights = tuple(float(value) for value in pending["weights"])
        if controller_weights != pending_weights:
            raise RuntimeError("controller weights do not match its pending decision")

    def _build_lineage(self, checkpoint_iteration: int) -> list[dict]:
        lineage = (
            []
            if self.loaded_state is None
            else [dict(item) for item in self.loaded_state["segment_lineage"]]
        )
        start = self.parent_checkpoint_iteration + 1
        if checkpoint_iteration < start:
            raise RuntimeError(
                "DMI exact checkpoint has no completed child iteration to commit"
            )
        lineage.append(
            {
                "run_id": self.run_id,
                "model_id": self.model_id,
                "valid_start_iteration": start,
                "valid_end_iteration": int(checkpoint_iteration),
            }
        )
        return _validate_lineage(
            lineage,
            checkpoint_iteration=checkpoint_iteration,
        )

    def prepare_checkpoint(self, checkpoint_iteration: int) -> dict:
        checkpoint_iteration = int(checkpoint_iteration)
        deadline = time.monotonic() + self.timeout_s
        _run_collective_local_stage(
            "runtime validation",
            lambda: self._validate_checkpoint_coordinates(checkpoint_iteration),
        )
        _run_collective_local_stage(
            "durable flush",
            lambda: self.handle.flush_and_wait(
                self._remaining(deadline, "durable flush")
            ),
        )
        dataset_state, dataset_hash = self._snapshot_dataset()
        self._validate_dataset_position(
            dataset_state,
            checkpoint_iteration=checkpoint_iteration,
        )
        installed_window_id = int(dataset_state["current_window_id"])
        controller_state, controller_hash = self._snapshot_controller(
            checkpoint_iteration=checkpoint_iteration,
            installed_window_id=installed_window_id,
            deadline=deadline,
        )
        self._validate_dataset_controller_contract(dataset_state, controller_state)
        capture_state = _run_collective_local_stage(
            "capture-state snapshot",
            lambda: self.handle.schedule_runtime.exact_resume_state_dict(
                checkpoint_iteration=checkpoint_iteration
            ),
        )
        capture_hashes = set(_all_gather_object(_sha256(capture_state)))
        if len(capture_hashes) != 1:
            raise RuntimeError("DMI capture state differs across distributed ranks")
        state = {
            "schema_version": _RESUME_STATE_VERSION,
            "checkpoint_iteration": checkpoint_iteration,
            "consumed_train_samples": int(self.args.consumed_train_samples),
            "consumed_valid_samples": int(
                getattr(self.args, "consumed_valid_samples", 0)
            ),
            "execution_contract": copy.deepcopy(_EXECUTION_CONTRACT),
            "segment_lineage": self._build_lineage(checkpoint_iteration),
            "database_identity": self._database_identity(),
            "dataset_contract": {
                "configuration": dataset_state["configuration"],
                "source_contract": dataset_state["source_contract"],
            },
            "dynamic_dataset_state": dataset_state,
            "dynamic_dataset_state_sha256": dataset_hash,
            "controller_state": controller_state,
            "controller_state_sha256": controller_hash,
            "capture_state": capture_state,
            "durable_flush_state": {
                "durable_through_iteration": checkpoint_iteration,
            },
        }
        state_hashes = set(_all_gather_object(_sha256(state)))
        if len(state_hashes) != 1:
            raise RuntimeError("DMI resume state differs across distributed ranks")
        self.printer(
            "[DMI] exact checkpoint prepared through iteration "
            f"{checkpoint_iteration}; state_sha256={next(iter(state_hashes))}"
        )
        return state

    def _validate_checkpoint_coordinates(self, checkpoint_iteration: int) -> None:
        if int(self.args.consumed_train_samples) != (
            checkpoint_iteration * int(self.args.global_batch_size)
        ):
            raise RuntimeError(
                "DMI exact checkpoint requires consumed_train_samples == "
                "iteration * global_batch_size"
            )
        if int(self.args.global_batch_size) <= 0:
            raise RuntimeError("DMI exact checkpoint requires a positive global batch size")
        if self.loaded_state is not None and checkpoint_iteration <= (
            self.parent_checkpoint_iteration
        ):
            raise RuntimeError("DMI exact checkpoint did not advance beyond its parent")

    def salvage_metadata(
        self,
        *,
        checkpoint_iteration: int,
        failure: BaseException,
    ) -> dict[str, object]:
        return {
            "checkpoint_kind": "model_salvage",
            "dmi_exact_resume": False,
            "dmi_failure_reason": f"{type(failure).__name__}: {failure}",
            "failed_exact_checkpoint_iteration": int(checkpoint_iteration),
            "last_exact_checkpoint_path": self._last_exact_checkpoint_path,
            "last_exact_checkpoint_iteration": self._last_exact_checkpoint_iteration,
            "parent_run_id": (
                None
                if self.loaded_state is None
                else self.loaded_state["segment_lineage"][-1]["run_id"]
            ),
            "parent_model_id": (
                None
                if self.loaded_state is None
                else self.loaded_state["segment_lineage"][-1]["model_id"]
            ),
        }

    def mark_checkpoint_committed(self, checkpoint_iteration: int, save_path: str) -> None:
        checkpoint_iteration = int(checkpoint_iteration)
        if checkpoint_iteration <= self._last_exact_checkpoint_iteration:
            raise RuntimeError("DMI exact checkpoint commit did not advance")
        self._last_exact_checkpoint_iteration = checkpoint_iteration
        self._last_exact_checkpoint_path = str(save_path)


def setup_dmi_exact_resume(
    args: Any,
    handle: Any,
    *,
    printer: Callable[[str], None] = print,
) -> DMIExactResumeManager | None:
    global _active_manager
    if not bool(getattr(args, "dmi_exact_resume", False)):
        return None
    if _active_manager is not None:
        raise RuntimeError("DMI exact-resume manager is already active")
    loaded = getattr(args, "_dmi_loaded_resume_state", None)
    manager = DMIExactResumeManager(
        args=args,
        handle=handle,
        printer=printer,
        loaded_state=loaded,
    )
    _active_manager = manager
    return manager


def prepare_dmi_exact_checkpoint(args: Any, checkpoint_iteration: int) -> dict | None:
    if not bool(getattr(args, "dmi_exact_resume", False)):
        return None
    if _active_manager is None:
        raise RuntimeError("DMI exact checkpoint requested before manager setup")
    return _active_manager.prepare_checkpoint(checkpoint_iteration)


def validate_dmi_exact_dataset_restore(args: Any) -> None:
    if not bool(getattr(args, "dmi_exact_resume", False)):
        return
    if _active_manager is None:
        raise RuntimeError("DMI exact dataset validation requires an active manager")
    _active_manager.validate_restored_dataset()


def build_dmi_salvage_metadata(
    args: Any,
    *,
    checkpoint_iteration: int,
    failure: BaseException,
) -> dict[str, object]:
    if _active_manager is None:
        return {
            "checkpoint_kind": "model_salvage",
            "dmi_exact_resume": False,
            "dmi_failure_reason": f"{type(failure).__name__}: {failure}",
            "failed_exact_checkpoint_iteration": int(checkpoint_iteration),
            "last_exact_checkpoint_path": getattr(args, "load", None),
            "last_exact_checkpoint_iteration": int(getattr(args, "iteration", 0)),
            "parent_run_id": None,
            "parent_model_id": None,
        }
    return _active_manager.salvage_metadata(
        checkpoint_iteration=checkpoint_iteration,
        failure=failure,
    )


def mark_dmi_exact_checkpoint_committed(
    args: Any,
    *,
    checkpoint_iteration: int,
    save_path: str,
) -> None:
    if not bool(getattr(args, "dmi_exact_resume", False)):
        return
    if _active_manager is None:
        raise RuntimeError("DMI exact checkpoint committed without an active manager")
    _active_manager.mark_checkpoint_committed(checkpoint_iteration, save_path)


def clear_active_dmi_exact_resume() -> None:
    global _active_manager
    _active_manager = None


__all__ = [
    "build_dmi_salvage_metadata",
    "capture_dmi_exact_loaded_rng_state",
    "clear_active_dmi_exact_resume",
    "configure_dmi_exact_execution",
    "install_loaded_resume_state",
    "mark_dmi_exact_checkpoint_committed",
    "prepare_dmi_exact_checkpoint",
    "restore_dmi_exact_loaded_rng_state",
    "setup_dmi_exact_resume",
    "validate_dmi_exact_dataset_restore",
    "validate_loaded_resume_state",
]
