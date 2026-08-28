"""Opt-in runtime-windowed dataset blend for Megatron SFT."""

from __future__ import annotations

import atexit
import hashlib
import json
import os
import queue
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import torch

from .dynamic_mixture_primitives import (
    DecisionHTTPClient,
    DecisionPrefetcher,
    MixtureDecision,
    SelectionWindow,
    WindowSelector,
    normalize_weights,
)
from megatron.core.datasets.blended_dataset import BlendedDataset
from megatron.core.datasets.blended_megatron_dataset_config import BlendedMegatronDatasetConfig
from megatron.core.datasets.megatron_dataset import MegatronDataset


_AUDIT_STOP = object()
_DYNAMIC_DATASET_STATE_VERSION = 1
_active_dynamic_train_dataset: "DynamicBlendedDataset | None" = None


def get_active_dynamic_train_dataset() -> "DynamicBlendedDataset | None":
    return _active_dynamic_train_dataset


class _AsyncJSONLAuditWriter:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._queue: queue.Queue[str | object] = queue.Queue()
        self._closed = False
        self._failure: BaseException | None = None
        self._thread = threading.Thread(
            target=self._run,
            name="dmi-mixture-selection-audit",
            daemon=True,
        )
        self._thread.start()
        atexit.register(self.close)

    def submit(self, value: dict[str, object]) -> None:
        if self._closed:
            raise RuntimeError("dynamic-mixture audit writer is closed")
        self._raise_if_failed()
        payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
        self._queue.put_nowait(payload)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._queue.put_nowait(_AUDIT_STOP)
        self._thread.join()
        self._raise_if_failed()

    def _raise_if_failed(self) -> None:
        if self._failure is not None:
            raise RuntimeError("dynamic-mixture audit writer failed") from self._failure

    def _run(self) -> None:
        try:
            with self.path.open("a", encoding="utf-8") as handle:
                while True:
                    item = self._queue.get()
                    if item is _AUDIT_STOP:
                        handle.flush()
                        os.fsync(handle.fileno())
                        return
                    handle.write(item)
                    handle.write("\n")
                    handle.flush()
        except BaseException as error:
            self._failure = error


class _SyncJSONLAuditWriter:
    """Durably write rank audit rows from one designated DataLoader worker."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._closed = False

    def submit(self, value: dict[str, object]) -> None:
        if self._closed:
            raise RuntimeError("dynamic-mixture audit writer is closed")
        payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(payload)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())

    def close(self) -> None:
        self._closed = True


class DynamicBlendedDataset(BlendedDataset):
    """Generate deterministic immutable blend windows from exact HTTP decisions."""

    def __init__(
        self,
        datasets: List[MegatronDataset],
        weights: List[Union[int, float]],
        size: Optional[int],
        config: BlendedMegatronDatasetConfig,
    ) -> None:
        global _active_dynamic_train_dataset
        if size is None:
            raise ValueError("dynamic mixture requires a finite train dataset size")
        if not datasets or len(datasets) != len(weights):
            raise ValueError("dynamic mixture requires one weight per source dataset")
        if any(type(dataset).__name__ != "SFTDataset" for dataset in datasets):
            raise TypeError("the first dynamic mixture path supports SFTDataset sources only")
        if _active_dynamic_train_dataset is not None:
            raise RuntimeError("a dynamic training dataset is already active in this process")

        self.datasets = datasets
        self.split = datasets[0].index_split
        self.weights = list(normalize_weights(weights))
        self.size = int(size)
        self.config = config

        self._run_id = _required_text(config, "dmi_dynamic_mixture_run_id")
        control_url = _required_text(config, "dmi_dynamic_mixture_control_url")
        self._window_iters = _required_positive_int(
            config, "dmi_dynamic_mixture_window_iters"
        )
        self._total_iters = _required_positive_int(
            config, "dmi_dynamic_mixture_total_iters"
        )
        self._global_batch_size = _required_positive_int(
            config, "dmi_dynamic_mixture_global_batch_size"
        )
        if self._total_iters % self._window_iters != 0:
            raise ValueError("dynamic mixture total iterations must divide into complete windows")
        self._total_windows = self._total_iters // self._window_iters
        self._samples_per_window = self._window_iters * self._global_batch_size
        required_samples = self._total_iters * self._global_batch_size
        if self.size < required_samples:
            raise ValueError(
                f"dynamic blend size {self.size} is smaller than required {required_samples}"
            )
        self._num_workers = int(
            getattr(config, "dmi_dynamic_mixture_num_workers", -1)
        )
        if self._num_workers < 0:
            raise ValueError("dynamic mixture requires nonnegative num_workers")
        if (
            self._num_workers > 0
            and getattr(config, "dmi_dynamic_mixture_resume_state", None) is not None
        ):
            raise ValueError(
                "dynamic-mixture checkpoint restore requires num_workers=0"
            )

        self._selector = WindowSelector(
            seed=int(config.random_seed),
            dataset_count=len(datasets),
            samples_per_window=self._samples_per_window,
            iterations_per_window=self._window_iters,
            global_batch_size=self._global_batch_size,
        )
        self._windows: dict[int, SelectionWindow] = {}
        self._current_window_id = 0
        self._lock = threading.Lock()
        self._cached_source_contract = self._build_source_contract()
        self._audit_path = self._build_audit_path(config)
        self._audit_writer: _AsyncJSONLAuditWriter | _SyncJSONLAuditWriter | None = None
        self._request_timeout_s = _required_positive_float(
            config, "dmi_dynamic_mixture_request_timeout_s"
        )
        self._feedback_timeout_s = _required_positive_float(
            config, "dmi_dynamic_mixture_feedback_timeout_s"
        )
        self._control_url = control_url
        self._rank = (
            torch.distributed.get_rank()
            if torch.distributed.is_initialized()
            else 0
        )
        self._runtime_pid = os.getpid()
        self._runtime_worker_id: int | None = None
        self._prefetcher: DecisionPrefetcher | None = None
        if self._num_workers == 0:
            self._audit_writer = (
                None
                if self._audit_path is None
                else _AsyncJSONLAuditWriter(self._audit_path)
            )
            self._prefetcher = self._new_prefetcher()

        resume_state = getattr(config, "dmi_dynamic_mixture_resume_state", None)
        if resume_state is None:
            initial = self._selector.build(
                window_id=1,
                weights=self.weights,
                decision_id=0,
            )
            self._install_window(initial, decision=None)
        else:
            if not isinstance(resume_state, dict):
                raise TypeError("dynamic-mixture resume state must be a mapping")
            self.load_state_dict(resume_state)
        if self._num_workers == 0 and self._current_window_id < self._total_windows:
            self._request_window(self._current_window_id + 1)

        _active_dynamic_train_dataset = self
        atexit.register(self.close)

    def _new_prefetcher(self) -> DecisionPrefetcher:
        return DecisionPrefetcher(
            DecisionHTTPClient(
                self._control_url,
                run_id=self._run_id,
                request_timeout_s=self._request_timeout_s,
                client_id=f"rank-{self._rank}",
            )
        )

    def __len__(self) -> int:
        return self.size

    def close(self) -> None:
        global _active_dynamic_train_dataset
        if self._prefetcher is not None:
            self._prefetcher.close()
            self._prefetcher = None
        if self._audit_writer is not None:
            self._audit_writer.close()
            self._audit_writer = None
        if _active_dynamic_train_dataset is self:
            _active_dynamic_train_dataset = None

    def state_dict(self) -> dict[str, object]:
        with self._lock:
            current = self._windows.get(self._current_window_id)
            if current is None:
                raise RuntimeError("dynamic mixture has no installed current window")
            return {
                "schema_version": _DYNAMIC_DATASET_STATE_VERSION,
                "configuration": {
                    "size": self.size,
                    "window_iters": self._window_iters,
                    "total_iters": self._total_iters,
                    "global_batch_size": self._global_batch_size,
                    "total_windows": self._total_windows,
                    "samples_per_window": self._samples_per_window,
                    "random_seed": int(self.config.random_seed),
                },
                "source_contract": self._cached_source_contract,
                "selector": self._selector.state_dict(),
                "current_window_id": self._current_window_id,
                "weights": list(self.weights),
                "current_window": _selection_window_to_dict(current),
            }

    def load_state_dict(self, state: dict[str, object]) -> None:
        expected = {
            "schema_version",
            "configuration",
            "source_contract",
            "selector",
            "current_window_id",
            "weights",
            "current_window",
        }
        if set(state) != expected:
            missing = sorted(expected - set(state))
            extra = sorted(set(state) - expected)
            raise ValueError(
                f"dynamic-dataset state fields mismatch: missing={missing}, extra={extra}"
            )
        if int(state["schema_version"]) != _DYNAMIC_DATASET_STATE_VERSION:
            raise ValueError(
                "unsupported dynamic-dataset state version: "
                f"{state['schema_version']!r}"
            )
        configuration = state["configuration"]
        if not isinstance(configuration, dict):
            raise TypeError("dynamic-dataset configuration must be a mapping")
        expected_configuration = {
            "size": self.size,
            "window_iters": self._window_iters,
            "total_iters": self._total_iters,
            "global_batch_size": self._global_batch_size,
            "total_windows": self._total_windows,
            "samples_per_window": self._samples_per_window,
            "random_seed": int(self.config.random_seed),
        }
        if configuration != expected_configuration:
            raise ValueError(
                "dynamic-dataset configuration mismatch: "
                f"checkpoint={configuration}, runtime={expected_configuration}"
            )
        source_contract = state["source_contract"]
        if source_contract != self._cached_source_contract:
            raise ValueError("dynamic-dataset source contract mismatch")
        selector_state = state["selector"]
        if not isinstance(selector_state, dict):
            raise TypeError("dynamic-dataset selector state must be a mapping")
        self._selector.load_state_dict(selector_state)
        current_window_id = int(state["current_window_id"])
        if not 1 <= current_window_id <= self._total_windows:
            raise ValueError("dynamic-dataset current window is outside the run")
        current_window = _selection_window_from_dict(state["current_window"])
        if current_window.window_id != current_window_id:
            raise ValueError("dynamic-dataset current window ID mismatch")
        if current_window.first_training_iteration != (
            current_window_id - 1
        ) * self._window_iters + 1:
            raise ValueError("dynamic-dataset current window iteration mismatch")
        if len(current_window.dataset_index) != self._samples_per_window:
            raise ValueError("dynamic-dataset restored dataset-index length mismatch")
        if len(current_window.dataset_sample_index) != self._samples_per_window:
            raise ValueError("dynamic-dataset restored source-index length mismatch")
        if tuple(current_window.counters_after) != self._selector.counters:
            raise ValueError("dynamic-dataset restored selector counters mismatch")
        if int(selector_state["last_window_id"]) != current_window_id:
            raise ValueError("dynamic-dataset restored selector window mismatch")
        weights = list(normalize_weights(state["weights"]))
        if tuple(weights) != current_window.weights:
            raise ValueError("dynamic-dataset restored weights mismatch")
        self._windows = {current_window_id: current_window}
        self._current_window_id = current_window_id
        self.weights = weights
        self._append_audit(
            {
                "kind": "selection_resume",
                "run_id": self._run_id,
                "window_id": current_window_id,
                "restored_at_unix_ns": time.time_ns(),
            }
        )

    def _build_source_contract(self) -> list[dict[str, object]]:
        result = []
        for dataset_id, dataset in enumerate(self.datasets):
            path = getattr(dataset, "dataset_path", None)
            path_text = None if path is None else str(Path(path).resolve())
            indices = np.asarray(
                getattr(dataset, "indices", np.arange(len(dataset))),
                dtype=np.int64,
            )
            low_level_dataset = getattr(dataset, "dataset", dataset)
            description_hash = getattr(dataset, "unique_description_hash", None)
            if description_hash is None:
                description = json.dumps(
                    getattr(dataset, "unique_identifiers", {}),
                    sort_keys=True,
                    default=str,
                )
                description_hash = hashlib.sha256(description.encode("utf-8")).hexdigest()
            result.append(
                {
                    "dataset_id": dataset_id,
                    "dataset_path": path_text,
                    "dataset_sha256": (
                        None if path_text is None else _sha256_file(Path(path_text))
                    ),
                    "length": int(len(dataset)),
                    "low_level_length": int(len(low_level_dataset)),
                    "indices_sha256": hashlib.sha256(indices.tobytes()).hexdigest(),
                    "unique_description_hash": str(description_hash),
                }
            )
        return result

    def __getitem__(self, idx: int) -> Dict[str, Union[int, np.ndarray]]:
        self._ensure_worker_runtime()
        absolute_index = int(idx)
        if absolute_index < 0:
            absolute_index += self.size
        if absolute_index < 0 or absolute_index >= self.size:
            raise IndexError(idx)
        window_id = absolute_index // self._samples_per_window + 1
        if window_id > self._total_windows:
            raise RuntimeError(
                f"training requested dynamic window {window_id}, but the configured "
                f"run ends at window {self._total_windows}"
            )
        window = self._ensure_window(window_id)
        local_index = absolute_index % self._samples_per_window
        dataset_id = int(window.dataset_index[local_index])
        absolute_source_index = int(window.dataset_sample_index[local_index])
        source_index = absolute_source_index % len(self.datasets[dataset_id])
        return {
            "dataset_id": np.int16(dataset_id),
            **self.datasets[dataset_id][source_index],
        }

    def _ensure_worker_runtime(self) -> None:
        if self._num_workers == 0:
            if self._prefetcher is None:
                raise RuntimeError("dynamic-mixture prefetcher is not initialized")
            return
        process_id = os.getpid()
        if self._runtime_pid == process_id and self._prefetcher is not None:
            return
        worker = torch.utils.data.get_worker_info()
        if worker is None:
            raise RuntimeError(
                "dynamic mixture configured with workers was accessed in the parent process"
            )
        self._runtime_pid = process_id
        self._runtime_worker_id = int(worker.id)
        self._lock = threading.Lock()
        self._prefetcher = self._new_prefetcher()
        if self._runtime_worker_id == 0 and self._audit_path is not None:
            self._audit_writer = _SyncJSONLAuditWriter(self._audit_path)
            current = self._windows.get(self._current_window_id)
            if current is None:
                raise RuntimeError("dynamic mixture has no initial worker window")
            self._record_window_audit(current, decision=None)
        if self._current_window_id < self._total_windows:
            self._request_window(self._current_window_id + 1)

    def _ensure_window(self, window_id: int) -> SelectionWindow:
        with self._lock:
            existing = self._windows.get(window_id)
            if existing is not None:
                return existing
            if window_id != self._current_window_id + 1:
                raise RuntimeError(
                    f"dynamic windows were accessed out of order: "
                    f"current={self._current_window_id}, requested={window_id}"
                )

        wait_started_ns = time.time_ns()
        if self._prefetcher is None:
            raise RuntimeError("dynamic-mixture prefetcher is not initialized")
        decision = self._prefetcher.wait(window_id, timeout_s=self._feedback_timeout_s)
        wait_finished_ns = time.time_ns()
        self._validate_decision(decision, window_id)
        generated = self._selector.build(
            window_id=window_id,
            weights=decision.weights,
            decision_id=decision.decision_id,
        )
        with self._lock:
            if window_id in self._windows:
                return self._windows[window_id]
            self._install_window(
                generated,
                decision=decision,
                wait_started_ns=wait_started_ns,
                wait_finished_ns=wait_finished_ns,
            )
            if window_id < self._total_windows:
                self._request_window(window_id + 1)
            return generated

    def _validate_decision(self, decision: MixtureDecision, window_id: int) -> None:
        expected_iteration = (window_id - 1) * self._window_iters + 1
        if decision.run_id != self._run_id:
            raise RuntimeError("dynamic mixture decision run_id mismatch")
        if decision.effective_window_id != window_id:
            raise RuntimeError("dynamic mixture decision window mismatch")
        if decision.effective_training_iteration != expected_iteration:
            raise RuntimeError(
                "dynamic mixture decision iteration mismatch: "
                f"expected={expected_iteration}, "
                f"actual={decision.effective_training_iteration}"
            )
        if len(decision.weights) != len(self.datasets):
            raise RuntimeError("dynamic mixture decision weight count mismatch")

    def _request_window(self, window_id: int) -> None:
        if window_id > self._total_windows:
            raise RuntimeError("attempted to request a decision beyond the terminal window")
        self._append_audit(
            {
                "kind": "decision_request",
                "run_id": self._run_id,
                "effective_window_id": window_id,
                "requested_at_unix_ns": time.time_ns(),
            }
        )
        if self._prefetcher is None:
            raise RuntimeError("dynamic-mixture prefetcher is not initialized")
        self._prefetcher.request(window_id)

    def _install_window(
        self,
        window: SelectionWindow,
        *,
        decision: MixtureDecision | None,
        wait_started_ns: int | None = None,
        wait_finished_ns: int | None = None,
    ) -> None:
        if window.window_id != self._current_window_id + 1:
            raise RuntimeError("dynamic mixture window installation order is invalid")
        self._windows[window.window_id] = window
        self._current_window_id = window.window_id
        self.weights = list(window.weights)
        self._record_window_audit(
            window,
            decision=decision,
            wait_started_ns=wait_started_ns,
            wait_finished_ns=wait_finished_ns,
        )

    def _record_window_audit(
        self,
        window: SelectionWindow,
        *,
        decision: MixtureDecision | None,
        wait_started_ns: int | None = None,
        wait_finished_ns: int | None = None,
    ) -> None:
        per_iteration_counts = []
        for offset in range(self._window_iters):
            start = offset * self._global_batch_size
            end = start + self._global_batch_size
            counts = np.bincount(
                window.dataset_index[start:end].astype(np.int64),
                minlength=len(self.datasets),
            )
            per_iteration_counts.append(
                {
                    "training_iteration_id": window.first_training_iteration + offset,
                    "counts": [int(value) for value in counts],
                }
            )
        self._append_audit(
            {
                "kind": "selection_window",
                "run_id": self._run_id,
                "window_id": window.window_id,
                "first_training_iteration": window.first_training_iteration,
                "decision_id": window.decision_id,
                "decision": None if decision is None else decision.to_dict(),
                "weights": list(window.weights),
                "dataset_index": [int(value) for value in window.dataset_index],
                "dataset_sample_index": [
                    int(value) for value in window.dataset_sample_index
                ],
                "counters_after": list(window.counters_after),
                "per_iteration_realized_counts": per_iteration_counts,
                "wait_started_unix_ns": wait_started_ns,
                "wait_finished_unix_ns": wait_finished_ns,
                "installed_at_unix_ns": time.time_ns(),
            }
        )

    def _build_audit_path(
        self,
        config: BlendedMegatronDatasetConfig,
    ) -> Path | None:
        configured = getattr(config, "dmi_dynamic_mixture_audit_dir", None)
        if configured is None or not str(configured).strip():
            return None
        audit_dir = Path(str(configured).strip())
        audit_dir.mkdir(parents=True, exist_ok=True)
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        return audit_dir / f"dynamic_mixture_rank{rank}.jsonl"

    def _append_audit(self, value: dict[str, object]) -> None:
        if self._audit_writer is not None:
            self._audit_writer.submit(value)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _selection_window_to_dict(window: SelectionWindow) -> dict[str, object]:
    return {
        "window_id": window.window_id,
        "first_training_iteration": window.first_training_iteration,
        "decision_id": window.decision_id,
        "weights": list(window.weights),
        "dataset_index": window.dataset_index.astype(np.int16, copy=False).tolist(),
        "dataset_sample_index": (
            window.dataset_sample_index.astype(np.int64, copy=False).tolist()
        ),
        "counters_after": list(window.counters_after),
    }


def _selection_window_from_dict(value: object) -> SelectionWindow:
    if not isinstance(value, dict):
        raise TypeError("selection-window state must be a mapping")
    expected = {
        "window_id",
        "first_training_iteration",
        "decision_id",
        "weights",
        "dataset_index",
        "dataset_sample_index",
        "counters_after",
    }
    if set(value) != expected:
        missing = sorted(expected - set(value))
        extra = sorted(set(value) - expected)
        raise ValueError(
            f"selection-window fields mismatch: missing={missing}, extra={extra}"
        )
    weights = normalize_weights(value["weights"])
    dataset_index = np.asarray(value["dataset_index"], dtype=np.int16)
    dataset_sample_index = np.asarray(
        value["dataset_sample_index"], dtype=np.int64
    )
    counters_after = tuple(int(item) for item in value["counters_after"])
    if dataset_index.ndim != 1 or dataset_sample_index.ndim != 1:
        raise ValueError("selection-window arrays must be one-dimensional")
    if len(dataset_index) != len(dataset_sample_index):
        raise ValueError("selection-window arrays have different lengths")
    if any(value < 0 for value in dataset_index):
        raise ValueError("selection-window dataset IDs must be nonnegative")
    if any(value < 0 for value in dataset_sample_index):
        raise ValueError("selection-window source indices must be nonnegative")
    if any(value < 0 for value in counters_after):
        raise ValueError("selection-window counters must be nonnegative")
    return SelectionWindow(
        window_id=int(value["window_id"]),
        first_training_iteration=int(value["first_training_iteration"]),
        decision_id=int(value["decision_id"]),
        weights=weights,
        dataset_index=dataset_index,
        dataset_sample_index=dataset_sample_index,
        counters_after=counters_after,
    )


def _required_text(config: object, name: str) -> str:
    value = getattr(config, name, None)
    if value is None or not str(value).strip():
        raise ValueError(f"dynamic mixture requires {name}")
    return str(value).strip()


def _required_positive_int(config: object, name: str) -> int:
    value = int(getattr(config, name, 0))
    if value <= 0:
        raise ValueError(f"dynamic mixture requires positive {name}")
    return value


def _required_positive_float(config: object, name: str) -> float:
    value = float(getattr(config, name, 0.0))
    if value <= 0.0:
        raise ValueError(f"dynamic mixture requires positive {name}")
    return value
