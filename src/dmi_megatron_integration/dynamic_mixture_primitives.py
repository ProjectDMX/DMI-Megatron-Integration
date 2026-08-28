"""Deterministic windowed mixture selection and exact-window decision delivery."""

from __future__ import annotations

import json
import math
import random
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from http import HTTPStatus
from typing import Mapping, Sequence

import numpy as np


DECISION_TYPES = frozenset({"UPDATE", "HOLD_INCOMPLETE"})
_WINDOW_SELECTOR_STATE_VERSION = 1


def _lists_to_tuples(value: object) -> object:
    if isinstance(value, list):
        return tuple(_lists_to_tuples(item) for item in value)
    return value


def normalize_weights(weights: Sequence[float]) -> tuple[float, ...]:
    values = tuple(float(value) for value in weights)
    if not values:
        raise ValueError("mixture weights must not be empty")
    if any(not math.isfinite(value) or value <= 0.0 for value in values):
        raise ValueError(f"mixture weights must be finite and positive: {values}")
    total = sum(values)
    return tuple(value / total for value in values)


@dataclass(frozen=True)
class MixtureDecision:
    run_id: str
    decision_id: int
    decision_type: str
    source_window_id: int
    source_window_end_iteration: int
    effective_window_id: int
    effective_training_iteration: int
    weights: tuple[float, ...]
    reason: str
    produced_at_unix_ns: int

    def __post_init__(self) -> None:
        if not self.run_id:
            raise ValueError("run_id must not be empty")
        if self.decision_id <= 0:
            raise ValueError("decision_id must be positive")
        if self.decision_type not in DECISION_TYPES:
            raise ValueError(f"unsupported decision_type: {self.decision_type!r}")
        if self.source_window_id <= 0:
            raise ValueError("source_window_id must be positive")
        if self.effective_window_id != self.source_window_id + 1:
            raise ValueError("a decision must target the immediately following window")
        if self.source_window_end_iteration <= 0 or self.effective_training_iteration <= 0:
            raise ValueError("decision iteration coordinates must be positive")
        normalized = normalize_weights(self.weights)
        if len(normalized) != len(self.weights):
            raise AssertionError("weight normalization changed the vector length")
        if any(abs(left - right) > 1e-12 for left, right in zip(normalized, self.weights)):
            raise ValueError("decision weights must already be normalized")
        if self.decision_type == "HOLD_INCOMPLETE" and not self.reason:
            raise ValueError("HOLD_INCOMPLETE requires a reason")
        if self.produced_at_unix_ns <= 0:
            raise ValueError("produced_at_unix_ns must be positive")

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["weights"] = list(self.weights)
        return result

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "MixtureDecision":
        fields = {
            "run_id",
            "decision_id",
            "decision_type",
            "source_window_id",
            "source_window_end_iteration",
            "effective_window_id",
            "effective_training_iteration",
            "weights",
            "reason",
            "produced_at_unix_ns",
        }
        if set(value) != fields:
            missing = sorted(fields - set(value))
            extra = sorted(set(value) - fields)
            raise ValueError(f"decision fields mismatch: missing={missing}, extra={extra}")
        raw_weights = value["weights"]
        if not isinstance(raw_weights, list):
            raise TypeError("decision weights must be a JSON array")
        return cls(
            run_id=str(value["run_id"]),
            decision_id=int(value["decision_id"]),
            decision_type=str(value["decision_type"]),
            source_window_id=int(value["source_window_id"]),
            source_window_end_iteration=int(value["source_window_end_iteration"]),
            effective_window_id=int(value["effective_window_id"]),
            effective_training_iteration=int(value["effective_training_iteration"]),
            weights=tuple(float(item) for item in raw_weights),
            reason=str(value["reason"]),
            produced_at_unix_ns=int(value["produced_at_unix_ns"]),
        )


@dataclass(frozen=True)
class SelectionWindow:
    window_id: int
    first_training_iteration: int
    decision_id: int
    weights: tuple[float, ...]
    dataset_index: np.ndarray
    dataset_sample_index: np.ndarray
    counters_after: tuple[int, ...]


class WindowSelector:
    """Generate bounded blend windows from one deterministic categorical stream."""

    def __init__(
        self,
        *,
        seed: int,
        dataset_count: int,
        samples_per_window: int,
        iterations_per_window: int,
        global_batch_size: int,
    ) -> None:
        if dataset_count <= 0:
            raise ValueError("dataset_count must be positive")
        if samples_per_window <= 0:
            raise ValueError("samples_per_window must be positive")
        if iterations_per_window <= 0 or global_batch_size <= 0:
            raise ValueError("window and batch dimensions must be positive")
        if samples_per_window != iterations_per_window * global_batch_size:
            raise ValueError(
                "samples_per_window must equal iterations_per_window * global_batch_size"
            )
        self.seed = int(seed)
        self.dataset_count = int(dataset_count)
        self.samples_per_window = int(samples_per_window)
        self.iterations_per_window = int(iterations_per_window)
        self.global_batch_size = int(global_batch_size)
        self._rng = random.Random(self.seed)
        self._counters = [0] * self.dataset_count
        self._last_window_id = 0

    @property
    def counters(self) -> tuple[int, ...]:
        return tuple(self._counters)

    def state_dict(self) -> dict[str, object]:
        return {
            "schema_version": _WINDOW_SELECTOR_STATE_VERSION,
            "seed": self.seed,
            "dataset_count": self.dataset_count,
            "samples_per_window": self.samples_per_window,
            "iterations_per_window": self.iterations_per_window,
            "global_batch_size": self.global_batch_size,
            "rng_state": self._rng.getstate(),
            "counters": list(self._counters),
            "last_window_id": self._last_window_id,
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        expected = {
            "schema_version",
            "seed",
            "dataset_count",
            "samples_per_window",
            "iterations_per_window",
            "global_batch_size",
            "rng_state",
            "counters",
            "last_window_id",
        }
        if set(state) != expected:
            missing = sorted(expected - set(state))
            extra = sorted(set(state) - expected)
            raise ValueError(
                f"window-selector state fields mismatch: missing={missing}, extra={extra}"
            )
        if int(state["schema_version"]) != _WINDOW_SELECTOR_STATE_VERSION:
            raise ValueError(
                "unsupported window-selector state version: "
                f"{state['schema_version']!r}"
            )
        dimensions = {
            "seed": self.seed,
            "dataset_count": self.dataset_count,
            "samples_per_window": self.samples_per_window,
            "iterations_per_window": self.iterations_per_window,
            "global_batch_size": self.global_batch_size,
        }
        mismatches = {
            name: (expected_value, int(state[name]))
            for name, expected_value in dimensions.items()
            if int(state[name]) != expected_value
        }
        if mismatches:
            raise ValueError(f"window-selector configuration mismatch: {mismatches}")
        raw_counters = state["counters"]
        if not isinstance(raw_counters, (list, tuple)):
            raise TypeError("window-selector counters must be a sequence")
        counters = [int(value) for value in raw_counters]
        if len(counters) != self.dataset_count or any(value < 0 for value in counters):
            raise ValueError("window-selector counters are invalid")
        last_window_id = int(state["last_window_id"])
        if last_window_id < 0:
            raise ValueError("window-selector last_window_id must be nonnegative")
        rng_state = _lists_to_tuples(state["rng_state"])
        if not isinstance(rng_state, tuple):
            raise TypeError("window-selector RNG state must decode to a tuple")
        probe = random.Random()
        probe.setstate(rng_state)
        self._rng.setstate(rng_state)
        self._counters = counters
        self._last_window_id = last_window_id

    def build(
        self,
        *,
        window_id: int,
        weights: Sequence[float],
        decision_id: int,
    ) -> SelectionWindow:
        if int(window_id) != self._last_window_id + 1:
            raise RuntimeError(
                f"selection windows must be generated in order: "
                f"last={self._last_window_id}, requested={window_id}"
            )
        normalized = normalize_weights(weights)
        if len(normalized) != self.dataset_count:
            raise ValueError(
                f"weight count {len(normalized)} does not match dataset count "
                f"{self.dataset_count}"
            )
        selected = self._rng.choices(
            range(self.dataset_count),
            weights=normalized,
            k=self.samples_per_window,
        )
        dataset_index = np.asarray(selected, dtype=np.int16)
        dataset_sample_index = np.empty(self.samples_per_window, dtype=np.int64)
        for index, dataset_id in enumerate(selected):
            dataset_sample_index[index] = self._counters[dataset_id]
            self._counters[dataset_id] += 1
        self._last_window_id = int(window_id)
        return SelectionWindow(
            window_id=int(window_id),
            first_training_iteration=(int(window_id) - 1) * self.iterations_per_window + 1,
            decision_id=int(decision_id),
            weights=normalized,
            dataset_index=dataset_index,
            dataset_sample_index=dataset_sample_index,
            counters_after=tuple(self._counters),
        )


class DecisionHTTPClient:
    def __init__(
        self,
        base_url: str,
        *,
        run_id: str,
        request_timeout_s: float,
        client_id: str = "",
    ) -> None:
        if not base_url:
            raise ValueError("base_url must not be empty")
        if not run_id:
            raise ValueError("run_id must not be empty")
        if request_timeout_s <= 0.0:
            raise ValueError("request_timeout_s must be positive")
        self.base_url = base_url.rstrip("/")
        self.run_id = run_id
        self.request_timeout_s = float(request_timeout_s)
        self.client_id = str(client_id)

    def request_once(self, effective_window_id: int) -> MixtureDecision | None:
        query = urllib.parse.urlencode(
            {
                "run_id": self.run_id,
                "effective_window_id": int(effective_window_id),
                "timeout_s": self.request_timeout_s,
                "client_id": self.client_id,
            }
        )
        request = urllib.request.Request(
            f"{self.base_url}/v1/decision?{query}",
            method="GET",
        )
        try:
            with urllib.request.urlopen(
                request,
                timeout=self.request_timeout_s + 5.0,
            ) as response:
                if response.status == HTTPStatus.NO_CONTENT:
                    return None
                if response.status != HTTPStatus.OK:
                    raise RuntimeError(
                        f"decision server returned HTTP {response.status}"
                    )
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            if error.code == HTTPStatus.NO_CONTENT:
                return None
            raise RuntimeError(f"decision HTTP request failed: {error}") from error
        if not isinstance(body, dict):
            raise TypeError("decision response must be a JSON object")
        decision = MixtureDecision.from_dict(body)
        if decision.run_id != self.run_id:
            raise RuntimeError("decision response run_id mismatch")
        if decision.effective_window_id != int(effective_window_id):
            raise RuntimeError("decision response window mismatch")
        return decision


class DecisionPrefetcher:
    """One always-on client thread that caches exact requested windows."""

    def __init__(self, client: DecisionHTTPClient) -> None:
        self.client = client
        self._condition = threading.Condition()
        self._requested: set[int] = set()
        self._pending: list[int] = []
        self._cache: dict[int, MixtureDecision] = {}
        self._failure: BaseException | None = None
        self._closed = False
        self._thread = threading.Thread(
            target=self._run,
            name="dmi-mixture-decision-client",
            daemon=True,
        )
        self._thread.start()

    def request(self, effective_window_id: int) -> None:
        window_id = int(effective_window_id)
        with self._condition:
            if window_id in self._requested:
                return
            if self._failure is not None:
                raise RuntimeError("decision prefetch thread failed") from self._failure
            self._requested.add(window_id)
            self._pending.append(window_id)
            self._condition.notify_all()

    def wait(self, effective_window_id: int, *, timeout_s: float) -> MixtureDecision:
        window_id = int(effective_window_id)
        self.request(window_id)
        deadline = time.monotonic() + float(timeout_s)
        with self._condition:
            while window_id not in self._cache:
                if self._failure is not None:
                    raise RuntimeError("decision prefetch thread failed") from self._failure
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    raise TimeoutError(
                        f"decision for effective window {window_id} missed "
                        f"the {timeout_s:.3f}s feedback deadline"
                    )
                self._condition.wait(remaining)
            return self._cache[window_id]

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._condition.notify_all()
        self._thread.join(timeout=self.client.request_timeout_s + 6.0)

    def _run(self) -> None:
        try:
            while True:
                with self._condition:
                    while not self._pending and not self._closed:
                        self._condition.wait()
                    if self._closed:
                        return
                    window_id = self._pending.pop(0)
                while True:
                    decision = self.client.request_once(window_id)
                    if decision is not None:
                        break
                    with self._condition:
                        if self._closed:
                            return
                with self._condition:
                    self._cache[window_id] = decision
                    self._condition.notify_all()
        except BaseException as error:
            with self._condition:
                self._failure = error
                self._condition.notify_all()
