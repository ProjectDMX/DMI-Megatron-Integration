"""Loss-slope mixture controller."""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass
from typing import Mapping, Sequence

from tools.sft_mixture.dynamic_mixture import (
    MixtureDecision,
    normalize_weights,
)

_LOSS_SLOPE_WINDOW_STATE_VERSION = 1
_LOSS_SLOPE_CONTROLLER_STATE_VERSION = 1

LOSS_SLOPE_CONTROLLER_MODE = "loss_slope"


@dataclass(frozen=True)
class LossSlopeSampleObservation:
    """One loss-only sample observation with no router/pathway payload."""

    training_iteration_id: int
    sample_coordinate: tuple[int, int, int, int]
    dataset_id: int
    loss_mean: float
    loss_token_count: int

    def __post_init__(self) -> None:
        if self.training_iteration_id <= 0:
            raise ValueError("training_iteration_id must be positive")
        if len(self.sample_coordinate) != 4:
            raise ValueError("sample_coordinate must contain dp, microbatch, sample, shard")
        if any(value < 0 for value in self.sample_coordinate) or self.dataset_id < 0:
            raise ValueError("sample and dataset coordinates must be nonnegative")
        if not math.isfinite(self.loss_mean):
            raise ValueError("loss_mean must be finite")
        if self.loss_token_count < 0:
            raise ValueError("loss_token_count must be nonnegative")


@dataclass(frozen=True)
class LossSlopeIterationMetric:
    controller_mode: str
    dataset_id: int
    window_id: int
    training_iteration_id: int
    status: str
    incomplete_reason: str
    sample_count: int
    positive_loss_sample_count: int
    target_token_count: int
    loss_value: float | None
    processed_at_unix_ns: int

    def __post_init__(self) -> None:
        if self.controller_mode != LOSS_SLOPE_CONTROLLER_MODE:
            raise ValueError("loss-slope iteration metric has the wrong controller mode")
        if self.status not in {"COMPLETE", "INCOMPLETE"}:
            raise ValueError("invalid loss-slope iteration status")
        if self.status == "COMPLETE" and self.loss_value is None:
            raise ValueError("complete loss-slope iteration metric requires a loss")
        if self.status == "INCOMPLETE" and not self.incomplete_reason:
            raise ValueError("incomplete loss-slope iteration metric requires a reason")
        if any(
            value < 0
            for value in (
                self.dataset_id,
                self.window_id,
                self.training_iteration_id,
                self.sample_count,
                self.positive_loss_sample_count,
                self.target_token_count,
                self.processed_at_unix_ns,
            )
        ):
            raise ValueError("loss-slope iteration metric counts must be nonnegative")
        if self.positive_loss_sample_count > self.sample_count:
            raise ValueError("positive-loss sample count exceeds sample count")
        if self.loss_value is not None and not math.isfinite(self.loss_value):
            raise ValueError("loss-slope iteration value must be finite")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class LossSlopeWindowIndicator:
    controller_mode: str
    dataset_id: int
    window_id: int
    window_start_iteration: int
    window_end_iteration: int
    terminal_window: bool
    status: str
    incomplete_reason: str
    iteration_point_count: int
    expected_iteration_point_count: int
    sample_count: int
    positive_loss_sample_count: int
    target_token_count: int
    loss_slope: float | None
    indicator: float | None
    processed_at_unix_ns: int

    def __post_init__(self) -> None:
        if self.controller_mode != LOSS_SLOPE_CONTROLLER_MODE:
            raise ValueError("loss-slope window indicator has the wrong controller mode")
        if self.status not in {"COMPLETE", "INCOMPLETE"}:
            raise ValueError("invalid loss-slope window status")
        if self.status == "COMPLETE":
            if self.loss_slope is None or self.indicator is None:
                raise ValueError("complete loss-slope window requires slope and indicator")
        elif not self.incomplete_reason:
            raise ValueError("incomplete loss-slope window requires a reason")
        if self.loss_slope is not None and not math.isfinite(self.loss_slope):
            raise ValueError("loss-slope slope must be finite")
        if self.indicator is not None and not math.isfinite(self.indicator):
            raise ValueError("loss-slope indicator must be finite")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class LossSlopeControlResult:
    decision: MixtureDecision | None
    indicators: tuple[LossSlopeWindowIndicator, ...]
    terminal_window: bool


def ordinary_loss_slope(points: Sequence[tuple[int, float]]) -> float:
    """Fit an unweighted OLS slope over ordered absolute iteration IDs."""

    if len(points) < 2:
        raise ValueError("ordinary loss slope requires at least two points")
    xs = [float(point[0]) for point in points]
    ys = [float(point[1]) for point in points]
    if xs != sorted(set(xs)):
        raise ValueError("loss-slope iterations must be sorted and unique")
    if any(not math.isfinite(value) for value in xs + ys):
        raise ValueError("loss-slope inputs must be finite")
    x_mean = sum(xs) / len(xs)
    y_mean = sum(ys) / len(ys)
    denominator = sum((value - x_mean) ** 2 for value in xs)
    if denominator <= 0.0:
        raise ValueError("loss-slope iterations have no variation")
    numerator = sum(
        (x - x_mean) * (y - y_mean) for x, y in zip(xs, ys)
    )
    slope = numerator / denominator
    if not math.isfinite(slope):
        raise ValueError("ordinary loss slope is nonfinite")
    return float(slope)


class LossSlopeDomainWindowState:
    """Strict per-domain, per-iteration token-weighted loss window."""

    def __init__(
        self,
        *,
        dataset_id: int,
        window_id: int,
        window_start_iteration: int,
        window_end_iteration: int,
        terminal_window: bool,
    ) -> None:
        if dataset_id < 0:
            raise ValueError("dataset_id must be nonnegative")
        if window_id <= 0 or window_start_iteration <= 0:
            raise ValueError("window coordinates must be positive")
        if window_end_iteration < window_start_iteration:
            raise ValueError("window end precedes its start")
        self.dataset_id = int(dataset_id)
        self.window_id = int(window_id)
        self.window_start_iteration = int(window_start_iteration)
        self.window_end_iteration = int(window_end_iteration)
        self.terminal_window = bool(terminal_window)
        self._iteration_metrics: list[LossSlopeIterationMetric] = []
        self._finalized = False

    @property
    def iteration_metrics(self) -> tuple[LossSlopeIterationMetric, ...]:
        return tuple(self._iteration_metrics)

    def add_iteration_samples(
        self,
        training_iteration_id: int,
        samples: Sequence[LossSlopeSampleObservation],
    ) -> LossSlopeIterationMetric:
        if self._finalized:
            raise RuntimeError("cannot add samples after loss-slope window finalization")
        iteration = int(training_iteration_id)
        expected_iteration = (
            self.window_start_iteration + len(self._iteration_metrics)
        )
        if iteration != expected_iteration:
            raise RuntimeError(
                "loss-slope iterations must be processed consecutively: "
                f"expected={expected_iteration}, actual={iteration}"
            )
        if not self.window_start_iteration <= iteration <= self.window_end_iteration:
            raise ValueError("loss-slope iteration is outside its control window")
        coordinates = [sample.sample_coordinate for sample in samples]
        if coordinates != sorted(set(coordinates)):
            raise RuntimeError("loss-slope sample coordinates must be sorted and unique")
        for sample in samples:
            if sample.training_iteration_id != iteration:
                raise ValueError("loss-slope samples span multiple iterations")
            if sample.dataset_id != self.dataset_id:
                raise ValueError("loss-slope domain state received another dataset")

        positive = [sample for sample in samples if sample.loss_token_count > 0]
        target_token_count = sum(sample.loss_token_count for sample in positive)
        if not samples:
            loss_value = None
            reason = "no_samples"
        elif target_token_count <= 0:
            loss_value = None
            reason = "no_target_tokens"
        else:
            loss_value = sum(
                sample.loss_mean * sample.loss_token_count for sample in positive
            ) / target_token_count
            reason = ""
        metric = LossSlopeIterationMetric(
            controller_mode=LOSS_SLOPE_CONTROLLER_MODE,
            dataset_id=self.dataset_id,
            window_id=self.window_id,
            training_iteration_id=iteration,
            status="COMPLETE" if loss_value is not None else "INCOMPLETE",
            incomplete_reason=reason,
            sample_count=len(samples),
            positive_loss_sample_count=len(positive),
            target_token_count=target_token_count,
            loss_value=None if loss_value is None else float(loss_value),
            processed_at_unix_ns=time.time_ns(),
        )
        self._iteration_metrics.append(metric)
        return metric

    def finalize(self) -> LossSlopeWindowIndicator:
        if self._finalized:
            raise RuntimeError("loss-slope window state was already finalized")
        expected_count = self.window_end_iteration - self.window_start_iteration + 1
        actual_iterations = [
            metric.training_iteration_id for metric in self._iteration_metrics
        ]
        expected_iterations = list(
            range(self.window_start_iteration, self.window_end_iteration + 1)
        )
        reasons: list[str] = []
        if actual_iterations != expected_iterations:
            reasons.append(
                f"observed_iterations={len(actual_iterations)}<{expected_count}"
            )
        incomplete_iterations = [
            metric.training_iteration_id
            for metric in self._iteration_metrics
            if metric.loss_value is None
        ]
        if incomplete_iterations:
            reasons.append(
                "missing_loss_iterations="
                + ",".join(str(value) for value in incomplete_iterations)
            )
        points = [
            (metric.training_iteration_id, float(metric.loss_value))
            for metric in self._iteration_metrics
            if metric.loss_value is not None
        ]
        slope = None
        indicator = None
        if not reasons:
            try:
                slope = ordinary_loss_slope(points)
            except ValueError as error:
                reasons.append(f"undefined_loss_slope:{error}")
            else:
                indicator = -slope
        self._finalized = True
        return LossSlopeWindowIndicator(
            controller_mode=LOSS_SLOPE_CONTROLLER_MODE,
            dataset_id=self.dataset_id,
            window_id=self.window_id,
            window_start_iteration=self.window_start_iteration,
            window_end_iteration=self.window_end_iteration,
            terminal_window=self.terminal_window,
            status="COMPLETE" if not reasons else "INCOMPLETE",
            incomplete_reason=";".join(reasons),
            iteration_point_count=len(points),
            expected_iteration_point_count=expected_count,
            sample_count=sum(metric.sample_count for metric in self._iteration_metrics),
            positive_loss_sample_count=sum(
                metric.positive_loss_sample_count
                for metric in self._iteration_metrics
            ),
            target_token_count=sum(
                metric.target_token_count for metric in self._iteration_metrics
            ),
            loss_slope=slope,
            indicator=indicator,
            processed_at_unix_ns=time.time_ns(),
        )

    def state_dict(self) -> dict[str, object]:
        if self._finalized:
            raise RuntimeError("cannot checkpoint a finalized loss-slope window")
        return {
            "schema_version": _LOSS_SLOPE_WINDOW_STATE_VERSION,
            "controller_mode": LOSS_SLOPE_CONTROLLER_MODE,
            "dataset_id": self.dataset_id,
            "window_id": self.window_id,
            "window_start_iteration": self.window_start_iteration,
            "window_end_iteration": self.window_end_iteration,
            "terminal_window": self.terminal_window,
            "iteration_metrics": [
                metric.to_dict() for metric in self._iteration_metrics
            ],
        }

    @classmethod
    def from_state_dict(
        cls,
        state: Mapping[str, object],
    ) -> "LossSlopeDomainWindowState":
        expected = {
            "schema_version",
            "controller_mode",
            "dataset_id",
            "window_id",
            "window_start_iteration",
            "window_end_iteration",
            "terminal_window",
            "iteration_metrics",
        }
        if set(state) != expected:
            raise ValueError("loss-slope window-state fields mismatch")
        if int(state["schema_version"]) != _LOSS_SLOPE_WINDOW_STATE_VERSION:
            raise ValueError("unsupported loss-slope window-state version")
        if state["controller_mode"] != LOSS_SLOPE_CONTROLLER_MODE:
            raise ValueError("loss-slope window-state controller mode mismatch")
        result = cls(
            dataset_id=int(state["dataset_id"]),
            window_id=int(state["window_id"]),
            window_start_iteration=int(state["window_start_iteration"]),
            window_end_iteration=int(state["window_end_iteration"]),
            terminal_window=bool(state["terminal_window"]),
        )
        raw_metrics = state["iteration_metrics"]
        if not isinstance(raw_metrics, (list, tuple)):
            raise TypeError("loss-slope iteration metrics must be a sequence")
        for raw in raw_metrics:
            if not isinstance(raw, Mapping):
                raise TypeError("loss-slope iteration metric must be a mapping")
            metric_fields = {
                "controller_mode",
                "dataset_id",
                "window_id",
                "training_iteration_id",
                "status",
                "incomplete_reason",
                "sample_count",
                "positive_loss_sample_count",
                "target_token_count",
                "loss_value",
                "processed_at_unix_ns",
            }
            if set(raw) != metric_fields:
                raise ValueError("loss-slope iteration-metric fields mismatch")
            metric = LossSlopeIterationMetric(
                controller_mode=str(raw["controller_mode"]),
                dataset_id=int(raw["dataset_id"]),
                window_id=int(raw["window_id"]),
                training_iteration_id=int(raw["training_iteration_id"]),
                status=str(raw["status"]),
                incomplete_reason=str(raw["incomplete_reason"]),
                sample_count=int(raw["sample_count"]),
                positive_loss_sample_count=int(raw["positive_loss_sample_count"]),
                target_token_count=int(raw["target_token_count"]),
                loss_value=(
                    None if raw["loss_value"] is None else float(raw["loss_value"])
                ),
                processed_at_unix_ns=int(raw["processed_at_unix_ns"]),
            )
            expected_iteration = (
                result.window_start_iteration + len(result._iteration_metrics)
            )
            if metric.training_iteration_id != expected_iteration:
                raise ValueError("restored loss-slope iterations are not consecutive")
            if metric.dataset_id != result.dataset_id:
                raise ValueError("restored loss-slope metric has the wrong dataset")
            if metric.window_id != result.window_id:
                raise ValueError("restored loss-slope metric has the wrong window")
            result._iteration_metrics.append(metric)
        return result


class LossSlopeMixtureController:
    """One-window signed decreasing-loss-slope mixture controller."""

    def __init__(
        self,
        *,
        run_id: str,
        dataset_ids: Sequence[int],
        initial_weights: Sequence[float],
        window_iters: int,
        total_windows: int,
        eta: float = 500.0,
        uniform_smoothing: float = 0.5,
    ) -> None:
        self.run_id = str(run_id)
        self.dataset_ids = tuple(int(value) for value in dataset_ids)
        if self.dataset_ids != tuple(sorted(set(self.dataset_ids))):
            raise ValueError("dataset_ids must be sorted and unique")
        self.previous_weights = normalize_weights(initial_weights)
        if len(self.previous_weights) != len(self.dataset_ids):
            raise ValueError("initial weight count does not match dataset count")
        if window_iters < 2 or total_windows <= 0:
            raise ValueError("loss-slope controller requires windows of at least two points")
        if eta <= 0.0 or not 0.0 <= uniform_smoothing <= 1.0:
            raise ValueError("invalid loss-slope controller update parameters")
        self.window_iters = int(window_iters)
        self.total_windows = int(total_windows)
        self.eta = float(eta)
        self.uniform_smoothing = float(uniform_smoothing)
        self._last_window_id = 0
        self._next_decision_id = 1

    def state_dict(self) -> dict[str, object]:
        return {
            "schema_version": _LOSS_SLOPE_CONTROLLER_STATE_VERSION,
            "controller_mode": LOSS_SLOPE_CONTROLLER_MODE,
            "run_id": self.run_id,
            "dataset_ids": list(self.dataset_ids),
            "previous_weights": list(self.previous_weights),
            "window_iters": self.window_iters,
            "total_windows": self.total_windows,
            "eta": self.eta,
            "uniform_smoothing": self.uniform_smoothing,
            "last_window_id": self._last_window_id,
            "next_decision_id": self._next_decision_id,
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        expected = {
            "schema_version",
            "controller_mode",
            "run_id",
            "dataset_ids",
            "previous_weights",
            "window_iters",
            "total_windows",
            "eta",
            "uniform_smoothing",
            "last_window_id",
            "next_decision_id",
        }
        if set(state) != expected:
            raise ValueError("loss-slope controller-state fields mismatch")
        if int(state["schema_version"]) != _LOSS_SLOPE_CONTROLLER_STATE_VERSION:
            raise ValueError("unsupported loss-slope controller-state version")
        if state["controller_mode"] != LOSS_SLOPE_CONTROLLER_MODE:
            raise ValueError("loss-slope controller-state mode mismatch")
        runtime = {
            "run_id": self.run_id,
            "dataset_ids": list(self.dataset_ids),
            "window_iters": self.window_iters,
            "total_windows": self.total_windows,
            "eta": self.eta,
            "uniform_smoothing": self.uniform_smoothing,
        }
        checkpoint = {
            "run_id": str(state["run_id"]),
            "dataset_ids": [int(value) for value in state["dataset_ids"]],
            "window_iters": int(state["window_iters"]),
            "total_windows": int(state["total_windows"]),
            "eta": float(state["eta"]),
            "uniform_smoothing": float(state["uniform_smoothing"]),
        }
        if checkpoint != runtime:
            raise ValueError(
                f"loss-slope controller configuration mismatch: "
                f"checkpoint={checkpoint}, runtime={runtime}"
            )
        weights = normalize_weights(state["previous_weights"])
        if len(weights) != len(self.dataset_ids):
            raise ValueError("loss-slope controller weight count mismatch")
        last_window_id = int(state["last_window_id"])
        next_decision_id = int(state["next_decision_id"])
        if not 0 <= last_window_id <= self.total_windows:
            raise ValueError("loss-slope last window is outside the run")
        if next_decision_id <= 0:
            raise ValueError("loss-slope next decision ID must be positive")
        self.previous_weights = weights
        self._last_window_id = last_window_id
        self._next_decision_id = next_decision_id

    def process_window(
        self,
        indicators: Mapping[int, LossSlopeWindowIndicator],
    ) -> LossSlopeControlResult:
        if set(indicators) != set(self.dataset_ids):
            raise ValueError("loss-slope indicator dataset coverage mismatch")
        window_ids = {item.window_id for item in indicators.values()}
        if len(window_ids) != 1:
            raise ValueError("loss-slope indicators span multiple windows")
        window_id = next(iter(window_ids))
        if window_id != self._last_window_id + 1:
            raise RuntimeError("loss-slope windows must be processed in order")
        terminal = window_id == self.total_windows
        if any(item.terminal_window != terminal for item in indicators.values()):
            raise ValueError("loss-slope terminal-window marker mismatch")
        if any(item.controller_mode != LOSS_SLOPE_CONTROLLER_MODE for item in indicators.values()):
            raise ValueError("loss-slope indicator mode mismatch")
        self._last_window_id = window_id
        ordered = tuple(indicators[dataset_id] for dataset_id in self.dataset_ids)
        if terminal:
            return LossSlopeControlResult(None, ordered, True)
        incomplete = [item for item in ordered if item.status != "COMPLETE"]
        if incomplete:
            reason = "|".join(
                f"dataset={item.dataset_id}:{item.incomplete_reason}"
                for item in incomplete
            )
            return LossSlopeControlResult(
                self._decision(
                    window_id=window_id,
                    decision_type="HOLD_INCOMPLETE",
                    weights=self.previous_weights,
                    reason=reason,
                ),
                ordered,
                False,
            )
        logits = [
            math.log(weight) + self.eta * float(item.indicator)
            for weight, item in zip(self.previous_weights, ordered)
        ]
        maximum = max(logits)
        exponentials = [math.exp(value - maximum) for value in logits]
        denominator = sum(exponentials)
        alpha = [value / denominator for value in exponentials]
        domain_count = len(alpha)
        updated = tuple(
            (1.0 - self.uniform_smoothing) * value
            + self.uniform_smoothing / domain_count
            for value in alpha
        )
        decision = self._decision(
            window_id=window_id,
            decision_type="UPDATE",
            weights=updated,
            reason="loss_slope",
        )
        self.previous_weights = decision.weights
        return LossSlopeControlResult(decision, ordered, False)

    def _decision(
        self,
        *,
        window_id: int,
        decision_type: str,
        weights: Sequence[float],
        reason: str,
    ) -> MixtureDecision:
        decision = MixtureDecision(
            run_id=self.run_id,
            decision_id=self._next_decision_id,
            decision_type=decision_type,
            source_window_id=window_id,
            source_window_end_iteration=window_id * self.window_iters,
            effective_window_id=window_id + 1,
            effective_training_iteration=window_id * self.window_iters + 1,
            weights=normalize_weights(weights),
            reason=reason,
            produced_at_unix_ns=time.time_ns(),
        )
        self._next_decision_id += 1
        return decision
