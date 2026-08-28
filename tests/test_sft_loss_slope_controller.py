import math

import pytest

from tools.sft_mixture.controller import (
    LOSS_SLOPE_CONTROLLER_MODE,
    LossSlopeDomainWindowState,
    LossSlopeMixtureController,
    LossSlopeSampleObservation,
    ordinary_loss_slope,
)


def _sample(iteration, index, dataset_id, loss, tokens):
    return LossSlopeSampleObservation(
        training_iteration_id=iteration,
        sample_coordinate=(0, index, 0, 0),
        dataset_id=dataset_id,
        loss_mean=loss,
        loss_token_count=tokens,
    )


def _complete_window(dataset_id, losses, *, window_id=1, terminal=False):
    start = (window_id - 1) * len(losses) + 1
    state = LossSlopeDomainWindowState(
        dataset_id=dataset_id,
        window_id=window_id,
        window_start_iteration=start,
        window_end_iteration=start + len(losses) - 1,
        terminal_window=terminal,
    )
    for offset, loss in enumerate(losses):
        iteration = start + offset
        state.add_iteration_samples(
            iteration,
            [
                _sample(iteration, 0, dataset_id, loss - 1.0, 1),
                _sample(iteration, 1, dataset_id, loss + 1.0 / 3.0, 3),
                _sample(iteration, 2, dataset_id, 100.0, 0),
            ],
        )
    return state


def test_loss_slope_uses_one_token_weighted_point_per_domain_iteration():
    state = _complete_window(0, (4.0, 3.5, 3.0))

    assert [
        metric.loss_value for metric in state.iteration_metrics
    ] == pytest.approx((4.0, 3.5, 3.0))
    assert [metric.sample_count for metric in state.iteration_metrics] == [3, 3, 3]
    assert [
        metric.positive_loss_sample_count for metric in state.iteration_metrics
    ] == [2, 2, 2]
    assert [metric.target_token_count for metric in state.iteration_metrics] == [4, 4, 4]

    indicator = state.finalize()
    assert indicator.controller_mode == LOSS_SLOPE_CONTROLLER_MODE
    assert indicator.status == "COMPLETE"
    assert indicator.iteration_point_count == 3
    assert indicator.loss_slope == pytest.approx(-0.5)
    assert indicator.indicator == pytest.approx(0.5)


def test_ordinary_loss_slope_uses_iteration_trend_without_uncertainty_scaling():
    assert ordinary_loss_slope(((10, 2.0), (11, 1.7), (12, 1.4))) == pytest.approx(
        -0.3
    )
    with pytest.raises(ValueError, match="sorted and unique"):
        ordinary_loss_slope(((2, 1.0), (1, 2.0)))


@pytest.mark.parametrize(
    ("losses", "expected_slope", "expected_indicator"),
    (
        ((3.0, 2.0, 1.0), -1.0, 1.0),
        ((2.0, 2.0, 2.0), 0.0, 0.0),
        ((1.0, 2.0, 3.0), 1.0, -1.0),
    ),
)
def test_loss_slope_indicator_preserves_decrease_direction(
    losses, expected_slope, expected_indicator
):
    indicator = _complete_window(0, losses).finalize()

    assert indicator.loss_slope == pytest.approx(expected_slope)
    assert indicator.indicator == pytest.approx(expected_indicator)


def test_loss_slope_controller_prefers_decrease_over_flat_over_increase():
    controller = LossSlopeMixtureController(
        run_id="run",
        dataset_ids=(0, 1, 2),
        initial_weights=(1.0 / 3.0,) * 3,
        window_iters=3,
        total_windows=2,
        eta=1.0,
        uniform_smoothing=0.5,
    )
    result = controller.process_window(
        {
            0: _complete_window(0, (3.0, 2.0, 1.0)).finalize(),
            1: _complete_window(1, (2.0, 2.0, 2.0)).finalize(),
            2: _complete_window(2, (1.0, 2.0, 3.0)).finalize(),
        }
    )

    assert result.decision is not None
    decreasing, flat, increasing = result.decision.weights
    assert decreasing > flat > increasing


def test_loss_slope_missing_domain_iteration_creates_no_fake_point_and_holds():
    states = {
        dataset_id: LossSlopeDomainWindowState(
            dataset_id=dataset_id,
            window_id=1,
            window_start_iteration=1,
            window_end_iteration=3,
            terminal_window=False,
        )
        for dataset_id in (0, 1)
    }
    for iteration in (1, 2, 3):
        states[0].add_iteration_samples(
            iteration, [_sample(iteration, 0, 0, 2.0 - iteration * 0.1, 8)]
        )
        states[1].add_iteration_samples(
            iteration,
            []
            if iteration == 2
            else [_sample(iteration, 0, 1, 2.0 - iteration * 0.2, 8)],
        )
    indicators = {key: state.finalize() for key, state in states.items()}
    assert indicators[1].iteration_point_count == 2
    assert indicators[1].status == "INCOMPLETE"
    assert "missing_loss_iterations=2" in indicators[1].incomplete_reason

    controller = LossSlopeMixtureController(
        run_id="run",
        dataset_ids=(0, 1),
        initial_weights=(0.5, 0.5),
        window_iters=3,
        total_windows=2,
    )
    result = controller.process_window(indicators)
    assert result.decision is not None
    assert result.decision.decision_type == "HOLD_INCOMPLETE"
    assert result.decision.weights == pytest.approx((0.5, 0.5))


def test_loss_slope_controller_applies_exact_selected_update_and_stops_at_terminal():
    controller = LossSlopeMixtureController(
        run_id="run",
        dataset_ids=(0, 1),
        initial_weights=(0.5, 0.5),
        window_iters=3,
        total_windows=2,
        eta=2.0,
        uniform_smoothing=0.5,
    )
    indicators = {
        0: _complete_window(0, (2.0, 1.9, 1.8)).finalize(),
        1: _complete_window(1, (2.0, 1.6, 1.2)).finalize(),
    }
    result = controller.process_window(indicators)
    assert result.decision is not None
    logits = (math.log(0.5) + 2.0 * 0.1, math.log(0.5) + 2.0 * 0.4)
    exponentials = [math.exp(value - max(logits)) for value in logits]
    alpha = [value / sum(exponentials) for value in exponentials]
    expected = tuple(0.5 * value + 0.25 for value in alpha)
    assert result.decision.decision_type == "UPDATE"
    assert result.decision.weights == pytest.approx(expected)

    terminal = {
        dataset_id: _complete_window(
            dataset_id,
            (1.5, 1.4, 1.3),
            window_id=2,
            terminal=True,
        ).finalize()
        for dataset_id in (0, 1)
    }
    terminal_result = controller.process_window(terminal)
    assert terminal_result.terminal_window
    assert terminal_result.decision is None


def test_loss_slope_serialized_state_rejects_mode_and_parameter_mismatch():
    state = _complete_window(0, (3.0, 2.8, 2.6))
    restored = LossSlopeDomainWindowState.from_state_dict(state.state_dict())
    assert [item.to_dict() for item in restored.iteration_metrics] == [
        item.to_dict() for item in state.iteration_metrics
    ]

    controller = LossSlopeMixtureController(
        run_id="run",
        dataset_ids=(0, 1),
        initial_weights=(0.5, 0.5),
        window_iters=3,
        total_windows=2,
    )
    checkpoint = controller.state_dict()
    wrong_mode = dict(checkpoint, controller_mode="loss_pathway")
    with pytest.raises(ValueError, match="mode mismatch"):
        controller.load_state_dict(wrong_mode)

    different_eta = LossSlopeMixtureController(
        run_id="run",
        dataset_ids=(0, 1),
        initial_weights=(0.5, 0.5),
        window_iters=3,
        total_windows=2,
        eta=499.0,
    )
    with pytest.raises(ValueError, match="configuration mismatch"):
        different_eta.load_state_dict(checkpoint)
