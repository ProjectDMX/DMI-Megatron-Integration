from __future__ import annotations

import argparse

import pytest
import torch

import tools.materialization.clickhouse as materializer
from tools.materialization.metrics import (
    levenshtein,
    pairwise_distance_sum_blockwise,
)


def _args(**overrides) -> argparse.Namespace:
    values = {
        "model_id": "training-run",
        "run_name": "analysis",
        "phase": "train",
        "expected_train_iters": 2,
        "expected_layer_count": 2,
        "expected_expert_count": 2,
        "expected_samples_per_iteration": 2,
        "pathway_threshold": 0.7,
        "consistency_eps": 1e-8,
        "pathway_window_size": [1],
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_materializer_rejects_payload_iteration_without_status() -> None:
    valid_identity = (-1, -1, -1, -1, -1, 0, 1, 0, -1)

    class FakeClient:
        def __init__(self):
            self.responses = [
                [(1, 0, 1, *valid_identity)],
                [(1,), (2,)],
                [],
                [],
            ]

        def execute(self, _query, _params, settings=None):
            assert settings == {"strings_as_bytes": 1}
            return self.responses.pop(0)

    args = _args(raw_db="raw", raw_table="training_payload")
    with pytest.raises(RuntimeError, match=r"no accepted attempt status: \[2\]"):
        materializer._read_accepted_attempts(FakeClient(), args)


def test_blockwise_edit_distance_matches_direct_pair_traversal() -> None:
    pathways = [
        (0, 1, 1_000_001, 2),
        (0, 2, 1_000_001, 2),
        (3, 2, 1_000_001, 4),
        (3, 2, 1_000_001, 4, 5),
        (0, 1, 1_000_001, 2),
    ]
    expected = sum(
        levenshtein(pathways[left], pathways[right])
        for left in range(len(pathways))
        for right in range(left + 1, len(pathways))
    )

    total, pair_count = pairwise_distance_sum_blockwise(pathways, block_size=2)
    assert total == expected
    assert pair_count == 10
    assert pairwise_distance_sum_blockwise(pathways, block_size=1) == (expected, 10)
    assert pairwise_distance_sum_blockwise(pathways, block_size=64) == (expected, 10)


def test_iteration_loss_materialization_is_exactly_token_weighted() -> None:
    args = _args()
    float_rows = [
        ("lm_per_sample_loss", "fwd", "train", 1, 0, 0, 0, -1, 0, 0, 1, 0, 0, 0, 2.0),
        ("lm_per_sample_loss", "fwd", "train", 1, 0, 0, 1, -1, 0, 0, 1, 0, 0, 0, 4.0),
        ("grad_norm", "iter", "train", 1, -1, -1, -1, -1, -1, 0, 1, 0, 0, -1, 3.5),
    ]
    int_rows = [
        (
            "lm_per_sample_loss_token_count",
            "fwd",
            "train",
            1,
            0,
            0,
            0,
            -1,
            0,
            0,
            1,
            0,
            0,
            0,
            1,
        ),
        (
            "lm_per_sample_loss_token_count",
            "fwd",
            "train",
            1,
            0,
            0,
            1,
            -1,
            0,
            0,
            1,
            0,
            0,
            0,
            3,
        ),
    ]

    loss_rows = materializer._materialize_iteration_loss_from_samples(
        float_rows, int_rows, args
    )
    grad_rows = materializer._materialize_direct_iteration_scalars(float_rows, args)

    assert loss_rows[0][5:] == ("lm_loss_iteration", 3.5)
    assert grad_rows[0][5:] == ("grad_norm", 3.5)


def test_iteration_loss_rejects_duplicate_rerun_coordinates() -> None:
    args = _args()
    mean = (
        "lm_per_sample_loss",
        "fwd",
        "train",
        1,
        0,
        0,
        0,
        -1,
        0,
        0,
        1,
        0,
        0,
        0,
        2.0,
    )
    count = (
        "lm_per_sample_loss_token_count",
        "fwd",
        "train",
        1,
        0,
        0,
        0,
        -1,
        0,
        0,
        1,
        0,
        0,
        0,
        2,
    )

    with pytest.raises(RuntimeError, match="Duplicate lm_per_sample_loss raw row"):
        materializer._materialize_iteration_loss_from_samples(
            [mean, mean], [count], args
        )


def test_expert_load_materialization_derives_counts_and_summary() -> None:
    args = _args()
    counts = {
        ("pre_drop_token_count", "train", 1, 0, 0): torch.tensor([6, 2]),
        ("post_drop_token_count", "train", 1, 0, 0): torch.tensor([4, 0]),
    }

    load_rows, summary_rows = materializer._materialize_expert_load(counts, args)

    assert [row[7:] for row in load_rows] == [
        ("pre_drop", 6),
        ("pre_drop", 2),
        ("post_drop", 4),
        ("post_drop", 0),
        ("dropped", 2),
        ("dropped", 2),
    ]
    assert summary_rows[0][6:] == (1, 1, 1.0, 1.0, 0.5)


def test_router_entropy_materialization_averages_samples() -> None:
    args = _args()
    rows = [
        ("router_token_entropy_mean", "fwd", "train", 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0.5),
        ("router_token_entropy_mean", "fwd", "train", 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1.5),
    ]

    result = materializer._materialize_router_entropy(rows, args)

    assert result == [("training-run", "analysis", "train", 1, 1, 0, 1.0)]


def test_window_20_materialization_grows_then_slides() -> None:
    args = _args(expected_train_iters=21, pathway_window_size=[1, 20])
    consistency = {
        (0, iteration): [float(iteration), float(iteration) + 0.5]
        for iteration in range(1, 22)
    }
    pathways = {
        (0, iteration): [(iteration, 1_000_001), (iteration + 100, 1_000_001)]
        for iteration in range(1, 22)
    }

    consistency_rows, distance_rows = materializer._materialize_pathway_windows(
        args,
        completed_iteration_ids=range(1, 22),
        consistency_by_dataset_iteration=consistency,
        pathways_by_dataset_iteration=pathways,
        window_sizes=(1, 20),
    )
    assert len(consistency_rows) == 42
    assert len(distance_rows) == 42

    rolling_consistency = consistency_rows[21:]
    rolling_distance = distance_rows[21:]
    assert rolling_consistency[0][3:10] == (1, 0, 20, 1, 1, 1, 2)
    assert rolling_distance[0][3:11] == (1, 0, 20, 1, 1, 1, 2, 1)
    assert rolling_consistency[19][3:10] == (20, 0, 20, 20, 1, 20, 40)
    assert rolling_distance[19][3:11] == (20, 0, 20, 20, 1, 20, 40, 780)
    assert rolling_consistency[20][3:10] == (21, 0, 20, 20, 2, 21, 40)
    assert rolling_distance[20][3:11] == (21, 0, 20, 20, 2, 21, 40, 780)


def test_pathway_windows_never_construct_cross_dataset_pairs() -> None:
    args = _args(expected_train_iters=2, pathway_window_size=[2])
    consistency = {
        (0, 1): [0.1],
        (1, 1): [0.2],
        (0, 2): [0.3],
    }
    pathways = {
        (0, 1): [(0, 1_000_001)],
        (1, 1): [(5, 1_000_001)],
        (0, 2): [(1, 1_000_001)],
    }

    consistency_rows, distance_rows = materializer._materialize_pathway_windows(
        args,
        completed_iteration_ids=(1, 2),
        consistency_by_dataset_iteration=consistency,
        pathways_by_dataset_iteration=pathways,
        window_sizes=(2,),
    )

    by_key = {(row[3], row[4]): row for row in distance_rows}
    assert set(by_key) == {(1, 0), (1, 1), (2, 0), (2, 1)}
    assert by_key[(1, 0)][10:13] == (0, 0.0, None)
    assert by_key[(1, 1)][10:13] == (0, 0.0, None)
    assert by_key[(2, 0)][9:11] == (2, 1)
    assert by_key[(2, 1)][9:13] == (1, 0, 0.0, None)

    consistency_by_key = {(row[3], row[4]): row for row in consistency_rows}
    assert consistency_by_key[(2, 1)][9] == 1
    assert consistency_by_key[(2, 1)][10:12] == (0.2, 0.2)


def test_router_probability_reader_rejects_dataset_mismatch_across_layers() -> None:
    args = _args(
        raw_db="raw",
        raw_table="payload",
        expected_train_iters=1,
        expected_samples_per_iteration=1,
    )
    tensor = torch.tensor([0.8, 0.2], dtype=torch.float32)
    encoded = (b"torch.float", [2], tensor.numpy().tobytes())
    rows = [
        ("fwd", "train", 1, 0, 0, 0, 0, 0, 0, 0, 3, *encoded),
        ("fwd", "train", 1, 0, 0, 0, 1, 0, 0, 0, 4, *encoded),
    ]

    class FakeClient:
        def execute(self, _query, _params, settings=None):
            assert settings == {"strings_as_bytes": 1}
            return rows

    with pytest.raises(RuntimeError, match="inconsistent dataset IDs across layers"):
        materializer._read_router_probability_samples(FakeClient(), args, [1])


def test_pathways_join_iteration_to_latest_prior_weight_state(monkeypatch) -> None:
    args = _args()
    sample = torch.tensor([[0.8, 0.2], [0.3, 0.7]], dtype=torch.float32)
    samples = {
        (0, 1): [((0, 0, 0, 0), sample), ((0, 1, 0, 0), sample.flip(-1))],
        (0, 2): [((0, 0, 0, 0), sample), ((0, 1, 0, 0), sample.flip(-1))],
    }
    requested_boundaries = []
    requested_states = []

    monkeypatch.setattr(
        materializer,
        "_read_router_probability_samples",
        lambda _client, _args, _accepted_ids: samples,
    )

    def fake_latest_state(_client, _args, *, before_iteration):
        requested_boundaries.append(before_iteration)
        return 0

    def fake_weights(_client, _args, *, state_id, accepted_attempts):
        assert accepted_attempts == {1: 0, 2: 0}
        requested_states.append(state_id)
        return {0: torch.eye(2), 1: torch.eye(2)}

    monkeypatch.setattr(
        materializer, "_latest_router_weight_state_id", fake_latest_state
    )
    monkeypatch.setattr(materializer, "_read_router_weight_state", fake_weights)

    consistency_rows, distance_rows = materializer._materialize_pathways(
        object(), args, {1: 0, 2: 0}
    )

    assert requested_boundaries == [1, 2]
    assert requested_states == [0, 0]
    assert len(consistency_rows) == 2
    assert len(distance_rows) == 2
    assert all(row[9] == 2 for row in consistency_rows)
    assert all(row[10] == 1 for row in distance_rows)


def test_materializer_refuses_raw_database_as_processed_database() -> None:
    with pytest.raises(ValueError, match="must be different"):
        materializer.main(
            [
                "--raw-db",
                "same_db",
                "--processed-db",
                "same_db",
                "--model-id",
                "model",
                "--run-name",
                "run",
                "--expected-train-iters",
                "1",
            ]
        )
