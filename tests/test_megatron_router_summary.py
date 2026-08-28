from __future__ import annotations

import pytest
import torch

from dmi_megatron_integration.hooks.megatron_router_summary import (
    expert_token_count_from_routing_map,
    router_probs_mean_from_logits,
    router_probs_mean_from_segmented_probabilities,
    router_token_entropy_mean_from_logits,
)


def test_router_probs_mean_softmax_ignores_suffix_padding():
    logits = torch.tensor(
        [
            [[2.0, 0.0], [0.0, 2.0]],
            [[0.0, 2.0], [2.0, 0.0]],
            [[100.0, -100.0], [100.0, -100.0]],
        ]
    )
    valid_count = torch.tensor([2, 1])

    out = router_probs_mean_from_logits(logits, valid_count, "softmax")

    expected0 = torch.softmax(logits[:2, 0], dim=-1, dtype=torch.float32).mean(dim=0)
    expected1 = torch.softmax(logits[:1, 1], dim=-1, dtype=torch.float32).mean(dim=0)
    expected = torch.stack([expected0, expected1], dim=0)
    torch.testing.assert_close(out, expected)


def test_router_probs_mean_sigmoid_normalizes_and_clamps_zero_count():
    logits = torch.tensor(
        [
            [[0.0, 1.0, 2.0], [5.0, 5.0, 5.0]],
            [[2.0, 1.0, 0.0], [9.0, 9.0, 9.0]],
        ]
    )
    valid_count = torch.tensor([2, 0])

    out = router_probs_mean_from_logits(logits, valid_count, "sigmoid")

    probs = torch.sigmoid(logits[:, 0].float())
    probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-20)
    expected = torch.stack([probs.mean(dim=0), torch.zeros(3)], dim=0)
    torch.testing.assert_close(out, expected)


def test_router_probs_mean_rejects_bad_shapes_and_score_function():
    with pytest.raises(ValueError, match=r"\[S, B, E\]"):
        router_probs_mean_from_logits(torch.zeros(2, 3), torch.ones(3, dtype=torch.long), "softmax")

    with pytest.raises(ValueError, match=r"\[B\]"):
        router_probs_mean_from_logits(torch.zeros(2, 3, 4), torch.ones(3, 1), "softmax")

    with pytest.raises(ValueError, match="batch dimension"):
        router_probs_mean_from_logits(torch.zeros(2, 3, 4), torch.ones(2, dtype=torch.long), "softmax")

    with pytest.raises(NotImplementedError, match="Unsupported"):
        router_probs_mean_from_logits(torch.zeros(2, 3, 4), torch.ones(3, dtype=torch.long), "sinkhorn")


def test_router_probs_mean_from_segmented_probabilities_uses_active_prefix():
    probabilities = torch.tensor(
        [
            [0.8, 0.2],
            [0.4, 0.6],
            [0.1, 0.9],
            [0.3, 0.7],
            [0.9, 0.1],
        ],
        dtype=torch.float32,
    )
    starts = torch.tensor([0, 3, 5], dtype=torch.int64)
    ends = torch.tensor([2, 5, 5], dtype=torch.int64)

    means, active_count = router_probs_mean_from_segmented_probabilities(
        probabilities,
        starts,
        ends,
    )

    torch.testing.assert_close(
        means,
        torch.tensor([[0.6, 0.4], [0.6, 0.4], [0.0, 0.0]]),
    )
    torch.testing.assert_close(active_count, torch.tensor([2], dtype=torch.int64))


def test_router_token_entropy_mean_uses_token_level_entropy():
    logits = torch.tensor(
        [
            [[3.0, 0.0], [0.0, 0.0]],
            [[0.0, 3.0], [100.0, -100.0]],
            [[100.0, -100.0], [100.0, -100.0]],
        ]
    )
    valid_count = torch.tensor([2, 1])

    out = router_token_entropy_mean_from_logits(logits, valid_count, "softmax")

    probs0 = torch.softmax(logits[:2, 0], dim=-1, dtype=torch.float32)
    probs1 = torch.softmax(logits[:1, 1], dim=-1, dtype=torch.float32)
    expected0 = -(probs0 * probs0.clamp_min(1e-20).log()).sum(dim=-1).mean()
    expected1 = -(probs1 * probs1.clamp_min(1e-20).log()).sum(dim=-1).mean()
    expected = torch.tensor([[expected0], [expected1]])
    torch.testing.assert_close(out, expected)


def test_expert_token_count_from_routing_map_returns_per_sample_counts():
    # Shape is flattened from [S, B, E], matching Megatron router.routing().
    # S=3, B=2, E=3.
    routing_map = torch.tensor(
        [
            [1, 0, 1],  # s0 b0
            [0, 1, 1],  # s0 b1
            [0, 1, 1],  # s1 b0
            [1, 0, 0],  # s1 b1, invalid for b1
            [1, 0, 0],  # s2 b0, invalid for b0
            [0, 0, 1],  # s2 b1, invalid for b1
        ],
        dtype=torch.bool,
    )
    valid_count = torch.tensor([2, 1])

    out = expert_token_count_from_routing_map(
        routing_map,
        valid_count,
        seq_length=3,
        batch_size=2,
    )

    expected = torch.tensor(
        [
            [1, 1, 2],
            [0, 1, 1],
        ],
        dtype=torch.int64,
    )
    torch.testing.assert_close(out, expected)
