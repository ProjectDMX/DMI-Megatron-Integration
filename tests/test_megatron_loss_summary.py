from __future__ import annotations

import pytest
import torch

from dmi_megatron_integration.hooks.megatron_loss_summary import (
    per_sample_loss_from_token_loss,
    per_segment_loss_from_token_loss,
)


def test_per_sample_loss_from_token_loss_masks_and_reduces():
    output_tensor = torch.tensor(
        [
            [1.0, 3.0, 100.0],
            [2.0, 4.0, 6.0],
        ],
        dtype=torch.float32,
    )
    loss_mask = torch.tensor(
        [
            [1.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )

    loss_mean, token_count = per_sample_loss_from_token_loss(output_tensor, loss_mask)

    torch.testing.assert_close(loss_mean, torch.tensor([[2.0], [4.0]]))
    torch.testing.assert_close(token_count, torch.tensor([[2], [2]], dtype=torch.int64))


def test_per_sample_loss_from_token_loss_rejects_non_dense_layout():
    output_tensor = torch.ones(2, 3, 4)
    loss_mask = torch.ones(2, 3)

    with pytest.raises(ValueError, match=r"dense \[B, S\]"):
        per_sample_loss_from_token_loss(output_tensor, loss_mask)


def test_per_sample_loss_from_token_loss_rejects_tuple_layout():
    with pytest.raises(NotImplementedError, match="tuple loss tensors"):
        per_sample_loss_from_token_loss((torch.ones(2, 3),), torch.ones(2, 3))  # type: ignore[arg-type]


def test_per_segment_loss_from_token_loss_uses_logical_conversation_ranges():
    output_tensor = torch.tensor(
        [[1.0, 3.0, 100.0, 5.0, 7.0, 200.0]],
        dtype=torch.float32,
    )
    loss_mask = torch.tensor(
        [[1.0, 1.0, 0.0, 1.0, 0.0, 0.0]],
        dtype=torch.float32,
    )
    starts = torch.tensor([0, 3, 5, 5], dtype=torch.int64)
    ends = torch.tensor([2, 5, 5, 5], dtype=torch.int64)

    outputs = per_segment_loss_from_token_loss(
        output_tensor,
        loss_mask,
        starts,
        ends,
    )

    (loss_mean, mean_rows), (token_count, count_rows) = outputs
    torch.testing.assert_close(
        loss_mean,
        torch.tensor([[2.0], [5.0], [0.0], [0.0]]),
    )
    torch.testing.assert_close(
        token_count,
        torch.tensor([[2], [1], [0], [0]], dtype=torch.int64),
    )
    torch.testing.assert_close(mean_rows, torch.tensor([2], dtype=torch.int64))
    torch.testing.assert_close(count_rows, torch.tensor([2], dtype=torch.int64))
