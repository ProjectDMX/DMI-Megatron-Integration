from __future__ import annotations

import pytest
import torch

from dmi_megatron_integration.hooks.megatron_vocab_logits import (
    vocab_logits_by_sample,
    vocab_logits_topk_by_sample,
)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_vocab_logits_by_sample_preserves_values_dtype_and_detaches(dtype):
    logits = torch.arange(2 * 3 * 5, dtype=torch.float32).reshape(2, 3, 5).to(dtype)
    logits.requires_grad_(True)

    captured = vocab_logits_by_sample(logits)

    assert captured.shape == (3, 2, 5)
    assert captured.dtype is dtype
    assert captured.requires_grad is False
    assert torch.equal(captured, logits.detach().transpose(0, 1))


def test_vocab_logits_by_sample_rejects_non_rank_three_input():
    with pytest.raises(ValueError, match=r"Expected vocabulary logits \[S, B, V\]"):
        vocab_logits_by_sample(torch.zeros(2, 3))


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_vocab_logits_topk_by_sample_matches_torch_topk(dtype):
    logits = torch.tensor(
        [
            [[-4.0, 2.0, 2.0, 9.0, 1.0], [3.0, -1.0, 8.0, 0.0, 7.0]],
            [[5.0, 4.0, 3.0, 2.0, 1.0], [-5.0, -2.0, -3.0, -1.0, -4.0]],
        ],
        dtype=dtype,
        requires_grad=True,
    )

    values, indices = vocab_logits_topk_by_sample(logits, k=3)
    expected_values, expected_indices = torch.topk(logits.detach(), 3, dim=-1)

    assert values.shape == (2, 2, 3)
    assert indices.shape == (2, 2, 3)
    assert values.dtype is dtype
    assert indices.dtype is torch.int32
    assert values.requires_grad is False
    assert indices.requires_grad is False
    assert torch.equal(values, expected_values.transpose(0, 1))
    assert torch.equal(indices, expected_indices.to(torch.int32).transpose(0, 1))
    assert torch.all(values[..., :-1] >= values[..., 1:])


@pytest.mark.parametrize("k", [1, 5])
def test_vocab_logits_topk_by_sample_supports_boundary_k(k):
    logits = torch.arange(10, dtype=torch.float32).reshape(1, 2, 5)

    values, indices = vocab_logits_topk_by_sample(logits, k=k)

    assert values.shape == (2, 1, k)
    assert indices.shape == (2, 1, k)


def test_local_topk_indices_and_shard_rank_reconstruct_global_topk():
    tp_world, local_vocab_size, k = 8, 16, 4
    vocab_size = tp_world * local_vocab_size
    generator = torch.Generator().manual_seed(1234)
    logits = torch.stack(
        [torch.randperm(vocab_size, generator=generator) for _ in range(6)]
    ).reshape(3, 2, vocab_size).to(torch.float32)
    candidate_values, candidate_ids = [], []
    for shard_rank, shard in enumerate(logits.chunk(tp_world, dim=-1)):
        values, indices = vocab_logits_topk_by_sample(shard, k=k)
        candidate_values.append(values)
        candidate_ids.append(indices + shard_rank * local_vocab_size)

    # Storage-side reconstruction from only local candidates and TP coordinates.
    values = torch.cat(candidate_values, dim=-1)
    ids = torch.cat(candidate_ids, dim=-1)
    merged_values, positions = values.topk(k, dim=-1)
    merged_ids = ids.gather(-1, positions)
    expected_values, expected_ids = logits.transpose(0, 1).topk(k, dim=-1)
    assert torch.equal(merged_values, expected_values)
    assert torch.equal(merged_ids, expected_ids.to(torch.int32))


@pytest.mark.parametrize("k", [0, -1, 6])
def test_vocab_logits_topk_by_sample_rejects_out_of_range_k(k):
    with pytest.raises(ValueError, match="must satisfy"):
        vocab_logits_topk_by_sample(torch.zeros(1, 2, 5), k=k)


@pytest.mark.parametrize("k", [True, 1.5, "1"])
def test_vocab_logits_topk_by_sample_rejects_non_integer_k(k):
    with pytest.raises(TypeError, match="must be an integer"):
        vocab_logits_topk_by_sample(torch.zeros(1, 2, 5), k=k)


def test_vocab_logits_topk_by_sample_rejects_non_rank_three_input():
    with pytest.raises(ValueError, match=r"Expected vocabulary logits \[S, B, V\]"):
        vocab_logits_topk_by_sample(torch.zeros(2, 5), k=1)
