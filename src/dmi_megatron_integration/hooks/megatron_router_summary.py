"""Preprocessing helpers for Megatron router summary hooks."""

from __future__ import annotations

import torch


def _validate_logits_and_counts(
    logits: torch.Tensor,
    valid_count: torch.Tensor,
) -> tuple[int, int, int, torch.Tensor]:
    if logits.dim() != 3:
        raise ValueError(f"Expected router logits [S, B, E], got {tuple(logits.shape)}")
    if valid_count.dim() != 1:
        raise ValueError(f"Expected valid_count [B], got {tuple(valid_count.shape)}")
    if logits.shape[1] != valid_count.shape[0]:
        raise ValueError(
            "Router logits batch dimension must match valid_count: "
            f"{logits.shape[1]} != {valid_count.shape[0]}"
        )
    seq_len, batch, experts = (int(logits.shape[0]), int(logits.shape[1]), int(logits.shape[2]))
    counts = valid_count.to(device=logits.device, dtype=torch.long)
    return seq_len, batch, experts, counts


def _router_probs_from_logits(logits: torch.Tensor, score_function: str) -> torch.Tensor:
    if score_function == "softmax":
        return torch.softmax(logits, dim=-1, dtype=torch.float32)
    if score_function == "sigmoid":
        probs = torch.sigmoid(logits.float())
        return probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-20)
    raise NotImplementedError(f"Unsupported Megatron router score function: {score_function}")


def _valid_mask(seq_len: int, counts: torch.Tensor) -> torch.Tensor:
    seq = torch.arange(seq_len, device=counts.device)[:, None]
    return seq < counts[None, :]


def router_probs_mean_from_logits(
    logits: torch.Tensor,
    valid_count: torch.Tensor,
    score_function: str,
) -> torch.Tensor:
    """Return per-sample mean full-router probabilities.

    Args:
        logits: Dense router logits with shape ``[S, B, E]``.
        valid_count: Number of valid low-S-prefix tokens per sample, shape ``[B]``.
        score_function: Megatron router score function, ``"softmax"`` or ``"sigmoid"``.
    """

    seq_len, _, _, counts = _validate_logits_and_counts(logits, valid_count)
    probs = _router_probs_from_logits(logits, score_function)
    valid = _valid_mask(seq_len, counts)
    masked = probs * valid.unsqueeze(-1).to(probs.dtype)
    denom = counts.clamp_min(1).to(dtype=probs.dtype).unsqueeze(-1)
    return masked.sum(dim=0) / denom


def router_probs_mean_from_segmented_probabilities(
    probabilities: torch.Tensor,
    sample_start_ptr: torch.Tensor,
    sample_end_ptr: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Average full router probabilities over packed logical conversations."""

    if probabilities.dim() != 2:
        raise ValueError(
            f"Expected packed router probabilities [T, E], got {tuple(probabilities.shape)}"
        )
    starts = sample_start_ptr.reshape(-1)
    ends = sample_end_ptr.reshape(-1)
    offsets = torch.stack((starts, ends), dim=-1).reshape(-1)
    probs = probabilities.detach().to(dtype=torch.float32)
    segment_sums = torch.segment_reduce(
        probs,
        "sum",
        offsets=offsets,
        axis=0,
        unsafe=True,
    )[::2]
    counts = (ends - starts).to(dtype=probs.dtype)
    means = segment_sums / counts.clamp_min(1).unsqueeze(-1)
    active_count = (ends > starts).sum(dtype=torch.int64).reshape(1)
    return means, active_count


def router_token_entropy_mean_from_logits(
    logits: torch.Tensor,
    valid_count: torch.Tensor,
    score_function: str,
) -> torch.Tensor:
    """Return per-sample mean token-level router entropy as ``[B, 1]``."""

    seq_len, _, _, counts = _validate_logits_and_counts(logits, valid_count)
    probs = _router_probs_from_logits(logits, score_function)
    entropy = -(probs * probs.clamp_min(1e-20).log()).sum(dim=-1)
    valid = _valid_mask(seq_len, counts)
    masked = entropy * valid.to(entropy.dtype)
    denom = counts.clamp_min(1).to(dtype=entropy.dtype)
    return (masked.sum(dim=0) / denom).unsqueeze(-1)


def expert_token_count_from_routing_map(
    routing_map: torch.Tensor,
    valid_count: torch.Tensor,
    *,
    seq_length: int,
    batch_size: int,
) -> torch.Tensor:
    """Return per-sample valid-token expert assignment counts as ``[B, E]``.

    ``routing_map`` is Megatron's dense boolean/int map with shape
    ``[S * B, E]``.  ``valid_count`` follows the DMI convention: valid tokens are
    the low-sequence prefix for each sample.
    """

    if routing_map.dim() != 2:
        raise ValueError(f"Expected routing_map [S*B, E], got {tuple(routing_map.shape)}")
    if valid_count.dim() != 1:
        raise ValueError(f"Expected valid_count [B], got {tuple(valid_count.shape)}")
    if int(valid_count.shape[0]) != int(batch_size):
        raise ValueError(
            f"valid_count batch dimension must match batch_size: {valid_count.shape[0]} != {batch_size}"
        )
    expected_tokens = int(seq_length) * int(batch_size)
    if int(routing_map.shape[0]) != expected_tokens:
        raise ValueError(
            "routing_map token dimension must match seq_length * batch_size: "
            f"{routing_map.shape[0]} != {expected_tokens}"
        )

    counts = valid_count.to(device=routing_map.device, dtype=torch.long)
    experts = int(routing_map.shape[1])
    routing_by_sample = routing_map.reshape(int(seq_length), int(batch_size), experts)
    valid = _valid_mask(int(seq_length), counts)
    masked = routing_by_sample.to(torch.int64) * valid.unsqueeze(-1).to(torch.int64)
    return masked.sum(dim=0)


__all__ = [
    "expert_token_count_from_routing_map",
    "router_probs_mean_from_logits",
    "router_probs_mean_from_segmented_probabilities",
    "router_token_entropy_mean_from_logits",
]
