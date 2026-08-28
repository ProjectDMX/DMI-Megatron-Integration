"""Megatron per-sample loss summaries for DMI training monitoring."""

from __future__ import annotations

import torch


def _segmented_sum(
    values: torch.Tensor,
    starts: torch.Tensor,
    ends: torch.Tensor,
) -> torch.Tensor:
    offsets = torch.stack((starts.reshape(-1), ends.reshape(-1)), dim=-1).reshape(-1)
    return torch.segment_reduce(
        values,
        "sum",
        offsets=offsets,
        axis=0,
        unsafe=True,
    )[::2]


def per_sample_loss_from_token_loss(
    output_tensor: torch.Tensor,
    loss_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return per-sample LM loss means and exact loss-token counts.

    Stage-1 loss monitoring intentionally supports only dense Megatron
    per-token loss tensors. Packed/THD, sequence-parallel tuple loss, and
    ModelOpt tuple loss layouts must fail clearly instead of silently producing
    misleading scalar rows.
    """

    if isinstance(output_tensor, tuple):
        raise NotImplementedError(
            "DMI loss-summary only supports dense [B, S] per-token loss tensors; "
            "tuple loss tensors are not supported yet"
        )
    if output_tensor.dim() != 2 or loss_mask.dim() != 2:
        raise ValueError(
            "DMI loss-summary only supports dense [B, S] per-token loss tensors "
            "and [B, S] loss masks"
        )
    if tuple(output_tensor.shape) != tuple(loss_mask.shape):
        raise ValueError(
            "DMI loss-summary output_tensor/loss_mask shape mismatch: "
            f"{tuple(output_tensor.shape)} != {tuple(loss_mask.shape)}"
        )

    token_loss = output_tensor.float()
    mask = loss_mask.float()
    loss_sum = (token_loss * mask).sum(dim=1)
    token_count = mask.sum(dim=1)
    loss_mean = loss_sum / token_count.clamp_min(1)
    return loss_mean[:, None], token_count.to(torch.int64)[:, None]


def per_segment_loss_from_token_loss(
    output_tensor: torch.Tensor,
    loss_mask: torch.Tensor,
    sample_start_ptr: torch.Tensor,
    sample_end_ptr: torch.Tensor,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Return target-token loss summaries for packed logical conversations."""

    if isinstance(output_tensor, tuple):
        raise NotImplementedError("DMI segmented loss does not support tuple loss tensors")
    if output_tensor.dim() != 2 or loss_mask.dim() != 2:
        raise ValueError("DMI segmented loss requires [B, S] loss and mask tensors")
    if tuple(output_tensor.shape) != tuple(loss_mask.shape):
        raise ValueError(
            "DMI segmented loss/mask shape mismatch: "
            f"{tuple(output_tensor.shape)} != {tuple(loss_mask.shape)}"
        )

    starts = sample_start_ptr.reshape(-1)
    ends = sample_end_ptr.reshape(-1)
    token_loss = output_tensor.detach().reshape(-1).float()
    mask = loss_mask.detach().reshape(-1).float()
    loss_sum = _segmented_sum(token_loss * mask, starts, ends)
    token_count = _segmented_sum(mask, starts, ends)
    loss_mean = loss_sum / token_count.clamp_min(1)
    active_count = (ends > starts).sum(dtype=torch.int64).reshape(1)
    return [
        (loss_mean[:, None], active_count),
        (token_count.to(torch.int64)[:, None], active_count),
    ]


__all__ = [
    "per_sample_loss_from_token_loss",
    "per_segment_loss_from_token_loss",
]
