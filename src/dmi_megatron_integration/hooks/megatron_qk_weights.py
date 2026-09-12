"""GPU preprocessing for Megatron Q/K projection-weight hooks."""

from __future__ import annotations

import torch


def qk_weight_from_fused_qkv(
    weight: torch.Tensor,
    *,
    num_query_groups: int,
    query_rows_per_group: int,
    head_dim: int,
    projection: str,
) -> torch.Tensor:
    """Select Q or K from local grouped QKV, without moving data off its device."""
    if projection not in ("q", "k"):
        raise ValueError(f"Unsupported Q/K projection: {projection!r}")
    hidden_size = int(weight.shape[1])
    grouped = weight.detach().view(
        num_query_groups,
        query_rows_per_group + 2 * head_dim,
        hidden_size,
    )
    start = 0 if projection == "q" else query_rows_per_group
    width = query_rows_per_group if projection == "q" else head_dim
    # Copy only the selected projection, including for a single KV group.
    # Production weights are CUDA-resident, so this is a contiguous GPU copy.
    return grouped[:, start : start + width, :].clone(
        memory_format=torch.contiguous_format
    ).view(num_query_groups * width, hidden_size)
