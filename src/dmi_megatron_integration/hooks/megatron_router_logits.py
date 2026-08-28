"""Layout-only preprocessing for raw Megatron router-logit capture."""

from __future__ import annotations

import torch


def router_logits_by_sample(logits: torch.Tensor) -> torch.Tensor:
    """Return raw ``[S, B, E]`` router logits as ``[B, S, E]`` records."""

    if logits.dim() != 3:
        raise ValueError(f"Expected router logits [S, B, E], got {tuple(logits.shape)}")
    return logits.transpose(0, 1)


__all__ = ["router_logits_by_sample"]
