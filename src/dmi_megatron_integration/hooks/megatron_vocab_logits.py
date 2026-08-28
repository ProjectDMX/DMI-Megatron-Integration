"""Preprocessing for Megatron language-model vocabulary-logit capture."""

from __future__ import annotations

import torch


def vocab_logits_by_sample(logits: torch.Tensor) -> torch.Tensor:
    """Return detached ``[S, B, V]`` LM logits as ``[B, S, V]`` records."""

    if logits.dim() != 3:
        raise ValueError(f"Expected vocabulary logits [S, B, V], got {tuple(logits.shape)}")
    return logits.detach().transpose(0, 1)


def vocab_logits_topk_by_sample(
    logits: torch.Tensor,
    *,
    k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return sorted vocabulary top-K values and int32 indices as ``[B, S, K]``."""

    if logits.dim() != 3:
        raise ValueError(f"Expected vocabulary logits [S, B, V], got {tuple(logits.shape)}")
    if isinstance(k, bool) or not isinstance(k, int):
        raise TypeError("Vocabulary-logit top-K must be an integer")
    vocab_size = int(logits.shape[-1])
    if k < 1 or k > vocab_size:
        raise ValueError(
            f"Vocabulary-logit top-K must satisfy 1 <= K <= {vocab_size}; got {k}"
        )
    values, indices = torch.topk(
        logits.detach(),
        k=k,
        dim=-1,
        largest=True,
        sorted=True,
    )
    return values.transpose(0, 1), indices.to(torch.int32).transpose(0, 1)


__all__ = ["vocab_logits_by_sample", "vocab_logits_topk_by_sample"]
