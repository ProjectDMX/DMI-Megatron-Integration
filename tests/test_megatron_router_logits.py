from __future__ import annotations

import pytest
import torch

from dmi_megatron_integration.hooks.megatron_router_logits import router_logits_by_sample


def test_router_logits_by_sample_only_changes_layout() -> None:
    logits = torch.arange(3 * 2 * 4, dtype=torch.float32).reshape(3, 2, 4)

    captured = router_logits_by_sample(logits)

    assert captured.shape == (2, 3, 4)
    assert captured.data_ptr() == logits.data_ptr()
    for batch in range(2):
        for sequence in range(3):
            torch.testing.assert_close(captured[batch, sequence], logits[sequence, batch])


def test_router_logits_by_sample_rejects_non_sequence_batch_expert_layout() -> None:
    with pytest.raises(ValueError, match=r"Expected router logits \[S, B, E\]"):
        router_logits_by_sample(torch.zeros(6, 4))
