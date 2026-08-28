from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from megatron.training.datasets.sft_dataset import IGNORE_INDEX, SFTDataset


def _packed_sample(
    *,
    tokens,
    targets,
    pad,
    eod,
    sequence_length,
    context_parallel_size=1,
):
    tokenizer = SimpleNamespace(
        pad=pad,
        eod=eod,
        tokenize_conversation=lambda *args, **kwargs: (
            np.asarray(tokens, dtype=np.int64),
            np.asarray(targets, dtype=np.int64),
        ),
    )
    dataset = object.__new__(SFTDataset)
    dataset.indices = np.asarray([0], dtype=np.int64)
    dataset.dataset = [
        [
            {"role": "system", "content": ""},
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "answer"},
        ]
    ]
    dataset.dataset_path = "synthetic.jsonl"
    dataset.num_samples = 1
    dataset.config = SimpleNamespace(
        tokenizer=tokenizer,
        sequence_length=sequence_length,
        reset_position_ids=False,
        reset_attention_mask=False,
        context_parallel_size=context_parallel_size,
        dmi_metadata_enabled=False,
        dmi_packed_max_conversations_per_row=None,
        dmi_micro_batch_size=1,
    )
    return dataset[0]


@pytest.mark.parametrize(
    ("pad", "eod", "context_parallel_size"),
    ((7, 7, 1), (7, 7, 2), (0, 7, 1)),
)
def test_padding_labels_are_ignored_without_masking_real_eos(
    pad, eod, context_parallel_size
):
    sample = _packed_sample(
        tokens=[101, 102, eod],
        targets=[IGNORE_INDEX, 102, eod],
        pad=pad,
        eod=eod,
        sequence_length=7,
        context_parallel_size=context_parallel_size,
    )

    assert sample["labels"].tolist() == [
        102,
        eod,
        IGNORE_INDEX,
        IGNORE_INDEX,
        IGNORE_INDEX,
        IGNORE_INDEX,
        IGNORE_INDEX,
    ]
    assert sample["loss_mask"].tolist() == [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    assert sample["loss_mask"][sample["labels"] == eod].tolist() == [1.0]


def test_truncation_sentinel_is_an_ignored_label_when_pad_equals_eos():
    sample = _packed_sample(
        tokens=[101, 102, 103, 7],
        targets=[IGNORE_INDEX, 102, 103, 7],
        pad=7,
        eod=7,
        sequence_length=3,
    )

    assert sample["tokens"].tolist() == [101, 102, 103]
    assert sample["labels"].tolist() == [102, 103, IGNORE_INDEX]
    assert sample["loss_mask"].tolist() == [1.0, 1.0, 0.0]
