from __future__ import annotations

import json

import pytest

from tools.sft_mixture.scan_conversation_bound import scan_paths
from tools.sft_mixture.segmentation import (
    ConversationRange,
    active_lengths,
    pack_fixed_capacity_ranges,
    split_merged_conversations,
)


def _messages(conversation_count: int) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    for index in range(conversation_count):
        messages.extend(
            [
                {"role": "system", "content": ""},
                {"role": "user", "content": f"q{index}"},
                {"role": "assistant", "content": f"a{index}"},
            ]
        )
    return messages


def test_split_merged_conversations_matches_system_boundaries():
    conversations = split_merged_conversations(_messages(3))

    assert len(conversations) == 3
    assert [item[1]["content"] for item in conversations] == ["q0", "q1", "q2"]


def test_fixed_capacity_ranges_use_noncopying_compatible_trailing_entries():
    starts, ends = pack_fixed_capacity_ranges(
        [ConversationRange(0, 3), ConversationRange(5, 8)],
        capacity=4,
    )

    assert starts == [0, 5, 8, 8]
    assert ends == [3, 8, 8, 8]
    assert active_lengths(starts, ends) == [3, 3]


def test_scan_paths_reports_global_bound_for_paths_and_globs(tmp_path):
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    first.write_text(
        json.dumps({"messages": _messages(1)}) + "\n"
        + json.dumps({"messages": _messages(2)}) + "\n"
    )
    second.write_text(json.dumps({"messages": _messages(3)}) + "\n")

    report = scan_paths([str(first), str(tmp_path / "second*.jsonl")])

    assert report["global_c_row_max"] == 3
    assert report["total_rows"] == 3
    assert report["global_max_location"]["path"] == str(second.resolve())
    assert report["global_max_location"]["line_number"] == 1


def test_scan_paths_rejects_blank_rows(tmp_path):
    source = tmp_path / "bad.jsonl"
    source.write_text(json.dumps({"messages": _messages(1)}) + "\n\n")

    with pytest.raises(ValueError, match="Blank JSONL row"):
        scan_paths([str(source)])
