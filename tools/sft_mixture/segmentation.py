"""Logical-conversation segmentation shared by preparation and runtime code."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence


SPLIT_RULE_VERSION = "megatron-sft-system-boundary-v1"


@dataclass(frozen=True)
class ConversationRange:
    """One logical conversation's valid physical token range."""

    start: int
    end: int

    def __post_init__(self) -> None:
        if self.start < 0 or self.end <= self.start:
            raise ValueError(f"Invalid conversation range [{self.start}, {self.end})")

    @property
    def length(self) -> int:
        return self.end - self.start


def split_merged_conversations(
    messages: Sequence[Mapping[str, object]],
) -> list[list[Mapping[str, object]]]:
    """Apply Megatron SFTDataset's system-message split rule exactly."""

    conversations: list[list[Mapping[str, object]]] = []
    current: list[Mapping[str, object]] = []
    for message_index, message in enumerate(messages):
        if not isinstance(message, Mapping):
            raise TypeError(f"messages[{message_index}] must be an object")
        role = message.get("role")
        if not isinstance(role, str):
            raise ValueError(f"messages[{message_index}].role must be a string")
        if role == "system":
            if current:
                conversations.append(current)
            current = [message]
        else:
            current.append(message)
    if current:
        conversations.append(current)
    return conversations


def count_conversations(messages: Sequence[Mapping[str, object]]) -> int:
    """Return the physical row's logical-conversation count."""

    return len(split_merged_conversations(messages))


def pack_fixed_capacity_ranges(
    ranges: Sequence[ConversationRange],
    *,
    capacity: int,
) -> tuple[list[int], list[int]]:
    """Build fixed-capacity starts/ends with zero-length trailing entries."""

    if capacity <= 0:
        raise ValueError("capacity must be positive")
    if len(ranges) > capacity:
        raise ValueError(
            f"Conversation count {len(ranges)} exceeds configured capacity {capacity}"
        )
    previous_end = 0
    for index, item in enumerate(ranges):
        if item.start < previous_end:
            raise ValueError(
                f"Conversation ranges must be ordered and non-overlapping: "
                f"range {index} starts at {item.start}, previous end is {previous_end}"
            )
        previous_end = item.end

    trailing = ranges[-1].end if ranges else 0
    starts = [item.start for item in ranges]
    ends = [item.end for item in ranges]
    starts.extend([trailing] * (capacity - len(ranges)))
    ends.extend([trailing] * (capacity - len(ranges)))
    return starts, ends


def active_lengths(starts: Sequence[int], ends: Sequence[int]) -> list[int]:
    """Return positive segment lengths after validating the fixed-capacity map."""

    if len(starts) != len(ends):
        raise ValueError("start/end pointer lengths must match")
    lengths: list[int] = []
    previous_end = 0
    saw_inactive = False
    for index, (start, end) in enumerate(zip(starts, ends)):
        start = int(start)
        end = int(end)
        if start < previous_end or end < start:
            raise ValueError(
                f"Invalid segment {index}: [{start}, {end}) after end {previous_end}"
            )
        if end == start:
            saw_inactive = True
        else:
            if saw_inactive:
                raise ValueError("Active segments must precede inactive capacity entries")
            lengths.append(end - start)
        previous_end = end
    return lengths


__all__ = [
    "SPLIT_RULE_VERSION",
    "ConversationRange",
    "active_lengths",
    "count_conversations",
    "pack_fixed_capacity_ranges",
    "split_merged_conversations",
]
