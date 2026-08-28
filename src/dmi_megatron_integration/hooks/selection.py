"""Hook-selection parsing for the Megatron integration."""

from __future__ import annotations


def parse_hook_selection(
    selection: str | None,
    *,
    default: str = "router-summary",
) -> set[str]:
    selected = {
        part.strip()
        for part in str(selection if selection is not None else default).split(",")
    }
    if "" in selected:
        raise ValueError(f"Invalid empty DMI hook selection entry: {selection!r}")
    return selected


__all__ = ["parse_hook_selection"]
