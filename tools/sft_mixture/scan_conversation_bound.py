#!/usr/bin/env python3
"""Scan normalized SFT JSONL rows for the packed conversation bound."""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
from pathlib import Path
from typing import Iterable

from .segmentation import SPLIT_RULE_VERSION, count_conversations


def _expand_inputs(patterns: Iterable[str]) -> list[Path]:
    paths: list[Path] = []
    seen: set[Path] = set()
    for pattern in patterns:
        matches = sorted(Path(match).resolve() for match in glob.glob(pattern))
        if not matches:
            candidate = Path(pattern).resolve()
            if candidate.is_file():
                matches = [candidate]
            else:
                raise FileNotFoundError(f"No SFT JSONL file matched {pattern!r}")
        for path in matches:
            if not path.is_file():
                raise FileNotFoundError(f"SFT input is not a file: {path}")
            if path not in seen:
                seen.add(path)
                paths.append(path)
    if not paths:
        raise ValueError("At least one SFT JSONL input is required")
    return paths


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def scan_paths(patterns: Iterable[str]) -> dict[str, object]:
    """Scan every selected row and return a machine-readable bound report."""

    source_reports: list[dict[str, object]] = []
    global_max = -1
    global_location: dict[str, object] | None = None

    for path in _expand_inputs(patterns):
        row_count = 0
        source_max = -1
        source_max_row = -1
        with path.open("r", encoding="utf-8") as handle:
            for row_index, line in enumerate(handle):
                if not line.strip():
                    raise ValueError(f"Blank JSONL row at {path}:{row_index + 1}")
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSON at {path}:{row_index + 1}: {exc}"
                    ) from exc
                messages = record.get("messages")
                if not isinstance(messages, list):
                    raise ValueError(
                        f"{path}:{row_index + 1} must contain a messages list"
                    )
                count = count_conversations(messages)
                if count <= 0:
                    raise ValueError(
                        f"{path}:{row_index + 1} contains no logical conversation"
                    )
                row_count += 1
                if count > source_max:
                    source_max = count
                    source_max_row = row_index
                if count > global_max:
                    global_max = count
                    global_location = {
                        "path": str(path),
                        "row_index": row_index,
                        "line_number": row_index + 1,
                    }
        if row_count == 0:
            raise ValueError(f"SFT input contains no rows: {path}")
        source_reports.append(
            {
                "path": str(path),
                "sha256": _sha256(path),
                "row_count": row_count,
                "max_conversations_per_row": source_max,
                "max_row_index": source_max_row,
                "max_line_number": source_max_row + 1,
            }
        )

    return {
        "kind": "dmi_sft_conversation_bound",
        "schema_version": 1,
        "split_rule_version": SPLIT_RULE_VERSION,
        "sources": source_reports,
        "source_count": len(source_reports),
        "total_rows": sum(int(item["row_count"]) for item in source_reports),
        "global_c_row_max": global_max,
        "global_max_location": global_location,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "inputs",
        nargs="+",
        help="JSONL paths or quoted glob patterns; every matched row is scanned",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    report = scan_paths(args.inputs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
