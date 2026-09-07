#!/usr/bin/env python3
"""Summarize D2H placement relative to recurring-window boundary kernels."""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


MARKER_FRAGMENT = "advance_d2h_window_boundary_kernel"
ISSUE_RE = re.compile(
    r"\[d2h_window\] issue version=(\d+) window=(\d+) occurrence=(\d+) "
    r"counter=(\d+) bytes=(\d+) minimum_record_probe=(\d+)"
)
COMPLETION_RE = re.compile(
    r"\[d2h_window\] complete version=(\d+) counter=(\d+) result=([^\s]+)"
)


@dataclass(frozen=True)
class Interval:
    start: int
    end: int


@dataclass(frozen=True)
class Copy:
    start: int
    end: int
    size: int
    key: tuple[int, int, int]
    stream_id: int


def _union(intervals: Iterable[Interval]) -> list[Interval]:
    merged: list[Interval] = []
    for current in sorted(intervals, key=lambda value: (value.start, value.end)):
        if current.end <= current.start:
            continue
        if not merged or current.start > merged[-1].end:
            merged.append(current)
        else:
            previous = merged[-1]
            merged[-1] = Interval(previous.start, max(previous.end, current.end))
    return merged


def _overlap(interval: Interval, intervals: list[Interval]) -> int:
    return sum(
        max(0, min(interval.end, other.end) - max(interval.start, other.start))
        for other in intervals
        if other.end > interval.start and other.start < interval.end
    )


def _inside_window(copy: Copy, windows: list[Interval], *, require_completion: bool) -> bool:
    for window in windows:
        if window.start <= copy.start < window.end:
            return not require_completion or copy.end <= window.end
    return False


def _table_names(connection: sqlite3.Connection) -> set[str]:
    return {
        str(row[0])
        for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
    }


def _copy_kind(connection: sqlite3.Connection, name: str) -> int:
    row = connection.execute(
        "SELECT id FROM ENUM_CUDA_MEMCPY_OPER WHERE name = ?", (name,)
    ).fetchone()
    if row is None:
        raise RuntimeError(f"Nsight export does not define {name}")
    return int(row[0])


def _load_copies(connection: sqlite3.Connection, minimum_bytes: int) -> list[Copy]:
    dtoh = _copy_kind(connection, "CUDA_MEMCPY_KIND_DTOH")
    rows = connection.execute(
        """
        SELECT start, end, bytes, COALESCE(globalPid, -1), deviceId, contextId, streamId
        FROM CUPTI_ACTIVITY_KIND_MEMCPY
        WHERE copyKind = ? AND bytes >= ?
        ORDER BY start
        """,
        (dtoh, minimum_bytes),
    )
    return [
        Copy(
            start=int(start),
            end=int(end),
            size=int(size),
            key=(int(global_pid), int(device_id), int(context_id)),
            stream_id=int(stream_id),
        )
        for start, end, size, global_pid, device_id, context_id, stream_id in rows
    ]


def _load_kernels(
    connection: sqlite3.Connection,
) -> dict[tuple[int, int, int], list[tuple[Interval, str]]]:
    rows = connection.execute(
        """
        SELECT k.start, k.end, COALESCE(k.globalPid, -1), k.deviceId, k.contextId, s.value
        FROM CUPTI_ACTIVITY_KIND_KERNEL AS k
        JOIN StringIds AS s ON s.id = k.demangledName
        ORDER BY k.start
        """
    )
    result: dict[tuple[int, int, int], list[tuple[Interval, str]]] = {}
    for start, end, global_pid, device_id, context_id, name in rows:
        key = (int(global_pid), int(device_id), int(context_id))
        result.setdefault(key, []).append((Interval(int(start), int(end)), str(name)))
    return result


def _pair_windows(markers: list[Interval]) -> tuple[list[Interval], bool]:
    windows = [
        Interval(markers[index].start, markers[index + 1].end)
        for index in range(0, len(markers) - 1, 2)
    ]
    return windows, len(markers) % 2 != 0


def summarize(path: Path, minimum_bytes: int) -> dict[str, object]:
    connection = sqlite3.connect(path)
    try:
        required = {
            "CUPTI_ACTIVITY_KIND_MEMCPY",
            "CUPTI_ACTIVITY_KIND_KERNEL",
            "ENUM_CUDA_MEMCPY_OPER",
            "StringIds",
        }
        missing = required - _table_names(connection)
        if missing:
            raise RuntimeError(f"{path} is missing Nsight tables: {sorted(missing)}")

        copies = _load_copies(connection, minimum_bytes)
        kernels = _load_kernels(connection)
    finally:
        connection.close()

    by_key: dict[tuple[int, int, int], list[Copy]] = {}
    for copy in copies:
        by_key.setdefault(copy.key, []).append(copy)

    contexts: list[dict[str, object]] = []
    aggregate = {
        "large_d2h_copies": 0,
        "large_d2h_bytes": 0,
        "markers": 0,
        "windows": 0,
        "training_span_d2h_copies": 0,
        "training_span_d2h_bytes": 0,
        "issued_inside_window_bytes": 0,
        "completed_inside_window_bytes": 0,
        "d2h_compute_overlap_ns": 0,
        "d2h_nccl_overlap_ns": 0,
    }

    all_keys = sorted(set(by_key) | set(kernels))
    for key in all_keys:
        context_copies = by_key.get(key, [])
        context_kernels = kernels.get(key, [])
        markers = [
            interval for interval, name in context_kernels if MARKER_FRAGMENT in name
        ]
        windows, has_unpaired_marker = _pair_windows(markers)
        if windows:
            span = Interval(windows[0].start, windows[-1].end)
            training_copies = [
                copy
                for copy in context_copies
                if copy.start >= span.start and copy.end <= span.end
            ]
            relevant_kernels = [
                (interval, name)
                for interval, name in context_kernels
                if interval.end > span.start and interval.start < span.end
            ]
        else:
            span = None
            training_copies = context_copies
            relevant_kernels = context_kernels

        nccl = _union(
            interval
            for interval, name in relevant_kernels
            if "nccl" in name.lower() or "msccl" in name.lower()
        )
        compute = _union(
            interval
            for interval, name in relevant_kernels
            if MARKER_FRAGMENT not in name
            and "nccl" not in name.lower()
            and "msccl" not in name.lower()
        )

        issued_bytes = sum(
            copy.size
            for copy in training_copies
            if _inside_window(copy, windows, require_completion=False)
        )
        completed_bytes = sum(
            copy.size
            for copy in training_copies
            if _inside_window(copy, windows, require_completion=True)
        )
        compute_overlap = sum(
            _overlap(Interval(copy.start, copy.end), compute) for copy in training_copies
        )
        nccl_overlap = sum(
            _overlap(Interval(copy.start, copy.end), nccl) for copy in training_copies
        )
        stream_bytes: dict[int, int] = {}
        for copy in training_copies:
            stream_bytes[copy.stream_id] = stream_bytes.get(copy.stream_id, 0) + copy.size

        context = {
            "global_pid": key[0],
            "device_id": key[1],
            "context_id": key[2],
            "large_d2h_copies": len(context_copies),
            "large_d2h_bytes": sum(copy.size for copy in context_copies),
            "markers": len(markers),
            "windows": len(windows),
            "has_unpaired_marker": has_unpaired_marker,
            "training_span_ns": None if span is None else [span.start, span.end],
            "training_span_d2h_copies": len(training_copies),
            "training_span_d2h_bytes": sum(copy.size for copy in training_copies),
            "issued_inside_window_bytes": issued_bytes,
            "completed_inside_window_bytes": completed_bytes,
            "d2h_compute_overlap_ns": compute_overlap,
            "d2h_nccl_overlap_ns": nccl_overlap,
            "d2h_bytes_by_stream": dict(sorted(stream_bytes.items())),
        }
        contexts.append(context)
        for field in aggregate:
            if field == "markers":
                aggregate[field] += len(markers)
            elif field == "windows":
                aggregate[field] += len(windows)
            else:
                aggregate[field] += int(context[field])

    training_bytes = aggregate["training_span_d2h_bytes"]
    has_windows = aggregate["windows"] > 0
    aggregate["issued_inside_window_fraction"] = (
        aggregate["issued_inside_window_bytes"] / training_bytes
        if has_windows and training_bytes
        else None
    )
    aggregate["completed_inside_window_fraction"] = (
        aggregate["completed_inside_window_bytes"] / training_bytes
        if has_windows and training_bytes
        else None
    )
    return {
        "path": str(path.resolve()),
        "minimum_copy_bytes": minimum_bytes,
        "aggregate": aggregate,
        "contexts": contexts,
    }


def summarize_debug_log(path: Path) -> dict[str, object]:
    text = path.read_text(encoding="utf-8", errors="replace")
    issues = ISSUE_RE.findall(text)
    completions = COMPLETION_RE.findall(text)
    results: dict[str, int] = {}
    for _, _, result in completions:
        results[result] = results.get(result, 0) + 1
    return {
        "path": str(path.resolve()),
        "definitions": text.count("[DMI] d2h_window define "),
        "issues": len(issues),
        "issued_bytes": sum(int(issue[4]) for issue in issues),
        "minimum_record_probes": sum(int(issue[5]) for issue in issues),
        "completions": len(completions),
        "completion_results": dict(sorted(results.items())),
        "terminal_fallback": "[d2h_window] terminal fallback" in text,
    }


def _bytes(value: int) -> str:
    return f"{value / (1024 ** 2):.2f} MiB"


def _fraction(value: object) -> str:
    return "n/a" if value is None else f"{float(value):.2%}"


def write_markdown(
    path: Path,
    summaries: dict[str, dict[str, object]],
    debug: dict[str, object],
) -> None:
    lines = [
        "# Recurring D2H Window Nsight Summary",
        "",
        "D2H events smaller than the configured threshold are excluded. For a trace with",
        "boundary markers, the training span runs from its first OPEN marker to its final",
        "CLOSE marker. Marker kernel duration makes the reported window boundary an Nsight",
        "timeline approximation.",
        "",
        "| Run | Large D2H | Markers | Windows | D2H in training span | "
        "Issued inside | Completed inside | Compute overlap | NCCL overlap |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, summary in summaries.items():
        aggregate = summary["aggregate"]
        assert isinstance(aggregate, dict)
        lines.append(
            "| {label} | {large} | {markers} | {windows} | {training} | {issued} | "
            "{completed} | {compute:.3f} ms | {nccl:.3f} ms |".format(
                label=label,
                large=_bytes(int(aggregate["large_d2h_bytes"])),
                markers=int(aggregate["markers"]),
                windows=int(aggregate["windows"]),
                training=_bytes(int(aggregate["training_span_d2h_bytes"])),
                issued=_fraction(aggregate["issued_inside_window_fraction"]),
                completed=_fraction(aggregate["completed_inside_window_fraction"]),
                compute=int(aggregate["d2h_compute_overlap_ns"]) / 1e6,
                nccl=int(aggregate["d2h_nccl_overlap_ns"]) / 1e6,
            )
        )
    lines.extend(
        [
            "",
            "The normal-batching trace has no recurring-window markers, so its inside-window",
            "columns are intentionally reported as `n/a`. Use the Nsight timeline for the",
            "aligned visual comparison.",
            "",
            "## Window debug log",
            "",
            f"- Pattern definitions: {debug['definitions']}",
            f"- D2H issues: {debug['issues']} ({_bytes(int(debug['issued_bytes']))})",
            f"- D2H completions: {debug['completions']}",
            f"- Completion results: `{json.dumps(debug['completion_results'], sort_keys=True)}`",
            f"- Minimum-record probes: {debug['minimum_record_probes']}",
            f"- Terminal fallback: {debug['terminal_fallback']}",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--normal", type=Path, required=True, help="Normal-batching SQLite export")
    parser.add_argument("--window", type=Path, required=True, help="Window-scheduled SQLite export")
    parser.add_argument("--window-log", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--minimum-copy-bytes",
        type=int,
        default=1024 * 1024,
        help="Ignore small unrelated D2H copies (default: 1 MiB)",
    )
    parser.add_argument(
        "--require-valid-window-run",
        action="store_true",
        help="Fail after writing output if markers/debug issues are missing or fallback occurred",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.minimum_copy_bytes < 1:
        raise ValueError("--minimum-copy-bytes must be positive")
    for path in (args.normal, args.window, args.window_log):
        if not path.is_file():
            raise FileNotFoundError(path)

    summaries = {
        "normal_batching": summarize(args.normal, args.minimum_copy_bytes),
        "window_scheduled": summarize(args.window, args.minimum_copy_bytes),
    }
    debug = summarize_debug_log(args.window_log)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps({"traces": summaries, "window_debug": debug}, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    write_markdown(args.output_dir / "summary.md", summaries, debug)
    print(args.output_dir / "summary.md")
    if args.require_valid_window_run:
        window_aggregate = summaries["window_scheduled"]["aggregate"]
        assert isinstance(window_aggregate, dict)
        marker_contexts = sum(
            int(context["markers"]) > 0
            for context in summaries["window_scheduled"]["contexts"]
        )
        unpaired = any(
            bool(context["has_unpaired_marker"])
            for context in summaries["window_scheduled"]["contexts"]
        )
        errors = []
        if int(summaries["normal_batching"]["aggregate"]["markers"]) != 0:
            errors.append("normal-batching trace unexpectedly contains boundary markers")
        if marker_contexts != 2:
            errors.append(f"expected boundary markers in two PP contexts, found {marker_contexts}")
        if unpaired:
            errors.append("window-scheduled trace contains an unpaired boundary marker")
        if int(debug["definitions"]) != 2:
            errors.append(
                f"expected one pattern definition per PP rank, found {debug['definitions']}"
            )
        if int(debug["issues"]) == 0:
            errors.append("window debug log contains no real D2H issue")
        if bool(debug["terminal_fallback"]):
            errors.append("window debug log reports terminal fallback")
        if errors:
            raise RuntimeError("; ".join(errors))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
