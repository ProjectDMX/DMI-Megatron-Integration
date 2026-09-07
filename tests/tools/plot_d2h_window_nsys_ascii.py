#!/usr/bin/env python3
"""Render compute, NCCL, recurring-window, and large-D2H Nsight timelines."""

from __future__ import annotations

import argparse
import sqlite3
from dataclasses import dataclass
from pathlib import Path


MARKER_FRAGMENT = "advance_d2h_window_boundary_kernel"


@dataclass(frozen=True)
class Interval:
    start: int
    end: int


@dataclass(frozen=True)
class Kernel:
    interval: Interval
    stream_id: int
    name: str


@dataclass(frozen=True)
class Copy:
    interval: Interval
    stream_id: int
    size: int


ContextKey = tuple[int, int, int]


def _table_names(connection: sqlite3.Connection) -> set[str]:
    return {
        str(row[0])
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        )
    }


def _copy_kind(connection: sqlite3.Connection, name: str) -> int:
    row = connection.execute(
        "SELECT id FROM ENUM_CUDA_MEMCPY_OPER WHERE name = ?", (name,)
    ).fetchone()
    if row is None:
        raise RuntimeError(f"Nsight export does not define {name}")
    return int(row[0])


def _load_trace(
    path: Path, minimum_copy_bytes: int
) -> tuple[dict[ContextKey, list[Kernel]], dict[ContextKey, list[Copy]]]:
    connection = sqlite3.connect(path)
    try:
        required = {
            "CUPTI_ACTIVITY_KIND_KERNEL",
            "CUPTI_ACTIVITY_KIND_MEMCPY",
            "ENUM_CUDA_MEMCPY_OPER",
            "StringIds",
        }
        missing = required - _table_names(connection)
        if missing:
            raise RuntimeError(f"{path} is missing Nsight tables: {sorted(missing)}")

        kernels: dict[ContextKey, list[Kernel]] = {}
        for row in connection.execute(
            """
            SELECT k.start, k.end, k.streamId, COALESCE(k.globalPid, -1),
                   k.deviceId, k.contextId, s.value
            FROM CUPTI_ACTIVITY_KIND_KERNEL AS k
            JOIN StringIds AS s ON s.id = k.demangledName
            ORDER BY k.start
            """
        ):
            start, end, stream_id, global_pid, device_id, context_id, name = row
            key = (int(global_pid), int(device_id), int(context_id))
            kernels.setdefault(key, []).append(
                Kernel(
                    interval=Interval(int(start), int(end)),
                    stream_id=int(stream_id),
                    name=str(name),
                )
            )

        dtoh = _copy_kind(connection, "CUDA_MEMCPY_KIND_DTOH")
        copies: dict[ContextKey, list[Copy]] = {}
        for row in connection.execute(
            """
            SELECT start, end, streamId, bytes, COALESCE(globalPid, -1),
                   deviceId, contextId
            FROM CUPTI_ACTIVITY_KIND_MEMCPY
            WHERE copyKind = ? AND bytes >= ?
            ORDER BY start
            """,
            (dtoh, minimum_copy_bytes),
        ):
            start, end, stream_id, size, global_pid, device_id, context_id = row
            key = (int(global_pid), int(device_id), int(context_id))
            copies.setdefault(key, []).append(
                Copy(
                    interval=Interval(int(start), int(end)),
                    stream_id=int(stream_id),
                    size=int(size),
                )
            )
    finally:
        connection.close()
    return kernels, copies


def _is_nccl(name: str) -> bool:
    lowered = name.lower()
    return "nccl" in lowered or "msccl" in lowered


def _intersect(left: Interval, right: Interval) -> Interval | None:
    start = max(left.start, right.start)
    end = min(left.end, right.end)
    return Interval(start, end) if end > start else None


def _clip(intervals: list[Interval], span: Interval) -> list[Interval]:
    result = []
    for interval in intervals:
        clipped = _intersect(interval, span)
        if clipped is not None:
            result.append(clipped)
    return result


def _pair_windows(markers: list[Interval]) -> list[Interval]:
    if len(markers) % 2:
        raise RuntimeError("recurring-window trace contains an unpaired marker")
    return [
        Interval(markers[index].start, markers[index + 1].end)
        for index in range(0, len(markers), 2)
    ]


def _dominant_compute_stream(kernels: list[Kernel], span: Interval) -> int | None:
    duration_by_stream: dict[int, int] = {}
    for kernel in kernels:
        if MARKER_FRAGMENT in kernel.name or _is_nccl(kernel.name):
            continue
        clipped = _intersect(kernel.interval, span)
        if clipped is None:
            continue
        duration_by_stream[kernel.stream_id] = (
            duration_by_stream.get(kernel.stream_id, 0) + clipped.end - clipped.start
        )
    if not duration_by_stream:
        return None
    return max(duration_by_stream, key=lambda stream: duration_by_stream[stream])


def _overlaps(left: list[Interval], right: list[Interval]) -> list[Interval]:
    result = []
    for first in left:
        for second in right:
            overlap = _intersect(first, second)
            if overlap is not None:
                result.append(overlap)
    return result


def _bin_range(interval: Interval, span: Interval, width: int) -> tuple[int, int]:
    duration = span.end - span.start
    begin_offset = interval.start - span.start
    end_offset = interval.end - span.start
    begin = min(width - 1, begin_offset * width // duration)
    end = min(
        width,
        max(begin + 1, (end_offset * width + duration - 1) // duration),
    )
    return begin, end


def _raster(intervals: list[Interval], span: Interval, width: int, symbol: str) -> str:
    cells = [" "] * width
    for interval in intervals:
        clipped = _intersect(interval, span)
        if clipped is None:
            continue
        begin, end = _bin_range(clipped, span, width)
        for index in range(begin, end):
            cells[index] = symbol
    return "".join(cells)


def _window_raster(windows: list[Interval], span: Interval, width: int) -> str:
    cells = list(_raster(windows, span, width, "-"))
    for window in windows:
        clipped = _intersect(window, span)
        if clipped is None:
            continue
        begin, end = _bin_range(clipped, span, width)
        last = end - 1
        if begin == last:
            cells[begin] = "W"
        else:
            cells[begin] = "[" if cells[begin] in {" ", "-"} else "W"
            cells[last] = "]" if cells[last] in {" ", "-"} else "W"
    return "".join(cells)


def _render_tracks(
    span: Interval,
    width: int,
    compute: list[Interval],
    d2h: list[Interval],
    nccl: list[Interval],
    windows: list[Interval] | None = None,
) -> list[str]:
    overlap = _overlaps(compute, d2h)
    compute_row = list(_raster(compute, span, width, "#"))
    d2h_row = _raster(d2h, span, width, "=")
    overlap_row = _raster(overlap, span, width, "!")
    # A coarse display bin may contain sequential compute and D2H intervals
    # without any real overlap. Suppress that compute glyph so vertical
    # alignment means overlap only when the exact-overlap row also marks it.
    for index in range(width):
        if d2h_row[index] == "=" and overlap_row[index] != "!":
            compute_row[index] = " "
    lines = []
    if windows:
        lines.append(f"{'windows':>12} |{_window_raster(windows, span, width)}|")
    lines.extend(
        [
            f"{'compute':>12} |{''.join(compute_row)}|",
            f"{'PCIe D2H':>12} |{d2h_row}|",
            f"{'overlap':>12} |{overlap_row}|",
            f"{'NCCL':>12} |{_raster(nccl, span, width, '~')}|",
        ]
    )
    return lines


def _render_context(
    label: str,
    key: ContextKey,
    kernels: list[Kernel],
    copies: list[Copy],
    width: int,
) -> list[str]:
    markers = [
        kernel.interval for kernel in kernels if MARKER_FRAGMENT in kernel.name
    ]
    windows = _pair_windows(markers)
    if windows:
        span = Interval(markers[0].start, markers[-1].end)
        span_source = "first OPEN to final CLOSE"
    elif copies:
        span = Interval(copies[0].interval.start, copies[-1].interval.end)
        span_source = "first to final large D2H"
    else:
        return []

    compute_stream = _dominant_compute_stream(kernels, span)
    all_compute = (
        [
            kernel.interval
            for kernel in kernels
            if kernel.stream_id == compute_stream
            and MARKER_FRAGMENT not in kernel.name
            and not _is_nccl(kernel.name)
        ]
        if compute_stream is not None
        else []
    )
    all_nccl = [kernel.interval for kernel in kernels if _is_nccl(kernel.name)]
    compute = _clip(all_compute, span)
    nccl = _clip(all_nccl, span)
    d2h = _clip([copy.interval for copy in copies], span)
    overlap = _overlaps(compute, d2h)
    shown_copies = [
        copy for copy in copies if _intersect(copy.interval, span) is not None
    ]
    d2h_streams = sorted(
        {copy.stream_id for copy in copies if _intersect(copy.interval, span) is not None}
    )
    compute_label = "none" if compute_stream is None else str(compute_stream)
    d2h_label = ",".join(map(str, d2h_streams)) or "none"
    span_ms = (span.end - span.start) / 1e6
    lines = [
        f"{label}: trace device {key[1]}, CUDA context {key[2]}",
        f"span: {span_ms:.3f} ms ({span_source}); one column ~= {span_ms / width:.3f} ms",
        "large D2H shown: {} copies, {:.2f} MiB; exact compute overlap: {:.3f} ms".format(
            len(shown_copies),
            sum(copy.size for copy in shown_copies) / (1024**2),
            sum(interval.end - interval.start for interval in overlap) / 1e6,
        ),
    ]
    lines.append(f"streams: compute={compute_label}, PCIe D2H={d2h_label}")
    lines.extend(_render_tracks(span, width, compute, d2h, nccl, windows))
    lines.append("")
    return lines


def render_trace(
    label: str, path: Path, minimum_copy_bytes: int, width: int
) -> list[str]:
    kernels_by_context, copies_by_context = _load_trace(path, minimum_copy_bytes)
    keys = sorted(copies_by_context)
    lines: list[str] = []
    for key in keys:
        context_lines = _render_context(
            label,
            key,
            kernels_by_context.get(key, []),
            copies_by_context.get(key, []),
            width,
        )
        lines.extend(context_lines)
    if not lines:
        raise RuntimeError(f"{path} contains no plottable large D2H context")
    return lines


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--normal", type=Path, required=True)
    parser.add_argument("--window", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--width", type=int, default=72)
    parser.add_argument("--minimum-copy-bytes", type=int, default=1024 * 1024)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not 40 <= args.width <= 240:
        raise ValueError("--width must be between 40 and 240")
    if args.minimum_copy_bytes < 1:
        raise ValueError("--minimum-copy-bytes must be positive")
    for path in (args.normal, args.window):
        if not path.is_file():
            raise FileNotFoundError(path)

    lines = [
        "Recurring D2H Window Nsight ASCII Timeline",
        "",
        "# = dominant non-NCCL compute-stream kernel",
        "= = PCIe D2H copy at or above the configured size threshold",
        "! = exact nanosecond overlap between PCIe D2H and compute",
        "~ = NCCL kernel",
        "[---] = recurring D2H window; W = a window narrower than one time bin",
        "A symbol means that at least one matching interval intersects that time bin.",
        "In a D2H bin, # is suppressed unless exact D2H/compute overlap exists.",
        "",
    ]
    lines.extend(render_trace("normal_batching", args.normal, args.minimum_copy_bytes, args.width))
    lines.extend(render_trace("window_scheduled", args.window, args.minimum_copy_bytes, args.width))
    text = "\n".join(lines).rstrip() + "\n"
    if args.output is None:
        print(text, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
        print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
