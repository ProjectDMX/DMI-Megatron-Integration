import importlib.util
import sys
from pathlib import Path


SCRIPT = Path(__file__).parent / "tools" / "plot_d2h_window_nsys_ascii.py"
SPEC = importlib.util.spec_from_file_location("plot_d2h_window_nsys_ascii", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
plot = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = plot
SPEC.loader.exec_module(plot)


def test_raster_preserves_an_interval_narrower_than_one_bin():
    rendered = plot._raster(
        [plot.Interval(10, 11)], plot.Interval(0, 100), width=10, symbol="D"
    )

    assert rendered == " D        "


def test_raster_does_not_spill_across_an_exact_bin_boundary():
    rendered = plot._raster(
        [plot.Interval(0, 10)], plot.Interval(0, 100), width=10, symbol="D"
    )

    assert rendered == "D         "


def test_window_raster_marks_a_window_narrower_than_one_bin():
    rendered = plot._window_raster(
        [plot.Interval(10, 11)], plot.Interval(0, 100), width=10
    )

    assert rendered == " W        "


def test_dominant_compute_stream_excludes_markers_and_nccl():
    span = plot.Interval(0, 100)
    kernels = [
        plot.Kernel(plot.Interval(0, 80), 7, "gemm"),
        plot.Kernel(plot.Interval(0, 90), 8, "ncclKernel"),
        plot.Kernel(plot.Interval(0, 95), 9, plot.MARKER_FRAGMENT),
        plot.Kernel(plot.Interval(0, 20), 10, "short_compute"),
    ]

    assert plot._dominant_compute_stream(kernels, span) == 7


def test_overlap_track_contains_only_exact_interval_intersections():
    left = [plot.Interval(0, 10), plot.Interval(20, 30)]
    right = [plot.Interval(9, 21), plot.Interval(40, 50)]

    assert plot._overlaps(left, right) == [
        plot.Interval(9, 10),
        plot.Interval(20, 21),
    ]


def test_tracks_do_not_imply_overlap_for_sequential_events_in_one_bin():
    rendered = plot._render_tracks(
        span=plot.Interval(0, 10),
        width=1,
        compute=[plot.Interval(0, 4)],
        d2h=[plot.Interval(6, 10)],
        nccl=[],
    )

    assert rendered[0] == "     compute | |"
    assert rendered[1] == "    PCIe D2H |=|"
    assert rendered[2] == "     overlap | |"


def test_tracks_mark_exact_overlap_on_all_three_rows():
    rendered = plot._render_tracks(
        span=plot.Interval(0, 10),
        width=1,
        compute=[plot.Interval(0, 7)],
        d2h=[plot.Interval(6, 10)],
        nccl=[],
    )

    assert rendered[0] == "     compute |#|"
    assert rendered[1] == "    PCIe D2H |=|"
    assert rendered[2] == "     overlap |!|"


def test_unpaired_window_marker_is_rejected():
    try:
        plot._pair_windows([plot.Interval(0, 1)])
    except RuntimeError as error:
        assert "unpaired marker" in str(error)
    else:
        raise AssertionError("expected an unpaired marker to be rejected")
