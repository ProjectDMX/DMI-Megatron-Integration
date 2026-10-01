#!/usr/bin/env python3
"""Run Megatron with an experiment-only PP=2 D2H window subset."""

from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path


def select_pp2_m2_windows(
    selection: str,
    pp_size: int,
    pp_rank: int,
    num_microbatches: int,
    period: int,
    windows: tuple[tuple[int, int], ...],
) -> tuple[int, tuple[tuple[int, int], ...]]:
    if selection == "all":
        return period, windows
    if pp_size != 2 or num_microbatches != 2 or len(windows) != 2:
        raise RuntimeError("the experiment-only window selector requires PP=2 and M=2")
    if selection == "tx_free":
        index = 1 if pp_rank == 0 else 0
    elif selection == "tx_dependency_stalled":
        index = 0 if pp_rank == 0 else 1
    else:
        raise ValueError(f"unsupported D2H window subset: {selection}")
    return period, (windows[index],)


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("usage: pretrain_gpt_with_d2h_window_subset.py PRETRAIN_GPT [ARGS...]")
    selection = os.environ.get("DMI_NSYS_WINDOW_SUBSET", "all")
    pretrain_gpt = Path(sys.argv[1]).resolve()
    if not pretrain_gpt.is_file():
        raise FileNotFoundError(pretrain_gpt)

    from dmi_megatron_integration import schedule_runtime

    original = schedule_runtime.non_interleaved_d2h_window_pattern

    def selected_pattern(
        pp_size: int, pp_rank: int, num_microbatches: int
    ) -> tuple[int, tuple[tuple[int, int], ...]]:
        period, windows = original(pp_size, pp_rank, num_microbatches)
        return select_pp2_m2_windows(
            selection, pp_size, pp_rank, num_microbatches, period, windows
        )

    schedule_runtime.non_interleaved_d2h_window_pattern = selected_pattern

    sys.argv = [str(pretrain_gpt), *sys.argv[2:]]
    runpy.run_path(str(pretrain_gpt), run_name="__main__")


if __name__ == "__main__":
    main()
