#!/usr/bin/env bash
set -euo pipefail

# This script is for the pure Megatron-LM baseline, not the DMI-patched
# Megatron submodule.  DMI_CONDA_ENV should name an environment that can run the
# pure Megatron-LM clone at DMI_BENCH_PURE_MEGATRON_ROOT.  The benchmark driver
# still lives in this DMI repo, but the launched framework code comes from the
# clean Megatron clone and DMI is forced off.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

export DMI_CONDA_ENV="${DMI_CONDA_ENV:-ring_offload_pure_megatron}"

if [[ "${CONDA_DEFAULT_ENV:-}" != "$DMI_CONDA_ENV" ]]; then
  if [[ -f "$HOME/miniforge3/etc/profile.d/conda.sh" ]]; then
    # shellcheck disable=SC1091
    source "$HOME/miniforge3/etc/profile.d/conda.sh"
    conda activate "$DMI_CONDA_ENV"
  fi
fi

BASE_COMMIT="${DMI_BENCH_PURE_MEGATRON_COMMIT:-a3a7a0c699876bb6699f47581c3ca6da764abf9d}"
CLEAN_ROOT="${DMI_BENCH_PURE_MEGATRON_ROOT:-$REPO_ROOT/artifacts/pure_megatron_baseline/Megatron-LM}"
if ! git -C "$CLEAN_ROOT" rev-parse --git-dir >/dev/null 2>&1; then
  echo "[DMI bench] ERROR: pure Megatron clone not found at $CLEAN_ROOT" >&2
  echo "[DMI bench]        clone it first; this script intentionally does not create a git worktree." >&2
  exit 1
fi
current_commit="$(git -C "$CLEAN_ROOT" rev-parse HEAD)"
expected_commit="$(git -C "$CLEAN_ROOT" rev-parse "$BASE_COMMIT")"
if [[ "$current_commit" != "$expected_commit" ]]; then
  echo "[DMI bench] ERROR: pure Megatron clone has commit $current_commit, expected $expected_commit" >&2
  exit 1
fi
if [[ -n "$(git -C "$CLEAN_ROOT" status --short)" ]]; then
  echo "[DMI bench] ERROR: pure Megatron clone has local changes:" >&2
  git -C "$CLEAN_ROOT" status --short >&2
  exit 1
fi

timestamp="$(date -u +%Y%m%d_%H%M%S)"
out_dir="${DMI_BENCH_OUT_DIR:-$REPO_ROOT/artifacts/megatron_bench/olmoe_pure_baseline_$timestamp}"

echo "[DMI bench] OLMoE pure Megatron baseline"
echo "  clean Megatron:  $CLEAN_ROOT"
echo "  base commit:     $BASE_COMMIT"
echo "  conda env:       $DMI_CONDA_ENV"
echo "                   must be able to run the pure Megatron-LM clone"
echo "  out_dir:         $out_dir"
echo "  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "  DMI_REAL_E2E_CUDA_VISIBLE_DEVICES=${DMI_REAL_E2E_CUDA_VISIBLE_DEVICES:-<unset>}"
echo

cmd=(
  python tests/run_megatron_dmi_benchmark.py
  --benchmark-model olmoe
  --megatron-root "$CLEAN_ROOT"
  --dmi-mode off-only
  --case "${DMI_BENCH_CASE:-eager_tp1_pp2}"
  --hook-selection "${DMI_BENCH_HOOK_SELECTION:-hidden-states}"
  --repeat "${DMI_BENCH_REPEAT:-1}"
  --default-train-iters "${DMI_BENCH_TRAIN_ITERS:-10}"
  --eval-iters "${DMI_BENCH_EVAL_ITERS:-1}"
  --eval-interval "${DMI_BENCH_EVAL_INTERVAL:-5}"
  --micro-batch-size "${DMI_BENCH_MICRO_BATCH_SIZE:-2}"
  --timeout-s "${DMI_BENCH_TIMEOUT_S:-900}"
  --out-dir "$out_dir"
)
if [[ -n "${DMI_BENCH_SEQ_LENGTH:-}" ]]; then
  cmd+=(--seq-length "$DMI_BENCH_SEQ_LENGTH")
fi

"${cmd[@]}" "$@"

echo "[DMI bench] pure Megatron baseline completed: $out_dir"
