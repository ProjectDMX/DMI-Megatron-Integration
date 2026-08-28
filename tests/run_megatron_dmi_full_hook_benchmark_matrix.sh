#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

export DMI_CONDA_ENV="${DMI_CONDA_ENV:-ring_offload}"

if [[ "${CONDA_DEFAULT_ENV:-}" != "$DMI_CONDA_ENV" ]]; then
  if [[ -f "$HOME/miniforge3/etc/profile.d/conda.sh" ]]; then
    # shellcheck disable=SC1091
    source "$HOME/miniforge3/etc/profile.d/conda.sh"
    conda activate "$DMI_CONDA_ENV"
  fi
fi

timestamp="$(date -u +%Y%m%d_%H%M%S)"
out_root="${DMI_BENCH_OUT_ROOT:-$REPO_ROOT/artifacts/megatron_bench/full_hook_matrix_$timestamp}"
mkdir -p "$out_root"

hooks=(
  router-summary
  loss-summary
  hidden-states
)

echo "[DMI bench] full Megatron hook benchmark matrix"
echo "  out_root=$out_root"
echo "  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "  DMI_REAL_E2E_CUDA_VISIBLE_DEVICES=${DMI_REAL_E2E_CUDA_VISIBLE_DEVICES:-<unset>}"
echo "  DMX_DB_HOST=${DMX_DB_HOST:-localhost}"
echo "  DMX_DB_PORT=${DMX_DB_PORT:-9000}"
echo "  hooks=${hooks[*]}"
echo

for hook in "${hooks[@]}"; do
  hook_dir="$out_root/$hook"
  echo "[DMI bench] start hook=$hook out=$hook_dir"
  python tests/run_megatron_dmi_benchmark.py \
    --hook-selection "$hook" \
    --out-dir "$hook_dir" \
    "$@"
  echo "[DMI bench] done hook=$hook"
  echo
done

echo "[DMI bench] full hook benchmark matrix completed: $out_root"
