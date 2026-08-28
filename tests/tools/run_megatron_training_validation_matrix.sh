#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

if [[ -z "${DMI_MEGATRON_PYTHON:-}" ]]; then
  echo "DMI_MEGATRON_PYTHON must name the qualification Python executable" >&2
  exit 2
fi
if [[ ! -x "$DMI_MEGATRON_PYTHON" ]]; then
  echo "DMI_MEGATRON_PYTHON is not executable: $DMI_MEGATRON_PYTHON" >&2
  exit 2
fi

export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export DMI_REAL_E2E_TIMEOUT_S="${DMI_REAL_E2E_TIMEOUT_S:-240}"
export DMI_REAL_E2E_CLICKHOUSE_TIMEOUT_S="${DMI_REAL_E2E_CLICKHOUSE_TIMEOUT_S:-20}"
if [[ -n "${DMI_REAL_E2E_CUDA_VISIBLE_DEVICES:-}" && -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  export CUDA_VISIBLE_DEVICES="$DMI_REAL_E2E_CUDA_VISIBLE_DEVICES"
fi

"$DMI_MEGATRON_PYTHON" -c '
import torch

if not torch.cuda.is_available():
    raise SystemExit("DMI Megatron validation requires CUDA, but torch.cuda.is_available() is false")
device_count = torch.cuda.device_count()
if device_count < 2:
    raise SystemExit(
        f"DMI Megatron validation requires at least two visible CUDA devices, found {device_count}"
    )
print(f"[DMI] CUDA preflight: {device_count} visible devices")
'

echo "[DMI] Megatron validation matrix"
echo "  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "  DMI_REAL_E2E_CUDA_VISIBLE_DEVICES=${DMI_REAL_E2E_CUDA_VISIBLE_DEVICES:-<unset>}"
echo "  DMX_DB_HOST=${DMX_DB_HOST:-localhost}"
echo "  DMX_DB_PORT=${DMX_DB_PORT:-9000}"
echo

echo "[DMI] Unit/regression checks for Megatron storage and runtime"
"$DMI_MEGATRON_PYTHON" -m pytest -q \
  tests/test_megatron_file_sink.py \
  tests/test_megatron_adapter.py \
  tests/test_megatron_metadata_context.py \
  tests/test_megatron_schedule_runtime.py \
  tests/test_megatron_startup.py \
  tests/test_megatron_e2e_clickhouse.py \
  tests/test_megatron_router_summary.py \
  tests/test_megatron_loss_summary.py \
  tests/test_megatron_training_materialization.py \
  tests/test_megatron_ep_reconstruction.py \
  tests/test_megatron_ep_clickhouse_reconstruction.py \
  tests/test_megatron_ep_topology_manifest.py

echo
echo "[DMI] Real hidden-state E2E: SEQ_PREFIX_PACK ClickHouse rows"
"$DMI_MEGATRON_PYTHON" -m pytest -q -s \
  tests/test_megatron_real_training_e2e.py::test_real_megatron_hidden_states_clickhouse_rows

echo
echo "[DMI] Real numeric E2E: eager file sink vs eager ClickHouse, router-summary + loss-summary"
"$DMI_MEGATRON_PYTHON" -m pytest -q -s \
  tests/test_megatron_real_training_e2e.py::test_real_megatron_router_and_loss_summary_file_sink_matches_clickhouse_numeric \
  tests/test_megatron_real_training_e2e.py::test_real_megatron_router_and_loss_summary_eval_only_file_sink_matches_clickhouse_numeric

echo
echo "[DMI] Real numeric E2E: CUDA graph ClickHouse vs eager file-sink reference"
echo "[DMI] Nonzero max error is printed as WARNING when within tolerance."
"$DMI_MEGATRON_PYTHON" -m pytest -q -s \
  tests/test_megatron_real_training_e2e.py::test_real_megatron_router_and_loss_summary_graph_matches_eager_file_sink_numeric

echo
echo "[DMI] Real iteration records: health signals, materialization, grad-norm semantics, and PP=2 router weights"
"$DMI_MEGATRON_PYTHON" -m pytest -q -s \
  tests/test_megatron_real_training_e2e.py::test_real_megatron_training_health_signals_clickhouse_rows \
  tests/test_megatron_real_training_e2e.py::test_real_megatron_grad_norm_preserves_megatron_value_when_clip_disabled \
  tests/test_megatron_real_training_e2e.py::test_real_megatron_router_weights_cover_pipeline_stages_once \
  tests/test_megatron_real_training_e2e.py::test_real_megatron_iteration_hooks_optimizer_modes

echo
echo "[DMI] Real rerun and DP contracts: stable PP=2 IDs, one DP=2 grad row, and pre-engine router rejection"
"$DMI_MEGATRON_PYTHON" -m pytest -q -s \
  tests/test_megatron_real_training_e2e.py::test_real_megatron_pp2_rerun_reuses_logical_training_ids \
  tests/test_megatron_real_training_e2e.py::test_real_megatron_dp2_emits_one_global_grad_norm \
  tests/test_megatron_real_training_e2e.py::test_real_megatron_dp2_rejects_router_weights_before_rows

echo
echo "[DMI] Real recompute identity: eager, selective-MoE local graph, and full-iteration graph"
"$DMI_MEGATRON_PYTHON" -m pytest -q -s \
  tests/test_megatron_real_training_e2e.py::test_real_megatron_retained_recompute_invocation_identity

echo
echo "[DMI] Real composite identity: PP=2 mixed provenance and rerun + recompute + provenance"
"$DMI_MEGATRON_PYTHON" -m pytest -q -s \
  tests/test_megatron_real_training_e2e.py::test_real_megatron_pp2_mixed_dataset_provenance \
  tests/test_megatron_real_training_e2e.py::test_real_megatron_combined_attempt_invocation_and_dataset_identity

echo
echo "[DMI] Real execution matrix: eager/local/full-iteration graphs, with and without supported recomputation, x TP=1/PP={1,2}"
echo "[DMI] Expected skips: PP=2 local and full-iteration graph cases, due native Megatron graph issues. Eager full recompute runs on PP=2."
"$DMI_MEGATRON_PYTHON" -m pytest -q -s \
  tests/test_megatron_real_training_e2e.py::test_real_megatron_training_router_and_loss_summary_exact_clickhouse_rows

echo
echo "[DMI] Real EP reconstruction E2E matrix"
"$DMI_MEGATRON_PYTHON" -m pytest -q -s \
  tests/test_megatron_ep_reconstruction_e2e.py

echo
echo "[DMI] Megatron validation matrix completed"
