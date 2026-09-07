#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNNER="$SCRIPT_DIR/run_qwen3_1p7b_pp2_d2h_window_nsys.sh"
NVIDIA_SMI="${DMI_NVIDIA_SMI_BIN:-nvidia-smi}"

POLL_SECONDS="${DMI_GPU_POLL_SECONDS:-10}"
MAX_USED_MEMORY_MIB="${DMI_GPU_MAX_USED_MEMORY_MIB:-1024}"
MAX_UTILIZATION_PERCENT="${DMI_GPU_MAX_UTILIZATION_PERCENT:-5}"

usage() {
    cat <<EOF
Usage: $0 [options]

Wait until two GPUs meet both availability thresholds, then launch the
Qwen3-1.7B PP=2 recurring-D2H-window Nsight experiment on those GPUs.

Options:
  --poll-seconds SECONDS             Poll interval (default: $POLL_SECONDS)
  --max-used-memory-mib MIB          Maximum used memory (default: $MAX_USED_MEMORY_MIB)
  --max-utilization-percent PERCENT  Maximum GPU utilization (default: $MAX_UTILIZATION_PERCENT)
  -h, --help                         Show this help
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --poll-seconds)
            [[ $# -ge 2 ]] || { echo "--poll-seconds requires a value" >&2; exit 2; }
            POLL_SECONDS="$2"
            shift 2
            ;;
        --max-used-memory-mib)
            [[ $# -ge 2 ]] || { echo "--max-used-memory-mib requires a value" >&2; exit 2; }
            MAX_USED_MEMORY_MIB="$2"
            shift 2
            ;;
        --max-utilization-percent)
            [[ $# -ge 2 ]] || { echo "--max-utilization-percent requires a value" >&2; exit 2; }
            MAX_UTILIZATION_PERCENT="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [[ ! "$POLL_SECONDS" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)$ ]] || \
   ! awk -v value="$POLL_SECONDS" 'BEGIN { exit !(value > 0) }'; then
    echo "--poll-seconds must be greater than zero" >&2
    exit 2
fi
if [[ ! "$MAX_USED_MEMORY_MIB" =~ ^[0-9]+$ ]]; then
    echo "--max-used-memory-mib must be a nonnegative integer" >&2
    exit 2
fi
if [[ ! "$MAX_UTILIZATION_PERCENT" =~ ^[0-9]+$ ]] || \
   (( MAX_UTILIZATION_PERCENT > 100 )); then
    echo "--max-utilization-percent must be an integer from 0 to 100" >&2
    exit 2
fi
if [[ ! -x "$RUNNER" ]]; then
    echo "Missing executable Part I runner: $RUNNER" >&2
    exit 1
fi
if ! command -v "$NVIDIA_SMI" >/dev/null 2>&1; then
    echo "Cannot find nvidia-smi executable: $NVIDIA_SMI" >&2
    exit 1
fi

echo "Waiting for two GPUs with used memory <= ${MAX_USED_MEMORY_MIB} MiB and utilization <= ${MAX_UTILIZATION_PERCENT}%"
echo "Polling every ${POLL_SECONDS} seconds"

while true; do
    if ! sample="$("$NVIDIA_SMI" \
        --query-gpu=index,memory.used,utilization.gpu \
        --format=csv,noheader,nounits)"; then
        echo "nvidia-smi query failed; retrying in ${POLL_SECONDS} seconds" >&2
        sleep "$POLL_SECONDS"
        continue
    fi

    free_gpus=()
    while IFS=',' read -r index used_memory utilization; do
        index="${index//[[:space:]]/}"
        used_memory="${used_memory//[[:space:]]/}"
        utilization="${utilization//[[:space:]]/}"
        if [[ "$index" =~ ^[0-9]+$ && "$used_memory" =~ ^[0-9]+$ && \
              "$utilization" =~ ^[0-9]+$ ]] && \
           (( used_memory <= MAX_USED_MEMORY_MIB && \
              utilization <= MAX_UTILIZATION_PERCENT )); then
            free_gpus+=("$index")
        fi
    done <<<"$sample"

    if (( ${#free_gpus[@]} >= 2 )); then
        selected="${free_gpus[0]},${free_gpus[1]}"
        echo "Selected GPUs: $selected"
        exec "$RUNNER" --gpus "$selected"
    fi

    timestamp="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "$timestamp: found ${#free_gpus[@]} eligible GPU(s); waiting"
    sleep "$POLL_SECONDS"
done
