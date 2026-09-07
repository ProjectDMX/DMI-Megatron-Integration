#!/usr/bin/env bash
set -euo pipefail

# Part I of RECURRING_D2H_WINDOWS_VERIFICATION_PLAN.md.
# This uses the Qwen3-1.7B architecture with random initialization and mock
# tokens. Traffic placement depends on the model shape and schedule, not on
# pretrained parameter values.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INTEGRATION_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
WORKSPACE_ROOT="$(cd "$INTEGRATION_ROOT/../.." && pwd)"
MEGATRON_ROOT="${DMI_NSYS_MEGATRON_ROOT:-$INTEGRATION_ROOT/third_party/megatron-lm}"
ENV_PREFIX="${DMI_NSYS_ENV_PREFIX:-$WORKSPACE_ROOT/.conda-env-megatron}"
NSYS="${DMI_NSYS_BIN:-/usr/local/cuda/bin/nsys}"
NVIDIA_SMI="${DMI_NVIDIA_SMI_BIN:-nvidia-smi}"

CUDA_DEVICES="${DMI_NSYS_CUDA_VISIBLE_DEVICES:-1,2}"
TRAIN_ITERS="${DMI_NSYS_TRAIN_ITERS:-11}"
SEQ_LENGTH="${DMI_NSYS_SEQ_LENGTH:-1024}"
MICRO_BATCH_SIZE="${DMI_NSYS_MICRO_BATCH_SIZE:-1}"
GLOBAL_BATCH_SIZE="${DMI_NSYS_GLOBAL_BATCH_SIZE:-2}"
RING_PAYLOAD_MB="${DMI_NSYS_RING_PAYLOAD_MB:-512}"
RING_PINNED_MB="${DMI_NSYS_RING_PINNED_MB:-512}"
RING_TASK_ENTRIES="${DMI_NSYS_RING_TASK_ENTRIES:-4096}"
CH_PARALLELISM="${DMI_NSYS_CH_PARALLELISM:-4}"
NORMAL_DRAIN_BYTE_THRESHOLD="${DMI_NSYS_NORMAL_DRAIN_BYTE_THRESHOLD:-67108864}"
TIMING_REVALIDATION_RETRY_INTERVAL_OCCURRENCES="${DMI_NSYS_TIMING_REVALIDATION_RETRY_INTERVAL_OCCURRENCES:-100}"

DB_HOST="${DMX_DB_HOST:-localhost}"
DB_PORT="${DMX_DB_PORT:-9000}"
DB_DATABASE="${DMX_DB_DATABASE:-default}"
TIMESTAMP="$(date -u +%Y%m%d_%H%M%S)"
OUT_DIR="${DMI_NSYS_OUT_DIR:-$INTEGRATION_ROOT/artifacts/recurring_d2h_windows/qwen3_1p7b_pp2_$TIMESTAMP}"
TABLE="${DMI_NSYS_TABLE:-dmi_qwen3_1p7b_pp2_nsys_$TIMESTAMP}"

usage() {
    echo "Usage: $0 [--gpus GPU0,GPU1]"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpus)
            if [[ $# -lt 2 ]]; then
                echo "--gpus requires a comma-separated pair, for example: --gpus 1,2" >&2
                exit 2
            fi
            CUDA_DEVICES="$2"
            shift 2
            ;;
        --gpus=*)
            CUDA_DEVICES="${1#*=}"
            shift
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

if [[ ! "$CUDA_DEVICES" =~ ^[0-9]+,[0-9]+$ ]]; then
    echo "--gpus must contain exactly two comma-separated GPU indices" >&2
    exit 2
fi
if [[ "${CUDA_DEVICES%,*}" == "${CUDA_DEVICES#*,}" ]]; then
    echo "--gpus must select two different GPU indices" >&2
    exit 2
fi

if [[ ! -d "$ENV_PREFIX" ]]; then
    echo "Missing repository-local environment: $ENV_PREFIX" >&2
    exit 1
fi
ENV_PREFIX="$(cd "$ENV_PREFIX" && pwd)"
PYTHON="$ENV_PREFIX/bin/python"
if [[ ! -x "$PYTHON" ]]; then
    echo "Missing repository-local Python: $PYTHON" >&2
    exit 1
fi
if ! resolved_nsys="$(command -v "$NSYS")"; then
    echo "Missing Nsight Systems CLI: $NSYS" >&2
    exit 1
fi
NSYS="$resolved_nsys"
if [[ ! -d "$MEGATRON_ROOT" ]]; then
    echo "Missing Megatron checkout: $MEGATRON_ROOT" >&2
    exit 1
fi
MEGATRON_ROOT="$(cd "$MEGATRON_ROOT" && pwd)"
if [[ ! -f "$MEGATRON_ROOT/pretrain_gpt.py" ]]; then
    echo "Missing Megatron entry point: $MEGATRON_ROOT/pretrain_gpt.py" >&2
    exit 1
fi
if ! resolved_nvidia_smi="$(command -v "$NVIDIA_SMI")"; then
    echo "Cannot find nvidia-smi executable: $NVIDIA_SMI" >&2
    exit 1
fi
NVIDIA_SMI="$resolved_nvidia_smi"
if [[ "$GLOBAL_BATCH_SIZE" -ne $((2 * MICRO_BATCH_SIZE)) ]]; then
    echo "GLOBAL_BATCH_SIZE must equal 2 * MICRO_BATCH_SIZE so PP=2 uses M=2" >&2
    exit 1
fi
if [[ ! "$TRAIN_ITERS" =~ ^[0-9]+$ ]] || (( TRAIN_ITERS < 11 )); then
    echo "DMI_NSYS_TRAIN_ITERS must be an integer greater than or equal to 11" >&2
    exit 1
fi
if [[ ! "$TIMING_REVALIDATION_RETRY_INTERVAL_OCCURRENCES" =~ ^[0-9]+$ ]] || \
   (( TIMING_REVALIDATION_RETRY_INTERVAL_OCCURRENCES < 100 )); then
    echo "DMI_NSYS_TIMING_REVALIDATION_RETRY_INTERVAL_OCCURRENCES must be an integer greater than or equal to 100" >&2
    exit 1
fi

mkdir -p "$OUT_DIR"
OUT_DIR="$(cd "$OUT_DIR" && pwd)"

available_gpus=()
while IFS= read -r index; do
    index="${index//[[:space:]]/}"
    [[ "$index" =~ ^[0-9]+$ ]] && available_gpus+=("$index")
done < <("$NVIDIA_SMI" --query-gpu=index --format=csv,noheader,nounits)
for requested in "${CUDA_DEVICES%,*}" "${CUDA_DEVICES#*,}"; do
    found=0
    for available in "${available_gpus[@]}"; do
        if [[ "$requested" == "$available" ]]; then
            found=1
            break
        fi
    done
    if (( found == 0 )); then
        echo "Requested GPU $requested is not present" >&2
        exit 1
    fi
done

export CUDA_HOME="$ENV_PREFIX"
export CUDA_VISIBLE_DEVICES="$CUDA_DEVICES"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export PYTHONPATH="$WORKSPACE_ROOT:$INTEGRATION_ROOT:$MEGATRON_ROOT${PYTHONPATH:+:$PYTHONPATH}"

# Megatron utilities may resolve additional paths from the process working
# directory. Keep the caller's working directory irrelevant.
cd "$MEGATRON_ROOT"

# Validate identifiers and create the existing tensor-record table schema.
"$PYTHON" - "$DB_HOST" "$DB_PORT" "$DB_DATABASE" "$TABLE" <<'PY'
import re
import sys

from clickhouse_driver import Client

host, port, database, table = sys.argv[1:]
identifier = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
if not identifier.fullmatch(database) or not identifier.fullmatch(table):
    raise SystemExit("database and table must be unquoted ClickHouse identifiers")

client = Client(host=host, port=int(port), database=database)
client.execute(f"CREATE DATABASE IF NOT EXISTS `{database}`")
client.execute(
    f"""
    CREATE TABLE IF NOT EXISTS `{database}`.`{table}` (
        `model_id` String,
        `act_name` String,
        `direction` String,
        `phase` String,
        `global_batch_id` Int64,
        `dp_rank` Int32,
        `microbatch_id` Int32,
        `sample_index` Int32,
        `layer_no` Int32,
        `shard_rank` Int32,
        `token_start` Int64,
        `token_end` Int64,
        `attempt_id` Int32,
        `invocation_id` Int32,
        `dataset_id` Int32,
        `dtype` String,
        `shape` Array(Int64),
        `bytes` String
    ) ENGINE = MergeTree
    PRIMARY KEY (
        `model_id`, `act_name`, `direction`, `phase`, `global_batch_id`, `dp_rank`,
        `microbatch_id`, `sample_index`, `layer_no`, `shard_rank`,
        `token_start`, `token_end`, `attempt_id`, `invocation_id`
    )
    ORDER BY (
        `model_id`, `act_name`, `direction`, `phase`, `global_batch_id`, `dp_rank`,
        `microbatch_id`, `sample_index`, `layer_no`, `shard_rank`,
        `token_start`, `token_end`, `attempt_id`, `invocation_id`
    )
    """
)
PY

cat >"$OUT_DIR/run_config.txt" <<EOF
model=Qwen3-1.7B architecture, random initialization
megatron_root=$MEGATRON_ROOT
cuda_visible_devices=$CUDA_DEVICES
train_iters=$TRAIN_ITERS
seq_length=$SEQ_LENGTH
micro_batch_size=$MICRO_BATCH_SIZE
global_batch_size=$GLOBAL_BATCH_SIZE
pipeline_model_parallel_size=2
tensor_model_parallel_size=1
ring_payload_mb=$RING_PAYLOAD_MB
ring_pinned_mb=$RING_PINNED_MB
ring_task_entries=$RING_TASK_ENTRIES
normal_drain_byte_threshold=$NORMAL_DRAIN_BYTE_THRESHOLD
timing_revalidation_retry_interval_occurrences=$TIMING_REVALIDATION_RETRY_INTERVAL_OCCURRENCES
clickhouse_host=$DB_HOST
clickhouse_port=$DB_PORT
clickhouse_database=$DB_DATABASE
clickhouse_table=$TABLE
EOF

base_command=(
    "$PYTHON" -m torch.distributed.run
    --standalone
    --nproc_per_node=2
    "$MEGATRON_ROOT/pretrain_gpt.py"
    --mock-data
    --tokenizer-type NullTokenizer
    --vocab-size 151936
    --make-vocab-size-divisible-by 128
    --num-layers 28
    --hidden-size 2048
    --ffn-hidden-size 6144
    --num-attention-heads 16
    --kv-channels 128
    --group-query-attention
    --num-query-groups 8
    --normalization RMSNorm
    --norm-epsilon 1e-6
    --qk-layernorm
    --position-embedding-type rope
    --rotary-percent 1.0
    --rotary-base 1000000
    --use-rotary-position-embeddings
    --swiglu
    --disable-bias-linear
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --attention-softmax-in-fp32
    --no-masked-softmax-fusion
    --seq-length "$SEQ_LENGTH"
    --max-position-embeddings 40960
    --micro-batch-size "$MICRO_BATCH_SIZE"
    --global-batch-size "$GLOBAL_BATCH_SIZE"
    --tensor-model-parallel-size 1
    --pipeline-model-parallel-size 2
    --train-iters "$TRAIN_ITERS"
    --eval-interval 100
    --eval-iters 0
    --log-interval 1
    --seed 1234
    --lr 1.0e-5
    --min-lr 1.0e-5
    --lr-decay-style constant
    --lr-decay-iters "$TRAIN_ITERS"
    --lr-warmup-iters 0
    --weight-decay 0.0
    --adam-beta1 0.9
    --adam-beta2 0.95
    --clip-grad 1.0
    --bf16
    --transformer-impl local
    --no-persist-layer-norm
    --no-gradient-accumulation-fusion
    --recompute-granularity selective
    --recompute-activations
    --recompute-modules core_attn
    --no-save-optim
    --no-save-rng
    --no-load-optim
    --no-load-rng
    --no-create-attention-mask-in-dataloader
    --dmi-enable
    --dmi-hook-selection hidden-states
    --dmi-db-host "$DB_HOST"
    --dmi-db-port "$DB_PORT"
    --dmi-db-database "$DB_DATABASE"
    --dmi-clickhouse-table "$TABLE"
    --dmi-ch-parallelism "$CH_PARALLELISM"
    --dmi-ring-payload-mb "$RING_PAYLOAD_MB"
    --dmi-ring-pinned-mb "$RING_PINNED_MB"
    --dmi-ring-task-entries "$RING_TASK_ENTRIES"
    --dmi-flush-every-n-train-iters 0
    --dmi-d2h-window-timing-revalidation-retry-interval-occurrences \
        "$TIMING_REVALIDATION_RETRY_INTERVAL_OCCURRENCES"
)

run_case() {
    local case_name="$1"
    local windows_enabled="$2"
    local debug_enabled="$3"
    local case_dir="$OUT_DIR/$case_name"
    local model_id="qwen3-1p7b-pp2-${case_name}-${TIMESTAMP}"
    local report_base="$case_dir/trace"
    local command_file="$case_dir/command.txt"
    local log_file="$case_dir/training.log"
    local command=("${base_command[@]}" --dmi-model-id "$model_id")

    if [[ "$windows_enabled" == "1" ]]; then
        command+=(--dmi-recurring-d2h-windows)
    fi
    if [[ "$debug_enabled" == "1" ]]; then
        command+=(--dmi-d2h-window-debug)
    fi

    mkdir -p "$case_dir"
    printf 'DMI_RECURRING_D2H_WINDOWS=%q DMI_D2H_WINDOW_DEBUG=%q ' \
        "$windows_enabled" "$debug_enabled" >"$command_file"
    printf '%q ' "$NSYS" profile \
        --trace=cuda,nvtx,osrt \
        --sample=none \
        --cpuctxsw=none \
        --cuda-event-trace=false \
        --force-overwrite=true \
        --export=sqlite \
        --output="$report_base" \
        "${command[@]}" >>"$command_file"
    printf '\n' >>"$command_file"

    echo "Running $case_name; output: $case_dir"
    DMI_ENABLE=1 \
    DMI_RECURRING_D2H_WINDOWS="$windows_enabled" \
    DMI_D2H_WINDOW_DEBUG="$debug_enabled" \
    DMI_DRAIN_FLUSH_PAYLOAD_RATIO=0 \
    DMI_DRAIN_FLUSH_TASK_RATIO=0 \
    DMI_DRAIN_FLUSH_BYTE_THRESHOLD="$NORMAL_DRAIN_BYTE_THRESHOLD" \
    DMI_DRAIN_FLUSH_ENTRY_THRESHOLD=0 \
    DMI_DRAIN_FLUSH_TIMEOUT_US=0 \
    "$NSYS" profile \
        --trace=cuda,nvtx,osrt \
        --sample=none \
        --cpuctxsw=none \
        --cuda-event-trace=false \
        --force-overwrite=true \
        --export=sqlite \
        --output="$report_base" \
        "${command[@]}" 2>&1 | tee "$log_file"

    "$PYTHON" - "$DB_HOST" "$DB_PORT" "$DB_DATABASE" "$TABLE" \
        "$model_id" "$TRAIN_ITERS" "$GLOBAL_BATCH_SIZE" <<'PY' \
        >"$case_dir/clickhouse_summary.txt"
import sys
import time

from clickhouse_driver import Client

host, port, database, table, model_id, train_iters, global_batch_size = sys.argv[1:]
expected = int(train_iters) * int(global_batch_size) * 28
client = Client(host=host, port=int(port), database=database)
deadline = time.monotonic() + 60.0
row = (0, 0)
while time.monotonic() < deadline:
    row = client.execute(
        f"""
        SELECT count(), sum(length(bytes))
        FROM `{database}`.`{table}`
        WHERE model_id = %(model_id)s
          AND act_name = 'hidden_states'
          AND direction = 'fwd'
          AND phase = 'train'
        """,
        {"model_id": model_id},
    )[0]
    if int(row[0]) == expected:
        break
    time.sleep(0.2)

print(f"model_id={model_id}")
print(f"expected_rows={expected}")
print(f"actual_rows={int(row[0])}")
print(f"payload_bytes={int(row[1])}")
if int(row[0]) != expected:
    raise SystemExit("hidden-state row count did not reach the expected value")
PY
}

run_case normal_batching 0 0
run_case window_scheduled 1 1

"$PYTHON" "$SCRIPT_DIR/analyze_d2h_window_nsys.py" \
    --normal "$OUT_DIR/normal_batching/trace.sqlite" \
    --window "$OUT_DIR/window_scheduled/trace.sqlite" \
    --window-log "$OUT_DIR/window_scheduled/training.log" \
    --output-dir "$OUT_DIR/analysis" \
    --require-valid-window-run

"$PYTHON" "$SCRIPT_DIR/plot_d2h_window_nsys_ascii.py" \
    --normal "$OUT_DIR/normal_batching/trace.sqlite" \
    --window "$OUT_DIR/window_scheduled/trace.sqlite" \
    --output "$OUT_DIR/analysis/timeline.txt"

echo "Part I traces and analysis: $OUT_DIR"
