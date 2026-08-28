#!/usr/bin/env python3
"""Launch the fixed real-OLMoE expert-contribution capture.

The launcher consumes a separately converted Megatron checkpoint.  It never
downloads or converts model weights.  ``--dry-run`` writes the complete run
manifest and resolved shell command without requiring the converted checkpoint
to exist or starting Megatron.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
MEGATRON_ROOT = REPO_ROOT / "third_party" / "megatron-lm"
GROKKING_SCRIPTS = REPO_ROOT / "tools" / "grokking" / "scripts"
if str(GROKKING_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(GROKKING_SCRIPTS))

from olmoe_megatron_conversion import (  # noqa: E402
    megatron_argv_from_olmoe,
)


DEFAULT_DATA_PREFIX = (
    REPO_ROOT
    / "artifacts"
    / "grokking_repro"
    / "megatron_datasets"
    / "math_open_web_math_1000"
    / "math_open_web_math_1000_text_document"
)
EXPECTED_HF_REPOSITORY = "allenai/OLMoE-1B-7B-0924"
EXPECTED_HF_REVISION = "6d84c48581ece794365f2b8e9cfb043c68ade9c5"

HOOK_SELECTION = "router-topk,moe-inverse-map,moe-packed-weighted-output"
HOOK_SELECTORS = tuple(HOOK_SELECTION.split(","))
PAYLOAD_ACT_NAMES = (
    "router_topk_expert_ids",
    "router_topk_weights",
    "moe_inverse_map",
    "moe_packed_weighted_output",
)
NO_RECOMPUTE_HOOKS = HOOK_SELECTION

# Fixed, audited workload.  These are deliberately not command-line knobs.
CUDA_VISIBLE_DEVICES = "1,2"
NPROC_PER_NODE = 2
TP_SIZE = 1
PP_SIZE = 1
EP_SIZE = 2
ETP_SIZE = 1
CP_SIZE = 1
SEQ_LENGTH = 512
MICRO_BATCH_SIZE = 1
GLOBAL_BATCH_SIZE = 2
TRAIN_ITERS = 10
FIXED_LR = "4.6159e-5"
MOE_AUX_LOSS_COEFF = "0.01"
RING_PAYLOAD_MB = 1024
RING_PINNED_MB = 1024
RING_TASK_ENTRIES = 65536
CH_PARALLELISM = 10
def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-dir", type=Path, required=True)
    parser.add_argument("--megatron-dir", type=Path, required=True)
    parser.add_argument("--data-prefix", type=Path, default=DEFAULT_DATA_PREFIX)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--run-name",
        default=None,
        help="Stable run name. Reuse the same value after auditing a dry run.",
    )
    parser.add_argument("--db-host", default=os.environ.get("DMX_DB_HOST", "localhost"))
    parser.add_argument("--db-port", type=int, default=int(os.environ.get("DMX_DB_PORT", "9000")))
    parser.add_argument("--dry-run", action="store_true")
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return _parser().parse_args(argv)


def _remove_option(argv: Sequence[str], flag: str, *, has_value: bool) -> list[str]:
    output: list[str] = []
    index = 0
    while index < len(argv):
        if argv[index] == flag:
            index += 2 if has_value else 1
            continue
        output.append(str(argv[index]))
        index += 1
    return output


def _replace_option(argv: Sequence[str], flag: str, value: object) -> list[str]:
    output = _remove_option(argv, flag, has_value=True)
    output.extend((flag, str(value)))
    return output


def _ensure_flag(argv: Sequence[str], flag: str) -> list[str]:
    output = _remove_option(argv, flag, has_value=False)
    output.append(flag)
    return output


def _option(argv: Sequence[str], flag: str) -> str:
    positions = [index for index, value in enumerate(argv) if value == flag]
    if len(positions) != 1:
        raise RuntimeError(f"Expected exactly one {flag}, found {len(positions)}")
    index = positions[0]
    if index + 1 >= len(argv):
        raise RuntimeError(f"Missing value after {flag}")
    return str(argv[index + 1])


def _build_megatron_args(
    *,
    hf_dir: Path,
    megatron_dir: Path,
    data_prefix: Path,
    model_id: str,
    database: str,
    table: str,
    base_argv: Sequence[str] | None = None,
) -> list[str]:
    if base_argv is None:
        base_argv = megatron_argv_from_olmoe(
            hf_dir=hf_dir,
            load_dir=megatron_dir,
            extra_args=(),
            ckpt_format="torch_dist",
        )[1:]
    argv = list(base_argv)
    argv = _remove_option(argv, "--mock-data", has_value=False)

    replacements = {
        "--data-path": data_prefix,
        "--split": "900,50,50",
        "--micro-batch-size": MICRO_BATCH_SIZE,
        "--global-batch-size": GLOBAL_BATCH_SIZE,
        "--tensor-model-parallel-size": TP_SIZE,
        "--pipeline-model-parallel-size": PP_SIZE,
        "--expert-model-parallel-size": EP_SIZE,
        "--expert-tensor-parallel-size": ETP_SIZE,
        "--context-parallel-size": CP_SIZE,
        "--seq-length": SEQ_LENGTH,
        "--max-position-embeddings": SEQ_LENGTH,
        "--train-iters": TRAIN_ITERS,
        "--eval-iters": 0,
        "--eval-interval": TRAIN_ITERS + 1,
        "--lr": FIXED_LR,
        "--min-lr": FIXED_LR,
        "--lr-decay-style": "constant",
        "--lr-warmup-iters": 0,
        "--lr-decay-iters": TRAIN_ITERS,
        "--moe-token-dispatcher-type": "alltoall",
        "--moe-router-load-balancing-type": "aux_loss",
        "--moe-aux-loss-coeff": MOE_AUX_LOSS_COEFF,
        "--cuda-graph-impl": "none",
        "--optimizer-offload-fraction": "1.0",
        "--main-grads-dtype": "bf16",
        "--main-params-dtype": "fp16",
        "--exp-avg-dtype": "bf16",
        "--exp-avg-sq-dtype": "bf16",
        "--recompute-granularity": "full",
        "--recompute-method": "uniform",
        "--recompute-num-layers": 1,
        "--dmi-hook-selection": HOOK_SELECTION,
        "--dmi-no-recompute-hook": NO_RECOMPUTE_HOOKS,
        "--dmi-model-id": model_id,
    }
    for flag, value in replacements.items():
        argv = _replace_option(argv, flag, value)

    argv = _replace_option(argv, "--dmi-db-database", database)
    argv = _replace_option(argv, "--dmi-clickhouse-table", table)
    argv = _replace_option(argv, "--dmi-ch-parallelism", CH_PARALLELISM)
    argv = _replace_option(argv, "--dmi-ring-payload-mb", RING_PAYLOAD_MB)
    argv = _replace_option(argv, "--dmi-ring-pinned-mb", RING_PINNED_MB)
    argv = _replace_option(argv, "--dmi-ring-task-entries", RING_TASK_ENTRIES)

    for flag in (
        "--finetune",
        "--bf16",
        "--use-distributed-optimizer",
        "--use-precision-aware-optimizer",
        "--optimizer-cpu-offload",
        "--use-torch-optimizer-for-cpu-offload",
        "--no-pin-cpu-grads",
        "--no-pin-cpu-params",
        "--dmi-enable",
    ):
        argv = _ensure_flag(argv, flag)

    _assert_fixed_megatron_args(argv)
    return argv


def _assert_fixed_megatron_args(argv: Sequence[str]) -> None:
    expected_options = {
        "--micro-batch-size": "1",
        "--global-batch-size": "2",
        "--tensor-model-parallel-size": "1",
        "--pipeline-model-parallel-size": "1",
        "--expert-model-parallel-size": "2",
        "--expert-tensor-parallel-size": "1",
        "--context-parallel-size": "1",
        "--seq-length": "512",
        "--split": "900,50,50",
        "--train-iters": "10",
        "--moe-token-dispatcher-type": "alltoall",
        "--moe-aux-loss-coeff": "0.01",
        "--cuda-graph-impl": "none",
        "--optimizer-offload-fraction": "1.0",
        "--main-grads-dtype": "bf16",
        "--main-params-dtype": "fp16",
        "--exp-avg-dtype": "bf16",
        "--exp-avg-sq-dtype": "bf16",
        "--recompute-granularity": "full",
        "--recompute-method": "uniform",
        "--recompute-num-layers": "1",
        "--dmi-hook-selection": HOOK_SELECTION,
        "--dmi-no-recompute-hook": NO_RECOMPUTE_HOOKS,
        "--dmi-ring-payload-mb": "1024",
        "--dmi-ring-pinned-mb": "1024",
        "--dmi-ring-task-entries": "65536",
    }
    mismatches = {
        flag: (expected, _option(argv, flag))
        for flag, expected in expected_options.items()
        if _option(argv, flag) != expected
    }
    if mismatches:
        raise RuntimeError(f"Fixed OLMoE setup was changed: {mismatches}")

    required_flags = {
        "--finetune",
        "--bf16",
        "--use-distributed-optimizer",
        "--use-precision-aware-optimizer",
        "--optimizer-cpu-offload",
        "--use-torch-optimizer-for-cpu-offload",
        "--no-pin-cpu-grads",
        "--no-pin-cpu-params",
        "--dmi-enable",
    }
    missing = sorted(required_flags - set(argv))
    if missing:
        raise RuntimeError(f"Fixed OLMoE setup is missing flags: {missing}")

    forbidden = {
        "--mock-data",
        "--save",
        "--sequence-parallel",
        "--moe-expert-capacity-factor",
        "--moe-pad-expert-input-to-capacity",
        "--moe-permute-fusion",
        "--fine-grained-activation-offloading",
        "--cpu-offloading",
        "--optimizer-cuda-graph",
    }
    present = sorted(forbidden & set(argv))
    if present:
        raise RuntimeError(f"Fixed OLMoE setup contains forbidden flags: {present}")


def _run_token(run_name: str) -> str:
    token = re.sub(r"[^A-Za-z0-9_]", "_", run_name).strip("_")
    if not token:
        raise ValueError("--run-name must contain at least one letter or digit")
    return token[:80]


def _default_run_name() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"olmoe_expert_contribution_{stamp}_{uuid.uuid4().hex[:8]}"


def _read_json_if_present(path: Path) -> dict[str, object] | None:
    if not path.is_file():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _input_status(hf_dir: Path, megatron_dir: Path, data_prefix: Path) -> dict[str, bool]:
    return {
        "hf_config": (hf_dir / "config.json").is_file(),
        "hf_source_manifest": (hf_dir / "source_manifest.json").is_file(),
        "megatron_conversion_manifest": (
            megatron_dir / "dmi_olmoe_hf_to_megatron_manifest.json"
        ).is_file(),
        "megatron_checkpoint": megatron_dir.is_dir()
        and (megatron_dir / "latest_checkpointed_iteration.txt").is_file(),
        "openwebmath_bin": Path(f"{data_prefix}.bin").is_file(),
        "openwebmath_idx": Path(f"{data_prefix}.idx").is_file(),
    }


def _validate_execution_inputs(
    *,
    hf_dir: Path,
    megatron_dir: Path,
    data_prefix: Path,
    source_manifest: dict[str, object] | None,
    conversion_manifest: dict[str, object] | None,
) -> None:
    status = _input_status(hf_dir, megatron_dir, data_prefix)
    missing = sorted(name for name, exists in status.items() if not exists)
    if missing:
        raise FileNotFoundError(f"OLMoE execution inputs are incomplete: {missing}")
    if source_manifest is None:
        raise FileNotFoundError(hf_dir / "source_manifest.json")
    if source_manifest.get("repository") != EXPECTED_HF_REPOSITORY:
        raise ValueError("HF source manifest has an unexpected repository")
    if source_manifest.get("revision") != EXPECTED_HF_REVISION:
        raise ValueError("HF source manifest has an unexpected immutable revision")
    _validate_conversion_binding(
        hf_dir=hf_dir,
        source_manifest_path=hf_dir / "source_manifest.json",
        source_manifest=source_manifest,
        conversion_manifest=conversion_manifest,
    )


def _validate_conversion_binding(
    *,
    hf_dir: Path,
    source_manifest_path: Path,
    source_manifest: dict[str, object],
    conversion_manifest: dict[str, object] | None,
) -> None:
    if conversion_manifest is None:
        raise FileNotFoundError("Megatron checkpoint conversion manifest is required")
    expected = {
        "iteration": 1,
        "ckpt_format": "torch_dist",
        "hf_repository": source_manifest.get("repository"),
        "hf_revision": source_manifest.get("revision"),
        "hf_source_manifest_sha256": _sha256(source_manifest_path),
    }
    mismatches = {
        field: (value, conversion_manifest.get(field))
        for field, value in expected.items()
        if conversion_manifest.get(field) != value
    }
    recorded_hf_dir = conversion_manifest.get("hf_dir")
    if not isinstance(recorded_hf_dir, str):
        mismatches["hf_dir"] = (str(hf_dir.resolve()), recorded_hf_dir)
    elif Path(recorded_hf_dir).expanduser().resolve() != hf_dir.resolve():
        mismatches["hf_dir"] = (str(hf_dir.resolve()), recorded_hf_dir)
    if mismatches:
        raise ValueError(f"Megatron checkpoint source binding mismatch: {mismatches}")


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _resolved_shell_command(environment: dict[str, str], command: Sequence[str]) -> str:
    assignments = " ".join(
        f"{name}={shlex.quote(value)}" for name, value in sorted(environment.items())
    )
    return f"{assignments} {shlex.join(command)}"


def _training_environment(
    topology_manifest_path: Path,
) -> tuple[dict[str, str], dict[str, str]]:
    inherited = os.environ.copy()
    pythonpath = os.pathsep.join((str(REPO_ROOT), str(MEGATRON_ROOT), str(GROKKING_SCRIPTS)))
    explicit = {
        "CUDA_DEVICE_MAX_CONNECTIONS": "1",
        "CUDA_VISIBLE_DEVICES": CUDA_VISIBLE_DEVICES,
        "DMI_ENABLE": "1",
        "DMI_TOPOLOGY_MANIFEST_PATH": str(topology_manifest_path),
        "PYTHONPATH": pythonpath,
        "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD": "1",
    }
    inherited.update(explicit)
    return inherited, explicit


def _manifest_inputs(
    *, hf_dir: Path, megatron_dir: Path, data_prefix: Path
) -> tuple[dict[str, object] | None, dict[str, object] | None, dict[str, object]]:
    source_path = hf_dir / "source_manifest.json"
    conversion_path = megatron_dir / "dmi_olmoe_hf_to_megatron_manifest.json"
    source = _read_json_if_present(source_path)
    conversion = _read_json_if_present(conversion_path)
    provenance = {
        "hf_source_manifest_path": str(source_path),
        "hf_source_manifest_sha256": _sha256(source_path),
        "hf_source_manifest": source,
        "immutable_hf_repository": source.get("repository") if source else None,
        "immutable_hf_revision": source.get("revision") if source else None,
        "megatron_conversion_manifest_path": str(conversion_path),
        "megatron_conversion_manifest": conversion,
        "input_status": _input_status(hf_dir, megatron_dir, data_prefix),
    }
    return source, conversion, provenance


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    hf_dir = args.hf_dir.expanduser().resolve()
    megatron_dir = args.megatron_dir.expanduser().resolve()
    data_prefix = args.data_prefix.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()

    run_name = args.run_name or _default_run_name()
    token = _run_token(run_name)
    run_dir = output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    model_id = f"olmoe-expert-contribution-{token}"
    database = f"dmi_olmoe_ep_{token}"
    table = f"training_payload_{token}"
    topology_manifest_path = run_dir / "ep_topology.json"
    log_path = run_dir / "megatron_train.log"

    megatron_args = _build_megatron_args(
        hf_dir=hf_dir,
        megatron_dir=megatron_dir,
        data_prefix=data_prefix,
        model_id=model_id,
        database=database,
        table=table,
    )
    megatron_args = _replace_option(megatron_args, "--dmi-db-host", args.db_host)
    megatron_args = _replace_option(megatron_args, "--dmi-db-port", args.db_port)
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc_per_node={NPROC_PER_NODE}",
        str(MEGATRON_ROOT / "pretrain_gpt.py"),
        *megatron_args,
    ]
    environment, explicit_environment = _training_environment(topology_manifest_path)
    shell_command = _resolved_shell_command(explicit_environment, command)

    source_manifest, conversion_manifest, provenance = _manifest_inputs(
        hf_dir=hf_dir,
        megatron_dir=megatron_dir,
        data_prefix=data_prefix,
    )
    manifest_path = run_dir / "run_manifest.json"
    command_path = run_dir / "resolved_command.sh"
    manifest: dict[str, object] = {
        "kind": "dmi_real_olmoe_expert_contribution_capture",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "dry_run" if args.dry_run else "prepared",
        "run_name": run_name,
        "run_dir": str(run_dir),
        "model_id": model_id,
        "database": database,
        "table": table,
        "hf_dir": str(hf_dir),
        "megatron_dir": str(megatron_dir),
        "data_prefix": str(data_prefix),
        "hook_selectors": list(HOOK_SELECTORS),
        "payload_act_names": list(PAYLOAD_ACT_NAMES),
        "no_recompute_hook_selectors": list(HOOK_SELECTORS),
        "topology_manifest_path": str(topology_manifest_path),
        "training_log": str(log_path),
        "resolved_command_path": str(command_path),
        "setup": {
            "cuda_visible_devices": CUDA_VISIBLE_DEVICES,
            "nproc_per_node": NPROC_PER_NODE,
            "tensor_parallel_size": TP_SIZE,
            "pipeline_parallel_size": PP_SIZE,
            "expert_parallel_size": EP_SIZE,
            "expert_tensor_parallel_size": ETP_SIZE,
            "context_parallel_size": CP_SIZE,
            "sequence_length": SEQ_LENGTH,
            "micro_batch_size": MICRO_BATCH_SIZE,
            "global_batch_size": GLOBAL_BATCH_SIZE,
            "train_iters": TRAIN_ITERS,
            "model_dtype": "bf16",
            "optimizer": "distributed Adam with 100% CPU offload",
            "optimizer_state_dtypes": {
                "main_grads": "bf16",
                "main_params": "fp16",
                "exp_avg": "bf16",
                "exp_avg_sq": "bf16",
            },
            "activation_recomputation": "full uniform, one layer per recompute unit",
            "activation_offload": False,
            "cuda_graphs": False,
            "moe_dispatch": "alltoall, dropless, unpadded, non-fused permutation",
            "moe_aux_loss_coeff": float(MOE_AUX_LOSS_COEFF),
            "learning_rate": float(FIXED_LR),
            "checkpoint_save": False,
            "finetune_semantics": True,
            "ring_payload_mb": RING_PAYLOAD_MB,
            "ring_pinned_mb": RING_PINNED_MB,
            "ring_task_entries": RING_TASK_ENTRIES,
        },
        "provenance": provenance,
        "environment": explicit_environment,
        "command": command,
        "shell_command": shell_command,
    }
    _write_json(manifest_path, manifest)
    command_path.write_text("#!/usr/bin/env bash\nset -euo pipefail\n" + shell_command + "\n")

    print(json.dumps(manifest, indent=2, sort_keys=True))
    if args.dry_run:
        return 0

    _validate_execution_inputs(
        hf_dir=hf_dir,
        megatron_dir=megatron_dir,
        data_prefix=data_prefix,
        source_manifest=source_manifest,
        conversion_manifest=conversion_manifest,
    )
    manifest["status"] = "running"
    manifest["started_at"] = datetime.now(timezone.utc).isoformat()
    _write_json(manifest_path, manifest)
    with log_path.open("w", encoding="utf-8") as log:
        result = subprocess.run(
            command,
            cwd=str(MEGATRON_ROOT),
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    manifest["completed_at"] = datetime.now(timezone.utc).isoformat()
    manifest["returncode"] = int(result.returncode)
    manifest["status"] = "complete" if result.returncode == 0 else "failed"
    _write_json(manifest_path, manifest)
    if result.returncode != 0:
        tail = log_path.read_text(encoding="utf-8", errors="replace")[-16000:]
        raise RuntimeError(f"OLMoE expert-contribution capture failed. Log tail:\n{tail}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
