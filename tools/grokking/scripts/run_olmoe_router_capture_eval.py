#!/usr/bin/env python3
"""Launch OLMoE Megatron eval with DMI router capture and dump ClickHouse rows.

This script is intentionally run-oriented:

1. create a unique model_id/table/run directory;
2. write database/table identifiers to disk before launching Megatron;
3. run Megatron with DMI router-summary capture enabled;
4. launch dump_clickhouse_training_rows.py to copy rows from ClickHouse to disk.

Raw JSONL is recorded as provenance only.  Real Megatron eval requires either
--mock-data or a Megatron indexed dataset prefix via --data-prefix.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

from olmoe_megatron_conversion import megatron_argv_from_olmoe


REPO_ROOT = Path(__file__).resolve().parents[3]
MEGATRON_ROOT = REPO_ROOT / "third_party" / "megatron-lm"
THIS_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--megatron-dir", type=Path, required=True)
    parser.add_argument("--hf-template-dir", type=Path, required=True)
    parser.add_argument(
        "--data-prefix",
        type=str,
        default=None,
        help="Megatron indexed dataset prefix. Required unless --mock-data is set.",
    )
    parser.add_argument(
        "--dataset-jsonl",
        type=Path,
        default=None,
        help="Raw JSONL provenance path. This is not passed directly to Megatron.",
    )
    parser.add_argument("--mock-data", action="store_true")
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "artifacts" / "grokking_repro" / "runs")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--database", default=os.environ.get("DMX_DB_DATABASE", "default"))
    parser.add_argument("--table", default=None)
    parser.add_argument("--db-host", default=os.environ.get("DMX_DB_HOST", "localhost"))
    parser.add_argument("--db-port", type=int, default=int(os.environ.get("DMX_DB_PORT", "9000")))
    parser.add_argument("--nproc-per-node", type=int, default=2)
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--pp-size", type=int, default=2)
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--global-batch-size", type=int, default=1)
    parser.add_argument("--seq-length", type=int, default=2048)
    parser.add_argument("--train-iters", type=int, default=1)
    parser.add_argument("--eval-iters", type=int, default=1)
    parser.add_argument("--split", default="1,49,50")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--ring-payload-mb", type=int, default=int(os.environ.get("DMI_RING_PAYLOAD_MB", "1024")))
    parser.add_argument("--ring-pinned-mb", type=int, default=int(os.environ.get("DMI_RING_PINNED_MB", "1024")))
    parser.add_argument("--ring-task-entries", type=int, default=int(os.environ.get("DMI_RING_TASK_ENTRIES", "65536")))
    parser.add_argument("--dump-chunk-size", type=int, default=1000)
    parser.add_argument("--dump-no-bytes", action="store_true")
    parser.add_argument("--overwrite-dump", action="store_true")
    parser.add_argument("--extra-megatron-arg", action="append", default=[])
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _remove_flag(argv: list[str], flag: str, *, has_value: bool) -> list[str]:
    out: list[str] = []
    i = 0
    while i < len(argv):
        if argv[i] == flag:
            i += 2 if has_value else 1
            continue
        out.append(argv[i])
        i += 1
    return out


def _replace_option(argv: list[str], flag: str, value: str) -> list[str]:
    argv = _remove_flag(argv, flag, has_value=True)
    argv.extend([flag, value])
    return argv


def _build_megatron_args(args: argparse.Namespace, *, model_id: str, table: str) -> list[str]:
    argv = megatron_argv_from_olmoe(
        hf_dir=args.hf_template_dir,
        load_dir=args.megatron_dir,
        extra_args=(),
        ckpt_format="torch_dist",
    )
    argv = argv[1:]  # drop pseudo program name

    if args.mock_data:
        if "--mock-data" not in argv:
            argv.append("--mock-data")
    else:
        if not args.data_prefix:
            raise ValueError("--data-prefix is required unless --mock-data is set")
        argv = _remove_flag(argv, "--mock-data", has_value=False)
        argv.extend(["--data-path", str(args.data_prefix), "--split", str(args.split)])

    replacements = {
        "--micro-batch-size": str(args.micro_batch_size),
        "--global-batch-size": str(args.global_batch_size),
        "--tensor-model-parallel-size": str(args.tp_size),
        "--pipeline-model-parallel-size": str(args.pp_size),
        "--seq-length": str(args.seq_length),
        "--train-iters": str(args.train_iters),
        "--eval-iters": str(args.eval_iters),
        "--seed": str(args.seed),
    }
    for flag, value in replacements.items():
        argv = _replace_option(argv, flag, value)

    argv.extend(
        [
            "--skip-train",
            "--eval-interval",
            str(max(1, args.train_iters + 1)),
            "--dmi-enable",
            "--dmi-model-id",
            model_id,
            "--dmi-db-host",
            str(args.db_host),
            "--dmi-db-port",
            str(args.db_port),
            "--dmi-db-database",
            str(args.database),
            "--dmi-clickhouse-table",
            table,
            "--dmi-ch-parallelism",
            "1",
            "--dmi-ring-payload-mb",
            str(args.ring_payload_mb),
            "--dmi-ring-pinned-mb",
            str(args.ring_pinned_mb),
            "--dmi-ring-task-entries",
            str(args.ring_task_entries),
        ]
    )
    argv.extend(args.extra_megatron_arg)
    return argv


def _write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    args.megatron_dir = args.megatron_dir.resolve()
    args.hf_template_dir = args.hf_template_dir.resolve()
    if args.dataset_jsonl is not None:
        args.dataset_jsonl = args.dataset_jsonl.resolve()
    if args.data_prefix is not None:
        args.data_prefix = str(Path(args.data_prefix).resolve())

    if not args.megatron_dir.exists():
        raise FileNotFoundError(args.megatron_dir)
    if not args.hf_template_dir.exists():
        raise FileNotFoundError(args.hf_template_dir)
    if args.dataset_jsonl is not None and not args.dataset_jsonl.is_file():
        raise FileNotFoundError(args.dataset_jsonl)

    run_suffix = uuid.uuid4().hex[:8]
    run_name = args.run_name or f"olmoe_1100k_router_eval_{run_suffix}"
    model_id = args.model_id or run_name
    table = args.table or f"dmi_olmoe_router_{run_suffix}"
    run_dir = (args.output_root / run_name).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    megatron_args = _build_megatron_args(args, model_id=model_id, table=table)
    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc_per_node={args.nproc_per_node}",
        str(MEGATRON_ROOT / "pretrain_gpt.py"),
        *megatron_args,
    ]
    dump_dir = run_dir / "clickhouse_dump"
    dump_cmd = [
        sys.executable,
        str(THIS_DIR / "dump_clickhouse_training_rows.py"),
        "--model-id",
        model_id,
        "--database",
        args.database,
        "--table",
        table,
        "--host",
        args.db_host,
        "--port",
        str(args.db_port),
        "--output-dir",
        str(dump_dir),
        "--chunk-size",
        str(args.dump_chunk_size),
    ]
    if args.dump_no_bytes:
        dump_cmd.append("--no-bytes")
    if args.overwrite_dump:
        dump_cmd.append("--overwrite")

    db_info = {
        "kind": "dmi_olmoe_router_capture_database_tables",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_id": model_id,
        "database": args.database,
        "tensor_table": table,
        "sealed_event_segments_table": f"{table}_sealed_event_segments",
        "current_event_state_table": f"{table}_current_event_state",
        "dump_dir": str(dump_dir),
    }
    _write_json(run_dir / "database_tables.json", db_info)
    (run_dir / "database_tables.txt").write_text(
        "\n".join(f"{key}={value}" for key, value in db_info.items()) + "\n",
        encoding="utf-8",
    )

    manifest = {
        "kind": "dmi_olmoe_router_capture_eval_run",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir),
        "model_id": model_id,
        "checkpoint_dir": str(args.megatron_dir),
        "hf_template_dir": str(args.hf_template_dir),
        "dataset_jsonl": str(args.dataset_jsonl) if args.dataset_jsonl is not None else None,
        "data_prefix": args.data_prefix,
        "mock_data": bool(args.mock_data),
        "database": args.database,
        "table": table,
        "db_host": args.db_host,
        "db_port": args.db_port,
        "cmd": cmd,
        "dump_cmd": dump_cmd,
    }
    _write_json(run_dir / "run_manifest.json", manifest)

    print(json.dumps(manifest, indent=2, sort_keys=True))
    if args.dry_run:
        return 0

    env = os.environ.copy()
    # The Megatron checkpoint used here is locally generated by our HF->Megatron
    # converter and may contain non-tensor metadata.  PyTorch 2.6 defaults
    # torch.load(weights_only=True), while Megatron's native checkpoint loader
    # calls torch.load without that argument.
    env.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    pythonpath_parts = [
        str(REPO_ROOT),
        str(MEGATRON_ROOT),
        str(THIS_DIR),
        env.get("PYTHONPATH", ""),
    ]
    env["PYTHONPATH"] = ":".join(part for part in pythonpath_parts if part)

    log_path = run_dir / "megatron_eval.log"
    with log_path.open("w", encoding="utf-8") as log:
        result = subprocess.run(
            cmd,
            cwd=str(MEGATRON_ROOT),
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if result.returncode != 0:
        tail = log_path.read_text(encoding="utf-8", errors="replace")[-12000:]
        raise RuntimeError(f"Megatron eval failed with code {result.returncode}. Log tail:\n{tail}")

    dump_log_path = run_dir / "clickhouse_dump.log"
    with dump_log_path.open("w", encoding="utf-8") as log:
        dump_result = subprocess.run(
            dump_cmd,
            cwd=str(REPO_ROOT),
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if dump_result.returncode != 0:
        tail = dump_log_path.read_text(encoding="utf-8", errors="replace")[-12000:]
        raise RuntimeError(f"ClickHouse dump failed with code {dump_result.returncode}. Log tail:\n{tail}")

    print(f"[DMI] Megatron log: {log_path}")
    print(f"[DMI] ClickHouse dump log: {dump_log_path}")
    print(f"[DMI] ClickHouse dump dir: {dump_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
