#!/usr/bin/env python3
"""Preprocess the downloaded math JSONL into Megatron indexed dataset files."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
MEGATRON_ROOT = REPO_ROOT / "third_party" / "megatron-lm"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--hf-tokenizer-dir", type=Path, required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "artifacts" / "grokking_repro" / "megatron_datasets",
    )
    parser.add_argument("--name", default=None)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--json-key", default="text")
    parser.add_argument("--append-eod", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    input_jsonl = args.input_jsonl.resolve()
    hf_tokenizer_dir = args.hf_tokenizer_dir.resolve()
    output_root = args.output_root.resolve()

    if not input_jsonl.is_file():
        raise FileNotFoundError(input_jsonl)
    if not hf_tokenizer_dir.exists():
        raise FileNotFoundError(hf_tokenizer_dir)
    if args.workers <= 0:
        raise ValueError("--workers must be positive")

    name = args.name or input_jsonl.parent.name
    output_dir = output_root / name
    output_prefix = output_dir / name
    data_prefix = f"{output_prefix}_{args.json_key}_document"
    bin_path = Path(f"{data_prefix}.bin")
    idx_path = Path(f"{data_prefix}.idx")
    manifest_path = output_dir / "dmi_preprocess_manifest.json"
    log_path = output_dir / "preprocess.log"

    if output_dir.exists() and not args.overwrite and (bin_path.exists() or idx_path.exists()):
        raise FileExistsError(f"{output_dir} already has preprocessed data; pass --overwrite to replace it")
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.overwrite:
        for path in (bin_path, idx_path, manifest_path, log_path):
            if path.exists():
                path.unlink()

    cmd = [
        sys.executable,
        str(MEGATRON_ROOT / "tools" / "preprocess_data.py"),
        "--input",
        str(input_jsonl),
        "--output-prefix",
        str(output_prefix),
        "--json-keys",
        args.json_key,
        "--tokenizer-type",
        "HuggingFaceTokenizer",
        "--tokenizer-model",
        str(hf_tokenizer_dir),
        "--workers",
        str(args.workers),
    ]
    if args.append_eod:
        cmd.append("--append-eod")

    manifest = {
        "kind": "dmi_grokking_megatron_preprocessed_dataset",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input_jsonl": str(input_jsonl),
        "hf_tokenizer_dir": str(hf_tokenizer_dir),
        "output_dir": str(output_dir),
        "output_prefix": str(output_prefix),
        "data_prefix": data_prefix,
        "bin_path": str(bin_path),
        "idx_path": str(idx_path),
        "json_key": args.json_key,
        "workers": args.workers,
        "append_eod": bool(args.append_eod),
        "cmd": cmd,
    }
    _write_json(manifest_path, manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))

    if args.dry_run:
        return 0

    env = os.environ.copy()
    pythonpath_parts = [
        str(REPO_ROOT),
        str(MEGATRON_ROOT),
        env.get("PYTHONPATH", ""),
    ]
    env["PYTHONPATH"] = ":".join(part for part in pythonpath_parts if part)
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
        raise RuntimeError(f"Megatron preprocessing failed with code {result.returncode}. Log tail:\n{tail}")
    if not bin_path.is_file() or not idx_path.is_file():
        raise FileNotFoundError(f"Expected {bin_path} and {idx_path} after preprocessing")
    print(f"[DMI] Megatron data prefix: {data_prefix}")
    print(f"[DMI] Preprocess log: {log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
