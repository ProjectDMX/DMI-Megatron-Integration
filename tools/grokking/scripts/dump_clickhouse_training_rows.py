#!/usr/bin/env python3
"""Dump DMI training ClickHouse rows to local JSONL files.

The dump is intentionally database-level rather than reader-level: it exports
the tensor table plus the two phase event tables so a run can be inspected even
if the shared ClickHouse database is later dropped or overwritten.
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import torch


TENSOR_COLUMNS = (
    "model_id",
    "act_name",
    "direction",
    "phase",
    "global_batch_id",
    "dp_rank",
    "microbatch_id",
    "sample_index",
    "layer_no",
    "shard_rank",
    "token_start",
    "token_end",
    "dtype",
    "shape",
    "bytes",
)

TENSOR_ORDER_COLUMNS = (
    "model_id",
    "act_name",
    "direction",
    "phase",
    "global_batch_id",
    "dp_rank",
    "microbatch_id",
    "sample_index",
    "layer_no",
    "shard_rank",
    "token_start",
    "token_end",
)

SEALED_COLUMNS = (
    "model_id",
    "phase",
    "training_iteration_id_start",
    "training_iteration_id_end",
    "global_batch_id_start",
    "global_batch_id_end",
    "execution_order_id_start",
    "execution_order_id_end",
)

CURRENT_COLUMNS = (
    "model_id",
    "execution_id_start",
    "training_iteration_id_start",
    "global_batch_id_start",
    "phase",
)

SCALAR_COLUMNS = (
    "model_id",
    "act_name",
    "direction",
    "phase",
    "global_batch_id",
    "dp_rank",
    "microbatch_id",
    "sample_index",
    "layer_no",
    "shard_rank",
    "token_start",
    "token_end",
    "value",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--database", default="default")
    parser.add_argument("--table", required=True)
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--username", default="default")
    parser.add_argument("--password", default="")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--chunk-size", type=int, default=1000)
    parser.add_argument(
        "--no-bytes",
        action="store_true",
        help="Do not dump tensor payload files. Metadata and byte lengths are still dumped.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing output directory.")
    return parser.parse_args()


def _ident(name: str) -> str:
    if not name or not name.replace("_", "").isalnum() or not name[0].isalpha() and name[0] != "_":
        raise ValueError(f"Invalid ClickHouse identifier: {name!r}")
    return f"`{name}`"


def _decode_string(value: Any) -> str:
    if isinstance(value, memoryview):
        value = value.tobytes()
    if isinstance(value, bytearray):
        value = bytes(value)
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="surrogateescape")
    return str(value)


def _bytes_value(value: Any) -> bytes:
    if isinstance(value, memoryview):
        return value.tobytes()
    if isinstance(value, bytearray):
        return bytes(value)
    if isinstance(value, bytes):
        return value
    raise TypeError(f"Expected bytes-like payload, got {type(value)!r}")


def _torch_dtype(dtype: str) -> torch.dtype:
    mapping = {
        "bool": torch.bool,
        "torch.bool": torch.bool,
        "uint8": torch.uint8,
        "torch.uint8": torch.uint8,
        "int8": torch.int8,
        "torch.int8": torch.int8,
        "int16": torch.int16,
        "torch.int16": torch.int16,
        "int32": torch.int32,
        "torch.int32": torch.int32,
        "int64": torch.int64,
        "torch.int64": torch.int64,
        "float16": torch.float16,
        "half": torch.float16,
        "torch.float16": torch.float16,
        "torch.half": torch.float16,
        "bfloat16": torch.bfloat16,
        "torch.bfloat16": torch.bfloat16,
        "float32": torch.float32,
        "float": torch.float32,
        "torch.float32": torch.float32,
        "torch.float": torch.float32,
        "float64": torch.float64,
        "double": torch.float64,
        "torch.float64": torch.float64,
        "torch.double": torch.float64,
    }
    try:
        return mapping[dtype]
    except KeyError as exc:
        raise ValueError(f"Unsupported tensor dtype in ClickHouse row: {dtype!r}") from exc


def _decode_tensor(payload: bytes, *, dtype: str, shape: list[int]) -> torch.Tensor:
    tensor_1d = torch.frombuffer(payload, dtype=_torch_dtype(dtype)).clone()
    expected = 1
    for dim in shape:
        expected *= int(dim)
    if tensor_1d.numel() != expected:
        raise ValueError(
            f"Tensor payload size mismatch: dtype={dtype}, shape={shape}, "
            f"expected {expected} elements, got {tensor_1d.numel()}"
        )
    return tensor_1d.reshape(shape)


def _row_to_record(columns: Iterable[str], row: tuple[Any, ...]) -> tuple[dict[str, Any], bytes | None]:
    record: dict[str, Any] = {}
    payload_bytes: bytes | None = None
    for name, value in zip(columns, row):
        if name == "bytes":
            payload = _bytes_value(value)
            record["bytes_len"] = len(payload)
            payload_bytes = payload
        elif name in {"model_id", "act_name", "direction", "phase", "dtype"}:
            record[name] = _decode_string(value)
        elif name == "value":
            record[name] = float(value) if isinstance(value, float) else int(value)
        elif name == "shape":
            record[name] = [int(x) for x in value]
        else:
            record[name] = int(value)
    return record, payload_bytes


def _count_rows(client: Any, *, database: str, table: str, model_id: str) -> int:
    rows = client.execute(
        f"SELECT count() FROM {_ident(database)}.{_ident(table)} WHERE `model_id` = %(model_id)s",
        {"model_id": model_id},
    )
    return int(rows[0][0])


def _dump_table(
    client: Any,
    *,
    database: str,
    table: str,
    model_id: str,
    columns: tuple[str, ...],
    order_columns: tuple[str, ...],
    output_dir: Path,
    prefix: str,
    chunk_size: int,
    payload_dir: Path | None = None,
) -> dict[str, Any]:
    total = _count_rows(client, database=database, table=table, model_id=model_id)
    files: list[dict[str, Any]] = []
    if total == 0:
        return {"table": table, "row_count": 0, "files": files}

    select_cols = ", ".join(_ident(c) for c in columns)
    order_by = ", ".join(_ident(c) for c in order_columns)
    for offset in range(0, total, chunk_size):
        limit = min(chunk_size, total - offset)
        sql = (
            f"SELECT {select_cols} FROM {_ident(database)}.{_ident(table)} "
            f"WHERE `model_id` = %(model_id)s "
            f"ORDER BY {order_by} "
            f"LIMIT %(limit)s OFFSET %(offset)s"
        )
        rows = client.execute(
            sql,
            {"model_id": model_id, "limit": int(limit), "offset": int(offset)},
        )
        path = output_dir / f"rows_{offset:012d}_{offset + len(rows):012d}.jsonl"
        with path.open("w", encoding="utf-8") as out:
            for row_index, row in enumerate(rows, start=offset):
                record, payload = _row_to_record(columns, row)
                if payload is not None and payload_dir is not None:
                    payload_rel = Path("tensor_payloads") / f"tensor_{row_index:012d}.pt"
                    payload_path = payload_dir / payload_rel.name
                    tensor = _decode_tensor(payload, dtype=record["dtype"], shape=record["shape"])
                    torch.save(tensor, payload_path)
                    record["payload_path"] = str(payload_rel)
                out.write(json.dumps(record, sort_keys=True) + "\n")
        files.append({"path": str(path), "row_count": len(rows), "offset": offset})
    return {"table": table, "row_count": total, "files": files}


def main() -> int:
    args = parse_args()
    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be positive")
    if args.output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"{args.output_dir} already exists; pass --overwrite to replace it")
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    tensor_rows_dir = args.output_dir / "tensor_rows"
    tensor_payloads_dir = args.output_dir / "tensor_payloads"
    scalar_float_rows_dir = args.output_dir / "scalar_float_rows"
    scalar_int_rows_dir = args.output_dir / "scalar_int_rows"
    sealed_dir = args.output_dir / "sealed_event_segments"
    current_dir = args.output_dir / "current_event_state"
    for path in (tensor_rows_dir, scalar_float_rows_dir, scalar_int_rows_dir, sealed_dir, current_dir):
        path.mkdir(parents=True, exist_ok=False)
    if not args.no_bytes:
        tensor_payloads_dir.mkdir(parents=True, exist_ok=False)

    from clickhouse_driver import Client

    client = Client(
        host=args.host,
        port=args.port,
        user=args.username,
        password=args.password,
        database=args.database,
        settings={"strings_as_bytes": 1},
    )

    tensor_table = args.table
    scalar_float_table = f"{args.table}_scalar_float"
    scalar_int_table = f"{args.table}_scalar_int"
    sealed_table = f"{args.table}_sealed_event_segments"
    current_table = f"{args.table}_current_event_state"

    manifest = {
        "kind": "dmi_training_clickhouse_dump",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_id": args.model_id,
        "database": args.database,
        "host": args.host,
        "port": args.port,
        "tensor_table": tensor_table,
        "scalar_float_table": scalar_float_table,
        "scalar_int_table": scalar_int_table,
        "sealed_event_segments_table": sealed_table,
        "current_event_state_table": current_table,
        "chunk_size": args.chunk_size,
        "payload_format": "torch_pt_file_v1" if not args.no_bytes else "metadata_only_v1",
        "include_tensor_payloads": not args.no_bytes,
        "layout": {
            "tensor_rows_dir": "tensor_rows",
            "tensor_payloads_dir": "tensor_payloads" if not args.no_bytes else None,
            "scalar_float_rows_dir": "scalar_float_rows",
            "scalar_int_rows_dir": "scalar_int_rows",
            "sealed_event_segments_dir": "sealed_event_segments",
            "current_event_state_dir": "current_event_state",
        },
        "tables": {},
    }

    manifest["tables"]["tensor"] = _dump_table(
        client,
        database=args.database,
        table=tensor_table,
        model_id=args.model_id,
        columns=TENSOR_COLUMNS,
        order_columns=TENSOR_ORDER_COLUMNS,
        output_dir=tensor_rows_dir,
        prefix="tensor_rows",
        chunk_size=args.chunk_size,
        payload_dir=None if args.no_bytes else tensor_payloads_dir,
    )
    manifest["tables"]["scalar_float"] = _dump_table(
        client,
        database=args.database,
        table=scalar_float_table,
        model_id=args.model_id,
        columns=SCALAR_COLUMNS,
        order_columns=TENSOR_ORDER_COLUMNS,
        output_dir=scalar_float_rows_dir,
        prefix="scalar_float_rows",
        chunk_size=args.chunk_size,
    )
    manifest["tables"]["scalar_int"] = _dump_table(
        client,
        database=args.database,
        table=scalar_int_table,
        model_id=args.model_id,
        columns=SCALAR_COLUMNS,
        order_columns=TENSOR_ORDER_COLUMNS,
        output_dir=scalar_int_rows_dir,
        prefix="scalar_int_rows",
        chunk_size=args.chunk_size,
    )
    manifest["tables"]["sealed_event_segments"] = _dump_table(
        client,
        database=args.database,
        table=sealed_table,
        model_id=args.model_id,
        columns=SEALED_COLUMNS,
        order_columns=("model_id", "phase", "training_iteration_id_start", "global_batch_id_start", "execution_order_id_start"),
        output_dir=sealed_dir,
        prefix="sealed_event_segments",
        chunk_size=args.chunk_size,
    )
    manifest["tables"]["current_event_state"] = _dump_table(
        client,
        database=args.database,
        table=current_table,
        model_id=args.model_id,
        columns=CURRENT_COLUMNS,
        order_columns=("model_id", "execution_id_start"),
        output_dir=current_dir,
        prefix="current_event_state",
        chunk_size=args.chunk_size,
    )

    manifest_path = args.output_dir / "dump_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
