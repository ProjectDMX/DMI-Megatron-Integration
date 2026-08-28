from __future__ import annotations

import json
import math
import os
import re
import subprocess
import sys
import time
import uuid
from pathlib import Path

import numpy
import pytest
import torch

from dmi_megatron_integration.records.reader import MegatronTrainingReader
from tests.test_megatron_e2e_clickhouse import (
    _clickhouse_client_or_skip,
    _create_training_table,
    _read_training_rows,
)


ROOT = Path(__file__).resolve().parents[1]
MEGATRON_ROOT = ROOT / "third_party" / "megatron-lm"


def _available_cuda_devices() -> int:
    visible = os.environ.get("DMI_REAL_E2E_CUDA_VISIBLE_DEVICES")
    if visible:
        return len([item for item in visible.split(",") if item.strip()])
    return torch.cuda.device_count()


def _query_count(
    client,
    *,
    database: str,
    table: str,
    model_id: str,
    act_name: str | None = "router_probs_mean",
) -> int:
    where_act = "" if act_name is None else "AND act_name = %(act_name)s"
    params = {"model_id": model_id}
    if act_name is not None:
        params["act_name"] = act_name
    rows = client.execute(
        f"""
        SELECT count()
        FROM `{database}`.`{table}`
        WHERE model_id = %(model_id)s
          {where_act}
          AND direction = 'fwd'
        """,
        params,
    )
    return int(rows[0][0])


def _query_scalar_count(
    client,
    *,
    database: str,
    table: str,
    model_id: str,
    act_name: str,
    direction: str = "fwd",
) -> int:
    rows = client.execute(
        f"""
        SELECT count()
        FROM `{database}`.`{table}`
        WHERE model_id = %(model_id)s
          AND act_name = %(act_name)s
          AND direction = %(direction)s
        """,
        {"model_id": model_id, "act_name": act_name, "direction": direction},
    )
    return int(rows[0][0])


def _query_act_count_all_directions(
    client,
    *,
    database: str,
    table: str,
    model_id: str,
    act_name: str,
) -> int:
    rows = client.execute(
        f"""
        SELECT count()
        FROM `{database}`.`{table}`
        WHERE model_id = %(model_id)s
          AND act_name = %(act_name)s
        """,
        {"model_id": model_id, "act_name": act_name},
    )
    return int(rows[0][0])


def _query_model_count(client, *, database: str, table: str, model_id: str) -> int:
    rows = client.execute(
        f"""
        SELECT count()
        FROM `{database}`.`{table}`
        WHERE model_id = %(model_id)s
        """,
        {"model_id": model_id},
    )
    return int(rows[0][0])


def _query_scalar_values(
    client,
    *,
    database: str,
    table: str,
    model_id: str,
    act_name: str,
    direction: str = "fwd",
):
    return client.execute(
        f"""
        SELECT phase, global_batch_id, microbatch_id, sample_index, layer_no, value
        FROM `{database}`.`{table}`
        WHERE model_id = %(model_id)s
          AND act_name = %(act_name)s
          AND direction = %(direction)s
        ORDER BY phase, global_batch_id, microbatch_id, sample_index
        """,
        {"model_id": model_id, "act_name": act_name, "direction": direction},
    )


def _query_rows(client, *, database: str, table: str, model_id: str):
    return client.execute(
        f"""
        SELECT
            model_id,
            act_name,
            direction,
            phase,
            global_batch_id,
            dp_rank,
            microbatch_id,
            sample_index,
            layer_no,
            shard_rank,
            token_start,
            token_end,
            dtype,
            shape,
            length(bytes)
        FROM `{database}`.`{table}`
        WHERE model_id = %(model_id)s
          AND act_name = 'router_probs_mean'
          AND direction = 'fwd'
        ORDER BY phase, global_batch_id, microbatch_id, layer_no, sample_index
        """,
        {"model_id": model_id},
    )


def _query_tensor_row_shapes(
    client,
    *,
    database: str,
    table: str,
    model_id: str,
    act_name: str,
):
    return client.execute(
        f"""
        SELECT dtype, shape, length(bytes)
        FROM `{database}`.`{table}`
        WHERE model_id = %(model_id)s
          AND act_name = %(act_name)s
          AND direction = 'fwd'
        ORDER BY phase, global_batch_id, microbatch_id, layer_no, sample_index
        """,
        {"model_id": model_id, "act_name": act_name},
    )


def _query_moe_hook_rows(
    client,
    *,
    database: str,
    table: str,
    model_id: str,
    act_names: tuple[str, ...],
):
    return client.execute(
        f"""
        SELECT
            act_name,
            direction,
            phase,
            global_batch_id,
            dp_rank,
            microbatch_id,
            sample_index,
            layer_no,
            shard_rank,
            token_start,
            token_end,
            attempt_id,
            invocation_id,
            dataset_id,
            dtype,
            shape,
            bytes
        FROM `{database}`.`{table}`
        WHERE model_id = %(model_id)s
          AND act_name IN %(act_names)s
          AND direction = 'fwd'
        ORDER BY act_name, layer_no, shard_rank, dp_rank, microbatch_id
        """,
        {"model_id": model_id, "act_names": act_names},
        settings={"strings_as_bytes": True},
    )


def _wait_for_exact_rows(client, *, database: str, table: str, model_id: str, expected: int):
    deadline = time.time() + float(os.environ.get("DMI_REAL_E2E_CLICKHOUSE_TIMEOUT_S", "20"))
    last_count = 0
    while time.time() < deadline:
        last_count = _query_count(client, database=database, table=table, model_id=model_id)
        if last_count == expected:
            return
        time.sleep(0.2)
    raise AssertionError(f"Expected {expected} rows, saw {last_count}")


def _wait_for_exact_act_rows(
    client,
    *,
    database: str,
    table: str,
    model_id: str,
    act_name: str,
    expected: int,
):
    deadline = time.time() + float(os.environ.get("DMI_REAL_E2E_CLICKHOUSE_TIMEOUT_S", "20"))
    last_count = 0
    while time.time() < deadline:
        last_count = _query_count(
            client,
            database=database,
            table=table,
            model_id=model_id,
            act_name=act_name,
        )
        if last_count == expected:
            return
        time.sleep(0.2)
    raise AssertionError(f"Expected {expected} {act_name} rows, saw {last_count}")


def _wait_for_exact_model_rows(
    client,
    *,
    database: str,
    table: str,
    model_id: str,
    expected: int,
):
    deadline = time.time() + float(os.environ.get("DMI_REAL_E2E_CLICKHOUSE_TIMEOUT_S", "20"))
    last_count = 0
    while time.time() < deadline:
        last_count = _query_model_count(
            client,
            database=database,
            table=table,
            model_id=model_id,
        )
        if last_count == expected:
            return
        time.sleep(0.2)
    raise AssertionError(f"Expected {expected} rows in {table}, saw {last_count}")


def _wait_for_exact_scalar_rows(
    client,
    *,
    database: str,
    table: str,
    model_id: str,
    act_name: str,
    expected: int,
    direction: str = "fwd",
):
    deadline = time.time() + float(os.environ.get("DMI_REAL_E2E_CLICKHOUSE_TIMEOUT_S", "20"))
    last_count = 0
    while time.time() < deadline:
        last_count = _query_scalar_count(
            client,
            database=database,
            table=table,
            model_id=model_id,
            act_name=act_name,
            direction=direction,
        )
        if last_count == expected:
            return
        time.sleep(0.2)
    raise AssertionError(f"Expected {expected} scalar rows, saw {last_count}")


def _tiny_megatron_router_summary_cmd(
    *,
    model_id: str,
    train_iters: int,
    eval_iters: int = 0,
    eval_interval: int | None = None,
    micro_batch_size: int,
    global_batch_size: int,
    nproc_per_node: int = 1,
    tp_size: int = 1,
    pp_size: int = 1,
    ep_size: int = 1,
    num_experts: int = 2,
    moe_router_topk: int = 1,
    moe_token_dispatcher_type: str = "allgather",
    transformer_impl: str = "local",
    clip_grad: float = 1.0,
    database: str | None = None,
    table: str | None = None,
    extra_args: list[str] | None = None,
) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc_per_node={nproc_per_node}",
        "pretrain_gpt.py",
        "--mock-data",
        "--tokenizer-type",
        "NullTokenizer",
        "--vocab-size",
        "128",
        "--num-layers",
        "2",
        "--hidden-size",
        "64",
        "--ffn-hidden-size",
        "128",
        "--num-attention-heads",
        "4",
        "--seq-length",
        "16",
        "--max-position-embeddings",
        "16",
        "--micro-batch-size",
        str(micro_batch_size),
        "--global-batch-size",
        str(global_batch_size),
        "--tensor-model-parallel-size",
        str(tp_size),
        "--pipeline-model-parallel-size",
        str(pp_size),
        "--expert-model-parallel-size",
        str(ep_size),
        "--train-iters",
        str(train_iters),
        "--eval-interval",
        str(eval_interval if eval_interval is not None else train_iters + 1),
        "--eval-iters",
        str(eval_iters),
        "--seed",
        "1234",
        "--lr",
        "1.0e-4",
        "--min-lr",
        "1.0e-5",
        "--lr-decay-iters",
        str(train_iters),
        "--lr-warmup-iters",
        "0",
        "--weight-decay",
        "0.0",
        "--adam-beta1",
        "0.9",
        "--adam-beta2",
        "0.95",
        "--init-method-std",
        "0.02",
        "--clip-grad",
        str(clip_grad),
        "--bf16",
        "--transformer-impl",
        transformer_impl,
        "--no-persist-layer-norm",
        "--no-gradient-accumulation-fusion",
        "--swiglu",
        "--disable-bias-linear",
        "--num-experts",
        str(num_experts),
        "--moe-router-topk",
        str(moe_router_topk),
        "--moe-token-dispatcher-type",
        str(moe_token_dispatcher_type),
        "--moe-router-pre-softmax",
        "--moe-router-load-balancing-type",
        "aux_loss",
        "--moe-aux-loss-coeff",
        "0.01",
        "--no-save-optim",
        "--no-save-rng",
        "--no-load-optim",
        "--no-load-rng",
        "--no-create-attention-mask-in-dataloader",
        "--dmi-enable",
        "--dmi-model-id",
        model_id,
        "--dmi-ring-payload-mb",
        os.environ.get("DMI_REAL_E2E_RING_PAYLOAD_MB", "64"),
        "--dmi-ring-pinned-mb",
        os.environ.get("DMI_REAL_E2E_RING_PINNED_MB", "64"),
        "--dmi-ring-task-entries",
        os.environ.get("DMI_REAL_E2E_RING_TASK_ENTRIES", "1024"),
    ]
    if database is not None and table is not None:
        cmd += [
            "--dmi-db-host",
            os.environ.get("DMX_DB_HOST", "localhost"),
            "--dmi-db-port",
            os.environ.get("DMX_DB_PORT", "9000"),
            "--dmi-db-database",
            database,
            "--dmi-clickhouse-table",
            table,
            "--dmi-ch-parallelism",
            os.environ.get("DMI_REAL_E2E_CH_PARALLELISM", "10"),
        ]
    if extra_args:
        cmd += extra_args
    return cmd


def _build_tiny_indexed_gpt_dataset(prefix: Path, *, token_offset: int) -> None:
    sys.path.insert(0, str(MEGATRON_ROOT))
    try:
        from megatron.core.datasets.indexed_dataset import IndexedDatasetBuilder
    finally:
        sys.path.pop(0)

    builder = IndexedDatasetBuilder(str(prefix) + ".bin", dtype=numpy.int32)
    for document_id in range(64):
        tokens = ((numpy.arange(80) + token_offset + document_id) % 120) + 1
        builder.add_document(tokens, [tokens.size])
    builder.finalize(str(prefix) + ".idx")


def _expected_phase_counts(
    *,
    train_iters: int,
    eval_iters: int,
    global_batch_size: int,
    num_moe_layers: int,
    eval_interval: int | None = None,
    skip_train: bool = False,
) -> dict[str, int]:
    if eval_iters == 0:
        valid_runs = 0
        test_runs = 0
    elif skip_train:
        valid_runs = 1
        test_runs = 1
    else:
        interval = eval_interval if eval_interval is not None else train_iters + 1
        valid_runs = train_iters // interval + 1
        test_runs = 1

    train_rows = 0 if skip_train else train_iters * global_batch_size * num_moe_layers
    valid_rows = valid_runs * eval_iters * global_batch_size * num_moe_layers
    test_rows = test_runs * eval_iters * global_batch_size * num_moe_layers
    return {"train": train_rows, "valid": valid_rows, "test": test_rows}


def _expected_sample_phase_counts(
    *,
    train_iters: int,
    eval_iters: int,
    global_batch_size: int,
    eval_interval: int | None = None,
    skip_train: bool = False,
) -> dict[str, int]:
    if eval_iters == 0:
        valid_runs = 0
        test_runs = 0
    elif skip_train:
        valid_runs = 1
        test_runs = 1
    else:
        interval = eval_interval if eval_interval is not None else train_iters + 1
        valid_runs = train_iters // interval + 1
        test_runs = 1

    train_rows = 0 if skip_train else train_iters * global_batch_size
    valid_rows = valid_runs * eval_iters * global_batch_size
    test_rows = test_runs * eval_iters * global_batch_size
    return {"train": train_rows, "valid": valid_rows, "test": test_rows}


def _expected_total_rows(**kwargs) -> int:
    return sum(_expected_phase_counts(**kwargs).values())


def _expected_total_sample_rows(**kwargs) -> int:
    return sum(_expected_sample_phase_counts(**kwargs).values())


def _run_megatron_cmd(cmd: list[str], *, env: dict[str, str], log_path: Path) -> None:
    child_env = dict(env)
    python_bin = str(Path(cmd[0]).resolve().parent)
    existing_path = child_env.get("PATH", "")
    child_env["PATH"] = (
        python_bin if not existing_path else os.pathsep.join((python_bin, existing_path))
    )
    with log_path.open("w", encoding="utf-8") as log:
        result = subprocess.run(
            cmd,
            cwd=MEGATRON_ROOT,
            env=child_env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=float(os.environ.get("DMI_REAL_E2E_TIMEOUT_S", "240")),
            check=False,
        )
    if result.returncode != 0:
        tail = log_path.read_text(encoding="utf-8", errors="replace")[-12000:]
        raise AssertionError(
            f"Real Megatron training command failed with code {result.returncode}. "
            f"Log tail:\n{tail}"
        )


def _read_file_sink_rows(root: Path):
    rows = []
    for rows_path in sorted(root.glob("rank*/rows.jsonl")):
        rank_dir = rows_path.parent
        for line in rows_path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            payload = torch.load(rank_dir / row["payload_file"], map_location="cpu", weights_only=True)
            rows.append((row, payload))
    return rows


def _read_file_sink_scalar_rows(root: Path, filename: str):
    rows = []
    for rows_path in sorted(root.glob(f"rank*/{filename}")):
        for line in rows_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _read_training_rows_all_phases(*, model_id: str, table: str, database: str):
    reader = MegatronTrainingReader(
        host=os.environ.get("DMX_DB_HOST", "localhost"),
        port=int(os.environ.get("DMX_DB_PORT", "9000")),
        username=os.environ.get("DMX_DB_USER", "default"),
        password=os.environ.get("DMX_DB_PASSWORD", ""),
        database=database,
        table=table,
    )
    rows = []
    try:
        for phase in ("train", "valid", "test"):
            rows.extend(
                reader.training_prefix_get(
                    (model_id, "router_probs_mean", "fwd", phase),
                    return_full_key_tuple=True,
                )
            )
        return rows
    finally:
        reader.close()


def _read_training_act_rows(
    *,
    model_id: str,
    table: str,
    database: str,
    act_name: str,
    direction: str = "fwd",
    phase: str = "train",
    raw: bool = False,
):
    reader = MegatronTrainingReader(
        host=os.environ.get("DMX_DB_HOST", "localhost"),
        port=int(os.environ.get("DMX_DB_PORT", "9000")),
        username=os.environ.get("DMX_DB_USER", "default"),
        password=os.environ.get("DMX_DB_PASSWORD", ""),
        database=database,
        table=table,
    )
    try:
        read = reader.training_raw_prefix_get if raw else reader.training_prefix_get
        return read(
            (model_id, act_name, direction, phase),
            return_full_key_tuple=True,
        )
    finally:
        reader.close()


def _read_training_scalar_act_rows(
    *,
    model_id: str,
    table: str,
    database: str,
    act_name: str,
    direction: str,
    phase: str = "train",
):
    reader = MegatronTrainingReader(
        host=os.environ.get("DMX_DB_HOST", "localhost"),
        port=int(os.environ.get("DMX_DB_PORT", "9000")),
        username=os.environ.get("DMX_DB_USER", "default"),
        password=os.environ.get("DMX_DB_PASSWORD", ""),
        database=database,
        table=table,
    )
    try:
        return reader.training_scalar_prefix_get(
            (model_id, act_name, direction, phase),
            scalar_kind="float",
            return_full_key_tuple=True,
        )
    finally:
        reader.close()


def _read_training_scalar_rows_all_phases(
    *,
    model_id: str,
    table: str,
    database: str,
    scalar_kind: str,
    act_name: str,
    direction: str = "fwd",
    raw: bool = False,
):
    reader = MegatronTrainingReader(
        host=os.environ.get("DMX_DB_HOST", "localhost"),
        port=int(os.environ.get("DMX_DB_PORT", "9000")),
        username=os.environ.get("DMX_DB_USER", "default"),
        password=os.environ.get("DMX_DB_PASSWORD", ""),
        database=database,
        table=table,
    )
    rows = []
    try:
        for phase in ("train", "valid", "test"):
            read = (
                reader.training_scalar_raw_prefix_get
                if raw
                else reader.training_scalar_prefix_get
            )
            rows.extend(
                read(
                    (model_id, act_name, direction, phase),
                    scalar_kind=scalar_kind,
                    return_full_key_tuple=True,
                )
            )
        return rows
    finally:
        reader.close()


def _read_file_sink_jsonl(root: Path, filename: str):
    rows = []
    for rows_path in sorted(root.glob(f"rank*/{filename}")):
        rows.extend(
            json.loads(line)
            for line in rows_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    return rows


def _query_clickhouse_json_rows(client, sql: str, params: dict[str, str], columns: tuple[str, ...]):
    return [dict(zip(columns, row)) for row in client.execute(sql, params)]


def _assert_file_sink_boundaries_match_clickhouse(
    *,
    client,
    database: str,
    table: str,
    model_id: str,
    file_sink_dir: Path,
) -> None:
    boundary_columns = (
        "model_id",
        "training_iteration_id",
        "phase",
        "eval_index",
        "boundary_type",
        "next_global_batch_id",
    )
    boundary_rows = _query_clickhouse_json_rows(
        client,
        f"""
        SELECT {", ".join(f"`{name}`" for name in boundary_columns)}
        FROM `{database}`.`{table}_eval_phase_boundary`
        WHERE model_id = %(model_id)s
        ORDER BY training_iteration_id, phase, eval_index, next_global_batch_id, boundary_type
        """,
        {"model_id": model_id},
        boundary_columns,
    )
    file_boundary_rows = sorted(
        _read_file_sink_jsonl(file_sink_dir, "eval_phase_boundary.jsonl"),
        key=lambda row: (
            row["training_iteration_id"],
            row["phase"],
            row["eval_index"],
            row["next_global_batch_id"],
            row["boundary_type"],
        ),
    )
    assert file_boundary_rows == boundary_rows


def _assert_training_table_counts(
    *,
    client,
    database: str,
    table: str,
    model_id: str,
    expected_tensor_rows: int,
    expected_scalar_float_rows: int,
    expected_scalar_int_rows: int | None = None,
    expected_attempt_status_rows: int = 0,
    expected_boundary_min: int = 0,
) -> None:
    if expected_scalar_int_rows is None:
        expected_scalar_int_rows = expected_scalar_float_rows
    tensor_rows = _query_count(
        client,
        database=database,
        table=table,
        model_id=model_id,
        act_name="router_probs_mean",
    )
    scalar_float_rows = _query_scalar_count(
        client,
        database=database,
        table=f"{table}_scalar_float",
        model_id=model_id,
        act_name="lm_per_sample_loss",
    )
    scalar_int_rows = _query_model_count(
        client,
        database=database,
        table=f"{table}_scalar_int",
        model_id=model_id,
    )
    attempt_status_rows = _query_scalar_count(
        client,
        database=database,
        table=f"{table}_scalar_int",
        model_id=model_id,
        act_name="iteration_attempt_status",
        direction="iter",
    )
    boundary_rows = _query_model_count(
        client,
        database=database,
        table=f"{table}_eval_phase_boundary",
        model_id=model_id,
    )
    all_tensor_rows = _query_model_count(
        client,
        database=database,
        table=table,
        model_id=model_id,
    )
    all_loss_rows = _query_act_count_all_directions(
        client,
        database=database,
        table=f"{table}_scalar_float",
        model_id=model_id,
        act_name="lm_per_sample_loss",
    )
    assert tensor_rows == expected_tensor_rows
    assert scalar_float_rows == expected_scalar_float_rows
    assert all_tensor_rows == expected_tensor_rows
    assert all_loss_rows == expected_scalar_float_rows
    assert scalar_int_rows == expected_scalar_int_rows + expected_attempt_status_rows
    assert attempt_status_rows == expected_attempt_status_rows
    assert boundary_rows >= expected_boundary_min


def _assert_tensor_maps_close_with_report(
    file_by_key,
    db_by_key,
    *,
    label: str = "file sink and ClickHouse",
    atol: float | None = None,
    rtol: float | None = None,
) -> None:
    if atol is None:
        atol = float(os.environ.get("DMI_NUMERIC_E2E_ATOL", "0"))
    if rtol is None:
        rtol = float(os.environ.get("DMI_NUMERIC_E2E_RTOL", "0"))
    max_abs_error = 0.0
    max_rel_error = 0.0
    max_error_key = None

    for key, db_tensor in db_by_key.items():
        file_tensor = file_by_key[key]
        abs_error = (file_tensor - db_tensor).abs()
        item_abs = float(abs_error.max().item()) if abs_error.numel() else 0.0
        denom = db_tensor.abs().clamp_min(1.0e-30)
        rel_error = abs_error / denom
        item_rel = float(rel_error.max().item()) if rel_error.numel() else 0.0
        if item_abs > max_abs_error:
            max_abs_error = item_abs
            max_error_key = key
        max_rel_error = max(max_rel_error, item_rel)

    if max_abs_error == 0.0:
        print(f"[DMI numeric E2E] {label} tensors are identical; max error = 0")
    else:
        print(
            f"[DMI numeric E2E] WARNING: {label} tensors differ within configured "
            f"tolerance: max_abs_error={max_abs_error:.8e}, "
            f"max_rel_error={max_rel_error:.8e}, key={max_error_key}, "
            f"atol={atol}, rtol={rtol}",
            flush=True,
        )

    for key, db_tensor in db_by_key.items():
        torch.testing.assert_close(file_by_key[key], db_tensor, rtol=rtol, atol=atol)


def _assert_scalar_maps_close_with_report(
    file_by_key,
    db_by_key,
    *,
    label: str = "file sink and ClickHouse",
    atol: float | None = None,
    rtol: float | None = None,
) -> None:
    if atol is None:
        atol = float(os.environ.get("DMI_NUMERIC_E2E_ATOL", "0"))
    if rtol is None:
        rtol = float(os.environ.get("DMI_NUMERIC_E2E_RTOL", "0"))
    max_abs_error = 0.0
    max_rel_error = 0.0
    max_error_key = None

    for key, db_value in db_by_key.items():
        file_value = float(file_by_key[key])
        db_value = float(db_value)
        abs_error = abs(file_value - db_value)
        rel_error = abs_error / max(abs(db_value), 1.0e-30)
        if abs_error > max_abs_error:
            max_abs_error = abs_error
            max_error_key = key
        max_rel_error = max(max_rel_error, rel_error)

    if max_abs_error == 0.0:
        print(f"[DMI numeric E2E] {label} scalars are identical; max error = 0")
    else:
        print(
            f"[DMI numeric E2E] WARNING: {label} scalars differ within configured "
            f"tolerance: max_abs_error={max_abs_error:.8e}, "
            f"max_rel_error={max_rel_error:.8e}, key={max_error_key}, "
            f"atol={atol}, rtol={rtol}",
            flush=True,
        )

    for key, db_value in db_by_key.items():
        torch.testing.assert_close(
            torch.tensor(float(file_by_key[key])),
            torch.tensor(float(db_value)),
            rtol=rtol,
            atol=atol,
        )


def _file_sink_row_key(row: dict) -> tuple:
    assert row["schema_version"] == 2
    return (
        row["model_id"],
        row["act_name"],
        row["direction"],
        row["phase"],
        row["global_batch_id"],
        row["dp_rank"],
        row["microbatch_id"],
        row["sample_index"],
        row["layer_no"],
        row["shard_rank"],
        row["token_start"],
        row["token_end"],
        row["attempt_id"],
        row["invocation_id"],
        row["dataset_id"],
    )


def _run_two_path_numeric_check(
    *,
    tmp_path: Path,
    database: str,
    table: str,
    model_id: str,
    train_iters: int,
    eval_iters: int,
    eval_interval: int | None = None,
    micro_batch_size: int,
    global_batch_size: int,
    expected_tensor_rows: int,
    expected_scalar_float_rows: int,
    extra_args: list[str] | None = None,
) -> None:
    client = _clickhouse_client_or_skip()
    file_sink_dir = tmp_path / "file_sink"

    base_env = os.environ.copy()
    base_env["PYTHONPATH"] = f"{ROOT}:{MEGATRON_ROOT}:{base_env.get('PYTHONPATH', '')}"
    base_env["DMI_ENABLE"] = "1"
    base_env.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", "1")
    if "DMI_REAL_E2E_CUDA_VISIBLE_DEVICES" in base_env:
        base_env["CUDA_VISIBLE_DEVICES"] = base_env["DMI_REAL_E2E_CUDA_VISIBLE_DEVICES"]

    client.execute(f"CREATE DATABASE IF NOT EXISTS `{database}`")
    client.execute(f"DROP TABLE IF EXISTS `{database}`.`{table}`")
    _create_training_table(client, database=database, table=table)

    try:
        combined_extra_args = ["--dmi-hook-selection", "router-summary,loss-summary"]
        if extra_args:
            combined_extra_args += extra_args
        expected_attempt_status_rows = (
            0 if "--skip-train" in combined_extra_args else train_iters
        )

        db_cmd = _tiny_megatron_router_summary_cmd(
            model_id=model_id,
            train_iters=train_iters,
            eval_iters=eval_iters,
            eval_interval=eval_interval,
            micro_batch_size=micro_batch_size,
            global_batch_size=global_batch_size,
            database=database,
            table=table,
            extra_args=combined_extra_args,
        )
        _run_megatron_cmd(
            db_cmd,
            env=base_env,
            log_path=tmp_path / "megatron_numeric_clickhouse.log",
        )
        _wait_for_exact_rows(
            client,
            database=database,
            table=table,
            model_id=model_id,
            expected=expected_tensor_rows,
        )
        _wait_for_exact_scalar_rows(
            client,
            database=database,
            table=f"{table}_scalar_float",
            model_id=model_id,
            act_name="lm_per_sample_loss",
            expected=expected_scalar_float_rows,
        )
        _wait_for_exact_scalar_rows(
            client,
            database=database,
            table=f"{table}_scalar_int",
            model_id=model_id,
            act_name="lm_per_sample_loss_token_count",
            expected=expected_scalar_float_rows,
        )
        _assert_training_table_counts(
            client=client,
            database=database,
            table=table,
            model_id=model_id,
            expected_tensor_rows=expected_tensor_rows,
            expected_scalar_float_rows=expected_scalar_float_rows,
            expected_attempt_status_rows=expected_attempt_status_rows,
        )

        file_env = dict(base_env)
        file_env["DMI_TEST_FILE_SINK_DIR"] = str(file_sink_dir)
        file_cmd = _tiny_megatron_router_summary_cmd(
            model_id=model_id,
            train_iters=train_iters,
            eval_iters=eval_iters,
            eval_interval=eval_interval,
            micro_batch_size=micro_batch_size,
            global_batch_size=global_batch_size,
            extra_args=combined_extra_args,
        )
        file_cmd[file_cmd.index("pretrain_gpt.py")] = str(
            ROOT / "tests" / "oracles" / "run_megatron_file_sink_oracle.py"
        )
        _run_megatron_cmd(
            file_cmd,
            env=file_env,
            log_path=tmp_path / "megatron_numeric_file_sink.log",
        )

        db_rows = _read_training_rows_all_phases(
            model_id=model_id,
            table=table,
            database=database,
        )
        file_rows = _read_file_sink_rows(file_sink_dir)
        db_scalar_rows = _read_training_scalar_rows_all_phases(
            model_id=model_id,
            table=table,
            database=database,
            scalar_kind="float",
            act_name="lm_per_sample_loss",
        )
        db_scalar_int_rows = _read_training_scalar_rows_all_phases(
            model_id=model_id,
            table=table,
            database=database,
            scalar_kind="int",
            act_name="lm_per_sample_loss_token_count",
        )
        db_attempt_status_rows = _read_training_scalar_rows_all_phases(
            model_id=model_id,
            table=table,
            database=database,
            scalar_kind="int",
            act_name="iteration_attempt_status",
            direction="iter",
        )
        file_scalar_rows = _read_file_sink_scalar_rows(file_sink_dir, "scalar_float_rows.jsonl")
        all_file_scalar_int_rows = _read_file_sink_scalar_rows(
            file_sink_dir, "scalar_int_rows.jsonl"
        )
        file_scalar_int_rows = [
            row
            for row in all_file_scalar_int_rows
            if row["act_name"] == "lm_per_sample_loss_token_count"
        ]
        file_attempt_status_rows = [
            row
            for row in all_file_scalar_int_rows
            if row["act_name"] == "iteration_attempt_status"
        ]

        assert len(db_rows) == expected_tensor_rows
        assert len(file_rows) == expected_tensor_rows
        assert len(db_scalar_rows) == expected_scalar_float_rows
        assert len(file_scalar_rows) == expected_scalar_float_rows
        assert len(db_scalar_int_rows) == expected_scalar_float_rows
        assert len(file_scalar_int_rows) == expected_scalar_float_rows
        assert len(db_attempt_status_rows) == expected_attempt_status_rows
        assert len(file_attempt_status_rows) == expected_attempt_status_rows
        assert len(all_file_scalar_int_rows) == (
            expected_scalar_float_rows + expected_attempt_status_rows
        )

        db_by_key = {key: tensor for key, tensor in db_rows}
        file_by_key = {}
        for row, tensor in file_rows:
            key = _file_sink_row_key(row)
            assert row["dtype"] == "torch.float"
            assert row["shape"] == [2]
            assert row["bytes"] == 8
            file_by_key[key] = tensor

        assert set(file_by_key) == set(db_by_key)
        _assert_tensor_maps_close_with_report(file_by_key, db_by_key)

        db_scalar_by_key = {key: value for key, value in db_scalar_rows}
        file_scalar_by_key = {_file_sink_row_key(row): float(row["value"]) for row in file_scalar_rows}
        assert set(file_scalar_by_key) == set(db_scalar_by_key)
        _assert_scalar_maps_close_with_report(file_scalar_by_key, db_scalar_by_key)
        db_scalar_int_by_key = {key: int(value) for key, value in db_scalar_int_rows}
        file_scalar_int_by_key = {
            _file_sink_row_key(row): int(row["value"])
            for row in file_scalar_int_rows
        }
        assert file_scalar_int_by_key == db_scalar_int_by_key
        db_attempt_status_by_key = {
            key: int(value) for key, value in db_attempt_status_rows
        }
        file_attempt_status_by_key = {
            _file_sink_row_key(row): int(row["value"])
            for row in file_attempt_status_rows
        }
        assert file_attempt_status_by_key == db_attempt_status_by_key
        assert set(db_attempt_status_by_key.values()) <= {1}
        _assert_file_sink_boundaries_match_clickhouse(
            client=client,
            database=database,
            table=table,
            model_id=model_id,
            file_sink_dir=file_sink_dir,
        )
    finally:
        client.execute(f"DROP TABLE IF EXISTS `{database}`.`{table}`")
        client.execute(f"DROP TABLE IF EXISTS `{database}`.`{table}_scalar_float`")
        client.execute(f"DROP TABLE IF EXISTS `{database}`.`{table}_scalar_int`")
        client.execute(f"DROP TABLE IF EXISTS `{database}`.`{table}_eval_phase_boundary`")
        client.disconnect()


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Real Megatron hidden-state E2E needs CUDA")
def test_real_megatron_hidden_states_clickhouse_rows(tmp_path):
    """Run real Megatron eval and verify full hidden-state tensor rows."""

    if _available_cuda_devices() < 1:
        pytest.skip("hidden-states E2E requires one CUDA device")

    client = _clickhouse_client_or_skip()
    database = os.environ.get("DMX_DB_DATABASE", "default")
    table = f"dmi_megatron_hidden_states_e2e_{uuid.uuid4().hex}"
    model_id = f"megatron-hidden-states-e2e-{uuid.uuid4().hex}"
    log_path = tmp_path / "megatron_real_training_hidden_states.log"

    train_iters = 1
    eval_iters = 1
    micro_batch_size = 2
    global_batch_size = 2
    num_layers = 2
    expected_rows = 2 * eval_iters * global_batch_size * num_layers

    client.execute(f"CREATE DATABASE IF NOT EXISTS `{database}`")
    client.execute(f"DROP TABLE IF EXISTS `{database}`.`{table}`")
    _create_training_table(client, database=database, table=table)

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{ROOT}:{MEGATRON_ROOT}:{env.get('PYTHONPATH', '')}"
    env["DMI_ENABLE"] = "1"
    env.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", "1")
    if "DMI_REAL_E2E_CUDA_VISIBLE_DEVICES" in env:
        env["CUDA_VISIBLE_DEVICES"] = env["DMI_REAL_E2E_CUDA_VISIBLE_DEVICES"]

    cmd = _tiny_megatron_router_summary_cmd(
        model_id=model_id,
        train_iters=train_iters,
        eval_iters=eval_iters,
        micro_batch_size=micro_batch_size,
        global_batch_size=global_batch_size,
        database=database,
        table=table,
        extra_args=[
            "--dmi-hook-selection",
            "hidden-states",
            "--skip-train",
            "--no-load-optim",
        ],
    )

    try:
        _run_megatron_cmd(cmd, env=env, log_path=log_path)
        _wait_for_exact_act_rows(
            client,
            database=database,
            table=table,
            model_id=model_id,
            act_name="hidden_states",
            expected=expected_rows,
        )
        assert _query_model_count(
            client,
            database=database,
            table=f"{table}_scalar_float",
            model_id=model_id,
        ) == 0
        assert _query_model_count(
            client,
            database=database,
            table=f"{table}_scalar_int",
            model_id=model_id,
        ) == 0
        assert _query_model_count(
            client,
            database=database,
            table=f"{table}_eval_phase_boundary",
            model_id=model_id,
        ) >= 1

        rows = _query_tensor_row_shapes(
            client,
            database=database,
            table=table,
            model_id=model_id,
            act_name="hidden_states",
        )
        assert len(rows) == expected_rows
        assert {row[0] for row in rows} == {"torch.bfloat16"}
        assert {tuple(row[1]) for row in rows} == {(16, 64)}
        assert {int(row[2]) for row in rows} == {16 * 64 * 2}
    finally:
        client.execute(f"DROP TABLE IF EXISTS `{database}`.`{table}`")
        client.execute(f"DROP TABLE IF EXISTS `{database}`.`{table}_scalar_float`")
        client.execute(f"DROP TABLE IF EXISTS `{database}`.`{table}_scalar_int`")
        client.execute(f"DROP TABLE IF EXISTS `{database}`.`{table}_eval_phase_boundary`")
        client.disconnect()


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Real Megatron vocabulary-logit E2E needs CUDA")
def test_real_megatron_vocab_logits_training_clickhouse_rows_and_boundary_flush(tmp_path):
    """Run real Megatron training and verify raw vocabulary logits plus boundary flushes."""

    if _available_cuda_devices() < 1:
        pytest.skip("vocab-logits E2E requires one CUDA device")

    client = _clickhouse_client_or_skip()
    database = os.environ.get("DMX_DB_DATABASE", "default")
    table = f"dmi_megatron_vocab_logits_e2e_{uuid.uuid4().hex}"
    model_id = f"megatron-vocab-logits-e2e-{uuid.uuid4().hex}"
    log_path = tmp_path / "megatron_real_training_vocab_logits.log"

    train_iters = 4
    micro_batch_size = 2
    global_batch_size = 2
    seq_length = 16
    # NullTokenizer adds its end-of-document token before Megatron pads the LM
    # head to the configured divisibility, so --vocab-size 128 becomes 256.
    padded_vocab_size = 256
    expected_rows = train_iters * global_batch_size

    client.execute(f"CREATE DATABASE IF NOT EXISTS `{database}`")
    client.execute(f"DROP TABLE IF EXISTS `{database}`.`{table}`")
    _create_training_table(client, database=database, table=table)

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{ROOT}:{MEGATRON_ROOT}:{env.get('PYTHONPATH', '')}"
    env["DMI_ENABLE"] = "1"
    env.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", "1")
    if "DMI_REAL_E2E_CUDA_VISIBLE_DEVICES" in env:
        env["CUDA_VISIBLE_DEVICES"] = env["DMI_REAL_E2E_CUDA_VISIBLE_DEVICES"]

    cmd = _tiny_megatron_router_summary_cmd(
        model_id=model_id,
        train_iters=train_iters,
        eval_iters=0,
        micro_batch_size=micro_batch_size,
        global_batch_size=global_batch_size,
        database=database,
        table=table,
        extra_args=[
            "--dmi-hook-selection",
            "vocab-logits",
            "--dmi-flush-every-n-train-iters",
            "2",
        ],
    )

    try:
        _run_megatron_cmd(cmd, env=env, log_path=log_path)
        _wait_for_exact_act_rows(
            client,
            database=database,
            table=table,
            model_id=model_id,
            act_name="vocab_logits",
            expected=expected_rows,
        )
        rows = _query_tensor_row_shapes(
            client,
            database=database,
            table=table,
            model_id=model_id,
            act_name="vocab_logits",
        )
        assert len(rows) == expected_rows
        assert {row[0] for row in rows} == {"torch.bfloat16"}
        assert {tuple(row[1]) for row in rows} == {(seq_length, padded_vocab_size)}
        assert {int(row[2]) for row in rows} == {seq_length * padded_vocab_size * 2}

        log_text = log_path.read_text(encoding="utf-8", errors="replace")
        flush_iterations = re.findall(
            r"\[DMI\] iteration-boundary flush iteration=(\d+) elapsed_s=",
            log_text,
        )
        assert flush_iterations == ["2", "4"]
    finally:
        client.execute(f"DROP TABLE IF EXISTS `{database}`.`{table}`")
        client.execute(f"DROP TABLE IF EXISTS `{database}`.`{table}_scalar_float`")
        client.execute(f"DROP TABLE IF EXISTS `{database}`.`{table}_scalar_int`")
        client.execute(f"DROP TABLE IF EXISTS `{database}`.`{table}_eval_phase_boundary`")
        client.disconnect()


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Real Megatron vocabulary top-K E2E needs CUDA")
def test_real_megatron_vocab_logits_topk_training_clickhouse_rows(tmp_path):
    """Run real Megatron training and verify both fixed-K vocabulary-logit outputs."""

    if _available_cuda_devices() < 1:
        pytest.skip("vocab-logits-topk E2E requires one CUDA device")

    client = _clickhouse_client_or_skip()
    database = os.environ.get("DMX_DB_DATABASE", "default")
    table = f"dmi_megatron_vocab_logits_topk_e2e_{uuid.uuid4().hex}"
    model_id = f"megatron-vocab-logits-topk-e2e-{uuid.uuid4().hex}"
    log_path = tmp_path / "megatron_real_training_vocab_logits_topk.log"

    train_iters = 4
    micro_batch_size = 2
    global_batch_size = 2
    seq_length = 16
    padded_vocab_size = 256
    top_k = 100
    expected_rows_per_output = train_iters * global_batch_size

    client.execute(f"CREATE DATABASE IF NOT EXISTS `{database}`")
    client.execute(f"DROP TABLE IF EXISTS `{database}`.`{table}`")
    _create_training_table(client, database=database, table=table)

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{ROOT}:{MEGATRON_ROOT}:{env.get('PYTHONPATH', '')}"
    env["DMI_ENABLE"] = "1"
    env.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", "1")
    if "DMI_REAL_E2E_CUDA_VISIBLE_DEVICES" in env:
        env["CUDA_VISIBLE_DEVICES"] = env["DMI_REAL_E2E_CUDA_VISIBLE_DEVICES"]

    cmd = _tiny_megatron_router_summary_cmd(
        model_id=model_id,
        train_iters=train_iters,
        eval_iters=0,
        micro_batch_size=micro_batch_size,
        global_batch_size=global_batch_size,
        database=database,
        table=table,
        extra_args=[
            "--dmi-hook-selection",
            "vocab-logits-topk",
            "--dmi-vocab-logits-top-k",
            str(top_k),
        ],
    )

    try:
        _run_megatron_cmd(cmd, env=env, log_path=log_path)
        for act_name in (
            "vocab_logits_topk_values",
            "vocab_logits_topk_indices",
        ):
            _wait_for_exact_act_rows(
                client,
                database=database,
                table=table,
                model_id=model_id,
                act_name=act_name,
                expected=expected_rows_per_output,
            )

        rows = client.execute(
            f"""
            SELECT
                act_name, phase, global_batch_id, dp_rank, microbatch_id,
                sample_index, layer_no, shard_rank, dtype, shape, bytes
            FROM `{database}`.`{table}`
            WHERE model_id = %(model_id)s
              AND act_name IN %(act_names)s
              AND direction = 'fwd'
            ORDER BY phase, global_batch_id, microbatch_id, sample_index, act_name
            """,
            {
                "model_id": model_id,
                "act_names": (
                    "vocab_logits_topk_values",
                    "vocab_logits_topk_indices",
                ),
            },
        )
        assert len(rows) == 2 * expected_rows_per_output
        by_key = {}
        for row in rows:
            act_name, *metadata, dtype, shape, payload = row
            key = tuple(metadata)
            by_key.setdefault(key, {})[str(act_name)] = (
                str(dtype),
                tuple(shape),
                payload,
            )
        assert len(by_key) == expected_rows_per_output
        for outputs in by_key.values():
            assert set(outputs) == {
                "vocab_logits_topk_values",
                "vocab_logits_topk_indices",
            }
            values_dtype, values_shape, values_bytes = outputs[
                "vocab_logits_topk_values"
            ]
            indices_dtype, indices_shape, indices_bytes = outputs[
                "vocab_logits_topk_indices"
            ]
            assert values_dtype == "torch.bfloat16"
            # The native ClickHouse sink canonically serializes at::kInt
            # (torch.int32) as "torch.int".
            assert indices_dtype == "torch.int"
            assert values_shape == (seq_length, top_k)
            assert indices_shape == (seq_length, top_k)
            assert len(values_bytes) == seq_length * top_k * 2
            assert len(indices_bytes) == seq_length * top_k * 4

            values = torch.frombuffer(
                bytearray(values_bytes), dtype=torch.bfloat16
            ).reshape(seq_length, top_k)
            indices = torch.frombuffer(
                bytearray(indices_bytes), dtype=torch.int32
            ).reshape(seq_length, top_k)
            assert torch.all(values[:, :-1] >= values[:, 1:])
            assert int(indices.min()) >= 0
            assert int(indices.max()) < padded_vocab_size
    finally:
        client.execute(f"DROP TABLE IF EXISTS `{database}`.`{table}`")
        client.execute(f"DROP TABLE IF EXISTS `{database}`.`{table}_scalar_float`")
        client.execute(f"DROP TABLE IF EXISTS `{database}`.`{table}_scalar_int`")
        client.execute(f"DROP TABLE IF EXISTS `{database}`.`{table}_eval_phase_boundary`")
        client.disconnect()


def _run_graph_vs_eager_file_sink_numeric_check(
    *,
    tmp_path: Path,
    database: str,
    table: str,
    model_id: str,
    train_iters: int,
    eval_iters: int,
    eval_interval: int,
    micro_batch_size: int,
    global_batch_size: int,
    expected_tensor_rows: int,
    expected_scalar_float_rows: int,
    reference_args: list[str] | None = None,
    graph_args: list[str],
    label: str,
    allow_graph_subset: bool = False,
    transformer_impl: str = "local",
    require_te_graph_capture: bool = False,
) -> None:
    client = _clickhouse_client_or_skip()
    file_sink_dir = tmp_path / f"file_sink_{label}"

    base_env = os.environ.copy()
    base_env["PYTHONPATH"] = f"{ROOT}:{MEGATRON_ROOT}:{base_env.get('PYTHONPATH', '')}"
    base_env["DMI_ENABLE"] = "1"
    base_env.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", "1")
    if "DMI_REAL_E2E_CUDA_VISIBLE_DEVICES" in base_env:
        base_env["CUDA_VISIBLE_DEVICES"] = base_env["DMI_REAL_E2E_CUDA_VISIBLE_DEVICES"]

    client.execute(f"CREATE DATABASE IF NOT EXISTS `{database}`")
    client.execute(f"DROP TABLE IF EXISTS `{database}`.`{table}`")
    _create_training_table(client, database=database, table=table)

    try:
        combined_reference_args = ["--dmi-hook-selection", "router-summary,loss-summary"]
        if reference_args:
            combined_reference_args += reference_args
        combined_graph_args = ["--dmi-hook-selection", "router-summary,loss-summary"]
        combined_graph_args += graph_args

        file_env = dict(base_env)
        file_env["DMI_TEST_FILE_SINK_DIR"] = str(file_sink_dir)
        file_cmd = _tiny_megatron_router_summary_cmd(
            model_id=model_id,
            train_iters=train_iters,
            eval_iters=eval_iters,
            eval_interval=eval_interval,
            micro_batch_size=micro_batch_size,
            global_batch_size=global_batch_size,
            transformer_impl=transformer_impl,
            extra_args=combined_reference_args,
        )
        file_cmd[file_cmd.index("pretrain_gpt.py")] = str(
            ROOT / "tests" / "oracles" / "run_megatron_file_sink_oracle.py"
        )
        _run_megatron_cmd(
            file_cmd,
            env=file_env,
            log_path=tmp_path / f"megatron_{label}_eager_file_sink.log",
        )

        db_cmd = _tiny_megatron_router_summary_cmd(
            model_id=model_id,
            train_iters=train_iters,
            eval_iters=eval_iters,
            eval_interval=eval_interval,
            micro_batch_size=micro_batch_size,
            global_batch_size=global_batch_size,
            transformer_impl=transformer_impl,
            database=database,
            table=table,
            extra_args=combined_graph_args,
        )
        graph_log_path = tmp_path / f"megatron_{label}_clickhouse.log"
        _run_megatron_cmd(
            db_cmd,
            env=base_env,
            log_path=graph_log_path,
        )
        if require_te_graph_capture:
            graph_log = graph_log_path.read_text(encoding="utf-8", errors="replace")
            assert "Time spent in CUDA Graphs capture" in graph_log
            assert (
                "TECudaGraphHelper: No graphable layers found. "
                "Skipping CUDA graph capture."
            ) not in graph_log
        _wait_for_exact_rows(
            client,
            database=database,
            table=table,
            model_id=model_id,
            expected=expected_tensor_rows,
        )
        _wait_for_exact_scalar_rows(
            client,
            database=database,
            table=f"{table}_scalar_float",
            model_id=model_id,
            act_name="lm_per_sample_loss",
            expected=expected_scalar_float_rows,
        )
        _wait_for_exact_scalar_rows(
            client,
            database=database,
            table=f"{table}_scalar_int",
            model_id=model_id,
            act_name="lm_per_sample_loss_token_count",
            expected=expected_scalar_float_rows,
        )
        _assert_training_table_counts(
            client=client,
            database=database,
            table=table,
            model_id=model_id,
            expected_tensor_rows=expected_tensor_rows,
            expected_scalar_float_rows=expected_scalar_float_rows,
            expected_attempt_status_rows=train_iters,
        )

        db_rows = _read_training_rows_all_phases(
            model_id=model_id,
            table=table,
            database=database,
        )
        file_rows = _read_file_sink_rows(file_sink_dir)
        db_scalar_rows = _read_training_scalar_rows_all_phases(
            model_id=model_id,
            table=table,
            database=database,
            scalar_kind="float",
            act_name="lm_per_sample_loss",
        )
        db_scalar_int_rows = _read_training_scalar_rows_all_phases(
            model_id=model_id,
            table=table,
            database=database,
            scalar_kind="int",
            act_name="lm_per_sample_loss_token_count",
        )
        db_attempt_status_rows = _read_training_scalar_rows_all_phases(
            model_id=model_id,
            table=table,
            database=database,
            scalar_kind="int",
            act_name="iteration_attempt_status",
            direction="iter",
        )
        file_scalar_rows = _read_file_sink_scalar_rows(file_sink_dir, "scalar_float_rows.jsonl")
        all_file_scalar_int_rows = _read_file_sink_scalar_rows(
            file_sink_dir, "scalar_int_rows.jsonl"
        )
        file_scalar_int_rows = [
            row
            for row in all_file_scalar_int_rows
            if row["act_name"] == "lm_per_sample_loss_token_count"
        ]
        file_attempt_status_rows = [
            row
            for row in all_file_scalar_int_rows
            if row["act_name"] == "iteration_attempt_status"
        ]

        assert len(db_rows) == expected_tensor_rows
        assert len(db_scalar_rows) == expected_scalar_float_rows
        assert len(db_scalar_int_rows) == expected_scalar_float_rows
        assert len(db_attempt_status_rows) == train_iters
        assert len(file_attempt_status_rows) == train_iters
        assert len(all_file_scalar_int_rows) == len(file_scalar_int_rows) + train_iters
        if allow_graph_subset:
            assert len(file_rows) >= expected_tensor_rows
            assert len(file_scalar_rows) >= expected_scalar_float_rows
            assert len(file_scalar_int_rows) >= expected_scalar_float_rows
        else:
            assert len(file_rows) == expected_tensor_rows
            assert len(file_scalar_rows) == expected_scalar_float_rows
            assert len(file_scalar_int_rows) == expected_scalar_float_rows

        db_by_key = {key: tensor for key, tensor in db_rows}
        file_by_key = {}
        for row, tensor in file_rows:
            key = _file_sink_row_key(row)
            file_by_key[key] = tensor

        db_scalar_by_key = {key: value for key, value in db_scalar_rows}
        file_scalar_by_key = {_file_sink_row_key(row): float(row["value"]) for row in file_scalar_rows}
        db_scalar_int_by_key = {key: int(value) for key, value in db_scalar_int_rows}
        file_scalar_int_by_key = {
            _file_sink_row_key(row): int(row["value"])
            for row in file_scalar_int_rows
        }
        db_attempt_status_by_key = {
            key: int(value) for key, value in db_attempt_status_rows
        }
        file_attempt_status_by_key = {
            _file_sink_row_key(row): int(row["value"])
            for row in file_attempt_status_rows
        }
        assert file_attempt_status_by_key == db_attempt_status_by_key
        assert set(db_attempt_status_by_key.values()) == {1}
        if allow_graph_subset:
            missing_from_file = set(db_by_key) - set(file_by_key)
            assert not missing_from_file
            missing_scalar_from_file = set(db_scalar_by_key) - set(file_scalar_by_key)
            assert not missing_scalar_from_file
            graph_phases = {key[3] for key in db_by_key}
            assert graph_phases == {"train", "valid", "test"}
            file_by_key = {key: file_by_key[key] for key in db_by_key}
            file_scalar_by_key = {key: file_scalar_by_key[key] for key in db_scalar_by_key}
            missing_int_from_file = set(db_scalar_int_by_key) - set(file_scalar_int_by_key)
            assert not missing_int_from_file
            file_scalar_int_by_key = {
                key: file_scalar_int_by_key[key] for key in db_scalar_int_by_key
            }
        else:
            assert set(file_by_key) == set(db_by_key)
            assert set(file_scalar_by_key) == set(db_scalar_by_key)
            assert set(file_scalar_int_by_key) == set(db_scalar_int_by_key)
        _assert_tensor_maps_close_with_report(
            file_by_key,
            db_by_key,
            label=f"eager file sink vs {label} ClickHouse",
            atol=float(os.environ.get("DMI_GRAPH_NUMERIC_E2E_ATOL", "5e-2")),
            rtol=float(os.environ.get("DMI_GRAPH_NUMERIC_E2E_RTOL", "1e-1")),
        )
        _assert_scalar_maps_close_with_report(
            file_scalar_by_key,
            db_scalar_by_key,
            label=f"eager file sink vs {label} ClickHouse",
            atol=float(os.environ.get("DMI_GRAPH_NUMERIC_E2E_ATOL", "5e-2")),
            rtol=float(os.environ.get("DMI_GRAPH_NUMERIC_E2E_RTOL", "1e-1")),
        )
        assert file_scalar_int_by_key == db_scalar_int_by_key
        if not allow_graph_subset:
            _assert_file_sink_boundaries_match_clickhouse(
                client=client,
                database=database,
                table=table,
                model_id=model_id,
                file_sink_dir=file_sink_dir,
            )
    finally:
        client.execute(f"DROP TABLE IF EXISTS `{database}`.`{table}`")
        client.execute(f"DROP TABLE IF EXISTS `{database}`.`{table}_scalar_float`")
        client.execute(f"DROP TABLE IF EXISTS `{database}`.`{table}_scalar_int`")
        client.execute(f"DROP TABLE IF EXISTS `{database}`.`{table}_eval_phase_boundary`")
        client.disconnect()


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Real Megatron numeric E2E needs CUDA")
def test_real_megatron_router_and_loss_summary_file_sink_matches_clickhouse_numeric(tmp_path):
    """Run real Megatron twice and compare combined hook rows against ClickHouse rows."""

    if _available_cuda_devices() < 1:
        pytest.skip("numeric E2E requires one CUDA device")

    database = os.environ.get("DMX_DB_DATABASE", "default")
    table = f"dmi_megatron_numeric_e2e_{uuid.uuid4().hex}"
    model_id = f"megatron-numeric-e2e-{uuid.uuid4().hex}"
    train_iters = int(os.environ.get("DMI_NUMERIC_E2E_TRAIN_ITERS", "2"))
    eval_iters = int(os.environ.get("DMI_NUMERIC_E2E_EVAL_ITERS", "1"))
    eval_interval = int(os.environ.get("DMI_NUMERIC_E2E_EVAL_INTERVAL", "1"))
    micro_batch_size = 2
    global_batch_size = 2
    expected_tensor_rows = _expected_total_rows(
        train_iters=train_iters,
        eval_iters=eval_iters,
        eval_interval=eval_interval,
        global_batch_size=global_batch_size,
        num_moe_layers=2,
    )
    expected_scalar_float_rows = _expected_total_sample_rows(
        train_iters=train_iters,
        eval_iters=eval_iters,
        eval_interval=eval_interval,
        global_batch_size=global_batch_size,
    )

    _run_two_path_numeric_check(
        tmp_path=tmp_path,
        database=database,
        table=table,
        model_id=model_id,
        train_iters=train_iters,
        eval_iters=eval_iters,
        eval_interval=eval_interval,
        micro_batch_size=micro_batch_size,
        global_batch_size=global_batch_size,
        expected_tensor_rows=expected_tensor_rows,
        expected_scalar_float_rows=expected_scalar_float_rows,
    )


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Real Megatron eval E2E needs CUDA")
def test_real_megatron_router_and_loss_summary_eval_only_file_sink_matches_clickhouse_numeric(tmp_path):
    """Run validation-only Megatron and compare combined hook rows against ClickHouse rows."""

    if _available_cuda_devices() < 1:
        pytest.skip("eval numeric E2E requires one CUDA device")

    database = os.environ.get("DMX_DB_DATABASE", "default")
    table = f"dmi_megatron_eval_numeric_e2e_{uuid.uuid4().hex}"
    model_id = f"megatron-eval-numeric-e2e-{uuid.uuid4().hex}"
    train_iters = 1
    eval_iters = 1
    micro_batch_size = 2
    global_batch_size = 2
    # Megatron --skip-train with eval_iters>0 runs both validation and test.
    expected_tensor_rows = _expected_total_rows(
        train_iters=train_iters,
        eval_iters=eval_iters,
        global_batch_size=global_batch_size,
        num_moe_layers=2,
        skip_train=True,
    )
    expected_scalar_float_rows = _expected_total_sample_rows(
        train_iters=train_iters,
        eval_iters=eval_iters,
        global_batch_size=global_batch_size,
        skip_train=True,
    )

    _run_two_path_numeric_check(
        tmp_path=tmp_path,
        database=database,
        table=table,
        model_id=model_id,
        train_iters=train_iters,
        eval_iters=eval_iters,
        eval_interval=None,
        micro_batch_size=micro_batch_size,
        global_batch_size=global_batch_size,
        expected_tensor_rows=expected_tensor_rows,
        expected_scalar_float_rows=expected_scalar_float_rows,
        extra_args=["--skip-train", "--no-load-optim"],
    )


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Real Megatron graph numeric E2E needs CUDA")
@pytest.mark.parametrize(
    (
        "label",
        "transformer_impl",
        "reference_args",
        "graph_args",
        "allow_graph_subset",
        "require_te_graph_capture",
    ),
    [
        pytest.param(
            "local_cuda_graph",
            "local",
            [
                "--moe-token-dispatcher-type",
                "alltoall",
                "--moe-expert-capacity-factor",
                "1.0",
                "--moe-pad-expert-input-to-capacity",
            ],
            [
                "--moe-token-dispatcher-type",
                "alltoall",
                "--moe-expert-capacity-factor",
                "1.0",
                "--moe-pad-expert-input-to-capacity",
                "--cuda-graph-impl",
                "local",
            ],
            False,
            False,
            id="local_cuda_graph",
        ),
        pytest.param(
            "full_iteration_cuda_graph",
            "local",
            [
                "--moe-token-dispatcher-type",
                "alltoall",
                "--moe-expert-capacity-factor",
                "1.0",
                "--moe-pad-expert-input-to-capacity",
            ],
            [
                "--moe-token-dispatcher-type",
                "alltoall",
                "--moe-expert-capacity-factor",
                "1.0",
                "--moe-pad-expert-input-to-capacity",
                "--cuda-graph-impl",
                "local",
                "--cuda-graph-scope",
                "full_iteration",
                "--no-check-for-nan-in-loss-and-grad",
            ],
            False,
            False,
            id="full_iteration_cuda_graph",
        ),
        pytest.param(
            "transformer_engine_scoped_cuda_graph",
            "transformer_engine",
            [
                "--moe-token-dispatcher-type",
                "alltoall",
            ],
            [
                "--moe-token-dispatcher-type",
                "alltoall",
                "--cuda-graph-impl",
                "transformer_engine",
                "--cuda-graph-scope",
                "attn",
                "moe_router",
                "moe_preprocess",
            ],
            False,
            True,
            id="transformer_engine_scoped_cuda_graph",
        ),
    ],
)
def test_real_megatron_router_and_loss_summary_graph_matches_eager_file_sink_numeric(
    tmp_path,
    label,
    transformer_impl,
    reference_args,
    graph_args,
    allow_graph_subset,
    require_te_graph_capture,
):
    """Compare graph-mode ClickHouse rows against eager file-sink reference."""

    if _available_cuda_devices() < 1:
        pytest.skip("graph numeric E2E requires one CUDA device")

    database = os.environ.get("DMX_DB_DATABASE", "default")
    table = f"dmi_megatron_graph_numeric_e2e_{label}_{uuid.uuid4().hex}"
    model_id = f"megatron-graph-numeric-e2e-{label}-{uuid.uuid4().hex}"
    default_train_iters = "6" if require_te_graph_capture else "3"
    train_iters = int(
        os.environ.get("DMI_GRAPH_NUMERIC_E2E_TRAIN_ITERS", default_train_iters)
    )
    if require_te_graph_capture and train_iters < 6:
        raise ValueError(
            "Transformer Engine graph E2E requires at least 6 train iterations"
        )
    eval_iters = int(os.environ.get("DMI_GRAPH_NUMERIC_E2E_EVAL_ITERS", "1"))
    eval_interval = int(os.environ.get("DMI_GRAPH_NUMERIC_E2E_EVAL_INTERVAL", "1"))
    micro_batch_size = 2
    global_batch_size = 2
    expected_tensor_rows = _expected_total_rows(
        train_iters=train_iters,
        eval_iters=eval_iters,
        eval_interval=eval_interval,
        global_batch_size=global_batch_size,
        num_moe_layers=2,
    )
    expected_scalar_float_rows = _expected_total_sample_rows(
        train_iters=train_iters,
        eval_iters=eval_iters,
        eval_interval=eval_interval,
        global_batch_size=global_batch_size,
    )
    _run_graph_vs_eager_file_sink_numeric_check(
        tmp_path=tmp_path,
        database=database,
        table=table,
        model_id=model_id,
        train_iters=train_iters,
        eval_iters=eval_iters,
        eval_interval=eval_interval,
        micro_batch_size=micro_batch_size,
        global_batch_size=global_batch_size,
        expected_tensor_rows=expected_tensor_rows,
        expected_scalar_float_rows=expected_scalar_float_rows,
        reference_args=reference_args,
        graph_args=graph_args,
        label=label,
        allow_graph_subset=allow_graph_subset,
        transformer_impl=transformer_impl,
        require_te_graph_capture=require_te_graph_capture,
    )


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Real Megatron E2E needs CUDA")
def test_real_megatron_loss_summary_clickhouse_rows(tmp_path):
    """Run real Megatron eval and verify loss-summary scalar rows."""

    if _available_cuda_devices() < 1:
        pytest.skip("loss-summary E2E requires one CUDA device")

    client = _clickhouse_client_or_skip()
    database = os.environ.get("DMX_DB_DATABASE", "default")
    table = f"dmi_megatron_loss_e2e_{uuid.uuid4().hex}"
    scalar_table = f"{table}_scalar_float"
    scalar_int_table = f"{table}_scalar_int"
    model_id = f"megatron-loss-e2e-{uuid.uuid4().hex}"
    log_path = tmp_path / "megatron_real_training_loss_summary.log"

    train_iters = 1
    eval_iters = 1
    micro_batch_size = 1
    global_batch_size = 1
    # Megatron --skip-train with eval_iters>0 runs both validation and test.
    expected_rows = 2 * eval_iters * global_batch_size

    client.execute(f"CREATE DATABASE IF NOT EXISTS `{database}`")
    client.execute(f"DROP TABLE IF EXISTS `{database}`.`{table}`")
    client.execute(f"DROP TABLE IF EXISTS `{database}`.`{scalar_table}`")
    _create_training_table(client, database=database, table=table)

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{ROOT}:{MEGATRON_ROOT}:{env.get('PYTHONPATH', '')}"
    env["DMI_ENABLE"] = "1"
    env.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", "1")
    if "DMI_REAL_E2E_CUDA_VISIBLE_DEVICES" in env:
        env["CUDA_VISIBLE_DEVICES"] = env["DMI_REAL_E2E_CUDA_VISIBLE_DEVICES"]

    cmd = _tiny_megatron_router_summary_cmd(
        model_id=model_id,
        train_iters=train_iters,
        eval_iters=eval_iters,
        micro_batch_size=micro_batch_size,
        global_batch_size=global_batch_size,
        database=database,
        table=table,
        extra_args=["--dmi-hook-selection", "loss-summary", "--skip-train", "--no-load-optim"],
    )

    try:
        _run_megatron_cmd(cmd, env=env, log_path=log_path)
        _wait_for_exact_scalar_rows(
            client,
            database=database,
            table=scalar_table,
            model_id=model_id,
            act_name="lm_per_sample_loss",
            expected=expected_rows,
        )
        rows = _query_scalar_values(
            client,
            database=database,
            table=scalar_table,
            model_id=model_id,
            act_name="lm_per_sample_loss",
        )
        assert len(rows) == expected_rows
        for phase, _global_batch_id, _microbatch_id, _sample_index, layer_no, value in rows:
            phase = phase.decode("utf-8") if isinstance(phase, bytes) else phase
            assert phase in {"valid", "test"}
            assert int(layer_no) == -1
            assert float(value) > 0.0
        _wait_for_exact_scalar_rows(
            client,
            database=database,
            table=scalar_int_table,
            model_id=model_id,
            act_name="lm_per_sample_loss_token_count",
            expected=expected_rows,
        )
    finally:
        client.execute(f"DROP TABLE IF EXISTS `{database}`.`{table}`")
        client.execute(f"DROP TABLE IF EXISTS `{database}`.`{scalar_table}`")
        client.execute(f"DROP TABLE IF EXISTS `{database}`.`{table}_scalar_int`")
        client.disconnect()


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Real Megatron training E2E needs CUDA")
def test_real_megatron_training_health_signals_clickhouse_rows(tmp_path):
    """Verify reusable router-health and iteration-level signals in ClickHouse."""

    if _available_cuda_devices() < 1:
        pytest.skip("training-health E2E requires one CUDA device")

    client = _clickhouse_client_or_skip()
    database = os.environ.get("DMX_DB_DATABASE", "default")
    table = f"dmi_megatron_training_health_e2e_{uuid.uuid4().hex}"
    scalar_table = f"{table}_scalar_float"
    processed_database = f"dmi_megatron_processed_e2e_{uuid.uuid4().hex}"
    model_id = f"megatron-training-health-e2e-{uuid.uuid4().hex}"
    log_path = tmp_path / "megatron_real_training_health_signals.log"

    train_iters = 2
    micro_batch_size = 2
    global_batch_size = 2
    num_moe_layers = 2
    seq_length = 16
    expected_hook_rows = train_iters * global_batch_size * num_moe_layers
    expected_sample_rows = train_iters * global_batch_size
    expected_router_weight_rows = (train_iters + 1) * num_moe_layers
    expected_tensor_rows = 3 * expected_hook_rows + expected_router_weight_rows
    expected_scalar_rows = expected_hook_rows + expected_sample_rows + train_iters
    expected_scalar_int_rows = expected_sample_rows + train_iters

    client.execute(f"CREATE DATABASE IF NOT EXISTS `{database}`")
    client.execute(f"DROP TABLE IF EXISTS `{database}`.`{table}`")
    client.execute(f"DROP TABLE IF EXISTS `{database}`.`{scalar_table}`")
    _create_training_table(client, database=database, table=table)

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{ROOT}:{MEGATRON_ROOT}:{env.get('PYTHONPATH', '')}"
    env["DMI_ENABLE"] = "1"
    env.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", "1")
    if "DMI_REAL_E2E_CUDA_VISIBLE_DEVICES" in env:
        env["CUDA_VISIBLE_DEVICES"] = env["DMI_REAL_E2E_CUDA_VISIBLE_DEVICES"]

    cmd = _tiny_megatron_router_summary_cmd(
        model_id=model_id,
        train_iters=train_iters,
        eval_iters=0,
        micro_batch_size=micro_batch_size,
        global_batch_size=global_batch_size,
        database=database,
        table=table,
        extra_args=[
            "--dmi-hook-selection",
            "router-summary,router-entropy,expert-counts,loss-summary,grad-norm,router-weights",
            "--moe-expert-capacity-factor",
            "0.5",
            "--moe-pad-expert-input-to-capacity",
            "--log-interval",
            "1",
        ],
    )

    try:
        _run_megatron_cmd(cmd, env=env, log_path=log_path)
        log_text = log_path.read_text(encoding="utf-8", errors="replace")
        assert "payload arrived with empty metadata FIFO" not in log_text
        assert "unconsumed metadata entries" not in log_text

        _wait_for_exact_model_rows(
            client,
            database=database,
            table=table,
            model_id=model_id,
            expected=expected_tensor_rows,
        )
        _wait_for_exact_model_rows(
            client,
            database=database,
            table=scalar_table,
            model_id=model_id,
            expected=expected_scalar_rows,
        )
        _wait_for_exact_model_rows(
            client,
            database=database,
            table=f"{table}_scalar_int",
            model_id=model_id,
            expected=expected_scalar_int_rows,
        )
        _wait_for_exact_scalar_rows(
            client,
            database=database,
            table=scalar_table,
            model_id=model_id,
            act_name="router_token_entropy_mean",
            expected=expected_hook_rows,
        )
        _wait_for_exact_scalar_rows(
            client,
            database=database,
            table=scalar_table,
            model_id=model_id,
            act_name="grad_norm",
            direction="iter",
            expected=train_iters,
        )
        _wait_for_exact_scalar_rows(
            client,
            database=database,
            table=f"{table}_scalar_int",
            model_id=model_id,
            act_name="iteration_attempt_status",
            direction="iter",
            expected=train_iters,
        )

        pre_rows = _read_training_act_rows(
            model_id=model_id,
            table=table,
            database=database,
            act_name="pre_drop_token_count",
        )
        post_rows = _read_training_act_rows(
            model_id=model_id,
            table=table,
            database=database,
            act_name="post_drop_token_count",
        )
        assert len(pre_rows) == expected_hook_rows
        assert len(post_rows) == expected_hook_rows

        def coordinate_without_act_name(key):
            return key[:1] + key[2:]

        pre_by_coordinate = {
            coordinate_without_act_name(key): value for key, value in pre_rows
        }
        post_by_coordinate = {
            coordinate_without_act_name(key): value for key, value in post_rows
        }
        assert set(pre_by_coordinate) == set(post_by_coordinate)
        for counts in pre_by_coordinate.values():
            assert counts.dtype == torch.int64
            assert tuple(counts.shape) == (2,)
            assert int(counts.sum().item()) == seq_length
        for coordinate, post_counts in post_by_coordinate.items():
            pre_counts = pre_by_coordinate[coordinate]
            assert post_counts.dtype == torch.int64
            assert tuple(post_counts.shape) == (2,)
            assert torch.all(post_counts >= 0)
            assert torch.all(post_counts <= pre_counts)
        assert sum(int(value.sum().item()) for value in post_by_coordinate.values()) < sum(
            int(value.sum().item()) for value in pre_by_coordinate.values()
        )

        entropy_rows = _read_training_scalar_act_rows(
            model_id=model_id,
            table=table,
            database=database,
            act_name="router_token_entropy_mean",
            direction="fwd",
        )
        assert len(entropy_rows) == expected_hook_rows
        for key, value in entropy_rows:
            assert key[3] == "train"
            assert key[8] in {0, 1}
            assert math.isfinite(float(value))
            assert 0.0 <= float(value) <= math.log(2.0)

        grad_rows = _read_training_scalar_act_rows(
            model_id=model_id,
            table=table,
            database=database,
            act_name="grad_norm",
            direction="iter",
        )
        assert len(grad_rows) == train_iters
        assert {key[4] for key, _value in grad_rows} == {1, 2}
        for key, value in grad_rows:
            assert key[3] == "train"
            assert key[5:10] == (-1, -1, -1, -1, -1)
            assert key[10:12] == (0, 1)
            assert key[12:] == (0, 0, -1)
            assert math.isfinite(float(value))
            assert float(value) >= 0.0

        status_rows = _read_training_scalar_rows_all_phases(
            model_id=model_id,
            table=table,
            database=database,
            scalar_kind="int",
            act_name="iteration_attempt_status",
            direction="iter",
        )
        assert len(status_rows) == train_iters
        assert {key[4] for key, _value in status_rows} == {1, 2}
        for key, value in status_rows:
            assert key[3] == "train"
            assert key[5:10] == (-1, -1, -1, -1, -1)
            assert key[10:12] == (0, 1)
            assert key[12:] == (0, 0, -1)
            assert int(value) == 1

        weight_rows = _read_training_act_rows(
            model_id=model_id,
            table=table,
            database=database,
            act_name="router_projection_weight",
            direction="iter",
        )
        assert len(weight_rows) == expected_router_weight_rows
        assert {key[4] for key, _value in weight_rows} == {0, 1, 2}
        assert {key[8] for key, _value in weight_rows} == {0, 1}
        for key, value in weight_rows:
            assert key[3] == "train"
            assert key[5:8] == (-1, -1, -1)
            assert key[9] == 0
            assert key[10:12] == (0, 1)
            assert key[12:] == (0, 0, -1)
            assert tuple(value.shape) == (2, 64)

        materialize_result = subprocess.run(
            [
                sys.executable,
                "-m",
                "tools.materialization",
                "--raw-db",
                database,
                "--processed-db",
                processed_database,
                "--raw-table",
                table,
                "--model-id",
                model_id,
                "--run-name",
                "real-e2e",
                "--expected-train-iters",
                str(train_iters),
                "--expected-layer-count",
                str(num_moe_layers),
                "--expected-expert-count",
                "2",
                "--expected-hidden-size",
                "64",
                "--expected-samples-per-iteration",
                str(global_batch_size),
                "--materialize-pathways",
                "--pathway-window-size",
                "1",
                "--drop-existing",
            ],
            cwd=ROOT,
            env={**env, "CUDA_VISIBLE_DEVICES": ""},
            text=True,
            capture_output=True,
            check=False,
            timeout=60,
        )
        assert materialize_result.returncode == 0, (
            materialize_result.stdout + materialize_result.stderr
        )
        processed_scalars = client.execute(
            f"""
            SELECT training_iteration_id, metric_name, metric_value
            FROM `{processed_database}`.`training_scalar`
            WHERE model_id = %(model_id)s AND run_name = 'real-e2e'
            ORDER BY training_iteration_id, metric_name
            """,
            {"model_id": model_id},
        )
        assert len(processed_scalars) == 2 * train_iters
        assert {str(row[1]) for row in processed_scalars} == {
            "grad_norm",
            "lm_loss_iteration",
        }
        native_losses = [
            float(value)
            for value in re.findall(r"lm loss:\s*([0-9.Ee+-]+)", log_text)
        ]
        assert len(native_losses) == train_iters
        processed_loss_by_iteration = {
            int(iteration): float(value)
            for iteration, metric_name, value in processed_scalars
            if str(metric_name) == "lm_loss_iteration"
        }
        # Megatron deliberately keeps iteration 1 in the logging accumulator.
        # With log_interval=1, its second printed value is mean(iteration 1, 2).
        native_iteration_losses = [
            native_losses[0],
            2.0 * native_losses[1] - native_losses[0],
            *native_losses[2:],
        ]
        for iteration, native_loss in enumerate(native_iteration_losses, start=1):
            assert processed_loss_by_iteration[iteration] == pytest.approx(
                native_loss,
                abs=5e-4,
            )
        for metric_table in ("pathway_consistency", "pathway_edit_distance"):
            rows = client.execute(
                f"""
                SELECT training_iteration_id, dataset_id, window_size_iterations,
                       effective_window_size_iterations
                FROM `{processed_database}`.`{metric_table}`
                WHERE model_id = %(model_id)s AND run_name = 'real-e2e'
                ORDER BY training_iteration_id
                """,
                {"model_id": model_id},
            )
            assert rows == [(1, 0, 1, 1), (2, 0, 1, 1)]

    finally:
        client.execute(f"DROP TABLE IF EXISTS `{database}`.`{table}`")
        client.execute(f"DROP TABLE IF EXISTS `{database}`.`{scalar_table}`")
        client.execute(f"DROP TABLE IF EXISTS `{database}`.`{table}_scalar_int`")
        client.execute(f"DROP TABLE IF EXISTS `{database}`.`{table}_eval_phase_boundary`")
        client.execute(f"DROP DATABASE IF EXISTS `{processed_database}`")
        client.disconnect()


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Real Megatron PP E2E needs CUDA")
def test_real_megatron_router_weights_cover_pipeline_stages_once(tmp_path):
    """Verify PP stages collectively emit every router weight exactly once per state."""

    if _available_cuda_devices() < 2:
        pytest.skip("router-weight PP E2E requires two CUDA devices")

    client = _clickhouse_client_or_skip()
    database = os.environ.get("DMX_DB_DATABASE", "default")
    table = f"dmi_megatron_router_weight_pp2_e2e_{uuid.uuid4().hex}"
    model_id = f"megatron-router-weight-pp2-e2e-{uuid.uuid4().hex}"
    log_path = tmp_path / "megatron_router_weight_pp2.log"
    train_iters = 1
    global_batch_size = 4
    num_moe_layers = 2
    expected_router_rows = (train_iters + 1) * num_moe_layers
    expected_summary_rows = train_iters * global_batch_size * num_moe_layers

    client.execute(f"CREATE DATABASE IF NOT EXISTS `{database}`")
    client.execute(f"DROP TABLE IF EXISTS `{database}`.`{table}`")
    _create_training_table(client, database=database, table=table)

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{ROOT}:{MEGATRON_ROOT}:{env.get('PYTHONPATH', '')}"
    env["DMI_ENABLE"] = "1"
    env.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", "1")
    if "DMI_REAL_E2E_CUDA_VISIBLE_DEVICES" in env:
        env["CUDA_VISIBLE_DEVICES"] = env["DMI_REAL_E2E_CUDA_VISIBLE_DEVICES"]

    cmd = _tiny_megatron_router_summary_cmd(
        model_id=model_id,
        train_iters=train_iters,
        micro_batch_size=2,
        global_batch_size=global_batch_size,
        nproc_per_node=2,
        pp_size=2,
        database=database,
        table=table,
        extra_args=["--dmi-hook-selection", "router-summary,router-weights"],
    )

    try:
        _run_megatron_cmd(cmd, env=env, log_path=log_path)
        log_text = log_path.read_text(encoding="utf-8", errors="replace")
        assert "payload arrived with empty metadata FIFO" not in log_text
        assert "unconsumed metadata entries" not in log_text

        _wait_for_exact_model_rows(
            client,
            database=database,
            table=table,
            model_id=model_id,
            expected=expected_router_rows + expected_summary_rows,
        )
        weight_rows = _read_training_act_rows(
            model_id=model_id,
            table=table,
            database=database,
            act_name="router_projection_weight",
            direction="iter",
        )
        assert len(weight_rows) == expected_router_rows
        assert {(key[4], key[8]) for key, _value in weight_rows} == {
            (0, 0),
            (0, 1),
            (1, 0),
            (1, 1),
        }
        assert len({key for key, _value in weight_rows}) == expected_router_rows
        for key, value in weight_rows:
            assert key[3] == "train"
            assert key[5:8] == (-1, -1, -1)
            assert key[9:12] == (0, 0, 1)
            assert tuple(value.shape) == (2, 64)
    finally:
        client.execute(f"DROP TABLE IF EXISTS `{database}`.`{table}`")
        client.execute(f"DROP TABLE IF EXISTS `{database}`.`{table}_scalar_float`")
        client.execute(f"DROP TABLE IF EXISTS `{database}`.`{table}_scalar_int`")
        client.execute(f"DROP TABLE IF EXISTS `{database}`.`{table}_eval_phase_boundary`")
        client.disconnect()


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Real Megatron rerun E2E needs CUDA")
def test_real_megatron_pp2_rerun_reuses_logical_training_ids(tmp_path):
    """Verify in-process reruns duplicate coordinates without shifting later IDs."""

    if _available_cuda_devices() < 2:
        pytest.skip("rerun PP E2E requires two CUDA devices")

    client = _clickhouse_client_or_skip()
    database = os.environ.get("DMX_DB_DATABASE", "default")
    table = f"dmi_megatron_rerun_pp2_e2e_{uuid.uuid4().hex}"
    model_id = f"megatron-rerun-pp2-e2e-{uuid.uuid4().hex}"
    log_path = tmp_path / "megatron_rerun_pp2.log"
    train_iters = 3
    global_batch_size = 4
    num_moe_layers = 2
    attempts_by_iteration = {1: 1, 2: 2, 3: 2}
    rows_per_attempt = global_batch_size * num_moe_layers
    expected_rows = sum(attempts_by_iteration.values()) * rows_per_attempt

    client.execute(f"CREATE DATABASE IF NOT EXISTS `{database}`")
    client.execute(f"DROP TABLE IF EXISTS `{database}`.`{table}`")
    _create_training_table(client, database=database, table=table)

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{ROOT}:{MEGATRON_ROOT}:{env.get('PYTHONPATH', '')}"
    env["DMI_ENABLE"] = "1"
    env.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", "1")
    if "DMI_REAL_E2E_CUDA_VISIBLE_DEVICES" in env:
        env["CUDA_VISIBLE_DEVICES"] = env["DMI_REAL_E2E_CUDA_VISIBLE_DEVICES"]

    cmd = _tiny_megatron_router_summary_cmd(
        model_id=model_id,
        train_iters=train_iters,
        micro_batch_size=2,
        global_batch_size=global_batch_size,
        nproc_per_node=2,
        pp_size=2,
        database=database,
        table=table,
        extra_args=[
            "--dmi-hook-selection",
            "router-summary",
            "--rerun-mode",
            "validate_results",
            "--error-injection-rate",
            "1",
            "--error-injection-type",
            "transient_error",
            "--no-check-for-nan-in-loss-and-grad",
            "--check-for-spiky-loss",
        ],
    )

    try:
        _run_megatron_cmd(cmd, env=env, log_path=log_path)
        log_text = log_path.read_text(encoding="utf-8", errors="replace")
        assert "payload arrived with empty metadata FIFO" not in log_text
        assert "unconsumed metadata entries" not in log_text
        _wait_for_exact_model_rows(
            client,
            database=database,
            table=table,
            model_id=model_id,
            expected=expected_rows,
        )
        raw_rows = _read_training_act_rows(
            model_id=model_id,
            table=table,
            database=database,
            act_name="router_probs_mean",
            direction="fwd",
            raw=True,
        )
        accepted_rows = _read_training_act_rows(
            model_id=model_id,
            table=table,
            database=database,
            act_name="router_probs_mean",
            direction="fwd",
        )
        assert len(raw_rows) == expected_rows
        assert len(accepted_rows) == train_iters * rows_per_attempt
        assert {key[4] for key, _value in raw_rows} == {1, 2, 3}
        assert {key[4] for key, _value in accepted_rows} == {1, 2, 3}
        for global_batch_id, attempt_count in attempts_by_iteration.items():
            iteration_rows = [
                key for key, _value in raw_rows if key[4] == global_batch_id
            ]
            assert len(iteration_rows) == attempt_count * rows_per_attempt
            assert {key[12] for key in iteration_rows} == set(range(attempt_count))
            assert {key[13:] for key in iteration_rows} == {(0, 0)}

        raw_status_rows = _read_training_scalar_rows_all_phases(
            model_id=model_id,
            table=table,
            database=database,
            scalar_kind="int",
            act_name="iteration_attempt_status",
            direction="iter",
            raw=True,
        )
        assert len(raw_status_rows) == sum(attempts_by_iteration.values())
        for key, value in raw_status_rows:
            global_batch_id = int(key[4])
            attempt_id = int(key[12])
            assert key[5:12] == (-1, -1, -1, -1, -1, 0, 1)
            assert key[13:] == (0, -1)
            expected_status = (
                1 if attempt_id == attempts_by_iteration[global_batch_id] - 1 else 0
            )
            assert int(value) == expected_status

        reader = MegatronTrainingReader(
            host=os.environ.get("DMX_DB_HOST", "localhost"),
            port=int(os.environ.get("DMX_DB_PORT", "9000")),
            username=os.environ.get("DMX_DB_USER", "default"),
            password=os.environ.get("DMX_DB_PASSWORD", ""),
            database=database,
            table=table,
        )
        try:
            assert reader.training_accepted_attempts(model_id) == {
                global_batch_id: attempt_count - 1
                for global_batch_id, attempt_count in attempts_by_iteration.items()
            }
        finally:
            reader.close()
    finally:
        client.execute(f"DROP TABLE IF EXISTS `{database}`.`{table}`")
        client.execute(f"DROP TABLE IF EXISTS `{database}`.`{table}_scalar_float`")
        client.execute(f"DROP TABLE IF EXISTS `{database}`.`{table}_scalar_int`")
        client.execute(f"DROP TABLE IF EXISTS `{database}`.`{table}_eval_phase_boundary`")
        client.disconnect()


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Real Megatron recompute E2E needs CUDA")
@pytest.mark.parametrize(
    ("case_name", "train_iters", "moe_args", "execution_args"),
    [
        pytest.param(
            "eager_full_recompute",
            2,
            ["--moe-token-dispatcher-type", "allgather"],
            [
                "--recompute-granularity",
                "full",
                "--recompute-method",
                "uniform",
                "--recompute-num-layers",
                "1",
                "--attention-dropout",
                "0",
                "--hidden-dropout",
                "0",
            ],
            id="eager",
        ),
        pytest.param(
            "local_cuda_graph_selective_recompute",
            3,
            ["--moe-token-dispatcher-type", "allgather"],
            [
                "--cuda-graph-impl",
                "local",
                "--cuda-graph-scope",
                "moe_router",
                "--spec",
                "dmi_megatron_integration._test_specs",
                "dmi_test_moe_layer_spec",
                "--recompute-granularity",
                "selective",
                "--recompute-modules",
                "moe",
                "--attention-dropout",
                "0",
                "--hidden-dropout",
                "0",
            ],
            id="local_cuda_graph",
        ),
        pytest.param(
            "full_iteration_cuda_graph_full_recompute",
            3,
            [
                "--moe-token-dispatcher-type",
                "alltoall",
                "--moe-expert-capacity-factor",
                "1.0",
                "--moe-pad-expert-input-to-capacity",
            ],
            [
                "--cuda-graph-impl",
                "local",
                "--cuda-graph-scope",
                "full_iteration",
                "--no-check-for-nan-in-loss-and-grad",
                "--recompute-granularity",
                "full",
                "--recompute-method",
                "uniform",
                "--recompute-num-layers",
                "1",
                "--attention-dropout",
                "0",
                "--hidden-dropout",
                "0",
            ],
            id="full_iteration_cuda_graph",
        ),
    ],
)
def test_real_megatron_retained_recompute_invocation_identity(
    tmp_path,
    case_name,
    train_iters,
    moe_args,
    execution_args,
):
    """Retained recompute rows remain semantic-forward and uniquely indexed."""

    if _available_cuda_devices() < 1:
        pytest.skip("retained-recompute E2E requires one CUDA device")

    client = _clickhouse_client_or_skip()
    database = os.environ.get("DMX_DB_DATABASE", "default")
    table = f"dmi_megatron_recompute_{case_name}_{uuid.uuid4().hex}"
    model_id = f"megatron-recompute-{case_name}-{uuid.uuid4().hex}"
    log_path = tmp_path / f"megatron_recompute_{case_name}.log"
    micro_batch_size = 2
    global_batch_size = 2
    num_moe_layers = 2
    expected_primary_rows = train_iters * global_batch_size * num_moe_layers
    expected_all_invocation_rows = 2 * expected_primary_rows

    client.execute(f"CREATE DATABASE IF NOT EXISTS `{database}`")
    client.execute(f"DROP TABLE IF EXISTS `{database}`.`{table}`")
    _create_training_table(client, database=database, table=table)

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{ROOT}:{MEGATRON_ROOT}:{env.get('PYTHONPATH', '')}"
    env["DMI_ENABLE"] = "1"
    env.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", "1")
    if "DMI_REAL_E2E_CUDA_VISIBLE_DEVICES" in env:
        env["CUDA_VISIBLE_DEVICES"] = env["DMI_REAL_E2E_CUDA_VISIBLE_DEVICES"]

    cmd = _tiny_megatron_router_summary_cmd(
        model_id=model_id,
        train_iters=train_iters,
        eval_iters=0,
        micro_batch_size=micro_batch_size,
        global_batch_size=global_batch_size,
        database=database,
        table=table,
        extra_args=[
            "--dmi-hook-selection",
            "router-summary",
            "--dmi-recompute-hook",
            "router-summary",
            *execution_args,
        ],
    )
    cmd += moe_args

    try:
        _run_megatron_cmd(cmd, env=env, log_path=log_path)
        log_text = log_path.read_text(encoding="utf-8", errors="replace")
        assert "payload arrived with empty metadata FIFO" not in log_text
        assert "unconsumed metadata entries" not in log_text
        _wait_for_exact_model_rows(
            client,
            database=database,
            table=table,
            model_id=model_id,
            expected=expected_all_invocation_rows,
        )
        _wait_for_exact_scalar_rows(
            client,
            database=database,
            table=f"{table}_scalar_int",
            model_id=model_id,
            act_name="iteration_attempt_status",
            direction="iter",
            expected=train_iters,
        )

        raw_rows = _read_training_act_rows(
            model_id=model_id,
            table=table,
            database=database,
            act_name="router_probs_mean",
            raw=True,
        )
        primary_rows = _read_training_act_rows(
            model_id=model_id,
            table=table,
            database=database,
            act_name="router_probs_mean",
        )
        assert len(raw_rows) == expected_all_invocation_rows
        assert len(primary_rows) == expected_primary_rows
        assert {key[2] for key, _value in raw_rows} == {"fwd"}
        assert {key[3] for key, _value in raw_rows} == {"train"}
        assert {key[12] for key, _value in raw_rows} == {0}
        assert {key[13] for key, _value in raw_rows} == {0, 1}
        assert {key[14] for key, _value in raw_rows} == {0}
        assert {key[13] for key, _value in primary_rows} == {0}

        emissions: dict[tuple[object, ...], dict[int, dict[int, torch.Tensor]]] = {}
        for key, value in raw_rows:
            producer_key = (
                key[4],
                key[5],
                key[6],
                key[8],
                key[9],
                key[10],
                key[11],
                key[12],
            )
            by_invocation = emissions.setdefault(producer_key, {})
            by_sample = by_invocation.setdefault(int(key[13]), {})
            assert int(key[7]) not in by_sample
            by_sample[int(key[7])] = value

        assert len(emissions) == train_iters * num_moe_layers
        for by_invocation in emissions.values():
            assert set(by_invocation) == {0, 1}
            assert set(by_invocation[0]) == {0, 1}
            assert set(by_invocation[1]) == {0, 1}
            for sample_index in (0, 1):
                torch.testing.assert_close(
                    by_invocation[0][sample_index],
                    by_invocation[1][sample_index],
                    rtol=0,
                    atol=0,
                )
    finally:
        client.execute(f"DROP TABLE IF EXISTS `{database}`.`{table}`")
        client.execute(f"DROP TABLE IF EXISTS `{database}`.`{table}_scalar_float`")
        client.execute(f"DROP TABLE IF EXISTS `{database}`.`{table}_scalar_int`")
        client.execute(f"DROP TABLE IF EXISTS `{database}`.`{table}_eval_phase_boundary`")
        client.disconnect()


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Real Megatron blend E2E needs CUDA")
def test_real_megatron_pp2_mixed_dataset_provenance(tmp_path):
    """Propagate exact per-sample source IDs from a real two-dataset blend."""

    if _available_cuda_devices() < 2:
        pytest.skip("mixed-dataset PP=2 E2E requires two CUDA devices")

    client = _clickhouse_client_or_skip()
    database = os.environ.get("DMX_DB_DATABASE", "default")
    table = f"dmi_megatron_dataset_provenance_{uuid.uuid4().hex}"
    model_id = f"megatron-dataset-provenance-{uuid.uuid4().hex}"
    log_path = tmp_path / "megatron_dataset_provenance.log"
    first_prefix = tmp_path / "source_0"
    second_prefix = tmp_path / "source_1"
    _build_tiny_indexed_gpt_dataset(first_prefix, token_offset=0)
    _build_tiny_indexed_gpt_dataset(second_prefix, token_offset=37)

    train_iters = 3
    micro_batch_size = 2
    global_batch_size = 4
    num_layers = 2
    expected_rows = train_iters * global_batch_size * num_layers

    client.execute(f"CREATE DATABASE IF NOT EXISTS `{database}`")
    client.execute(f"DROP TABLE IF EXISTS `{database}`.`{table}`")
    _create_training_table(client, database=database, table=table)

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{ROOT}:{MEGATRON_ROOT}:{env.get('PYTHONPATH', '')}"
    env["DMI_ENABLE"] = "1"
    env.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", "1")
    if "DMI_REAL_E2E_CUDA_VISIBLE_DEVICES" in env:
        env["CUDA_VISIBLE_DEVICES"] = env["DMI_REAL_E2E_CUDA_VISIBLE_DEVICES"]

    cmd = _tiny_megatron_router_summary_cmd(
        model_id=model_id,
        train_iters=train_iters,
        micro_batch_size=micro_batch_size,
        global_batch_size=global_batch_size,
        nproc_per_node=2,
        pp_size=2,
        database=database,
        table=table,
        extra_args=["--dmi-hook-selection", "router-summary"],
    )
    cmd.remove("--mock-data")
    cmd += [
        "--data-path",
        "0.5",
        str(first_prefix),
        "0.5",
        str(second_prefix),
        "--split",
        "100,0,0",
    ]

    try:
        _run_megatron_cmd(cmd, env=env, log_path=log_path)
        log_text = log_path.read_text(encoding="utf-8", errors="replace")
        assert "payload arrived with empty metadata FIFO" not in log_text
        assert "unconsumed metadata entries" not in log_text
        _wait_for_exact_model_rows(
            client,
            database=database,
            table=table,
            model_id=model_id,
            expected=expected_rows,
        )

        rows = _read_training_act_rows(
            model_id=model_id,
            table=table,
            database=database,
            act_name="router_probs_mean",
            raw=True,
        )
        assert len(rows) == expected_rows
        observed: dict[tuple[int, int, int, int], int] = {}
        for key, _value in rows:
            coordinate = (int(key[4]), int(key[6]), int(key[8]), int(key[7]))
            assert coordinate not in observed
            observed[coordinate] = int(key[14])
            assert key[12:14] == (0, 0)

        expected = {
            (global_batch_id, microbatch_id, layer_no, sample_index): sample_index
            for global_batch_id in range(1, train_iters + 1)
            for microbatch_id in range(global_batch_size // micro_batch_size)
            for layer_no in range(num_layers)
            for sample_index in range(micro_batch_size)
        }
        assert observed == expected
    finally:
        client.execute(f"DROP TABLE IF EXISTS `{database}`.`{table}`")
        client.execute(f"DROP TABLE IF EXISTS `{database}`.`{table}_scalar_float`")
        client.execute(f"DROP TABLE IF EXISTS `{database}`.`{table}_scalar_int`")
        client.execute(f"DROP TABLE IF EXISTS `{database}`.`{table}_eval_phase_boundary`")
        client.disconnect()


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Combined identity E2E needs CUDA")
def test_real_megatron_combined_attempt_invocation_and_dataset_identity(tmp_path):
    """Compose rerun, retained recompute, and dataset provenance identities."""

    if _available_cuda_devices() < 2:
        pytest.skip("combined identity PP=2 E2E requires two CUDA devices")

    client = _clickhouse_client_or_skip()
    database = os.environ.get("DMX_DB_DATABASE", "default")
    table = f"dmi_megatron_combined_identity_{uuid.uuid4().hex}"
    model_id = f"megatron-combined-identity-{uuid.uuid4().hex}"
    log_path = tmp_path / "megatron_combined_identity.log"
    first_prefix = tmp_path / "combined_source_0"
    second_prefix = tmp_path / "combined_source_1"
    _build_tiny_indexed_gpt_dataset(first_prefix, token_offset=0)
    _build_tiny_indexed_gpt_dataset(second_prefix, token_offset=53)

    train_iters = 3
    micro_batch_size = 2
    global_batch_size = 4
    num_layers = 2
    attempts_by_iteration = {1: 1, 2: 2, 3: 2}
    rows_per_invocation = global_batch_size * num_layers
    expected_raw_rows = 2 * sum(attempts_by_iteration.values()) * rows_per_invocation
    expected_primary_rows = train_iters * rows_per_invocation

    client.execute(f"CREATE DATABASE IF NOT EXISTS `{database}`")
    client.execute(f"DROP TABLE IF EXISTS `{database}`.`{table}`")
    _create_training_table(client, database=database, table=table)

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{ROOT}:{MEGATRON_ROOT}:{env.get('PYTHONPATH', '')}"
    env["DMI_ENABLE"] = "1"
    env.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", "1")
    if "DMI_REAL_E2E_CUDA_VISIBLE_DEVICES" in env:
        env["CUDA_VISIBLE_DEVICES"] = env["DMI_REAL_E2E_CUDA_VISIBLE_DEVICES"]

    cmd = _tiny_megatron_router_summary_cmd(
        model_id=model_id,
        train_iters=train_iters,
        micro_batch_size=micro_batch_size,
        global_batch_size=global_batch_size,
        nproc_per_node=2,
        pp_size=2,
        database=database,
        table=table,
        extra_args=[
            "--dmi-hook-selection",
            "router-summary",
            "--dmi-recompute-hook",
            "router-summary",
            "--recompute-granularity",
            "full",
            "--recompute-method",
            "uniform",
            "--recompute-num-layers",
            "1",
            "--attention-dropout",
            "0",
            "--hidden-dropout",
            "0",
            "--rerun-mode",
            "validate_results",
            "--error-injection-rate",
            "1",
            "--error-injection-type",
            "transient_error",
            "--no-check-for-nan-in-loss-and-grad",
            "--check-for-spiky-loss",
        ],
    )
    cmd.remove("--mock-data")
    cmd += [
        "--data-path",
        "0.5",
        str(first_prefix),
        "0.5",
        str(second_prefix),
        "--split",
        "100,0,0",
    ]

    try:
        _run_megatron_cmd(cmd, env=env, log_path=log_path)
        log_text = log_path.read_text(encoding="utf-8", errors="replace")
        assert "payload arrived with empty metadata FIFO" not in log_text
        assert "unconsumed metadata entries" not in log_text
        _wait_for_exact_model_rows(
            client,
            database=database,
            table=table,
            model_id=model_id,
            expected=expected_raw_rows,
        )

        raw_rows = _read_training_act_rows(
            model_id=model_id,
            table=table,
            database=database,
            act_name="router_probs_mean",
            raw=True,
        )
        primary_rows = _read_training_act_rows(
            model_id=model_id,
            table=table,
            database=database,
            act_name="router_probs_mean",
        )
        assert len(raw_rows) == expected_raw_rows
        assert len(primary_rows) == expected_primary_rows

        observed = {
            (
                int(key[4]),
                int(key[12]),
                int(key[13]),
                int(key[6]),
                int(key[8]),
                int(key[7]),
                int(key[14]),
            )
            for key, _value in raw_rows
        }
        expected = {
            (
                global_batch_id,
                attempt_id,
                invocation_id,
                microbatch_id,
                layer_no,
                sample_index,
                sample_index,
            )
            for global_batch_id, attempt_count in attempts_by_iteration.items()
            for attempt_id in range(attempt_count)
            for invocation_id in (0, 1)
            for microbatch_id in range(global_batch_size // micro_batch_size)
            for layer_no in range(num_layers)
            for sample_index in range(micro_batch_size)
        }
        assert observed == expected

        accepted_coordinates = {
            (int(key[4]), int(key[12]), int(key[13]), int(key[14]))
            for key, _value in primary_rows
        }
        assert accepted_coordinates == {
            (global_batch_id, attempt_count - 1, 0, dataset_id)
            for global_batch_id, attempt_count in attempts_by_iteration.items()
            for dataset_id in (0, 1)
        }

        status_rows = _read_training_scalar_rows_all_phases(
            model_id=model_id,
            table=table,
            database=database,
            scalar_kind="int",
            act_name="iteration_attempt_status",
            direction="iter",
            raw=True,
        )
        assert len(status_rows) == sum(attempts_by_iteration.values())
        assert {
            (int(key[4]), int(key[12]), int(value)) for key, value in status_rows
        } == {
            (
                global_batch_id,
                attempt_id,
                1 if attempt_id == attempt_count - 1 else 0,
            )
            for global_batch_id, attempt_count in attempts_by_iteration.items()
            for attempt_id in range(attempt_count)
        }
    finally:
        client.execute(f"DROP TABLE IF EXISTS `{database}`.`{table}`")
        client.execute(f"DROP TABLE IF EXISTS `{database}`.`{table}_scalar_float`")
        client.execute(f"DROP TABLE IF EXISTS `{database}`.`{table}_scalar_int`")
        client.execute(f"DROP TABLE IF EXISTS `{database}`.`{table}_eval_phase_boundary`")
        client.disconnect()


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Real Megatron DP E2E needs CUDA")
def test_real_megatron_dp2_emits_one_global_grad_norm(tmp_path):
    """Verify DP_RANK_0 emission stores one semantic global grad row at DP=2."""

    if _available_cuda_devices() < 2:
        pytest.skip("grad-norm DP E2E requires two CUDA devices")

    client = _clickhouse_client_or_skip()
    database = os.environ.get("DMX_DB_DATABASE", "default")
    table = f"dmi_megatron_grad_dp2_e2e_{uuid.uuid4().hex}"
    model_id = f"megatron-grad-dp2-e2e-{uuid.uuid4().hex}"
    log_path = tmp_path / "megatron_grad_dp2.log"

    client.execute(f"CREATE DATABASE IF NOT EXISTS `{database}`")
    client.execute(f"DROP TABLE IF EXISTS `{database}`.`{table}`")
    _create_training_table(client, database=database, table=table)

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{ROOT}:{MEGATRON_ROOT}:{env.get('PYTHONPATH', '')}"
    env["DMI_ENABLE"] = "1"
    env.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", "1")
    if "DMI_REAL_E2E_CUDA_VISIBLE_DEVICES" in env:
        env["CUDA_VISIBLE_DEVICES"] = env["DMI_REAL_E2E_CUDA_VISIBLE_DEVICES"]

    cmd = _tiny_megatron_router_summary_cmd(
        model_id=model_id,
        train_iters=1,
        micro_batch_size=1,
        global_batch_size=2,
        nproc_per_node=2,
        database=database,
        table=table,
        extra_args=["--dmi-hook-selection", "grad-norm"],
    )

    try:
        _run_megatron_cmd(cmd, env=env, log_path=log_path)
        _wait_for_exact_scalar_rows(
            client,
            database=database,
            table=f"{table}_scalar_float",
            model_id=model_id,
            act_name="grad_norm",
            direction="iter",
            expected=1,
        )
        rows = _read_training_scalar_act_rows(
            model_id=model_id,
            table=table,
            database=database,
            act_name="grad_norm",
            direction="iter",
        )
        assert len(rows) == 1
        key, value = rows[0]
        assert key[4:12] == (1, -1, -1, -1, -1, -1, 0, 1)
        assert math.isfinite(float(value))
    finally:
        client.execute(f"DROP TABLE IF EXISTS `{database}`.`{table}`")
        client.execute(f"DROP TABLE IF EXISTS `{database}`.`{table}_scalar_float`")
        client.execute(f"DROP TABLE IF EXISTS `{database}`.`{table}_scalar_int`")
        client.execute(f"DROP TABLE IF EXISTS `{database}`.`{table}_eval_phase_boundary`")
        client.disconnect()


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Real Megatron EP E2E needs CUDA")
def test_real_megatron_ep2_moe_payloads_reach_clickhouse(tmp_path):
    """Verify router and packed MoE payloads from both EP ranks reach ClickHouse."""

    if _available_cuda_devices() < 2:
        pytest.skip("MoE payload EP=2 E2E requires two CUDA devices")

    client = _clickhouse_client_or_skip()
    database = os.environ.get("DMX_DB_DATABASE", "default")
    table = f"dmi_megatron_moe_payload_ep2_e2e_{uuid.uuid4().hex}"
    model_id = f"megatron-moe-payload-ep2-e2e-{uuid.uuid4().hex}"
    log_path = tmp_path / "megatron_moe_payload_ep2.log"
    act_names = (
        "router_topk_expert_ids",
        "router_topk_weights",
        "moe_inverse_map",
        "moe_packed_weighted_output",
    )

    client.execute(f"CREATE DATABASE IF NOT EXISTS `{database}`")
    client.execute(f"DROP TABLE IF EXISTS `{database}`.`{table}`")
    _create_training_table(client, database=database, table=table)

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{ROOT}:{MEGATRON_ROOT}:{env.get('PYTHONPATH', '')}"
    env["DMI_ENABLE"] = "1"
    env["DMI_TOPOLOGY_MANIFEST_PATH"] = str(tmp_path / "topology.json")
    env.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", "1")
    if "DMI_REAL_E2E_CUDA_VISIBLE_DEVICES" in env:
        env["CUDA_VISIBLE_DEVICES"] = env["DMI_REAL_E2E_CUDA_VISIBLE_DEVICES"]

    cmd = _tiny_megatron_router_summary_cmd(
        model_id=model_id,
        train_iters=1,
        micro_batch_size=1,
        global_batch_size=2,
        nproc_per_node=2,
        ep_size=2,
        num_experts=4,
        moe_router_topk=2,
        moe_token_dispatcher_type="alltoall",
        database=database,
        table=table,
        extra_args=[
            "--dmi-hook-selection",
            "router-topk,moe-inverse-map,moe-packed-weighted-output",
        ],
    )

    try:
        _run_megatron_cmd(cmd, env=env, log_path=log_path)
        for act_name in act_names:
            _wait_for_exact_act_rows(
                client,
                database=database,
                table=table,
                model_id=model_id,
                act_name=act_name,
                expected=4,
            )

        rows = _query_moe_hook_rows(
            client,
            database=database,
            table=table,
            model_id=model_id,
            act_names=act_names,
        )
        assert len(rows) == 16
        by_act = {act_name: [] for act_name in act_names}
        for row in rows:
            by_act[row[0].decode("utf-8")].append(row)
            assert row[1].decode("utf-8") == "fwd"
            assert row[2].decode("utf-8") == "train"

        expected_layer_shards = {
            (layer_no, shard_rank)
            for layer_no in (0, 1)
            for shard_rank in (0, 1)
        }
        for act_name in act_names:
            assert {
                (int(row[7]), int(row[8])) for row in by_act[act_name]
            } == expected_layer_shards

        dtype_map = {
            "torch.long": torch.int64,
            "torch.bfloat16": torch.bfloat16,
            "torch.float": torch.float32,
        }

        def decode(row):
            dtype_name = row[14].decode("utf-8")
            assert dtype_name in dtype_map
            dtype = dtype_map[dtype_name]
            shape = tuple(int(value) for value in row[15])
            payload = bytearray(row[16])
            expected_bytes = math.prod(shape) * torch.empty((), dtype=dtype).element_size()
            assert len(payload) == expected_bytes
            return torch.frombuffer(payload, dtype=dtype).clone().reshape(shape)

        for row in by_act["router_topk_expert_ids"]:
            assert row[14].decode("utf-8") == "torch.long"
            expert_ids = decode(row)
            assert tuple(expert_ids.shape) == (16, 2)
            assert int(expert_ids.min()) >= 0
            assert int(expert_ids.max()) < 4

        for row in by_act["router_topk_weights"]:
            assert row[14].decode("utf-8") in {"torch.bfloat16", "torch.float"}
            weights = decode(row)
            assert tuple(weights.shape) == (16, 2)
            assert bool(torch.isfinite(weights).all())

        for row in by_act["moe_inverse_map"]:
            assert tuple(int(row[index]) for index in (6, 9, 10, 13)) == (-1, -1, -1, -1)
            assert row[14].decode("utf-8") == "torch.long"
            inverse_map = decode(row)
            assert tuple(inverse_map.shape) == (32,)
            assert torch.equal(
                inverse_map.sort().values,
                torch.arange(16, dtype=torch.int64).repeat_interleave(2),
            )

        weighted_rows_by_layer = {0: [], 1: []}
        for row in by_act["moe_packed_weighted_output"]:
            assert tuple(int(row[index]) for index in (6, 9, 10, 13)) == (-1, -1, -1, -1)
            assert row[14].decode("utf-8") == "torch.bfloat16"
            weighted_output = decode(row)
            assert weighted_output.dim() == 2
            assert int(weighted_output.shape[1]) == 64
            assert bool(torch.isfinite(weighted_output).all())
            weighted_rows_by_layer[int(row[7])].append(int(weighted_output.shape[0]))
        for layer_rows in weighted_rows_by_layer.values():
            assert sum(layer_rows) == 64
    finally:
        client.execute(f"DROP TABLE IF EXISTS `{database}`.`{table}`")
        client.execute(f"DROP TABLE IF EXISTS `{database}`.`{table}_scalar_float`")
        client.execute(f"DROP TABLE IF EXISTS `{database}`.`{table}_scalar_int`")
        client.execute(f"DROP TABLE IF EXISTS `{database}`.`{table}_eval_phase_boundary`")
        client.disconnect()


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Real Megatron DP E2E needs CUDA")
def test_real_megatron_dp2_rejects_router_weights_before_rows(tmp_path):
    """Verify the DP>1 router-weight guard fails before any DMI payload is emitted."""

    if _available_cuda_devices() < 2:
        pytest.skip("router-weight DP rejection E2E requires two CUDA devices")

    client = _clickhouse_client_or_skip()
    database = os.environ.get("DMX_DB_DATABASE", "default")
    table = f"dmi_megatron_router_dp2_reject_e2e_{uuid.uuid4().hex}"
    model_id = f"megatron-router-dp2-reject-e2e-{uuid.uuid4().hex}"
    log_path = tmp_path / "megatron_router_dp2_reject.log"

    client.execute(f"CREATE DATABASE IF NOT EXISTS `{database}`")
    client.execute(f"DROP TABLE IF EXISTS `{database}`.`{table}`")
    _create_training_table(client, database=database, table=table)

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{ROOT}:{MEGATRON_ROOT}:{env.get('PYTHONPATH', '')}"
    env["DMI_ENABLE"] = "1"
    env.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", "1")
    if "DMI_REAL_E2E_CUDA_VISIBLE_DEVICES" in env:
        env["CUDA_VISIBLE_DEVICES"] = env["DMI_REAL_E2E_CUDA_VISIBLE_DEVICES"]

    cmd = _tiny_megatron_router_summary_cmd(
        model_id=model_id,
        train_iters=1,
        micro_batch_size=1,
        global_batch_size=2,
        nproc_per_node=2,
        database=database,
        table=table,
        extra_args=["--dmi-hook-selection", "router-weights"],
    )

    try:
        with log_path.open("w", encoding="utf-8") as log:
            result = subprocess.run(
                cmd,
                cwd=MEGATRON_ROOT,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=float(os.environ.get("DMI_REAL_E2E_TIMEOUT_S", "240")),
                check=False,
            )
        assert result.returncode != 0
        log_text = log_path.read_text(encoding="utf-8", errors="replace")
        assert "router-weights requires data-parallel world size exactly 1" in log_text
        assert _query_count(
            client,
            database=database,
            table=table,
            model_id=model_id,
            act_name=None,
        ) == 0
    finally:
        client.execute(f"DROP TABLE IF EXISTS `{database}`.`{table}`")
        client.execute(f"DROP TABLE IF EXISTS `{database}`.`{table}_scalar_float`")
        client.execute(f"DROP TABLE IF EXISTS `{database}`.`{table}_scalar_int`")
        client.execute(f"DROP TABLE IF EXISTS `{database}`.`{table}_eval_phase_boundary`")
        client.disconnect()


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Real Megatron optimizer E2E needs CUDA")
@pytest.mark.parametrize(
    ("optimizer_mode", "train_iters", "optimizer_args"),
    [
        pytest.param(
            "distributed",
            2,
            ["--use-distributed-optimizer"],
            id="distributed",
        ),
        pytest.param(
            "cpu_offload",
            2,
            [
                "--use-distributed-optimizer",
                "--use-precision-aware-optimizer",
                "--optimizer-cpu-offload",
                "--use-torch-optimizer-for-cpu-offload",
                "--main-grads-dtype",
                "bf16",
                "--main-params-dtype",
                "fp16",
                "--exp-avg-dtype",
                "bf16",
                "--exp-avg-sq-dtype",
                "bf16",
            ],
            id="cpu_offload",
        ),
        pytest.param(
            "optimizer_cuda_graph",
            3,
            ["--optimizer-cuda-graph", "--cuda-graph-warmup-steps", "1"],
            id="optimizer_cuda_graph",
        ),
    ],
)
def test_real_megatron_iteration_hooks_optimizer_modes(
    tmp_path,
    optimizer_mode,
    train_iters,
    optimizer_args,
):
    """Verify iteration hooks remain outside supported optimizer implementations."""

    if _available_cuda_devices() < 1:
        pytest.skip("optimizer-mode E2E requires one CUDA device")

    client = _clickhouse_client_or_skip()
    database = os.environ.get("DMX_DB_DATABASE", "default")
    table = f"dmi_megatron_{optimizer_mode}_e2e_{uuid.uuid4().hex}"
    model_id = f"megatron-{optimizer_mode}-e2e-{uuid.uuid4().hex}"
    log_path = tmp_path / f"megatron_{optimizer_mode}.log"
    num_moe_layers = 2
    expected_weight_rows = (train_iters + 1) * num_moe_layers

    client.execute(f"CREATE DATABASE IF NOT EXISTS `{database}`")
    client.execute(f"DROP TABLE IF EXISTS `{database}`.`{table}`")
    _create_training_table(client, database=database, table=table)

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{ROOT}:{MEGATRON_ROOT}:{env.get('PYTHONPATH', '')}"
    env["DMI_ENABLE"] = "1"
    env.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", "1")
    if "DMI_REAL_E2E_CUDA_VISIBLE_DEVICES" in env:
        env["CUDA_VISIBLE_DEVICES"] = env["DMI_REAL_E2E_CUDA_VISIBLE_DEVICES"]

    cmd = _tiny_megatron_router_summary_cmd(
        model_id=model_id,
        train_iters=train_iters,
        micro_batch_size=1,
        global_batch_size=1,
        database=database,
        table=table,
        extra_args=[
            "--dmi-hook-selection",
            "grad-norm,router-weights",
            *optimizer_args,
        ],
    )

    try:
        _run_megatron_cmd(cmd, env=env, log_path=log_path)
        log_text = log_path.read_text(encoding="utf-8", errors="replace")
        assert "payload arrived with empty metadata FIFO" not in log_text
        assert "unconsumed metadata entries" not in log_text
        _wait_for_exact_model_rows(
            client,
            database=database,
            table=table,
            model_id=model_id,
            expected=expected_weight_rows,
        )
        _wait_for_exact_scalar_rows(
            client,
            database=database,
            table=f"{table}_scalar_float",
            model_id=model_id,
            act_name="grad_norm",
            direction="iter",
            expected=train_iters,
        )
        weight_rows = _read_training_act_rows(
            model_id=model_id,
            table=table,
            database=database,
            act_name="router_projection_weight",
            direction="iter",
        )
        assert len(weight_rows) == expected_weight_rows
        assert {key[4] for key, _value in weight_rows} == set(
            range(train_iters + 1)
        )
        assert {key[8] for key, _value in weight_rows} == {0, 1}
    finally:
        client.execute(f"DROP TABLE IF EXISTS `{database}`.`{table}`")
        client.execute(f"DROP TABLE IF EXISTS `{database}`.`{table}_scalar_float`")
        client.execute(f"DROP TABLE IF EXISTS `{database}`.`{table}_scalar_int`")
        client.execute(f"DROP TABLE IF EXISTS `{database}`.`{table}_eval_phase_boundary`")
        client.disconnect()


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Real Megatron training E2E needs CUDA")
def test_real_megatron_grad_norm_preserves_megatron_value_when_clip_disabled(tmp_path):
    """Store the common optimizer's returned norm even when clipping is disabled."""

    if _available_cuda_devices() < 1:
        pytest.skip("iteration-metric guard E2E requires one CUDA device")

    client = _clickhouse_client_or_skip()
    database = os.environ.get("DMX_DB_DATABASE", "default")
    table = f"dmi_megatron_iteration_guard_e2e_{uuid.uuid4().hex}"
    scalar_table = f"{table}_scalar_float"
    model_id = f"megatron-iteration-guard-e2e-{uuid.uuid4().hex}"
    log_path = tmp_path / "megatron_iteration_metric_guards.log"

    client.execute(f"CREATE DATABASE IF NOT EXISTS `{database}`")
    client.execute(f"DROP TABLE IF EXISTS `{database}`.`{table}`")
    client.execute(f"DROP TABLE IF EXISTS `{database}`.`{scalar_table}`")
    _create_training_table(client, database=database, table=table)

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{ROOT}:{MEGATRON_ROOT}:{env.get('PYTHONPATH', '')}"
    env["DMI_ENABLE"] = "1"
    env.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", "1")
    if "DMI_REAL_E2E_CUDA_VISIBLE_DEVICES" in env:
        env["CUDA_VISIBLE_DEVICES"] = env["DMI_REAL_E2E_CUDA_VISIBLE_DEVICES"]

    cmd = _tiny_megatron_router_summary_cmd(
        model_id=model_id,
        train_iters=1,
        eval_iters=0,
        micro_batch_size=1,
        global_batch_size=1,
        clip_grad=0.0,
        database=database,
        table=table,
        extra_args=["--dmi-hook-selection", "grad-norm"],
    )

    try:
        _run_megatron_cmd(cmd, env=env, log_path=log_path)
        _wait_for_exact_scalar_rows(
            client,
            database=database,
            table=scalar_table,
            model_id=model_id,
            act_name="grad_norm",
            direction="iter",
            expected=1,
        )
        rows = _read_training_scalar_act_rows(
            model_id=model_id,
            table=table,
            database=database,
            act_name="grad_norm",
            direction="iter",
        )
        assert len(rows) == 1
        key, value = rows[0]
        assert key[4:12] == (1, -1, -1, -1, -1, -1, 0, 1)
        log_text = log_path.read_text(encoding="utf-8", errors="replace")
        match = re.search(r"grad norm:\s*([0-9.Ee+-]+)", log_text)
        assert match is not None
        assert float(value) == pytest.approx(float(match.group(1)), abs=5e-4)
    finally:
        client.execute(f"DROP TABLE IF EXISTS `{database}`.`{table}`")
        client.execute(f"DROP TABLE IF EXISTS `{database}`.`{scalar_table}`")
        client.execute(f"DROP TABLE IF EXISTS `{database}`.`{table}_scalar_int`")
        client.execute(f"DROP TABLE IF EXISTS `{database}`.`{table}_eval_phase_boundary`")
        client.disconnect()


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Real Megatron training E2E needs CUDA")
@pytest.mark.parametrize(
    ("case_name", "moe_args", "extra_args", "default_train_iters", "pp2_skip_reason"),
    [
        pytest.param("eager", ["--moe-token-dispatcher-type", "allgather"], [], 2, None, id="eager"),
        pytest.param(
            "eager_full_recompute",
            ["--moe-token-dispatcher-type", "allgather"],
            [
                "--recompute-granularity",
                "full",
                "--recompute-method",
                "uniform",
                "--recompute-num-layers",
                "1",
                "--attention-dropout",
                "0",
                "--hidden-dropout",
                "0",
            ],
            2,
            None,
            id="eager_full_recompute",
        ),
        pytest.param(
            "local_cuda_graph",
            [
                "--moe-token-dispatcher-type",
                "alltoall",
                "--moe-expert-capacity-factor",
                "1.0",
                "--moe-pad-expert-input-to-capacity",
            ],
            ["--cuda-graph-impl", "local"],
            3,
            "Native Megatron PP>1 with local per-layer CUDA graphs currently "
            "fails before DMI verification: graph replay can return a view "
            "pipeline output, then deallocate_output_tensor asserts because "
            "deallocate_pipeline_outputs is enabled and views cannot be "
            "pseudo-deallocated.",
            id="local_cuda_graph",
        ),
        pytest.param(
            "local_cuda_graph_selective_recompute",
            ["--moe-token-dispatcher-type", "allgather"],
            [
                "--cuda-graph-impl",
                "local",
                "--cuda-graph-scope",
                "moe_router",
                "--recompute-granularity",
                "selective",
                "--recompute-modules",
                "moe",
                "--attention-dropout",
                "0",
                "--hidden-dropout",
                "0",
            ],
            3,
            "Native Megatron PP>1 with local per-layer CUDA graphs currently "
            "fails before DMI verification: graph replay can return a view "
            "pipeline output, then deallocate_output_tensor asserts because "
            "deallocate_pipeline_outputs is enabled and views cannot be "
            "pseudo-deallocated.",
            id="local_cuda_graph_selective_recompute",
        ),
        pytest.param(
            "full_iteration_cuda_graph",
            [
                "--moe-token-dispatcher-type",
                "alltoall",
                "--moe-expert-capacity-factor",
                "1.0",
                "--moe-pad-expert-input-to-capacity",
            ],
            [
                "--cuda-graph-impl",
                "local",
                "--cuda-graph-scope",
                "full_iteration",
                "--no-check-for-nan-in-loss-and-grad",
            ],
            3,
            "Native Megatron PP>1 with full-iteration CUDA graphs currently "
            "fails before DMI verification: pipeline P2P communication calls "
            "torch.cuda.synchronize() during CUDA graph capture, which is not "
            "capture-safe.",
            id="full_iteration_cuda_graph",
        ),
        pytest.param(
            "full_iteration_cuda_graph_full_recompute",
            [
                "--moe-token-dispatcher-type",
                "alltoall",
                "--moe-expert-capacity-factor",
                "1.0",
                "--moe-pad-expert-input-to-capacity",
            ],
            [
                "--cuda-graph-impl",
                "local",
                "--cuda-graph-scope",
                "full_iteration",
                "--no-check-for-nan-in-loss-and-grad",
                "--recompute-granularity",
                "full",
                "--recompute-method",
                "uniform",
                "--recompute-num-layers",
                "1",
                "--attention-dropout",
                "0",
                "--hidden-dropout",
                "0",
            ],
            3,
            "Native Megatron PP>1 with full-iteration CUDA graphs currently "
            "fails before DMI verification: pipeline P2P communication calls "
            "torch.cuda.synchronize() during CUDA graph capture, which is not "
            "capture-safe.",
            id="full_iteration_cuda_graph_full_recompute",
        ),
    ],
)
@pytest.mark.parametrize(
    ("parallel_name", "nproc_per_node", "tp_size", "pp_size", "num_microbatches"),
    [
        pytest.param("tp1_pp1", 1, 1, 1, 1, id="tp1_pp1"),
        pytest.param("tp1_pp2", 2, 1, 2, 2, id="tp1_pp2"),
    ],
)
def test_real_megatron_training_router_and_loss_summary_exact_clickhouse_rows(
    tmp_path,
    case_name,
    moe_args,
    extra_args,
    default_train_iters,
    pp2_skip_reason,
    parallel_name,
    nproc_per_node,
    tp_size,
    pp_size,
    num_microbatches,
):
    """Run real Megatron training and verify exact combined-hook ClickHouse rows."""

    if _available_cuda_devices() < nproc_per_node:
        pytest.skip(f"{parallel_name} requires {nproc_per_node} CUDA devices")
    if pp_size > 1 and pp2_skip_reason is not None:
        pytest.skip(pp2_skip_reason)

    client = _clickhouse_client_or_skip()
    database = os.environ.get("DMX_DB_DATABASE", "default")
    table = f"dmi_megatron_real_{case_name}_{parallel_name}_e2e_{uuid.uuid4().hex}"
    model_id = f"megatron-real-{case_name}-{parallel_name}-e2e-{uuid.uuid4().hex}"
    log_path = tmp_path / f"megatron_real_training_{case_name}_{parallel_name}.log"

    env_key = f"DMI_REAL_E2E_{case_name.upper()}_TRAIN_ITERS"
    train_iters = int(os.environ.get(env_key, str(default_train_iters)))
    eval_iters = int(os.environ.get("DMI_REAL_E2E_EVAL_ITERS", "1"))
    eval_interval = int(os.environ.get("DMI_REAL_E2E_EVAL_INTERVAL", "1"))
    micro_batch_size = 2
    global_batch_size = micro_batch_size * num_microbatches
    num_moe_layers = 2
    expected_tensor_phase_counts = _expected_phase_counts(
        train_iters=train_iters,
        eval_iters=eval_iters,
        eval_interval=eval_interval,
        global_batch_size=global_batch_size,
        num_moe_layers=num_moe_layers,
    )
    expected_scalar_phase_counts = _expected_sample_phase_counts(
        train_iters=train_iters,
        eval_iters=eval_iters,
        eval_interval=eval_interval,
        global_batch_size=global_batch_size,
    )
    expected_tensor_rows = sum(expected_tensor_phase_counts.values())
    expected_scalar_float_rows = sum(expected_scalar_phase_counts.values())

    client.execute(f"CREATE DATABASE IF NOT EXISTS `{database}`")
    client.execute(f"DROP TABLE IF EXISTS `{database}`.`{table}`")
    _create_training_table(client, database=database, table=table)

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{ROOT}:{MEGATRON_ROOT}:{env.get('PYTHONPATH', '')}"
    env["DMI_ENABLE"] = "1"
    env.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", "1")
    if "DMI_REAL_E2E_CUDA_VISIBLE_DEVICES" in env:
        env["CUDA_VISIBLE_DEVICES"] = env["DMI_REAL_E2E_CUDA_VISIBLE_DEVICES"]

    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc_per_node={nproc_per_node}",
        "pretrain_gpt.py",
        "--mock-data",
        "--tokenizer-type",
        "NullTokenizer",
        "--vocab-size",
        "128",
        "--num-layers",
        str(num_moe_layers),
        "--hidden-size",
        "64",
        "--ffn-hidden-size",
        "128",
        "--num-attention-heads",
        "4",
        "--seq-length",
        "16",
        "--max-position-embeddings",
        "16",
        "--micro-batch-size",
        str(micro_batch_size),
        "--global-batch-size",
        str(global_batch_size),
        "--tensor-model-parallel-size",
        str(tp_size),
        "--pipeline-model-parallel-size",
        str(pp_size),
        "--train-iters",
        str(train_iters),
        "--eval-interval",
        str(eval_interval),
        "--eval-iters",
        str(eval_iters),
        "--seed",
        "1234",
        "--lr",
        "1.0e-4",
        "--min-lr",
        "1.0e-5",
        "--lr-decay-iters",
        str(train_iters),
        "--lr-warmup-iters",
        "0",
        "--weight-decay",
        "0.0",
        "--adam-beta1",
        "0.9",
        "--adam-beta2",
        "0.95",
        "--init-method-std",
        "0.02",
        "--clip-grad",
        "1.0",
        "--bf16",
        "--transformer-impl",
        "local",
        "--no-persist-layer-norm",
        "--no-gradient-accumulation-fusion",
        "--swiglu",
        "--disable-bias-linear",
        "--num-experts",
        "2",
        "--moe-router-topk",
        "1",
        "--moe-router-pre-softmax",
        "--moe-router-load-balancing-type",
        "aux_loss",
        "--moe-aux-loss-coeff",
        "0.01",
        "--no-save-optim",
        "--no-save-rng",
        "--no-load-optim",
        "--no-load-rng",
        "--no-create-attention-mask-in-dataloader",
        "--dmi-enable",
        "--dmi-hook-selection",
        "router-summary,loss-summary",
        "--dmi-model-id",
        model_id,
        "--dmi-db-host",
        os.environ.get("DMX_DB_HOST", "localhost"),
        "--dmi-db-port",
        os.environ.get("DMX_DB_PORT", "9000"),
        "--dmi-db-database",
        database,
        "--dmi-clickhouse-table",
        table,
        "--dmi-ch-parallelism",
        os.environ.get("DMI_REAL_E2E_CH_PARALLELISM", "10"),
        "--dmi-ring-payload-mb",
        os.environ.get("DMI_REAL_E2E_RING_PAYLOAD_MB", "64"),
        "--dmi-ring-pinned-mb",
        os.environ.get("DMI_REAL_E2E_RING_PINNED_MB", "64"),
        "--dmi-ring-task-entries",
        os.environ.get("DMI_REAL_E2E_RING_TASK_ENTRIES", "1024"),
    ] + moe_args + extra_args

    try:
        with log_path.open("w", encoding="utf-8") as log:
            result = subprocess.run(
                cmd,
                cwd=MEGATRON_ROOT,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=float(os.environ.get("DMI_REAL_E2E_TIMEOUT_S", "240")),
                check=False,
            )
        if result.returncode != 0:
            tail = log_path.read_text(encoding="utf-8", errors="replace")[-12000:]
            raise AssertionError(
                f"Real Megatron training E2E failed with code {result.returncode}. "
                f"Log tail:\n{tail}"
            )
        log_text = log_path.read_text(encoding="utf-8", errors="replace")
        assert "payload arrived with empty metadata FIFO" not in log_text
        assert "unconsumed metadata entries" not in log_text

        _wait_for_exact_rows(
            client,
            database=database,
            table=table,
            model_id=model_id,
            expected=expected_tensor_rows,
        )
        _wait_for_exact_scalar_rows(
            client,
            database=database,
            table=f"{table}_scalar_float",
            model_id=model_id,
            act_name="lm_per_sample_loss",
            expected=expected_scalar_float_rows,
        )
        _assert_training_table_counts(
            client=client,
            database=database,
            table=table,
            model_id=model_id,
            expected_tensor_rows=expected_tensor_rows,
            expected_scalar_float_rows=expected_scalar_float_rows,
            expected_attempt_status_rows=train_iters,
        )
        rows = _query_rows(client, database=database, table=table, model_id=model_id)
        scalar_rows = _query_scalar_values(
            client,
            database=database,
            table=f"{table}_scalar_float",
            model_id=model_id,
            act_name="lm_per_sample_loss",
        )

        assert len(rows) == expected_tensor_rows
        assert len(scalar_rows) == expected_scalar_float_rows
        tensor_keys = [tuple(row[:12]) for row in rows]
        scalar_keys = [tuple(row[:5]) for row in scalar_rows]
        assert len(set(tensor_keys)) == len(tensor_keys)
        assert len(set(scalar_keys)) == len(scalar_keys)
        actual_phase_counts = {
            phase: sum(1 for row in rows if row[3] == phase)
            for phase in ("train", "valid", "test")
        }
        actual_scalar_phase_counts = {
            phase: sum(1 for row in scalar_rows if row[0] == phase)
            for phase in ("train", "valid", "test")
        }
        assert actual_phase_counts == expected_tensor_phase_counts
        assert actual_scalar_phase_counts == expected_scalar_phase_counts
        assert {row[3] for row in rows} == {
            phase for phase, count in expected_tensor_phase_counts.items() if count > 0
        }
        assert {row[4] for row in rows if row[3] == "train"} == set(range(1, train_iters + 1))
        assert {row[6] for row in rows} == set(range(num_microbatches))
        assert {row[7] for row in rows} == {0, 1}
        assert {row[8] for row in rows} == {0, 1}
        assert {row[9] for row in rows} == {0}
        assert {row[12] for row in rows} == {"torch.float"}
        assert {tuple(row[13]) for row in rows} == {(2,)}
        assert {row[14] for row in rows} == {8}
        for row in rows:
            token_start = int(row[10])
            token_end = int(row[11])
            assert token_start == 0
            assert 0 < token_end <= 16
        for phase, _global_batch_id, _microbatch_id, _sample_index, layer_no, value in scalar_rows:
            assert phase in {"train", "valid", "test"}
            assert int(layer_no) == -1
            assert float(value) > 0.0
    finally:
        client.execute(f"DROP TABLE IF EXISTS `{database}`.`{table}`")
        client.execute(f"DROP TABLE IF EXISTS `{database}`.`{table}_scalar_float`")
        client.execute(f"DROP TABLE IF EXISTS `{database}`.`{table}_scalar_int`")
        client.execute(f"DROP TABLE IF EXISTS `{database}`.`{table}_eval_phase_boundary`")
        client.disconnect()
