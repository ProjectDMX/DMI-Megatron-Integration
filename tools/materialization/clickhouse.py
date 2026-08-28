#!/usr/bin/env python3
"""Materialize raw DMI Megatron rows into analysis-ready training tables."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from typing import Any, Iterable, Sequence

import torch

from tools.materialization.metrics import (
    coefficient_of_variation,
    levenshtein,
    materialize_pathway_windows,
    pairwise_distance_sum_blockwise,
    paper_pathway_sequence,
    pathway_consistency,
    pathway_tokens,
)
from dmi.api.v1 import CHClickhouseDriverReadOnly


EXPERT_COUNT_ACTS = ("pre_drop_token_count", "post_drop_token_count")
ROUTER_ENTROPY_ACT = "router_token_entropy_mean"
ROUTER_PROBS_ACT = "router_probs_mean"
ROUTER_WEIGHT_ACT = "router_projection_weight"
LOSS_MEAN_ACT = "lm_per_sample_loss"
LOSS_TOKEN_COUNT_ACT = "lm_per_sample_loss_token_count"
ATTEMPT_STATUS_ACT = "iteration_attempt_status"
DEFAULT_PATHWAY_WINDOWS = (20,)

# Preserve the private names used by older analysis scripts while keeping the
# implementations in the pure metrics module.
_levenshtein = levenshtein
_pairwise_distance_sum_blockwise = pairwise_distance_sum_blockwise

TABLE_COLUMNS: dict[str, tuple[tuple[str, str], ...]] = {
    "training_scalar": (
        ("model_id", "String"),
        ("run_name", "String"),
        ("phase", "String"),
        ("training_iteration_id", "UInt64"),
        ("global_batch_id", "UInt64"),
        ("metric_name", "LowCardinality(String)"),
        ("metric_value", "Float64"),
    ),
    "expert_load": (
        ("model_id", "String"),
        ("run_name", "String"),
        ("phase", "String"),
        ("training_iteration_id", "UInt64"),
        ("global_batch_id", "UInt64"),
        ("layer_id", "Int32"),
        ("expert_id", "UInt16"),
        ("load_kind", "LowCardinality(String)"),
        ("token_count", "UInt64"),
    ),
    "expert_load_summary": (
        ("model_id", "String"),
        ("run_name", "String"),
        ("phase", "String"),
        ("training_iteration_id", "UInt64"),
        ("global_batch_id", "UInt64"),
        ("layer_id", "Int32"),
        ("active_expert_count", "UInt16"),
        ("dead_expert_count", "UInt16"),
        ("load_cv", "Float64"),
        ("max_load_share", "Float64"),
        ("drop_rate", "Float64"),
    ),
    "router_entropy": (
        ("model_id", "String"),
        ("run_name", "String"),
        ("phase", "String"),
        ("training_iteration_id", "UInt64"),
        ("global_batch_id", "UInt64"),
        ("layer_id", "Int32"),
        ("entropy_mean", "Float64"),
    ),
    "pathway_consistency": (
        ("model_id", "String"),
        ("run_name", "String"),
        ("phase", "LowCardinality(String)"),
        ("training_iteration_id", "UInt64"),
        ("dataset_id", "Int32"),
        ("window_size_iterations", "UInt16"),
        ("effective_window_size_iterations", "UInt16"),
        ("window_start_iteration", "UInt64"),
        ("window_end_iteration", "UInt64"),
        ("sample_count", "UInt32"),
        ("consistency_sum", "Float64"),
        ("mean_pathway_consistency", "Float64"),
        ("consistency_eps", "Float64"),
    ),
    "pathway_edit_distance": (
        ("model_id", "String"),
        ("run_name", "String"),
        ("phase", "LowCardinality(String)"),
        ("training_iteration_id", "UInt64"),
        ("dataset_id", "Int32"),
        ("window_size_iterations", "UInt16"),
        ("effective_window_size_iterations", "UInt16"),
        ("window_start_iteration", "UInt64"),
        ("window_end_iteration", "UInt64"),
        ("sample_count", "UInt32"),
        ("pair_count", "UInt64"),
        ("pathway_edit_distance_sum", "Float64"),
        ("mean_pathway_edit_distance", "Nullable(Float64)"),
        ("pathway_threshold", "Float64"),
    ),
}

TABLE_ORDER_BY = {
    "training_scalar": "(model_id, run_name, phase, training_iteration_id, metric_name)",
    "expert_load": (
        "(model_id, run_name, phase, layer_id, load_kind, "
        "training_iteration_id, expert_id)"
    ),
    "expert_load_summary": "(model_id, run_name, phase, layer_id, training_iteration_id)",
    "router_entropy": "(model_id, run_name, phase, layer_id, training_iteration_id)",
    "pathway_consistency": (
        "(model_id, run_name, phase, dataset_id, "
        "window_size_iterations, training_iteration_id)"
    ),
    "pathway_edit_distance": (
        "(model_id, run_name, phase, dataset_id, "
        "window_size_iterations, training_iteration_id)"
    ),
}


def _ident(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value):
        raise ValueError(f"Invalid ClickHouse identifier: {value!r}")
    return value


def _q(value: str) -> str:
    return f"`{_ident(value)}`"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--user", default="default")
    parser.add_argument("--password", default="")
    parser.add_argument("--raw-db", required=True)
    parser.add_argument("--processed-db", required=True)
    parser.add_argument("--raw-table", default="training_payload")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--phase", default="train", choices=["train", "valid", "test"])
    parser.add_argument("--expected-train-iters", type=int, default=0)
    parser.add_argument("--materialize-pathways", action="store_true")
    parser.add_argument("--pathway-window-size", action="append", type=int, default=None)
    parser.add_argument("--pathway-threshold", type=float, default=0.7)
    parser.add_argument("--consistency-eps", type=float, default=1e-8)
    parser.add_argument("--expected-layer-count", type=int, default=16)
    parser.add_argument("--expected-expert-count", type=int, default=64)
    parser.add_argument("--expected-hidden-size", type=int, default=2048)
    parser.add_argument("--expected-samples-per-iteration", type=int, default=2)
    parser.add_argument("--insert-batch-rows", type=int, default=50_000)
    parser.add_argument("--drop-existing", action="store_true")
    return parser.parse_args(argv)


def _client(args: argparse.Namespace):
    from clickhouse_driver import Client

    return Client(
        host=args.host,
        port=int(args.port),
        user=args.user,
        password=args.password,
        database=args.raw_db,
    )


def _raw_execute(client: Any, query: str, params: dict[str, Any]):
    return client.execute(query, params, settings={"strings_as_bytes": 1})


def _decode_cell(value: Any) -> Any:
    if isinstance(value, memoryview):
        value = value.tobytes()
    if isinstance(value, bytearray):
        value = bytes(value)
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="surrogateescape")
    return value


def _decode_tensor(dtype: Any, shape: Any, payload: Any) -> torch.Tensor:
    return CHClickhouseDriverReadOnly.torch_decode(dtype, shape, payload).clone()


def _accepted_attempt_subquery(args: argparse.Namespace) -> str:
    scalar_int_table = f"{args.raw_table}_scalar_int"
    return f"""
        SELECT global_batch_id, attempt_id
        FROM {_q(args.raw_db)}.{_q(scalar_int_table)}
        WHERE model_id = %(model_id)s
          AND act_name = '{ATTEMPT_STATUS_ACT}'
          AND direction = 'iter'
          AND phase = 'train'
          AND invocation_id = 0
          AND dataset_id = -1
          AND value = 1
    """


def _read_accepted_attempts(
    client: Any,
    args: argparse.Namespace,
) -> dict[int, int]:
    scalar_int_table = f"{args.raw_table}_scalar_int"
    rows = _raw_execute(
        client,
        f"""
        SELECT global_batch_id, attempt_id, value,
               dp_rank, microbatch_id, sample_index, layer_no, shard_rank,
               token_start, token_end, invocation_id, dataset_id
        FROM {_q(args.raw_db)}.{_q(scalar_int_table)}
        WHERE model_id = %(model_id)s
          AND act_name = '{ATTEMPT_STATUS_ACT}'
          AND direction = 'iter'
          AND phase = 'train'
        ORDER BY global_batch_id, attempt_id
        """,
        {"model_id": args.model_id},
    )
    statuses: dict[int, dict[int, int]] = defaultdict(dict)
    for row in rows:
        gid, attempt_id, value = int(row[0]), int(row[1]), int(row[2])
        identity = tuple(int(item) for item in row[3:])
        expected_identity = (-1, -1, -1, -1, -1, 0, 1, 0, -1)
        if identity != expected_identity:
            raise RuntimeError(
                f"Invalid attempt status identity for gid={gid}, attempt={attempt_id}: "
                f"{identity}"
            )
        if value not in {-1, 0, 1}:
            raise RuntimeError(
                f"Invalid attempt status value for gid={gid}, attempt={attempt_id}: {value}"
            )
        if attempt_id in statuses[gid]:
            raise RuntimeError(
                f"Duplicate attempt status for gid={gid}, attempt={attempt_id}"
            )
        statuses[gid][attempt_id] = value

    accepted: dict[int, int] = {}
    for gid, attempts in statuses.items():
        accepted_ids = [attempt for attempt, value in attempts.items() if value == 1]
        if len(accepted_ids) > 1:
            raise RuntimeError(f"Multiple accepted attempts for gid={gid}: {accepted_ids}")
        if len(accepted_ids) == 1:
            accepted[gid] = accepted_ids[0]
        elif -1 not in attempts.values():
            raise RuntimeError(f"Training iteration {gid} has no accepted attempt")

    represented_iterations: set[int] = set()
    for table in (
        args.raw_table,
        f"{args.raw_table}_scalar_float",
        f"{args.raw_table}_scalar_int",
    ):
        represented_iterations.update(
            int(row[0])
            for row in _raw_execute(
                client,
                f"""
                SELECT DISTINCT global_batch_id
                FROM {_q(args.raw_db)}.{_q(table)}
                WHERE model_id = %(model_id)s
                  AND phase = 'train'
                  AND global_batch_id != 0
                """,
                {"model_id": args.model_id},
            )
        )
    missing = sorted(
        gid
        for gid in represented_iterations
        if gid not in accepted and -1 not in statuses.get(gid, {}).values()
    )
    if missing:
        raise RuntimeError(
            "Training payload iterations have no accepted attempt status: "
            f"{missing}"
        )
    if not accepted:
        raise RuntimeError("No accepted DMI training attempts were found")
    return accepted


def _create_tables(
    client: Any,
    processed_db: str,
    *,
    drop_existing: bool,
    include_pathways: bool,
) -> tuple[str, ...]:
    client.execute(f"CREATE DATABASE IF NOT EXISTS {_q(processed_db)}")
    tables = (
        "training_scalar",
        "expert_load",
        "expert_load_summary",
        "router_entropy",
    )
    if include_pathways:
        tables += ("pathway_consistency", "pathway_edit_distance")
    if drop_existing:
        for table in tables:
            client.execute(f"DROP TABLE IF EXISTS {_q(processed_db)}.{_q(table)}")

    for table in tables:
        definitions = ",\n".join(
            f"          {_q(name)} {column_type}"
            for name, column_type in TABLE_COLUMNS[table]
        )
        client.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {_q(processed_db)}.{_q(table)} (
{definitions}
            )
            ENGINE = MergeTree
            ORDER BY {TABLE_ORDER_BY[table]}
            """
        )
        actual = tuple(
            (str(row[0]), str(row[1]))
            for row in client.execute(
                f"DESCRIBE TABLE {_q(processed_db)}.{_q(table)}"
            )
        )
        if actual != TABLE_COLUMNS[table]:
            raise RuntimeError(
                f"Existing schema mismatch for {processed_db}.{table}: "
                f"actual={actual}, expected={TABLE_COLUMNS[table]}"
            )
    return tables


def _read_expert_counts(client: Any, args: argparse.Namespace):
    rows = _raw_execute(
        client,
        f"""
        SELECT
          act_name, direction, phase, global_batch_id, dp_rank, microbatch_id,
          sample_index, layer_no, shard_rank, attempt_id, invocation_id,
          dataset_id, dtype, shape, bytes
        FROM {_q(args.raw_db)}.{_q(args.raw_table)}
        WHERE model_id = %(model_id)s
          AND phase = %(phase)s
          AND act_name IN %(acts)s
          AND invocation_id = 0
          AND (global_batch_id, attempt_id) IN ({_accepted_attempt_subquery(args)})
        ORDER BY act_name, global_batch_id, direction, layer_no,
                 dp_rank, microbatch_id, sample_index, shard_rank
        """,
        {"model_id": args.model_id, "phase": args.phase, "acts": EXPERT_COUNT_ACTS},
    )
    directions = {str(_decode_cell(row[1])) for row in rows}
    if directions != {"fwd"}:
        raise RuntimeError(
            "Expert-count materialization requires FWD-only raw rows; "
            f"got directions={sorted(directions)}"
        )
    counts: dict[tuple[str, str, int, int, int], torch.Tensor] = {}
    seen_coordinates: set[tuple[Any, ...]] = set()
    seen_acts: set[str] = set()
    for row in rows:
        act_name = str(_decode_cell(row[0]))
        phase = str(_decode_cell(row[2]))
        global_batch_id = int(row[3])
        coordinate = (
            act_name,
            phase,
            global_batch_id,
            *(int(value) for value in row[4:9]),
            int(row[9]),
            int(row[10]),
        )
        if coordinate in seen_coordinates:
            raise RuntimeError(f"Duplicate expert-count raw row: {coordinate}")
        seen_coordinates.add(coordinate)
        layer_no = int(row[7])
        shard_rank = int(row[8])
        dataset_id = int(row[11])
        if dataset_id < -1:
            raise RuntimeError(f"Invalid dataset_id in expert-count row: {dataset_id}")
        tensor = _decode_tensor(row[12], row[13], row[14]).to(torch.int64).reshape(-1)
        if tensor.numel() != int(args.expected_expert_count) or torch.any(tensor < 0):
            raise RuntimeError(
                f"Invalid {act_name} tensor at gid={global_batch_id}, layer={layer_no}: "
                f"shape={tuple(tensor.shape)}, minimum={int(tensor.min().item())}"
            )
        key = (act_name, phase, global_batch_id, layer_no, shard_rank)
        if key not in counts:
            counts[key] = torch.zeros_like(tensor, dtype=torch.int64)
        counts[key] += tensor
        seen_acts.add(act_name)
    missing = set(EXPERT_COUNT_ACTS) - seen_acts
    if missing:
        raise RuntimeError(f"Missing expert-count raw rows for: {sorted(missing)}")
    return counts


def _read_scalar_rows(client: Any, args: argparse.Namespace):
    scalar_table = f"{args.raw_table}_scalar_float"
    return _raw_execute(
        client,
        f"""
        SELECT
          act_name, direction, phase, global_batch_id, dp_rank, microbatch_id,
          sample_index, layer_no, shard_rank, token_start, token_end,
          attempt_id, invocation_id, dataset_id, value
        FROM {_q(args.raw_db)}.{_q(scalar_table)}
        WHERE model_id = %(model_id)s
          AND phase = %(phase)s
          AND invocation_id = 0
          AND (global_batch_id, attempt_id) IN ({_accepted_attempt_subquery(args)})
        ORDER BY act_name, global_batch_id, direction, dp_rank,
                 microbatch_id, sample_index, layer_no, shard_rank
        """,
        {"model_id": args.model_id, "phase": args.phase},
    )


def _read_scalar_int_rows(client: Any, args: argparse.Namespace):
    scalar_table = f"{args.raw_table}_scalar_int"
    return _raw_execute(
        client,
        f"""
        SELECT
          act_name, direction, phase, global_batch_id, dp_rank, microbatch_id,
          sample_index, layer_no, shard_rank, token_start, token_end,
          attempt_id, invocation_id, dataset_id, value
        FROM {_q(args.raw_db)}.{_q(scalar_table)}
        WHERE model_id = %(model_id)s
          AND phase = %(phase)s
          AND invocation_id = 0
          AND (global_batch_id, attempt_id) IN ({_accepted_attempt_subquery(args)})
        ORDER BY act_name, global_batch_id, direction, dp_rank,
                 microbatch_id, sample_index, layer_no, shard_rank
        """,
        {"model_id": args.model_id, "phase": args.phase},
    )


def _materialize_direct_iteration_scalars(
    rows: list[tuple[Any, ...]], args: argparse.Namespace
):
    out = []
    seen: set[tuple[str, str, int]] = set()
    for row in rows:
        act_name = str(_decode_cell(row[0]))
        direction = str(_decode_cell(row[1]))
        phase = str(_decode_cell(row[2]))
        gid = int(row[3])
        value = float(row[14])
        if direction != "iter":
            continue
        if act_name != "grad_norm":
            continue
        identity = tuple(int(value) for value in row[4:14])
        expected_identity = (-1, -1, -1, -1, -1, 0, 1, int(row[11]), 0, -1)
        if identity != expected_identity:
            raise RuntimeError(
                f"Invalid PER_ITERATION scalar identity for {act_name}: "
                f"{identity}, expected {expected_identity}"
            )
        key = (act_name, phase, gid)
        if key in seen:
            raise RuntimeError(f"Duplicate iteration scalar row: {key}")
        if not math.isfinite(value):
            raise RuntimeError(f"Nonfinite iteration scalar row: key={key}, value={value}")
        seen.add(key)
        out.append((args.model_id, args.run_name, phase, gid, gid, act_name, value))
    if not out:
        raise RuntimeError("Missing grad_norm PER_ITERATION rows with direction='iter'")
    return out


def _sample_scalar_map(
    rows: list[tuple[Any, ...]],
    *,
    act_name: str,
    integer: bool,
) -> dict[tuple[Any, ...], float | int]:
    values: dict[tuple[Any, ...], float | int] = {}
    for row in rows:
        if str(_decode_cell(row[0])) != act_name:
            continue
        direction = str(_decode_cell(row[1]))
        phase = str(_decode_cell(row[2]))
        if direction != "fwd":
            raise RuntimeError(f"{act_name} requires FWD rows, got {direction!r}")
        coordinate = (phase, *(int(value) for value in row[3:14]))
        if coordinate in values:
            raise RuntimeError(f"Duplicate {act_name} raw row: {coordinate}")
        layer_no = int(row[7])
        shard_rank = int(row[8])
        if layer_no != -1 or shard_rank != 0:
            raise RuntimeError(
                f"{act_name} requires layer_no=-1 and shard_rank=0; "
                f"got layer_no={layer_no}, shard_rank={shard_rank}"
            )
        value = int(row[14]) if integer else float(row[14])
        if integer:
            if value < 0:
                raise RuntimeError(f"Negative loss-token count at {coordinate}: {value}")
        elif not math.isfinite(value):
            raise RuntimeError(f"Nonfinite per-sample loss at {coordinate}: {value}")
        values[coordinate] = value
    if not values:
        raise RuntimeError(f"Missing raw rows for {act_name!r}")
    return values


def _materialize_iteration_loss_from_samples(
    scalar_float_rows: list[tuple[Any, ...]],
    scalar_int_rows: list[tuple[Any, ...]],
    args: argparse.Namespace,
) -> list[tuple[Any, ...]]:
    means = _sample_scalar_map(
        scalar_float_rows,
        act_name=LOSS_MEAN_ACT,
        integer=False,
    )
    counts = _sample_scalar_map(
        scalar_int_rows,
        act_name=LOSS_TOKEN_COUNT_ACT,
        integer=True,
    )
    if set(means) != set(counts):
        missing_counts = sorted(set(means) - set(counts))
        missing_means = sorted(set(counts) - set(means))
        raise RuntimeError(
            "Loss mean/count sample identities do not match: "
            f"missing_counts={missing_counts[:5]}, missing_means={missing_means[:5]}"
        )

    weighted_sum: dict[tuple[str, int], float] = defaultdict(float)
    token_sum: dict[tuple[str, int], int] = defaultdict(int)
    for coordinate, mean_value in means.items():
        phase = str(coordinate[0])
        gid = int(coordinate[1])
        count = int(counts[coordinate])
        weighted_sum[(phase, gid)] += float(mean_value) * count
        token_sum[(phase, gid)] += count

    out = []
    for (phase, gid), total_loss in sorted(weighted_sum.items()):
        total_tokens = token_sum[(phase, gid)]
        if total_tokens <= 0:
            raise RuntimeError(
                f"Iteration loss has zero total loss-token count: phase={phase}, gid={gid}"
            )
        out.append(
            (
                args.model_id,
                args.run_name,
                phase,
                gid,
                gid,
                "lm_loss_iteration",
                total_loss / total_tokens,
            )
        )
    return out


def _materialize_router_entropy(rows: list[tuple[Any, ...]], args: argparse.Namespace):
    entropy_rows = [
        row for row in rows if str(_decode_cell(row[0])) == ROUTER_ENTROPY_ACT
    ]
    directions = {str(_decode_cell(row[1])) for row in entropy_rows}
    if directions != {"fwd"}:
        raise RuntimeError(
            "Router-entropy materialization requires FWD-only raw rows; "
            f"got directions={sorted(directions)}"
        )
    grouped: dict[tuple[str, int, int], list[float]] = defaultdict(list)
    seen: set[tuple[Any, ...]] = set()
    for row in entropy_rows:
        phase = str(_decode_cell(row[2]))
        gid = int(row[3])
        coordinate = (phase, gid, *(int(value) for value in row[4:14]))
        if coordinate in seen:
            raise RuntimeError(f"Duplicate router-entropy raw row: {coordinate}")
        seen.add(coordinate)
        value = float(row[14])
        if not math.isfinite(value):
            raise RuntimeError(
                f"Nonfinite router entropy at gid={gid}, layer={int(row[7])}: {value}"
            )
        grouped[(phase, gid, int(row[7]))].append(value)
    if not grouped:
        raise RuntimeError(f"Missing router entropy rows for act_name={ROUTER_ENTROPY_ACT!r}")
    out = []
    for (phase, gid, layer_no), values in sorted(grouped.items()):
        if len(values) != int(args.expected_samples_per_iteration):
            raise RuntimeError(
                f"Router entropy gid={gid}, layer={layer_no} has {len(values)} samples; "
                f"expected {args.expected_samples_per_iteration}"
            )
        out.append(
            (args.model_id, args.run_name, phase, gid, gid, layer_no, sum(values) / len(values))
        )
    return out


def _materialize_expert_load(
    counts: dict[tuple[str, str, int, int, int], torch.Tensor],
    args: argparse.Namespace,
):
    by_base: dict[tuple[str, int, int], dict[str, torch.Tensor]] = defaultdict(dict)
    for (act_name, phase, gid, layer_no, _shard_rank), tensor in counts.items():
        kind = "pre_drop" if act_name == "pre_drop_token_count" else "post_drop"
        base = (phase, gid, layer_no)
        if kind not in by_base[base]:
            by_base[base][kind] = torch.zeros_like(tensor, dtype=torch.int64)
        by_base[base][kind] += tensor

    load_rows = []
    summary_rows = []
    for (phase, gid, layer_no), data in sorted(by_base.items()):
        if "pre_drop" not in data or "post_drop" not in data:
            raise RuntimeError(
                f"Missing pre/post expert counts for phase={phase} gid={gid} layer={layer_no}"
            )
        pre = data["pre_drop"].to(torch.int64)
        post = data["post_drop"].to(torch.int64)
        if pre.shape != post.shape:
            raise RuntimeError(
                f"pre/post count shape mismatch for phase={phase} gid={gid} layer={layer_no}"
            )
        if torch.any(post > pre):
            raise RuntimeError(
                "post-drop expert counts exceed pre-drop counts for "
                f"phase={phase} gid={gid} layer={layer_no}"
            )
        dropped = pre - post
        for load_kind, values in (("pre_drop", pre), ("post_drop", post), ("dropped", dropped)):
            for expert_id, token_count in enumerate(values.tolist()):
                load_rows.append(
                    (
                        args.model_id,
                        args.run_name,
                        phase,
                        gid,
                        gid,
                        int(layer_no),
                        int(expert_id),
                        load_kind,
                        int(token_count),
                    )
                )
        post_sum = int(post.sum().item())
        pre_sum = int(pre.sum().item())
        active = int((post > 0).sum().item())
        dead = int((post == 0).sum().item())
        max_share = float(post.max().item() / post_sum) if post_sum > 0 else 0.0
        drop_rate = float(dropped.sum().item() / pre_sum) if pre_sum > 0 else 0.0
        summary_rows.append(
            (
                args.model_id,
                args.run_name,
                phase,
                gid,
                gid,
                int(layer_no),
                active,
                dead,
                coefficient_of_variation(post),
                max_share if math.isfinite(max_share) else 0.0,
                drop_rate if math.isfinite(drop_rate) else 0.0,
            )
        )
    return load_rows, summary_rows


def _check_expected_iterations(
    args: argparse.Namespace,
    rows: list[tuple[Any, ...]],
    label: str,
    accepted_iteration_ids: Sequence[int],
) -> None:
    if args.expected_train_iters <= 0:
        return
    gids = {int(row[3]) for row in rows}
    expected = set(int(item) for item in accepted_iteration_ids)
    if gids != expected:
        raise RuntimeError(
            f"{label} iteration IDs are {sorted(gids)}, "
            f"expected accepted IDs {sorted(expected)}"
        )


def _check_scalar_metrics(
    args: argparse.Namespace,
    rows: list[tuple[Any, ...]],
    accepted_iteration_ids: Sequence[int],
) -> None:
    required = {"lm_loss_iteration", "grad_norm"}
    expected = set(int(item) for item in accepted_iteration_ids)
    by_metric: dict[str, list[int]] = defaultdict(list)
    for row in rows:
        by_metric[str(row[5])].append(int(row[3]))
    for metric in required:
        ids = by_metric.get(metric, [])
        if len(ids) != len(set(ids)):
            raise RuntimeError(f"training_scalar metric={metric!r} contains duplicate iterations")
        if set(ids) != expected:
            raise RuntimeError(
                f"training_scalar metric={metric!r} IDs are {sorted(ids)}, "
                f"expected accepted IDs {sorted(expected)}"
            )


SampleCoordinate = tuple[int, int, int, int]
DatasetIteration = tuple[int, int]


def _read_router_probability_samples(
    client: Any,
    args: argparse.Namespace,
    accepted_iteration_ids: Sequence[int],
) -> dict[DatasetIteration, list[tuple[SampleCoordinate, torch.Tensor]]]:
    rows = _raw_execute(
        client,
        f"""
        SELECT
          direction, phase, global_batch_id, dp_rank, microbatch_id,
          sample_index, layer_no, shard_rank, attempt_id, invocation_id,
          dataset_id, dtype, shape, bytes
        FROM {_q(args.raw_db)}.{_q(args.raw_table)}
        WHERE model_id = %(model_id)s
          AND act_name = %(act_name)s
          AND phase = %(phase)s
          AND direction = 'fwd'
          AND invocation_id = 0
          AND (global_batch_id, attempt_id) IN ({_accepted_attempt_subquery(args)})
        ORDER BY global_batch_id, dp_rank, microbatch_id,
                 sample_index, shard_rank, layer_no
        """,
        {
            "model_id": args.model_id,
            "act_name": ROUTER_PROBS_ACT,
            "phase": args.phase,
        },
    )
    grouped: dict[
        tuple[int, SampleCoordinate], dict[int, tuple[int, torch.Tensor]]
    ] = defaultdict(dict)
    for row in rows:
        direction = str(_decode_cell(row[0]))
        phase = str(_decode_cell(row[1]))
        if direction != "fwd" or phase != args.phase:
            raise RuntimeError(
                f"Unexpected router-probability identity: direction={direction!r}, phase={phase!r}"
            )
        gid = int(row[2])
        coordinate = (int(row[3]), int(row[4]), int(row[5]), int(row[7]))
        layer_no = int(row[6])
        dataset_id = int(row[10])
        if dataset_id < -1:
            raise RuntimeError(
                f"Invalid router-probability dataset_id={dataset_id} at "
                f"gid={gid}, sample={coordinate}, layer={layer_no}"
            )
        layers = grouped[(gid, coordinate)]
        if layer_no in layers:
            raise RuntimeError(
                f"Duplicate router probability row for gid={gid}, sample={coordinate}, "
                f"layer={layer_no}"
            )
        tensor = _decode_tensor(row[11], row[12], row[13]).float().reshape(-1)
        if tensor.numel() != int(args.expected_expert_count):
            raise RuntimeError(
                f"Router probability gid={gid}, layer={layer_no} has "
                f"{tensor.numel()} experts; expected {args.expected_expert_count}"
            )
        if not torch.isfinite(tensor).all() or torch.any(tensor < 0):
            raise RuntimeError(
                f"Router probability gid={gid}, layer={layer_no} is nonfinite or negative"
            )
        layers[layer_no] = (dataset_id, tensor)

    by_dataset_iteration: dict[
        DatasetIteration, list[tuple[SampleCoordinate, torch.Tensor]]
    ] = defaultdict(list)
    expected_layers = list(range(int(args.expected_layer_count)))
    for (gid, coordinate), layer_map in sorted(grouped.items()):
        if sorted(layer_map) != expected_layers:
            raise RuntimeError(
                f"Router sample gid={gid}, coordinate={coordinate} has layers "
                f"{sorted(layer_map)}, expected {expected_layers}"
            )
        dataset_ids = {layer_map[layer][0] for layer in expected_layers}
        if len(dataset_ids) != 1:
            raise RuntimeError(
                f"Router sample gid={gid}, coordinate={coordinate} has inconsistent "
                f"dataset IDs across layers: {sorted(dataset_ids)}"
            )
        dataset_id = dataset_ids.pop()
        by_dataset_iteration[(dataset_id, gid)].append(
            (
                coordinate,
                torch.stack(
                    [layer_map[layer][1] for layer in expected_layers], dim=0
                ),
            )
        )

    expected_iterations = {int(value) for value in accepted_iteration_ids}
    represented_iterations = {gid for _dataset_id, gid in by_dataset_iteration}
    if represented_iterations != expected_iterations:
        raise RuntimeError(
            f"Router probability iterations are {sorted(represented_iterations)}, "
            f"expected accepted IDs {sorted(expected_iterations)}"
        )
    sample_count_by_iteration: dict[int, int] = defaultdict(int)
    for (_dataset_id, gid), samples in by_dataset_iteration.items():
        samples.sort(key=lambda item: item[0])
        sample_count_by_iteration[gid] += len(samples)
    for gid, sample_count in sample_count_by_iteration.items():
        if sample_count != int(args.expected_samples_per_iteration):
            raise RuntimeError(
                f"Router probability iteration {gid} has {sample_count} samples, "
                f"expected {args.expected_samples_per_iteration}"
            )
    return dict(by_dataset_iteration)


def _read_router_weight_state(
    client: Any,
    args: argparse.Namespace,
    *,
    state_id: int,
    accepted_attempts: dict[int, int],
) -> dict[int, torch.Tensor]:
    rows = _raw_execute(
        client,
        f"""
        SELECT
          direction, phase, global_batch_id, dp_rank, microbatch_id,
          sample_index, layer_no, shard_rank, token_start, token_end,
          attempt_id, invocation_id, dataset_id, dtype, shape, bytes
        FROM {_q(args.raw_db)}.{_q(args.raw_table)}
        WHERE model_id = %(model_id)s
          AND act_name = %(act_name)s
          AND global_batch_id = %(state_id)s
          AND invocation_id = 0
        ORDER BY layer_no
        """,
        {
            "model_id": args.model_id,
            "act_name": ROUTER_WEIGHT_ACT,
            "state_id": state_id,
        },
    )
    expected_attempt_id = 0 if state_id == 0 else accepted_attempts.get(state_id)
    if expected_attempt_id is None:
        raise RuntimeError(
            f"Router weight state {state_id} does not belong to an accepted attempt"
        )
    expected_identity = (
        "iter",
        "train",
        state_id,
        -1,
        -1,
        -1,
        0,
        0,
        1,
        expected_attempt_id,
        0,
        -1,
    )
    weights: dict[int, torch.Tensor] = {}
    for row in rows:
        layer_no = int(row[6])
        identity = (
            str(_decode_cell(row[0])),
            str(_decode_cell(row[1])),
            int(row[2]),
            int(row[3]),
            int(row[4]),
            int(row[5]),
            int(row[7]),
            int(row[8]),
            int(row[9]),
            int(row[10]),
            int(row[11]),
            int(row[12]),
        )
        if identity != expected_identity:
            raise RuntimeError(
                f"Router weight state={state_id}, layer={layer_no} has identity={identity}, "
                f"expected={expected_identity}"
            )
        if layer_no in weights:
            raise RuntimeError(
                f"Duplicate router weight row for state={state_id}, layer={layer_no}"
            )
        tensor = _decode_tensor(row[13], row[14], row[15])
        expected_shape = (
            int(args.expected_expert_count),
            int(args.expected_hidden_size),
        )
        if not tensor.dtype.is_floating_point or tuple(tensor.shape) != expected_shape:
            raise RuntimeError(
                f"Router weight state={state_id}, layer={layer_no} has "
                f"dtype={tensor.dtype}, shape={tuple(tensor.shape)}, "
                f"expected floating {expected_shape}"
            )
        weights[layer_no] = tensor
    expected_layers = set(range(int(args.expected_layer_count)))
    if set(weights) != expected_layers:
        raise RuntimeError(
            f"Router weight state {state_id} has layers {sorted(weights)}, "
            f"expected {sorted(expected_layers)}"
        )
    return weights


def _latest_router_weight_state_id(
    client: Any,
    args: argparse.Namespace,
    *,
    before_iteration: int,
) -> int:
    rows = _raw_execute(
        client,
        f"""
        SELECT maxOrNull(global_batch_id)
        FROM {_q(args.raw_db)}.{_q(args.raw_table)}
        WHERE model_id = %(model_id)s
          AND act_name = %(act_name)s
          AND direction = 'iter'
          AND phase = 'train'
          AND invocation_id = 0
          AND dataset_id = -1
          AND global_batch_id < %(before_iteration)s
          AND (
            (global_batch_id = 0 AND attempt_id = 0)
            OR (global_batch_id, attempt_id) IN ({_accepted_attempt_subquery(args)})
          )
        """,
        {
            "model_id": args.model_id,
            "act_name": ROUTER_WEIGHT_ACT,
            "before_iteration": int(before_iteration),
        },
    )
    state_id = rows[0][0] if rows else None
    if state_id is None:
        raise RuntimeError(
            f"No router-weight snapshot precedes training iteration {before_iteration}"
        )
    return int(state_id)


def _materialize_pathway_windows(
    args: argparse.Namespace,
    *,
    completed_iteration_ids: Sequence[int],
    consistency_by_dataset_iteration: dict[DatasetIteration, list[float]],
    pathways_by_dataset_iteration: dict[
        DatasetIteration, list[tuple[int, ...]]
    ],
    window_sizes: Sequence[int],
) -> tuple[list[tuple[Any, ...]], list[tuple[Any, ...]]]:
    return materialize_pathway_windows(
        model_id=args.model_id,
        run_name=args.run_name,
        phase=args.phase,
        completed_iteration_ids=completed_iteration_ids,
        consistency_eps=float(args.consistency_eps),
        pathway_threshold=float(args.pathway_threshold),
        consistency_by_dataset_iteration=consistency_by_dataset_iteration,
        pathways_by_dataset_iteration=pathways_by_dataset_iteration,
        window_sizes=window_sizes,
    )


def _materialize_pathways(
    client: Any,
    args: argparse.Namespace,
    accepted_attempts: dict[int, int],
) -> tuple[list[tuple[Any, ...]], list[tuple[Any, ...]]]:
    completed_iteration_ids = sorted(accepted_attempts)
    samples_by_dataset_iteration = _read_router_probability_samples(
        client,
        args,
        completed_iteration_ids,
    )
    consistency_by_dataset_iteration: dict[DatasetIteration, list[float]] = {}
    pathways_by_dataset_iteration: dict[
        DatasetIteration, list[tuple[int, ...]]
    ] = {}

    layers = list(range(int(args.expected_layer_count)))
    weights_by_iteration: dict[int, dict[int, torch.Tensor]] = {}
    for iteration in completed_iteration_ids:
        state_id = _latest_router_weight_state_id(
            client,
            args,
            before_iteration=iteration,
        )
        weights_by_iteration[iteration] = _read_router_weight_state(
            client,
            args,
            state_id=state_id,
            accepted_attempts=accepted_attempts,
        )

    for (dataset_id, iteration), samples in sorted(
        samples_by_dataset_iteration.items()
    ):
        data = torch.stack([tensor for _coordinate, tensor in samples], dim=0)
        scores = pathway_consistency(
            data,
            layers,
            weights_by_iteration[iteration],
            eps=float(args.consistency_eps),
        )
        score_values = [float(value) for value in scores.tolist()]
        if any(not math.isfinite(value) for value in score_values):
            raise RuntimeError(f"Nonfinite pathway consistency at iteration {iteration}")
        consistency_by_dataset_iteration[(dataset_id, iteration)] = score_values

        iteration_pathways = []
        for sample_tensor in data:
            layer_tokens = [
                pathway_tokens(sample_tensor[layer_no], float(args.pathway_threshold))
                for layer_no in layers
            ]
            iteration_pathways.append(paper_pathway_sequence(layer_tokens))
        pathways_by_dataset_iteration[(dataset_id, iteration)] = iteration_pathways

    window_sizes = tuple(sorted(set(args.pathway_window_size or DEFAULT_PATHWAY_WINDOWS)))
    return _materialize_pathway_windows(
        args,
        completed_iteration_ids=completed_iteration_ids,
        consistency_by_dataset_iteration=consistency_by_dataset_iteration,
        pathways_by_dataset_iteration=pathways_by_dataset_iteration,
        window_sizes=window_sizes,
    )


def _replace_run_rows(
    client: Any,
    args: argparse.Namespace,
    tables: Iterable[str],
) -> None:
    for table in tables:
        client.execute(
            f"""
            ALTER TABLE {_q(args.processed_db)}.{_q(table)}
            DELETE WHERE model_id = %(model_id)s
              AND run_name = %(run_name)s
              AND phase = %(phase)s
            """,
            {
                "model_id": args.model_id,
                "run_name": args.run_name,
                "phase": args.phase,
            },
            settings={"mutations_sync": 2},
        )


def _insert_rows(
    client: Any,
    *,
    processed_db: str,
    table: str,
    rows: list[tuple[Any, ...]],
    batch_rows: int,
) -> None:
    if batch_rows <= 0:
        raise ValueError("insert batch size must be positive")
    columns = ", ".join(_q(name) for name, _column_type in TABLE_COLUMNS[table])
    for start in range(0, len(rows), batch_rows):
        client.execute(
            f"INSERT INTO {_q(processed_db)}.{_q(table)} ({columns}) VALUES",
            rows[start : start + batch_rows],
        )


def _validate_inserted_counts(
    client: Any,
    args: argparse.Namespace,
    rows_by_table: dict[str, list[tuple[Any, ...]]],
) -> None:
    for table, rows in rows_by_table.items():
        actual = int(
            client.execute(
                f"""
                SELECT count()
                FROM {_q(args.processed_db)}.{_q(table)}
                WHERE model_id = %(model_id)s
                  AND run_name = %(run_name)s
                  AND phase = %(phase)s
                """,
                {
                    "model_id": args.model_id,
                    "run_name": args.run_name,
                    "phase": args.phase,
                },
            )[0][0]
        )
        if actual != len(rows):
            raise RuntimeError(
                f"Committed {args.processed_db}.{table} row count is {actual}, "
                f"expected {len(rows)}"
            )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.raw_db == args.processed_db:
        raise ValueError("--raw-db and --processed-db must be different databases")
    if args.expected_train_iters <= 0:
        raise ValueError("--expected-train-iters must be positive")
    if args.expected_layer_count <= 1:
        raise ValueError("--expected-layer-count must be greater than one")
    if (
        args.expected_expert_count <= 0
        or args.expected_hidden_size <= 0
        or args.expected_samples_per_iteration <= 0
    ):
        raise ValueError("Expected expert, hidden, and sample counts must be positive")
    if not 0.0 < args.pathway_threshold <= 1.0:
        raise ValueError("--pathway-threshold must be in (0, 1]")
    if not math.isfinite(args.consistency_eps) or args.consistency_eps <= 0.0:
        raise ValueError("--consistency-eps must be finite and positive")
    if args.phase != "train":
        raise ValueError("Version-2 training materialization currently requires phase=train")

    client = _client(args)
    try:
        accepted_attempts = _read_accepted_attempts(client, args)
        accepted_iteration_ids = sorted(accepted_attempts)
        if len(accepted_iteration_ids) != int(args.expected_train_iters):
            raise RuntimeError(
                f"Found {len(accepted_iteration_ids)} accepted training iterations "
                f"{accepted_iteration_ids}, expected {args.expected_train_iters}"
            )
        tables = _create_tables(
            client,
            args.processed_db,
            drop_existing=bool(args.drop_existing),
            include_pathways=bool(args.materialize_pathways),
        )

        counts = _read_expert_counts(client, args)
        scalar_raw_rows = _read_scalar_rows(client, args)
        scalar_int_raw_rows = _read_scalar_int_rows(client, args)
        training_scalar_rows = _materialize_direct_iteration_scalars(
            scalar_raw_rows, args
        )
        training_scalar_rows.extend(
            _materialize_iteration_loss_from_samples(
                scalar_raw_rows,
                scalar_int_raw_rows,
                args,
            )
        )
        training_scalar_rows.sort(key=lambda row: (row[3], row[5]))
        router_entropy_rows = _materialize_router_entropy(scalar_raw_rows, args)
        expert_load_rows, expert_summary_rows = _materialize_expert_load(counts, args)

        _check_expected_iterations(
            args,
            training_scalar_rows,
            "training_scalar",
            accepted_iteration_ids,
        )
        _check_expected_iterations(
            args,
            expert_summary_rows,
            "expert_load_summary",
            accepted_iteration_ids,
        )
        _check_expected_iterations(
            args,
            router_entropy_rows,
            "router_entropy",
            accepted_iteration_ids,
        )
        _check_scalar_metrics(args, training_scalar_rows, accepted_iteration_ids)

        rows_by_table: dict[str, list[tuple[Any, ...]]] = {
            "training_scalar": training_scalar_rows,
            "expert_load": expert_load_rows,
            "expert_load_summary": expert_summary_rows,
            "router_entropy": router_entropy_rows,
        }
        if args.materialize_pathways:
            consistency_rows, distance_rows = _materialize_pathways(
                client,
                args,
                accepted_attempts,
            )
            rows_by_table["pathway_consistency"] = consistency_rows
            rows_by_table["pathway_edit_distance"] = distance_rows

        if not args.drop_existing:
            _replace_run_rows(client, args, tables)
        for table in tables:
            _insert_rows(
                client,
                processed_db=args.processed_db,
                table=table,
                rows=rows_by_table[table],
                batch_rows=int(args.insert_batch_rows),
            )
        _validate_inserted_counts(client, args, rows_by_table)
    finally:
        client.disconnect()

    summary = {table: len(rows) for table, rows in rows_by_table.items()}
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
