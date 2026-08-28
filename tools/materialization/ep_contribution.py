#!/usr/bin/env python3
"""Derive expert-contribution metrics from reconstructed Megatron MoE rows."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Protocol, Sequence

import torch

from dmi_megatron_integration.materialization.ep_clickhouse_reconstruction import (
    reconstruct_moe_clickhouse_rows,
)
from dmi_megatron_integration.materialization.ep_reconstruction import (
    ReconstructedMoEInvocation,
)
from dmi_megatron_integration.topology.ep_topology_manifest import (
    FrozenMegatronEPTopologyManifest,
    load_ep_topology_manifest,
)
from dmi_megatron_integration.records.reader import MegatronTrainingReader


_ACT_NAMES = (
    "router_topk_expert_ids",
    "router_topk_weights",
    "moe_inverse_map",
    "moe_packed_weighted_output",
)
CSV_COLUMNS = (
    "global_moe_layer",
    "valid_token_count",
    "magnitude_tie_token_count",
    "router_rank_dominance_counts",
    "top_weight_tie_token_count",
    "top_route_alignment_q1",
    "top_route_alignment_median",
    "top_route_alignment_q3",
)


class _TrainingTensorReader(Protocol):
    def training_prefix_get(
        self,
        prefix_key: tuple,
        *,
        return_full_key_tuple: bool = True,
        include_all_attempts: bool = False,
        include_all_invocations: bool = False,
    ) -> list[tuple[tuple, torch.Tensor]]: ...


@dataclass(frozen=True)
class ExpertContributionRow:
    global_moe_layer: int
    valid_token_count: int
    magnitude_tie_token_count: int
    router_rank_dominance_counts: tuple[int, ...]
    top_weight_tie_token_count: int
    top_route_alignment_q1: float
    top_route_alignment_median: float
    top_route_alignment_q3: float


@dataclass(frozen=True)
class ExpertContributionValidation:
    schema_version: int
    status: str
    model_id: str
    phase: str
    expected_step_count: int
    observed_step_count: int
    expected_global_batch_ids: tuple[int, ...]
    observed_global_batch_ids: tuple[int, ...]
    expected_global_moe_layer_ids: tuple[int, ...]
    observed_global_moe_layer_ids: tuple[int, ...]
    expected_layer_count: int
    observed_layer_count: int
    expected_microbatches_per_step: int
    observed_microbatch_invocation_counts: tuple[tuple[int, int, int], ...]
    primary_forward_invocation_count: int
    source_domain_count: int
    output_row_count: int
    valid_token_count: int
    router_rank_1_dominance_count: int
    magnitude_tie_token_count: int
    top_weight_tie_token_count: int
    source_row_count_by_act: Mapping[str, int]


class ExpertContributionValidationError(ValueError):
    """A failed fixed-run grid validation with an auditable JSON payload."""

    def __init__(
        self, message: str, validation: ExpertContributionValidation
    ) -> None:
        super().__init__(message)
        self.validation = validation


def read_primary_forward_rows(
    reader: _TrainingTensorReader,
    *,
    model_id: str,
    phase: str = "train",
) -> dict[str, tuple[tuple[tuple, torch.Tensor], ...]]:
    """Read accepted-attempt, invocation-zero forward rows without modifying them."""

    if not model_id:
        raise ValueError("model_id must not be empty")
    if phase not in {"train", "valid", "test"}:
        raise ValueError("phase must be train, valid, or test")
    return {
        act_name: tuple(
            reader.training_prefix_get(
                (model_id, act_name, "fwd", phase),
                return_full_key_tuple=True,
                include_all_attempts=False,
                include_all_invocations=False,
            )
        )
        for act_name in _ACT_NAMES
    }


def _linear_quantile(values: Sequence[float], quantile: float) -> float:
    if not values:
        raise ValueError("Cannot take a quantile of an empty sequence")
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be within [0, 1]")
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


def analyze_expert_contributions(
    invocations: Sequence[ReconstructedMoEInvocation],
    *,
    expected_model_id: str,
    expected_phase: str = "train",
    expected_step_count: int,
    expected_layer_ids: Sequence[int],
    expected_microbatches_per_step: int,
    source_row_count_by_act: Mapping[str, int] | None = None,
) -> tuple[tuple[ExpertContributionRow, ...], ExpertContributionValidation]:
    """Aggregate token metrics by global MoE layer after validating the run grid."""

    if not invocations:
        raise ValueError("No reconstructed MoE invocations were provided")
    if not expected_model_id:
        raise ValueError("expected_model_id must not be empty")
    if expected_step_count <= 0:
        raise ValueError("expected_step_count must be positive")
    if expected_microbatches_per_step <= 0:
        raise ValueError("expected_microbatches_per_step must be positive")
    normalized_layer_ids = tuple(sorted(int(layer_id) for layer_id in expected_layer_ids))
    if not normalized_layer_ids or len(set(normalized_layer_ids)) != len(
        normalized_layer_ids
    ):
        raise ValueError("expected_layer_ids must contain unique layer IDs")

    seen_keys = set()
    attempts_by_batch: dict[int, set[int]] = defaultdict(set)
    invocations_by_group: dict[
        tuple[int, int], list[ReconstructedMoEInvocation]
    ] = defaultdict(list)
    for invocation in invocations:
        key = invocation.key
        if key in seen_keys:
            raise ValueError(f"Duplicate reconstructed invocation: {key}")
        seen_keys.add(key)
        if key.model_id != expected_model_id:
            raise ValueError("Reconstructed invocation model_id disagrees with the request")
        if key.phase != expected_phase or key.direction != "fwd":
            raise ValueError("Only forward invocations from the requested phase are valid")
        attempts_by_batch[key.global_batch_id].add(key.attempt_id)
        invocations_by_group[(key.global_batch_id, key.layer_no)].append(invocation)

    ambiguous_batches = sorted(
        batch for batch, attempts in attempts_by_batch.items() if len(attempts) != 1
    )
    if ambiguous_batches:
        raise ValueError(
            "More than one attempt was supplied for global batches "
            f"{ambiguous_batches}"
        )

    observed_batch_ids = tuple(
        sorted({batch_id for batch_id, _ in invocations_by_group})
    )
    first_batch_id = observed_batch_ids[0]
    expected_batch_ids = tuple(
        range(first_batch_id, first_batch_id + expected_step_count)
    )
    observed_layer_ids = tuple(
        sorted({layer_id for _, layer_id in invocations_by_group})
    )
    expected_grid = {
        (batch_id, layer_id)
        for batch_id in expected_batch_ids
        for layer_id in normalized_layer_ids
    }
    observed_grid = set(invocations_by_group)
    invocation_counts = tuple(
        (batch_id, layer_id, len(invocations_by_group[(batch_id, layer_id)]))
        for batch_id, layer_id in sorted(observed_grid)
    )

    row_counts = {
        name: int((source_row_count_by_act or {}).get(name, 0)) for name in _ACT_NAMES
    }

    def fail_grid(message: str) -> None:
        raise ExpertContributionValidationError(
            message,
            ExpertContributionValidation(
                schema_version=3,
                status="failed",
                model_id=expected_model_id,
                phase=expected_phase,
                expected_step_count=expected_step_count,
                observed_step_count=len(observed_batch_ids),
                expected_global_batch_ids=expected_batch_ids,
                observed_global_batch_ids=observed_batch_ids,
                expected_global_moe_layer_ids=normalized_layer_ids,
                observed_global_moe_layer_ids=observed_layer_ids,
                expected_layer_count=len(normalized_layer_ids),
                observed_layer_count=len(observed_layer_ids),
                expected_microbatches_per_step=expected_microbatches_per_step,
                observed_microbatch_invocation_counts=invocation_counts,
                primary_forward_invocation_count=len(invocations),
                source_domain_count=sum(
                    len(invocation.source_domains) for invocation in invocations
                ),
                output_row_count=0,
                valid_token_count=0,
                router_rank_1_dominance_count=0,
                magnitude_tie_token_count=0,
                top_weight_tie_token_count=0,
                source_row_count_by_act=row_counts,
            ),
        )

    if observed_batch_ids != expected_batch_ids:
        fail_grid(
            "Observed global_batch_id values do not match the required consecutive range: "
            f"expected={expected_batch_ids}, observed={observed_batch_ids}"
        )

    if observed_layer_ids != normalized_layer_ids:
        fail_grid(
            "Observed global MoE layer IDs disagree with the topology manifest: "
            f"expected={normalized_layer_ids}, observed={observed_layer_ids}"
        )

    if observed_grid != expected_grid:
        missing = sorted(expected_grid.difference(observed_grid))
        extra = sorted(observed_grid.difference(expected_grid))
        fail_grid(
            f"Incomplete batch-by-layer grid: missing={missing}, extra={extra}"
        )

    invalid_counts = tuple(
        count
        for count in invocation_counts
        if count[2] != expected_microbatches_per_step
    )
    if invalid_counts:
        fail_grid(
            "Batch-layer microbatch invocation counts disagree with the fixed run: "
            f"expected={expected_microbatches_per_step}, observed={invalid_counts}"
        )

    grouped: dict[int, dict[str, object]] = {}
    source_domain_count = 0

    for invocation in invocations:
        key = invocation.key
        group = grouped.setdefault(
            key.layer_no,
            {
                "valid": 0,
                "magnitude_ties": 0,
                "rank_counts": None,
                "top_weight_ties": 0,
                "alignments": [],
            },
        )
        for domain in invocation.source_domains:
            source_domain_count += 1
            weights = domain.selected_weights.detach().to(
                device="cpu", dtype=torch.float64
            )
            expert_ids = domain.selected_expert_ids.detach().to(device="cpu")
            weighted_outputs = domain.weighted_outputs.detach().to(
                device="cpu", dtype=torch.float64
            )
            combined = domain.combined_output.detach().to(
                device="cpu", dtype=torch.float64
            )

            if weights.ndim != 2 or tuple(expert_ids.shape) != tuple(weights.shape):
                raise ValueError("Selected expert IDs and weights must have shape [token, top_k]")
            token_count, top_k = (int(value) for value in weights.shape)
            if token_count == 0 or top_k == 0:
                raise ValueError("Every reconstructed source domain must contain valid tokens")
            if weighted_outputs.ndim < 3 or tuple(weighted_outputs.shape[:2]) != (
                token_count,
                top_k,
            ):
                raise ValueError(
                    "Weighted outputs must have shape [token, top_k, ...]"
                )
            if int(combined.shape[0]) != token_count or tuple(combined.shape[1:]) != tuple(
                weighted_outputs.shape[2:]
            ):
                raise ValueError("Combined-output shape disagrees with weighted outputs")
            if len(domain.token_coordinates) != token_count:
                raise ValueError("Token-coordinate extent disagrees with reconstructed tensors")
            if not bool(torch.isfinite(weights).all().item()):
                raise ValueError("Selected routing weights contain a non-finite value")
            if not bool(torch.isfinite(weighted_outputs).all().item()):
                raise ValueError("Weighted expert outputs contain a non-finite value")
            if not bool(torch.isfinite(combined).all().item()):
                raise ValueError("Combined expert outputs contain a non-finite value")

            flat_outputs = weighted_outputs.reshape(token_count, top_k, -1)
            flat_combined = combined.reshape(token_count, -1)
            contribution_norms = torch.linalg.vector_norm(flat_outputs, dim=2)
            row_indices = torch.arange(token_count, dtype=torch.int64)

            maximum_norms = contribution_norms.max(dim=1).values
            unique_magnitude = (
                contribution_norms == maximum_norms.unsqueeze(1)
            ).sum(dim=1) == 1
            contribution_slots = torch.argmax(contribution_norms, dim=1)
            winning_weights = weights[row_indices, contribution_slots]
            # Competition ranking gives tied router weights the same rank:
            # rank = 1 + the number of strictly larger weights.  The stored
            # count index is zero-based, so no arbitrary slot-order tie break
            # enters this statistic.
            dominant_router_ranks = (
                weights > winning_weights.unsqueeze(1)
            ).sum(dim=1)
            domain_rank_counts = torch.bincount(
                dominant_router_ranks[unique_magnitude],
                minlength=top_k,
            )
            magnitude_ties = int((~unique_magnitude).sum().item())

            maximum_weights = weights.max(dim=1).values
            unique_top_weight = (
                weights == maximum_weights.unsqueeze(1)
            ).sum(dim=1) == 1
            top_weight_ties = int((~unique_top_weight).sum().item())
            router_slots = torch.argmax(weights, dim=1)
            alignment_indices = row_indices[unique_top_weight]
            top_outputs = flat_outputs[
                alignment_indices,
                router_slots[unique_top_weight],
            ]
            alignment_combined = flat_combined[unique_top_weight]
            top_norms = torch.linalg.vector_norm(top_outputs, dim=1)
            combined_norms = torch.linalg.vector_norm(alignment_combined, dim=1)
            if bool((top_norms == 0).any().item()) or bool(
                (combined_norms == 0).any().item()
            ):
                raise ValueError("Cosine alignment has a zero-norm operand")
            alignments = (
                (top_outputs * alignment_combined).sum(dim=1)
                / (top_norms * combined_norms)
            )
            if not bool(torch.isfinite(alignments).all().item()):
                raise ValueError("Top-route alignment contains a non-finite value")

            group["valid"] = int(group["valid"]) + token_count
            group["magnitude_ties"] = (
                int(group["magnitude_ties"]) + magnitude_ties
            )
            group["top_weight_ties"] = (
                int(group["top_weight_ties"]) + top_weight_ties
            )
            rank_counts = group["rank_counts"]
            if rank_counts is None:
                rank_counts = [0] * top_k
                group["rank_counts"] = rank_counts
            if not isinstance(rank_counts, list) or len(rank_counts) != top_k:
                raise ValueError("top_k must remain constant within each global MoE layer")
            for rank, count in enumerate(domain_rank_counts.tolist()):
                rank_counts[rank] += int(count)
            cast_alignments = group["alignments"]
            assert isinstance(cast_alignments, list)
            cast_alignments.extend(float(value) for value in alignments.tolist())

    rows: list[ExpertContributionRow] = []
    for layer_no, values in sorted(grouped.items()):
        alignments = values["alignments"]
        assert isinstance(alignments, list)
        rank_counts = values["rank_counts"]
        if not isinstance(rank_counts, list) or not rank_counts:
            raise ValueError("Router-rank dominance counts are missing")
        valid_count = int(values["valid"])
        magnitude_tie_count = int(values["magnitude_ties"])
        top_weight_tie_count = int(values["top_weight_ties"])
        if magnitude_tie_count >= valid_count:
            raise ValueError("Every token has a tied maximum weighted-output norm")
        if sum(rank_counts) != valid_count - magnitude_tie_count:
            raise ValueError(
                "Router-rank dominance counts do not cover every unique maximum"
            )
        if top_weight_tie_count >= valid_count or not alignments:
            raise ValueError("Every token has a tied maximum router weight")
        q1 = _linear_quantile(alignments, 0.25)
        median = _linear_quantile(alignments, 0.5)
        q3 = _linear_quantile(alignments, 0.75)
        if not all(math.isfinite(value) for value in (q1, median, q3)):
            raise ValueError("Top-route alignment quantiles contain a non-finite value")
        rows.append(
            ExpertContributionRow(
                global_moe_layer=layer_no,
                valid_token_count=valid_count,
                magnitude_tie_token_count=magnitude_tie_count,
                router_rank_dominance_counts=tuple(rank_counts),
                top_weight_tie_token_count=top_weight_tie_count,
                top_route_alignment_q1=q1,
                top_route_alignment_median=median,
                top_route_alignment_q3=q3,
            )
        )

    validation = ExpertContributionValidation(
        schema_version=3,
        status="passed",
        model_id=expected_model_id,
        phase=expected_phase,
        expected_step_count=expected_step_count,
        observed_step_count=len(observed_batch_ids),
        expected_global_batch_ids=expected_batch_ids,
        observed_global_batch_ids=observed_batch_ids,
        expected_global_moe_layer_ids=normalized_layer_ids,
        observed_global_moe_layer_ids=observed_layer_ids,
        expected_layer_count=len(normalized_layer_ids),
        observed_layer_count=len(observed_layer_ids),
        expected_microbatches_per_step=expected_microbatches_per_step,
        observed_microbatch_invocation_counts=invocation_counts,
        primary_forward_invocation_count=len(invocations),
        source_domain_count=source_domain_count,
        output_row_count=len(rows),
        valid_token_count=sum(row.valid_token_count for row in rows),
        router_rank_1_dominance_count=sum(
            row.router_rank_dominance_counts[0] for row in rows
        ),
        magnitude_tie_token_count=sum(
            row.magnitude_tie_token_count for row in rows
        ),
        top_weight_tie_token_count=sum(
            row.top_weight_tie_token_count for row in rows
        ),
        source_row_count_by_act=row_counts,
    )
    return tuple(rows), validation


def analyze_clickhouse_rows(
    manifest: FrozenMegatronEPTopologyManifest,
    rows_by_act: Mapping[str, Sequence[tuple[tuple, torch.Tensor]]],
    *,
    phase: str = "train",
    expected_step_count: int,
    expected_microbatches_per_step: int,
) -> tuple[tuple[ExpertContributionRow, ...], ExpertContributionValidation]:
    """Reconstruct persisted rows and derive contribution metrics in memory."""

    invocations = reconstruct_moe_clickhouse_rows(manifest, rows_by_act)
    return analyze_expert_contributions(
        invocations,
        expected_model_id=manifest.model_id,
        expected_phase=phase,
        expected_step_count=expected_step_count,
        expected_layer_ids=tuple(
            placement.layer_no for placement in manifest.layer_placements
        ),
        expected_microbatches_per_step=expected_microbatches_per_step,
        source_row_count_by_act={name: len(rows_by_act.get(name, ())) for name in _ACT_NAMES},
    )


def write_expert_contribution_outputs(
    rows: Sequence[ExpertContributionRow],
    validation: ExpertContributionValidation,
    *,
    csv_path: str | os.PathLike[str],
    validation_json_path: str | os.PathLike[str],
) -> None:
    """Write deterministic CSV metrics and their validation summary."""

    csv_destination = Path(csv_path)
    json_destination = Path(validation_json_path)
    csv_destination.parent.mkdir(parents=True, exist_ok=True)
    json_destination.parent.mkdir(parents=True, exist_ok=True)
    with csv_destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            serialized = asdict(row)
            serialized["router_rank_dominance_counts"] = json.dumps(
                row.router_rank_dominance_counts,
                separators=(",", ":"),
            )
            writer.writerow(serialized)
    _write_validation_json(validation, json_destination)


def _write_validation_json(
    validation: ExpertContributionValidation,
    destination: str | os.PathLike[str],
) -> None:
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(asdict(validation), handle, indent=2, sort_keys=True)
        handle.write("\n")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--user", default="default")
    parser.add_argument("--password", default="")
    parser.add_argument("--database", required=True)
    parser.add_argument("--table", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--phase", default="train", choices=("train", "valid", "test"))
    parser.add_argument("--expected-step-count", type=int, default=10)
    parser.add_argument("--expected-microbatches-per-step", type=int, default=1)
    parser.add_argument("--topology-manifest", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--validation-json", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = load_ep_topology_manifest(args.topology_manifest)
    if manifest.model_id != args.model_id:
        raise ValueError("CLI model_id disagrees with the topology manifest")
    reader = MegatronTrainingReader(
        host=args.host,
        port=args.port,
        username=args.user,
        password=args.password,
        database=args.database,
        table=args.table,
    )
    try:
        rows_by_act = read_primary_forward_rows(
            reader,
            model_id=args.model_id,
            phase=args.phase,
        )
    finally:
        reader.close()
    try:
        rows, validation = analyze_clickhouse_rows(
            manifest,
            rows_by_act,
            phase=args.phase,
            expected_step_count=args.expected_step_count,
            expected_microbatches_per_step=args.expected_microbatches_per_step,
        )
    except ExpertContributionValidationError as exc:
        _write_validation_json(exc.validation, args.validation_json)
        raise
    write_expert_contribution_outputs(
        rows,
        validation,
        csv_path=args.output_csv,
        validation_json_path=args.validation_json,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CSV_COLUMNS",
    "ExpertContributionRow",
    "ExpertContributionValidation",
    "ExpertContributionValidationError",
    "analyze_clickhouse_rows",
    "analyze_expert_contributions",
    "main",
    "read_primary_forward_rows",
    "write_expert_contribution_outputs",
]
