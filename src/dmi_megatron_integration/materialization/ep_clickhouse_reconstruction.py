"""Assemble persisted Megatron EP payload rows for offline reconstruction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import torch

from .ep_reconstruction import (
    InverseMapShard,
    MoEExecutionKey,
    PackedWeightedOutputShard,
    ReconstructedMoEInvocation,
    RouterExpertIdsShard,
    RouterWeightsShard,
    SourceTokenCoordinate,
    reconstruct_moe_invocation,
)
from ..topology.ep_topology_manifest import (
    FrozenMegatronEPTopologyManifest,
)
from ..records.reader import TRAINING_ROW_COORDINATE_COLUMN_NAMES


_EXPERT_IDS = "router_topk_expert_ids"
_ROUTER_WEIGHTS = "router_topk_weights"
_INVERSE_MAP = "moe_inverse_map"
_PACKED_OUTPUT = "moe_packed_weighted_output"
_REQUIRED_ACT_NAMES = (_EXPERT_IDS, _ROUTER_WEIGHTS, _INVERSE_MAP, _PACKED_OUTPUT)


@dataclass(frozen=True)
class _TrainingTensorRow:
    key: MoEExecutionKey
    act_name: str
    dp_rank: int
    sample_index: int
    shard_rank: int
    token_start: int
    token_end: int
    dataset_id: int
    tensor: torch.Tensor


def _text(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _parse_row(coordinates: tuple, tensor: torch.Tensor) -> _TrainingTensorRow:
    if len(coordinates) != len(TRAINING_ROW_COORDINATE_COLUMN_NAMES):
        raise ValueError(
            "Training row coordinate count disagrees with the ClickHouse schema"
        )
    values = dict(zip(TRAINING_ROW_COORDINATE_COLUMN_NAMES, coordinates))
    key = MoEExecutionKey(
        model_id=_text(values["model_id"]),
        phase=_text(values["phase"]),
        global_batch_id=int(values["global_batch_id"]),
        attempt_id=int(values["attempt_id"]),
        microbatch_id=int(values["microbatch_id"]),
        layer_no=int(values["layer_no"]),
        direction=_text(values["direction"]),
    )
    return _TrainingTensorRow(
        key=key,
        act_name=_text(values["act_name"]),
        dp_rank=int(values["dp_rank"]),
        sample_index=int(values["sample_index"]),
        shard_rank=int(values["shard_rank"]),
        token_start=int(values["token_start"]),
        token_end=int(values["token_end"]),
        dataset_id=int(values["dataset_id"]),
        tensor=tensor,
    )


def _tp_coordinate(
    manifest: FrozenMegatronEPTopologyManifest, producer_rank: int
) -> tuple[int, int]:
    owning = [group for group in manifest.tp_groups if producer_rank in group]
    if len(owning) != 1:
        raise ValueError(f"Producer rank {producer_rank} has no unique TP group")
    group = owning[0]
    return group.index(producer_rank), len(group)


def _semantic_row_key(row: _TrainingTensorRow) -> tuple[int, int, int, int, int, int]:
    return (
        row.shard_rank,
        row.dp_rank,
        row.sample_index,
        row.token_start,
        row.token_end,
        row.dataset_id,
    )


def _unique_rows(
    rows: Sequence[_TrainingTensorRow], name: str
) -> dict[tuple[int, int, int, int, int, int], _TrainingTensorRow]:
    indexed: dict[tuple[int, int, int, int, int, int], _TrainingTensorRow] = {}
    for row in rows:
        position = _semantic_row_key(row)
        if position in indexed:
            raise ValueError(f"Duplicate {name} row at {position}")
        indexed[position] = row
    return indexed


def _assemble_router_shards(
    manifest: FrozenMegatronEPTopologyManifest,
    key: MoEExecutionKey,
    ids_rows: Sequence[_TrainingTensorRow],
    weight_rows: Sequence[_TrainingTensorRow],
) -> tuple[tuple[RouterExpertIdsShard, ...], tuple[RouterWeightsShard, ...]]:
    topology = manifest.topology_for_layer(key.layer_no)
    producer_ranks = {
        rank for dispatch_group in topology.dispatch_groups for rank in dispatch_group
    }
    ids_index = _unique_rows(ids_rows, "router expert-ID")
    weights_index = _unique_rows(weight_rows, "router weight")
    if set(ids_index) != set(weights_index):
        raise ValueError("Router expert-ID and weight row coordinates disagree")

    ids_by_rank: dict[int, list[_TrainingTensorRow]] = {}
    weights_by_rank: dict[int, list[_TrainingTensorRow]] = {}
    for position, ids_row in ids_index.items():
        ids_by_rank.setdefault(ids_row.shard_rank, []).append(ids_row)
        weights_by_rank.setdefault(ids_row.shard_rank, []).append(weights_index[position])
    if set(ids_by_rank) != producer_ranks:
        raise ValueError("Router rows do not cover exactly the manifest producer ranks")

    for tp_group in manifest.tp_groups:
        stage_members = tuple(rank for rank in tp_group if rank in producer_ranks)
        if not stage_members:
            continue
        if len(stage_members) != len(tp_group):
            raise ValueError("A TP group crosses the selected pipeline stage")
        reference = tuple(
            sorted(
                (
                    row.sample_index,
                    row.token_start,
                    row.token_end,
                    row.dataset_id,
                )
                for row in ids_by_rank[stage_members[0]]
            )
        )
        for producer_rank in stage_members[1:]:
            candidate = tuple(
                sorted(
                    (
                        row.sample_index,
                        row.token_start,
                        row.token_end,
                        row.dataset_id,
                    )
                    for row in ids_by_rank[producer_rank]
                )
            )
            if candidate != reference:
                raise ValueError(
                    f"TP peers disagree on router sample metadata in group {tp_group}"
                )

    ids_shards: list[RouterExpertIdsShard] = []
    weight_shards: list[RouterWeightsShard] = []
    local_seq_extent_by_rank: dict[int, int] = {}
    for producer_rank in sorted(producer_ranks):
        paired = sorted(
            zip(ids_by_rank[producer_rank], weights_by_rank[producer_rank]),
            key=lambda pair: pair[0].sample_index,
        )
        sample_indices = [ids_row.sample_index for ids_row, _ in paired]
        if sample_indices != list(range(len(sample_indices))):
            raise ValueError(
                f"Producer rank {producer_rank} router sample indices are not contiguous"
            )
        if not paired:
            raise ValueError(f"Producer rank {producer_rank} has no router rows")

        tp_rank, tp_size = _tp_coordinate(manifest, producer_rank)
        if tp_size > 1 and not manifest.sequence_parallel:
            raise ValueError("TP-sharded router rows require sequence parallelism")

        local_seq_extent: int | None = None
        dense_dp_rank = topology.dense_dp_rank_by_global_rank[producer_rank]
        for ids_row, weights_row in paired:
            if ids_row.dp_rank != dense_dp_rank or weights_row.dp_rank != dense_dp_rank:
                raise ValueError(
                    f"Producer rank {producer_rank} has the wrong dense-DP coordinate"
                )
            if ids_row.tensor.ndim != 2 or weights_row.tensor.ndim != 2:
                raise ValueError("Persisted router rows must have shape [local_seq, top_k]")
            if tuple(ids_row.tensor.shape) != tuple(weights_row.tensor.shape):
                raise ValueError("Persisted router ID and weight row shapes disagree")
            if int(ids_row.tensor.shape[1]) != topology.top_k:
                raise ValueError("Persisted router row top-k disagrees with the manifest")
            row_seq_extent = int(ids_row.tensor.shape[0])
            if local_seq_extent is None:
                local_seq_extent = row_seq_extent
            elif row_seq_extent != local_seq_extent:
                raise ValueError("Router rows on one producer have different sequence extents")
            valid_count = ids_row.token_end - ids_row.token_start
            if valid_count <= 0:
                raise ValueError("Persisted router rows must represent at least one valid token")
            if valid_count > row_seq_extent * tp_size:
                raise ValueError("Router valid-token count exceeds the reconstructed TP extent")

        assert local_seq_extent is not None
        local_seq_extent_by_rank[producer_rank] = local_seq_extent
        compact_ids: list[torch.Tensor] = []
        compact_weights: list[torch.Tensor] = []
        token_coordinates: list[SourceTokenCoordinate] = []
        source_flat_indices: list[int] = []
        batch_size = len(paired)
        for local_token_index in range(local_seq_extent):
            for ids_row, weights_row in paired:
                global_token_index = (
                    ids_row.token_start
                    + tp_rank * local_seq_extent
                    + local_token_index
                )
                if global_token_index >= ids_row.token_end:
                    continue
                compact_ids.append(ids_row.tensor[local_token_index])
                compact_weights.append(weights_row.tensor[local_token_index])
                token_coordinates.append(
                    SourceTokenCoordinate(
                        dataset_id=ids_row.dataset_id,
                        sample_index=ids_row.sample_index,
                        token_index=global_token_index,
                    )
                )
                source_flat_indices.append(
                    local_token_index * batch_size + ids_row.sample_index
                )

        ids_template = paired[0][0].tensor
        weights_template = paired[0][1].tensor
        ids_tensor = (
            torch.stack(compact_ids, dim=0)
            if compact_ids
            else ids_template.new_empty((0, topology.top_k))
        )
        weights_tensor = (
            torch.stack(compact_weights, dim=0)
            if compact_weights
            else weights_template.new_empty((0, topology.top_k))
        )
        ids_shards.append(
            RouterExpertIdsShard(
                key=key,
                producer_rank=producer_rank,
                dense_dp_rank=dense_dp_rank,
                token_coordinates=tuple(token_coordinates),
                tensor=ids_tensor,
                source_flat_indices=tuple(source_flat_indices),
            )
        )
        weight_shards.append(
            RouterWeightsShard(
                key=key,
                producer_rank=producer_rank,
                dense_dp_rank=dense_dp_rank,
                token_coordinates=tuple(token_coordinates),
                tensor=weights_tensor,
            )
        )

    ids_shard_by_rank = {shard.producer_rank: shard for shard in ids_shards}
    for tp_group in manifest.tp_groups:
        stage_members = tuple(rank for rank in tp_group if rank in producer_ranks)
        if not stage_members:
            continue
        peer_extents = {local_seq_extent_by_rank[rank] for rank in stage_members}
        if len(peer_extents) != 1:
            raise ValueError(
                f"TP/SP peers disagree on local sequence extent in group {tp_group}"
            )

        expected_coordinates = {
            SourceTokenCoordinate(
                dataset_id=row.dataset_id,
                sample_index=row.sample_index,
                token_index=token_index,
            )
            for row in ids_by_rank[stage_members[0]]
            for token_index in range(row.token_start, row.token_end)
        }
        actual_coordinates = [
            coordinate
            for rank in stage_members
            for coordinate in ids_shard_by_rank[rank].token_coordinates
        ]
        if len(actual_coordinates) != len(set(actual_coordinates)):
            raise ValueError(f"TP/SP token slices overlap in group {tp_group}")
        if set(actual_coordinates) != expected_coordinates:
            raise ValueError(
                f"TP/SP token slices do not exactly cover declared ranges in group {tp_group}"
            )
    return tuple(ids_shards), tuple(weight_shards)


def _execution_payloads_by_rank(
    rows: Sequence[_TrainingTensorRow],
    *,
    expected_ranks: set[int],
    name: str,
) -> dict[int, torch.Tensor]:
    result: dict[int, torch.Tensor] = {}
    for row in rows:
        if (
            row.dp_rank != -1
            or row.sample_index != -1
            or row.token_start != -1
            or row.token_end != -1
            or row.dataset_id != -1
        ):
            raise ValueError(f"{name} row does not use PER_EXECUTION coordinates")
        if row.shard_rank in result:
            raise ValueError(f"Duplicate {name} row for producer {row.shard_rank}")
        result[row.shard_rank] = row.tensor
    if set(result) != expected_ranks:
        raise ValueError(f"{name} rows do not cover exactly the manifest producer ranks")
    return result


def reconstruct_moe_clickhouse_rows(
    manifest: FrozenMegatronEPTopologyManifest,
    rows_by_act: Mapping[str, Sequence[tuple[tuple, torch.Tensor]]],
) -> tuple[ReconstructedMoEInvocation, ...]:
    """Reconstruct every complete MoE invocation in persisted training rows."""

    missing = [name for name in _REQUIRED_ACT_NAMES if name not in rows_by_act]
    if missing:
        raise ValueError(f"Missing MoE payload kinds: {missing}")

    grouped: dict[str, dict[MoEExecutionKey, list[_TrainingTensorRow]]] = {
        name: {} for name in _REQUIRED_ACT_NAMES
    }
    for act_name in _REQUIRED_ACT_NAMES:
        for coordinates, tensor in rows_by_act[act_name]:
            row = _parse_row(coordinates, tensor)
            if row.act_name != act_name:
                raise ValueError(
                    f"Row act_name {row.act_name!r} was supplied under {act_name!r}"
                )
            if row.key.model_id != manifest.model_id:
                raise ValueError("Training row model_id disagrees with the topology manifest")
            grouped[act_name].setdefault(row.key, []).append(row)

    key_sets = [set(grouped[name]) for name in _REQUIRED_ACT_NAMES]
    if not key_sets[0] or any(keys != key_sets[0] for keys in key_sets[1:]):
        raise ValueError("MoE payload kinds do not contain the same execution keys")

    def key_order(key: MoEExecutionKey) -> tuple[object, ...]:
        return (
            key.model_id,
            key.phase,
            key.global_batch_id,
            key.attempt_id,
            key.microbatch_id,
            key.layer_no,
            key.direction,
        )

    reconstructed: list[ReconstructedMoEInvocation] = []
    for key in sorted(key_sets[0], key=key_order):
        topology = manifest.topology_for_layer(key.layer_no)
        producer_ranks = {
            rank for dispatch_group in topology.dispatch_groups for rank in dispatch_group
        }
        expert_id_shards, weight_shards = _assemble_router_shards(
            manifest,
            key,
            grouped[_EXPERT_IDS][key],
            grouped[_ROUTER_WEIGHTS][key],
        )
        inverse_by_rank = _execution_payloads_by_rank(
            grouped[_INVERSE_MAP][key],
            expected_ranks=producer_ranks,
            name="inverse-map",
        )
        packed_by_rank = _execution_payloads_by_rank(
            grouped[_PACKED_OUTPUT][key],
            expected_ranks=producer_ranks,
            name="packed-output",
        )
        reconstructed.append(
            reconstruct_moe_invocation(
                topology,
                expert_id_shards=expert_id_shards,
                weight_shards=weight_shards,
                inverse_map_shards=tuple(
                    InverseMapShard(key, rank, inverse_by_rank[rank])
                    for rank in sorted(producer_ranks)
                ),
                packed_output_shards=tuple(
                    PackedWeightedOutputShard(key, rank, packed_by_rank[rank])
                    for rank in sorted(producer_ranks)
                ),
            )
        )
    return tuple(reconstructed)


__all__ = ["reconstruct_moe_clickhouse_rows"]
