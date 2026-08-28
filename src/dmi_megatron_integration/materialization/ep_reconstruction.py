"""Offline reconstruction of dropless Megatron MoE expert outputs.

This module contains only the topology-normalized reconstruction kernel.  It
does not query ClickHouse or define the JSON representation of the frozen run
manifest.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import torch


@dataclass(frozen=True)
class MoEExecutionKey:
    """Execution coordinates shared across the four MoE payload kinds.

    ``invocation_id`` is intentionally absent: DMI allocates it independently
    for each hook runtime, so it cannot associate records emitted by different
    hooks.  Under the supported one-MoE-invocation-per-coordinate contract,
    repeated rows at this key are ambiguous and must be rejected by the row
    assembler.
    """

    model_id: str
    phase: str
    global_batch_id: int
    attempt_id: int
    microbatch_id: int
    layer_no: int
    direction: str


@dataclass(frozen=True, order=True)
class SourceTokenCoordinate:
    """Global semantic identity of one source token within a dense-DP domain."""

    dataset_id: int
    sample_index: int
    token_index: int


@dataclass(frozen=True)
class MoEParallelTopology:
    """Topology-normalized subset of the frozen run manifest.

    ``ep_groups`` are ordered at fixed ETP coordinates. ``etp_groups`` are
    ordered at fixed EP coordinates. ``dispatch_groups`` are the actual
    ETP-by-EP producer sets and prevent joins across expert-DP domains.
    """

    context_parallel_size: int
    top_k: int
    global_expert_to_ep_rank: tuple[int, ...]
    local_expert_order_by_ep_rank: Mapping[int, tuple[int, ...]]
    ep_groups: tuple[tuple[int, ...], ...]
    etp_groups: tuple[tuple[int, ...], ...]
    dispatch_groups: tuple[tuple[int, ...], ...]
    dense_dp_rank_by_global_rank: Mapping[int, int]
    expert_dp_rank_by_global_rank: Mapping[int, int]
    dispatcher_type: str = "alltoall"
    permutation_mode: str = "non_fused"
    dropless: bool = True
    padded: bool = False

    def __post_init__(self) -> None:
        if self.context_parallel_size <= 0:
            raise ValueError("context_parallel_size must be positive")
        if self.top_k <= 0:
            raise ValueError("top_k must be positive")
        if not self.global_expert_to_ep_rank:
            raise ValueError("global_expert_to_ep_rank must not be empty")
        if self.top_k > len(self.global_expert_to_ep_rank):
            raise ValueError("top_k cannot exceed the global expert count")

        def partition(groups: tuple[tuple[int, ...], ...], name: str) -> set[int]:
            if not groups:
                raise ValueError(f"{name} must not be empty")
            seen: set[int] = set()
            for group in groups:
                if not group:
                    raise ValueError(f"{name} contains an empty group")
                if len(set(group)) != len(group):
                    raise ValueError(f"{name} contains a duplicate rank")
                overlap = seen.intersection(group)
                if overlap:
                    raise ValueError(
                        f"{name} assigns ranks to multiple groups: {sorted(overlap)}"
                    )
                seen.update(group)
            return seen

        dispatch_ranks = partition(self.dispatch_groups, "dispatch_groups")
        ep_ranks = partition(self.ep_groups, "ep_groups")
        etp_ranks = partition(self.etp_groups, "etp_groups")
        if ep_ranks != dispatch_ranks or etp_ranks != dispatch_ranks:
            raise ValueError(
                "EP, ETP, and dispatch groups must partition the same producer ranks"
            )

        ep_sizes = {len(group) for group in self.ep_groups}
        etp_sizes = {len(group) for group in self.etp_groups}
        if len(ep_sizes) != 1 or len(etp_sizes) != 1:
            raise ValueError("EP and ETP group sizes must be uniform")
        ep_size = next(iter(ep_sizes))
        etp_size = next(iter(etp_sizes))

        dispatch_index: dict[int, int] = {}
        for group_index, group in enumerate(self.dispatch_groups):
            if len(group) != ep_size * etp_size:
                raise ValueError(
                    "Each dispatch group must contain exactly EP-size times ETP-size ranks"
                )
            for rank in group:
                dispatch_index[rank] = group_index

        ep_rank_by_global_rank: dict[int, int] = {}
        for group in self.ep_groups:
            owning_dispatches = {dispatch_index[rank] for rank in group}
            if len(owning_dispatches) != 1:
                raise ValueError("An EP group crosses dispatch groups")
            for ep_rank, global_rank in enumerate(group):
                ep_rank_by_global_rank[global_rank] = ep_rank

        etp_rank_by_global_rank: dict[int, int] = {}
        for group in self.etp_groups:
            owning_dispatches = {dispatch_index[rank] for rank in group}
            if len(owning_dispatches) != 1:
                raise ValueError("An ETP group crosses dispatch groups")
            for etp_rank, global_rank in enumerate(group):
                etp_rank_by_global_rank[global_rank] = etp_rank

        for group in self.ep_groups:
            if len({etp_rank_by_global_rank[rank] for rank in group}) != 1:
                raise ValueError("An EP group must have one fixed ETP coordinate")
        for group in self.etp_groups:
            if len({ep_rank_by_global_rank[rank] for rank in group}) != 1:
                raise ValueError("An ETP group must have one fixed EP coordinate")

        for dispatch_group_index, dispatch_group in enumerate(self.dispatch_groups):
            member_set = set(dispatch_group)
            contained_ep_groups = [
                group for group in self.ep_groups if set(group).issubset(member_set)
            ]
            contained_etp_groups = [
                group for group in self.etp_groups if set(group).issubset(member_set)
            ]
            if len(contained_ep_groups) != etp_size:
                raise ValueError(
                    f"Dispatch group {dispatch_group_index} has the wrong EP-group count"
                )
            if len(contained_etp_groups) != ep_size:
                raise ValueError(
                    f"Dispatch group {dispatch_group_index} has the wrong ETP-group count"
                )

        if set(self.dense_dp_rank_by_global_rank) != dispatch_ranks:
            raise ValueError("dense-DP rank mapping must cover every producer exactly")
        if set(self.expert_dp_rank_by_global_rank) != dispatch_ranks:
            raise ValueError("expert-DP rank mapping must cover every producer exactly")
        for group in self.dispatch_groups:
            if len({self.expert_dp_rank_by_global_rank[rank] for rank in group}) != 1:
                raise ValueError("A dispatch group crosses expert-DP coordinates")

        expected_ep_ranks = set(range(ep_size))
        if set(self.local_expert_order_by_ep_rank) != expected_ep_ranks:
            raise ValueError("local expert order must be provided for every EP rank")
        seen_experts: set[int] = set()
        for ep_rank in range(ep_size):
            local_order = self.local_expert_order_by_ep_rank[ep_rank]
            if len(set(local_order)) != len(local_order):
                raise ValueError(f"EP rank {ep_rank} has duplicate local experts")
            expected = {
                expert
                for expert, owner in enumerate(self.global_expert_to_ep_rank)
                if owner == ep_rank
            }
            if set(local_order) != expected:
                raise ValueError(
                    f"EP rank {ep_rank} local expert order disagrees with placement"
                )
            seen_experts.update(local_order)
        if seen_experts != set(range(len(self.global_expert_to_ep_rank))):
            raise ValueError("Local expert orders do not cover every global expert")


@dataclass(frozen=True)
class RouterExpertIdsShard:
    key: MoEExecutionKey
    producer_rank: int
    dense_dp_rank: int
    token_coordinates: tuple[SourceTokenCoordinate, ...]
    tensor: torch.Tensor
    source_flat_indices: tuple[int, ...] = ()


@dataclass(frozen=True)
class RouterWeightsShard:
    key: MoEExecutionKey
    producer_rank: int
    dense_dp_rank: int
    token_coordinates: tuple[SourceTokenCoordinate, ...]
    tensor: torch.Tensor


@dataclass(frozen=True)
class InverseMapShard:
    key: MoEExecutionKey
    producer_rank: int
    tensor: torch.Tensor


@dataclass(frozen=True)
class PackedWeightedOutputShard:
    key: MoEExecutionKey
    producer_rank: int
    tensor: torch.Tensor


@dataclass(frozen=True)
class ReconstructedSourceDomain:
    dense_dp_rank: int
    token_coordinates: tuple[SourceTokenCoordinate, ...]
    selected_expert_ids: torch.Tensor
    selected_weights: torch.Tensor
    weighted_outputs: torch.Tensor
    combined_output: torch.Tensor


@dataclass(frozen=True)
class ReconstructedMoEInvocation:
    key: MoEExecutionKey
    source_domains: tuple[ReconstructedSourceDomain, ...]


def reconstruct_moe_invocation(
    topology: MoEParallelTopology,
    *,
    expert_id_shards: Sequence[RouterExpertIdsShard],
    weight_shards: Sequence[RouterWeightsShard],
    inverse_map_shards: Sequence[InverseMapShard],
    packed_output_shards: Sequence[PackedWeightedOutputShard],
) -> ReconstructedMoEInvocation:
    """Reconstruct one logical MoE invocation for the supported CP=1 path."""

    if topology.context_parallel_size != 1:
        raise NotImplementedError(
            "MoE reconstruction currently supports only context_parallel_size == 1"
        )
    if topology.dispatcher_type != "alltoall":
        raise ValueError("Only the AlltoAll dispatcher is supported")
    if topology.permutation_mode != "non_fused":
        raise ValueError("Only the non-fused permutation path is supported")
    if not topology.dropless or topology.padded:
        raise ValueError("Reconstruction requires dropless, unpadded MoE execution")

    record_sequences = (
        tuple(expert_id_shards),
        tuple(weight_shards),
        tuple(inverse_map_shards),
        tuple(packed_output_shards),
    )
    if any(not records for records in record_sequences):
        raise ValueError("All four captured payload kinds are required")
    keys = {record.key for records in record_sequences for record in records}
    if len(keys) != 1:
        raise ValueError("Captured payloads do not share one execution key")
    execution_key = next(iter(keys))

    def records_by_rank(records: Sequence[object], name: str) -> dict[int, object]:
        result: dict[int, object] = {}
        for record in records:
            producer_rank = int(getattr(record, "producer_rank"))
            if producer_rank in result:
                raise ValueError(f"Duplicate {name} payload for rank {producer_rank}")
            result[producer_rank] = record
        return result

    ids_by_rank = records_by_rank(expert_id_shards, "expert-ID")
    weights_by_rank = records_by_rank(weight_shards, "router-weight")
    inverse_by_rank = records_by_rank(inverse_map_shards, "inverse-map")
    packed_by_rank = records_by_rank(packed_output_shards, "packed-output")
    producer_ranks = {
        rank for group in topology.dispatch_groups for rank in group
    }
    for name, records in (
        ("expert-ID", ids_by_rank),
        ("router-weight", weights_by_rank),
        ("inverse-map", inverse_by_rank),
        ("packed-output", packed_by_rank),
    ):
        if set(records) != producer_ranks:
            missing = sorted(producer_ranks.difference(records))
            extra = sorted(set(records).difference(producer_ranks))
            raise ValueError(f"{name} producer mismatch: missing={missing}, extra={extra}")

    integral_dtypes = {
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    }
    source_routes: dict[int, list[tuple[int, int, int]]] = {}
    source_shapes: dict[int, tuple[int, int]] = {}
    for producer_rank in sorted(producer_ranks):
        ids_record = ids_by_rank[producer_rank]
        weights_record = weights_by_rank[producer_rank]
        inverse_record = inverse_by_rank[producer_rank]
        assert isinstance(ids_record, RouterExpertIdsShard)
        assert isinstance(weights_record, RouterWeightsShard)
        assert isinstance(inverse_record, InverseMapShard)

        expected_dense_dp_rank = topology.dense_dp_rank_by_global_rank[producer_rank]
        if ids_record.dense_dp_rank != expected_dense_dp_rank:
            raise ValueError(f"Expert-ID dense-DP rank mismatch for rank {producer_rank}")
        if weights_record.dense_dp_rank != expected_dense_dp_rank:
            raise ValueError(f"Router-weight dense-DP rank mismatch for rank {producer_rank}")
        if ids_record.token_coordinates != weights_record.token_coordinates:
            raise ValueError(f"Router token coordinates disagree for rank {producer_rank}")
        if len(set(ids_record.token_coordinates)) != len(ids_record.token_coordinates):
            raise ValueError(f"Duplicate source token coordinate on rank {producer_rank}")

        ids = ids_record.tensor
        weights = weights_record.tensor
        inverse_map = inverse_record.tensor
        if ids.dtype not in integral_dtypes or ids.ndim != 2:
            raise ValueError("Expert IDs must be a rank-2 integer tensor")
        if ids.shape[1] != topology.top_k:
            raise ValueError("Expert-ID top-k dimension disagrees with topology")
        if ids.shape[0] != len(ids_record.token_coordinates):
            raise ValueError("Expert-ID token extent disagrees with token coordinates")
        source_flat_indices = (
            ids_record.source_flat_indices
            if ids_record.source_flat_indices
            else tuple(range(int(ids.shape[0])))
        )
        if len(source_flat_indices) != int(ids.shape[0]):
            raise ValueError("Source-flat-index extent disagrees with expert IDs")
        if len(set(source_flat_indices)) != len(source_flat_indices):
            raise ValueError("Source-flat indices must be unique on each producer")
        if any(index < 0 for index in source_flat_indices):
            raise ValueError("Source-flat indices must be nonnegative")
        if not weights.is_floating_point() or tuple(weights.shape) != tuple(ids.shape):
            raise ValueError("Router weights must be floating point and match expert IDs")
        if not bool(torch.isfinite(weights).all().item()):
            raise ValueError("Router weights contain a non-finite value")
        if inverse_map.dtype not in integral_dtypes or inverse_map.ndim != 1:
            raise ValueError("Inverse map must be a rank-1 integer tensor")
        route_count = int(ids.numel())
        if inverse_map.numel() != route_count:
            raise ValueError("Inverse-map extent disagrees with source route count")

        ids_i64 = ids.to(dtype=torch.int64)
        if ids_i64.numel():
            if int(ids_i64.min().item()) < 0 or int(ids_i64.max().item()) >= len(
                topology.global_expert_to_ep_rank
            ):
                raise ValueError("Expert ID is outside the configured global expert range")
        if topology.top_k > 1 and not bool((ids_i64[:, 1:] > ids_i64[:, :-1]).all().item()):
            raise ValueError("Selected expert IDs must be strictly increasing per token")

        # Megatron's inverse map stores the source-token index for each route
        # in expert-major order.  Keep the unique flattened route occurrence
        # separately because top-k routing repeats each source-token index.
        route_order = torch.argsort(ids_i64.reshape(-1), stable=True)
        expected_inverse = torch.tensor(
            [
                source_flat_indices[route_index // topology.top_k]
                for route_index in route_order.to(device="cpu").tolist()
            ],
            dtype=torch.int64,
        )
        if not torch.equal(
            inverse_map.to(dtype=torch.int64, device="cpu"),
            expected_inverse.to(device="cpu"),
        ):
            raise ValueError(f"Captured inverse map disagrees with routing on rank {producer_rank}")

        flat_ids = ids_i64.reshape(-1).to(device="cpu")
        source_routes[producer_rank] = [
            (producer_rank, original_route_index, int(flat_ids[original_route_index].item()))
            for original_route_index in route_order.to(device="cpu").tolist()
        ]
        source_shapes[producer_rank] = (int(ids.shape[0]), int(ids.shape[1]))

    ep_rank_by_global_rank: dict[int, int] = {}
    dispatch_index_by_global_rank: dict[int, int] = {}
    for dispatch_index, group in enumerate(topology.dispatch_groups):
        for rank in group:
            dispatch_index_by_global_rank[rank] = dispatch_index
    for group in topology.ep_groups:
        for ep_rank, global_rank in enumerate(group):
            ep_rank_by_global_rank[global_rank] = ep_rank

    local_layout_by_rank: dict[int, list[tuple[int, int, int]]] = {}
    for ep_group in topology.ep_groups:
        for destination_ep_rank, destination_rank in enumerate(ep_group):
            received_routes: list[tuple[int, int, int]] = []
            for source_rank in ep_group:
                received_routes.extend(
                    route
                    for route in source_routes[source_rank]
                    if topology.global_expert_to_ep_rank[route[2]]
                    == destination_ep_rank
                )
            local_order = topology.local_expert_order_by_ep_rank[
                destination_ep_rank
            ]
            local_expert_position = {
                global_expert: position
                for position, global_expert in enumerate(local_order)
            }
            local_layout_by_rank[destination_rank] = sorted(
                received_routes,
                key=lambda route: local_expert_position[route[2]],
            )

    route_outputs: dict[tuple[int, int], torch.Tensor] = {}
    output_tail_shape: tuple[int, ...] | None = None
    output_dtype: torch.dtype | None = None
    output_device: torch.device | None = None
    for etp_group in topology.etp_groups:
        if len({dispatch_index_by_global_rank[rank] for rank in etp_group}) != 1:
            raise ValueError("ETP group crosses dispatch groups")
        if len({ep_rank_by_global_rank[rank] for rank in etp_group}) != 1:
            raise ValueError("ETP group crosses destination EP coordinates")
        common_layout = [
            route
            for rank in etp_group
            for route in local_layout_by_rank[rank]
        ]
        destination_ep_rank = ep_rank_by_global_rank[etp_group[0]]
        local_expert_position = {
            global_expert: position
            for position, global_expert in enumerate(
                topology.local_expert_order_by_ep_rank[destination_ep_rank]
            )
        }
        # Megatron gathers the ETP/TP contributions before permutation 2.
        # Therefore the final packed rows are stably grouped by local expert
        # across the concatenated ETP rank order, not merely within each rank.
        common_layout = sorted(
            common_layout,
            key=lambda route: local_expert_position[route[2]],
        )
        partials: list[torch.Tensor] = []
        for producer_rank in etp_group:
            packed_record = packed_by_rank[producer_rank]
            assert isinstance(packed_record, PackedWeightedOutputShard)
            packed = packed_record.tensor
            if not packed.is_floating_point() or packed.ndim < 2:
                raise ValueError("Packed weighted output must be a floating tensor of rank >= 2")
            if int(packed.shape[0]) != len(common_layout):
                raise ValueError(
                    f"Packed row count disagrees with reconstructed layout on rank {producer_rank}"
                )
            tail_shape = tuple(int(value) for value in packed.shape[1:])
            if output_tail_shape is None:
                output_tail_shape = tail_shape
                output_dtype = packed.dtype
                output_device = packed.device
            elif (
                tail_shape != output_tail_shape
                or packed.dtype != output_dtype
                or packed.device != output_device
            ):
                raise ValueError("Packed output shapes, dtypes, and devices must agree")
            partials.append(packed)
        summed = partials[0].clone()
        for partial in partials[1:]:
            summed.add_(partial)
        for row_index, route in enumerate(common_layout):
            route_key = (route[0], route[1])
            if route_key in route_outputs:
                raise ValueError(f"Route {route_key} is assigned more than one packed row")
            route_outputs[route_key] = summed[row_index]

    expected_route_keys = {
        (producer_rank, route_index)
        for producer_rank, (token_count, top_k) in source_shapes.items()
        for route_index in range(token_count * top_k)
    }
    if set(route_outputs) != expected_route_keys:
        missing = sorted(expected_route_keys.difference(route_outputs))
        extra = sorted(set(route_outputs).difference(expected_route_keys))
        raise ValueError(f"Packed-row route bijection failed: missing={missing}, extra={extra}")

    rows_by_dense_dp: dict[
        int,
        list[
            tuple[
                SourceTokenCoordinate,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
            ]
        ],
    ] = {}
    for producer_rank in sorted(producer_ranks):
        ids_record = ids_by_rank[producer_rank]
        weights_record = weights_by_rank[producer_rank]
        assert isinstance(ids_record, RouterExpertIdsShard)
        assert isinstance(weights_record, RouterWeightsShard)
        token_count, top_k = source_shapes[producer_rank]
        domain_rows = rows_by_dense_dp.setdefault(ids_record.dense_dp_rank, [])
        for token_index, coordinate in enumerate(ids_record.token_coordinates):
            route_rows = torch.stack(
                [
                    route_outputs[(producer_rank, token_index * top_k + slot)]
                    for slot in range(top_k)
                ],
                dim=0,
            )
            domain_rows.append(
                (
                    coordinate,
                    ids_record.tensor[token_index],
                    weights_record.tensor[token_index],
                    route_rows,
                )
            )
    source_domains: list[ReconstructedSourceDomain] = []
    for dense_dp_rank in sorted(rows_by_dense_dp):
        rows = sorted(rows_by_dense_dp[dense_dp_rank], key=lambda row: row[0])
        coordinates = tuple(row[0] for row in rows)
        if len(set(coordinates)) != len(coordinates):
            raise ValueError(
                f"Dense-DP domain {dense_dp_rank} has duplicate global token coordinates"
            )
        selected_expert_ids = torch.stack([row[1] for row in rows], dim=0)
        selected_weights = torch.stack([row[2] for row in rows], dim=0)
        weighted_outputs = torch.stack([row[3] for row in rows], dim=0)
        source_domains.append(
            ReconstructedSourceDomain(
                dense_dp_rank=dense_dp_rank,
                token_coordinates=coordinates,
                selected_expert_ids=selected_expert_ids,
                selected_weights=selected_weights,
                weighted_outputs=weighted_outputs,
                combined_output=weighted_outputs.sum(dim=1),
            )
        )

    return ReconstructedMoEInvocation(
        key=execution_key,
        source_domains=tuple(source_domains),
    )
