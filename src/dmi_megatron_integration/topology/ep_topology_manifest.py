"""Frozen Megatron topology manifests for offline MoE reconstruction."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from ..materialization.ep_reconstruction import MoEParallelTopology


_SCHEMA_VERSION = 1
_GROUP_FIELDS = (
    "tp_groups",
    "pp_groups",
    "dp_groups",
    "cp_groups",
    "ep_groups",
    "etp_groups",
    "dispatch_groups",
    "expert_dp_groups",
)


def _validate_group(group: Sequence[int], name: str) -> tuple[int, ...]:
    normalized = tuple(int(rank) for rank in group)
    if not normalized:
        raise ValueError(f"{name} must not be empty")
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{name} contains duplicate ranks")
    return normalized


def _validate_partition(
    groups: Sequence[Sequence[int]],
    name: str,
    expected_ranks: set[int] | None = None,
) -> set[int]:
    if not groups:
        raise ValueError(f"{name} must not be empty")
    seen: set[int] = set()
    sizes: set[int] = set()
    for index, raw_group in enumerate(groups):
        group = _validate_group(raw_group, f"{name}[{index}]")
        overlap = seen.intersection(group)
        if overlap:
            raise ValueError(f"{name} assigns ranks more than once: {sorted(overlap)}")
        seen.update(group)
        sizes.add(len(group))
    if len(sizes) != 1:
        raise ValueError(f"{name} must have a uniform group size")
    if expected_ranks is not None and seen != expected_ranks:
        raise ValueError(f"{name} does not partition the run's global ranks")
    return seen


def _canonical_partition(
    local_groups: Iterable[Sequence[int]], name: str, expected_ranks: set[int]
) -> tuple[tuple[int, ...], ...]:
    unique = {_validate_group(group, name) for group in local_groups}
    groups = tuple(sorted(unique))
    _validate_partition(groups, name, expected_ranks)
    return groups


def _rank_in_group(global_rank: int, group: Sequence[int], name: str) -> int:
    try:
        return tuple(group).index(int(global_rank))
    except ValueError as exc:
        raise ValueError(f"global rank {global_rank} is absent from its {name}") from exc


def _require_bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a JSON boolean")
    return value


@dataclass(frozen=True, order=True)
class MoELayerFragment:
    """One local model chunk's global MoE-layer placement."""

    layer_no: int
    scope_id: int

    def __post_init__(self) -> None:
        if self.layer_no < 0:
            raise ValueError("MoE layer_no must be non-negative")
        if self.scope_id < 0:
            raise ValueError("MoE scope_id must be non-negative")


@dataclass(frozen=True)
class MegatronEPTopologyFragment:
    """Topology facts visible to one Megatron global rank."""

    model_id: str
    global_rank: int
    tp_group: tuple[int, ...]
    pp_group: tuple[int, ...]
    dp_group: tuple[int, ...]
    cp_group: tuple[int, ...]
    ep_group: tuple[int, ...]
    etp_group: tuple[int, ...]
    dispatch_group: tuple[int, ...]
    expert_dp_group: tuple[int, ...]
    moe_layers: tuple[MoELayerFragment, ...]
    local_expert_order: tuple[int, ...] | None
    sequence_parallel: bool
    top_k: int
    dispatcher_type: str
    permutation_mode: str
    etp_composition: str
    dropless: bool
    padded: bool

    def __post_init__(self) -> None:
        if not self.model_id:
            raise ValueError("model_id must not be empty")
        if self.global_rank < 0:
            raise ValueError("global_rank must be non-negative")
        for field_name in (
            "tp_group",
            "pp_group",
            "dp_group",
            "cp_group",
            "ep_group",
            "etp_group",
            "dispatch_group",
            "expert_dp_group",
        ):
            group = _validate_group(getattr(self, field_name), field_name)
            if self.global_rank not in group:
                raise ValueError(f"global rank {self.global_rank} is absent from {field_name}")
        if self.top_k <= 0:
            raise ValueError("top_k must be positive")
        if self.local_expert_order is not None:
            if not self.local_expert_order:
                raise ValueError("local_expert_order must not be empty")
            if len(set(self.local_expert_order)) != len(self.local_expert_order):
                raise ValueError("local_expert_order contains duplicates")


@dataclass(frozen=True, order=True)
class MoELayerPlacement:
    layer_no: int
    pp_rank: int
    scope_id: int

    def __post_init__(self) -> None:
        if self.layer_no < 0:
            raise ValueError("MoE layer_no must be non-negative")
        if self.pp_rank < 0:
            raise ValueError("MoE pp_rank must be non-negative")
        if self.scope_id < 0:
            raise ValueError("MoE scope_id must be non-negative")


@dataclass(frozen=True)
class FrozenMegatronEPTopologyManifest:
    """Canonical run-level topology consumed by offline reconstruction."""

    model_id: str
    tp_groups: tuple[tuple[int, ...], ...]
    pp_groups: tuple[tuple[int, ...], ...]
    dp_groups: tuple[tuple[int, ...], ...]
    cp_groups: tuple[tuple[int, ...], ...]
    ep_groups: tuple[tuple[int, ...], ...]
    etp_groups: tuple[tuple[int, ...], ...]
    dispatch_groups: tuple[tuple[int, ...], ...]
    expert_dp_groups: tuple[tuple[int, ...], ...]
    layer_placements: tuple[MoELayerPlacement, ...]
    local_expert_order_by_ep_rank: tuple[tuple[int, ...], ...]
    sequence_parallel: bool
    top_k: int
    dispatcher_type: str
    permutation_mode: str
    etp_composition: str
    dropless: bool
    padded: bool

    def __post_init__(self) -> None:
        if not self.model_id:
            raise ValueError("model_id must not be empty")
        all_ranks = _validate_partition(self.tp_groups, "tp_groups")
        for name in _GROUP_FIELDS[1:]:
            _validate_partition(getattr(self, name), name, all_ranks)
        if tuple(sorted(self.layer_placements)) != self.layer_placements:
            raise ValueError("layer_placements must be sorted")
        layer_numbers = [placement.layer_no for placement in self.layer_placements]
        if not layer_numbers or len(set(layer_numbers)) != len(layer_numbers):
            raise ValueError("Each MoE layer must have exactly one placement")
        pp_size = len(self.pp_groups[0])
        if any(not 0 <= placement.pp_rank < pp_size for placement in self.layer_placements):
            raise ValueError("A MoE layer has an invalid PP rank")
        ep_size = len(self.ep_groups[0])
        if len(self.local_expert_order_by_ep_rank) != ep_size:
            raise ValueError("Local expert order must cover every EP rank")
        experts: set[int] = set()
        for ep_rank, local_order in enumerate(self.local_expert_order_by_ep_rank):
            if not local_order or len(set(local_order)) != len(local_order):
                raise ValueError(f"Invalid local expert order for EP rank {ep_rank}")
            overlap = experts.intersection(local_order)
            if overlap:
                raise ValueError(f"Experts assigned to multiple EP ranks: {sorted(overlap)}")
            experts.update(local_order)
        if experts != set(range(len(experts))):
            raise ValueError("Global expert IDs must be contiguous from zero")
        if self.top_k <= 0 or self.top_k > len(experts):
            raise ValueError("top_k is incompatible with the global expert count")
        if self.dispatcher_type != "alltoall":
            raise ValueError("Only the AlltoAll dispatcher is supported")
        if self.permutation_mode != "non_fused":
            raise ValueError("Only non-fused permutation is supported")
        if self.etp_composition != "matching_row_sum":
            raise ValueError("Only matching-row ETP summation is supported")
        if not self.dropless or self.padded:
            raise ValueError("The manifest requires dropless, unpadded execution")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": _SCHEMA_VERSION,
            "model_id": self.model_id,
            "groups": {
                name: [list(group) for group in getattr(self, name)] for name in _GROUP_FIELDS
            },
            "layer_placements": [
                {
                    "layer_no": placement.layer_no,
                    "pp_rank": placement.pp_rank,
                    "scope_id": placement.scope_id,
                }
                for placement in self.layer_placements
            ],
            "local_expert_order_by_ep_rank": [
                list(order) for order in self.local_expert_order_by_ep_rank
            ],
            "sequence_parallel": self.sequence_parallel,
            "top_k": self.top_k,
            "dispatcher_type": self.dispatcher_type,
            "permutation_mode": self.permutation_mode,
            "etp_composition": self.etp_composition,
            "dropless": self.dropless,
            "padded": self.padded,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "FrozenMegatronEPTopologyManifest":
        if int(value["schema_version"]) != _SCHEMA_VERSION:
            raise ValueError(f"Unsupported topology schema version: {value['schema_version']}")
        groups = value["groups"]
        return cls(
            model_id=str(value["model_id"]),
            **{
                name: tuple(tuple(int(rank) for rank in group) for group in groups[name])
                for name in _GROUP_FIELDS
            },
            layer_placements=tuple(
                MoELayerPlacement(
                    layer_no=int(item["layer_no"]),
                    pp_rank=int(item["pp_rank"]),
                    scope_id=int(item["scope_id"]),
                )
                for item in value["layer_placements"]
            ),
            local_expert_order_by_ep_rank=tuple(
                tuple(int(expert) for expert in order)
                for order in value["local_expert_order_by_ep_rank"]
            ),
            sequence_parallel=_require_bool(value["sequence_parallel"], "sequence_parallel"),
            top_k=int(value["top_k"]),
            dispatcher_type=str(value["dispatcher_type"]),
            permutation_mode=str(value["permutation_mode"]),
            etp_composition=str(value["etp_composition"]),
            dropless=_require_bool(value["dropless"], "dropless"),
            padded=_require_bool(value["padded"], "padded"),
        )

    def topology_for_layer(self, layer_no: int) -> MoEParallelTopology:
        placements = {placement.layer_no: placement for placement in self.layer_placements}
        if int(layer_no) not in placements:
            raise KeyError(f"MoE layer {layer_no} is absent from the manifest")
        target_pp_rank = placements[int(layer_no)].pp_rank

        pp_rank_by_global_rank: dict[int, int] = {}
        for group in self.pp_groups:
            for pp_rank, global_rank in enumerate(group):
                pp_rank_by_global_rank[global_rank] = pp_rank

        def groups_on_stage(groups: tuple[tuple[int, ...], ...]) -> tuple[tuple[int, ...], ...]:
            selected: list[tuple[int, ...]] = []
            for group in groups:
                stage_ranks = {pp_rank_by_global_rank[rank] for rank in group}
                if len(stage_ranks) != 1:
                    raise ValueError("A reconstruction process group crosses PP stages")
                if next(iter(stage_ranks)) == target_pp_rank:
                    selected.append(group)
            if not selected:
                raise ValueError(f"No producer group owns MoE layer {layer_no}")
            return tuple(selected)

        ep_groups = groups_on_stage(self.ep_groups)
        etp_groups = groups_on_stage(self.etp_groups)
        dispatch_groups = groups_on_stage(self.dispatch_groups)
        producer_ranks = {rank for group in dispatch_groups for rank in group}

        def rank_map(groups: tuple[tuple[int, ...], ...]) -> dict[int, int]:
            result: dict[int, int] = {}
            for group in groups:
                for local_rank, global_rank in enumerate(group):
                    if global_rank in producer_ranks:
                        result[global_rank] = local_rank
            if set(result) != producer_ranks:
                raise ValueError("A producer is absent from a required rank mapping")
            return result

        expert_to_ep: dict[int, int] = {}
        local_orders: dict[int, tuple[int, ...]] = {}
        for ep_rank, order in enumerate(self.local_expert_order_by_ep_rank):
            local_orders[ep_rank] = order
            for expert in order:
                expert_to_ep[expert] = ep_rank
        global_expert_to_ep_rank = tuple(
            expert_to_ep[expert] for expert in range(len(expert_to_ep))
        )

        return MoEParallelTopology(
            context_parallel_size=len(self.cp_groups[0]),
            top_k=self.top_k,
            global_expert_to_ep_rank=global_expert_to_ep_rank,
            local_expert_order_by_ep_rank=local_orders,
            ep_groups=ep_groups,
            etp_groups=etp_groups,
            dispatch_groups=dispatch_groups,
            dense_dp_rank_by_global_rank=rank_map(self.dp_groups),
            expert_dp_rank_by_global_rank=rank_map(self.expert_dp_groups),
            dispatcher_type=self.dispatcher_type,
            permutation_mode=self.permutation_mode,
            dropless=self.dropless,
            padded=self.padded,
        )


def assemble_ep_topology_manifest(
    fragments: Sequence[MegatronEPTopologyFragment],
) -> FrozenMegatronEPTopologyManifest:
    """Validate WORLD rank fragments and assemble one canonical manifest."""

    if not fragments:
        raise ValueError("At least one topology fragment is required")
    ranks = [int(fragment.global_rank) for fragment in fragments]
    if len(set(ranks)) != len(ranks):
        raise ValueError("Duplicate topology fragment for a global rank")
    all_ranks = set(ranks)

    def common(field_name: str) -> Any:
        values = {getattr(fragment, field_name) for fragment in fragments}
        if len(values) != 1:
            raise ValueError(f"Ranks disagree on {field_name}")
        return next(iter(values))

    groups = {
        plural_name: _canonical_partition(
            (getattr(fragment, plural_name[:-1]) for fragment in fragments),
            plural_name,
            all_ranks,
        )
        for plural_name in _GROUP_FIELDS
    }

    placements: dict[int, MoELayerPlacement] = {}
    expert_orders: dict[int, tuple[int, ...]] = {}
    for fragment in fragments:
        pp_rank = _rank_in_group(fragment.global_rank, fragment.pp_group, "PP group")
        ep_rank = _rank_in_group(fragment.global_rank, fragment.ep_group, "EP group")
        for local_layer in fragment.moe_layers:
            placement = MoELayerPlacement(local_layer.layer_no, pp_rank, local_layer.scope_id)
            previous = placements.setdefault(local_layer.layer_no, placement)
            if previous != placement:
                raise ValueError(f"Ranks disagree on placement for MoE layer {local_layer.layer_no}")
        if fragment.local_expert_order is not None:
            previous_order = expert_orders.setdefault(ep_rank, fragment.local_expert_order)
            if previous_order != fragment.local_expert_order:
                raise ValueError(f"Ranks disagree on local expert order for EP rank {ep_rank}")

    ep_size = len(groups["ep_groups"][0])
    if set(expert_orders) != set(range(ep_size)):
        raise ValueError("Rank fragments do not establish every EP rank's expert order")

    return FrozenMegatronEPTopologyManifest(
        model_id=str(common("model_id")),
        **groups,
        layer_placements=tuple(sorted(placements.values())),
        local_expert_order_by_ep_rank=tuple(expert_orders[index] for index in range(ep_size)),
        sequence_parallel=bool(common("sequence_parallel")),
        top_k=int(common("top_k")),
        dispatcher_type=str(common("dispatcher_type")),
        permutation_mode=str(common("permutation_mode")),
        etp_composition=str(common("etp_composition")),
        dropless=bool(common("dropless")),
        padded=bool(common("padded")),
    )


def load_ep_topology_manifest(path: str | os.PathLike[str]) -> FrozenMegatronEPTopologyManifest:
    with Path(path).open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, Mapping):
        raise TypeError("Topology manifest root must be a JSON object")
    return FrozenMegatronEPTopologyManifest.from_dict(value)


def write_ep_topology_manifest(
    path: str | os.PathLike[str], manifest: FrozenMegatronEPTopologyManifest
) -> None:
    """Atomically create a frozen manifest or accept identical existing content."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if load_ep_topology_manifest(destination) == manifest:
            return
        raise ValueError(f"Existing topology manifest differs: {destination}")

    payload = json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n"
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_name = stream.name
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, destination)
        temporary_name = None
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)


__all__ = [
    "FrozenMegatronEPTopologyManifest",
    "MegatronEPTopologyFragment",
    "MoELayerFragment",
    "MoELayerPlacement",
    "assemble_ep_topology_manifest",
    "load_ep_topology_manifest",
    "write_ep_topology_manifest",
]
