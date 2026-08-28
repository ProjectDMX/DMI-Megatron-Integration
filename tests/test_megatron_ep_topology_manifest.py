from __future__ import annotations

from dataclasses import replace

import pytest

from dmi_megatron_integration.topology.ep_topology_manifest import (
    FrozenMegatronEPTopologyManifest,
    MegatronEPTopologyFragment,
    MoELayerFragment,
    assemble_ep_topology_manifest,
    load_ep_topology_manifest,
    write_ep_topology_manifest,
)


def _owning_group(rank: int, groups: tuple[tuple[int, ...], ...]) -> tuple[int, ...]:
    matches = [group for group in groups if rank in group]
    assert len(matches) == 1
    return matches[0]


def _fragments() -> list[MegatronEPTopologyFragment]:
    tp_groups = tuple((rank,) for rank in range(8))
    pp_groups = ((0, 4), (1, 5), (2, 6), (3, 7))
    dp_groups = ((0, 2), (1, 3), (4, 6), (5, 7))
    cp_groups = tuple((rank,) for rank in range(8))
    # Reversed inner order proves that canonicalization preserves group rank order.
    ep_groups = ((1, 0), (3, 2), (5, 4), (7, 6))
    etp_groups = tuple((rank,) for rank in range(8))
    dispatch_groups = ep_groups
    expert_dp_groups = dp_groups

    result: list[MegatronEPTopologyFragment] = []
    for rank in range(8):
        pp_group = _owning_group(rank, pp_groups)
        pp_rank = pp_group.index(rank)
        ep_group = _owning_group(rank, ep_groups)
        ep_rank = ep_group.index(rank)
        result.append(
            MegatronEPTopologyFragment(
                model_id="controlled",
                global_rank=rank,
                tp_group=_owning_group(rank, tp_groups),
                pp_group=pp_group,
                dp_group=_owning_group(rank, dp_groups),
                cp_group=_owning_group(rank, cp_groups),
                ep_group=ep_group,
                etp_group=_owning_group(rank, etp_groups),
                dispatch_group=_owning_group(rank, dispatch_groups),
                expert_dp_group=_owning_group(rank, expert_dp_groups),
                moe_layers=(MoELayerFragment(layer_no=pp_rank, scope_id=0),),
                local_expert_order=(ep_rank,),
                sequence_parallel=False,
                top_k=1,
                dispatcher_type="alltoall",
                permutation_mode="non_fused",
                etp_composition="matching_row_sum",
                dropless=True,
                padded=False,
            )
        )
    return result


def test_assemble_preserves_group_order_and_derives_layer_topology() -> None:
    manifest = assemble_ep_topology_manifest(_fragments())

    assert manifest.ep_groups[0] == (1, 0)
    assert [(item.layer_no, item.pp_rank, item.scope_id) for item in manifest.layer_placements] == [
        (0, 0, 0),
        (1, 1, 0),
    ]
    assert manifest.local_expert_order_by_ep_rank == ((0,), (1,))

    topology = manifest.topology_for_layer(0)
    assert topology.ep_groups == ((1, 0), (3, 2))
    assert topology.dispatch_groups == ((1, 0), (3, 2))
    assert topology.global_expert_to_ep_rank == (0, 1)
    assert topology.dense_dp_rank_by_global_rank == {0: 0, 1: 0, 2: 1, 3: 1}
    assert topology.expert_dp_rank_by_global_rank == {0: 0, 1: 0, 2: 1, 3: 1}


def test_rank_disagreement_is_rejected() -> None:
    fragments = _fragments()
    fragments[7] = replace(fragments[7], sequence_parallel=True)

    with pytest.raises(ValueError, match="sequence_parallel"):
        assemble_ep_topology_manifest(fragments)


def test_missing_rank_fragment_is_rejected() -> None:
    with pytest.raises(ValueError, match="does not partition"):
        assemble_ep_topology_manifest(_fragments()[:-1])


def test_manifest_round_trip_and_conflicting_existing_file(tmp_path) -> None:
    manifest = assemble_ep_topology_manifest(_fragments())
    path = tmp_path / "topology.json"

    write_ep_topology_manifest(path, manifest)
    assert load_ep_topology_manifest(path) == manifest
    write_ep_topology_manifest(path, manifest)

    conflicting = replace(manifest, top_k=2)
    with pytest.raises(ValueError, match="differs"):
        write_ep_topology_manifest(path, conflicting)


def test_context_parallel_size_is_retained_for_reconstruction_guard() -> None:
    fragments = _fragments()
    cp_groups = ((0, 1), (2, 3), (4, 5), (6, 7))
    fragments = [
        replace(fragment, cp_group=_owning_group(fragment.global_rank, cp_groups))
        for fragment in fragments
    ]

    topology = assemble_ep_topology_manifest(fragments).topology_for_layer(0)
    assert topology.context_parallel_size == 2


def test_loaded_manifest_rejects_negative_placement() -> None:
    value = assemble_ep_topology_manifest(_fragments()).to_dict()
    value["layer_placements"][0]["scope_id"] = -1

    with pytest.raises(ValueError, match="scope_id"):
        FrozenMegatronEPTopologyManifest.from_dict(value)


def test_loaded_manifest_requires_real_json_booleans() -> None:
    value = assemble_ep_topology_manifest(_fragments()).to_dict()
    value["dropless"] = "false"

    with pytest.raises(TypeError, match="dropless"):
        FrozenMegatronEPTopologyManifest.from_dict(value)
