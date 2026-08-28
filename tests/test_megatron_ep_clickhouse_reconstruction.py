from __future__ import annotations

import pytest
import torch

from dmi_megatron_integration.materialization.ep_clickhouse_reconstruction import (
    reconstruct_moe_clickhouse_rows,
)
from dmi_megatron_integration.materialization.ep_reconstruction import SourceTokenCoordinate
from dmi_megatron_integration.topology.ep_topology_manifest import (
    FrozenMegatronEPTopologyManifest,
    MoELayerPlacement,
)


def _manifest() -> FrozenMegatronEPTopologyManifest:
    return FrozenMegatronEPTopologyManifest(
        model_id="controlled-sp-etp",
        tp_groups=((0, 1),),
        pp_groups=((0,), (1,)),
        dp_groups=((0,), (1,)),
        cp_groups=((0,), (1,)),
        ep_groups=((0,), (1,)),
        etp_groups=((0, 1),),
        dispatch_groups=((0, 1),),
        expert_dp_groups=((0,), (1,)),
        layer_placements=(MoELayerPlacement(layer_no=0, pp_rank=0, scope_id=0),),
        local_expert_order_by_ep_rank=((0, 1),),
        sequence_parallel=True,
        top_k=1,
        dispatcher_type="alltoall",
        permutation_mode="non_fused",
        etp_composition="matching_row_sum",
        dropless=True,
        padded=False,
    )


def _coordinates(
    act_name: str,
    *,
    shard_rank: int,
    dp_rank: int,
    sample_index: int,
    token_start: int,
    token_end: int,
    dataset_id: int,
    invocation_id: int = 2,
) -> tuple:
    return (
        "controlled-sp-etp",
        act_name,
        "fwd",
        "train",
        3,
        dp_rank,
        0,
        sample_index,
        0,
        shard_rank,
        token_start,
        token_end,
        0,
        invocation_id,
        dataset_id,
    )


def _router_row(
    act_name: str,
    *,
    shard_rank: int,
    sample_index: int,
    dataset_id: int,
    tensor: torch.Tensor,
    invocation_id: int = 2,
) -> tuple[tuple, torch.Tensor]:
    return (
        _coordinates(
            act_name,
            shard_rank=shard_rank,
            dp_rank=0,
            sample_index=sample_index,
            token_start=0,
            token_end=4,
            dataset_id=dataset_id,
            invocation_id=invocation_id,
        ),
        tensor,
    )


def _execution_row(
    act_name: str,
    *,
    shard_rank: int,
    tensor: torch.Tensor,
    invocation_id: int = 2,
) -> tuple[tuple, torch.Tensor]:
    return (
        _coordinates(
            act_name,
            shard_rank=shard_rank,
            dp_rank=-1,
            sample_index=-1,
            token_start=-1,
            token_end=-1,
            dataset_id=-1,
            invocation_id=invocation_id,
        ),
        tensor,
    )


def test_clickhouse_rows_restore_sp_coordinates_and_global_etp_expert_order() -> None:
    ids_by_sample = (
        torch.tensor([[0], [1]], dtype=torch.int64),
        torch.tensor([[1], [0]], dtype=torch.int64),
    )
    weights_by_sample = (
        torch.tensor([[1.0], [1.0]]),
        torch.tensor([[1.0], [1.0]]),
    )
    rows_by_act = {
        "router_topk_expert_ids": [
            _router_row(
                "router_topk_expert_ids",
                shard_rank=rank,
                sample_index=sample,
                dataset_id=10 + sample,
                tensor=ids_by_sample[sample],
                invocation_id=2,
            )
            for rank in (0, 1)
            for sample in (0, 1)
        ],
        "router_topk_weights": [
            _router_row(
                "router_topk_weights",
                shard_rank=rank,
                sample_index=sample,
                dataset_id=10 + sample,
                tensor=weights_by_sample[sample],
                invocation_id=5,
            )
            for rank in (0, 1)
            for sample in (0, 1)
        ],
        "moe_inverse_map": [
            _execution_row(
                "moe_inverse_map",
                shard_rank=rank,
                tensor=torch.tensor([0, 3, 1, 2], dtype=torch.int64),
                invocation_id=7,
            )
            for rank in (0, 1)
        ],
        "moe_packed_weighted_output": [
            _execution_row(
                "moe_packed_weighted_output",
                shard_rank=0,
                tensor=torch.arange(1, 9, dtype=torch.float32).reshape(8, 1),
                invocation_id=11,
            ),
            _execution_row(
                "moe_packed_weighted_output",
                shard_rank=1,
                tensor=(100 * torch.arange(1, 9, dtype=torch.float32)).reshape(8, 1),
                invocation_id=11,
            ),
        ],
    }

    reconstructed = reconstruct_moe_clickhouse_rows(_manifest(), rows_by_act)

    assert len(reconstructed) == 1
    invocation = reconstructed[0]
    assert len(invocation.source_domains) == 1
    domain = invocation.source_domains[0]
    assert domain.token_coordinates == (
        SourceTokenCoordinate(10, 0, 0),
        SourceTokenCoordinate(10, 0, 1),
        SourceTokenCoordinate(10, 0, 2),
        SourceTokenCoordinate(10, 0, 3),
        SourceTokenCoordinate(11, 1, 0),
        SourceTokenCoordinate(11, 1, 1),
        SourceTokenCoordinate(11, 1, 2),
        SourceTokenCoordinate(11, 1, 3),
    )
    torch.testing.assert_close(
        domain.combined_output,
        torch.tensor([[101.0], [606.0], [303.0], [808.0], [505.0], [202.0], [707.0], [404.0]]),
    )

    rows_by_act["moe_inverse_map"].append(
        _execution_row(
            "moe_inverse_map",
            shard_rank=0,
            tensor=torch.tensor([0, 3, 1, 2], dtype=torch.int64),
            invocation_id=99,
        )
    )
    with pytest.raises(ValueError, match="Duplicate inverse-map row for producer 0"):
        reconstruct_moe_clickhouse_rows(_manifest(), rows_by_act)


def test_clickhouse_rows_reject_inconsistent_tp_peer_sample_metadata() -> None:
    ids = torch.tensor([[0], [1]], dtype=torch.int64)
    weights = torch.ones((2, 1), dtype=torch.float32)
    rows_by_act = {
        "router_topk_expert_ids": [
            _router_row(
                "router_topk_expert_ids",
                shard_rank=rank,
                sample_index=sample,
                dataset_id=(99 if rank == 1 and sample == 0 else 10 + sample),
                tensor=ids,
            )
            for rank in (0, 1)
            for sample in (0, 1)
        ],
        "router_topk_weights": [
            _router_row(
                "router_topk_weights",
                shard_rank=rank,
                sample_index=sample,
                dataset_id=(99 if rank == 1 and sample == 0 else 10 + sample),
                tensor=weights,
            )
            for rank in (0, 1)
            for sample in (0, 1)
        ],
        "moe_inverse_map": [
            _execution_row(
                "moe_inverse_map",
                shard_rank=rank,
                tensor=torch.tensor([0, 1, 0, 1], dtype=torch.int64),
            )
            for rank in (0, 1)
        ],
        "moe_packed_weighted_output": [
            _execution_row(
                "moe_packed_weighted_output",
                shard_rank=rank,
                tensor=torch.zeros((8, 1), dtype=torch.float32),
            )
            for rank in (0, 1)
        ],
    }

    with pytest.raises(ValueError, match="TP peers disagree on router sample metadata"):
        reconstruct_moe_clickhouse_rows(_manifest(), rows_by_act)


def test_clickhouse_rows_reject_inconsistent_tp_sp_local_extents() -> None:
    ids_by_rank = {
        0: torch.tensor([[0], [1]], dtype=torch.int64),
        1: torch.tensor([[0], [1], [0]], dtype=torch.int64),
    }
    rows_by_act = {
        "router_topk_expert_ids": [
            _router_row(
                "router_topk_expert_ids",
                shard_rank=rank,
                sample_index=sample,
                dataset_id=10 + sample,
                tensor=ids_by_rank[rank],
            )
            for rank in (0, 1)
            for sample in (0, 1)
        ],
        "router_topk_weights": [
            _router_row(
                "router_topk_weights",
                shard_rank=rank,
                sample_index=sample,
                dataset_id=10 + sample,
                tensor=torch.ones_like(ids_by_rank[rank], dtype=torch.float32),
            )
            for rank in (0, 1)
            for sample in (0, 1)
        ],
        "moe_inverse_map": [
            _execution_row(
                "moe_inverse_map",
                shard_rank=rank,
                tensor=torch.tensor([0, 1, 0, 1], dtype=torch.int64),
            )
            for rank in (0, 1)
        ],
        "moe_packed_weighted_output": [
            _execution_row(
                "moe_packed_weighted_output",
                shard_rank=rank,
                tensor=torch.zeros((8, 1), dtype=torch.float32),
            )
            for rank in (0, 1)
        ],
    }

    with pytest.raises(
        ValueError, match="TP/SP peers disagree on local sequence extent"
    ):
        reconstruct_moe_clickhouse_rows(_manifest(), rows_by_act)
