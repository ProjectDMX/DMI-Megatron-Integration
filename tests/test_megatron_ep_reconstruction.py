from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from dmi_megatron_integration.materialization.ep_reconstruction import (
    InverseMapShard,
    MoEExecutionKey,
    MoEParallelTopology,
    PackedWeightedOutputShard,
    RouterExpertIdsShard,
    RouterWeightsShard,
    SourceTokenCoordinate,
    reconstruct_moe_invocation,
)


def test_ep2_reconstructs_exact_source_weighted_outputs() -> None:
    key = MoEExecutionKey(
        model_id="controlled",
        phase="train",
        global_batch_id=7,
        attempt_id=0,
        microbatch_id=0,
        layer_no=3,
        direction="fwd",
    )
    topology = MoEParallelTopology(
        context_parallel_size=1,
        top_k=2,
        global_expert_to_ep_rank=(0, 0, 1, 1),
        local_expert_order_by_ep_rank={0: (0, 1), 1: (2, 3)},
        ep_groups=((0, 1),),
        etp_groups=((0,), (1,)),
        dispatch_groups=((0, 1),),
        dense_dp_rank_by_global_rank={0: 0, 1: 1},
        expert_dp_rank_by_global_rank={0: 0, 1: 0},
    )
    rank0_coordinates = (
        SourceTokenCoordinate(dataset_id=10, sample_index=0, token_index=0),
        SourceTokenCoordinate(dataset_id=10, sample_index=0, token_index=1),
    )
    rank1_coordinates = (
        SourceTokenCoordinate(dataset_id=10, sample_index=1, token_index=0),
        SourceTokenCoordinate(dataset_id=10, sample_index=1, token_index=1),
    )
    result = reconstruct_moe_invocation(
        topology,
        expert_id_shards=(
            RouterExpertIdsShard(
                key=key,
                producer_rank=0,
                dense_dp_rank=0,
                token_coordinates=rank0_coordinates,
                tensor=torch.tensor([[0, 2], [1, 3]], dtype=torch.int64),
            ),
            RouterExpertIdsShard(
                key=key,
                producer_rank=1,
                dense_dp_rank=1,
                token_coordinates=rank1_coordinates,
                tensor=torch.tensor([[2, 3], [0, 1]], dtype=torch.int64),
            ),
        ),
        weight_shards=(
            RouterWeightsShard(
                key=key,
                producer_rank=0,
                dense_dp_rank=0,
                token_coordinates=rank0_coordinates,
                tensor=torch.tensor([[0.6, 0.4], [0.7, 0.3]]),
            ),
            RouterWeightsShard(
                key=key,
                producer_rank=1,
                dense_dp_rank=1,
                token_coordinates=rank1_coordinates,
                tensor=torch.tensor([[0.8, 0.2], [0.55, 0.45]]),
            ),
        ),
        inverse_map_shards=(
            InverseMapShard(
                key=key,
                producer_rank=0,
                tensor=torch.tensor([0, 1, 0, 1], dtype=torch.int64),
            ),
            InverseMapShard(
                key=key,
                producer_rank=1,
                tensor=torch.tensor([1, 1, 0, 0], dtype=torch.int64),
            ),
        ),
        packed_output_shards=(
            PackedWeightedOutputShard(
                key=key,
                producer_rank=0,
                tensor=torch.tensor(
                    [[1.0, 2.0], [121.0, 122.0], [21.0, 22.0], [131.0, 132.0]]
                ),
            ),
            PackedWeightedOutputShard(
                key=key,
                producer_rank=1,
                tensor=torch.tensor(
                    [[11.0, 12.0], [101.0, 102.0], [31.0, 32.0], [111.0, 112.0]]
                ),
            ),
        ),
    )

    assert result.key == key
    assert tuple(domain.dense_dp_rank for domain in result.source_domains) == (0, 1)
    rank0, rank1 = result.source_domains
    assert rank0.token_coordinates == rank0_coordinates
    assert rank1.token_coordinates == rank1_coordinates
    assert torch.equal(
        rank0.selected_expert_ids,
        torch.tensor([[0, 2], [1, 3]], dtype=torch.int64),
    )
    assert torch.equal(
        rank1.selected_expert_ids,
        torch.tensor([[2, 3], [0, 1]], dtype=torch.int64),
    )
    torch.testing.assert_close(
        rank0.weighted_outputs,
        torch.tensor(
            [
                [[1.0, 2.0], [11.0, 12.0]],
                [[21.0, 22.0], [31.0, 32.0]],
            ]
        ),
    )
    torch.testing.assert_close(
        rank1.weighted_outputs,
        torch.tensor(
            [
                [[101.0, 102.0], [111.0, 112.0]],
                [[121.0, 122.0], [131.0, 132.0]],
            ]
        ),
    )
    torch.testing.assert_close(
        rank0.combined_output,
        torch.tensor([[12.0, 14.0], [52.0, 54.0]]),
    )
    torch.testing.assert_close(
        rank1.combined_output,
        torch.tensor([[212.0, 214.0], [252.0, 254.0]]),
    )


def test_etp_partials_sum_in_common_packed_row_order() -> None:
    key = MoEExecutionKey("controlled", "train", 8, 0, 0, 4, "fwd")
    topology = MoEParallelTopology(
        context_parallel_size=1,
        top_k=1,
        global_expert_to_ep_rank=(0, 0),
        local_expert_order_by_ep_rank={0: (0, 1)},
        ep_groups=((0,), (1,)),
        etp_groups=((0, 1),),
        dispatch_groups=((0, 1),),
        dense_dp_rank_by_global_rank={0: 0, 1: 1},
        expert_dp_rank_by_global_rank={0: 0, 1: 0},
    )
    coordinate0 = (SourceTokenCoordinate(20, 0, 0),)
    coordinate1 = (SourceTokenCoordinate(20, 1, 0),)
    result = reconstruct_moe_invocation(
        topology,
        expert_id_shards=(
            RouterExpertIdsShard(
                key, 0, 0, coordinate0, torch.tensor([[0]], dtype=torch.int64)
            ),
            RouterExpertIdsShard(
                key, 1, 1, coordinate1, torch.tensor([[1]], dtype=torch.int64)
            ),
        ),
        weight_shards=(
            RouterWeightsShard(key, 0, 0, coordinate0, torch.tensor([[1.0]])),
            RouterWeightsShard(key, 1, 1, coordinate1, torch.tensor([[1.0]])),
        ),
        inverse_map_shards=(
            InverseMapShard(key, 0, torch.tensor([0], dtype=torch.int64)),
            InverseMapShard(key, 1, torch.tensor([0], dtype=torch.int64)),
        ),
        packed_output_shards=(
            PackedWeightedOutputShard(
                key, 0, torch.tensor([[1.0, 2.0], [3.0, 4.0]])
            ),
            PackedWeightedOutputShard(
                key, 1, torch.tensor([[10.0, 20.0], [30.0, 40.0]])
            ),
        ),
    )

    torch.testing.assert_close(
        result.source_domains[0].weighted_outputs,
        torch.tensor([[[11.0, 22.0]]]),
    )
    torch.testing.assert_close(
        result.source_domains[1].weighted_outputs,
        torch.tensor([[[33.0, 44.0]]]),
    )


def test_expert_dp_dispatch_groups_remain_isolated() -> None:
    key = MoEExecutionKey("controlled", "train", 9, 0, 0, 5, "fwd")
    topology = MoEParallelTopology(
        context_parallel_size=1,
        top_k=1,
        global_expert_to_ep_rank=(0, 0),
        local_expert_order_by_ep_rank={0: (0, 1)},
        ep_groups=((0,), (1,)),
        etp_groups=((0,), (1,)),
        dispatch_groups=((0,), (1,)),
        dense_dp_rank_by_global_rank={0: 0, 1: 1},
        expert_dp_rank_by_global_rank={0: 0, 1: 1},
    )
    coordinate0 = (SourceTokenCoordinate(30, 0, 0),)
    coordinate1 = (SourceTokenCoordinate(30, 1, 0),)
    result = reconstruct_moe_invocation(
        topology,
        expert_id_shards=(
            RouterExpertIdsShard(
                key, 0, 0, coordinate0, torch.tensor([[0]], dtype=torch.int64)
            ),
            RouterExpertIdsShard(
                key, 1, 1, coordinate1, torch.tensor([[1]], dtype=torch.int64)
            ),
        ),
        weight_shards=(
            RouterWeightsShard(key, 0, 0, coordinate0, torch.tensor([[1.0]])),
            RouterWeightsShard(key, 1, 1, coordinate1, torch.tensor([[1.0]])),
        ),
        inverse_map_shards=(
            InverseMapShard(key, 0, torch.tensor([0], dtype=torch.int64)),
            InverseMapShard(key, 1, torch.tensor([0], dtype=torch.int64)),
        ),
        packed_output_shards=(
            PackedWeightedOutputShard(key, 0, torch.tensor([[5.0, 6.0]])),
            PackedWeightedOutputShard(key, 1, torch.tensor([[7.0, 8.0]])),
        ),
    )

    assert tuple(domain.dense_dp_rank for domain in result.source_domains) == (0, 1)
    torch.testing.assert_close(
        result.source_domains[0].weighted_outputs,
        torch.tensor([[[5.0, 6.0]]]),
    )
    torch.testing.assert_close(
        result.source_domains[1].weighted_outputs,
        torch.tensor([[[7.0, 8.0]]]),
    )


def test_context_parallel_greater_than_one_fails_before_record_validation() -> None:
    topology = MoEParallelTopology(
        context_parallel_size=1,
        top_k=1,
        global_expert_to_ep_rank=(0,),
        local_expert_order_by_ep_rank={0: (0,)},
        ep_groups=((0,),),
        etp_groups=((0,),),
        dispatch_groups=((0,),),
        dense_dp_rank_by_global_rank={0: 0},
        expert_dp_rank_by_global_rank={0: 0},
    )

    with pytest.raises(NotImplementedError, match="context_parallel_size == 1"):
        reconstruct_moe_invocation(
            replace(topology, context_parallel_size=2),
            expert_id_shards=(),
            weight_shards=(),
            inverse_map_shards=(),
            packed_output_shards=(),
        )


def test_inverse_map_must_equal_runtime_source_token_order() -> None:
    key = MoEExecutionKey("controlled", "train", 10, 0, 0, 6, "fwd")
    topology = MoEParallelTopology(
        context_parallel_size=1,
        top_k=1,
        global_expert_to_ep_rank=(0,),
        local_expert_order_by_ep_rank={0: (0,)},
        ep_groups=((0,),),
        etp_groups=((0,),),
        dispatch_groups=((0,),),
        dense_dp_rank_by_global_rank={0: 0},
        expert_dp_rank_by_global_rank={0: 0},
    )
    coordinates = (SourceTokenCoordinate(40, 0, 0),)

    with pytest.raises(ValueError, match="Captured inverse map disagrees"):
        reconstruct_moe_invocation(
            topology,
            expert_id_shards=(
                RouterExpertIdsShard(
                    key, 0, 0, coordinates, torch.tensor([[0]], dtype=torch.int64)
                ),
            ),
            weight_shards=(
                RouterWeightsShard(key, 0, 0, coordinates, torch.tensor([[1.0]])),
            ),
            inverse_map_shards=(
                InverseMapShard(key, 0, torch.tensor([1], dtype=torch.int64)),
            ),
            packed_output_shards=(
                PackedWeightedOutputShard(key, 0, torch.tensor([[1.0, 2.0]])),
            ),
        )


def test_inverse_map_accepts_valid_tokens_with_padding_gaps() -> None:
    key = MoEExecutionKey("controlled", "train", 11, 0, 0, 7, "fwd")
    topology = MoEParallelTopology(
        context_parallel_size=1,
        top_k=1,
        global_expert_to_ep_rank=(0,),
        local_expert_order_by_ep_rank={0: (0,)},
        ep_groups=((0,),),
        etp_groups=((0,),),
        dispatch_groups=((0,),),
        dense_dp_rank_by_global_rank={0: 0},
        expert_dp_rank_by_global_rank={0: 0},
    )
    coordinates = (
        SourceTokenCoordinate(50, 0, 0),
        SourceTokenCoordinate(50, 0, 2),
    )

    result = reconstruct_moe_invocation(
        topology,
        expert_id_shards=(
            RouterExpertIdsShard(
                key,
                0,
                0,
                coordinates,
                torch.tensor([[0], [0]], dtype=torch.int64),
                source_flat_indices=(0, 2),
            ),
        ),
        weight_shards=(
            RouterWeightsShard(key, 0, 0, coordinates, torch.ones((2, 1))),
        ),
        inverse_map_shards=(
            InverseMapShard(key, 0, torch.tensor([0, 2], dtype=torch.int64)),
        ),
        packed_output_shards=(
            PackedWeightedOutputShard(
                key,
                0,
                torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            ),
        ),
    )

    assert result.source_domains[0].token_coordinates == coordinates
    torch.testing.assert_close(
        result.source_domains[0].combined_output,
        torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
    )
