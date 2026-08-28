from __future__ import annotations

import pytest
import torch

from dmi_megatron_integration.metadata_context import (
    DMIMetadataContext,
    DMIMetadataDirection,
    DMIMetadataFieldSpec,
    DMIMetadataPropagator,
    LocalMetadataPropagator,
    TorchDistributedMetadataPropagator,
    dataset_id_field_spec,
    valid_count_field_spec,
)
from dmi_megatron_integration.hooks.specs import DimSpec


def test_context_allocates_stable_source_and_current_buffers():
    ctx = DMIMetadataContext(
        max_num_microbatches=3,
        max_batch_size=4,
        num_scopes=2,
        device="cpu",
    )
    source_ptr = ctx.source("valid_count", 0).data_ptr()
    current_ptr = ctx.current("valid_count", DMIMetadataDirection.FWD, 1).data_ptr()

    ctx.begin_iteration(2)
    ctx.ingest_microbatch(0, {"valid_count": torch.tensor([5, 4])})
    ctx.enter_scope(DMIMetadataDirection.FWD, 1, 0)
    ctx.end_iteration()
    ctx.begin_iteration(1)

    assert ctx.source("valid_count", 0).data_ptr() == source_ptr
    assert ctx.current("valid_count", DMIMetadataDirection.FWD, 1).data_ptr() == current_ptr


def test_ingest_microbatch_zero_fills_inactive_batch_entries():
    ctx = DMIMetadataContext(
        max_num_microbatches=2,
        max_batch_size=4,
        num_scopes=1,
        device="cpu",
    )

    ctx.begin_iteration(2)
    ctx.ingest_microbatch(0, {"valid_count": [7, 3]})

    assert torch.equal(ctx.source("valid_count", 0), torch.tensor([7, 3, 0, 0]))
    assert ctx.source_cpu("valid_count", 0) == (7, 3, 0, 0)


def test_ingest_microbatch_preserves_cpu_metadata_separately():
    ctx = DMIMetadataContext(
        max_num_microbatches=1,
        max_batch_size=3,
        num_scopes=1,
        device="cpu",
    )

    ctx.begin_iteration(1)
    ctx.ingest_microbatch(
        0,
        {"valid_count": torch.tensor([9, 8])},
        cpu_fields={"valid_count": [7, 6]},
    )

    assert torch.equal(ctx.source("valid_count", 0), torch.tensor([9, 8, 0]))
    assert ctx.source_cpu("valid_count", 0) == (7, 6, 0)


def test_load_source_microbatch_does_not_require_active_iteration():
    ctx = DMIMetadataContext(
        max_num_microbatches=2,
        max_batch_size=3,
        num_scopes=1,
        device="cpu",
    )

    ctx.load_source_microbatch(
        1,
        {"valid_count": [5, 4]},
        cpu_fields={"valid_count": [3, 2]},
    )

    assert torch.equal(ctx.source("valid_count", 1), torch.tensor([5, 4, 0]))
    assert ctx.source_cpu("valid_count", 1) == (3, 2, 0)


def test_begin_iteration_can_preserve_preloaded_source_buffers():
    ctx = DMIMetadataContext(
        max_num_microbatches=1,
        max_batch_size=2,
        num_scopes=1,
        device="cpu",
    )

    ctx.load_source_microbatch(0, {"valid_count": [8, 7]})
    ctx.begin_iteration(1, clear_buffers=False)

    assert torch.equal(ctx.source("valid_count", 0), torch.tensor([8, 7]))
    assert ctx.source_cpu("valid_count", 0) == (8, 7)


def test_enter_scope_copies_current_metadata_by_direction_and_scope():
    ctx = DMIMetadataContext(
        max_num_microbatches=2,
        max_batch_size=3,
        num_scopes=2,
        device="cpu",
    )

    ctx.begin_iteration(2)
    ctx.ingest_microbatch(0, {"valid_count": [1, 2, 3]})
    ctx.ingest_microbatch(1, {"valid_count": [4, 5, 6]})
    ctx.enter_scope("fwd", 0, 0)
    ctx.enter_scope(DMIMetadataDirection.BWD, 1, 1)

    assert torch.equal(ctx.current("valid_count", "fwd", 0), torch.tensor([1, 2, 3]))
    assert torch.equal(ctx.current("valid_count", "bwd", 1), torch.tensor([4, 5, 6]))
    assert torch.equal(ctx.current("valid_count", "fwd", 1), torch.tensor([0, 0, 0]))


def test_context_supports_additional_field_specs():
    spec = DMIMetadataFieldSpec(
        name="sample_offset",
        shape=[DimSpec.BATCH],
        dtype=torch.int64,
    )
    ctx = DMIMetadataContext(
        max_num_microbatches=1,
        max_batch_size=2,
        num_scopes=1,
        field_specs=[spec],
        device="cpu",
    )

    ctx.begin_iteration(1)
    ctx.ingest_microbatch(0, {"sample_offset": torch.tensor([10, 20], dtype=torch.int64)})
    ctx.enter_scope("fwd", 0, 0)

    current = ctx.current("sample_offset", "fwd", 0)
    assert current.dtype == torch.int64
    assert torch.equal(current, torch.tensor([10, 20], dtype=torch.int64))


def test_integer_metadata_fields_share_one_packet_as_zero_copy_views():
    ctx = DMIMetadataContext(
        max_num_microbatches=1,
        max_batch_size=3,
        num_scopes=1,
        field_specs=(valid_count_field_spec(), dataset_id_field_spec()),
        device="cpu",
    )
    ctx.begin_iteration(1)
    ctx.ingest_microbatch(
        0,
        {"valid_count": [7, 5, 3]},
        cpu_fields={"valid_count": [7, 5, 3], "dataset_id": [0, 2, 1]},
    )

    packets = ctx.active_cpu_packets(0)
    assert set(packets) == {torch.int64}
    packet = packets[torch.int64]
    valid_counts = ctx.source_cpu_tensor("valid_count", 0)
    dataset_ids = ctx.source_cpu_tensor("dataset_id", 0)
    assert packet.tolist() == [7, 5, 3, 0, 2, 1]
    assert valid_counts.untyped_storage().data_ptr() == packet.untyped_storage().data_ptr()
    assert dataset_ids.untyped_storage().data_ptr() == packet.untyped_storage().data_ptr()
    assert valid_counts.data_ptr() == packet.data_ptr()
    assert dataset_ids.data_ptr() == packet.data_ptr() + 3 * packet.element_size()


def test_cpu_packet_layout_is_independent_of_gpu_visibility():
    visible_ctx = DMIMetadataContext(
        max_num_microbatches=1,
        max_batch_size=3,
        num_scopes=1,
        field_specs=(
            valid_count_field_spec(gpu_visible=True),
            dataset_id_field_spec(),
        ),
        device="cpu",
    )
    cpu_only_ctx = DMIMetadataContext(
        max_num_microbatches=1,
        max_batch_size=3,
        num_scopes=1,
        field_specs=(
            valid_count_field_spec(gpu_visible=False),
            dataset_id_field_spec(),
        ),
        device="cuda",
    )

    visible_ctx.begin_iteration(1)
    visible_ctx.ingest_microbatch(
        0,
        {"valid_count": [7, 5, 3]},
        cpu_fields={"valid_count": [7, 5, 3], "dataset_id": [0, 2, 1]},
    )
    cpu_only_ctx.begin_iteration(1)
    cpu_only_ctx.ingest_microbatch(
        0,
        {},
        cpu_fields={"valid_count": [7, 5, 3], "dataset_id": [0, 2, 1]},
    )

    visible_packet = visible_ctx.active_cpu_packets(0)[torch.int64]
    cpu_only_packet = cpu_only_ctx.active_cpu_packets(0)[torch.int64]
    assert visible_packet.shape == cpu_only_packet.shape == (6,)
    assert visible_packet.tolist() == cpu_only_packet.tolist() == [7, 5, 3, 0, 2, 1]
    for name, expected_offset in (("valid_count", 0), ("dataset_id", 3)):
        visible_field = visible_ctx.source_cpu_tensor(name, 0)
        cpu_only_field = cpu_only_ctx.source_cpu_tensor(name, 0)
        visible_offset = (
            visible_field.data_ptr() - visible_packet.data_ptr()
        ) // visible_packet.element_size()
        cpu_only_offset = (
            cpu_only_field.data_ptr() - cpu_only_packet.data_ptr()
        ) // cpu_only_packet.element_size()
        assert visible_offset == cpu_only_offset == expected_offset

    assert cpu_only_ctx._source_buffers == {}
    assert cpu_only_ctx._current_buffers == {}
    with pytest.raises(KeyError, match="valid_count"):
        cpu_only_ctx.source("valid_count", 0)


def test_inactive_dataset_field_adds_no_packet_elements():
    ctx = DMIMetadataContext(
        max_num_microbatches=1,
        max_batch_size=3,
        num_scopes=1,
        field_specs=(valid_count_field_spec(), dataset_id_field_spec()),
        device="cpu",
    )
    ctx.set_active_fields(("valid_count",))
    ctx.begin_iteration(1)
    ctx.ingest_microbatch(0, {"valid_count": [7, 5, 3]})

    packets = ctx.active_cpu_packets(0)
    assert len(packets) == 1
    assert packets[torch.int64].tolist() == [7, 5, 3]


def test_constant_only_phase_with_cpu_dataset_field_posts_no_collective():
    ctx = DMIMetadataContext(
        max_num_microbatches=1,
        max_batch_size=2,
        num_scopes=1,
        field_specs=(dataset_id_field_spec(),),
        device="cpu",
    )
    ctx.set_active_fields(())
    dist = FakeDist()
    propagator = TorchDistributedMetadataPropagator(
        ctx,
        rank=0,
        pp_source_rank=0,
        tp_source_rank=0,
        pp_cpu_ranks=[0, 1],
        tp_cpu_ranks=[0],
        pp_cpu_group="pp_cpu",
        tp_cpu_group=None,
        dist_module=dist,
    )

    propagator.begin_iteration(1)
    propagator.ingest_microbatch(0, {})
    propagator.end_iteration()

    assert ctx.active_cpu_packets(0) == {}
    assert dist.broadcast_calls == []


def test_empty_field_specs_means_no_metadata_fields():
    ctx = DMIMetadataContext(
        max_num_microbatches=1,
        max_batch_size=2,
        num_scopes=1,
        field_specs=(),
        device="cpu",
    )

    assert ctx.field_specs == {}
    ctx.begin_iteration(1)
    ctx.ingest_microbatch(0, {})
    ctx.enter_scope("fwd", 0, 0)
    with pytest.raises(KeyError, match="valid_count"):
        ctx.source_cpu("valid_count", 0)


def test_context_rejects_missing_and_oversized_fields():
    ctx = DMIMetadataContext(
        max_num_microbatches=1,
        max_batch_size=2,
        num_scopes=1,
        device="cpu",
    )

    ctx.begin_iteration(1)
    with pytest.raises(KeyError, match="valid_count"):
        ctx.ingest_microbatch(0, {})
    with pytest.raises(ValueError, match="too many elements"):
        ctx.ingest_microbatch(0, {"valid_count": [1, 2, 3]})


def test_context_rejects_out_of_range_iteration_ids():
    ctx = DMIMetadataContext(
        max_num_microbatches=1,
        max_batch_size=2,
        num_scopes=1,
        device="cpu",
    )

    with pytest.raises(ValueError, match="exceeds"):
        ctx.begin_iteration(2)

    ctx.begin_iteration(1)
    with pytest.raises(IndexError, match="microbatch_id"):
        ctx.ingest_microbatch(1, {"valid_count": [1]})
    with pytest.raises(IndexError, match="scope_id"):
        ctx.enter_scope("fwd", 1, 0)


def test_local_metadata_propagator_wraps_context_lifecycle():
    ctx = DMIMetadataContext(
        max_num_microbatches=2,
        max_batch_size=3,
        num_scopes=2,
        device="cpu",
    )
    propagator = LocalMetadataPropagator(ctx)

    propagator.begin_iteration(2)
    propagator.ingest_microbatch(0, {"valid_count": [8, 7]})
    propagator.wait_microbatch(0)
    propagator.enter_scope("fwd", 1, 0)

    assert torch.equal(ctx.current("valid_count", "fwd", 1), torch.tensor([8, 7, 0]))

    propagator.end_iteration()
    assert ctx.active_num_microbatches == 0


def test_local_metadata_propagator_wait_checks_active_microbatch():
    ctx = DMIMetadataContext(
        max_num_microbatches=1,
        max_batch_size=2,
        num_scopes=1,
        device="cpu",
    )
    propagator = LocalMetadataPropagator(ctx)

    propagator.begin_iteration(1)
    with pytest.raises(IndexError, match="microbatch_id"):
        propagator.wait_microbatch(1)


def test_metadata_propagator_interface_is_abstract():
    ctx = DMIMetadataContext(
        max_num_microbatches=1,
        max_batch_size=2,
        num_scopes=1,
        device="cpu",
    )

    with pytest.raises(TypeError):
        DMIMetadataPropagator(ctx)


class FakeWork:
    def __init__(self, name: str) -> None:
        self.name = name
        self.wait_count = 0

    def wait(self):
        self.wait_count += 1


class FakeDist:
    def __init__(self) -> None:
        self.broadcast_calls = []

    def broadcast(self, tensor, src, group=None, async_op=False):
        work = FakeWork(f"broadcast:{len(self.broadcast_calls)}")
        self.broadcast_calls.append((tensor, src, group, async_op, work))
        if async_op:
            return work
        return work


def test_torch_distributed_receiver_preposts_pp_broadcast_and_waits_microbatch():
    ctx = DMIMetadataContext(
        max_num_microbatches=2,
        max_batch_size=3,
        num_scopes=1,
        device="cpu",
    )
    dist = FakeDist()
    propagator = TorchDistributedMetadataPropagator(
        ctx,
        rank=2,
        pp_source_rank=0,
        tp_source_rank=2,
        pp_cpu_ranks=[0, 2],
        tp_cpu_ranks=[2],
        pp_cpu_group="pp_cpu",
        tp_cpu_group=None,
        dist_module=dist,
    )

    propagator.begin_iteration(2)

    assert len(dist.broadcast_calls) == 2
    assert dist.broadcast_calls[0][0].data_ptr() == ctx.source_cpu_tensor("valid_count", 0).data_ptr()
    assert dist.broadcast_calls[0][1:4] == (0, "pp_cpu", True)
    assert dist.broadcast_calls[1][0].data_ptr() == ctx.source_cpu_tensor("valid_count", 1).data_ptr()

    ctx.source_cpu_tensor("valid_count", 0).copy_(torch.tensor([9, 8, 7]))
    propagator.enter_scope("fwd", 0, 0)

    assert dist.broadcast_calls[0][4].wait_count == 1
    assert dist.broadcast_calls[1][4].wait_count == 0
    assert torch.equal(ctx.current("valid_count", "fwd", 0), torch.tensor([9, 8, 7]))

    propagator.end_iteration()
    assert dist.broadcast_calls[1][4].wait_count == 1
    assert ctx.active_num_microbatches == 0


def test_torch_distributed_receiver_reuses_pp_metadata_for_backward_scope():
    ctx = DMIMetadataContext(
        max_num_microbatches=1,
        max_batch_size=2,
        num_scopes=1,
        device="cpu",
    )
    dist = FakeDist()
    propagator = TorchDistributedMetadataPropagator(
        ctx,
        rank=1,
        pp_source_rank=0,
        tp_source_rank=1,
        pp_cpu_ranks=[0, 1],
        tp_cpu_ranks=[1],
        pp_cpu_group="pp_cpu",
        tp_cpu_group=None,
        dist_module=dist,
    )

    propagator.begin_iteration(1)
    ctx.source_cpu_tensor("valid_count", 0).copy_(torch.tensor([9, 8]))

    propagator.enter_scope("fwd", 0, 0)
    propagator.enter_scope("bwd", 0, 0)

    assert len(dist.broadcast_calls) == 1
    assert dist.broadcast_calls[0][4].wait_count == 1
    assert torch.equal(ctx.current("valid_count", "fwd", 0), torch.tensor([9, 8]))
    assert torch.equal(ctx.current("valid_count", "bwd", 0), torch.tensor([9, 8]))


def test_torch_distributed_source_ingests_and_broadcasts_to_pp_receivers():
    ctx = DMIMetadataContext(
        max_num_microbatches=2,
        max_batch_size=3,
        num_scopes=1,
        device="cpu",
    )
    dist = FakeDist()
    propagator = TorchDistributedMetadataPropagator(
        ctx,
        rank=0,
        pp_source_rank=0,
        tp_source_rank=0,
        pp_cpu_ranks=[0, 2, 4],
        tp_cpu_ranks=[0],
        pp_cpu_group="pp_cpu",
        tp_cpu_group=None,
        dist_module=dist,
    )

    propagator.begin_iteration(2)
    propagator.ingest_microbatch(1, {"valid_count": [5, 4]})
    propagator.enter_scope("fwd", 0, 1)

    assert torch.equal(ctx.source("valid_count", 1), torch.tensor([5, 4, 0]))
    assert torch.equal(ctx.current("valid_count", "fwd", 0), torch.tensor([5, 4, 0]))
    assert len(dist.broadcast_calls) == 1
    assert dist.broadcast_calls[0][0].data_ptr() == ctx.source_cpu_tensor("valid_count", 1).data_ptr()
    assert dist.broadcast_calls[0][1:4] == (0, "pp_cpu", True)

    propagator.end_iteration()
    assert dist.broadcast_calls[0][4].wait_count == 1


def test_torch_distributed_tp_broadcast_runs_after_pp_wait():
    ctx = DMIMetadataContext(
        max_num_microbatches=1,
        max_batch_size=2,
        num_scopes=1,
        device="cpu",
    )
    dist = FakeDist()
    propagator = TorchDistributedMetadataPropagator(
        ctx,
        rank=3,
        pp_source_rank=0,
        tp_source_rank=2,
        pp_cpu_ranks=[0, 2],
        tp_cpu_ranks=[2, 3],
        pp_cpu_group=None,
        tp_cpu_group="tp_cpu",
        dist_module=dist,
    )

    propagator.begin_iteration(1)
    ctx.source_cpu_tensor("valid_count", 0).copy_(torch.tensor([6, 5]))
    propagator.enter_scope("fwd", 0, 0)

    assert len(dist.broadcast_calls) == 1
    assert dist.broadcast_calls[0][0].data_ptr() == ctx.source_cpu_tensor("valid_count", 0).data_ptr()
    assert dist.broadcast_calls[0][1:4] == (2, "tp_cpu", False)
    assert torch.equal(ctx.current("valid_count", "fwd", 0), torch.tensor([6, 5]))


def test_torch_distributed_non_participant_only_checks_ids():
    ctx = DMIMetadataContext(
        max_num_microbatches=1,
        max_batch_size=2,
        num_scopes=1,
        device="cpu",
    )
    dist = FakeDist()
    propagator = TorchDistributedMetadataPropagator(
        ctx,
        rank=5,
        pp_source_rank=0,
        tp_source_rank=1,
        pp_cpu_ranks=[0, 1],
        tp_cpu_ranks=[1],
        dist_module=dist,
    )

    propagator.begin_iteration(1)
    propagator.ingest_microbatch(0, {"valid_count": [1, 2]})
    propagator.wait_microbatch(0)

    assert dist.broadcast_calls == []
