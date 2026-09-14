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


def test_ingest_microbatch_uses_cpu_authority_for_gpu_counts():
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

    assert torch.equal(ctx.source("valid_count", 0), torch.tensor([7, 6, 0]))
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

    assert torch.equal(ctx.source("valid_count", 1), torch.tensor([3, 2, 0]))
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


def test_cpu_only_metadata_lifecycle_never_accesses_cuda(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("CPU-only metadata accessed CUDA")

    monkeypatch.setattr(torch.cuda, "current_stream", forbidden)
    monkeypatch.setattr(torch.cuda, "Event", forbidden)
    ctx = DMIMetadataContext(
        max_num_microbatches=1,
        max_batch_size=2,
        num_scopes=2,
        field_specs=(valid_count_field_spec(gpu_visible=False), dataset_id_field_spec()),
        device="cuda",
    )
    for counts in ([3, 2], [1, 4]):
        ctx.begin_iteration(1)
        ctx.ingest_microbatch(
            0, {}, cpu_fields={"valid_count": counts, "dataset_id": [5, 6]},
        )
        ctx.enter_scope("fwd", 0, 0)
        ctx.enter_scope("bwd", 1, 0)
        assert ctx.source_cpu("valid_count", 0) == tuple(counts)
        assert ctx.prepared_packing(0).global_counts.prefix == (0, counts[0], 5)
        ctx.end_iteration()


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


def test_torch_distributed_receiver_preposts_pp_broadcast_and_waits_microbatch(monkeypatch):
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

    ctx.register_valid_count_prefix()
    propagator.begin_iteration(2)

    assert len(dist.broadcast_calls) == 2
    assert dist.broadcast_calls[0][0].data_ptr() == ctx.source_cpu_tensor("valid_count", 0).data_ptr()
    assert dist.broadcast_calls[0][1:4] == (0, "pp_cpu", True)
    assert dist.broadcast_calls[1][0].data_ptr() == ctx.source_cpu_tensor("valid_count", 1).data_ptr()

    work = dist.broadcast_calls[0][4]
    original_wait = work.wait

    def complete_receive():
        assert ctx.prepared_packing(0) is None
        ctx.source_cpu_tensor("valid_count", 0).copy_(torch.tensor([9, 8, 7]))
        original_wait()

    monkeypatch.setattr(work, "wait", complete_receive)
    propagator.enter_scope("fwd", 0, 0)

    assert dist.broadcast_calls[0][4].wait_count == 1
    assert dist.broadcast_calls[1][4].wait_count == 0
    assert torch.equal(ctx.current("valid_count", "fwd", 0), torch.tensor([9, 8, 7]))
    assert ctx.prepared_packing(0).global_counts.prefix == (0, 9, 17, 24)
    assert ctx.current("valid_count_prefix", "fwd", 0).tolist() == [0, 9, 17, 24]

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


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("tp_rank", [0, 1])
def test_owned_prefix_uploads_refresh_equal_totals_and_scope_views(device, tp_rank, monkeypatch):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA metadata upload test")
    from tests.test_megatron_adapter import _tp_sequence_distributed_info

    def forbid_gpu_scan(*args, **kwargs):
        raise AssertionError("packing metadata must not call torch.cumsum")

    monkeypatch.setattr(torch, "cumsum", forbid_gpu_scan)
    ctx = DMIMetadataContext(
        max_num_microbatches=2, max_batch_size=3, num_scopes=2,
        dims={DimSpec.SEQ: 8}, device=device,
        megatron_distributed_info=_tp_sequence_distributed_info(
            tp_rank=tp_rank, sequence_parallel_enabled=True,
        ),
        tp_sequence_sharded_enabled=True,
    )
    names = ["valid_count", "tp_seq_sharded_valid_count"]
    for local in (False, True):
        name = ctx.register_valid_count_prefix(tp_local=local)
        pointer = ctx.current(name, "fwd", 0).data_ptr()
        assert ctx.register_valid_count_prefix(tp_local=local) == name
        assert ctx.current(name, "fwd", 0).data_ptr() == pointer
    staging_pointers = {name: buf.data_ptr() for name, buf in ctx._pinned_count_buffers.items()}
    if device == "cuda":
        assert all(buf.is_pinned() for buf in ctx._pinned_count_buffers.values())
    previous = None
    for counts in ([3, 2], [1, 4], [7, 2], [0, 0], [8, 8]):
        ctx.begin_iteration(2)
        ctx.ingest_microbatch(0, {"valid_count": counts})
        ctx.ingest_microbatch(1, {"valid_count": [0, 1]})
        prepared = ctx.prepared_packing(0)
        assert prepared.global_counts.counts == (*counts, 0)
        if previous is not None:
            assert prepared is not previous
        previous = prepared
        for direction, scope, mb in (("fwd", 0, 0), ("bwd", 0, 1), ("fwd", 1, 1), ("bwd", 1, 0)):
            ctx.enter_scope(direction, scope, mb)
            snapshot = ctx.prepared_packing(mb)
            assert snapshot is ctx.prepared_packing(mb)
            for name in names:
                expected = (snapshot.global_counts if name == "valid_count" else
                            snapshot.counts_for(tp_rank * 4, 4))
                assert ctx.current(name, direction, scope).cpu().tolist() == list(expected.counts)
                assert ctx.current(name + "_prefix", direction, scope).cpu().tolist() == list(expected.prefix)
                assert expected.token_count == expected.prefix[-1]
        assert staging_pointers == {name: buf.data_ptr() for name, buf in ctx._pinned_count_buffers.items()}
        ctx.end_iteration()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA metadata graph test")
@pytest.mark.parametrize("tp_rank", [0, 1])
def test_owned_prefix_graph_replay_uses_refreshed_sources(tp_rank):
    from tests.test_megatron_adapter import _tp_sequence_distributed_info

    ctx = DMIMetadataContext(
        max_num_microbatches=2, max_batch_size=2, num_scopes=1,
        dims={DimSpec.SEQ: 8}, device="cuda",
        megatron_distributed_info=_tp_sequence_distributed_info(
            tp_rank=tp_rank, sequence_parallel_enabled=True,
        ),
        tp_sequence_sharded_enabled=True,
    )
    names = ["valid_count", "valid_count_prefix", "tp_seq_sharded_valid_count",
             "tp_seq_sharded_valid_count_prefix"]
    ctx.register_valid_count_prefix()
    ctx.register_valid_count_prefix(tp_local=True)
    ctx.begin_iteration(2)
    for mb in (0, 1):
        ctx.ingest_microbatch(mb, {"valid_count": [3, 2]})
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        snapshots = []
        for direction, mb in (("fwd", 0), ("bwd", 1)):
            ctx.enter_scope(direction, 0, mb)
            snapshots.append([ctx.current(name, direction, 0).clone() for name in names])
    torch.cuda.current_stream().wait_stream(stream)
    # Repeated asynchronous replacements exercise the same pinned/source slots.
    # Only the assertions read back GPU data; the production path does not.
    for counts in ([1, 4], [7, 2], [0, 0], [8, 8]):
        ctx.ingest_microbatch(0, {"valid_count": counts})
        ctx.ingest_microbatch(1, {"valid_count": list(reversed(counts))})
        graph.replay()
        for mb, tensors in enumerate(snapshots):
            snapshot = ctx.prepared_packing(mb)
            global_counts = snapshot.global_counts
            local_counts = snapshot.counts_for(tp_rank * 4, 4)
            expected = [global_counts.counts, global_counts.prefix, local_counts.counts, local_counts.prefix]
            assert [tensor.cpu().tolist() for tensor in tensors] == [list(value) for value in expected]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA asynchronous upload test")
def test_owned_prefix_async_source_and_staging_reuse():
    ctx = DMIMetadataContext(
        max_num_microbatches=1, max_batch_size=2, num_scopes=1, device="cuda",
    )
    ctx.register_valid_count_prefix()
    ctx.begin_iteration(1)
    upload = torch.cuda.Stream()
    consume = torch.cuda.Stream()
    upload.wait_stream(torch.cuda.current_stream())
    consume.wait_stream(torch.cuda.current_stream())
    snapshots = []
    cases = ((3, 2), (1, 4), (7, 2), (0, 0), (8, 8)) * 4
    # Reuse one source/staging slot across streams without per-case GPU reads
    # or whole-device synchronization. Scope copies protect source reuse;
    # consumers remain ordered on their own stream before current-view reuse.
    for counts in cases:
        with torch.cuda.stream(upload):
            torch.cuda._sleep(100_000)
            ctx.ingest_microbatch(0, {"valid_count": counts})
        with torch.cuda.stream(consume):
            ctx.enter_scope("fwd", 0, 0)
            snapshots.append((ctx.current("valid_count", "fwd", 0).clone(),
                              ctx.current("valid_count_prefix", "fwd", 0).clone()))
    consume.synchronize()
    for counts, (actual_counts, actual_prefix) in zip(cases, snapshots):
        assert actual_counts.cpu().tolist() == list(counts)
        assert actual_prefix.cpu().tolist() == [0, counts[0], counts[0] + counts[1]]
