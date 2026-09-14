"""Exact eager/replay sizing with independent global and TP-local counts."""

import gc
import weakref

import pytest
import torch

from dmi.api.v1 import (
    HookOutput, ProducerPlan, ProducerPlanEntry, RecordType, StepReservation,
    TransportSpec, TransportType,
)
from dmi.hooks.producer_plan import _align_up
from dmi_megatron_integration.adapter import (
    _CapturedSizing, _PreparedPacking, _ProducerSemantics, _make_packed_size_fn,
    MegatronTrainingContext, MegatronEventCoordinates, MegatronFullIterationPlan,
)
from dmi_megatron_integration.metadata_context import DMIMetadataContext, _prepare_valid_counts
from dmi_megatron_integration.records.metadata import MegatronRecordMetadata
from tests.test_megatron_adapter import FakeEngine, TinyModel, _make_adaptor, DimSpec, HookPhase


def _packed(width=4, local=False):
    spec = TransportSpec("packed", transport_type=TransportType.SEQ_PREFIX_PACK,
                         feature_bytes=width * 4, output_shape=(-1, width))
    entry = ProducerPlanEntry.from_output(
        output_id=1 << 16, output_spec=spec,
        output=HookOutput(torch.empty(4 if local else 8, 2, width)),
    )
    semantic = _ProducerSemantics(
        output_id=entry.output_id, act_name=spec.name, layer_no=0,
        record_type=RecordType.PER_SAMPLE, transport_type=spec.transport_type,
        record_dp_rank=None, record_shard_rank=None, need_token_range=True,
        suppress_recompute=True, record_direction="fwd",
        tp_sequence_start=4 if local else None, tp_sequence_length=4 if local else None,
        packed_size_fn=_make_packed_size_fn(width * 4, local=local),
    )
    return entry, semantic


@pytest.mark.parametrize("width,mode", [(4, "simplified"), (1, "recursive")])
def test_graph_count_sources_and_individual_alignment(width, mode):
    global_entry, global_semantic = _packed(width)
    local_entry, local_semantic = _packed(width, local=True)
    sizing = _CapturedSizing((global_entry, local_entry), (global_semantic, local_semantic))
    assert sizing.mode == mode
    fn = sizing.size_fn
    for counts in ((7, 2), (1, 4), (0, 0), (8, 8)):
        packing = _PreparedPacking(_prepare_valid_counts(counts))
        total, sizes = sizing.resolve((packing, packing))
        expected = (_align_up(sum(counts) * width * 4),
                    _align_up(sum(max(0, value - 4) for value in counts) * width * 4))
        assert sizes == expected
        assert total == sum(expected)
        assert sizing.size_fn is fn
    assert sizing._parameters is None


@pytest.mark.parametrize("second_width,mode", [(4, "simplified"), (8, "recursive")])
def test_full_iteration_uniform_and_nonuniform_coefficients(second_width, mode):
    first, first_semantic = _packed(4)
    second, second_semantic = _packed(second_width)
    sizing = _CapturedSizing((first, second), (first_semantic, second_semantic), (0, 1))
    assert sizing.mode == mode
    results = []
    for count_pair in (((6, 0), (2, 0)), ((2, 0), (6, 0))):
        packings = tuple(_PreparedPacking(_prepare_valid_counts(counts)) for counts in count_pair)
        total, sizes = sizing.resolve(packings)
        expected = (sum(count_pair[0]) * 16, sum(count_pair[1]) * second_width * 4)
        assert sizes == expected
        assert total == sum(expected)
        results.append(total)
    assert (results[0] == results[1]) == (mode == "simplified")
    if mode == "simplified":
        assert sizing.coefficients == (16, 0, 0)  # Not 32: two MB counts are summed.


def test_unaligned_constant_prevents_simplification():
    entry, semantic = _packed()
    constant = ProducerPlanEntry.from_output(
        output_id=entry.output_id, output_spec=TransportSpec("scalar"),
        output=HookOutput(torch.empty(1)),
    )
    sizing = _CapturedSizing((entry, constant), (semantic, semantic))
    assert sizing.mode == "recursive"
    packing = _PreparedPacking(_prepare_valid_counts((3, 2)))
    assert sizing.resolve((packing, packing)) == (96, (80, 16))


def test_graph_cache_is_owned_by_exact_plan_and_released():
    engine = FakeEngine()
    model = TinyModel()
    adaptor = _make_adaptor(engine, "test", dims={DimSpec.BATCH: 2, DimSpec.NUM_EXPERTS: 4})
    adaptor.attach_model(model, hook_selection="router-summary")
    adaptor.begin_capture_plan(warmup_enabled=True, capture_direction=HookPhase.FWD)
    model.router0(torch.ones(2, 4))
    plan = adaptor.finish_capture_plan()
    state = adaptor._plan_state(plan)
    ref = weakref.ref(plan)
    key = id(plan)
    adaptor.prepare_replay_capacity_only(plan, plan_direction=HookPhase.FWD, live_direction=HookPhase.FWD)
    assert adaptor._plan_state(plan) is state
    del plan
    gc.collect()
    assert ref() is None
    assert key not in adaptor._plan_states_by_id


def test_prefix_same_total_does_not_reuse_different_sample_boundaries():
    first = _prepare_valid_counts((3, 2))
    second = _prepare_valid_counts((1, 4))
    assert first.token_count == second.token_count == 5
    assert first.prefix == (0, 3, 5)
    assert second.prefix == (0, 1, 5)
    with pytest.raises(ValueError, match="non-negative"):
        _prepare_valid_counts((-1, 2))


def test_packed_capacity_and_replay_use_live_counts_not_unpacked_bound():
    entry, semantic = _packed()
    plan = ProducerPlan((entry,))
    engine = FakeEngine()
    engine.payload_cap = engine.staging_cap = 128
    adaptor = _make_adaptor(engine, "sizing")
    adaptor._remember_plan(plan, (semantic,))
    assert entry.aligned_reservation_bytes == 256
    prior_metadata = []
    for iteration, counts in enumerate(((3, 2), (0, 0), (8, 8), (1, 4))):
        ctx = MegatronTrainingContext(
            global_batch_id=iteration, microbatch_id=0, valid_counts=counts,
            dataset_ids=(iteration, iteration + 1),
        )
        adaptor.set_current_event(ctx)
        expected = StepReservation.OVERSIZED if counts == (8, 8) else StepReservation.RESERVED
        before = len(engine.record_runtime.reservation_calls)
        assert adaptor.prepare_replay_capacity_only(
            plan, plan_direction="fwd", live_direction="fwd",
        ) is expected
        assert len(engine.record_runtime.reservation_calls) == before
        assert adaptor.prepare_replay(plan, plan_direction="fwd") is expected
        if expected is StepReservation.RESERVED:
            assert engine.record_runtime.reservation_calls[-1] == (sum(counts) * 16, 1)
            metadata = engine.record_runtime.replay_calls[-1][1][0]
            assert metadata.valid_counts == counts
            prior_metadata.append(metadata)
        else:
            assert len(engine.record_runtime.reservation_calls) == before
    assert prior_metadata[0].valid_counts == (3, 2)
    assert prior_metadata[0].dataset_ids == (0, 1)
    assert entry.aligned_reservation_bytes == 256


def test_full_iteration_exact_reservations_use_each_supplied_context():
    entry, semantic = _packed()
    physical = ProducerPlan((entry, entry))
    events = tuple(MegatronEventCoordinates(microbatch_id=i, direction="fwd") for i in (0, 1))
    plan = MegatronFullIterationPlan.from_plan_and_events(physical, events, (semantic, semantic))
    engine = FakeEngine()
    engine.payload_cap = engine.staging_cap = 192
    adaptor = _make_adaptor(engine, "sizing")
    # This unrelated active event must never replace explicitly supplied contexts.
    adaptor.set_current_event(MegatronTrainingContext(99, 9, (8, 8)))
    contexts = tuple(MegatronTrainingContext(7, i, counts, dataset_ids=(i, i))
                     for i, counts in enumerate(((3, 2), (1, 4))))
    assert adaptor.prepare_full_iteration_replay(plan, contexts) is StepReservation.RESERVED
    assert engine.record_runtime.reservation_calls == [(160, 2)]
    submitted_plan, metadata = engine.record_runtime.replay_calls[0]
    assert submitted_plan is plan.producer_plan
    assert [item.valid_counts for item in metadata] == [(3, 2), (1, 4)]
    assert [item.global_batch_id for item in metadata] == [7, 7]
    assert [item.microbatch_id for item in metadata] == [0, 1]


def test_normalized_metadata_snapshots_lists_and_reuses_tuples(monkeypatch):
    counts, datasets = [3, 2], [10, 20]
    ctx = MegatronTrainingContext(7, 0, counts, datasets, attempt_id=2)
    engine = FakeEngine()
    adaptor = _make_adaptor(engine, "metadata")
    _, semantic = _packed()
    adaptor.set_current_event(ctx)
    original = MegatronRecordMetadata.__post_init__
    normalized_calls = []

    def tracked(self, normalized):
        normalized_calls.append(normalized)
        original(self, normalized)

    monkeypatch.setattr(MegatronRecordMetadata, "__post_init__", tracked)
    first = adaptor._record_metadata(ctx, semantic)
    second = adaptor._record_metadata(ctx, semantic)
    assert normalized_calls == [True, True]
    assert first.valid_counts is second.valid_counts is ctx.valid_counts
    assert first.dataset_ids is second.dataset_ids is ctx.dataset_ids
    counts[:] = [1, 4]
    datasets[:] = [30, 40]
    following = MegatronTrainingContext(8, 0, counts, datasets, direction="bwd", attempt_id=3)
    adaptor.set_current_event(following)
    third = adaptor._record_metadata(following, semantic)
    assert first.valid_counts == (3, 2) and first.dataset_ids == (10, 20)
    assert first.global_batch_id == 7 and first.attempt_id == 2
    assert third.valid_counts == (1, 4) and third.dataset_ids == (30, 40)
    assert third.global_batch_id == 8 and third.attempt_id == 3
    assert ctx._packing.global_counts.prefix == (0, 3, 5)
    assert following._packing.global_counts.prefix == (0, 1, 5)
    adaptor.clear_current_event()
    assert adaptor._event_packing is None


def test_prepared_microbatch_snapshot_shared_across_contexts_and_replaced():
    owner = DMIMetadataContext(
        max_num_microbatches=1, max_batch_size=2, num_scopes=2, device="cpu",
    )
    owner.begin_iteration(1)
    owner.ingest_microbatch(0, {"valid_count": [7, 2]})
    packing = owner.prepared_packing(0)
    fwd = MegatronTrainingContext(1, 0, (7, 2), _packing=packing)
    bwd = MegatronTrainingContext(1, 0, (7, 2), direction="bwd", scope_id=1, _packing=packing)
    assert fwd._packing is bwd._packing is packing
    local = fwd._packing.counts_for(4, 4)
    assert bwd._packing.counts_for(4, 4) is local
    assert local.counts == (3, 0) and local.prefix == (0, 3, 3)
    with pytest.raises(ValueError, match="disagree"):
        MegatronTrainingContext(1, 0, (1, 4), _packing=packing)
    owner.ingest_microbatch(0, {"valid_count": [1, 4]})
    assert owner.prepared_packing(0) is not packing
    assert owner.prepared_packing(0).global_counts.prefix == (0, 1, 5)
    assert fwd._packing.global_counts.prefix == (0, 7, 9)
    owner.end_iteration()
    assert owner.prepared_packing(0) is None
