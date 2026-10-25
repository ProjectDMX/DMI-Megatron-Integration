"""Eager packed metadata: cached validation and opt-in direct source rows."""
import pytest
import torch
from dmi.api.v1 import TransportType
from dmi_megatron_integration.adapter import MegatronTrainingContext
from dmi_megatron_integration.hooks.specs import (
    MegatronDistributedInfo, MegatronMetadataField, ShardPolicy,
)
from dmi_megatron_integration.metadata_context import DMIMetadataContext
from tests.test_megatron_adapter import (
    FakeEngine, _make_adaptor, _make_hook, MegatronHookSpec, MegatronOutputSpec,
    MegatronHookBinding, DimSpec,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def packed_hook(name="packed"):
    return _make_hook(MegatronHookSpec(name=name, layer_no=0, outputs=[
        MegatronOutputSpec(name=name, input_shape=[DimSpec.SEQ, DimSpec.BATCH, 4],
                           output_shape=[-1, 4], dtype=torch.float32,
                           transport_type=TransportType.SEQ_PREFIX_PACK)], preprocess=None))


def consumer_hook():
    # Reads the current valid-count buffer itself (like router_probs_mean).
    return _make_hook(MegatronHookSpec(
        name="consumer", layer_no=0,
        outputs=[MegatronOutputSpec(name="consumer", input_shape=[DimSpec.BATCH], dtype=torch.float32)],
        preprocess=lambda x, valid: x,
        preprocess_metadata_fields=frozenset({MegatronMetadataField.VALID_COUNT}),
    ))


def build(*hooks, enabled=True):
    context = DMIMetadataContext(max_num_microbatches=2, max_batch_size=2, num_scopes=1,
                                 dims={DimSpec.SEQ: 8}, device=torch.device("cuda"))
    context.set_active_fields(("valid_count",))
    adaptor = _make_adaptor(FakeEngine(), "test", dims={DimSpec.SEQ: 8, DimSpec.BATCH: 2})
    adaptor.attach_hooks(model_hooks=tuple(MegatronHookBinding(hook=h) for h in hooks),
                         iteration_hooks=(), metadata_context=context)
    adaptor.configure_eager_direct_packed_metadata(enabled)
    return context, adaptor


def ingest(context, counts_by_microbatch):
    context.begin_iteration(len(counts_by_microbatch))
    for microbatch, counts in enumerate(counts_by_microbatch):
        context.ingest_microbatch(microbatch, {"valid_count": counts})


def test_direct_scope_skips_packed_copies_and_hook_reads_source_rows():
    hook = packed_hook()
    context, adaptor = build(hook)
    ingest(context, [(3, 2), (8, 1)])
    assert adaptor.eager_direct_packed_allowed("valid")
    x = torch.zeros(8, 2, 4, device="cuda")
    for microbatch, expected in ((0, (3, 2)), (1, (8, 1))):
        context.skip_packed_current_copies = True
        context.enter_scope("fwd", 0, microbatch)
        context.skip_packed_current_copies = False
        adaptor.set_current_event(MegatronTrainingContext(
            global_batch_id=1, microbatch_id=microbatch, valid_counts=expected, phase="valid",
            _packing=context.prepared_packing(microbatch)))
        adaptor.set_direct_packed_microbatch(microbatch)
        output = hook.spec.preprocess(x)
        valid, prefix = output.producer_meta
        assert valid.data_ptr() == context.source("valid_count", microbatch).data_ptr()
        assert valid.tolist() == list(expected)
        assert prefix.tolist() == [0, expected[0], sum(expected)]
        # Current buffers were not touched by the direct scope.
        assert context.current("valid_count", "fwd", 0).tolist() == [0, 0]


def test_non_direct_scope_still_copies_and_uses_current_buffers():
    hook = packed_hook()
    context, adaptor = build(hook, enabled=False)
    ingest(context, [(3, 2)])
    assert not adaptor.eager_direct_packed_allowed("valid")
    context.enter_scope("fwd", 0, 0)
    output = hook.spec.preprocess(torch.zeros(8, 2, 4, device="cuda"))
    assert output.producer_meta[0] is hook.valid_count_fwd
    assert hook.valid_count_fwd.tolist() == [3, 2]


def test_direct_mode_disabled_by_train_phase_or_direct_consumer():
    _, adaptor = build(packed_hook())
    assert not adaptor.eager_direct_packed_allowed("train")
    _, adaptor = build(packed_hook(), consumer_hook())
    assert not adaptor.eager_direct_packed_allowed("valid")


def test_cached_validation_still_sees_rebound_tensors_and_rejects_bad_ones():
    hook = packed_hook()
    _, adaptor = build(hook, enabled=False)
    x = torch.zeros(8, 2, 4, device="cuda")
    first = hook.spec.preprocess(x).producer_meta[0]
    hook.valid_count_fwd = torch.tensor([1, 1], device="cuda")
    assert hook.spec.preprocess(x).producer_meta[0] is hook.valid_count_fwd is not first
    hook.valid_count_fwd = torch.tensor([1.0, 1.0], device="cuda")
    with pytest.raises(TypeError, match="int64"):
        hook.spec.preprocess(x)


def tp1_router_like_hook():
    info = MegatronDistributedInfo(
        global_rank=0, world_size=1, tp_rank=0, tp_world_size=1, pp_rank=0, pp_world_size=1,
        dp_rank=0, dp_world_size=1, ep_rank=0, ep_world_size=1, cp_rank=0, cp_world_size=1,
        sequence_parallel_enabled=False)
    hook = _make_hook(MegatronHookSpec(
        name="router_like", layer_no=0, shard_policy=ShardPolicy.TP_SEQUENCE_SHARDED,
        outputs=[MegatronOutputSpec(name="router_like", input_shape=[DimSpec.SEQ, DimSpec.BATCH, 4],
                                    output_shape=[-1, 4], dtype=torch.float32,
                                    transport_type=TransportType.SEQ_PREFIX_PACK)],
        preprocess=None))
    hook.megatron_distributed_info = info
    context = DMIMetadataContext(max_num_microbatches=1, max_batch_size=2, num_scopes=1,
                                 dims={DimSpec.SEQ: 8}, device=torch.device("cuda"),
                                 megatron_distributed_info=info, tp_sequence_sharded_enabled=True)
    context.set_active_fields(("valid_count",))
    adaptor = _make_adaptor(FakeEngine(), "test", dims={DimSpec.SEQ: 8, DimSpec.BATCH: 2})
    adaptor.attach_hooks(model_hooks=(MegatronHookBinding(hook=hook),), iteration_hooks=(),
                         metadata_context=context)
    adaptor.configure_eager_direct_packed_metadata(True)
    return hook, context, adaptor


def test_tp1_sequence_sharded_hook_skips_narrow_and_reads_local_source_rows():
    hook, context, adaptor = tp1_router_like_hook()
    ingest(context, [(5, 7)])
    x = torch.zeros(8, 2, 4, device="cuda")
    output = hook.spec.preprocess(x)
    assert output.tensor is x  # no no-op narrow view
    context.skip_packed_current_copies = True
    context.enter_scope("fwd", 0, 0)
    context.skip_packed_current_copies = False
    adaptor.set_direct_packed_microbatch(0)
    valid, prefix = hook.spec.preprocess(x).producer_meta
    assert valid.data_ptr() == context.source("tp_seq_sharded_valid_count", 0).data_ptr()
    assert valid.tolist() == [5, 7] and prefix.tolist() == [0, 5, 12]
    assert context.current("tp_seq_sharded_valid_count", "fwd", 0).tolist() == [0, 0]


def test_direct_iteration_uses_value_cached_counts_without_source_uploads():
    hook = packed_hook()
    context, adaptor = build(hook)
    context.direct_packed_iteration = True
    ingest(context, [(3, 2), (3, 2)])
    assert context.source("valid_count", 0).tolist() == [0, 0]  # no per-microbatch upload
    first = context.direct_count("valid_count_prefix", 0)
    assert first is context.direct_count("valid_count_prefix", 1)  # cached by value
    x = torch.zeros(8, 2, 4, device="cuda")
    context.skip_packed_current_copies = True
    context.enter_scope("fwd", 0, 1)
    context.skip_packed_current_copies = False
    adaptor.set_direct_packed_microbatch(1)
    valid, prefix = hook.spec.preprocess(x).producer_meta
    assert valid.tolist() == [3, 2] and prefix.tolist() == [0, 3, 5]
    assert prefix is first


def test_non_direct_scope_after_direct_ingest_uploads_source_rows_first():
    hook = packed_hook()
    context, adaptor = build(hook)
    context.direct_packed_iteration = True
    ingest(context, [(4, 1)])
    context.enter_scope("fwd", 0, 0)  # not a direct scope: must see real counts
    assert context.source("valid_count", 0).tolist() == [4, 1]
    assert hook.valid_count_fwd.tolist() == [4, 1]
    assert hook.valid_count_prefix_fwd.tolist() == [0, 4, 5]
