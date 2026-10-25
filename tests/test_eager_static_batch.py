"""Opt-in eager static batch: one reservation per confirmed static microbatch."""
import pytest
import torch
from dmi.api.v1 import StepReservation, TransportType
from dmi_megatron_integration.adapter import MegatronTrainingContext
from tests.test_megatron_adapter import (
    FakeEngine, FakeRecordRuntime, _make_adaptor, _make_hook, MegatronHookSpec,
    MegatronOutputSpec, MegatronHookBinding, DimSpec,
)


class RecordingRuntime(FakeRecordRuntime):
    """Report reserved records as RESERVED and keep per-record submissions."""

    def __init__(self):
        super().__init__()
        self.records = []
        self.submit_calls = 0

    def _emit_prepared_output(self, entry, metadata, output, *, reservation_bytes):
        super()._emit_prepared_output(entry, metadata, output, reservation_bytes=reservation_bytes)
        self.records.append((entry, metadata, reservation_bytes))
        self.submit_calls += 1
        return StepReservation.RESERVED

    def prepare_replay(self, plan, metadata, *, reservation_bytes=None):
        result = super().prepare_replay(plan, metadata, reservation_bytes=reservation_bytes)
        self.records.extend(zip(plan.entries, tuple(metadata), reservation_bytes))
        self.submit_calls += 1
        return result


def build(enabled=True, payload_cap=1 << 30):
    engine = FakeEngine()
    engine.record_runtime = RecordingRuntime()
    engine.payload_cap = engine.staging_cap = payload_cap
    adaptor = _make_adaptor(engine, 'test', dims={DimSpec.SEQ: 4, DimSpec.BATCH: 2})
    packed = _make_hook(MegatronHookSpec(name='packed', layer_no=0, outputs=[
        MegatronOutputSpec(name='packed', input_shape=[DimSpec.SEQ, DimSpec.BATCH, 4],
                           output_shape=[-1, 4], dtype=torch.float32,
                           transport_type=TransportType.SEQ_PREFIX_PACK)], preprocess=None))
    loss = _make_hook(MegatronHookSpec(name='loss', layer_no=1, outputs=[
        MegatronOutputSpec(name='loss_mean', input_shape=[DimSpec.BATCH, 1], dtype=torch.float32),
        MegatronOutputSpec(name='loss_tokens', input_shape=[DimSpec.BATCH, 1], dtype=torch.int64),
    ], preprocess=lambda a, b: (a, b)))
    adaptor.attach_hooks(model_hooks=(MegatronHookBinding(hook=packed), MegatronHookBinding(hook=loss)),
                         iteration_hooks=())
    adaptor.configure_eager_static_batch(enabled)
    return engine, adaptor, packed, loss


COUNTS = ((3, 1), (4, 4), (0, 2), (2, 3), (1, 1), (4, 0))


def run_microbatch(adaptor, packed, loss, index, counts, *, phase='valid', fire_loss=True):
    packed.valid_count_fwd = torch.tensor(counts)
    packed.valid_count_prefix_fwd = torch.tensor([0, counts[0], sum(counts)])
    for tensor in (packed.valid_count_fwd, packed.valid_count_prefix_fwd):
        PTRS[tensor.data_ptr()] = tensor.tolist()
    ctx = MegatronTrainingContext(global_batch_id=7, microbatch_id=index, valid_counts=counts,
                                  phase=phase)
    adaptor.set_current_event(ctx)
    if adaptor.hook_runtime.eager_static_batch_enabled:
        adaptor.hook_runtime.begin_static_microbatch(ctx)
    packed(torch.arange(32.).reshape(4, 2, 4))
    if fire_loss:
        loss(torch.ones(2, 1), torch.ones(2, 1, dtype=torch.int64))


def finish(adaptor):
    adaptor.hook_runtime.finish_static_microbatch()
    adaptor.clear_current_event()


def submissions(enabled, phase='valid'):
    engine, adaptor, packed, loss = build(enabled)
    per_microbatch = []
    for index, counts in enumerate(COUNTS):
        before = engine.record_runtime.submit_calls
        run_microbatch(adaptor, packed, loss, index, counts, phase=phase)
        per_microbatch.append(engine.record_runtime.submit_calls - before)
    finish(adaptor)
    return engine, per_microbatch


def test_batch_matches_per_hook_records_and_uses_one_submit_after_two_learning_microbatches():
    off, off_calls = submissions(False)
    on, on_calls = submissions(True)
    assert off_calls == [3] * len(COUNTS)
    assert on_calls == [3, 3] + [1] * (len(COUNTS) - 2)
    assert on.record_runtime.records == off.record_runtime.records
    # Every output still launches its producer exactly once.
    assert len(on.record_runtime.dispatch_calls) == len(off.record_runtime.dispatch_calls)
    assert len(on.record_runtime.dispatch_calls) == 3 * len(COUNTS)


def test_train_phase_never_batches():
    _, calls = submissions(True, phase='train')
    assert calls == [3] * len(COUNTS)


def test_dynamic_output_set_is_ineligible():
    engine, adaptor, packed, loss = build()
    calls = []
    for index, counts in enumerate(COUNTS):
        before = engine.record_runtime.submit_calls
        run_microbatch(adaptor, packed, loss, index, counts, fire_loss=index % 2 == 0)
        calls.append(engine.record_runtime.submit_calls - before)
    finish(adaptor)
    assert calls == [3, 1, 3, 1, 3, 1]
    assert not engine.record_runtime.replay_calls


def test_batch_larger_than_ring_capacity_uses_per_hook_path():
    # Microbatch 2 needs 64 bytes (fits); microbatch 3 needs 112 (> capacity).
    engine, adaptor, packed, loss = build(payload_cap=100)
    calls = []
    for index, counts in enumerate(COUNTS[:4]):
        before = engine.record_runtime.submit_calls
        run_microbatch(adaptor, packed, loss, index, counts)
        calls.append(engine.record_runtime.submit_calls - before)
    finish(adaptor)
    assert calls == [3, 3, 1, 3]
    assert len(engine.record_runtime.replay_calls) == 1


def test_missing_output_after_reservation_raises():
    _, adaptor, packed, loss = build()
    for index, counts in enumerate(COUNTS[:2]):
        run_microbatch(adaptor, packed, loss, index, counts)
    run_microbatch(adaptor, packed, loss, 2, COUNTS[2], fire_loss=False)
    with pytest.raises(RuntimeError, match='reserved 3 outputs but 1 fired'):
        finish(adaptor)


def test_out_of_order_or_reshaped_output_raises():
    _, adaptor, packed, loss = build()
    for index, counts in enumerate(COUNTS[:2]):
        run_microbatch(adaptor, packed, loss, index, counts)
    ctx = MegatronTrainingContext(global_batch_id=7, microbatch_id=2, valid_counts=(1, 1),
                                  phase='valid')
    adaptor.set_current_event(ctx)
    adaptor.hook_runtime.begin_static_microbatch(ctx)
    with pytest.raises(RuntimeError, match='does not match the reserved plan'):
        loss(torch.ones(2, 1), torch.ones(2, 1, dtype=torch.int64))
    adaptor.hook_runtime.reset_static_batches()
    for index, counts in enumerate(COUNTS[:2]):
        run_microbatch(adaptor, packed, loss, index, counts)
    adaptor.set_current_event(ctx)
    adaptor.hook_runtime.begin_static_microbatch(ctx)
    packed.valid_count_fwd = torch.tensor((1, 1))
    packed.valid_count_prefix_fwd = torch.tensor([0, 1, 2])
    with pytest.raises(RuntimeError, match='does not match the reserved plan'):
        packed(torch.zeros(4, 2, 8))


def test_disabled_runtime_keeps_no_static_state():
    engine, adaptor, packed, loss = build(False)
    for index, counts in enumerate(COUNTS):
        run_microbatch(adaptor, packed, loss, index, counts)
    runtime = adaptor.hook_runtime
    assert runtime._static_recording is None and not runtime._static_candidates
    assert not runtime._static_plans and not engine.record_runtime.replay_calls


def test_other_reservations_inside_armed_batch_raise():
    _, adaptor, packed, loss = build()
    for index, counts in enumerate(COUNTS[:2]):
        run_microbatch(adaptor, packed, loss, index, counts)
    ctx = MegatronTrainingContext(global_batch_id=7, microbatch_id=2, valid_counts=(1, 1),
                                  phase='valid')
    adaptor.set_current_event(ctx)
    adaptor.hook_runtime.begin_static_microbatch(ctx)
    plan = adaptor.hook_runtime._static_armed.plan
    with pytest.raises(RuntimeError, match='inside an eager static batch'):
        adaptor.prepare_replay(plan, ctx, plan_direction='fwd')


class LaunchRecordingRuntime(RecordingRuntime):
    """Keeps HookPointV1's real _dispatch so the fast launch can arm."""

    def bind_hook(self, hook, *, hook_runtime, gate_tensor=None, gate_value=0):
        super().bind_hook(hook, hook_runtime=hook_runtime, gate_tensor=gate_tensor,
                          gate_value=gate_value)
        del hook._dispatch


PTRS = {}


class FakeLauncher:
    """Stands in for the native StaticLauncher: checks slots, records launches."""

    calls = []

    def __init__(self, engine):
        self.slots, self.counts = [], {}

    def add(self, packed, shape, dtype, device, batch, feature_bytes, gate_ptr, gate_value):
        self.slots.append((packed, tuple(shape), dtype, feature_bytes, gate_value))
        return len(self.slots) - 1

    def set_counts(self, slot, valid_ptr, prefix_ptr):
        self.counts[slot] = (PTRS[valid_ptr], PTRS[prefix_ptr])

    def launch(self, slot, tensor):
        packed, shape, dtype, feature_bytes, gate_value = self.slots[slot]
        assert tuple(tensor.shape) == shape and tensor.dtype == dtype
        valid, prefix = self.counts[slot] if packed else (None, None)
        self.calls.append(("seq" if packed else "id", tensor.tolist(), valid, prefix,
                           feature_bytes if packed else None, gate_value))


def record_ops(monkeypatch):
    import types
    import dmi_megatron_integration.adapter as adapter_module
    calls = []

    def seq(payload, tensor, valid, prefix, feature_bytes, gate, gate_value):
        calls.append(("seq", tensor.tolist(), valid.tolist(), prefix.tolist(), feature_bytes, gate_value))

    def ident(payload, tensor, gate, gate_value):
        calls.append(("id", tensor.tolist(), None, None, None, gate_value))
    monkeypatch.setattr(torch.ops.ring, "record_producer_seq_prefix_pack", seq, raising=False)
    monkeypatch.setattr(torch.ops.ring, "record_producer", ident, raising=False)
    FakeLauncher.calls = calls
    monkeypatch.setattr(adapter_module, "_native_extension",
                        lambda: types.SimpleNamespace(StaticLauncher=FakeLauncher))
    return calls


def build_launch(enabled=True):
    import types
    engine = FakeEngine()
    engine.record_runtime = LaunchRecordingRuntime()
    engine.record_runtime._transport = types.SimpleNamespace(_ring_engine=object(), null_offload=False)
    adaptor = _make_adaptor(engine, 'test', dims={DimSpec.SEQ: 4, DimSpec.BATCH: 2})
    packed = _make_hook(MegatronHookSpec(name='packed', layer_no=0, outputs=[
        MegatronOutputSpec(name='packed', input_shape=[DimSpec.SEQ, DimSpec.BATCH, 4],
                           output_shape=[-1, 4], dtype=torch.float32,
                           transport_type=TransportType.SEQ_PREFIX_PACK)], preprocess=None))
    loss = _make_hook(MegatronHookSpec(name='loss', layer_no=1, outputs=[
        MegatronOutputSpec(name='loss_mean', input_shape=[DimSpec.BATCH, 1], dtype=torch.float32),
        MegatronOutputSpec(name='loss_tokens', input_shape=[DimSpec.BATCH, 1], dtype=torch.int64),
    ], preprocess=lambda a, b: (a * 2, b)))
    adaptor.attach_hooks(model_hooks=(MegatronHookBinding(hook=packed), MegatronHookBinding(hook=loss)),
                         iteration_hooks=())
    adaptor.configure_eager_static_batch(enabled)
    return engine, adaptor, packed, loss


def test_fast_launch_matches_per_hook_dispatch_and_skips_runtime_calls(monkeypatch):
    calls_on = record_ops(monkeypatch)
    on, adaptor, packed, loss = build_launch()
    prepare_calls = []
    original = adaptor.hook_runtime.prepare_output
    monkeypatch.setattr(adaptor.hook_runtime, 'prepare_output',
                        lambda **kw: prepare_calls.append(1) or original(**kw))
    per_microbatch = []
    for index, counts in enumerate(COUNTS):
        before = len(prepare_calls)
        run_microbatch(adaptor, packed, loss, index, counts)
        per_microbatch.append(len(prepare_calls) - before)
    finish(adaptor)
    # Learning microbatches use the per-hook path; armed ones only launch.
    assert per_microbatch == [3, 3] + [0] * (len(COUNTS) - 2)
    launched_on = list(calls_on)
    calls_on.clear()
    off, adaptor_off, packed_off, loss_off = build_launch(enabled=False)
    for index, counts in enumerate(COUNTS):
        run_microbatch(adaptor_off, packed_off, loss_off, index, counts)
    launched_off = list(calls_on)
    assert launched_on == launched_off
    assert len(launched_on) == 3 * len(COUNTS)
    assert on.record_runtime.records == off.record_runtime.records


def test_fast_launch_mismatch_falls_to_normal_path_and_raises(monkeypatch):
    record_ops(monkeypatch)
    _, adaptor, packed, loss = build_launch()
    for index, counts in enumerate(COUNTS[:2]):
        run_microbatch(adaptor, packed, loss, index, counts)
    ctx = MegatronTrainingContext(global_batch_id=7, microbatch_id=2, valid_counts=(1, 1),
                                  phase='valid')
    adaptor.set_current_event(ctx)
    adaptor.hook_runtime.begin_static_microbatch(ctx)
    assert adaptor.hook_runtime.fast_forward is not None
    with pytest.raises(RuntimeError, match='does not match the reserved plan'):
        loss(torch.ones(2, 1), torch.ones(2, 1, dtype=torch.int64))


def test_disabled_hook_inside_armed_batch_is_a_no_op(monkeypatch):
    calls = record_ops(monkeypatch)
    _, adaptor, packed, loss = build_launch()
    for index, counts in enumerate(COUNTS[:2]):
        run_microbatch(adaptor, packed, loss, index, counts)
    extra = _make_hook(MegatronHookSpec(name='extra', layer_no=5, outputs=[
        MegatronOutputSpec(name='extra', input_shape=[DimSpec.BATCH], dtype=torch.float32)],
        preprocess=None, enabled_by=frozenset({'never'})))
    run_microbatch(adaptor, packed, loss, 2, COUNTS[2])
    before = len(calls)
    extra.enabled = False
    extra(torch.ones(2))  # not bound, not planned: untouched
    assert len(calls) == before
    finish(adaptor)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_graphed_loss_summary_replay_is_bitwise_identical_and_falls_back():
    from dmi_megatron_integration.adapter import _GraphedPreprocess
    from dmi_megatron_integration.hooks.megatron_loss_summary import per_sample_loss_from_token_loss as fn
    graphed = _GraphedPreprocess(fn)
    for seed in range(4):
        g = torch.Generator(device="cuda").manual_seed(seed)
        loss = torch.rand(2, 512, device="cuda", generator=g) * 10
        mask = (torch.rand(2, 512, device="cuda", generator=g) > 0.3).float()
        expected = fn(loss, mask)
        actual = graphed(loss, mask)
        assert all(torch.equal(a, e) for a, e in zip(actual, expected))
    other = graphed(torch.ones(3, 8, device="cuda"), torch.ones(3, 8, device="cuda"))
    assert other[0].shape == (3, 1)  # different shape: eager fallback, graph untouched


def test_module_forward_hooks_keep_the_normal_call_path(monkeypatch):
    record_ops(monkeypatch)
    _, adaptor, packed, loss = build_launch()
    for index, counts in enumerate(COUNTS[:2]):
        run_microbatch(adaptor, packed, loss, index, counts)
    seen = []
    handle = packed.register_forward_hook(lambda module, args, out: seen.append(1))
    run_microbatch(adaptor, packed, loss, 2, COUNTS[2])
    finish(adaptor)
    handle.remove()
    assert seen == [1]


def test_profiler_keeps_the_normal_call_path_and_identical_records(monkeypatch):
    # Under a profiler (likewise torch.compile/jit tracing or stream capture),
    # armed microbatches skip fast_forward: every hook goes through
    # nn.Module.__call__, prepare_output and the torch.ops producers.
    calls = record_ops(monkeypatch)
    on, adaptor, packed, loss = build_launch()
    prepare_calls = []
    original = adaptor.hook_runtime.prepare_output
    monkeypatch.setattr(adaptor.hook_runtime, 'prepare_output',
                        lambda **kw: prepare_calls.append(1) or original(**kw))
    per_microbatch = []
    with torch.profiler.profile():
        for index, counts in enumerate(COUNTS):
            before = len(prepare_calls)
            run_microbatch(adaptor, packed, loss, index, counts)
            per_microbatch.append(len(prepare_calls) - before)
    finish(adaptor)
    assert per_microbatch == [3] * len(COUNTS)
    launched_profiled = list(calls)
    calls.clear()
    off, adaptor_off, packed_off, loss_off = build_launch(enabled=False)
    for index, counts in enumerate(COUNTS):
        run_microbatch(adaptor_off, packed_off, loss_off, index, counts)
    assert launched_profiled == list(calls)
    assert on.record_runtime.records == off.record_runtime.records
