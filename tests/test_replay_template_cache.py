"""Graph-replay reservation templates (MegatronAdaptor._prepare_cached_replay).

The same replay sequence runs once with the template path and once with the
general path only; records, ring payload bytes, expected record counts and
reservation results must be identical. Covers repeated replays across
iterations/microbatches, a recomputation replay (invocation id 1), changing
valid counts, and an oversized plan."""
import json

import pytest
import torch
from dmi.api.v1 import DropConfig, MonitoringConfig, MonitoringEngine, RingConfig, StepReservation
from dmi_megatron_integration.adapter import MegatronAdaptor, MegatronHookBinding, MegatronTrainingContext
from dmi_megatron_integration.hooks.specs import DimSpec, HookPhase, MegatronHookSpec, MegatronOutputSpec
from dmi_megatron_integration.records.format import MegatronRecordFormat
from tests.test_megatron_adapter import _make_hook

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

# (global batch, microbatch, valid counts, replays of this microbatch)
SCHEDULE = (
    (1, 0, (1, 1), 1), (1, 1, (1, 1), 2), (1, 2, (1, 1), 1),
    (2, 0, (1, 1), 1), (2, 1, (1, 0), 1), (2, 2, (1, 1), 2),
    (3, 0, (1, 1), 1), (3, 1, (0, 0), 1), (3, 2, (1, 1), 1),
)


def _run(tmp_path, *, cached, ring_bytes=1 << 20, width=4, graphed=True):
    cfg = RingConfig()
    cfg.payload_ring_bytes = cfg.pinned_staging_bytes = ring_bytes
    cfg.task_ring_entries = 256
    engine = MonitoringEngine(config=MonitoringConfig(storage_backend="drop", drop=DropConfig(str(tmp_path))),
                              ring_config=cfg, record_mode_v1=True)
    results, counts, hits = [], [], []
    try:
        fmt = MegatronRecordFormat("replay_cache")
        runtime = engine.create_record_runtime(fmt)
        hook = _make_hook(MegatronHookSpec(name="replay_probe", layer_no=3, outputs=(MegatronOutputSpec(
            name="replay_probe", input_shape=(DimSpec.BATCH, width), dtype=torch.float32),)),
            suppress_recompute=False)
        adaptor = MegatronAdaptor(engine, runtime, "replay_cache", dims={DimSpec.BATCH: 2})
        adaptor.attach_hooks(model_hooks=(MegatronHookBinding(hook=hook),), iteration_hooks=(),
                             activation_recompute_enabled=True)
        assert adaptor.invocation_allocator is not None
        original_replay, original_emit = adaptor._prepare_cached_replay, adaptor._emit_cached

        def cached_replay(*args):
            result = original_replay(*args) if cached else None
            hits.append(result is not None)
            return result

        def cached_emit(*args, **kwargs):
            result = original_emit(*args, **kwargs) if cached else None
            hits.append(result is not None)
            return result

        class NoEntryCache(dict):
            def __setitem__(self, key, value):
                pass

        adaptor._prepare_cached_replay = cached_replay
        adaptor._emit_cached = cached_emit
        if not cached:
            adaptor._eager_entries = NoEntryCache()
        local = torch.zeros(2, width, device="cuda")
        graph = plan = None
        batch_open = None
        for batch, microbatch, valid, replays in SCHEDULE:
            if batch != batch_open:
                if batch_open is not None:
                    adaptor.end_attempt(attempt_id=0)
                    counts.append(fmt.take_expected_count("train", batch_open, 0))
                adaptor.begin_attempt(phase="train", global_batch_id=batch, attempt_id=0)
                fmt.count_records = True
                batch_open = batch
            ctx = MegatronTrainingContext(global_batch_id=batch, microbatch_id=microbatch, valid_counts=valid)
            adaptor.set_current_event(ctx)
            if graphed and graph is None:
                adaptor.begin_capture_plan(warmup_enabled=True, capture_direction=HookPhase.FWD)
                graph = torch.cuda.CUDAGraph()
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.graph(graph, stream=stream):
                    hook(local)
                torch.cuda.current_stream().wait_stream(stream)
                plan = adaptor.finish_capture_plan()
            for replay in range(replays):
                local.copy_(torch.arange(2 * width, dtype=torch.float32, device="cuda").reshape(2, width)
                            + 1000 * batch + 100 * microbatch + 10 * replay)
                if not graphed:
                    hook(local)
                    continue
                result = adaptor.prepare_replay(plan, plan_direction=HookPhase.FWD)
                results.append(result)
                if result is StepReservation.RESERVED:
                    graph.replay()
            adaptor.clear_current_event()
        adaptor.end_attempt(attempt_id=0)
        counts.append(fmt.take_expected_count("train", batch_open, 0))
        torch.cuda.synchronize()
        engine.flush_and_wait()
        payload = runtime._transport._ring_payload[:ring_bytes].cpu().clone()
    finally:
        engine.close()
    events = [json.loads(line) for line in (tmp_path / "rank_00000/events.jsonl").read_text().splitlines()]
    rows = [r for e in events if e["type"] == "record" for r in e["rows"] if r.get("act_name") == "replay_probe"]
    return rows, payload, counts, results, hits


def test_template_path_matches_general_path(tmp_path):
    fast = _run(tmp_path / "fast", cached=True)
    general = _run(tmp_path / "general", cached=False)
    rows, payload, counts, results, hits = fast
    assert rows == general[0]
    assert torch.equal(payload, general[1])
    assert counts == general[2]
    assert results == general[3]
    # The template path was taken, and the recomputation replays used id 1.
    assert sum(hits) >= 4
    assert sorted({r["invocation_id"] for r in rows}) == [0, 1]
    recomputed = [(r["global_batch_id"], r["microbatch_id"]) for r in rows if r["invocation_id"] == 1]
    assert sorted(set(recomputed)) == [(1, 1), (2, 2)]


def test_oversized_plan_matches_general_path(tmp_path):
    # A 2x4096 float32 replay (32 KiB) cannot fit a 4 KiB ring.
    fast = _run(tmp_path / "fast", cached=True, ring_bytes=4096, width=4096)
    general = _run(tmp_path / "general", cached=False, ring_bytes=4096, width=4096)
    assert fast[3] == general[3]
    assert set(fast[3]) == {StepReservation.OVERSIZED}
    assert fast[0] == general[0] == []
    assert fast[2] == general[2]


def test_eager_template_path_matches_general_path(tmp_path):
    fast = _run(tmp_path / "fast", cached=True, graphed=False)
    general = _run(tmp_path / "general", cached=False, graphed=False)
    rows, payload, counts, _, hits = fast
    assert rows == general[0]
    assert torch.equal(payload, general[1])
    assert counts == general[2]
    assert sum(hits) >= 4
    assert sorted({r["invocation_id"] for r in rows}) == [0, 1]


def test_eager_oversized_cpu_direct_matches_general_path(tmp_path):
    fast = _run(tmp_path / "fast", cached=True, ring_bytes=4096, width=4096, graphed=False)
    general = _run(tmp_path / "general", cached=False, ring_bytes=4096, width=4096, graphed=False)
    assert fast[0] == general[0] and len(fast[0]) > 0
    assert fast[2] == general[2]
