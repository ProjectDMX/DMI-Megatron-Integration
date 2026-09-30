"""Source selection, topology composition, packing and eager-record regression."""
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from dmi.api.v1 import HookOutput, OutputSizingMode, RecordType, TransportType
from dmi_megatron_integration.hooks.source_sampling import (
    BUILTIN_ROUND_ROBIN, EP_OUTPUT, ExpertSourceCapture, SourceSampling,
    read_source_sampling, round_robin, segment_ranges,
)
from dmi_megatron_integration.materialization.ep_clickhouse_reconstruction import reconstruct_moe_clickhouse_rows
from dmi_megatron_integration.records.schema import TRAINING_ROW_COORDINATE_COLUMN_NAMES
from dmi_megatron_integration.topology.ep_topology_manifest import (
    FrozenMegatronEPTopologyManifest, MoELayerPlacement,
)


def last_sources(iteration, num_sources, count):
    return tuple(range(num_sources - count, num_sources))


def test_round_robin_and_custom_plugin():
    assert [round_robin(i, 8, 2, 1) for i in range(5)] == [
        (1, 2), (3, 4), (5, 6), (7, 0), (1, 2)]
    policy = SourceSampling(BUILTIN_ROUND_ROBIN, {"count": 2, "offset": 1})
    assert policy.select(4, 8) == (0, 7)
    assert SourceSampling.from_dict(policy.to_dict()).select(4, 8) == (0, 7)
    custom = SourceSampling(f"{__name__}.last_sources", {"count": 2})
    assert custom.select(100, 8) == (6, 7)
    with pytest.raises(ValueError):
        policy.select(1, 1)


def test_unsupported_and_disabled_hooks_do_not_load_plugin(tmp_path):
    config = tmp_path / "hooks.yaml"
    config.write_text("hooks:\n  hidden_states:\n    source_sampling:\n      function: nonexistent.plugin\n")
    with pytest.warns(UserWarning, match="hidden_states.*ignoring"):
        assert read_source_sampling(str(config), ep_enabled=True) is None
    config.write_text("hooks:\n  moe_packed_weighted_output:\n    source_sampling:\n      function: nonexistent.plugin\n      args: {}\n")
    assert read_source_sampling(str(config), ep_enabled=False) is None


def _fixture(ep, etp, edp, tp, pp, blocks, selected_count):
    """Build independently labelled route values, then emulate physical packing."""
    width = ep * etp * edp
    assert width % tp == 0
    world = width * pp
    def rank(p, d, e, t):
        return p * width + d * ep * etp + e * etp + t
    manifest = FrozenMegatronEPTopologyManifest(
        model_id="sampling-test",
        tp_groups=tuple(tuple(p * width + d * tp + t for t in range(tp))
                        for p in range(pp) for d in range(width // tp)),
        pp_groups=tuple(tuple(p * width + r for p in range(pp)) for r in range(width)),
        dp_groups=tuple(tuple(p * width + d * tp + t for d in range(width // tp))
                        for p in range(pp) for t in range(tp)),
        cp_groups=tuple((r,) for r in range(world)),
        ep_groups=tuple(tuple(rank(p, d, e, t) for e in range(ep))
                        for p in range(pp) for d in range(edp) for t in range(etp)),
        etp_groups=tuple(tuple(rank(p, d, e, t) for t in range(etp))
                         for p in range(pp) for d in range(edp) for e in range(ep)),
        dispatch_groups=tuple(tuple(rank(p, d, e, t) for e in range(ep) for t in range(etp))
                              for p in range(pp) for d in range(edp)),
        expert_dp_groups=tuple(tuple(rank(p, d, e, t) for d in range(edp))
                               for p in range(pp) for e in range(ep) for t in range(etp)),
        layer_placements=tuple(MoELayerPlacement(p * 2 + v, p, v) for p in range(pp) for v in range(2)),
        local_expert_order_by_ep_rank=tuple(tuple(e * blocks + b for b in range(blocks)) for e in range(ep)),
        sequence_parallel=tp > 1, top_k=min(2, ep * blocks), dispatcher_type="alltoall",
        permutation_mode="non_fused", etp_composition="matching_row_sum", dropless=True, padded=False,
    )
    policy = SourceSampling(BUILTIN_ROUND_ROBIN, {"count": selected_count, "offset": 1})
    full, sampled, expected = {}, {}, {}
    def add(target, name, tensor, p, r, layer, batch, per_sample):
        coords = dict(model_id=manifest.model_id, act_name=name, direction="fwd", phase="train",
                      global_batch_id=batch, attempt_id=0, microbatch_id=0, layer_no=layer,
                      invocation_id=0, shard_rank=r, dp_rank=(r % width) // tp if per_sample else -1,
                      dataset_id=11 if per_sample else -1, sample_index=0 if per_sample else -1,
                      token_start=0 if per_sample else -1, token_end=3 * tp if per_sample else -1)
        target.setdefault(name, []).append((tuple(coords[k] for k in TRAINING_ROW_COORDINATE_COLUMN_NAMES), tensor))
    for batch in (1, 2, 3):
        for p in range(pp):
            for v in range(2):
                layer = p * 2 + v
                routes = {}
                for d in range(edp):
                    for t in range(etp):
                        for e in range(ep):
                            r = rank(p, d, e, t)
                            ids = torch.tensor([sorted({(r + j + k) % (ep * blocks) for k in range(manifest.top_k)})
                                                for j in range(3)], dtype=torch.int64)
                            routes[r] = [(j, int(expert)) for j in range(3) for expert in ids[j]]
                            inverse = torch.tensor([j for j, ex in sorted(routes[r], key=lambda route: route[1])], dtype=torch.int64)
                            for target in (full, sampled):
                                add(target, "router_topk_expert_ids", ids, p, r, layer, batch, True)
                                add(target, "moe_inverse_map", inverse, p, r, layer, batch, False)
                selected = set()
                for d in range(edp):
                    ordered = tuple(rank(p, d, e, t) for t in range(etp) for e in range(ep))
                    selected.update(ordered[u] for u in policy.select(batch, len(ordered)))
                    for e in range(ep):
                        layout = [(r, j, expert) for expert in range(e * blocks, (e + 1) * blocks)
                                  for r in ordered for j, ex in routes[r] if ex == expert]
                        vals = torch.tensor([[r * 100 + j * 10 + expert, batch + layer + 1]
                                             for r, j, expert in layout], dtype=torch.float32).reshape(-1, 2) / etp
                        mask = torch.tensor([r in selected for r, _, _ in layout], dtype=torch.bool)
                        for t in range(etp):
                            dest = rank(p, d, e, t)
                            add(full, EP_OUTPUT, vals, p, dest, layer, batch, False)
                            add(sampled, EP_OUTPUT, vals[mask], p, dest, layer, batch, False)
                for r, route_list in routes.items():
                    for j in range(3):
                        truth = torch.tensor([[r * 100 + j * 10 + ex, batch + layer + 1]
                                              for jj, ex in route_list if jj == j], dtype=torch.float32)
                        expected[(batch, layer, (r % width) // tp, (r % tp) * 3 + j)] = (truth, r in selected)
    return manifest, replace(manifest, hook_capture={EP_OUTPUT: {"source_sampling": policy.to_dict()}}), full, sampled, expected


@pytest.mark.parametrize("ep,etp,edp,tp,pp,blocks", [
    (1, 1, 1, 1, 1, 1), (2, 1, 2, 1, 2, 1), (1, 2, 1, 2, 1, 2),
    (2, 1, 1, 2, 2, 2), (2, 2, 2, 4, 2, 1), (3, 2, 2, 3, 2, 2),
])
@pytest.mark.parametrize("selection", ["one", "multiple", "all"])
def test_full_and_sampled_combined_topologies(ep, etp, edp, tp, pp, blocks, selection):
    count = {"one": 1, "multiple": min(2, ep * etp), "all": ep * etp}[selection]
    manifest, sampled_manifest, full, sampled, expected = _fixture(ep, etp, edp, tp, pp, blocks, count)
    for current_manifest, records, is_sampled in ((manifest, full, False), (sampled_manifest, sampled, True)):
        restored = FrozenMegatronEPTopologyManifest.from_dict(current_manifest.to_dict())
        observed = {}
        for invocation in reconstruct_moe_clickhouse_rows(restored, records):
            for domain in invocation.source_domains:
                for index, coord in enumerate(domain.token_coordinates):
                    key = (invocation.key.global_batch_id, invocation.key.layer_no, domain.dense_dp_rank, coord.token_index)
                    observed[key] = domain.weighted_outputs[index]
        wanted = {key: value for key, (value, keep) in expected.items() if keep or not is_sampled}
        assert observed.keys() == wanted.keys()
        for key in wanted:
            torch.testing.assert_close(observed[key], wanted[key])
    old = manifest.to_dict()
    old.pop("hook_capture")
    assert FrozenMegatronEPTopologyManifest.from_dict(old).hook_capture == {}


def test_ranges_and_eager_size_are_actual_not_fractional():
    policy = SourceSampling(BUILTIN_ROUND_ROBIN, {"count": 1, "offset": 1})
    counts = torch.tensor([[2, 1, 3], [1, 4, 2]], dtype=torch.int64)
    dispatcher = SimpleNamespace(num_global_tokens_per_local_expert=counts.T.contiguous())
    capture = ExpertSourceCapture(policy, dispatcher)
    ctx = SimpleNamespace(global_batch_id=1)
    capture.hook = SimpleNamespace(_hook_runtime=SimpleNamespace(adaptor=SimpleNamespace(current_context=ctx)))
    values = torch.arange(26, dtype=torch.float32).reshape(13, 2)
    output = capture(values, counts)
    assert output.tensor is values
    assert capture.output_nbytes() == 5 * 2 * 4
    assert output.producer_meta[0].tolist() == [2, 7]
    assert output.producer_meta[1].tolist() == [3, 11]
    counts.zero_()
    dispatcher.num_global_tokens_per_local_expert.zero_()
    assert capture(values[:0], counts).producer_meta[0].tolist() == [0, 0]
    assert capture.output_nbytes() == 0


def test_explicit_ranges_and_per_execution_sizing():
    from dmi_megatron_integration.startup import _make_hook, _validate_hook_contract
    from dmi_megatron_integration.hooks.specs import MegatronHookSpec, MegatronOutputSpec, HookPhase, DimSpec
    from dmi_megatron_integration.adapter import MegatronAdaptor, _ProducerSemantics, MegatronTrainingContext
    from dmi_megatron_integration.records.format import MegatronRecordFormat
    from tests.test_megatron_adapter import FakeEngine, _make_adaptor
    counts = torch.tensor([2, 7]), torch.tensor([3, 11])
    nbytes = 40
    output_spec = MegatronOutputSpec(EP_OUTPUT, (DimSpec.ACTUAL_TOKEN_PACKED, 2), torch.float32,
        output_shape=(DimSpec.ACTUAL_TOKEN_PACKED, 2), transport_type=TransportType.SEGMENTED_PACK,
        sizing_mode=OutputSizingMode.RUNTIME_SIZED, segment_ranges_from_preprocess=True,
        eager_nbytes=lambda: nbytes)
    policy = MegatronHookSpec("sample", 0, (output_spec,), record_type=RecordType.PER_EXECUTION, need_token_range=False)
    hook = _make_hook(policy, hook_phase=HookPhase.FWD)
    _validate_hook_contract(hook)
    assert not policy.binding_metadata_fields
    engine = FakeEngine()
    runtime = engine.record_runtime
    adapter = _make_adaptor(engine, "sample-test")
    adapter.current_context = MegatronTrainingContext(global_batch_id=1, microbatch_id=0, valid_counts=())
    physical = output_spec.resolve({}, record_type=RecordType.PER_EXECUTION)
    output = HookOutput(torch.ones(13, 2), counts)
    enriched = adapter._enrich_output(hook, physical, output)
    assert enriched.producer_meta[0] is counts[0]
    semantic = _ProducerSemantics(1 << 16, EP_OUTPUT, 0, RecordType.PER_EXECUTION, TransportType.SEGMENTED_PACK,
        -1, 0, False, True, "fwd", eager_nbytes=lambda: nbytes)
    adapter.emit_immediate_output(semantics=semantic, output_spec=physical, output=enriched)
    call = runtime.emit_calls[-1]
    entry, metadata = call[0], call[1]
    assert entry.output_shape == (5, 2)
    assert entry.reservation_upper_bytes == 40
    assert len(MegatronRecordFormat("test").encode(metadata, entry).rows) == 1
    hook.sample_start_ptr_fwd, hook.sample_end_ptr_fwd = counts
    assert adapter._enrich_output(hook, physical, HookOutput(output.tensor)).producer_meta[0] is counts[0]
    with pytest.raises(ValueError, match="starts and ends"):
        adapter._enrich_output(hook, physical, HookOutput(output.tensor, (counts[0],)))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU range replay test")
def test_ranges_replay_with_changed_counts_and_selection():
    counts = torch.tensor([[2, 1, 3], [1, 4, 2]], device="cuda", dtype=torch.int64)
    selected = torch.tensor([1], device="cuda", dtype=torch.int64)
    starts = torch.empty(2, device="cuda", dtype=torch.int64)
    ends = torch.empty_like(starts)
    with torch.cuda.stream(torch.cuda.Stream()):
        segment_ranges(counts, selected, starts, ends)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        segment_ranges(counts, selected, starts, ends)
    for cpu_counts, source, expected in [
        ([[2, 1, 3], [1, 4, 2]], 1, ([2, 7], [3, 11])),
        ([[0, 4, 1], [2, 0, 3]], 2, ([4, 7], [5, 10])),
        ([[0, 0, 0], [0, 0, 0]], 0, ([0, 0], [0, 0])),
    ]:
        counts.copy_(torch.tensor(cpu_counts, dtype=torch.int64))
        selected.fill_(source)
        graph.replay()
        assert (starts.tolist(), ends.tolist()) == expected


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires native GPU transport")
def test_sampled_gpu_payloads_reach_storage_including_empty():
    """Synthetic dispatch counts exercise real segmented transport, not a gather mock."""
    import os
    import uuid
    from dmi_megatron_integration.adapter import MegatronAdaptor, MegatronTrainingContext
    from dmi_megatron_integration.hooks.specs import DimSpec
    from dmi_megatron_integration.records.format import MegatronRecordFormat
    from dmi_megatron_integration.startup import _install_moe_packed_weighted_output_hooks
    from tests.test_megatron_e2e_clickhouse import _build_training_engine, _clickhouse_client_or_skip
    from tests.test_megatron_ep_reconstruction_e2e import _read_moe_rows
    from tests.test_megatron_real_training_e2e import _wait_for_exact_act_rows

    client = _clickhouse_client_or_skip()
    database = os.environ.get("DMX_DB_DATABASE", "default")
    table = f"dmi_sampled_payload_{uuid.uuid4().hex}"
    dispatcher = type("MoEAlltoAllTokenDispatcher", (), {})()
    dispatcher.cudagraph_attrs = []
    layer = type("MoELayer", (torch.nn.Module,), {})()
    layer.layer_number = 1
    layer.config = SimpleNamespace(params_dtype=torch.float32)
    layer.token_dispatcher = dispatcher
    layer.dmi_moe_packed_weighted_output = None
    policy = SourceSampling(BUILTIN_ROUND_ROBIN, {"count": 2})
    _install_moe_packed_weighted_output_hooks(layer, policy)
    engine = None
    try:
        engine = _build_training_engine(model_id=table, table=table, database=database)
        runtime = engine.create_record_runtime(MegatronRecordFormat(table))
        adapter = MegatronAdaptor(engine, runtime, table, dims={DimSpec.HIDDEN: 4})
        from dmi_megatron_integration.adapter import MegatronHookBinding
        adapter.attach_hooks(model_hooks=(MegatronHookBinding(
            hook=layer.dmi_moe_packed_weighted_output, record_dp_rank=-1, record_shard_rank=0),),
            iteration_hooks=())
        truth = {}
        for batch, counts in enumerate(([[2, 1, 3], [1, 4, 2]], [[0, 2, 1], [0, 3, 2]],
                                       [[0, 0, 0], [0, 0, 0]]), start=1):
            cpu = torch.tensor(counts, dtype=torch.int64)
            dispatcher.num_global_tokens_per_local_expert = cpu.T.contiguous()
            values = torch.arange(int(cpu.sum()) * 4, dtype=torch.float32, device="cuda").reshape(-1, 4)
            ctx = MegatronTrainingContext(model_id=table, global_batch_id=batch, microbatch_id=0, valid_counts=())
            adapter.set_current_event(ctx)
            try:
                layer.dmi_moe_packed_weighted_output(values, cpu.cuda())
            finally:
                adapter.clear_current_event()
            offsets = torch.cat((torch.zeros(1, dtype=torch.int64), cpu.flatten().cumsum(0)))
            chosen = policy.select(batch, 3)
            truth[batch] = torch.cat([values[offsets[b * 3 + u]:offsets[b * 3 + u + 1]].cpu()
                                      for b in range(2) for u in chosen])
        torch.cuda.synchronize()
        engine.flush_and_wait()
        _wait_for_exact_act_rows(client, database=database, table=table, model_id=table,
                                act_name=EP_OUTPUT, expected=3)
        from dmi_megatron_integration.records.reader import MegatronTrainingReader
        reader = MegatronTrainingReader(host=os.environ.get("DMX_DB_HOST", "localhost"),
            port=int(os.environ.get("DMX_DB_PORT", "9000")), database=database, table=table)
        try:
            rows = reader.training_raw_prefix_get((table, EP_OUTPUT, "fwd", "train"), return_full_key_tuple=True)
        finally:
            reader.close()
        for key, payload in rows:
            coords = dict(zip(TRAINING_ROW_COORDINATE_COLUMN_NAMES, key))
            torch.testing.assert_close(payload, truth[coords["global_batch_id"]], rtol=0, atol=0)
    finally:
        if engine is not None:
            engine.close()
        for (name,) in client.execute("SELECT name FROM system.tables WHERE database=%(db)s AND startsWith(name, %(prefix)s)",
                                       {"db": database, "prefix": table}):
            client.execute(f"DROP TABLE `{database}`.`{name}`")
        client.disconnect()


@pytest.mark.parametrize("shape", [(0, 4), (2, 0), (0,)])
def test_reader_decodes_empty_sampled_records(shape):
    from dmi_megatron_integration.records.reader import MegatronTrainingReader
    decoded = MegatronTrainingReader.torch_decode("torch.bfloat16", shape, b"")
    assert decoded.shape == shape and decoded.dtype == torch.bfloat16
    with pytest.raises(ValueError, match="zero-size"):
        MegatronTrainingReader.torch_decode("torch.bfloat16", (2, 4), b"")
