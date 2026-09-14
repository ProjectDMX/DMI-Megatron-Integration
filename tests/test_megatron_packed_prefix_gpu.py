"""Rank-local TP=2 packing qualification without Megatron's batch loader."""

import os
import uuid

import pytest
import torch

from dmi.api.v1 import StepReservation, TransportType
from dmi_megatron_integration.adapter import MegatronAdaptor, MegatronHookBinding, MegatronTrainingContext
from dmi_megatron_integration.hooks.specs import DimSpec, HookPhase, MegatronHookSpec, MegatronOutputSpec, ShardPolicy
from dmi_megatron_integration.metadata_context import DMIMetadataContext
from dmi_megatron_integration.records.format import MegatronRecordFormat
from tests.test_megatron_adapter import _make_hook, _tp_sequence_distributed_info
from tests.test_megatron_e2e_clickhouse import _build_training_engine, _clickhouse_client_or_skip
from tests.test_megatron_real_training_e2e import _read_training_act_rows, _wait_for_exact_act_rows


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Two CUDA devices required")
@pytest.mark.parametrize("tp_rank", [0, 1])
@pytest.mark.parametrize("graph_mode", [False, True], ids=["eager", "local_graph"])
def test_tp2_rank_local_packed_payloads_with_changing_counts(tp_rank, graph_mode, monkeypatch):
    client = _clickhouse_client_or_skip()
    database = os.environ.get("DMX_DB_DATABASE", "default")
    table = "dmi_prefix_gpu_" + uuid.uuid4().hex
    model_id = table
    engine = None
    with torch.cuda.device(tp_rank):
        try:
            engine = _build_training_engine(model_id=model_id, table=table, database=database)
            runtime = engine.create_record_runtime(MegatronRecordFormat(table))
            info = _tp_sequence_distributed_info(tp_rank=tp_rank, sequence_parallel_enabled=True)
            metadata = DMIMetadataContext(
                max_num_microbatches=1, max_batch_size=2, num_scopes=1,
                dims={DimSpec.SEQ: 8}, device=torch.device("cuda", tp_rank),
                megatron_distributed_info=info, tp_sequence_sharded_enabled=True,
            )
            hook = _make_hook(MegatronHookSpec(
                name="prefix_probe", layer_no=0, shard_policy=ShardPolicy.TP_SEQUENCE_SHARDED,
                outputs=(MegatronOutputSpec(
                    name="prefix_probe", input_shape=(DimSpec.SEQ, DimSpec.BATCH, 4),
                    output_shape=(DimSpec.ACTUAL_TOKEN_PACKED, 4), dtype=torch.float32,
                    transport_type=TransportType.SEQ_PREFIX_PACK,
                ),),
            ))
            hook.megatron_distributed_info = info
            adaptor = MegatronAdaptor(engine, runtime, model_id, dims={DimSpec.SEQ: 8, DimSpec.BATCH: 2})
            adaptor.attach_hooks(
                model_hooks=(MegatronHookBinding(hook=hook, record_shard_rank=tp_rank),),
                iteration_hooks=(), metadata_context=metadata,
            )
            reservations = []
            original_reserve = runtime._transport.reserve_record

            def record_reserve(items):
                reservations.append(tuple(items))
                return original_reserve(items)

            monkeypatch.setattr(runtime._transport, "reserve_record", record_reserve)
            global_input = torch.arange(8 * 2 * 4, dtype=torch.float32).reshape(8, 2, 4)
            local_input = global_input[tp_rank * 4:(tp_rank + 1) * 4].to("cuda")
            expected = {}
            graph = None
            plan = None
            for iteration, counts in enumerate(((3, 2), (1, 4), (7, 2), (0, 0), (8, 8))):
                metadata.begin_iteration(1)
                metadata.ingest_microbatch(0, {"valid_count": counts})
                metadata.enter_scope("fwd", 0, 0)
                packing = metadata.prepared_packing(0)
                ctx = MegatronTrainingContext(
                    global_batch_id=iteration, microbatch_id=0, valid_counts=counts,
                    dataset_ids=(10 + iteration, 20 + iteration), shard_rank=tp_rank,
                    _packing=packing,
                )
                adaptor.set_current_event(ctx)
                if graph_mode and iteration == 1:
                    adaptor.begin_capture_plan(warmup_enabled=True, capture_direction=HookPhase.FWD)
                    graph = torch.cuda.CUDAGraph()
                    # torch.cuda.graph's default stream is process-wide; this
                    # parametrized test captures on two different devices.
                    capture_stream = torch.cuda.Stream(device=tp_rank)
                    capture_stream.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.graph(graph, stream=capture_stream):
                        assert torch.cuda.current_device() == tp_rank
                        metadata.enter_scope("fwd", 0, 0)
                        hook(local_input)
                    torch.cuda.current_stream().wait_stream(capture_stream)
                    plan = adaptor.finish_capture_plan()
                if graph is None:
                    hook(local_input)
                else:
                    assert adaptor.prepare_replay(plan, plan_direction=HookPhase.FWD) is StepReservation.RESERVED
                    graph.replay()
                local_counts = packing.counts_for(tp_rank * 4, 4).counts
                assert reservations[-1] == ((sum(local_counts) * 16, True),)
                for sample, count in enumerate(local_counts):
                    if count:
                        key = (iteration, sample)
                        expected[key] = (global_input[tp_rank * 4:tp_rank * 4 + count, sample], ctx.dataset_ids[sample])
                torch.cuda.synchronize()
                engine.flush_and_wait()
                adaptor.clear_current_event()
                metadata.end_iteration()
            assert len(reservations) == 5
            _wait_for_exact_act_rows(client, database=database, table=table, model_id=model_id,
                                    act_name="prefix_probe", expected=len(expected))
            rows = _read_training_act_rows(model_id=model_id, table=table, database=database,
                                           act_name="prefix_probe", direction="fwd", raw=True)
            assert len(rows) == len(expected)
            for key, tensor in rows:
                reference, dataset_id = expected[key[4], key[7]]
                assert key[9] == tp_rank
                assert key[10:12] == (tp_rank * 4, tp_rank * 4 + reference.shape[0])
                assert key[14] == dataset_id
                assert torch.equal(tensor, reference)
        finally:
            if engine is not None:
                engine.close()
            for suffix in ("", "_scalar_float", "_scalar_int", "_eval_phase_boundary"):
                client.execute(f"DROP TABLE IF EXISTS `{database}`.`{table}{suffix}`")
            client.disconnect()
