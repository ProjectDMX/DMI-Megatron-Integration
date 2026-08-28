from __future__ import annotations

import json
from dataclasses import replace
from typing import Any

import torch

from dmi.api.v1 import (
    HookOutput,
    HookPointV1,
    OutputStorage,
    ProducerPlanBuilder,
    RecordRuntime,
    RecordType,
    StepReservation,
    TransportSpec,
    TransportType,
)
from dmi_megatron_integration.hooks.specs import (
    DimSpec,
    MegatronHookSpec,
    MegatronMetadataField,
    MegatronOutputSpec,
)
from dmi_megatron_integration.records.format import (
    EVALUATION_BOUNDARY_CELL_TYPES,
    EVALUATION_BOUNDARY_LAYOUT_NAME,
    EVALUATION_BOUNDARY_NBYTES,
    MegatronRecordFormat,
    evaluation_boundary_row,
    required_record_metadata_fields,
)
from dmi_megatron_integration.records.metadata import MegatronRecordMetadata
from tests.support.megatron_file_sink import MegatronTestFileSink


class _ImmediateRecordRuntime:
    def __init__(
        self,
        record_runtime: RecordRuntime[MegatronRecordMetadata],
        metadata: MegatronRecordMetadata,
    ) -> None:
        self.record_runtime = record_runtime
        self.metadata = metadata

    def should_emit(self, hook: HookPointV1) -> bool:
        del hook
        return True

    def prepare_output(
        self,
        *,
        hook: HookPointV1,
        output_index: int,
        output_id: int,
        output_spec: TransportSpec,
        output: HookOutput,
    ) -> StepReservation:
        del hook, output_index
        entry = ProducerPlanBuilder().record_output(
            output_id=output_id,
            output_spec=output_spec,
            output=output,
        )
        return self.record_runtime.emit_output(
            entry,
            replace(self.metadata, act_name=output_spec.name),
            output,
        )


def _emit(
    sink: MegatronTestFileSink,
    spec: MegatronHookSpec,
    metadata: MegatronRecordMetadata,
    *inputs: Any,
    dims: dict[Any, int] | None = None,
) -> None:
    record_runtime = RecordRuntime(sink, MegatronRecordFormat("unused_file_oracle"))
    hook = HookPointV1(spec.resolve(dims))
    hook_runtime = _ImmediateRecordRuntime(record_runtime, metadata)
    record_runtime.bind_hook(hook, hook_runtime=hook_runtime)
    hook(*inputs)


def _rows(path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
    ]


def test_required_record_metadata_fields_are_output_specific():
    def fields(
        transport_type: TransportType,
        *,
        need_token_range: bool = False,
        dynamic_dataset_provenance: bool = False,
    ) -> frozenset[MegatronMetadataField]:
        return required_record_metadata_fields(
            record_type=RecordType.PER_SAMPLE,
            need_token_range=need_token_range,
            transport_type=transport_type,
            dynamic_dataset_provenance=dynamic_dataset_provenance,
        )

    assert fields(TransportType.IDENTITY) == frozenset()
    assert fields(
        TransportType.IDENTITY,
        need_token_range=True,
    ) == frozenset({MegatronMetadataField.VALID_COUNT})
    assert fields(TransportType.PREFIX_STRIP) == frozenset(
        {MegatronMetadataField.VALID_COUNT}
    )
    assert fields(TransportType.SEQ_PREFIX_PACK) == frozenset(
        {MegatronMetadataField.VALID_COUNT}
    )
    assert fields(TransportType.SEGMENTED_PACK) == frozenset(
        {MegatronMetadataField.VALID_COUNT}
    )
    assert fields(
        TransportType.IDENTITY,
        dynamic_dataset_provenance=True,
    ) == frozenset({MegatronMetadataField.DATASET_ID})


def test_megatron_file_sink_writes_training_rows(tmp_path):
    sink = MegatronTestFileSink(tmp_path, rank=0)
    payload = torch.tensor(
        [[0.1, 0.2, 0.7], [0.3, 0.3, 0.4]], dtype=torch.float32
    )
    _emit(
        sink,
        MegatronHookSpec(
            name="router_probs_mean",
            layer_no=2,
            outputs=(
                MegatronOutputSpec(
                    name="router_probs_mean",
                    input_shape=(2, 3),
                    dtype=torch.float32,
                ),
            ),
        ),
        MegatronRecordMetadata(
            model_id="model-a",
            act_name="router_probs_mean",
            direction="fwd",
            phase="train",
            global_batch_id=7,
            dp_rank=0,
            microbatch_id=1,
            layer_no=2,
            shard_rank=0,
            token_start=10,
            valid_counts=(4, 2),
            dataset_ids=(4, 9),
            attempt_id=2,
            invocation_id=3,
        ),
        payload,
    )
    sink.close()

    rows = _rows(tmp_path / "rank000" / "rows.jsonl")
    assert len(rows) == 2
    assert [row["sample_index"] for row in rows] == [0, 1]
    assert [row["dataset_id"] for row in rows] == [4, 9]
    assert [row["token_end"] for row in rows] == [14, 12]
    assert rows[0]["phase"] == "train"
    assert rows[0]["schema_version"] == 2
    assert rows[0]["attempt_id"] == 2
    assert rows[0]["invocation_id"] == 3
    assert rows[0]["shape"] == [3]
    assert rows[0]["dtype"] == "torch.float"
    for row, expected in zip(rows, payload):
        torch.testing.assert_close(
            torch.load(
                tmp_path / "rank000" / row["payload_file"],
                weights_only=True,
            ),
            expected,
        )


def test_megatron_file_sink_writes_prefix_strip_tensor_rows(tmp_path):
    sink = MegatronTestFileSink(tmp_path, rank=0)
    active_count = torch.tensor([2], dtype=torch.int64)
    payload = torch.tensor(
        [[0.1, 0.9], [0.3, 0.7], [99.0, 99.0]],
        dtype=torch.float32,
    )

    def with_active_count(tensor: torch.Tensor):
        return tensor, active_count

    _emit(
        sink,
        MegatronHookSpec(
            name="router_probs_mean",
            layer_no=2,
            outputs=(
                MegatronOutputSpec(
                    name="router_probs_mean",
                    input_shape=(3, 2),
                    dtype=torch.float32,
                    transport_type=TransportType.PREFIX_STRIP,
                ),
            ),
            preprocess=with_active_count,
        ),
        MegatronRecordMetadata(
            model_id="model-a",
            act_name="router_probs_mean",
            direction="fwd",
            phase="train",
            global_batch_id=7,
            dp_rank=0,
            microbatch_id=1,
            layer_no=2,
            shard_rank=0,
            token_start=10,
            valid_counts=(3, 1, 0),
            dataset_ids=(4, 9, 12),
        ),
        payload,
    )
    sink.close()

    rows = _rows(tmp_path / "rank000" / "rows.jsonl")
    assert [row["sample_index"] for row in rows] == [0, 1]
    assert [row["dataset_id"] for row in rows] == [4, 9]
    assert [row["token_end"] for row in rows] == [13, 11]
    for row, expected in zip(rows, payload[:2]):
        torch.testing.assert_close(
            torch.load(
                tmp_path / "rank000" / row["payload_file"],
                weights_only=True,
            ),
            expected,
        )


def test_megatron_file_sink_writes_seq_prefix_pack_rows(tmp_path):
    sink = MegatronTestFileSink(tmp_path, rank=0)
    source = torch.tensor(
        [
            [[0.0, 0.5], [10.0, 10.5]],
            [[1.0, 1.5], [11.0, 11.5]],
            [[2.0, 2.5], [12.0, 12.5]],
        ],
        dtype=torch.float32,
    )
    valid_count = torch.tensor([3, 1], dtype=torch.int64)
    prefix = torch.tensor([0, 3, 4], dtype=torch.int64)

    def with_counts(tensor: torch.Tensor):
        return tensor, valid_count, prefix

    _emit(
        sink,
        MegatronHookSpec(
            name="hidden_states",
            layer_no=-1,
            outputs=(
                MegatronOutputSpec(
                    name="hidden_states",
                    input_shape=(3, 2, 2),
                    output_shape=(DimSpec.ACTUAL_TOKEN_PACKED, 2),
                    dtype=torch.float32,
                    transport_type=TransportType.SEQ_PREFIX_PACK,
                ),
            ),
            preprocess=with_counts,
        ),
        MegatronRecordMetadata(
            model_id="model-a",
            act_name="hidden_states",
            direction="fwd",
            phase="valid",
            global_batch_id=3,
            dp_rank=0,
            microbatch_id=0,
            layer_no=-1,
            shard_rank=0,
            token_start=0,
            valid_counts=(3, 1),
            dataset_ids=(7, 8),
        ),
        source,
    )
    sink.close()

    rows = _rows(tmp_path / "rank000" / "rows.jsonl")
    assert [row["shape"] for row in rows] == [[3, 2], [1, 2]]
    assert [row["token_end"] for row in rows] == [3, 1]
    assert [row["dataset_id"] for row in rows] == [7, 8]
    torch.testing.assert_close(
        torch.load(tmp_path / "rank000" / rows[0]["payload_file"], weights_only=True),
        source[:, 0],
    )
    torch.testing.assert_close(
        torch.load(tmp_path / "rank000" / rows[1]["payload_file"], weights_only=True),
        source[:1, 1],
    )


def test_megatron_file_sink_writes_loss_summary_scalar_rows(tmp_path):
    sink = MegatronTestFileSink(tmp_path, rank=0)
    _emit(
        sink,
        MegatronHookSpec(
            name="lm_per_sample_loss",
            layer_no=-1,
            outputs=(
                MegatronOutputSpec(
                    name="lm_per_sample_loss",
                    input_shape=(2, 1),
                    dtype=torch.float32,
                    storage=OutputStorage.SCALAR_FLOAT,
                ),
                MegatronOutputSpec(
                    name="lm_per_sample_loss_token_count",
                    input_shape=(2, 1),
                    dtype=torch.int64,
                    storage=OutputStorage.SCALAR_INT,
                ),
            ),
            need_token_range=False,
        ),
        MegatronRecordMetadata(
            model_id="model-a",
            act_name="lm_per_sample_loss",
            direction="fwd",
            phase="valid",
            global_batch_id=9,
            dp_rank=0,
            microbatch_id=0,
            layer_no=-1,
            shard_rank=0,
            token_start=0,
            valid_counts=(3, 2),
            dataset_ids=(5, 6),
        ),
        torch.tensor([[1.25], [2.5]], dtype=torch.float32),
        torch.tensor([[3], [2]], dtype=torch.int64),
    )
    sink.close()

    rows = _rows(tmp_path / "rank000" / "scalar_float_rows.jsonl")
    assert [row["value"] for row in rows] == [1.25, 2.5]
    assert [row["dataset_id"] for row in rows] == [5, 6]
    assert [row["token_end"] for row in rows] == [1, 1]
    assert all(row["act_name"] == "lm_per_sample_loss" for row in rows)

    int_rows = _rows(tmp_path / "rank000" / "scalar_int_rows.jsonl")
    assert [row["value"] for row in int_rows] == [3, 2]
    assert [row["dataset_id"] for row in int_rows] == [5, 6]
    assert [row["token_end"] for row in int_rows] == [1, 1]
    assert all(
        row["act_name"] == "lm_per_sample_loss_token_count"
        for row in int_rows
    )


def test_megatron_file_sink_writes_packed_loss_summary_scalar_rows(tmp_path):
    sink = MegatronTestFileSink(tmp_path, rank=0)
    active_count = torch.tensor([2], dtype=torch.int64)

    def packed_loss_summary(_input: torch.Tensor):
        return [
            (
                torch.tensor([[1.25], [2.5], [99.0]], dtype=torch.float32),
                active_count,
            ),
            (
                torch.tensor([[4], [2], [99]], dtype=torch.int64),
                active_count,
            ),
        ]

    _emit(
        sink,
        MegatronHookSpec(
            name="lm_per_sample_loss",
            layer_no=-1,
            outputs=(
                MegatronOutputSpec(
                    name="lm_per_sample_loss",
                    input_shape=(3, 1),
                    dtype=torch.float32,
                    transport_type=TransportType.PREFIX_STRIP,
                    storage=OutputStorage.SCALAR_FLOAT,
                ),
                MegatronOutputSpec(
                    name="lm_per_sample_loss_token_count",
                    input_shape=(3, 1),
                    dtype=torch.int64,
                    transport_type=TransportType.PREFIX_STRIP,
                    storage=OutputStorage.SCALAR_INT,
                ),
            ),
            preprocess=packed_loss_summary,
            need_token_range=False,
        ),
        MegatronRecordMetadata(
            model_id="model-a",
            act_name="lm_per_sample_loss",
            direction="fwd",
            phase="train",
            global_batch_id=9,
            dp_rank=0,
            microbatch_id=0,
            layer_no=-1,
            shard_rank=0,
            token_start=0,
            valid_counts=(4, 2, 0),
            dataset_ids=(5, 6, 7),
        ),
        torch.empty((), dtype=torch.float32),
    )
    sink.close()

    float_rows = _rows(tmp_path / "rank000" / "scalar_float_rows.jsonl")
    assert [row["sample_index"] for row in float_rows] == [0, 1]
    assert [row["dataset_id"] for row in float_rows] == [5, 6]
    assert [row["token_end"] for row in float_rows] == [1, 1]
    assert [row["value"] for row in float_rows] == [1.25, 2.5]

    int_rows = _rows(tmp_path / "rank000" / "scalar_int_rows.jsonl")
    assert [row["sample_index"] for row in int_rows] == [0, 1]
    assert [row["dataset_id"] for row in int_rows] == [5, 6]
    assert [row["token_end"] for row in int_rows] == [1, 1]
    assert [row["value"] for row in int_rows] == [4, 2]


def test_megatron_file_sink_submits_iteration_scalar_float(tmp_path):
    sink = MegatronTestFileSink(tmp_path, rank=0)
    _emit(
        sink,
        MegatronHookSpec(
            name="grad_norm",
            layer_no=-1,
            outputs=(
                MegatronOutputSpec(
                    name="grad_norm",
                    input_shape=(1,),
                    dtype=torch.float32,
                    storage=OutputStorage.SCALAR_FLOAT,
                ),
            ),
            need_token_range=False,
            record_type=RecordType.PER_ITERATION,
        ),
        MegatronRecordMetadata(
            model_id="model-a",
            act_name="grad_norm",
            direction="iter",
            phase="train",
            global_batch_id=7,
            dp_rank=-1,
            microbatch_id=-1,
            layer_no=-1,
            shard_rank=-1,
            token_start=0,
            attempt_id=4,
        ),
        torch.tensor([3.5], dtype=torch.float32),
    )
    sink.close()

    rows = _rows(tmp_path / "rank000" / "scalar_float_rows.jsonl")
    assert len(rows) == 1
    assert rows[0]["value"] == 3.5
    assert rows[0]["attempt_id"] == 4
    assert rows[0]["sample_index"] == -1
    assert rows[0]["dataset_id"] == -1
    assert rows[0]["token_start"] == 0
    assert rows[0]["token_end"] == 1


def test_megatron_file_sink_writes_unsplit_execution_tensor(tmp_path):
    sink = MegatronTestFileSink(tmp_path, rank=0)
    payload = torch.tensor([2, 0, 3, 1], dtype=torch.int64)
    _emit(
        sink,
        MegatronHookSpec(
            name="moe_inverse_map",
            layer_no=2,
            outputs=(
                MegatronOutputSpec(
                    name="moe_inverse_map",
                    input_shape=(DimSpec.ACTUAL_TOKEN_PACKED,),
                    output_shape=(DimSpec.ACTUAL_TOKEN_PACKED,),
                    dtype=torch.int64,
                ),
            ),
            need_token_range=False,
            record_type=RecordType.PER_EXECUTION,
        ),
        MegatronRecordMetadata(
            model_id="model-a",
            act_name="moe_inverse_map",
            direction="fwd",
            phase="train",
            global_batch_id=7,
            dp_rank=1,
            microbatch_id=3,
            layer_no=2,
            shard_rank=4,
            token_start=-1,
            attempt_id=5,
            invocation_id=6,
        ),
        payload,
    )
    sink.close()

    rows = _rows(tmp_path / "rank000" / "rows.jsonl")
    assert len(rows) == 1
    row = rows[0]
    assert row["act_name"] == "moe_inverse_map"
    assert row["dp_rank"] == 1
    assert row["microbatch_id"] == 3
    assert row["layer_no"] == 2
    assert row["shard_rank"] == 4
    assert row["attempt_id"] == 5
    assert row["invocation_id"] == 6
    assert row["sample_index"] == -1
    assert row["dataset_id"] == -1
    assert row["token_start"] == -1
    assert row["token_end"] == -1
    assert row["shape"] == [4]
    torch.testing.assert_close(
        torch.load(tmp_path / "rank000" / row["payload_file"], weights_only=True),
        payload,
    )


def test_megatron_file_sink_writes_eval_boundary_rows(tmp_path):
    sink = MegatronTestFileSink(tmp_path, rank=0)
    for values in (
        evaluation_boundary_row(
            model_id="model-a",
            training_iteration_id=2,
            phase="valid",
            eval_index=0,
            boundary_type="entry",
            next_global_batch_id=1,
        ),
        evaluation_boundary_row(
            model_id="model-a",
            training_iteration_id=2,
            phase="valid",
            eval_index=0,
            boundary_type="exit",
            next_global_batch_id=2,
        ),
        evaluation_boundary_row(
            model_id="model-a",
            training_iteration_id=3,
            phase="test",
            eval_index=0,
            boundary_type="entry",
            next_global_batch_id=1,
        ),
    ):
        sink.submit_record(
            EVALUATION_BOUNDARY_LAYOUT_NAME,
            values,
            EVALUATION_BOUNDARY_CELL_TYPES,
            nbytes=EVALUATION_BOUNDARY_NBYTES,
        )
    sink.close()

    assert _rows(tmp_path / "rank000" / "eval_phase_boundary.jsonl") == [
        {
            "model_id": "model-a",
            "training_iteration_id": 2,
            "phase": "valid",
            "eval_index": 0,
            "boundary_type": "entry",
            "next_global_batch_id": 1,
        },
        {
            "model_id": "model-a",
            "training_iteration_id": 2,
            "phase": "valid",
            "eval_index": 0,
            "boundary_type": "exit",
            "next_global_batch_id": 2,
        },
        {
            "model_id": "model-a",
            "training_iteration_id": 3,
            "phase": "test",
            "eval_index": 0,
            "boundary_type": "entry",
            "next_global_batch_id": 1,
        },
    ]


def test_megatron_file_sink_rejects_invalid_eval_boundary(tmp_path):
    sink = MegatronTestFileSink(tmp_path, rank=0)
    try:
        sink.submit_record(
            EVALUATION_BOUNDARY_LAYOUT_NAME,
            ("model-a", 1, "train", 0, "entry", 1),
            EVALUATION_BOUNDARY_CELL_TYPES,
            nbytes=EVALUATION_BOUNDARY_NBYTES,
        )
    except ValueError as exc:
        assert "phase must be valid or test" in str(exc)
    else:
        raise AssertionError("MegatronTestFileSink should reject train eval boundary")
