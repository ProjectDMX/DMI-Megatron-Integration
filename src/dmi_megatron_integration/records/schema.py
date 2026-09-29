"""Schema-v3 ClickHouse layouts owned by the Megatron integration."""

from __future__ import annotations

from dmi.api.v1 import (
    RecordCellType,
    RecordColumn,
    RecordLayout,
    RecordSchema,
)


TRAINING_SCHEMA_VERSION = 3
TRAINING_PRIMARY_KEY_COLUMN_NAMES = (
    "model_id",
    "act_name",
    "direction",
    "phase",
    "global_batch_id",
    "dp_rank",
    "microbatch_id",
    "sample_index",
    "layer_no",
    "shard_rank",
    "token_start",
    "token_end",
    "attempt_id",
    "invocation_id",
)
TRAINING_PROVENANCE_COLUMN_NAMES = ("dataset_id",)
TRAINING_ROW_COORDINATE_COLUMN_NAMES = (
    TRAINING_PRIMARY_KEY_COLUMN_NAMES + TRAINING_PROVENANCE_COLUMN_NAMES
)

TENSOR_LAYOUT_NAME = "tensor"
SCALAR_FLOAT_LAYOUT_NAME = "scalar_float"
SCALAR_INT_LAYOUT_NAME = "scalar_int"
EVALUATION_BOUNDARY_LAYOUT_NAME = "eval_phase_boundary"


def _coordinate_columns() -> tuple[RecordColumn, ...]:
    return (
        RecordColumn("model_id", RecordCellType.STRING),
        RecordColumn("act_name", RecordCellType.STRING),
        RecordColumn("direction", RecordCellType.STRING),
        RecordColumn("phase", RecordCellType.STRING),
        RecordColumn("global_batch_id", RecordCellType.INT64),
        RecordColumn("dp_rank", RecordCellType.INT32),
        RecordColumn("microbatch_id", RecordCellType.INT32),
        RecordColumn("sample_index", RecordCellType.INT32),
        RecordColumn("layer_no", RecordCellType.INT32),
        RecordColumn("shard_rank", RecordCellType.INT32),
        RecordColumn("token_start", RecordCellType.INT64),
        RecordColumn("token_end", RecordCellType.INT64),
        RecordColumn("attempt_id", RecordCellType.INT32),
        RecordColumn("invocation_id", RecordCellType.INT32),
        RecordColumn("dataset_id", RecordCellType.INT32),
        RecordColumn("producer_rank", RecordCellType.INT32),
    )


def build_training_schema(
    base_table: str,
    *,
    index_granularity: int = 8192,
) -> RecordSchema:
    """Build the Megatron schema-v3 payload and lifecycle tables for one base name."""

    coordinates = _coordinate_columns()
    tensor_layout = RecordLayout(
        name=TENSOR_LAYOUT_NAME,
        table=base_table,
        columns=coordinates
        + (
            RecordColumn(
                "value",
                RecordCellType.TENSOR,
                dtype_column="dtype",
                shape_column="shape",
                bytes_column="bytes",
            ),
        ),
        primary_key=TRAINING_PRIMARY_KEY_COLUMN_NAMES,
        order_by=TRAINING_PRIMARY_KEY_COLUMN_NAMES,
    )
    scalar_float_layout = RecordLayout(
        name=SCALAR_FLOAT_LAYOUT_NAME,
        table=f"{base_table}_scalar_float",
        columns=coordinates + (RecordColumn("value", RecordCellType.FLOAT64),),
        primary_key=TRAINING_PRIMARY_KEY_COLUMN_NAMES,
        order_by=TRAINING_PRIMARY_KEY_COLUMN_NAMES,
    )
    scalar_int_layout = RecordLayout(
        name=SCALAR_INT_LAYOUT_NAME,
        table=f"{base_table}_scalar_int",
        columns=coordinates + (RecordColumn("value", RecordCellType.INT64),),
        primary_key=TRAINING_PRIMARY_KEY_COLUMN_NAMES,
        order_by=TRAINING_PRIMARY_KEY_COLUMN_NAMES,
    )
    boundary_key = (
        "model_id",
        "training_iteration_id",
        "phase",
        "eval_index",
        "boundary_type",
        "next_global_batch_id",
    )
    boundary_layout = RecordLayout(
        name=EVALUATION_BOUNDARY_LAYOUT_NAME,
        table=f"{base_table}_eval_phase_boundary",
        columns=(
            RecordColumn("model_id", RecordCellType.STRING),
            RecordColumn("training_iteration_id", RecordCellType.INT64),
            RecordColumn("phase", RecordCellType.STRING),
            RecordColumn("eval_index", RecordCellType.INT32),
            RecordColumn("boundary_type", RecordCellType.STRING),
            RecordColumn("next_global_batch_id", RecordCellType.INT64),
        ),
        primary_key=boundary_key,
        order_by=boundary_key,
    )
    iteration_columns = (
        ("model_id", RecordCellType.STRING), ("phase", RecordCellType.STRING),
        ("global_batch_id", RecordCellType.INT64), ("training_iteration_id", RecordCellType.INT64),
        ("eval_index", RecordCellType.INT32), ("producer_rank", RecordCellType.INT32),
        ("attempt_id", RecordCellType.INT32), ("status", RecordCellType.INT32),
        ("weights_updated", RecordCellType.INT32), ("expected_tensor_count", RecordCellType.INT64),
    )
    phase_columns = (
        ("model_id", RecordCellType.STRING), ("phase", RecordCellType.STRING),
        ("training_iteration_id", RecordCellType.INT64), ("eval_index", RecordCellType.INT32),
        ("producer_rank", RecordCellType.INT32), ("boundary_type", RecordCellType.STRING),
        ("global_batch_id_start", RecordCellType.INT64), ("global_batch_id_end", RecordCellType.INT64),
        ("expected_tensor_count", RecordCellType.INT64),
    )
    metadata_layouts = tuple(
        RecordLayout(name=name, table=f"{base_table}_{name}",
                     columns=tuple(RecordColumn(n, t) for n, t in columns),
                     primary_key=("model_id", "phase", "producer_rank"),
                     order_by=("model_id", "phase", "producer_rank"))
        for name, columns in (("iteration_metadata", iteration_columns), ("phase_metadata", phase_columns))
    )
    topology_layout = RecordLayout(
        name="topology_manifest", table=f"{base_table}_topology_manifest",
        columns=(RecordColumn("model_id", RecordCellType.STRING), RecordColumn("manifest_json", RecordCellType.STRING)),
        primary_key=("model_id",), order_by=("model_id",),
    )
    capture_topology_layout = RecordLayout(
        name="capture_topology", table=f"{base_table}_capture_topology",
        columns=(RecordColumn("model_id", RecordCellType.STRING),) + tuple(
            RecordColumn(name, RecordCellType.INT32) for name in (
                "producer_rank", "tp_rank", "tp_world_size", "pp_rank", "dp_rank", "cp_rank"
            )
        ) + (RecordColumn("vocab_partition_size", RecordCellType.INT64),
             RecordColumn("weight_layout_json", RecordCellType.STRING)),
        primary_key=("model_id", "producer_rank"), order_by=("model_id", "producer_rank"),
    )
    return RecordSchema(
        layouts=(
            tensor_layout,
            scalar_float_layout,
            scalar_int_layout,
            boundary_layout,
        ) + metadata_layouts + (topology_layout, capture_topology_layout),
        index_granularity=index_granularity,
    )


__all__ = [
    "EVALUATION_BOUNDARY_LAYOUT_NAME",
    "SCALAR_FLOAT_LAYOUT_NAME",
    "SCALAR_INT_LAYOUT_NAME",
    "TENSOR_LAYOUT_NAME",
    "TRAINING_PRIMARY_KEY_COLUMN_NAMES",
    "TRAINING_PROVENANCE_COLUMN_NAMES",
    "TRAINING_ROW_COORDINATE_COLUMN_NAMES",
    "TRAINING_SCHEMA_VERSION",
    "build_training_schema",
]
