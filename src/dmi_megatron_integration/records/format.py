"""Encode Megatron coordinates into public DMI record descriptors."""

from __future__ import annotations

from math import prod

import torch

from dmi.api.v1 import (
    OutputStorage,
    PayloadSlice,
    ProducerPlanEntry,
    RecordDescriptor,
    RecordSchema,
    RecordType,
    TransportType,
)

from ..hooks.specs import MegatronMetadataField
from .metadata import MegatronRecordMetadata
from .schema import (
    EVALUATION_BOUNDARY_LAYOUT_NAME,
    SCALAR_FLOAT_LAYOUT_NAME,
    SCALAR_INT_LAYOUT_NAME,
    TENSOR_LAYOUT_NAME,
    build_training_schema,
)


EVALUATION_BOUNDARY_CELL_TYPES = (
    "string",
    "int64",
    "string",
    "int32",
    "string",
    "int64",
)
EVALUATION_BOUNDARY_NBYTES = 8


def required_record_metadata_fields(
    *,
    record_type: RecordType,
    need_token_range: bool,
    transport_type: TransportType,
    dynamic_dataset_provenance: bool,
) -> frozenset[MegatronMetadataField]:
    fields: set[MegatronMetadataField] = set()
    if record_type is RecordType.PER_SAMPLE:
        if need_token_range or transport_type in (
            TransportType.PREFIX_STRIP,
            TransportType.SEQ_PREFIX_PACK,
            TransportType.SEGMENTED_PACK,
        ):
            fields.add(MegatronMetadataField.VALID_COUNT)
        if dynamic_dataset_provenance:
            fields.add(MegatronMetadataField.DATASET_ID)
    return frozenset(fields)


def evaluation_boundary_row(
    *,
    model_id: str,
    training_iteration_id: int,
    phase: str,
    eval_index: int,
    boundary_type: str,
    next_global_batch_id: int,
) -> tuple[object, ...]:
    """Return one validated row for the established evaluation-boundary table."""

    if phase not in {"valid", "test"}:
        raise ValueError("evaluation boundary phase must be 'valid' or 'test'")
    if boundary_type not in {"entry", "exit"}:
        raise ValueError("evaluation boundary type must be 'entry' or 'exit'")
    training_iteration_id = int(training_iteration_id)
    eval_index = int(eval_index)
    next_global_batch_id = int(next_global_batch_id)
    if training_iteration_id < 1 or eval_index < 0 or next_global_batch_id < 1:
        raise ValueError(
            "evaluation boundary IDs must be positive and eval_index non-negative"
        )
    if eval_index >= 1 << 31:
        raise ValueError("evaluation boundary eval_index is outside Int32 range")
    return (
        str(model_id),
        training_iteration_id,
        str(phase),
        eval_index,
        str(boundary_type),
        next_global_batch_id,
    )


class MegatronRecordFormat:
    """Schema-v3 descriptor encoder for Megatron's GPU record path."""

    def __init__(self, base_table: str, *, index_granularity: int = 8192, producer_rank: int = 0) -> None:
        self.producer_rank = int(producer_rank)
        self._expected_records = {}
        self.count_records = False
        self._schema = build_training_schema(
            base_table,
            index_granularity=index_granularity,
        )

    @property
    def schema(self) -> RecordSchema:
        return self._schema

    def encode(
        self,
        metadata: MegatronRecordMetadata,
        entry: ProducerPlanEntry,
    ) -> RecordDescriptor:
        if not isinstance(metadata, MegatronRecordMetadata):
            raise TypeError("metadata must be MegatronRecordMetadata")
        if not isinstance(entry, ProducerPlanEntry):
            raise TypeError("entry must be ProducerPlanEntry")
        rows = self._rows(metadata, entry)
        if self.count_records and metadata.attempt_id >= 0 and metadata.act_name != "iteration_attempt_status":
            key = (metadata.phase, metadata.global_batch_id, metadata.attempt_id)
            self._expected_records[key] = self._expected_records.get(key, 0) + len(rows)
        return RecordDescriptor(
            layout=self._layout_name(entry.storage),
            rows=rows,
            output_id=entry.output_id,
        )

    def _bind_native_encoder(self, transport):
        # Subclasses/custom encoders retain their public encode() implementation.
        compile_layout = getattr(transport, "_compile_record_layout", None)
        if type(self) is not MegatronRecordFormat or "encode" in self.__dict__ or compile_layout is None:
            return None
        layouts = {storage: compile_layout(self._layout_name(storage), (7, 10, 11, 14), 16)
                   for storage in OutputStorage}

        def encode_native(metadata, entry):
            if not isinstance(metadata, MegatronRecordMetadata):
                raise TypeError("metadata must be MegatronRecordMetadata")
            if not isinstance(entry, ProducerPlanEntry):
                raise TypeError("entry must be ProducerPlanEntry")
            rows = self._row_specs(metadata, entry)
            common = self._common_coordinates(metadata) if rows else (None,) * 12
            if self.count_records and metadata.attempt_id >= 0 and metadata.act_name != "iteration_attempt_status":
                key = (metadata.phase, metadata.global_batch_id, metadata.attempt_id)
                self._expected_records[key] = self._expected_records.get(key, 0) + len(rows)
            return (layouts[entry.storage], common, rows, entry.output_id)

        return encode_native

    def take_expected_count(self, phase: str, batch: int, attempt: int) -> int:
        return self._expected_records.pop((phase, batch, attempt), 0)

    def _rows(self, metadata, entry):
        # The reference/custom path uses the same splitting arithmetic, but
        # materializes the public Python descriptor objects as before.
        return tuple(
            self._coordinates(metadata, sample_index=coords[0], token_start=coords[1],
                              token_end=coords[2], dataset_id=coords[3])
            + (PayloadSlice(offset_bytes=offset, nbytes=nbytes, storage=entry.storage,
                            dtype=dtype, shape=shape),)
            for coords, offset, nbytes, dtype, shape in self._row_specs(metadata, entry)
        )

    def _row_specs(self, metadata, entry):
        if entry.record_type is RecordType.PER_SAMPLE:
            return self._per_sample_specs(metadata, entry)
        if entry.record_type not in (RecordType.PER_ITERATION, RecordType.PER_EXECUTION):
            raise ValueError(f"unsupported record type: {entry.record_type!r}")
        if entry.transport_type is not TransportType.IDENTITY and not (
            entry.record_type is RecordType.PER_EXECUTION
            and entry.transport_type is TransportType.SEGMENTED_PACK
        ):
            raise ValueError("unsplit Megatron records require IDENTITY transport")
        if metadata.valid_counts:
            raise ValueError("unsplit Megatron records must not carry valid_counts")
        if metadata.dataset_ids:
            raise ValueError("unsplit Megatron records must not carry dataset_ids")
        if entry.storage in (OutputStorage.SCALAR_FLOAT, OutputStorage.SCALAR_INT):
            if not entry.output_shape:
                raise ValueError("unsplit Megatron scalar output must be at least 1-D")
            self._require_scalar_element_shape(entry.output_shape)
        per_execution = entry.record_type is RecordType.PER_EXECUTION
        start = -1 if per_execution else int(metadata.token_start)
        end = -1 if per_execution else int(metadata.token_start) + 1
        payload = self._payload_values(entry, offset_bytes=0, nbytes=self._entry_bytes(entry),
                                       shape=entry.output_shape)
        return (((-1, start, end, -1), *payload),)

    def _per_sample_rows(self, metadata, entry):
        return self._rows(metadata, entry)

    def _per_sample_specs(self, metadata, entry):
        counts = self._record_counts(metadata, entry)
        self._validate_per_sample_shape(entry, counts)
        if metadata.dataset_ids and len(metadata.dataset_ids) != len(counts):
            raise ValueError("dataset_ids length must match the record count")
        datasets = metadata.dataset_ids or (0,) * len(counts)
        rows = []
        packed_offset = 0
        active_index = 0
        start = int(metadata.token_start)
        for sample_index, (valid_count, dataset_id) in enumerate(zip(counts, datasets)):
            if valid_count <= 0:
                continue
            dataset_id = int(dataset_id)
            if not -1 <= dataset_id < 1 << 31:
                raise ValueError("dataset_id is outside the supported Int32 range")
            payload, packed_offset, active_index = self._sample_payload_values(
                entry, sample_index=sample_index, valid_count=valid_count,
                packed_offset=packed_offset, active_index=active_index,
            )
            rows.append(((sample_index, start, start + valid_count, dataset_id), *payload))
        return tuple(rows)

    @staticmethod
    def _validate_per_sample_shape(
        entry: ProducerPlanEntry,
        counts: tuple[int, ...],
    ) -> None:
        if entry.storage in (OutputStorage.SCALAR_FLOAT, OutputStorage.SCALAR_INT):
            return
        if not entry.output_shape:
            raise ValueError("per-sample tensor output requires at least one dimension")
        if entry.transport_type is TransportType.IDENTITY:
            expected = len(counts)
        elif entry.transport_type is TransportType.PREFIX_STRIP:
            expected = len(counts)
        elif entry.transport_type in (
            TransportType.SEQ_PREFIX_PACK,
            TransportType.SEGMENTED_PACK,
        ):
            expected = sum(max(0, count) for count in counts)
        else:
            raise ValueError(
                "unsupported per-sample Megatron transport: "
                f"{entry.transport_type.value}"
            )
        actual = int(entry.output_shape[0])
        if actual >= 0 and actual != expected:
            raise ValueError(
                "per-sample Megatron payload row mismatch: "
                f"transport={entry.transport_type.value}, "
                f"output_rows={actual}, expected={expected}"
            )

    @staticmethod
    def _record_counts(
        metadata: MegatronRecordMetadata,
        entry: ProducerPlanEntry,
    ) -> tuple[int, ...]:
        if metadata.valid_counts:
            return metadata.valid_counts
        if entry.transport_type in (
            TransportType.PREFIX_STRIP,
            TransportType.SEQ_PREFIX_PACK,
            TransportType.SEGMENTED_PACK,
        ):
            raise ValueError("packed Megatron records require valid_counts")
        if not entry.output_shape:
            return (1,)
        return (1,) * max(0, int(entry.output_shape[0]))

    def _sample_payload_values(
        self,
        entry: ProducerPlanEntry,
        *,
        sample_index: int,
        valid_count: int,
        packed_offset: int,
        active_index: int,
    ) -> tuple[tuple, int, int]:
        element_size = entry.element_size
        if entry.storage in (OutputStorage.SCALAR_FLOAT, OutputStorage.SCALAR_INT):
            self._validate_scalar_dtype(entry.storage, entry.dtype)
            if not entry.output_shape:
                raise ValueError("per-sample scalar output requires a sample dimension")
            self._require_scalar_element_shape(entry.output_shape[1:])
            if entry.transport_type is TransportType.IDENTITY:
                offset_bytes = sample_index * element_size
                next_active_index = active_index
            elif entry.transport_type is TransportType.PREFIX_STRIP:
                offset_bytes = active_index * element_size
                next_active_index = active_index + 1
            else:
                raise ValueError(
                    "Megatron scalar records require IDENTITY or PREFIX_STRIP "
                    "transport"
                )
            return (
                self._payload_values(
                    entry,
                    offset_bytes=offset_bytes,
                    nbytes=element_size,
                    shape=(),
                ),
                packed_offset,
                next_active_index,
            )

        if entry.transport_type is TransportType.IDENTITY:
            if not entry.output_shape or entry.output_shape[0] < 0:
                raise ValueError(
                    "per-sample IDENTITY output requires a fixed first dimension"
                )
            row_shape = entry.output_shape[1:]
            row_bytes = int(prod(row_shape)) * element_size
            return (
                self._payload_values(
                    entry,
                    offset_bytes=sample_index * row_bytes,
                    nbytes=row_bytes,
                    shape=row_shape,
                ),
                packed_offset,
                active_index,
            )

        if entry.transport_type is TransportType.PREFIX_STRIP:
            row_bytes = int(entry.transport_args[0])
            row_shape = entry.output_shape[1:]
            value = self._payload_values(
                entry,
                offset_bytes=active_index * row_bytes,
                nbytes=row_bytes,
                shape=row_shape,
            )
            return value, packed_offset, active_index + 1

        if entry.transport_type in (
            TransportType.SEQ_PREFIX_PACK,
            TransportType.SEGMENTED_PACK,
        ):
            feature_bytes = int(entry.transport_args[0])
            nbytes = valid_count * feature_bytes
            value = self._payload_values(
                entry,
                offset_bytes=packed_offset,
                nbytes=nbytes,
                shape=(valid_count, *entry.output_shape[1:]),
            )
            return value, packed_offset + nbytes, active_index

        raise ValueError(
            f"unsupported per-sample Megatron transport: {entry.transport_type.value}"
        )

    def _sample_payload_slice(self, entry, **kwargs):
        values, packed_offset, active_index = self._sample_payload_values(entry, **kwargs)
        offset, nbytes, dtype, shape = values
        return (PayloadSlice(offset_bytes=offset, nbytes=nbytes, storage=entry.storage,
                             dtype=dtype, shape=shape), packed_offset, active_index)

    @classmethod
    def _payload_values(cls, entry, *, offset_bytes, nbytes, shape):
        cls._validate_scalar_dtype(entry.storage, entry.dtype)
        offset_bytes = int(offset_bytes)
        nbytes = None if nbytes is None else int(nbytes)
        shape = tuple(int(dim) for dim in shape) if entry.storage is OutputStorage.TENSOR else ()
        if offset_bytes < 0:
            raise ValueError("PayloadSlice.offset_bytes must be non-negative")
        if nbytes is not None and nbytes < 0:
            raise ValueError("PayloadSlice.nbytes must be non-negative")
        if sum(dim == -1 for dim in shape) > 1 or any(dim < -1 for dim in shape):
            raise ValueError("PayloadSlice.shape supports at most one -1")
        if entry.dtype is None:
            raise ValueError("PayloadSlice requires dtype")
        return (offset_bytes, nbytes, entry.dtype, shape)

    @classmethod
    def _payload_slice(cls, entry, **kwargs):
        offset, nbytes, dtype, shape = cls._payload_values(entry, **kwargs)
        return PayloadSlice(offset_bytes=offset, nbytes=nbytes, storage=entry.storage,
                            dtype=dtype, shape=shape)

    @staticmethod
    def _validate_scalar_dtype(storage: OutputStorage, dtype: torch.dtype) -> None:
        if storage is OutputStorage.SCALAR_FLOAT and not dtype.is_floating_point:
            raise ValueError("scalar-float Megatron output requires a floating dtype")
        if storage is OutputStorage.SCALAR_INT and dtype not in {
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
        }:
            raise ValueError("scalar-int Megatron output requires an integer dtype")

    @staticmethod
    def _require_scalar_element_shape(shape: tuple[int, ...]) -> None:
        if any(dimension < 0 for dimension in shape) or prod(shape) != 1:
            raise ValueError("Megatron scalar output must contain exactly one value per row")

    def _common_coordinates(self, metadata):
        attempt_id = int(metadata.attempt_id)
        invocation_id = int(metadata.invocation_id)
        initial_weight = (attempt_id == -1 and metadata.direction == "iter"
                          and metadata.act_name in {"query_projection_weight",
                              "key_projection_weight", "router_projection_weight"})
        if not initial_weight and not 0 <= attempt_id < 1 << 31:
            raise ValueError("attempt_id is outside the supported Int32 range")
        if not 0 <= invocation_id < 1 << 31:
            raise ValueError("invocation_id is outside the supported Int32 range")
        return (metadata.model_id, metadata.act_name, metadata.direction, metadata.phase,
                int(metadata.global_batch_id), int(metadata.dp_rank), int(metadata.microbatch_id),
                int(metadata.layer_no), int(metadata.shard_rank), attempt_id, invocation_id,
                self.producer_rank)

    def _coordinates(self, metadata, *, sample_index, token_start, token_end, dataset_id):
        common = self._common_coordinates(metadata)
        dataset_id = int(dataset_id)
        if not -1 <= dataset_id < 1 << 31:
            raise ValueError("dataset_id is outside the supported Int32 range")
        return (*common[:7], int(sample_index), *common[7:9], int(token_start), int(token_end),
                *common[9:11], dataset_id, common[11])

    @staticmethod
    def _layout_name(storage: OutputStorage) -> str:
        if storage is OutputStorage.TENSOR:
            return TENSOR_LAYOUT_NAME
        if storage is OutputStorage.SCALAR_FLOAT:
            return SCALAR_FLOAT_LAYOUT_NAME
        if storage is OutputStorage.SCALAR_INT:
            return SCALAR_INT_LAYOUT_NAME
        raise ValueError(f"unsupported output storage: {storage!r}")

    @staticmethod
    def _entry_bytes(entry: ProducerPlanEntry) -> int | None:
        if -1 in entry.output_shape:
            return None
        element_size = entry.element_size
        return int(prod(entry.output_shape)) * element_size


__all__ = [
    "EVALUATION_BOUNDARY_CELL_TYPES",
    "EVALUATION_BOUNDARY_LAYOUT_NAME",
    "EVALUATION_BOUNDARY_NBYTES",
    "MegatronRecordFormat",
    "evaluation_boundary_row",
    "required_record_metadata_fields",
]
