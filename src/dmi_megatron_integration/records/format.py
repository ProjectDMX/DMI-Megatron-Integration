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
    """Schema-v2 descriptor encoder for Megatron's GPU record path."""

    def __init__(self, base_table: str, *, index_granularity: int = 8192) -> None:
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
        return RecordDescriptor(
            layout=self._layout_name(entry.storage),
            rows=self._rows(metadata, entry),
            output_id=entry.output_id,
        )

    def _rows(
        self,
        metadata: MegatronRecordMetadata,
        entry: ProducerPlanEntry,
    ) -> tuple[tuple[object, ...], ...]:
        if entry.record_type is RecordType.PER_SAMPLE:
            return self._per_sample_rows(metadata, entry)
        if entry.record_type not in (
            RecordType.PER_ITERATION,
            RecordType.PER_EXECUTION,
        ):
            raise ValueError(f"unsupported record type: {entry.record_type!r}")
        if entry.transport_type is not TransportType.IDENTITY:
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
        token_start = -1 if per_execution else int(metadata.token_start)
        token_end = -1 if per_execution else int(metadata.token_start) + 1
        value = self._payload_slice(
            entry,
            offset_bytes=0,
            nbytes=self._entry_bytes(entry),
            shape=entry.output_shape,
        )
        return (
            self._coordinates(
                metadata,
                sample_index=-1,
                token_start=token_start,
                token_end=token_end,
                dataset_id=-1,
            )
            + (value,),
        )

    def _per_sample_rows(
        self,
        metadata: MegatronRecordMetadata,
        entry: ProducerPlanEntry,
    ) -> tuple[tuple[object, ...], ...]:
        counts = self._record_counts(metadata, entry)
        self._validate_per_sample_shape(entry, counts)
        if metadata.dataset_ids and len(metadata.dataset_ids) != len(counts):
            raise ValueError("dataset_ids length must match the record count")
        datasets = metadata.dataset_ids or (0,) * len(counts)

        rows: list[tuple[object, ...]] = []
        packed_offset = 0
        active_index = 0
        for sample_index, (valid_count, dataset_id) in enumerate(
            zip(counts, datasets)
        ):
            if valid_count <= 0:
                continue
            value, packed_offset, active_index = self._sample_payload_slice(
                entry,
                sample_index=sample_index,
                valid_count=valid_count,
                packed_offset=packed_offset,
                active_index=active_index,
            )
            rows.append(
                self._coordinates(
                    metadata,
                    sample_index=sample_index,
                    token_start=int(metadata.token_start),
                    token_end=int(metadata.token_start) + valid_count,
                    dataset_id=dataset_id,
                )
                + (value,)
            )
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
        if entry.storage in (OutputStorage.SCALAR_FLOAT, OutputStorage.SCALAR_INT):
            if entry.transport_type is TransportType.PREFIX_STRIP:
                if not metadata.valid_counts:
                    raise ValueError(
                        "PREFIX_STRIP scalar Megatron records require valid_counts"
                    )
                return tuple(
                    1 if int(count) > 0 else 0 for count in metadata.valid_counts
                )
            if not entry.output_shape:
                return (1,)
            return (1,) * max(0, int(entry.output_shape[0]))
        if metadata.valid_counts:
            return metadata.valid_counts
        if entry.transport_type in (
            TransportType.SEQ_PREFIX_PACK,
            TransportType.SEGMENTED_PACK,
        ):
            raise ValueError("packed Megatron records require valid_counts")
        if not entry.output_shape:
            return (1,)
        return (1,) * max(0, int(entry.output_shape[0]))

    def _sample_payload_slice(
        self,
        entry: ProducerPlanEntry,
        *,
        sample_index: int,
        valid_count: int,
        packed_offset: int,
        active_index: int,
    ) -> tuple[PayloadSlice, int, int]:
        element_size = int(torch.empty((), dtype=entry.dtype).element_size())
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
                self._payload_slice(
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
                self._payload_slice(
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
            value = self._payload_slice(
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
            value = self._payload_slice(
                entry,
                offset_bytes=packed_offset,
                nbytes=nbytes,
                shape=(valid_count, *entry.output_shape[1:]),
            )
            return value, packed_offset + nbytes, active_index

        raise ValueError(
            f"unsupported per-sample Megatron transport: {entry.transport_type.value}"
        )

    @classmethod
    def _payload_slice(
        cls,
        entry: ProducerPlanEntry,
        *,
        offset_bytes: int,
        nbytes: int | None,
        shape: tuple[int, ...],
    ) -> PayloadSlice:
        cls._validate_scalar_dtype(entry.storage, entry.dtype)
        return PayloadSlice(
            offset_bytes=int(offset_bytes),
            nbytes=None if nbytes is None else int(nbytes),
            storage=entry.storage,
            dtype=entry.dtype,
            shape=shape if entry.storage is OutputStorage.TENSOR else (),
        )

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

    @staticmethod
    def _coordinates(
        metadata: MegatronRecordMetadata,
        *,
        sample_index: int,
        token_start: int,
        token_end: int,
        dataset_id: int,
    ) -> tuple[object, ...]:
        attempt_id = int(metadata.attempt_id)
        invocation_id = int(metadata.invocation_id)
        dataset_id = int(dataset_id)
        if not 0 <= attempt_id < 1 << 31:
            raise ValueError("attempt_id is outside the supported Int32 range")
        if not 0 <= invocation_id < 1 << 31:
            raise ValueError("invocation_id is outside the supported Int32 range")
        if not -1 <= dataset_id < 1 << 31:
            raise ValueError("dataset_id is outside the supported Int32 range")
        return (
            metadata.model_id,
            metadata.act_name,
            metadata.direction,
            metadata.phase,
            int(metadata.global_batch_id),
            int(metadata.dp_rank),
            int(metadata.microbatch_id),
            int(sample_index),
            int(metadata.layer_no),
            int(metadata.shard_rank),
            int(token_start),
            int(token_end),
            attempt_id,
            invocation_id,
            dataset_id,
        )

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
        element_size = int(torch.empty((), dtype=entry.dtype).element_size())
        return int(prod(entry.output_shape)) * element_size


__all__ = [
    "EVALUATION_BOUNDARY_CELL_TYPES",
    "EVALUATION_BOUNDARY_LAYOUT_NAME",
    "EVALUATION_BOUNDARY_NBYTES",
    "MegatronRecordFormat",
    "evaluation_boundary_row",
    "required_record_metadata_fields",
]
