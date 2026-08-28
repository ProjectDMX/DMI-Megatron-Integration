"""Test-only encoded-record sink for Megatron numeric verification."""

from __future__ import annotations

import json
import os
from collections import deque
from pathlib import Path
from typing import Any, Sequence

import torch

from dmi.api.v1 import (
    OutputStorage,
    PayloadSlice,
    RecordCellType,
    RecordDescriptor,
    RecordRuntime,
    RingCapacities,
    StepReservation,
    TransportType,
)
from dmi_megatron_integration.records.format import (
    EVALUATION_BOUNDARY_CELL_TYPES,
    EVALUATION_BOUNDARY_LAYOUT_NAME,
    EVALUATION_BOUNDARY_NBYTES,
)
from dmi_megatron_integration.records.schema import (
    TRAINING_ROW_COORDINATE_COLUMN_NAMES,
    TRAINING_SCHEMA_VERSION,
)


_METADATA_COLUMNS = TRAINING_ROW_COORDINATE_COLUMN_NAMES

_DTYPE_NAMES: dict[torch.dtype, str] = {
    torch.float32: "torch.float",
    torch.float64: "torch.double",
    torch.float16: "torch.half",
    torch.bfloat16: "torch.bfloat16",
    torch.uint8: "torch.uint8",
    torch.int8: "torch.int8",
    torch.int16: "torch.short",
    torch.int32: "torch.int",
    torch.int64: "torch.long",
    torch.bool: "torch.bool",
    torch.complex64: "torch.cfloat",
    torch.complex128: "torch.cdouble",
}
try:
    _DTYPE_NAMES[torch.complex32] = "torch.chalf"
except AttributeError:
    pass


def _rank_for_path() -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return int(torch.distributed.get_rank())
    if "RANK" in os.environ:
        return int(os.environ["RANK"])
    return 0


class MegatronTestFileSink:
    """Fake record transport and host sink used only by the file oracle.

    Every reservation reports ``OVERSIZED``. The public ``RecordRuntime``
    therefore publishes the integration's ordinary ``RecordDescriptor`` and
    invokes the same CPU-direct contract used by a real oversized producer.
    """

    def __init__(self, root_dir: str | os.PathLike[str], *, rank: int | None = None) -> None:
        self.root_dir = Path(root_dir)
        self.rank = _rank_for_path() if rank is None else int(rank)
        self.rank_dir = self.root_dir / f"rank{self.rank:03d}"
        self.rank_dir.mkdir(parents=True, exist_ok=True)
        self.rows_path = self.rank_dir / "rows.jsonl"
        self.scalar_float_rows_path = self.rank_dir / "scalar_float_rows.jsonl"
        self.scalar_int_rows_path = self.rank_dir / "scalar_int_rows.jsonl"
        self.eval_phase_boundary_path = self.rank_dir / "eval_phase_boundary.jsonl"
        self.null_offload = False
        self._payload = torch.empty(0, dtype=torch.uint8)
        self._descriptors: deque[RecordDescriptor] = deque()
        self._schema: Any | None = None
        self._row_id = 0
        self._closed = False

    def _record_payload_tensor(self) -> torch.Tensor:
        return self._payload

    def configure_record_schema(self, schema: Any) -> None:
        if self._schema is not None:
            raise RuntimeError("record schema is already configured")
        self._schema = schema

    def reserve_record(self, reservation_items: Any) -> int:
        items = tuple(reservation_items)
        if len(items) != 1:
            raise RuntimeError("file oracle accepts one eager producer at a time")
        return int(StepReservation.OVERSIZED)

    def push_record_descriptors(self, descriptors: Any) -> None:
        self._descriptors.extend(tuple(descriptors))

    def submit_record_cpu_direct(self, output: Any, entry: Any) -> None:
        if not self._descriptors:
            raise RuntimeError("file-oracle payload arrived without a descriptor")
        descriptor = self._descriptors.popleft()
        payload = self._record_cpu_tensor(output, entry)
        self._write_descriptor(descriptor, payload)

    def flush_records_and_wait(self, timeout_s: float) -> None:
        if float(timeout_s) <= 0:
            raise ValueError("timeout_s must be positive")
        if self._descriptors:
            raise RuntimeError(
                f"file oracle has {len(self._descriptors)} unconsumed descriptors"
            )

    def submit_record(
        self,
        layout: str,
        values: Sequence[Any],
        cell_types: Sequence[str],
        *,
        nbytes: int,
    ) -> None:
        if str(layout) != EVALUATION_BOUNDARY_LAYOUT_NAME:
            raise ValueError(f"unsupported evaluation-boundary file-oracle layout: {layout!r}")
        if int(nbytes) != EVALUATION_BOUNDARY_NBYTES:
            raise ValueError(
                "evaluation-boundary records must account for one Int64 payload"
            )
        if tuple(str(item) for item in cell_types) != EVALUATION_BOUNDARY_CELL_TYPES:
            raise ValueError("evaluation-boundary cell types do not match the record schema")
        if len(values) != 6:
            raise ValueError("evaluation-boundary row must contain six values")
        row = {
            "model_id": str(values[0]),
            "training_iteration_id": int(values[1]),
            "phase": str(values[2]),
            "eval_index": int(values[3]),
            "boundary_type": str(values[4]),
            "next_global_batch_id": int(values[5]),
        }
        self._validate_boundary(row)
        self._append_json(self.eval_phase_boundary_path, row)

    def close(self) -> None:
        if self._closed:
            return
        self.flush_records_and_wait(1.0)
        self._closed = True

    def _write_descriptor(
        self,
        descriptor: RecordDescriptor,
        payload: torch.Tensor,
    ) -> None:
        schema = self._schema
        if schema is None:
            raise RuntimeError("file oracle record schema is not configured")
        layout = schema.layout(descriptor.layout)
        byte_payload = payload.contiguous().view(torch.uint8).reshape(-1)
        for raw_row in descriptor.rows:
            row = {
                column.name: self._materialize_cell(cell, byte_payload)
                for column, cell in zip(layout.columns, raw_row)
            }
            self._write_training_row(layout, row)

    def _write_training_row(self, layout: Any, row: dict[str, Any]) -> None:
        metadata = {
            "schema_version": int(row.get("schema_version", TRAINING_SCHEMA_VERSION)),
            **{name: row[name] for name in _METADATA_COLUMNS},
        }
        tensor_columns = [
            column
            for column in layout.columns
            if column.type is RecordCellType.TENSOR
        ]
        if tensor_columns:
            if len(tensor_columns) != 1:
                raise ValueError("file oracle requires exactly one tensor value column")
            tensor = row[tensor_columns[0].name]
            if not isinstance(tensor, torch.Tensor):
                raise TypeError("file-oracle tensor row did not materialize a Tensor")
            tensor = tensor.contiguous().clone()
            row_id = self._row_id
            self._row_id += 1
            sample_index = int(metadata["sample_index"])
            payload_name = f"payload_{row_id:012d}_sample{sample_index:06d}.pt"
            torch.save(tensor, self.rank_dir / payload_name)
            metadata.update(
                {
                    "row_id": row_id,
                    "dtype": self._dtype_name(tensor.dtype),
                    "shape": [int(dim) for dim in tensor.shape],
                    "bytes": int(tensor.numel()) * int(tensor.element_size()),
                    "payload_file": payload_name,
                }
            )
            self._append_json(self.rows_path, metadata)
            return

        value_columns = [
            column
            for column in layout.columns
            if column.name not in _METADATA_COLUMNS
            and column.name != "schema_version"
        ]
        if len(value_columns) != 1 or value_columns[0].name != "value":
            raise ValueError("file oracle requires exactly one scalar value column")
        value_column = value_columns[0]
        value = row["value"]
        if value_column.type is RecordCellType.FLOAT64:
            metadata["value"] = float(value)
            self._append_json(self.scalar_float_rows_path, metadata)
            return
        if value_column.type is RecordCellType.INT64:
            metadata["value"] = int(value)
            self._append_json(self.scalar_int_rows_path, metadata)
            return
        raise TypeError(f"unsupported file-oracle scalar type: {value_column.type!r}")

    @staticmethod
    def _materialize_cell(cell: Any, byte_payload: torch.Tensor) -> Any:
        if not isinstance(cell, PayloadSlice):
            return cell
        begin = int(cell.offset_bytes)
        end = byte_payload.numel() if cell.nbytes is None else begin + int(cell.nbytes)
        raw = byte_payload[begin:end].clone()
        element_size = int(torch.empty((), dtype=cell.dtype).element_size())
        if raw.numel() % element_size != 0:
            raise ValueError("PayloadSlice bytes do not divide by the declared dtype")
        values = raw.view(cell.dtype)
        if cell.storage is OutputStorage.SCALAR_FLOAT:
            if values.numel() != 1 or not values.dtype.is_floating_point:
                raise TypeError("scalar-float PayloadSlice must contain one floating value")
            return float(values.item())
        if cell.storage is OutputStorage.SCALAR_INT:
            if values.numel() != 1 or values.dtype.is_floating_point or values.dtype is torch.bool:
                raise TypeError("scalar-int PayloadSlice must contain one integer value")
            return int(values.item())

        shape = list(cell.shape)
        if -1 in shape:
            known = 1
            for dim in shape:
                if dim != -1:
                    known *= int(dim)
            if known <= 0 or values.numel() % known != 0:
                raise ValueError("dynamic PayloadSlice shape cannot be inferred")
            shape[shape.index(-1)] = values.numel() // known
        if not shape:
            shape = [values.numel()]
        return values.reshape(tuple(shape))

    @staticmethod
    def _record_cpu_tensor(output: Any, entry: Any) -> torch.Tensor:
        """Match DMI's public record CPU-direct producer transformation."""

        source = output.tensor.detach().cpu().contiguous()
        transport_type = entry.transport_type
        if transport_type is TransportType.IDENTITY:
            return source

        byte_source = source.view(torch.uint8).reshape(-1)
        if transport_type is TransportType.PREFIX_STRIP:
            row_count = int(output.producer_meta[0].detach().cpu().item())
            row_bytes = int(entry.transport_args[0])
            nbytes = min(byte_source.numel(), max(0, row_count) * row_bytes)
            return byte_source[:nbytes].clone()

        if transport_type is TransportType.CHUNKED:
            counts = output.producer_meta[0].detach().cpu().reshape(-1)
            chunks = int(counts.numel())
            if chunks <= 0 or byte_source.numel() % chunks != 0:
                raise ValueError("CHUNKED CPU fallback requires equal input chunks")
            chunk_bytes = byte_source.numel() // chunks
            pieces = []
            for index, value in enumerate(counts.tolist()):
                count = max(0, min(chunk_bytes, int(value)))
                begin = index * chunk_bytes
                pieces.append(byte_source[begin : begin + count])
            return torch.cat(pieces).contiguous() if pieces else byte_source[:0].clone()

        if transport_type is TransportType.SEQ_PREFIX_PACK:
            if source.dim() < 2:
                raise ValueError("SEQ_PREFIX_PACK requires [S, B, ...] input")
            counts = output.producer_meta[0].detach().cpu().reshape(-1)
            if source.size(1) != counts.numel():
                raise ValueError("SEQ_PREFIX_PACK valid-count length mismatch")
            prefix = output.producer_meta[1].detach().cpu().reshape(-1)
            batch = int(counts.numel())
            if prefix.numel() != batch + 1:
                raise ValueError("SEQ_PREFIX_PACK prefix length must equal batch + 1")
            feature_bytes = int(entry.transport_args[0])
            physical_batch_bytes = batch * feature_bytes
            if batch <= 0 or feature_bytes <= 0 or byte_source.numel() % physical_batch_bytes != 0:
                raise ValueError("SEQ_PREFIX_PACK payload has incompatible feature bytes")
            max_rows = byte_source.numel() // feature_bytes
            total_rows = max(0, min(max_rows, int(prefix[-1].item())))
            prefix_values = [int(value) for value in prefix.tolist()]
            pieces = []
            for row in range(total_rows):
                sample = 0
                while sample + 1 < batch and row >= prefix_values[sample + 1]:
                    sample += 1
                sample_row = row - prefix_values[sample]
                begin = (sample_row * batch + sample) * feature_bytes
                pieces.append(byte_source[begin : begin + feature_bytes])
            return torch.cat(pieces).contiguous() if pieces else byte_source[:0].clone()

        if transport_type is TransportType.SEGMENTED_PACK:
            starts = output.producer_meta[0].detach().cpu().reshape(-1)
            ends = output.producer_meta[1].detach().cpu().reshape(-1)
            if starts.numel() == 0 or starts.numel() != ends.numel():
                raise ValueError("SEGMENTED_PACK start/end length mismatch")
            feature_bytes = int(entry.transport_args[0])
            if feature_bytes <= 0 or byte_source.numel() % feature_bytes != 0:
                raise ValueError("SEGMENTED_PACK payload has incompatible feature bytes")
            input_rows = byte_source.numel() // feature_bytes
            pieces = []
            for start, end in zip(starts.tolist(), ends.tolist()):
                first_row = max(0, min(input_rows, int(start)))
                last_row = max(first_row, min(input_rows, int(end)))
                for source_row in range(first_row, last_row):
                    begin = source_row * feature_bytes
                    pieces.append(byte_source[begin : begin + feature_bytes])
                    if len(pieces) == input_rows:
                        break
                if len(pieces) == input_rows:
                    break
            return torch.cat(pieces).contiguous() if pieces else byte_source[:0].clone()

        raise ValueError(f"unsupported transport type: {transport_type!r}")

    @staticmethod
    def _validate_boundary(row: dict[str, Any]) -> None:
        if not row["model_id"]:
            raise ValueError("file-oracle eval boundary requires non-empty model_id")
        if row["phase"] not in {"valid", "test"}:
            raise ValueError("file-oracle eval boundary phase must be valid or test")
        if row["boundary_type"] not in {"entry", "exit"}:
            raise ValueError("file-oracle eval boundary type must be entry or exit")
        if (
            row["training_iteration_id"] < 1
            or row["eval_index"] < 0
            or row["next_global_batch_id"] < 1
        ):
            raise ValueError("file-oracle eval boundary ids must be positive")

    @staticmethod
    def _append_json(path: Path, row: dict[str, Any]) -> None:
        with path.open("a", encoding="utf-8") as out:
            out.write(json.dumps(row, sort_keys=True) + "\n")

    @staticmethod
    def _dtype_name(dtype: torch.dtype) -> str:
        if dtype not in _DTYPE_NAMES:
            raise TypeError(f"unsupported file-oracle dtype: {dtype}")
        return _DTYPE_NAMES[dtype]


class MegatronTestRecordEngine:
    """Minimal MonitoringEngine contract backed by ``MegatronTestFileSink``."""

    def __init__(self, sink: MegatronTestFileSink) -> None:
        self.sink = sink
        self._runtime_created = False
        self._closed = False

    @property
    def capture_enabled(self) -> bool:
        return not self.sink.null_offload

    def set_capture_enabled(self, enabled: bool) -> None:
        self.sink.null_offload = not bool(enabled)

    def create_record_runtime(self, record_format: Any) -> RecordRuntime[Any]:
        if self._runtime_created:
            raise RuntimeError("a test record runtime is already active")
        self._runtime_created = True
        return RecordRuntime(self.sink, record_format)

    def ring_capacities(self) -> RingCapacities:
        return RingCapacities(
            payload_bytes=1 << 60,
            staging_bytes=1 << 60,
            task_entries=1 << 30,
        )

    def flush_and_wait(self, timeout_s: float = 600.0) -> None:
        self.sink.flush_records_and_wait(timeout_s)

    def close(self) -> None:
        if self._closed:
            return
        self.sink.close()
        self._closed = True


__all__ = ["MegatronTestFileSink", "MegatronTestRecordEngine"]
