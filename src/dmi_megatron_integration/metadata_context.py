"""DMI metadata context for Megatron training hooks.

This module stores small tensors needed by HookPointV1 preprocessing.  It does
not import Megatron-LM and does not know Megatron schedules; schedule patches
decide when to ingest a microbatch and when a microbatch enters a hook-owning
scope.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Mapping, Sequence

import torch

from .hooks.specs import DimSpec


class DMIMetadataDirection(Enum):
    FWD = "fwd"
    BWD = "bwd"


@dataclass(frozen=True)
class DMIMetadataFieldSpec:
    name: str
    shape: Sequence[int | DimSpec | str]
    dtype: torch.dtype
    source_scope: str = "microbatch"
    current_scope: str = "direction_scope"
    gpu_visible: bool = True

    def resolved_shape(self, dims: Mapping[str | DimSpec, int]) -> tuple[int, ...]:
        out: list[int] = []
        for dim in self.shape:
            if isinstance(dim, int):
                out.append(int(dim))
            elif dim in dims:
                out.append(int(dims[dim]))
            elif isinstance(dim, DimSpec) and dim.value in dims:
                out.append(int(dims[dim.value]))
            else:
                raise KeyError(f"Unresolved metadata dim {dim!r} for {self.name}")
        return tuple(out)


def valid_count_field_spec(
    size: int | DimSpec = DimSpec.BATCH,
    *,
    gpu_visible: bool = True,
) -> DMIMetadataFieldSpec:
    return DMIMetadataFieldSpec(
        name="valid_count",
        shape=[size],
        dtype=torch.int64,
        gpu_visible=gpu_visible,
    )


def dataset_id_field_spec(size: int | DimSpec = DimSpec.BATCH) -> DMIMetadataFieldSpec:
    return DMIMetadataFieldSpec(
        name="dataset_id",
        shape=[size],
        dtype=torch.int64,
        gpu_visible=False,
    )


def segment_metadata_field_spec(segment_capacity: int) -> DMIMetadataFieldSpec:
    if segment_capacity <= 0:
        raise ValueError("segment_capacity must be positive")
    return DMIMetadataFieldSpec(
        name="segment_metadata",
        shape=[2 * int(segment_capacity)],
        dtype=torch.int64,
    )


class DMIMetadataContext:
    """Stable-address source/current buffers for DMI preprocessing metadata."""

    def __init__(
        self,
        *,
        max_num_microbatches: int,
        max_batch_size: int,
        num_scopes: int,
        field_specs: Sequence[DMIMetadataFieldSpec] | None = None,
        dims: Mapping[str | DimSpec, int] | None = None,
        device: torch.device | str | int | None = None,
    ) -> None:
        if max_num_microbatches <= 0:
            raise ValueError("max_num_microbatches must be positive")
        if max_batch_size <= 0:
            raise ValueError("max_batch_size must be positive")
        if num_scopes <= 0:
            raise ValueError("num_scopes must be positive")

        self.max_num_microbatches = int(max_num_microbatches)
        self.max_batch_size = int(max_batch_size)
        self.num_scopes = int(num_scopes)
        if isinstance(device, int):
            self.device = torch.device("cuda", device)
        else:
            self.device = torch.device(device) if device is not None else torch.device("cuda")
        self.dims: dict[str | DimSpec, int] = {DimSpec.BATCH: self.max_batch_size}
        if dims is not None:
            self.dims.update(dict(dims))
        self.dims[DimSpec.BATCH] = self.max_batch_size
        self.dims[DimSpec.BATCH.value] = self.max_batch_size

        specs = (
            (valid_count_field_spec(),)
            if field_specs is None
            else tuple(field_specs)
        )
        if len({spec.name for spec in specs}) != len(specs):
            raise ValueError("metadata field names must be unique")
        self.field_specs: dict[str, DMIMetadataFieldSpec] = {spec.name: spec for spec in specs}
        self.active_num_microbatches = 0
        self._direction_to_index = {
            DMIMetadataDirection.FWD.value: 0,
            DMIMetadataDirection.BWD.value: 1,
        }

        self._source_buffers: dict[str, torch.Tensor] = {}
        self._source_cpu_buffers: dict[str, torch.Tensor] = {}
        self._source_cpu_packets: dict[torch.dtype, torch.Tensor] = {}
        self._field_packet_ranges: dict[str, tuple[torch.dtype, int, int]] = {}
        self._current_buffers: dict[str, torch.Tensor] = {}
        wire_offsets: dict[torch.dtype, int] = {}
        resolved_shapes: dict[str, tuple[int, ...]] = {}
        for spec in specs:
            if spec.dtype not in (torch.int64, torch.float64):
                raise TypeError(
                    f"DMI CPU metadata field {spec.name!r} must use Int64 or Float64"
                )
            shape = spec.resolved_shape(self.dims)
            resolved_shapes[spec.name] = shape
            numel = 1
            for dim in shape:
                numel *= int(dim)
            offset = wire_offsets.get(spec.dtype, 0)
            self._field_packet_ranges[spec.name] = (spec.dtype, offset, numel)
            wire_offsets[spec.dtype] = offset + numel

        for dtype, packet_length in wire_offsets.items():
            self._source_cpu_packets[dtype] = torch.zeros(
                (self.max_num_microbatches, packet_length),
                dtype=dtype,
                device="cpu",
            )
        for spec in specs:
            dtype, offset, numel = self._field_packet_ranges[spec.name]
            shape = resolved_shapes[spec.name]
            self._source_cpu_buffers[spec.name] = self._source_cpu_packets[dtype].narrow(
                1, offset, numel
            ).view(self.max_num_microbatches, *shape)
            if spec.gpu_visible:
                self._source_buffers[spec.name] = torch.zeros(
                    (self.max_num_microbatches, *shape),
                    dtype=spec.dtype,
                    device=self.device,
                )
                self._current_buffers[spec.name] = torch.zeros(
                    (len(self._direction_to_index), self.num_scopes, *shape),
                    dtype=spec.dtype,
                    device=self.device,
                )
        self._active_field_names = frozenset(self.field_specs)

    def set_active_fields(self, names: Sequence[str]) -> None:
        active = tuple(str(name) for name in names)
        if len(active) != len(set(active)):
            raise ValueError("DMI active metadata field names must be unique")
        unknown = set(active) - set(self.field_specs)
        if unknown:
            raise KeyError(f"Unknown DMI metadata fields: {sorted(unknown)}")
        self._active_field_names = frozenset(active)

    @property
    def active_field_names(self) -> frozenset[str]:
        return self._active_field_names

    def active_cpu_packets(self, microbatch_id: int) -> dict[torch.dtype, torch.Tensor]:
        self._check_allocated_microbatch_id(microbatch_id)
        packet_ends: dict[torch.dtype, int] = {}
        for name in self._active_field_names:
            dtype, offset, numel = self._field_packet_ranges[name]
            packet_ends[dtype] = max(packet_ends.get(dtype, 0), offset + numel)
        return {
            dtype: self._source_cpu_packets[dtype][microbatch_id].narrow(0, 0, end)
            for dtype, end in packet_ends.items()
            if end > 0
        }

    def begin_iteration(self, active_num_microbatches: int, *, clear_buffers: bool = True) -> None:
        if active_num_microbatches < 0:
            raise ValueError("active_num_microbatches must be non-negative")
        if active_num_microbatches > self.max_num_microbatches:
            raise ValueError(
                "active_num_microbatches exceeds max_num_microbatches: "
                f"{active_num_microbatches} > {self.max_num_microbatches}"
            )
        self.active_num_microbatches = int(active_num_microbatches)
        if not clear_buffers:
            return
        for buf in self._source_cpu_buffers.values():
            buf.zero_()
        for buf in self._source_buffers.values():
            buf.zero_()
        for buf in self._current_buffers.values():
            buf.zero_()

    def ingest_microbatch(
        self,
        microbatch_id: int,
        fields: Mapping[str, torch.Tensor | Sequence[int] | int],
        cpu_fields: Mapping[str, torch.Tensor | Sequence[int] | int] | None = None,
    ) -> None:
        self._check_microbatch_id(microbatch_id)
        self.load_source_microbatch(microbatch_id, fields, cpu_fields=cpu_fields)

    def load_source_microbatch(
        self,
        microbatch_id: int,
        fields: Mapping[str, torch.Tensor | Sequence[int] | int],
        cpu_fields: Mapping[str, torch.Tensor | Sequence[int] | int] | None = None,
    ) -> None:
        self._check_allocated_microbatch_id(microbatch_id)
        for name in self._active_field_names:
            spec = self.field_specs[name]
            if name not in fields and (cpu_fields is None or name not in cpu_fields):
                raise KeyError(f"Missing DMI metadata field {name!r}")
            cpu_value_source = (
                fields[name]
                if cpu_fields is None or name not in cpu_fields
                else cpu_fields[name]
            )
            cpu_target = self._source_cpu_buffers[name][microbatch_id]
            cpu_target.zero_()
            cpu_value = self._normalize_cpu_field_value(cpu_value_source, spec)
            if cpu_value.numel() > cpu_target.numel():
                raise ValueError(
                    f"DMI CPU metadata field {name!r} has too many elements: "
                    f"{cpu_value.numel()} > {cpu_target.numel()}"
                )
            cpu_target.view(-1)[: cpu_value.numel()].copy_(cpu_value.view(-1))

            if not spec.gpu_visible:
                continue
            if name not in fields:
                raise KeyError(f"Missing GPU-visible DMI metadata field {name!r}")
            target = self._source_buffers[name][microbatch_id]
            target.zero_()
            value = self._normalize_field_value(fields[name], spec)
            if value.numel() > target.numel():
                raise ValueError(
                    f"DMI metadata field {name!r} has too many elements: "
                    f"{value.numel()} > {target.numel()}"
                )
            target.view(-1)[: value.numel()].copy_(value.view(-1))

    def enter_scope(
        self,
        direction: DMIMetadataDirection | str,
        scope_id: int,
        microbatch_id: int,
    ) -> None:
        direction_idx = self._direction_index(direction)
        self._check_scope_id(scope_id)
        self._check_microbatch_id(microbatch_id)
        for name in self._active_field_names:
            if name not in self._current_buffers:
                continue
            self._current_buffers[name][direction_idx, scope_id].copy_(
                self._source_buffers[name][microbatch_id]
            )

    def current(
        self,
        name: str,
        direction: DMIMetadataDirection | str,
        scope_id: int,
    ) -> torch.Tensor:
        if name not in self._current_buffers:
            raise KeyError(f"Unknown DMI metadata field {name!r}")
        direction_idx = self._direction_index(direction)
        self._check_scope_id(scope_id)
        return self._current_buffers[name][direction_idx, scope_id]

    def source(self, name: str, microbatch_id: int) -> torch.Tensor:
        if name not in self._source_buffers:
            raise KeyError(f"Unknown DMI metadata field {name!r}")
        self._check_allocated_microbatch_id(microbatch_id)
        return self._source_buffers[name][microbatch_id]

    def source_cpu_tensor(self, name: str, microbatch_id: int) -> torch.Tensor:
        if name not in self._source_cpu_buffers:
            raise KeyError(f"Unknown DMI metadata field {name!r}")
        self._check_allocated_microbatch_id(microbatch_id)
        return self._source_cpu_buffers[name][microbatch_id]

    def source_cpu(self, name: str, microbatch_id: int) -> tuple[int, ...]:
        tensor = self.source_cpu_tensor(name, microbatch_id)
        return tuple(int(x) for x in tensor.view(-1).tolist())

    def sync_source_gpu_from_cpu(self, name: str, microbatch_id: int) -> None:
        if name not in self.field_specs:
            raise KeyError(f"Unknown DMI metadata field {name!r}")
        self._check_allocated_microbatch_id(microbatch_id)
        if name not in self._source_buffers:
            raise ValueError(f"DMI metadata field {name!r} is CPU-only")
        self._source_buffers[name][microbatch_id].copy_(
            self._source_cpu_buffers[name][microbatch_id],
            non_blocking=True,
        )

    def end_iteration(self) -> None:
        self.active_num_microbatches = 0

    def _normalize_field_value(
        self,
        value: torch.Tensor | Sequence[int] | int,
        spec: DMIMetadataFieldSpec,
    ) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            return value.to(device=self.device, dtype=spec.dtype, non_blocking=True).contiguous()
        return torch.as_tensor(value, dtype=spec.dtype, device=self.device).contiguous()

    @staticmethod
    def _normalize_cpu_field_value(
        value: torch.Tensor | Sequence[int] | int,
        spec: DMIMetadataFieldSpec,
    ) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            if value.is_cuda:
                raise RuntimeError(
                    f"DMI CPU metadata field {spec.name!r} must not be recovered from CUDA"
                )
            return value.to(device="cpu", dtype=spec.dtype).contiguous()
        return torch.as_tensor(value, dtype=spec.dtype, device="cpu").contiguous()

    def _direction_index(self, direction: DMIMetadataDirection | str) -> int:
        key = direction.value if isinstance(direction, DMIMetadataDirection) else str(direction)
        if key not in self._direction_to_index:
            raise KeyError(f"Unknown DMI metadata direction {direction!r}")
        return self._direction_to_index[key]

    def _check_microbatch_id(self, microbatch_id: int) -> None:
        if microbatch_id < 0 or microbatch_id >= self.active_num_microbatches:
            raise IndexError(
                f"microbatch_id {microbatch_id} outside active iteration "
                f"[0, {self.active_num_microbatches})"
            )

    def _check_allocated_microbatch_id(self, microbatch_id: int) -> None:
        if microbatch_id < 0 or microbatch_id >= self.max_num_microbatches:
            raise IndexError(
                f"microbatch_id {microbatch_id} outside allocated range "
                f"[0, {self.max_num_microbatches})"
            )

    def _check_scope_id(self, scope_id: int) -> None:
        if scope_id < 0 or scope_id >= self.num_scopes:
            raise IndexError(f"scope_id {scope_id} outside [0, {self.num_scopes})")


class DMIMetadataPropagator(ABC):
    """Schedule-facing API for moving metadata into hook-visible buffers."""

    def __init__(self, context: DMIMetadataContext) -> None:
        self.context = context

    @property
    @abstractmethod
    def is_metadata_source_rank(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def begin_iteration(self, active_num_microbatches: int) -> None:
        raise NotImplementedError

    @abstractmethod
    def ingest_microbatch(
        self,
        microbatch_id: int,
        fields: Mapping[str, torch.Tensor | Sequence[int] | int],
        cpu_fields: Mapping[str, torch.Tensor | Sequence[int] | int] | None = None,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def wait_microbatch(self, microbatch_id: int) -> None:
        raise NotImplementedError

    @abstractmethod
    def enter_scope(
        self,
        direction: DMIMetadataDirection | str,
        scope_id: int,
        microbatch_id: int,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def end_iteration(self) -> None:
        raise NotImplementedError


class LocalMetadataPropagator(DMIMetadataPropagator):
    """No-distributed metadata propagator for local source/hook ownership."""

    @property
    def is_metadata_source_rank(self) -> bool:
        return True

    def begin_iteration(self, active_num_microbatches: int) -> None:
        self.context.begin_iteration(active_num_microbatches)

    def ingest_microbatch(
        self,
        microbatch_id: int,
        fields: Mapping[str, torch.Tensor | Sequence[int] | int],
        cpu_fields: Mapping[str, torch.Tensor | Sequence[int] | int] | None = None,
    ) -> None:
        self.context.ingest_microbatch(microbatch_id, fields, cpu_fields=cpu_fields)

    def wait_microbatch(self, microbatch_id: int) -> None:
        self.context._check_microbatch_id(microbatch_id)

    def enter_scope(
        self,
        direction: DMIMetadataDirection | str,
        scope_id: int,
        microbatch_id: int,
    ) -> None:
        self.wait_microbatch(microbatch_id)
        self.context.enter_scope(direction, scope_id, microbatch_id)

    def end_iteration(self) -> None:
        self.context.end_iteration()


class PerDPCPUMetadataPropagator(DMIMetadataPropagator):
    """Per-DP CPU metadata propagation for Megatron training hooks.

    Metadata is CPU-authoritative.  The canonical source is PP0/TP0 in one
    model-parallel island.  PP propagation uses async CPU broadcast over a
    DMI-owned Gloo group containing TP0 ranks across PP stages.  TP propagation
    uses sync CPU broadcast over a DMI-owned Gloo group inside each PP stage.
    """

    def __init__(
        self,
        context: DMIMetadataContext,
        *,
        rank: int,
        pp_source_rank: int,
        tp_source_rank: int,
        pp_cpu_ranks: Sequence[int],
        tp_cpu_ranks: Sequence[int],
        pp_cpu_group=None,
        tp_cpu_group=None,
        dist_module=None,
    ) -> None:
        super().__init__(context)
        self.rank = int(rank)
        self.pp_source_rank = int(pp_source_rank)
        self.tp_source_rank = int(tp_source_rank)
        self.pp_cpu_ranks = tuple(int(rank) for rank in pp_cpu_ranks)
        self.tp_cpu_ranks = tuple(int(rank) for rank in tp_cpu_ranks)
        self.pp_cpu_group = pp_cpu_group
        self.tp_cpu_group = tp_cpu_group
        self.dist = dist_module if dist_module is not None else torch.distributed
        self._pp_handles: dict[tuple[int, torch.dtype], object] = {}
        self._pp_send_handles: list[object] = []
        self._pp_synced_microbatches: set[int] = set()
        self._gpu_synced_microbatches: set[int] = set()
        self._tp_synced_microbatches: set[int] = set()

    @property
    def is_pp_source_rank(self) -> bool:
        return self.rank == self.pp_source_rank

    @property
    def is_metadata_source_rank(self) -> bool:
        return self.is_pp_source_rank

    @property
    def is_pp_member_rank(self) -> bool:
        return self.rank in self.pp_cpu_ranks

    @property
    def is_tp_member_rank(self) -> bool:
        return self.rank in self.tp_cpu_ranks

    @property
    def is_tp_source_rank(self) -> bool:
        return self.rank == self.tp_source_rank

    def begin_iteration(self, active_num_microbatches: int) -> None:
        self._wait_all(self._pp_send_handles)
        self._wait_all(self._pp_handles.values())
        self._pp_send_handles.clear()
        self._pp_handles.clear()
        self._pp_synced_microbatches.clear()
        self._gpu_synced_microbatches.clear()
        self._tp_synced_microbatches.clear()
        self.context.begin_iteration(active_num_microbatches)

        if not self.is_pp_member_rank or self.is_pp_source_rank:
            return
        if self.pp_cpu_group is None:
            raise RuntimeError("DMI PP CPU metadata propagation requires a PP CPU group")
        for microbatch_id in range(self.context.active_num_microbatches):
            for dtype, packet in self.context.active_cpu_packets(microbatch_id).items():
                handle = self.dist.broadcast(
                    packet,
                    src=self.pp_source_rank,
                    group=self.pp_cpu_group,
                    async_op=True,
                )
                self._pp_handles[(microbatch_id, dtype)] = handle

    def ingest_microbatch(
        self,
        microbatch_id: int,
        fields: Mapping[str, torch.Tensor | Sequence[int] | int],
        cpu_fields: Mapping[str, torch.Tensor | Sequence[int] | int] | None = None,
    ) -> None:
        if self.is_pp_source_rank:
            self.context.ingest_microbatch(microbatch_id, fields, cpu_fields=cpu_fields)
            if len(self.pp_cpu_ranks) > 1:
                if self.pp_cpu_group is None:
                    raise RuntimeError("DMI PP CPU metadata propagation requires a PP CPU group")
                for packet in self.context.active_cpu_packets(microbatch_id).values():
                    handle = self.dist.broadcast(
                        packet,
                        src=self.pp_source_rank,
                        group=self.pp_cpu_group,
                        async_op=True,
                    )
                    self._pp_send_handles.append(handle)
            return

        self.context._check_microbatch_id(microbatch_id)

    def wait_microbatch(self, microbatch_id: int) -> None:
        self.context._check_microbatch_id(microbatch_id)
        if not self.is_tp_member_rank:
            return

        self._wait_pp_microbatch(microbatch_id)
        self._sync_tp_microbatch(microbatch_id)
        if microbatch_id not in self._gpu_synced_microbatches:
            for name in self.context.active_field_names:
                if not self.context.field_specs[name].gpu_visible:
                    continue
                self.context.sync_source_gpu_from_cpu(name, microbatch_id)
            self._gpu_synced_microbatches.add(microbatch_id)

    def enter_scope(
        self,
        direction: DMIMetadataDirection | str,
        scope_id: int,
        microbatch_id: int,
    ) -> None:
        self.wait_microbatch(microbatch_id)
        self.context.enter_scope(direction, scope_id, microbatch_id)

    def end_iteration(self) -> None:
        self._wait_all(self._pp_handles.values())
        self._wait_all(self._pp_send_handles)
        self._pp_handles.clear()
        self._pp_send_handles.clear()
        self._pp_synced_microbatches.clear()
        self._gpu_synced_microbatches.clear()
        self._tp_synced_microbatches.clear()
        self.context.end_iteration()

    def _wait_pp_microbatch(self, microbatch_id: int) -> None:
        if microbatch_id in self._pp_synced_microbatches:
            return
        if self.is_pp_member_rank and not self.is_pp_source_rank:
            for dtype in self.context.active_cpu_packets(microbatch_id):
                key = (microbatch_id, dtype)
                handle = self._pp_handles.pop(key, None)
                if handle is None:
                    raise RuntimeError(
                        "DMI PP CPU metadata broadcast was not posted for "
                        f"microbatch {microbatch_id}, packet {dtype}"
                    )
                handle.wait()
        self._pp_synced_microbatches.add(microbatch_id)

    def _sync_tp_microbatch(self, microbatch_id: int) -> None:
        if microbatch_id in self._tp_synced_microbatches:
            return
        if len(self.tp_cpu_ranks) > 1:
            if self.tp_cpu_group is None:
                raise RuntimeError("DMI TP CPU metadata propagation requires a TP CPU group")
            for packet in self.context.active_cpu_packets(microbatch_id).values():
                self.dist.broadcast(
                    packet,
                    src=self.tp_source_rank,
                    group=self.tp_cpu_group,
                )
        self._tp_synced_microbatches.add(microbatch_id)

    @staticmethod
    def _wait_all(handles) -> None:
        for handle in list(handles):
            handle.wait()


TorchDistributedMetadataPropagator = PerDPCPUMetadataPropagator


__all__ = [
    "DMIMetadataContext",
    "DMIMetadataDirection",
    "DMIMetadataFieldSpec",
    "DMIMetadataPropagator",
    "LocalMetadataPropagator",
    "PerDPCPUMetadataPropagator",
    "TorchDistributedMetadataPropagator",
    "dataset_id_field_spec",
    "valid_count_field_spec",
    "segment_metadata_field_spec",
]
