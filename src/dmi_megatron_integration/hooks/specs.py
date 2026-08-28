"""Megatron policy layered over DMI's physical record-hook contract."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, IntEnum
from math import prod
from typing import Any, Callable, Mapping, Sequence

import torch

from dmi.api.v1 import (
    HookSpecV1,
    OutputStorage,
    RecordType,
    TransportSpec,
    TransportType,
)


class DPEmissionPolicy(IntEnum):
    ALL_DP_RANKS = 0
    DP_RANK_0 = 1


class ShardPolicy(Enum):
    REPLICATED = "replicated"
    GLOBAL_RANK_SHARDED = "global_rank_sharded"
    TP_SHARDED = "tp_sharded"
    EP_SHARDED = "ep_sharded"
    DP_SHARDED = "dp_sharded"
    CP_SHARDED = "cp_sharded"


class HookLayerPlacement(IntEnum):
    EVERY_LAYER = 0
    LAYER_SET = 1
    NO_LAYER_FIRST_PP = 2
    NO_LAYER_LAST_PP = 3


class HookRuntimeMode(Enum):
    EAGER_IMMEDIATE = "eager_immediate"
    CAPTURE_RECORD = "capture_record"
    REPLAY_PLANNED = "replay_planned"


class HookInputLayout(Enum):
    SEQ_BATCH = "seq_batch"
    PACKED_SEGMENTED = "packed_segmented"


class MegatronMetadataField(Enum):
    VALID_COUNT = "valid_count"
    SEGMENT_METADATA = "segment_metadata"
    DATASET_ID = "dataset_id"


class HookPhase(Enum):
    FWD = 1
    BWD = 2
    ITERATION = 3


class DimSpec(Enum):
    SEQ = "seq"
    BATCH = "batch"
    HIDDEN = "hidden"
    NUM_EXPERTS = "num_experts"
    VOCAB = "vocab"
    ACTUAL_TOKEN_PACKED = "actual_token_packed"


SymbolicDim = int | DimSpec | str


@dataclass(frozen=True, slots=True)
class MegatronOutputSpec:
    """Megatron's symbolic description of one public physical output."""

    name: str
    input_shape: Sequence[SymbolicDim]
    dtype: torch.dtype
    output_shape: Sequence[SymbolicDim] | None = None
    transport_type: TransportType = TransportType.IDENTITY
    storage: OutputStorage = OutputStorage.TENSOR
    row_bytes: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "input_shape", tuple(self.input_shape))
        if self.output_shape is not None:
            object.__setattr__(self, "output_shape", tuple(self.output_shape))
        if self.transport_type is not TransportType.PREFIX_STRIP and self.row_bytes is not None:
            raise ValueError("row_bytes is valid only for PREFIX_STRIP")

    @property
    def transport_metadata_fields(self) -> frozenset[MegatronMetadataField]:
        if self.transport_type is TransportType.SEQ_PREFIX_PACK:
            return frozenset({MegatronMetadataField.VALID_COUNT})
        if self.transport_type is TransportType.SEGMENTED_PACK:
            return frozenset({MegatronMetadataField.SEGMENT_METADATA})
        return frozenset()

    def resolved_input_shape(
        self,
        dims: Mapping[str | DimSpec, int] | None = None,
    ) -> tuple[int, ...]:
        return self._resolve_shape(self.input_shape, dims)

    def resolved_output_shape(
        self,
        dims: Mapping[str | DimSpec, int] | None = None,
    ) -> tuple[int, ...]:
        shape = self.input_shape if self.output_shape is None else self.output_shape
        return self._resolve_shape(shape, dims)

    def _resolve_shape(
        self,
        shape: Sequence[SymbolicDim],
        dims: Mapping[str | DimSpec, int] | None = None,
    ) -> tuple[int, ...]:
        dims = dims or {}
        resolved: list[int] = []
        for dim in shape:
            if isinstance(dim, int):
                resolved.append(dim)
            elif dim in dims:
                resolved.append(int(dims[dim]))
            elif dim is DimSpec.ACTUAL_TOKEN_PACKED:
                resolved.append(-1)
            elif isinstance(dim, DimSpec) and dim.value in dims:
                resolved.append(int(dims[dim.value]))
            else:
                raise KeyError(f"Unresolved output dim {dim!r} for {self.name}")
        return tuple(resolved)

    def resolved_row_bytes(
        self,
        dims: Mapping[str | DimSpec, int] | None = None,
    ) -> int:
        if self.row_bytes is not None:
            return int(self.row_bytes)
        shape = self.resolved_input_shape(dims)
        if not shape:
            raise ValueError(
                f"PREFIX_STRIP output {self.name} needs at least one dimension"
            )
        return int(prod(shape[1:])) * self.element_size()

    def element_size(self) -> int:
        return int(torch.empty((), dtype=self.dtype).element_size())

    def seq_prefix_pack_feature_bytes(
        self,
        dims: Mapping[str | DimSpec, int] | None = None,
    ) -> int:
        shape = self.resolved_input_shape(dims)
        if len(shape) < 2:
            raise ValueError(
                f"SEQ_PREFIX_PACK output {self.name} requires [S, B, ...] input"
            )
        return int(prod(shape[2:])) * self.element_size()

    def seq_prefix_pack_output_shape(
        self,
        valid_counts: Sequence[int],
        dims: Mapping[str | DimSpec, int] | None = None,
    ) -> tuple[int, ...]:
        input_shape = self.resolved_input_shape(dims)
        if len(input_shape) < 2:
            raise ValueError(
                f"SEQ_PREFIX_PACK output {self.name} requires [S, B, ...] input"
            )
        batch = int(input_shape[1])
        if len(valid_counts) != batch:
            raise ValueError(
                f"SEQ_PREFIX_PACK output {self.name} expected valid_count length "
                f"{batch}, got {len(valid_counts)}"
            )
        total = sum(max(0, int(value)) for value in valid_counts)
        return (int(total), *tuple(int(value) for value in input_shape[2:]))

    def segmented_pack_feature_bytes(
        self,
        dims: Mapping[str | DimSpec, int] | None = None,
    ) -> int:
        shape = self.resolved_input_shape(dims)
        if not shape:
            raise ValueError(
                f"SEGMENTED_PACK output {self.name} requires [R, ...] input"
            )
        return int(prod(shape[1:])) * self.element_size()

    def segmented_pack_output_shape(
        self,
        segment_lengths: Sequence[int],
        dims: Mapping[str | DimSpec, int] | None = None,
    ) -> tuple[int, ...]:
        input_shape = self.resolved_input_shape(dims)
        if not input_shape:
            raise ValueError(
                f"SEGMENTED_PACK output {self.name} requires [R, ...] input"
            )
        total = sum(max(0, int(value)) for value in segment_lengths)
        return (int(total), *tuple(int(value) for value in input_shape[1:]))

    def resolve(
        self,
        dims: Mapping[str | DimSpec, int] | None,
        *,
        record_type: RecordType,
    ) -> TransportSpec:
        """Resolve symbolic Megatron dimensions into one public DMI output."""

        row_bytes = None
        feature_bytes = None
        if self.transport_type is TransportType.PREFIX_STRIP:
            row_bytes = self.resolved_row_bytes(dims)
        elif self.transport_type is TransportType.SEQ_PREFIX_PACK:
            feature_bytes = self.seq_prefix_pack_feature_bytes(dims)
        elif self.transport_type is TransportType.SEGMENTED_PACK:
            feature_bytes = self.segmented_pack_feature_bytes(dims)
        output_shape = None
        if self.transport_type in (
            TransportType.SEQ_PREFIX_PACK,
            TransportType.SEGMENTED_PACK,
        ):
            output_shape = self.resolved_output_shape(dims)
        return TransportSpec(
            name=self.name,
            transport_type=self.transport_type,
            storage=self.storage,
            record_type=record_type,
            output_shape=output_shape,
            row_bytes=row_bytes,
            feature_bytes=feature_bytes,
        )


@dataclass(frozen=True, slots=True)
class MegatronHookSpec:
    """Megatron placement and row policy for one public physical hook."""

    name: str
    layer_no: int
    outputs: Sequence[MegatronOutputSpec]
    preprocess: Callable[..., Any] | None = None
    preprocess_metadata_fields: frozenset[MegatronMetadataField] = frozenset()
    shard_policy: ShardPolicy = ShardPolicy.REPLICATED
    layer_placement: HookLayerPlacement = HookLayerPlacement.EVERY_LAYER
    layer_selector: tuple[int, ...] | None = None
    enabled_by: frozenset[str] = frozenset()
    need_token_range: bool = True
    record_type: RecordType = RecordType.PER_SAMPLE
    dp_emission: DPEmissionPolicy = DPEmissionPolicy.ALL_DP_RANKS
    supported_layouts: frozenset[HookInputLayout] = frozenset(
        {HookInputLayout.SEQ_BATCH}
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "outputs", tuple(self.outputs))
        object.__setattr__(self, "enabled_by", frozenset(self.enabled_by))
        object.__setattr__(self, "supported_layouts", frozenset(self.supported_layouts))
        preprocess_metadata_fields = frozenset(self.preprocess_metadata_fields)
        if not all(
            isinstance(field, MegatronMetadataField)
            for field in preprocess_metadata_fields
        ):
            raise TypeError(
                "preprocess_metadata_fields must contain MegatronMetadataField values"
            )
        if MegatronMetadataField.DATASET_ID in preprocess_metadata_fields:
            raise ValueError("dataset_id is record metadata and cannot bind to a hook")
        object.__setattr__(
            self,
            "preprocess_metadata_fields",
            preprocess_metadata_fields,
        )
        if self.layer_selector is not None:
            object.__setattr__(
                self,
                "layer_selector",
                tuple(int(layer) for layer in self.layer_selector),
            )

    @property
    def binding_metadata_fields(self) -> frozenset[MegatronMetadataField]:
        fields = set(self.preprocess_metadata_fields)
        for output in self.outputs:
            fields.update(output.transport_metadata_fields)
        return frozenset(fields)

    def resolve(
        self,
        dims: Mapping[str | DimSpec, int] | None = None,
    ) -> HookSpecV1:
        """Produce the concrete public DMI hook bound to the record runtime."""

        return HookSpecV1(
            name=self.name,
            outputs=tuple(
                output.resolve(dims, record_type=self.record_type)
                for output in self.outputs
            ),
            preprocess=self.preprocess,
            enabled_by=self.enabled_by,
        )


__all__ = [
    "DPEmissionPolicy",
    "DimSpec",
    "HookInputLayout",
    "HookLayerPlacement",
    "HookPhase",
    "HookRuntimeMode",
    "MegatronMetadataField",
    "MegatronHookSpec",
    "MegatronOutputSpec",
    "ShardPolicy",
]
