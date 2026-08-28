"""Megatron semantic coordinates for one public DMI producer output."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class MegatronRecordMetadata:
    """Coordinates shared by rows materialized from one producer output."""

    model_id: str
    act_name: str
    direction: str
    phase: str
    global_batch_id: int
    dp_rank: int
    microbatch_id: int
    layer_no: int
    shard_rank: int
    token_start: int
    valid_counts: tuple[int, ...] = ()
    dataset_ids: tuple[int, ...] = ()
    attempt_id: int = 0
    invocation_id: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "valid_counts",
            tuple(int(value) for value in self.valid_counts),
        )
        object.__setattr__(
            self,
            "dataset_ids",
            tuple(int(value) for value in self.dataset_ids),
        )


__all__ = ["MegatronRecordMetadata"]
