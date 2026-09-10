"""Persist the runtime validation-dataset ordinal-to-path mapping."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Sequence


SCHEMA_VERSION = 1
FILE_SUFFIX = ".validation_dataset_id_map.json"


def _destination(output_dir: str | os.PathLike[str], run_id: str) -> Path:
    run_id = str(run_id).strip()
    if not run_id or run_id in {".", ".."} or Path(run_id).name != run_id:
        raise ValueError("DMI run_id must be a nonempty filename-safe component")
    return Path(output_dir).expanduser() / f"{run_id}{FILE_SUFFIX}"


def _dataset_path(dataset: Any) -> str:
    path = getattr(dataset, "dataset_path", None)
    if path is None or not str(path).strip():
        raise ValueError("validation dataset has no runtime dataset_path")
    return str(path)


def write_validation_dataset_id_map(
    output_dir: str | os.PathLike[str],
    *,
    run_id: str,
    validation_datasets: Sequence[Any],
    global_rank: int,
) -> Path | None:
    """Write one immutable mapping on global rank zero.

    Megatron assigns ``dataset_id`` by enumerating this same ordered validation
    dataset list. Data parallelism shards samples inside each dataset and does
    not alter this list or its ordinals.
    """

    if int(global_rank) != 0:
        return None

    destination = _destination(output_dir, run_id)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "run_id": str(run_id).strip(),
        "datasets": [
            {"dataset_id": dataset_id, "dataset_path": _dataset_path(dataset)}
            for dataset_id, dataset in enumerate(validation_datasets)
        ],
    }
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"

    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.read_text(encoding="utf-8") == encoded:
            return destination
        raise RuntimeError(
            f"validation dataset ID map already exists with different content: {destination}"
        )

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)
    return destination


__all__ = ["write_validation_dataset_id_map"]
