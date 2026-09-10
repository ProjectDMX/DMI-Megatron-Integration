from types import SimpleNamespace

import pytest

from dmi_megatron_integration.validation_dataset_map import (
    write_validation_dataset_id_map,
)


def test_global_rank_zero_writes_runtime_dataset_map(tmp_path):
    datasets = [
        SimpleNamespace(dataset_path="/validation/math.KT"),
        SimpleNamespace(dataset_path="/validation/math.NT"),
    ]

    path = write_validation_dataset_id_map(
        tmp_path,
        run_id="olmoe_step1200000",
        validation_datasets=datasets,
        global_rank=0,
    )

    assert path == tmp_path / "olmoe_step1200000.validation_dataset_id_map.json"
    assert path.read_text() == (
        '{\n'
        '  "datasets": [\n'
        '    {\n'
        '      "dataset_id": 0,\n'
        '      "dataset_path": "/validation/math.KT"\n'
        '    },\n'
        '    {\n'
        '      "dataset_id": 1,\n'
        '      "dataset_path": "/validation/math.NT"\n'
        '    }\n'
        '  ],\n'
        '  "run_id": "olmoe_step1200000",\n'
        '  "schema_version": 1\n'
        '}\n'
    )


def test_nonzero_rank_does_not_write(tmp_path):
    result = write_validation_dataset_id_map(
        tmp_path,
        run_id="olmoe_step1200000",
        validation_datasets=[SimpleNamespace(dataset_path="/validation/math.KT")],
        global_rank=1,
    )
    assert result is None
    assert list(tmp_path.iterdir()) == []


def test_existing_different_map_is_rejected(tmp_path):
    first = [SimpleNamespace(dataset_path="/validation/math.KT")]
    second = [SimpleNamespace(dataset_path="/validation/math.NT")]
    write_validation_dataset_id_map(
        tmp_path,
        run_id="run",
        validation_datasets=first,
        global_rank=0,
    )
    with pytest.raises(RuntimeError, match="different content"):
        write_validation_dataset_id_map(
            tmp_path,
            run_id="run",
            validation_datasets=second,
            global_rank=0,
        )


def test_run_id_must_be_filename_safe(tmp_path):
    with pytest.raises(ValueError, match="filename-safe"):
        write_validation_dataset_id_map(
            tmp_path,
            run_id="parent/run",
            validation_datasets=[],
            global_rank=0,
        )
