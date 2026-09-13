"""Drop configuration and a short real two-rank training run without storage."""
import json
import os
from types import SimpleNamespace

import pytest
import torch

from dmi_megatron_integration.startup import resolve_megatron_dmi_config


def test_drop_settings_are_independent_of_debug_and_cli_overrides_environment():
    env = {"DMI_STORAGE_BACKEND": "drop", "DMI_DROP_BASE_FOLDER": "/tmp/drop",
           "DMI_TIMING_ENABLED": "1", "DMI_RING_METRICS_ENABLED": "0",
           "DMI_D2H_WINDOW_DEBUG": "1"}
    cfg = resolve_megatron_dmi_config(environ=env)
    assert cfg.storage_backend == "drop"
    assert cfg.drop.base_folder == "/tmp/drop"
    assert cfg.drop.timing_enabled and not cfg.drop.ring_metrics_enabled
    cfg = resolve_megatron_dmi_config(SimpleNamespace(
        dmi_timing_enabled=False, dmi_ring_metrics_enabled=True), environ=env)
    assert not cfg.drop.timing_enabled and cfg.drop.ring_metrics_enabled
    cfg = resolve_megatron_dmi_config(environ={"DMI_D2H_WINDOW_DEBUG": "1"})
    assert not cfg.drop.timing_enabled and not cfg.drop.ring_metrics_enabled


@pytest.mark.parametrize("backend", ["auto", "native", "drop"])
@pytest.mark.parametrize("timing,ring", [(False, False), (True, False), (False, True), (True, True)])
def test_builder_warns_for_conflicting_measurement_flags(backend, timing, ring, monkeypatch, capsys):
    from dmi_megatron_integration import startup

    # Exercise CLI/environment resolution and the real config builder without
    # constructing a GPU ring or connecting a database.
    config = resolve_megatron_dmi_config(environ={
        "DMI_STORAGE_BACKEND": backend,
        "DMI_DROP_BASE_FOLDER": "/unused",
        "DMI_TIMING_ENABLED": str(int(timing)),
        "DMI_RING_METRICS_ENABLED": str(int(ring)),
    })
    monkeypatch.setattr(startup, "MonitoringEngine", lambda **kwargs: kwargs)
    engine, host = startup._build_engine(config, "warning-test", None)
    captured = capsys.readouterr()
    assert captured.out == ""
    if backend != "drop" and (timing or ring):
        assert captured.err.count("[DMI] WARNING:") == 1
        assert f"storage_backend={backend!r}" in captured.err
        assert ("timing_enabled" in captured.err) == timing
        assert ("ring_metrics_enabled" in captured.err) == ring
    else:
        assert captured.err == ""
    assert host is None
    assert not engine["ring_config"].ring_metrics_enabled
    if backend == "auto" and not timing and not ring:
        assert engine["config"] is None


@pytest.mark.gpu
@pytest.mark.parametrize("measurements", [False, True])
def test_real_two_rank_drop_training(tmp_path, measurements):
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        pytest.skip("two CUDA devices required")
    from tests.test_megatron_real_training_e2e import (
        _tiny_megatron_router_summary_cmd, _run_megatron_cmd,
    )
    output = tmp_path / "records"
    flags = ["--dmi-storage-backend", "drop", "--dmi-drop-base-folder", str(output)]
    if measurements:
        flags += ["--dmi-timing-enabled", "--dmi-ring-metrics-enabled"]
    cmd = _tiny_megatron_router_summary_cmd(
        model_id="drop-real", train_iters=2, micro_batch_size=1,
        global_batch_size=4, nproc_per_node=2, pp_size=2, extra_args=flags,
    )
    env = {key: value for key, value in os.environ.items() if not key.startswith("DMI_")}
    env["DMI_DB_HOST"] = "unreachable.invalid"  # Explicit drop must never connect.
    env["OMP_NUM_THREADS"] = "1"
    _run_megatron_cmd(cmd, env=env, log_path=tmp_path / "train.log")
    for rank in (0, 1):
        events = [json.loads(line) for line in
            (output / f"rank_{rank:05d}" / "events.jsonl").read_text().splitlines()]
        assert all(event["rank"] == rank for event in events)
        records = [event for event in events if event["type"] == "record"]
        assert records
        rows = [row for event in records for row in event["rows"]]
        assert all("bytes" not in row for row in rows)
        assert {1, 2} <= {row["global_batch_id"] for row in rows if row["phase"] == "train"}
        starts = [event for event in events if event["type"] == "iteration_start"]
        ends = [event for event in events if event["type"] == "iteration_end"]
        rings = [event for event in events if event["type"] == "ring_metrics"]
        if measurements:
            assert [event["iteration"] for event in starts] == [1, 2]
            assert [event["iteration"] for event in ends] == [1, 2]
            assert starts[0]["time_ns"] == 0
            assert starts[1]["time_ns"] >= ends[0]["time_ns"] > 0
            assert all(event["duration_ns"] > 0 for event in ends)
            assert any("cpu_arrival_ns" in event for event in records)
            assert len(rings) == 2
            assert all(event["payload_high_water_bytes"] > 0 for event in rings)
        else:
            assert not starts and not ends and not rings
            assert all("cpu_arrival_ns" not in event for event in records)
