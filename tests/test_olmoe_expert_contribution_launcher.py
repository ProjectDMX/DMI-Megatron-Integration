import hashlib
import json
import sys
from pathlib import Path

import pytest

from examples import run_olmoe_expert_contribution as launcher


def _base_args() -> list[str]:
    return [
        "--use-mcore-models",
        "--mock-data",
        "--micro-batch-size",
        "99",
        "--global-batch-size",
        "99",
        "--bf16",
        "--moe-token-dispatcher-type",
        "allgather",
        "--load",
        "/checkpoint",
        "--ckpt-format",
        "torch_dist",
    ]


def _option(command: list[str], flag: str) -> str:
    index = command.index(flag)
    return command[index + 1]


def test_fixed_command_uses_real_data_ep2_and_four_payload_contract(tmp_path: Path) -> None:
    command = launcher._build_megatron_args(
        hf_dir=tmp_path / "hf",
        megatron_dir=tmp_path / "megatron",
        data_prefix=tmp_path / "openwebmath",
        model_id="olmoe-real",
        database="olmoe_db",
        table="olmoe_table",
        base_argv=_base_args(),
    )

    assert "--mock-data" not in command
    assert _option(command, "--data-path") == str(tmp_path / "openwebmath")
    assert _option(command, "--split") == "900,50,50"
    assert _option(command, "--expert-model-parallel-size") == "2"
    assert _option(command, "--expert-tensor-parallel-size") == "1"
    assert _option(command, "--context-parallel-size") == "1"
    assert _option(command, "--train-iters") == "10"
    assert _option(command, "--dmi-hook-selection") == launcher.HOOK_SELECTION
    assert _option(command, "--dmi-no-recompute-hook") == launcher.HOOK_SELECTION
    assert launcher.HOOK_SELECTORS == (
        "router-topk",
        "moe-inverse-map",
        "moe-packed-weighted-output",
    )
    assert launcher.PAYLOAD_ACT_NAMES == (
        "router_topk_expert_ids",
        "router_topk_weights",
        "moe_inverse_map",
        "moe_packed_weighted_output",
    )


def test_fixed_command_uses_recompute_cpu_optimizer_and_final_aux_coefficient(
    tmp_path: Path,
) -> None:
    command = launcher._build_megatron_args(
        hf_dir=tmp_path / "hf",
        megatron_dir=tmp_path / "megatron",
        data_prefix=tmp_path / "openwebmath",
        model_id="olmoe-real",
        database="olmoe_db",
        table="olmoe_table",
        base_argv=_base_args(),
    )

    assert _option(command, "--moe-aux-loss-coeff") == "0.01"
    assert _option(command, "--recompute-granularity") == "full"
    assert _option(command, "--recompute-method") == "uniform"
    assert _option(command, "--recompute-num-layers") == "1"
    assert _option(command, "--optimizer-offload-fraction") == "1.0"
    assert _option(command, "--main-grads-dtype") == "bf16"
    assert _option(command, "--main-params-dtype") == "fp16"
    assert "--optimizer-cpu-offload" in command
    assert "--no-pin-cpu-grads" in command
    assert "--no-pin-cpu-params" in command
    assert "--fine-grained-activation-offloading" not in command
    assert _option(command, "--cuda-graph-impl") == "none"
    assert "--save" not in command


def test_topology_manifest_is_an_explicit_environment_input(tmp_path: Path) -> None:
    inherited, explicit = launcher._training_environment(tmp_path / "ep_topology.json")

    assert explicit["CUDA_VISIBLE_DEVICES"] == "1,2"
    assert explicit["DMI_TOPOLOGY_MANIFEST_PATH"] == str(tmp_path / "ep_topology.json")
    assert explicit["DMI_ENABLE"] == "1"
    assert inherited.items() >= explicit.items()
    assert not {
        "CUDA_HOME",
        "CUDA_PATH",
        "CURAND_HOME",
        "LD_LIBRARY_PATH",
        "LD_PRELOAD",
        "NVRTC_HOME",
        "PATH",
        "PYTHON_EXEC",
    } & explicit.keys()


def test_dry_run_uses_current_python_without_creating_a_launcher(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(launcher, "_build_megatron_args", lambda **_kwargs: [])
    monkeypatch.setattr(launcher, "_manifest_inputs", lambda **_kwargs: (None, None, {}))

    assert launcher.main(
        [
            "--dry-run",
            "--hf-dir",
            str(tmp_path / "hf"),
            "--megatron-dir",
            str(tmp_path / "megatron"),
            "--data-prefix",
            str(tmp_path / "data"),
            "--output-root",
            str(tmp_path / "runs"),
            "--run-name",
            "portable",
        ]
    ) == 0

    run_dir = tmp_path / "runs" / "portable"
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["command"][0] == sys.executable
    assert "cuda12_python_launcher" not in manifest
    assert not (run_dir / "cuda12_python").exists()


@pytest.mark.parametrize("missing_flag", ["--hf-dir", "--megatron-dir", "--output-root"])
def test_machine_specific_paths_must_be_configured(missing_flag: str) -> None:
    configured_paths = {
        "--hf-dir": "/configured/hf",
        "--megatron-dir": "/configured/megatron",
        "--output-root": "/configured/runs",
    }
    argv = [
        item
        for flag, value in configured_paths.items()
        if flag != missing_flag
        for item in (flag, value)
    ]

    with pytest.raises(SystemExit, match="2"):
        launcher.parse_args(argv)


def _write_source_manifest(hf_dir: Path) -> tuple[Path, dict[str, object]]:
    hf_dir.mkdir()
    source = {
        "repository": launcher.EXPECTED_HF_REPOSITORY,
        "revision": launcher.EXPECTED_HF_REVISION,
    }
    path = hf_dir / "source_manifest.json"
    path.write_text(json.dumps(source, sort_keys=True) + "\n", encoding="utf-8")
    return path, source


def test_conversion_binding_accepts_exact_downloaded_source(tmp_path: Path) -> None:
    hf_dir = tmp_path / "hf"
    source_path, source = _write_source_manifest(hf_dir)
    digest = hashlib.sha256(source_path.read_bytes()).hexdigest()
    conversion = {
        "iteration": 1,
        "ckpt_format": "torch_dist",
        "hf_dir": str(hf_dir.resolve()),
        "hf_repository": launcher.EXPECTED_HF_REPOSITORY,
        "hf_revision": launcher.EXPECTED_HF_REVISION,
        "hf_source_manifest_sha256": digest,
    }

    launcher._validate_conversion_binding(
        hf_dir=hf_dir,
        source_manifest_path=source_path,
        source_manifest=source,
        conversion_manifest=conversion,
    )


def test_conversion_binding_rejects_different_source_hash(tmp_path: Path) -> None:
    hf_dir = tmp_path / "hf"
    source_path, source = _write_source_manifest(hf_dir)
    conversion = {
        "iteration": 1,
        "ckpt_format": "torch_dist",
        "hf_dir": str(hf_dir.resolve()),
        "hf_repository": launcher.EXPECTED_HF_REPOSITORY,
        "hf_revision": launcher.EXPECTED_HF_REVISION,
        "hf_source_manifest_sha256": "0" * 64,
    }

    try:
        launcher._validate_conversion_binding(
            hf_dir=hf_dir,
            source_manifest_path=source_path,
            source_manifest=source,
            conversion_manifest=conversion,
        )
    except ValueError as error:
        assert "hf_source_manifest_sha256" in str(error)
    else:
        raise AssertionError("A converted checkpoint from a different source was accepted")
