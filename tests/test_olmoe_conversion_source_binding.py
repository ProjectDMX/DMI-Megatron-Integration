import hashlib
import json
import sys
from pathlib import Path


SCRIPTS = (
    Path(__file__).resolve().parents[1]
    / "tools"
    / "grokking"
    / "scripts"
)
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import convert_olmoe_hf_to_megatron_native as converter


def test_converter_binds_output_to_required_source_manifest(tmp_path: Path) -> None:
    hf_dir = tmp_path / "hf"
    hf_dir.mkdir()
    source_path = hf_dir / "source_manifest.json"
    source_path.write_text(
        json.dumps(
            {
                "repository": "allenai/OLMoE-1B-7B-0924",
                "revision": "immutable-commit",
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    binding = converter.load_hf_source_binding(hf_dir)

    assert binding == {
        "hf_dir": str(hf_dir.resolve()),
        "hf_source_manifest_path": str(source_path.resolve()),
        "hf_source_manifest_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "hf_repository": "allenai/OLMoE-1B-7B-0924",
        "hf_revision": "immutable-commit",
    }


def test_converter_rejects_hf_directory_without_source_manifest(tmp_path: Path) -> None:
    hf_dir = tmp_path / "hf"
    hf_dir.mkdir()

    try:
        converter.load_hf_source_binding(hf_dir)
    except FileNotFoundError as error:
        assert "source_manifest.json" in str(error)
    else:
        raise AssertionError("Conversion accepted an HF directory without provenance")
