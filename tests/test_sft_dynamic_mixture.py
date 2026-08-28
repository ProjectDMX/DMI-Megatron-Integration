import json
import os
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

os.environ.setdefault("FLASHINFER_WORKSPACE_BASE", "/tmp/dmi-flashinfer-tests")

ROOT = Path(__file__).resolve().parents[1]
MEGATRON_ROOT = ROOT / "third_party" / "megatron-lm"
for path in (str(ROOT), str(MEGATRON_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from dmi_megatron_integration.dynamic_mixture import (  # noqa: E402
    DynamicBlendedDataset,
)
from dmi_megatron_integration.dynamic_mixture_primitives import (  # noqa: E402
    DecisionHTTPClient,
    MixtureDecision,
    WindowSelector,
)
from tools.sft_mixture.dynamic_mixture import (  # noqa: E402
    DecisionHTTPServer,
    DecisionStore,
)


def _decision(*, window_id: int, weights=(0.25, 0.75)) -> MixtureDecision:
    return MixtureDecision(
        run_id="run",
        decision_id=window_id - 1,
        decision_type="UPDATE",
        source_window_id=window_id - 1,
        source_window_end_iteration=(window_id - 1) * 2,
        effective_window_id=window_id,
        effective_training_iteration=(window_id - 1) * 2 + 1,
        weights=weights,
        reason="",
        produced_at_unix_ns=1,
    )


def test_window_selector_chunking_matches_one_random_choices_stream():
    selector = WindowSelector(
        seed=42,
        dataset_count=2,
        samples_per_window=4,
        iterations_per_window=2,
        global_batch_size=2,
    )
    first = selector.build(window_id=1, weights=(0.4, 0.6), decision_id=0)
    second = selector.build(window_id=2, weights=(0.4, 0.6), decision_id=1)

    expected = random.Random(42).choices(range(2), weights=(0.4, 0.6), k=8)
    actual = first.dataset_index.tolist() + second.dataset_index.tolist()
    assert actual == expected

    seen = [0, 0]
    expected_source_indices = []
    for dataset_id in expected:
        expected_source_indices.append(seen[dataset_id])
        seen[dataset_id] += 1
    actual_source_indices = (
        first.dataset_sample_index.tolist() + second.dataset_sample_index.tolist()
    )
    assert actual_source_indices == expected_source_indices
    assert second.counters_after == tuple(seen)


def test_http_long_poll_returns_only_the_exact_window():
    store = DecisionStore()
    server = DecisionHTTPServer("127.0.0.1", 0, store)
    server.start()
    host, port = server.address
    client = DecisionHTTPClient(
        f"http://{host}:{port}",
        run_id="run",
        request_timeout_s=0.05,
    )
    try:
        assert client.request_once(2) is None
        decision = _decision(window_id=2)
        store.publish(decision)
        assert client.request_once(2) == decision
        assert client.request_once(3) is None
    finally:
        server.close()


def test_restored_pending_decision_is_available_without_republishing():
    parent_decision = _decision(window_id=2)
    parent = DecisionStore()
    parent.publish(parent_decision)
    state = parent.state_dict(
        run_id="run",
        pending_effective_window_id=2,
    )

    published = []
    child = DecisionStore(on_publish=published.append)
    restored = child.load_state_dict(state, run_id="child")

    assert restored is not None
    assert restored.run_id == "child"
    assert child.get("child", 2) == restored
    assert published == []


def test_dynamic_dataset_applies_decision_only_to_its_future_window(tmp_path):
    class SFTDataset:
        index_split = SimpleNamespace(name="train")
        unique_identifiers = {"kind": "fake"}

        def __init__(self, source: int):
            self.source = source

        def __len__(self):
            return 100

        def __getitem__(self, index):
            return {"source": self.source, "source_sample": int(index)}

    store = DecisionStore()
    server = DecisionHTTPServer("127.0.0.1", 0, store)
    server.start()
    host, port = server.address
    config = SimpleNamespace(
        random_seed=42,
        dmi_dynamic_mixture_run_id="run",
        dmi_dynamic_mixture_control_url=f"http://{host}:{port}",
        dmi_dynamic_mixture_window_iters=2,
        dmi_dynamic_mixture_total_iters=4,
        dmi_dynamic_mixture_global_batch_size=2,
        dmi_dynamic_mixture_num_workers=0,
        dmi_dynamic_mixture_request_timeout_s=0.05,
        dmi_dynamic_mixture_feedback_timeout_s=2.0,
        dmi_dynamic_mixture_audit_dir=str(tmp_path),
    )
    dataset = DynamicBlendedDataset(
        [SFTDataset(0), SFTDataset(1)],
        [0.75, 0.25],
        8,
        config,
    )
    try:
        first_window_sources = [dataset[index]["source"] for index in range(4)]
        expected_first = random.Random(42).choices(
            range(2), weights=(0.75, 0.25), k=4
        )
        assert first_window_sources == expected_first

        decision = _decision(window_id=2, weights=(0.25, 0.75))
        store.publish(decision)
        second_window_sources = [dataset[index]["source"] for index in range(4, 8)]
        rng = random.Random(42)
        rng.choices(range(2), weights=(0.75, 0.25), k=4)
        expected_second = rng.choices(range(2), weights=(0.25, 0.75), k=4)
        assert second_window_sources == expected_second

        dataset.close()
        records = [
            json.loads(line)
            for line in (tmp_path / "dynamic_mixture_rank0.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        windows = [record for record in records if record["kind"] == "selection_window"]
        assert [record["window_id"] for record in windows] == [1, 2]
        assert windows[1]["decision"]["decision_id"] == 1
        assert not any(
            record.get("effective_window_id") == 3
            for record in records
            if record["kind"] == "decision_request"
        )
    finally:
        dataset.close()
        server.close()


def test_dynamic_dataset_audit_is_optional(tmp_path):
    class SFTDataset:
        index_split = SimpleNamespace(name="train")
        unique_identifiers = {"kind": "fake"}

        def __len__(self):
            return 100

        def __getitem__(self, index):
            return {"source_sample": int(index)}

    store = DecisionStore()
    server = DecisionHTTPServer("127.0.0.1", 0, store)
    server.start()
    host, port = server.address
    config = SimpleNamespace(
        random_seed=42,
        dmi_dynamic_mixture_run_id="run",
        dmi_dynamic_mixture_control_url=f"http://{host}:{port}",
        dmi_dynamic_mixture_window_iters=1,
        dmi_dynamic_mixture_total_iters=1,
        dmi_dynamic_mixture_global_batch_size=1,
        dmi_dynamic_mixture_num_workers=0,
        dmi_dynamic_mixture_request_timeout_s=0.05,
        dmi_dynamic_mixture_feedback_timeout_s=2.0,
        dmi_dynamic_mixture_audit_dir=None,
    )
    dataset = DynamicBlendedDataset(
        [SFTDataset(), SFTDataset()],
        [0.5, 0.5],
        1,
        config,
    )
    try:
        assert dataset[0]["source_sample"] == 0
        assert list(tmp_path.iterdir()) == []
    finally:
        dataset.close()
        server.close()


def test_dynamic_dataset_four_workers_reconstruct_identical_future_windows(tmp_path):
    class SFTDataset:
        index_split = SimpleNamespace(name="train")
        unique_identifiers = {"kind": "multiworker-fake"}

        def __init__(self, source):
            self.source = source

        def __len__(self):
            return 100

        def __getitem__(self, index):
            return {"source": self.source, "source_sample": int(index)}

    store = DecisionStore()
    server = DecisionHTTPServer("127.0.0.1", 0, store)
    server.start()
    host, port = server.address
    config = SimpleNamespace(
        random_seed=42,
        dmi_dynamic_mixture_run_id="run",
        dmi_dynamic_mixture_control_url=f"http://{host}:{port}",
        dmi_dynamic_mixture_window_iters=2,
        dmi_dynamic_mixture_total_iters=4,
        dmi_dynamic_mixture_global_batch_size=2,
        dmi_dynamic_mixture_num_workers=4,
        dmi_dynamic_mixture_request_timeout_s=0.05,
        dmi_dynamic_mixture_feedback_timeout_s=2.0,
        dmi_dynamic_mixture_audit_dir=str(tmp_path),
        dmi_dynamic_mixture_resume_state=None,
    )
    dataset = DynamicBlendedDataset(
        [SFTDataset(0), SFTDataset(1)],
        [0.75, 0.25],
        8,
        config,
    )
    store.publish(_decision(window_id=2, weights=(0.25, 0.75)))
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        persistent_workers=False,
    )
    try:
        actual_sources = [int(batch["source"].item()) for batch in loader]
        rng = random.Random(42)
        expected_sources = rng.choices(range(2), weights=(0.75, 0.25), k=4)
        expected_sources += rng.choices(range(2), weights=(0.25, 0.75), k=4)
        assert actual_sources == expected_sources

        records = [
            json.loads(line)
            for line in (tmp_path / "dynamic_mixture_rank0.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        windows = [row for row in records if row["kind"] == "selection_window"]
        assert [row["window_id"] for row in windows] == [1, 2]
        assert windows[1]["decision"]["decision_id"] == 1
        requests = [row for row in records if row["kind"] == "decision_request"]
        assert [row["effective_window_id"] for row in requests] == [2]
    finally:
        dataset.close()
        server.close()


def test_dynamic_dataset_workers_reject_checkpoint_restore(tmp_path):
    class SFTDataset:
        index_split = SimpleNamespace(name="train")
        unique_identifiers = {"kind": "restore-fake"}

        def __len__(self):
            return 100

        def __getitem__(self, index):
            return {"source_sample": int(index)}

    config = SimpleNamespace(
        random_seed=42,
        dmi_dynamic_mixture_run_id="run",
        dmi_dynamic_mixture_control_url="http://127.0.0.1:1",
        dmi_dynamic_mixture_window_iters=2,
        dmi_dynamic_mixture_total_iters=4,
        dmi_dynamic_mixture_global_batch_size=2,
        dmi_dynamic_mixture_num_workers=4,
        dmi_dynamic_mixture_request_timeout_s=0.05,
        dmi_dynamic_mixture_feedback_timeout_s=2.0,
        dmi_dynamic_mixture_audit_dir=str(tmp_path),
        dmi_dynamic_mixture_resume_state={"schema_version": 1},
    )
    with pytest.raises(ValueError, match="checkpoint restore requires num_workers=0"):
        DynamicBlendedDataset(
            [SFTDataset(), SFTDataset()],
            [0.5, 0.5],
            8,
            config,
        )
