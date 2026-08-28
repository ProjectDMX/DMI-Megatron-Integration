from __future__ import annotations

import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

MEGATRON_ROOT = Path(__file__).resolve().parents[1] / "third_party" / "megatron-lm"
sys.path.insert(0, str(MEGATRON_ROOT))

import dmi_megatron_integration.exact_resume as exact_resume
from dmi_megatron_integration.exact_resume import (
    DMIExactResumeManager,
    capture_dmi_exact_loaded_rng_state,
    restore_dmi_exact_loaded_rng_state,
    validate_loaded_resume_state,
)
from dmi_megatron_integration.schedule_runtime import MegatronScheduleRuntime
from megatron.core.rerun_state_machine import (
    RerunDataIterator,
    RerunMode,
    RerunState,
    RerunStateMachine,
)


def _completed_runtime(iteration: int, *, aborted: bool = False):
    runtime = MegatronScheduleRuntime(object())
    runtime.enter_phase(
        "train",
        training_iteration_id_start=1,
        global_batch_id_start=1,
    )
    runtime.begin_logical_iteration(iteration + (1 if aborted else 0))
    runtime.begin_attempt(0)
    runtime.finish_attempt(-1 if aborted else 1)
    runtime.finish_logical_iteration()
    return runtime


def test_capture_state_normalizes_ordinary_and_aborted_boundaries():
    ordinary = _completed_runtime(5)
    ordinary_state = ordinary.exact_resume_state_dict(checkpoint_iteration=5)
    assert ordinary_state["next_training_global_batch_id"] == 6

    aborted = _completed_runtime(5, aborted=True)
    aborted_state = aborted.exact_resume_state_dict(checkpoint_iteration=5)
    assert aborted_state["next_training_global_batch_id"] == 6

    restored = MegatronScheduleRuntime(object())
    restored.load_exact_resume_state_dict(
        aborted_state,
        checkpoint_iteration=5,
    )
    assert restored.global_batch_id == 5
    assert restored.phase == "train"


def test_active_rerun_loads_buffer_before_state_dispatch():
    machine = RerunStateMachine(mode=RerunMode.VALIDATE_RESULTS)
    machine.state = RerunState.WILL_RERUN_FROM_CHECKPOINT
    machine.data_iterator_checkpoints = [
        {
            "saved_microbatches": ["checkpointed-batch"],
            "replaying": True,
            "replay_pos": 0,
        }
    ]
    iterator = RerunDataIterator(iter(["underlying-next-batch"]))

    assert machine.should_run_forward_backward(iterator) is True
    assert next(iterator) == "checkpointed-batch"
    assert machine.data_iterator_checkpoints is None


def _dataset_state() -> dict:
    return {
        "schema_version": 1,
        "configuration": {
            "size": 1536,
            "window_iters": 2,
            "total_iters": 12,
            "global_batch_size": 128,
            "total_windows": 6,
            "samples_per_window": 256,
            "random_seed": 42,
        },
        "source_contract": [{"dataset_id": 0}],
        "selector": {"last_window_id": 3},
        "current_window_id": 3,
        "weights": [0.25, 0.25, 0.25, 0.25],
        "current_window": {"window_id": 3},
    }


def _controller_state() -> dict:
    return {
        "schema_version": 1,
        "source_run_id": "child-run",
        "source_model_id": "child-model",
        "configuration": {},
        "checkpoint_iteration": 5,
        "processed_through_iteration": 5,
        "installed_window_id": 3,
        "controller": {
            "previous_weights": [0.25, 0.25, 0.25, 0.25],
        },
        "active_window_id": 3,
        "active_states": {},
        "decision_store": {
            "schema_version": 1,
            "pending_decision": None,
        },
        "cumulative_decision_count": 0,
        "cumulative_update_count": 0,
    }


class _FakeHandle:
    def __init__(self, runtime):
        self.model_id = "child-model"
        self.schedule_runtime = runtime
        self.config = SimpleNamespace(
            db_database="raw_db",
            clickhouse_table="raw_table",
        )
        self.flushes = []

    def flush_and_wait(self, timeout_s):
        self.flushes.append(timeout_s)


def _args():
    return SimpleNamespace(
        dmi_exact_checkpoint_timeout_s=600.0,
        dmi_dynamic_mixture_run_id="child-run",
        dmi_dynamic_mixture_control_url="http://controller",
        dmi_enable=True,
        deterministic_mode=True,
        dataloader_type="single",
        data_parallel_size=1,
        num_workers=0,
        rampup_batch_size=None,
        async_save=False,
        dmi_db_host="localhost",
        dmi_exact_processed_database=None,
        dmi_exact_control_database=None,
        global_batch_size=128,
        consumed_train_samples=640,
        consumed_valid_samples=0,
        iteration=0,
        load=None,
        _dmi_exact_execution_contract=exact_resume._EXECUTION_CONTRACT.copy(),
    )


def test_manager_prepares_one_consistent_resume_state(monkeypatch):
    for name, value in exact_resume._DETERMINISTIC_ENVIRONMENT.items():
        monkeypatch.setenv(name, value)
    runtime = _completed_runtime(5)
    handle = _FakeHandle(runtime)
    dataset_state = _dataset_state()
    controller_state = _controller_state()
    monkeypatch.setattr(
        exact_resume,
        "get_active_dynamic_train_dataset",
        lambda: SimpleNamespace(state_dict=lambda: dataset_state),
    )
    monkeypatch.setattr(
        exact_resume,
        "_post_json",
        lambda *args, **kwargs: {
            "state": controller_state,
            "sha256": exact_resume._sha256(controller_state),
        },
    )
    manager = DMIExactResumeManager(
        args=_args(),
        handle=handle,
        printer=lambda _: None,
        loaded_state=None,
    )

    state = manager.prepare_checkpoint(5)

    assert state["segment_lineage"] == [
        {
            "run_id": "child-run",
            "model_id": "child-model",
            "valid_start_iteration": 1,
            "valid_end_iteration": 5,
        }
    ]
    assert state["durable_flush_state"]["durable_through_iteration"] == 5
    assert state["dynamic_dataset_state_sha256"] == exact_resume._sha256(
        dataset_state
    )
    assert state["execution_contract"] == exact_resume._EXECUTION_CONTRACT
    assert handle.flushes


def test_configure_exact_execution_disables_adaptive_torchscript(monkeypatch):
    calls = []
    monkeypatch.setattr(
        torch._C,
        "_jit_set_profiling_executor",
        lambda enabled: calls.append(("executor", enabled)),
    )
    monkeypatch.setattr(
        torch._C,
        "_jit_set_profiling_mode",
        lambda enabled: calls.append(("mode", enabled)),
    )
    args = SimpleNamespace(dmi_exact_resume=True)

    exact_resume.configure_dmi_exact_execution(args)

    assert calls == [("executor", False), ("mode", False)]
    assert args._dmi_exact_execution_contract == exact_resume._EXECUTION_CONTRACT


def test_manager_rejects_missing_deterministic_mode(monkeypatch):
    for name, value in exact_resume._DETERMINISTIC_ENVIRONMENT.items():
        monkeypatch.setenv(name, value)
    args = _args()
    args.deterministic_mode = False

    with pytest.raises(RuntimeError, match="requires --deterministic-mode"):
        DMIExactResumeManager(
            args=args,
            handle=_FakeHandle(_completed_runtime(1)),
            printer=lambda _: None,
            loaded_state=None,
        )


@pytest.mark.parametrize("name", sorted(exact_resume._DETERMINISTIC_ENVIRONMENT))
def test_manager_rejects_incomplete_deterministic_environment(monkeypatch, name):
    for env_name, value in exact_resume._DETERMINISTIC_ENVIRONMENT.items():
        monkeypatch.setenv(env_name, value)
    monkeypatch.delenv(name)

    with pytest.raises(RuntimeError, match=rf"requires {name}="):
        DMIExactResumeManager(
            args=_args(),
            handle=_FakeHandle(_completed_runtime(1)),
            printer=lambda _: None,
            loaded_state=None,
        )


def test_loaded_state_rejects_non_durable_checkpoint():
    dataset_state = _dataset_state()
    controller_state = _controller_state()
    state = {
        "schema_version": 2,
        "checkpoint_iteration": 5,
        "consumed_train_samples": 640,
        "consumed_valid_samples": 0,
        "execution_contract": exact_resume._EXECUTION_CONTRACT.copy(),
        "segment_lineage": [
            {
                "run_id": "parent-run",
                "model_id": "parent-model",
                "valid_start_iteration": 1,
                "valid_end_iteration": 5,
            }
        ],
        "database_identity": {},
        "dataset_contract": {},
        "dynamic_dataset_state": dataset_state,
        "dynamic_dataset_state_sha256": exact_resume._sha256(dataset_state),
        "controller_state": controller_state,
        "controller_state_sha256": exact_resume._sha256(controller_state),
        "capture_state": {},
        "durable_flush_state": {"durable_through_iteration": 4},
    }
    args = SimpleNamespace(
        finetune=False,
        no_load_optim=False,
        no_load_rng=False,
        pretrained_checkpoint=None,
        global_batch_size=128,
        _dmi_exact_execution_contract=exact_resume._EXECUTION_CONTRACT.copy(),
    )

    with pytest.raises(ValueError, match="not durable"):
        validate_loaded_resume_state(
            args,
            state,
            checkpoint_iteration=5,
            release=False,
        )


def test_exact_resume_restores_rng_after_post_load_setup(monkeypatch):
    class _Tracker:
        def __init__(self):
            self.states = {"model-parallel-rng": torch.tensor([31, 32], dtype=torch.uint8)}

        def get_states(self):
            return self.states

        def set_states(self, states):
            self.states = states

    tracker = _Tracker()
    cuda_state = torch.tensor([41, 42], dtype=torch.uint8)
    restored_cuda_state = None

    monkeypatch.setattr(
        "megatron.core.tensor_parallel.get_cuda_rng_tracker",
        lambda: tracker,
    )
    monkeypatch.setattr(
        "megatron.core.tensor_parallel.is_graph_safe_cuda_rng_tracker",
        lambda _: False,
    )
    monkeypatch.setattr(
        "megatron.core.tensor_parallel.convert_cuda_rng_state",
        lambda value, to_graphable=False: value,
    )
    monkeypatch.setattr(torch.cuda, "get_rng_state", lambda: cuda_state.clone())

    def _set_cuda_rng_state(value):
        nonlocal restored_cuda_state
        restored_cuda_state = value.clone()

    monkeypatch.setattr(torch.cuda, "set_rng_state", _set_cuda_rng_state)

    args = SimpleNamespace(
        dmi_exact_resume=True,
        _dmi_loaded_resume_state={"checkpoint_iteration": 5},
    )
    random.seed(7)
    np.random.seed(11)
    torch.manual_seed(13)
    expected_random = random.getstate()
    expected_numpy = np.random.get_state()
    expected_torch = torch.get_rng_state().clone()
    expected_tracker = tracker.states["model-parallel-rng"].clone()

    capture_dmi_exact_loaded_rng_state(args)

    random.seed(101)
    np.random.seed(103)
    torch.manual_seed(107)
    cuda_state.fill_(0)
    tracker.states["model-parallel-rng"].fill_(0)

    restore_dmi_exact_loaded_rng_state(args)

    assert random.getstate() == expected_random
    actual_numpy = np.random.get_state()
    assert actual_numpy[0] == expected_numpy[0]
    assert np.array_equal(actual_numpy[1], expected_numpy[1])
    assert actual_numpy[2:] == expected_numpy[2:]
    assert torch.equal(torch.get_rng_state(), expected_torch)
    assert torch.equal(restored_cuda_state, torch.tensor([41, 42], dtype=torch.uint8))
    assert torch.equal(tracker.states["model-parallel-rng"], expected_tracker)
    assert not hasattr(args, "_dmi_exact_loaded_rng_state")
