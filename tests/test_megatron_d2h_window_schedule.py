"""Exercise the real non-interleaved schedule with CPU-only P2P/compute fakes."""

from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

from megatron.core.pipeline_parallel import schedules
from dmi_megatron_integration import schedule_runtime as runtime_module
from dmi_megatron_integration.schedule_runtime import (
    MegatronScheduleRuntime,
    dmi_prepare_d2h_windows,
    set_active_megatron_schedule_runtime,
)


class FakeRecords:
    def __init__(self, events):
        self.events = events
        self.counter = 0

    def define_d2h_window_pattern(self, **kwargs):
        self.events.append(("define", kwargs))
        self.counter = 0
        return True

    def advance_boundary(self):
        self.counter += 1
        self.events.append(("open" if self.counter % 2 else "close",))


class FakeP2P:
    def __init__(self, p, r, config, events):
        self.total_stages = p
        self.current_stage = r
        self.is_pp_first_stage = r == 0
        self.is_pp_last_stage = r == p - 1
        self.config = config
        self.events = events

    def _call(self, name, noop):
        self.events.append(("noop" if noop else name,))
        return torch.zeros(1)

    def recv_forward(self, shapes, first):
        return self._call("recv_forward", first)

    def recv_backward(self, shapes, last):
        return self._call("recv_backward", last)

    def send_forward(self, tensor, last):
        self._call("send_forward", last)

    def send_backward(self, tensor, first):
        self._call("send_backward", first)

    def send_forward_recv_backward(self, tensor, shapes, last):
        return self._call("send_forward_recv_backward", last)

    def send_backward_recv_forward(self, tensor, shapes, first):
        return self._call("send_backward_recv_forward", first)


@pytest.fixture
def schedule_harness(monkeypatch):
    events = []
    records = FakeRecords(events)
    runtime = MegatronScheduleRuntime(SimpleNamespace())
    runtime.configure_d2h_windows(enabled=True)
    runtime.adaptor = SimpleNamespace(record_runtime=records)
    set_active_megatron_schedule_runtime(runtime)

    @contextmanager
    def no_sync():
        events.append(("disable_grad_sync",))
        yield
        events.append(("enable_grad_sync",))

    config = SimpleNamespace(
        overlap_p2p_comm=False, timers=None, no_sync_func=no_sync,
        finalize_model_grads_func=lambda *a, **kw: events.append(("finalize",)),
        grad_sync_func=lambda *a: events.append(("grad_sync",)),
        num_microbatches_with_partial_activation_checkpoints=None,
        deallocate_pipeline_outputs=True, calculate_per_token_loss=False,
    )
    monkeypatch.setattr(schedules, "get_model_config", lambda model: config)
    monkeypatch.setattr(schedules, "get_tensor_shapes", lambda **kw: (1, 1, 1))
    monkeypatch.setattr(schedules, "clear_embedding_activation_buffer", lambda *a: None)
    monkeypatch.setattr(schedules, "finish_embedding_wgrad_compute", lambda *a: None)
    monkeypatch.setattr(schedules, "dmi_begin_iteration", lambda *a, **kw: None)
    monkeypatch.setattr(schedules, "dmi_end_iteration", lambda: None)
    monkeypatch.setattr(
        schedules, "dmi_set_current_event",
        lambda *a: events.append(("metadata", *a)),
    )
    monkeypatch.setattr(schedules, "dmi_enter_current_scope", lambda: None)
    monkeypatch.setattr(
        schedules, "deallocate_output_tensor",
        lambda *a: events.append(("deallocate",)),
    )

    def forward(*a, **kw):
        events.append(("forward",))
        return torch.zeros(1), 0

    def backward(*a, **kw):
        events.append(("backward",))
        return torch.zeros(1)

    monkeypatch.setattr(schedules, "forward_step", forward)
    monkeypatch.setattr(schedules, "backward_step", backward)
    real_zeros = torch.zeros
    monkeypatch.setattr(
        schedules.torch, "zeros",
        lambda *a, **kw: real_zeros(*a, **{**kw, "device": "cpu"}),
    )

    def run(p, r, m, *, enabled=True, forward_only=False, multimodule=False):
        runtime.configure_d2h_windows(enabled=enabled)
        monkeypatch.setattr(schedules.parallel_state, "get_pipeline_model_parallel_world_size", lambda: p)
        monkeypatch.setattr(schedules.parallel_state, "get_pipeline_model_parallel_rank", lambda: r)
        func = schedules.forward_backward_pipelining_without_interleaving
        if not forward_only:
            dmi_prepare_d2h_windows(func, m)
        pg = schedules.ProcessGroupCollection(tp=None, cp=SimpleNamespace(size=lambda: 1))
        if multimodule:
            pg = object.__new__(schedules.MultiModuleProcessGroupCollection)
        func(
            forward_step_func=None, data_iterator=iter(()),
            model=SimpleNamespace(parameters=lambda: ()),
            num_microbatches=m, seq_length=1, micro_batch_size=1,
            forward_only=forward_only,
            p2p_communicator=FakeP2P(p, r, config, events), pg_collection=pg,
        )

    yield run, events, runtime, records
    set_active_megatron_schedule_runtime(None)


@pytest.mark.parametrize("p,r", [(p, r) for p in (2, 3, 4) for r in range(p)])
@pytest.mark.parametrize("m", [1, 2, 5])
def test_non_interleaved_boundary_order_and_period(schedule_harness, p, r, m):
    run, events, runtime, records = schedule_harness
    run(p, r, m)
    assert events[0][0] == "define"
    definition = events[0][1]
    w = m if r in (0, p - 1) else 2 * m
    assert definition["period"] == 2 * w
    assert definition["windows"] == tuple((2 * i + 1, 2 * i + 2) for i in range(w))

    # A second iteration reuses the pattern and must produce another full period.
    run(p, r, m)
    assert sum(e[0] == "define" for e in events) == 1
    assert records.counter == 4 * w
    assert sum(e[0] == "forward" for e in events) == 2 * m
    assert sum(e[0] == "backward" for e in events) == 2 * m
    open_window = False
    starts = {"recv_forward", "recv_backward", "send_forward_recv_backward",
              "send_backward_recv_forward"}
    for i, event in enumerate(events):
        name = event[0]
        if name == "open":
            assert not open_window
            assert events[i + 1][0] in starts
            open_window = True
        elif name in starts:
            assert open_window and events[i - 1][0] == "open"
        elif name == "close":
            assert open_window
            # In particular, CLOSE is not placed at P2P return, before metadata.
            assert events[i - 1][0] == "metadata"
            assert events[i + 1][0] in ("forward", "backward")
            open_window = False
        elif name in ("forward", "backward", "send_forward", "send_backward", "grad_sync", "finalize"):
            assert not open_window, (p, r, m, i, event)
    assert not open_window


@pytest.mark.parametrize("forward_only,enabled", [(True, True), (False, False)])
def test_no_markers_in_forward_only_or_disabled_run(schedule_harness, forward_only, enabled):
    run, events, _, _ = schedule_harness
    run(2, 0, 2, enabled=enabled, forward_only=forward_only)
    assert not any(e[0] in ("open", "close", "define") for e in events)


def test_multi_module_rejected_with_windows_active(schedule_harness):
    run, events, _, _ = schedule_harness
    with pytest.raises(NotImplementedError, match="multi-module"):
        run(2, 0, 2, multimodule=True)
    assert not any(e[0] in ("forward", "backward", "open") for e in events)


def test_interleaved_opt_in_warns_once_without_defining(schedule_harness, capsys):
    _, events, runtime, _ = schedule_harness
    for _ in range(3):
        dmi_prepare_d2h_windows(schedules.forward_backward_pipelining_with_interleaving, 2)
    assert events == []
    assert not runtime.d2h_windows_active
    assert capsys.readouterr().out.count("unsupported for interleaved/VPP") == 1


def test_interleaved_disabled_is_silent(schedule_harness, capsys):
    _, events, runtime, _ = schedule_harness
    runtime.configure_d2h_windows(enabled=False)
    dmi_prepare_d2h_windows(schedules.forward_backward_pipelining_with_interleaving, 2)
    assert events == []
    assert capsys.readouterr().out == ""


def test_recurring_window_cli_defaults_and_resolution():
    from argparse import ArgumentParser
    from megatron.training.arguments import _add_dmi_args
    from dmi_megatron_integration.startup import resolve_megatron_dmi_config

    parser = _add_dmi_args(ArgumentParser())
    absent = parser.parse_args([])
    assert absent.dmi_recurring_d2h_windows is None
    assert absent.dmi_d2h_window_debug is None
    env = {"DMI_RECURRING_D2H_WINDOWS": "true", "DMI_D2H_WINDOW_DEBUG": "true"}
    cfg = resolve_megatron_dmi_config(absent, environ=env)
    assert cfg.recurring_d2h_windows_enabled and cfg.d2h_window_debug
    args = parser.parse_args([
        "--dmi-recurring-d2h-windows", "--dmi-d2h-window-debug",
        "--dmi-d2h-window-timing-revalidation-retry-interval-occurrences", "5",
        "--dmi-d2h-window-minimum-record-probe-retry-interval-occurrences", "6",
        "--dmi-d2h-window-capacity-flush-fallback-threshold", "7",
        "--dmi-d2h-window-capacity-flush-count-reset-interval-periods", "8",
    ])
    cfg = resolve_megatron_dmi_config(args, environ={})
    assert cfg.recurring_d2h_windows_enabled and cfg.d2h_window_debug
    assert cfg.d2h_window_timing_revalidation_retry_interval_occurrences == 5
    assert cfg.d2h_window_minimum_record_probe_retry_interval_occurrences == 6
    assert cfg.d2h_window_capacity_flush_fallback_threshold == 7
    assert cfg.d2h_window_capacity_flush_count_reset_interval_periods == 8


@pytest.mark.parametrize("next_m", [2, 3])
def test_training_loop_prepares_before_first_call_and_retry(schedule_harness, monkeypatch, next_m):
    from megatron.training import training

    _, events, runtime, _ = schedule_harness
    attempt = -1

    def should_run(data):
        nonlocal attempt
        attempt += 1
        return attempt < 2

    def schedule(**kwargs):
        events.append(("schedule", kwargs["num_microbatches"]))
        return []

    def begin_attempt(value):
        runtime._active_attempt_id = value
        events.append(("begin_attempt", value))

    monkeypatch.setattr(schedules, "forward_backward_pipelining_without_interleaving", schedule)
    monkeypatch.setattr(schedules.parallel_state, "get_pipeline_model_parallel_world_size", lambda: 2)
    monkeypatch.setattr(schedules.parallel_state, "get_pipeline_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(runtime_module, "dmi_begin_attempt", begin_attempt)
    monkeypatch.setattr(runtime_module, "dmi_finish_attempt", lambda status: events.append(("finish_attempt", status)))
    monkeypatch.setattr(training, "get_args", lambda: SimpleNamespace(
        save_dgrads_interval=None, save_wgrads_interval=None,
        reuse_grad_buf_for_mxfp8_param_ag=False,
        seq_length=1, micro_batch_size=1, decoder_seq_length=None,
    ))
    monkeypatch.setattr(training, "get_timers", lambda: None)
    monkeypatch.setattr(training, "get_rerun_state_machine", lambda: SimpleNamespace(
        should_run_forward_backward=should_run,
        # Exit after exercising both attempts; no optimizer/GPU work is needed.
        should_checkpoint_and_exit=lambda: (False, True, 0),
    ))
    monkeypatch.setattr(training, "get_num_microbatches", lambda: 2 if attempt == 0 else next_m)
    monkeypatch.setattr(training, "has_nvidia_modelopt", False)
    training._train_step_impl(
        None, iter(()), [SimpleNamespace(zero_grad_buffer=lambda: None)],
        SimpleNamespace(zero_grad=lambda: None), None, None, schedule,
        iteration=0, dmi_handle=object(),
    )
    expected = ["begin_attempt", "define", "schedule", "finish_attempt", "begin_attempt"]
    if next_m != 2:
        expected.append("define")
    expected.extend(["schedule", "finish_attempt"])
    assert [e[0] for e in events] == expected
    assert [e[1] for e in events if e[0] == "schedule"] == [2, next_m]
