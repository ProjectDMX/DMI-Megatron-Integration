"""Online ClickHouse processor and HTTP mixture-decision service."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import queue
import re
import threading
import time
from collections import Counter
from typing import Any, Mapping, Sequence

from tools.sft_mixture.controller import (
    LOSS_SLOPE_CONTROLLER_MODE,
    LossSlopeDomainWindowState,
    LossSlopeIterationMetric,
    LossSlopeMixtureController,
    LossSlopeSampleObservation,
    LossSlopeWindowIndicator,
)
from tools.sft_mixture.dynamic_mixture import (
    DecisionHTTPServer,
    DecisionStore,
    MixtureDecision,
)


LOSS_ACT = "lm_per_sample_loss"
LOSS_COUNT_ACT = "lm_per_sample_loss_token_count"


class IncompleteIterationError(RuntimeError):
    pass


_AUDIT_STOP = object()


class _AsyncClickHouseAuditWriter:
    """Own one ClickHouse client and persist queued audit rows off request paths."""

    def __init__(self, client_args: dict[str, object]) -> None:
        self._client_args = dict(client_args)
        self._queue: queue.Queue[
            tuple[str, list[tuple[object, ...]]] | object
        ] = queue.Queue()
        self._started = threading.Event()
        self._closed = False
        self._failure: BaseException | None = None
        self._thread = threading.Thread(
            target=self._run,
            name="dmi-mixture-audit-writer",
            daemon=True,
        )
        self._thread.start()
        self._started.wait()
        self._raise_if_failed()

    def submit(self, query: str, rows: list[tuple[object, ...]]) -> None:
        if self._closed:
            raise RuntimeError("mixture audit writer is closed")
        self._raise_if_failed()
        self._queue.put_nowait((str(query), rows))

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._queue.put_nowait(_AUDIT_STOP)
        self._thread.join()
        self._raise_if_failed()

    def _raise_if_failed(self) -> None:
        if self._failure is not None:
            raise RuntimeError("mixture audit writer failed") from self._failure

    def _run(self) -> None:
        client = None
        try:
            from clickhouse_driver import Client

            client = Client(**self._client_args)
            self._started.set()
            while True:
                item = self._queue.get()
                if item is _AUDIT_STOP:
                    return
                query, rows = item
                client.execute(query, rows)
        except BaseException as error:
            self._failure = error
            self._started.set()
        finally:
            if client is not None:
                client.disconnect()


def _ident(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value):
        raise ValueError(f"invalid ClickHouse identifier: {value!r}")
    return value


def _q(value: str) -> str:
    return f"`{_ident(value)}`"


def _decode(value: Any) -> Any:
    if isinstance(value, memoryview):
        value = value.tobytes()
    if isinstance(value, bytearray):
        value = bytes(value)
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="strict")
    return value


class ClickHouseControlStorage:
    def __init__(
        self,
        *,
        host: str,
        port: int,
        user: str,
        password: str,
        raw_database: str,
        raw_table: str,
        processed_database: str | None,
        control_database: str | None,
        run_id: str,
        model_id: str,
        enable_audit: bool = False,
        controller_mode: str = LOSS_SLOPE_CONTROLLER_MODE,
        loss_slope_eta: float = 500.0,
        loss_slope_uniform_smoothing: float = 0.5,
    ) -> None:
        from clickhouse_driver import Client

        self.raw_database = _ident(raw_database)
        self.raw_table = _ident(raw_table)
        self.enable_audit = bool(enable_audit)
        if self.enable_audit and (not processed_database or not control_database):
            raise ValueError(
                "optional auditing requires processed_database and control_database"
            )
        self.processed_database = (
            None if processed_database is None else _ident(processed_database)
        )
        self.control_database = (
            None if control_database is None else _ident(control_database)
        )
        self.run_id = str(run_id)
        self.model_id = str(model_id)
        if controller_mode != LOSS_SLOPE_CONTROLLER_MODE:
            raise ValueError(f"unsupported controller mode: {controller_mode!r}")
        self.controller_mode = LOSS_SLOPE_CONTROLLER_MODE
        self.loss_slope_eta = float(loss_slope_eta)
        self.loss_slope_uniform_smoothing = float(loss_slope_uniform_smoothing)
        self._client_args = {
            "host": host,
            "port": int(port),
            "user": user,
            "password": password,
            "settings": {"strings_as_bytes": 1},
        }
        self._audit_client_args = {
            "host": host,
            "port": int(port),
            "user": user,
            "password": password,
        }
        self._processor_client = Client(
            **self._client_args,
        )
        self._audit_writer: _AsyncClickHouseAuditWriter | None = None

    @property
    def audit_enabled(self) -> bool:
        return self.enable_audit

    def initialize_output(self) -> None:
        if not self.enable_audit:
            return
        assert self.processed_database is not None
        assert self.control_database is not None
        self._processor_client.execute(
            f"CREATE DATABASE IF NOT EXISTS {_q(self.processed_database)}"
        )
        self._processor_client.execute(
            f"CREATE DATABASE IF NOT EXISTS {_q(self.control_database)}"
        )
        self._initialize_loss_slope_output()

    def _initialize_loss_slope_output(self) -> None:
        assert self.processed_database is not None
        assert self.control_database is not None
        self._processor_client.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {_q(self.processed_database)}.loss_slope_iteration_metrics (
                controller_mode LowCardinality(String),
                run_id String,
                model_id String,
                dataset_id Int32,
                window_id UInt32,
                training_iteration_id UInt64,
                status LowCardinality(String),
                incomplete_reason String,
                sample_count UInt32,
                positive_loss_sample_count UInt32,
                target_token_count UInt64,
                loss_value Nullable(Float64),
                processed_at_unix_ns UInt64
            )
            ENGINE = MergeTree
            ORDER BY (run_id, training_iteration_id, dataset_id)
            """
        )
        self._processor_client.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {_q(self.processed_database)}.loss_slope_window_indicators (
                controller_mode LowCardinality(String),
                run_id String,
                model_id String,
                dataset_id Int32,
                window_id UInt32,
                window_start_iteration UInt64,
                window_end_iteration UInt64,
                terminal_window UInt8,
                status LowCardinality(String),
                incomplete_reason String,
                iteration_point_count UInt32,
                expected_iteration_point_count UInt32,
                sample_count UInt32,
                positive_loss_sample_count UInt32,
                target_token_count UInt64,
                loss_slope Nullable(Float64),
                indicator Nullable(Float64),
                eta Float64,
                uniform_smoothing Float64,
                processed_at_unix_ns UInt64
            )
            ENGINE = MergeTree
            ORDER BY (run_id, window_id, dataset_id)
            """
        )
        self._processor_client.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {_q(self.processed_database)}.loss_slope_realized_mixture (
                controller_mode LowCardinality(String),
                run_id String,
                model_id String,
                training_iteration_id UInt64,
                dataset_id Int32,
                sample_count UInt32
            )
            ENGINE = MergeTree
            ORDER BY (run_id, training_iteration_id, dataset_id)
            """
        )
        self._processor_client.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {_q(self.control_database)}.loss_slope_mixture_decisions (
                controller_mode LowCardinality(String),
                run_id String,
                model_id String,
                decision_id UInt32,
                decision_type LowCardinality(String),
                source_window_id UInt32,
                source_window_end_iteration UInt64,
                effective_window_id UInt32,
                effective_training_iteration UInt64,
                weights Array(Float64),
                reason String,
                eta Float64,
                uniform_smoothing Float64,
                produced_at_unix_ns UInt64
            )
            ENGINE = MergeTree
            ORDER BY (run_id, effective_window_id, decision_id)
            """
        )
        self._processor_client.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {_q(self.control_database)}.loss_slope_decision_requests (
                controller_mode LowCardinality(String),
                run_id String,
                model_id String,
                effective_window_id UInt32,
                client_id String,
                status LowCardinality(String),
                event_at_unix_ns UInt64
            )
            ENGINE = MergeTree
            ORDER BY (run_id, effective_window_id, client_id, event_at_unix_ns)
            """
        )
        output_tables = [
            (self.processed_database, "loss_slope_iteration_metrics"),
            (self.processed_database, "loss_slope_window_indicators"),
            (self.processed_database, "loss_slope_realized_mixture"),
            (self.control_database, "loss_slope_mixture_decisions"),
            (self.control_database, "loss_slope_decision_requests"),
        ]
        for database, table in output_tables:
            count = self._processor_client.execute(
                f"SELECT count() FROM {_q(database)}.{_q(table)} "
                "WHERE run_id = %(run_id)s",
                {"run_id": self.run_id},
            )[0][0]
            if int(count) != 0:
                raise RuntimeError(
                    f"output already contains run_id={self.run_id!r} "
                    f"in {database}.{table}"
                )
        self._audit_writer = _AsyncClickHouseAuditWriter(self._audit_client_args)

    def raw_tables_ready(self) -> bool:
        names = {
            self.raw_table,
            f"{self.raw_table}_scalar_float",
            f"{self.raw_table}_scalar_int",
        }
        rows = self._processor_client.execute(
            "SELECT name FROM system.tables WHERE database = %(database)s",
            {"database": self.raw_database},
        )
        actual = {str(_decode(row[0])) for row in rows}
        return names <= actual

    def accepted_attempt(self, training_iteration_id: int) -> int | None:
        rows = self._processor_client.execute(
            f"""
            SELECT attempt_id, value
            FROM {_q(self.raw_database)}.{_q(self.raw_table + "_scalar_int")}
            WHERE model_id = %(model_id)s
              AND act_name = 'iteration_attempt_status'
              AND direction = 'iter'
              AND phase = 'train'
              AND global_batch_id = %(iteration)s
              AND invocation_id = 0
              AND dataset_id = -1
            ORDER BY attempt_id
            """,
            {
                "model_id": self.model_id,
                "iteration": int(training_iteration_id),
            },
        )
        by_attempt: dict[int, int] = {}
        for attempt_id, value in rows:
            attempt = int(attempt_id)
            if attempt in by_attempt:
                raise RuntimeError(
                    f"duplicate iteration status for iteration={training_iteration_id}, "
                    f"attempt={attempt}"
                )
            by_attempt[attempt] = int(value)
        accepted = [attempt for attempt, value in by_attempt.items() if value == 1]
        if len(accepted) > 1:
            raise RuntimeError(
                f"multiple accepted attempts for iteration {training_iteration_id}: "
                f"{accepted}"
            )
        return accepted[0] if accepted else None

    def read_iteration(
        self,
        *,
        training_iteration_id: int,
        attempt_id: int,
        expected_samples: int,
        expected_layers: int,
        expected_experts: int,
    ) -> list[LossSlopeSampleObservation]:
        return self._read_loss_slope_iteration(
            training_iteration_id=training_iteration_id,
            attempt_id=attempt_id,
            expected_samples=expected_samples,
        )

    def _read_loss_slope_iteration(
        self,
        *,
        training_iteration_id: int,
        attempt_id: int,
        expected_samples: int,
    ) -> list[LossSlopeSampleObservation]:
        common_params = {
            "model_id": self.model_id,
            "iteration": int(training_iteration_id),
            "attempt_id": int(attempt_id),
        }
        loss_rows = self._processor_client.execute(
            f"""
            SELECT dp_rank, microbatch_id, sample_index, layer_no, shard_rank,
                   token_start, token_end, dataset_id, value
            FROM {_q(self.raw_database)}.{_q(self.raw_table + "_scalar_float")}
            WHERE model_id = %(model_id)s
              AND act_name = '{LOSS_ACT}'
              AND direction = 'fwd' AND phase = 'train'
              AND global_batch_id = %(iteration)s
              AND attempt_id = %(attempt_id)s AND invocation_id = 0
            ORDER BY dp_rank, microbatch_id, sample_index, shard_rank
            """,
            common_params,
        )
        count_rows = self._processor_client.execute(
            f"""
            SELECT dp_rank, microbatch_id, sample_index, layer_no, shard_rank,
                   token_start, token_end, dataset_id, value
            FROM {_q(self.raw_database)}.{_q(self.raw_table + "_scalar_int")}
            WHERE model_id = %(model_id)s
              AND act_name = '{LOSS_COUNT_ACT}'
              AND direction = 'fwd' AND phase = 'train'
              AND global_batch_id = %(iteration)s
              AND attempt_id = %(attempt_id)s AND invocation_id = 0
            ORDER BY dp_rank, microbatch_id, sample_index, shard_rank
            """,
            common_params,
        )
        losses = _parse_sample_scalar_rows(loss_rows, LOSS_ACT, integer=False)
        counts = _parse_sample_scalar_rows(count_rows, LOSS_COUNT_ACT, integer=True)
        if len(losses) < expected_samples or len(counts) < expected_samples:
            raise IncompleteIterationError(
                f"iteration {training_iteration_id} has loss/count rows "
                f"{len(losses)}/{len(counts)}, expected {expected_samples}"
            )
        if len(losses) != expected_samples or len(counts) != expected_samples:
            raise RuntimeError(
                f"iteration {training_iteration_id} has unexpected loss/count rows "
                f"{len(losses)}/{len(counts)}, expected {expected_samples}"
            )
        if set(losses) != set(counts):
            raise RuntimeError(
                f"iteration {training_iteration_id} loss/count coordinates differ"
            )
        observations = []
        for coordinate in sorted(losses):
            loss_dataset_id, loss_value = losses[coordinate]
            count_dataset_id, count_value = counts[coordinate]
            if loss_dataset_id != count_dataset_id:
                raise RuntimeError(
                    f"loss/count dataset_id mismatch at coordinate={coordinate}"
                )
            observations.append(
                LossSlopeSampleObservation(
                    training_iteration_id=int(training_iteration_id),
                    sample_coordinate=coordinate,
                    dataset_id=loss_dataset_id,
                    loss_mean=float(loss_value),
                    loss_token_count=int(count_value),
                )
            )
        return observations

    def insert_loss_slope_iteration_metrics(
        self,
        metrics: Sequence[LossSlopeIterationMetric],
    ) -> None:
        if not self.enable_audit:
            return
        if self.controller_mode != LOSS_SLOPE_CONTROLLER_MODE:
            raise RuntimeError("loss-slope metrics cannot be written in another mode")
        assert self.processed_database is not None
        rows = [
            (
                metric.controller_mode,
                self.run_id,
                self.model_id,
                metric.dataset_id,
                metric.window_id,
                metric.training_iteration_id,
                metric.status,
                metric.incomplete_reason,
                metric.sample_count,
                metric.positive_loss_sample_count,
                metric.target_token_count,
                metric.loss_value,
                metric.processed_at_unix_ns,
            )
            for metric in metrics
        ]
        self._submit_audit(
            f"INSERT INTO {_q(self.processed_database)}.loss_slope_iteration_metrics VALUES",
            rows,
        )

    def insert_loss_slope_window_indicators(
        self,
        indicators: Sequence[LossSlopeWindowIndicator],
    ) -> None:
        if not self.enable_audit:
            return
        if self.controller_mode != LOSS_SLOPE_CONTROLLER_MODE:
            raise RuntimeError("loss-slope indicators cannot be written in another mode")
        assert self.processed_database is not None
        rows = [
            (
                item.controller_mode,
                self.run_id,
                self.model_id,
                item.dataset_id,
                item.window_id,
                item.window_start_iteration,
                item.window_end_iteration,
                int(item.terminal_window),
                item.status,
                item.incomplete_reason,
                item.iteration_point_count,
                item.expected_iteration_point_count,
                item.sample_count,
                item.positive_loss_sample_count,
                item.target_token_count,
                item.loss_slope,
                item.indicator,
                self.loss_slope_eta,
                self.loss_slope_uniform_smoothing,
                item.processed_at_unix_ns,
            )
            for item in indicators
        ]
        self._submit_audit(
            f"INSERT INTO {_q(self.processed_database)}.loss_slope_window_indicators VALUES",
            rows,
        )

    def insert_realized(
        self,
        training_iteration_id: int,
        observations: Sequence[LossSlopeSampleObservation],
    ) -> None:
        if not self.enable_audit:
            return
        assert self.processed_database is not None
        counts = Counter(observation.dataset_id for observation in observations)
        rows = [
            (
                LOSS_SLOPE_CONTROLLER_MODE,
                self.run_id,
                self.model_id,
                int(training_iteration_id),
                int(dataset_id),
                int(sample_count),
            )
            for dataset_id, sample_count in sorted(counts.items())
        ]
        self._submit_audit(
            f"INSERT INTO {_q(self.processed_database)}.loss_slope_realized_mixture VALUES",
            rows,
        )

    def insert_decision(self, decision: MixtureDecision) -> None:
        if not self.enable_audit:
            return
        assert self.control_database is not None
        row = (
            LOSS_SLOPE_CONTROLLER_MODE,
            self.run_id,
            self.model_id,
            decision.decision_id,
            decision.decision_type,
            decision.source_window_id,
            decision.source_window_end_iteration,
            decision.effective_window_id,
            decision.effective_training_iteration,
            list(decision.weights),
            decision.reason,
            self.loss_slope_eta,
            self.loss_slope_uniform_smoothing,
            decision.produced_at_unix_ns,
        )
        self._submit_audit(
            f"INSERT INTO {_q(self.control_database)}.loss_slope_mixture_decisions VALUES",
            [row],
        )

    def insert_request(
        self,
        run_id: str,
        effective_window_id: int,
        client_id: str,
        status: str,
    ) -> None:
        if not self.enable_audit:
            return
        if run_id != self.run_id:
            raise RuntimeError("decision request run_id mismatch")
        assert self.control_database is not None
        row = (
            LOSS_SLOPE_CONTROLLER_MODE,
            self.run_id,
            self.model_id,
            int(effective_window_id),
            str(client_id),
            str(status),
            time.time_ns(),
        )
        self._submit_audit(
            f"INSERT INTO {_q(self.control_database)}.loss_slope_decision_requests VALUES",
            [row],
        )

    def _submit_audit(
        self,
        query: str,
        rows: list[tuple[object, ...]],
    ) -> None:
        if self._audit_writer is None:
            raise RuntimeError("optional audit writer is not initialized")
        self._audit_writer.submit(query, rows)

    def close(self) -> None:
        try:
            if self._audit_writer is not None:
                self._audit_writer.close()
                self._audit_writer = None
        finally:
            self._processor_client.disconnect()


def _parse_sample_scalar_rows(
    rows: Sequence[tuple[Any, ...]],
    act_name: str,
    *,
    integer: bool,
) -> dict[tuple[int, int, int, int], tuple[int, float | int]]:
    values: dict[tuple[int, int, int, int], tuple[int, float | int]] = {}
    for row in rows:
        coordinate = tuple(int(value) for value in (row[0], row[1], row[2], row[4]))
        if coordinate in values:
            raise RuntimeError(f"duplicate {act_name} row at {coordinate}")
        if any(value < 0 for value in coordinate):
            raise RuntimeError(f"invalid {act_name} coordinate: {coordinate}")
        if int(row[3]) != -1 or int(row[4]) != 0:
            raise RuntimeError(
                f"{act_name} requires layer_no=-1 and shard_rank=0"
            )
        dataset_id = int(row[7])
        if dataset_id < 0:
            raise RuntimeError(f"{act_name} row has no dataset_id")
        value = int(row[8]) if integer else float(row[8])
        if integer and value < 0:
            raise RuntimeError(f"{act_name} is negative at {coordinate}")
        if not integer and not math.isfinite(value):
            raise RuntimeError(f"{act_name} is nonfinite at {coordinate}")
        values[coordinate] = (dataset_id, value)
    return values


class OnlineMixtureService:
    def __init__(
        self,
        *,
        storage: ClickHouseControlStorage,
        host: str,
        port: int,
        run_id: str,
        model_id: str,
        dataset_ids: Sequence[int],
        initial_weights: Sequence[float],
        first_iteration: int,
        train_iters: int,
        window_iters: int,
        global_batch_size: int,
        expected_layers: int,
        expected_experts: int,
        poll_interval_s: float,
        startup_timeout_s: float,
        feedback_deadline_s: float,
        pathway_threshold: float,
        minimum_conversations: int,
        minimum_target_tokens: int,
        required_update_count: int,
        controller_mode: str = LOSS_SLOPE_CONTROLLER_MODE,
        loss_slope_eta: float = 500.0,
        loss_slope_uniform_smoothing: float = 0.5,
        resume_state: Mapping[str, object] | None = None,
        serve_after_complete: bool = False,
    ) -> None:
        if train_iters % window_iters != 0:
            raise ValueError("train_iters must divide into complete control windows")
        if first_iteration <= 0 or first_iteration > train_iters:
            raise ValueError("first_iteration must be inside the configured training run")
        self.storage = storage
        self.first_iteration = int(first_iteration)
        self.train_iters = int(train_iters)
        self.window_iters = int(window_iters)
        self.global_batch_size = int(global_batch_size)
        self.expected_layers = int(expected_layers)
        self.expected_experts = int(expected_experts)
        self.poll_interval_s = float(poll_interval_s)
        self.startup_timeout_s = float(startup_timeout_s)
        self.feedback_deadline_s = float(feedback_deadline_s)
        self.pathway_threshold = float(pathway_threshold)
        self.minimum_conversations = int(minimum_conversations)
        self.minimum_target_tokens = int(minimum_target_tokens)
        self.required_update_count = int(required_update_count)
        self.serve_after_complete = bool(serve_after_complete)
        self.dataset_ids = tuple(int(value) for value in dataset_ids)
        if controller_mode != LOSS_SLOPE_CONTROLLER_MODE:
            raise ValueError(f"unsupported controller mode: {controller_mode!r}")
        self.controller_mode = LOSS_SLOPE_CONTROLLER_MODE
        storage_mode = getattr(storage, "controller_mode", self.controller_mode)
        if storage_mode != self.controller_mode:
            raise ValueError("storage and service controller modes differ")
        self.loss_slope_eta = float(loss_slope_eta)
        self.loss_slope_uniform_smoothing = float(loss_slope_uniform_smoothing)
        self.controller = LossSlopeMixtureController(
            run_id=run_id,
            dataset_ids=self.dataset_ids,
            initial_weights=initial_weights,
            window_iters=window_iters,
            total_windows=train_iters // window_iters,
            eta=self.loss_slope_eta,
            uniform_smoothing=self.loss_slope_uniform_smoothing,
        )
        audit_enabled = bool(storage.audit_enabled)
        self.store = DecisionStore(
            on_publish=self.storage.insert_decision if audit_enabled else None,
            on_request=self.storage.insert_request if audit_enabled else None,
        )
        self.model_id = str(model_id)
        self._state_condition = threading.Condition(threading.RLock())
        self._active_window_id: int | None = None
        self._active_states: dict[int, LossSlopeDomainWindowState] = {}
        self._processed_through_iteration = self.first_iteration - 1
        self._cumulative_decision_count = 0
        self._cumulative_update_count = 0
        self._pending_store_state: dict[str, object] | None = None
        self._completed = False
        self._shutdown_requested = False
        self._service_failure: BaseException | None = None
        self._processed_iterations: list[int] = []
        self._iteration_visibility: list[dict[str, object]] = []
        self._iteration_processing: list[dict[str, object]] = []
        self._window_processing: list[dict[str, object]] = []
        self._decisions: list[dict[str, object]] = []
        if resume_state is not None:
            self.load_state_dict(resume_state)
        self.http = DecisionHTTPServer(
            host,
            port,
            self.store,
            checkpoint_callback=self.checkpoint_snapshot,
            shutdown_callback=self.request_shutdown,
        )

    def state_dict(
        self,
        *,
        checkpoint_iteration: int,
        installed_window_id: int,
    ) -> dict[str, object]:
        return self._loss_slope_state_dict(
            checkpoint_iteration=checkpoint_iteration,
            installed_window_id=installed_window_id,
        )

    def _loss_slope_state_dict(
        self,
        *,
        checkpoint_iteration: int,
        installed_window_id: int,
    ) -> dict[str, object]:
        checkpoint_iteration = int(checkpoint_iteration)
        installed_window_id = int(installed_window_id)
        if self._processed_through_iteration != checkpoint_iteration:
            raise RuntimeError("loss-slope controller snapshot iteration mismatch")
        if not 1 <= installed_window_id <= self.controller.total_windows:
            raise ValueError("installed selection window is outside the run")
        pending_store = self.store.state_dict(
            run_id=self.controller.run_id,
            pending_effective_window_id=installed_window_id + 1,
        )
        pending_raw = pending_store["pending_decision"]
        if pending_raw is not None:
            pending = MixtureDecision.from_dict(pending_raw)
            if pending.source_window_id != installed_window_id:
                raise RuntimeError("pending loss-slope decision source window mismatch")
            if self.controller.previous_weights != pending.weights:
                raise RuntimeError("loss-slope controller and pending weights differ")
        return {
            "schema_version": 2,
            "controller_mode": LOSS_SLOPE_CONTROLLER_MODE,
            "source_run_id": self.controller.run_id,
            "source_model_id": self.model_id,
            "configuration": {
                "controller_mode": LOSS_SLOPE_CONTROLLER_MODE,
                "train_iters": self.train_iters,
                "window_iters": self.window_iters,
                "global_batch_size": self.global_batch_size,
                "required_update_count": self.required_update_count,
                "dataset_ids": list(self.dataset_ids),
                "eta": self.loss_slope_eta,
                "uniform_smoothing": self.loss_slope_uniform_smoothing,
            },
            "checkpoint_iteration": checkpoint_iteration,
            "processed_through_iteration": self._processed_through_iteration,
            "installed_window_id": installed_window_id,
            "controller": self.controller.state_dict(),
            "active_window_id": self._active_window_id,
            "active_states": {
                str(dataset_id): state.state_dict()
                for dataset_id, state in self._active_states.items()
            },
            "decision_store": pending_store,
            "cumulative_decision_count": self._cumulative_decision_count,
            "cumulative_update_count": self._cumulative_update_count,
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        self._load_loss_slope_state_dict(state)

    def _load_loss_slope_state_dict(self, state: Mapping[str, object]) -> None:
        expected = {
            "schema_version",
            "controller_mode",
            "source_run_id",
            "source_model_id",
            "configuration",
            "checkpoint_iteration",
            "processed_through_iteration",
            "installed_window_id",
            "controller",
            "active_window_id",
            "active_states",
            "decision_store",
            "cumulative_decision_count",
            "cumulative_update_count",
        }
        if set(state) != expected:
            raise ValueError("loss-slope online-controller state fields mismatch")
        if int(state["schema_version"]) != 2:
            raise ValueError("unsupported loss-slope online-controller state version")
        if state["controller_mode"] != LOSS_SLOPE_CONTROLLER_MODE:
            raise ValueError("loss-slope online-controller state mode mismatch")
        configuration = state["configuration"]
        if not isinstance(configuration, Mapping):
            raise TypeError("loss-slope online-controller configuration must be a mapping")
        runtime_configuration = {
            "controller_mode": LOSS_SLOPE_CONTROLLER_MODE,
            "train_iters": self.train_iters,
            "window_iters": self.window_iters,
            "global_batch_size": self.global_batch_size,
            "required_update_count": self.required_update_count,
            "dataset_ids": list(self.dataset_ids),
            "eta": self.loss_slope_eta,
            "uniform_smoothing": self.loss_slope_uniform_smoothing,
        }
        if dict(configuration) != runtime_configuration:
            raise ValueError(
                "loss-slope online-controller configuration mismatch: "
                f"checkpoint={dict(configuration)}, runtime={runtime_configuration}"
            )
        checkpoint_iteration = int(state["checkpoint_iteration"])
        processed_through = int(state["processed_through_iteration"])
        if checkpoint_iteration != processed_through:
            raise ValueError("loss-slope checkpoint and processed iterations differ")
        if self.first_iteration != checkpoint_iteration + 1:
            raise ValueError("loss-slope resumed service starts after the wrong iteration")
        controller_state = state["controller"]
        if not isinstance(controller_state, Mapping):
            raise TypeError("loss-slope controller state must be a mapping")
        self.controller.load_state_dict(controller_state)
        raw_active_window = state["active_window_id"]
        active_window_id = (
            None if raw_active_window is None else int(raw_active_window)
        )
        raw_active_states = state["active_states"]
        if not isinstance(raw_active_states, Mapping):
            raise TypeError("loss-slope active states must be a mapping")
        if active_window_id is None:
            if raw_active_states:
                raise ValueError("loss-slope active states exist without a window")
            active_states = {}
        else:
            if set(raw_active_states) != {str(value) for value in self.dataset_ids}:
                raise ValueError("loss-slope active-state dataset coverage mismatch")
            active_states = {
                dataset_id: LossSlopeDomainWindowState.from_state_dict(
                    raw_active_states[str(dataset_id)]
                )
                for dataset_id in self.dataset_ids
            }
            if any(
                item.window_id != active_window_id
                for item in active_states.values()
            ):
                raise ValueError("loss-slope active window IDs disagree")
        decision_store = state["decision_store"]
        if not isinstance(decision_store, dict):
            raise TypeError("loss-slope decision-store state must be a mapping")
        self._active_window_id = active_window_id
        self._active_states = active_states
        self._processed_through_iteration = processed_through
        self._cumulative_decision_count = int(state["cumulative_decision_count"])
        self._cumulative_update_count = int(state["cumulative_update_count"])
        if (
            self._cumulative_decision_count < 0
            or self._cumulative_update_count < 0
            or self._cumulative_update_count > self._cumulative_decision_count
        ):
            raise ValueError("loss-slope cumulative decision counts are invalid")
        self._pending_store_state = decision_store

    def checkpoint_snapshot(
        self,
        checkpoint_iteration: int,
        installed_window_id: int,
        timeout_s: float,
    ) -> dict[str, object]:
        if timeout_s <= 0.0:
            raise ValueError("checkpoint timeout must be positive")
        deadline = time.monotonic() + float(timeout_s)
        with self._state_condition:
            while self._processed_through_iteration < int(checkpoint_iteration):
                if self._service_failure is not None:
                    raise RuntimeError("controller service failed") from self._service_failure
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    raise TimeoutError(
                        "controller did not process through checkpoint iteration "
                        f"{checkpoint_iteration}"
                    )
                self._state_condition.wait(remaining)
            if self._processed_through_iteration > int(checkpoint_iteration):
                raise RuntimeError(
                    "controller processed beyond the requested checkpoint: "
                    f"processed={self._processed_through_iteration}, "
                    f"requested={checkpoint_iteration}"
                )
            state = self.state_dict(
                checkpoint_iteration=checkpoint_iteration,
                installed_window_id=installed_window_id,
            )
        payload = json.dumps(
            state, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return {
            "state": state,
            "sha256": hashlib.sha256(payload).hexdigest(),
        }

    def request_shutdown(self) -> None:
        with self._state_condition:
            self._shutdown_requested = True
            self._state_condition.notify_all()

    def run(self) -> dict[str, object]:
        self.storage.initialize_output()
        self.http.start()
        started = time.monotonic()
        try:
            if self._pending_store_state is not None:
                self.store.load_state_dict(
                    self._pending_store_state,
                    run_id=self.controller.run_id,
                )
                self._pending_store_state = None
            while not self.storage.raw_tables_ready():
                if time.monotonic() - started > self.startup_timeout_s:
                    raise TimeoutError("raw DMI ClickHouse tables did not appear")
                time.sleep(self.poll_interval_s)

            pending_seen_at: dict[int, float] = {}
            final_iteration = self.train_iters
            for iteration in range(self.first_iteration, final_iteration + 1):
                accepted_detected_at_unix_ns = None
                accepted_status_query_duration_ns = 0
                payload_query_decode_duration_ns = 0
                while True:
                    status_query_started_at = time.perf_counter_ns()
                    attempt_id = self.storage.accepted_attempt(iteration)
                    accepted_status_query_duration_ns += (
                        time.perf_counter_ns() - status_query_started_at
                    )
                    if attempt_id is None:
                        time.sleep(self.poll_interval_s)
                        continue
                    if accepted_detected_at_unix_ns is None:
                        accepted_detected_at_unix_ns = time.time_ns()
                    pending_seen_at.setdefault(iteration, time.monotonic())
                    query_started_at = time.perf_counter_ns()
                    try:
                        observations = self.storage.read_iteration(
                            training_iteration_id=iteration,
                            attempt_id=attempt_id,
                            expected_samples=self.global_batch_size,
                            expected_layers=self.expected_layers,
                            expected_experts=self.expected_experts,
                        )
                    except IncompleteIterationError:
                        payload_query_decode_duration_ns += (
                            time.perf_counter_ns() - query_started_at
                        )
                        elapsed = time.monotonic() - pending_seen_at[iteration]
                        if elapsed > self.feedback_deadline_s:
                            raise TimeoutError(
                                f"iteration {iteration} rows missed the "
                                f"{self.feedback_deadline_s:.3f}s feedback deadline"
                            )
                        time.sleep(self.poll_interval_s)
                        continue
                    payload_query_decode_duration_ns += (
                        time.perf_counter_ns() - query_started_at
                    )
                    break

                rows_complete_at_unix_ns = time.time_ns()
                with self._state_condition:
                    self._process_iteration(
                        iteration=iteration,
                        attempt_id=attempt_id,
                        observations=observations,
                        accepted_detected_at_unix_ns=int(
                            accepted_detected_at_unix_ns
                        ),
                        rows_complete_at_unix_ns=rows_complete_at_unix_ns,
                        accepted_status_query_duration_ns=(
                            accepted_status_query_duration_ns
                        ),
                        payload_query_decode_duration_ns=(
                            payload_query_decode_duration_ns
                        ),
                    )
                    self._processed_through_iteration = iteration
                    self._state_condition.notify_all()

            if self._active_window_id is not None or self._active_states:
                raise RuntimeError("controller finished with an unsealed active window")
            with self._state_condition:
                self._completed = True
                self._state_condition.notify_all()
                while self.serve_after_complete and not self._shutdown_requested:
                    self._state_condition.wait()
            result = {
                "kind": "dmi_online_mixture_service_result",
                "run_id": self.controller.run_id,
                "model_id": self.model_id,
                "processed_iterations": self._processed_iterations,
                "iteration_visibility": self._iteration_visibility,
                "iteration_processing": self._iteration_processing,
                "window_processing": self._window_processing,
                "decision_count": self._cumulative_decision_count,
                "update_count": self._cumulative_update_count,
                "decisions": self._decisions,
                "terminal_window_id": self.controller.total_windows,
            }
            result.update(
                {
                    "controller_mode": LOSS_SLOPE_CONTROLLER_MODE,
                    "policy": {
                        "indicator": "-loss_slope",
                        "eta": self.loss_slope_eta,
                        "uniform_smoothing": (
                            self.loss_slope_uniform_smoothing
                        ),
                        "window_iters": self.window_iters,
                    },
                    "actionable_window_count": (
                        self.controller.total_windows - 1
                    ),
                    "hold_count": (
                        self._cumulative_decision_count
                        - self._cumulative_update_count
                    ),
                }
            )
            return result
        except BaseException as error:
            with self._state_condition:
                self._service_failure = error
                self._state_condition.notify_all()
            raise
        finally:
            self.http.close()
            self.storage.close()

    def _process_iteration(
        self,
        *,
        iteration: int,
        attempt_id: int,
        observations: Sequence[LossSlopeSampleObservation],
        accepted_detected_at_unix_ns: int,
        rows_complete_at_unix_ns: int,
        accepted_status_query_duration_ns: int,
        payload_query_decode_duration_ns: int,
    ) -> None:
        self._process_loss_slope_iteration(
            iteration=iteration,
            attempt_id=attempt_id,
            observations=observations,
            accepted_detected_at_unix_ns=accepted_detected_at_unix_ns,
            rows_complete_at_unix_ns=rows_complete_at_unix_ns,
            accepted_status_query_duration_ns=accepted_status_query_duration_ns,
            payload_query_decode_duration_ns=payload_query_decode_duration_ns,
        )

    def _process_loss_slope_iteration(
        self,
        *,
        iteration: int,
        attempt_id: int,
        observations: Sequence[LossSlopeSampleObservation],
        accepted_detected_at_unix_ns: int,
        rows_complete_at_unix_ns: int,
        accepted_status_query_duration_ns: int,
        payload_query_decode_duration_ns: int,
    ) -> None:
        if any(
            not isinstance(observation, LossSlopeSampleObservation)
            for observation in observations
        ):
            raise TypeError("loss-slope service received a non-loss-slope observation")
        self._processed_iterations.append(iteration)
        self._iteration_visibility.append(
            {
                "training_iteration_id": iteration,
                "attempt_id": attempt_id,
                "accepted_detected_at_unix_ns": accepted_detected_at_unix_ns,
                "rows_complete_at_unix_ns": rows_complete_at_unix_ns,
                "accepted_to_complete_ns": (
                    rows_complete_at_unix_ns - accepted_detected_at_unix_ns
                ),
            }
        )
        self.storage.insert_realized(iteration, observations)
        window_id = (iteration - 1) // self.window_iters + 1
        window_first_iteration = (window_id - 1) * self.window_iters + 1
        window_last_iteration = window_id * self.window_iters
        if self._active_window_id is None:
            terminal = window_id == self.controller.total_windows
            self._active_states = {
                dataset_id: LossSlopeDomainWindowState(
                    dataset_id=dataset_id,
                    window_id=window_id,
                    window_start_iteration=window_first_iteration,
                    window_end_iteration=window_last_iteration,
                    terminal_window=terminal,
                )
                for dataset_id in self.dataset_ids
            }
            self._active_window_id = window_id
        elif self._active_window_id != window_id:
            raise RuntimeError(
                f"loss-slope controller advanced to window {window_id} with "
                f"window {self._active_window_id} still active"
            )

        samples_by_dataset: dict[int, list[LossSlopeSampleObservation]] = {
            dataset_id: [] for dataset_id in self.dataset_ids
        }
        for observation in observations:
            try:
                samples_by_dataset[observation.dataset_id].append(observation)
            except KeyError as exc:
                raise RuntimeError(
                    f"iteration {iteration} contains unknown "
                    f"dataset_id={observation.dataset_id}"
                ) from exc

        incremental_started_at = time.perf_counter_ns()
        iteration_metrics = []
        for dataset_id in self.dataset_ids:
            state = self._active_states[dataset_id]
            if not isinstance(state, LossSlopeDomainWindowState):
                raise TypeError("loss-slope service has a non-loss-slope window state")
            iteration_metrics.append(
                state.add_iteration_samples(
                    iteration,
                    samples_by_dataset[dataset_id],
                )
            )
        incremental_update_duration_ns = (
            time.perf_counter_ns() - incremental_started_at
        )
        self.storage.insert_loss_slope_iteration_metrics(iteration_metrics)
        self._iteration_processing.append(
            {
                "controller_mode": LOSS_SLOPE_CONTROLLER_MODE,
                "training_iteration_id": iteration,
                "attempt_id": attempt_id,
                "sample_count": len(observations),
                "accepted_status_query_duration_ns": (
                    accepted_status_query_duration_ns
                ),
                "payload_query_decode_duration_ns": (
                    payload_query_decode_duration_ns
                ),
                "query_decode_duration_ns": (
                    accepted_status_query_duration_ns
                    + payload_query_decode_duration_ns
                ),
                "loss_aggregation_duration_ns": incremental_update_duration_ns,
            }
        )
        if iteration != window_last_iteration:
            return

        processing_started_at_unix_ns = time.time_ns()
        terminal = window_id == self.controller.total_windows
        finalization_started_at = time.perf_counter_ns()
        indicators = {}
        for dataset_id in self.dataset_ids:
            state = self._active_states[dataset_id]
            if not isinstance(state, LossSlopeDomainWindowState):
                raise TypeError("loss-slope service has a non-loss-slope window state")
            indicators[dataset_id] = state.finalize()
        finalization_duration_ns = time.perf_counter_ns() - finalization_started_at
        self._active_states = {}
        self._active_window_id = None
        self.storage.insert_loss_slope_window_indicators(list(indicators.values()))
        control_started_at = time.perf_counter_ns()
        result = self.controller.process_window(indicators)
        control_processing_duration_ns = time.perf_counter_ns() - control_started_at
        publication_started_at = time.perf_counter_ns()
        if result.decision is not None:
            self.store.publish(result.decision)
            self._decisions.append(result.decision.to_dict())
            self._cumulative_decision_count += 1
            if result.decision.decision_type == "UPDATE":
                self._cumulative_update_count += 1
        decision_publication_duration_ns = (
            time.perf_counter_ns() - publication_started_at
        )
        processing_finished_at_unix_ns = time.time_ns()
        self._window_processing.append(
            {
                "controller_mode": LOSS_SLOPE_CONTROLLER_MODE,
                "window_id": window_id,
                "window_end_iteration": window_last_iteration,
                "terminal_window": terminal,
                "processing_started_at_unix_ns": processing_started_at_unix_ns,
                "processing_finished_at_unix_ns": processing_finished_at_unix_ns,
                "processing_duration_ns": (
                    processing_finished_at_unix_ns
                    - processing_started_at_unix_ns
                ),
                "finalization_duration_ns": finalization_duration_ns,
                "loss_control_processing_duration_ns": (
                    control_processing_duration_ns
                ),
                "decision_publication_duration_ns": (
                    decision_publication_duration_ns
                ),
                "decision_produced_at_unix_ns": (
                    None
                    if result.decision is None
                    else result.decision.produced_at_unix_ns
                ),
            }
        )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--user", default="default")
    parser.add_argument("--password", default="")
    parser.add_argument("--raw-database", required=True)
    parser.add_argument("--raw-table", default="dmi_training_tensors")
    parser.add_argument("--processed-database")
    parser.add_argument("--control-database")
    parser.add_argument(
        "--enable-audit",
        action="store_true",
        help="Persist optional controller and HTTP audit records asynchronously.",
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument(
        "--controller-mode",
        choices=(LOSS_SLOPE_CONTROLLER_MODE,),
        default=LOSS_SLOPE_CONTROLLER_MODE,
    )
    parser.add_argument("--loss-slope-eta", type=float, default=500.0)
    parser.add_argument(
        "--loss-slope-uniform-smoothing", type=float, default=0.5
    )
    parser.add_argument("--http-host", default="127.0.0.1")
    parser.add_argument("--http-port", type=int, default=8765)
    parser.add_argument("--dataset-ids", default="0,1,2,3")
    parser.add_argument("--initial-weights", default="0.25,0.25,0.25,0.25")
    parser.add_argument("--first-iteration", type=int, default=1)
    parser.add_argument("--train-iters", type=int, default=12)
    parser.add_argument("--window-iters", type=int, default=2)
    parser.add_argument("--global-batch-size", type=int, default=128)
    parser.add_argument("--expected-layers", type=int, default=24)
    parser.add_argument("--expected-experts", type=int, default=32)
    parser.add_argument("--poll-interval-s", type=float, default=0.2)
    parser.add_argument("--startup-timeout-s", type=float, default=300.0)
    parser.add_argument("--feedback-deadline-s", type=float, default=600.0)
    parser.add_argument("--pathway-threshold", type=float, default=0.7)
    parser.add_argument("--minimum-conversations", type=int, default=24)
    parser.add_argument("--minimum-target-tokens", type=int, default=256)
    parser.add_argument("--required-update-count", type=int, default=3)
    parser.add_argument(
        "--resume-state-json",
        help="Controller-state JSON extracted from a committed DMI checkpoint.",
    )
    parser.add_argument(
        "--serve-after-complete",
        action="store_true",
        help="Keep checkpoint and decision HTTP endpoints alive until graceful shutdown.",
    )
    parser.add_argument("--result-json", default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    dataset_ids = tuple(int(value) for value in args.dataset_ids.split(","))
    initial_weights = tuple(float(value) for value in args.initial_weights.split(","))
    storage = ClickHouseControlStorage(
        host=args.host,
        port=args.port,
        user=args.user,
        password=args.password,
        raw_database=args.raw_database,
        raw_table=args.raw_table,
        processed_database=args.processed_database,
        control_database=args.control_database,
        run_id=args.run_id,
        model_id=args.model_id,
        enable_audit=args.enable_audit,
        controller_mode=args.controller_mode,
        loss_slope_eta=args.loss_slope_eta,
        loss_slope_uniform_smoothing=args.loss_slope_uniform_smoothing,
    )
    resume_state = None
    if args.resume_state_json:
        with open(args.resume_state_json, "r", encoding="utf-8") as handle:
            resume_state = json.load(handle)
        if not isinstance(resume_state, dict):
            raise TypeError("controller resume-state file must contain a JSON object")
    service = OnlineMixtureService(
        storage=storage,
        host=args.http_host,
        port=args.http_port,
        run_id=args.run_id,
        model_id=args.model_id,
        dataset_ids=dataset_ids,
        initial_weights=initial_weights,
        first_iteration=args.first_iteration,
        train_iters=args.train_iters,
        window_iters=args.window_iters,
        global_batch_size=args.global_batch_size,
        expected_layers=args.expected_layers,
        expected_experts=args.expected_experts,
        poll_interval_s=args.poll_interval_s,
        startup_timeout_s=args.startup_timeout_s,
        feedback_deadline_s=args.feedback_deadline_s,
        pathway_threshold=args.pathway_threshold,
        minimum_conversations=args.minimum_conversations,
        minimum_target_tokens=args.minimum_target_tokens,
        required_update_count=args.required_update_count,
        controller_mode=args.controller_mode,
        loss_slope_eta=args.loss_slope_eta,
        loss_slope_uniform_smoothing=args.loss_slope_uniform_smoothing,
        resume_state=resume_state,
        serve_after_complete=args.serve_after_complete,
    )
    result = service.run()
    payload = json.dumps(result, indent=2, sort_keys=True)
    print(payload)
    if args.result_json:
        with open(args.result_json, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
