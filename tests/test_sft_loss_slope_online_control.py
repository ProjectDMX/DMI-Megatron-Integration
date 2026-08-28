import math

import pytest

from tools.sft_mixture.controller import (
    LOSS_SLOPE_CONTROLLER_MODE,
    LossSlopeSampleObservation,
)
from tools.sft_mixture.online_control import (
    ClickHouseControlStorage,
    OnlineMixtureService,
)


def _scalar_row(*, microbatch, dataset_id, value):
    return (0, microbatch, 0, -1, 0, -1, -1, dataset_id, value)


class _ReadClient:
    def __init__(self, loss_rows, count_rows):
        self.loss_rows = loss_rows
        self.count_rows = count_rows
        self.queries = []

    def execute(self, query, *_args):
        self.queries.append(query)
        if "lm_per_sample_loss_token_count" in query:
            return self.count_rows
        if "lm_per_sample_loss" in query:
            return self.loss_rows
        raise AssertionError(f"unexpected ClickHouse query: {query}")


def _storage_with_read_client(monkeypatch, loss_rows, count_rows):
    class FakeClient:
        def __init__(self, **_kwargs):
            return None

        def disconnect(self):
            return None

    monkeypatch.setattr("clickhouse_driver.Client", FakeClient)
    storage = ClickHouseControlStorage(
        host="localhost",
        port=9000,
        user="default",
        password="",
        raw_database="raw",
        raw_table="payload",
        processed_database=None,
        control_database=None,
        run_id="run",
        model_id="model",
        enable_audit=False,
        controller_mode=LOSS_SLOPE_CONTROLLER_MODE,
    )
    client = _ReadClient(loss_rows, count_rows)
    storage._processor_client = client
    return storage, client


def test_loss_slope_storage_reads_only_loss_and_count_without_router_query(monkeypatch):
    storage, client = _storage_with_read_client(
        monkeypatch,
        [
            _scalar_row(microbatch=0, dataset_id=0, value=2.0),
            _scalar_row(microbatch=1, dataset_id=1, value=3.0),
        ],
        [
            _scalar_row(microbatch=0, dataset_id=0, value=7),
            _scalar_row(microbatch=1, dataset_id=1, value=11),
        ],
    )

    observations = storage.read_iteration(
        training_iteration_id=1,
        attempt_id=0,
        expected_samples=2,
        expected_layers=24,
        expected_experts=32,
    )

    assert [item.dataset_id for item in observations] == [0, 1]
    assert [item.loss_mean for item in observations] == [2.0, 3.0]
    assert [item.loss_token_count for item in observations] == [7, 11]
    assert len(client.queries) == 2
    assert not any("router_probs_mean" in query for query in client.queries)
    assert not any("FROM `raw`.`payload`" in query for query in client.queries)


def test_loss_slope_audit_uses_dedicated_mode_named_tables(monkeypatch):
    class FakeClient:
        instances = []

        def __init__(self, **_kwargs):
            self.executions = []
            self.instances.append(self)

        def execute(self, query, *args):
            self.executions.append((query, args))
            if "SELECT count()" in query:
                return [(0,)]
            return []

        def disconnect(self):
            return None

    monkeypatch.setattr("clickhouse_driver.Client", FakeClient)
    storage = ClickHouseControlStorage(
        host="localhost",
        port=9000,
        user="default",
        password="",
        raw_database="raw",
        raw_table="payload",
        processed_database="processed",
        control_database="control",
        run_id="run",
        model_id="model",
        enable_audit=True,
        controller_mode=LOSS_SLOPE_CONTROLLER_MODE,
        loss_slope_eta=500.0,
        loss_slope_uniform_smoothing=0.5,
    )
    storage.initialize_output()
    storage.insert_request("run", 2, "rank-0", "DELIVERED")
    storage.close()

    processor, writer = FakeClient.instances
    create_text = "\n".join(query for query, _args in processor.executions)
    assert "loss_slope_iteration_metrics" in create_text
    assert "loss_slope_window_indicators" in create_text
    assert "loss_slope_realized_mixture" in create_text
    assert "loss_slope_mixture_decisions" in create_text
    assert "loss_slope_decision_requests" in create_text
    assert "domain_window_metrics" not in create_text
    request_inserts = [
        (query, args)
        for query, args in writer.executions
        if "loss_slope_decision_requests" in query
    ]
    assert len(request_inserts) == 1
    assert request_inserts[0][1][0][0][0] == LOSS_SLOPE_CONTROLLER_MODE


@pytest.mark.parametrize(
    ("loss_rows", "count_rows", "message"),
    (
        (
            [
                _scalar_row(microbatch=0, dataset_id=0, value=2.0),
                _scalar_row(microbatch=0, dataset_id=0, value=2.0),
            ],
            [
                _scalar_row(microbatch=0, dataset_id=0, value=7),
                _scalar_row(microbatch=1, dataset_id=1, value=11),
            ],
            "duplicate lm_per_sample_loss row",
        ),
        (
            [
                _scalar_row(microbatch=0, dataset_id=0, value=2.0),
                _scalar_row(microbatch=1, dataset_id=1, value=3.0),
            ],
            [
                _scalar_row(microbatch=0, dataset_id=1, value=7),
                _scalar_row(microbatch=1, dataset_id=1, value=11),
            ],
            "dataset_id mismatch",
        ),
        (
            [
                _scalar_row(microbatch=0, dataset_id=0, value=math.inf),
                _scalar_row(microbatch=1, dataset_id=1, value=3.0),
            ],
            [
                _scalar_row(microbatch=0, dataset_id=0, value=7),
                _scalar_row(microbatch=1, dataset_id=1, value=11),
            ],
            "is nonfinite",
        ),
        (
            [
                _scalar_row(microbatch=0, dataset_id=-1, value=2.0),
                _scalar_row(microbatch=1, dataset_id=1, value=3.0),
            ],
            [
                _scalar_row(microbatch=0, dataset_id=-1, value=7),
                _scalar_row(microbatch=1, dataset_id=1, value=11),
            ],
            "has no dataset_id",
        ),
    ),
)
def test_loss_slope_storage_fails_fast_on_corrupt_rows(
    monkeypatch, loss_rows, count_rows, message
):
    storage, _client = _storage_with_read_client(
        monkeypatch, loss_rows, count_rows
    )
    with pytest.raises(RuntimeError, match=message):
        storage.read_iteration(
            training_iteration_id=1,
            attempt_id=0,
            expected_samples=2,
            expected_layers=24,
            expected_experts=32,
        )


class _LossSlopeFakeStorage:
    controller_mode = LOSS_SLOPE_CONTROLLER_MODE

    def __init__(self, *, omit_domain_at=None):
        self.audit_enabled = True
        self.omit_domain_at = omit_domain_at
        self.iteration_metrics = []
        self.window_indicators = []
        self.realized = []
        self.decisions = []
        self.requests = []
        self.closed = False

    def initialize_output(self):
        return None

    def raw_tables_ready(self):
        return True

    def accepted_attempt(self, _iteration):
        return 0

    def read_iteration(self, *, training_iteration_id, **_kwargs):
        observations = []
        for dataset_id in (0, 1):
            if self.omit_domain_at == (dataset_id, training_iteration_id):
                continue
            sample_count = 2
            if (
                self.omit_domain_at is not None
                and self.omit_domain_at[1] == training_iteration_id
                and self.omit_domain_at[0] != dataset_id
            ):
                sample_count = 4
            for local_index in range(sample_count):
                observations.append(
                    LossSlopeSampleObservation(
                        training_iteration_id=training_iteration_id,
                        sample_coordinate=(
                            0,
                            dataset_id * 2 + local_index,
                            0,
                            0,
                        ),
                        dataset_id=dataset_id,
                        loss_mean=(
                            4.0
                            - (dataset_id + 1) * 0.1 * training_iteration_id
                            + local_index * 0.01
                        ),
                        loss_token_count=10 + local_index,
                    )
                )
        return observations

    def insert_loss_slope_iteration_metrics(self, metrics):
        self.iteration_metrics.extend(metrics)

    def insert_loss_slope_window_indicators(self, indicators):
        self.window_indicators.extend(indicators)

    def insert_realized(self, iteration, observations):
        self.realized.append((iteration, tuple(observations)))

    def insert_decision(self, decision):
        self.decisions.append(decision)

    def insert_request(self, run_id, window_id, client_id, status):
        self.requests.append((run_id, window_id, client_id, status))

    def close(self):
        self.closed = True


def _service(storage, *, global_batch_size=4):
    return OnlineMixtureService(
        storage=storage,
        host="127.0.0.1",
        port=0,
        run_id="run",
        model_id="model",
        dataset_ids=(0, 1),
        initial_weights=(0.5, 0.5),
        first_iteration=1,
        train_iters=4,
        window_iters=2,
        global_batch_size=global_batch_size,
        expected_layers=24,
        expected_experts=32,
        poll_interval_s=0.001,
        startup_timeout_s=1.0,
        feedback_deadline_s=1.0,
        pathway_threshold=0.7,
        minimum_conversations=0,
        minimum_target_tokens=0,
        required_update_count=1,
        controller_mode=LOSS_SLOPE_CONTROLLER_MODE,
        loss_slope_eta=500.0,
        loss_slope_uniform_smoothing=0.5,
    )


def test_loss_slope_online_service_updates_first_window_and_names_policy():
    storage = _LossSlopeFakeStorage()
    result = _service(storage).run()

    assert result["controller_mode"] == LOSS_SLOPE_CONTROLLER_MODE
    assert result["policy"] == {
        "indicator": "-loss_slope",
        "eta": 500.0,
        "uniform_smoothing": 0.5,
        "window_iters": 2,
    }
    assert result["decision_count"] == 1
    assert result["update_count"] == 1
    assert result["hold_count"] == 0
    assert len(storage.iteration_metrics) == 8
    assert len(storage.window_indicators) == 4
    assert storage.decisions[0].effective_training_iteration == 3
    assert storage.closed


def test_loss_slope_online_service_accepts_valid_missing_domain_as_hold():
    storage = _LossSlopeFakeStorage(omit_domain_at=(1, 2))
    result = _service(storage).run()

    assert result["decision_count"] == 1
    assert result["update_count"] == 0
    assert result["hold_count"] == 1
    assert storage.decisions[0].decision_type == "HOLD_INCOMPLETE"
    assert "missing_loss_iterations=2" in storage.decisions[0].reason
