from __future__ import annotations

import os
import time
import uuid

import pytest
import torch
from torch import nn

from dmi.api.v1 import (
    ClickHouseClientConfig,
    DMXHostEngine,
    HookPointV1,
    MonitoringEngine,
    OutputStorage,
    RingConfig,
    StageConfig,
)
from dmi_megatron_integration.adapter import MegatronAdaptor, MegatronTrainingContext
from dmi_megatron_integration.hooks.specs import (
    DimSpec,
    HookPhase,
    MegatronHookSpec,
    MegatronOutputSpec,
)
from dmi_megatron_integration.records.format import MegatronRecordFormat
from dmi_megatron_integration.records.reader import MegatronTrainingReader


class TinyMegatronHookModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.router_summary = HookPointV1()
        self.router_summary._dmi_megatron_spec = MegatronHookSpec(
            name="router_summary",
            layer_no=0,
            outputs=[
                MegatronOutputSpec(
                    name="router_probs_mean",
                    input_shape=[DimSpec.BATCH, DimSpec.NUM_EXPERTS],
                    dtype=torch.float32,
                )
            ],
            preprocess=lambda x: x,
            enabled_by=frozenset({"router-summary"}),
        )
        self.router_summary.hook_phase = HookPhase.FWD
        self.router_summary.suppress_recompute = True
        self.loss_summary = HookPointV1()
        self.loss_summary._dmi_megatron_spec = MegatronHookSpec(
            name="loss_summary",
            layer_no=-1,
            outputs=[
                MegatronOutputSpec(
                    name="lm_per_sample_loss",
                    input_shape=[DimSpec.BATCH, 1],
                    dtype=torch.float32,
                    storage=OutputStorage.SCALAR_FLOAT,
                )
            ],
            preprocess=lambda x: x,
            enabled_by=frozenset({"loss-summary"}),
            need_token_range=False,
        )
        self.loss_summary.hook_phase = HookPhase.FWD
        self.loss_summary.suppress_recompute = True

    def forward(self, router_probs_mean: torch.Tensor, loss_summary: torch.Tensor) -> torch.Tensor:
        self.router_summary(router_probs_mean)
        self.loss_summary(loss_summary)
        return router_probs_mean


def _clickhouse_client_or_skip():
    try:
        import clickhouse_driver
    except ImportError:
        pytest.skip("clickhouse-driver is required for ClickHouse E2E tests")

    host = os.environ.get("DMX_DB_HOST", "localhost")
    port = int(os.environ.get("DMX_DB_PORT", "9000"))
    user = os.environ.get("DMX_DB_USER", "default")
    password = os.environ.get("DMX_DB_PASSWORD", "")
    try:
        client = clickhouse_driver.Client(host=host, port=port, user=user, password=password)
        client.execute("SELECT 1")
    except Exception as exc:
        pytest.skip(f"ClickHouse is not reachable at {host}:{port}: {exc}")
    return client


def _build_training_engine(*, model_id: str, table: str, database: str) -> MonitoringEngine:
    record_format = MegatronRecordFormat(table)
    ch_cfg = ClickHouseClientConfig()
    ch_cfg.host = os.environ.get("DMX_DB_HOST", "localhost")
    ch_cfg.port = int(os.environ.get("DMX_DB_PORT", "9000"))
    ch_cfg.username = os.environ.get("DMX_DB_USER", "default")
    ch_cfg.password = os.environ.get("DMX_DB_PASSWORD", "")
    ch_cfg.database = database
    ch_cfg.create_database_if_missing = True

    host_engine = DMXHostEngine(
        StageConfig.clickhouse_records(
            ch_cfg,
            record_format.schema,
            parallelism=int(os.environ.get("DMI_E2E_CH_PARALLELISM", "10")),
            name="clickhouse_training_records",
        )
    )

    ring_cfg = RingConfig()
    ring_cfg.payload_ring_bytes = int(os.environ.get("DMI_E2E_RING_PAYLOAD_MB", "64")) * 1024 * 1024
    ring_cfg.pinned_staging_bytes = int(os.environ.get("DMI_E2E_RING_PINNED_MB", "64")) * 1024 * 1024
    ring_cfg.task_ring_entries = int(os.environ.get("DMI_E2E_RING_TASK_ENTRIES", "1024"))

    return MonitoringEngine(
        config=None,
        model_id=model_id,
        host_engine=host_engine,
        ring_config=ring_cfg,
    )


def _create_training_table(client, *, database: str, table: str) -> None:
    client.execute(
        f"""
        CREATE TABLE IF NOT EXISTS `{database}`.`{table}` (
            `model_id` String,
            `act_name` String,
            `direction` String,
            `phase` String,
            `global_batch_id` Int64,
            `dp_rank` Int32,
            `microbatch_id` Int32,
            `sample_index` Int32,
            `layer_no` Int32,
            `shard_rank` Int32,
            `token_start` Int64,
            `token_end` Int64,
            `attempt_id` Int32,
            `invocation_id` Int32,
            `dataset_id` Int32,
            `dtype` String,
            `shape` Array(Int64),
            `bytes` String
        ) ENGINE = MergeTree
        PRIMARY KEY (
            `model_id`, `act_name`, `direction`, `phase`, `global_batch_id`, `dp_rank`,
            `microbatch_id`, `sample_index`, `layer_no`, `shard_rank`,
            `token_start`, `token_end`, `attempt_id`, `invocation_id`
        )
        ORDER BY (
            `model_id`, `act_name`, `direction`, `phase`, `global_batch_id`, `dp_rank`,
            `microbatch_id`, `sample_index`, `layer_no`, `shard_rank`,
            `token_start`, `token_end`, `attempt_id`, `invocation_id`
        )
        """
    )


def _read_training_rows(*, model_id: str, table: str, database: str):
    reader = MegatronTrainingReader(
        host=os.environ.get("DMX_DB_HOST", "localhost"),
        port=int(os.environ.get("DMX_DB_PORT", "9000")),
        username=os.environ.get("DMX_DB_USER", "default"),
        password=os.environ.get("DMX_DB_PASSWORD", ""),
        database=database,
        table=table,
    )
    try:
        return reader.training_raw_prefix_get(
            (model_id, "router_probs_mean", "fwd", "train"),
            return_full_key_tuple=True,
        )
    finally:
        reader.close()


def _wait_for_rows(*, model_id: str, table: str, database: str, expected_rows: int):
    deadline = time.time() + float(os.environ.get("DMI_E2E_CLICKHOUSE_TIMEOUT_S", "10"))
    last_rows = []
    while time.time() < deadline:
        rows = _read_training_rows(model_id=model_id, table=table, database=database)
        if len(rows) >= expected_rows:
            return rows
        last_rows = rows
        time.sleep(0.1)
    return last_rows


def _read_training_scalar_rows(*, model_id: str, table: str, database: str):
    reader = MegatronTrainingReader(
        host=os.environ.get("DMX_DB_HOST", "localhost"),
        port=int(os.environ.get("DMX_DB_PORT", "9000")),
        username=os.environ.get("DMX_DB_USER", "default"),
        password=os.environ.get("DMX_DB_PASSWORD", ""),
        database=database,
        table=table,
    )
    try:
        return reader.training_scalar_raw_prefix_get(
            (model_id, "lm_per_sample_loss", "fwd", "train"),
            scalar_kind="float",
            return_full_key_tuple=True,
        )
    finally:
        reader.close()


def _wait_for_scalar_rows(*, model_id: str, table: str, database: str, expected_rows: int):
    deadline = time.time() + float(os.environ.get("DMI_E2E_CLICKHOUSE_TIMEOUT_S", "10"))
    last_rows = []
    while time.time() < deadline:
        rows = _read_training_scalar_rows(model_id=model_id, table=table, database=database)
        if len(rows) >= expected_rows:
            return rows
        last_rows = rows
        time.sleep(0.1)
    return last_rows


def _wait_for_training_tables(client, *, database: str, table: str) -> set[str]:
    expected = {
        table,
        f"{table}_scalar_float",
        f"{table}_scalar_int",
        f"{table}_eval_phase_boundary",
    }
    deadline = time.time() + float(os.environ.get("DMI_E2E_CLICKHOUSE_TIMEOUT_S", "10"))
    found: set[str] = set()
    while time.time() < deadline:
        found = {
            str(row[0])
            for row in client.execute(
                "SELECT name FROM system.tables "
                "WHERE database = %(database)s AND name IN %(names)s",
                {"database": database, "names": tuple(expected)},
            )
        }
        if found == expected:
            return found
        time.sleep(0.1)
    return found


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Training schema E2E needs CUDA")
def test_training_schema_initialization_is_per_target():
    """Each distinct training table is initialized once within one process."""

    client = _clickhouse_client_or_skip()
    database = os.environ.get("DMX_DB_DATABASE", "default")
    tables = [
        f"dmi_training_schema_target_a_{uuid.uuid4().hex}",
        f"dmi_training_schema_target_b_{uuid.uuid4().hex}",
    ]
    engine = None
    client.execute(f"CREATE DATABASE IF NOT EXISTS `{database}`")
    try:
        for table in tables:
            engine = _build_training_engine(
                model_id=f"schema-target-{uuid.uuid4().hex}",
                table=table,
                database=database,
            )
            assert _wait_for_training_tables(
                client,
                database=database,
                table=table,
            ) == {
                table,
                f"{table}_scalar_float",
                f"{table}_scalar_int",
                f"{table}_eval_phase_boundary",
            }
            engine.close()
            engine = None
    finally:
        if engine is not None:
            engine.close()
        for table in tables:
            client.execute(f"DROP TABLE IF EXISTS `{database}`.`{table}`")
            client.execute(f"DROP TABLE IF EXISTS `{database}`.`{table}_scalar_float`")
            client.execute(f"DROP TABLE IF EXISTS `{database}`.`{table}_scalar_int`")
            client.execute(f"DROP TABLE IF EXISTS `{database}`.`{table}_eval_phase_boundary`")
        client.disconnect()


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Megatron DMI E2E needs CUDA")
def test_megatron_router_summary_training_clickhouse_e2e():
    """Supported E2E: [B,E] router summary -> ring -> training ClickHouse rows."""

    client = _clickhouse_client_or_skip()
    database = os.environ.get("DMX_DB_DATABASE", "default")
    table = f"dmi_training_e2e_{uuid.uuid4().hex}"
    model_id = f"megatron-e2e-{uuid.uuid4().hex}"
    engine = None

    client.execute(f"CREATE DATABASE IF NOT EXISTS `{database}`")
    client.execute(f"DROP TABLE IF EXISTS `{database}`.`{table}`")
    _create_training_table(client, database=database, table=table)

    try:
        engine = _build_training_engine(model_id=model_id, table=table, database=database)
        model = TinyMegatronHookModel().cuda().eval()
        record_runtime = engine.create_record_runtime(MegatronRecordFormat(table))
        adaptor = MegatronAdaptor(
            engine,
            record_runtime,
            model_id,
            dims={DimSpec.BATCH: 2, DimSpec.NUM_EXPERTS: 3},
        )
        adaptor.attach_model(model, hook_selection="router-summary,loss-summary")

        ctx = MegatronTrainingContext(
            model_id=model_id,
            global_batch_id=7,
            microbatch_id=1,
            valid_counts=[4, 2],
            direction="fwd",
            dp_rank=0,
            shard_rank=0,
            token_start=10,
        )
        payload = torch.tensor(
            [[0.10, 0.20, 0.70], [0.30, 0.30, 0.40]],
            dtype=torch.float32,
            device="cuda",
        )
        loss_payload = torch.tensor([[1.25], [2.5]], dtype=torch.float32, device="cuda")
        adaptor.set_current_event(ctx)
        try:
            with torch.no_grad():
                model(payload, loss_payload)
        finally:
            adaptor.clear_current_event()

        torch.cuda.synchronize()
        engine.flush_and_wait()

        rows = _wait_for_rows(
            model_id=model_id,
            table=table,
            database=database,
            expected_rows=2,
        )
        assert len(rows) == 2

        rows_by_sample = {key[7]: (key, tensor) for key, tensor in rows}
        assert set(rows_by_sample) == {0, 1}

        key0, tensor0 = rows_by_sample[0]
        key1, tensor1 = rows_by_sample[1]

        assert key0 == (
            model_id,
            "router_probs_mean",
            "fwd",
            "train",
            7,
            0,
            1,
            0,
            0,
            0,
            10,
            14,
            0,
            0,
            0,
        )
        assert key1 == (
            model_id,
            "router_probs_mean",
            "fwd",
            "train",
            7,
            0,
            1,
            1,
            0,
            0,
            10,
            12,
            0,
            0,
            0,
        )
        torch.testing.assert_close(tensor0, payload[0].cpu(), rtol=0, atol=0)
        torch.testing.assert_close(tensor1, payload[1].cpu(), rtol=0, atol=0)

        scalar_rows = _wait_for_scalar_rows(
            model_id=model_id,
            table=table,
            database=database,
            expected_rows=2,
        )
        assert len(scalar_rows) == 2
        scalar_by_sample = {key[7]: (key, value) for key, value in scalar_rows}
        assert set(scalar_by_sample) == {0, 1}
        scalar_key0, value0 = scalar_by_sample[0]
        scalar_key1, value1 = scalar_by_sample[1]
        assert scalar_key0 == (
            model_id,
            "lm_per_sample_loss",
            "fwd",
            "train",
            7,
            0,
            1,
            0,
            -1,
            0,
            0,
            1,
            0,
            0,
            0,
        )
        assert scalar_key1 == (
            model_id,
            "lm_per_sample_loss",
            "fwd",
            "train",
            7,
            0,
            1,
            1,
            -1,
            0,
            0,
            1,
            0,
            0,
            0,
        )
        assert value0 == pytest.approx(1.25)
        assert value1 == pytest.approx(2.5)

        scalar_int_count = client.execute(
            f"""
            SELECT count()
            FROM `{database}`.`{table}_scalar_int`
            WHERE model_id = %(model_id)s
            """,
            {"model_id": model_id},
        )
        assert scalar_int_count == [(0,)]
        boundary_count = client.execute(
            f"""
            SELECT count()
            FROM `{database}`.`{table}_eval_phase_boundary`
            WHERE model_id = %(model_id)s
            """,
            {"model_id": model_id},
        )
        assert boundary_count == [(0,)]
    finally:
        if engine is not None:
            engine.close()
        client.execute(f"DROP TABLE IF EXISTS `{database}`.`{table}`")
        client.execute(f"DROP TABLE IF EXISTS `{database}`.`{table}_scalar_float`")
        client.execute(f"DROP TABLE IF EXISTS `{database}`.`{table}_scalar_int`")
        client.execute(f"DROP TABLE IF EXISTS `{database}`.`{table}_eval_phase_boundary`")
        client.disconnect()
