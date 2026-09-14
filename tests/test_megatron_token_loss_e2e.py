"""Actual ring/storage coverage for the separate per-token loss hook."""

from __future__ import annotations

import os
import uuid

import pytest
import torch

from tests import test_megatron_real_training_e2e as e2e


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Megatron token-loss E2E needs CUDA")
@pytest.mark.parametrize(
    "tp,dp,pp,ep,with_summary,fused",
    [
        (1, 1, 1, 1, False, False),
        (1, 1, 1, 1, True, True),
        (1, 2, 1, 2, True, False),
        (2, 1, 1, 2, True, False),
        (1, 1, 2, 1, True, False),
    ],
    ids=["standalone", "with_summary_fused", "dp2_ep2", "tp2_sp_ep2", "pp2"],
)
def test_token_loss_training_records(tmp_path, tp, dp, pp, ep, with_summary, fused):
    nproc = tp * dp * pp
    if e2e._available_cuda_devices() < nproc:
        pytest.skip(f"requires {nproc} CUDA devices")
    client = e2e._clickhouse_client_or_skip()
    database = os.environ.get("DMX_DB_DATABASE", "default")
    table = f"dmi_token_loss_{uuid.uuid4().hex}"
    model_id = table
    batch = 2 * dp
    train_iters = 2
    expected = train_iters * batch
    selection = "token-loss,loss-summary" if with_summary else "token-loss"
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{e2e.ROOT}:{e2e.MEGATRON_ROOT}:{env.get('PYTHONPATH', '')}"
    env["DMI_ENABLE"] = "1"
    env.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", "1")
    if "DMI_REAL_E2E_CUDA_VISIBLE_DEVICES" in env:
        env["CUDA_VISIBLE_DEVICES"] = env["DMI_REAL_E2E_CUDA_VISIBLE_DEVICES"]
    cmd = e2e._tiny_megatron_router_summary_cmd(
        model_id=model_id, train_iters=train_iters, eval_iters=0,
        micro_batch_size=1, global_batch_size=batch,
        nproc_per_node=nproc, tp_size=tp, pp_size=pp, ep_size=ep,
        moe_token_dispatcher_type="alltoall" if ep > 1 else "allgather",
        transformer_impl="transformer_engine" if nproc > 1 else "local",
        database=database, table=table,
        extra_args=[
            "--dmi-hook-selection", selection,
            "--expert-tensor-parallel-size", "1", "--attention-backend", "unfused",
            *(["--sequence-parallel"] if tp > 1 else []),
            *(["--cross-entropy-loss-fusion", "--cross-entropy-fusion-impl", "native"] if fused else []),
        ],
    )
    try:
        client.execute(f"CREATE DATABASE IF NOT EXISTS `{database}`")
        e2e._create_training_table(client, database=database, table=table)
        e2e._run_megatron_cmd(cmd, env=env, log_path=tmp_path / "token_loss.log")
        e2e._wait_for_exact_act_rows(
            client, database=database, table=table, model_id=model_id,
            expected=expected, act_name="lm_per_token_loss",
        )
        rows = e2e._read_training_act_rows(
            database=database, table=table, model_id=model_id, act_name="lm_per_token_loss",
        )
        assert len(rows) == len(dict(rows)) == expected
        assert {key[5] for key, _ in rows} == set(range(dp))
        for key, value in rows:
            assert key[8] == -1 and key[9] == key[5]
            assert key[10:12] == (0, 16)
            assert tuple(value.shape) == (16,)
            assert value.dtype is torch.float32
            assert torch.isfinite(value).all() and (value > 0).all()
        if with_summary:
            e2e._wait_for_exact_scalar_rows(
                client, database=database, table=f"{table}_scalar_float", model_id=model_id,
                act_name="lm_per_sample_loss", expected=expected,
            )
            summaries = e2e._read_training_scalar_act_rows(
                database=database, table=table, model_id=model_id,
                direction="fwd", act_name="lm_per_sample_loss",
            )
            # Match sample identity; summary records have no per-token range.
            by_sample = {key[2:10]: float(value) for key, value in summaries}
            assert len(by_sample) == expected
            for key, value in rows:
                assert float(value.mean()) == pytest.approx(by_sample[key[2:10]], abs=1e-6)
        else:
            assert e2e._query_scalar_count(
                client, database=database, table=f"{table}_scalar_float", model_id=model_id,
                act_name="lm_per_sample_loss",
            ) == 0
    finally:
        for suffix in ("", "_scalar_float", "_scalar_int", "_eval_phase_boundary"):
            client.execute(f"DROP TABLE IF EXISTS `{database}`.`{table}{suffix}`")
        client.disconnect()
