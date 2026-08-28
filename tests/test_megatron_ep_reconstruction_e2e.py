from __future__ import annotations

import os
import uuid
from pathlib import Path

import pytest
import torch

from dmi_megatron_integration.materialization.ep_clickhouse_reconstruction import (
    reconstruct_moe_clickhouse_rows,
)
from dmi_megatron_integration.topology.ep_topology_manifest import load_ep_topology_manifest
from dmi_megatron_integration.records.reader import MegatronTrainingReader
from tests.test_megatron_e2e_clickhouse import (
    _clickhouse_client_or_skip,
    _create_training_table,
)
from tests.test_megatron_real_training_e2e import (
    MEGATRON_ROOT,
    ROOT,
    _available_cuda_devices,
    _run_megatron_cmd,
    _tiny_megatron_router_summary_cmd,
    _wait_for_exact_act_rows,
)


_ACT_NAMES = (
    "router_topk_expert_ids",
    "router_topk_weights",
    "moe_inverse_map",
    "moe_packed_weighted_output",
)


def _read_moe_rows(
    *,
    model_id: str,
    database: str,
    table: str,
) -> dict[str, list[tuple[tuple, torch.Tensor]]]:
    reader = MegatronTrainingReader(
        host=os.environ.get("DMX_DB_HOST", "localhost"),
        port=int(os.environ.get("DMX_DB_PORT", "9000")),
        username=os.environ.get("DMX_DB_USER", "default"),
        password=os.environ.get("DMX_DB_PASSWORD", ""),
        database=database,
        table=table,
    )
    try:
        return {
            act_name: reader.training_prefix_get(
                (model_id, act_name, "fwd", "train"),
                return_full_key_tuple=True,
            )
            for act_name in _ACT_NAMES
        }
    finally:
        reader.close()


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Real Megatron EP E2E needs CUDA")
def test_real_megatron_ep2_reconstructs_from_clickhouse(tmp_path: Path) -> None:
    if _available_cuda_devices() < 2:
        pytest.skip("MoE reconstruction EP=2 E2E requires two CUDA devices")

    client = _clickhouse_client_or_skip()
    database = os.environ.get("DMX_DB_DATABASE", "default")
    table = f"dmi_megatron_moe_reconstruction_ep2_e2e_{uuid.uuid4().hex}"
    model_id = f"megatron-moe-reconstruction-ep2-e2e-{uuid.uuid4().hex}"
    log_path = tmp_path / "megatron_moe_reconstruction_ep2.log"
    oracle_dir = tmp_path / "native_moe_oracle"
    topology_manifest_path = tmp_path / "topology.json"

    client.execute(f"CREATE DATABASE IF NOT EXISTS `{database}`")
    client.execute(f"DROP TABLE IF EXISTS `{database}`.`{table}`")
    _create_training_table(client, database=database, table=table)

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{ROOT}:{MEGATRON_ROOT}:{env.get('PYTHONPATH', '')}"
    env["DMI_ENABLE"] = "1"
    env["DMI_EP_ORACLE_DIR"] = str(oracle_dir)
    env["DMI_TOPOLOGY_MANIFEST_PATH"] = str(topology_manifest_path)
    env.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", "1")
    if "DMI_REAL_E2E_CUDA_VISIBLE_DEVICES" in env:
        env["CUDA_VISIBLE_DEVICES"] = env["DMI_REAL_E2E_CUDA_VISIBLE_DEVICES"]

    cmd = _tiny_megatron_router_summary_cmd(
        model_id=model_id,
        train_iters=1,
        micro_batch_size=1,
        global_batch_size=2,
        nproc_per_node=2,
        ep_size=2,
        num_experts=4,
        moe_router_topk=2,
        moe_token_dispatcher_type="alltoall",
        database=database,
        table=table,
        extra_args=[
            "--dmi-hook-selection",
            "router-topk,moe-inverse-map,moe-packed-weighted-output",
        ],
    )
    entrypoint_index = cmd.index("pretrain_gpt.py")
    cmd[entrypoint_index] = str(
        ROOT / "tests" / "oracles" / "run_megatron_ep_reconstruction_oracle.py"
    )

    try:
        _run_megatron_cmd(cmd, env=env, log_path=log_path)
        topology_manifest = load_ep_topology_manifest(topology_manifest_path)
        assert topology_manifest.model_id == model_id
        assert topology_manifest.dp_groups == ((0, 1),)
        for act_name in _ACT_NAMES:
            _wait_for_exact_act_rows(
                client,
                database=database,
                table=table,
                model_id=model_id,
                act_name=act_name,
                expected=4,
            )

        rows = _read_moe_rows(model_id=model_id, database=database, table=table)
        reconstructed = reconstruct_moe_clickhouse_rows(topology_manifest, rows)
        reconstructed_by_layer = {
            invocation.key.layer_no: invocation for invocation in reconstructed
        }
        assert set(reconstructed_by_layer) == {0, 1}

        oracle_by_rank: dict[int, dict[int, list[torch.Tensor]]] = {}
        for producer_rank in (0, 1):
            oracle_path = oracle_dir / f"rank_{producer_rank}.pt"
            assert oracle_path.is_file()
            oracle_by_rank[producer_rank] = torch.load(
                oracle_path,
                map_location="cpu",
                weights_only=True,
            )

        for layer_no in (0, 1):
            topology = topology_manifest.topology_for_layer(layer_no)
            domains_by_dp = {
                domain.dense_dp_rank: domain
                for domain in reconstructed_by_layer[layer_no].source_domains
            }
            assert set(domains_by_dp) == {0, 1}
            for producer_rank in (0, 1):
                oracle_calls = oracle_by_rank[producer_rank].get(layer_no)
                assert oracle_calls is not None
                assert len(oracle_calls) == 1
                native_output = oracle_calls[0]
                assert native_output.ndim == 3
                native_flat = native_output.reshape(-1, native_output.shape[-1])
                reconstructed_output = domains_by_dp[
                    topology.dense_dp_rank_by_global_rank[producer_rank]
                ].combined_output.cpu()
                assert tuple(reconstructed_output.shape) == tuple(native_flat.shape)
                assert torch.equal(reconstructed_output, native_flat)
    finally:
        client.execute(f"DROP TABLE IF EXISTS `{database}`.`{table}`")
        client.execute(f"DROP TABLE IF EXISTS `{database}`.`{table}_scalar_float`")
        client.execute(f"DROP TABLE IF EXISTS `{database}`.`{table}_scalar_int`")
        client.execute(f"DROP TABLE IF EXISTS `{database}`.`{table}_eval_phase_boundary`")
        client.disconnect()


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Real Megatron EP E2E needs CUDA")
def test_real_megatron_tp2_sp_etp2_reconstructs_from_clickhouse(tmp_path: Path) -> None:
    """Reconstruct sequence-sharded inputs and ETP-summed expert outputs."""

    if _available_cuda_devices() < 2:
        pytest.skip("MoE reconstruction TP=2/SP/ETP=2 E2E requires two CUDA devices")

    client = _clickhouse_client_or_skip()
    database = os.environ.get("DMX_DB_DATABASE", "default")
    table = f"dmi_megatron_moe_reconstruction_tp2_sp_etp2_e2e_{uuid.uuid4().hex}"
    model_id = f"megatron-moe-reconstruction-tp2-sp-etp2-e2e-{uuid.uuid4().hex}"
    log_path = tmp_path / "megatron_moe_reconstruction_tp2_sp_etp2.log"
    oracle_dir = tmp_path / "native_moe_oracle"
    topology_manifest_path = tmp_path / "topology.json"

    client.execute(f"CREATE DATABASE IF NOT EXISTS `{database}`")
    client.execute(f"DROP TABLE IF EXISTS `{database}`.`{table}`")
    _create_training_table(client, database=database, table=table)

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{ROOT}:{MEGATRON_ROOT}:{env.get('PYTHONPATH', '')}"
    env["DMI_ENABLE"] = "1"
    env["DMI_EP_ORACLE_DIR"] = str(oracle_dir)
    env["DMI_TOPOLOGY_MANIFEST_PATH"] = str(topology_manifest_path)
    env.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", "1")
    if "DMI_REAL_E2E_CUDA_VISIBLE_DEVICES" in env:
        env["CUDA_VISIBLE_DEVICES"] = env["DMI_REAL_E2E_CUDA_VISIBLE_DEVICES"]

    cmd = _tiny_megatron_router_summary_cmd(
        model_id=model_id,
        train_iters=1,
        micro_batch_size=1,
        global_batch_size=1,
        nproc_per_node=2,
        tp_size=2,
        ep_size=1,
        num_experts=4,
        moe_router_topk=2,
        moe_token_dispatcher_type="alltoall",
        transformer_impl="transformer_engine",
        database=database,
        table=table,
        extra_args=[
            "--context-parallel-size",
            "1",
            "--expert-tensor-parallel-size",
            "2",
            "--sequence-parallel",
            "--dmi-hook-selection",
            "router-topk,moe-inverse-map,moe-packed-weighted-output",
        ],
    )
    entrypoint_index = cmd.index("pretrain_gpt.py")
    cmd[entrypoint_index] = str(
        ROOT / "tests" / "oracles" / "run_megatron_ep_reconstruction_oracle.py"
    )

    try:
        _run_megatron_cmd(cmd, env=env, log_path=log_path)
        topology_manifest = load_ep_topology_manifest(topology_manifest_path)
        assert topology_manifest.model_id == model_id
        assert topology_manifest.sequence_parallel
        assert topology_manifest.tp_groups == ((0, 1),)
        assert topology_manifest.etp_groups == ((0, 1),)
        assert topology_manifest.ep_groups == ((0,), (1,))
        for act_name in _ACT_NAMES:
            _wait_for_exact_act_rows(
                client,
                database=database,
                table=table,
                model_id=model_id,
                act_name=act_name,
                expected=4,
            )

        rows = _read_moe_rows(model_id=model_id, database=database, table=table)
        reconstructed = reconstruct_moe_clickhouse_rows(topology_manifest, rows)
        reconstructed_by_step = {}
        for invocation in reconstructed:
            step = (invocation.key.layer_no, invocation.key.microbatch_id)
            assert step not in reconstructed_by_step
            reconstructed_by_step[step] = invocation
        assert set(reconstructed_by_step) == {
            (layer_no, 0)
            for layer_no in (0, 1)
        }

        oracle_by_rank: dict[int, dict[int, list[torch.Tensor]]] = {}
        for producer_rank in (0, 1):
            oracle_path = oracle_dir / f"rank_{producer_rank}.pt"
            assert oracle_path.is_file()
            oracle_by_rank[producer_rank] = torch.load(
                oracle_path,
                map_location="cpu",
                weights_only=True,
            )

        for layer_no in (0, 1):
            for producer_rank in (0, 1):
                oracle_calls = oracle_by_rank[producer_rank].get(layer_no)
                assert oracle_calls is not None
                assert len(oracle_calls) == 1
            tp_group = topology_manifest.tp_groups[0]
            native_output = torch.cat(
                [oracle_by_rank[producer_rank][layer_no][0] for producer_rank in tp_group],
                dim=0,
            )
            native_flat = native_output.reshape(-1, native_output.shape[-1])
            invocation = reconstructed_by_step[(layer_no, 0)]
            assert len(invocation.source_domains) == 1
            reconstructed_output = invocation.source_domains[0].combined_output.cpu()
            assert tuple(reconstructed_output.shape) == tuple(native_flat.shape)
            assert torch.equal(reconstructed_output, native_flat)
    finally:
        client.execute(f"DROP TABLE IF EXISTS `{database}`.`{table}`")
        client.execute(f"DROP TABLE IF EXISTS `{database}`.`{table}_scalar_float`")
        client.execute(f"DROP TABLE IF EXISTS `{database}`.`{table}_scalar_int`")
        client.execute(f"DROP TABLE IF EXISTS `{database}`.`{table}_eval_phase_boundary`")
        client.disconnect()


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Real Megatron EP E2E needs CUDA")
def test_real_megatron_tp2_sp_ep2_reconstructs_from_clickhouse(tmp_path: Path) -> None:
    """Reconstruct SP-sharded source tokens across an EP=2 dispatch group."""

    if _available_cuda_devices() < 2:
        pytest.skip("MoE reconstruction TP=2/SP/EP=2 E2E requires two CUDA devices")

    client = _clickhouse_client_or_skip()
    database = os.environ.get("DMX_DB_DATABASE", "default")
    table = f"dmi_megatron_moe_reconstruction_tp2_sp_ep2_e2e_{uuid.uuid4().hex}"
    model_id = f"megatron-moe-reconstruction-tp2-sp-ep2-e2e-{uuid.uuid4().hex}"
    log_path = tmp_path / "megatron_moe_reconstruction_tp2_sp_ep2.log"
    oracle_dir = tmp_path / "native_moe_oracle"
    topology_manifest_path = tmp_path / "topology.json"

    client.execute(f"CREATE DATABASE IF NOT EXISTS `{database}`")
    client.execute(f"DROP TABLE IF EXISTS `{database}`.`{table}`")
    _create_training_table(client, database=database, table=table)

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{ROOT}:{MEGATRON_ROOT}:{env.get('PYTHONPATH', '')}"
    env["DMI_ENABLE"] = "1"
    env["DMI_EP_ORACLE_DIR"] = str(oracle_dir)
    env["DMI_TOPOLOGY_MANIFEST_PATH"] = str(topology_manifest_path)
    env.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", "1")
    if "DMI_REAL_E2E_CUDA_VISIBLE_DEVICES" in env:
        env["CUDA_VISIBLE_DEVICES"] = env["DMI_REAL_E2E_CUDA_VISIBLE_DEVICES"]

    cmd = _tiny_megatron_router_summary_cmd(
        model_id=model_id,
        train_iters=1,
        micro_batch_size=1,
        global_batch_size=1,
        nproc_per_node=2,
        tp_size=2,
        ep_size=2,
        num_experts=4,
        moe_router_topk=2,
        moe_token_dispatcher_type="alltoall",
        transformer_impl="transformer_engine",
        database=database,
        table=table,
        extra_args=[
            "--context-parallel-size",
            "1",
            "--expert-tensor-parallel-size",
            "1",
            "--sequence-parallel",
            "--dmi-hook-selection",
            "router-topk,moe-inverse-map,moe-packed-weighted-output",
        ],
    )
    entrypoint_index = cmd.index("pretrain_gpt.py")
    cmd[entrypoint_index] = str(
        ROOT / "tests" / "oracles" / "run_megatron_ep_reconstruction_oracle.py"
    )

    try:
        _run_megatron_cmd(cmd, env=env, log_path=log_path)
        topology_manifest = load_ep_topology_manifest(topology_manifest_path)
        assert topology_manifest.model_id == model_id
        assert topology_manifest.sequence_parallel
        assert topology_manifest.tp_groups == ((0, 1),)
        assert topology_manifest.ep_groups == ((0, 1),)
        assert topology_manifest.etp_groups == ((0,), (1,))
        assert topology_manifest.dispatch_groups == ((0, 1),)
        for act_name in _ACT_NAMES:
            _wait_for_exact_act_rows(
                client,
                database=database,
                table=table,
                model_id=model_id,
                act_name=act_name,
                expected=4,
            )

        rows = _read_moe_rows(model_id=model_id, database=database, table=table)
        reconstructed = reconstruct_moe_clickhouse_rows(topology_manifest, rows)
        reconstructed_by_step = {}
        for invocation in reconstructed:
            step = (invocation.key.layer_no, invocation.key.microbatch_id)
            assert step not in reconstructed_by_step
            reconstructed_by_step[step] = invocation
        assert set(reconstructed_by_step) == {(layer_no, 0) for layer_no in (0, 1)}

        oracle_by_rank: dict[int, dict[int, list[torch.Tensor]]] = {}
        for producer_rank in (0, 1):
            oracle_path = oracle_dir / f"rank_{producer_rank}.pt"
            assert oracle_path.is_file()
            oracle_by_rank[producer_rank] = torch.load(
                oracle_path,
                map_location="cpu",
                weights_only=True,
            )

        for layer_no in (0, 1):
            for producer_rank in (0, 1):
                oracle_calls = oracle_by_rank[producer_rank].get(layer_no)
                assert oracle_calls is not None
                assert len(oracle_calls) == 1
            native_output = torch.cat(
                [oracle_by_rank[producer_rank][layer_no][0] for producer_rank in (0, 1)],
                dim=0,
            )
            native_flat = native_output.reshape(-1, native_output.shape[-1])
            invocation = reconstructed_by_step[(layer_no, 0)]
            assert len(invocation.source_domains) == 1
            reconstructed_output = invocation.source_domains[0].combined_output.cpu()
            assert tuple(reconstructed_output.shape) == tuple(native_flat.shape)
            assert torch.equal(reconstructed_output, native_flat)
    finally:
        client.execute(f"DROP TABLE IF EXISTS `{database}`.`{table}`")
        client.execute(f"DROP TABLE IF EXISTS `{database}`.`{table}_scalar_float`")
        client.execute(f"DROP TABLE IF EXISTS `{database}`.`{table}_scalar_int`")
        client.execute(f"DROP TABLE IF EXISTS `{database}`.`{table}_eval_phase_boundary`")
        client.disconnect()


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Real Megatron PP E2E needs CUDA")
def test_real_megatron_pp2_reconstructs_owner_layers_from_clickhouse(tmp_path: Path) -> None:
    """Reconstruct each global MoE layer from its owning pipeline stage."""

    if _available_cuda_devices() < 2:
        pytest.skip("MoE reconstruction PP=2 E2E requires two CUDA devices")

    client = _clickhouse_client_or_skip()
    database = os.environ.get("DMX_DB_DATABASE", "default")
    table = f"dmi_megatron_moe_reconstruction_pp2_e2e_{uuid.uuid4().hex}"
    model_id = f"megatron-moe-reconstruction-pp2-e2e-{uuid.uuid4().hex}"
    log_path = tmp_path / "megatron_moe_reconstruction_pp2.log"
    oracle_dir = tmp_path / "native_moe_oracle"
    topology_manifest_path = tmp_path / "topology.json"

    client.execute(f"CREATE DATABASE IF NOT EXISTS `{database}`")
    client.execute(f"DROP TABLE IF EXISTS `{database}`.`{table}`")
    _create_training_table(client, database=database, table=table)

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{ROOT}:{MEGATRON_ROOT}:{env.get('PYTHONPATH', '')}"
    env["DMI_ENABLE"] = "1"
    env["DMI_EP_ORACLE_DIR"] = str(oracle_dir)
    env["DMI_TOPOLOGY_MANIFEST_PATH"] = str(topology_manifest_path)
    env.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", "1")
    if "DMI_REAL_E2E_CUDA_VISIBLE_DEVICES" in env:
        env["CUDA_VISIBLE_DEVICES"] = env["DMI_REAL_E2E_CUDA_VISIBLE_DEVICES"]

    cmd = _tiny_megatron_router_summary_cmd(
        model_id=model_id,
        train_iters=1,
        micro_batch_size=1,
        global_batch_size=1,
        nproc_per_node=2,
        pp_size=2,
        ep_size=1,
        num_experts=4,
        moe_router_topk=2,
        moe_token_dispatcher_type="alltoall",
        transformer_impl="transformer_engine",
        database=database,
        table=table,
        extra_args=[
            "--context-parallel-size",
            "1",
            "--dmi-hook-selection",
            "router-topk,moe-inverse-map,moe-packed-weighted-output",
        ],
    )
    entrypoint_index = cmd.index("pretrain_gpt.py")
    cmd[entrypoint_index] = str(
        ROOT / "tests" / "oracles" / "run_megatron_ep_reconstruction_oracle.py"
    )

    try:
        _run_megatron_cmd(cmd, env=env, log_path=log_path)
        topology_manifest = load_ep_topology_manifest(topology_manifest_path)
        assert topology_manifest.model_id == model_id
        assert topology_manifest.pp_groups == ((0, 1),)
        assert {
            (placement.layer_no, placement.pp_rank)
            for placement in topology_manifest.layer_placements
        } == {(0, 0), (1, 1)}
        for act_name in _ACT_NAMES:
            _wait_for_exact_act_rows(
                client,
                database=database,
                table=table,
                model_id=model_id,
                act_name=act_name,
                expected=2,
            )

        rows = _read_moe_rows(model_id=model_id, database=database, table=table)
        reconstructed = reconstruct_moe_clickhouse_rows(topology_manifest, rows)
        reconstructed_by_layer = {
            invocation.key.layer_no: invocation for invocation in reconstructed
        }
        assert set(reconstructed_by_layer) == {0, 1}

        oracle_by_rank: dict[int, dict[int, list[torch.Tensor]]] = {}
        for producer_rank in (0, 1):
            oracle_path = oracle_dir / f"rank_{producer_rank}.pt"
            assert oracle_path.is_file()
            oracle_by_rank[producer_rank] = torch.load(
                oracle_path,
                map_location="cpu",
                weights_only=True,
            )

        pp_group = topology_manifest.pp_groups[0]
        for placement in topology_manifest.layer_placements:
            owner_rank = pp_group[placement.pp_rank]
            oracle_calls = oracle_by_rank[owner_rank].get(placement.layer_no)
            assert oracle_calls is not None
            assert len(oracle_calls) == 1
            native_output = oracle_calls[0]
            native_flat = native_output.reshape(-1, native_output.shape[-1])
            invocation = reconstructed_by_layer[placement.layer_no]
            assert len(invocation.source_domains) == 1
            reconstructed_output = invocation.source_domains[0].combined_output.cpu()
            assert tuple(reconstructed_output.shape) == tuple(native_flat.shape)
            assert torch.equal(reconstructed_output, native_flat)
            for other_rank in set(pp_group) - {owner_rank}:
                assert placement.layer_no not in oracle_by_rank[other_rank]
    finally:
        client.execute(f"DROP TABLE IF EXISTS `{database}`.`{table}`")
        client.execute(f"DROP TABLE IF EXISTS `{database}`.`{table}_scalar_float`")
        client.execute(f"DROP TABLE IF EXISTS `{database}`.`{table}_scalar_int`")
        client.execute(f"DROP TABLE IF EXISTS `{database}`.`{table}_eval_phase_boundary`")
        client.disconnect()


@pytest.mark.slow
@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Real Megatron expert-DP E2E needs CUDA"
)
def test_real_megatron_expert_dp2_keeps_dispatch_groups_isolated(tmp_path: Path) -> None:
    """Reconstruct two independent expert-DP replicas without cross-group joins."""

    if _available_cuda_devices() < 2:
        pytest.skip("MoE reconstruction expert-DP=2 E2E requires two CUDA devices")

    client = _clickhouse_client_or_skip()
    database = os.environ.get("DMX_DB_DATABASE", "default")
    table = f"dmi_megatron_moe_reconstruction_expert_dp2_e2e_{uuid.uuid4().hex}"
    model_id = f"megatron-moe-reconstruction-expert-dp2-e2e-{uuid.uuid4().hex}"
    log_path = tmp_path / "megatron_moe_reconstruction_expert_dp2.log"
    oracle_dir = tmp_path / "native_moe_oracle"
    topology_manifest_path = tmp_path / "topology.json"

    client.execute(f"CREATE DATABASE IF NOT EXISTS `{database}`")
    client.execute(f"DROP TABLE IF EXISTS `{database}`.`{table}`")
    _create_training_table(client, database=database, table=table)

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{ROOT}:{MEGATRON_ROOT}:{env.get('PYTHONPATH', '')}"
    env["DMI_ENABLE"] = "1"
    env["DMI_EP_ORACLE_DIR"] = str(oracle_dir)
    env["DMI_TOPOLOGY_MANIFEST_PATH"] = str(topology_manifest_path)
    env.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", "1")
    if "DMI_REAL_E2E_CUDA_VISIBLE_DEVICES" in env:
        env["CUDA_VISIBLE_DEVICES"] = env["DMI_REAL_E2E_CUDA_VISIBLE_DEVICES"]

    cmd = _tiny_megatron_router_summary_cmd(
        model_id=model_id,
        train_iters=1,
        micro_batch_size=1,
        global_batch_size=2,
        nproc_per_node=2,
        ep_size=1,
        num_experts=4,
        moe_router_topk=2,
        moe_token_dispatcher_type="alltoall",
        transformer_impl="transformer_engine",
        database=database,
        table=table,
        extra_args=[
            "--context-parallel-size",
            "1",
            "--dmi-hook-selection",
            "router-topk,moe-inverse-map,moe-packed-weighted-output",
        ],
    )
    entrypoint_index = cmd.index("pretrain_gpt.py")
    cmd[entrypoint_index] = str(
        ROOT / "tests" / "oracles" / "run_megatron_ep_reconstruction_oracle.py"
    )

    try:
        _run_megatron_cmd(cmd, env=env, log_path=log_path)
        topology_manifest = load_ep_topology_manifest(topology_manifest_path)
        assert topology_manifest.model_id == model_id
        assert topology_manifest.dp_groups == ((0, 1),)
        assert topology_manifest.expert_dp_groups == ((0, 1),)
        assert topology_manifest.ep_groups == ((0,), (1,))
        assert topology_manifest.etp_groups == ((0,), (1,))
        assert topology_manifest.dispatch_groups == ((0,), (1,))
        for act_name in _ACT_NAMES:
            _wait_for_exact_act_rows(
                client,
                database=database,
                table=table,
                model_id=model_id,
                act_name=act_name,
                expected=4,
            )

        rows = _read_moe_rows(model_id=model_id, database=database, table=table)
        reconstructed = reconstruct_moe_clickhouse_rows(topology_manifest, rows)
        reconstructed_by_layer = {
            invocation.key.layer_no: invocation for invocation in reconstructed
        }
        assert set(reconstructed_by_layer) == {0, 1}

        oracle_by_rank: dict[int, dict[int, list[torch.Tensor]]] = {}
        for producer_rank in (0, 1):
            oracle_path = oracle_dir / f"rank_{producer_rank}.pt"
            assert oracle_path.is_file()
            oracle_by_rank[producer_rank] = torch.load(
                oracle_path,
                map_location="cpu",
                weights_only=True,
            )

        for layer_no in (0, 1):
            topology = topology_manifest.topology_for_layer(layer_no)
            assert topology.expert_dp_rank_by_global_rank == {0: 0, 1: 1}
            domains_by_dp = {
                domain.dense_dp_rank: domain
                for domain in reconstructed_by_layer[layer_no].source_domains
            }
            assert set(domains_by_dp) == {0, 1}
            for producer_rank in (0, 1):
                oracle_calls = oracle_by_rank[producer_rank].get(layer_no)
                assert oracle_calls is not None
                assert len(oracle_calls) == 1
                native_output = oracle_calls[0]
                native_flat = native_output.reshape(-1, native_output.shape[-1])
                dense_dp_rank = topology.dense_dp_rank_by_global_rank[producer_rank]
                reconstructed_output = domains_by_dp[dense_dp_rank].combined_output.cpu()
                assert tuple(reconstructed_output.shape) == tuple(native_flat.shape)
                assert torch.equal(reconstructed_output, native_flat)
    finally:
        client.execute(f"DROP TABLE IF EXISTS `{database}`.`{table}`")
        client.execute(f"DROP TABLE IF EXISTS `{database}`.`{table}_scalar_float`")
        client.execute(f"DROP TABLE IF EXISTS `{database}`.`{table}_scalar_int`")
        client.execute(f"DROP TABLE IF EXISTS `{database}`.`{table}_eval_phase_boundary`")
        client.disconnect()
