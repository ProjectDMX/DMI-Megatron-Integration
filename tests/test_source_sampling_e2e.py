"""Sampled EP transport and reconstruction against native Megatron outputs."""
from __future__ import annotations

import os
import uuid
from pathlib import Path

import pytest
import torch
import yaml

from dmi_megatron_integration.hooks.source_sampling import BUILTIN_ROUND_ROBIN, EP_OUTPUT, SourceSampling
from dmi_megatron_integration.materialization.ep_clickhouse_reconstruction import reconstruct_moe_clickhouse_rows
from dmi_megatron_integration.topology.ep_topology_manifest import load_ep_topology_manifest
from tests.test_megatron_ep_reconstruction_e2e import _read_moe_rows
from tests.test_megatron_e2e_clickhouse import _clickhouse_client_or_skip, _create_training_table
from tests.test_megatron_real_training_e2e import (
    ROOT, MEGATRON_ROOT, _available_cuda_devices, _run_megatron_cmd,
    _tiny_megatron_router_summary_cmd, _wait_for_exact_act_rows,
)


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
@pytest.mark.parametrize("graph", [False, True], ids=["eager", "te_graph"])
@pytest.mark.parametrize("world,tp,pp,ep,etp,experts,count", [
    (1, 1, 1, 1, 1, 1, 1), (1, 1, 1, 1, 1, 2, 1),
    (2, 1, 1, 2, 1, 2, 1), (2, 1, 1, 2, 1, 4, 1),
    (2, 1, 1, 2, 1, 4, 2), (2, 2, 1, 1, 2, 2, 1),
    (2, 2, 1, 2, 1, 4, 1), (2, 1, 2, 1, 1, 2, 1),
    (2, 1, 1, 1, 1, 2, 1),
    (1, 1, 1, 1, 1, 2, None),
    (2, 1, 1, 2, 1, 4, None), (2, 2, 1, 1, 2, 2, None),
])
def test_sampled_training_reconstructs_native(tmp_path: Path, world, tp, pp, ep, etp, experts, count, graph, layer_indices=None, router_selection="router-topk"):
    if _available_cuda_devices() < world:
        pytest.skip(f"Requires {world} available GPUs")
    if graph and experts == 1:
        pytest.skip("Megatron rejects MoE graph scopes for a one-expert model")
    client = _clickhouse_client_or_skip()
    database = os.environ.get("DMX_DB_DATABASE", "default")
    table = f"dmi_sampled_ep_test_{uuid.uuid4().hex}"
    model_id = table
    oracle_dir = tmp_path / "oracle"
    manifest_path = tmp_path / "topology.json"
    config = tmp_path / "hooks.yaml"
    policy = SourceSampling(BUILTIN_ROUND_ROBIN, {"count": count}) if count is not None else None
    config.write_text(yaml.safe_dump({"hooks": {EP_OUTPUT: {
        "source_sampling": policy.to_dict() if policy else None,
    }}}))
    dp = world // (tp * pp)
    args = ["--expert-tensor-parallel-size", str(etp), "--context-parallel-size", "1",
            "--dmi-hook-selection", f"{router_selection},moe-inverse-map,moe-packed-weighted-output",
            "--dmi-hook-config", str(config), "--log-interval", "1", "--split", "100,0,0",
            "--attention-dropout", "0", "--hidden-dropout", "0",
            "--attention-backend", "fused"]
    if layer_indices is not None:
        args += ["--dmi-layer-indices", *map(str, layer_indices),
                 "--dmi-no-recompute-hook", "moe-packed-weighted-output"]
    if graph:
        args += ["--cuda-graph-impl", "transformer_engine", "--cuda-graph-scope",
                 "attn", "moe_preprocess", "moe_router", "--cuda-graph-warmup-steps", "1"]
    if tp > 1:
        args += ["--sequence-parallel"]
    cmd = _tiny_megatron_router_summary_cmd(
        model_id=model_id, train_iters=3, micro_batch_size=1, global_batch_size=dp,
        nproc_per_node=world, tp_size=tp, pp_size=pp, ep_size=ep,
        num_experts=experts, moe_router_topk=min(2, experts) if world > 1 else 1,
        moe_token_dispatcher_type="alltoall", transformer_impl="transformer_engine",
        database=database, table=table, extra_args=args,
    )
    cmd[cmd.index("pretrain_gpt.py")] = str(ROOT / "tests/oracles/run_megatron_ep_reconstruction_oracle.py")
    env = os.environ.copy()
    env.update(PYTHONPATH=f"{ROOT}:{MEGATRON_ROOT}:{env.get('PYTHONPATH', '')}",
               DMI_ENABLE="1", DMI_EP_ORACLE_DIR=str(oracle_dir),
               DMI_TOPOLOGY_MANIFEST_PATH=str(manifest_path), CUDA_DEVICE_MAX_CONNECTIONS="1")
    if "DMI_REAL_E2E_CUDA_VISIBLE_DEVICES" in env:
        env["CUDA_VISIBLE_DEVICES"] = env["DMI_REAL_E2E_CUDA_VISIBLE_DEVICES"]
    client.execute(f"CREATE DATABASE IF NOT EXISTS `{database}`")
    _create_training_table(client, database=database, table=table)
    try:
        _run_megatron_cmd(cmd, env=env, log_path=tmp_path / "training.log")
        manifest = load_ep_topology_manifest(manifest_path)
        assert manifest.hook_capture[EP_OUTPUT]["source_sampling"] == (policy.to_dict() if policy else None)
        for name in ("router_topk_expert_ids", "moe_inverse_map", EP_OUTPUT):
            _wait_for_exact_act_rows(client, database=database, table=table, model_id=model_id,
                                    act_name=name, expected=3 * (world // pp) * (2 if layer_indices is None else len(layer_indices)))
        rows = _read_moe_rows(model_id=model_id, database=database, table=table)
        if router_selection == "router-topk-expert-ids":
            # The shared reader also queries optional routing weights. They
            # must be absent when only IDs were requested.
            assert rows.pop("router_topk_weights") == []
        invocations = reconstruct_moe_clickhouse_rows(manifest, rows)
        assert len(invocations) == 3 * (2 if layer_indices is None else len(layer_indices))
        if layer_indices is not None:
            assert {invocation.key.layer_no for invocation in invocations} == set(layer_indices)
        oracle = {r: torch.load(oracle_dir / f"rank_{r}.pt", map_location="cpu", weights_only=True)
                  for r in range(world)}
        for invocation in invocations:
            layer, batch = invocation.key.layer_no, invocation.key.global_batch_id
            topology = manifest.topology_for_layer(layer)
            selected = {topology.ordered_sources(group)[u] for group in topology.dispatch_groups
                        for u in (policy.select(batch, len(group)) if policy else range(len(group)))}
            expected_tokens = set()
            for rank in selected:
                dp_rank = topology.dense_dp_rank_by_global_rank[rank]
                tp_rank = next(group.index(rank) for group in manifest.tp_groups if rank in group)
                local = oracle[rank][layer][batch - 1].reshape(-1, 64)
                expected_tokens.update((dp_rank, tp_rank * local.shape[0] + i) for i in range(local.shape[0]))
            observed = set()
            for domain in invocation.source_domains:
                group = next(group for group in manifest.tp_groups
                             if group[0] in topology.dense_dp_rank_by_global_rank
                             and topology.dense_dp_rank_by_global_rank[group[0]] == domain.dense_dp_rank)
                native = torch.cat([oracle[r][layer][batch - 1] for r in group], dim=0).reshape(-1, 64)
                positions = [coord.token_index for coord in domain.token_coordinates]
                observed.update((domain.dense_dp_rank, i) for i in positions)
                torch.testing.assert_close(domain.combined_output.cpu(), native[positions], rtol=0, atol=0)
            assert observed == expected_tokens
    finally:
        tables = client.execute("SELECT name FROM system.tables WHERE database=%(db)s AND startsWith(name, %(prefix)s)",
                                {"db": database, "prefix": table})
        for (name,) in tables:
            client.execute(f"DROP TABLE `{database}`.`{name}`")
        client.disconnect()


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
@pytest.mark.parametrize("graph", [False, True], ids=["eager", "te_graph"])
@pytest.mark.parametrize("count", [1, None], ids=["sampled", "full"])
def test_expert_ids_only_training_reconstructs_native(tmp_path, graph, count):
    # EP reconstruction needs IDs without routing weights. Exercise the actual
    # single-output hook through transport, storage, and reconstruction.
    test_sampled_training_reconstructs_native(
        tmp_path, world=2, tp=2, pp=1, ep=2, etp=1, experts=4,
        count=count, graph=graph, router_selection="router-topk-expert-ids",
    )


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
@pytest.mark.parametrize("graph", [False, True], ids=["eager", "te_graph"])
@pytest.mark.parametrize("pp,count,experts", [(1, None, 4), (1, 1, 4), (2, None, 4), (2, 1, 4), (1, 1, 2)])
def test_layer_filtered_ep_reconstructs_native(tmp_path, graph, pp, count, experts):
    # PP=2 selects the last stage only; the first stage must still complete
    # setup/readiness with no selected EP hooks and no local recompute match.
    test_sampled_training_reconstructs_native(tmp_path, world=2, tp=1, pp=pp,
        ep=2 if pp == 1 else 1, etp=1, experts=experts, count=count, graph=graph,
        layer_indices=(1,))
