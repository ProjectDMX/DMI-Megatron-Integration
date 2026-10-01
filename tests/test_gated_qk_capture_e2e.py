"""Opt-in real hybrid-model training and Q/K transport/reconstruction checks."""
import os
import uuid

import pytest
import torch

from tests.test_megatron_e2e_clickhouse import _clickhouse_client_or_skip
from tests.test_megatron_real_training_e2e import (
    ROOT, MEGATRON_ROOT, _tiny_megatron_router_summary_cmd, _run_megatron_cmd,
    _read_merged_weight_rows,
)


pytestmark = pytest.mark.skipif(os.environ.get('DMI_TEST_GATED_QK_E2E') != '1',
                               reason='Set DMI_TEST_GATED_QK_E2E=1 for hybrid training')


@pytest.mark.parametrize('world,tp,pp,graph', [
    (2, 2, 1, False), (2, 2, 1, True),
    (2, 1, 1, False), (2, 1, 2, False),
])
def test_hybrid_qk_matches_preupdate_model_weights(tmp_path, world, tp, pp, graph):
    if torch.cuda.device_count() < world:
        pytest.skip('Two GPUs required')
    client = _clickhouse_client_or_skip()
    database = 'gated_qk_' + uuid.uuid4().hex
    oracle = tmp_path / 'oracle'
    extra = [
        '--num-layers', '8', '--hidden-size', '128', '--ffn-hidden-size', '256',
        '--num-attention-heads', '8', '--kv-channels', '32',
        '--group-query-attention', '--num-query-groups', '1', '--qk-layernorm',
        '--attention-output-gate', '--experimental-attention-variant', 'gated_delta_net',
        '--linear-attention-freq', '4', '--linear-key-head-dim', '32',
        '--linear-value-head-dim', '32', '--linear-num-key-heads', '4',
        '--linear-num-value-heads', '8', '--linear-conv-kernel-dim', '4',
        '--moe-ffn-hidden-size', '64', '--moe-shared-expert-intermediate-size', '64',
        '--moe-shared-expert-gate', '--moe-grouped-gemm', '--moe-router-dtype', 'fp32',
        '--expert-tensor-parallel-size', str(tp),
        '--normalization', 'RMSNorm', '--position-embedding-type', 'rope',
        '--rotary-percent', '0.25', '--rotary-base', '10000000',
        '--untie-embeddings-and-output-weights', '--attention-backend', 'fused',
        '--seq-length', '64', '--max-position-embeddings', '64',
        '--dmi-hook-selection', 'q-weights,k-weights',
        '--log-interval', '1', '--split', '100,0,0', '--num-workers', '0',
        '--optimizer', 'adam', '--context-parallel-size', '1',
        '--attention-dropout', '0', '--hidden-dropout', '0',
        '--use-distributed-optimizer', '--overlap-grad-reduce', '--overlap-param-gather',
        '--no-batch-p2p-sync', '--save-interval', '1000000',
    ]
    if tp > 1:
        extra += ['--sequence-parallel']
    if graph:
        extra += ['--cuda-graph-impl', 'transformer_engine', '--cuda-graph-scope',
                  'attn', 'moe_preprocess', 'moe_router', '--cuda-graph-warmup-steps', '1']
    cmd = _tiny_megatron_router_summary_cmd(
        model_id='run', train_iters=3, micro_batch_size=1, global_batch_size=4,
        nproc_per_node=world, tp_size=tp, pp_size=pp, ep_size=1,
        num_experts=4, moe_router_topk=2, moe_token_dispatcher_type='alltoall',
        transformer_impl='transformer_engine', database=database, table='raw', extra_args=extra)
    # Keep TE kernels and Megatron's Apex-backed output-layer gradient fusion.
    cmd.remove('--no-gradient-accumulation-fusion')
    cmd[cmd.index('pretrain_gpt.py')] = str(ROOT / 'tests/oracles/run_megatron_gated_weight_oracle.py')
    env = os.environ.copy()
    env.update(PYTHONPATH=f'{ROOT}:{MEGATRON_ROOT}:' + env.get('PYTHONPATH', ''),
               DMI_ENABLE='1', DMI_WEIGHT_ORACLE_DIR=str(oracle), CUDA_DEVICE_MAX_CONNECTIONS='1')
    client.execute('CREATE DATABASE ' + database)
    (tmp_path / 'command.txt').write_text('\n'.join(cmd))
    try:
        _run_megatron_cmd(cmd, env=env, log_path=tmp_path / 'training.log')
        if graph:
            assert 'Time spent in CUDA Graphs capture' in (tmp_path / 'training.log').read_text()
        expected = {}
        for step in range(4):
            by_layer = {}
            for rank in range(world):
                saved = torch.load(oracle / f'rank{rank}_step{step}.pt', weights_only=True)
                for layer, weight in saved['weights'].items():
                    shards = by_layer.setdefault(layer, {})
                    if saved['tp_rank'] in shards:
                        torch.testing.assert_close(shards[saved['tp_rank']], weight, rtol=0, atol=0)
                    shards[saved['tp_rank']] = weight
            assert set(by_layer) == {3, 7}
            for layer, shards in by_layer.items():
                assert set(shards) == set(range(tp))
                fused = torch.cat([shards[t] for t in range(tp)])
                assert fused.shape == (576, 128)  # Q256 + gate256 + K32 + V32
                expected[step, layer, 'query_projection_weight'] = fused[:256]
                expected[step, layer, 'key_projection_weight'] = fused[512:544]
        for name in ('query_projection_weight', 'key_projection_weight'):
            rows = _read_merged_weight_rows(model_id='run', table='raw', database=database, act_name=name)
            assert len(rows) == 8
            for key, value in rows:
                torch.testing.assert_close(value, expected[key[4], key[8], name], rtol=0, atol=0)
            for layer in (3, 7):
                torch.testing.assert_close(expected[0, layer, name], expected[1, layer, name], rtol=0, atol=0)
                assert not torch.equal(expected[1, layer, name], expected[3, layer, name])
    finally:
        client.execute('DROP DATABASE ' + database + ' SYNC')
        client.disconnect()
