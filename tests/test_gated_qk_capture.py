"""Gated attention physical-shard selection, independent of forward counts."""
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from dmi_megatron_integration.hooks.weight_capture import discover_weight_captures
from dmi_megatron_integration.materialization.reconstruction import merge_weight_shards
from tests.test_megatron_qk_weights import SelfAttention, _attention, _rank
from tests.test_weight_capture import materialize


@pytest.mark.parametrize('heads,groups,dim,hidden,tp', [
    (4, 4, 2, 7, 2), (8, 2, 2, 7, 2), (8, 1, 2, 7, 2),
    (16, 2, 256, 2048, 8),
])
@pytest.mark.parametrize('gated', [False, True])
@pytest.mark.parametrize('selected', [{'q-weights'}, {'k-weights'}, {'q-weights', 'k-weights'}])
def test_physical_shards_reconstruct_projection_without_gate_or_value(
        heads, groups, dim, hidden, tp, gated, selected):
    # Build the reference from separate projections, not production byte maps.
    q = (torch.arange(heads * dim * hidden) % 97).reshape(heads * dim, hidden).float()
    k = (torch.arange(groups * dim * hidden) % 89 + 200).reshape(groups * dim, hidden).float()
    parts = [q.reshape(groups, -1, hidden)]
    if gated:
        parts.append(torch.full_like(parts[0], -1000))
    parts += [k.reshape(groups, dim, hidden), torch.full((groups, dim, hidden), -2000.)]
    fused = torch.cat(parts, dim=1).reshape(-1, hidden)
    captures = []
    for rank, local in enumerate(fused.chunk(tp)):
        module = SelfAttention()
        module.layer_number = 1
        module.hidden_size_per_attention_head = dim
        module.config = SimpleNamespace(num_attention_heads=heads, num_query_groups=groups,
                                       hidden_size=hidden, attention_output_gate=gated)
        module.linear_qkv = nn.Module()
        module.linear_qkv.weight = nn.Parameter(local.clone())
        captures.extend(discover_weight_captures(module, _rank(global_rank=rank,
            tp_rank=rank, tp_world_size=tp), selected))
    rows, topology = materialize(captures)
    expected = {'query_projection_weight': q, 'key_projection_weight': k}
    merged, = merge_weight_shards(rows, topology)
    assert len(merged) == len(selected)
    for row in merged:
        torch.testing.assert_close(row.value, expected[row.act_name], rtol=0, atol=0)
    assert sum(r['value'].numel() for r in rows) == sum(r.value.numel()*4 for r in merged)
    if tp > groups and 'k-weights' in selected:
        assert any(r['act_name'] == 'key_projection_weight' and not r['value'].numel()
                   for r in rows)


@pytest.mark.parametrize('selected_layers', [None, (), (3,), (7,), (0, 1, 2)])
def test_hybrid_discovery_preserves_global_layers_and_skips_linear_attention(selected_layers):
    class GatedDeltaNet(nn.Module):
        def __init__(self, layer):
            super().__init__()
            self.layer_number = layer
            self.in_proj = nn.Linear(7, 24)
            # Discovery must not inspect linear attention as a Q/K source.
            self.config = SimpleNamespace(fp8=True)
    stages = []
    refs = {}
    for pp in range(2):
        root = nn.Module()
        layers = []
        for layer in range(pp * 4, (pp + 1) * 4):
            if (layer + 1) % 4:
                layers.append(GatedDeltaNet(layer + 1))
            else:
                attention, q, k = _attention(layer=layer+1, attention_output_gate=True)
                layers.append(attention)
                refs[layer] = {'query_projection_weight': q, 'key_projection_weight': k}
        root.layers = nn.ModuleList(layers)
        captures = discover_weight_captures(root, _rank(global_rank=pp, pp_rank=pp,
            pp_world_size=2), {'q-weights', 'k-weights'}, capture_layers=selected_layers)
        stages.extend(captures)
    expected_layers = {3, 7} if selected_layers is None else {3, 7}.intersection(selected_layers)
    assert {c.layer_no for c in stages} == expected_layers
    assert len(stages) == 2 * len(expected_layers)
    for layer in expected_layers:
        rows, topology = materialize([c for c in stages if c.layer_no == layer])
        for row in merge_weight_shards(rows, topology)[0]:
            torch.testing.assert_close(row.value, refs[layer][row.act_name], rtol=0, atol=0)


def test_gated_discovery_rejects_an_ungated_physical_parameter():
    model, _, _ = _attention()
    model.config.attention_output_gate = True
    with pytest.raises(ValueError, match='Weight shape/layout'):
        discover_weight_captures(model, _rank(), {'q-weights', 'k-weights'})
