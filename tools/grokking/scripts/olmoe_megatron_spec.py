"""External Megatron spec for HF OLMoE checkpoints.

This keeps Megatron-LM source untouched while matching OLMoE's attention
layout: HF OLMoE applies Q/K RMSNorm on the full projected hidden dimension
before reshaping into attention heads, while Megatron's stock qk_layernorm is
per-head.
"""

from __future__ import annotations

from functools import partial

import torch
from torch import nn

from megatron.core.models.backends import LocalSpecProvider
from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.core.transformer.attention import SelfAttention, SelfAttentionSubmodules
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.moe.moe_layer import MoELayer, MoESubmodules
from megatron.core.transformer.moe.shared_experts import SharedExpertMLP
from megatron.core.transformer.mlp import MLPSubmodules
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_layer import TransformerLayer, TransformerLayerSubmodules
from megatron.core.typed_torch import apply_module


class OlmoeFullHiddenRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-5, dtype: torch.dtype | None = None):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=dtype))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class OlmoeSelfAttention(SelfAttention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.q_layernorm = OlmoeFullHiddenRMSNorm(
            self.hidden_size_per_attention_head * self.num_attention_heads_per_partition,
            eps=self.config.layernorm_epsilon,
            dtype=self.config.params_dtype,
        )
        self.k_layernorm = OlmoeFullHiddenRMSNorm(
            self.hidden_size_per_attention_head * self.num_query_groups_per_partition,
            eps=self.config.layernorm_epsilon,
            dtype=self.config.params_dtype,
        )

    def get_query_key_value_tensors(
        self,
        hidden_states: torch.Tensor,
        key_value_states: torch.Tensor | None = None,
        output_gate: bool = False,
        split_qkv: bool = True,
    ):
        if key_value_states is not None:
            raise NotImplementedError("OLMoE self-attention conversion does not support cross-attention")
        if output_gate:
            raise NotImplementedError("OLMoE self-attention conversion does not support output_gate")
        if not split_qkv:
            raise NotImplementedError("OLMoE self-attention conversion requires split_qkv=True")
        if self.world_size != 1:
            raise NotImplementedError("OLMoE full-hidden Q/K RMSNorm spec currently supports TP=1")

        mixed_qkv, _ = apply_module(self.linear_qkv)(hidden_states)
        num_query_heads_per_group = (
            self.num_attention_heads_per_partition // self.num_query_groups_per_partition
        )
        new_tensor_shape = mixed_qkv.size()[:-1] + (
            self.num_query_groups_per_partition,
            (num_query_heads_per_group + 2) * self.hidden_size_per_attention_head,
        )
        mixed_qkv = mixed_qkv.view(*new_tensor_shape)

        query, key, value = torch.split(
            mixed_qkv,
            [
                num_query_heads_per_group * self.hidden_size_per_attention_head,
                self.hidden_size_per_attention_head,
                self.hidden_size_per_attention_head,
            ],
            dim=3,
        )

        query_shape = query.shape
        key_shape = key.shape
        query = apply_module(self.q_layernorm)(query.reshape(*query.shape[:2], -1))
        key = apply_module(self.k_layernorm)(key.reshape(*key.shape[:2], -1))
        query = query.view(*query_shape)
        key = key.view(*key_shape)

        query = query.reshape(query.size(0), query.size(1), -1, self.hidden_size_per_attention_head)
        return query, key, value


_backend = LocalSpecProvider()
_mlp_submodules = MLPSubmodules(
    linear_fc1=_backend.column_parallel_linear(),
    linear_fc2=_backend.row_parallel_linear(),
    activation_func=_backend.activation_func(),
)
_moe_spec = ModuleSpec(
    module=MoELayer,
    submodules=MoESubmodules(
        experts=_backend.grouped_mlp_modules(False),
        shared_experts=partial(SharedExpertMLP, submodules=_mlp_submodules),
    ),
    metainfo={"fuse_pre_mlp_layernorm": False},
)

_olmoe_layer_submodules = TransformerLayerSubmodules(
    input_layernorm=_backend.layer_norm(rms_norm=True, has_residual=True),
    self_attention=ModuleSpec(
        module=OlmoeSelfAttention,
        params={"attn_mask_type": AttnMaskType.causal},
        submodules=SelfAttentionSubmodules(
            linear_qkv=_backend.column_parallel_linear(),
            core_attention=_backend.core_attention(),
            linear_proj=_backend.row_parallel_linear(),
            q_layernorm=None,
            k_layernorm=None,
        ),
    ),
    self_attn_bda=get_bias_dropout_add,
    pre_mlp_layernorm=_backend.layer_norm(has_residual=True),
    mlp=_moe_spec,
    mlp_bda=get_bias_dropout_add,
)

olmoe_layer_spec = ModuleSpec(module=TransformerLayer, submodules=_olmoe_layer_submodules)
