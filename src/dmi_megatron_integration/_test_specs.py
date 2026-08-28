"""Megatron model specs used by DMI's real integration tests."""

from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_submodules
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_layer import MoETransformerLayer


dmi_test_moe_layer_spec = ModuleSpec(
    module=MoETransformerLayer,
    submodules=get_gpt_layer_local_submodules(
        num_experts=2,
        moe_grouped_gemm=False,
        qk_layernorm=False,
        multi_latent_attention=False,
        normalization="LayerNorm",
    ),
)
