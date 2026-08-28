#!/usr/bin/env python3
"""Convert a Megatron-native OLMoE checkpoint back to HF safetensors."""

from __future__ import annotations

import argparse
import json
import sys
from collections import OrderedDict
from pathlib import Path

import torch

from olmoe_megatron_conversion import (
    add_megatron_to_path,
    build_megatron_model,
    configure_nvidia_library_path,
    copy_hf_sidecar_files,
    iter_decoder_layers,
    megatron_argv_from_olmoe,
    olmoe_config,
    save_hf_safetensors,
    unpack_qkv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--megatron-dir", type=Path, required=True)
    parser.add_argument("--hf-template-dir", type=Path, required=True)
    parser.add_argument("--save-dir", type=Path, required=True)
    parser.add_argument("--max-shard-size-gb", type=float, default=5.0)
    parser.add_argument("--ckpt-format", choices=["torch_dist", "torch"], default="torch_dist")
    parser.add_argument("--extra-megatron-arg", action="append", default=[])
    parser.add_argument("--manifest-name", default="")
    return parser.parse_args()


def megatron_to_hf_state_dict(model, config) -> OrderedDict[str, torch.Tensor]:
    state = OrderedDict()
    if hasattr(model, "embedding"):
        state["model.embed_tokens.weight"] = model.embedding.word_embeddings.weight.detach().cpu().contiguous()
    if hasattr(model, "output_layer"):
        state["lm_head.weight"] = model.output_layer.weight.detach().cpu().contiguous()
    if hasattr(model, "decoder") and hasattr(model.decoder, "final_layernorm"):
        state["model.norm.weight"] = model.decoder.final_layernorm.weight.detach().cpu().contiguous()

    for layer_idx, layer in iter_decoder_layers(model):
        hf_prefix = f"model.layers.{layer_idx}"
        attn = layer.self_attention
        q, k, v = unpack_qkv(
            attn.linear_qkv.weight.detach().cpu().contiguous(),
            config.num_attention_heads,
            config.num_key_value_heads,
        )
        state[f"{hf_prefix}.self_attn.q_proj.weight"] = q
        state[f"{hf_prefix}.self_attn.k_proj.weight"] = k
        state[f"{hf_prefix}.self_attn.v_proj.weight"] = v
        state[f"{hf_prefix}.self_attn.o_proj.weight"] = (
            attn.linear_proj.weight.detach().cpu().contiguous()
        )
        state[f"{hf_prefix}.input_layernorm.weight"] = (
            layer.input_layernorm.weight.detach().cpu().contiguous()
        )
        state[f"{hf_prefix}.self_attn.q_norm.weight"] = (
            attn.q_layernorm.weight.detach().cpu().contiguous()
        )
        state[f"{hf_prefix}.self_attn.k_norm.weight"] = (
            attn.k_layernorm.weight.detach().cpu().contiguous()
        )
        state[f"{hf_prefix}.post_attention_layernorm.weight"] = (
            layer.pre_mlp_layernorm.weight.detach().cpu().contiguous()
        )

        mlp = layer.mlp
        state[f"{hf_prefix}.mlp.gate.weight"] = mlp.router.weight.detach().cpu().contiguous()
        experts = mlp.experts.local_experts
        if len(experts) != config.num_experts:
            raise ValueError(
                f"Expected all {config.num_experts} experts in this standalone reverse converter, "
                f"but Megatron local_experts has {len(experts)}. Run reverse conversion with EP=1."
            )
        for expert_idx, expert in enumerate(experts):
            fc1 = expert.linear_fc1.weight.detach().cpu().contiguous()
            gate, up = torch.chunk(fc1, 2, dim=0)
            state[f"{hf_prefix}.mlp.experts.{expert_idx}.gate_proj.weight"] = gate.contiguous()
            state[f"{hf_prefix}.mlp.experts.{expert_idx}.up_proj.weight"] = up.contiguous()
            state[f"{hf_prefix}.mlp.experts.{expert_idx}.down_proj.weight"] = (
                expert.linear_fc2.weight.detach().cpu().contiguous()
            )

    expected_layers = set(range(config.num_hidden_layers))
    seen_layers = {idx for idx, _ in iter_decoder_layers(model)}
    if seen_layers != expected_layers:
        raise ValueError(
            "Standalone HF export requires all layers in the local model. "
            f"Expected {sorted(expected_layers)}, got {sorted(seen_layers)}. Run with PP=1."
        )
    return state


def allow_trusted_megatron_common_state_load() -> None:
    original_load = torch.load

    def trusted_load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original_load(*args, **kwargs)

    torch.load = trusted_load


def main() -> None:
    args = parse_args()
    args.megatron_dir = args.megatron_dir.resolve()
    args.hf_template_dir = args.hf_template_dir.resolve()
    args.save_dir = args.save_dir.resolve()

    configure_nvidia_library_path()
    add_megatron_to_path()

    sys.argv = megatron_argv_from_olmoe(
        hf_dir=args.hf_template_dir,
        load_dir=args.megatron_dir,
        extra_args=args.extra_megatron_arg,
        ckpt_format=args.ckpt_format,
    )

    from megatron.training.initialize import initialize_megatron
    from megatron.training.checkpointing import load_checkpoint

    initialize_megatron()
    config = olmoe_config(args.hf_template_dir)
    model = build_megatron_model()
    allow_trusted_megatron_common_state_load()
    load_checkpoint([model], None, None)
    model.eval()

    state = megatron_to_hf_state_dict(model, config)
    args.save_dir.mkdir(parents=True, exist_ok=True)
    copy_hf_sidecar_files(args.hf_template_dir, args.save_dir)
    save_hf_safetensors(state, args.save_dir, args.max_shard_size_gb)

    manifest = {
        "kind": "dmi_olmoe_megatron_to_hf_conversion",
        "megatron_dir": str(args.megatron_dir),
        "hf_template_dir": str(args.hf_template_dir),
        "save_dir": str(args.save_dir),
        "ckpt_format": args.ckpt_format,
        "architecture": type(config).__name__,
        "num_layers": config.num_hidden_layers,
        "hidden_size": config.hidden_size,
        "num_experts": config.num_experts,
        "num_tensors": len(state),
    }
    if args.manifest_name:
        (args.save_dir / args.manifest_name).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
