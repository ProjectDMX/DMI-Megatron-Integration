#!/usr/bin/env python3
"""Convert an HF OLMoE checkpoint into a Megatron-native checkpoint.

This script is standalone: it does not modify or use Megatron's checkpoint
converter plugins.  It builds an OLMoE-shaped Megatron model, copies HF tensors
into matching Megatron parameters, and calls Megatron's native save_checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch

from olmoe_megatron_conversion import (
    HFTensorStore,
    add_megatron_to_path,
    build_megatron_model,
    configure_nvidia_library_path,
    copy_tensor,
    iter_decoder_layers,
    megatron_argv_from_olmoe,
    olmoe_config,
    pack_qkv,
)


SOURCE_MANIFEST_NAME = "source_manifest.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_hf_source_binding(hf_dir: Path) -> dict[str, str]:
    """Read the immutable source identity required by checkpoint conversion."""

    resolved_hf_dir = hf_dir.expanduser().resolve()
    manifest_path = resolved_hf_dir / SOURCE_MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"HF conversion requires {SOURCE_MANIFEST_NAME}: {manifest_path}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError(f"HF source manifest must be a JSON object: {manifest_path}")
    repository = manifest.get("repository")
    revision = manifest.get("revision")
    if not isinstance(repository, str) or not repository:
        raise ValueError("HF source manifest is missing repository")
    if not isinstance(revision, str) or not revision:
        raise ValueError("HF source manifest is missing immutable revision")
    return {
        "hf_dir": str(resolved_hf_dir),
        "hf_source_manifest_path": str(manifest_path),
        "hf_source_manifest_sha256": _sha256(manifest_path),
        "hf_repository": repository,
        "hf_revision": revision,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-dir", type=Path, required=True)
    parser.add_argument("--save-dir", type=Path, required=True)
    parser.add_argument("--iteration", type=int, default=1)
    parser.add_argument("--ckpt-format", choices=["torch_dist", "torch"], default="torch_dist")
    parser.add_argument("--extra-megatron-arg", action="append", default=[])
    parser.add_argument("--manifest-name", default="dmi_olmoe_hf_to_megatron_manifest.json")
    return parser.parse_args()


def copy_olmoe_to_megatron(model, store: HFTensorStore, config) -> None:
    if hasattr(model, "embedding"):
        copy_tensor(model.embedding.word_embeddings.weight.data, store.get("model.embed_tokens.weight"))
    if hasattr(model, "output_layer"):
        copy_tensor(model.output_layer.weight.data, store.get("lm_head.weight"))
    if hasattr(model, "decoder") and hasattr(model.decoder, "final_layernorm"):
        copy_tensor(model.decoder.final_layernorm.weight.data, store.get("model.norm.weight"))

    for layer_idx, layer in iter_decoder_layers(model):
        hf_prefix = f"model.layers.{layer_idx}"
        attn = layer.self_attention

        q = store.get(f"{hf_prefix}.self_attn.q_proj.weight")
        k = store.get(f"{hf_prefix}.self_attn.k_proj.weight")
        v = store.get(f"{hf_prefix}.self_attn.v_proj.weight")
        qkv = pack_qkv(q, k, v, config.num_attention_heads, config.num_key_value_heads)
        copy_tensor(attn.linear_qkv.weight.data, qkv)
        copy_tensor(attn.linear_proj.weight.data, store.get(f"{hf_prefix}.self_attn.o_proj.weight"))

        copy_tensor(layer.input_layernorm.weight.data, store.get(f"{hf_prefix}.input_layernorm.weight"))
        copy_tensor(attn.q_layernorm.weight.data, store.get(f"{hf_prefix}.self_attn.q_norm.weight"))
        copy_tensor(attn.k_layernorm.weight.data, store.get(f"{hf_prefix}.self_attn.k_norm.weight"))
        copy_tensor(layer.pre_mlp_layernorm.weight.data, store.get(f"{hf_prefix}.post_attention_layernorm.weight"))

        mlp = layer.mlp
        copy_tensor(mlp.router.weight.data, store.get(f"{hf_prefix}.mlp.gate.weight"))
        experts = mlp.experts.local_experts
        if len(experts) != config.num_experts:
            raise ValueError(
                f"Expected all {config.num_experts} experts in this standalone converter, "
                f"but Megatron local_experts has {len(experts)}. Run conversion with EP=1."
            )
        for expert_idx, expert in enumerate(experts):
            gate = store.get(f"{hf_prefix}.mlp.experts.{expert_idx}.gate_proj.weight")
            up = store.get(f"{hf_prefix}.mlp.experts.{expert_idx}.up_proj.weight")
            down = store.get(f"{hf_prefix}.mlp.experts.{expert_idx}.down_proj.weight")
            copy_tensor(expert.linear_fc1.weight.data, torch.cat([gate, up], dim=0))
            copy_tensor(expert.linear_fc2.weight.data, down)


def main() -> None:
    args = parse_args()
    args.hf_dir = args.hf_dir.resolve()
    args.save_dir = args.save_dir.resolve()
    source_binding = load_hf_source_binding(args.hf_dir)

    configure_nvidia_library_path()
    add_megatron_to_path()

    sys.argv = megatron_argv_from_olmoe(
        hf_dir=args.hf_dir,
        save_dir=args.save_dir,
        extra_args=args.extra_megatron_arg,
        ckpt_format=args.ckpt_format,
    )

    from megatron.training.initialize import initialize_megatron
    from megatron.training.checkpointing import save_checkpoint

    initialize_megatron()
    config = olmoe_config(args.hf_dir)
    model = build_megatron_model()
    model.eval()

    store = HFTensorStore(args.hf_dir)
    try:
        with torch.no_grad():
            copy_olmoe_to_megatron(model, store, config)
    finally:
        store.close()

    save_checkpoint(args.iteration, [model], None, None, num_floating_point_operations_so_far=0)

    manifest = {
        "kind": "dmi_olmoe_hf_to_megatron_conversion",
        **source_binding,
        "save_dir": str(args.save_dir),
        "iteration": args.iteration,
        "ckpt_format": args.ckpt_format,
        "architecture": type(config).__name__,
        "num_layers": config.num_hidden_layers,
        "hidden_size": config.hidden_size,
        "num_experts": config.num_experts,
        "num_experts_per_tok": config.num_experts_per_tok,
    }
    args.save_dir.mkdir(parents=True, exist_ok=True)
    (args.save_dir / args.manifest_name).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
