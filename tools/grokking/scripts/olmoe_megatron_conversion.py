"""Shared helpers for standalone OLMoE HF <-> Megatron checkpoint conversion."""

from __future__ import annotations

import json
import os
import shutil
import sys
import ctypes
from collections import OrderedDict
from pathlib import Path
from typing import Iterable

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoConfig


REPO_ROOT = Path(__file__).resolve().parents[3]
MEGATRON_ROOT = REPO_ROOT / "third_party" / "megatron-lm"


def add_megatron_to_path() -> None:
    root = str(MEGATRON_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


def configure_nvidia_library_path() -> None:
    """Expose CUDA pip-wheel libraries to TE when scripts are run from conda envs."""
    try:
        import site

        site_packages = Path(site.getsitepackages()[0])
    except Exception:
        return

    lib_dirs = [
        site_packages / "nvidia" / name / "lib"
        for name in ("cublas", "cudnn", "cuda_runtime", "cuda_nvrtc", "nvjitlink", "nccl")
    ]
    existing = [str(path) for path in lib_dirs if path.exists()]
    if not existing:
        return
    current = os.environ.get("LD_LIBRARY_PATH", "")
    parts = existing + ([current] if current else [])
    os.environ["LD_LIBRARY_PATH"] = ":".join(parts)

    for relative_path in (
        "nvidia/cuda_runtime/lib/libcudart.so.12",
        "nvidia/cublas/lib/libcublas.so.12",
        "nvidia/cublas/lib/libcublasLt.so.12",
        "nvidia/cudnn/lib/libcudnn.so.9",
        "nvidia/nccl/lib/libnccl.so.2",
    ):
        lib_path = site_packages / relative_path
        if lib_path.exists():
            ctypes.CDLL(str(lib_path), mode=ctypes.RTLD_GLOBAL)


def load_hf_index(hf_dir: Path) -> dict[str, Path]:
    hf_dir = hf_dir.resolve()
    index_path = hf_dir / "model.safetensors.index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text())
        return {key: hf_dir / filename for key, filename in index["weight_map"].items()}

    single_file = hf_dir / "model.safetensors"
    if single_file.exists():
        with safe_open(single_file, framework="pt", device="cpu") as handle:
            return {key: single_file for key in handle.keys()}

    mapping: dict[str, Path] = {}
    for shard in sorted(hf_dir.glob("*.safetensors")):
        with safe_open(shard, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                mapping[key] = shard
    if not mapping:
        raise FileNotFoundError(f"No safetensors checkpoint files found in {hf_dir}")
    return mapping


class HFTensorStore:
    def __init__(self, hf_dir: Path):
        self.hf_dir = hf_dir.resolve()
        self.key_to_file = load_hf_index(self.hf_dir)
        self._handles: dict[Path, safe_open] = {}

    def get(self, key: str) -> torch.Tensor:
        if key not in self.key_to_file:
            raise KeyError(f"HF tensor not found: {key}")
        path = self.key_to_file[key]
        handle = self._handles.get(path)
        if handle is None:
            handle = safe_open(path, framework="pt", device="cpu")
            self._handles[path] = handle
        return handle.get_tensor(key)

    def close(self) -> None:
        self._handles.clear()


def copy_tensor(dst: torch.Tensor, src: torch.Tensor) -> None:
    if tuple(dst.shape) != tuple(src.shape):
        raise ValueError(f"shape mismatch for destination {tuple(dst.shape)} and source {tuple(src.shape)}")
    dst.copy_(src.to(device=dst.device, dtype=dst.dtype))


def olmoe_config(hf_dir: Path):
    config = AutoConfig.from_pretrained(str(hf_dir))
    if type(config).__name__ != "OlmoeConfig":
        raise TypeError(f"Expected OlmoeConfig, got {type(config).__name__}")
    return config


def megatron_argv_from_olmoe(
    *,
    hf_dir: Path,
    save_dir: Path | None = None,
    load_dir: Path | None = None,
    extra_args: Iterable[str] = (),
    ckpt_format: str = "torch_dist",
) -> list[str]:
    config = olmoe_config(hf_dir)
    argv = [
        "olmoe_conversion.py",
        "--use-mcore-models",
        "--micro-batch-size",
        "1",
        "--global-batch-size",
        "1",
        "--bf16",
        "--no-gradient-accumulation-fusion",
        "--no-persist-layer-norm",
        "--no-masked-softmax-fusion",
        "--disable-bias-linear",
        "--untie-embeddings-and-output-weights",
        "--position-embedding-type",
        "rope",
        "--no-rope-fusion",
        "--normalization",
        "RMSNorm",
        "--swiglu",
        "--num-layers",
        str(config.num_hidden_layers),
        "--hidden-size",
        str(config.hidden_size),
        "--ffn-hidden-size",
        str(config.intermediate_size),
        "--num-attention-heads",
        str(config.num_attention_heads),
        "--num-query-groups",
        str(config.num_key_value_heads),
        "--num-experts",
        str(config.num_experts),
        "--moe-ffn-hidden-size",
        str(config.intermediate_size),
        "--moe-router-topk",
        str(config.num_experts_per_tok),
        "--moe-router-dtype",
        "fp32",
        "--moe-router-score-function",
        "softmax",
        "--moe-router-pre-softmax",
        "--moe-token-dispatcher-type",
        "alltoall",
        "--moe-router-load-balancing-type",
        "aux_loss",
        "--seq-length",
        str(config.max_position_embeddings),
        "--max-position-embeddings",
        str(config.max_position_embeddings),
        "--tokenizer-type",
        "HuggingFaceTokenizer",
        "--tokenizer-model",
        str(hf_dir),
        "--vocab-size",
        str(config.vocab_size),
        "--make-vocab-size-divisible-by",
        "128",
        "--rotary-percent",
        "1.0",
        "--rotary-base",
        str(int(config.rope_theta)),
        "--transformer-impl",
        "local",
        "--spec",
        "olmoe_megatron_spec",
        "olmoe_layer_spec",
        "--mock-data",
        "--train-iters",
        "1",
        "--no-load-optim",
        "--no-load-rng",
        "--no-save-optim",
        "--no-save-rng",
        "--ckpt-format",
        ckpt_format,
        "--no-one-logger",
    ]
    if save_dir is not None:
        argv.extend(["--save", str(save_dir), "--save-interval", "1"])
    if load_dir is not None:
        argv.extend(["--load", str(load_dir), "--auto-detect-ckpt-format", "--exit-on-missing-checkpoint"])
    argv.extend(list(extra_args))
    return argv


def build_megatron_model():
    from megatron.core import mpu
    from model_provider import model_provider
    from gpt_builders import gpt_builder

    pre_process = mpu.is_pipeline_first_stage()
    post_process = mpu.is_pipeline_last_stage()
    return model_provider(gpt_builder, pre_process=pre_process, post_process=post_process)


def pack_qkv(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, num_heads: int, num_query_groups: int):
    hidden_size = q.shape[1]
    head_dim = q.shape[0] // num_heads
    queries_per_group = num_heads // num_query_groups
    q_grouped = q.reshape(num_query_groups, queries_per_group * head_dim, hidden_size)
    k_grouped = k.reshape(num_query_groups, head_dim, hidden_size)
    v_grouped = v.reshape(num_query_groups, head_dim, hidden_size)
    return torch.cat([q_grouped, k_grouped, v_grouped], dim=1).reshape(-1, hidden_size)


def unpack_qkv(qkv: torch.Tensor, num_heads: int, num_query_groups: int):
    hidden_size = qkv.shape[1]
    head_dim = hidden_size // num_heads
    queries_per_group = num_heads // num_query_groups
    grouped = qkv.reshape(num_query_groups, queries_per_group * head_dim + 2 * head_dim, hidden_size)
    q = grouped[:, : queries_per_group * head_dim, :].reshape(num_heads * head_dim, hidden_size)
    k_start = queries_per_group * head_dim
    k = grouped[:, k_start : k_start + head_dim, :].reshape(num_query_groups * head_dim, hidden_size)
    v = grouped[:, k_start + head_dim :, :].reshape(num_query_groups * head_dim, hidden_size)
    return q.contiguous(), k.contiguous(), v.contiguous()


def unwrap_model_dict_key(key: str) -> str:
    for prefix in ("module.",):
        if key.startswith(prefix):
            key = key[len(prefix) :]
    return key


def iter_decoder_layers(model) -> list[tuple[int, object]]:
    decoder = getattr(model, "decoder", None)
    if decoder is None:
        return []
    layers = getattr(decoder, "layers", [])
    result = []
    for local_idx, layer in enumerate(layers):
        global_idx = getattr(layer, "layer_number", local_idx + 1) - 1
        result.append((global_idx, layer))
    return result


def save_hf_safetensors(state_dict: OrderedDict[str, torch.Tensor], out_dir: Path, max_shard_size_gb: float) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    max_bytes = int(max_shard_size_gb * 1024**3)
    shards: list[OrderedDict[str, torch.Tensor]] = []
    current: OrderedDict[str, torch.Tensor] = OrderedDict()
    current_bytes = 0
    total_bytes = 0
    for key, tensor in state_dict.items():
        tensor = tensor.detach().cpu().contiguous()
        nbytes = tensor.numel() * tensor.element_size()
        if current and current_bytes + nbytes > max_bytes:
            shards.append(current)
            current = OrderedDict()
            current_bytes = 0
        current[key] = tensor
        current_bytes += nbytes
        total_bytes += nbytes
    if current:
        shards.append(current)

    weight_map = {}
    if len(shards) == 1:
        filename = "model.safetensors"
        save_file(shards[0], out_dir / filename, metadata={"format": "pt"})
        weight_map = {key: filename for key in shards[0]}
    else:
        width = max(5, len(str(len(shards))))
        for idx, shard in enumerate(shards, start=1):
            filename = f"model-{idx:0{width}d}-of-{len(shards):0{width}d}.safetensors"
            save_file(shard, out_dir / filename, metadata={"format": "pt"})
            for key in shard:
                weight_map[key] = filename
        index = {"metadata": {"total_size": total_bytes}, "weight_map": weight_map}
        (out_dir / "model.safetensors.index.json").write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")


def copy_hf_sidecar_files(template_dir: Path, out_dir: Path) -> None:
    skip_suffixes = (".safetensors", ".bin", ".pt")
    skip_names = {"model.safetensors.index.json", "pytorch_model.bin.index.json"}
    for path in template_dir.iterdir():
        if path.name in skip_names or path.name.startswith("model-"):
            continue
        if path.suffix in skip_suffixes:
            continue
        dst = out_dir / path.name
        if path.is_dir():
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(path, dst)
        else:
            shutil.copy2(path, dst)
