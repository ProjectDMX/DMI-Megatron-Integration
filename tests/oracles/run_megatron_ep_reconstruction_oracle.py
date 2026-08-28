"""Run tiny Megatron training while retaining native MoE outputs as an oracle."""

from __future__ import annotations

import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

import pretrain_gpt as base_pretrain


_ORACLE_OUTPUTS: dict[int, list[torch.Tensor]] = defaultdict(list)


def _record_moe_output(layer_no: int):
    def record(_module: torch.nn.Module, _inputs: tuple[Any, ...], outputs: Any) -> None:
        if not isinstance(outputs, tuple) or len(outputs) != 2:
            raise TypeError("Controlled eager MoELayer output must be (output, mlp_bias)")
        output, mlp_bias = outputs
        if not isinstance(output, torch.Tensor):
            raise TypeError("Controlled eager MoELayer output[0] must be a tensor")
        if mlp_bias is not None:
            raise ValueError("Controlled MoE reconstruction oracle requires mlp_bias=None")
        _ORACLE_OUTPUTS[layer_no].append(output.detach().cpu().clone())

    return record


def _oracle_model_provider(
    pre_process: bool = True,
    post_process: bool = True,
    vp_stage: int | None = None,
    config=None,
    pg_collection=None,
):
    model = base_pretrain.model_provider(
        base_pretrain.gpt_builder,
        pre_process,
        post_process,
        vp_stage,
        config=config,
        pg_collection=pg_collection,
    )
    for module in model.modules():
        if module.__class__.__name__ != "MoELayer":
            continue
        layer_number = getattr(module, "layer_number", None)
        if layer_number is None:
            raise ValueError("Controlled MoE oracle requires a global layer number")
        module.register_forward_hook(_record_moe_output(int(layer_number) - 1))
    return model


def main() -> None:
    oracle_dir_text = os.environ.get("DMI_EP_ORACLE_DIR")
    if not oracle_dir_text:
        raise RuntimeError("DMI_EP_ORACLE_DIR is required")
    oracle_dir = Path(oracle_dir_text)
    oracle_dir.mkdir(parents=True, exist_ok=True)

    now = time.time()
    base_pretrain.set_startup_timestamps(program_start=now, main_entry=now)
    base_pretrain.train_valid_test_datasets_provider.is_distributed = True
    base_pretrain.train_valid_test_datasets_provider.dmi_standard_dataset_provider = True

    base_pretrain.pretrain(
        base_pretrain.train_valid_test_datasets_provider,
        _oracle_model_provider,
        base_pretrain.ModelType.encoder_or_decoder,
        base_pretrain.forward_step,
        args_defaults={"tokenizer_type": "GPT2BPETokenizer"},
        extra_args_provider=(
            base_pretrain.add_modelopt_args if base_pretrain.has_nvidia_modelopt else None
        ),
        store=None,
        get_embedding_ranks=base_pretrain.get_embedding_ranks,
    )

    rank = int(os.environ["RANK"])
    if not _ORACLE_OUTPUTS:
        raise RuntimeError(f"Rank {rank} did not observe a MoELayer output")
    torch.save(dict(_ORACLE_OUTPUTS), oracle_dir / f"rank_{rank}.pt")


if __name__ == "__main__":
    main()
