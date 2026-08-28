#!/usr/bin/env python3
"""Run a tiny greedy generation sanity check on converted Megatron OLMoE."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from convert_olmoe_megatron_native_to_hf import allow_trusted_megatron_common_state_load
from olmoe_megatron_conversion import (
    add_megatron_to_path,
    build_megatron_model,
    configure_nvidia_library_path,
    megatron_argv_from_olmoe,
)

DEFAULT_PROMPTS = [
    "1 + 10 =",
    "2 + 3 =",
    "7 - 4 =",
    "The capital of France is",
    "The capital of Germany is",
    "The opposite of hot is",
    "The chemical symbol for water is",
]

# Expected answer starts, for human inspection:
# 11, 5, 3, Paris, Berlin, cold, H2O/h2o.


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--megatron-dir", type=Path, required=True)
    parser.add_argument("--hf-template-dir", type=Path, required=True)
    parser.add_argument("--prompt", action="append", default=[])
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--ckpt-format", choices=["torch_dist", "torch"], default="torch_dist")
    parser.add_argument("--extra-megatron-arg", action="append", default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.megatron_dir = args.megatron_dir.resolve()
    args.hf_template_dir = args.hf_template_dir.resolve()
    prompts = args.prompt or DEFAULT_PROMPTS

    configure_nvidia_library_path()
    add_megatron_to_path()

    sys.argv = megatron_argv_from_olmoe(
        hf_dir=args.hf_template_dir,
        load_dir=args.megatron_dir,
        extra_args=args.extra_megatron_arg,
        ckpt_format=args.ckpt_format,
    )

    from megatron.core.inference.contexts.static_context import StaticInferenceContext
    from megatron.core.inference.engines import StaticInferenceEngine
    from megatron.core.inference.model_inference_wrappers.gpt.gpt_inference_wrapper import (
        GPTInferenceWrapper,
    )
    from megatron.core.inference.sampling_params import SamplingParams
    from megatron.core.inference.text_generation_controllers.text_generation_controller import (
        TextGenerationController,
    )
    from megatron.training.checkpointing import load_checkpoint
    from megatron.training.initialize import initialize_megatron
    from megatron.training import get_args
    from megatron.core.tokenizers.utils.build_tokenizer import build_tokenizer

    initialize_megatron()
    megatron_args = get_args()
    model = build_megatron_model()
    allow_trusted_megatron_common_state_load()
    load_checkpoint([model], None, None)
    model.cuda(torch.cuda.current_device())
    model.eval()

    tokenizer = build_tokenizer(megatron_args)
    inference_context = StaticInferenceContext(
        megatron_args.inference_max_requests,
        megatron_args.inference_max_seq_length,
    )
    wrapped_model = GPTInferenceWrapper(model, inference_context)
    controller = TextGenerationController(
        inference_wrapped_model=wrapped_model,
        tokenizer=tokenizer,
    )
    engine = StaticInferenceEngine(
        text_generation_controller=controller,
        legacy=megatron_args.use_legacy_static_engine,
    )
    sampling_params = SamplingParams(
        top_k=1,
        num_tokens_to_generate=args.max_new_tokens,
        termination_id=tokenizer.eod,
    )
    results = engine.generate(prompts=prompts, sampling_params=sampling_params)

    if torch.distributed.get_rank() == 0:
        for result in results:
            print(f"PROMPT: {result.prompt}")
            print(f"GENERATED: {result.generated_text}")


if __name__ == "__main__":
    main()
