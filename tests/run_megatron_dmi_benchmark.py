#!/usr/bin/env python3
"""Run paired Megatron DMI off/on benchmark jobs.

This is a benchmark driver, not a pytest.  It launches real Megatron commands
and writes machine-readable JSONL rows for each run plus pair-level overhead
rows in the summary markdown.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MEGATRON_ROOT = ROOT / "third_party" / "megatron-lm"
GROKKING_SCRIPTS = ROOT / "tools" / "grokking" / "scripts"
if str(GROKKING_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(GROKKING_SCRIPTS))

from olmoe_megatron_conversion import megatron_argv_from_olmoe
from transformers import AutoConfig

ITER_RE = re.compile(r"elapsed time per iteration \(ms\):\s*([0-9.]+)")
ITERATION_RECORD_RE = re.compile(
    r"iteration\s+(\d+)/\s*\d+.*?elapsed time per iteration \(ms\):\s*([0-9.]+)"
)
ITERATION_FLUSH_RE = re.compile(
    r"\[DMI\] iteration-boundary flush iteration=(\d+) elapsed_s=([0-9.]+)"
)
EVALUATE_TIMER_RE = re.compile(r"evaluate\s+\.{3,}:\s+\(([0-9.]+),\s+([0-9.]+)\)")
EVALUATE_PHASE_RE = re.compile(r"validation loss at iteration .* on (validation set|test set)")
MEM_RE = re.compile(
    r"memory \(MB\).*allocated:\s*([0-9.]+).*max allocated:\s*([0-9.]+).*"
    r"reserved:\s*([0-9.]+).*max reserved:\s*([0-9.]+)"
)


@dataclass(frozen=True)
class BenchmarkCase:
    graph_mode: str
    parallel_name: str
    nproc_per_node: int
    tp_size: int
    pp_size: int
    num_microbatches: int
    moe_args: tuple[str, ...]
    graph_args: tuple[str, ...]
    train_iters: int
    skip_reason: str | None = None

    @property
    def case_id(self) -> str:
        return f"{self.graph_mode}_{self.parallel_name}"


@dataclass(frozen=True)
class HookBenchmarkSpec:
    selection: str
    table_suffix: str
    storage: str
    act_name: str
    per_layer: bool
    additional_act_names: tuple[str, ...] = ()

    @property
    def act_names(self) -> tuple[str, ...]:
        return (self.act_name, *self.additional_act_names)


HOOK_BENCHMARK_SPECS: dict[str, HookBenchmarkSpec] = {
    "vocab-logits": HookBenchmarkSpec(
        selection="vocab-logits",
        table_suffix="",
        storage="tensor",
        act_name="vocab_logits",
        per_layer=False,
    ),
    "vocab-logits-topk": HookBenchmarkSpec(
        selection="vocab-logits-topk",
        table_suffix="",
        storage="tensor",
        act_name="vocab_logits_topk_values",
        additional_act_names=("vocab_logits_topk_indices",),
        per_layer=False,
    ),
    "router-summary": HookBenchmarkSpec(
        selection="router-summary",
        table_suffix="",
        storage="tensor",
        act_name="router_probs_mean",
        per_layer=True,
    ),
    "router-logits": HookBenchmarkSpec(
        selection="router-logits",
        table_suffix="",
        storage="tensor",
        act_name="router_logits",
        per_layer=True,
    ),
    "hidden-states": HookBenchmarkSpec(
        selection="hidden-states",
        table_suffix="",
        storage="tensor",
        act_name="hidden_states",
        per_layer=True,
    ),
    "loss-summary": HookBenchmarkSpec(
        selection="loss-summary",
        table_suffix="_scalar_float",
        storage="scalar_float",
        act_name="lm_per_sample_loss",
        per_layer=False,
    ),
}


def _cases(default_train_iters: int) -> list[BenchmarkCase]:
    eager_moe = ("--moe-token-dispatcher-type", "allgather")
    graph_moe = (
        "--moe-token-dispatcher-type",
        "alltoall",
        "--moe-expert-capacity-factor",
        "1.0",
        "--moe-pad-expert-input-to-capacity",
    )
    te_partial_moe = ("--moe-token-dispatcher-type", "alltoall")
    local_graph_skip = (
        "Native Megatron PP>1 with local per-layer CUDA graphs currently fails "
        "before DMI verification."
    )
    full_graph_skip = (
        "Native Megatron PP>1 with full-iteration CUDA graphs currently fails "
        "during capture-safe pipeline communication."
    )
    specs = [
        ("eager", eager_moe, ("--cuda-graph-impl", "none"), 2, None),
        (
            "te_cuda_graph_attn",
            eager_moe,
            (
                "--cuda-graph-impl",
                "transformer_engine",
                "--cuda-graph-scope",
                "attn",
            ),
            6,
            None,
        ),
        (
            "te_cuda_graph_attn_moe_router_moe_preprocess",
            te_partial_moe,
            (
                "--cuda-graph-impl",
                "transformer_engine",
                "--cuda-graph-scope",
                "attn",
                "moe_router",
                "moe_preprocess",
            ),
            6,
            None,
        ),
        ("local_cuda_graph", graph_moe, ("--cuda-graph-impl", "local"), 3, local_graph_skip),
        (
            "full_iteration_cuda_graph",
            graph_moe,
            (
                "--cuda-graph-impl",
                "local",
                "--cuda-graph-scope",
                "full_iteration",
                "--no-check-for-nan-in-loss-and-grad",
            ),
            3,
            full_graph_skip,
        ),
    ]
    parallel = [
        ("tp1_pp1", 1, 1, 1, 1),
        ("tp1_pp2", 2, 1, 2, 2),
    ]
    out: list[BenchmarkCase] = []
    for graph_mode, moe_args, graph_args, fallback_iters, pp2_skip in specs:
        train_iters = max(int(default_train_iters), int(fallback_iters))
        for parallel_name, nproc, tp, pp, num_microbatches in parallel:
            skip = pp2_skip if pp > 1 and pp2_skip is not None else None
            out.append(
                BenchmarkCase(
                    graph_mode=graph_mode,
                    parallel_name=parallel_name,
                    nproc_per_node=nproc,
                    tp_size=tp,
                    pp_size=pp,
                    num_microbatches=num_microbatches,
                    moe_args=tuple(moe_args),
                    graph_args=tuple(graph_args),
                    train_iters=train_iters,
                    skip_reason=skip,
                )
            )
    return out


def _expected_phase_counts(
    *,
    train_iters: int,
    eval_iters: int,
    eval_interval: int,
    global_batch_size: int,
    num_moe_layers: int,
) -> dict[str, int]:
    valid_runs = train_iters // eval_interval + 1 if eval_iters > 0 else 0
    test_runs = 1 if eval_iters > 0 else 0
    return {
        "train": train_iters * global_batch_size * num_moe_layers,
        "valid": valid_runs * eval_iters * global_batch_size * num_moe_layers,
        "test": test_runs * eval_iters * global_batch_size * num_moe_layers,
    }


def _expected_sample_phase_counts(
    *,
    train_iters: int,
    eval_iters: int,
    eval_interval: int,
    global_batch_size: int,
) -> dict[str, int]:
    valid_runs = train_iters // eval_interval + 1 if eval_iters > 0 else 0
    test_runs = 1 if eval_iters > 0 else 0
    return {
        "train": train_iters * global_batch_size,
        "valid": valid_runs * eval_iters * global_batch_size,
        "test": test_runs * eval_iters * global_batch_size,
    }


def _expected_counts_for_hook(
    hook_spec: HookBenchmarkSpec,
    *,
    train_iters: int,
    eval_iters: int,
    eval_interval: int,
    global_batch_size: int,
    num_moe_layers: int,
    eval_only: bool = False,
) -> dict[str, int]:
    if hook_spec.per_layer:
        if eval_only:
            return {
                "train": 0,
                "valid": eval_iters * global_batch_size * num_moe_layers,
                "test": eval_iters * global_batch_size * num_moe_layers,
            }
        return _expected_phase_counts(
            train_iters=train_iters,
            eval_iters=eval_iters,
            eval_interval=eval_interval,
            global_batch_size=global_batch_size,
            num_moe_layers=num_moe_layers,
        )
    if eval_only:
        return {
            "train": 0,
            "valid": eval_iters * global_batch_size,
            "test": eval_iters * global_batch_size,
        }
    return _expected_sample_phase_counts(
        train_iters=train_iters,
        eval_iters=eval_iters,
        eval_interval=eval_interval,
        global_batch_size=global_batch_size,
    )


def _num_layers_for_benchmark(args: argparse.Namespace) -> int:
    if args.benchmark_model == "mock":
        return int(args.mock_num_layers)
    config = AutoConfig.from_pretrained(str(Path(args.olmoe_hf_template_dir).resolve()))
    return int(config.num_hidden_layers)


def _seq_length_for_benchmark(args: argparse.Namespace) -> int:
    if args.seq_length is not None:
        return int(args.seq_length)
    if args.benchmark_model == "mock":
        return 16
    config = AutoConfig.from_pretrained(str(Path(args.olmoe_hf_template_dir).resolve()))
    return int(config.max_position_embeddings)


def _base_cmd(
    *,
    case: BenchmarkCase,
    train_iters: int,
    eval_iters: int,
    eval_interval: int,
    micro_batch_size: int,
    global_batch_size: int,
    seq_length: int,
    seed: int,
    mock_num_layers: int = 2,
    mock_hidden_size: int = 64,
    mock_ffn_hidden_size: int = 128,
    mock_num_attention_heads: int = 4,
    mock_num_experts: int = 2,
    mock_moe_router_topk: int = 1,
    log_interval: int | None = None,
    no_masked_softmax_fusion: bool = False,
) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc_per_node={case.nproc_per_node}",
        "pretrain_gpt.py",
        "--mock-data",
        "--tokenizer-type",
        "NullTokenizer",
        "--vocab-size",
        "128",
        "--num-layers",
        str(mock_num_layers),
        "--hidden-size",
        str(mock_hidden_size),
        "--ffn-hidden-size",
        str(mock_ffn_hidden_size),
        "--num-attention-heads",
        str(mock_num_attention_heads),
        "--seq-length",
        str(seq_length),
        "--max-position-embeddings",
        str(seq_length),
        "--micro-batch-size",
        str(micro_batch_size),
        "--global-batch-size",
        str(global_batch_size),
        "--tensor-model-parallel-size",
        str(case.tp_size),
        "--pipeline-model-parallel-size",
        str(case.pp_size),
        "--train-iters",
        str(train_iters),
        "--eval-interval",
        str(eval_interval),
        "--eval-iters",
        str(eval_iters),
        "--seed",
        str(seed),
        "--lr",
        "1.0e-4",
        "--min-lr",
        "1.0e-5",
        "--lr-decay-iters",
        str(train_iters),
        "--lr-warmup-iters",
        "0",
        "--weight-decay",
        "0.0",
        "--adam-beta1",
        "0.9",
        "--adam-beta2",
        "0.95",
        "--init-method-std",
        "0.02",
        "--clip-grad",
        "1.0",
        "--bf16",
        "--transformer-impl",
        "local",
        "--no-persist-layer-norm",
        "--no-gradient-accumulation-fusion",
        "--swiglu",
        "--disable-bias-linear",
        "--num-experts",
        str(mock_num_experts),
        "--moe-router-topk",
        str(mock_moe_router_topk),
        "--moe-router-pre-softmax",
        "--moe-router-load-balancing-type",
        "aux_loss",
        "--moe-aux-loss-coeff",
        "0.01",
        "--no-save-optim",
        "--no-save-rng",
        "--no-load-optim",
        "--no-load-rng",
        "--no-create-attention-mask-in-dataloader",
        *case.moe_args,
        *case.graph_args,
    ]
    if log_interval is not None:
        cmd.extend(["--log-interval", str(log_interval)])
    if no_masked_softmax_fusion:
        cmd.append("--no-masked-softmax-fusion")
    return cmd


def _remove_flag(argv: list[str], flag: str, *, has_value: bool) -> list[str]:
    out: list[str] = []
    i = 0
    while i < len(argv):
        if argv[i] == flag:
            i += 2 if has_value else 1
            continue
        out.append(argv[i])
        i += 1
    return out


def _replace_option(argv: list[str], flag: str, value: str) -> list[str]:
    argv = _remove_flag(argv, flag, has_value=True)
    argv.extend([flag, value])
    return argv


def _add_flag_once(argv: list[str], flag: str) -> list[str]:
    if flag not in argv:
        argv.append(flag)
    return argv


def _olmoe_base_cmd(
    *,
    case: BenchmarkCase,
    train_iters: int,
    eval_iters: int,
    eval_interval: int,
    micro_batch_size: int,
    global_batch_size: int,
    seq_length: int,
    seed: int,
    args: argparse.Namespace,
) -> list[str]:
    checkpoint_dir = Path(args.olmoe_megatron_dir).resolve()
    hf_template_dir = Path(args.olmoe_hf_template_dir).resolve()
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"OLMoE Megatron checkpoint not found: {checkpoint_dir}")
    if not hf_template_dir.exists():
        raise FileNotFoundError(f"OLMoE HF template directory not found: {hf_template_dir}")

    argv = megatron_argv_from_olmoe(
        hf_dir=hf_template_dir,
        load_dir=checkpoint_dir,
        extra_args=(),
        ckpt_format="torch_dist",
    )[1:]

    if args.olmoe_mock_data:
        argv = _add_flag_once(argv, "--mock-data")
        argv = _remove_flag(argv, "--data-path", has_value=True)
        argv = _remove_flag(argv, "--split", has_value=True)
    else:
        data_prefix = args.olmoe_data_prefix
        if not data_prefix:
            raise ValueError("--olmoe-data-prefix is required unless --olmoe-mock-data is set")
        prefix = Path(data_prefix).resolve()
        if not prefix.with_suffix(".idx").exists() or not prefix.with_suffix(".bin").exists():
            raise FileNotFoundError(f"OLMoE dataset prefix missing .idx/.bin files: {prefix}")
        argv = _remove_flag(argv, "--mock-data", has_value=False)
        argv = _replace_option(argv, "--data-path", str(prefix))
        argv = _replace_option(argv, "--split", str(args.olmoe_split))

    replacements = {
        "--micro-batch-size": str(micro_batch_size),
        "--global-batch-size": str(global_batch_size),
        "--tensor-model-parallel-size": str(case.tp_size),
        "--pipeline-model-parallel-size": str(case.pp_size),
        "--expert-model-parallel-size": "1",
        "--context-parallel-size": "1",
        "--seq-length": str(seq_length),
        "--train-iters": str(train_iters),
        "--eval-iters": str(eval_iters),
        "--seed": str(seed),
    }
    for flag, value in replacements.items():
        argv = _replace_option(argv, flag, value)

    argv = _remove_flag(argv, "--moe-token-dispatcher-type", has_value=True)
    argv = _remove_flag(argv, "--cuda-graph-impl", has_value=True)
    argv = _remove_flag(argv, "--cuda-graph-scope", has_value=True)
    argv = _remove_flag(argv, "--no-check-for-nan-in-loss-and-grad", has_value=False)
    argv.extend(case.moe_args)
    argv.extend(case.graph_args)

    if args.olmoe_eval_only:
        argv = _add_flag_once(argv, "--skip-train")
        argv = _replace_option(argv, "--eval-interval", str(max(1, train_iters + 1)))
    else:
        training_replacements = {
            "--attention-backend": "unfused",
            "--attention-dropout": "0.0",
            "--hidden-dropout": "0.0",
            "--moe-aux-loss-coeff": "0.01",
            "--data-cache-path": str(Path(args.olmoe_data_cache_path).resolve()),
            "--dataloader-type": "single",
            "--num-workers": "0",
            "--log-interval": str(args.log_interval if args.log_interval is not None else 1),
            "--lr": "1.0e-5",
            "--min-lr": "1.0e-5",
            "--lr-decay-style": "constant",
            "--lr-decay-iters": str(train_iters),
            "--lr-warmup-iters": "0",
            "--weight-decay": "0.0",
            "--adam-beta1": "0.9",
            "--adam-beta2": "0.95",
            "--init-method-std": "0.02",
            "--clip-grad": "1.0",
        }
        for flag, value in training_replacements.items():
            argv = _replace_option(argv, flag, value)
        for flag in (
            "--finetune",
            "--no-pin-cpu-grads",
            "--no-pin-cpu-params",
        ):
            argv = _add_flag_once(argv, flag)
        argv = _remove_flag(argv, "--skip-train", has_value=False)
        argv = _replace_option(argv, "--eval-interval", str(eval_interval))

    return [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc_per_node={case.nproc_per_node}",
        "pretrain_gpt.py",
        *argv,
    ]


def _dmi_args(
    *,
    model_id: str,
    database: str,
    table: str,
    hook_selection: str,
    args: argparse.Namespace,
) -> list[str]:
    out = [
        "--dmi-enable",
        "--dmi-model-id",
        model_id,
        "--dmi-hook-selection",
        hook_selection,
        "--dmi-db-host",
        args.db_host,
        "--dmi-db-port",
        str(args.db_port),
        "--dmi-db-database",
        database,
        "--dmi-clickhouse-table",
        table,
        "--dmi-ch-parallelism",
        str(args.dmi_ch_parallelism),
        "--dmi-ring-payload-mb",
        str(args.dmi_ring_payload_mb),
        "--dmi-ring-pinned-mb",
        str(args.dmi_ring_pinned_mb),
        "--dmi-ring-task-entries",
        str(args.dmi_ring_task_entries),
        "--dmi-flush-every-n-train-iters",
        str(args.dmi_flush_every_n_train_iters),
    ]
    if args.dmi_vocab_logits_top_k is not None:
        out.extend(
            ["--dmi-vocab-logits-top-k", str(args.dmi_vocab_logits_top_k)]
        )
    return out


_DMI_VALUE_FLAGS = {
    "--dmi-model-id",
    "--dmi-hook-selection",
    "--dmi-db-host",
    "--dmi-db-port",
    "--dmi-db-database",
    "--dmi-clickhouse-table",
    "--dmi-ch-parallelism",
    "--dmi-ring-payload-mb",
    "--dmi-ring-pinned-mb",
    "--dmi-ring-task-entries",
    "--dmi-flush-every-n-train-iters",
    "--dmi-vocab-logits-top-k",
}

_DMI_BOOL_FLAGS = {
    "--dmi-enable",
}


def _workload_argv(cmd: list[str]) -> list[str]:
    """Return the framework workload argv used for controlled comparisons.

    The Python executable, torchrun launcher path, and DMI-only options are not
    part of the Megatron workload. Everything from pretrain_gpt.py onward must
    match for benchmark rows to be comparable.
    """
    try:
        start = cmd.index("pretrain_gpt.py")
    except ValueError as exc:
        raise ValueError(f"Cannot find pretrain_gpt.py in command: {cmd}") from exc
    out: list[str] = []
    i = start
    while i < len(cmd):
        token = cmd[i]
        if token in _DMI_BOOL_FLAGS:
            i += 1
            continue
        if token in _DMI_VALUE_FLAGS:
            i += 2
            continue
        out.append(token)
        i += 1
    return out


def _env(
    *,
    dmi_enabled: bool,
    benchmark_model: str = "mock",
    megatron_root: Path = DEFAULT_MEGATRON_ROOT,
) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{ROOT}:{megatron_root}:{GROKKING_SCRIPTS}:{env.get('PYTHONPATH', '')}"
    env.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", "1")
    if benchmark_model == "olmoe":
        env.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    if "DMI_REAL_E2E_CUDA_VISIBLE_DEVICES" in env:
        env["CUDA_VISIBLE_DEVICES"] = env["DMI_REAL_E2E_CUDA_VISIBLE_DEVICES"]
    if dmi_enabled:
        env["DMI_ENABLE"] = "1"
        env["DMI_DRAIN_FLUSH_PAYLOAD_RATIO"] = "0"
        env["DMI_DRAIN_FLUSH_TASK_RATIO"] = "0"
        env["DMI_DRAIN_FLUSH_BYTE_THRESHOLD"] = "0"
        env["DMI_DRAIN_FLUSH_ENTRY_THRESHOLD"] = "0"
        env["DMI_DRAIN_FLUSH_TIMEOUT_US"] = "0"
    else:
        env.pop("DMI_ENABLE", None)
        env.pop("DMI_TEST_FILE_SINK_DIR", None)
    return env


def _parse_log(log_path: Path) -> dict[str, Any]:
    text = log_path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    iter_ms = [float(x) for x in ITER_RE.findall(text)]
    iteration_ms_by_id = {
        int(iteration): float(elapsed_ms)
        for iteration, elapsed_ms in ITERATION_RECORD_RE.findall(text)
    }
    flush_s_by_iteration = {
        int(iteration): float(elapsed_s)
        for iteration, elapsed_s in ITERATION_FLUSH_RE.findall(text)
    }
    mem_matches = [tuple(float(v) for v in match) for match in MEM_RE.findall(text)]
    eval_blocks = _parse_evaluate_timer_blocks(lines)
    eval_max_ms_values = [float(block["max_ms"]) for block in eval_blocks]
    warmed_eval_blocks = eval_blocks[1:]
    warmed_eval_max_ms_values = [float(block["max_ms"]) for block in warmed_eval_blocks]
    out: dict[str, Any] = {
        "iteration_time_ms_values": iter_ms,
        "iteration_time_ms_by_iteration": iteration_ms_by_id,
        "iteration_flush_s_by_iteration": flush_s_by_iteration,
        "iteration_time_ms_mean": statistics.mean(iter_ms) if iter_ms else None,
        "iteration_time_ms_p50": statistics.median(iter_ms) if iter_ms else None,
        "iteration_time_ms_p95": _percentile(iter_ms, 95) if iter_ms else None,
        "evaluate_timer_blocks": eval_blocks,
        "evaluate_timer_max_ms_values": eval_max_ms_values,
        "evaluate_timer_max_ms_sum": sum(eval_max_ms_values) if eval_max_ms_values else None,
        "evaluate_timer_max_ms_mean": statistics.mean(eval_max_ms_values) if eval_max_ms_values else None,
        "evaluate_timer_max_ms_p50": statistics.median(eval_max_ms_values) if eval_max_ms_values else None,
        "evaluate_timer_max_ms_p95": _percentile(eval_max_ms_values, 95) if eval_max_ms_values else None,
        "evaluate_timer_discarded_warmup_blocks": eval_blocks[:1],
        "evaluate_timer_warmed_blocks": warmed_eval_blocks,
        "evaluate_timer_warmed_max_ms_values": warmed_eval_max_ms_values,
        "evaluate_timer_warmed_max_ms_sum": (
            sum(warmed_eval_max_ms_values) if warmed_eval_max_ms_values else None
        ),
        "evaluate_timer_warmed_max_ms_mean": (
            statistics.mean(warmed_eval_max_ms_values) if warmed_eval_max_ms_values else None
        ),
        "evaluate_timer_warmed_max_ms_p50": (
            statistics.median(warmed_eval_max_ms_values) if warmed_eval_max_ms_values else None
        ),
        "evaluate_timer_warmed_max_ms_p95": (
            _percentile(warmed_eval_max_ms_values, 95) if warmed_eval_max_ms_values else None
        ),
        "max_allocated_mb": max((m[1] for m in mem_matches), default=None),
        "max_reserved_mb": max((m[3] for m in mem_matches), default=None),
    }
    return out


def _parse_evaluate_timer_blocks(lines: list[str]) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    for index, line in enumerate(lines):
        match = EVALUATE_TIMER_RE.search(line)
        if match is None:
            continue
        phase = "unknown"
        for next_line in lines[index + 1 : index + 8]:
            phase_match = EVALUATE_PHASE_RE.search(next_line)
            if phase_match is None:
                continue
            phase = "valid" if phase_match.group(1) == "validation set" else "test"
            break
        blocks.append(
            {
                "phase": phase,
                "min_ms": float(match.group(1)),
                "max_ms": float(match.group(2)),
            }
        )
    return blocks


def _percentile(values: list[float], pct: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("empty values")
    idx = int(round((pct / 100.0) * (len(ordered) - 1)))
    return ordered[idx]


def _clickhouse_client(args: argparse.Namespace):
    from clickhouse_driver import Client

    return Client(host=args.db_host, port=int(args.db_port), database=args.db_database)


def _create_training_table(client, *, database: str, table: str) -> None:
    client.execute(f"CREATE DATABASE IF NOT EXISTS `{database}`")
    client.execute(
        f"""
        CREATE TABLE IF NOT EXISTS `{database}`.`{table}` (
            `model_id` String,
            `act_name` String,
            `direction` String,
            `phase` String,
            `global_batch_id` Int64,
            `dp_rank` Int32,
            `microbatch_id` Int32,
            `sample_index` Int32,
            `layer_no` Int32,
            `shard_rank` Int32,
            `token_start` Int64,
            `token_end` Int64,
            `attempt_id` Int32,
            `invocation_id` Int32,
            `dataset_id` Int32,
            `dtype` String,
            `shape` Array(Int64),
            `bytes` String
        ) ENGINE = MergeTree
        PRIMARY KEY (
            `model_id`, `act_name`, `direction`, `phase`, `global_batch_id`, `dp_rank`,
            `microbatch_id`, `sample_index`, `layer_no`, `shard_rank`,
            `token_start`, `token_end`, `attempt_id`, `invocation_id`
        )
        ORDER BY (
            `model_id`, `act_name`, `direction`, `phase`, `global_batch_id`, `dp_rank`,
            `microbatch_id`, `sample_index`, `layer_no`, `shard_rank`,
            `token_start`, `token_end`, `attempt_id`, `invocation_id`
        )
        """
    )


def _create_training_scalar_float_table(client, *, database: str, table: str) -> None:
    client.execute(f"CREATE DATABASE IF NOT EXISTS `{database}`")
    client.execute(
        f"""
        CREATE TABLE IF NOT EXISTS `{database}`.`{table}_scalar_float` (
            `model_id` String,
            `act_name` String,
            `direction` String,
            `phase` String,
            `global_batch_id` Int64,
            `dp_rank` Int32,
            `microbatch_id` Int32,
            `sample_index` Int32,
            `layer_no` Int32,
            `shard_rank` Int32,
            `token_start` Int64,
            `token_end` Int64,
            `attempt_id` Int32,
            `invocation_id` Int32,
            `dataset_id` Int32,
            `value` Float64
        ) ENGINE = MergeTree
        PRIMARY KEY (
            `model_id`, `act_name`, `direction`, `phase`, `global_batch_id`, `dp_rank`,
            `microbatch_id`, `sample_index`, `layer_no`, `shard_rank`,
            `token_start`, `token_end`, `attempt_id`, `invocation_id`
        )
        ORDER BY (
            `model_id`, `act_name`, `direction`, `phase`, `global_batch_id`, `dp_rank`,
            `microbatch_id`, `sample_index`, `layer_no`, `shard_rank`,
            `token_start`, `token_end`, `attempt_id`, `invocation_id`
        )
        """
    )


def _query_rows(
    client,
    *,
    database: str,
    table: str,
    model_id: str,
    hook_spec: HookBenchmarkSpec,
) -> tuple[int, dict[str, int]]:
    query_table = table + hook_spec.table_suffix
    rows = client.execute(
        f"""
        SELECT phase, count()
        FROM `{database}`.`{query_table}`
        WHERE model_id = %(model_id)s
          AND act_name IN %(act_names)s
          AND direction = 'fwd'
        GROUP BY phase
        """,
        {"model_id": model_id, "act_names": hook_spec.act_names},
    )
    phase_counts = {str(phase): int(count) for phase, count in rows}
    return sum(phase_counts.values()), phase_counts


def _run_one(
    *,
    case: BenchmarkCase,
    repetition: int,
    dmi_enabled: bool,
    args: argparse.Namespace,
    out_dir: Path,
    client,
) -> dict[str, Any]:
    hook_spec = HOOK_BENCHMARK_SPECS[str(args.hook_selection)]
    megatron_root = Path(args.megatron_root).resolve()
    if not megatron_root.exists():
        raise FileNotFoundError(f"Megatron root not found: {megatron_root}")
    pair_id = case.case_id
    hook_label = hook_spec.selection.replace("-", "_")
    run_id = f"{pair_id}_{hook_label}_r{repetition:02d}_{'dmi_on' if dmi_enabled else 'dmi_off'}_{uuid.uuid4().hex[:8]}"
    model_id = f"bench-{run_id}"
    table = f"{args.table_prefix}_{run_id}"
    log_path = out_dir / "logs" / f"{run_id}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    micro_batch_size = int(args.micro_batch_size)
    global_batch_size = micro_batch_size * int(case.num_microbatches)
    train_iters = int(case.train_iters)
    eval_iters = int(args.eval_iters)
    num_layers = _num_layers_for_benchmark(args)
    seq_length = _seq_length_for_benchmark(args)
    eval_only = bool(args.benchmark_model == "olmoe" and args.olmoe_eval_only)
    eval_interval = max(1, train_iters + 1) if eval_only else int(args.eval_interval)
    expected_phase_counts = _expected_counts_for_hook(
        hook_spec,
        train_iters=train_iters,
        eval_iters=eval_iters,
        eval_interval=eval_interval,
        global_batch_size=global_batch_size,
        num_moe_layers=num_layers,
        eval_only=eval_only,
    )
    if len(hook_spec.act_names) > 1:
        expected_phase_counts = {
            phase: count * len(hook_spec.act_names)
            for phase, count in expected_phase_counts.items()
        }
    expected_phase_counts = {k: v for k, v in expected_phase_counts.items() if v != 0}
    expected_rows = sum(expected_phase_counts.values())
    if args.benchmark_model == "mock":
        cmd = _base_cmd(
            case=case,
            train_iters=train_iters,
            eval_iters=eval_iters,
            eval_interval=eval_interval,
            micro_batch_size=micro_batch_size,
            global_batch_size=global_batch_size,
            seq_length=seq_length,
            seed=int(args.seed),
            mock_num_layers=int(args.mock_num_layers),
            mock_hidden_size=int(args.mock_hidden_size),
            mock_ffn_hidden_size=int(args.mock_ffn_hidden_size),
            mock_num_attention_heads=int(args.mock_num_attention_heads),
            mock_num_experts=int(args.mock_num_experts),
            mock_moe_router_topk=int(args.mock_moe_router_topk),
            log_interval=args.log_interval,
            no_masked_softmax_fusion=bool(args.no_masked_softmax_fusion),
        )
    elif args.benchmark_model == "olmoe":
        cmd = _olmoe_base_cmd(
            case=case,
            train_iters=train_iters,
            eval_iters=eval_iters,
            eval_interval=eval_interval,
            micro_batch_size=micro_batch_size,
            global_batch_size=global_batch_size,
            seq_length=seq_length,
            seed=int(args.seed),
            args=args,
        )
    else:
        raise ValueError(f"Unsupported benchmark model: {args.benchmark_model!r}")
    if dmi_enabled and not args.dry_run:
        _create_training_table(client, database=args.db_database, table=table)
        _create_training_scalar_float_table(client, database=args.db_database, table=table)
        cmd += _dmi_args(
            model_id=model_id,
            database=args.db_database,
            table=table,
            hook_selection=hook_spec.selection,
            args=args,
        )
    elif dmi_enabled:
        cmd += _dmi_args(
            model_id=model_id,
            database=args.db_database,
            table=table,
            hook_selection=hook_spec.selection,
            args=args,
        )

    record: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "case_id": run_id,
        "pair_id": pair_id,
        "repetition": int(repetition),
        "dmi_enabled": bool(dmi_enabled),
        "graph_mode": case.graph_mode,
        "parallel_name": case.parallel_name,
        "tp_size": case.tp_size,
        "pp_size": case.pp_size,
        "dp_size": 1,
        "ep_size": 1,
        "seq_length": seq_length,
        "micro_batch_size": micro_batch_size,
        "global_batch_size": global_batch_size,
        "train_iters": train_iters,
        "eval_iters": eval_iters,
        "eval_interval": eval_interval,
        "seed": int(args.seed),
        "benchmark_model": str(args.benchmark_model),
        "megatron_root": str(megatron_root),
        "dmi_mode": str(args.dmi_mode),
        "num_layers": int(num_layers),
        "hook_selection": hook_spec.selection,
        "hook_act_name": hook_spec.act_name,
        "hook_act_names": hook_spec.act_names,
        "hook_storage": hook_spec.storage,
        "expected_clickhouse_rows": expected_rows if dmi_enabled else None,
        "expected_phase_counts": expected_phase_counts if dmi_enabled else None,
        "model_id": model_id if dmi_enabled else None,
        "clickhouse_table": table if dmi_enabled else None,
        "log_path": str(log_path),
        "cmd": cmd,
        "workload_argv": _workload_argv(cmd),
    }

    if args.dry_run:
        record.update({"status": "dry_run", "elapsed_wall_s": 0.0})
        return record

    start = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log:
        result = subprocess.run(
            cmd,
            cwd=megatron_root,
            env=_env(
                dmi_enabled=dmi_enabled,
                benchmark_model=str(args.benchmark_model),
                megatron_root=megatron_root,
            ),
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=float(args.timeout_s),
            check=False,
        )
    elapsed = time.perf_counter() - start
    record["elapsed_wall_s"] = elapsed
    record["returncode"] = int(result.returncode)
    record.update(_parse_log(log_path))
    if result.returncode != 0:
        record["status"] = "failed"
        record["log_tail"] = log_path.read_text(encoding="utf-8", errors="replace")[-4000:]
        return record
    if dmi_enabled and int(args.dmi_flush_every_n_train_iters) > 0:
        flush_interval = int(args.dmi_flush_every_n_train_iters)
        expected_flush_ids = list(range(flush_interval, train_iters + 1, flush_interval))
        actual_flush_ids = sorted(int(x) for x in record["iteration_flush_s_by_iteration"])
        if actual_flush_ids != expected_flush_ids:
            record["status"] = "boundary_flush_mismatch"
            record["expected_boundary_flush_iterations"] = expected_flush_ids
            record["actual_boundary_flush_iterations"] = actual_flush_ids
            return record
        retained_ids = list(range(6, train_iters + 1))
        iteration_ms = record["iteration_time_ms_by_iteration"]
        missing_retained = [iteration for iteration in retained_ids if iteration not in iteration_ms]
        if missing_retained:
            record["status"] = "missing_iteration_timings"
            record["missing_retained_iteration_ids"] = missing_retained
            return record
        boundary_ids = [iteration for iteration in retained_ids if iteration % flush_interval == 0]
        non_boundary_ids = [iteration for iteration in retained_ids if iteration % flush_interval != 0]
        record["retained_iteration_ids"] = retained_ids
        record["boundary_iteration_ids"] = boundary_ids
        record["non_boundary_iteration_ids"] = non_boundary_ids
        record["retained_iteration_time_ms_values"] = [iteration_ms[x] for x in retained_ids]
        record["boundary_iteration_time_ms_values"] = [iteration_ms[x] for x in boundary_ids]
        record["non_boundary_iteration_time_ms_values"] = [iteration_ms[x] for x in non_boundary_ids]
    samples = global_batch_size * train_iters
    tokens = samples * seq_length
    record["samples_per_s"] = samples / elapsed if elapsed > 0 else None
    record["tokens_per_s"] = tokens / elapsed if elapsed > 0 else None

    if dmi_enabled:
        deadline = time.time() + float(args.clickhouse_timeout_s)
        actual_rows = 0
        actual_phase_counts: dict[str, int] = {}
        while time.time() < deadline:
            actual_rows, actual_phase_counts = _query_rows(
                client,
                database=args.db_database,
                table=table,
                model_id=model_id,
                hook_spec=hook_spec,
            )
            if actual_rows == expected_rows:
                break
            time.sleep(0.2)
        record["clickhouse_rows"] = actual_rows
        record["clickhouse_phase_counts"] = actual_phase_counts
        if actual_rows != expected_rows or actual_phase_counts != expected_phase_counts:
            record["status"] = "row_count_mismatch"
            return record

    record["status"] = "passed"
    return record


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def _fmt_float(value: float | None, suffix: str = "") -> str:
    if value is None:
        return "n/a"
    return f"{value:.3f}{suffix}"


def _fmt_pct(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.2%}"


def _write_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    by_run: dict[tuple[str, str, int], dict[bool, dict[str, Any]]] = {}
    for row in rows:
        if row.get("status") in {"skipped"}:
            continue
        key = (
            str(row["pair_id"]),
            str(row.get("hook_selection", "n/a")),
            int(row.get("repetition", 1)),
        )
        by_run.setdefault(key, {})[bool(row["dmi_enabled"])] = row

    lines = ["# Megatron DMI Benchmark Summary", ""]
    lines.append("## Per-Repetition Results")
    lines.append("")
    lines.append("| Pair | Hook | Rep | Off status | On status | Off wall s | On wall s | Wall overhead | Off warmed eval ms | On warmed eval ms | Warmed eval overhead | Rows |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")

    overheads_by_pair: dict[str, list[float]] = {}
    eval_overheads_by_pair: dict[str, list[float]] = {}
    off_s_by_pair: dict[str, list[float]] = {}
    on_s_by_pair: dict[str, list[float]] = {}
    off_eval_ms_by_pair: dict[str, list[float]] = {}
    on_eval_ms_by_pair: dict[str, list[float]] = {}
    rows_by_pair: dict[str, list[int]] = {}
    for pair_id, hook_selection, repetition in sorted(by_run):
        off = by_run[(pair_id, hook_selection, repetition)].get(False)
        on = by_run[(pair_id, hook_selection, repetition)].get(True)
        if off is None or on is None:
            continue
        off_s = float(off.get("elapsed_wall_s") or 0.0)
        on_s = float(on.get("elapsed_wall_s") or 0.0)
        overhead = ((on_s - off_s) / off_s) if off_s > 0 else None
        off_eval_ms = off.get("evaluate_timer_warmed_max_ms_sum")
        on_eval_ms = on.get("evaluate_timer_warmed_max_ms_sum")
        eval_overhead = (
            (float(on_eval_ms) - float(off_eval_ms)) / float(off_eval_ms)
            if off_eval_ms is not None and on_eval_ms is not None and float(off_eval_ms) > 0
            else None
        )
        if overhead is not None and off.get("status") == "passed" and on.get("status") == "passed":
            overheads_by_pair.setdefault(pair_id, []).append(overhead)
            off_s_by_pair.setdefault(pair_id, []).append(off_s)
            on_s_by_pair.setdefault(pair_id, []).append(on_s)
            if eval_overhead is not None:
                eval_overheads_by_pair.setdefault(pair_id, []).append(eval_overhead)
                off_eval_ms_by_pair.setdefault(pair_id, []).append(float(off_eval_ms))
                on_eval_ms_by_pair.setdefault(pair_id, []).append(float(on_eval_ms))
            if on.get("clickhouse_rows") is not None:
                rows_by_pair.setdefault(pair_id, []).append(int(on["clickhouse_rows"]))
        lines.append(
            "| {pair} | {hook} | {rep} | {off_status} | {on_status} | {off_s:.3f} | {on_s:.3f} | {overhead} | {off_eval} | {on_eval} | {eval_overhead} | {rows} |".format(
                pair=pair_id,
                hook=hook_selection,
                rep=repetition,
                off_status=off.get("status"),
                on_status=on.get("status"),
                off_s=off_s,
                on_s=on_s,
                overhead="n/a" if overhead is None else f"{overhead:.2%}",
                off_eval="n/a" if off_eval_ms is None else f"{float(off_eval_ms):.2f}",
                on_eval="n/a" if on_eval_ms is None else f"{float(on_eval_ms):.2f}",
                eval_overhead="n/a" if eval_overhead is None else f"{eval_overhead:.2%}",
                rows=on.get("clickhouse_rows", "n/a"),
            )
        )

    if overheads_by_pair:
        lines.append("")
        lines.append("## Aggregate Passed Runs")
        lines.append("")
        lines.append("| Pair | Hook | Reps | Off mean s | On mean s | Wall overhead mean | Wall overhead std | Warmed eval off mean ms | Warmed eval on mean ms | Warmed eval overhead mean | Warmed eval overhead std | Rows |")
        lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for pair_id in sorted(overheads_by_pair):
            overheads = overheads_by_pair[pair_id]
            overhead_std = statistics.stdev(overheads) if len(overheads) > 1 else 0.0
            eval_overheads = eval_overheads_by_pair.get(pair_id, [])
            eval_overhead_mean = statistics.mean(eval_overheads) if eval_overheads else None
            eval_overhead_std = statistics.stdev(eval_overheads) if len(eval_overheads) > 1 else 0.0
            row_values = sorted(set(rows_by_pair.get(pair_id, [])))
            row_text = ",".join(str(v) for v in row_values) if row_values else "n/a"
            lines.append(
                "| {pair} | {hook} | {reps} | {off_mean} | {on_mean} | {ov_mean} | {ov_std} | {eval_off_mean} | {eval_on_mean} | {eval_ov_mean} | {eval_ov_std} | {rows} |".format(
                    pair=pair_id,
                    hook=next(
                        (
                            str(row.get("hook_selection"))
                            for row in rows
                            if row.get("pair_id") == pair_id and row.get("hook_selection") is not None
                        ),
                        "n/a",
                    ),
                    reps=len(overheads),
                    off_mean=_fmt_float(statistics.mean(off_s_by_pair[pair_id])),
                    on_mean=_fmt_float(statistics.mean(on_s_by_pair[pair_id])),
                    ov_mean=_fmt_pct(statistics.mean(overheads)),
                    ov_std=_fmt_pct(overhead_std),
                    eval_off_mean=_fmt_float(
                        statistics.mean(off_eval_ms_by_pair[pair_id]) if pair_id in off_eval_ms_by_pair else None,
                        " ms",
                    ),
                    eval_on_mean=_fmt_float(
                        statistics.mean(on_eval_ms_by_pair[pair_id]) if pair_id in on_eval_ms_by_pair else None,
                        " ms",
                    ),
                    eval_ov_mean=_fmt_pct(eval_overhead_mean),
                    eval_ov_std=_fmt_pct(eval_overhead_std if eval_overheads else None),
                    rows=row_text,
                )
            )
    lines.append("")
    skipped = [row for row in rows if row.get("status") == "skipped"]
    if skipped:
        lines.append("## Skipped")
        lines.append("")
        for row in skipped:
            lines.append(f"- {row['pair_id']}: {row['skip_reason']}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--case", action="append", default=None, help="Run only case_id, e.g. eager_tp1_pp1")
    parser.add_argument(
        "--hook-selection",
        choices=sorted(HOOK_BENCHMARK_SPECS),
        default=os.environ.get("DMI_BENCH_HOOK_SELECTION", "router-summary"),
        help="Single DMI hook selection to benchmark.",
    )
    parser.add_argument(
        "--benchmark-model",
        choices=("mock", "olmoe"),
        default=os.environ.get("DMI_BENCH_MODEL", "mock"),
        help="Benchmark workload model. mock is the tiny regression model; olmoe uses the converted OLMoE checkpoint.",
    )
    parser.add_argument(
        "--megatron-root",
        default=os.environ.get("DMI_BENCH_MEGATRON_ROOT", str(DEFAULT_MEGATRON_ROOT)),
        help="Megatron checkout used to launch pretrain_gpt.py.",
    )
    parser.add_argument(
        "--dmi-mode",
        choices=("paired", "off-only", "on-only"),
        default=os.environ.get("DMI_BENCH_DMI_MODE", "paired"),
        help="Run DMI off/on pair, only pure baseline, or only monitored run.",
    )
    parser.add_argument("--repeat", type=int, default=int(os.environ.get("DMI_BENCH_REPEAT", "1")))
    parser.add_argument("--default-train-iters", type=int, default=int(os.environ.get("DMI_BENCH_TRAIN_ITERS", "10")))
    parser.add_argument("--eval-iters", type=int, default=int(os.environ.get("DMI_BENCH_EVAL_ITERS", "1")))
    parser.add_argument("--eval-interval", type=int, default=int(os.environ.get("DMI_BENCH_EVAL_INTERVAL", "5")))
    parser.add_argument(
        "--seq-length",
        type=int,
        default=int(os.environ["DMI_BENCH_SEQ_LENGTH"]) if "DMI_BENCH_SEQ_LENGTH" in os.environ else None,
    )
    parser.add_argument("--micro-batch-size", type=int, default=int(os.environ.get("DMI_BENCH_MICRO_BATCH_SIZE", "2")))
    parser.add_argument("--mock-num-layers", type=int, default=2)
    parser.add_argument("--mock-hidden-size", type=int, default=64)
    parser.add_argument("--mock-ffn-hidden-size", type=int, default=128)
    parser.add_argument("--mock-num-attention-heads", type=int, default=4)
    parser.add_argument("--mock-num-experts", type=int, default=2)
    parser.add_argument("--mock-moe-router-topk", type=int, default=1)
    parser.add_argument("--log-interval", type=int, default=None)
    parser.add_argument("--no-masked-softmax-fusion", action="store_true")
    parser.add_argument("--seed", type=int, default=int(os.environ.get("DMI_BENCH_SEED", "1234")))
    parser.add_argument("--timeout-s", type=float, default=float(os.environ.get("DMI_BENCH_TIMEOUT_S", "300")))
    parser.add_argument("--clickhouse-timeout-s", type=float, default=float(os.environ.get("DMI_BENCH_CLICKHOUSE_TIMEOUT_S", "30")))
    parser.add_argument("--db-host", default=os.environ.get("DMX_DB_HOST", "localhost"))
    parser.add_argument("--db-port", type=int, default=int(os.environ.get("DMX_DB_PORT", "9000")))
    parser.add_argument("--db-database", default=os.environ.get("DMX_DB_DATABASE", "default"))
    parser.add_argument("--table-prefix", default=os.environ.get("DMI_BENCH_TABLE_PREFIX", "dmi_megatron_bench"))
    parser.add_argument("--dmi-ring-payload-mb", type=int, default=int(os.environ.get("DMI_REAL_E2E_RING_PAYLOAD_MB", "64")))
    parser.add_argument("--dmi-ring-pinned-mb", type=int, default=int(os.environ.get("DMI_REAL_E2E_RING_PINNED_MB", "64")))
    parser.add_argument("--dmi-ring-task-entries", type=int, default=int(os.environ.get("DMI_REAL_E2E_RING_TASK_ENTRIES", "1024")))
    parser.add_argument(
        "--dmi-flush-every-n-train-iters",
        type=int,
        default=int(os.environ.get("DMI_BENCH_FLUSH_EVERY_N_TRAIN_ITERS", "0")),
        help="Force a durable DMI flush after every N accepted training iterations; zero disables it.",
    )
    parser.add_argument(
        "--dmi-vocab-logits-top-k",
        type=int,
        default=(
            int(os.environ["DMI_VOCAB_LOGITS_TOP_K"])
            if "DMI_VOCAB_LOGITS_TOP_K" in os.environ
            else None
        ),
        help="Fixed K for the vocab-logits-topk hook.",
    )
    parser.add_argument("--dmi-ch-parallelism", type=int, default=10)
    parser.add_argument(
        "--olmoe-megatron-dir",
        default=os.environ.get(
            "DMI_BENCH_OLMOE_MEGATRON_DIR",
            str(ROOT / "artifacts" / "grokking_repro" / "megatron_checkpoints" / "standalone_olmoe_1100k_local_tp1_pp1"),
        ),
    )
    parser.add_argument(
        "--olmoe-hf-template-dir",
        default=os.environ.get(
            "DMI_BENCH_OLMOE_HF_TEMPLATE_DIR",
            str(ROOT / "artifacts" / "grokking_repro" / "checkpoints" / "olmoe_1b7b_step1100000_tokens4613B"),
        ),
    )
    parser.add_argument(
        "--olmoe-data-prefix",
        default=os.environ.get(
            "DMI_BENCH_OLMOE_DATA_PREFIX",
            str(
                ROOT
                / "artifacts"
                / "grokking_repro"
                / "megatron_datasets"
                / "math_open_web_math_1000"
                / "math_open_web_math_1000_text_document"
            ),
        ),
    )
    parser.add_argument(
        "--olmoe-data-cache-path",
        default=os.environ.get(
            "DMI_BENCH_OLMOE_DATA_CACHE_PATH",
            str(ROOT / "artifacts" / "grokking_repro" / "megatron_datasets" / "index_cache"),
        ),
    )
    parser.add_argument("--olmoe-split", default=os.environ.get("DMI_BENCH_OLMOE_SPLIT", "1,49,50"))
    parser.add_argument("--olmoe-mock-data", action="store_true")
    parser.add_argument(
        "--olmoe-train",
        action="store_false",
        dest="olmoe_eval_only",
        help="Run OLMoE train/eval instead of eval-only. Default is eval-only.",
    )
    parser.set_defaults(olmoe_eval_only=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if int(args.repeat) < 1:
        raise ValueError("--repeat must be >= 1")
    if args.hook_selection not in HOOK_BENCHMARK_SPECS:
        raise ValueError(f"Unsupported benchmark hook selection: {args.hook_selection!r}")
    if args.hook_selection == "vocab-logits-topk":
        if args.dmi_vocab_logits_top_k is None:
            raise ValueError("vocab-logits-topk benchmark requires --dmi-vocab-logits-top-k")
    elif args.dmi_vocab_logits_top_k is not None:
        raise ValueError(
            "--dmi-vocab-logits-top-k requires --hook-selection vocab-logits-topk"
        )
    if args.dmi_mode == "off-only":
        dmi_modes = (False,)
    elif args.dmi_mode == "on-only":
        dmi_modes = (True,)
    else:
        dmi_modes = (False, True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir) if args.out_dir else ROOT / "artifacts" / "megatron_bench" / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)

    selected = set(args.case or [])
    cases = [case for case in _cases(args.default_train_iters) if not selected or case.case_id in selected]
    rows: list[dict[str, Any]] = []
    client = None if args.dry_run or args.dmi_mode == "off-only" else _clickhouse_client(args)
    try:
        for case in cases:
            if case.skip_reason is not None:
                rows.append(
                    {
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "pair_id": case.case_id,
                        "repetition": None,
                        "graph_mode": case.graph_mode,
                        "parallel_name": case.parallel_name,
                        "dmi_enabled": None,
                        "status": "skipped",
                        "skip_reason": case.skip_reason,
                    }
                )
                print(f"[DMI bench] skip {case.case_id}: {case.skip_reason}", flush=True)
                continue
            for repetition in range(1, int(args.repeat) + 1):
                for dmi_enabled in dmi_modes:
                    label = "on" if dmi_enabled else "off"
                    print(
                        f"[DMI bench] run {case.case_id} rep {repetition}/{args.repeat} DMI {label}",
                        flush=True,
                    )
                    row = _run_one(
                        case=case,
                        repetition=repetition,
                        dmi_enabled=dmi_enabled,
                        args=args,
                        out_dir=out_dir,
                        client=client,
                    )
                    rows.append(row)
                    print(
                        f"[DMI bench] {case.case_id} rep {repetition} DMI {label}: {row.get('status')} "
                        f"elapsed={row.get('elapsed_wall_s')}",
                        flush=True,
                    )
                    _write_jsonl(out_dir / "benchmark_results.jsonl", rows)
                    _write_summary(out_dir / "benchmark_summary.md", rows)
                    if row.get("status") not in {"passed", "dry_run"}:
                        return 1
    finally:
        if client is not None:
            client.disconnect()

    _write_jsonl(out_dir / "benchmark_results.jsonl", rows)
    _write_summary(out_dir / "benchmark_summary.md", rows)
    print(f"[DMI bench] wrote {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
