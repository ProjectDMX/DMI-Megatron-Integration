"""Monitoring preprocessors must not attach their work to training autograd."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from dmi_megatron_integration.hooks.megatron_loss_summary import per_sample_loss_from_token_loss
from dmi_megatron_integration.hooks.megatron_router_logits import router_logits_by_sample
from dmi_megatron_integration.hooks.megatron_router_summary import (
    router_probs_mean_from_logits,
    router_token_entropy_mean_from_logits,
)


CASES = ("raw", "mean_softmax", "mean_sigmoid", "entropy_softmax", "entropy_sigmoid", "loss", "topk")


def _case(name, *, dtype, device):
    if name == "loss":
        loss = torch.arange(6, dtype=dtype, device=device).reshape(2, 3).requires_grad_()
        mask = torch.tensor([[1, 0, 1], [1, 1, 0]], dtype=dtype, device=device).requires_grad_()
        return per_sample_loss_from_token_loss, (loss, mask)

    logits = (torch.arange(24, dtype=dtype, device=device) / 10).reshape(3, 2, 4).requires_grad_()
    counts = torch.tensor([3, 1], dtype=torch.long, device=device)
    if name == "raw":
        return lambda value: (router_logits_by_sample(value),), (logits,)
    if name.startswith("mean_"):
        score = name.removeprefix("mean_")
        return lambda value: (router_probs_mean_from_logits(value, counts, score),), (logits,)
    if name.startswith("entropy_"):
        score = name.removeprefix("entropy_")
        return lambda value: (router_token_entropy_mean_from_logits(value, counts, score),), (logits,)
    if name == "topk":
        megatron_root = str(Path(__file__).resolve().parents[1] / "third_party/megatron-lm")
        if megatron_root not in sys.path:
            sys.path.insert(0, megatron_root)
        from megatron.core.transformer.moe.router import TopKRouter

        routing = torch.tensor(
            [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 0, 0],
             [0, 1, 0, 0], [1, 1, 0, 0], [0, 0, 1, 1]],
            dtype=torch.bool, device=device,
        )
        return lambda value: TopKRouter._dmi_router_topk_from_routing(
            SimpleNamespace(topk=2), value.reshape(6, 4), routing, 3, 2,
        ), (logits,)
    raise AssertionError(name)


@pytest.mark.parametrize("name", CASES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_preprocess_detaches_before_work_and_preserves_training_gradients(name, dtype):
    fn, inputs = _case(name, dtype=dtype, device="cpu")
    before = [value.detach().clone() for value in inputs]
    with torch.no_grad():
        expected = fn(*inputs)
    saved = []

    def pack(tensor):
        saved.append(tensor)
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        outputs = fn(*inputs)
    assert saved == [], "monitoring must not save tensors for backward"
    for output, reference in zip(outputs, expected):
        assert not output.requires_grad and output.grad_fn is None
        torch.testing.assert_close(output, reference)
    for value, original in zip(inputs, before):
        assert value.requires_grad
        torch.testing.assert_close(value, original)
    if name == "raw":
        assert outputs[0].data_ptr() == inputs[0].data_ptr()

    # Training continues to use the original tensors, not the detached views.
    sum(value.square().sum() for value in inputs).backward()
    for value in inputs:
        torch.testing.assert_close(value.grad, 2 * value.detach())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA Graph test requires CUDA")
@pytest.mark.parametrize("name", CASES)
def test_detached_preprocess_cuda_graph_replay_uses_current_inputs(name):
    fn, inputs = _case(name, dtype=torch.float32, device="cuda")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            fn(*inputs)
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outputs = fn(*inputs)
    for _ in range(2):
        graph.replay()
        torch.cuda.synchronize()
        with torch.no_grad():
            expected = fn(*inputs)
        for output, reference in zip(outputs, expected):
            assert output.is_cuda
            assert not output.requires_grad and output.grad_fn is None
            torch.testing.assert_close(output, reference)
        with torch.no_grad():
            inputs[0].mul_(1.25).add_(0.125)
