from __future__ import annotations

import sys
from pathlib import Path
from types import MethodType, SimpleNamespace

import torch
from torch import nn


MEGATRON_ROOT = Path(__file__).resolve().parents[1] / "third_party" / "megatron-lm"
if str(MEGATRON_ROOT) not in sys.path:
    sys.path.insert(0, str(MEGATRON_ROOT))

from megatron.core.models.gpt.gpt_model import GPTModel


class _FakeOutputLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.sequence_parallel = False

    def forward(self, hidden_states, *, weight=None, runtime_gather_output=None):
        del weight, runtime_gather_output
        return hidden_states, None


def _fake_gpt(hook, *, topk_hook=None):
    model = GPTModel.__new__(GPTModel)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(
        mtp_num_layers=0,
        use_mup=True,
        mup_output_mult=2.0,
        config_logger_dir="",
    )
    model.post_process = True
    model.share_embeddings_and_output_weights = False
    model.output_layer = _FakeOutputLayer()
    model.dmi_vocab_logits = hook
    model.dmi_vocab_logits_topk = topk_hook

    def compute_loss(self, labels, logits):
        del self, labels
        return logits.sum()

    model.compute_language_model_loss = MethodType(compute_loss, model)
    return model


def _postprocess(model, hidden_states):
    return model._postprocess(
        hidden_states=hidden_states,
        input_ids=None,
        position_ids=None,
        labels=torch.zeros((1, 2), dtype=torch.long),
        rotary_pos_emb=None,
        rotary_pos_cos=None,
        rotary_pos_sin=None,
        mtp_in_postprocess=False,
        loss_mask=None,
        decoder_input=None,
        attention_mask=None,
        inference_params=None,
        packed_seq_params=None,
        sequence_len_offset=None,
        runtime_gather_output=None,
        extra_block_kwargs=None,
        inference_context=None,
        is_spec_decode=None,
    )


def test_gpt_vocab_hook_observes_post_scale_pre_loss_logits_without_changing_gradient():
    captured = []
    hooked_input = torch.arange(6, dtype=torch.float32).reshape(2, 1, 3).requires_grad_(True)
    reference_input = hooked_input.detach().clone().requires_grad_(True)

    hooked_loss = _postprocess(_fake_gpt(captured.append), hooked_input)
    reference_loss = _postprocess(_fake_gpt(None), reference_input)
    hooked_loss.backward()
    reference_loss.backward()

    assert len(captured) == 1
    assert torch.equal(captured[0], hooked_input.detach() * 2.0)
    assert hooked_loss.detach() == reference_loss.detach()
    assert torch.equal(hooked_input.grad, reference_input.grad)


def test_gpt_vocab_hook_is_not_called_without_labels():
    captured = []
    model = _fake_gpt(captured.append)
    logits = model._postprocess(
        hidden_states=torch.ones((2, 1, 3)),
        input_ids=None,
        position_ids=None,
        labels=None,
        rotary_pos_emb=None,
        rotary_pos_cos=None,
        rotary_pos_sin=None,
        mtp_in_postprocess=False,
        loss_mask=None,
        decoder_input=None,
        attention_mask=None,
        inference_params=None,
        packed_seq_params=None,
        sequence_len_offset=None,
        runtime_gather_output=None,
        extra_block_kwargs=None,
        inference_context=None,
        is_spec_decode=None,
    )

    assert captured == []
    assert torch.equal(logits, torch.full((1, 2, 3), 2.0))


def test_gpt_vocab_topk_hook_observes_same_post_scale_pre_loss_logits():
    dense_captured = []
    topk_captured = []
    hidden_states = torch.arange(6, dtype=torch.float32).reshape(2, 1, 3)

    loss = _postprocess(
        _fake_gpt(dense_captured.append, topk_hook=topk_captured.append),
        hidden_states,
    )

    expected = hidden_states * 2.0
    assert loss == expected.sum()
    assert len(dense_captured) == 1
    assert len(topk_captured) == 1
    assert torch.equal(dense_captured[0], expected)
    assert torch.equal(topk_captured[0], expected)
