# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""A checkpoint recompute only needs the autograd graph of the layer it re-runs, so LoRA_MLP skips act(e, g)
and down_proj there, in eager layers and in compiled decoder blocks. Gradients must stay bit-identical."""

import types

import pytest
from real_accelerator import has_real_cuda
import torch


def _active(fast_lora):
    return fast_lora._MLP_OUTPUT_UNUSED[0]


def test_skip_needs_own_checkpoint_recompute_with_grad(monkeypatch):
    from unsloth.kernels import fast_lora

    def gate(own_checkpoint, grad, recompute):
        monkeypatch.setattr(fast_lora, "in_gradient_checkpoint_recompute", lambda: recompute)
        layer = types.SimpleNamespace(gradient_checkpointing = own_checkpoint)
        with torch.set_grad_enabled(grad), fast_lora.mlp_output_unused(layer):
            return _active(fast_lora)

    assert gate(True, True, True)
    # Not checkpointed alone (inside a larger block), forward inside an outer checkpoint, or no recompute.
    assert not gate(False, True, True)
    assert not gate(True, False, True)
    assert not gate(True, True, False)
    assert not _active(fast_lora)


def test_flag_is_cleared_when_the_mlp_raises(monkeypatch):
    from unsloth.kernels import fast_lora

    monkeypatch.setattr(fast_lora, "in_gradient_checkpoint_recompute", lambda: True)
    layer = types.SimpleNamespace(gradient_checkpointing = True)
    with pytest.raises(RuntimeError), torch.enable_grad(), fast_lora.mlp_output_unused(layer):
        raise RuntimeError
    assert not _active(fast_lora)


def _model(model_name):
    from unsloth import FastLanguageModel

    # max_seq_length >= 512 selects Unsloth's own checkpointing, the one that marks its recompute.
    model, _ = FastLanguageModel.from_pretrained(
        model_name, max_seq_length = 512, load_in_4bit = False, dtype = torch.bfloat16
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r = 8,
        lora_alpha = 8,
        lora_dropout = 0,
        bias = "none",
        target_modules = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        use_gradient_checkpointing = "unsloth",
        random_state = 0,
    )
    generator = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "lora_B" in name:
                # Zero B leaves every A gradient zero, which would compare equal for any skip.
                param.copy_(torch.randn(param.shape, generator = generator) * 0.02)
    model.train()
    ids = torch.randint(0, model.config.vocab_size, (2, 32), generator = generator).cuda()
    return model, ids


def _grads(model, ids):
    model.zero_grad(set_to_none = True)
    # The math backend is deterministic, so the two runs can be compared bit for bit.
    with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
        model(input_ids = ids, labels = ids, use_cache = False).loss.backward()
    return {n: p.grad.detach().clone() for n, p in model.named_parameters() if p.grad is not None}


@pytest.mark.gpu
@pytest.mark.skipif(not has_real_cuda(), reason = "CUDA Triton kernels required")
@pytest.mark.parametrize("block_compile", ["0", "auto"])
@pytest.mark.parametrize(
    "model_name",
    [
        "trl-internal-testing/tiny-Qwen3ForCausalLM",
        "trl-internal-testing/tiny-LlamaForCausalLM-3.2",
    ],
)
def test_recompute_skips_mlp_tail_with_identical_grads(model_name, block_compile, monkeypatch):
    from unsloth.kernels import fast_lora

    # Read when get_peft_model installs the per-block compile: "0" keeps every layer eager.
    monkeypatch.setenv("UNSLOTH_BLOCK_COMPILE_LEGACY", block_compile)
    real = fast_lora.in_gradient_checkpoint_recompute
    raised = []

    def recompute():
        # Called only outside compiled code: by the eager layer, or by the block wrapper.
        active = real()
        raised.append(active)
        return active

    monkeypatch.setattr(fast_lora, "in_gradient_checkpoint_recompute", lambda: False)
    model, ids = _model(model_name)
    ref = _grads(model, ids)
    monkeypatch.setattr(fast_lora, "in_gradient_checkpoint_recompute", recompute)
    grads = _grads(model, ids)

    # The skip is entered once per layer, in its recompute.
    assert sum(raised) == model.config.num_hidden_layers, raised
    assert ref.keys() == grads.keys() and any("lora_A" in n for n in ref)
    assert all(ref[n].abs().sum() > 0 for n in ref)
    mismatched = [n for n in ref if not torch.equal(ref[n], grads[n])]
    assert not mismatched, mismatched
