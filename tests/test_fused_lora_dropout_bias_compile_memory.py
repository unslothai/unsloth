# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Dropout / lora_B bias fusion must not cost graph breaks or resident memory versus PEFT.

Before torch 2.11 the fused LoRA Functions run behind torch._dynamo.disable (fast_lora._apply), so a
dropout or bias projection routed into them while compiling turns a PEFT forward Dynamo traced into a
graph break (fullgraph=True raises). Those configurations stay on PEFT while compiling there; plain
LoRA keeps its existing routing. Eagerly, the dropout / bias path dequantizes 4-bit weights into a
fresh tensor rather than the per-device global scratch, which PEFT never kept resident.
"""

from __future__ import annotations

import pytest
import torch
import unsloth  # noqa: F401

from real_accelerator import has_real_cuda
from unsloth.kernels import fast_lora
from unsloth.kernels import utils as kernel_utils

H, I = 32, 64
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


class _Block(torch.nn.Module):
    def __init__(self, linear):
        super().__init__()
        for name in TARGETS:
            shape = (
                (I, H)
                if name == "down_proj"
                else ((H, I) if name in ("gate_proj", "up_proj") else (H, H))
            )
            setattr(self, name, linear(*shape))
        self.act_fn = torch.nn.SiLU()


def _peft_block(
    dropout,
    lora_bias,
    linear = None,
    device = "cpu",
    dtype = torch.float32,
):
    from peft import LoraConfig, get_peft_model

    torch.manual_seed(3407)
    block = _Block(linear or (lambda i, o: torch.nn.Linear(i, o, bias = False)))
    block = block.to(dtype = dtype).to(device = device)
    if linear is not None:
        block.is_loaded_in_4bit = True
    cfg = LoraConfig(
        r = 4,
        lora_alpha = 8,
        lora_dropout = dropout,
        target_modules = TARGETS,
        init_lora_weights = False,
        lora_bias = lora_bias,
    )
    model = get_peft_model(block, cfg)
    for name, p in model.named_parameters():
        p.requires_grad_("lora_" in name)
    model.train()
    return model, model.base_model.model


def _emulate_torch_before_2_11(monkeypatch):
    # What importing fast_lora does on torch < 2.11.
    def _apply(function, *args):
        return function.apply(*args)

    monkeypatch.setattr(fast_lora, "TRACE_LORA_FUNCTIONS", False)
    monkeypatch.setattr(fast_lora, "_apply", torch._dynamo.disable(_apply))


CASES = [(0.1, False), (0.0, True), (0.1, True)]


@pytest.mark.parametrize("dropout,lora_bias", CASES)
@pytest.mark.parametrize("path", ["o", "qkv"])
def test_fullgraph_compiles_before_torch_2_11(monkeypatch, dropout, lora_bias, path):
    _emulate_torch_before_2_11(monkeypatch)
    model, block = _peft_block(dropout, lora_bias)
    if path == "o":
        fn = lambda X: fast_lora.apply_lora_o(block, X)
    else:
        fn = lambda X: sum(fast_lora.apply_lora_qkv(block, X, inplace = False))
    torch._dynamo.reset()
    compiled = torch.compile(fn, backend = "eager", fullgraph = True)
    X = torch.randn(2, 3, H, requires_grad = True)
    compiled(X * 1).sum().backward()
    assert X.grad is not None and torch.isfinite(X.grad).all()

    # Eagerly the same configuration still runs fused.
    calls = []
    real = fast_lora._apply
    monkeypatch.setattr(fast_lora, "_apply", lambda f, *a: calls.append(f) or real(f, *a))
    fn(torch.randn(2, 3, H))
    assert len(calls) == 1
    torch._dynamo.reset()


@pytest.mark.parametrize("dropout,lora_bias", CASES)
def test_no_new_graph_breaks_before_torch_2_11(monkeypatch, dropout, lora_bias):
    from torch._dynamo.utils import counters

    _emulate_torch_before_2_11(monkeypatch)
    model, block = _peft_block(dropout, lora_bias)
    torch._dynamo.reset()
    counters.clear()
    compiled = torch.compile(lambda X: fast_lora.apply_lora_o(block, X), backend = "eager")
    compiled(torch.randn(2, 3, H))
    assert sum(counters["graph_break"].values()) == 0, dict(counters["graph_break"])
    torch._dynamo.reset()


@pytest.mark.parametrize("dropout,lora_bias", CASES)
@pytest.mark.parametrize("path", ["o", "qkv", "mlp"])
def test_dropout_and_bias_paths_keep_no_global_dequant_scratch(
    monkeypatch, dropout, lora_bias, path
):
    if not has_real_cuda():
        pytest.skip("needs a CUDA GPU for bitsandbytes 4-bit weights")
    bnb = pytest.importorskip("bitsandbytes")
    linear = lambda i, o: bnb.nn.Linear4bit(
        i, o, bias = False, compute_dtype = torch.float16, quant_type = "nf4"
    )
    model, block = _peft_block(dropout, lora_bias, linear, "cuda", torch.float16)
    calls = []
    real = fast_lora._apply
    monkeypatch.setattr(fast_lora, "_apply", lambda f, *a: calls.append(f) or real(f, *a))
    monkeypatch.setattr(kernel_utils, "_SCRATCH", {})
    X = torch.randn(2, 3, H, device = "cuda", dtype = torch.float16, requires_grad = True)
    fn = {
        "o": lambda X: fast_lora.apply_lora_o(block, X),
        "qkv": lambda X: sum(fast_lora.apply_lora_qkv(block, X, inplace = False)),
        "mlp": lambda X: fast_lora.apply_lora_mlp_swiglu(block, X, inplace = False),
    }[path]
    with torch.autocast("cuda", dtype = torch.float16):
        out = fn(X * 1)
    out.float().sum().backward()
    assert len(calls) == 1, "the fused kernel did not run"
    assert X.grad is not None
    assert not any(key[0] == "weight" for key in kernel_utils._SCRATCH), list(kernel_utils._SCRATCH)
