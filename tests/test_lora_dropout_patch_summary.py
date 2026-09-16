# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""The patched layers summary says why the fused LoRA kernels were skipped.

unsloth#2076. patch_peft_model only installs the fused kernels when
`lora_dropout == 0 and bias == "none"`, so anyone who passes lora_dropout = 0.1 (or a bias
term) gets

    Unsloth 2026.x patched 32 layers with 0 QKV layers, 0 O layers and 0 MLP layers.

which reads as a failed patch even though training is fine. _fused_lora_skip_reason supplies
the missing clause. It is a separate function so the wording can be tested without loading
a checkpoint, and the gate parity test below is what stops the two from drifting apart.
"""

from __future__ import annotations

import ast
import inspect
import itertools
import textwrap

import pytest
import torch
import unsloth  # noqa: F401

from unsloth.models import llama as llama_module
from unsloth.models.llama import _fused_lora_skip_reason


def test_no_reason_when_the_fused_kernels_were_installed():
    assert _fused_lora_skip_reason(0, "none") == ""
    assert _fused_lora_skip_reason(0.0, "none") == ""


def test_lora_dropout_is_named_with_its_value():
    reason = _fused_lora_skip_reason(0.1, "none")
    assert "lora_dropout = 0.1" in reason
    assert "bias" not in reason
    assert "Training is unaffected." in reason


def test_bias_is_named_with_its_value():
    reason = _fused_lora_skip_reason(0, "all")
    assert "bias = 'all'" in reason
    assert "lora_dropout" not in reason


def test_both_reasons_are_joined():
    reason = _fused_lora_skip_reason(0.05, "lora_only")
    assert "lora_dropout = 0.05" in reason
    assert "bias = 'lora_only'" in reason
    assert " and " in reason


def test_reason_appends_cleanly_to_the_summary_sentence():
    """The summary line ends in a full stop and the reason is appended straight onto it,
    so the reason must start with a space and must not start a new sentence mid word."""
    reason = _fused_lora_skip_reason(0.1, "none")
    assert reason.startswith(" ")
    assert reason.endswith(".")
    assert "  " not in ("...0 MLP layers." + reason)


@pytest.mark.parametrize(
    "lora_dropout,bias",
    list(itertools.product([0, 0.0, 0.1, 0.5], ["none", "all", "lora_only"])),
)
def test_reason_is_non_empty_exactly_when_the_fused_gate_is_closed(lora_dropout, bias):
    """Parity with the `lora_dropout == 0 and bias == "none"` gate in patch_peft_model."""
    fused_installed = lora_dropout == 0 and bias == "none"
    assert bool(_fused_lora_skip_reason(lora_dropout, bias)) is not fused_installed


def _gate_tests(source: str) -> list[str]:
    """Every `if` test in patch_peft_model that reads both lora_dropout and bias."""
    tree = ast.parse(textwrap.dedent(source))
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        names = {n.id for n in ast.walk(node.test) if isinstance(n, ast.Name)}
        if {"lora_dropout", "bias"} <= names:
            out.append(ast.unparse(node.test))
    return out


def test_patch_peft_model_still_gates_the_fused_kernels_on_the_same_two_values():
    """If the gate grows a third condition, _fused_lora_skip_reason has to grow with it or
    the summary starts reporting zero counts with no reason again."""
    source = inspect.getsource(llama_module.FastLlamaModel.patch_peft_model)
    assert _gate_tests(source) == ["lora_dropout == 0 and bias == 'none'"]


def test_the_summary_call_carries_the_reason():
    source = inspect.getsource(llama_module.FastLlamaModel.patch_peft_model)
    assert "unfused_reason = _fused_lora_skip_reason(lora_dropout, bias)" in source
    assert "MLP layers.{unfused_reason}" in source


@pytest.mark.gpu
@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason = "loads a real checkpoint through FastLanguageModel; needs an accelerator",
)
def test_summary_reason_is_logged_for_a_real_model():
    """unsloth#2076 end to end: the reason has to reach the user's console."""
    import logging

    from unsloth import FastLanguageModel

    records = []

    class _Capture(logging.Handler):
        def emit(self, record):
            records.append(record.getMessage())

    model, _tokenizer = FastLanguageModel.from_pretrained(
        "unsloth/Llama-3.2-1B-Instruct",
        max_seq_length = 512,
        load_in_4bit = True,
    )
    # warning_once (patched onto logging.Logger by transformers) dedupes process wide.
    getattr(llama_module.logger.warning_once, "cache_clear", lambda: None)()

    # Attach to the logger llama.py actually writes to. Naming it here rather than the
    # root logger keeps the test correct whether or not transformers has switched off
    # propagation on its own library logger.
    handler = _Capture()
    target = llama_module.logger
    target.addHandler(handler)
    previous_level = target.level
    target.setLevel(logging.WARNING)
    try:
        model = FastLanguageModel.get_peft_model(
            model,
            r = 8,
            lora_alpha = 16,
            lora_dropout = 0.1,
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"],
            random_state = 0,
        )
    finally:
        target.removeHandler(handler)
        target.setLevel(previous_level)

    summary = [m for m in records if "MLP layers." in m]
    assert summary, records
    assert "lora_dropout = 0.1" in summary[-1], summary[-1]
