# Unsloth - 2x faster, 60% less VRAM LLM training and finetuning
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.

"""unsloth#409: the fused LoRA kernels must be declined under FSDP.

`apply_lora_qkv` / `apply_lora_o` / `apply_lora_mlp` read `.weight` off the
projection modules and matmul it themselves, so they never go through the module
call FSDP hooks its unshard onto. Under FSDP those weights are shard views: on a
2-rank FULL_SHARD run a LoRA A of shape (8, 896) arrives as a 1-D tensor of 3584
elements, and `matmul_lora` dies with

    RuntimeError: size mismatch, got input (20), mat (20x896), vec (3584)

which is the torch 2.14 spelling of #409's
`setStorage ... out of bounds for storage of size 0`.

Measured on 2 GPUs, Qwen2.5-0.5B LoRA SFT, `accelerate launch` with an FSDP1
config: before, every run died in `matmul_lora`; after (together with the
empty-logits sentinel fix), both ranks log `[3.5723, 3.9427, 3.5217, 3.4233]`.
The FSDP2 config on the same branch logs `[3.5781, 3.9453, 3.5234, 3.4297]`,
the fused path's own answer, so the fallback is tracking it.

No GPU: the probe is env-driven and the wiring check is textual.
"""

from __future__ import annotations

import ast
import os
from pathlib import Path

import pytest

from unsloth.models.loader_utils import fsdp_will_wrap

_LLAMA = Path(__file__).resolve().parent.parent / "unsloth" / "models" / "llama.py"

_FSDP_ENV = (
    "ACCELERATE_USE_FSDP",
    "FSDP_VERSION",
    "UNSLOTH_FORCE_FUSED_LORA",
)


@pytest.fixture(autouse = True)
def _clean_env(monkeypatch):
    for key in _FSDP_ENV:
        monkeypatch.delenv(key, raising = False)


def test_a_plain_launch_keeps_the_fused_kernels(monkeypatch):
    """`accelerate launch` without an FSDP config exports neither variable, and a
    single-GPU run exports nothing at all. Both must keep the fast path."""
    assert fsdp_will_wrap() is False
    monkeypatch.setenv("ACCELERATE_MIXED_PRECISION", "bf16")
    assert fsdp_will_wrap() is False


@pytest.mark.parametrize("value", ["true", "True", "1", "yes", "ON"])
def test_accelerate_use_fsdp_declines(monkeypatch, value):
    """What `accelerate launch --config_file <fsdp.yaml>` really exports; measured
    as the literal string "true"."""
    monkeypatch.setenv("ACCELERATE_USE_FSDP", value)
    assert fsdp_will_wrap() is True


@pytest.mark.parametrize("value", ["false", "0", "", "no"])
def test_a_falsy_accelerate_use_fsdp_keeps_the_fused_kernels(monkeypatch, value):
    monkeypatch.setenv("ACCELERATE_USE_FSDP", value)
    assert fsdp_will_wrap() is False


@pytest.mark.parametrize("value,expected", [("1", True), ("2", True), ("0", False), ("", False)])
def test_fsdp_version_alone_is_enough(monkeypatch, value, expected):
    """A torchrun user who sets the FSDP_* contract by hand, and the "0" a config
    writes to say it is not FSDP."""
    monkeypatch.setenv("FSDP_VERSION", value)
    assert fsdp_will_wrap() is expected


def test_the_override_wins_over_every_signal(monkeypatch):
    """`UNSLOTH_FORCE_FUSED_LORA=1` is how you measure what the fallback costs."""
    monkeypatch.setenv("ACCELERATE_USE_FSDP", "true")
    monkeypatch.setenv("FSDP_VERSION", "2")
    monkeypatch.setenv("UNSLOTH_FORCE_FUSED_LORA", "1")
    assert fsdp_will_wrap() is False


def test_a_live_accelerator_state_is_consulted_first(monkeypatch):
    """patch_peft_model usually runs before the Trainer builds its Accelerator, so
    the env is the normal signal. A caller who built one themselves must still be
    read: the env carries nothing in that case."""
    accelerate_state = pytest.importorskip("accelerate.state")
    shared = accelerate_state.AcceleratorState._shared_state
    monkeypatch.setitem(shared, "distributed_type", "DistributedType.FSDP")
    assert fsdp_will_wrap() is True


def test_the_fused_lora_install_is_gated_on_the_probe():
    """The three fused installs share one `if`, so gating that one `if` is the
    whole fix. Re-splitting them without carrying the gate turns this red."""
    source = _LLAMA.read_text(encoding = "utf-8")
    assert "fsdp_will_wrap" in source, "llama.py no longer probes for FSDP"

    tree = ast.parse(source)
    guarded = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        test = ast.unparse(node.test)
        if "fused_lora_declined_for_fsdp" not in test:
            continue
        body = ast.unparse(node)
        if "apply_qkv" in body or "apply_o" in body or "_apply_lora_mlp" in body:
            guarded.append(test)
    assert guarded, (
        "no `if` guarding apply_qkv / apply_o / apply_lora_mlp consults "
        "fused_lora_declined_for_fsdp; under FSDP the fused kernels will read "
        "shard views again"
    )


def test_the_declined_message_names_the_override():
    """A user who sees the slowdown has to be able to find the way back."""
    source = _LLAMA.read_text(encoding = "utf-8")
    index = source.index("fused_lora_declined_for_fsdp")
    window = source[index : index + 1200]
    assert "UNSLOTH_FORCE_FUSED_LORA" in window
