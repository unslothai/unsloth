# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""CPU tests: the diffusion LoRA trainers' torch.compile needs inductor's host C++ compiler on Windows. Without one
(a stock install has no ``cl``) the run must train eager with a warning, not die at the first step."""

from __future__ import annotations

import sys

import pytest

from core.training import diffusion_dit_trainer as dit
from core.training.diffusion_train_common import DiffusionLoraConfig


class _Transformer:
    def __init__(self):
        self.compiled = 0

    def compile_repeated_blocks(self, **kwargs):
        self.compiled += 1


def _cfg():
    return DiffusionLoraConfig(base_model = "b", data_dir = "d", output_dir = "o")


def _compile(
    monkeypatch,
    platform,
    compiler,
    rocm = True,
):
    import torch._inductor.cpp_builder as cpp_builder

    monkeypatch.setattr(dit, "torch_is_rocm", lambda: rocm)

    def lookup():
        if compiler is None:
            raise RuntimeError("Compiler: cl is not found.")
        return compiler

    monkeypatch.setattr(sys, "platform", platform)
    monkeypatch.setattr(cpp_builder, "get_cpp_compiler", lookup)
    events = []
    t = _Transformer()
    compiled = dit._maybe_compile_transformer(
        t, _cfg(), False, "cuda", events.append, base_precision = "bf16"
    )
    return compiled, t.compiled, [e for e in events if e.get("type") == "warning"]


def test_windows_without_a_cpp_compiler_trains_eager_with_a_warning(monkeypatch):
    compiled, calls, warnings = _compile(monkeypatch, "win32", None)
    assert compiled is False
    assert calls == 0
    assert len(warnings) == 1
    assert "cl is not found" in warnings[0]["message"]
    assert "eager" in warnings[0]["message"]


@pytest.mark.parametrize("platform, compiler", [("win32", "cl"), ("linux", None), ("linux", "g++")])
def test_compile_unchanged_where_the_compiler_resolves_or_off_windows(
    monkeypatch, platform, compiler
):
    compiled, calls, warnings = _compile(monkeypatch, platform, compiler)
    assert compiled is True
    assert calls == 1
    assert warnings == []


def test_windows_cuda_keeps_its_compile(monkeypatch):
    # Unmeasured on Windows CUDA: the gate must not turn off a compile that may have been running there.
    compiled, calls, warnings = _compile(monkeypatch, "win32", None, rocm = False)
    assert compiled is True
    assert calls == 1
    assert warnings == []
