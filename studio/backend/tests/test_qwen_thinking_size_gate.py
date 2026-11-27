# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Slices the inline Qwen thinking-size gate out of llama_cpp source, so control flow is tested too."""

import re
import textwrap
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parent.parent / "core" / "inference" / "llama_cpp.py"
_START = 'mid = (model_identifier or "").lower()'
_END = "thinking_default = False"


def _gate_source() -> str:
    lines = _SRC.read_text(encoding = "utf-8").splitlines()
    start = next(i for i, line in enumerate(lines) if line.strip() == _START)
    end = next(i for i, line in enumerate(lines[start:], start) if line.strip() == _END)
    return textwrap.dedent("\n".join(lines[start : end + 1]))


_GATE = compile(_gate_source(), str(_SRC), "exec")


def _thinking_default_off(model_identifier: str) -> bool:
    scope = {"re": re, "model_identifier": model_identifier, "thinking_default": True}
    exec(_GATE, scope)
    return scope["thinking_default"] is False


@pytest.mark.parametrize(
    "model_id",
    [
        "unsloth/Qwen3.5-35B-A3B-GGUF",
        "unsloth/Qwen3.6-35B-A3B-MTP-GGUF",
        "unsloth/Qwen3.5-122B-A10B-GGUF",
    ],
)
def test_moe_total_params_win_over_active_params(model_id):
    # extract_model_size_b() prefers A3B and reads 35B-A3B as 3B.
    assert _thinking_default_off(model_id) is False


@pytest.mark.parametrize(
    "model_id",
    [
        "unsloth/Qwen3.5-4B-GGUF",
        "unsloth/Qwen3.5-0.8B-GGUF",
        "unsloth/Qwen3.5-9B-GGUF",
        "/models/Qwen3.5-4B-GGUF/UD-Q4_K_XL",
        "/c/models--unsloth--Qwen3.5-4B-GGUF/snapshots/bfc15c3",
        "C:\\models\\Qwen3.5-4B.gguf",
        "unsloth/Qwen3.5-4B/",
        "C:\\models\\Qwen3.5-4B\\",
    ],
)
def test_sub_9b_turns_thinking_off(model_id):
    assert _thinking_default_off(model_id) is True


@pytest.mark.parametrize(
    "model_id",
    [
        "/models/8bit/qwen3.6-27b.gguf",
        "/models/8b/qwen3.6-27b.gguf",
        "/models/8b/Qwen3.5-35B-A3B/UD-Q4_K_XL",
        "/models/4b/Qwen3.6-27B-GGUF/snapshots/bfc15c3",
    ],
)
def test_size_like_directory_does_not_shadow_the_real_size(model_id):
    assert _thinking_default_off(model_id) is False


@pytest.mark.parametrize(
    "model_id",
    ["unsloth/Qwen3-4B-GGUF", "unsloth/gemma-4-12b-it-GGUF", "unsloth/Qwen3.5-9.5B-GGUF", ""],
)
def test_other_models_are_never_gated(model_id):
    assert _thinking_default_off(model_id) is False


@pytest.mark.parametrize(
    "model_id", ["Qwen3.5-4 B-GGUF", "unsloth/Qwen3.5-800M-GGUF", "unsloth/Qwen3.5-4 B"]
)
def test_spacing_and_millions_match_extract_model_size_b(model_id):
    # extract_model_size_b allows \s* before the unit and M suffixes; the inline matcher must too.
    assert _thinking_default_off(model_id) is True


@pytest.mark.parametrize(
    "quant",
    ["Q4_K_M", "Q3_K_M", "IQ3_M", "UD-Q4_K_XL", "Q8_0", "BF16", "MXFP4"],
)
def test_quant_subdirs_never_read_as_a_size(quant):
    # The M suffix is the risk here: a quant name is the leaf segment, so it is scanned first.
    assert _thinking_default_off(f"unsloth/Qwen3.5-35B-A3B-GGUF/{quant}") is False
    assert _thinking_default_off(f"unsloth/Qwen3.5-4B-GGUF/{quant}") is True
