# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The Qwen3.5/3.6 thinking-default size gate in llama_cpp.py's launch path.

The gate is inline in _build_command with no seam, so the pattern is read back out of the
source and exercised here. Keep this in sync with the two frontend mirrors.
"""

import re
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parent.parent / "core" / "inference" / "llama_cpp.py"


def _gate_pattern() -> str:
    src = _SRC.read_text(encoding = "utf-8")
    match = re.search(r'size_re = r"([^"]+)"', src)
    assert match, "size gate pattern not found in llama_cpp.py"
    return match.group(1)


def _thinking_default_off(model_identifier: str) -> bool:
    mid = (model_identifier or "").lower()
    if "qwen3.5" not in mid and "qwen3.6" not in mid:
        return False
    size_re = _gate_pattern()
    mid_slash = mid.replace("\\", "/")
    size_match = re.search(size_re, mid_slash.split("/")[-1]) or re.search(size_re, mid_slash)
    return bool(size_match) and float(size_match.group(1)) < 9


@pytest.mark.parametrize(
    "model_id",
    [
        "unsloth/Qwen3.5-35B-A3B-GGUF",
        "unsloth/Qwen3.6-35B-A3B-MTP-GGUF",
        "unsloth/Qwen3.5-122B-A10B-GGUF",
    ],
)
def test_moe_total_params_win_over_active_params(model_id):
    # extract_model_size_b() prefers A3B and reads 35B-A3B as 3B, which turned thinking off
    # on a medium-tier model.
    assert _thinking_default_off(model_id) is False


@pytest.mark.parametrize(
    "model_id",
    [
        "unsloth/Qwen3.5-4B-GGUF",
        "unsloth/Qwen3.5-0.8B-GGUF",
        # Directory identifiers: auto-switch passes a snapshot dir, scan folders a quant subdir.
        "/models/Qwen3.5-4B-GGUF/UD-Q4_K_XL",
        "/c/models--unsloth--Qwen3.5-4B-GGUF/snapshots/bfc15c3",
        "C:\\models\\Qwen3.5-4B.gguf",
        "unsloth/Qwen3.5-4B/",
        "C:\\models\\Qwen3.5-4B\\",
    ],
)
def test_sub_9b_turns_thinking_off(model_id):
    assert _thinking_default_off(model_id) is True


@pytest.mark.parametrize("model_id", ["/models/8bit/qwen3.6-27b.gguf", "/models/8b/qwen3.6-27b.gguf"])
def test_size_like_directory_does_not_shadow_the_real_size(model_id):
    assert _thinking_default_off(model_id) is False


@pytest.mark.parametrize("model_id", ["unsloth/Qwen3-4B-GGUF", "unsloth/gemma-4-12b-it-GGUF", ""])
def test_other_models_are_never_gated(model_id):
    assert _thinking_default_off(model_id) is False
