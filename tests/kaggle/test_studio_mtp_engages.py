# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Selecting the MTP repo is a REQUEST; the drafter engaging is the result.

`--chat-model unsloth/Qwen3.5-2B-MTP-GGUF` was chosen because multi-token
prediction is a distinct serving path in llama.cpp. But when the companion is
missing, the binary was built without MTP, or the drafter is downgraded for
VRAM, the main GGUF still loads and still generates -- so every inference
assertion stays green while the path the repo was chosen for never ran. That is
the shape this directory keeps being caught by: nothing red, and a claim nobody
made.

`spec_drafter_kind` is what settles it, and `spec_fallback_reason` names WHICH
of those happened, since "llama.cpp has no MTP" and "the drafter was downgraded
for VRAM" are different findings and only one is about this leg.
"""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PAYLOAD = ROOT / "tests" / "kaggle" / "studio_gpu" / "run_studio_gpu.py"
SRC = PAYLOAD.read_text(encoding = "utf-8")
TREE = ast.parse(SRC)


def _func(name: str) -> ast.FunctionDef:
    for node in ast.walk(TREE):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{name} is gone")


def test_the_default_chat_model_is_the_mtp_repo():
    """If this stops being an MTP repo the assertion below becomes vacuous, so
    it fails here rather than passing silently."""
    assert "unsloth/Qwen3.5-2B-MTP-GGUF" in SRC


def test_the_status_fields_are_read_from_the_load():
    body = ast.unparse(_func("load_model"))
    for field in ("spec_drafter_kind", "spec_fallback_reason", "llama_cpp_supports_mtp"):
        assert field in body, f"{field} is never read, so the drafter cannot be checked"


def test_a_drafter_that_never_engaged_is_a_failure():
    """The claim itself. Without this, an MTP repo whose drafter fell back
    serves ordinary decoding and reports success."""
    func = _func("assert_gpu_inference")
    body = ast.unparse(func)
    assert "drafter_kind" in body, "the drafter kind is never compared"
    guards = [
        ast.unparse(node.test)
        for node in ast.walk(func)
        if isinstance(node, ast.If) and "drafter_kind" in ast.unparse(node.test)
    ]
    assert guards, "spec_drafter_kind appears but gates nothing"
    assert any(
        "'mtp'" in guard or '"mtp"' in guard for guard in guards
    ), "the drafter kind is read but never required to be mtp"
    appends = [
        ast.unparse(node)
        for node in ast.walk(func)
        if isinstance(node, ast.Call)
        and "failures" in ast.unparse(node)
        and "append" in ast.unparse(node)
        and "drafter" in ast.unparse(node)
    ]
    assert appends, "a fallback is noticed but not recorded as a failure"


def test_the_check_is_scoped_to_an_mtp_model():
    """A leg pointed at a plain GGUF must not go red for not doing MTP: that is
    the correct behaviour there, and a rule that cannot pass is deleted."""
    func = _func("assert_gpu_inference")
    guards = [
        ast.unparse(node.test)
        for node in ast.walk(func)
        if isinstance(node, ast.If) and "drafter_kind" in ast.unparse(node.test)
    ]
    assert any("MTP" in guard and "chat_model" in guard for guard in guards), (
        "the MTP requirement is unconditional, so a plain-GGUF run fails on "
        "behaviour that is correct for it"
    )


def test_the_reason_is_reported_and_not_just_the_verdict():
    """'MTP is off' sends the reader nowhere. The reason distinguishes a build
    without MTP from a VRAM downgrade, and only one of those is actionable."""
    func = _func("assert_gpu_inference")
    body = ast.unparse(func)
    assert "fallback_reason" in body
    assert "supports_mtp" in body
