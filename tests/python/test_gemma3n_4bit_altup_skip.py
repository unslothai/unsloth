# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team.

"""Gemma 3n on-the-fly 4-bit keeps the AltUp coefficient Linears in 16-bit: AltUp clamps
prediction_coefs.weight.data in place every training step, which fails on packed uint8 weights."""

import ast
from pathlib import Path

VISION_PATH = Path(__file__).resolve().parents[2] / "unsloth" / "models" / "vision.py"
ALTUP = {"correction_coefs", "prediction_coefs", "altup_projections", "altup_unembed_projections"}


def _architecture_skip_modules():
    tree = ast.parse(VISION_PATH.read_text(encoding = "utf-8"))
    node = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "_architecture_skip_modules"
    )
    ns = {}
    exec(compile(ast.Module([node], []), str(VISION_PATH), "exec"), ns)
    return ns["_architecture_skip_modules"]


def test_gemma3n_keeps_altup_coefficients_unquantized():
    skip = _architecture_skip_modules()
    for model_types in (
        ["gemma3n"],
        ["gemma3n", "gemma3n_text", "gemma3n_audio"],
        ["gemma3n_text"],
    ):
        assert ALTUP <= set(skip(model_types)), model_types


def test_other_models_unchanged():
    skip = _architecture_skip_modules()
    for model_types in (["gemma3"], ["gemma4"], ["llama"], [], None):
        assert not ALTUP & set(skip(model_types)), model_types


def test_altup_linears_match_the_transformers_skip_matcher():
    """transformers 5 skips a Linear only when a key matches its full name (suffix or start-anchored regex)."""
    import pytest

    utils = pytest.importorskip("transformers.quantizers.quantizers_utils")
    should_convert = getattr(utils, "should_convert_module", None)
    if should_convert is None:
        pytest.skip("transformers < 5 matches skip keys per path component")
    skip = _architecture_skip_modules()(["gemma3n"])
    for name in (
        "model.language_model.altup_projections.0",
        "model.language_model.altup_projections.2",
        "model.language_model.altup_unembed_projections.1",
        "model.language_model.layers.0.altup.prediction_coefs",
        "model.language_model.layers.3.altup.correction_coefs",
    ):
        assert not should_convert(name, skip), name
    for name in (
        "model.language_model.layers.0.self_attn.q_proj",
        "model.language_model.layers.0.mlp.down_proj",
    ):
        assert should_convert(name, skip), name
