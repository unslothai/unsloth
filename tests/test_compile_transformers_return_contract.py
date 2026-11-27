# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team.

# loader.py unpacks `model_types, supports_sdpa = ...`: every early return must be a pair.

import pytest


@pytest.fixture
def utils(monkeypatch):
    monkeypatch.delenv("UNSLOTH_FORCE_CUSTOM_DTYPE", raising = False)
    import unsloth.models._utils as U
    return U


def test_torch_below_24_returns_model_types_and_false(utils, monkeypatch):
    monkeypatch.setattr(utils, "torch_version", "2.3.1")
    model_types, compiled = utils.unsloth_compile_transformers(
        dtype = None,
        model_name = "unsloth/tiny",
        model_types = ["llama", "qwen3"],
    )
    assert model_types == ["llama", "qwen3"]
    assert compiled is False


def test_trust_remote_code_returns_model_types_and_false(utils, monkeypatch):
    monkeypatch.setattr(utils, "torch_version", "2.10.0")
    model_types, compiled = utils.unsloth_compile_transformers(
        dtype = None,
        model_name = "unsloth/tiny",
        model_types = ["llama"],
        trust_remote_code = True,
    )
    assert model_types == ["llama"]
    assert compiled is False


def test_disable_returns_deduped_model_types_and_false(utils, monkeypatch):
    monkeypatch.setattr(utils, "torch_version", "2.10.0")
    model_types, compiled = utils.unsloth_compile_transformers(
        dtype = None,
        model_name = "unsloth/tiny",
        model_types = ["llama", "llama", "qwen3"],
        disable = True,
    )
    assert model_types == ["llama", "qwen3"]
    assert compiled is False
