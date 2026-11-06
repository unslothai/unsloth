# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

import ast
import importlib.util
import sys
import types
from pathlib import Path

MODELS = Path(__file__).resolve().parents[1] / "unsloth" / "models"


def _helper(monkeypatch):
    package = types.ModuleType("_fi_models")
    package.__path__ = [str(MODELS)]
    spec = importlib.util.spec_from_file_location("_fi_models.mapper", MODELS / "mapper.py")
    mapper = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "_fi_models", package)
    monkeypatch.setitem(sys.modules, "_fi_models.mapper", mapper)
    spec.loader.exec_module(mapper)
    tree = ast.parse((MODELS / "_utils.py").read_text(encoding = "utf-8"))
    node = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "_vllm_plain_bnb_or_16bit"
    )
    namespace = {"__package__": "_fi_models", "__name__": "_fi_models._utils"}
    exec(compile(ast.Module(body = [node], type_ignores = []), "_utils.py", "exec"), namespace)
    return namespace["_vllm_plain_bnb_or_16bit"]


def test_published_plain_bnb_repo_is_kept(monkeypatch):
    f = _helper(monkeypatch)
    assert f("unsloth/gemma-3-12b-it-unsloth-bnb-4bit") == "unsloth/gemma-3-12b-it-bnb-4bit"
    assert f("unsloth/Qwen3-0.6B-unsloth-bnb-4bit") == "unsloth/Qwen3-0.6B-bnb-4bit"


def test_dynamic_only_models_fall_back_to_16bit(monkeypatch):
    # No -bnb-4bit repo is published for these: the old rename hit a 404.
    f = _helper(monkeypatch)
    assert f("unsloth/Muse-Glimmer-30B-unsloth-bnb-4bit") == "unsloth/Muse-Glimmer-30B"
    assert f("unsloth/gemma-4-E2B-it-unsloth-bnb-4bit") == "unsloth/gemma-4-E2B-it"
    assert f("unsloth/Qwen3.8-27B-unsloth-bnb-4bit") == "unsloth/Qwen3.8-27B"


def test_unknown_repo_keeps_the_old_rename(monkeypatch):
    f = _helper(monkeypatch)
    assert f("someone/foo-unsloth-bnb-4bit") == "someone/foo-bnb-4bit"
