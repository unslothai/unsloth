# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
"""Exercise the loader's actual resolution blocks without loading GPU weights."""

import ast
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).parents[2]
LOADER = ROOT / "unsloth/models/loader.py"


def _loader(cls):
    tree = ast.parse(LOADER.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls)
    return next(
        n for n in node.body if isinstance(n, ast.FunctionDef) and n.name == "from_pretrained"
    )


@pytest.mark.parametrize("cls", ["FastLanguageModel", "FastModel"])
@pytest.mark.parametrize("exact,prequant", [(False, True), (True, True), (False, False)])
def test_reports_actual_repo_after_mapping_and_capability_normalization(cls, exact, prequant):
    function = _loader(cls)
    start = next(
        i for i, n in enumerate(function.body) if ast.unparse(n) == "old_model_name = model_name"
    )
    # Stop before provider downloads/config probes. This executes the production
    # mapping and reporting statements, not a second implementation of the mapper.
    end = next(
        i
        for i in range(start, len(function.body))
        if "USE_MODELSCOPE" in ast.unparse(function.body[i])
    )
    reports = []
    mapping_calls = []

    def mapper(name, **kwargs):
        mapping_calls.append(name)
        return "org/resolved-unsloth-bnb-4bit"

    scope = dict(
        model_name = "org/requested",
        use_exact_model_name = exact,
        load_in_4bit = True,
        load_in_8bit = False,
        load_in_16bit = False,
        load_in_fp8 = False,
        token = None,
        trust_remote_code = False,
        get_model_name = mapper,
        ALLOW_PREQUANTIZED_MODELS = prequant,
        _strip_unsloth_bnb_4bit_suffix = lambda name: name.removesuffix("-unsloth-bnb-4bit"),
        on_model_resolved = reports.append,
        os = os,
        kwargs = {},
        quantization_config = None,
    )
    module = ast.Module(body = function.body[start:end], type_ignores = [])
    exec(compile(ast.fix_missing_locations(module), str(LOADER), "exec"), scope)
    expected = (
        "org/requested"
        if exact
        else "org/resolved-unsloth-bnb-4bit"
        if prequant
        else "org/resolved"
    )
    assert reports == [expected]
    assert scope["model_name"] == expected
    assert mapping_calls == ([] if exact else ["org/requested"])


@pytest.mark.parametrize("cls", ["FastLanguageModel", "FastModel"])
def test_adapter_base_resolution_updates_report_before_base_config_load(cls):
    function = _loader(cls)
    block = next(
        n for n in function.body if isinstance(n, ast.If) and ast.unparse(n.test) == "is_peft"
    )
    # The first three statements select, map and normalize the adapter's base.
    # Include the notification that must follow, stopping before precision setup.
    end = next(
        i
        for i, n in enumerate(block.body)
        if isinstance(n, ast.If) and ast.unparse(n.test).startswith("model_name.lower().endswith")
    )
    reports = []
    scope = dict(
        peft_config = SimpleNamespace(base_model_name_or_path = "org/base"),
        use_exact_model_name = False,
        load_in_4bit = True,
        load_in_fp8 = False,
        token = None,
        trust_remote_code = False,
        ALLOW_PREQUANTIZED_MODELS = True,
        get_model_name = lambda *a, **k: "org/base-unsloth-bnb-4bit",
        on_model_resolved = reports.append,
    )
    module = ast.Module(body = block.body[:end], type_ignores = [])
    exec(compile(ast.fix_missing_locations(module), str(LOADER), "exec"), scope)
    assert reports == ["org/base-unsloth-bnb-4bit"]


def test_callback_is_consumed_by_loaders_and_forwarded_only_to_delegated_loader():
    for cls in ("FastLanguageModel", "FastModel"):
        function = _loader(cls)
        assert "on_model_resolved" in [a.arg for a in function.args.args + function.args.kwonlyargs]
        for call in (n for n in ast.walk(function) if isinstance(n, ast.Call)):
            if ast.unparse(call.func) == "FastModel.from_pretrained":
                assert any(k.arg == "on_model_resolved" for k in call.keywords)
            elif ast.unparse(call.func) in (
                "dispatch_model.from_pretrained",
                "FastBaseModel.from_pretrained",
            ):
                assert not any(k.arg == "on_model_resolved" for k in call.keywords)


def test_training_forwards_observer_through_each_loader_and_retry():
    path = ROOT / "studio/backend/core/training/trainer.py"
    tree = ast.parse(path.read_text())
    load = next(
        n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "load_model"
    )
    calls = [
        n
        for n in ast.walk(load)
        if isinstance(n, ast.Call)
        and ast.unparse(n.func)
        in (
            "FastLanguageModel.from_pretrained",
            "FastModel.from_pretrained",
            "FastVisionModel.from_pretrained",
            "self.load_model",
        )
    ]
    assert len(calls) == 9
    for call in calls:
        assert any(
            k.arg == "on_model_resolved" and ast.unparse(k.value) == "on_model_resolved"
            for k in call.keywords
        )
