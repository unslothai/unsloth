# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
"""Inkling 2 under the parallel axes: tensor / pipeline parallel are refused from the config (before any weight
loads) and by the capability table, with the reason and the data parallel alternatives; DDP / FSDP2 stay allowed."""

import ast
import importlib.util
import pathlib
import sys
import types

import pytest

_ROOT = pathlib.Path(__file__).resolve().parents[1]


def _by_path(name, rel):
    spec = importlib.util.spec_from_file_location(name, _ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _inkling2():
    return _by_path("_unsloth_inkling2_par", "unsloth/models/inkling2.py")


def _capabilities():
    # By path, as a submodule of a bare `unsloth` namespace (importing unsloth itself needs a GPU).
    if "unsloth" not in sys.modules:
        pkg = types.ModuleType("unsloth")
        pkg.__path__ = [str(_ROOT / "unsloth")]
        sys.modules["unsloth"] = pkg
    return _by_path("unsloth.parallel_capabilities", "unsloth/parallel_capabilities.py")


def _config(model_type):
    text = types.SimpleNamespace(model_type = model_type + "_text")
    return types.SimpleNamespace(model_type = model_type, text_config = text)


@pytest.mark.parametrize("model_type", ["inkling2", "inkling2_loop"])
@pytest.mark.parametrize(
    "env, words",
    [
        (
            {"UNSLOTH_TP_SIZE": "2"},
            ["tensor parallel (UNSLOTH_TP_SIZE=2)", "wrapper modules", "UNSLOTH_TP_SIZE"],
        ),
        (
            {"UNSLOTH_PP_SIZE": "2"},
            [
                "pipeline parallel (UNSLOTH_PP_SIZE=2)",
                "block attention residuals",
                "UNSLOTH_PP_SIZE",
            ],
        ),
        (
            {"UNSLOTH_TP_SIZE": "2", "UNSLOTH_PP_SIZE": "2"},
            ["tensor parallel", "pipeline parallel", "UNSLOTH_TP_SIZE / UNSLOTH_PP_SIZE"],
        ),
    ],
)
def test_tp_pp_refused_from_the_config(monkeypatch, model_type, env, words):
    ink = _inkling2()
    for key in ("UNSLOTH_TP_SIZE", "UNSLOTH_PP_SIZE"):
        monkeypatch.delenv(key, raising = False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    with pytest.raises(NotImplementedError) as info:
        ink.raise_if_inkling2_parallel_unsupported(_config(model_type))
    message = str(info.value)
    for word in words + [
        'UNSLOTH_PARALLEL="dp=N"',
        'UNSLOTH_PARALLEL="fsdp=1"',
        'device_map = "balanced"',
    ]:
        assert word in message, (word, message)
    # Known only by model type (an adapter load resolves the base's types, not its config).
    with pytest.raises(NotImplementedError):
        ink.raise_if_inkling2_parallel_unsupported(None, [model_type])


def test_data_parallel_and_other_models_pass(monkeypatch):
    ink = _inkling2()
    for key in ("UNSLOTH_TP_SIZE", "UNSLOTH_PP_SIZE"):
        monkeypatch.delenv(key, raising = False)
    ink.raise_if_inkling2_parallel_unsupported(
        _config("inkling2_loop")
    )  # DDP / FSDP2 / one process
    monkeypatch.setenv("UNSLOTH_TP_SIZE", "1")
    monkeypatch.setenv("UNSLOTH_PP_SIZE", "1")
    ink.raise_if_inkling2_parallel_unsupported(_config("inkling2"))
    monkeypatch.setenv("UNSLOTH_TP_SIZE", "2")
    ink.raise_if_inkling2_parallel_unsupported(_config("qwen3_moe"), ["qwen3_moe"])


def test_loader_refuses_in_the_inkling2_branch_before_loading():
    """FastModel.from_pretrained's model-type branch for Inkling 2 calls the refusal (config time, before the
    transformers load the tensor parallel sharded loader wraps)."""
    tree = ast.parse((_ROOT / "unsloth" / "models" / "loader.py").read_text())
    found = False
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        test = ast.unparse(node.test)
        if test.replace("'", '"') != '"inkling2" in model_types_all':
            continue
        calls = [
            ast.unparse(n.func)
            for stmt in node.body
            for n in ast.walk(stmt)
            if isinstance(n, ast.Call)
        ]
        found = "raise_if_inkling2_parallel_unsupported" in calls
    assert found


@pytest.mark.parametrize("family", ["inkling2", "inkling2_loop"])
def test_capability_table_refuses_tp_pp(family):
    cap = _capabilities()
    for layout in (dict(tp = 2), dict(pp = 2), dict(pp = 2, pp_schedule = "gpipe")):
        with pytest.raises(Exception) as info:
            cap.check(cap.Query(family = family, moe = True, expert_lora = True, **layout))
        assert "Inkling 2" in str(info.value) and "fsdp=1" in str(info.value)
    # data parallel (DDP / FSDP2) is not refused by the rule
    for layout in (dict(dp = 2), dict(dp = 2, fsdp2 = True)):
        rules = [r for r in cap.RULES if r.applies(cap.Query(family = family, **layout))]
        assert not rules, [r.when for r in rules]
    assert "Inkling 2" in cap.markdown_table("rules")
