# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
"""Inkling 2 under the parallel axes: tensor parallel with a tp size that does not divide the attention head
counts, and pipeline parallel composed with tensor / data parallel, are refused from the config (before any
weight loads) and by the capability table, with the reason and the alternatives; DDP / FSDP2 / tensor parallel /
pipeline parallel alone stay allowed."""

import ast
import importlib.util
import pathlib
import sys
import types

import pytest

_ROOT = pathlib.Path(__file__).resolve().parents[1]


def _by_path(name, rel, monkeypatch):
    spec = importlib.util.spec_from_file_location(name, _ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, mod)  # restored after the test: no package state leaks
    spec.loader.exec_module(mod)
    return mod


def _inkling2(monkeypatch):
    return _by_path("_unsloth_inkling2_par", "unsloth/models/inkling2.py", monkeypatch)


def _capabilities(monkeypatch):
    # By path, as a submodule of a bare `unsloth` namespace (importing unsloth itself needs a GPU).
    if "unsloth" not in sys.modules:
        pkg = types.ModuleType("unsloth")
        pkg.__path__ = [str(_ROOT / "unsloth")]
        monkeypatch.setitem(sys.modules, "unsloth", pkg)
    return _by_path(
        "unsloth.parallel_capabilities", "unsloth/parallel_capabilities.py", monkeypatch
    )


def _config(model_type):
    text = types.SimpleNamespace(model_type = model_type + "_text")
    return types.SimpleNamespace(model_type = model_type, text_config = text)


@pytest.mark.parametrize("model_type", ["inkling2", "inkling2_loop"])
@pytest.mark.parametrize(
    "env, words",
    [
        (
            {"UNSLOTH_TP_SIZE": "2", "UNSLOTH_PP_SIZE": "2"},
            ["pipeline parallel x tensor parallel", "UNSLOTH_PP_SIZE / UNSLOTH_TP_SIZE"],
        ),
    ],
)
def test_pp_x_tp_refused_from_the_config(monkeypatch, model_type, env, words):
    ink = _inkling2(monkeypatch)
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


def _headed_config(
    model_type,
    heads = 24,
    kv_heads = 8,
):
    text = types.SimpleNamespace(
        model_type = model_type + "_text",
        num_attention_heads = heads,
        num_key_value_heads = kv_heads,
        swa_num_attention_heads = heads,
        swa_num_key_value_heads = kv_heads,
    )
    return types.SimpleNamespace(model_type = model_type, text_config = text)


@pytest.mark.parametrize("model_type", ["inkling2", "inkling2_loop"])
def test_tp_allowed_when_it_divides_the_heads(monkeypatch, model_type):
    ink = _inkling2(monkeypatch)
    monkeypatch.delenv("UNSLOTH_PP_SIZE", raising = False)
    for tp in ("2", "4", "8"):  # Mini / Small: 24 / 48 query heads, 8 key / value heads
        monkeypatch.setenv("UNSLOTH_TP_SIZE", tp)
        ink.raise_if_inkling2_parallel_unsupported(_headed_config(model_type))
        ink.raise_if_inkling2_parallel_unsupported(
            _config(model_type)
        )  # sizes unknown: tp_native checks them
        ink.raise_if_inkling2_parallel_unsupported(None, [model_type])
    monkeypatch.setenv("UNSLOTH_TP_SIZE", "3")
    with pytest.raises(NotImplementedError) as info:
        ink.raise_if_inkling2_parallel_unsupported(_headed_config(model_type))
    message = str(info.value)
    for word in (
        "UNSLOTH_TP_SIZE=3",
        "num_key_value_heads = 8",
        "Choose a tp size",
        'UNSLOTH_PARALLEL="fsdp=1"',
    ):
        assert word in message, (word, message)


def test_data_parallel_and_other_models_pass(monkeypatch):
    ink = _inkling2(monkeypatch)
    for key in ("UNSLOTH_TP_SIZE", "UNSLOTH_PP_SIZE"):
        monkeypatch.delenv(key, raising = False)
    ink.raise_if_inkling2_parallel_unsupported(
        _config("inkling2_loop")
    )  # DDP / FSDP2 / one process
    monkeypatch.setenv("UNSLOTH_TP_SIZE", "1")
    monkeypatch.setenv("UNSLOTH_PP_SIZE", "1")
    ink.raise_if_inkling2_parallel_unsupported(_config("inkling2"))
    monkeypatch.setenv(
        "UNSLOTH_PP_SIZE", "2"
    )  # pipeline parallel: models/inkling2.py Inkling2PipelineAdapter
    ink.raise_if_inkling2_parallel_unsupported(_config("inkling2_loop"))
    monkeypatch.setenv("UNSLOTH_PP_SIZE", "1")
    monkeypatch.setenv("UNSLOTH_TP_SIZE", "2")
    ink.raise_if_inkling2_parallel_unsupported(_config("qwen3_moe"), ["qwen3_moe"])


def test_loader_refuses_in_the_inkling2_branch_before_loading():
    """FastModel.from_pretrained's model-type branch for Inkling 2 calls the refusal (config time, before the
    transformers load the tensor parallel sharded loader wraps)."""
    tree = ast.parse((_ROOT / "unsloth" / "models" / "loader.py").read_text(encoding = "utf-8"))
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
def test_capability_table_inkling2_rules(monkeypatch, family):
    cap = _capabilities(monkeypatch)
    for layout in (dict(tp = 2, pp = 2), dict(pp = 2, dp = 2), dict(pp = 2, dp = 2, fsdp2 = True)):
        with pytest.raises(Exception) as info:
            cap.check(cap.Query(family = family, moe = True, expert_lora = True, **layout))
        assert "Inkling 2" in str(info.value) and "fsdp=1" in str(info.value)
    # pipeline parallel alone has its cell (tests/test_pp_inkling2_gloo.py)
    for schedule in ("gpipe", "1f1b"):
        for mode in ("lora", "qlora"):
            query = cap.Query(
                family = family,
                moe = True,
                expert_lora = True,
                pp = 2,
                pp_schedule = schedule,
                mode = mode,
                trainer = "sft",
                attention = "eager",
                memory = "unsloth_offload",
            )
            cap.check(query)
            assert cap.cell_status(query)[0] == "supported", cap.cell_status(query)
    # data parallel (DDP / FSDP2), tensor parallel and pipeline parallel alone are not refused by the rule
    for layout in (dict(dp = 2), dict(dp = 2, fsdp2 = True), dict(tp = 2), dict(pp = 2)):
        rules = [r for r in cap.RULES if r.applies(cap.Query(family = family, **layout))]
        assert not rules, [r.when for r in rules]
    assert "Inkling 2" in cap.markdown_table("rules")
