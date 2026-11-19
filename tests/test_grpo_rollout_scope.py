# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Multi-turn GRPO: every vLLM engine call of one tool-calling `_generate` runs inside one vllm_rollout_scope, so the
adapter is folded (or the vLLM LoRA request built) once per rollout step, not once per turn. CPU only: the helpers are
lifted from rl_replacements.py via ast, the engine is a stub."""

from __future__ import annotations

import ast
import contextlib
import os
import re
import textwrap
import types
from pathlib import Path

import pytest

SOURCE_PATH = Path(__file__).resolve().parents[1] / "unsloth" / "models" / "rl_replacements.py"
NAMES = (
    "_unsloth_rollout_scope",
    "_unsloth_wrap_generate_in_rollout_scope",
    "_unsloth_patch_grpo_truncation_check",
    "_guard_vllm_sync_reload_for_shared_weights",
    "grpo_trainer__generate",
)


def _load():
    tree = ast.parse(SOURCE_PATH.read_text(encoding = "utf-8"))
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in NAMES]
    namespace = {"re": re, "os": os, "textwrap": textwrap, "ast": ast}
    exec(compile(ast.Module(body = nodes, type_ignores = []), str(SOURCE_PATH), "exec"), namespace)
    return namespace


_GENERATE = """
    def _generate(self, prompts):
        for turn in range(3):
            self.vllm_generation.llm.generate(prompts)
            if turn == 0:
                assert self.vllm_generation.llm.generate is not None
        return "done"
"""


class _FakeLLM:
    shared_weights = True

    def __init__(self):
        self.inside = []

    def generate(
        self,
        prompts,
        lora_request = None,
    ):
        self.inside.append(getattr(self, "_unsloth_rollout_scope", None) is not None)
        return ["out"]


def _trainer_class(ns):
    source = ns["grpo_trainer__generate"]("_generate", _GENERATE)
    class_ns = dict(ns)
    exec("class _Trainer:\n" + source, class_ns)
    return class_ns["_Trainer"]


def _make(ns, **attrs):
    trainer = _trainer_class(ns)()
    llm = _FakeLLM()
    trainer.vllm_generation = types.SimpleNamespace(llm = llm)
    trainer.use_vllm = True
    trainer.tools = [lambda x: x]
    trainer.rollout_func = None
    for k, v in attrs.items():
        setattr(trainer, k, v)
    return trainer, llm


def test_every_engine_call_of_a_tool_rollout_is_inside_one_scope():
    pytest.importorskip("unsloth_zoo.vllm_utils").vllm_rollout_scope
    ns = _load()
    trainer, llm = _make(ns)
    assert trainer._generate(["p"]) == "done"
    assert llm.inside == [True, True, True]
    assert getattr(llm, "_unsloth_rollout_scope", None) is None  # closed afterwards


@pytest.mark.parametrize(
    "case", ["no_tools", "rollout_func", "not_vllm", "not_shared", "kill_switch"]
)
def test_scope_stays_off_where_it_cannot_help_or_is_unsafe(case, monkeypatch):
    pytest.importorskip("unsloth_zoo.vllm_utils").vllm_rollout_scope
    ns = _load()
    attrs = {
        "no_tools": {"tools": []},
        "rollout_func": {"rollout_func": lambda prompts, trainer: None},
        "not_vllm": {"use_vllm": False},
    }.get(case, {})
    trainer, llm = _make(ns, **attrs)
    if case == "not_shared":
        llm.shared_weights = False
    if case == "kill_switch":
        monkeypatch.setenv("UNSLOTH_VLLM_ROLLOUT_SCOPE", "0")
    trainer._generate(["p"])
    assert llm.inside == [False, False, False]


def test_old_zoo_without_the_scope_is_a_no_op(monkeypatch):
    import sys

    fake = types.ModuleType("unsloth_zoo.vllm_utils")
    monkeypatch.setitem(sys.modules, "unsloth_zoo.vllm_utils", fake)
    ns = _load()
    trainer, llm = _make(ns)
    assert trainer._generate(["p"]) == "done"
    assert llm.inside == [False, False, False]


def _trl_generate_sources():
    """`_generate` of the installed TRL (and of any TRL tree named in UNSLOTH_TEST_TRL_DIRS)."""
    paths = []
    try:
        import trl
        paths.append(Path(trl.__file__).parent / "trainer" / "grpo_trainer.py")
    except Exception:
        pass
    for d in filter(None, os.environ.get("UNSLOTH_TEST_TRL_DIRS", "").split(os.pathsep)):
        paths.append(Path(d) / "trainer" / "grpo_trainer.py")
    out = []
    for path in paths:
        if not path.exists():
            continue
        text = path.read_text(encoding = "utf-8")
        for node in ast.walk(ast.parse(text)):
            if isinstance(node, ast.ClassDef) and node.name == "GRPOTrainer":
                for item in node.body:
                    if isinstance(item, ast.FunctionDef) and item.name == "_generate":
                        out.append(
                            (str(path), textwrap.indent(ast.get_source_segment(text, item), "    "))
                        )
    return out


def test_real_trl_generate_is_wrapped_and_still_compiles():
    sources = _trl_generate_sources()
    if not sources:
        pytest.skip("no TRL GRPOTrainer._generate available")
    ns = _load()
    for path, source in sources:
        patched = ns["grpo_trainer__generate"]("_generate", source)
        tree = ast.parse("class _T:\n" + patched)
        methods = {f.name: f for f in tree.body[0].body if isinstance(f, ast.FunctionDef)}
        assert set(methods) == {"_generate", "_unsloth_generate_in_scope"}, path
        wrapper = ast.unparse(methods["_generate"])
        assert (
            "_unsloth_rollout_scope(self)" in wrapper
            and "_unsloth_generate_in_scope(*args, **kwargs)" in wrapper
        )
        # Idempotent: a second pass does not nest another wrapper.
        assert ns["_unsloth_wrap_generate_in_rollout_scope"](patched) == patched
