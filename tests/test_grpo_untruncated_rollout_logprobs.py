# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Colocated vLLM + IS correction requests untruncated rollout logprobs; otherwise off."""

from __future__ import annotations

import ast
import importlib.util
import re
import sys
import textwrap
import types
from pathlib import Path
from types import SimpleNamespace

import pytest


def _patched_trl():
    spec = (
        importlib.util.find_spec("trl.trainer.grpo_trainer")
        if importlib.util.find_spec("trl")
        else None
    )
    if spec is None or spec.origin is None:
        pytest.skip("trl is not installed")
    source = Path(spec.origin).read_text(encoding = "utf-8")
    node = next(
        n
        for n in ast.walk(ast.parse(source))
        if isinstance(n, ast.FunctionDef) and n.name == "_generate_and_score_completions"
    )
    function = "    " + ast.get_source_segment(source, node)
    if "vllm_importance_sampling_correction" not in function:
        pytest.skip("this TRL has no vLLM importance sampling correction")
    from unsloth.models.rl_replacements import grpo_trainer__generate_and_score_completions

    patched = grpo_trainer__generate_and_score_completions(
        "_generate_and_score_completions", function
    )
    ast.parse(textwrap.dedent(patched))
    return patched


def _block(patched):
    match = re.search(
        r"\n( +)try:\n\1    from unsloth_zoo\.vllm_utils import set_vllm_untruncated_logprobs.*?"
        r"\n\1    self\._unsloth_vllm_untruncated_logprobs = False\n",
        patched,
        re.S,
    )
    assert match is not None
    assert patched.count("set_vllm_untruncated_logprobs") == 1
    header = re.search(r"def _generate_and_score_completions\([^)]*\)[^:\n]*:[ \t]*\n", patched)
    assert match.start() == header.end() - 1
    first_generate = min(
        i for i in (patched.find(x, header.end()) for x in ("generate(", "_generate(")) if i != -1
    )
    assert match.end() < first_generate
    return "def run(self):\n" + textwrap.indent(textwrap.dedent(match.group(0)), "    ")


@pytest.mark.parametrize(
    "use_vllm, correction, colocated, zoo, expected",
    [
        (True, True, True, True, True),
        (
            True,
            True,
            "vllm_generation",
            True,
            True,
        ),  # TRL >= 1.x keeps the colocated engine on vllm_generation
        (True, False, True, True, False),
        (False, True, True, True, False),
        (
            True,
            True,
            False,
            True,
            False,
        ),  # vllm_mode = "server": logprobs come from another process
        (True, True, True, False, False),  # unsloth_zoo without the patch
    ],
)
def test_generation_turns_on_untruncated_logprobs(
    monkeypatch, use_vllm, correction, colocated, zoo, expected
):
    block = _block(_patched_trl())
    module = types.ModuleType("unsloth_zoo.vllm_utils")
    calls = []
    if zoo:
        module.set_vllm_untruncated_logprobs = (
            lambda on, llm = None: calls.append((on, llm is not None)) or on
        )
    monkeypatch.setitem(sys.modules, "unsloth_zoo.vllm_utils", module)
    ns = {}
    exec(block, ns)
    trainer = SimpleNamespace(
        use_vllm = use_vllm,
        vllm_importance_sampling_correction = correction,
        llm = object() if colocated is True else None,
    )
    if colocated == "vllm_generation":
        trainer.vllm_generation = SimpleNamespace(llm = object())
    ns["run"](trainer)
    assert trainer._unsloth_vllm_untruncated_logprobs is expected
    if zoo:
        assert calls == [(bool(use_vllm and correction and colocated), bool(colocated))]
