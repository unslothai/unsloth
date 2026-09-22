# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

from __future__ import annotations

import ast
import inspect
import os
from dataclasses import dataclass, field

import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
SOURCE_PATH = os.path.join(REPO_ROOT, "unsloth", "models", "rl_replacements.py")
HELPER = "grpo_update_SamplingParams"


def _read_source() -> str:
    with open(SOURCE_PATH, "r", encoding = "utf-8") as fh:
        return fh.read()


def _load_helper():
    try:
        import unsloth.models.rl_replacements as rl
    except Exception:
        rl = None
    if rl is not None:
        return getattr(rl, HELPER)
    tree = ast.parse(_read_source())
    node = next(
        (n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == HELPER),
        None,
    )
    assert node is not None, f"{HELPER} is not defined in rl_replacements.py"
    namespace = {"inspect": inspect}
    exec(compile(ast.Module(body = [node], type_ignores = []), SOURCE_PATH, "exec"), namespace)
    return namespace[HELPER]


@dataclass
class SamplingParams:
    n: int = 1
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    min_p: float = 0.0
    seed: int | None = None
    max_tokens: int = 16
    stop: list[str] | None = None
    include_stop_str_in_output: bool = False
    logprobs: int | None = None
    _real_n: int | None = field(default = None, repr = False)


EOS = "<|im_end|>"


def _trl_generation_kwargs():
    return {
        "n": 8,
        "repetition_penalty": 1.0,
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": -1,
        "min_p": 0.0,
        "max_tokens": 1024,
        "truncate_prompt_tokens": 512,
        "guided_decoding": None,
        "logprobs": 0,
    }


@pytest.fixture(scope = "module")
def helper():
    return _load_helper()


def test_notebook_scalar_fields_reach_generation(helper):
    generation_kwargs = _trl_generation_kwargs()
    user = SamplingParams(
        min_p = 0.1,
        top_p = 1.0,
        top_k = -1,
        seed = 3407,
        stop = [EOS],
        include_stop_str_in_output = True,
    )
    result = helper(SamplingParams, generation_kwargs, user)
    assert result["min_p"] == 0.1
    assert result["include_stop_str_in_output"] is True
    assert result["stop"] == [EOS]
    assert "seed" not in result
    assert result["n"] == generation_kwargs["n"]
    assert result["max_tokens"] == generation_kwargs["max_tokens"]
    assert "repetition_penalty" not in result
    assert "_real_n" not in result
    SamplingParams(**result)


def test_default_sampling_params_overlay_nothing(helper):
    generation_kwargs = _trl_generation_kwargs()
    result = helper(SamplingParams, generation_kwargs, SamplingParams())
    expected = {
        k: v for k, v in generation_kwargs.items() if k in SamplingParams.__dataclass_fields__
    }
    assert result == expected


def test_set_kwargs_take_precedence_over_field_diff(helper):
    generation_kwargs = _trl_generation_kwargs()
    user = SamplingParams(min_p = 0.1, seed = 3407, n = 4)
    user._set_kwargs = {"min_p": 0.2, "seed": 3407, "n": 4, "not_a_field": 1}
    result = helper(SamplingParams, generation_kwargs, user)
    assert result["min_p"] == 0.2
    assert "seed" not in result
    assert result["n"] == generation_kwargs["n"]
    assert "not_a_field" not in result


@pytest.mark.parametrize("use_set_kwargs", [False, True])
def test_trl_owned_fields_are_not_overridden(helper, use_set_kwargs):
    generation_kwargs = _trl_generation_kwargs()
    overrides = {"temperature": 0.6, "max_tokens": 4096, "logprobs": 5, "min_p": 0.1}
    user = SamplingParams(**overrides)
    if use_set_kwargs:
        user._set_kwargs = dict(overrides)
    result = helper(SamplingParams, generation_kwargs, user)
    assert result["temperature"] == generation_kwargs["temperature"]
    assert result["max_tokens"] == generation_kwargs["max_tokens"]
    assert result["logprobs"] == generation_kwargs["logprobs"]
    assert result["min_p"] == 0.1


def test_repo_injects_the_local_helper():
    src = _read_source()
    assert f"def {HELPER}(SamplingParams, generation_kwargs, vllm_sampling_params = None):" in src
    assert f'RL_REPLACEMENTS["{HELPER}"]' not in src
    assert f'RL_PRE_ITEMS["grpo_trainer"].append(inspect.getsource({HELPER}))' in src
