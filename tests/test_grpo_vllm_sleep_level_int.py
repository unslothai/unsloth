# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""TRL < 0.23 has no vLLM sleep of its own, so Unsloth wraps each self.llm.generate with wake_up / sleep. The
level came from os.environ unconverted: a set VLLM_SLEEP_MODE reached LLM.sleep as a string, which vLLM's
level == 1 test never matches (weights discarded instead of offloaded) and vLLM 0.29 rejects (level >= 1).
The level is now an int read before wake_up: unset / blank is 1, and anything but 0, 1, 2 fails before the
engine is touched (not after a finished rollout)."""

import os
import textwrap
import types

import pytest

_LEGACY_SOURCE = textwrap.dedent(
    """
    def _generate_and_score_completions(self, inputs):
        all_outputs = self.llm.generate(all_prompts_text, sampling_params=None, use_tqdm=False)
        return all_outputs
    """
)


def _generated():
    from unsloth.models.rl_replacements import grpo_trainer__generate_and_score_completions

    source = grpo_trainer__generate_and_score_completions(
        "_generate_and_score_completions", _LEGACY_SOURCE
    )
    assert "self.llm.sleep(" in source and "self.llm.wake_up()" in source, source
    # Only the wake_up / generate / sleep wrap runs: the other injected helpers are stubbed.
    scope = {
        "os": os,
        "all_prompts_text": ["p"],
        "_unsloth_reject_grpo_image_list": lambda inputs: None,
    }
    exec(source, scope)
    return scope["_generate_and_score_completions"]


class _LLM:
    def __init__(self, sleep_mode):
        self.calls = []
        config = types.SimpleNamespace(enable_sleep_mode = sleep_mode)
        self.llm_engine = types.SimpleNamespace(
            vllm_config = types.SimpleNamespace(model_config = config)
        )

    def wake_up(self):
        self.calls.append("wake_up")

    def generate(self, *args, **kwargs):
        self.calls.append("generate")
        return ["out"]

    def sleep(self, level):
        self.calls.append(("sleep", type(level).__name__, level))


def _run(
    monkeypatch,
    value,
    sleep_mode = True,
):
    if value is None:
        monkeypatch.delenv("VLLM_SLEEP_MODE", raising = False)
    else:
        monkeypatch.setenv("VLLM_SLEEP_MODE", value)
    llm = _LLM(sleep_mode)
    trainer = types.SimpleNamespace(llm = llm)
    try:
        _generated()(trainer, None)
    except ValueError as exc:
        return llm.calls, str(exc)
    return llm.calls, None


@pytest.mark.parametrize(
    "value, expected",
    [(None, 1), ("", 1), (" \r\n", 1), ("0", 0), ("1", 1), ("2", 2), (" 2 ", 2)],
)
def test_sleep_level_is_the_int_vllm_expects(monkeypatch, value, expected):
    calls, error = _run(monkeypatch, value)
    assert error is None, error
    assert calls == ["wake_up", "generate", ("sleep", "int", expected)], calls


@pytest.mark.parametrize("value", ["false", "-1", "3", "1.5"])
def test_unsupported_sleep_level_fails_before_the_engine_is_touched(monkeypatch, value):
    calls, error = _run(monkeypatch, value)
    assert error is not None and "VLLM_SLEEP_MODE must be 0, 1 or 2" in error, error
    assert calls == [], calls


@pytest.mark.parametrize("value", [None, "", "2", "false"])
def test_sleep_mode_disabled_only_generates(monkeypatch, value):
    calls, error = _run(monkeypatch, value, sleep_mode = False)
    assert error is None and calls == ["generate"], (calls, error)
