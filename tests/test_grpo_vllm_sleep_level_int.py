# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""TRL < 0.23 has no vLLM sleep of its own, so Unsloth wraps each self.llm.generate with wake_up / sleep. The
level came from os.environ unconverted: a set VLLM_SLEEP_MODE reached LLM.sleep as a string, which vLLM's
level == 1 test never matches (weights discarded instead of offloaded) and vLLM 0.29 rejects (level >= 1)."""

import os
import re
import textwrap

import pytest

_LEGACY_SOURCE = textwrap.dedent(
    """
    def _generate_and_score_completions(self, inputs):
        device = self.accelerator.device
        all_outputs = self.llm.generate(all_prompts_text, sampling_params=sampling_params, use_tqdm=False)
        return all_outputs
    """
)


def _sleep_levels():
    from unsloth.models.rl_replacements import grpo_trainer__generate_and_score_completions

    source = grpo_trainer__generate_and_score_completions(
        "_generate_and_score_completions", _LEGACY_SOURCE
    )
    levels = re.findall(r"self\.llm\.sleep\((.*)\)\s*$", source, flags = re.MULTILINE)
    assert levels, source
    return levels


def test_generated_sleep_level_is_converted_to_int():
    for level in _sleep_levels():
        assert re.fullmatch(r"int\(os\.environ\.get\('VLLM_SLEEP_MODE', 1\)\)", level), level


@pytest.mark.parametrize("value, expected", [(None, 1), ("1", 1), ("2", 2), ("0", 0)])
def test_generated_sleep_level_evaluates_to_the_int_vllm_expects(monkeypatch, value, expected):
    if value is None:
        monkeypatch.delenv("VLLM_SLEEP_MODE", raising = False)
    else:
        monkeypatch.setenv("VLLM_SLEEP_MODE", value)
    for level in _sleep_levels():
        result = eval(level, {"os": os})
        assert type(result) is int and result == expected, (level, result)
