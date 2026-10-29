# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Expected strings: transformers apply_chat_template renders from unsloth/Starling-LM-7B-beta,
01-ai/Yi-6B-Chat and LiquidAI/LFM2(.5)-1.2B, hard coded for offline runs."""

import os
import re

import pytest
from jinja2 import Environment


CHAT_TEMPLATES_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "unsloth",
    "chat_templates.py",
)


def _extract_template(name):
    src = open(CHAT_TEMPLATES_PATH, encoding = "utf-8").read()
    pattern = rf"{re.escape(name)}\s*=\s*\\\n(\"\"\"|''')(.*?)\1"
    m = re.search(pattern, src, flags = re.DOTALL)
    assert m, f"Could not extract {name} from chat_templates.py"
    return m.group(2)


# transformers renders with trim_blocks + lstrip_blocks; plain Environment = other Jinja consumers.
ENVIRONMENTS = {
    "transformers": dict(trim_blocks = True, lstrip_blocks = True),
    "plain": dict(),
}

CONVERSATION = [
    {"role": "system", "content": "SYS"},
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hi there"},
    {"role": "user", "content": "Q2"},
]

EXPECTED = {
    "starling_template": (
        "<s>",
        "<s>GPT4 Correct System: SYS<|end_of_turn|>"
        "GPT4 Correct User: Hello<|end_of_turn|>"
        "GPT4 Correct Assistant: Hi there<|end_of_turn|>"
        "GPT4 Correct User: Q2<|end_of_turn|>",
        "GPT4 Correct Assistant:",
    ),
    "yi_chat_template": (
        "<|startoftext|>",
        "<|im_start|>system\nSYS<|im_end|>\n"
        "<|im_start|>user\nHello<|im_end|>\n"
        "<|im_start|>assistant\nHi there<|im_end|>\n"
        "<|im_start|>user\nQ2<|im_end|>\n",
        "<|im_start|>assistant\n",
    ),
    "liquid_lfm2_template": (
        "<|startoftext|>",
        "<|startoftext|><|im_start|>system\nSYS<|im_end|>\n"
        "<|im_start|>user\nHello<|im_end|>\n"
        "<|im_start|>assistant\nHi there<|im_end|>\n"
        "<|im_start|>user\nQ2<|im_end|>\n",
        "<|im_start|>assistant\n",
    ),
}


@pytest.mark.parametrize("env_name", sorted(ENVIRONMENTS))
@pytest.mark.parametrize("add_generation_prompt", [False, True])
@pytest.mark.parametrize("template_name", sorted(EXPECTED))
def test_template_matches_official_render(template_name, add_generation_prompt, env_name):
    bos_token, expected, generation_prompt = EXPECTED[template_name]
    if add_generation_prompt:
        expected += generation_prompt
    tmpl = Environment(**ENVIRONMENTS[env_name]).from_string(_extract_template(template_name))
    out = tmpl.render(
        messages = CONVERSATION,
        bos_token = bos_token,
        add_generation_prompt = add_generation_prompt,
    )
    assert out == expected
