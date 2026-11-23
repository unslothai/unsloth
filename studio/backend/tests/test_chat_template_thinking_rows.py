# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import ast
from pathlib import Path

import pytest
from datasets import Dataset

from utils.datasets import format_and_template_dataset

_QWEN3_TEMPLATE = """
{%- for message in messages %}
{%- if message.role == 'assistant' %}
{%- set reasoning_content = '' %}
{%- set content = message.content %}
{%- if message.reasoning_content is string %}{%- set reasoning_content = message.reasoning_content %}
{%- elif '</think>' in content %}
{%- set reasoning_content = content.split('</think>')[0].split('<think>')[-1].strip('\\n') %}
{%- set content = content.split('</think>')[-1].lstrip('\\n') %}
{%- endif %}
{{- '<|im_start|>assistant\\n<think>\\n' + reasoning_content + '\\n</think>\\n\\n' + content + '<|im_end|>\\n' }}
{%- else %}
{{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>\\n' }}
{%- endif %}
{%- endfor %}
"""

_GPT_OSS_TEMPLATE = """
{%- for message in messages %}
{%- if message.role == 'assistant' and message.thinking %}
{{- '<|start|>assistant<|channel|>analysis<|message|>' + message.thinking + '<|end|>' }}
{%- endif %}
{{- '<|start|>' + message.role + '<|message|>' + message.content + '<|end|>' }}
{%- endfor %}
"""

_REPO_ROOT = Path(__file__).resolve().parents[3]


def _unsloth_template(name):
    source = (_REPO_ROOT / "unsloth" / "chat_templates.py").read_text(encoding = "utf-8")
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == name:
            return ast.literal_eval(node.value)
    raise AssertionError(name)


_THINKING = "The user asks in French about the capital. It is Paris."


class _JinjaTokenizer:
    eos_token = ""

    def __init__(self, template):
        self.chat_template = template

    def apply_chat_template(self, conversation, **_kwargs):
        sandbox = pytest.importorskip("jinja2.sandbox")
        env = sandbox.ImmutableSandboxedEnvironment(trim_blocks = True, lstrip_blocks = True)
        return env.from_string(self.chat_template).render(messages = conversation)


def _multilingual_thinking_rows(answer = "Paris."):
    return Dataset.from_list(
        [
            {
                "reasoning_language": "French",
                "messages": [
                    {"role": "system", "content": "reasoning language: French", "thinking": None},
                    {
                        "role": "user",
                        "content": "Quelle est la capitale de la France ?",
                        "thinking": None,
                    },
                    {"role": "assistant", "content": answer, "thinking": _THINKING},
                ],
            }
        ]
    )


def _training_text(template, answer = "Paris."):
    result = format_and_template_dataset(
        _multilingual_thinking_rows(answer),
        model_name = "stub-model",
        tokenizer = _JinjaTokenizer(template),
        num_proc = 1,
    )
    assert result["success"] is True, result["errors"]
    return result["dataset"][0]["text"]


def test_dataset_thinking_trains_inside_qwen_think_block():
    text = _training_text(_QWEN3_TEMPLATE)

    assert f"<|im_start|>assistant\n<think>\n{_THINKING}\n</think>\n\nParis.<|im_end|>" in text


def test_dataset_thinking_trains_inside_gemma4_thought_channel():
    text = _training_text(_unsloth_template("gemma4_template"))

    assert f"<|turn>model\n<|channel>thought\n{_THINKING}\n<channel|>Paris.<turn|>" in text


def test_dataset_thinking_renders_once_on_a_template_reading_thinking():
    text = _training_text(_GPT_OSS_TEMPLATE)

    assert text.count(_THINKING) == 1


def test_dataset_thinking_already_inline_in_content_renders_once():
    text = _training_text(_QWEN3_TEMPLATE, answer = f"<think>\n{_THINKING}\n</think>\n\nParis.")

    assert text.count(_THINKING) == 1
    assert f"<think>\n{_THINKING}\n</think>\n\nParis.<|im_end|>" in text
