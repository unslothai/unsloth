# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import pytest
from datasets import Dataset

from utils.datasets import apply_chat_template_to_dataset

_LLAMA3_TEMPLATE = """
{%- for message in messages %}
{{- '<|start_header_id|>' + message['role'] + '<|end_header_id|>\\n\\n' + message['content'] | trim + '<|eot_id|>' }}
{%- endfor %}
"""

_QWEN3_2507_TEMPLATE = """
{%- for message in messages %}
{%- if message.content is string %}{%- set content = message.content %}{%- else %}{%- set content = '' %}{%- endif %}
{{- '<|im_start|>' + message.role + '\\n' + content + '<|im_end|>\\n' }}
{%- endfor %}
"""

_QWEN3_TEMPLATE = """
{%- for message in messages %}
{{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>\\n' }}
{%- endfor %}
"""

_IDEFICS3_TEMPLATE = """
{%- for message in messages %}
{{- message['role'] + ':' }}{%- for line in message['content'] %}{%- if line['type'] == 'text' %}{{ line['text'] }}{%- endif %}{%- endfor %}
{%- endfor %}
"""


class _JinjaTokenizer:
    eos_token = ""

    def __init__(self, template):
        self.chat_template = template

    def apply_chat_template(
        self,
        conversation,
        add_generation_prompt = False,
        **_kwargs,
    ):
        sandbox = pytest.importorskip("jinja2.sandbox")
        env = sandbox.ImmutableSandboxedEnvironment(trim_blocks = True, lstrip_blocks = True)
        return env.from_string(self.chat_template).render(
            messages = conversation, add_generation_prompt = add_generation_prompt
        )


class _JinjaProcessor(_JinjaTokenizer):
    tokenizer = None


def _format(
    rows,
    template,
    tokenizer_class = _JinjaTokenizer,
):
    dataset_info = {
        "dataset": Dataset.from_list([{"messages": row} for row in rows]),
        "detected_format": "chatml_messages",
        "final_format": "chatml_messages",
        "chat_column": "messages",
        "is_standardized": True,
        "warnings": [],
    }
    return apply_chat_template_to_dataset(dataset_info, tokenizer_class(template), num_proc = 1)


def _parts(*texts):
    return [{"type": "text", "text": text} for text in texts]


@pytest.mark.parametrize(
    "template",
    [_LLAMA3_TEMPLATE, _QWEN3_2507_TEMPLATE, _QWEN3_TEMPLATE],
    ids = ["llama3", "qwen3-2507", "qwen3"],
)
def test_text_content_parts_train_as_their_text(template):
    row = [
        {"role": "system", "content": _parts("be brief")},
        {"role": "user", "content": _parts("hi there", "how are you?")},
        {"role": "assistant", "content": _parts("hello back")},
    ]

    result = _format([row], template)

    assert result["success"] is True
    text = result["dataset"][0]["text"]
    assert "be brief" in text
    assert "hi there\nhow are you?" in text
    assert "hello back" in text
    assert "'type'" not in text


def test_content_parts_and_plain_strings_render_the_same():
    plain = [
        {"role": "user", "content": "hi there"},
        {"role": "assistant", "content": "hello back"},
    ]
    parts = [{**message, "content": _parts(message["content"])} for message in plain]

    texts = [_format([row], _LLAMA3_TEMPLATE)["dataset"][0]["text"] for row in (plain, parts)]

    assert texts[0] == texts[1]


def test_processor_keeps_text_parts_for_list_only_templates():
    row = [
        {"role": "user", "content": _parts("hi there")},
        {"role": "assistant", "content": _parts("hello back")},
    ]

    result = _format([row], _IDEFICS3_TEMPLATE, _JinjaProcessor)

    assert result["dataset"][0]["text"] == "user:hi thereassistant:hello back"
