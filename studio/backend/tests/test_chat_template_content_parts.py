# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import pytest
from datasets import Dataset

from utils.datasets import apply_chat_template_to_dataset, format_and_template_dataset

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


@pytest.mark.parametrize("tokenizer_class", [_JinjaProcessor, _JinjaTokenizer])
def test_native_alpaca_keeps_text_for_list_only_templates(tokenizer_class):
    dataset = Dataset.from_list(
        [{"system": "Be brief.", "instruction": "Say hi", "input": "in French", "output": "Salut"}]
    )
    result = format_and_template_dataset(
        dataset,
        model_name = "Idefics3ForConditionalGeneration",
        tokenizer = tokenizer_class(_IDEFICS3_TEMPLATE),
        prefer_native_alpaca = True,
        num_proc = 1,
    )

    assert result["success"] is True
    assert result["dataset"][0]["text"] == "system:Be brief.user:Say hi\n\nin Frenchassistant:Salut"


@pytest.mark.parametrize("template_adds_eos", [True, False])
def test_native_alpaca_renders_an_existing_assistant_eos_once(template_adds_eos):
    tokenizer = _JinjaTokenizer(
        _QWEN3_TEMPLATE if template_adds_eos else "{{ messages[-1].content }}"
    )
    tokenizer.eos_token = "<|im_end|>"
    dataset = Dataset.from_list([{"instruction": "Echo", "output": "done<|im_end|>"}])
    result = format_and_template_dataset(
        dataset,
        model_name = "Qwen3ForCausalLM",
        tokenizer = tokenizer,
        prefer_native_alpaca = True,
        num_proc = 1,
    )

    assert result["success"] is True
    text = result["dataset"][0]["text"]
    assert text.rstrip().endswith("done<|im_end|>")


def test_native_alpaca_drops_an_eos_the_turn_terminator_replaces():
    # Gemma (<eos> vs <end_of_turn>) and base Qwen (<|endoftext|> vs <|im_end|>).
    tokenizer = _JinjaTokenizer(_QWEN3_TEMPLATE)
    tokenizer.eos_token = "<|endoftext|>"
    dataset = Dataset.from_list([{"instruction": "Echo", "output": "done<|endoftext|>"}])
    result = format_and_template_dataset(
        dataset,
        model_name = "Qwen3ForCausalLM",
        tokenizer = tokenizer,
        prefer_native_alpaca = True,
        num_proc = 1,
    )

    assert result["success"] is True
    text = result["dataset"][0]["text"]
    assert text.rstrip().endswith("done<|im_end|>")
    assert "<|endoftext|>" not in text


def test_native_alpaca_terminates_rows_when_the_template_does_not():
    tokenizer = _JinjaTokenizer("{{ messages[-1].content }}")
    tokenizer.eos_token = "<|im_end|>"
    dataset = Dataset.from_list([{"instruction": "Echo", "output": "done"}])
    result = format_and_template_dataset(
        dataset,
        model_name = "Qwen3ForCausalLM",
        tokenizer = tokenizer,
        prefer_native_alpaca = True,
        num_proc = 1,
    )

    assert result["dataset"][0]["text"] == "done<|im_end|>"


def test_native_alpaca_keeps_its_template_for_later_splits():
    from utils.datasets.chat_templates import _CHOSEN_TEMPLATE_ATTR

    tokenizer = _JinjaTokenizer(_QWEN3_TEMPLATE)
    tokenizer.eos_token = "<|im_end|>"
    format_and_template_dataset(
        Dataset.from_list([{"instruction": "Echo", "output": "done"}]),
        model_name = "Qwen3ForCausalLM",
        tokenizer = tokenizer,
        prefer_native_alpaca = True,
        num_proc = 1,
    )

    assert getattr(tokenizer, _CHOSEN_TEMPLATE_ATTR) == ("Qwen3ForCausalLM", _QWEN3_TEMPLATE)


def test_native_alpaca_accepts_templates_that_rewrite_the_answer():
    template = (
        "{%- for m in messages %}{{ m.role }}:"
        "{{ m.content.split('</think>')[-1].lstrip() if m.role == 'assistant' else m.content }}"
        "{%- endfor %}"
    )
    result = format_and_template_dataset(
        Dataset.from_list([{"instruction": "Q", "output": "<think>reason</think>\n42"}]),
        model_name = "Qwen3ForCausalLM",
        tokenizer = _JinjaTokenizer(template),
        prefer_native_alpaca = True,
        num_proc = 1,
    )

    assert result["success"] is True
    assert result["dataset"][0]["text"] == "user:Qassistant:42"


def test_native_alpaca_sends_strings_to_a_processors_string_template():
    template = "{%- for m in messages %}{{ m.role }}:{{ m.content | trim }}|{%- endfor %}"
    result = format_and_template_dataset(
        Dataset.from_list([{"instruction": "Say hi", "output": "Salut"}]),
        model_name = "Gemma3ForConditionalGeneration",
        tokenizer = _JinjaProcessor(template),
        prefer_native_alpaca = True,
        num_proc = 1,
    )

    assert result["dataset"][0]["text"] == "user:Say hi|assistant:Salut|"
