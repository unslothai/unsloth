# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

"""Gemma-4 26B / 31B empty thought channel on non-thinking model turns; E2B / E4B unchanged.

unsloth import needs a GPU, so template code is pulled out via ast.
"""

import ast
import os

import pytest
from jinja2.exceptions import TemplateError
from jinja2.sandbox import ImmutableSandboxedEnvironment

CHAT_TEMPLATES_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "unsloth",
    "chat_templates.py",
)
_SOURCE = open(CHAT_TEMPLATES_PATH, encoding = "utf-8").read()
_TREE = ast.parse(_SOURCE)

EMPTY = "<|channel>thought\n<channel|>"


def _load():
    wanted_assign = {
        "gemma4_template",
        "gemma4_thinking_template",
        "_gemma4_model_turn",
        "gemma4_empty_thought_template",
        "GEMMA4_TEMPLATE_NAMES",
        "gemma4_empty_thought_ollama",
    }
    namespace = {}
    for node in _TREE.body:
        if isinstance(node, ast.Assign) and any(
            getattr(target, "id", None) in wanted_assign for target in node.targets
        ):
            exec(compile(ast.Module([node], []), CHAT_TEMPLATES_PATH, "exec"), namespace)
        elif isinstance(node, ast.FunctionDef) and node.name == "_gemma4_wants_empty_thought":
            exec(compile(ast.Module([node], []), CHAT_TEMPLATES_PATH, "exec"), namespace)
    return namespace


NS = _load()


def _render(template, messages, **kwargs):
    env = ImmutableSandboxedEnvironment(trim_blocks = True, lstrip_blocks = True)
    env.globals["raise_exception"] = lambda msg: (_ for _ in ()).throw(TemplateError(msg))
    ctx = {"messages": messages, "add_generation_prompt": False, "bos_token": "<bos>"}
    ctx.update(kwargs)
    return env.from_string("{{ bos_token }}" + template).render(**ctx)


class _Holder:
    def __init__(self, chat_template):
        self.chat_template = chat_template


# Tails of google/gemma-4-26B-A4B-it and google/gemma-4-E2B-it chat_template.jinja (verbatim)
GOOGLE_26B_TAIL = (
    "{%- if add_generation_prompt -%}\n"
    "    {%- if ns.prev_message_type != 'tool_response' and ns.prev_message_type != 'tool_call' -%}\n"
    "        {{- '<|turn>model\\n' -}}\n"
    "        {%- if not enable_thinking -%}\n"
    "            {{- '<|channel>thought\\n<channel|>' -}}\n"
    "        {%- endif -%}\n"
    "    {%- elif ns.prev_message_type == 'tool_response' and enable_thinking -%}\n"
    "        {{- '<|channel>thought\\n' -}}\n"
    "    {%- endif -%}\n"
    "{%- endif -%}\n"
)
GOOGLE_E2B_TAIL = (
    "{%- if add_generation_prompt -%}\n"
    "    {%- if ns.prev_message_type != 'tool_response' and ns.prev_message_type != 'tool_call' -%}\n"
    "        {{- '<|turn>model\\n' -}}\n"
    "    {%- elif ns.prev_message_type == 'tool_response' and enable_thinking -%}\n"
    "        {{- '<|channel>thought\\n' -}}\n"
    "    {%- endif -%}\n"
    "{%- endif -%}\n"
)

CONVO = [
    {"role": "system", "content": "Be brief."},
    {"role": "user", "content": "Hi"},
    {"role": "assistant", "content": "Hello!"},
    {"role": "user", "content": "2+2?"},
    {"role": "assistant", "content": "4"},
]


def test_detects_26b_31b_template():
    assert NS["_gemma4_wants_empty_thought"](_Holder(GOOGLE_26B_TAIL))


def test_ignores_e2b_template():
    assert not NS["_gemma4_wants_empty_thought"](_Holder(GOOGLE_E2B_TAIL))


def test_ignores_missing_or_non_string_templates():
    assert not NS["_gemma4_wants_empty_thought"](None, _Holder(None), _Holder(123))


def test_ignores_unsloth_thinking_template():
    # Re-calling get_chat_template on an E2B tokenizer already carrying it must not switch.
    tmpl = "{{ bos_token }}" + NS["gemma4_thinking_template"]
    assert not NS["_gemma4_wants_empty_thought"](_Holder(tmpl))


def test_detects_own_output_on_recall():
    tmpl = "{{ bos_token }}" + NS["gemma4_empty_thought_template"]
    assert NS["_gemma4_wants_empty_thought"](_Holder(tmpl))


def test_processor_dict_template():
    assert NS["_gemma4_wants_empty_thought"](_Holder({"default": GOOGLE_26B_TAIL}))


def test_multi_turn_with_system_prompt():
    out = _render(NS["gemma4_empty_thought_template"], CONVO)
    assert out == (
        "<bos><|turn>system\nBe brief.<turn|>\n"
        "<|turn>user\nHi<turn|>\n"
        f"<|turn>model\n{EMPTY}Hello!<turn|>\n"
        "<|turn>user\n2+2?<turn|>\n"
        f"<|turn>model\n{EMPTY}4<turn|>\n"
    )


def test_generation_prompt_matches_google_26b():
    msgs = [{"role": "user", "content": "Hi"}]
    out = _render(NS["gemma4_empty_thought_template"], msgs, add_generation_prompt = True)
    assert out == f"<bos><|turn>user\nHi<turn|>\n<|turn>model\n{EMPTY}"
    out = _render(
        NS["gemma4_empty_thought_template"], msgs, add_generation_prompt = True, enable_thinking = True
    )
    assert out == "<bos><|turn>system\n<|think|>\n<turn|>\n<|turn>user\nHi<turn|>\n<|turn>model\n"


def test_training_text_is_prefix_consistent_with_generation_prompt():
    full = _render(NS["gemma4_empty_thought_template"], CONVO[:3])
    prompt = _render(NS["gemma4_empty_thought_template"], CONVO[:2], add_generation_prompt = True)
    assert full == prompt + "Hello!<turn|>\n"


def test_list_content():
    msgs = [
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "What?"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "A cat."}]},
    ]
    out = _render(NS["gemma4_empty_thought_template"], msgs)
    assert out == f"<bos><|turn>user\n<|image|>What?<turn|>\n<|turn>model\n{EMPTY}A cat.<turn|>\n"


@pytest.mark.parametrize("key", ["reasoning_content", "reasoning"])
def test_final_turn_reasoning_becomes_the_thought_channel(key):
    msgs = [
        {"role": "user", "content": "2+2?"},
        {"role": "assistant", "content": "4", key: "2+2=4"},
    ]
    out = _render(NS["gemma4_empty_thought_template"], msgs)
    assert (
        out
        == "<bos><|turn>user\n2+2?<turn|>\n<|turn>model\n<|channel>thought\n2+2=4\n<channel|>4<turn|>\n"
    )


def test_reasoning_before_the_last_user_turn_is_dropped_like_the_model_template():
    msgs = [
        {"role": "user", "content": "2+2?"},
        {"role": "assistant", "content": "4", "reasoning_content": "2+2=4"},
        {"role": "user", "content": "3+3?"},
        {"role": "assistant", "content": "6"},
    ]
    out = _render(NS["gemma4_empty_thought_template"], msgs)
    assert "2+2=4" not in out
    assert out.count(EMPTY) == 2


@pytest.mark.parametrize(
    "content",
    [
        "<|channel>thought\n2+2=4<channel|>4",
        [{"type": "text", "text": "<|channel>thought\nx<channel|>4"}],
    ],
)
def test_inline_thoughts_are_stripped_and_get_the_empty_channel(content):
    msgs = [{"role": "user", "content": "2+2?"}, {"role": "assistant", "content": content}]
    out = _render(NS["gemma4_empty_thought_template"], msgs)
    assert out == f"<bos><|turn>user\n2+2?<turn|>\n<|turn>model\n{EMPTY}4<turn|>\n"


def test_ollama_generation_prompt_has_the_empty_channel():
    ollama = NS["gemma4_empty_thought_ollama"]
    assert "{{ if $last }}<|turn>model\n" + EMPTY + "{{ end }}" in ollama


def test_ollama_history_assistant_turns_have_the_empty_channel():
    ollama = NS["gemma4_empty_thought_ollama"]
    assert '{{- if eq .Role "assistant" }}<|turn>model\n' + EMPTY + "{{ .Content }}" in ollama
    assert "<|turn>assistant" not in ollama
    assert ollama.count(EMPTY) == 2


def test_enable_thinking_is_unchanged():
    new = _render(NS["gemma4_empty_thought_template"], CONVO, enable_thinking = True)
    old = _render(NS["gemma4_thinking_template"], CONVO, enable_thinking = True)
    assert new == old
    assert EMPTY not in new


def test_only_model_turns_change():
    new = _render(NS["gemma4_empty_thought_template"], CONVO)
    old = _render(NS["gemma4_thinking_template"], CONVO)
    assert new.replace(EMPTY, "") == old


def test_roles_must_alternate_still_raises():
    with pytest.raises(TemplateError):
        _render(NS["gemma4_empty_thought_template"], [{"role": "assistant", "content": "x"}])


def test_get_chat_template_switches_only_gemma4_names():
    body = next(
        node
        for node in _TREE.body
        if isinstance(node, ast.FunctionDef) and node.name == "get_chat_template"
    )
    src = ast.get_source_segment(_SOURCE, body)
    assert "_gemma4_wants_empty_thought(_processor, old_tokenizer)" in src
    assert "type_chat_template in GEMMA4_TEMPLATE_NAMES" in src
    assert set(NS["GEMMA4_TEMPLATE_NAMES"]) == {
        "gemma-4",
        "gemma4",
        "gemma-4-thinking",
        "gemma4-thinking",
    }
