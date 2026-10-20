# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Mistral Small 3.2 tool definitions must survive training-shaped rows (unslothai/unsloth#3096).

The 2025 Unsloth template only emitted [AVAILABLE_TOOLS] when the conversation ended on a user or
tool turn, so every SFT row ending in an assistant reply silently dropped the tool list.
mistral_common places it before the last user message in both serving and finetuning mode.
"""

import copy
import json
from pathlib import Path

import pytest

import unsloth.tokenizer_utils as tu

jinja2 = pytest.importorskip("jinja2")

OLD = (Path(__file__).parent / "fixtures" / "mistral_small_3_2_tools_2025.jinja").read_text(
    encoding = "utf-8"
)
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
        },
    }
]
TOOLS_TEXT = "[AVAILABLE_TOOLS]" + json.dumps(TOOLS) + "[/AVAILABLE_TOOLS]"
S = {"role": "system", "content": "sys"}


def U(t):
    return {"role": "user", "content": t}


def A(t):
    return {"role": "assistant", "content": t}


def TC(i):
    call = {
        "id": i,
        "type": "function",
        "function": {"name": "get_weather", "arguments": {"city": "X"}},
    }
    return {"role": "assistant", "content": "", "tool_calls": [call]}


def TR(i):
    return {"role": "tool", "tool_call_id": i, "content": "22C"}


TRAIN = {
    "text reply": [S, U("q"), A("a")],
    "tool call, result, reply": [S, U("q"), TC("abc123def"), TR("abc123def"), A("a")],
    "ends in tool call": [S, U("q"), TC("abc123def")],
    "multi-turn": [S, U("hi"), A("hello"), U("q"), TC("abc123def"), TR("abc123def"), A("a")],
    "tool turn then plain turn": [
        S,
        U("q"),
        TC("abc123def"),
        TR("abc123def"),
        A("a"),
        U("thx"),
        A("np"),
    ],
}
SERVE = {
    "user": [S, U("q")],
    "after tool result": [S, U("q"), TC("abc123def"), TR("abc123def")],
    "multi-turn": [S, U("hi"), A("hello"), U("q")],
    "multi-turn after tool result": [
        S,
        U("hi"),
        A("hello"),
        U("q"),
        TC("abc123def"),
        TR("abc123def"),
    ],
    "after a full tool turn": [S, U("q"), TC("abc123def"), TR("abc123def"), A("a"), U("thx")],
}


def render(template, messages, tools):
    env = jinja2.Environment(extensions = ["jinja2.ext.loopcontrols"])
    env.policies["json.dumps_kwargs"] = {"sort_keys": False}

    def raise_exception(msg):
        raise jinja2.exceptions.TemplateError(msg)

    return env.from_string(template).render(
        messages = copy.deepcopy(messages),
        tools = tools,
        bos_token = "<s>",
        eos_token = "</s>",
        raise_exception = raise_exception,
    )


def expected_tools_position(text):
    # mistral_common: the tool list sits immediately before the last [INST].
    return text.count(TOOLS_TEXT) == 1 and text.rfind("[INST]") == text.find(TOOLS_TEXT) + len(
        TOOLS_TEXT
    )


NEW = tu._rewrite_mistral_tools_chat_template(OLD)


def test_rewrite_applies_and_is_idempotent():
    assert NEW is not OLD
    assert "append_tools" not in NEW
    assert tu._rewrite_mistral_tools_chat_template(NEW) is NEW


@pytest.mark.parametrize("name", list(TRAIN))
def test_training_rows_keep_tools(name):
    assert TOOLS_TEXT not in render(OLD, TRAIN[name], TOOLS)
    assert expected_tools_position(render(NEW, TRAIN[name], TOOLS))


@pytest.mark.parametrize("name", list(SERVE))
def test_serving_prompts_unchanged(name):
    old = render(OLD, SERVE[name], TOOLS)
    assert old == render(NEW, SERVE[name], TOOLS)
    assert expected_tools_position(old)


@pytest.mark.parametrize("messages", list(TRAIN.values()) + list(SERVE.values()))
def test_no_tools_unchanged(messages):
    assert render(OLD, messages, None) == render(NEW, messages, None)


def test_other_templates_untouched():
    for template in (
        None,
        "",
        "{{ messages }}",
        OLD.replace("{#- If directly called tools in first turn, prepend to user #}", ""),
    ):
        assert tu._rewrite_mistral_tools_chat_template(template) is template


def test_fixes_processor_and_inner_tokenizer():
    class Obj:
        pass

    proc, inner = Obj(), Obj()
    proc.tokenizer, proc.chat_template, inner.chat_template = inner, OLD, OLD
    tu._fix_mistral_tools_chat_template(proc)
    assert proc.chat_template == NEW and inner.chat_template == NEW
