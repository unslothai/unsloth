# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Mistral templates raise unless every tool-call id is nine alphanumerics; the replayed history must satisfy them."""

import re

import jinja2

from core.inference.chat_template_helpers import model_markup, neutralize_control_markup_in_messages

# The tool-call parts of Mistral-Nemo-Instruct-2407's template, as shipped in its GGUF.
MISTRAL_TEMPLATE = """{%- for message in messages %}
{%- if message.role == "user" %}[INST]{{ message.content }}[/INST]
{%- elif message.tool_calls is defined and message.tool_calls is not none %}[TOOL_CALLS][
{%- for tool_call in message.tool_calls %}
{%- if not tool_call.id is defined or tool_call.id|length != 9 %}
{{- raise_exception("Tool call IDs should be alphanumeric strings with length 9!") }}
{%- endif %}{"name": "{{ tool_call.function.name }}", "id": "{{ tool_call.id }}"}
{%- endfor %}]</s>
{%- elif message.role == "tool" %}
{%- if not message.tool_call_id is defined or message.tool_call_id|length != 9 %}
{{- raise_exception("Tool call IDs should be alphanumeric strings with length 9!") }}
{%- endif %}[TOOL_RESULTS]{"content": {{ message.content|tojson }}, "call_id": "{{ message.tool_call_id }}"}[/TOOL_RESULTS]
{%- elif message.role == "assistant" %}{{ message.content }}</s>
{%- endif %}
{%- endfor %}"""

# llama-server's id, with the suffix Studio adds to keep ids unique across runs.
LONG_ID = "8THmZjw2H0blOXVXDgxwoeHwjlfDRG70:6b0b264d-a19d-47c9-80fa-b6bea04f31dd"


def _exchange(*ids):
    messages = [{"role": "user", "content": "What day is it?"}]
    for call_id in ids:
        messages.append(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {"name": "web_search", "arguments": "{}"},
                    }
                ],
            }
        )
        messages.append(
            {"role": "tool", "name": "web_search", "tool_call_id": call_id, "content": "Thursday"}
        )
    return messages


def _render(messages):
    def raise_exception(message):
        raise jinja2.exceptions.TemplateError(message)

    env = jinja2.Environment()
    env.globals["raise_exception"] = raise_exception
    return env.from_string(MISTRAL_TEMPLATE).render(messages = messages)


def test_mistral_template_is_detected():
    markup = model_markup(MISTRAL_TEMPLATE)
    assert markup is not None and markup.nine_char_tool_call_ids


def test_long_ids_are_replayed_as_nine_alphanumerics_that_still_pair():
    markup = model_markup(MISTRAL_TEMPLATE)
    out = neutralize_control_markup_in_messages(_exchange(LONG_ID, "call_0"), None, markup)
    call_ids = [m["tool_calls"][0]["id"] for m in out if m.get("tool_calls")]
    result_ids = [m["tool_call_id"] for m in out if m.get("role") == "tool"]
    assert call_ids == result_ids
    assert all(re.fullmatch(r"[A-Za-z0-9]{9}", value) for value in call_ids)
    assert len(set(call_ids)) == 2
    rendered = _render(out)
    assert rendered.count("[TOOL_RESULTS]") == 2


def test_the_mapping_is_stable_across_rounds():
    markup = model_markup(MISTRAL_TEMPLATE)
    first = neutralize_control_markup_in_messages(_exchange(LONG_ID), None, markup)
    second = neutralize_control_markup_in_messages(_exchange(LONG_ID, "other"), None, markup)
    assert first[1]["tool_calls"][0]["id"] == second[1]["tool_calls"][0]["id"]


def test_an_id_that_already_fits_is_kept():
    markup = model_markup(MISTRAL_TEMPLATE)
    out = neutralize_control_markup_in_messages(_exchange("abc123XYZ"), None, markup)
    assert out[1]["tool_calls"][0]["id"] == "abc123XYZ"


def test_other_templates_keep_their_ids():
    chatml = (
        "{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{% endfor %}"
    )
    markup = model_markup(chatml)
    assert not markup.nine_char_tool_call_ids
    out = neutralize_control_markup_in_messages(_exchange(LONG_ID), None, markup)
    assert out[1]["tool_calls"][0]["id"] == LONG_ID
