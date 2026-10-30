# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""A template that renders every replayed reasoning trace must get that reasoning back on every turn.

GLM-5.x sets ``clear_thinking`` false by default, so llama-server renders the reasoning of each
earlier answer. Dropping it once the next user message arrives changed the prompt right after the
first answered query, and llama-server could not reuse its prompt cache: Claude Desktop re-prefilled
its whole conversation (tens of thousands of tokens) on every message.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import jinja2

from core.inference.anthropic_compat import anthropic_messages_to_openai
from core.inference.llama_cpp import detect_reasoning_flags, template_replays_reasoning_by_default

# The reasoning-replay core of GLM-5.3-Flash's template, verbatim.
GLM_TEMPLATE = """{%- set clear_thinking = clear_thinking if clear_thinking is defined else false -%}
{%- set ns = namespace(last_user_index=-1) -%}
{%- for m in messages %}{%- if m.role == 'user' %}{%- set ns.last_user_index = loop.index0 -%}{%- endif %}{%- endfor %}
{%- for m in messages -%}
{%- if m.role == 'system' %}<|system|>{{ m.content }}
{%- elif m.role == 'user' %}<|user|>{{ m.content }}
{%- elif m.role == 'assistant' %}<|assistant|>
{%- set content = m.content or '' %}
{%- if m.reasoning_content is string %}{%- set reasoning_content = m.reasoning_content %}
{%- elif '</think>' in content %}{%- set reasoning_content = content.split('</think>')[0].split('<think>')[-1] %}{%- endif %}
{%- if (not clear_thinking or loop.index0 > ns.last_user_index) and reasoning_content is defined -%}
{{ '<think>' + reasoning_content + '</think>' }}
{%- else -%}{{ '</think>' }}{%- endif -%}
{{ content }}
{%- endif -%}
{%- endfor -%}
{%- if add_generation_prompt %}<|assistant|><think>{% endif -%}"""


def _render(messages, generation = True):
    return jinja2.Template(GLM_TEMPLATE).render(messages = messages, add_generation_prompt = generation)


def test_glm_templates_are_detected_as_replaying_reasoning():
    assert template_replays_reasoning_by_default(GLM_TEMPLATE) is True
    flipped = GLM_TEMPLATE.replace("else false", "else true")
    assert template_replays_reasoning_by_default(flipped) is False
    assert template_replays_reasoning_by_default("{{ messages }}") is False
    assert template_replays_reasoning_by_default(None) is False


def test_the_global_reasoning_flags_stay_as_upstream():
    """Only the Claude Desktop gateway acts on the GLM detection; load/status flags are unchanged."""
    flags = detect_reasoning_flags(GLM_TEMPLATE, "unsloth/GLM-5.3-Flash-GGUF")
    assert flags["preserve_thinking_default"] is False
    assert flags["supports_preserve_thinking"] is False


def _claude_turns():
    system = [{"type": "text", "text": "You are Claude. " * 50}]
    first = [{"role": "user", "content": "List the files."}]
    answered = first + [
        {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "I should call ls.", "signature": ""},
                {"type": "tool_use", "id": "toolu_1", "name": "ls", "input": {}},
            ],
        },
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "a.py"}],
        },
        {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "One file.", "signature": ""},
                {"type": "text", "text": "There is a.py."},
            ],
        },
    ]
    second = answered + [{"role": "user", "content": "Open it."}]
    return system, answered, second


def test_the_next_message_extends_the_prompt_the_last_one_was_built_on():
    system, answered, second = _claude_turns()
    preserve = template_replays_reasoning_by_default(GLM_TEMPLATE)  # what the gateway path uses
    # What the model read at the end of turn one (tool loop in progress: thinking kept)...
    before = _render(anthropic_messages_to_openai(answered[:3], system, preserve_thinking = preserve))
    # ...must be a prefix of what turn two sends, so llama-server reuses the cache.
    after = _render(anthropic_messages_to_openai(second, system, preserve_thinking = preserve))
    assert after.startswith(before)


def test_without_preservation_the_prompt_breaks_after_the_first_query():
    system, answered, second = _claude_turns()
    before = _render(anthropic_messages_to_openai(answered[:3], system))
    after = _render(anthropic_messages_to_openai(second, system))
    assert not after.startswith(before)  # the regression this file guards against


def test_claude_desktops_per_request_billing_header_is_not_part_of_the_prompt():
    def system(cch):
        return [
            {
                "type": "text",
                "text": f"x-anthropic-billing-header: cc_version=2.3.1.abc; cch={cch};",
            },
            {"type": "text", "text": "You are Claude Code."},
        ]

    msgs = [{"role": "user", "content": "hi"}]
    strip = {"strip_billing_header": True}
    first = anthropic_messages_to_openai(msgs, system("1a2b3"), **strip)
    second = anthropic_messages_to_openai(msgs, system("9f8e7"), **strip)
    assert first == second
    assert first[0] == {"role": "system", "content": "You are Claude Code."}
    assert anthropic_messages_to_openai(
        msgs, "x-anthropic-billing-header: cch=1;\nBe brief.", **strip
    )[0] == {"role": "system", "content": "Be brief."}
    assert (
        anthropic_messages_to_openai(msgs, "x-anthropic-billing-header: cch=1;", **strip)[0]["role"]
        == "user"
    )


def test_other_clients_drop_the_leading_billing_line_as_upstream():
    # Upstream (#15845) strips the first billing line for every client.
    msgs = [{"role": "user", "content": "hi"}]
    text = "x-anthropic-billing-header: cc_version=1; cch=abc;\nYou are helpful"
    assert anthropic_messages_to_openai(msgs, text)[0] == {
        "role": "system",
        "content": "You are helpful",
    }
    blocks = [
        {"type": "text", "text": "x-anthropic-billing-header: cch=1;"},
        {"type": "text", "text": "Hi"},
    ]
    assert anthropic_messages_to_openai(msgs, blocks)[0]["content"] == "Hi"


class _Headers(dict):
    def get(
        self,
        key,
        default = None,
    ):
        return {k.lower(): v for k, v in self.items()}.get(key.lower(), default)


class _Request:
    def __init__(self, headers):
        self.headers = _Headers(headers)


class _Backend:
    preserve_thinking_default = False
    replays_reasoning_by_default = True


class _Payload:
    preserve_thinking = None


def test_only_the_claude_desktop_gateway_keeps_glm_reasoning_by_default():
    from routes.inference import _anthropic_preserve_thinking, _from_claude_desktop_gateway
    from utils.claude_desktop import GATEWAY_CLIENT_HEADER, GATEWAY_CLIENT_VALUE

    gateway = _Request({GATEWAY_CLIENT_HEADER: GATEWAY_CLIENT_VALUE})
    assert _from_claude_desktop_gateway(gateway) is True
    assert _from_claude_desktop_gateway(_Request({})) is False
    assert _from_claude_desktop_gateway(None) is False
    assert _anthropic_preserve_thinking(_Backend(), _Payload(), gateway) is True
    # Claude Code and every other client: upstream's launch default.
    assert _anthropic_preserve_thinking(_Backend(), _Payload(), _Request({})) is False
    assert _anthropic_preserve_thinking(_Backend(), _Payload()) is False
    # An explicit request still wins either way.
    explicit = _Payload()
    explicit.preserve_thinking = False
    assert _anthropic_preserve_thinking(_Backend(), explicit, gateway) is False
    non_glm = _Backend()
    non_glm.replays_reasoning_by_default = False
    assert _anthropic_preserve_thinking(non_glm, _Payload(), gateway) is False
