# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from pathlib import Path
from types import SimpleNamespace

import pytest
from jinja2 import Environment

from core.inference.template_capabilities import template_supports_tools


@pytest.mark.parametrize(
    "template",
    [
        "{# message.tool_calls are unsupported #}{{ message.content }}",
        "{{ 'message.tool_calls are unsupported' }}",
        "{{ \"message['tool_calls'] is unsupported\" }}",
        "{{ '{% if tools %}example{% endif %}' }}",
        "{# {% if tools %}example{% endif %} #}",
        "{% raw %}{% if tools %}example{% endif %}{% endraw %}",
        "{% if not tools %}plain{% endif %}",
        "{% if tools is none %}plain{% endif %}",
        "{% if tools is undefined %}plain{% endif %}",
        "{{ not tools }}",
        "{{ tools is not none }}",
        "{{ tools | length }}",
        "{% if tools %}{{ tools is defined }}{% endif %}",
        "{% if tools is not none %}{{ raise_exception('tools unsupported') }}{% endif %}",
        "{% if tools %}{{ raise_exception('unsupported: ' ~ tools) }}{% endif %}",
        "{% if message.role == 'tool' %}{{ raise_exception('unsupported role') }}{% endif %}",
        "{% if message.role != 'tool' %}{{ message.content }}{% endif %}",
        "{% if not message.tool_calls %}{{ message.content }}{% endif %}",
        "{% if builtin_tools or tools_in_user_message %}plain{% endif %}",
        "{% set unused = tools %}{{ message.content }}",
        "{% if tools %}",
    ],
)
def test_non_tool_templates_stay_disabled(template):
    assert template_supports_tools(template) is False


@pytest.mark.parametrize(
    "template",
    [
        "{{ tools | tojson }}",
        "{% set available_tools = tools %}{{ available_tools | tojson }}",
        "{% if tools and not available_tools %}{% set available_tools = tools %}"
        "{% endif %}{% if available_tools %}{{ available_tools | tojson }}{% endif %}",
        "{% for tool in tools %}{{ tool | tojson }}{% endfor %}",
        "{% if tools is not none %}{{ tools | tojson }}{% endif %}",
        "{% if documents %}docs{% elif tools %}{{ tools | tojson }}{% endif %}",
        "{%+ if tools +%}{{ tools | tojson }}{% endif %}",
        "{% if message.role == 'tool' %}{{ message.content }}{% endif %}",
        "{% if 'tool' == message['role'] %}{{ message.content }}{% endif %}",
        "{% if message.tool_calls %}{{ message.tool_calls | tojson }}{% endif %}",
        "{% if tool_calls is defined %}{{ tool_calls | tojson }}{% endif %}",
        "{% generation %}{{ message.tool_calls | tojson }}{% endgeneration %}",
        "{% for message in messages %}{% if loop.index > 1 %}{% break %}{% endif %}"
        "{{ message.tool_calls | tojson }}{% endfor %}",
        "{{ '{% raw %}' }}{{ tools | tojson }}{{ '{% endraw %}' }}",
    ],
)
def test_executable_tool_templates_are_detected(template):
    assert template_supports_tools(template) is True


def _published_template(name):
    return (Path(__file__).parent / "data" / "chat_templates" / f"{name}.jinja").read_text()


@pytest.mark.parametrize("name", ["granite-3.3", "phi-4-mini"])
def test_detection_does_not_bypass_the_safetensors_parser_gate(name):
    from routes.inference import _detect_safetensors_features

    template = _published_template(name)
    assert template_supports_tools(template) is True
    flags = _detect_safetensors_features(SimpleNamespace(active_model_name = name), template)
    assert flags["supports_tools"] is False


@pytest.mark.parametrize("name, emits_schema", [("granite-3.3", True), ("phi-4-mini", False)])
def test_published_templates_render_studios_tool_argument(name, emits_schema):
    from core.inference.chat_template_helpers import apply_chat_template_for_generation

    class Tokenizer:
        chat_template = _published_template(name)

        def apply_chat_template(self, messages, **kwargs):
            return (
                Environment()
                .from_string(self.chat_template)
                .render(messages = messages, eos_token = "", **kwargs)
            )

    tokenizer = Tokenizer()
    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hi"},
    ]
    tools = [{"type": "function", "function": {"name": "get_weather", "parameters": {}}}]
    with_tools = apply_chat_template_for_generation(tokenizer, messages, tools = tools)
    without_tools = apply_chat_template_for_generation(tokenizer, messages)
    assert ("get_weather" in with_tools) is emits_schema
    assert (with_tools != without_tools) is emits_schema
