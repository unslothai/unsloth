# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A template that enforces a reasoning_effort allowlist publishes it, and its raise reaches the client."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from core.inference.llama_cpp import detect_reasoning_flags  # noqa: E402
import routes.inference as inference_routes  # noqa: E402


# The reasoning_effort block of unsloth/Mistral-Small-4-119B-2603's chat_template.jinja.
MISTRAL_SMALL_4_TEMPLATE = """
{%- set reasoning_effort = reasoning_effort if reasoning_effort is defined and reasoning_effort is not none else 'none' %}
{%- if reasoning_effort not in ['none', 'high'] %}
    {{- raise_exception('reasoning_effort must be either "none" or "high"') }}
{%- endif %}
{%- set model_settings = '[MODEL_SETTINGS]{"reasoning_effort": "' + reasoning_effort + '"}[/MODEL_SETTINGS]' %}
{%- if message['role'] not in ['user', 'assistant'] %}
    {{- raise_exception('User message must have a string or a list of chunks in content') }}
{%- endif %}
"""

GPT_OSS_TEMPLATE = """
{%- if reasoning_effort is not defined %}
    {%- set reasoning_effort = "medium" %}
{%- endif %}
{{- "Reasoning: " + reasoning_effort + "\\n\\n" }}
"""

# llama-server's error body for the raise above, as returned by build b11985.
_MISTRAL_RAISE_BODY = json.dumps(
    {
        "error": {
            "code": 500,
            "message": (
                "\n------------\nWhile executing CallExpression at line 187, column 24 in source:\n"
                "...one', 'high'] %}↵    {{- raise_exception('reasoning_effort must be "
                'either "none"...\n                                           ^\n'
                'Error: Jinja Exception: reasoning_effort must be either "none" or "high"'
            ),
            "type": "server_error",
        }
    }
)


def _llama_500(message_body: str) -> RuntimeError:
    return RuntimeError(f"llama-server returned 500: {message_body}")


def test_enforced_allowlist_is_published():
    flags = detect_reasoning_flags(
        MISTRAL_SMALL_4_TEMPLATE, "unsloth/Mistral-Small-4-119B-2603-GGUF"
    )
    assert flags["reasoning_style"] == "reasoning_effort"
    assert flags["reasoning_effort_levels"] == ["none", "high"]


def test_allowlist_of_only_none_is_not_a_ladder():
    template = (
        "{%- if reasoning_effort not in ['none'] %}{{- raise_exception('off only') }}{%- endif %}"
    )
    assert detect_reasoning_flags(template)["reasoning_effort_levels"] == []


@pytest.mark.parametrize(
    ("template", "expected"),
    [
        (GPT_OSS_TEMPLATE, []),
        (
            "{%- if reasoning_effort == 'low' %}l{%- elif reasoning_effort == 'high' %}h{%- endif %}",
            ["low", "high"],
        ),
        # A membership branch that groups levels is not a guard: high stays offered.
        (
            "{%- if reasoning_effort in ['low', 'medium'] %}short"
            "{%- elif reasoning_effort == 'high' %}long{%- endif %}",
            ["low", "medium", "high"],
        ),
        # Raising on membership rejects the listed level, so it must not become the only choice.
        (
            "{%- if reasoning_effort in ['high'] %}{{- raise_exception('high is not supported') }}{%- endif %}"
            "{%- if reasoning_effort == 'low' %}l{%- endif %}",
            ["low", "high"],
        ),
        # `not in` without a raise only formats.
        (
            "{%- if reasoning_effort not in ['low', 'high'] %}default{%- endif %}"
            "{%- if reasoning_effort == 'medium' %}m{%- endif %}",
            ["low", "medium", "high"],
        ),
    ],
)
def test_templates_without_a_validation_guard_keep_the_scan(template, expected):
    assert detect_reasoning_flags(template)["reasoning_effort_levels"] == expected


def test_allowlist_ignores_levels_quoted_elsewhere():
    template = (
        "{%- if tool_choice == 'none' %}{%- endif %}"
        "{%- if reasoning_effort not in ['low', 'medium', 'high'] %}"
        "{{- raise_exception('unsupported reasoning_effort') }}{%- endif %}"
    )
    assert detect_reasoning_flags(template)["reasoning_effort_levels"] == ["low", "medium", "high"]


@pytest.fixture
def mistral_template_loaded(monkeypatch):
    monkeypatch.setattr(inference_routes, "_loaded_chat_template", lambda: MISTRAL_SMALL_4_TEMPLATE)


def test_template_raise_with_quotes_reaches_the_client(mistral_template_loaded):
    exc = _llama_500(_MISTRAL_RAISE_BODY)
    expected = 'An internal error occurred: reasoning_effort must be either "none" or "high"'
    assert inference_routes._friendly_error(exc) == expected
    assert inference_routes._openai_stream_error_chunk(exc)["error"]["message"] == expected


def test_template_raise_from_an_override_reaches_the_client(monkeypatch):
    class _Backend:
        chat_template = (
            "{{ bos_token }}{% for message in messages %}{{ message['content'] }}{% endfor %}"
        )
        chat_template_override = MISTRAL_SMALL_4_TEMPLATE

    monkeypatch.setattr(inference_routes, "get_llama_cpp_backend", lambda: _Backend())
    assert inference_routes._friendly_error(_llama_500(_MISTRAL_RAISE_BODY)) == (
        'An internal error occurred: reasoning_effort must be either "none" or "high"'
    )


def test_template_raise_without_quotes_still_reaches_the_client(mistral_template_loaded):
    body = json.dumps(
        {
            "error": {
                "code": 500,
                "message": "Error: Jinja Exception: User message must have a string or a list of chunks in content",
                "type": "server_error",
            }
        }
    )
    assert inference_routes._friendly_error(_llama_500(body)) == (
        "An internal error occurred: User message must have a string or a list of chunks in content"
    )


def test_server_text_not_in_the_template_stays_hidden(mistral_template_loaded):
    body = json.dumps(
        {
            "error": {
                "code": 500,
                "message": 'Error: Jinja Exception: failed to open "/home/user/secret/model.gguf"',
                "type": "server_error",
            }
        }
    )
    assert inference_routes._friendly_error(_llama_500(body)) == "An internal error occurred"


def test_undecodable_body_keeps_the_raw_text_cut(mistral_template_loaded):
    raw = "Jinja Exception: User message must have a string or a list of chunks in content\ntrailer"
    assert inference_routes._friendly_error(_llama_500(raw)) == (
        "An internal error occurred: User message must have a string or a list of chunks in content"
    )
