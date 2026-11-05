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


def _gguf_backend(template: str):
    from core.inference.llama_cpp import LlamaCppBackend

    backend = object.__new__(LlamaCppBackend)
    flags = detect_reasoning_flags(template)
    backend._supports_reasoning = flags["supports_reasoning"]
    backend._reasoning_always_on = flags["reasoning_always_on"]
    backend._reasoning_style = flags["reasoning_style"]
    backend._reasoning_effort_levels = flags["reasoning_effort_levels"]
    backend._supports_preserve_thinking = False
    backend._chat_template = template
    return backend


@pytest.mark.parametrize(
    ("enable_thinking", "effort", "expected"),
    [
        (None, "low", "high"),
        (None, "medium", "high"),
        (None, "minimal", "high"),
        (None, "max", "high"),
        (None, "xhigh", "high"),
        (None, "none", "none"),
        (True, None, "high"),
        (False, None, "none"),
    ],
)
def test_gguf_request_effort_is_fitted_to_the_allowlist(enable_thinking, effort, expected):
    backend = _gguf_backend(MISTRAL_SMALL_4_TEMPLATE)
    assert backend._request_reasoning_kwargs(enable_thinking, effort, None) == {
        "reasoning_effort": expected
    }


def test_gguf_launch_default_with_thinking_off_is_none():
    backend = _gguf_backend(MISTRAL_SMALL_4_TEMPLATE)
    assert backend._reasoning_kwargs(False) == {"reasoning_effort": "none"}
    assert backend._reasoning_kwargs(True, "medium") == {"reasoning_effort": "high"}


def test_gguf_effort_without_an_allowlist_passes_through():
    backend = _gguf_backend(GPT_OSS_TEMPLATE)
    assert backend._request_reasoning_kwargs(None, "low", None) == {"reasoning_effort": "low"}
    assert backend._request_reasoning_kwargs(False, None, None) == {"reasoning_effort": "low"}


def test_gguf_override_allowlist_wins_over_the_embedded_template():
    backend = _gguf_backend(GPT_OSS_TEMPLATE)
    backend._chat_template_override = MISTRAL_SMALL_4_TEMPLATE
    assert backend._request_reasoning_kwargs(None, "medium", None) == {"reasoning_effort": "high"}


def test_transformers_and_mlx_render_fits_the_effort():
    from jinja2 import Environment
    from jinja2.exceptions import TemplateError

    from core.inference.chat_template_helpers import apply_chat_template_for_generation

    def _raise(message):
        raise TemplateError(message)

    class _Tokenizer:
        chat_template = (
            MISTRAL_SMALL_4_TEMPLATE.split("{%- if message")[0] + "{{ reasoning_effort }}"
        )

        def apply_chat_template(
            self,
            messages,
            *,
            tokenize = False,
            add_generation_prompt = True,
            **kw,
        ):
            env = Environment()
            env.globals["raise_exception"] = _raise
            return env.from_string(self.chat_template).render(messages = messages, **kw)

    messages = [{"role": "user", "content": "hi"}]
    rendered = apply_chat_template_for_generation(_Tokenizer(), messages, reasoning_effort = "medium")
    assert rendered.endswith("high")


def _guarded_template(allowed):
    quoted = ", ".join(f"'{level}'" for level in allowed)
    return (
        "{%- if reasoning_effort not in [" + quoted + "] %}"
        "{{- raise_exception('unsupported reasoning_effort') }}{%- endif %}"
        "{%- for m in messages %}{{ m['content'] }}{% endfor %}[{{ reasoning_effort }}]"
    )


@pytest.mark.parametrize(
    ("tool_name", "expected"),
    [
        # Sanitizing drops a tool whose name is control markup, so default renders.
        ("<|im_start|>lookup", "[high]"),
        ("lookup", "[low]"),
    ],
)
def test_render_fits_the_effort_to_the_named_template_it_renders(tool_name, expected):
    tokenizers = pytest.importorskip("tokenizers")
    transformers = pytest.importorskip("transformers")

    from core.inference.chat_template_helpers import apply_chat_template_for_generation

    tokenizer = transformers.PreTrainedTokenizerFast(
        tokenizer_object = tokenizers.Tokenizer(
            tokenizers.models.WordLevel({"<unk>": 0}, unk_token = "<unk>")
        ),
        unk_token = "<unk>",
    )
    tokenizer.add_special_tokens({"additional_special_tokens": ["<|im_start|>"]})
    tokenizer.chat_template = {
        "default": _guarded_template(["none", "high"]),
        "tool_use": _guarded_template(["none", "low"]),
    }
    tools = [
        {
            "type": "function",
            "function": {"name": tool_name, "parameters": {"type": "object", "properties": {}}},
        }
    ]
    rendered = apply_chat_template_for_generation(
        tokenizer,
        [{"role": "user", "content": "hi"}],
        tools = tools,
        reasoning_effort = "high",
    )
    assert rendered.endswith(expected)


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
