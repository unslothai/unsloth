# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
"""Quotes break the Jinja literal and backslashes rewrite text, so {system_message} must be escaped."""

import jinja2
import pytest
from jinja2.sandbox import ImmutableSandboxedEnvironment

from unsloth.chat_templates import (
    CHAT_TEMPLATES,
    DEFAULT_SYSTEM_MESSAGE,
    _change_system_message,
    _escape_jinja_literal,
)


TEMPLATES_WITH_SYSTEM_MESSAGE = sorted(
    name
    for name, entry in CHAT_TEMPLATES.items()
    if isinstance(entry[0], str) and "{system_message}" in entry[0]
)

MESSAGES = [
    ("apostrophe", "Answer the user's question."),
    ("double", 'He said "hi" today.'),
    ("both_quotes", """mix ' and " here"""),
    ("latex", r"Put it in \boxed{\frac{1}{2}}."),
    ("windows", r"C:\Users\me"),
    ("trailing", "ends with a backslash \\"),
    ("crlf", "line one\r\nline two"),
    ("group_ref", r"use \1 for the group"),
    ("jinja", "{{ 7*6 }} and {% raw %}x{% endraw %}"),
]


def _render(template):
    environment = ImmutableSandboxedEnvironment(trim_blocks = True, lstrip_blocks = True)
    return environment.from_string(template).render(
        messages = [{"role": "user", "content": "Hi"}],
        bos_token = "<s>",
        eos_token = "</s>",
        add_generation_prompt = False,
    )


def test_templates_with_system_message_were_found():
    assert len(TEMPLATES_WITH_SYSTEM_MESSAGE) >= 10


@pytest.mark.parametrize("name", TEMPLATES_WITH_SYSTEM_MESSAGE)
@pytest.mark.parametrize("label, system_message", MESSAGES, ids = [m[0] for m in MESSAGES])
def test_system_message_survives_the_jinja_literal(name, label, system_message):
    template, used = _change_system_message(CHAT_TEMPLATES[name][0], name, system_message)
    assert used == system_message, "the returned message must be the raw one"
    assert system_message in _render(template)


@pytest.mark.parametrize("name", TEMPLATES_WITH_SYSTEM_MESSAGE)
def test_default_system_message_renders_verbatim(name):
    default = DEFAULT_SYSTEM_MESSAGE[name]
    template, _ = _change_system_message(CHAT_TEMPLATES[name][0], name, None)
    assert default in _render(template)


@pytest.mark.parametrize("name", ["vicuna", "vicuna_old", "vicuna old"])
def test_vicuna_default_has_a_plain_apostrophe(name):
    assert "\\" not in DEFAULT_SYSTEM_MESSAGE[name]
    assert "'s questions." in _render(
        _change_system_message(CHAT_TEMPLATES[name][0], name, None)[0]
    )


@pytest.mark.parametrize("quote", ["'", '"'])
@pytest.mark.parametrize("label, text", MESSAGES, ids = [m[0] for m in MESSAGES])
def test_escape_round_trips_in_either_quote_style(quote, label, text):
    # llama-3.1 uses "..." where others use '...', so one escaper must cover both.
    template = "{{ " + quote + _escape_jinja_literal(text) + quote + " }}"
    assert jinja2.Environment().from_string(template).render() == text
