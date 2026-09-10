# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tool detection reads Jinja syntax, not the spelling of it.

The detector is an over-approximation on purpose, so the tests are split three ways:
what it must get right, what it deliberately answers loosely, and what it must never
do (raise, or disagree with itself over line endings).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from jinja2 import Environment

_backend_root = Path(__file__).resolve().parent.parent
if str(_backend_root) not in sys.path:
    sys.path.insert(0, str(_backend_root))

from core.inference.template_capabilities import (  # noqa: E402
    template_supports_tools,
)

TOOLS = [
    {"type": "function", "function": {"name": "get_weather", "parameters": {}}},
    {"type": "function", "function": {"name": "get_time", "parameters": {}}},
]


def _published(name):
    return (Path(__file__).parent / "data" / "chat_templates" / f"{name}.jinja").read_text(
        encoding = "utf-8"
    )


def _renders_catalog(template, **context):
    """Ground truth: does rendering this actually put a tool name in the output?"""
    environment = Environment(extensions = ["jinja2.ext.loopcontrols", "jinja2.ext.do"])
    output = environment.from_string(template).render(tools = list(TOOLS), **context)
    return "get_weather" in output


# ── the published templates this exists for ─────────────────────────────────
@pytest.mark.parametrize(
    ("name", "detected"),
    [
        # Aliases the catalog and renders the alias. The old substring scan matched
        # none of its spellings and reported it tool-less, which is the bug.
        ("granite-3.3", True),
        # Names a `tools` variable but never renders anything derived from it.
        ("phi-4-mini", False),
    ],
)
def test_published_templates(name, detected):
    assert template_supports_tools(_published(name)) is detected


def test_granite_really_does_render_its_catalog():
    """Pins the ground truth rather than the detector, so this test still means
    something if the detector changes."""
    environment = Environment(extensions = ["jinja2.ext.loopcontrols", "jinja2.ext.do"])
    rendered = environment.from_string(_published("granite-3.3")).render(
        tools = list(TOOLS),
        messages = [{"role": "user", "content": "what is the weather"}],
        documents = [],
        controls = {},
        thinking = False,
        add_generation_prompt = True,
        strftime_now = lambda fmt: "January 01, 2026",
    )
    assert "get_weather" in rendered


# ── the spellings a guard comes in ──────────────────────────────────────────
@pytest.mark.parametrize(
    ("template", "detected"),
    [
        ("{% if tools %}{{ tools|tojson }}{% endif %}", True),
        ("{%- if tools -%}{{ tools|tojson }}{%- endif -%}", True),
        ("{% if tools is defined and tools %}{{ tools|tojson }}{% endif %}", True),
        ("{% if tools and not available_tools %}{{ tools|tojson }}{% endif %}", True),
        # A guarded branch advertises tools even when its body is only prose.
        ("{% if tools %}You may call the tools listed above.{% endif %}", True),
        # Tool results coming back from the model.
        ("{% for m in messages %}{{ m.tool_calls|tojson }}{% endfor %}", True),
        ("{% if message.role == 'tool' %}{{ message.content }}{% endif %}", True),
        ("{% if message['role'] == 'tool' %}{{ message.content }}{% endif %}", True),
        # Nothing to do with tools.
        ("{% for m in messages %}{{ m.content }}{% endfor %}", False),
        ("{% if enable_thinking %}<think>{% endif %}", False),
        ("", False),
        ("plain text, no jinja at all", False),
    ],
)
def test_guard_spellings(template, detected):
    assert template_supports_tools(template) is detected


# ── carrying the catalog through names ──────────────────────────────────────
@pytest.mark.parametrize(
    ("template", "detected"),
    [
        ("{% set catalog = tools %}{{ catalog|tojson }}", True),
        ("{% set a = tools %}{% set b = a %}{{ b|tojson }}", True),
        ("{% if tools %}{% set c = tools %}{% endif %}{{ c|tojson }}", True),
        ("{% set catalog = tools %}{{ catalog|length }}", False),
        # A name rebound to something that is not the caller's catalog stops being
        # one. THUDM/glm-4-9b-chat reads `tools` off a message, and
        # ibm-granite/granite-guardian-3.1-8b sets it to none; neither consumes the
        # catalog Studio would pass in.
        ("{% set tools = item['tools'] %}{{ tools|tojson }}", False),
        ("{% set tools = none %}{{ tools }}", False),
        ("{% set catalog = tools %}{% set catalog = [] %}{{ catalog|tojson }}", False),
        # Handing the catalog to a container puts it in that container.
        ("{% set c = [] %}{% do c.append(tools) %}{{ c|tojson }}", True),
        ("{% set c = [] %}{% set _ = c.append(tools) %}{{ c|tojson }}", True),
        ("{% set ns = namespace(c=[]) %}{% do ns.c.extend(tools) %}{{ ns.c|tojson }}", True),
        ("{% set c = [] %}{% do c.append(messages) %}{{ c|tojson }}", False),
    ],
)
def test_names_carrying_the_catalog(template, detected):
    assert template_supports_tools(template) is detected


@pytest.mark.parametrize(
    "template",
    [
        "{% set catalog = tools %}{{ catalog|tojson }}",
        "{% set a = tools %}{% set b = a %}{{ b|tojson }}",
        "{% if tools %}{{ tools|tojson }}{% endif %}",
        "{% for t in tools %}{{ t|tojson }}{% endfor %}",
        "{% set c = [] %}{% do c.append(tools) %}{{ c|tojson }}",
    ],
)
def test_positives_match_a_real_render(template):
    """Every case claimed positive here really does emit a tool name."""
    assert _renders_catalog(template, item = {}, message = {}, messages = [])
    assert template_supports_tools(template) is True


# ── deliberate over-approximation ───────────────────────────────────────────
@pytest.mark.parametrize(
    "template",
    [
        # A branch that can never run.
        "{% if false %}{{ tools|tojson }}{% endif %}",
        # A catalog put somewhere and then taken back out again.
        "{% set c = [] %}{% do c.append(tools) %}{% do c.clear() %}{{ c|tojson }}",
        # Iterating a mapping yields its keys, not the catalog under them.
        "{% for key in {'weather': tools} %}{{ key }}{% endfor %}",
    ],
)
def test_known_over_approximations_answer_yes(template):
    """These render no schema, and the detector says yes anyway.

    A value handed to a container is assumed to stay there, a branch is walked whether
    or not it can run, and a mapping's values are not told apart from its keys.
    Tracking any of that properly needs a much larger analysis, and none of it is
    reachable from the 120 published templates checked. The error runs towards showing
    a tool control that the backend then re-checks, rather than hiding one that works,
    so it is left here on purpose rather than papered over.
    """
    assert not _renders_catalog(template, item = {}, message = {}, messages = [])
    assert template_supports_tools(template) is True


def test_a_guarded_branch_counts_even_without_naming_the_catalog():
    """`{% if tools %}<prose>{% endif %}` renders no schema but is still telling the
    model it has tools, so it counts. Asserted because it looks like a false positive
    and is not one."""
    template = "{% if tools %}You may call tools.{% endif %}"
    assert not _renders_catalog(template)
    assert template_supports_tools(template) is True


# ── robustness ──────────────────────────────────────────────────────────────
@pytest.mark.parametrize(
    "template",
    [
        "{% if tools %}",  # unterminated
        "{{ tools",  # unterminated expression
        "{%",
        "{% if tools %}" * 400 + "x" + "{% endif %}" * 400,  # deeply nested
        "{{ " + "(" * 300 + "tools" + ")" * 300 + " }}",  # deeply nested expression
        "{{ tools|tojson }}" * 5000,  # very long
        "{% unknown_tag %}{{ tools|tojson }}",
    ],
)
def test_hostile_templates_never_raise(template):
    """This runs on the GGUF metadata read and the llama-server launch, so it must
    return a verdict rather than stop a model from loading."""
    assert template_supports_tools(template) in (True, False)


@pytest.mark.parametrize("template", [None, 123, b"bytes", {"default": "x"}, [{"name": "x"}]])
def test_non_string_templates_are_turned_away(template):
    """Hugging Face ships named-template maps and Hermes-style lists. These reach the
    detector unhashable, so the type check has to sit outside the cache."""
    assert template_supports_tools(template) is False


@pytest.mark.parametrize(
    "template",
    [
        "{% if tools %}{{ tools|tojson }}{% endif %}",
        "{% set catalog = tools %}{{ catalog|tojson }}",
        "{% for m in messages %}{{ m.content }}{% endfor %}",
    ],
)
def test_line_endings_and_byte_order_marks_do_not_change_the_verdict(template):
    """A Windows checkout and a file saved with a BOM have to agree with everyone
    else, which is the whole point of reading syntax rather than bytes."""
    base = template_supports_tools(template)
    assert template_supports_tools(template.replace("\n", "\r\n")) is base
    assert template_supports_tools("﻿" + template) is base


def test_the_generation_tag_still_parses():
    """`{% generation %}` marks the assistant span for training masks and is not a
    real Jinja tag. Without the extension the template fails to parse and the
    fail-closed branch would turn tools off. HuggingFaceTB/SmolLM3-3B needs this."""
    template = (
        "{% if tools %}{{ tools|tojson }}{% endif %}"
        "{% generation %}{{ message.content }}{% endgeneration %}"
    )
    assert template_supports_tools(template) is True


def test_unrelated_conditions_do_not_degrade():
    """Nothing here forks a state per condition, so a template with many unrelated
    guards costs no more than a linear walk. Qwen3-Coder-30B is the reason to keep
    checking: it is long enough that any per-branch blow-up would reach it."""
    template = (
        "".join("{%% if flag%d %%}plain{%% endif %%}" % index for index in range(300))
        + "{% if tools %}{{ tools|tojson }}{% endif %}"
    )
    assert template_supports_tools(template) is True


def test_the_cache_is_keyed_on_the_template_text():
    template = "{% if tools %}{{ tools|tojson }}{% endif %}"
    template_supports_tools.cache_clear()
    assert template_supports_tools(template) is True
    assert template_supports_tools("".join([template])) is True
    assert template_supports_tools.cache_info().hits >= 1
