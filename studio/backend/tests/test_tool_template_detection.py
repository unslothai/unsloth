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
        # Accumulates the catalog into a namespace field inside the guard and renders
        # that field outside it, so nothing is emitted where the walk can see the
        # guard. The marker scan matched its `{%- if tools -%}` and this has to agree.
        ("lfm2-tool", True),
        # Names a `tools` variable but never renders anything derived from it.
        ("phi-4-mini", False),
    ],
)
def test_published_templates(name, detected):
    assert template_supports_tools(_published(name)) is detected


def test_lfm2_really_does_render_its_catalog():
    """Pins the ground truth rather than the detector. LiquidAI/LFM2-1.2B-Tool is a
    tool-calling model, and reporting it tool-less is the bug this file is about."""
    environment = Environment(extensions = ["jinja2.ext.loopcontrols", "jinja2.ext.do"])
    rendered = environment.from_string(_published("lfm2-tool")).render(
        tools = list(TOOLS),
        messages = [{"role": "user", "content": "what is the weather"}],
        bos_token = "<s>",
        add_generation_prompt = True,
    )
    assert "get_weather" in rendered


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
        # one. THUDM/glm-4-9b-chat reads `tools` off a message, inside the same
        # branch that then renders it, so it does not consume what Studio passes in.
        ("{% set tools = item['tools'] %}{{ tools|tojson }}", False),
        ("{% set tools = none %}{{ tools }}", False),
        ("{% set catalog = tools %}{% set catalog = [] %}{{ catalog|tojson }}", False),
        # A loop over the catalog hands each item to the loop variable, and a
        # namespace field written from one holds the catalog. LiquidAI's LFM2 builds
        # its whole tool block this way, outside any guard the walk can see.
        ("{% set ns = namespace(p='') %}{% for t in tools %}"
         "{% set ns.p = ns.p + (t|tojson) %}{% endfor %}{{ ns.p }}", True),
        ("{% set ns = namespace(c=none) %}{% set ns.c = tools %}{{ ns.c|tojson }}", True),
        ("{% set ns = namespace(c=none) %}{% set ns.c = messages %}{{ ns.c|tojson }}", False),
        ("{% set a, b = tools, none %}{{ a|tojson }}", True),
        # A rebinding inside a branch that may not run must not follow the walk out:
        # with the branch skipped the caller's catalog is still there and still
        # renders. The unconditional cases above stay False, which is what keeps
        # glm-4-9b-chat and granite-guardian's own rebindings meaningful.
        ("{% if legacy %}{% set tools = none %}{% endif %}{{ tools|tojson }}", True),
        ("{% if legacy %}{% set tools = none %}{% else %}{{ tools|tojson }}{% endif %}", True),
        ("{% if a %}{% if b %}{% set tools = none %}{% endif %}{% endif %}{{ tools|tojson }}", True),
        ("{% if a %}x{% elif b %}{% set tools = none %}{% endif %}{{ tools|tojson }}", True),
        ("{% for m in messages %}{% set tools = m.tools %}{% endfor %}{{ tools|tojson }}", True),
        ("{% macro unused() %}{% set tools = none %}{% endmacro %}{{ tools|tojson }}", True),
        # A rebinding on EVERY arm is still a rebinding: there is no path left where
        # the catalog survives, so an exhaustive chain has to kill it. Without this
        # the branch rule above would turn every `{% if %}/{% else %}` into a yes.
        ("{% if x %}{% set tools = none %}{% else %}{% set tools = none %}{% endif %}"
         "{{ tools|tojson }}", False),
        ("{% set c = tools %}{% if x %}{% set c = [] %}{% else %}{% set c = [] %}"
         "{% endif %}{{ c|tojson }}", False),
        ("{% if x %}{% set tools = none %}{% elif y %}{% set tools = none %}"
         "{% else %}{% set tools = none %}{% endif %}{{ tools|tojson }}", False),
        # One arm leaving it alone is a path where it survives.
        ("{% if x %}{% set tools = none %}{% else %}prose{% endif %}{{ tools|tojson }}", True),
        # The loop variable is undefined after `{% endfor %}`, so it must not carry
        # the catalog out with it.
        ("{% for t in tools %}{% endfor %}{{ t|tojson }}", False),
        ("{% for t in tools %}{% endfor %}"
         "{% for t in messages %}{{ t.content }}{% endfor %}", False),
        # A tuple target binds and rebinds the same way a plain name does. Reading it
        # only one way round would let `{% set tools, flag = none, false %}` keep the
        # catalog it just threw away.
        ("{% set tools, flag = none, false %}{{ tools|tojson }}", False),
        ("{% set a, b = tools, none %}{% set a, b = none, none %}{{ a|tojson }}", False),
        # A branch that only runs when tools exist stores tool-conditional content,
        # whatever the value is. The marker scan matched this shape, so a
        # catalog-only rule would lose tool support the old code had.
        ("{% set intro = '' %}{% if tools %}{% set intro = 'You may call functions.' %}"
         "{% endif %}{{ intro }}", True),
        ("{% set intro = '' %}{% if enable_thinking %}{% set intro = 'Think.' %}"
         "{% endif %}{{ intro }}", False),
        # The tool-role check can sit inside the output expression itself, and a
        # tools-gated branch can store its instructions through a side effect. The
        # marker scan matched both spellings.
        ("{{ message.content if message.role == 'tool' else '' }}", True),
        ("{{ message.content if message.role != 'tool' else '' }}", False),
        ("{{ message.content if message.role == 'user' else '' }}", False),
        ("{% set ns = namespace(lines=[]) %}{% if tools %}"
         "{% do ns.lines.append('You may call functions') %}{% endif %}"
         "{{ ns.lines|join(',') }}", True),
        ("{% set ns = namespace(lines=[]) %}{% if enable_thinking %}"
         "{% do ns.lines.append('Think') %}{% endif %}{{ ns.lines|join(',') }}", False),
        # A loop's inline filter is a guard like any other. The marker scan matched
        # this spelling of the tool-role check, so missing it loses tool support.
        ("{% for m in messages if m.role == 'tool' %}{{ m.content }}{% endfor %}", True),
        ('{% for m in messages if m["role"] == "tool" %}{{ m.content }}{% endfor %}', True),
        ("{% for m in messages if m.role != 'tool' %}{{ m.content }}{% endfor %}", False),
        ('{% for m in messages if m.role == "user" %}{{ m.content }}{% endfor %}', False),
        # `{% with %}` binds for its block and for nothing else.
        ("{% with catalog = tools %}{{ catalog|tojson }}{% endwith %}", True),
        ("{% with tools = none %}{{ tools|tojson }}{% endwith %}", False),
        ("{% with tools = none %}x{% endwith %}{{ tools|tojson }}", True),
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
        "{% set ns = namespace(p='') %}{% for t in tools %}"
        "{% set ns.p = ns.p + (t|tojson) %}{% endfor %}{{ ns.p }}",
        "{% set ns = namespace(c=none) %}{% set ns.c = tools %}{{ ns.c|tojson }}",
        "{% set a, b = tools, none %}{{ a|tojson }}",
        "{% if legacy %}{% set tools = none %}{% endif %}{{ tools|tojson }}",
        "{% macro unused() %}{% set tools = none %}{% endmacro %}{{ tools|tojson }}",
        "{% with catalog = tools %}{{ catalog|tojson }}{% endwith %}",
        "{% with tools = none %}x{% endwith %}{{ tools|tojson }}",
    ],
)
def test_positives_match_a_real_render(template):
    """Every case claimed positive here really does emit a tool name."""
    assert _renders_catalog(template, item = {}, message = {}, messages = [], legacy = False)
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


def test_a_string_subclass_cannot_escape_the_fail_closed_branch():
    """`isinstance` lets a subclass through, and then its own `__hash__` and
    `__contains__` run before the try: the cache hashes the argument and the early-out
    does `"tool" not in template`. Narrowing to str is what keeps those from reaching
    a caller that has no except around the model load."""

    class NoHash(str):
        __hash__ = None

    class BadContains(str):
        def __contains__(self, item):
            raise ValueError("injected")

    template = "{% if tools %}{{ tools|tojson }}{% endif %}"
    assert template_supports_tools(NoHash(template)) is True
    assert template_supports_tools(BadContains(template)) is True


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
