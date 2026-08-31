# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A GGUF whose chat template uses numeric member access ("m.content.0.output", which
zai-org/GLM-5.3 and its re-quants ship) launches with a repaired copy: llama-server's
Jinja rejects that form, and the throw leaves replayed tool calls unrenderable.
"""

from __future__ import annotations

import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import core.inference.llama_cpp as llama_cpp  # noqa: E402
from core.inference.chat_template_helpers import repair_numeric_member_access  # noqa: E402


_GLM53_PROBE_BREAKER = "{%- if m.content and m.content.0.output is defined -%}1{%- endif -%}"


def test_the_probe_breaking_access_is_rewritten():
    assert repair_numeric_member_access(_GLM53_PROBE_BREAKER) == (
        "{%- if m.content and m.content[0].output is defined -%}1{%- endif -%}"
    )


def test_a_chained_lookup_keeps_the_rest_of_the_expression():
    assert repair_numeric_member_access("{{ entry.output.0.type == 'tool_reference' }}") == (
        "{{ entry.output[0].type == 'tool_reference' }}"
    )


def test_a_subscript_result_is_rewritten_too():
    assert repair_numeric_member_access("{{ messages[i].0.role }}") == "{{ messages[i][0].role }}"


def test_a_chain_of_numeric_lookups_is_rewritten_whole():
    # Repairing only the head leaves "[0].1", which llama-server still throws on.
    assert repair_numeric_member_access("{{ rows.0.1 }}") == "{{ rows[0][1] }}"
    assert repair_numeric_member_access("{{ a.0.b.10 }}") == "{{ a[0].b[10] }}"


def test_whitespace_around_the_dot_is_still_numeric_member_access():
    # llama.cpp's lexer skips the spaces and throws on "x . 0" exactly as on "x.0".
    assert repair_numeric_member_access("{{ messages[i] . 0.role }}") == "{{ messages[i][0].role }}"
    assert repair_numeric_member_access("{{ m.content .0.type }}") == "{{ m.content[0].type }}"
    assert repair_numeric_member_access("{{ rows . 0 . 1 }}") == "{{ rows[0][1] }}"


def test_every_site_in_one_template_is_rewritten():
    template = (
        "{{ m.content.0.type }}{{ tr.output.0.type }}"
        + _GLM53_PROBE_BREAKER
        + "{{ entry.output.0.type }}"
    )
    assert ".0" not in repair_numeric_member_access(template)


def test_a_raw_block_keeps_the_braces_it_prints():
    # {% raw %} emits its body verbatim, so rewriting there edits the prompt, not code.
    assert repair_numeric_member_access("{% raw %}{{ example.0 }}{% endraw %}") is None
    assert repair_numeric_member_access("{%- raw -%}{{ example.0 }}{%- endraw -%}") is None


def test_a_raw_body_is_literal_all_the_way_to_its_terminator():
    # Jinja interprets nothing inside raw, so the "{#" here is text and the {% endraw %}
    # it appears to wrap really does close the block.
    assert repair_numeric_member_access("{% raw %}{# {% endraw %} #}{{ m.content.0 }}") == (
        "{% raw %}{# {% endraw %} #}{{ m.content[0] }}"
    )


def test_a_raw_block_that_never_closes_repairs_nothing_after_it():
    assert repair_numeric_member_access("{% raw %}{{ e.0 }}") is None


def test_raw_tags_that_are_only_comment_text_do_not_open_a_raw_block():
    # Matching the tags in the source would read the middle expression as verbatim and
    # leave llama-server the numeric member it rejects.
    assert repair_numeric_member_access("{# {% raw %} #}{{ m.content.0 }}{# {% endraw %} #}") == (
        "{# {% raw %} #}{{ m.content[0] }}{# {% endraw %} #}"
    )


def test_a_real_site_outside_a_raw_block_is_still_repaired():
    assert (
        repair_numeric_member_access("{% raw %}{{ example.0 }}{% endraw %}{{ m.content.0.output }}")
        == "{% raw %}{{ example.0 }}{% endraw %}{{ m.content[0].output }}"
    )


def test_jinja_syntax_a_template_prints_as_an_example_is_left_alone():
    # The quoted "}}" ends the literal, not the expression, so example.0 is prompt text.
    assert repair_numeric_member_access('{{ "{{ example.0 }}" }}') is None
    assert repair_numeric_member_access("{{ '{% if x.0 %}' }}") is None


def test_a_literal_brace_does_not_hide_a_later_real_site():
    assert repair_numeric_member_access('{{ "a }} b" }}{{ m.content.0.type }}') == (
        '{{ "a }} b" }}{{ m.content[0].type }}'
    )


def test_a_comment_is_not_worth_a_relaunch():
    # Nothing in a comment is rendered, so repairing it would only force the model through
    # --chat-template-file for a template that needed nothing.
    assert repair_numeric_member_access("{# {{ a.0 }} #}") is None


def test_an_unterminated_block_is_not_treated_as_code():
    assert repair_numeric_member_access('{{ "unterminated.0 ') is None


def test_a_renderable_template_is_left_alone():
    clean = "{% for m in messages %}{{ m.content }}{% endfor %}"
    assert repair_numeric_member_access(clean) is None


def test_a_float_literal_is_not_indexing():
    assert repair_numeric_member_access("{% if temperature > 0.5 %}{{ 1.0 }}{% endif %}") is None


def test_quoted_text_the_template_prints_is_untouched():
    assert repair_numeric_member_access('{{ "see step.0 below" }}') is None


def test_prompt_text_outside_jinja_is_untouched():
    prose = "Rate this v1.0 release{{ m.content }}"
    assert repair_numeric_member_access(prose) is None


def test_empty_and_non_string_templates():
    assert repair_numeric_member_access("") is None
    assert repair_numeric_member_access(None) is None


def _backend(embedded):
    backend = llama_cpp.LlamaCppBackend.__new__(llama_cpp.LlamaCppBackend)
    backend._chat_template = embedded
    backend._model_identifier = "unsloth/GLM-5.3-GGUF"
    return backend


_BROKEN_EMBEDDED = "{% for m in messages %}" + _GLM53_PROBE_BREAKER + "{% endfor %}"


def test_a_broken_embedded_template_launches_as_a_repaired_copy():
    effective = _backend(_BROKEN_EMBEDDED)._effective_chat_template(None)
    assert "m.content[0].output" in effective
    assert "m.content.0.output" not in effective


def test_a_renderable_embedded_template_launches_untouched():
    assert _backend("{{ m.content }}")._effective_chat_template(None) is None


def test_an_explicit_override_wins_and_is_never_repaired():
    override = "{{ m.content.0.output }}"
    assert _backend(_BROKEN_EMBEDDED)._effective_chat_template(override) == override


def test_a_gguf_with_no_embedded_template_is_not_a_failure():
    assert _backend(None)._effective_chat_template(None) is None
