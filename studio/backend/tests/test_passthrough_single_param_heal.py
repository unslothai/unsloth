# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A low-bit quant of a model that writes <function=NAME><parameter=KEY> calls often keeps the
function and the value but garbles the parameter opener. For a declared tool with exactly one
required string parameter the value has one place to go, so that shape is repaired; everything
else, and every model whose template does not teach the format, keeps the behaviour it had."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

import core.inference.passthrough_healing as ph  # noqa: E402
from core.inference.passthrough_healing import (  # noqa: E402
    HealTools,
    StreamToolCallHealer,
    backend_chat_templates,
    heal_gate,
    heal_openai_message,
    heal_openai_message_events,
    nudge_should_retry,
    single_param_heal_enabled,
)

# The part of such a template that matters: it spells the call format out for the model.
XML_TEMPLATE = (
    "{%- if tools %}reply in the following format:\n<tool_call>\n<function=example_function_name>\n"
    "<parameter=example_parameter_1>\nvalue_1\n</parameter>\n</function>\n</tool_call>{%- endif %}"
)
JSON_TEMPLATE = (
    '{%- if tools %}<tool_call>\n{"name": <function-name>, "arguments": <args-json-object>}\n'
    "</tool_call>{%- endif %}"
)


def _tool(name, properties, required):
    return {
        "type": "function",
        "function": {
            "name": name,
            "parameters": {"type": "object", "properties": properties, "required": required},
        },
    }


BASH = _tool("bash", {"command": {"type": "string"}, "timeout": {"type": "integer"}}, ["command"])
FETCH = _tool("fetch", {"method": {"type": "string"}, "url": {"type": "string"}}, ["method", "url"])
SLEEP = _tool("sleep", {"seconds": {"type": "integer"}}, ["seconds"])
TOOLS = [BASH, FETCH, SLEEP]


def _gate(
    tools = TOOLS,
    template = XML_TEMPLATE,
    **kwargs,
):
    return heal_gate(None, tools, chat_template = template, **kwargs)


def _heal(
    text,
    *,
    tools = TOOLS,
    template = XML_TEMPLATE,
    allow_incomplete = True,
):
    msg = {"role": "assistant", "content": text}
    healed = heal_openai_message(msg, _gate(tools, template), tools, allow_incomplete)
    return healed, msg


def _calls(msg):
    return [
        (call["function"]["name"], json.loads(call["function"]["arguments"]))
        for call in msg.get("tool_calls") or []
    ]


def _events(
    text,
    *,
    tools = TOOLS,
    template = XML_TEMPLATE,
    allow_incomplete = True,
):
    msg = {"role": "assistant", "content": text}
    return heal_openai_message_events(msg, _gate(tools, template), tools, allow_incomplete)


def _stream(
    text,
    size,
    *,
    tools = TOOLS,
    template = XML_TEMPLATE,
):
    healer = StreamToolCallHealer(_gate(tools, template), tools)
    events = []
    for start in range(0, len(text), size):
        events.extend(healer.feed(text[start : start + size]))
    events.extend(healer.finalize())
    calls = [
        (value["function"]["name"], json.loads(value["function"]["arguments"]))
        for kind, value in events
        if kind == "tool_call"
    ]
    return calls, "".join(value for kind, value in events if kind == "text")


def _block(
    header,
    opener,
    value = "ls -la",
):
    lines = ["<tool_call>", header, *([opener] if opener else []), value]
    return "\n".join([*lines, "</parameter>", "</function>", "</tool_call>"])


# Each is one garbled shape with the standard pieces around it left intact.
GARBLED = [
    pytest.param(_block("<function=bash>", "<=command>"), id = "opener_equals_key"),
    pytest.param(_block("<function=bash>", "<command>"), id = "opener_bare_tag"),
    pytest.param(_block("<function=bash>", "command:"), id = "opener_key_colon"),
    pytest.param(_block("<function=bash>", "command"), id = "opener_key_word"),
    pytest.param(_block("<function=bash>", "=command>"), id = "opener_cut_front"),
    pytest.param(_block("<function=bash>", "command>"), id = "opener_cut_front_bare"),
    pytest.param(_block("<function=bash>", "<function=command>"), id = "opener_function_key"),
    pytest.param(_block("<function=bash>", "<parameter>command>"), id = "opener_parameter_gt"),
    pytest.param(_block("<function=bash>", "<parameter=bash>"), id = "opener_parameter_tool_name"),
    pytest.param(_block("<function=bash>", "<bash>"), id = "opener_tool_name_tag"),
    pytest.param(_block("<function=bash>", None), id = "no_opener"),
    pytest.param(
        _block("<function=bash>", "<parameter=bash>\n<parameter=command>"),
        id = "duplicated_parameter_tool_name",
    ),
    pytest.param(
        _block("<function=bash>", "<function=bash>\n<parameter=command>"), id = "duplicated_header"
    ),
    pytest.param(_block("<function=bash=", "<parameter=command>"), id = "header_trailing_equals"),
    pytest.param(_block("<function=bash:", "<parameter=command>"), id = "header_trailing_colon"),
    pytest.param(_block("<function=bash", "<parameter=command>"), id = "header_unclosed"),
    pytest.param(_block("<=bash>", "<parameter=command>"), id = "header_equals_name"),
    pytest.param(_block("<bash>", "<parameter=command>"), id = "header_bare_name"),
    pytest.param(_block("<functionbash>", "<parameter=command>"), id = "header_no_equals"),
    pytest.param(_block("<function=bash=command>", None), id = "header_with_key"),
    pytest.param(_block("<function=bash:command>", None), id = "header_with_key_colon"),
    pytest.param(_block("<parameter=bash>", "<parameter=command>"), id = "header_as_parameter"),
    pytest.param(_block("<function=bash=", "<command>"), id = "header_and_opener_garbled"),
    pytest.param(_block("<function=bash", "\n<parameter=command>"), id = "blank_line_before_opener"),
]


class TestGarbledOpenerIsRepaired:
    @pytest.mark.parametrize("text", GARBLED)
    def test_message(self, text):
        healed, msg = _heal(text)
        assert healed
        assert _calls(msg) == [("bash", {"command": "ls -la"})]
        assert msg["content"] is None

    @pytest.mark.parametrize("text", GARBLED)
    def test_truncation_safe_mode_takes_a_closed_block(self, text):
        healed, msg = _heal(text, allow_incomplete = False)
        assert healed
        assert _calls(msg) == [("bash", {"command": "ls -la"})]

    @pytest.mark.parametrize("text", GARBLED)
    @pytest.mark.parametrize("size", [1, 7, 10_000])
    def test_stream(self, text, size):
        calls, relayed = _stream(text, size)
        assert calls == [("bash", {"command": "ls -la"})]
        assert relayed.strip() == ""

    def test_multiline_value_is_kept_verbatim(self):
        value = "grep -n foo bar.py\n  echo done\n\ncat <<'EOF'\nx = 1\nEOF"
        healed, msg = _heal(_block("<function=bash>", "<command>", value))
        assert healed
        assert _calls(msg) == [("bash", {"command": value})]

    def test_prose_around_the_block_is_relayed(self):
        text = "Listing it.\n\n" + _block("<function=bash>", "<=command>") + "\nDone."
        healed, msg = _heal(text)
        assert _calls(msg) == [("bash", {"command": "ls -la"})]
        assert msg["content"] == "Listing it.\n\n\nDone."

    def test_only_the_garbled_call_of_several_changes(self):
        good = (
            "<tool_call>\n<function=bash>\n<parameter=command>\ngrep -n foo bar.py\n</parameter>\n"
            "</function>\n</tool_call>"
        )
        text = good + "\n" + _block("<function=bash>", "<command>") + "\n" + good
        healed, msg = _heal(text)
        assert healed
        assert _calls(msg) == [
            ("bash", {"command": "grep -n foo bar.py"}),
            ("bash", {"command": "ls -la"}),
            ("bash", {"command": "grep -n foo bar.py"}),
        ]
        assert [call["id"] for call in msg["tool_calls"]] == ["call_0", "call_1", "call_2"]

    def test_unwrapped_block_with_a_plain_header(self):
        text = "<function=bash>\n<=command>\nls -la\n</parameter>\n</function>"
        healed, msg = _heal(text)
        assert healed
        assert _calls(msg) == [("bash", {"command": "ls -la"})]

    def test_wrapper_left_open_keeps_the_wrapper_text_like_a_normal_call(self):
        garbled = "<tool_call>\n<function=bash>\n<=command>\nls -la\n</parameter>\n</function>"
        normal = (
            "<tool_call>\n<function=bash>\n<parameter=command>\nls -la\n</parameter>\n</function>"
        )
        assert _heal(garbled)[1] == _heal(normal)[1]

    def test_the_repair_spares_a_nudge_retry(self):
        text = _block("<function=bash=", "<parameter=command>")
        data = {"choices": [{"message": {"role": "assistant", "content": text}}]}
        assert nudge_should_retry(data, heal_gate(None, TOOLS), TOOLS)
        assert not nudge_should_retry(data, _gate(), TOOLS)


def _unchanged(
    text,
    *,
    tools = TOOLS,
    allow_incomplete = True,
):
    """The rule on must behave exactly as the rule off: message events and every stream chunking."""
    on = _events(text, tools = tools, allow_incomplete = allow_incomplete)
    off = _events(text, tools = tools, template = None, allow_incomplete = allow_incomplete)
    assert on == off
    for size in (1, 7, 10_000):
        assert _stream(text, size, tools = tools) == _stream(text, size, tools = tools, template = None)
    return on


class TestRejectedRepairsKeepTheOldBehaviour:
    def test_think_close_and_a_second_wrapper_are_not_pulled_into_the_value(self):
        # The false repair a looser rule made: the first block never gets its value, and what
        # follows it is reasoning markup and another call, not an argument.
        text = (
            "<tool_call>\n<function=bash>\n</think>\n\n<tool_call>\n<function=bash>\n"
            "<parameter=command>\nls -la\n</parameter>\n</function>\n</tool_call>"
        )
        events = _unchanged(text)
        arguments = [json.loads(v["function"]["arguments"]) for k, v in events if k == "tool_call"]
        assert all(
            "think" not in json.dumps(a) and "tool_call" not in json.dumps(a) for a in arguments
        )

    @pytest.mark.parametrize(
        "markup",
        [
            "<tool_call>",
            "</tool_call>",
            "<function=bash>",
            "</function>",
            "<parameter=command>",
            "</parameter>",
            "<think>",
            "</think>",
            "[TOOL_CALLS]",
            "<|tool_call>",
            "<|content_invoke_tool_json|>",
            "<|im_end|>",
            "<tool_response>",
        ],
    )
    def test_markup_inside_the_value(self, markup):
        _unchanged(_block("<function=bash>", "<=command>", f"echo a\n{markup}\necho b"))

    @pytest.mark.parametrize(
        "value",
        ["", " ", "\n\n", "\t\n"],
    )
    def test_empty_value(self, value):
        _unchanged(_block("<function=bash>", "<=command>", value))

    @pytest.mark.parametrize(
        "text",
        [
            "<tool_call>",
            "<tool_call>\n</tool_call>",
            "<tool_call>\n<function=bash>\n</function>\n</tool_call>",
            "<tool_call>\n<function=bash>\n<command>\n</parameter>\n</function>\n</tool_call>",
            "<tool_call>\n<function=bash>\ncommand\n</parameter>\n</function>\n</tool_call>",
            "<tool_call>\n<function=bash>\nbash\n</parameter>\n</function>\n</tool_call>",
        ],
    )
    def test_empty_body(self, text):
        _unchanged(text)

    def test_two_parameter_closers(self):
        _unchanged(
            "<tool_call>\n<function=bash>\n<=command>\nls -la\n</parameter>\n5\n</parameter>\n"
            "</function>\n</tool_call>"
        )

    def test_no_parameter_closer(self):
        _unchanged("<tool_call>\n<function=bash>\n<=command>\nls -la\n</function>\n</tool_call>")

    def test_no_function_closer(self):
        _unchanged("<tool_call>\n<function=bash>\n<=command>\nls -la\n</parameter>\n</tool_call>")

    def test_text_between_the_closers(self):
        _unchanged(
            "<tool_call>\n<function=bash>\n<=command>\nls -la\n</parameter>\nnote\n</function>\n"
            "</tool_call>"
        )

    def test_two_required_parameters(self):
        _unchanged(_block("<function=fetch>", "<=url>", "http://localhost/x"))

    def test_non_string_parameter(self):
        _unchanged(_block("<function=sleep>", "<=seconds>", "5"))

    @pytest.mark.parametrize(
        "prop",
        [{}, {"type": "integer"}, {"type": "array"}, {"type": "string", "enum": ["a", "b"]}, True],
    )
    def test_parameter_not_a_free_string(self, prop):
        tools = [_tool("bash", {"command": prop}, ["command"])]
        _unchanged(_block("<function=bash>", "<=command>"), tools = tools)

    def test_undeclared_tool(self):
        _unchanged(_block("<function=shell>", "<=command>"))
        _unchanged(_block("<function=shell=", "<parameter=command>"))

    def test_tool_excluded_by_tool_choice(self):
        choice = {"type": "function", "function": {"name": "fetch"}}
        text = _block("<function=bash>", "<=command>")
        msg_on = {"role": "assistant", "content": text}
        msg_off = {"role": "assistant", "content": text}
        on = heal_gate(None, TOOLS, choice, chat_template = XML_TEMPLATE)
        assert on == {"fetch"} and on.single_string_param
        heal_openai_message(msg_on, on, TOOLS)
        heal_openai_message(msg_off, heal_gate(None, TOOLS, choice), TOOLS)
        assert msg_on == msg_off

    def test_key_naming_another_tool_is_ambiguous(self):
        tools = [BASH, _tool("command", {"line": {"type": "string"}}, ["line"])]
        _unchanged(_block("<function=bash>", "<function=command>"), tools = tools)

    def test_tool_name_naming_another_parameter_is_ambiguous(self):
        tools = [
            _tool("bash", {"command": {"type": "string"}, "bash": {"type": "string"}}, ["command"])
        ]
        _unchanged(_block("<function=bash>", "<parameter=bash>"), tools = tools)

    @pytest.mark.parametrize(
        "opener",
        [
            "<ash>",
            "<",
            "<=",
            "=",
            "<parameter=",
            "<function>",
            "<parameter>",
            "ommand>",
            "ommand:",
            "function",
        ],
    )
    def test_opener_outside_the_allowlist(self, opener):
        _unchanged(_block("<function=bash>", opener))

    def test_two_garbled_opener_lines(self):
        _unchanged(_block("<function=bash>", "<function=bash>\n<parameter=bash>"))

    def test_abandoned_header_owns_the_text_up_to_its_closer(self):
        # The second block cannot be told apart from text of the first, which never closed.
        _unchanged("<tool_call>\n<function=bash>\nnote\n" + _block("<function=bash>", "<=command>"))
        _unchanged(
            "<function=bash>\n<=command>\necho a\n<function=bash>\necho b\n</parameter>\n</function>"
        )

    def test_first_line_that_only_looks_like_a_label_is_kept(self):
        python = _tool("python", {"code": {"type": "string"}}, ["code"])
        code = "try:\n    x = 1\nexcept ValueError:\n    x = 2"
        text = (
            "<tool_call>\n<function=python>\n<=code>\n"
            + code
            + "\n</parameter>\n</function>\n</tool_call>"
        )
        healed, msg = _heal(text, tools = [python])
        assert healed
        assert _calls(msg) == [("python", {"code": code})]

    @pytest.mark.parametrize(
        "value",
        [
            "<ls -la",
            "=ls -la",
            "> out.txt",
            "/bash -c ls",
            'command="ls -la"',
            "command: ls -la",
            "command=ls -la",
            '{"command": "ls -la"}',
        ],
    )
    def test_value_opening_with_opener_residue(self, value):
        _unchanged(_block("<function=bash=", None, value))

    def test_garbled_header_needs_the_wrapper(self):
        _unchanged("<function=bash=\n<parameter=command>\nls -la\n</parameter>\n</function>")
        _unchanged("<bash>\n<parameter=command>\nls -la\n</parameter>\n</function>")

    def test_block_inside_reasoning(self):
        text = "<think>\n" + _block("<function=bash>", "<=command>") + "\n</think>\nDone."
        assert _events(text) is None
        assert _events(text, allow_incomplete = False) is None
        # A stream has relayed the <think> by then; the repair goes exactly as far as a well-formed
        # call does there, no further.
        normal = (
            "<think>\n" + _block("<function=bash>", "<parameter=command>") + "\n</think>\nDone."
        )
        for size in (1, 7, 10_000):
            assert _stream(text, size) == _stream(normal, size)

    def test_block_quoted_in_another_call_argument(self):
        writer = _tool("write", {"content": {"type": "string"}}, ["content"])
        text = (
            "<tool_call>\n<function=write>\n<parameter=content>\nexample:\n"
            + _block("<function=bash>", "<=command>")
            + "\n</parameter>\n</function>\n</tool_call>"
        )
        _unchanged(text, tools = [BASH, writer])

    def test_block_cut_off_by_the_token_limit(self):
        text = "<tool_call>\n<function=bash>\n<=command>\nls -la\n</parameter>"
        _unchanged(text, allow_incomplete = False)
        _unchanged(text)

    @pytest.mark.parametrize("text", GARBLED)
    def test_template_without_the_format(self, text):
        on = _events(text, template = JSON_TEMPLATE)
        assert on == _events(text, template = None)


class TestHealthyAndOtherFormatsAreUntouched:
    VALID = [
        "<tool_call>\n<function=bash>\n<parameter=command>\nls -la\n</parameter>\n</function>\n</tool_call>",
        "<function=bash>\n<parameter=command>\ngrep -n foo bar.py\n</parameter>\n</function>",
        "<tool_call>\n<function=bash>\n<parameter=command>\nls -la\n</parameter>\n"
        "<parameter=timeout>\n5\n</parameter>\n</function>\n</tool_call>",
        "<tool_call>\n<function=bash>\n<parameter=timeout>\n5\n</parameter>\n"
        "<parameter=command>\nls -la\n</parameter>\n</function>\n</tool_call>",
        "<tool_call>\n<function=fetch>\n<parameter=method>\nGET\n</parameter>\n<parameter=url>\n"
        "http://localhost/x\n</parameter>\n</function>\n</tool_call>",
        '<tool_call>\n<function=bash>\n{"command": "ls -la"}\n</function>\n</tool_call>',
        # The wrapper never closes: the call still parses on its own.
        "<tool_call>\n<function=bash>\n<parameter=command>\nls -la\n</parameter>\n</function>",
        "<tool_call>\n\n<function=bash>\n<parameter=command>\nls -la\n</parameter>\n</function>\nDone.",
    ]
    PROSE = [
        "Plain prose that mentions `<tool_call>` and nothing else.",
        "The format is <tool_call> then <function=bash> then the command, closed by </function>.",
        "Use `<function=bash>` with a `command`.",
    ]
    OTHER_FORMATS = [
        '<tool_call>\n{"name": "bash", "arguments": {"command": "ls -la"}}\n</tool_call>',
        '<tool_call>\n{"name": "bash", "arguments": {}}\n</tool_call>',
        '<tool_call>{"name": "bash", "arguments": "ls -la"}</tool_call>',
        '<|tool_call>call:bash{command:<|"|>ls -la<|"|>}<tool_call|>',
        "<|tool_call>call:bash{}<tool_call|>",
        '[TOOL_CALLS]bash[ARGS]{"command": "ls -la"}',
        "[TOOL_CALLS]bash[ARGS]{}",
        '[TOOL_CALLS][{"name": "bash", "arguments": {}}]',
        'bash<|content_invoke_tool_json|>{"name": "bash", "args": {}}<|end_message|>',
    ]

    @pytest.mark.parametrize("text", VALID + PROSE + OTHER_FORMATS)
    def test_events_identical(self, text):
        _unchanged(text)

    @pytest.mark.parametrize("text", VALID)
    def test_valid_call_never_reaches_the_repair(self, text, monkeypatch):
        def refuse(*_args, **_kwargs):
            raise AssertionError("a healthy message must not be scanned for a repair")

        monkeypatch.setattr(ph, "_single_param_block", refuse)
        msg = {"role": "assistant", "content": text}
        before = json.dumps(_events(text, template = None))
        assert json.dumps(heal_openai_message_events(msg, _gate(), TOOLS)) == before

    @pytest.mark.parametrize("text", VALID)
    @pytest.mark.parametrize("size", [1, 7, 10_000])
    def test_valid_stream_is_never_reparsed_for_a_repair(self, text, size, monkeypatch):
        expected = _stream(text, size, template = None)
        inner = ph.parse_tool_calls_from_text
        seen = []

        def counting(content, **kwargs):
            seen.append(content)
            return inner(content, **kwargs)

        monkeypatch.setattr(ph, "parse_tool_calls_from_text", counting)
        _stream(text, size, template = None)
        plain = len(seen)
        del seen[:]
        assert _stream(text, size) == expected
        assert len(seen) == plain

    def test_well_formed_block_is_not_a_repair_even_when_scanned(self):
        # A garbled neighbour makes the scan run; the healthy block is left to the normal parser.
        good = (
            "<tool_call>\n<function=bash>\n<parameter=command>\nls -la\n</parameter>\n</function>"
        )
        text = good + "\n" + _block("<function=bash>", "<command>", "grep -n foo bar.py")
        parsed, spans = ph.parse_tool_calls_from_text(text, allow_incomplete = True, with_spans = True)
        repaired = ph._repair_single_string_params(
            text, parsed, spans, _gate(), ph._tool_schemas_by_name(TOOLS)
        )
        assert repaired is not None
        assert [text[s:e] for s, e in repaired[2]] == [
            _block("<function=bash>", "<command>", "grep -n foo bar.py")
        ]

    def test_native_format_path_is_not_entered(self, monkeypatch):
        def refuse(*_args, **_kwargs):
            raise AssertionError("a native format is healed by its own parser")

        monkeypatch.setattr(ph, "_repair_single_string_params", refuse)
        native = ph.native_tool_format("<minimax:tool_call>")
        msg = {"role": "assistant", "content": _block("<function=bash>", "<=command>")}
        assert heal_openai_message_events(msg, _gate(), TOOLS, native = native) is None


class TestGate:
    def test_xml_template_turns_the_rule_on(self):
        names = _gate()
        assert names == {"bash", "fetch", "sleep"}
        assert isinstance(names, HealTools) and names.single_string_param

    @pytest.mark.parametrize("template", [None, "", JSON_TEMPLATE, {"default": JSON_TEMPLATE}, 7])
    def test_other_templates_return_the_plain_names(self, template):
        names = heal_gate(None, TOOLS, chat_template = template)
        assert names == {"bash", "fetch", "sleep"}
        assert type(names) is set

    @pytest.mark.parametrize(
        "template",
        [
            {"default": JSON_TEMPLATE, "tool_use": XML_TEMPLATE},
            [{"name": "default", "template": XML_TEMPLATE}],
            (None, XML_TEMPLATE),
        ],
    )
    def test_template_shapes(self, template):
        assert single_param_heal_enabled(template)
        assert _gate(template = template).single_string_param

    def test_no_template_argument_is_the_old_gate(self):
        assert type(heal_gate(None, TOOLS)) is set

    def test_request_and_process_opt_outs_still_win(self, monkeypatch):
        assert heal_gate(False, TOOLS, chat_template = XML_TEMPLATE) is None
        assert heal_gate(None, TOOLS, "none", chat_template = XML_TEMPLATE) is None
        assert (
            heal_gate(
                None, TOOLS, response_format = {"type": "json_object"}, chat_template = XML_TEMPLATE
            )
            is None
        )
        monkeypatch.setattr(ph, "_HEALING_DISABLED", True)
        assert heal_gate(None, TOOLS, chat_template = XML_TEMPLATE) is None

    def test_decision_is_cached_per_template(self):
        template = XML_TEMPLATE + "{# cache probe #}"
        before = ph._template_writes_xml_parameters.cache_info()
        for _ in range(5):
            assert single_param_heal_enabled(template)
        after = ph._template_writes_xml_parameters.cache_info()
        assert after.misses == before.misses + 1
        assert after.hits == before.hits + 4

    def test_backend_templates(self):
        class Backend:
            chat_template = JSON_TEMPLATE
            chat_template_override = XML_TEMPLATE

        assert backend_chat_templates(Backend()) == (JSON_TEMPLATE, XML_TEMPLATE)
        assert backend_chat_templates(object()) == (None, None)
        assert _gate(template = backend_chat_templates(Backend())).single_string_param
        assert type(_gate(template = backend_chat_templates(object()))) is set


class TestKillSwitch:
    def test_env_is_read_at_import(self):
        import importlib
        import os

        saved = os.environ.get("UNSLOTH_DISABLE_SINGLE_PARAM_HEAL")
        os.environ["UNSLOTH_DISABLE_SINGLE_PARAM_HEAL"] = "1"
        try:
            reloaded = importlib.reload(ph)
            assert reloaded._SINGLE_PARAM_HEAL_DISABLED is True
            assert not reloaded.single_param_heal_enabled(XML_TEMPLATE)
            assert type(reloaded.heal_gate(None, TOOLS, chat_template = XML_TEMPLATE)) is set
        finally:
            if saved is None:
                os.environ.pop("UNSLOTH_DISABLE_SINGLE_PARAM_HEAL", None)
            else:
                os.environ["UNSLOTH_DISABLE_SINGLE_PARAM_HEAL"] = saved
            importlib.reload(ph)
        assert ph._SINGLE_PARAM_HEAL_DISABLED is False

    @pytest.mark.parametrize("text", GARBLED)
    def test_switch_restores_the_old_behaviour(self, text, monkeypatch):
        names = _gate()
        off = _events(text, template = None)
        monkeypatch.setattr(ph, "_SINGLE_PARAM_HEAL_DISABLED", True)
        # Even names gated before the switch flipped do not repair.
        msg = {"role": "assistant", "content": text}
        assert heal_openai_message_events(msg, names, TOOLS) == off
        healer = StreamToolCallHealer(names, TOOLS)
        plain = StreamToolCallHealer(set(names), TOOLS)
        assert healer.feed(text) + healer.finalize() == plain.feed(text) + plain.finalize()
        assert type(_gate()) is set


import asyncio  # noqa: E402
import threading  # noqa: E402
from types import SimpleNamespace  # noqa: E402

import httpx  # noqa: E402

from core.inference.api_monitor import ApiMonitor  # noqa: E402
from models.inference import ChatCompletionRequest, ChatMessage  # noqa: E402
from routes.inference import (  # noqa: E402
    _openai_passthrough_non_streaming,
    _openai_passthrough_stream,
)

ROUTE_GARBLED = _block("<function=bash>", "<=command>")


def _route_payload(**kwargs):
    defaults = dict(
        model = "default",
        messages = [ChatMessage(role = "user", content = "hi")],
        tools = [BASH],
    )
    defaults.update(kwargs)
    return ChatCompletionRequest(**defaults)


def _route_backend(template):
    """What the routes read off the llama-server backend; no template attributes when None."""
    backend = SimpleNamespace(
        base_url = "http://llama.test",
        context_length = 4096,
        _request_reasoning_kwargs = lambda *_args, **_kwargs: None,
    )
    if template is not None:
        backend.chat_template = template
        backend.chat_template_override = None
    return backend


class _ScriptedClient:
    def __init__(self, body):
        self.body = body

    async def post(
        self,
        _url,
        json = None,
        timeout = None,
        headers = None,
    ):
        return httpx.Response(200, json = self.body)

    async def aclose(self):
        pass


def _route_non_streaming(monkeypatch, payload, template, content):
    import routes.inference as inf_mod

    body = {
        "id": "chatcmpl-up",
        "object": "chat.completion",
        "created": 1,
        "model": "gguf",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
    }
    monkeypatch.setattr(inf_mod, "_cancelable_nonstreaming_client", lambda: _ScriptedClient(body))

    async def _run():
        response = await _openai_passthrough_non_streaming(
            _route_backend(template), payload, "gguf", monitor_id = None
        )
        return json.loads(response.body)

    return asyncio.run(_run())["choices"][0]


def _route_stream(
    monkeypatch,
    payload,
    template,
    content,
    size = 9,
):
    import routes.inference as inf_mod

    class Request:
        async def is_disconnected(self):
            return False

    lines = [
        'data: {"id":"c1","model":"gguf","created":1,"choices":[{"index":0,"delta":{"content":%s}}]}'
        % json.dumps(content[start : start + size])
        for start in range(0, len(content), size)
    ]
    lines += [
        'data: {"id":"c1","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}',
        "data: [DONE]",
    ]

    async def fake_send(*_args, **_kwargs):
        return httpx.Response(200, content = b"")

    async def fake_items(*_args, **_kwargs):
        for line in lines:
            yield line

    monkeypatch.setattr(inf_mod, "_send_stream_with_preheader_cancel", fake_send)
    monkeypatch.setattr(inf_mod, "_aiter_llama_stream_items", fake_items)
    monkeypatch.setattr(inf_mod, "api_monitor", ApiMonitor(max_entries = 3))

    async def _run():
        response = await _openai_passthrough_stream(
            Request(),
            threading.Event(),
            _route_backend(template),
            payload,
            "gguf",
            "chatcmpl-test",
            monitor_id = None,
        )
        return [chunk async for chunk in response.body_iterator]

    deltas = [
        choice.get("delta") or {}
        for chunk in asyncio.run(_run())
        for line in chunk.splitlines()
        if line.startswith("data: ") and line[6:] != "[DONE]"
        for choice in json.loads(line[6:]).get("choices", [])
    ]
    calls = [
        (tc["function"]["name"], json.loads(tc["function"]["arguments"]))
        for delta in deltas
        for tc in delta.get("tool_calls") or []
    ]
    return calls, "".join(delta.get("content") or "" for delta in deltas)


class TestLlamaServerRoutesGateOnTheLoadedTemplate:
    def test_non_streaming_repairs_under_an_xml_template(self, monkeypatch):
        choice = _route_non_streaming(monkeypatch, _route_payload(), XML_TEMPLATE, ROUTE_GARBLED)
        (call,) = choice["message"]["tool_calls"]
        assert call["function"]["name"] == "bash"
        assert json.loads(call["function"]["arguments"]) == {"command": "ls -la"}
        assert choice["message"]["content"] is None
        assert choice["finish_reason"] == "tool_calls"

    @pytest.mark.parametrize("template", [None, JSON_TEMPLATE])
    def test_non_streaming_keeps_the_old_result_otherwise(self, monkeypatch, template):
        choice = _route_non_streaming(monkeypatch, _route_payload(), template, ROUTE_GARBLED)
        expected = {"role": "assistant", "content": ROUTE_GARBLED}
        heal_openai_message(expected, heal_gate(None, [BASH]), [BASH])
        assert choice["message"]["tool_calls"] == expected["tool_calls"]
        assert json.loads(choice["message"]["tool_calls"][0]["function"]["arguments"]) == {}

    def test_non_streaming_request_opt_out_relays_verbatim(self, monkeypatch):
        choice = _route_non_streaming(
            monkeypatch, _route_payload(auto_heal_tool_calls = False), XML_TEMPLATE, ROUTE_GARBLED
        )
        assert choice["message"]["content"] == ROUTE_GARBLED
        assert "tool_calls" not in choice["message"]

    def test_non_streaming_kill_switch(self, monkeypatch):
        monkeypatch.setattr(ph, "_SINGLE_PARAM_HEAL_DISABLED", True)
        choice = _route_non_streaming(monkeypatch, _route_payload(), XML_TEMPLATE, ROUTE_GARBLED)
        assert json.loads(choice["message"]["tool_calls"][0]["function"]["arguments"]) == {}

    def test_streaming_repairs_under_an_xml_template(self, monkeypatch):
        calls, text = _route_stream(monkeypatch, _route_payload(), XML_TEMPLATE, ROUTE_GARBLED)
        assert calls == [("bash", {"command": "ls -la"})]
        assert text.strip() == ""

    @pytest.mark.parametrize("template", [None, JSON_TEMPLATE])
    def test_streaming_keeps_the_old_result_otherwise(self, monkeypatch, template):
        calls, text = _route_stream(monkeypatch, _route_payload(), template, ROUTE_GARBLED)
        assert (calls, text) == _stream(ROUTE_GARBLED, 9, tools = [BASH], template = None)
        assert calls == [("bash", {})]
