# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A tool round overlaps only when none of its calls can stop and ask."""

import json

import pytest

from state import tool_policy
from state.tool_policy import round_may_prompt


def _call(name, arguments):
    return {"id": f"c_{name}", "type": "function", "function": {"name": name, "arguments": arguments}}


def _gate(calls, mode, **kw):
    return round_may_prompt(
        calls,
        confirm_tool_calls = True,
        bypass_permissions = False,
        permission_mode = mode,
        **kw,
    )


@pytest.fixture
def isolation(monkeypatch):
    def set_to(value):
        import core.inference.os_sandbox as os_sandbox

        monkeypatch.setattr(os_sandbox, "cached_tool_isolation", lambda _name: value)

    return set_to


def test_auto_reads_the_wire_shape_without_raising():
    calls = [_call("web_search", json.dumps({"query": "tides"})), _call("web_search", "{}")]
    assert _gate(calls, "auto") is False
    calls.append(_call("web_search", json.dumps({"url": "https://example.com"})))
    assert _gate(calls, "auto") is True, "a url fetch asks"


def test_unreadable_arguments_or_a_raising_classifier_keep_the_order():
    assert _gate([_call("web_search", "{not json")], "auto") is True

    def boom(_name, _args):
        raise RuntimeError("classifier bug")

    assert _gate([_call("python", json.dumps({"code": "1"}))], "auto", is_high_risk = boom) is True


@pytest.mark.parametrize(("isolated", "asks"), [(True, False), (False, True), (None, True)])
def test_off_still_counts_an_unisolated_python(isolation, isolated, asks):
    isolation(isolated)
    risky = json.dumps({"command": "cat ~/.ssh/id_rsa"})
    calls = [_call("web_search", json.dumps({"query": "x"})), _call("terminal", risky)]
    assert _gate(calls, "off") is asks
    assert _gate(calls, "off", sandbox_level = "low") is True, "Low never claims isolation"
    calls[1] = _call("terminal", json.dumps({"command": "ls"}))
    assert _gate(calls, "off") is False, "off asks only where the call itself would"


def test_off_with_ordinary_calls_overlaps(isolation):
    isolation(False)
    assert _gate([_call("web_search", json.dumps({"query": "x"}))] * 2, "off") is False
    assert _gate([_call("web_search", "{oops")], "off") is False


def test_no_confirmation_never_gates():
    calls = [_call("python", "{oops")]
    assert round_may_prompt(calls, confirm_tool_calls = False, bypass_permissions = False, permission_mode = "ask") is False
    assert round_may_prompt(calls, confirm_tool_calls = True, bypass_permissions = True, permission_mode = "ask") is False
    assert _gate(calls, "ask") is True


def test_it_agrees_with_the_per_call_decision(isolation):
    """The round gate must never say "no prompt" for a call the per-call check then prompts."""
    isolation(False)
    for mode in ("ask", "auto", "off"):
        for level in ("high", "low"):
            for name, args in [("python", {"code": "import os"}), ("terminal", {"command": "ls"}),
                               ("terminal", {"command": "cat ~/.ssh/id_rsa"}),
                               ("web_search", {"query": "q"}), ("web_search", {"url": "https://a.b"})]:
                per_call = tool_policy.needs_tool_confirmation(
                    confirm_tool_calls = True, bypass_permissions = False, permission_mode = mode,
                    name = name, arguments = args, sandbox_level = level,
                )
                gated = _gate([_call(name, json.dumps(args))], mode, sandbox_level = level)
                assert gated is per_call, (mode, level, name, args)
