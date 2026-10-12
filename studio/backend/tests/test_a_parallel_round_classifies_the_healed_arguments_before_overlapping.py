# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A round decides whether it may overlap on the arguments `prepare_call` classifies.

The preflight read the raw dict, while `prepare_call` heals it first: a web_search `href` heals
to a `url`, a page fetch that asks under "auto", so the raw search let a sibling start before
the approval was put in front of the user.
"""

import inspect
import json

from core.inference import llama_cpp, studio_tool_loop
from core.inference.tool_loop_controller import ToolLoopController
from core.inference.tools import WEB_SEARCH_TOOL, is_high_risk_tool_call
from state.tool_policy import round_may_prompt


def _call(arguments):
    return {"id": "c", "type": "function", "function": {"name": "web_search", "arguments": arguments}}


def _gate(calls, controller = None):
    return round_may_prompt(
        calls,
        confirm_tool_calls = True,
        bypass_permissions = False,
        permission_mode = "auto",
        is_high_risk = is_high_risk_tool_call,
        healed = None if controller is None else controller.healed_arguments,
    )


def test_a_healed_url_fetch_keeps_the_round_in_order():
    controller = ToolLoopController(tools = [WEB_SEARCH_TOOL])
    calls = [
        _call(json.dumps({"query": "tides"})),
        _call(json.dumps({"href": "https://example.com"})),
    ]
    assert controller.healed_arguments(calls[1]) == ("web_search", {"url": "https://example.com"})
    assert _gate(calls, controller) is True, "the fetch asks, so nothing may start before it"


def test_plain_searches_still_overlap():
    controller = ToolLoopController(tools = [WEB_SEARCH_TOOL])
    calls = [_call(json.dumps({"query": "tides"})), _call(json.dumps({"q": "moon"}))]
    assert _gate(calls, controller) is False


def test_a_healer_that_raises_keeps_the_order():
    def boom(_call):
        raise RuntimeError("healer bug")

    assert (
        round_may_prompt(
            [_call(json.dumps({"query": "tides"}))],
            confirm_tool_calls = True,
            bypass_permissions = False,
            permission_mode = "auto",
            is_high_risk = is_high_risk_tool_call,
            healed = boom,
        )
        is True
    )


def test_both_loops_gate_on_the_healed_arguments():
    for module, name in ((studio_tool_loop, "controller"), (llama_cpp, "tool_controller")):
        source = inspect.getsource(module)
        gate = source.index("_approval_gate = round_may_prompt(")
        assert f'healed = getattr({name}, "healed_arguments", None),' in source[gate : gate + 600]
