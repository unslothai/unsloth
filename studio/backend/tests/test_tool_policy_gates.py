# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""
Tests for `_effective_enable_tools` -- folds the process-level `tool_policy`
over a request's `enable_tools` field.

Truth table (override x default x payload.enable_tools -> effective):
  override=None  + default=None + payload=None  -> None
  override=None  + default=True + payload=None  -> True   (launcher default)
  override=None  + default=True + payload=False -> False  (request opts out)
  override=None  + payload=True                 -> True
  override=None  + payload=False                -> False
  override=True  + payload=*                    -> True   (--enable-tools)
  override=False + payload=*                    -> False  (--disable-tools)
"""

import os
import sys
from types import SimpleNamespace

_backend = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, _backend)

import pytest

from routes.inference import (
    _effective_enable_tools,
    _request_states_tool_intent,
    _tools_on_by_launcher_default_only,
)
from state.tool_policy import reset_tool_policy, set_tool_policy, set_tool_policy_default


@pytest.fixture(autouse = True)
def _reset():
    reset_tool_policy()
    yield
    reset_tool_policy()


def _payload(value):
    return SimpleNamespace(enable_tools = value)


class TestEffectiveEnableTools:
    @pytest.mark.parametrize(
        "payload_value,expected",
        [(None, None), (True, True), (False, False)],
    )
    def test_no_policy_falls_through_to_payload(self, payload_value, expected):
        assert _effective_enable_tools(_payload(payload_value)) == expected

    @pytest.mark.parametrize("payload_value", [None, True, False])
    def test_policy_true_overrides_any_payload(self, payload_value):
        set_tool_policy(True)
        assert _effective_enable_tools(_payload(payload_value)) is True

    @pytest.mark.parametrize("payload_value", [None, True, False])
    def test_policy_false_overrides_any_payload(self, payload_value):
        set_tool_policy(False)
        assert _effective_enable_tools(_payload(payload_value)) is False


class TestLauncherDefault:
    """`unsloth studio [run]` installs a tools-on default (not an override), so a
    request that never mentions tools gets them and one that says false does not."""

    def test_omitted_falls_back_to_default(self):
        set_tool_policy_default(True)
        assert _effective_enable_tools(_payload(None)) is True

    def test_explicit_false_beats_default(self):
        set_tool_policy_default(True)
        assert _effective_enable_tools(_payload(False)) is False

    def test_explicit_true_matches_default(self):
        set_tool_policy_default(True)
        assert _effective_enable_tools(_payload(True)) is True

    @pytest.mark.parametrize("payload_value", [None, True, False])
    def test_disable_tools_override_beats_default(self, payload_value):
        set_tool_policy_default(True)
        set_tool_policy(False)
        assert _effective_enable_tools(_payload(payload_value)) is False

    @pytest.mark.parametrize("payload_value", [None, True, False])
    def test_enable_tools_override_beats_default(self, payload_value):
        set_tool_policy_default(True)
        set_tool_policy(True)
        assert _effective_enable_tools(_payload(payload_value)) is True

    def test_force_disabled_context_beats_default(self):
        from state.tool_policy import tools_force_disabled
        set_tool_policy_default(True)
        with tools_force_disabled():
            assert _effective_enable_tools(_payload(None)) is False


def _msg(role = "user", tool_calls = None):
    return SimpleNamespace(role = role, tool_calls = tool_calls, content = "hi")


def _req(**kw):
    fields = dict(
        enable_tools = None,
        mcp_enabled = None,
        tool_choice = None,
        tools = None,
        response_format = None,
        messages = [_msg()],
        model_extra = {},
    )
    fields.update(kw)
    return SimpleNamespace(**fields)


def _sf_tools_on(payload):
    """The safetensors gate's resolution, as routes.inference applies it."""
    on = _effective_enable_tools(payload)
    if on and _tools_on_by_launcher_default_only(payload) and _request_states_tool_intent(payload):
        on = False
    return on


class TestRequestStatesToolIntent:
    """A request that uses the standard OpenAI tool fields has stated its intent,
    so the launcher default must not answer for it."""

    def test_plain_chat_states_nothing(self):
        assert _request_states_tool_intent(_req()) is False

    def test_tool_choice_none_is_a_withdrawal(self):
        assert _request_states_tool_intent(_req(tool_choice = "none")) is True

    def test_client_catalog_is_intent(self):
        assert _request_states_tool_intent(_req(tools = [{"function": {"name": "f"}}])) is True

    def test_tool_result_history_is_intent(self):
        assert _request_states_tool_intent(_req(messages = [_msg(role = "tool")])) is True
        assert _request_states_tool_intent(_req(messages = [_msg(tool_calls = [{}])])) is True

    def test_response_format_is_a_contract(self):
        # The tool loop would break structured output; the GGUF passthrough
        # already exempts these requests from the policy.
        payload = _req(response_format = {"type": "json_object"})
        assert _request_states_tool_intent(payload) is True

    def test_empty_tools_reads_as_omitted(self):
        # bool(payload.tools) is the GGUF router's own reading in
        # _takes_tool_passthrough; both paths treat [] like an absent catalog.
        assert _request_states_tool_intent(_req(tools = [])) is False


class TestLauncherDefaultOnly:
    def test_true_when_nothing_asked_and_no_override(self):
        set_tool_policy_default(True)
        assert _tools_on_by_launcher_default_only(_req()) is True

    @pytest.mark.parametrize("payload", [_req(enable_tools = True), _req(mcp_enabled = True)])
    def test_false_when_the_request_asked(self, payload):
        set_tool_policy_default(True)
        assert _tools_on_by_launcher_default_only(payload) is False

    @pytest.mark.parametrize("override", [True, False])
    def test_false_under_a_cli_override(self, override):
        set_tool_policy_default(True)
        set_tool_policy(override)
        assert _tools_on_by_launcher_default_only(_req()) is False


class TestSafetensorsGateHonorsStatedIntent:
    """The safetensors path has no llama-server passthrough to fall back on, so
    the launcher default must not withdraw tool_choice: "none" or take a client
    catalog into Unsloth's own loop."""

    def test_plain_chat_still_gets_the_default(self):
        set_tool_policy_default(True)
        assert _sf_tools_on(_req()) is True

    def test_tool_choice_none_is_honored(self):
        set_tool_policy_default(True)
        assert _sf_tools_on(_req(tool_choice = "none")) is False

    def test_client_catalog_keeps_the_passthrough(self):
        set_tool_policy_default(True)
        assert _sf_tools_on(_req(tools = [{"function": {"name": "f"}}])) is False

    def test_tool_result_history_keeps_the_passthrough(self):
        set_tool_policy_default(True)
        assert _sf_tools_on(_req(messages = [_msg(role = "tool")])) is False

    def test_response_format_keeps_structured_output(self):
        set_tool_policy_default(True)
        payload = _req(response_format = {"type": "json_object"})
        assert _sf_tools_on(payload) is False

    def test_explicit_ask_still_claims_a_catalog(self):
        set_tool_policy_default(True)
        payload = _req(enable_tools = True, tools = [{"function": {"name": "f"}}])
        assert _sf_tools_on(payload) is True

    def test_cli_enable_tools_is_unchanged(self):
        # Pre-existing --enable-tools behavior on this path is untouched.
        set_tool_policy_default(True)
        set_tool_policy(True)
        assert _sf_tools_on(_req(tools = [{"function": {"name": "f"}}])) is True

    def test_cli_disable_tools_still_wins(self):
        set_tool_policy_default(True)
        set_tool_policy(False)
        assert _sf_tools_on(_req()) is False
