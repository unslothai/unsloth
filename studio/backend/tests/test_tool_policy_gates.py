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

from routes.inference import _effective_enable_tools
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
