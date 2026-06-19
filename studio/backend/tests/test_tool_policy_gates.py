# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""
Tests for `_effective_enable_tools` -- the helper that folds the
process-level `tool_policy` over a request's `enable_tools` field.

Truth table (policy x payload.enable_tools -> effective):
  policy=None  + payload=None  -> None
  policy=None  + payload=True  -> True
  policy=None  + payload=False -> False
  policy=True  + payload=*     -> True
  policy=False + payload=*     -> False
"""

import os
import sys
from types import SimpleNamespace

_backend = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, _backend)

import pytest

from routes.inference import _effective_enable_tools
from state.tool_policy import reset_tool_policy, set_tool_policy


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
