# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""
Tests for the process-level server-side tool policy used by `unsloth run`.

The policy has three states:
  None  -> no CLI override (default; honor per-request enable_tools)
  True  -> CLI forced tools on
  False -> CLI forced tools off
"""

import os
import sys

_backend = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, _backend)

import pytest

from state.tool_policy import (
    get_tool_policy,
    reset_tool_policy,
    set_tool_policy,
)


@pytest.fixture(autouse = True)
def _reset():
    reset_tool_policy()
    yield
    reset_tool_policy()


class TestToolPolicy:
    def test_default_is_none(self):
        assert get_tool_policy() is None

    def test_set_true_then_get(self):
        set_tool_policy(True)
        assert get_tool_policy() is True

    def test_set_false_then_get(self):
        set_tool_policy(False)
        assert get_tool_policy() is False

    def test_set_none_clears(self):
        set_tool_policy(True)
        set_tool_policy(None)
        assert get_tool_policy() is None

    def test_reset_clears(self):
        set_tool_policy(False)
        reset_tool_policy()
        assert get_tool_policy() is None

    def test_rejects_non_optional_bool(self):
        with pytest.raises(TypeError):
            set_tool_policy("true")  # type: ignore[arg-type]
