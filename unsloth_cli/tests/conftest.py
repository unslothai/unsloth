# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Shared fixtures for the unsloth_cli tests."""

import sys
import types

import pytest


@pytest.fixture(autouse = True)
def _plain_cli_output(monkeypatch):
    """Remove FORCE_COLOR: it beats NO_COLOR, and Rich's ANSI escapes split asserted substrings."""
    for var in ("FORCE_COLOR", "CLICOLOR_FORCE"):
        monkeypatch.delenv(var, raising = False)
    monkeypatch.setenv("NO_COLOR", "1")
    monkeypatch.setenv("TERM", "dumb")
    # UNSLOTH_DEBUG makes the catalog re-raise source failures; the one test that wants it sets it.
    monkeypatch.delenv("UNSLOTH_DEBUG", raising = False)


@pytest.fixture
def stub_tool_policy_state(monkeypatch):
    """Stub state.tool_policy so run() never depends on studio/backend already being on sys.path."""
    state_mod = types.ModuleType("state")
    tp_mod = types.ModuleType("state.tool_policy")
    tp_mod.set_tool_policy = lambda *a, **k: None
    tp_mod.set_tool_policy_default = lambda *a, **k: None
    state_mod.tool_policy = tp_mod
    monkeypatch.setitem(sys.modules, "state", state_mod)
    monkeypatch.setitem(sys.modules, "state.tool_policy", tp_mod)
