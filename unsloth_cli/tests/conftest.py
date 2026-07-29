# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Shared fixtures for the unsloth_cli tests."""

import sys
import types

import pytest


@pytest.fixture
def stub_tool_policy_state(monkeypatch):
    """Stub the backend's `state.tool_policy`, which run() imports in-venv.

    It lives under studio/backend, so it only imports once something has put
    that directory on sys.path. Tests that reach the in-venv branch of run()
    used to get that for free from whichever file ran earlier and did it as a
    side effect, which made them pass only in a full-directory run.
    """
    state_mod = types.ModuleType("state")
    tp_mod = types.ModuleType("state.tool_policy")
    tp_mod.set_tool_policy = lambda *a, **k: None
    state_mod.tool_policy = tp_mod
    monkeypatch.setitem(sys.modules, "state", state_mod)
    monkeypatch.setitem(sys.modules, "state.tool_policy", tp_mod)
