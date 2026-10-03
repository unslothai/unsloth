# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Shared fixtures for the unsloth_cli tests."""

# --- studio home isolation (issue #9586, channel 3) -----------------------------------
# Redirect the HOME the studio root is inferred FROM, at module scope, before any test
# module imports the production code.
#
# Module scope, not a fixture: unsloth_cli/commands/studio.py binds
# `STUDIO_HOME, _STUDIO_HOME_IS_CUSTOM = _resolve_studio_home()` at import, and
# studio/backend/auth/storage.py binds `DB_PATH = auth_db_path()` the same way. Measured,
# an autouse fixture and pytest_configure both run AFTER those bindings and left auth.db
# in the real ~/.unsloth/studio/auth/. This is the placement tests/conftest.py already
# uses for its compile-cache block, for the same reason.
#
# HOME rather than UNSLOTH_STUDIO_HOME, which is the obvious lever and the wrong one:
# _resolve_studio_home() returns whether the home was explicitly set, and
# _fail_if_install_damaged() puts `UNSLOTH_STUDIO_HOME=... ` into the repair command it
# prints when it was. Setting that variable to isolate the tests therefore changes the
# output other tests in this root assert on -- measured, it breaks
# test_a_no_torch_install_keeps_that_mode_in_the_reinstall. The variable is an input to
# the behaviour under test, so it cannot also be the isolation mechanism.
#
# No teardown, deliberately: the path is redirected rather than cleaned up, so a
# hard-killed worker cannot leave residue in the real home.
import os as _os  # noqa: E402
import tempfile as _tempfile  # noqa: E402

_ISOLATED_HOME = _tempfile.mkdtemp(prefix = "unsloth-cli-tests-home-")
# Both names: POSIX resolves ~ through HOME, Windows through USERPROFILE.
_os.environ["HOME"] = _ISOLATED_HOME
_os.environ["USERPROFILE"] = _ISOLATED_HOME
# --------------------------------------------------------------------------------------

import sys
import types

import pytest


@pytest.fixture(autouse = True)
def _plain_cli_output(monkeypatch):
    """Keep Typer/Rich from colouring the output these tests assert on.

    Typer renders usage and parameter errors through Rich, which emits ANSI
    escapes as soon as FORCE_COLOR is set -- and a runner that exports it (as
    ours does) splits a plain substring like "Invalid value for
    '--gpu-memory-mode'" across escape sequences, so `in result.output` stops
    matching even though the message is right there. Setting NO_COLOR is not
    enough on its own: FORCE_COLOR still wins, so it has to be removed.
    """
    for var in ("FORCE_COLOR", "CLICOLOR_FORCE"):
        monkeypatch.delenv(var, raising = False)
    monkeypatch.setenv("NO_COLOR", "1")
    monkeypatch.setenv("TERM", "dumb")
    # UNSLOTH_DEBUG makes the catalog re-raise a failing source instead of reporting it, so a
    # developer who exports it fails every test that drives a source into a raise. The one
    # test that wants it sets it itself.
    monkeypatch.delenv("UNSLOTH_DEBUG", raising = False)


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
    tp_mod.set_tool_policy_default = lambda *a, **k: None
    state_mod.tool_policy = tp_mod
    monkeypatch.setitem(sys.modules, "state", state_mod)
    monkeypatch.setitem(sys.modules, "state.tool_policy", tp_mod)
