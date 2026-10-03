# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Shared fixtures for the unsloth_cli tests."""

import sys
import tempfile
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


@pytest.fixture(autouse = True)
def _contain_tempdir(monkeypatch, tmp_path):
    """Point tempfile at a per-test directory (issue #9586, channel 4).

    `_start_studio_server()` opens
    `Path(tempfile.gettempdir()) / f"unsloth-start-server-{os.getpid()}.log"`, so a test
    that drives it for real leaves that file in the shared tempdir. The name carries the
    pid, so it accumulates rather than collides -- measured, a full `unsloth_cli/tests`
    run under `-n 4` left one per worker and nothing else.

    A fixture is early enough here, unlike the studio-home case in the same conftest:
    gettempdir() is called inside the function rather than bound at import.

    `tempfile.tempdir` is the documented override and beats TMPDIR/TEMP/TMP, which
    gettempdir() consults only once and then caches. Redirecting rather than cleaning up
    is deliberate: pytest owns tmp_path, so a hard-killed worker cannot leave residue in
    the shared tempdir the way a `finally` unlink could.
    """
    private = tmp_path / "tempdir"
    private.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(private))
