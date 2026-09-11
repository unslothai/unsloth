# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Shared fixtures for the unsloth_cli tests."""

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


@pytest.fixture(autouse = True)
def _no_cloudflared_download(monkeypatch, tmp_path):
    """Stop a unit test fetching the 40 MB cloudflared binary (issue #9586, channel 3a).

    The `--secure` path calls a helper that loads studio/backend/cloudflare_tunnel.py by
    file path and asks `ensure_cloudflared()` whether a tunnel could start. That call
    DOWNLOADS the binary when none is found: measured on Linux, eight tests across
    test_studio_secure_flag.py and test_studio_cloudflare_flag.py each fetched 39,799,316
    bytes into ~/.unsloth/studio/bin/cloudflared. The suite then depends on network
    reachability and writes a large file outside anything it owns.

    Satisfied through PATH rather than by patching the module. The helper does
    `spec_from_file_location("studio.backend.cloudflare_tunnel", ...)` and execs a FRESH
    module object on every call, so `monkeypatch.setattr` on an imported
    `cloudflare_tunnel` reaches a different object entirely and the download still happens
    -- measured. `find_cloudflared()` consults `shutil.which("cloudflared")` first and
    returns early, so a stub on PATH satisfies every copy of the module, and the real
    lookup, cache-path and platform-asset code still runs.

    Both names: shutil.which keys off os.name for PATHEXT, which these tests do not
    monkeypatch even where they do set sys.platform.
    """
    import os as _os

    stub_dir = tmp_path / "fake-bin"
    stub_dir.mkdir()
    for name in ("cloudflared", "cloudflared.exe"):
        stub = stub_dir / name
        # Contents are irrelevant: shutil.which checks existence and the executable
        # bit, and nothing ever runs this file.
        stub.write_text("", encoding = "utf-8")
        stub.chmod(0o755)
    monkeypatch.setenv("PATH", str(stub_dir) + _os.pathsep + _os.environ.get("PATH", ""))
