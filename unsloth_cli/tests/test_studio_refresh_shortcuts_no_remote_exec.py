# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The launcher refresh must only ever run an installer that shipped with Unsloth.

`unsloth studio update` re-runs install.sh / install.ps1 with --shortcuts-only. That
used to fall back to downloading https://unsloth.ai/install.sh and piping it into
`bash -s` when no installer was on disk, which is the normal state of a PyPI install:
anyone able to tamper with that response got code execution as the updating user, on a
trust anchor weaker than the one the package itself came from.

pyproject ships the installers under <data>/share/unsloth instead, so the local exec
path covers wheel installs too and the network fallback is gone.
"""

from __future__ import annotations

import sys
import sysconfig
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _studio():
    from unsloth_cli.commands import studio as _studio_mod
    return _studio_mod


def test_candidates_cover_wheel_installs():
    """A wheel install has no repo root, so the data-files copy must be searched."""
    studio = _studio()
    candidates = studio._installer_script_candidates("install.sh")
    shipped = Path(sysconfig.get_path("data")) / "share" / "unsloth" / "install.sh"
    assert shipped in candidates
    assert all(c.name == "install.sh" for c in candidates)
    assert len(candidates) == len(set(candidates)), "duplicate lookups"


def test_local_repo_is_preferred(monkeypatch):
    monkeypatch.setenv("STUDIO_LOCAL_REPO", "/tmp/checkout")
    studio = _studio()
    assert studio._installer_script_candidates("install.sh")[0] == Path("/tmp/checkout/install.sh")


def test_shipped_installer_is_executed(monkeypatch, tmp_path):
    """The refresh runs the on-disk script by path, with --shortcuts-only."""
    studio = _studio()
    installer = tmp_path / "install.sh"
    installer.write_text("#!/bin/sh\nexit 0\n")

    monkeypatch.setattr(studio.platform, "system", lambda: "Linux")
    monkeypatch.setattr(studio, "_installer_script_candidates", lambda name: [tmp_path / name])
    calls = []

    class _Result:
        returncode = 0

    monkeypatch.setattr(
        studio.subprocess, "run", lambda argv, **kwargs: calls.append(argv) or _Result()
    )
    studio._refresh_desktop_shortcuts()
    assert calls == [["bash", str(installer), "--shortcuts-only"]]


def test_missing_installer_skips_instead_of_fetching(monkeypatch, tmp_path, capsys):
    """No installer on disk means no refresh -- never a download-and-run."""
    studio = _studio()
    monkeypatch.setattr(studio.platform, "system", lambda: "Linux")
    monkeypatch.setattr(studio, "_installer_script_candidates", lambda name: [tmp_path / name])

    def _boom(*args, **kwargs):
        raise AssertionError("launcher refresh must not open a network connection")

    monkeypatch.setattr(studio.urllib.request, "urlopen", _boom)
    monkeypatch.setattr(
        studio.subprocess,
        "run",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("nothing to execute")),
    )
    studio._refresh_desktop_shortcuts()
    assert "skipped" in capsys.readouterr().out


def test_no_installer_urls_remain():
    """Guard against the fetch-and-exec fallback being reintroduced."""
    source = (_REPO_ROOT / "unsloth_cli" / "commands" / "studio.py").read_text()
    refresh = source.split("def _installer_script_candidates", 1)[1]
    refresh = refresh.split("@studio_app.command", 1)[0]
    assert "urlopen" not in refresh
    assert 'bash", "-s"' not in refresh
