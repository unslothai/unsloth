# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""`unsloth studio --api-max-concurrency` on the plain-server path.

The cap rides UNSLOTH_API_MAX_CONCURRENCY in the environment because the child
argv built for the re-exec does not carry the flag. Set after the re-exec
branch it never reaches the server that enforces it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner


_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_ENV = "UNSLOTH_API_MAX_CONCURRENCY"


def _studio():
    from unsloth_cli.commands import studio as _studio_mod
    return _studio_mod


class _ExecCaptured(SystemExit):
    def __init__(self, argv):
        super().__init__(0)
        self.argv = list(argv)


def _invoke_studio_default(monkeypatch, args):
    """Run plain `unsloth studio` up to (and stopping at) the venv re-exec."""
    import typer as _typer

    studio_mod = _studio()
    captured = []

    monkeypatch.setattr(sys, "prefix", "/nonexistent/outer/venv")
    monkeypatch.setattr(studio_mod, "_ensure_studio_env_exported", lambda: None)
    fake_venv = Path("/fake/studio/venv/unsloth_studio")
    monkeypatch.setattr(studio_mod, "_studio_venv_python", lambda: fake_venv / "bin" / "python")
    monkeypatch.setattr(studio_mod, "_find_run_py", lambda: Path("/fake/studio/run.py"))
    monkeypatch.setattr(
        studio_mod, "_find_frontend_dist", lambda: Path("/fake/studio/frontend/dist")
    )
    monkeypatch.setattr(sys, "platform", "linux")

    def fake_execvp(file, argv):
        captured.append(list(argv))
        raise _ExecCaptured(argv)

    monkeypatch.setattr(studio_mod.os, "execvp", fake_execvp)

    app = _typer.Typer()
    app.command()(studio_mod.studio_default)
    CliRunner().invoke(app, args, catch_exceptions = True)
    return captured


def test_studio_default_sets_the_cap_env_before_the_reexec(monkeypatch):
    """The re-exec never returns, so an env var readable afterwards was set before it."""
    studio_mod = _studio()
    monkeypatch.delenv(_ENV, raising = False)

    captured = _invoke_studio_default(monkeypatch, ["--api-max-concurrency", "2"])

    assert len(captured) == 1, captured
    assert studio_mod.os.environ[_ENV] == "2"


@pytest.mark.parametrize("subcommand", ["run", "setup"])
def test_studio_rejects_the_cap_flag_before_a_subcommand(subcommand):
    """Typer does not forward parent options, so `unsloth studio --api-max-concurrency N run`
    would silently drop the cap the user asked for (mirrors --parallel)."""
    import typer as _typer

    studio_mod = _studio()
    app = _typer.Typer()
    app.add_typer(studio_mod.studio_app, name = "studio")

    result = CliRunner().invoke(
        app, ["studio", "--api-max-concurrency", "2", subcommand, "--model", "X"]
    )

    assert result.exit_code == 2, result.output
    combined = (result.output or "") + (getattr(result, "stderr", "") or "")
    assert "--api-max-concurrency" in combined, combined
