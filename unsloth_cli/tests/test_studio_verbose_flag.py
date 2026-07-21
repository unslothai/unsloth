# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for the `--verbose/-v` Unsloth flag: option registration on both the
plain callback and the `run` subcommand, re-exec forwarding, the access-log
env override, and rejection before a subcommand. Modeled on
test_studio_secure_flag.py."""

from __future__ import annotations

import sys
from pathlib import Path

from typer.testing import CliRunner

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_DEDUP = "UNSLOTH_STUDIO_ACCESS_LOG_DEDUP_MS"
_POLL = "UNSLOTH_STUDIO_ACCESS_LOG_POLL_DEDUP_MS"
_BASE = ["--model", "unsloth/Qwen3-1.7B-GGUF"]


def _studio():
    from unsloth_cli.commands import studio as _studio_mod
    return _studio_mod


# ── option registration ──────────────────────────────────────────────


def test_run_exposes_verbose_option_default_off():
    import inspect

    opt = inspect.signature(_studio().run).parameters["verbose"].default
    decls = set(getattr(opt, "param_decls", []) or [])
    assert "--verbose" in decls and "-v" in decls
    assert getattr(opt, "default", None) is False


def test_studio_default_exposes_verbose_option_default_off():
    import inspect

    opt = inspect.signature(_studio().studio_default).parameters["verbose"].default
    decls = set(getattr(opt, "param_decls", []) or [])
    assert "--verbose" in decls and "-v" in decls
    assert getattr(opt, "default", None) is False


# ── re-exec capture plumbing (mirrors test_studio_secure_flag.py) ─────


class _ExecCaptured(SystemExit):
    def __init__(self, argv):
        super().__init__(0)
        self.argv = list(argv)


def _invoke_run(monkeypatch, args):
    import typer as _typer

    studio_mod = _studio()
    captured = []
    monkeypatch.setattr(sys, "prefix", "/nonexistent/outer/venv")
    fake_venv = Path("/fake/studio/venv/unsloth_studio")
    monkeypatch.setattr(
        studio_mod, "_studio_venv_python", lambda: fake_venv / "bin" / "python"
    )
    fake_bin = fake_venv / "bin" / "unsloth"
    real_is_file = Path.is_file
    monkeypatch.setattr(
        Path,
        "is_file",
        lambda self: True if str(self) == str(fake_bin) else real_is_file(self),
    )
    from unsloth_cli import _tool_policy as _tp_mod

    monkeypatch.setattr(
        _tp_mod,
        "resolve_tool_policy",
        lambda host, flag, yes, silent: False if flag is None else bool(flag),
    )
    monkeypatch.setattr(sys, "platform", "linux")

    def fake_execvp(file, argv):
        captured.append(list(argv))
        raise _ExecCaptured(argv)

    monkeypatch.setattr(studio_mod.os, "execvp", fake_execvp)
    app = _typer.Typer()
    app.command(
        context_settings = {"allow_extra_args": True, "ignore_unknown_options": True},
    )(studio_mod.run)
    CliRunner().invoke(app, args, catch_exceptions = True)
    return captured


# ── re-exec forwarding + env override ─────────────────────────────────


def test_run_verbose_sets_env_and_forwards_on_reexec(monkeypatch):
    monkeypatch.delenv(_DEDUP, raising = False)
    monkeypatch.delenv(_POLL, raising = False)
    captured = _invoke_run(monkeypatch, _BASE + ["--verbose"])
    assert len(captured) == 1, captured
    assert "--verbose" in captured[0], captured[0]
    import os as _os

    assert _os.environ.get(_DEDUP) == "0"
    assert _os.environ.get(_POLL) == "0"


def test_run_without_verbose_leaves_env_unset(monkeypatch):
    monkeypatch.delenv(_DEDUP, raising = False)
    monkeypatch.delenv(_POLL, raising = False)
    captured = _invoke_run(monkeypatch, _BASE)
    assert len(captured) == 1, captured
    assert "--verbose" not in captured[0], captured[0]
    assert "--log-verbose" not in captured[0], captured[0]
    import os as _os

    assert _os.environ.get(_DEDUP) is None
    assert _os.environ.get(_POLL) is None


def test_run_verbose_preserves_llama_server_verbosity(monkeypatch):
    # Unsloth consumes --verbose but still forwards llama-server's own verbosity.
    monkeypatch.delenv(_DEDUP, raising = False)
    monkeypatch.delenv(_POLL, raising = False)
    captured = _invoke_run(monkeypatch, _BASE + ["--verbose"])
    assert len(captured) == 1, captured
    assert "--log-verbose" in captured[0], captured[0]


def test_run_verbose_does_not_duplicate_existing_llama_verbose(monkeypatch):
    monkeypatch.delenv(_DEDUP, raising = False)
    monkeypatch.delenv(_POLL, raising = False)
    captured = _invoke_run(monkeypatch, _BASE + ["--verbose", "--log-verbose"])
    assert len(captured) == 1, captured
    assert captured[0].count("--log-verbose") == 1, captured[0]


# ── --verbose before a subcommand is rejected ─────────────────────────


def test_studio_default_rejects_verbose_with_subcommand():
    import typer as _typer

    studio_mod = _studio()
    app = _typer.Typer()
    app.add_typer(studio_mod.studio_app, name = "studio")
    result = CliRunner().invoke(app, ["studio", "--verbose", "run", "--model", "X"])
    assert result.exit_code == 2, result.output
    combined = (result.output or "") + (getattr(result, "stderr", "") or "")
    assert "--verbose" in combined, combined
