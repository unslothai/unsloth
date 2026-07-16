# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for the `--secure/--no-secure` Studio flag: option registration,
re-exec/run_server forwarding, the forced 127.0.0.1 bind, and rejection
alongside --no-cloudflare or before a subcommand. Modeled on
test_studio_cloudflare_flag.py."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _studio():
    from unsloth_cli.commands import studio as _studio_mod
    return _studio_mod


_BASE = ["--model", "unsloth/Qwen3-1.7B-GGUF"]


# ── option registration ──────────────────────────────────────────────


def test_run_exposes_secure_option_default_off():
    import inspect

    opt = inspect.signature(_studio().run).parameters["secure"].default
    decls = set(getattr(opt, "param_decls", []) or [])
    assert "--secure/--no-secure" in decls
    assert getattr(opt, "default", None) is False


def test_studio_default_exposes_secure_option_default_off():
    import inspect

    opt = inspect.signature(_studio().studio_default).parameters["secure"].default
    decls = set(getattr(opt, "param_decls", []) or [])
    assert "--secure/--no-secure" in decls
    assert getattr(opt, "default", None) is False


def test_secure_exposes_hidden_not_secure_alias():
    # --not-secure is a hidden, deprecated alias for --no-secure on both commands.
    import inspect
    for fn in (_studio().run, _studio().studio_default):
        opt = inspect.signature(fn).parameters["not_secure"].default
        decls = set(getattr(opt, "param_decls", []) or [])
        assert "--not-secure" in decls
        assert getattr(opt, "hidden", False) is True
        assert getattr(opt, "default", None) is False


# ── re-exec capture plumbing (mirrors test_studio_cloudflare_flag.py) ─


class _ExecCaptured(SystemExit):
    def __init__(self, argv):
        super().__init__(0)
        self.argv = list(argv)


def _install_run_reexec_capture(monkeypatch):
    studio_mod = _studio()
    captured = []
    monkeypatch.setattr(sys, "prefix", "/nonexistent/outer/venv")
    fake_venv = Path("/fake/studio/venv/unsloth_studio")
    monkeypatch.setattr(studio_mod, "_studio_venv_python", lambda: fake_venv / "bin" / "python")
    # A built frontend dist is present so the public-launch UI check passes
    # deterministically (independent of whether the repo dist was built).
    monkeypatch.setattr(
        studio_mod, "_find_frontend_dist", lambda: Path("/fake/studio/frontend/dist")
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
    return captured


def _invoke_run(monkeypatch, args):
    import typer as _typer

    captured = _install_run_reexec_capture(monkeypatch)
    app = _typer.Typer()
    app.command(
        context_settings = {"allow_extra_args": True, "ignore_unknown_options": True},
    )(_studio().run)
    CliRunner().invoke(app, args, catch_exceptions = True)
    return captured


def _invoke_studio_default(monkeypatch, args):
    import typer as _typer

    studio_mod = _studio()
    captured = []
    monkeypatch.setattr(sys, "prefix", "/nonexistent/outer/venv")
    monkeypatch.setattr(studio_mod, "_ensure_studio_env_exported", lambda: None)
    fake_venv = Path("/fake/studio/venv/unsloth_studio")
    monkeypatch.setattr(studio_mod, "_studio_venv_python", lambda: fake_venv / "bin" / "python")
    monkeypatch.setattr(studio_mod, "_find_run_py", lambda: Path("/fake/studio/run.py"))
    # A built frontend dist is present so the public-launch UI check passes; this
    # suite exercises flag forwarding, not the missing-dist lockout guard.
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


# ── re-exec forwarding ────────────────────────────────────────────────


@pytest.mark.parametrize(
    "user_flag,expected,unexpected",
    [
        (None, "--no-secure", "--secure"),  # default off
        ("--secure", "--secure", "--no-secure"),
        ("--no-secure", "--no-secure", "--secure"),
        ("--not-secure", "--no-secure", "--secure"),  # deprecated alias -> canonical
    ],
)
def test_run_reexec_forwards_secure_polarity(monkeypatch, user_flag, expected, unexpected):
    extras = [user_flag] if user_flag else []
    captured = _invoke_run(monkeypatch, _BASE + extras)
    assert len(captured) == 1, captured
    argv = captured[0]
    assert expected in argv and unexpected not in argv, argv


def test_run_secure_forces_localhost_in_reexec(monkeypatch):
    # `unsloth studio run -H 0.0.0.0 --secure` must re-exec with --host 127.0.0.1.
    captured = _invoke_run(monkeypatch, _BASE + ["-H", "0.0.0.0", "--secure"])
    assert len(captured) == 1, captured
    argv = captured[0]
    assert "--secure" in argv
    assert argv[argv.index("--host") + 1] == "127.0.0.1", argv


def test_studio_default_reexec_forwards_secure(monkeypatch):
    captured = _invoke_studio_default(monkeypatch, ["-H", "0.0.0.0", "--secure"])
    assert len(captured) == 1, captured
    argv = captured[0]
    assert "--secure" in argv
    # studio_default also forces the loopback bind under --secure.
    assert argv[argv.index("--host") + 1] == "127.0.0.1", argv


def test_run_secure_warns_when_host_overridden(monkeypatch):
    # -H 0.0.0.0 --secure forces the loopback bind; warn (not error) that -H is
    # ignored so it does not silently read as "secure and on the network".
    import typer as _typer

    _install_run_reexec_capture(monkeypatch)
    app = _typer.Typer()
    app.command(
        context_settings = {"allow_extra_args": True, "ignore_unknown_options": True},
    )(_studio().run)
    result = CliRunner().invoke(app, _BASE + ["-H", "0.0.0.0", "--secure"], catch_exceptions = True)
    combined = (result.output or "") + (getattr(result, "stderr", "") or "")
    assert "ignores -H" in combined, combined


def test_run_secure_no_warning_when_already_loopback(monkeypatch):
    # --secure with an already-loopback -H must not warn about ignoring -H.
    import typer as _typer

    _install_run_reexec_capture(monkeypatch)
    app = _typer.Typer()
    app.command(
        context_settings = {"allow_extra_args": True, "ignore_unknown_options": True},
    )(_studio().run)
    result = CliRunner().invoke(app, _BASE + ["-H", "127.0.0.1", "--secure"], catch_exceptions = True)
    combined = (result.output or "") + (getattr(result, "stderr", "") or "")
    assert "ignores -H" not in combined, combined


def test_studio_default_not_secure_alias_forwards_no_secure(monkeypatch):
    # --not-secure on `unsloth studio` forwards the canonical --no-secure.
    captured = _invoke_studio_default(monkeypatch, ["--not-secure"])
    assert len(captured) == 1, captured
    argv = captured[0]
    assert "--no-secure" in argv and "--secure" not in argv, argv


@pytest.mark.parametrize(
    "argv_order,expected,unexpected",
    [
        # --not-secure tracks --no-secure: the last secure flag on argv wins,
        # matching the backend BooleanOptionalAction.
        (["--secure", "--not-secure"], "--no-secure", "--secure"),
        (["--not-secure", "--secure"], "--secure", "--no-secure"),
    ],
)
def test_run_not_secure_alias_respects_last_wins(monkeypatch, argv_order, expected, unexpected):
    monkeypatch.setattr(sys, "argv", ["unsloth", "studio", "run", *argv_order])
    captured = _invoke_run(monkeypatch, _BASE + argv_order)
    assert len(captured) == 1, captured
    argv = captured[0]
    assert expected in argv and unexpected not in argv, argv


# ── in-venv path forwards secure + forced host into run_server ────────


class _RunServerCaptured(SystemExit):
    def __init__(self, kwargs):
        super().__init__(0)
        self.kwargs = dict(kwargs)


def test_run_in_venv_passes_secure_and_forces_host(monkeypatch, tmp_path):
    import types

    studio_mod = _studio()
    # Real STUDIO_HOME with an already-changed admin (must_change_password=0) so
    # the pre-exposure gate is a no-op and the in-venv path reaches run_server.
    # (The gate now fails closed if it cannot open the auth DB, so a fake path
    # would refuse the launch before this assertion.)
    monkeypatch.setattr(studio_mod, "STUDIO_HOME", tmp_path)
    _seed = studio_mod._connect_auth_db()
    studio_mod._ensure_cli_default_admin(_seed)
    _seed.execute("UPDATE auth_user SET must_change_password = 0")
    _seed.commit()
    _seed.close()

    fake_venv = tmp_path / "unsloth_studio"
    monkeypatch.setattr(sys, "prefix", str(fake_venv))

    from unsloth_cli import _tool_policy as _tp_mod

    monkeypatch.setattr(
        _tp_mod,
        "resolve_tool_policy",
        lambda host, flag, yes, silent: False if flag is None else bool(flag),
    )

    captured: dict = {}

    def fake_run_server(**kwargs):
        captured.update(kwargs)
        raise _RunServerCaptured(kwargs)

    fake_backend_run = sys.modules.setdefault(
        "studio.backend.run", types.ModuleType("studio.backend.run")
    )
    fake_backend_run.run_server = fake_run_server
    fake_backend_run._resolve_external_ip = lambda: "127.0.0.1"
    monkeypatch.setattr(studio_mod, "_RUN_MODULE", fake_backend_run)

    import typer as _typer

    app = _typer.Typer()
    app.command(
        context_settings = {"allow_extra_args": True, "ignore_unknown_options": True},
    )(studio_mod.run)
    CliRunner().invoke(app, _BASE + ["-H", "0.0.0.0", "--secure"], catch_exceptions = True)

    assert captured.get("secure") is True, captured
    assert captured.get("host") == "127.0.0.1", captured


# ── --secure + --no-cloudflare is rejected ───────────────────────────


def test_run_secure_rejects_no_cloudflare(monkeypatch):
    studio_mod = _studio()
    import typer as _typer

    app = _typer.Typer()
    app.command(
        context_settings = {"allow_extra_args": True, "ignore_unknown_options": True},
    )(studio_mod.run)
    result = CliRunner().invoke(app, _BASE + ["--secure", "--no-cloudflare"])
    assert result.exit_code == 2, result.output


def test_studio_default_rejects_secure_with_subcommand():
    import typer as _typer

    studio_mod = _studio()
    app = _typer.Typer()
    app.add_typer(studio_mod.studio_app, name = "studio")
    result = CliRunner().invoke(app, ["studio", "--secure", "run", "--model", "X"])
    assert result.exit_code == 2, result.output
    combined = (result.output or "") + (getattr(result, "stderr", "") or "")
    assert "--secure" in combined, combined


# ── secure resolves tools against the loopback bind (tools stay ON) ──


def test_run_secure_resolves_tools_against_loopback(monkeypatch):
    # --secure is a loopback bind behind an authenticated tunnel, so tools resolve
    # against 127.0.0.1 (ON): the child gets --enable-tools, not --disable-tools.
    studio_mod = _studio()
    monkeypatch.setattr(sys, "prefix", "/nonexistent/outer/venv")
    fake_venv = Path("/fake/studio/venv/unsloth_studio")
    monkeypatch.setattr(studio_mod, "_studio_venv_python", lambda: fake_venv / "bin" / "python")
    # A built frontend dist is present so the public-launch UI check passes
    # deterministically (independent of whether the repo dist was built).
    monkeypatch.setattr(
        studio_mod, "_find_frontend_dist", lambda: Path("/fake/studio/frontend/dist")
    )
    fake_bin = fake_venv / "bin" / "unsloth"
    real_is_file = Path.is_file
    monkeypatch.setattr(
        Path,
        "is_file",
        lambda self: True if str(self) == str(fake_bin) else real_is_file(self),
    )
    monkeypatch.setattr(sys, "platform", "linux")

    from unsloth_cli import _tool_policy as _tp_mod

    calls = []

    def rec(host, flag, yes, silent):
        calls.append(host)
        return True if flag is None else bool(flag)  # default ON everywhere

    monkeypatch.setattr(_tp_mod, "resolve_tool_policy", rec)

    captured = []

    def fake_execvp(file, argv):
        captured.append(list(argv))
        raise _ExecCaptured(argv)

    monkeypatch.setattr(studio_mod.os, "execvp", fake_execvp)

    import typer as _typer

    app = _typer.Typer()
    app.command(
        context_settings = {"allow_extra_args": True, "ignore_unknown_options": True},
    )(studio_mod.run)
    CliRunner().invoke(app, _BASE + ["-H", "0.0.0.0", "--secure"], catch_exceptions = True)

    # Resolved against the forced-loopback bind, not the public 0.0.0.0 exposure.
    assert calls and calls[0] == "127.0.0.1", calls
    assert len(captured) == 1, captured
    assert "--enable-tools" in captured[0] and "--disable-tools" not in captured[0], captured[0]


def test_run_secure_enable_tools_no_auto_yes(monkeypatch):
    # No prompt now, so a secure --enable-tools forwards --enable-tools but not
    # --yes (only an explicit --yes is forwarded).
    captured = _invoke_run(monkeypatch, _BASE + ["-H", "0.0.0.0", "--secure", "--enable-tools"])
    assert len(captured) == 1, captured
    argv = captured[0]
    assert "--enable-tools" in argv, argv
    assert "--yes" not in argv, argv


# ── plain `unsloth studio` exposes + forwards --enable-tools/--disable-tools ──


def test_studio_default_exposes_enable_tools_option_default_none():
    import inspect

    opt = inspect.signature(_studio().studio_default).parameters["enable_tools"].default
    decls = set(getattr(opt, "param_decls", []) or [])
    assert "--enable-tools/--disable-tools" in decls
    assert opt.default is None  # tri-state: omitted -> leave policy unset (tools on)


def test_studio_default_forwards_disable_tools(monkeypatch):
    captured = _invoke_studio_default(monkeypatch, ["--disable-tools"])
    assert len(captured) == 1, captured
    assert "--disable-tools" in captured[0] and "--enable-tools" not in captured[0], captured[0]


def test_studio_default_forwards_enable_tools(monkeypatch):
    captured = _invoke_studio_default(monkeypatch, ["--enable-tools"])
    assert len(captured) == 1, captured
    assert "--enable-tools" in captured[0] and "--disable-tools" not in captured[0], captured[0]


def test_studio_default_no_tool_flag_omits_both(monkeypatch):
    # No flag -> neither flag forwarded; run.py leaves the policy unset (tools on).
    captured = _invoke_studio_default(monkeypatch, [])
    assert len(captured) == 1, captured
    assert "--enable-tools" not in captured[0] and "--disable-tools" not in captured[0], captured[0]


def test_studio_default_rejects_enable_tools_with_subcommand():
    import typer as _typer

    studio_mod = _studio()
    app = _typer.Typer()
    app.add_typer(studio_mod.studio_app, name = "studio")
    result = CliRunner().invoke(app, ["studio", "--enable-tools", "run", "--model", "X"])
    assert result.exit_code == 2, result.output
    combined = (result.output or "") + (getattr(result, "stderr", "") or "")
    assert "--enable-tools" in combined, combined


def test_run_tool_help_reflects_default_on_everywhere():
    # Help must match the new policy (tools on everywhere, no prompt).
    import inspect

    params = inspect.signature(_studio().run).parameters
    tools_help = params["enable_tools"].default.help or ""
    assert "on for every bind" in tools_help, tools_help
    assert "0.0.0.0" not in tools_help, tools_help

    yes_help = params["yes"].default.help or ""
    assert "Skip the 0.0.0.0" not in yes_help, yes_help
    assert "no longer prompts" in yes_help, yes_help
