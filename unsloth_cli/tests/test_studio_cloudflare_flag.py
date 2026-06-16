# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for the `--cloudflare/--no-cloudflare` Studio flag.

Pins the typer Option (default on) on both `unsloth studio` and
`unsloth studio run`, and that the chosen polarity reaches the re-exec'd
child and run_server. Modeled on test_studio_run_parallel_flag.py.
"""

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


def test_run_exposes_cloudflare_option_default_on():
    import inspect

    sig = inspect.signature(_studio().run)
    assert "cloudflare" in sig.parameters
    opt = sig.parameters["cloudflare"].default
    decls = set(getattr(opt, "param_decls", []) or [])
    assert "--cloudflare/--no-cloudflare" in decls
    assert getattr(opt, "default", None) is True


def test_studio_default_exposes_cloudflare_option_default_on():
    import inspect

    sig = inspect.signature(_studio().studio_default)
    assert "cloudflare" in sig.parameters
    opt = sig.parameters["cloudflare"].default
    assert getattr(opt, "default", None) is True


# ── re-exec forwarding: `unsloth studio run` ─────────────────────────


class _ExecCaptured(SystemExit):
    def __init__(self, argv):
        super().__init__(0)
        self.argv = list(argv)


def _install_run_reexec_capture(monkeypatch, *, platform = "linux"):
    studio_mod = _studio()
    captured = []

    monkeypatch.setattr(sys, "prefix", "/nonexistent/outer/venv")
    fake_venv = Path("/fake/studio/venv/unsloth_studio")
    monkeypatch.setattr(studio_mod, "_studio_venv_python", lambda: fake_venv / "bin" / "python")
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
    monkeypatch.setattr(sys, "platform", platform)

    def fake_execvp(file, argv):
        captured.append(list(argv))
        raise _ExecCaptured(argv)

    monkeypatch.setattr(studio_mod.os, "execvp", fake_execvp)
    return captured


def _invoke_run(monkeypatch, args):
    import typer as _typer

    studio_mod = _studio()
    captured = _install_run_reexec_capture(monkeypatch)
    app = _typer.Typer()
    app.command(
        context_settings = {"allow_extra_args": True, "ignore_unknown_options": True},
    )(studio_mod.run)
    CliRunner().invoke(app, args, catch_exceptions = True)
    return captured


@pytest.mark.parametrize(
    "user_flag,expected,unexpected",
    [
        (None, "--cloudflare", "--no-cloudflare"),  # default on
        ("--cloudflare", "--cloudflare", "--no-cloudflare"),
        ("--no-cloudflare", "--no-cloudflare", "--cloudflare"),
    ],
)
def test_run_reexec_forwards_cloudflare_polarity(monkeypatch, user_flag, expected, unexpected):
    extras = [user_flag] if user_flag else []
    captured = _invoke_run(monkeypatch, _BASE + extras)
    assert len(captured) == 1, captured
    argv = captured[0]
    assert expected in argv, f"expected {expected} in child argv; got {argv}"
    assert unexpected not in argv, f"unexpected {unexpected} in child argv; got {argv}"


# ── re-exec forwarding: plain `unsloth studio` ───────────────────────


def _invoke_studio_default(
    monkeypatch,
    args,
    *,
    platform = "linux",
):
    import typer as _typer

    studio_mod = _studio()
    captured = []

    monkeypatch.setattr(sys, "prefix", "/nonexistent/outer/venv")
    monkeypatch.setattr(studio_mod, "_ensure_studio_env_exported", lambda: None)
    fake_venv = Path("/fake/studio/venv/unsloth_studio")
    monkeypatch.setattr(studio_mod, "_studio_venv_python", lambda: fake_venv / "bin" / "python")
    monkeypatch.setattr(studio_mod, "_find_run_py", lambda: Path("/fake/studio/run.py"))
    monkeypatch.setattr(studio_mod, "_find_frontend_dist", lambda: None)
    monkeypatch.setattr(sys, "platform", platform)

    def fake_execvp(file, argv):
        captured.append(list(argv))
        raise _ExecCaptured(argv)

    monkeypatch.setattr(studio_mod.os, "execvp", fake_execvp)

    app = _typer.Typer()
    app.command()(studio_mod.studio_default)
    CliRunner().invoke(app, args, catch_exceptions = True)
    return captured


@pytest.mark.parametrize(
    "user_flag,expected,unexpected",
    [
        (None, "--cloudflare", "--no-cloudflare"),
        ("--no-cloudflare", "--no-cloudflare", "--cloudflare"),
    ],
)
def test_studio_default_reexec_forwards_cloudflare(monkeypatch, user_flag, expected, unexpected):
    extras = [user_flag] if user_flag else []
    captured = _invoke_studio_default(monkeypatch, ["-H", "0.0.0.0"] + extras)
    assert len(captured) == 1, captured
    argv = captured[0]
    assert expected in argv, f"expected {expected}; got {argv}"
    assert unexpected not in argv, f"unexpected {unexpected}; got {argv}"


# ── in-venv path forwards cloudflare into run_server ─────────────────


class _RunServerCaptured(SystemExit):
    def __init__(self, kwargs):
        super().__init__(0)
        self.kwargs = dict(kwargs)


@pytest.mark.parametrize("user_flag,expected", [(None, True), ("--no-cloudflare", False)])
def test_run_in_venv_passes_cloudflare_to_run_server(monkeypatch, user_flag, expected):
    import types

    studio_mod = _studio()
    fake_venv = Path("/fake/studio/venv/unsloth_studio")
    monkeypatch.setattr(sys, "prefix", str(fake_venv))
    monkeypatch.setattr(studio_mod, "STUDIO_HOME", fake_venv.parent)

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
    # run() loads the backend via _load_run_module() (by file path); inject the
    # mock as the cached run module so the stubbed run_server is used.
    monkeypatch.setattr(studio_mod, "_RUN_MODULE", fake_backend_run)

    state_mod = types.ModuleType("state")
    tp_mod = types.ModuleType("state.tool_policy")
    tp_mod.set_tool_policy = lambda *a, **k: None
    state_mod.tool_policy = tp_mod
    monkeypatch.setitem(sys.modules, "state", state_mod)
    monkeypatch.setitem(sys.modules, "state.tool_policy", tp_mod)

    import typer as _typer

    app = _typer.Typer()
    app.command(
        context_settings = {"allow_extra_args": True, "ignore_unknown_options": True},
    )(studio_mod.run)
    extras = [user_flag] if user_flag else []
    CliRunner().invoke(app, _BASE + extras, catch_exceptions = True)

    assert captured.get("cloudflare") is expected, captured


# ── parent-level --no-cloudflare with a subcommand is rejected ───────


def test_studio_default_rejects_no_cloudflare_with_subcommand(monkeypatch):
    # `unsloth studio --no-cloudflare run ...` would not reach the subcommand,
    # so it must error (mirrors --parallel) rather than silently still tunnel.
    import typer as _typer

    studio_mod = _studio()
    app = _typer.Typer()
    app.add_typer(studio_mod.studio_app, name = "studio")
    result = CliRunner().invoke(app, ["studio", "--no-cloudflare", "run", "--model", "X"])
    assert result.exit_code == 2, result.output
    combined = (result.output or "") + (getattr(result, "stderr", "") or "")
    assert "--no-cloudflare" in combined, combined


# ── run() tears the server + tunnel down if startup aborts ───────────


def test_run_in_venv_shuts_down_on_startup_abort(monkeypatch):
    import types

    studio_mod = _studio()
    fake_venv = Path("/fake/studio/venv/unsloth_studio")
    monkeypatch.setattr(sys, "prefix", str(fake_venv))
    monkeypatch.setattr(studio_mod, "STUDIO_HOME", fake_venv.parent)

    from unsloth_cli import _tool_policy as _tp_mod

    monkeypatch.setattr(
        _tp_mod,
        "resolve_tool_policy",
        lambda host, flag, yes, silent: False if flag is None else bool(flag),
    )

    class _App:
        class state:
            server_port = 8888

    shutdown_calls = []
    backend = sys.modules.setdefault("studio.backend.run", types.ModuleType("studio.backend.run"))
    backend.run_server = lambda **k: _App()
    backend._resolve_external_ip = lambda: "1.2.3.4"
    backend._server = object()
    backend._shutdown_event = None
    backend._graceful_shutdown = lambda server: shutdown_calls.append(server)
    # run() loads the backend via _load_run_module() (by file path); inject the
    # mock as the cached run module so the stubbed symbols are used.
    monkeypatch.setattr(studio_mod, "_RUN_MODULE", backend)

    # set_tool_policy is imported as `from state.tool_policy import set_tool_policy`.
    state_mod = sys.modules.setdefault("state", types.ModuleType("state"))
    tp_mod = sys.modules.setdefault("state.tool_policy", types.ModuleType("state.tool_policy"))
    tp_mod.set_tool_policy = lambda *a, **k: None
    state_mod.tool_policy = tp_mod

    # Force the health check to fail so startup aborts after run_server().
    monkeypatch.setattr(studio_mod, "_wait_for_server", lambda *a, **k: False)

    import typer as _typer

    app = _typer.Typer()
    app.command(
        context_settings = {"allow_extra_args": True, "ignore_unknown_options": True},
    )(studio_mod.run)
    result = CliRunner().invoke(app, _BASE + ["-H", "0.0.0.0"], catch_exceptions = True)

    assert result.exit_code == 1, result.output
    assert len(shutdown_calls) == 1, "startup abort must call _graceful_shutdown"


def test_run_in_venv_sets_tool_policy_before_server_start(monkeypatch):
    import types

    studio_mod = _studio()
    fake_venv = Path("/fake/studio/venv/unsloth_studio")
    monkeypatch.setattr(sys, "prefix", str(fake_venv))
    monkeypatch.setattr(studio_mod, "STUDIO_HOME", fake_venv.parent)

    from unsloth_cli import _tool_policy as _tp_mod

    monkeypatch.setattr(
        _tp_mod,
        "resolve_tool_policy",
        lambda host, flag, yes, silent: False,
    )

    calls = []

    class _App:
        class state:
            server_port = 8888

    def _run_server(**_kwargs):
        calls.append(("run_server", None))
        return _App()

    backend = types.ModuleType("studio.backend.run")
    backend.run_server = _run_server
    backend._resolve_external_ip = lambda: "1.2.3.4"
    backend._server = object()
    backend._shutdown_event = None
    backend._graceful_shutdown = lambda server: calls.append(("shutdown", server))
    monkeypatch.setitem(sys.modules, "studio.backend.run", backend)
    monkeypatch.setattr(studio_mod, "_RUN_MODULE", backend)

    state_mod = types.ModuleType("state")
    tp_mod = types.ModuleType("state.tool_policy")
    tp_mod.set_tool_policy = lambda value: calls.append(("policy", value))
    state_mod.tool_policy = tp_mod
    monkeypatch.setitem(sys.modules, "state", state_mod)
    monkeypatch.setitem(sys.modules, "state.tool_policy", tp_mod)

    monkeypatch.setattr(studio_mod, "_wait_for_server", lambda *a, **k: False)

    import typer as _typer

    app = _typer.Typer()
    app.command(
        context_settings = {"allow_extra_args": True, "ignore_unknown_options": True},
    )(studio_mod.run)
    result = CliRunner().invoke(app, _BASE + ["--disable-tools"], catch_exceptions = True)

    assert result.exit_code == 1, result.output
    assert calls[:2] == [("policy", False), ("run_server", None)]
