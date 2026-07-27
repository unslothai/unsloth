# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for the `--cloudflare/--no-cloudflare` Unsloth flag.

Pins the typer Option (tri-state, default off / None) on both `unsloth studio`
and `unsloth studio run`, and that the chosen polarity reaches the re-exec'd
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


def test_run_exposes_cloudflare_option_default_off():
    import inspect

    sig = inspect.signature(_studio().run)
    assert "cloudflare" in sig.parameters
    opt = sig.parameters["cloudflare"].default
    decls = set(getattr(opt, "param_decls", []) or [])
    assert "--cloudflare/--no-cloudflare" in decls
    assert getattr(opt, "default", "missing") is None


def test_studio_default_exposes_cloudflare_option_default_off():
    import inspect

    sig = inspect.signature(_studio().studio_default)
    assert "cloudflare" in sig.parameters
    opt = sig.parameters["cloudflare"].default
    assert getattr(opt, "default", "missing") is None


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
    "extra_flags,expected,unexpected",
    [
        # Default (no flag) forwards --no-cloudflare explicitly so a mixed-version
        # child venv (old default: --cloudflare on) can't re-enable the tunnel.
        ([], "--no-cloudflare", "--cloudflare"),
        (["--cloudflare"], "--cloudflare", "--no-cloudflare"),
        (["--no-cloudflare"], "--no-cloudflare", "--cloudflare"),
        # --secure implies the tunnel; never forward --no-cloudflare with it.
        (["--secure"], None, "--no-cloudflare"),
    ],
)
def test_run_reexec_forwards_cloudflare_polarity(monkeypatch, extra_flags, expected, unexpected):
    captured = _invoke_run(monkeypatch, _BASE + extra_flags)
    assert len(captured) == 1, captured
    argv = captured[0]
    if expected is not None:
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
    # A built frontend dist is present so the public-launch UI check passes; this
    # suite exercises flag forwarding, not the missing-dist lockout guard.
    monkeypatch.setattr(
        studio_mod, "_find_frontend_dist", lambda: Path("/fake/studio/frontend/dist")
    )
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
    "extra_flags,expected,unexpected",
    [
        # Default (no flag) forwards --no-cloudflare explicitly: _find_run_py can fall
        # back to an older studio-venv run.py (default on), so a mixed install must
        # not re-enable the tunnel.
        ([], "--no-cloudflare", "--cloudflare"),
        (["--cloudflare"], "--cloudflare", "--no-cloudflare"),
        (["--no-cloudflare"], "--no-cloudflare", "--cloudflare"),
        # --secure implies the tunnel; never forward --no-cloudflare with it.
        (["--secure"], None, "--no-cloudflare"),
    ],
)
def test_studio_default_reexec_forwards_cloudflare(monkeypatch, extra_flags, expected, unexpected):
    captured = _invoke_studio_default(monkeypatch, ["-H", "0.0.0.0"] + extra_flags)
    assert len(captured) == 1, captured
    argv = captured[0]
    if expected is not None:
        assert expected in argv, f"expected {expected}; got {argv}"
    assert unexpected not in argv, f"unexpected {unexpected}; got {argv}"


# ── in-venv path forwards cloudflare into run_server ─────────────────


class _RunServerCaptured(SystemExit):
    def __init__(self, kwargs):
        super().__init__(0)
        self.kwargs = dict(kwargs)


@pytest.mark.parametrize(
    "user_flag,expected",
    [(None, None), ("--cloudflare", True), ("--no-cloudflare", False)],
)
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


def test_run_display_host_and_url_helpers_cover_ipv6_wildcard():
    import types

    studio_mod = _studio()
    run_mod = types.SimpleNamespace(_resolve_external_ip = lambda: "198.51.100.7")

    assert studio_mod._display_host_for_bind(run_mod, "0.0.0.0") == "198.51.100.7"
    assert studio_mod._display_host_for_bind(run_mod, "::") == "198.51.100.7"
    assert studio_mod._url_host("2001:db8::7") == "[2001:db8::7]"
    assert studio_mod._url_host("127.0.0.1") == "127.0.0.1"


def test_run_cloudflare_notice_uses_external_host_policy():
    import types

    studio_mod = _studio()
    calls = []
    run_mod = types.SimpleNamespace(
        _verify_global_reachability = lambda host, port: calls.append(("verify", host, port)),
        _print_cloudflare_line = lambda **kw: calls.append(("print", kw)),
    )

    studio_mod._emit_run_cloudflare_notice(run_mod, "0.0.0.0", "198.51.100.7", 8888, False)
    assert calls == [
        ("verify", "198.51.100.7", 8888),
        ("print", {"secure": False, "loopback_host": "127.0.0.1"}),
    ]

    calls.clear()
    studio_mod._emit_run_cloudflare_notice(run_mod, "::", "198.51.100.7", 8888, False)
    assert calls == [
        ("verify", "198.51.100.7", 8888),
        ("print", {"secure": False, "loopback_host": "::1"}),
    ]

    calls.clear()
    studio_mod._emit_run_cloudflare_notice(run_mod, "127.0.0.1", "127.0.0.1", 8888, False)
    assert calls == []


def test_run_silent_emits_cloudflare_notice_for_external_bind(monkeypatch):
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

    class _Done(SystemExit):
        pass

    class _ShutdownEvent:
        def is_set(self):
            raise _Done(0)

        def wait(self, timeout = None):
            return None

    class _App:
        class state:
            server_port = 8888
            cloudflare_url = "https://x.trycloudflare.com"

    calls = []
    backend = types.ModuleType("studio.backend.run")
    backend.run_server = lambda **_kwargs: _App()
    backend._resolve_external_ip = lambda: "198.51.100.7"
    backend._verify_global_reachability = lambda host, port: calls.append(("verify", host, port))
    backend._print_cloudflare_line = lambda **kw: calls.append(("print", kw))
    backend._server = object()
    backend._shutdown_event = _ShutdownEvent()
    backend._graceful_shutdown = lambda server: calls.append(("shutdown", server))
    monkeypatch.setitem(sys.modules, "studio.backend.run", backend)
    monkeypatch.setattr(studio_mod, "_RUN_MODULE", backend)

    state_mod = types.ModuleType("state")
    tp_mod = types.ModuleType("state.tool_policy")
    tp_mod.set_tool_policy = lambda *a, **k: None
    state_mod.tool_policy = tp_mod
    monkeypatch.setitem(sys.modules, "state", state_mod)
    monkeypatch.setitem(sys.modules, "state.tool_policy", tp_mod)

    monkeypatch.setattr(studio_mod, "_wait_for_server", lambda *a, **k: True)
    monkeypatch.setattr(studio_mod, "_create_api_key_inprocess", lambda name: "sk-test")
    monkeypatch.setattr(
        studio_mod,
        "_load_model_via_http",
        lambda **_kwargs: {"model": "unsloth/Qwen3-1.7B-GGUF", "context_length": 4096},
    )

    import typer as _typer

    app = _typer.Typer()
    app.command(
        context_settings = {"allow_extra_args": True, "ignore_unknown_options": True},
    )(studio_mod.run)
    result = CliRunner().invoke(
        app,
        _BASE + ["--silent", "-H", "0.0.0.0"],
        catch_exceptions = True,
    )

    assert result.exit_code == 0, result.output
    assert ("verify", "198.51.100.7", 8888) in calls
    assert ("print", {"secure": False, "loopback_host": "127.0.0.1"}) in calls


# ── parent-level --cloudflare/--no-cloudflare with a subcommand is rejected ─


@pytest.mark.parametrize("flag", ["--cloudflare", "--no-cloudflare"])
def test_studio_default_rejects_cloudflare_flag_with_subcommand(monkeypatch, flag):
    # `unsloth studio --cloudflare run ...` (or --no-cloudflare) would not reach the
    # subcommand, so it must error (mirrors --parallel) rather than silently drop it.
    import typer as _typer

    studio_mod = _studio()
    app = _typer.Typer()
    app.add_typer(studio_mod.studio_app, name = "studio")
    result = CliRunner().invoke(app, ["studio", flag, "run", "--model", "X"])
    assert result.exit_code == 2, result.output
    combined = (result.output or "") + (getattr(result, "stderr", "") or "")
    assert flag in combined, combined


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
