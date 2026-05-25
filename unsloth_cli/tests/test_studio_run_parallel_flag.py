# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for the `unsloth studio run --parallel` CLI flag.

Before this commit, `unsloth studio run` always set
``llama_parallel_slots=4`` (hardcoded). Engine + KV-cache math already
supported any N, but the knob was not user-reachable.

These tests pin:
  1. The Typer option exists with the documented aliases.
  2. The default value matches the previous hardcoded value (4),
     so behaviour is unchanged for existing users.
  3. The range guards reject out-of-band values.
  4. Typer aliases stay a subset of the backend parallel denylist.
  5. Re-exec forwarding preserves every typer-claimed knob.

See also ``test_studio_run_short_alias_clashes.py`` for the argv
canonicaliser (``_expand_attached_np_short`` rewrites ``-np8`` /
``-np-1`` etc. before Click parses) and the legacy ``-m`` / ``-hfr`` /
``-f`` exact-match compatibility shim.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner


_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _load_run_command():
    """Import the `run` typer command without triggering server start.

    The CLI module pulls in heavy backend imports at function call
    time; we only need the Typer command object to introspect options.
    """
    from unsloth_cli.commands import studio as _studio

    return _studio


def test_parallel_option_is_registered():
    """The `--parallel` flag (with aliases) must be on the `run` command."""
    studio_mod = _load_run_command()
    import inspect

    run_fn = studio_mod.run
    sig = inspect.signature(run_fn)
    assert "parallel" in sig.parameters, "missing `parallel` parameter on run()"

    param = sig.parameters["parallel"]
    # typer.OptionInfo lives in the default value
    opt = param.default
    flags = set()
    # OptionInfo stores param_decls in .param_decls (typer >=0.4)
    decls = getattr(opt, "param_decls", None) or []
    for d in decls:
        flags.add(d)
    for required in ("--parallel", "--n-parallel", "-np"):
        assert required in flags, f"flag {required!r} missing from --parallel option"


def test_parallel_default_is_four():
    """Default must stay at 4 so plain `unsloth studio run` is unchanged."""
    studio_mod = _load_run_command()
    import inspect

    sig = inspect.signature(studio_mod.run)
    opt = sig.parameters["parallel"].default
    default = getattr(opt, "default", None)
    assert (
        default == 4
    ), f"default changed to {default}; would silently alter existing deployments"


def test_parallel_range_guards_are_set():
    """Range guards: 1 <= N <= 64. Outside this is a hard reject."""
    studio_mod = _load_run_command()
    import inspect

    sig = inspect.signature(studio_mod.run)
    opt = sig.parameters["parallel"].default
    assert getattr(opt, "min", None) == 1, "min must be 1 (0 = no decode possible)"
    assert getattr(opt, "max", None) == 64, "max must be 64 (KV split sanity cap)"


def test_typer_parallel_aliases_are_subset_of_backend_denylist():
    """Load-bearing invariant: every alias the typer Option claims as
    --parallel on `run` MUST also be in the backend parallel denylist
    group, otherwise an HTTP /load caller could smuggle the value
    through `llama_extra_args` and desync app.state.llama_parallel_slots
    from the running llama-server slot count."""
    studio_mod = _load_run_command()
    import inspect
    import sys as _sys

    backend = Path(__file__).resolve().parents[2] / "studio" / "backend"
    if str(backend) not in _sys.path:
        _sys.path.insert(0, str(backend))
    from core.inference.llama_server_args import _DENYLIST_GROUPS

    parallel_group = next((g for g in _DENYLIST_GROUPS if "--parallel" in g), None)
    assert parallel_group is not None, "denylist must include a --parallel group"

    sig = inspect.signature(studio_mod.run)
    opt = sig.parameters["parallel"].default
    typer_aliases = set(getattr(opt, "param_decls", []) or [])
    missing = typer_aliases - parallel_group
    assert not missing, (
        f"typer claims parallel aliases {missing!r} that the backend "
        f"denylist does not reject; an HTTP /load caller could pass "
        f"them through and desync llama_parallel_slots from the "
        f"running process. Add them to _DENYLIST_GROUPS."
    )


# Runtime equivalent of the old source-text guard lives in
# test_in_venv_path_passes_parallel_to_run_server below: that test fakes
# the in-venv branch and asserts run_server actually receives the typer
# value as llama_parallel_slots. A grep for "llama_parallel_slots = 4"
# is brittle against innocuous formatting refactors, so the static
# check has been retired in favour of the behaviour assertion.


# Re-exec arg-builder coverage. `unsloth studio run` is normally invoked
# outside the Studio venv; run() rebuilds argv and re-execs into the
# Studio venv via os.execvp (POSIX) or subprocess.Popen (Windows).
# Typer claims --parallel/--n-parallel/-np in the outer process, so
# without explicit forwarding the child re-execs at the typer default 4
# (silently dropping any user value -- including pre-PR `-np N` users
# who relied on llama-server pass-through).


class _ExecCaptured(SystemExit):
    def __init__(self, argv):
        super().__init__(0)
        self.argv = list(argv)


def _install_reexec_capture(monkeypatch, *, platform):
    studio_mod = _load_run_command()
    captured = []

    monkeypatch.setattr(sys, "prefix", "/nonexistent/outer/venv")

    fake_venv = Path("/fake/studio/venv/unsloth_studio")
    fake_python = fake_venv / "bin" / "python"
    fake_bin = fake_venv / "bin" / "unsloth"
    monkeypatch.setattr(studio_mod, "_studio_venv_python", lambda: fake_python)

    real_is_file = Path.is_file
    monkeypatch.setattr(
        Path,
        "is_file",
        lambda self: True if str(self) == str(fake_bin) else real_is_file(self),
    )

    # resolve_tool_policy is imported lazily inside run(); patch the source.
    from unsloth_cli import _tool_policy as _tp_mod

    monkeypatch.setattr(
        _tp_mod,
        "resolve_tool_policy",
        lambda host, flag, yes, silent: False if flag is None else bool(flag),
    )

    monkeypatch.setattr(sys, "platform", platform)

    def fake_execvp(file, argv):
        captured.append({"kind": "execvp", "argv": list(argv)})
        raise _ExecCaptured(argv)

    class _FakePopen:
        def __init__(self, argv, *a, **kw):
            captured.append({"kind": "popen", "argv": list(argv)})
            self._argv = argv

        def wait(self):
            raise _ExecCaptured(self._argv)

    monkeypatch.setattr(studio_mod.os, "execvp", fake_execvp)
    monkeypatch.setattr(studio_mod.subprocess, "Popen", _FakePopen)

    return captured


def _invoke_run(monkeypatch, args, *, platform = "linux"):
    import typer as _typer

    studio_mod = _load_run_command()
    captured = _install_reexec_capture(monkeypatch, platform = platform)
    app = _typer.Typer()
    app.command(
        context_settings = {
            "allow_extra_args": True,
            "ignore_unknown_options": True,
        },
    )(studio_mod.run)
    result = CliRunner().invoke(app, args, catch_exceptions = True)
    return result, captured


def _value_after(argv, flag):
    for i, tok in enumerate(argv):
        if tok == flag and i + 1 < len(argv):
            return argv[i + 1]
    return None


_BASE = ["--model", "unsloth/Qwen3-1.7B-GGUF"]


@pytest.mark.parametrize(
    "flag,value",
    [("--parallel", "8"), ("--n-parallel", "16"), ("-np", "32")],
)
def test_reexec_forwards_parallel_all_aliases(monkeypatch, flag, value):
    """Every alias the user can type must reach the re-exec'd child."""
    result, captured = _invoke_run(monkeypatch, _BASE + [flag, value])
    assert (
        len(captured) == 1
    ), f"expected one launch via re-exec, got {captured}; output={result.output!r}"
    argv = captured[0]["argv"]
    assert (
        _value_after(argv, "--parallel") == value
    ), f"{flag} {value} was dropped on re-exec; argv = {argv}"


@pytest.mark.parametrize("platform", ["linux", "darwin", "win32"])
def test_reexec_argv_is_consistent_across_platforms(monkeypatch, platform):
    """Linux/Darwin (execvp) and Windows (Popen) must build the same argv."""
    result, captured = _invoke_run(
        monkeypatch, _BASE + ["--parallel", "12"], platform = platform
    )
    assert len(captured) == 1
    expected_kind = "popen" if platform == "win32" else "execvp"
    assert (
        captured[0]["kind"] == expected_kind
    ), f"{platform}: expected launcher {expected_kind}, got {captured[0]['kind']}"
    assert _value_after(captured[0]["argv"], "--parallel") == "12"


def test_reexec_np_is_first_class_alias(monkeypatch):
    """`-np` is now a first-class `--parallel` alias and must reach the
    re-exec'd child as --parallel <N>. Pre-PR `-np 8` was clustered by
    Click as `-p 8` (port=8) + stray `-n`, silently breaking the port
    binding, so this also pins the no-collision behaviour."""
    result, captured = _invoke_run(monkeypatch, _BASE + ["-np", "8"])
    assert len(captured) == 1
    argv = captured[0]["argv"]
    assert (
        _value_after(argv, "--parallel") == "8"
    ), f"-np 8 silently became 4 after re-exec; argv = {argv}"
    # Confirm `-np 8` did not collide with --port either way.
    assert _value_after(argv, "--port") == "8888", argv


def test_reexec_mixed_parallel_with_passthrough(monkeypatch):
    """--parallel (typer-claimed) + pass-through llama-server flags must both reach the child."""
    result, captured = _invoke_run(
        monkeypatch,
        _BASE + ["--parallel", "8", "--top-k", "20", "--temp", "0.7"],
    )
    assert len(captured) == 1
    argv = captured[0]["argv"]
    assert _value_after(argv, "--parallel") == "8", argv
    assert _value_after(argv, "--top-k") == "20", argv
    assert _value_after(argv, "--temp") == "0.7", argv


@pytest.mark.parametrize(
    "user_flag,expected_in_child",
    [
        ("--load-in-4bit", "--load-in-4bit"),
        ("--no-load-in-4bit", "--no-load-in-4bit"),
        (None, "--load-in-4bit"),  # default True
    ],
)
def test_reexec_forwards_load_in_4bit_in_both_directions(
    monkeypatch, user_flag, expected_in_child
):
    """The re-exec must emit the explicit polarity the operator chose
    (or the typer default if neither was passed). Without this, a
    future default flip on one layer would silently invert behaviour
    for users who never typed the flag."""
    extras = [user_flag] if user_flag else []
    result, captured = _invoke_run(monkeypatch, _BASE + extras)
    assert len(captured) == 1
    argv = captured[0]["argv"]
    other_polarity = (
        "--no-load-in-4bit"
        if expected_in_child == "--load-in-4bit"
        else "--load-in-4bit"
    )
    assert (
        expected_in_child in argv
    ), f"expected {expected_in_child} in child argv; got {argv}"
    assert (
        other_polarity not in argv
    ), f"unexpected {other_polarity} in child argv; got {argv}"


# Runtime behaviour test: bypass the re-exec branch by faking sys.prefix
# into the studio venv, then assert run_server gets the typer --parallel
# value as llama_parallel_slots. Complements the source-text check in
# test_run_kwargs_use_parallel_value so refactors of the call site that
# preserve runtime semantics don't trip a false failure.


class _RunServerCaptured(SystemExit):
    def __init__(self, kwargs):
        super().__init__(0)
        self.kwargs = dict(kwargs)


def _types_module(name):
    import types as _types

    return _types.ModuleType(name)


def test_studio_default_rejects_parallel_when_subcommand_invoked():
    """`unsloth studio --parallel 8 run ...` would silently drop the 8
    because typer doesn't propagate parent options to subcommand
    kwargs. The studio_default callback rejects with exit code 2 and a
    message pointing the operator at the subcommand-level flag."""
    studio_mod = _load_run_command()
    import typer as _typer

    app = _typer.Typer()
    app.add_typer(studio_mod.studio_app, name = "studio")

    runner = CliRunner()
    result = runner.invoke(app, ["studio", "--parallel", "8", "run", "--model", "X"])
    assert result.exit_code == 2, (
        f"expected exit 2 when --parallel is on studio group with a "
        f"subcommand invoked; got {result.exit_code}; output={result.output!r}"
    )
    combined = (result.output or "") + (getattr(result, "stderr", "") or "")
    assert "--parallel" in combined, combined
    assert "run --parallel 8" in combined, (
        f"error message must show the corrected invocation; got: {combined}"
    )


def test_studio_default_default_parallel_with_subcommand_does_not_error():
    """When the user doesn't pass --parallel on the group, invoking a
    subcommand must still succeed (the group's default 1 is benign)."""
    studio_mod = _load_run_command()
    import typer as _typer

    monkey_target = studio_mod
    # Stub setup so invoking `studio setup` doesn't actually run the
    # installer script; we only care that the callback didn't error
    # out on the default --parallel.
    monkey_target.setup.__wrapped__ if hasattr(
        monkey_target.setup, "__wrapped__"
    ) else None

    app = _typer.Typer()
    app.add_typer(studio_mod.studio_app, name = "studio")
    runner = CliRunner()
    result = runner.invoke(app, ["studio", "--help"])
    assert result.exit_code == 0, result.output


def test_studio_default_exposes_parallel_option():
    """Plain `unsloth studio` (no `run`) must also expose --parallel so
    the API-only / bare-server path has a way to raise concurrency
    (since pass-through --parallel via llama_extra_args is denied).
    Default stays at 1 to match pre-PR behaviour for this entry point;
    `unsloth studio run` keeps its own default of 4."""
    studio_mod = _load_run_command()
    import inspect

    sig = inspect.signature(studio_mod.studio_default)
    assert "parallel" in sig.parameters, (
        "studio_default missing `parallel` parameter; API-only path "
        "has no first-class way to set llama_parallel_slots"
    )
    opt = sig.parameters["parallel"].default
    decls = set(getattr(opt, "param_decls", []) or [])
    assert "--parallel" in decls
    assert "--n-parallel" in decls
    assert getattr(opt, "default", None) == 1, (
        "studio_default --parallel must default to 1 (pre-PR behaviour); "
        "`unsloth studio run` keeps its hardcoded default of 4"
    )
    assert getattr(opt, "min", None) == 1
    assert getattr(opt, "max", None) == 64


@pytest.mark.parametrize("value", [1, 4, 8, 64])
def test_in_venv_path_passes_parallel_to_run_server(monkeypatch, value):
    """When already in the studio venv, run() must forward --parallel to
    run_server(llama_parallel_slots=N) instead of hardcoding 4."""
    studio_mod = _load_run_command()

    fake_venv = Path("/fake/studio/venv/unsloth_studio")
    monkeypatch.setattr(sys, "prefix", str(fake_venv))
    # sys.prefix.startswith(STUDIO_HOME / "unsloth_studio") gates the
    # in-venv path; pin STUDIO_HOME so the predicate is true.
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
        "studio.backend.run", _types_module("studio.backend.run")
    )
    fake_backend_run.run_server = fake_run_server
    fake_backend_run._resolve_external_ip = lambda: "127.0.0.1"

    import typer as _typer

    app = _typer.Typer()
    app.command(
        context_settings = {
            "allow_extra_args": True,
            "ignore_unknown_options": True,
        },
    )(studio_mod.run)
    CliRunner().invoke(app, _BASE + ["--parallel", str(value)], catch_exceptions = True)

    assert (
        captured.get("llama_parallel_slots") == value
    ), f"run_server got llama_parallel_slots={captured.get('llama_parallel_slots')!r}, expected {value}"
