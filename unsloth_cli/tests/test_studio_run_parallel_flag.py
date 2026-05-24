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


def test_run_kwargs_use_parallel_value(monkeypatch):
    """run_kwargs passed to run_server must reflect the --parallel value
    (no hardcoded 4 remaining after the flag was added)."""
    studio_mod = _load_run_command()
    import textwrap

    src = textwrap.dedent(Path(studio_mod.__file__).read_text())
    # Pre-fix line was: run_kwargs = dict(... llama_parallel_slots = 4)
    assert "llama_parallel_slots = 4" not in src, (
        "found hardcoded `llama_parallel_slots = 4` after the parallel "
        "flag landed -- run_kwargs must pull from the typer option"
    )
    assert (
        "llama_parallel_slots = parallel" in src
    ), "run_kwargs must use the parallel variable from the typer option"


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
    result = CliRunner(mix_stderr = False).invoke(app, args, catch_exceptions = True)
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


def test_reexec_pre_pr_np_passthrough_regression(monkeypatch):
    """Pre-PR `-np 8` worked because typer ignored it and llama-server
    last-wins parsing made it stick. Post-PR typer claims `-np`; without
    forwarding the child silently boots with 4."""
    result, captured = _invoke_run(monkeypatch, _BASE + ["-np", "8"])
    assert len(captured) == 1
    assert _value_after(captured[0]["argv"], "--parallel") == "8", (
        "REGRESSION: -np 8 silently became 4 after re-exec; argv = "
        f"{captured[0]['argv']}"
    )


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
