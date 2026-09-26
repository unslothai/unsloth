# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""`unsloth run` must not re-exec itself forever when the studio venv is reached through a symlink."""

from __future__ import annotations

import os
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


def _symlinked_venv(tmp_path):
    real = tmp_path / "disk2" / "unsloth_studio"
    real.mkdir(parents = True)
    home = tmp_path / "home"
    home.mkdir()
    link = home / "unsloth_studio"
    try:
        link.symlink_to(real, target_is_directory = True)
    except (OSError, NotImplementedError) as e:
        pytest.skip(f"symlinks unavailable here: {e}")
    return real, link


def test_resolved_prefix_counts_as_symlinked_venv(tmp_path, monkeypatch):
    studio_mod = _studio()
    real, link = _symlinked_venv(tmp_path)
    monkeypatch.setattr(sys, "prefix", str(real))
    assert studio_mod._running_in_studio_venv(link)
    assert not sys.prefix.startswith(str(link))


def test_link_prefix_counts_as_real_venv(tmp_path, monkeypatch):
    studio_mod = _studio()
    real, link = _symlinked_venv(tmp_path)
    monkeypatch.setattr(sys, "prefix", str(link))
    assert studio_mod._running_in_studio_venv(real)
    assert studio_mod._running_in_studio_venv(link)


def test_sibling_with_shared_name_prefix_is_not_the_venv(tmp_path, monkeypatch):
    studio_mod = _studio()
    venv = tmp_path / "unsloth_studio"
    sibling = tmp_path / "unsloth_studio2"
    venv.mkdir()
    sibling.mkdir()
    monkeypatch.setattr(sys, "prefix", str(sibling))
    assert not studio_mod._running_in_studio_venv(venv)


def test_prefix_inside_venv_and_unrelated_prefix(tmp_path, monkeypatch):
    studio_mod = _studio()
    venv = tmp_path / "unsloth_studio"
    (venv / "sub").mkdir(parents = True)
    monkeypatch.setattr(sys, "prefix", str(venv / "sub"))
    assert studio_mod._running_in_studio_venv(venv)
    monkeypatch.setattr(sys, "prefix", str(tmp_path / "elsewhere"))
    assert not studio_mod._running_in_studio_venv(venv)


def test_missing_paths_do_not_raise(tmp_path, monkeypatch):
    studio_mod = _studio()
    monkeypatch.setattr(sys, "prefix", str(tmp_path / "gone" / "venv"))
    assert not studio_mod._running_in_studio_venv(tmp_path / "also_gone" / "unsloth_studio")


def test_symlink_loop_does_not_raise(tmp_path, monkeypatch):
    studio_mod = _studio()

    def _loop(self, *a, **k):
        raise RuntimeError(f"Symlink loop from {self!r}")

    monkeypatch.setattr(Path, "resolve", _loop)
    monkeypatch.setattr(sys, "prefix", str(tmp_path / "outer"))
    assert not studio_mod._running_in_studio_venv(tmp_path / "unsloth_studio")
    monkeypatch.setattr(sys, "prefix", str(tmp_path / "unsloth_studio"))
    assert studio_mod._running_in_studio_venv(tmp_path / "unsloth_studio")


class _Execd(SystemExit):
    def __init__(self):
        super().__init__(0)


def _run_app(
    monkeypatch,
    *,
    prefix,
    guard_env = None,
    studio_home = None,
):
    """Drive run() up to its re-exec with a fake venv that the prefix is NOT inside."""
    import typer

    studio_mod = _studio()
    monkeypatch.setattr(sys, "prefix", prefix)
    if studio_home is not None:
        monkeypatch.setattr(studio_mod, "STUDIO_HOME", studio_home)
    fake_venv = (studio_home or Path("/fake/studio/venv")) / "unsloth_studio"
    fake_python = fake_venv / "bin" / "python"
    fake_bin = fake_python.parent / "unsloth"
    monkeypatch.setattr(studio_mod, "_studio_venv_python", lambda: fake_python)
    real_is_file = Path.is_file
    monkeypatch.setattr(
        Path, "is_file", lambda self: True if str(self) == str(fake_bin) else real_is_file(self)
    )
    from unsloth_cli import _tool_policy as _tp_mod

    monkeypatch.setattr(_tp_mod, "resolve_tool_policy", lambda host, flag, yes, silent: False)
    monkeypatch.setattr(sys, "platform", "linux")
    execs = []

    def fake_execvp(file, argv):
        execs.append((list(argv), os.environ.get(studio_mod._STUDIO_REEXEC_ENV)))
        raise _Execd()

    monkeypatch.setattr(studio_mod.os, "execvp", fake_execvp)
    if guard_env is None:
        monkeypatch.delenv(studio_mod._STUDIO_REEXEC_ENV, raising = False)
    else:
        monkeypatch.setenv(studio_mod._STUDIO_REEXEC_ENV, guard_env)
    app = typer.Typer()
    app.command(context_settings = {"allow_extra_args": True, "ignore_unknown_options": True})(
        studio_mod.run
    )
    result = CliRunner().invoke(app, ["--model", "unsloth/Qwen3-1.7B-GGUF"], catch_exceptions = True)
    return studio_mod, result, execs


def test_reexec_marks_the_child(monkeypatch):
    studio_mod, result, execs = _run_app(monkeypatch, prefix = "/nonexistent/outer/venv")
    assert len(execs) == 1, result.output
    argv, guard = execs[0]
    assert argv[1:3] == ["studio", "run"]
    assert guard == "1", "the re-exec'd child must see the marker"
    assert studio_mod._STUDIO_REEXEC_ENV not in os.environ, "the parent must not keep it"


def test_marked_child_outside_the_venv_stops_instead_of_looping(monkeypatch):
    studio_mod, result, execs = _run_app(
        monkeypatch, prefix = "/nonexistent/outer/venv", guard_env = "1"
    )
    assert execs == [], "a marked child must not re-exec again"
    assert result.exit_code == 1
    assert "not re-launching again" in result.output
    assert studio_mod._STUDIO_REEXEC_ENV not in os.environ


_posix_exec_only = pytest.mark.skipif(os.name == "nt", reason = "POSIX exec branch")


@_posix_exec_only
def test_unlinked_venv_still_execs_the_console_script(monkeypatch):
    _, result, execs = _run_app(monkeypatch, prefix = "/nonexistent/outer/venv")
    assert len(execs) == 1, result.output
    assert execs[0][0][0] == "/fake/studio/venv/unsloth_studio/bin/unsloth"


@_posix_exec_only
def test_symlinked_venv_execs_through_the_linked_interpreter(tmp_path, monkeypatch):
    # An older venv CLI would loop via the console script (resolved sys.prefix).
    studio_mod = _studio()
    real, link = _symlinked_venv(tmp_path)
    _, result, execs = _run_app(
        monkeypatch, prefix = "/nonexistent/outer/venv", studio_home = link.parent
    )
    assert len(execs) == 1, result.output
    argv = execs[0][0]
    assert argv[:3] == [str(link / "bin" / "python"), "-c", studio_mod._WINDOWS_CLI_ENTRYPOINT]
    assert argv[3:5] == ["studio", "run"]
