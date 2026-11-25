# SPDX-License-Identifier: AGPL-3.0-only

"""Per-call work a sandboxed tool launch used to repeat; each test fails on the code before the fix."""

from __future__ import annotations

import importlib.util
import os
import py_compile
import sys

import pytest

from core.inference import tools


def _shim() -> str:
    return os.path.join(tools._SANDBOX_SITE_DIR, "sitecustomize.py")


def test_the_sandbox_shim_has_bytecode_before_a_child_starts(monkeypatch, tmp_path):
    # The sandboxes mount sandbox_site read-only, so only Studio can cache the shim's bytecode.
    monkeypatch.setattr(sys, "dont_write_bytecode", False)
    cfile = importlib.util.cache_from_source(_shim())
    try:
        os.unlink(cfile)
    except FileNotFoundError:
        pass
    monkeypatch.setattr(tools, "_sandbox_site_bytecode_for", None)
    tools._build_safe_env(str(tmp_path))
    assert os.path.isfile(cfile)
    assert tools._pyc_matches_source(cfile, os.stat(_shim()))


def test_the_shim_is_compiled_once_per_version(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "dont_write_bytecode", False)
    monkeypatch.setattr(tools, "_sandbox_site_bytecode_for", None)
    tools._build_safe_env(str(tmp_path))
    compiled = []
    real = py_compile.compile
    monkeypatch.setattr(py_compile, "compile", lambda *a, **k: compiled.append(a) or real(*a, **k))
    for _ in range(5):
        tools._build_safe_env(str(tmp_path))
    assert compiled == []


def test_a_stale_shim_pyc_is_rewritten(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "dont_write_bytecode", False)
    cfile = importlib.util.cache_from_source(_shim())
    os.makedirs(os.path.dirname(cfile), exist_ok = True)
    with open(cfile, "wb") as stream:
        stream.write(importlib.util.MAGIC_NUMBER + b"\0" * 12)  # wrong mtime and size
    assert not tools._pyc_matches_source(cfile, os.stat(_shim()))
    monkeypatch.setattr(tools, "_sandbox_site_bytecode_for", None)
    tools._build_safe_env(str(tmp_path))
    assert tools._pyc_matches_source(cfile, os.stat(_shim()))


@pytest.mark.parametrize(
    ("name", "servable"),
    [
        ("report.csv", True),
        ("été.txt", True),
        ("\U0001f600.png", True),
        ("x\ud800y", False),
        ("é\udfff", False),
        ("..", False),
        ("a/b", False),
        ("tab\there", False),
    ],
)
def test_servable_segment_keeps_its_rules(name, servable):
    assert tools._servable_segment(name) is servable


@pytest.mark.skipif(sys.platform != "linux", reason = "the bubblewrap runtime binds are Linux only")
def test_runtime_read_paths_resolve_each_candidate_once(monkeypatch, tmp_path):
    from core.inference import sandbox_linux

    prefix = sys.prefix
    for name in ("base_prefix", "exec_prefix", "base_exec_prefix"):
        monkeypatch.setattr(sys, name, prefix)
    target = os.path.join(prefix, "bin")
    seen = []
    real = os.path.realpath
    monkeypatch.setattr(
        os.path,
        "realpath",
        lambda p, *a, **k: (seen.append(p) if p == target else None) or real(p, *a, **k),
    )
    sandbox_linux._runtime_read_paths(str(tmp_path), ())
    # Four prefixes spell the same venv; resolving its bin/ four times bought nothing.
    assert seen.count(target) == 1
