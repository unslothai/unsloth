# SPDX-License-Identifier: AGPL-3.0-only

"""Per-call work a sandboxed tool launch used to repeat; each test fails on the code before the fix."""

from __future__ import annotations

import importlib.util
import os
import py_compile
import subprocess
import sys

import pytest

from core.inference import tools


@pytest.fixture
def shim(monkeypatch, tmp_path):
    """A private copy of the shim, so parallel workers never share one cache file."""
    site_dir = tmp_path / "sandbox_site"
    site_dir.mkdir()
    with open(os.path.join(tools._SANDBOX_SITE_DIR, "sitecustomize.py"), "rb") as stream:
        (site_dir / "sitecustomize.py").write_bytes(stream.read())
    monkeypatch.setattr(tools, "_SANDBOX_SITE_DIR", str(site_dir))
    monkeypatch.setattr(tools, "_sandbox_site_bytecode_for", None)
    monkeypatch.setattr(sys, "dont_write_bytecode", False)
    return str(site_dir / "sitecustomize.py")


def _matches(shim) -> bool:
    with open(shim, "rb") as stream:
        return tools._pyc_matches_source(tools._sandbox_site_pyc(shim), stream.read())


def test_the_sandbox_shim_has_bytecode_before_a_child_starts(shim, tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    tools._build_safe_env(str(work))
    assert os.path.isfile(tools._sandbox_site_pyc(shim))
    assert _matches(shim)


def test_the_shim_is_compiled_once_per_version(shim, tmp_path, monkeypatch):
    compiled = []
    real = py_compile.compile
    monkeypatch.setattr(py_compile, "compile", lambda *a, **k: compiled.append(a) or real(*a, **k))
    for _ in range(5):
        tools._build_safe_env(str(tmp_path))
    assert len(compiled) == 1


def test_the_pyc_lands_where_a_default_child_looks(shim, monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "pycache_prefix", str(tmp_path / "elsewhere"))
    tools._ensure_sandbox_site_bytecode()
    tag = sys.implementation.cache_tag
    expected = os.path.join(os.path.dirname(shim), "__pycache__", f"sitecustomize.{tag}.pyc")
    assert os.path.isfile(expected)
    assert not (tmp_path / "elsewhere").exists()


def test_a_stale_shim_pyc_is_rewritten(shim, tmp_path):
    cfile = tools._sandbox_site_pyc(shim)
    os.makedirs(os.path.dirname(cfile), exist_ok = True)
    with open(cfile, "wb") as stream:
        stream.write(
            importlib.util.MAGIC_NUMBER + b"\0" * 12
        )  # timestamp mode, wrong mtime and size
    assert not _matches(shim)
    tools._build_safe_env(str(tmp_path))
    assert _matches(shim)


def test_a_same_size_shim_edit_never_runs_the_old_bytecode(shim, tmp_path):
    # Timestamp pycs keep whole seconds: a same-size edit inside one second would load stale code.
    with open(shim, "w") as stream:
        stream.write("VALUE = 'A'\n")
    second = 1_900_000_000 * 10**9
    os.utime(shim, ns = (second, second))
    tools._ensure_sandbox_site_bytecode()
    with open(shim, "w") as stream:
        stream.write("VALUE = 'B'\n")
    os.utime(shim, ns = (second, second + 500_000_000))
    tools._ensure_sandbox_site_bytecode()
    probe = subprocess.run(
        [sys.executable, "-B", "-c", "import sitecustomize; print(sitecustomize.VALUE)"],
        env = {**os.environ, "PYTHONPATH": os.path.dirname(shim)},
        capture_output = True,
        text = True,
        check = True,
    )
    assert probe.stdout.strip() == "B"


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
    assert seen.count(target) == 1


def test_one_launch_runs_inside_one_scan_memo_scope(monkeypatch):
    from core.inference import os_sandbox
    from utils.scan_memo import scan_memo_active

    seen = []
    monkeypatch.setattr(
        os_sandbox, "_prepare_tool_launch", lambda plan: seen.append(scan_memo_active())
    )
    os_sandbox.prepare_tool_launch(object())
    os_sandbox.prepare_tool_launch(object())
    # Active while the policy is built, and closed between launches so nothing carries over.
    assert seen == [True, True]
    assert not scan_memo_active()
