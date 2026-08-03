# SPDX-License-Identifier: AGPL-3.0-only
"""Generic Triton must not shadow torch's XPU Triton.

Both distributions own the top-level ``triton`` package, and resolving unsloth against a
pinned ``+xpu`` torch pulls both (uv reports ``pytorch-triton-xpu 3.5.0`` alongside
``triton 3.7.1``), so the CUDA-oriented build can land last and ``torch.compile`` then loads
the wrong library on an Intel GPU.

The swap lives in ``install_python_stack.py`` rather than ``install.sh`` because install.sh
runs setup.sh, which runs this module: one copy covers the fresh install and
``unsloth studio update``, which never touches install.sh.

Two things here are easy to get wrong and are asserted by execution rather than by reading:

* the ORDER. Fetch, then uninstall, then install. Uninstalling last deletes the shared paths
  the XPU build just wrote, because those paths are in generic triton's own RECORD.
* the venv has no pip. ``uv venv`` is created without ``--seed``, so a fresh venv cannot run
  ``pip download`` at all, and without a bootstrap the swap silently never happens.
"""

import subprocess
import sys
import types
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[2]
STACK = REPO / "studio/install_python_stack.py"


def _load(monkeypatch, *, spec, generic, has_pip = True, ensurepip_works = True,
          download_ok = True, drops_wheel = True):
    """Import the module with the world stubbed, and return (module, action log)."""
    log: list[str] = []

    mod = types.ModuleType("_stack_under_test")
    src = STACK.read_text(encoding = "utf-8")
    # Only the three helpers are needed; importing the whole module would run the installer.
    start = src.index("def _ensure_venv_pip() -> bool:")
    end = src.index("def _ensure_cpu_torch() -> None:")
    body = src[start:end]
    assert "_ensure_xpu_triton" in body, "extraction lost the swap"

    import glob as _glob
    import os as _os
    import shutil as _shutil
    import tempfile as _tempfile

    pip_state = {"present": has_pip}

    def fake_run(cmd, **kw):
        joined = " ".join(str(c) for c in cmd)
        if "-m pip --version" in joined or ("pip" in cmd and "--version" in cmd):
            return subprocess.CompletedProcess(cmd, 0 if pip_state["present"] else 1)
        if "ensurepip" in joined:
            log.append("ENSUREPIP")
            if ensurepip_works:
                pip_state["present"] = True
            return subprocess.CompletedProcess(cmd, 0)
        if "importlib.metadata" in joined:
            out = f"SPEC={spec}\nGENERIC={generic}\n".encode()
            return subprocess.CompletedProcess(cmd, 0, stdout = out)
        if "download" in cmd:
            log.append("DOWNLOAD")
            if download_ok and drops_wheel:
                target = cmd[cmd.index("-d") + 1]
                Path(target, "pytorch_triton_xpu-3.5.0-py3-none-any.whl").write_bytes(b"")
            return subprocess.CompletedProcess(cmd, 0 if download_ok else 1, stdout = b"")
        if "uninstall" in cmd:
            log.append("UNINSTALL")
            return subprocess.CompletedProcess(cmd, 0)
        return subprocess.CompletedProcess(cmd, 0, stdout = b"")

    def fake_pip_install_try(label, *args, **kw):
        if label.startswith("pip"):
            log.append("BOOTSTRAP")
            if ensurepip_works:
                pip_state["present"] = True
            return pip_state["present"]
        log.append("INSTALL")
        return True

    ns = {
        "subprocess": types.SimpleNamespace(
            run = fake_run,
            CompletedProcess = subprocess.CompletedProcess,
            TimeoutExpired = subprocess.TimeoutExpired,
            DEVNULL = subprocess.DEVNULL,
            PIPE = subprocess.PIPE,
            STDOUT = subprocess.STDOUT,
        ),
        "sys": sys, "glob": _glob, "os": _os, "shutil": _shutil, "tempfile": _tempfile,
        "NO_TORCH": False, "IS_MACOS": False, "IS_WINDOWS": False,
        "_explicit_xpu_torch_index_url": lambda: "https://download.pytorch.org/whl/xpu",
        "pip_install_try": fake_pip_install_try,
        "_red": lambda s: s,
        "print": lambda *a, **k: log.append("WARN") if a and "left in place" in str(a[0]) else None,
    }
    exec(compile(body, str(STACK), "exec"), ns)
    mod.__dict__.update(ns)
    return mod, log


def _run(monkeypatch, **kw):
    mod, log = _load(monkeypatch, **kw)
    mod.__dict__["_ensure_xpu_triton"]()
    return log


class TestXpuTritonSwap:
    def test_orders_fetch_uninstall_install(self, monkeypatch):
        # The whole point: the uninstall sits between the fetch and the install.
        log = _run(monkeypatch, spec = "pytorch-triton-xpu==3.5.0", generic = "3.7.1")
        assert log == ["DOWNLOAD", "UNINSTALL", "INSTALL"]

    def test_handles_the_triton_xpu_rename(self, monkeypatch):
        # torch 2.10 renamed the distribution; the spec is read from torch, never hardcoded.
        log = _run(monkeypatch, spec = "triton-xpu==3.6.0", generic = "3.7.1")
        assert log == ["DOWNLOAD", "UNINSTALL", "INSTALL"]

    def test_bootstraps_pip_when_the_venv_has_none(self, monkeypatch):
        # uv venv has no --seed, so a fresh venv cannot run pip download at all.
        log = _run(monkeypatch, spec = "pytorch-triton-xpu==3.5.0", generic = "3.7.1",
                   has_pip = False)
        assert log[0] == "ENSUREPIP"
        assert log[-3:] == ["DOWNLOAD", "UNINSTALL", "INSTALL"]

    def test_falls_back_to_installing_pip(self, monkeypatch):
        log = _run(monkeypatch, spec = "pytorch-triton-xpu==3.5.0", generic = "3.7.1",
                   has_pip = False, ensurepip_works = False)
        # ensurepip failed, so it tries a real pip install; that fails too here, and the swap
        # must warn rather than uninstall with nothing to install from.
        assert "BOOTSTRAP" in log
        assert "UNINSTALL" not in log

    @pytest.mark.parametrize(
        "spec, generic",
        [
            ("pytorch-triton-xpu==3.5.0", ""),   # nothing shadowing it
            ("triton==3.7.1", "3.7.1"),          # torch is not the +xpu wheel
            ("", "3.7.1"),                       # torch declares no triton at all
        ],
    )
    def test_leaves_a_healthy_venv_alone(self, monkeypatch, spec, generic):
        assert _run(monkeypatch, spec = spec, generic = generic) == []

    def test_a_dead_mirror_removes_nothing(self, monkeypatch):
        # Warn and leave the venv working; never uninstall with nothing to install from.
        log = _run(monkeypatch, spec = "pytorch-triton-xpu==3.5.0", generic = "3.7.1",
                   download_ok = False)
        assert "UNINSTALL" not in log and "INSTALL" not in log

    def test_a_successful_exit_with_no_wheel_removes_nothing(self, monkeypatch):
        # The exit code alone is not enough: no wheel on disk means nothing to install from.
        log = _run(monkeypatch, spec = "pytorch-triton-xpu==3.5.0", generic = "3.7.1",
                   drops_wheel = False)
        assert "UNINSTALL" not in log and "INSTALL" not in log


class TestPlatformGuards:
    @pytest.mark.parametrize("flag", ["NO_TORCH", "IS_MACOS", "IS_WINDOWS"])
    def test_skipped_where_it_does_not_apply(self, monkeypatch, flag):
        # Windows is setup.ps1's job; macOS has no XPU; --no-torch touches no wheels.
        mod, log = _load(monkeypatch, spec = "pytorch-triton-xpu==3.5.0", generic = "3.7.1")
        mod.__dict__[flag] = True
        mod.__dict__["_ensure_xpu_triton"].__globals__[flag] = True
        mod.__dict__["_ensure_xpu_triton"]()
        assert log == []


def test_the_swap_is_wired_in_at_both_repair_points():
    # The final repair pass would otherwise silently undo the first.
    src = STACK.read_text(encoding = "utf-8")
    assert src.count("        _ensure_xpu_triton()") == 2


def test_install_sh_does_not_carry_a_second_copy():
    # It used to. install.sh runs setup.sh, which runs this module, so a copy there is both
    # redundant and a place for the two to drift apart.
    assert "replace generic Triton" not in (REPO / "install.sh").read_text(encoding = "utf-8")
