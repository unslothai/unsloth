# SPDX-License-Identifier: AGPL-3.0-only
"""Generic Triton must not shadow torch's XPU Triton.

Both distributions own the top-level ``triton`` package, and resolving unsloth against a
pinned ``+xpu`` torch pulls both (uv reports ``pytorch-triton-xpu 3.5.0`` alongside
``triton 3.7.1``), so the CUDA-oriented build can land last and ``torch.compile`` then loads
the wrong library on an Intel GPU.

The swap lives in ``install_python_stack.py`` rather than ``install.sh`` because install.sh
runs setup.sh, which runs this module: one copy covers the fresh install and
``unsloth studio update``, which never touches install.sh.

Three things here are easy to get wrong and are asserted by execution rather than by reading:

* the ORDER. Fetch, then uninstall, then install. Uninstalling last deletes the shared paths
  the XPU build just wrote, because those paths are in generic triton's own RECORD.
* the venv has no pip. ``uv venv`` is created without ``--seed``, so a fresh venv cannot run
  ``pip download`` at all, and without a bootstrap the swap silently never happens.
* the pin is ONE-SHOT. ``UNSLOTH_TORCH_INDEX_FAMILY=xpu ./install.sh`` leaves nothing behind
  in the environment, so a later plain ``unsloth studio update`` must recognise the installed
  ``+xpu`` wheel instead. See TestTheInstalledWheelIsThePin.
"""

import subprocess
import sys
import types
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[2]
STACK = REPO / "studio/install_python_stack.py"


def _load(
    monkeypatch,
    tmp_path,
    *,
    spec,
    generic,
    has_pip = True,
    ensurepip_works = True,
    download_ok = True,
    drops_wheel = True,
    uninstall_ok = True,
    install_ok = True,
    pinned = True,
    torch_label = "2.9.1+xpu",
):
    """Import the module with the world stubbed, and return (module, action log)."""
    log: list[str] = []

    mod = types.ModuleType("_stack_under_test")
    src = STACK.read_text(encoding = "utf-8")
    # Only these helpers are needed; importing the whole module would run the installer.
    start = src.index("def _installed_torch_version_label() -> str:")
    end = src.index("def _ensure_cpu_torch() -> None:")
    body = src[start:end]
    assert "_ensure_xpu_triton" in body, "extraction lost the swap"
    assert "_ensure_venv_pip" in body, "extraction lost the pip bootstrap"

    import glob as _glob
    import importlib.util as _importlib_util
    import os as _os
    import re as _re
    import shutil as _shutil
    import tempfile as _tempfile

    # A real torch/version.py on disk, so the label read is executed rather than faked.
    # Only the LOCATION step is stubbed: find_spec would otherwise resolve this process's
    # own torch (or none at all).
    _pkg = tmp_path / "torch"
    _pkg.mkdir()
    (_pkg / "__init__.py").write_text("raise AssertionError('torch must never be imported')\n")
    if torch_label is not None:
        (_pkg / "version.py").write_text(
            f"from typing import Optional\n__version__ = '{torch_label}'\ndebug = False\n"
        )
    monkeypatch.setattr(
        _importlib_util,
        "find_spec",
        lambda name: (
            types.SimpleNamespace(origin = str(_pkg / "__init__.py"))
            if name == "torch" and torch_label is not None
            else None
        ),
    )

    pip_state = {"present": has_pip}
    index_urls: list[str] = []

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
            index_urls.append(cmd[cmd.index("--index-url") + 1])
            if download_ok and drops_wheel:
                target = cmd[cmd.index("-d") + 1]
                Path(target, "pytorch_triton_xpu-3.5.0-py3-none-any.whl").write_bytes(b"")
            return subprocess.CompletedProcess(cmd, 0 if download_ok else 1, stdout = b"")
        if "uninstall" in cmd:
            log.append("UNINSTALL")
            return subprocess.CompletedProcess(cmd, 0 if uninstall_ok else 1)
        return subprocess.CompletedProcess(cmd, 0, stdout = b"")

    def fake_pip_install_try(label, *args, **kw):
        if label.startswith("pip"):
            log.append("BOOTSTRAP")
            if ensurepip_works:
                pip_state["present"] = True
            return pip_state["present"]
        log.append("INSTALL")
        return True

    def fake_pip_install(label, *args, **kw):
        # The real one exits the process via run(); that is what keeps the completion
        # manifest unwritten, so the stub raises SystemExit rather than returning.
        log.append("INSTALL")
        if not install_ok:
            raise SystemExit(1)

    ns = {
        "subprocess": types.SimpleNamespace(
            run = fake_run,
            CompletedProcess = subprocess.CompletedProcess,
            TimeoutExpired = subprocess.TimeoutExpired,
            DEVNULL = subprocess.DEVNULL,
            PIPE = subprocess.PIPE,
            STDOUT = subprocess.STDOUT,
        ),
        "sys": sys,
        "glob": _glob,
        "importlib": types.SimpleNamespace(util = _importlib_util, invalidate_caches = lambda: None),
        "os": _os,
        "re": _re,
        "shutil": _shutil,
        "tempfile": _tempfile,
        "Path": Path,
        "NO_TORCH": False,
        "IS_MACOS": False,
        "IS_WINDOWS": False,
        "_PYTORCH_WHL_BASE": "https://download.pytorch.org/whl",
        "_explicit_xpu_torch_index_url": (
            (lambda: "https://download.pytorch.org/whl/xpu") if pinned else (lambda: None)
        ),
        "pip_install_try": fake_pip_install_try,
        "pip_install": fake_pip_install,
        "_red": lambda s: s,
        "print": lambda *a, **k: log.append("WARN") if a and "left in place" in str(a[0]) else None,
    }
    exec(compile(body, str(STACK), "exec"), ns)
    mod.__dict__.update(ns)
    mod.__dict__["_test_index_urls"] = index_urls
    return mod, log


def _run(monkeypatch, tmp_path, **kw):
    mod, log = _load(monkeypatch, tmp_path, **kw)
    mod.__dict__["_ensure_xpu_triton"]()
    return log


class TestXpuTritonSwap:
    def test_orders_fetch_uninstall_install(self, monkeypatch, tmp_path):
        # The whole point: the uninstall sits between the fetch and the install.
        log = _run(monkeypatch, tmp_path, spec = "pytorch-triton-xpu==3.5.0", generic = "3.7.1")
        assert log == ["DOWNLOAD", "UNINSTALL", "INSTALL"]

    def test_handles_the_triton_xpu_rename(self, monkeypatch, tmp_path):
        # torch 2.10 renamed the distribution; the spec is read from torch, never hardcoded.
        log = _run(monkeypatch, tmp_path, spec = "triton-xpu==3.6.0", generic = "3.7.1")
        assert log == ["DOWNLOAD", "UNINSTALL", "INSTALL"]

    def test_bootstraps_pip_when_the_venv_has_none(self, monkeypatch, tmp_path):
        # uv venv has no --seed, so a fresh venv cannot run pip download at all.
        log = _run(
            monkeypatch, tmp_path, spec = "pytorch-triton-xpu==3.5.0", generic = "3.7.1", has_pip = False
        )
        assert log[0] == "ENSUREPIP"
        assert log[-3:] == ["DOWNLOAD", "UNINSTALL", "INSTALL"]

    def test_falls_back_to_installing_pip(self, monkeypatch, tmp_path):
        log = _run(
            monkeypatch,
            tmp_path,
            spec = "pytorch-triton-xpu==3.5.0",
            generic = "3.7.1",
            has_pip = False,
            ensurepip_works = False,
        )
        # ensurepip failed, so it tries a real pip install; that fails too here, and the swap
        # must warn rather than uninstall with nothing to install from.
        assert "BOOTSTRAP" in log
        assert "UNINSTALL" not in log

    @pytest.mark.parametrize(
        "spec, generic",
        [
            ("pytorch-triton-xpu==3.5.0", ""),  # nothing shadowing it
            ("triton==3.7.1", "3.7.1"),  # torch is not the +xpu wheel
            ("", "3.7.1"),  # torch declares no triton at all
        ],
    )
    def test_leaves_a_healthy_venv_alone(self, monkeypatch, tmp_path, spec, generic):
        assert _run(monkeypatch, tmp_path, spec = spec, generic = generic) == []

    def test_a_dead_mirror_removes_nothing(self, monkeypatch, tmp_path):
        # Warn and leave the venv working; never uninstall with nothing to install from.
        log = _run(
            monkeypatch,
            tmp_path,
            spec = "pytorch-triton-xpu==3.5.0",
            generic = "3.7.1",
            download_ok = False,
        )
        assert "UNINSTALL" not in log and "INSTALL" not in log

    def test_a_successful_exit_with_no_wheel_removes_nothing(self, monkeypatch, tmp_path):
        # The exit code alone is not enough: no wheel on disk means nothing to install from.
        log = _run(
            monkeypatch,
            tmp_path,
            spec = "pytorch-triton-xpu==3.5.0",
            generic = "3.7.1",
            drops_wheel = False,
        )
        assert "UNINSTALL" not in log and "INSTALL" not in log


class TestFailedSwapIsNotSurvivable:
    def test_a_failed_uninstall_changes_nothing(self, monkeypatch, tmp_path):
        # A read-only or locked venv leaves generic triton registered. Installing over it would
        # let a later upgrade of that distribution delete the shared files again, and every
        # dependency pass would repeat the swap.
        log = _run(
            monkeypatch,
            tmp_path,
            spec = "pytorch-triton-xpu==3.5.0",
            generic = "3.7.1",
            uninstall_ok = False,
        )
        assert log == ["DOWNLOAD", "UNINSTALL"]
        assert "INSTALL" not in log

    def test_a_failed_install_propagates(self, monkeypatch, tmp_path):
        # The uninstall already took the shared files, so a warning would let the caller write a
        # completion manifest over a venv whose torch.compile is broken -- and the next update
        # would fast-path past it, since no generic distribution is left to trigger on.
        with pytest.raises(SystemExit):
            _run(
                monkeypatch,
                tmp_path,
                spec = "pytorch-triton-xpu==3.5.0",
                generic = "3.7.1",
                install_ok = False,
            )


class TestTheInstalledWheelIsThePin:
    """`UNSLOTH_TORCH_INDEX_FAMILY=xpu ./install.sh` is a ONE-SHOT pin.

    It is gone from the environment by the next plain `unsloth studio update`, yet that
    update's dependency pass can pull generic triton back in (unsloth declares triton as a
    core dep). Gating the swap on the pin alone therefore leaves every already-installed
    XPU venv shadowed forever. The +xpu wheel on disk is the durable signal, and setup.sh
    already raises the bitsandbytes floor off exactly that.
    """

    def test_swaps_with_no_pin_when_torch_is_the_xpu_wheel(self, monkeypatch, tmp_path):
        log = _run(
            monkeypatch,
            tmp_path,
            spec = "pytorch-triton-xpu==3.5.0",
            generic = "3.7.1",
            pinned = False,
        )
        assert log == ["DOWNLOAD", "UNINSTALL", "INSTALL"]

    def test_no_pin_falls_back_to_the_default_xpu_index(self, monkeypatch, tmp_path):
        mod, _ = _load(
            monkeypatch, tmp_path, spec = "pytorch-triton-xpu==3.5.0", generic = "3.7.1", pinned = False
        )
        mod.__dict__["_ensure_xpu_triton"]()
        assert mod.__dict__["_test_index_urls"] == ["https://download.pytorch.org/whl/xpu"]

    @pytest.mark.parametrize("label", ["2.9.1+cu128", "2.9.1+rocm6.4", "2.9.1", "", None])
    def test_no_pin_and_no_xpu_wheel_does_nothing(self, monkeypatch, tmp_path, label):
        # No pin and no +xpu torch is an ordinary CUDA/ROCm/CPU venv; generic triton is
        # correct there, and removing it would break torch.compile on the supported path.
        assert (
            _run(
                monkeypatch,
                tmp_path,
                spec = "pytorch-triton-xpu==3.5.0",
                generic = "3.7.1",
                pinned = False,
                torch_label = label,
            )
            == []
        )

    def test_the_label_is_read_off_disk_not_imported(self, monkeypatch, tmp_path):
        # The fake torch/__init__.py raises; reaching the swap at all proves the label came
        # from version.py. `import torch` loads the SYCL runtime and can wedge on a stalled
        # Intel driver, which is the exact host this code runs on.
        mod, _ = _load(
            monkeypatch, tmp_path, spec = "pytorch-triton-xpu==3.5.0", generic = "3.7.1", pinned = False
        )
        assert mod.__dict__["_installed_torch_version_label"]() == "2.9.1+xpu"

    def test_an_explicit_pin_still_wins(self, monkeypatch, tmp_path):
        # A pinned mirror must be used verbatim, not replaced by the default index.
        mod, _ = _load(monkeypatch, tmp_path, spec = "pytorch-triton-xpu==3.5.0", generic = "3.7.1")
        mod.__dict__["_ensure_xpu_triton"]()
        assert mod.__dict__["_test_index_urls"] == ["https://download.pytorch.org/whl/xpu"]


class TestPlatformGuards:
    @pytest.mark.parametrize("flag", ["NO_TORCH", "IS_MACOS", "IS_WINDOWS"])
    def test_skipped_where_it_does_not_apply(self, monkeypatch, tmp_path, flag):
        # Windows is setup.ps1's job; macOS has no XPU; --no-torch touches no wheels.
        mod, log = _load(monkeypatch, tmp_path, spec = "pytorch-triton-xpu==3.5.0", generic = "3.7.1")
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
