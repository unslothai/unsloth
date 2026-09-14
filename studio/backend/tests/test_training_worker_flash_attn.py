# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import ast
import builtins
import importlib.metadata
import inspect
import os
import subprocess
import sys
import types
from typing import Any
from unittest import mock

import pytest

from core.training import worker

# The runtime install is Linux-only, so elsewhere these return before any status.
linux_only = pytest.mark.skipif(
    not sys.platform.startswith("linux"),
    reason = "the runtime flash-attn install is gated to Linux",
)

# causal-conv1d is NOT Linux-gated: the installer bails out on `sys.platform == "win32"`
# alone (no prebuilt wheel for Windows) and runs everywhere else, macOS included.
# linux_only here would skip cases that legitimately pass off Linux.
not_on_windows = pytest.mark.skipif(
    sys.platform == "win32",
    reason = "mirrors the sys.platform == 'win32' bail-out in _ensure_causal_conv1d_fast_path",
)


@pytest.fixture(autouse = True)
def _clear_offline_environment(monkeypatch):
    monkeypatch.delenv("HF_HUB_OFFLINE", raising = False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising = False)


@pytest.fixture(autouse = True)
def _restore_fla_tilelang_environment():
    """monkeypatch.delenv on an absent var records no undo, so the guard's setdefault leaks it."""
    names = ("FLA_TILELANG", worker._FAST_PATH_HOOKS_SKIP_ENV)
    saved = {name: os.environ.get(name) for name in names}
    yield
    for name, value in saved.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value


def _missing_flash_attn_import():
    real_import = builtins.__import__

    def fake_import(
        name,
        globals = None,
        locals = None,
        fromlist = (),
        level = 0,
    ):
        if name == "flash_attn":
            raise ImportError
        return real_import(name, globals, locals, fromlist, level)

    return fake_import


def _flash_attn_import_until_installed(state: dict[str, bool]):
    """Import stub for a flash_attn that appears only once ``state['installed']`` is set.

    The installers verify the import, so a mock that keeps failing models a broken wheel,
    not a working one: "the wheel worked" means flipping the flag when install_wheel runs.
    """
    real_import = builtins.__import__

    def fake_import(
        name,
        globals = None,
        locals = None,
        fromlist = (),
        level = 0,
    ):
        if name == "flash_attn":
            if not state.get("installed"):
                raise ImportError
            # A stub, not a real import: flash_attn is not installed in the test env, and
            # whether it happens to be is not what these tests are about.
            return types.ModuleType("flash_attn")
        return real_import(name, globals, locals, fromlist, level)

    return fake_import


def _missing_module_import(missing: str):
    real_import = builtins.__import__

    def fake_import(
        name,
        globals = None,
        locals = None,
        fromlist = (),
        level = 0,
    ):
        if name == missing:
            raise ImportError
        return real_import(name, globals, locals, fromlist, level)

    return fake_import


class TestIsImportableIsolated:
    """The probe runs in a child so a bad native extension cannot take the worker with it."""

    def test_clean_exit_is_importable(self, monkeypatch):
        monkeypatch.setattr(worker._sp, "run", lambda *a, **k: subprocess.CompletedProcess(a[0], 0))
        assert worker._is_importable_isolated("flash_attn") is True

    def test_import_error_exit_is_not_importable(self, monkeypatch):
        monkeypatch.setattr(worker._sp, "run", lambda *a, **k: subprocess.CompletedProcess(a[0], 1))
        assert worker._is_importable_isolated("flash_attn") is False

    def test_fatal_signal_is_not_importable(self, monkeypatch):
        # SIGSEGV in the extension's initialiser: rc -11, and no Python exception to catch.
        monkeypatch.setattr(
            worker._sp, "run", lambda *a, **k: subprocess.CompletedProcess(a[0], -11)
        )
        assert worker._is_importable_isolated("flash_attn") is False

    def test_timeout_is_not_importable(self, monkeypatch):
        def _hang(*_args, **_kwargs):
            raise subprocess.TimeoutExpired(cmd = "python", timeout = 300)

        monkeypatch.setattr(worker._sp, "run", _hang)
        assert worker._is_importable_isolated("flash_attn") is False

    def test_the_module_name_is_passed_as_an_argument(self, monkeypatch):
        captured: list[list[str]] = []

        def _record(cmd, **_kwargs):
            captured.append(list(cmd))
            return subprocess.CompletedProcess(cmd, 0)

        monkeypatch.setattr(worker._sp, "run", _record)
        worker._is_importable_isolated("flash_attn")

        # argv, not string-formatted into the -c body.
        assert captured[0][-1] == "flash_attn"
        assert "import flash_attn" not in " ".join(captured[0])


class TestNoExitLeavesAnUnusableInstall:
    """Every unsuccessful exit discards the distribution, not just the ones with a call.

    The metadata gate in unsloth/models/_utils.py imports the extension in process, so
    anything left behind is loaded anyway. Four rounds of review found four separate exits
    that forgot to clean up, which is why this is enforced in one place.
    """

    def _run(self, monkeypatch, *, run_side_effect):
        removals: list[list[str]] = []

        def _spy(cmd, **kwargs):
            if "uninstall" in cmd:
                removals.append(list(cmd))
                return subprocess.CompletedProcess(cmd, 0, "")
            return run_side_effect(cmd, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _missing_flash_attn_import())
        monkeypatch.setattr(worker, "_is_importable_isolated", lambda name: False)
        # Metadata says it is installed, which is exactly the state that must not survive.
        monkeypatch.setattr(worker, "_distribution_present", lambda name: True)
        monkeypatch.setattr(worker, "flash_attn_wheel_url", lambda env: None)
        monkeypatch.setattr(worker, "url_exists", lambda url: False)
        monkeypatch.setattr(worker.shutil, "which", lambda name: None)
        monkeypatch.setattr(worker, "_send_status", lambda q, m: None)
        monkeypatch.setattr(worker._sp, "run", _spy)

        installed = worker._install_package_wheel_first(
            event_queue = [],
            import_name = "flash_attn",
            display_name = "flash-attn",
            pypi_name = "flash-attn",
            pypi_spec = "flash-attn",
        )
        return installed, removals

    def test_a_failed_fallback_install_still_cleans_up(self, monkeypatch):
        installed, removals = self._run(
            monkeypatch,
            run_side_effect = lambda cmd, **kw: subprocess.CompletedProcess(cmd, 1, "boom"),
        )
        assert installed is False
        assert removals, "a non-zero fallback exit must not leave the distribution installed"

    def test_a_timed_out_fallback_still_cleans_up(self, monkeypatch):
        """Only the ROCm source build sets a timeout, so this is the path that can raise."""
        removals: list[list[str]] = []

        def _spy(cmd, **kwargs):
            if "uninstall" in cmd:
                removals.append(list(cmd))
                return subprocess.CompletedProcess(cmd, 0, "")
            raise subprocess.TimeoutExpired(cmd = cmd, timeout = 1800)

        monkeypatch.setattr(builtins, "__import__", _missing_flash_attn_import())
        monkeypatch.setattr(worker, "_is_importable_isolated", lambda name: False)
        monkeypatch.setattr(worker, "_distribution_present", lambda name: True)
        monkeypatch.setattr(
            worker,
            "probe_torch_wheel_env",
            lambda timeout = 30: {
                "python_tag": "cp313",
                "torch_mm": "2.10",
                "cuda_major": "",
                "hip_version": "6.2",
                "cxx11abi": "TRUE",
                "platform_tag": "linux_x86_64",
            },
        )
        monkeypatch.setattr(worker, "flash_attn_wheel_url", lambda env: None)
        monkeypatch.setattr(worker, "url_exists", lambda url: False)
        # hipcc present, so the ROCm build is attempted rather than skipped.
        monkeypatch.setattr(worker.shutil, "which", lambda name: "/opt/rocm/bin/hipcc")
        monkeypatch.setattr(worker, "_send_status", lambda q, m: None)
        monkeypatch.setattr(worker._sp, "run", _spy)

        installed = worker._install_package_wheel_first(
            event_queue = [],
            import_name = "flash_attn",
            display_name = "flash-attn",
            pypi_name = "flash-attn",
            pypi_spec = "flash-attn",
        )

        assert installed is False
        assert removals, "a timed-out build must not leave the distribution installed"

    def test_nothing_installed_means_nothing_to_remove(self, monkeypatch):
        """The discard is state-based, so a clean failure does not run a pointless uninstall."""
        removals: list[list[str]] = []

        def _spy(cmd, **_kwargs):
            if "uninstall" in cmd:
                removals.append(list(cmd))
            return subprocess.CompletedProcess(cmd, 1, "")

        monkeypatch.setattr(builtins, "__import__", _missing_flash_attn_import())
        monkeypatch.setattr(worker, "_is_importable_isolated", lambda name: False)
        monkeypatch.setattr(worker, "_distribution_present", lambda name: False)
        monkeypatch.setattr(worker, "flash_attn_wheel_url", lambda env: None)
        monkeypatch.setattr(worker, "url_exists", lambda url: False)
        monkeypatch.setattr(worker.shutil, "which", lambda name: None)
        monkeypatch.setattr(worker, "_send_status", lambda q, m: None)
        monkeypatch.setattr(worker._sp, "run", _spy)

        worker._install_package_wheel_first(
            event_queue = [],
            import_name = "flash_attn",
            display_name = "flash-attn",
            pypi_name = "flash-attn",
            pypi_spec = "flash-attn",
        )

        assert removals == []

    def test_a_working_package_is_never_touched(self, monkeypatch):
        """The guard runs before the cleanup, so an install that works is left alone."""
        calls: list[list[str]] = []
        monkeypatch.setattr(worker, "_is_importable", lambda name: True)
        monkeypatch.setattr(worker._sp, "run", lambda cmd, **kw: calls.append(list(cmd)))

        installed = worker._install_package_wheel_first(
            event_queue = [],
            import_name = "flash_attn",
            display_name = "flash-attn",
            pypi_name = "flash-attn",
        )

        assert installed is True
        assert calls == [], "a working package must not be probed, installed or removed"


def test_should_try_runtime_flash_attn_install_threshold_and_skip(monkeypatch):
    monkeypatch.delenv(worker._FLASH_ATTN_SKIP_ENV, raising = False)
    assert worker._should_try_runtime_flash_attn_install(32767) is False
    assert worker._should_try_runtime_flash_attn_install(32768) is sys.platform.startswith("linux")

    monkeypatch.setenv(worker._FLASH_ATTN_SKIP_ENV, "1")
    assert worker._should_try_runtime_flash_attn_install(32768) is False


@linux_only
def test_runtime_flash_attn_prefers_prebuilt_wheel(monkeypatch):
    statuses: list[str] = []
    state: dict[str, bool] = {"installed": False}

    def _install(*_args, **_kwargs):
        state["installed"] = True
        return [("pip", subprocess.CompletedProcess(["pip"], 0, ""))]

    monkeypatch.delenv(worker._FLASH_ATTN_SKIP_ENV, raising = False)
    monkeypatch.setattr(builtins, "__import__", _flash_attn_import_until_installed(state))
    # The post-install probe runs in a child; model it off the same flag.
    monkeypatch.setattr(worker, "_is_importable_isolated", lambda name: state["installed"])
    monkeypatch.setattr(
        worker,
        "flash_attn_wheel_url",
        lambda env: "https://example.com/fa.whl",
    )
    monkeypatch.setattr(worker, "url_exists", lambda url: True)
    monkeypatch.setattr(
        worker,
        "_send_status",
        lambda queue, message: statuses.append(message),
    )
    monkeypatch.setattr(worker, "install_wheel", _install)

    worker._ensure_flash_attn_for_long_context(event_queue = [], max_seq_length = 32768)

    assert statuses == ["Installing flash-attn for faster training..."]


@linux_only
def test_runtime_flash_attn_wheel_that_does_not_import_falls_back(monkeypatch):
    """A wrong-arch/ABI wheel installs with rc=0 and then will not load.

    The Blackwell shape (#5420, #6961): trusting the exit code left a flash_attn that
    raised on import mid-training. It must be treated as not installed.
    """
    statuses: list[str] = []
    pypi_calls: list[list[str]] = []

    monkeypatch.delenv(worker._FLASH_ATTN_SKIP_ENV, raising = False)
    # Never becomes importable, however the install exits.
    monkeypatch.setattr(builtins, "__import__", _missing_flash_attn_import())
    monkeypatch.setattr(worker, "_is_importable_isolated", lambda name: False)
    monkeypatch.setattr(
        worker,
        "flash_attn_wheel_url",
        lambda env: "https://example.com/fa.whl",
    )
    monkeypatch.setattr(worker, "url_exists", lambda url: True)
    monkeypatch.setattr(
        worker,
        "_send_status",
        lambda queue, message: statuses.append(message),
    )
    monkeypatch.setattr(
        worker,
        "install_wheel",
        lambda *args, **kwargs: [("pip", subprocess.CompletedProcess(["pip"], 0, ""))],
    )
    monkeypatch.setattr(worker.shutil, "which", lambda name: None)
    monkeypatch.setattr(
        worker._sp,
        "run",
        lambda cmd, **kwargs: pypi_calls.append(cmd) or subprocess.CompletedProcess(cmd, 1, ""),
    )

    worker._ensure_flash_attn_for_long_context(event_queue = [], max_seq_length = 32768)

    # The wheel is not silently trusted: it falls through to the PyPI path.
    assert "Installing flash-attn from PyPI for long-context training..." in statuses
    installs = [cmd for cmd in pypi_calls if "install" in cmd]
    assert installs, "expected a PyPI install attempt after the wheel failed to import"
    # The rejected wheel is removed first. Installing over it is a no-op (pip reports it as
    # already satisfied), and --force-reinstall would widen the transaction to torch.
    assert any("uninstall" in cmd for cmd in pypi_calls), pypi_calls
    assert not any("--force-reinstall" in cmd for cmd in installs), installs


@linux_only
def test_runtime_flash_attn_rejected_wheel_is_not_reported_installed(monkeypatch):
    """A no-op fallback must not be read as success.

    pip exits 0 on "Requirement already satisfied" and uv on "Would make no changes", so
    the exit code alone would report the rejected wheel as a working install.
    """
    statuses: list[str] = []

    monkeypatch.delenv(worker._FLASH_ATTN_SKIP_ENV, raising = False)
    monkeypatch.setattr(builtins, "__import__", _missing_flash_attn_import())
    monkeypatch.setattr(worker, "_is_importable_isolated", lambda name: False)
    # The discard is state-based, so the installed-but-broken state has to be stated here.
    # Without this the test only passes on a machine that happens to have flash-attn.
    monkeypatch.setattr(worker, "_distribution_present", lambda name: True)
    monkeypatch.setattr(
        worker,
        "flash_attn_wheel_url",
        lambda env: "https://example.com/fa.whl",
    )
    monkeypatch.setattr(worker, "url_exists", lambda url: True)
    monkeypatch.setattr(
        worker,
        "_send_status",
        lambda queue, message: statuses.append(message),
    )
    monkeypatch.setattr(
        worker,
        "install_wheel",
        lambda *args, **kwargs: [("pip", subprocess.CompletedProcess(["pip"], 0, ""))],
    )
    monkeypatch.setattr(worker.shutil, "which", lambda name: None)
    # The fallback "succeeds" the way an already-satisfied install does: rc=0, nothing done.
    monkeypatch.setattr(
        worker._sp,
        "run",
        lambda cmd, **kwargs: subprocess.CompletedProcess(cmd, 0, "Requirement already satisfied"),
    )

    installed = worker._install_package_wheel_first(
        event_queue = [],
        import_name = "flash_attn",
        display_name = "flash-attn",
        pypi_name = "flash-attn",
        wheel_url_builder = worker.flash_attn_wheel_url,
        pypi_spec = "flash-attn",
        pypi_status_message = "Installing flash-attn from PyPI for long-context training...",
    )

    assert installed is False
    # and it is removed, not left where _package_available would advertise it.
    assert "flash-attn is not usable on this GPU; removed it" in statuses


@linux_only
def test_runtime_flash_attn_says_so_when_the_rejected_install_cannot_be_removed(monkeypatch):
    """A distribution still on disk is not the same state as never having installed one."""
    statuses: list[str] = []

    monkeypatch.delenv(worker._FLASH_ATTN_SKIP_ENV, raising = False)
    monkeypatch.setattr(builtins, "__import__", _missing_flash_attn_import())
    monkeypatch.setattr(worker, "_is_importable_isolated", lambda name: False)
    # State the installed-but-broken state explicitly; see the note above.
    monkeypatch.setattr(worker, "_distribution_present", lambda name: True)
    monkeypatch.setattr(worker, "flash_attn_wheel_url", lambda env: None)
    monkeypatch.setattr(worker, "url_exists", lambda url: False)
    monkeypatch.setattr(worker.shutil, "which", lambda name: None)
    monkeypatch.setattr(
        worker,
        "_send_status",
        lambda queue, message: statuses.append(message),
    )
    # Install exits 0; every uninstall attempt fails (read-only or locked site-packages).
    monkeypatch.setattr(
        worker._sp,
        "run",
        lambda cmd, **kw: subprocess.CompletedProcess(cmd, 1 if "uninstall" in cmd else 0, ""),
    )

    installed = worker._install_package_wheel_first(
        event_queue = [],
        import_name = "flash_attn",
        display_name = "flash-attn",
        pypi_name = "flash-attn",
        pypi_spec = "flash-attn",
    )

    assert installed is False
    assert any("could not be removed" in s for s in statuses), statuses
    assert not any("removed it" in s for s in statuses), statuses


@linux_only
def test_runtime_flash_attn_falls_back_to_pypi(monkeypatch):
    calls: list[list[str]] = []
    statuses: list[str] = []
    state: dict[str, bool] = {"installed": False}

    monkeypatch.delenv(worker._FLASH_ATTN_SKIP_ENV, raising = False)
    monkeypatch.setattr(builtins, "__import__", _flash_attn_import_until_installed(state))
    monkeypatch.setattr(
        worker,
        "probe_torch_wheel_env",
        lambda timeout = 30: {
            "python_tag": "cp313",
            "torch_mm": "2.10",
            "cuda_major": "13",
            "cxx11abi": "TRUE",
            "platform_tag": "linux_x86_64",
        },
    )
    monkeypatch.setattr(
        worker,
        "flash_attn_wheel_url",
        lambda env: "https://example.com/fa.whl",
    )
    monkeypatch.setattr(worker, "url_exists", lambda url: False)
    monkeypatch.setattr(worker.shutil, "which", lambda name: None)
    monkeypatch.setattr(worker, "_is_importable_isolated", lambda name: state["installed"])
    monkeypatch.setattr(
        worker,
        "_send_status",
        lambda queue, message: statuses.append(message),
    )
    monkeypatch.setattr(worker, "install_wheel", mock.Mock())

    def fake_run(cmd, **kwargs):
        calls.append(list(cmd))
        # A real install makes the module importable; the post-install check requires it.
        state["installed"] = True
        return subprocess.CompletedProcess(cmd, 0, "")

    monkeypatch.setattr(worker._sp, "run", fake_run)

    worker._ensure_flash_attn_for_long_context(event_queue = [], max_seq_length = 32768)

    assert statuses == ["Installing flash-attn from PyPI for long-context training..."]
    # No wheel was installed here (url_exists is False), so nothing needs replacing.
    assert calls == [[sys.executable, "-m", "pip", "install", "flash-attn"]]


def test_runtime_flash_attn_skip_env_avoids_all_install_work(monkeypatch):
    monkeypatch.setenv(worker._FLASH_ATTN_SKIP_ENV, "1")
    monkeypatch.setattr(worker._sp, "run", mock.Mock())

    worker._ensure_flash_attn_for_long_context(event_queue = [], max_seq_length = 32768)

    worker._sp.run.assert_not_called()


@pytest.mark.parametrize("offline_variable", ["HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"])
def test_wheel_first_install_skips_all_install_work_offline(monkeypatch, offline_variable):
    monkeypatch.delenv("HF_HUB_OFFLINE", raising = False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising = False)
    monkeypatch.setenv(offline_variable, "1")
    monkeypatch.setattr(builtins, "__import__", _missing_module_import("missing_fast_path"))
    probe = mock.Mock()
    url_probe = mock.Mock()
    wheel_install = mock.Mock()
    process_run = mock.Mock()
    monkeypatch.setattr(worker, "probe_torch_wheel_env", probe)
    monkeypatch.setattr(worker, "url_exists", url_probe)
    monkeypatch.setattr(worker, "install_wheel", wheel_install)
    monkeypatch.setattr(worker._sp, "run", process_run)

    installed = worker._install_package_wheel_first(
        event_queue = [],
        import_name = "missing_fast_path",
        display_name = "missing-fast-path",
        pypi_name = "missing-fast-path",
        pypi_version = "1.0.0",
        filename_prefix = "missing_fast_path",
        release_tag = "v1.0.0",
        release_base_url = "https://example.invalid/releases",
    )

    assert installed is False
    probe.assert_not_called()
    url_probe.assert_not_called()
    wheel_install.assert_not_called()
    process_run.assert_not_called()


def test_wheel_first_install_uses_existing_package_offline(monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "true")
    probe = mock.Mock()
    monkeypatch.setattr(worker, "probe_torch_wheel_env", probe)

    installed = worker._install_package_wheel_first(
        event_queue = [],
        import_name = "sys",
        display_name = "sys",
        pypi_name = "sys",
        pypi_version = "1.0.0",
    )

    assert installed is True
    probe.assert_not_called()


@not_on_windows
def test_causal_conv1d_fast_path_preserves_wheel_first_install_args(monkeypatch):
    install_mock = mock.Mock(return_value = True)
    monkeypatch.setattr(worker, "_install_package_wheel_first", install_mock)

    worker._ensure_causal_conv1d_fast_path(
        event_queue = [],
        model_name = "tiiuae/Falcon-H1-0.5B-Instruct",
    )

    install_mock.assert_called_once_with(
        event_queue = [],
        import_name = "causal_conv1d",
        display_name = "causal-conv1d",
        pypi_name = "causal-conv1d",
        pypi_version = worker._CAUSAL_CONV1D_PACKAGE_VERSION,
        filename_prefix = "causal_conv1d",
        release_tag = worker._CAUSAL_CONV1D_RELEASE_TAG,
        release_base_url = "https://github.com/Dao-AILab/causal-conv1d/releases/download",
    )


@not_on_windows
def test_causal_conv1d_fast_path_includes_qwen3_6_variants(monkeypatch):
    install_mock = mock.Mock(return_value = True)
    monkeypatch.setattr(worker, "_install_package_wheel_first", install_mock)

    worker._ensure_causal_conv1d_fast_path(
        event_queue = [],
        model_name = "unsloth/Qwen3.6-4B",
    )
    worker._ensure_causal_conv1d_fast_path(
        event_queue = [],
        model_name = "unsloth/Qwen3_6-4B",
    )

    assert install_mock.call_count == 2


def test_mamba_ssm_path_preserves_wheel_first_install_args(monkeypatch):
    install_mock = mock.Mock(return_value = True)
    monkeypatch.setattr(worker, "_install_package_wheel_first", install_mock)

    worker._ensure_mamba_ssm(
        event_queue = [],
        model_name = "tiiuae/Falcon-H1-0.5B-Instruct",
    )

    install_mock.assert_called_once_with(
        event_queue = [],
        import_name = "mamba_ssm",
        display_name = "mamba-ssm",
        pypi_name = "mamba-ssm",
        pypi_version = worker._MAMBA_SSM_PACKAGE_VERSION,
        filename_prefix = "mamba_ssm",
        release_tag = worker._MAMBA_SSM_RELEASE_TAG,
        release_base_url = "https://github.com/state-spaces/mamba/releases/download",
    )


class _FakeQueue(list):
    """List with `.put` so worker._send_status can send into it in tests."""

    def put(self, item):
        self.append(item)


def _make_fake_gate(initial_return: bool):
    """Callable mimicking transformers' lru_cache-decorated gates.

    Tracks call count and exposes `cache_clear`. Flip `.next_return` to
    mimic install-then-True behaviour.
    """

    class Gate:
        def __init__(self, initial: bool) -> None:
            self.next_return = initial
            self.call_count = 0
            self.cache_clear_count = 0

        def __call__(self) -> bool:
            self.call_count += 1
            return self.next_return

        def cache_clear(self) -> None:
            self.cache_clear_count += 1

    return Gate(initial_return)


def _patch_iu_gate(monkeypatch, conv_gate):
    """Drop a fake causal-conv1d gate onto transformers.utils.import_utils."""
    from transformers.utils import import_utils as _iu
    monkeypatch.setattr(_iu, "is_causal_conv1d_available", conv_gate)


def test_hook_leaves_causal_gate_unchanged_for_unrelated_model(monkeypatch):
    conv_gate = _make_fake_gate(initial_return = False)
    _patch_iu_gate(monkeypatch, conv_gate)
    conv_install = mock.Mock(return_value = True)
    monkeypatch.setattr(worker, "_install_package_wheel_first", conv_install)
    monkeypatch.delenv(worker._FAST_PATH_HOOKS_SKIP_ENV, raising = False)

    worker._install_fast_path_hooks(
        event_queue = _FakeQueue(),
        model_name = "unsloth/Llama-3.2-1B-Instruct",
    )

    from transformers.utils import import_utils as _iu

    assert _iu.is_causal_conv1d_available is conv_gate
    assert _iu.is_causal_conv1d_available() is False
    conv_install.assert_not_called()


@not_on_windows
def test_hook_installs_when_gate_returns_false(monkeypatch):
    conv_gate = _make_fake_gate(initial_return = False)
    _patch_iu_gate(monkeypatch, conv_gate)

    def _conv_install_side_effect(**kw):
        conv_gate.next_return = True
        return True

    conv_install = mock.Mock(side_effect = _conv_install_side_effect)
    monkeypatch.setattr(worker, "_install_package_wheel_first", conv_install)
    monkeypatch.delenv(worker._FAST_PATH_HOOKS_SKIP_ENV, raising = False)

    worker._install_fast_path_hooks(
        event_queue = _FakeQueue(),
        model_name = "tiiuae/Falcon-H1-0.5B-Instruct",
    )

    from transformers.utils import import_utils as _iu

    # Gate wrapped; calling it should drive the install.
    assert _iu.is_causal_conv1d_available() is True
    conv_install.assert_called_once()


def test_hook_skips_install_when_gate_already_true(monkeypatch):
    """Gate already True -> zero install work."""
    conv_gate = _make_fake_gate(initial_return = True)
    _patch_iu_gate(monkeypatch, conv_gate)

    conv_install = mock.Mock()
    monkeypatch.setattr(worker, "_install_package_wheel_first", conv_install)
    monkeypatch.delenv(worker._FAST_PATH_HOOKS_SKIP_ENV, raising = False)

    worker._install_fast_path_hooks(
        event_queue = _FakeQueue(),
        model_name = "tiiuae/Falcon-H1-0.5B-Instruct",
    )

    from transformers.utils import import_utils as _iu

    assert _iu.is_causal_conv1d_available() is True
    conv_install.assert_not_called()


def test_hook_idempotent_on_repeat_call(monkeypatch):
    conv_gate = _make_fake_gate(initial_return = False)
    _patch_iu_gate(monkeypatch, conv_gate)

    def _conv_install_side_effect(**kw):
        conv_gate.next_return = True
        return True

    conv_install = mock.Mock(side_effect = _conv_install_side_effect)
    monkeypatch.setattr(worker, "_install_package_wheel_first", conv_install)
    monkeypatch.delenv(worker._FAST_PATH_HOOKS_SKIP_ENV, raising = False)

    worker._install_fast_path_hooks(
        event_queue = _FakeQueue(),
        model_name = "tiiuae/Falcon-H1-0.5B-Instruct",
    )

    from transformers.utils import import_utils as _iu

    # First call: hook fires. Later calls: must not re-trigger the installer.
    _iu.is_causal_conv1d_available()
    _iu.is_causal_conv1d_available()
    _iu.is_causal_conv1d_available()
    assert conv_install.call_count == 1


def test_hook_handles_install_failure_gracefully(monkeypatch):
    conv_gate = _make_fake_gate(initial_return = False)
    _patch_iu_gate(monkeypatch, conv_gate)

    def raising_install(**kw):
        raise RuntimeError("pip failed to fetch wheel")

    monkeypatch.setattr(worker, "_install_package_wheel_first", raising_install)
    monkeypatch.delenv(worker._FAST_PATH_HOOKS_SKIP_ENV, raising = False)

    worker._install_fast_path_hooks(
        event_queue = _FakeQueue(),
        model_name = "tiiuae/Falcon-H1-0.5B-Instruct",
    )

    from transformers.utils import import_utils as _iu

    # Must not raise; returns False so transformers uses the torch loop.
    assert _iu.is_causal_conv1d_available() is False


def test_hook_can_be_disabled_via_env(monkeypatch):
    conv_gate = _make_fake_gate(initial_return = False)
    _patch_iu_gate(monkeypatch, conv_gate)

    conv_install = mock.Mock()
    monkeypatch.setattr(worker, "_install_package_wheel_first", conv_install)
    monkeypatch.setenv(worker._FAST_PATH_HOOKS_SKIP_ENV, "1")

    worker._install_fast_path_hooks(
        event_queue = _FakeQueue(),
        model_name = "tiiuae/Falcon-H1-0.5B-Instruct",
    )

    from transformers.utils import import_utils as _iu

    # Hook not installed; the gate remains the fake.
    assert _iu.is_causal_conv1d_available is conv_gate
    conv_install.assert_not_called()


def test_hook_clears_lru_cache_before_first_check(monkeypatch):
    conv_gate = _make_fake_gate(initial_return = True)
    _patch_iu_gate(monkeypatch, conv_gate)

    monkeypatch.setattr(worker, "_install_package_wheel_first", lambda **kw: None)
    monkeypatch.delenv(worker._FAST_PATH_HOOKS_SKIP_ENV, raising = False)

    worker._install_fast_path_hooks(
        event_queue = _FakeQueue(),
        model_name = "tiiuae/Falcon-H1-0.5B-Instruct",
    )
    from transformers.utils import import_utils as _iu

    _iu.is_causal_conv1d_available()
    # Wrapper called cache_clear at least once before delegating.
    assert conv_gate.cache_clear_count >= 1


def test_hook_rewrites_previously_imported_module_bindings(monkeypatch):
    """Modeling files bind is_causal_conv1d_available locally via
    `from ... import is_X`. Reassigning the attribute on import_utils alone
    misses those; the hook installer sweeps sys.modules and rebinds them.
    """
    conv_gate = _make_fake_gate(initial_return = False)
    _patch_iu_gate(monkeypatch, conv_gate)

    # Fake modeling module that did `from ... import is_causal_conv1d_available`.
    fake_mod = sys.modules.setdefault(
        "_test_fake_modeling_falcon_h1", type(sys)("_test_fake_modeling_falcon_h1")
    )
    fake_mod.is_causal_conv1d_available = conv_gate

    def fake_install(**kw):
        conv_gate.next_return = True
        return True

    monkeypatch.setattr(worker, "_install_package_wheel_first", fake_install)
    monkeypatch.delenv(worker._FAST_PATH_HOOKS_SKIP_ENV, raising = False)

    worker._install_fast_path_hooks(
        event_queue = _FakeQueue(),
        model_name = "tiiuae/Falcon-H1-0.5B-Instruct",
    )

    # The fake module's local binding is rewritten to the wrapper.
    assert fake_mod.is_causal_conv1d_available is not conv_gate
    # Calling through the fake module's reference triggers install.
    assert fake_mod.is_causal_conv1d_available() is True

    del sys.modules["_test_fake_modeling_falcon_h1"]


def test_hook_skips_when_import_utils_unavailable(monkeypatch):
    """If transformers.utils.import_utils can't be imported, the hook
    installer must log and return cleanly rather than crash the worker."""
    real_import = builtins.__import__

    def fake_import(name, *a, **kw):
        if name == "transformers.utils" or name == "transformers.utils.import_utils":
            raise ImportError("transformers missing in worker venv")
        return real_import(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.delenv(worker._FAST_PATH_HOOKS_SKIP_ENV, raising = False)

    # Should not raise.
    worker._install_fast_path_hooks(
        event_queue = _FakeQueue(),
        model_name = "tiiuae/Falcon-H1-0.5B-Instruct",
    )


def test_hook_trusts_installer_bool_not_metadata(monkeypatch):
    """If pip exits 0 but deep imports fail, the installer returns False; the hook
    must propagate that False even though the metadata-only gate flipped True, so
    transformers takes the torch fallback.
    """
    conv_gate = _make_fake_gate(initial_return = False)
    _patch_iu_gate(monkeypatch, conv_gate)

    def _bad_install(**kw):
        conv_gate.next_return = True  # metadata says yes after pip
        return False  # but deep import is broken

    fake_install = mock.Mock(side_effect = _bad_install)
    monkeypatch.setattr(worker, "_install_package_wheel_first", fake_install)
    monkeypatch.delenv(worker._FAST_PATH_HOOKS_SKIP_ENV, raising = False)

    worker._install_fast_path_hooks(
        event_queue = _FakeQueue(),
        model_name = "tiiuae/Falcon-H1-0.5B-Instruct",
    )

    from transformers.utils import import_utils as _iu

    # Hook MUST return False (installer's verdict), not True (metadata lies).
    assert _iu.is_causal_conv1d_available() is False
    fake_install.assert_called_once()


def test_rebind_does_not_trigger_module_getattr(monkeypatch):
    """The rebind sweep must use __dict__, not getattr(), to avoid invoking
    transformers' lazy module __getattr__ which spits out hundreds of
    "Accessing X from .models..." warnings.
    """
    original = object()
    replacement = object()

    class _GetattrTripwire(type(sys)):
        getattr_called = False

        def __getattr__(self, name):
            type(self).getattr_called = True
            raise AttributeError(name)

    lazy = _GetattrTripwire("_lazy_test_module")
    sys.modules["_lazy_test_module"] = lazy
    try:
        # No `is_causal_conv1d_available` in __dict__, so the sweep must NOT
        # trip the tripwire.
        worker._rebind_in_already_imported_modules(
            attr_name = "is_causal_conv1d_available",
            old_obj = original,
            new_obj = replacement,
        )
        assert (
            not _GetattrTripwire.getattr_called
        ), "Rebind sweep invoked __getattr__ - should use __dict__ probe"
    finally:
        sys.modules.pop("_lazy_test_module", None)


def test_run_training_process_eagerly_installs_causal_conv1d_in_normal_mode():
    """SSM modeling files use lazy_load_kernel and never call
    is_causal_conv1d_available(), so the hook won't fire; the orchestrator must
    always run the eager installer. Reads the worker source and asserts the eager
    install happens before the hooks are wired.
    """
    import inspect

    src = inspect.getsource(worker.run_training_process)
    # Orchestration block. Match the call, not its formatting, so wrapping the args over
    # several lines or adding a keyword does not break this.
    assert "_ensure_causal_conv1d_fast_path(" in src
    assert "_install_fast_path_hooks(" in src
    eager_pos = src.find("_ensure_causal_conv1d_fast_path(")
    hooks_pos = src.find("_install_fast_path_hooks(")
    assert eager_pos < hooks_pos, (
        "_ensure_causal_conv1d_fast_path must be called BEFORE the hooks are "
        "wired, so SSM models that bypass is_causal_conv1d_available() still "
        "get the eager install"
    )


# unsloth_zoo vendors the FLA kernels as `fla`; the worker must never pip install them.
_NEVER_PIP_INSTALLED = ("flash-linear-attention", "fla-core", "tilelang", "apache-tvm-ffi")


@not_on_windows
@pytest.mark.parametrize("uv_path", ["/usr/bin/uv", None], ids = ["uv", "no-uv"])
def test_no_install_path_pips_flash_linear_attention_or_tilelang(monkeypatch, uv_path):
    """Every pip argv the worker builds, and none of them may name the vendored stack.

    Parametrized over uv: the pip and the uv branch of every installer build different argvs.
    """
    calls: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        calls.append(list(cmd))
        return subprocess.CompletedProcess(cmd, 0, "")

    monkeypatch.delenv(worker._FLASH_ATTN_SKIP_ENV, raising = False)
    monkeypatch.delenv(worker._FAST_PATH_HOOKS_SKIP_ENV, raising = False)
    monkeypatch.setattr(worker._sp, "run", fake_run)
    monkeypatch.setattr(worker, "_is_importable", lambda name: False)
    monkeypatch.setattr(worker, "_is_importable_isolated", lambda name: True)
    monkeypatch.setattr(worker, "probe_torch_wheel_env", lambda timeout = 30: {})
    monkeypatch.setattr(worker, "url_exists", lambda url: False)
    monkeypatch.setattr(worker, "install_wheel", mock.Mock())
    monkeypatch.setattr(worker, "flash_attn_wheel_url", lambda env: None)
    monkeypatch.setattr(worker.shutil, "which", lambda name: uv_path if name == "uv" else None)
    monkeypatch.setattr(worker, "_send_status", lambda *a, **k: None)

    worker._ensure_causal_conv1d_fast_path(
        event_queue = [],
        model_name = "unsloth/Qwen3.5-2B",
        required = True,
    )
    worker._ensure_mamba_ssm(
        event_queue = [],
        model_name = "tiiuae/Falcon-H1-0.5B-Instruct",
    )
    if sys.platform.startswith("linux"):
        worker._ensure_flash_attn_for_long_context(event_queue = [], max_seq_length = 65536)

    conv_gate = _make_fake_gate(initial_return = False)
    _patch_iu_gate(monkeypatch, conv_gate)
    worker._install_fast_path_hooks(
        event_queue = _FakeQueue(),
        model_name = "unsloth/Qwen3.5-2B",
        install_causal_conv1d = True,
    )
    from transformers.utils import import_utils as _iu

    _iu.is_causal_conv1d_available()

    assert calls, "the doubles must let at least one pip install through"
    flat = " ".join(" ".join(call) for call in calls)
    for package in _NEVER_PIP_INSTALLED:
        assert package not in flat, f"{package} must never be pip installed: {calls}"


# The only worker.py strings allowed to name the stack: these prose log lines, plus the
# sole argument of an import probe. A bare "tilelang" anywhere else is an unpinned pip spec.
_FLA_PROSE_LOG_LINES = (
    "flash-linear-attention fast path importable: %s",
    "flash-linear-attention is not importable; continuing on the pure-torch path: %s",
)
_FLA_PROBE_CALLS = ("importlib.util.find_spec", "importlib.metadata.version")


def _attribute_chain(node: ast.AST) -> str:
    """Dotted source spelling of a call target, e.g. `importlib.util.find_spec`."""
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return ""
    parts.append(node.id)
    return ".".join(reversed(parts))


def test_worker_source_never_pins_the_vendored_stack():
    """worker.py may name the stack in a log line or an import probe, never in a pip spec."""
    tree = ast.parse(inspect.getsource(worker))
    probe_arguments = {
        id(call.args[0])
        for call in ast.walk(tree)
        if isinstance(call, ast.Call)
        and len(call.args) == 1
        and not call.keywords
        and _attribute_chain(call.func) in _FLA_PROBE_CALLS
    }
    for node in ast.walk(tree):
        if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
            continue
        if node.value in _FLA_PROSE_LOG_LINES or id(node) in probe_arguments:
            continue
        for package in _NEVER_PIP_INSTALLED:
            assert not node.value.startswith(
                package
            ), f"worker.py names {package} in a string constant: {node.value!r}"


def _stub_fla_modules(monkeypatch):
    """Register a walkable `fla` package exposing the two submodules the probe imports."""
    fla = types.ModuleType("fla")
    fla.__path__ = []
    for name in ("fla", "fla.modules", "fla.ops", "fla.ops.gated_delta_rule"):
        monkeypatch.setitem(sys.modules, name, fla if name == "fla" else types.ModuleType(name))


def test_flash_linear_attention_importable_true_when_vendored_fla_imports(monkeypatch):
    """unsloth_zoo injects the vendored kernels as `fla`; the probe must report them present."""
    _stub_fla_modules(monkeypatch)

    assert worker._flash_linear_attention_importable() is True


def test_flash_linear_attention_importable_false_and_warns_when_import_raises(monkeypatch):
    """No fla at all: the probe answers False and says the run drops to the pure-torch path."""
    for name in ("fla", "fla.modules", "fla.ops", "fla.ops.gated_delta_rule"):
        monkeypatch.setitem(sys.modules, name, None)
    # worker.logger is a structlog BoundLogger, so caplog never sees it.
    warnings: list[str] = []
    monkeypatch.setattr(worker.logger, "warning", lambda msg, *a: warnings.append(msg % a))

    assert worker._flash_linear_attention_importable() is False
    assert any("pure-torch path" in line for line in warnings), warnings


def _force_torch_hip(monkeypatch, hip: str | None):
    """Make the guard's lazily imported torch look like a ROCm (or CUDA) build."""
    import torch

    monkeypatch.setattr(torch.version, "hip", hip, raising = False)
    # __version__ too: on a ROCm host its rocm tag would keep the guard active in the CUDA case.
    monkeypatch.setattr(
        torch, "__version__", "2.12.1+rocm6.4" if hip else "2.12.1+cu130", raising = False
    )


def test_install_fast_path_hooks_sets_fla_tilelang_zero_on_hip(monkeypatch):
    """tilelang 0.1.8 has no HIP GEMM, so a pre-existing pip tilelang must not be dispatched to."""
    monkeypatch.delenv("FLA_TILELANG", raising = False)
    monkeypatch.delenv(worker._FAST_PATH_HOOKS_SKIP_ENV, raising = False)
    _force_torch_hip(monkeypatch, "6.4.43483")
    _patch_iu_gate(monkeypatch, _make_fake_gate(initial_return = True))
    monkeypatch.setattr(worker, "_install_package_wheel_first", lambda **kw: True)

    worker._install_fast_path_hooks(event_queue = _FakeQueue(), model_name = "unsloth/Qwen3.5-2B")

    assert os.environ.get("FLA_TILELANG") == "0"


def test_install_fast_path_hooks_guards_tilelang_even_when_hooks_are_skipped(monkeypatch):
    """The opt-out skips the hooks, not the ROCm / tvm-ffi protection."""
    monkeypatch.delenv("FLA_TILELANG", raising = False)
    monkeypatch.setenv(worker._FAST_PATH_HOOKS_SKIP_ENV, "1")
    _force_torch_hip(monkeypatch, "6.4.43483")
    gate = _make_fake_gate(initial_return = True)
    _patch_iu_gate(monkeypatch, gate)

    worker._install_fast_path_hooks(event_queue = _FakeQueue(), model_name = "unsloth/Qwen3.5-2B")

    assert os.environ.get("FLA_TILELANG") == "0"


def test_install_fast_path_hooks_sets_fla_tilelang_zero_on_rocm_tagged_torch(monkeypatch):
    """AMD SDK / Radeon wheels can leave torch.version.hip unset and only tag __version__."""
    monkeypatch.delenv("FLA_TILELANG", raising = False)
    monkeypatch.delenv(worker._FAST_PATH_HOOKS_SKIP_ENV, raising = False)
    _force_torch_hip(monkeypatch, None)
    import torch

    monkeypatch.setattr(torch, "__version__", "2.11.0+rocm7.1", raising = False)
    _patch_iu_gate(monkeypatch, _make_fake_gate(initial_return = True))
    monkeypatch.setattr(worker, "_install_package_wheel_first", lambda **kw: True)

    worker._install_fast_path_hooks(event_queue = _FakeQueue(), model_name = "unsloth/Qwen3.5-2B")

    assert os.environ.get("FLA_TILELANG") == "0"


def _force_tvm_ffi(monkeypatch, *, tilelang_present: bool, tvm_ffi_version: str | None):
    """Fake the two probes _guard_fla_tilelang uses to spot a leftover TileLang stack."""
    real_find_spec = worker.importlib.util.find_spec
    real_version = worker.importlib.metadata.version

    def fake_find_spec(name, *a, **kw):
        if name == "tilelang":
            return object() if tilelang_present else None
        return real_find_spec(name, *a, **kw)

    def fake_version(name, *a, **kw):
        if name == "apache-tvm-ffi":
            if tvm_ffi_version is None:
                raise importlib.metadata.PackageNotFoundError(name)
            return tvm_ffi_version
        return real_version(name, *a, **kw)

    monkeypatch.setattr(worker.importlib.util, "find_spec", fake_find_spec)
    monkeypatch.setattr(worker.importlib.metadata, "version", fake_version)


@pytest.mark.parametrize(
    "tilelang_present, tvm_ffi_version, expected",
    [
        (True, "0.1.10", "0"),
        (True, "0.1.11", "0"),
        (False, "0.1.10", None),
        (True, "0.1.9", None),
        (True, None, None),
    ],
    ids = ["broken-0.1.10", "broken-0.1.11", "no-tilelang", "healthy", "tvm-ffi-missing"],
)
def test_guard_fla_tilelang_disables_only_for_a_broken_tvm_ffi_with_tilelang(
    monkeypatch, tilelang_present, tvm_ffi_version, expected
):
    """apache-tvm-ffi 0.1.10/0.1.11 fault under TileLang; only that pair may flip the env."""
    monkeypatch.delenv("FLA_TILELANG", raising = False)
    _force_torch_hip(monkeypatch, None)
    _force_tvm_ffi(monkeypatch, tilelang_present = tilelang_present, tvm_ffi_version = tvm_ffi_version)

    worker._guard_fla_tilelang()

    assert os.environ.get("FLA_TILELANG") == expected


def test_guard_fla_tilelang_respects_user_override_on_a_broken_tvm_ffi(monkeypatch):
    """A user who set FLA_TILELANG=1 keeps it even with the faulting tvm-ffi installed."""
    monkeypatch.setenv("FLA_TILELANG", "1")
    _force_torch_hip(monkeypatch, None)
    _force_tvm_ffi(monkeypatch, tilelang_present = True, tvm_ffi_version = "0.1.10")

    worker._guard_fla_tilelang()

    assert os.environ["FLA_TILELANG"] == "1"


def test_guard_fla_tilelang_does_not_log_disabling_under_a_user_override(monkeypatch):
    """setdefault is a no-op when FLA_TILELANG=1 is preset, so a disabling line would be a lie."""
    monkeypatch.setenv("FLA_TILELANG", "1")
    _force_torch_hip(monkeypatch, None)
    _force_tvm_ffi(monkeypatch, tilelang_present = True, tvm_ffi_version = "0.1.10")
    # worker.logger is a structlog BoundLogger, so caplog never sees it.
    messages: list[str] = []
    monkeypatch.setattr(worker.logger, "info", lambda msg, *a: messages.append(msg % a))

    worker._guard_fla_tilelang()

    assert not any("Disabling" in line for line in messages), messages
    assert any("Keeping FLA_TILELANG=1" in line for line in messages), messages


def test_install_fast_path_hooks_respects_user_fla_tilelang_override(monkeypatch):
    """If the user set FLA_TILELANG (even on HIP), don't overwrite; they may have a HIP-aware fork."""
    monkeypatch.setenv("FLA_TILELANG", "1")
    monkeypatch.delenv(worker._FAST_PATH_HOOKS_SKIP_ENV, raising = False)
    _force_torch_hip(monkeypatch, "6.4.43483")
    _patch_iu_gate(monkeypatch, _make_fake_gate(initial_return = True))
    monkeypatch.setattr(worker, "_install_package_wheel_first", lambda **kw: True)

    worker._install_fast_path_hooks(event_queue = _FakeQueue(), model_name = "unsloth/Qwen3.5-2B")

    assert os.environ["FLA_TILELANG"] == "1"


def test_install_fast_path_hooks_does_not_set_fla_tilelang_on_cuda(monkeypatch):
    """CUDA path must NOT set FLA_TILELANG (tilelang is wanted there)."""
    monkeypatch.delenv("FLA_TILELANG", raising = False)
    monkeypatch.delenv(worker._FAST_PATH_HOOKS_SKIP_ENV, raising = False)
    _force_torch_hip(monkeypatch, None)
    # Pin the second probe too, so a tilelang that happens to be in the test venv can't flip it.
    _force_tvm_ffi(monkeypatch, tilelang_present = False, tvm_ffi_version = None)
    _patch_iu_gate(monkeypatch, _make_fake_gate(initial_return = True))
    monkeypatch.setattr(worker, "_install_package_wheel_first", lambda **kw: True)

    worker._install_fast_path_hooks(event_queue = _FakeQueue(), model_name = "unsloth/Qwen3.5-2B")

    assert os.environ.get("FLA_TILELANG") is None


def _isdir_for_layout(*existing: str):
    """os.path.isdir replacement treating only the given absolute paths as
    directories, to simulate which gcc runtime / C++ header dirs exist."""
    valid = set(existing)

    def fake_isdir(path: str) -> bool:
        return path in valid

    return fake_isdir


def test_hipcc_gcc_install_dir_picks_highest_with_headers(monkeypatch):
    """gcc-14 has runtime but no /usr/include/c++/14; loop falls through
    to gcc-13 which has both. The exact Ubuntu 24.04 layout."""
    monkeypatch.setattr(sys, "platform", "linux")
    import platform as _platform

    monkeypatch.setattr(_platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(
        worker.os.path,
        "isdir",
        _isdir_for_layout(
            "/usr/lib/gcc/x86_64-linux-gnu/14/include",  # runtime present
            # but no /usr/include/c++/14 — typical Ubuntu 24.04 default
            "/usr/lib/gcc/x86_64-linux-gnu/13/include",
            "/usr/include/c++/13",  # libstdc++-13-dev installed
        ),
    )
    assert worker._hipcc_gcc_install_dir() == "/usr/lib/gcc/x86_64-linux-gnu/13"


def test_hipcc_gcc_install_dir_picks_14_when_headers_exist(monkeypatch):
    """If the user has libstdc++-14-dev installed, prefer gcc-14."""
    monkeypatch.setattr(sys, "platform", "linux")
    import platform as _platform

    monkeypatch.setattr(_platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(
        worker.os.path,
        "isdir",
        _isdir_for_layout(
            "/usr/lib/gcc/x86_64-linux-gnu/14/include",
            "/usr/include/c++/14",
        ),
    )
    assert worker._hipcc_gcc_install_dir() == "/usr/lib/gcc/x86_64-linux-gnu/14"


def test_hipcc_gcc_install_dir_returns_none_when_no_match(monkeypatch):
    """No gcc dir has both halves → return None and skip env injection
    rather than guessing wrong and causing a confusing build failure."""
    monkeypatch.setattr(sys, "platform", "linux")
    import platform as _platform

    monkeypatch.setattr(_platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(worker.os.path, "isdir", lambda path: False)
    assert worker._hipcc_gcc_install_dir() is None


def test_hipcc_gcc_install_dir_returns_none_on_non_linux(monkeypatch):
    """Don't probe gcc layout on macOS / Windows — early-return."""
    monkeypatch.setattr(sys, "platform", "darwin")

    def _isdir_should_not_be_called(_path):
        raise AssertionError("isdir should not be called on non-Linux")

    monkeypatch.setattr(worker.os.path, "isdir", _isdir_should_not_be_called)
    assert worker._hipcc_gcc_install_dir() is None


def test_hipcc_gcc_install_dir_returns_none_on_non_x86_64(monkeypatch):
    """ROCm clang-20 on aarch64 has a different libstdc++ layout."""
    monkeypatch.setattr(sys, "platform", "linux")
    import platform as _platform

    monkeypatch.setattr(_platform, "machine", lambda: "aarch64")
    assert worker._hipcc_gcc_install_dir() is None


def _make_hip_install_env(monkeypatch, *, gcc_dir: str | None):
    """Scaffolding for end-to-end tests of the HIP source-build branch of
    _install_package_wheel_first: package not installed, no prebuilt
    wheel, hipcc on PATH, fake env reports HIP torch."""
    monkeypatch.setattr(builtins, "__import__", _missing_module_import("causal_conv1d"))
    monkeypatch.setattr(
        worker,
        "probe_torch_wheel_env",
        lambda timeout = 30: {
            "hip_version": "7.13.26176",
            "python_tag": "cp312",
            "torch_mm": "2.11",
            "cxx11abi": "TRUE",
            "platform_tag": "linux_x86_64",
        },
    )
    monkeypatch.setattr(worker, "direct_wheel_url", lambda **kw: None)
    monkeypatch.setattr(
        worker.shutil,
        "which",
        lambda name: "/opt/rocm/bin/hipcc" if name == "hipcc" else None,
    )
    monkeypatch.setattr(worker, "_send_status", lambda *a, **k: None)
    monkeypatch.setattr(worker, "_hipcc_gcc_install_dir", lambda: gcc_dir)


def test_install_injects_gcc_install_dir_on_hip_source_build(monkeypatch):
    """HIP source-build with no user-set HIPCC_COMPILE_FLAGS_APPEND →
    subprocess env carries --gcc-install-dir=<detected path>."""
    monkeypatch.delenv("HIPCC_COMPILE_FLAGS_APPEND", raising = False)
    _make_hip_install_env(monkeypatch, gcc_dir = "/usr/lib/gcc/x86_64-linux-gnu/13")

    captured: dict[str, str] = {}

    def fake_run(cmd, **kwargs):
        captured.update(kwargs.get("env") or {})
        return subprocess.CompletedProcess(cmd, 0, "")

    monkeypatch.setattr(worker._sp, "run", fake_run)

    worker._install_package_wheel_first(
        event_queue = [],
        import_name = "causal_conv1d",
        display_name = "causal-conv1d",
        pypi_name = "causal-conv1d",
        pypi_version = "1.6.2.post1",
        filename_prefix = "causal_conv1d",
        release_tag = "v1.6.2.post1",
        release_base_url = "https://example.com",
    )

    assert (
        captured.get("HIPCC_COMPILE_FLAGS_APPEND")
        == "--gcc-install-dir=/usr/lib/gcc/x86_64-linux-gnu/13"
    )


def test_install_appends_to_existing_hipcc_compile_flags(monkeypatch):
    """User has HIPCC_COMPILE_FLAGS_APPEND='-O3 -DFOO' → final value keeps
    the user's flags AND appends --gcc-install-dir."""
    monkeypatch.setenv("HIPCC_COMPILE_FLAGS_APPEND", "-O3 -DFOO")
    _make_hip_install_env(monkeypatch, gcc_dir = "/usr/lib/gcc/x86_64-linux-gnu/13")

    captured: dict[str, str] = {}

    def fake_run(cmd, **kwargs):
        captured.update(kwargs.get("env") or {})
        return subprocess.CompletedProcess(cmd, 0, "")

    monkeypatch.setattr(worker._sp, "run", fake_run)

    worker._install_package_wheel_first(
        event_queue = [],
        import_name = "causal_conv1d",
        display_name = "causal-conv1d",
        pypi_name = "causal-conv1d",
        pypi_version = "1.6.2.post1",
        filename_prefix = "causal_conv1d",
        release_tag = "v1.6.2.post1",
        release_base_url = "https://example.com",
    )

    assert captured.get("HIPCC_COMPILE_FLAGS_APPEND") == (
        "-O3 -DFOO --gcc-install-dir=/usr/lib/gcc/x86_64-linux-gnu/13"
    )


def test_install_respects_user_gcc_install_dir(monkeypatch):
    """User explicitly set --gcc-install-dir=… already → don't touch it.
    Avoids two competing --gcc-install-dir flags on the clang command line."""
    monkeypatch.setenv(
        "HIPCC_COMPILE_FLAGS_APPEND",
        "--gcc-install-dir=/opt/custom/gcc-13",
    )
    _make_hip_install_env(monkeypatch, gcc_dir = "/usr/lib/gcc/x86_64-linux-gnu/13")

    captured: dict[str, str] = {}

    def fake_run(cmd, **kwargs):
        captured.update(kwargs.get("env") or {})
        return subprocess.CompletedProcess(cmd, 0, "")

    monkeypatch.setattr(worker._sp, "run", fake_run)

    worker._install_package_wheel_first(
        event_queue = [],
        import_name = "causal_conv1d",
        display_name = "causal-conv1d",
        pypi_name = "causal-conv1d",
        pypi_version = "1.6.2.post1",
        filename_prefix = "causal_conv1d",
        release_tag = "v1.6.2.post1",
        release_base_url = "https://example.com",
    )

    assert captured["HIPCC_COMPILE_FLAGS_APPEND"] == "--gcc-install-dir=/opt/custom/gcc-13"


def test_install_does_not_inject_env_on_cuda(monkeypatch):
    """CUDA path (no hip_version in env) → no HIP flag injected."""
    monkeypatch.delenv("HIPCC_COMPILE_FLAGS_APPEND", raising = False)
    monkeypatch.setattr(builtins, "__import__", _missing_module_import("causal_conv1d"))
    monkeypatch.setattr(
        worker,
        "probe_torch_wheel_env",
        lambda timeout = 30: {
            "python_tag": "cp312",
            "torch_mm": "2.11",
            "cuda_major": "12",
            "cxx11abi": "TRUE",
            "platform_tag": "linux_x86_64",
        },
    )
    monkeypatch.setattr(worker, "direct_wheel_url", lambda **kw: None)
    monkeypatch.setattr(worker.shutil, "which", lambda name: None)
    monkeypatch.setattr(worker, "_send_status", lambda *a, **k: None)
    # _hipcc_gcc_install_dir must not be called on CUDA.
    monkeypatch.setattr(
        worker,
        "_hipcc_gcc_install_dir",
        lambda: (_ for _ in ()).throw(AssertionError("must not run on CUDA")),
    )

    captured: dict[str, Any] = {}

    def fake_run(cmd, **kwargs):
        captured.update(kwargs.get("env") or {})
        return subprocess.CompletedProcess(cmd, 0, "")

    monkeypatch.setattr(worker._sp, "run", fake_run)

    worker._install_package_wheel_first(
        event_queue = [],
        import_name = "causal_conv1d",
        display_name = "causal-conv1d",
        pypi_name = "causal-conv1d",
        pypi_version = "1.6.2.post1",
        filename_prefix = "causal_conv1d",
        release_tag = "v1.6.2.post1",
        release_base_url = "https://example.com",
    )

    # env is always passed (to force UTF-8), but never the HIP flag.
    assert "HIPCC_COMPILE_FLAGS_APPEND" not in captured
