# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import builtins
import subprocess
import sys
from typing import Any
from unittest import mock

from core.training import worker


def _missing_flash_attn_import():
    real_import = builtins.__import__

    def fake_import(name, globals = None, locals = None, fromlist = (), level = 0):
        if name == "flash_attn":
            raise ImportError
        return real_import(name, globals, locals, fromlist, level)

    return fake_import


def _missing_module_import(missing: str):
    real_import = builtins.__import__

    def fake_import(name, globals = None, locals = None, fromlist = (), level = 0):
        if name == missing:
            raise ImportError
        return real_import(name, globals, locals, fromlist, level)

    return fake_import


def test_should_try_runtime_flash_attn_install_threshold_and_skip(monkeypatch):
    monkeypatch.delenv(worker._FLASH_ATTN_SKIP_ENV, raising = False)
    assert worker._should_try_runtime_flash_attn_install(32767) is False
    assert worker._should_try_runtime_flash_attn_install(
        32768
    ) is sys.platform.startswith("linux")

    monkeypatch.setenv(worker._FLASH_ATTN_SKIP_ENV, "1")
    assert worker._should_try_runtime_flash_attn_install(32768) is False


def test_runtime_flash_attn_prefers_prebuilt_wheel(monkeypatch):
    statuses: list[str] = []

    monkeypatch.delenv(worker._FLASH_ATTN_SKIP_ENV, raising = False)
    monkeypatch.setattr(worker, "has_blackwell_gpu", lambda: False)
    monkeypatch.setattr(builtins, "__import__", _missing_flash_attn_import())
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

    worker._ensure_flash_attn_for_long_context(event_queue = [], max_seq_length = 32768)

    assert statuses == ["Installing prebuilt flash-attn wheel..."]


def test_runtime_flash_attn_falls_back_to_pypi(monkeypatch):
    calls: list[list[str]] = []
    statuses: list[str] = []

    monkeypatch.delenv(worker._FLASH_ATTN_SKIP_ENV, raising = False)
    monkeypatch.setattr(worker, "has_blackwell_gpu", lambda: False)
    monkeypatch.setattr(builtins, "__import__", _missing_flash_attn_import())
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
    monkeypatch.setattr(
        worker,
        "_send_status",
        lambda queue, message: statuses.append(message),
    )
    monkeypatch.setattr(worker, "install_wheel", mock.Mock())

    def fake_run(cmd, stdout = None, stderr = None, text = None):
        calls.append(list(cmd))
        return subprocess.CompletedProcess(cmd, 0, "")

    monkeypatch.setattr(worker._sp, "run", fake_run)

    worker._ensure_flash_attn_for_long_context(event_queue = [], max_seq_length = 32768)

    assert statuses == ["Installing flash-attn from PyPI for long-context training..."]
    assert calls == [[sys.executable, "-m", "pip", "install", "flash-attn"]]


def test_runtime_flash_attn_skip_env_avoids_all_install_work(monkeypatch):
    monkeypatch.setenv(worker._FLASH_ATTN_SKIP_ENV, "1")
    monkeypatch.setattr(worker._sp, "run", mock.Mock())

    worker._ensure_flash_attn_for_long_context(event_queue = [], max_seq_length = 32768)

    worker._sp.run.assert_not_called()


def test_runtime_flash_attn_skips_on_blackwell(monkeypatch):
    statuses: list[str] = []
    install_mock = mock.Mock()

    monkeypatch.delenv(worker._FLASH_ATTN_SKIP_ENV, raising = False)
    monkeypatch.setattr(
        worker, "_should_try_runtime_flash_attn_install", lambda max_seq: True
    )
    monkeypatch.setattr(worker, "has_blackwell_gpu", lambda: True)
    monkeypatch.setattr(worker, "_install_package_wheel_first", install_mock)
    monkeypatch.setattr(
        worker,
        "_send_status",
        lambda queue, message: statuses.append(message),
    )

    worker._ensure_flash_attn_for_long_context(event_queue = [], max_seq_length = 65536)

    install_mock.assert_not_called()
    assert len(statuses) == 1
    assert "Blackwell" in statuses[0]


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


# ────────────────────────────────────────────────────────────────────
# HIP source-build gcc-install-dir coverage (h34v3nzc0dex Strix Halo).
# Ubuntu 24.04 ships gcc-14's runtime dir without /usr/include/c++/14,
# so ROCm clang-20 picks it and fails with 'cstdlib' file not found
# when building causal-conv1d (or any other HIP source fallback).
# _hipcc_gcc_install_dir() finds a gcc dir that has both halves; the
# _install_package_wheel_first HIP branch passes it to clang via
# HIPCC_COMPILE_FLAGS_APPEND. Parallel to bbf004c's setup.sh fix for
# the llama.cpp HIP build (PR #5301).
# ────────────────────────────────────────────────────────────────────


def _isdir_for_layout(*existing: str):
    """Return an os.path.isdir replacement that only treats the given
    absolute paths as directories. Lets a test simulate exactly which
    gcc runtime dirs and C++ header dirs exist on the host."""
    valid = set(existing)

    def fake_isdir(path: str) -> bool:
        return path in valid

    return fake_isdir


def test_hipcc_gcc_install_dir_picks_highest_with_headers(monkeypatch):
    """gcc-14 has runtime but no /usr/include/c++/14; loop falls through
    to gcc-13 which has both. This is the exact Ubuntu 24.04 layout."""
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
    """No gcc dir has both halves → return None and skip the env injection
    rather than guessing wrong and surfacing a confusing build failure."""
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
    """Common scaffolding for tests that exercise the HIP source-build
    branch of _install_package_wheel_first end-to-end. The package isn't
    installed yet, no prebuilt wheel exists, hipcc is on PATH, and the
    fake env reports an HIP torch."""
    monkeypatch.setattr(
        builtins, "__import__", _missing_module_import("causal_conv1d")
    )
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
    _make_hip_install_env(
        monkeypatch, gcc_dir = "/usr/lib/gcc/x86_64-linux-gnu/13"
    )

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
    """User has HIPCC_COMPILE_FLAGS_APPEND='-O3 -DFOO' set → final value
    keeps the user's flags AND adds --gcc-install-dir at the end."""
    monkeypatch.setenv("HIPCC_COMPILE_FLAGS_APPEND", "-O3 -DFOO")
    _make_hip_install_env(
        monkeypatch, gcc_dir = "/usr/lib/gcc/x86_64-linux-gnu/13"
    )

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
    _make_hip_install_env(
        monkeypatch, gcc_dir = "/usr/lib/gcc/x86_64-linux-gnu/13"
    )

    captured: dict[str, str] | None = {"_called": "no"}

    def fake_run(cmd, **kwargs):
        env = kwargs.get("env")
        if env is not None:
            captured.clear()
            captured.update(env)
        else:
            captured["_called"] = "yes_no_env"
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

    # subprocess.run was invoked without env override (the user already
    # set HIPCC_COMPILE_FLAGS_APPEND with --gcc-install-dir, so we left
    # the env alone — the existing value is inherited normally).
    assert captured == {"_called": "yes_no_env"}


def test_install_does_not_inject_env_on_cuda(monkeypatch):
    """CUDA path (no hip_version in env) → no env override at all."""
    monkeypatch.delenv("HIPCC_COMPILE_FLAGS_APPEND", raising = False)
    monkeypatch.setattr(
        builtins, "__import__", _missing_module_import("causal_conv1d")
    )
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
    # If _hipcc_gcc_install_dir were called on CUDA we'd want to know.
    monkeypatch.setattr(
        worker,
        "_hipcc_gcc_install_dir",
        lambda: (_ for _ in ()).throw(
            AssertionError("must not run on CUDA")
        ),
    )

    captured: dict[str, Any] = {}

    def fake_run(cmd, **kwargs):
        captured["env_in_kwargs"] = "env" in kwargs
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

    # CUDA branch never sets the env, never invokes the gcc helper.
    assert captured.get("env_in_kwargs") is False
