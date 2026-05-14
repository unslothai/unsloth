# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import builtins
import subprocess
import sys
from unittest import mock

from core.training import worker


def _missing_flash_attn_import():
    real_import = builtins.__import__

    def fake_import(name, globals = None, locals = None, fromlist = (), level = 0):
        if name == "flash_attn":
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
