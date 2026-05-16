# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import builtins
import subprocess
import sys
from unittest import mock

from core.training import worker
from utils.wheel_utils import (
    CAUSAL_CONV1D_SPEC,
    FLASH_ATTN_SPEC,
    FLASH_LINEAR_ATTN_SPEC,
    MAMBA_SSM_SPEC,
)


def _record_train_time_installs(monkeypatch, tilelang_mock = None):
    calls = []

    def fake_install(spec, **kwargs):
        calls.append((spec, kwargs))
        return True

    if tilelang_mock is None:
        tilelang_mock = mock.Mock(return_value = False)
    monkeypatch.setattr(worker, "has_blackwell_gpu", lambda: False)
    monkeypatch.setattr(worker, "install_optional_kernel", fake_install)
    monkeypatch.setattr(worker, "_ensure_tilelang_backend", tilelang_mock)
    return calls


def test_should_try_runtime_flash_attn_install_threshold_and_skip(monkeypatch):
    monkeypatch.delenv(worker._FLASH_ATTN_SKIP_ENV, raising = False)
    assert worker._should_try_runtime_flash_attn_install(32768) is False
    assert worker._should_try_runtime_flash_attn_install(
        32769
    ) is sys.platform.startswith("linux")

    monkeypatch.setenv(worker._FLASH_ATTN_SKIP_ENV, "1")
    assert worker._should_try_runtime_flash_attn_install(32769) is False


def test_causal_conv1d_models_also_trigger_fla(monkeypatch):
    monkeypatch.delenv(worker._FLASH_ATTN_SKIP_ENV, raising = False)
    calls = _record_train_time_installs(monkeypatch)

    worker._install_train_time_optional_kernels(
        event_queue = [],
        model_name = "unsloth/Qwen3-Next-8B",
        max_seq_length = 2048,
    )

    specs = [spec for spec, _ in calls]
    assert specs == [CAUSAL_CONV1D_SPEC, FLASH_LINEAR_ATTN_SPEC]


def test_tilelang_backend_runs_after_fla_for_qwen3_5(monkeypatch):
    monkeypatch.delenv(worker._FLASH_ATTN_SKIP_ENV, raising = False)
    monkeypatch.delenv(worker._TILELANG_SKIP_ENV, raising = False)
    tilelang_mock = mock.Mock(return_value = True)
    calls = _record_train_time_installs(monkeypatch, tilelang_mock)
    event_queue = []

    worker._install_train_time_optional_kernels(
        event_queue = event_queue,
        model_name = "unsloth/Qwen3.5-2B",
        max_seq_length = 2048,
    )

    assert [spec for spec, _ in calls] == [CAUSAL_CONV1D_SPEC, FLASH_LINEAR_ATTN_SPEC]
    tilelang_mock.assert_called_once_with(
        event_queue,
        model_name = "unsloth/Qwen3.5-2B",
        blackwell = False,
    )


def test_causal_conv1d_condition_includes_qwen3_6_variants(monkeypatch):
    monkeypatch.delenv(worker._FLASH_ATTN_SKIP_ENV, raising = False)
    calls = _record_train_time_installs(monkeypatch)

    worker._install_train_time_optional_kernels(
        event_queue = [],
        model_name = "unsloth/Qwen3.6-4B",
        max_seq_length = 2048,
    )
    worker._install_train_time_optional_kernels(
        event_queue = [],
        model_name = "unsloth/Qwen3_6-4B",
        max_seq_length = 2048,
    )

    assert [spec for spec, _ in calls] == [
        CAUSAL_CONV1D_SPEC,
        FLASH_LINEAR_ATTN_SPEC,
        CAUSAL_CONV1D_SPEC,
        FLASH_LINEAR_ATTN_SPEC,
    ]


def test_ssm_models_trigger_mamba(monkeypatch):
    monkeypatch.delenv(worker._FLASH_ATTN_SKIP_ENV, raising = False)
    calls = _record_train_time_installs(monkeypatch)

    worker._install_train_time_optional_kernels(
        event_queue = [],
        model_name = "tiiuae/Falcon-H1-0.5B-Instruct",
        max_seq_length = 2048,
    )

    specs = [spec for spec, _ in calls]
    assert MAMBA_SSM_SPEC in specs


def test_runtime_flash_attn_uses_strictly_greater_than_32k(monkeypatch):
    monkeypatch.delenv(worker._FLASH_ATTN_SKIP_ENV, raising = False)
    monkeypatch.setattr(worker, "has_blackwell_gpu", lambda: False)
    calls = _record_train_time_installs(monkeypatch)

    worker._install_train_time_optional_kernels(
        event_queue = [],
        model_name = "unsloth/Llama-3.1-8B",
        max_seq_length = 32768,
    )

    assert [spec for spec, _ in calls] == []

    worker._install_train_time_optional_kernels(
        event_queue = [],
        model_name = "unsloth/Llama-3.1-8B",
        max_seq_length = 32769,
    )

    specs = [spec for spec, _ in calls]
    if sys.platform.startswith("linux"):
        assert specs == [FLASH_ATTN_SPEC]
    else:
        assert specs == []


def test_train_time_installs_allow_pypi_fallback(monkeypatch):
    monkeypatch.delenv(worker._FLASH_ATTN_SKIP_ENV, raising = False)
    monkeypatch.setattr(worker, "has_blackwell_gpu", lambda: False)
    calls = _record_train_time_installs(monkeypatch)

    worker._install_train_time_optional_kernels(
        event_queue = [],
        model_name = "tiiuae/Falcon-H1-0.5B-Instruct",
        max_seq_length = 32769,
    )

    assert calls
    assert all(kwargs["allow_pypi_fallback"] is True for _, kwargs in calls)


def test_train_time_flash_attn_failure_is_non_fatal_and_warns(monkeypatch):
    statuses: list[str] = []

    monkeypatch.delenv(worker._FLASH_ATTN_SKIP_ENV, raising = False)
    monkeypatch.setattr(worker, "has_blackwell_gpu", lambda: False)
    monkeypatch.setattr(
        worker,
        "_send_status",
        lambda queue, message: statuses.append(message),
    )

    def fake_install(spec, **kwargs):
        return spec is not FLASH_ATTN_SPEC

    monkeypatch.setattr(worker, "install_optional_kernel", fake_install)

    worker._install_train_time_optional_kernels(
        event_queue = [],
        model_name = "unsloth/Llama-3.1-8B",
        max_seq_length = 32769,
    )

    if sys.platform.startswith("linux"):
        assert statuses == ["Continuing without flash-attn"]
    else:
        assert statuses == []


def test_runtime_flash_attn_skip_env_avoids_install(monkeypatch):
    monkeypatch.setenv(worker._FLASH_ATTN_SKIP_ENV, "1")
    install_mock = mock.Mock()
    monkeypatch.setattr(worker, "install_optional_kernel", install_mock)

    worker._install_train_time_optional_kernels(
        event_queue = [],
        model_name = "unsloth/Llama-3.1-8B",
        max_seq_length = 32769,
    )

    install_mock.assert_not_called()


def test_runtime_flash_attn_skips_on_blackwell(monkeypatch):
    statuses: list[str] = []
    install_mock = mock.Mock(return_value = True)

    monkeypatch.delenv(worker._FLASH_ATTN_SKIP_ENV, raising = False)
    monkeypatch.setattr(worker, "has_blackwell_gpu", lambda: True)
    monkeypatch.setattr(worker, "install_optional_kernel", install_mock)
    monkeypatch.setattr(
        worker,
        "_send_status",
        lambda queue, message: statuses.append(message),
    )

    worker._install_train_time_optional_kernels(
        event_queue = [],
        model_name = "unsloth/Llama-3.1-8B",
        max_seq_length = 32769,
    )

    if sys.platform.startswith("linux"):
        install_mock.assert_not_called()
        assert len(statuses) == 1
        assert "Blackwell" in statuses[0]
    else:
        install_mock.assert_not_called()
        assert statuses == []


def test_flash_linear_attention_skips_on_blackwell_but_causal_conv1d_runs(
    monkeypatch,
):
    statuses: list[str] = []
    calls = _record_train_time_installs(monkeypatch)

    monkeypatch.delenv(worker._FLASH_ATTN_SKIP_ENV, raising = False)
    monkeypatch.setattr(worker, "has_blackwell_gpu", lambda: True)
    monkeypatch.setattr(
        worker,
        "_send_status",
        lambda queue, message: statuses.append(message),
    )

    worker._install_train_time_optional_kernels(
        event_queue = [],
        model_name = "unsloth/Qwen3-Next-8B",
        max_seq_length = 2048,
    )

    assert [spec for spec, _ in calls] == [CAUSAL_CONV1D_SPEC]
    assert len(statuses) == 1
    assert "flash-linear-attention" in statuses[0]
    assert "Blackwell" in statuses[0]


def test_blackwell_does_not_block_causal_conv1d_or_mamba(monkeypatch):
    statuses: list[str] = []
    calls = _record_train_time_installs(monkeypatch)

    monkeypatch.delenv(worker._FLASH_ATTN_SKIP_ENV, raising = False)
    monkeypatch.setattr(worker, "has_blackwell_gpu", lambda: True)
    monkeypatch.setattr(
        worker,
        "_send_status",
        lambda queue, message: statuses.append(message),
    )

    worker._install_train_time_optional_kernels(
        event_queue = [],
        model_name = "tiiuae/Falcon-H1-0.5B-Instruct",
        max_seq_length = 32769,
    )

    specs = [spec for spec, _ in calls]
    assert CAUSAL_CONV1D_SPEC in specs
    assert MAMBA_SSM_SPEC in specs
    assert FLASH_LINEAR_ATTN_SPEC not in specs
    assert FLASH_ATTN_SPEC not in specs
    assert any("flash-linear-attention" in message for message in statuses)
    if sys.platform.startswith("linux"):
        assert any("flash-attn" in message for message in statuses)


def test_tilelang_backend_installs_pinned_pair_for_qwen3_5(monkeypatch):
    monkeypatch.delenv(worker._TILELANG_SKIP_ENV, raising = False)
    monkeypatch.setattr(worker.shutil, "which", lambda name: "/usr/bin/uv")
    run_mock = mock.Mock(return_value = subprocess.CompletedProcess(["uv"], 0, ""))
    monkeypatch.setattr(worker.subprocess, "run", run_mock)

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name in ("tilelang", "tvm_ffi"):
            raise ImportError
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    statuses: list[str] = []
    monkeypatch.setattr(worker, "_send_status", lambda queue, msg: statuses.append(msg))

    assert (
        worker._ensure_tilelang_backend(
            event_queue = [],
            model_name = "unsloth/Qwen3.5-2B",
            blackwell = False,
        )
        is True
    )

    run_mock.assert_called_once()
    cmd = run_mock.call_args.args[0]
    assert cmd[:5] == ["uv", "pip", "install", "--python", sys.executable]
    assert f"apache-tvm-ffi=={worker._APACHE_TVM_FFI_PACKAGE_VERSION}" in cmd
    assert f"tilelang=={worker._TILELANG_PACKAGE_VERSION}" in cmd
    assert "--no-deps" not in cmd
    assert "--upgrade" not in cmd
    assert "--reinstall" not in cmd
    assert any("TileLang backend" in message for message in statuses)


def test_tilelang_backend_skips_for_blackwell(monkeypatch):
    monkeypatch.delenv(worker._TILELANG_SKIP_ENV, raising = False)
    run_mock = mock.Mock()
    monkeypatch.setattr(worker.subprocess, "run", run_mock)

    assert (
        worker._ensure_tilelang_backend(
            event_queue = [],
            model_name = "unsloth/Qwen3.5-2B",
            blackwell = True,
        )
        is False
    )

    run_mock.assert_not_called()


def test_tilelang_backend_skips_for_ssm_models(monkeypatch):
    monkeypatch.delenv(worker._TILELANG_SKIP_ENV, raising = False)
    run_mock = mock.Mock()
    monkeypatch.setattr(worker.subprocess, "run", run_mock)

    for name in (
        "tiiuae/Falcon-H1-0.5B-Instruct",
        "nvidia/Nemotron-H-8B-Base",
        "ibm-granite/granite-4.0-h-tiny",
        "meta-llama/Llama-3.2-1B-Instruct",
    ):
        assert (
            worker._ensure_tilelang_backend(
                event_queue = [],
                model_name = name,
                blackwell = False,
            )
            is False
        )

    run_mock.assert_not_called()


def test_tilelang_backend_skips_via_env(monkeypatch):
    monkeypatch.setenv(worker._TILELANG_SKIP_ENV, "1")
    run_mock = mock.Mock()
    monkeypatch.setattr(worker.subprocess, "run", run_mock)

    assert (
        worker._ensure_tilelang_backend(
            event_queue = [],
            model_name = "unsloth/Qwen3.5-2B",
            blackwell = False,
        )
        is False
    )

    run_mock.assert_not_called()


def test_tilelang_backend_failure_is_non_fatal(monkeypatch):
    monkeypatch.delenv(worker._TILELANG_SKIP_ENV, raising = False)
    monkeypatch.setattr(worker.shutil, "which", lambda name: None)
    run_mock = mock.Mock(
        return_value = subprocess.CompletedProcess(["pip"], 1, "install failed")
    )
    monkeypatch.setattr(worker.subprocess, "run", run_mock)

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name in ("tilelang", "tvm_ffi"):
            raise ImportError
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    statuses: list[str] = []
    monkeypatch.setattr(worker, "_send_status", lambda queue, msg: statuses.append(msg))

    assert (
        worker._ensure_tilelang_backend(
            event_queue = [],
            model_name = "unsloth/Qwen3.5-2B",
            blackwell = False,
        )
        is False
    )

    run_mock.assert_called_once()
    assert any("failed" in message.lower() for message in statuses)
