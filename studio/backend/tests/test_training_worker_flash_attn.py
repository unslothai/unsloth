# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import sys
from unittest import mock

from core.training import worker
from utils.wheel_utils import (
    CAUSAL_CONV1D_SPEC,
    FLASH_ATTN_SPEC,
    FLASH_LINEAR_ATTN_SPEC,
    MAMBA_SSM_SPEC,
)


def _record_train_time_installs(monkeypatch):
    calls = []

    def fake_install(spec, **kwargs):
        calls.append((spec, kwargs))
        return True

    monkeypatch.setattr(worker, "install_optional_kernel", fake_install)
    return calls


def test_should_try_runtime_flash_attn_install_threshold_and_skip(monkeypatch):
    monkeypatch.delenv(worker._FLASH_ATTN_SKIP_ENV, raising = False)
    assert worker._should_try_runtime_flash_attn_install(16384) is False
    assert worker._should_try_runtime_flash_attn_install(
        16385
    ) is sys.platform.startswith("linux")

    monkeypatch.setenv(worker._FLASH_ATTN_SKIP_ENV, "1")
    assert worker._should_try_runtime_flash_attn_install(16385) is False


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


def test_runtime_flash_attn_uses_strictly_greater_than_16k(monkeypatch):
    monkeypatch.delenv(worker._FLASH_ATTN_SKIP_ENV, raising = False)
    monkeypatch.setattr(worker, "has_blackwell_gpu", lambda: False)
    calls = _record_train_time_installs(monkeypatch)

    worker._install_train_time_optional_kernels(
        event_queue = [],
        model_name = "unsloth/Llama-3.1-8B",
        max_seq_length = 16384,
    )

    assert [spec for spec, _ in calls] == []

    worker._install_train_time_optional_kernels(
        event_queue = [],
        model_name = "unsloth/Llama-3.1-8B",
        max_seq_length = 16385,
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
        max_seq_length = 16385,
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
        max_seq_length = 16385,
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
        max_seq_length = 16385,
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
        max_seq_length = 16385,
    )

    if sys.platform.startswith("linux"):
        install_mock.assert_not_called()
        assert len(statuses) == 1
        assert "Blackwell" in statuses[0]
    else:
        install_mock.assert_not_called()
        assert statuses == []
