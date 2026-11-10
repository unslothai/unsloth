# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Flex attention is dropped on a ROCm card this Triton cannot build for (gfx101x on Triton 3.9)."""

import sys
import types
from unittest import mock

import pytest

import unsloth.models._utils as U


@pytest.fixture(autouse = True)
def _fresh_cache():
    # macOS / Windows runners have no Triton; the probe itself is stubbed per test.
    triton = sys.modules.get("triton") or types.ModuleType("triton")
    with (
        mock.patch.dict(U._TRITON_RUNS_ON_DEVICE, clear = True),
        mock.patch.dict(sys.modules, {"triton": triton}),
    ):
        yield


def _hip(
    on,
    count = 1,
    current = 0,
):
    return mock.patch.multiple(
        U.torch.cuda,
        is_available = lambda: count > 0,
        device_count = lambda: count,
        current_device = lambda: current,
    ), mock.patch.object(U.torch.version, "hip", "7.14" if on else None, create = True)


def _probe(results):
    calls = []

    def probe(index):
        calls.append(index)
        if isinstance(results[index], Exception):
            raise results[index]
        return results[index]

    probe.calls = calls
    return mock.patch.object(U, "_triton_probe", probe), calls


def test_cuda_builds_never_probe():
    cuda, hip = _hip(False)
    patch, calls = _probe({0: RuntimeError("unreachable")})
    with cuda, hip, patch, mock.patch.object(U.torch, "__version__", "2.14.1+cu130"):
        assert U.triton_runs_on_device(0) is True
    assert calls == []


def test_a_target_triton_rejects_is_refused_once():
    cuda, hip = _hip(True)
    patch, calls = _probe(
        {0: RuntimeError("error: unsupported target: 'gfx1010'\nPassManager::run failed")}
    )
    with cuda, hip, patch:
        assert U.triton_runs_on_device(0) is False
        assert U.triton_runs_on_device(0) is False
        assert U._flex_attention_gpu_is_supported() is False
    assert calls == [0]


def test_a_card_triton_runs_on_keeps_flex():
    cuda, hip = _hip(True)
    patch, calls = _probe({0: True})
    with cuda, hip, patch:
        assert U._flex_attention_gpu_is_supported() is True
    assert calls == [0]


@pytest.mark.parametrize("current, expected", [(0, True), (1, False)])
def test_only_the_current_card_is_probed(current, expected):
    """Probing every visible card would open a HIP context on each of them."""
    cuda, hip = _hip(True, count = 2, current = current)
    patch, calls = _probe({0: True, 1: RuntimeError("PassManager::run failed")})
    with cuda, hip, patch:
        assert U._flex_attention_gpu_is_supported() is expected
    assert calls == [current]


def test_a_rocm_build_without_a_card_is_left_alone():
    cuda, hip = _hip(True, count = 0)
    patch, calls = _probe({})
    with cuda, hip, patch:
        assert U._flex_attention_gpu_is_supported() is True
    assert calls == []


def test_no_triton_keeps_the_old_answer():
    cuda, hip = _hip(True)
    patch, calls = _probe({0: RuntimeError("unreachable")})
    with cuda, hip, patch, mock.patch.dict(sys.modules, {"triton": None}):
        assert U.triton_runs_on_device(0) is True
    assert calls == []
