# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The checkpointing budget and the dequant cache stay off on a discrete Windows ROCm card until measured: WDDM shows
the budgets the whole card as free and spills an overflow to host RAM instead of raising, so the OOM step-down that
protects CUDA never fires. A unified-memory APU (gfx1151, where both were measured), Linux ROCm and CUDA keep them."""

from __future__ import annotations

import sys

import pytest
import torch

import core.inference.host_pinning as hp
import core.training.diffusion_dit_trainer as dit
import core.training.diffusion_gc_budget as gb


class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.func = None

    def enable_gradient_checkpointing(self, gradient_checkpointing_func = None):
        self.func = gradient_checkpointing_func


@pytest.fixture
def host(monkeypatch):
    """(platform, rocm, integrated) -> configures a tag-only ROCm wheel (no torch.version.hip) or a CUDA one."""
    monkeypatch.delenv(gb.GC_MODE_ENV, raising = False)
    monkeypatch.delenv("UNSLOTH_DIFFUSION_DEQUANT_CACHE", raising = False)

    def set_host(platform, rocm, integrated):
        monkeypatch.setattr(sys, "platform", platform)
        monkeypatch.setattr(torch.version, "hip", None, raising = False)
        monkeypatch.setattr(torch, "__version__", "2.11.0+rocm7.14.0" if rocm else "2.11.0+cu128")
        monkeypatch.setattr(hp, "device_is_integrated", lambda device = None: integrated)

    return set_host


def _budget_on():
    return gb.enable_budgeted_checkpointing(_Model(), "cuda") is not None


def test_discrete_windows_rocm_keeps_the_budgets_off(host):
    host("win32", True, False)
    assert gb.rocm_windows_discrete() is True
    assert not _budget_on()
    assert not dit._dequant_cache_wanted(True, "cuda", False)


@pytest.mark.parametrize(
    "platform, rocm, integrated",
    [
        ("win32", True, True),
        ("linux", True, False),
        ("win32", False, False),
        ("linux", False, False),
    ],
)
def test_apu_linux_rocm_and_cuda_keep_the_budgets(host, platform, rocm, integrated):
    host(platform, rocm, integrated)
    assert gb.rocm_windows_discrete() is False
    assert _budget_on()
    assert dit._dequant_cache_wanted(True, "cuda", False)


def test_explicit_opt_in_on_discrete_windows_rocm(host, monkeypatch):
    host("win32", True, False)
    monkeypatch.setenv(gb.GC_MODE_ENV, "auto")
    assert _budget_on()
    monkeypatch.setenv("UNSLOTH_DIFFUSION_DEQUANT_CACHE", "1")
    assert dit._dequant_cache_wanted(True, "cuda", False)
    monkeypatch.setenv(gb.GC_MODE_ENV, "plain")
    monkeypatch.setenv("UNSLOTH_DIFFUSION_DEQUANT_CACHE", "0")
    assert not _budget_on() and not dit._dequant_cache_wanted(True, "cuda", False)


def test_rocm_detection_goes_through_is_rocm_torch():
    import inspect
    src = inspect.getsource(gb.rocm_windows_discrete)
    assert "is_rocm_torch" in src and "version.hip" not in src
