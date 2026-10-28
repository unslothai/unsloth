# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""On Windows ROCm, mem_get_info's free does not track residency, so the memory guards go blind."""

from __future__ import annotations

import sys
import types

import pytest

from core.inference import diffusion_memory as dm
from utils.hardware import hardware as hw


def _shared_setup_1(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "torch",
        _fake_torch(24 * GiB, free_bytes = 24 * GiB, reserved_bytes = 20 * GiB),
    )


MiB = 1024**2
GiB = 1024**3


def _fake_torch(
    total_bytes,
    *,
    free_bytes,
    reserved_bytes = 0,
    allocated_bytes = None,
):
    """A fake torch whose driver free reading can disagree with allocator accounting."""

    class _Props:
        def __init__(self):
            self.name = "AMD Radeon RX 7900 XTX"
            self.total_memory = total_bytes
            self.integrated = False

    allocated = reserved_bytes if allocated_bytes is None else allocated_bytes

    t = types.ModuleType("torch")
    t.__version__ = "2.11.0+rocm7.13"
    t.version = types.SimpleNamespace(hip = "7.13", cuda = None)
    t.cuda = types.SimpleNamespace(
        is_available = lambda: True,
        device_count = lambda: 1,
        current_device = lambda: 0,
        get_device_properties = lambda i = 0: _Props(),
        mem_get_info = lambda i = None: (free_bytes, total_bytes),
        memory_reserved = lambda i = None: reserved_bytes,
        memory_allocated = lambda i = None: allocated,
    )
    return t


@pytest.fixture
def win_rocm(monkeypatch):
    """Windows ROCm, as the hardware module's own sentinel predicate reads it."""
    monkeypatch.setattr(hw, "IS_ROCM", True)
    monkeypatch.setattr(hw.sys, "platform", "win32")
    return monkeypatch


class _Target:
    """The minimal shape snapshot_device_memory() reads off a diffusion state."""

    device = "cuda"
    backend = "diffusers"


def test_activation_guard_fires_on_a_full_card_that_reports_itself_empty(win_rocm, monkeypatch):
    """On Windows WDDM spills to host RAM rather than raising, so the guard must fire on a full card."""
    _shared_setup_1(monkeypatch)

    memory = dm.snapshot_device_memory(_Target())
    assert memory.free_mib == 4 * 1024

    message = dm.image_activation_shortfall_message(
        device_memory = memory, width = 1536, height = 1536, family = "sdxl"
    )
    assert message is not None
    with pytest.raises(dm.ImageActivationShortfallError):
        dm.raise_on_image_activation_shortfall(
            device_memory = memory, width = 1536, height = 1536, family = "sdxl"
        )


def test_activation_guard_still_silent_at_the_default_resolution(win_rocm, monkeypatch):
    """The correction must not turn the guard into a nuisance: a request at or
    below what the load itself budgeted is exempt by the `needed <= planned` arm,
    which the tighter free reading does not touch."""
    _shared_setup_1(monkeypatch)
    memory = dm.snapshot_device_memory(_Target())
    assert (
        dm.image_activation_shortfall_message(
            device_memory = memory, width = 1024, height = 1024, family = "sdxl"
        )
        is None
    )


def test_linux_rocm_reading_is_untouched(monkeypatch):
    """Control: the same numbers off Windows keep the driver's free reading, so
    Linux ROCm and NVIDIA behaviour is byte-identical to before."""
    monkeypatch.setattr(hw, "IS_ROCM", True)
    monkeypatch.setattr(hw.sys, "platform", "linux")
    monkeypatch.setitem(
        sys.modules,
        "torch",
        _fake_torch(24 * GiB, free_bytes = 9 * GiB, reserved_bytes = 20 * GiB),
    )
    assert dm.snapshot_device_memory(_Target()).free_mib == 9 * 1024


def test_windows_nvidia_reading_is_untouched(monkeypatch):
    """The cap is ROCm-gated: CUDA's free reading is trustworthy on Windows."""
    monkeypatch.setattr(hw, "IS_ROCM", False)
    monkeypatch.setattr(hw.sys, "platform", "win32")
    monkeypatch.setitem(
        sys.modules,
        "torch",
        _fake_torch(24 * GiB, free_bytes = 9 * GiB, reserved_bytes = 20 * GiB),
    )
    assert dm.snapshot_device_memory(_Target()).free_mib == 9 * 1024


def test_memory_plan_budget_sees_the_corrected_free(win_rocm, monkeypatch):
    """Planning budgets from the corrected free value, not the sentinel that reports the card empty."""
    _shared_setup_1(monkeypatch)
    memory = dm.settled_snapshot_device_memory(_Target(), attempts = 1)
    assert memory.free_mib == 4 * 1024
    assert dm._safe_device_budget_mib(memory) < 4 * 1024


def test_reclaimable_snapshot_credits_the_cache_back(win_rocm, monkeypatch):
    """The cap is against RESERVED, and the per-generation snapshot adds torch's
    reclaimable cache back, so a process holding 20 GiB reserved of which 6 GiB is
    cached lands on total - allocated rather than on a pessimistic floor."""
    monkeypatch.setitem(
        sys.modules,
        "torch",
        _fake_torch(
            24 * GiB,
            free_bytes = 24 * GiB,
            reserved_bytes = 20 * GiB,
            allocated_bytes = 14 * GiB,
        ),
    )
    assert dm.reclaimable_snapshot_device_memory(_Target()).free_mib == 10 * 1024


def test_trusted_mem_get_info_caps_free_at_unreserved_bytes(win_rocm, monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "torch",
        _fake_torch(16 * GiB, free_bytes = 16 * GiB, reserved_bytes = 11 * GiB),
    )
    assert hw.trusted_mem_get_info() == (5 * GiB, 16 * GiB)
    monkeypatch.setitem(
        sys.modules,
        "torch",
        _fake_torch(16 * GiB, free_bytes = 15 * GiB, reserved_bytes = 11 * GiB),
    )
    assert hw.trusted_mem_get_info() == (5 * GiB, 16 * GiB)
    monkeypatch.setitem(
        sys.modules,
        "torch",
        _fake_torch(16 * GiB, free_bytes = 2 * GiB, reserved_bytes = 11 * GiB),
    )
    assert hw.trusted_mem_get_info() == (2 * GiB, 16 * GiB)


def test_trusted_mem_get_info_is_a_no_op_without_the_sentinel(monkeypatch):
    monkeypatch.setattr(hw, "IS_ROCM", True)
    monkeypatch.setattr(hw.sys, "platform", "linux")
    monkeypatch.setitem(
        sys.modules,
        "torch",
        _fake_torch(16 * GiB, free_bytes = 15 * GiB, reserved_bytes = 11 * GiB),
    )
    assert hw.trusted_mem_get_info() == (15 * GiB, 16 * GiB)


def test_trusted_mem_get_info_falls_back_when_the_allocator_cannot_answer(win_rocm, monkeypatch):
    """No allocator accounting to cap against leaves the driver figure as the only
    reading there is, rather than a fabricated zero."""
    torch_mod = _fake_torch(16 * GiB, free_bytes = 16 * GiB)

    def _boom(i = None):
        raise RuntimeError("no allocator")

    torch_mod.cuda.memory_reserved = _boom
    monkeypatch.setitem(sys.modules, "torch", torch_mod)
    assert hw.trusted_mem_get_info() == (16 * GiB, 16 * GiB)


def test_trusted_mem_get_info_accepts_an_explicit_device_and_module(win_rocm, monkeypatch):
    """llama.cpp slot fitting probes per ordinal and the video preflight passes the
    resolved device module, so both call shapes have to work."""
    torch_mod = _fake_torch(16 * GiB, free_bytes = 16 * GiB, reserved_bytes = 4 * GiB)
    monkeypatch.setitem(sys.modules, "torch", torch_mod)
    assert hw.trusted_mem_get_info(0) == (12 * GiB, 16 * GiB)
    assert hw.trusted_mem_get_info(0, module = torch_mod.cuda) == (12 * GiB, 16 * GiB)
