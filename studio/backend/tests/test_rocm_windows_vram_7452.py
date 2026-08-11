# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Regression tests for issue #7452 -- "Stops Reading VRAM Total / Usage on RDNA3".

Reporter: the same host as #7072 (AMD Radeon PRO W7900 45 GiB + W7500 7.98 GiB,
Windows 10, ROCm 7.13, torch 2.11.0+rocm7.13), six days after the #7072 fix
(#7238) shipped. The two issues are a regression pair, not duplicates.

#7238 stopped fabricating a per-device usage when the LUID performance counters
cannot be attributed to torch ordinals: Windows shares no key between the two, so
usage is paired by capacity ranking and kept only when capacity FORCES it. On
this reporter's hardware the smaller card is 7.98 GiB, so every usage at or below
7.98 GiB -- idle, and every small model -- is swappable between the two cards and
reports Unknown. That is honest per device, but the System tab's aggregate VRAM
tile also went Unknown, and the aggregate does NOT depend on the pairing: the sum
of a permutation is the same whichever way round it is. His screenshot shows the
per-device totals intact (45.0 / 7.98 GiB) with used, free and percent all
Unknown, which is exactly this.

Mocks: torch, the performance counter and the platform are all faked, because
there is no AMD GPU and no Windows or ROCm CI in this repository. The fakes are
imported from test_rocm_windows_vram_7072 so both halves of the regression pair
are driven by one fixture shape.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from utils.hardware import hardware as hw

_TESTS_DIR = str(Path(__file__).resolve().parent)
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

from test_rocm_windows_vram_7072 import (  # noqa: E402, F401  (win_rocm is a fixture)
    GB,
    MiB,
    _adapter_output,
    _fake_torch,
    _subprocess_run,
    win_rocm,
)

# The reporter's exact figures: a W7900 and a W7500 sitting idle, plus the
# Windows Basic Render Driver placeholder counter that every such host carries.
REPORTER_DEVICES = [("AMD Radeon PRO W7900", 45.0 * GB), ("AMD Radeon PRO W7500", 7.98 * GB)]
IDLE_ADAPTERS = [
    ("luid_0x00000000_0x0000d1e2_phys_0", 0.22 * GB),  # W7900 idle desktop
    ("luid_0x00000000_0x0000e34a_phys_0", 0.14 * GB),  # W7500 idle desktop
    ("luid_0x00000000_0x0000f001_phys_0", 3 * MiB),  # Basic Render Driver
]


# ----------------------------------------------------------------------------- #
# The System tab tile: aggregate usage survives an unattributable pairing
# ----------------------------------------------------------------------------- #
def test_system_tab_reports_aggregate_when_pairing_is_ambiguous(win_rocm, monkeypatch):
    """0.22 + 0.14 GiB across a 45/7.98 GiB pair: neither usage is capacity-forced,
    so per device it stays Unknown, but the visible set's total is exactly 0.36 GiB
    either way round. Before the fix the whole tile read Unknown (#7452)."""
    monkeypatch.setitem(
        sys.modules, "torch", _fake_torch(REPORTER_DEVICES, free_equals_total = True)
    )
    monkeypatch.setattr(
        hw.subprocess, "run", _subprocess_run(adapter_output = _adapter_output(IDLE_ADAPTERS))
    )

    result = hw.get_visible_gpu_utilization()
    devices = result["devices"]
    assert len(devices) == 2
    assert sorted(d["vram_total_gb"] for d in devices) == [7.98, 45.0]
    # Per device the ranking cannot tell the two apart, so #7238's invariant holds.
    assert all(d["vram_used_gb"] is None for d in devices)
    # But the aggregate is pairing-independent, so the tile has a real number.
    assert result["vram_used_gb_aggregate"] == pytest.approx(0.36, abs = 0.01)


def test_gpu_utilization_payload_carries_the_aggregate(win_rocm, monkeypatch):
    """The floating monitor reads get_gpu_utilization(), so the same figure has to
    reach that payload and not only the System tab's."""
    monkeypatch.setitem(
        sys.modules, "torch", _fake_torch(REPORTER_DEVICES, free_equals_total = True)
    )
    monkeypatch.setattr(
        hw.subprocess, "run", _subprocess_run(adapter_output = _adapter_output(IDLE_ADAPTERS))
    )

    result = hw.get_gpu_utilization()
    assert result["vram_used_gb_aggregate"] == pytest.approx(0.36, abs = 0.01)
    # The legacy primary mirror must not be overwritten by the aggregate.
    assert result["vram_total_gb"] == 45.0
    assert result["vram_used_gb"] is None


def test_loaded_card_keeps_both_the_forced_device_value_and_the_aggregate(win_rocm, monkeypatch):
    """#7072's own case: a model resident on the W7900. The 40 GiB is capacity-forced
    onto the 45 GiB card and stays per device, and the aggregate adds the idle card's
    0.5 GiB, so the tile shows the real total rather than only the attributed part."""
    monkeypatch.setitem(
        sys.modules, "torch", _fake_torch(REPORTER_DEVICES, free_equals_total = True)
    )
    loaded = [
        ("luid_0x00000000_0x0000d1e2_phys_0", 40.0 * GB),
        ("luid_0x00000000_0x0000e34a_phys_0", 0.5 * GB),
        ("luid_0x00000000_0x0000f001_phys_0", 3 * MiB),
    ]
    monkeypatch.setattr(
        hw.subprocess, "run", _subprocess_run(adapter_output = _adapter_output(loaded))
    )

    result = hw.get_visible_gpu_utilization()
    by_idx = {d["index"]: d for d in result["devices"]}
    assert by_idx[0]["vram_used_gb"] == pytest.approx(40.0, abs = 0.01)
    assert by_idx[1]["vram_used_gb"] is None
    assert result["vram_used_gb_aggregate"] == pytest.approx(40.5, abs = 0.01)


def test_no_aggregate_when_the_counter_is_unavailable(win_rocm, monkeypatch):
    """A localized or missing counter set must stay Unknown rather than become 0:
    that fabricated zero is the #7072 symptom this pair started from."""
    monkeypatch.setitem(
        sys.modules, "torch", _fake_torch(REPORTER_DEVICES, free_equals_total = True)
    )
    monkeypatch.setattr(hw.subprocess, "run", _subprocess_run(adapter_output = "__NONE__\n"))

    result = hw.get_visible_gpu_utilization()
    assert len(result["devices"]) == 2
    assert result["vram_used_gb_aggregate"] is None


# ----------------------------------------------------------------------------- #
# The aggregate rule itself (pure unit)
# ----------------------------------------------------------------------------- #
def test_aggregate_requires_a_counter_per_visible_device():
    agg = hw._rocm_windows_aggregate_used_bytes
    # One counter per visible card: the sum is the visible set's, whatever the pairing.
    assert agg([0.22 * GB, 0.14 * GB], [45 * GB, 8 * GB]) == pytest.approx(0.36 * GB)
    # Sub-threshold placeholders drop, leaving exactly one counter per card.
    assert agg([0.22 * GB, 0.14 * GB, 3 * MiB], [45 * GB, 8 * GB]) == pytest.approx(0.36 * GB)
    # Fewer counters than cards: a card has no reading, so the sum is not the total.
    assert agg([5 * GB], [45 * GB, 8 * GB]) is None
    # More active counters than visible cards: an adapter outside the visibility
    # mask is in the list and its usage is not ours to add.
    assert agg([40 * GB, 7 * GB, 6 * GB], [45 * GB, 8 * GB]) is None
    # Every counter below the noise floor: which is the placeholder is unknowable.
    assert agg([50 * MiB, 10 * MiB, 5 * MiB], [45 * GB, 8 * GB]) is None
    assert agg([], [45 * GB, 8 * GB]) is None
    assert agg([1 * GB], []) is None


def test_aggregate_rejects_a_usage_larger_than_any_visible_card():
    """A masked 45 GiB card at 40 GiB beside a visible idle 8 GiB card: the counts
    can match by accident, but 40 GiB cannot be on an 8 GiB card, so the counter set
    is not the visible set and the sum would be a hidden GPU's."""
    agg = hw._rocm_windows_aggregate_used_bytes
    assert agg([40 * GB, 10 * MiB], [8 * GB]) is None
    assert agg([40 * GB, 6 * GB], [45 * GB, 4 * GB]) is None
    # Counter order must not matter.
    assert agg([6 * GB, 40 * GB], [45 * GB, 4 * GB]) is None
