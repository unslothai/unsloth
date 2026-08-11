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

# The reporter's cards, and his idle used figures as #7072's screenshot showed
# them (0.22 and 0.14 GiB, which is the 53.0 GiB tile reading 0.36 GiB). One
# counter instance per visible card and no others, which is the only shape the
# aggregate is emitted for.
REPORTER_DEVICES = [("AMD Radeon PRO W7900", 45.0 * GB), ("AMD Radeon PRO W7500", 7.98 * GB)]
IDLE_ADAPTERS = [
    ("luid_0x00000000_0x0000d1e2_phys_0", 0.22 * GB),  # W7900 idle desktop
    ("luid_0x00000000_0x0000e34a_phys_0", 0.14 * GB),  # W7500 idle desktop
]


# ----------------------------------------------------------------------------- #
# The System tab tile: aggregate usage survives an unattributable pairing
# ----------------------------------------------------------------------------- #
def test_system_tab_reports_aggregate_when_pairing_is_ambiguous(win_rocm, monkeypatch):
    """0.22 + 0.14 GiB across a 45/7.98 GiB pair: neither usage is capacity-forced,
    so per device it stays Unknown, but the visible set's total is exactly 0.36 GiB
    either way round. Before the fix the whole tile read Unknown (#7452)."""
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(REPORTER_DEVICES, free_equals_total = True))
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
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(REPORTER_DEVICES, free_equals_total = True))
    monkeypatch.setattr(
        hw.subprocess, "run", _subprocess_run(adapter_output = _adapter_output(IDLE_ADAPTERS))
    )

    result = hw.get_gpu_utilization()
    assert result["vram_used_gb_aggregate"] == pytest.approx(0.36, abs = 0.01)
    # The legacy primary mirror must not be overwritten by the aggregate.
    assert result["vram_total_gb"] == 45.0
    assert result["vram_used_gb"] is None


def test_loaded_card_agrees_with_the_per_device_figures(win_rocm, monkeypatch):
    """#7072's own case: a model resident on the W7900. 40 GiB exceeds the smaller
    card, so the ranking is forced and both rows get a value; the aggregate must then
    equal their sum, or the tile would disagree with the rows underneath it."""
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(REPORTER_DEVICES, free_equals_total = True))
    loaded = [
        ("luid_0x00000000_0x0000d1e2_phys_0", 40.0 * GB),
        ("luid_0x00000000_0x0000e34a_phys_0", 0.5 * GB),
    ]
    monkeypatch.setattr(
        hw.subprocess, "run", _subprocess_run(adapter_output = _adapter_output(loaded))
    )

    result = hw.get_visible_gpu_utilization()
    by_idx = {d["index"]: d for d in result["devices"]}
    assert by_idx[0]["vram_used_gb"] == pytest.approx(40.0, abs = 0.01)
    assert by_idx[1]["vram_used_gb"] == pytest.approx(0.5, abs = 0.01)
    assert result["vram_used_gb_aggregate"] == pytest.approx(40.5, abs = 0.01)
    assert result["vram_used_gb_aggregate"] == pytest.approx(
        sum(d["vram_used_gb"] for d in result["devices"]), abs = 0.01
    )


def test_no_aggregate_when_the_counter_is_unavailable(win_rocm, monkeypatch):
    """A localized or missing counter set must stay Unknown rather than become 0:
    that fabricated zero is the #7072 symptom this pair started from."""
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(REPORTER_DEVICES, free_equals_total = True))
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
    # Two identical cards make the ranking degenerate, and the sum does not care.
    assert agg([10 * GB, 3 * GB], [24 * GB, 24 * GB]) == pytest.approx(13 * GB)
    # Fewer counters than cards: a card has no reading, so the sum is not the total.
    assert agg([5 * GB], [45 * GB, 8 * GB]) is None
    # More counters than visible cards: an adapter in the list is not one of ours.
    assert agg([40 * GB, 7 * GB, 6 * GB], [45 * GB, 8 * GB]) is None
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


def test_aggregate_never_sums_bytes_that_are_not_on_a_visible_card():
    """The reason an unexplained instance is refused rather than filtered out.

    ``Get-Counter`` lists every WDDM adapter and the instance names carry no
    vendor, LUID or PCI key, so a counter cannot be told apart from a foreign
    adapter's. Dropping the small ones to force a 1:1 count keeps the foreign
    reading and drops the quiet visible card whenever the foreign adapter is the
    busier of the two, and the host total then reports bytes that are on no
    visible card. Each case below is one that a noise filter would have summed.
    """
    agg = hw._rocm_windows_aggregate_used_bytes
    pair = [45 * GB, 8 * GB]
    # A third AMD card hidden by HIP_VISIBLE_DEVICES, busy at 5 GiB, while the
    # visible 8 GiB card idles below the 64 MiB noise floor. Truth is 30.03 GiB.
    assert agg([30 * GB, 5 * GB, 30 * MiB], pair) is None
    # An NVIDIA card in the same box. Truth is 30.02 GiB.
    assert agg([30 * GB, 6 * GB, 20 * MiB], pair) is None
    # An integrated display GPU holding a 1 GiB carveout. Truth is 30.01 GiB.
    assert agg([30 * GB, 1 * GB, 10 * MiB], pair) is None
    # A Basic Render Driver / Remote Display placeholder ABOVE the cutoff.
    assert agg([30 * GB, 200 * MiB, 20 * MiB], pair) is None
    # One visible card, idle, beside a busy foreign adapter: the worst case, since
    # a single card's capacity admits almost any foreign reading. Truth is 30 MiB.
    assert agg([6 * GB, 30 * MiB], [45 * GB]) is None
    assert agg([6 * GB, 30 * MiB, 3 * MiB], [45 * GB]) is None


def test_aggregate_is_stable_across_a_changing_instance_list(win_rocm, monkeypatch):
    """Counters come and go between polls (a placeholder adapter appears, a card
    is masked mid-session). Every poll is judged on its own list, so the tile
    alternates between the real figure and Unknown, never between two figures."""
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(REPORTER_DEVICES, free_equals_total = True))
    polls = [
        (IDLE_ADAPTERS, pytest.approx(0.36, abs = 0.01)),
        (IDLE_ADAPTERS + [("luid_0x00000000_0x0000f001_phys_0", 3 * MiB)], None),
        (IDLE_ADAPTERS + [("luid_0x00000000_0x0000f002_phys_0", 2.0 * GB)], None),
        (IDLE_ADAPTERS, pytest.approx(0.36, abs = 0.01)),
    ]
    for adapters, expected in polls:
        monkeypatch.setattr(
            hw.subprocess, "run", _subprocess_run(adapter_output = _adapter_output(adapters))
        )
        result = hw.get_visible_gpu_utilization()
        assert result["vram_used_gb_aggregate"] == expected


def test_aggregate_tolerates_a_wddm_spill_over_the_smaller_card(win_rocm, monkeypatch):
    """WDDM satisfies an overrun from host RAM, so a usage can exceed the card it
    sits on. That only has to not fabricate a number: the reading above the LARGEST
    visible capacity is refused, and one that still fits the ranking is summed as
    reported rather than clamped."""
    agg = hw._rocm_windows_aggregate_used_bytes
    assert agg([9 * GB, 0.3 * GB], [45 * GB, 8 * GB]) == pytest.approx(9.3 * GB)
    assert agg([46 * GB, 2 * GB], [45 * GB, 8 * GB]) is None


# ----------------------------------------------------------------------------- #
# The aggregate and the rows have to describe the same cards
# ----------------------------------------------------------------------------- #


def _merged_gpu_info(monkeypatch, visibility, utilization):
    """Run main's payload merge over two stubbed probes and return the training half.

    Both probes are function-local imports from utils.hardware, so they are patched
    on the package. The module-level cache is cleared first or a neighbouring test's
    payload is returned instead of this one.
    """
    import main
    import utils.hardware as uh

    monkeypatch.setattr(uh, "get_backend_visible_gpu_info", lambda: visibility)
    monkeypatch.setattr(uh, "get_visible_gpu_utilization", lambda: utilization)
    monkeypatch.setattr(main, "_system_gpu_cache", None, raising = False)
    gpu_info, _ = main._get_cached_system_gpu_info(main.logger)
    return gpu_info


def _probe_pair(visible_indices, util_indices, aggregate):
    visibility = {
        "available": True,
        "backend": "rocm",
        "devices": [
            {"index": i, "name": f"card{i}", "memory_total_gb": 45.0 if i == 0 else 7.98}
            for i in visible_indices
        ],
    }
    utilization = {
        "backend": "rocm",
        "devices": [{"index": i, "vram_used_gb": None} for i in util_indices],
        "vram_used_gb_aggregate": aggregate,
    }
    return visibility, utilization


def test_aggregate_is_dropped_when_the_probes_enumerate_different_cards(monkeypatch):
    """The tile divides the aggregate by the SUMMED totals of the rows it shows, so
    a total taken over cards the rows do not list reads above 100 percent and the
    free figure clamps to 0. metrics_match cannot catch it: on the only path that
    sets an aggregate, Windows ROCm, both probes label themselves "rocm", so it is
    unconditionally true and constrains nothing about which cards were counted.

    The two probes really do enumerate independently -- the visibility side calls
    mem_get_info per device and drops one that raises, the aggregate side reads
    torch properties only and keeps it -- so the sets can differ with both probes
    reporting success.
    """
    visibility, utilization = _probe_pair([0], [0, 1], 46.0)
    gpu_info = _merged_gpu_info(monkeypatch, visibility, utilization)
    shown_total = sum(d["memory_total_gb"] for d in gpu_info["devices"])
    assert utilization["vram_used_gb_aggregate"] > shown_total  # the wrong number
    assert gpu_info["vram_used_gb_aggregate"] is None


def test_aggregate_survives_when_both_probes_name_the_same_cards(monkeypatch):
    """The reporter's own host, and the case the fix exists for: identical index
    sets, so the aggregate is a total over exactly the rows on screen."""
    visibility, utilization = _probe_pair([0, 1], [0, 1], 0.36)
    gpu_info = _merged_gpu_info(monkeypatch, visibility, utilization)
    assert gpu_info["vram_used_gb_aggregate"] == 0.36


def test_aggregate_is_dropped_when_the_sets_merely_overlap(monkeypatch):
    """Equal counts are not enough. Two cards each, but index 1 against index 2:
    the aggregate counts a card that has no row, and a row has no counter."""
    visibility, utilization = _probe_pair([0, 1], [0, 2], 46.0)
    gpu_info = _merged_gpu_info(monkeypatch, visibility, utilization)
    assert gpu_info["vram_used_gb_aggregate"] is None
