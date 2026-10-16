# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Windows ROCm aggregate VRAM must survive capacity pairing, since a permuted sum is unchanged."""

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

REPORTER_DEVICES = [("AMD Radeon PRO W7900", 45.0 * GB), ("AMD Radeon PRO W7500", 7.98 * GB)]
IDLE_ADAPTERS = [
    ("luid_0x00000000_0x0000d1e2_phys_0", 0.22 * GB),
    ("luid_0x00000000_0x0000e34a_phys_0", 0.14 * GB),
]


def test_system_tab_reports_aggregate_when_pairing_is_ambiguous(win_rocm, monkeypatch):
    """0.22 + 0.14 GiB across a 45/7.98 GiB pair: neither usage is capacity-forced, so
    per device stays Unknown, but the total is 0.36 GiB either way round. Before the
    fix the whole tile read Unknown (#7452)."""
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(REPORTER_DEVICES, free_equals_total = True))
    monkeypatch.setattr(
        hw.subprocess, "run", _subprocess_run(adapter_output = _adapter_output(IDLE_ADAPTERS))
    )

    result = hw.get_visible_gpu_utilization()
    devices = result["devices"]
    assert len(devices) == 2
    assert sorted(d["vram_total_gb"] for d in devices) == [7.98, 45.0]
    assert all(d["vram_used_gb"] is None for d in devices)
    assert result["vram_used_gb_aggregate"] == pytest.approx(0.36, abs = 0.01)


def test_gpu_utilization_payload_carries_the_aggregate(win_rocm, monkeypatch):
    """The floating monitor reads get_gpu_utilization(), so the figure has to reach
    that payload too, not only the System tab's."""
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(REPORTER_DEVICES, free_equals_total = True))
    monkeypatch.setattr(
        hw.subprocess, "run", _subprocess_run(adapter_output = _adapter_output(IDLE_ADAPTERS))
    )

    result = hw.get_gpu_utilization()
    assert result["vram_used_gb_aggregate"] == pytest.approx(0.36, abs = 0.01)
    assert result["vram_total_gb"] == 45.0
    assert result["vram_used_gb"] is None


def test_loaded_card_agrees_with_the_per_device_figures(win_rocm, monkeypatch):
    """#7072's own case: a model resident on the W7900. 40 GiB exceeds the smaller
    card, so the ranking is forced and both rows get a value; the aggregate must equal
    their sum or the tile disagrees with the rows underneath it."""
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
    """A localized or missing counter set stays Unknown rather than becoming 0: that
    fabricated zero is the #7072 symptom this pair started from."""
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(REPORTER_DEVICES, free_equals_total = True))
    monkeypatch.setattr(hw.subprocess, "run", _subprocess_run(adapter_output = "__NONE__\n"))

    result = hw.get_visible_gpu_utilization()
    assert len(result["devices"]) == 2
    assert result["vram_used_gb_aggregate"] is None


def test_aggregate_requires_a_counter_per_visible_device():
    agg = hw._rocm_windows_aggregate_used_bytes
    assert agg([0.22 * GB, 0.14 * GB], [45 * GB, 8 * GB]) == pytest.approx(0.36 * GB)
    assert agg([10 * GB, 3 * GB], [24 * GB, 24 * GB]) == pytest.approx(13 * GB)
    assert agg([5 * GB], [45 * GB, 8 * GB]) is None
    assert agg([40 * GB, 7 * GB, 6 * GB], [45 * GB, 8 * GB]) is None
    assert agg([], [45 * GB, 8 * GB]) is None
    assert agg([1 * GB], []) is None


def test_aggregate_rejects_a_usage_larger_than_any_visible_card():
    """A masked 45 GiB card at 40 GiB beside a visible idle 8 GiB card: the counts can
    match by accident, but 40 GiB cannot be on an 8 GiB card, so the counter set is
    not the visible set and the sum would be a hidden GPU's."""
    agg = hw._rocm_windows_aggregate_used_bytes
    assert agg([40 * GB, 10 * MiB], [8 * GB]) is None
    assert agg([40 * GB, 6 * GB], [45 * GB, 4 * GB]) is None
    assert agg([6 * GB, 40 * GB], [45 * GB, 4 * GB]) is None


def test_aggregate_never_sums_bytes_that_are_not_on_a_visible_card():
    """Counters cannot be tied to a visible card, so an unexplained instance is refused, not filtered."""
    agg = hw._rocm_windows_aggregate_used_bytes
    pair = [45 * GB, 8 * GB]
    assert agg([30 * GB, 5 * GB, 30 * MiB], pair) is None
    assert agg([30 * GB, 6 * GB, 20 * MiB], pair) is None
    assert agg([30 * GB, 1 * GB, 10 * MiB], pair) is None
    assert agg([30 * GB, 200 * MiB, 20 * MiB], pair) is None
    assert agg([6 * GB, 30 * MiB], [45 * GB]) is None
    assert agg([6 * GB, 30 * MiB, 3 * MiB], [45 * GB]) is None


def test_aggregate_is_stable_across_a_changing_instance_list(win_rocm, monkeypatch):
    """Counters come and go between polls (a placeholder adapter appears, a card is
    masked mid-session). Every poll is judged on its own list, so the tile alternates
    between the real figure and Unknown, never between two figures."""
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
    """WDDM satisfies an overrun from host RAM, so a usage can exceed the card it sits
    on. That only has to not fabricate a number: a reading above the LARGEST visible
    capacity is refused, one that still fits the ranking is summed as reported."""
    agg = hw._rocm_windows_aggregate_used_bytes
    assert agg([9 * GB, 0.3 * GB], [45 * GB, 8 * GB]) == pytest.approx(9.3 * GB)
    assert agg([46 * GB, 2 * GB], [45 * GB, 8 * GB]) is None


def _merged_gpu_info(monkeypatch, visibility, utilization):
    """Main's payload merge over stubbed probes, with the cache cleared so no neighbour's payload
    returns."""
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
    """The tile divides by the summed totals of shown rows, so the probes must enumerate the same cards."""
    visibility, utilization = _probe_pair([0], [0, 1], 46.0)
    gpu_info = _merged_gpu_info(monkeypatch, visibility, utilization)
    shown_total = sum(d["memory_total_gb"] for d in gpu_info["devices"])
    assert utilization["vram_used_gb_aggregate"] > shown_total
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
