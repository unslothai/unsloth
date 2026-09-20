# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
"""A full-GPU pin that Windows served from system RAM must not pass silently.

Every rung of the spawn loop's recovery ladder is gated on the child exiting non-zero.
On Windows an over-subscribed CUDA allocation does not fail: since driver 536.40 the WDDM
memory manager serves it from host RAM over PCIe, so llama-server starts, answers /health,
and decodes 5-10x slower forever. The ladder is therefore dead on the one platform that
needs it (#11349, #11336).

Measured on an RTX PRO 6000 for the arithmetic these tests encode: Qwen3-VL-8B-Instruct
needs a 9.00 GiB f16 KV cache at -c 65536 on top of 4.80 GiB of weights, and on Linux with
12 GiB free that allocation fails outright ("cudaMalloc failed: out of memory") rather than
spilling. These tests pin the Windows half: the check fires on a real shortfall, abstains
on every ambiguous reading, and stays a no-op off Windows.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_TESTS_DIR = Path(__file__).resolve().parent
_BACKEND_DIR = str(_TESTS_DIR.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from core.inference import llama_cpp as mod
from core.inference.llama_cpp import LlamaCppBackend

GIB = 1024**3
MIB = 1024 * 1024

# The reported case: weights + f16 KV at -c 65536, the two terms that must be resident.
QWEN3_VL_8B_FLOOR = int(4.80 * GIB) + 9 * GIB

# What the discrete full-GPU pin emits. The check reads the argv, not the plan, so the
# retry rungs that rewrite it disarm on their own.
PIN_ARGV = ["llama-server", "-m", "x.gguf", "-ngl", "-1", "--fit", "off"]


class _Backend:
    """Just the residency surface of LlamaCppBackend, over a stubbed VRAM probe.

    ``rows`` is what the probe answers AFTER the child is up; ``baseline_rows`` what it
    answered just before the spawn. Two readings because the real check samples its
    baseline immediately before Popen, not at plan time.
    """

    def __init__(
        self,
        rows,
        baseline_rows = ((0, 16384.0, 16384),),
        shared = None,
        baseline_shared = None,
    ):
        self._pin_resident_floor_bytes = None
        self._pin_baseline_free_mib = None
        self._pin_baseline_shared_usage = None
        self._pin_gpu_indices = None
        self._warnings = []
        self._rows = rows
        self._baseline_rows = baseline_rows
        self._sampled = False
        # Counter readings before and after the spawn. Default None on both, i.e. the
        # counter is unavailable, which is the device-delta fallback most tests exercise.
        self._baseline_shared = baseline_shared
        self._shared = shared
        self._shared_reads = 0

    _arm_residency_check = LlamaCppBackend.__dict__["_arm_residency_check"]
    _argv_claims_full_offload = LlamaCppBackend.__dict__["_argv_claims_full_offload"]
    _shared_usage_growth_bytes = LlamaCppBackend.__dict__["_shared_usage_growth_bytes"]
    _sample_residency_baseline = LlamaCppBackend.__dict__["_sample_residency_baseline"]
    _verify_vram_residency = LlamaCppBackend.__dict__["_verify_vram_residency"]
    _start_residency_check = LlamaCppBackend.__dict__["_start_residency_check"]

    def _shared_gpu_memory_bytes(self):
        self._shared_reads += 1
        return self._baseline_shared if self._shared_reads == 1 else self._shared

    def _get_gpu_memory(self, *a, **k):
        if not self._sampled:
            self._sampled = True
            return self._baseline_rows
        return self._rows

    def _record_load_warning(self, message):
        self._warnings.append(message)

    @staticmethod
    def _nvml_library():
        return object()  # NVIDIA present

    @staticmethod
    def _integrated_cuda_gpu_ids():
        return set()  # discrete card


@pytest.fixture
def on_windows(monkeypatch):
    monkeypatch.setattr(mod.os, "name", "nt", raising = False)


def _arm(backend, floor = QWEN3_VL_8B_FLOOR):
    """Arm and take the pre-spawn baseline, as the launch path does."""
    backend._arm_residency_check(floor, [0])
    backend._sample_residency_baseline(PIN_ARGV)


def test_spill_is_reported(on_windows):
    """3 GiB short of the floor, on a card whose free memory barely moved."""
    # 16384 free before; 16384 - 11469 = 4915 MiB taken, against a ~14.1 GiB floor.
    backend = _Backend([(0, 16384.0 - 11469.0, 16384)])
    _arm(backend)
    message = backend._verify_vram_residency()
    assert message is not None, "a 9 GiB shortfall was not reported"
    assert "shared system memory" in message
    assert "Prefer No Sysmem Fallback" in message
    assert backend._warnings == [message], "the advisory did not reach memory_warning"


def test_healthy_load_is_silent(on_windows):
    """Everything resident: free memory drops by the whole floor."""
    used_mib = QWEN3_VL_8B_FLOOR / MIB
    backend = _Backend([(0, 16384.0 - used_mib, 16384)])
    _arm(backend)
    assert backend._verify_vram_residency() is None
    assert backend._warnings == []


def test_small_shortfall_is_tolerated(on_windows):
    """A 5% gap is allocator rounding and a stale baseline, not a spill.

    The floor deliberately under-counts (no mmproj, no CUDA context reserve), so a
    healthy load routinely reads a little under it. Flagging that would train users to
    ignore the warning.
    """
    used_mib = (QWEN3_VL_8B_FLOOR * 0.95) / MIB
    backend = _Backend([(0, 16384.0 - used_mib, 16384)])
    _arm(backend)
    assert backend._verify_vram_residency() is None


def test_sub_gib_shortfall_is_tolerated(on_windows):
    """Both thresholds must trip. A small model 900 MiB short stays quiet."""
    floor = 3 * GIB
    used_mib = (floor - 900 * MIB) / MIB  # 29% short, but under the 1 GiB floor
    backend = _Backend([(0, 8192.0 - used_mib, 8192)], baseline_rows = ((0, 8192.0, 8192),))
    backend._arm_residency_check(floor, [0])
    backend._sample_residency_baseline(PIN_ARGV)
    assert backend._verify_vram_residency() is None


def test_baseline_is_taken_after_a_replaced_model_is_torn_down(on_windows):
    """The reason the baseline is not the plan-time reading.

    Loading over a resident model plans while the old child still holds ~8 GiB and
    spawns once the teardown released it. Sampling before Popen sees the freed card, so
    the new child's own allocation is measured; sampling at plan time would book the
    teardown as memory this load failed to take and cry spill on a healthy load.
    """
    used_mib = QWEN3_VL_8B_FLOOR / MIB
    backend = _Backend(
        [(0, 16384.0 - used_mib, 16384)],
        baseline_rows = ((0, 16384.0, 16384),),  # post-teardown: card is free
    )
    # Plan-time reading, 8 GiB still held by the outgoing model. Passed to arm, and
    # deliberately ignored by it.
    backend._arm_residency_check(QWEN3_VL_8B_FLOOR, [0])
    backend._sample_residency_baseline(PIN_ARGV)
    assert backend._pin_baseline_free_mib == {0: 16384.0}, "arm reused the stale reading"
    assert backend._verify_vram_residency() is None, "teardown was booked as a shortfall"


# ── The direct signal: '\GPU Adapter Memory(*)\Shared Usage' ────────────────────
# nvidia-smi on WDDM reports DEDICATED VRAM only, so the spilled bytes appear in neither
# memory.used nor memory.free. This counter is the one place Windows states them, and it
# is what Task Manager's "shared GPU memory" shows.

NVIDIA_LUID = "luid_0x00000000_0x0000c350_phys_0"
IGPU_LUID = "luid_0x00000000_0x00001234_phys_0"


def _shared(nvidia_mib, igpu_mib = 64):
    return {
        NVIDIA_LUID: int(nvidia_mib * MIB),
        IGPU_LUID: int(igpu_mib * MIB),
    }


def test_the_74_mib_case_the_inferred_floor_could_never_see(on_windows):
    """The measurement that motivated the whole check.

    Qwen3-VL-8B at -c 65536 on a 16 GiB card overshoots by about 74 MiB. That is three
    orders of magnitude under the 1 GiB inferred floor, so the device-delta path is
    structurally blind to it. The direct counter is not.
    """
    used_mib = (QWEN3_VL_8B_FLOOR / MIB) - 74  # 74 MiB short of the floor
    backend = _Backend(
        [(0, 16384.0 - used_mib, 16384)],
        baseline_shared = _shared(120),
        shared = _shared(120 + 74),  # the spill, as Windows reports it
    )
    _arm(backend)
    message = backend._verify_vram_residency()
    assert message is not None, "the direct counter missed a 74 MiB spill"
    assert backend._warnings == [message]


def test_the_inferred_path_alone_is_blind_to_it(on_windows):
    """Same load, counter unavailable. Documents the sensitivity that is given up."""
    used_mib = (QWEN3_VL_8B_FLOOR / MIB) - 74
    backend = _Backend([(0, 16384.0 - used_mib, 16384)])  # no counter readings
    _arm(backend)
    assert backend._verify_vram_residency() is None


def test_shared_growth_below_the_counter_threshold_is_tolerated(on_windows):
    """Under 32 MiB is desktop compositing jitter over the load window, not a spill."""
    used_mib = (QWEN3_VL_8B_FLOOR / MIB) - 20
    backend = _Backend(
        [(0, 16384.0 - used_mib, 16384)],
        baseline_shared = _shared(120),
        shared = _shared(140),  # +20 MiB
    )
    _arm(backend)
    assert backend._verify_vram_residency() is None


def test_another_app_growing_shared_memory_is_not_our_spill(on_windows):
    """The counter is per-adapter. Conjunction with the device side is what saves us.

    Shared usage jumps 500 MiB, but the child got everything its floor demanded, so the
    growth belongs to some other application. Reporting here would blame this load for a
    browser opening a tab.
    """
    used_mib = QWEN3_VL_8B_FLOOR / MIB  # fully resident
    backend = _Backend(
        [(0, 16384.0 - used_mib, 16384)],
        baseline_shared = _shared(120),
        shared = _shared(620),
    )
    _arm(backend)
    assert backend._verify_vram_residency() is None
    assert backend._warnings == []


def test_growth_on_two_adapters_is_ambiguous_and_falls_back(on_windows):
    """Attributing the spill to a card is the point; a guess would name the wrong one.

    Both adapters grew past the threshold, so the direct signal abstains. The device
    delta still decides, and here it is a real multi-GiB shortfall, so this still reports
    -- via the fallback, not the counter.
    """
    used_mib = (QWEN3_VL_8B_FLOOR / MIB) - 4096  # 4 GiB short
    backend = _Backend(
        [(0, 16384.0 - used_mib, 16384)],
        baseline_shared = _shared(120, igpu_mib = 64),
        shared = _shared(2120, igpu_mib = 1064),  # +2000 and +1000 MiB
    )
    _arm(backend)
    assert backend._verify_vram_residency() is not None


def test_counter_failure_degrades_to_the_device_delta(on_windows):
    """A missing, localised or slow Get-Counter must not cost the fallback verdict."""
    used_mib = (QWEN3_VL_8B_FLOOR / MIB) - 4096
    backend = _Backend([(0, 16384.0 - used_mib, 16384)])

    def _boom():
        raise RuntimeError("Get-Counter: counter name not found")

    backend._shared_gpu_memory_bytes = _boom
    backend._arm_residency_check(QWEN3_VL_8B_FLOOR, [0])
    # A raising counter must not escape the baseline thread, and must not cost the
    # inferred verdict. Both halves swallow it, so the same stub can stay in place.
    backend._sample_residency_baseline(PIN_ARGV)
    assert backend._pin_baseline_shared_usage is None
    assert backend._verify_vram_residency() is not None


def test_the_fit_on_retry_disarms_the_check(on_windows):
    """The retry rungs rewrite the argv, and the check must follow them.

    The `--fit off` -> `--fit on` rung says in its own comment that it "gives up the
    confirmed full offload": llama.cpp's fitter may then place whole layers in host RAM.
    That is a shortfall by design. Measuring the retry against the original pin's floor
    would report a spill on every single one of them.
    """
    used_mib = 4096.0  # fitter left most of it on CPU
    backend = _Backend([(0, 16384.0 - used_mib, 16384)])
    backend._arm_residency_check(QWEN3_VL_8B_FLOOR, [0])
    retry_argv = ["llama-server", "-m", "x.gguf", "-ngl", "-1", "--fit", "on"]
    backend._sample_residency_baseline(retry_argv)
    assert backend._pin_resident_floor_bytes is None, "the --fit on retry stayed armed"
    assert backend._verify_vram_residency() is None
    assert backend._warnings == []


def test_a_manual_layer_count_disarms_the_check(on_windows):
    """An explicit -ngl is a manual placement that owes no full-offload promise."""
    backend = _Backend([(0, 16384.0 - 2048.0, 16384)])
    backend._arm_residency_check(QWEN3_VL_8B_FLOOR, [0])
    backend._sample_residency_baseline(
        ["llama-server", "-m", "x.gguf", "--gpu-layers", "20", "--fit", "off"]
    )
    assert backend._pin_resident_floor_bytes is None


def test_a_user_extra_overriding_the_pin_disarms_the_check(on_windows):
    """Last-wins, as llama.cpp reads it: a trailing user --fit on takes the placement."""
    backend = _Backend([(0, 16384.0 - 2048.0, 16384)])
    backend._arm_residency_check(QWEN3_VL_8B_FLOOR, [0])
    backend._sample_residency_baseline(PIN_ARGV + ["--fit", "on"])
    assert backend._pin_resident_floor_bytes is None


def test_the_spill_plan_disarms_the_check(on_windows):
    """-ot deliberately leaves named tensors on the host."""
    backend = _Backend([(0, 16384.0 - 4096.0, 16384)])
    backend._arm_residency_check(QWEN3_VL_8B_FLOOR, [0])
    backend._sample_residency_baseline(
        ["llama-server", "-m", "x.gguf", "-ot", "blk\\.[0-9]+\\.ffn=CPU", "--fit", "on"]
    )
    assert backend._pin_resident_floor_bytes is None


def test_abstains_when_free_memory_did_not_drop(on_windows):
    """A concurrent release can leave more free than before. That is not evidence."""
    backend = _Backend([(0, 16384.0 + 512.0, 16384)])
    _arm(backend)
    assert backend._verify_vram_residency() is None
    assert backend._warnings == []


def test_abstains_when_a_pinned_card_vanishes(on_windows):
    """A missing row is a broken probe, not a spill."""
    backend = _Backend([(1, 4096.0, 16384)])
    _arm(backend)
    assert backend._verify_vram_residency() is None


def test_probe_failure_never_touches_a_running_server(on_windows):
    backend = _Backend([])

    def _boom(*a, **k):
        raise RuntimeError("nvidia-smi wedged")

    backend._get_gpu_memory = _boom
    _arm(backend)
    assert backend._verify_vram_residency() is None
    assert backend._warnings == []


def test_not_armed_off_windows(monkeypatch):
    """Linux fails the allocation outright, so the existing ladder owns it."""
    monkeypatch.setattr(mod.os, "name", "posix", raising = False)
    backend = _Backend([(0, 16384.0 - 100.0, 16384)])
    _arm(backend)
    assert backend._pin_resident_floor_bytes is None
    assert backend._verify_vram_residency() is None


def test_not_armed_on_integrated_gpu(on_windows, monkeypatch):
    """A shared pool has no device/host split for this to measure."""
    backend = _Backend([(0, 16384.0 - 100.0, 16384)])
    monkeypatch.setattr(backend, "_integrated_cuda_gpu_ids", lambda: {0}, raising = False)
    _arm(backend)
    assert backend._pin_resident_floor_bytes is None


def test_not_armed_without_nvidia(on_windows, monkeypatch):
    """The advice names an NVIDIA Control Panel setting; AMD is a different mechanism."""
    backend = _Backend([(0, 16384.0 - 100.0, 16384)])
    monkeypatch.setattr(backend, "_nvml_library", lambda: None, raising = False)
    _arm(backend)
    assert backend._pin_resident_floor_bytes is None


def test_disarms_without_a_baseline_for_every_pinned_card(on_windows):
    """A partial sum under-counts and would read as a spill."""
    backend = _Backend([(0, 100.0, 16384)], baseline_rows = ((0, 16384.0, 16384),))
    backend._arm_residency_check(QWEN3_VL_8B_FLOOR, [0, 1])
    backend._sample_residency_baseline(PIN_ARGV)  # card 1 never reported
    assert backend._pin_resident_floor_bytes is None
    assert backend._verify_vram_residency() is None


def test_disarms_when_the_baseline_probe_fails(on_windows):
    """No baseline, no verdict. A check that guesses is worse than no check."""
    backend = _Backend([(0, 100.0, 16384)])

    def _boom(*a, **k):
        raise RuntimeError("nvidia-smi wedged")

    backend._arm_residency_check(QWEN3_VL_8B_FLOOR, [0])
    backend._get_gpu_memory = _boom
    backend._sample_residency_baseline(PIN_ARGV)
    assert backend._pin_resident_floor_bytes is None
    assert backend._verify_vram_residency() is None
    assert backend._warnings == []


def test_state_is_disarmed_after_one_read(on_windows):
    """One load's numbers must never be charged against the next."""
    backend = _Backend([(0, 16384.0 - 11469.0, 16384)])
    _arm(backend)
    assert backend._verify_vram_residency() is not None
    assert backend._pin_resident_floor_bytes is None
    assert backend._pin_baseline_free_mib is None
    assert backend._verify_vram_residency() is None, "a second read re-reported the load"


def test_start_spawns_no_thread_when_unarmed(on_windows, monkeypatch):
    """No thread and no probe on the overwhelming majority of loads.

    Asserted by counting Thread constructions, not by observing an absence: an earlier
    version of this test checked a list nothing ever appended to and passed vacuously.
    """
    started = []
    monkeypatch.setattr(
        mod.threading,
        "Thread",
        lambda *a, **k: started.append(k.get("name")) or _NeverStarts(),
    )

    def _probe_threads():
        return [n for n in started if n == "vram-residency-check"]

    backend = _Backend([(0, 1.0, 16384)])  # never armed
    backend._start_residency_check()
    assert _probe_threads() == [], "an unarmed load still spawned the probe thread"

    _arm(backend)
    backend._start_residency_check()
    assert _probe_threads() == ["vram-residency-check"], "an armed load did not spawn the probe"


class _NeverStarts:
    """Stands in for threading.Thread: started and joined, but never runs."""

    def start(self):
        pass

    def join(self, timeout = None):
        pass
