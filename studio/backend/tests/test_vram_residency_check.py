# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
"""A full-GPU pin that Windows served from system RAM must not pass silently."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_TESTS_DIR = Path(__file__).resolve().parent
_BACKEND_DIR = str(_TESTS_DIR.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))

from core.inference import llama_cpp as mod
from core.inference.llama_cpp import LlamaCppBackend

GIB = 1024**3
MIB = 1024 * 1024

QWEN3_VL_8B_FLOOR = int(4.80 * GIB) + 9 * GIB

PIN_ARGV = ["llama-server", "-m", "x.gguf", "-ngl", "-1", "--fit", "off"]


class _Backend:
    """Just the residency surface of LlamaCppBackend, over a stubbed VRAM probe."""

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
        self._baseline_shared = baseline_shared
        self._shared = shared
        self._shared_reads = 0

    _arm_residency_check = LlamaCppBackend.__dict__["_arm_residency_check"]
    _argv_claims_full_offload = LlamaCppBackend.__dict__["_argv_claims_full_offload"]
    _shared_usage_growth_bytes = LlamaCppBackend.__dict__["_shared_usage_growth_bytes"]
    _sample_residency_baseline = LlamaCppBackend.__dict__["_sample_residency_baseline"]
    _verify_vram_residency = LlamaCppBackend.__dict__["_verify_vram_residency"]
    _kv_layout_of = LlamaCppBackend.__dict__["_kv_layout_of"]

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
    def _sysmem_fallback_risk(binary = None):
        return True  # CUDA build

    def _wait_for_vram_settle(
        self,
        since_kill = 0.0,
        **_kw,
    ):
        self._settle_calls = getattr(self, "_settle_calls", []) + [since_kill]

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
    """A 5% gap is allocator rounding and a stale baseline, not a spill."""
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
    """The reason the baseline is not the plan-time reading."""
    used_mib = QWEN3_VL_8B_FLOOR / MIB
    backend = _Backend(
        [(0, 16384.0 - used_mib, 16384)],
        baseline_rows = ((0, 16384.0, 16384),),  # post-teardown: card is free
    )
    backend._arm_residency_check(QWEN3_VL_8B_FLOOR, [0])
    backend._sample_residency_baseline(PIN_ARGV)
    assert backend._pin_baseline_free_mib == {0: 16384.0}, "arm reused the stale reading"
    assert backend._verify_vram_residency() is None, "teardown was booked as a shortfall"


NVIDIA_LUID = "luid_0x00000000_0x0000c350_phys_0"
IGPU_LUID = "luid_0x00000000_0x00001234_phys_0"


def _shared(
    nvidia_mib,
    igpu_mib = 64,
    nvidia_dedicated_mib = 0,
):
    return {
        NVIDIA_LUID: int(nvidia_mib * MIB),
        IGPU_LUID: int(igpu_mib * MIB),
        "dedicated|" + NVIDIA_LUID: int(nvidia_dedicated_mib * MIB),
        "dedicated|" + IGPU_LUID: 128 * MIB,
    }


def test_the_74_mib_case_the_inferred_floor_could_never_see(on_windows):
    """The measurement that motivated the whole check."""
    used_mib = (QWEN3_VL_8B_FLOOR / MIB) + 600  # context + compute buffers mask the gap
    backend = _Backend(
        [(0, 300.0, 16384)],
        baseline_rows = ((0, used_mib + 300.0, 16384),),
        baseline_shared = _shared(120),
        shared = _shared(120 + 74, nvidia_dedicated_mib = 12000),  # the spill, as Windows reports it
    )
    _arm(backend)
    message = backend._verify_vram_residency()
    assert message is not None, "the direct counter missed a 74 MiB spill"
    assert "About 74 MiB" in message
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
        [(0, 300.0, 16384)],
        baseline_rows = ((0, used_mib + 300.0, 16384),),
        baseline_shared = _shared(120),
        shared = _shared(140, nvidia_dedicated_mib = 12000),  # +20 MiB
    )
    _arm(backend)
    assert backend._verify_vram_residency() is None


def test_another_app_growing_shared_memory_is_not_our_spill(on_windows):
    """The counter is per-adapter: a rise needs a device-side shortfall to count."""
    used_mib = QWEN3_VL_8B_FLOOR / MIB  # fully resident
    backend = _Backend(
        [(0, 16384.0 - used_mib, 16384)],
        baseline_shared = _shared(120),
        shared = _shared(620, nvidia_dedicated_mib = 12000),
    )
    _arm(backend)
    assert backend._verify_vram_residency() is None
    assert backend._warnings == []


def test_growth_on_two_adapters_is_ambiguous_and_falls_back(on_windows):
    """Attributing the spill to a card is the point; a guess would name the wrong one."""
    used_mib = (QWEN3_VL_8B_FLOOR / MIB) - 4096  # 4 GiB short
    backend = _Backend(
        [(0, 16384.0 - used_mib, 16384)],
        baseline_shared = _shared(120, igpu_mib = 64),
        shared = _shared(2120, igpu_mib = 1064, nvidia_dedicated_mib = 12000),  # +2000 and +1000 MiB
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
    backend._sample_residency_baseline(PIN_ARGV)
    assert backend._pin_baseline_shared_usage is None
    assert backend._verify_vram_residency() is not None


def test_the_fit_on_retry_disarms_the_check(on_windows):
    """The retry rungs rewrite the argv, and the check must follow them."""
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


@pytest.mark.parametrize(
    "extra, env",
    [
        (["--cpu-moe"], {}),
        (["-ot", "exps=CPU"], {}),
        (["--n-cpu-moe", "30"], {}),
        (["-nkvo"], {}),
        (["--device", "none"], {}),
        (["--gpu-layers=10"], {}),
        (["--fit=on"], {}),
        ([], {"LLAMA_ARG_CPU_MOE": "1"}),
        ([], {"LLAMA_ARG_NO_KV_OFFLOAD": "1"}),
    ],
)
def test_user_host_placement_after_the_pin_is_not_a_spill(on_windows, extra, env):
    """User extras follow the pin, so host placement they ask for is not a spill."""
    backend = _Backend([(0, 16384.0 - 3000.0, 16384)])
    backend._arm_residency_check(QWEN3_VL_8B_FLOOR, [0])
    backend._sample_residency_baseline(PIN_ARGV + extra, env)
    assert backend._verify_vram_residency() is None
    assert backend._warnings == []


def test_a_plain_pin_under_a_clean_env_stays_armed(on_windows):
    backend = _Backend([(0, 16384.0 - 3000.0, 16384)])
    backend._arm_residency_check(QWEN3_VL_8B_FLOOR, [0])
    backend._sample_residency_baseline(PIN_ARGV + ["--n-cpu-moe", "0"], {})
    assert backend._verify_vram_residency() is not None


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


def test_an_igpu_growing_shared_memory_is_not_the_nvidia_spill(on_windows):
    """Optimus: the display iGPU's shared usage moves on its own; only the loaded card counts."""
    used_mib = (QWEN3_VL_8B_FLOOR / MIB) - 300  # benign: tok_embd stays on the host
    backend = _Backend(
        [(0, 16384.0 - used_mib, 16384)],
        baseline_shared = _shared(120, igpu_mib = 64),
        shared = _shared(120, igpu_mib = 564, nvidia_dedicated_mib = 12000),
    )
    _arm(backend)
    assert backend._verify_vram_residency() is None
    assert backend._warnings == []


def test_a_spill_beside_igpu_churn_is_still_reported(on_windows):
    used_mib = (QWEN3_VL_8B_FLOOR / MIB) - 300
    backend = _Backend(
        [(0, 300.0, 16384)],
        baseline_rows = ((0, used_mib + 300.0, 16384),),
        baseline_shared = _shared(120, igpu_mib = 64),
        shared = _shared(420, igpu_mib = 564, nvidia_dedicated_mib = 12000),
    )
    _arm(backend)
    assert backend._verify_vram_residency() is not None


def test_not_armed_for_a_non_cuda_build(on_windows):
    """A Vulkan build on an NVIDIA host has no CUDA sysmem fallback, and other indices."""
    backend = _Backend([(0, 16384.0 - 3000.0, 16384)])
    backend._sysmem_fallback_risk = lambda binary = None: False
    backend._arm_residency_check(QWEN3_VL_8B_FLOOR, [0], "/vulkan/llama-server")
    assert backend._pin_resident_floor_bytes is None


def test_the_advice_offers_q8_0_only_for_an_f16_cache(on_windows):
    for cache, offered in (("f16", True), (None, True), ("q8_0", False), ("q4_0", False)):
        backend = _Backend([(0, 16384.0 - 11469.0, 16384)])
        backend._arm_residency_check(QWEN3_VL_8B_FLOOR, [0], None, cache)
        backend._sample_residency_baseline(PIN_ARGV, {})
        message = backend._verify_vram_residency()
        assert ("q8_0" in message) is offered, cache


def test_a_spill_is_appended_to_an_earlier_notice(on_windows):
    backend = _Backend([(0, 16384.0 - 11469.0, 16384)])
    backend._last_load_warning = "Earlier notice."
    backend._amend_load_warning = lambda note: setattr(
        backend, "_last_load_warning", backend._last_load_warning + note
    )
    _arm(backend)
    message = backend._verify_vram_residency()
    assert backend._last_load_warning == "Earlier notice. " + message


def test_growth_counts_only_as_many_adapters_as_were_pinned():
    """Another dGPU app growing dedicated and shared memory is not this one-card load."""
    other = "luid_0x00000000_0x0000beef_phys_0"
    before = {NVIDIA_LUID: 0, other: 0, "dedicated|" + NVIDIA_LUID: 0, "dedicated|" + other: 0}
    after = {
        NVIDIA_LUID: 0,
        other: 500 * MIB,
        "dedicated|" + NVIDIA_LUID: 12000 * MIB,
        "dedicated|" + other: 800 * MIB,
    }
    assert LlamaCppBackend._shared_usage_growth_bytes(before, after, 1) == 0
    assert LlamaCppBackend._shared_usage_growth_bytes(before, after, 2) == 500 * MIB


def test_fallback_risk_is_cached_per_resolved_binary(monkeypatch):
    monkeypatch.setattr(mod.sys, "platform", "win32")
    monkeypatch.setattr(LlamaCppBackend, "_SYSMEM_FALLBACK_RISK", {})
    libs = {"/vulkan/llama-server": {"vulkan"}, "/cuda/llama-server": {"cuda"}}
    selected = ["/vulkan/llama-server"]
    monkeypatch.setattr(
        LlamaCppBackend,
        "_find_llama_server_binary",
        staticmethod(lambda include_denied = False: selected[0]),
    )
    monkeypatch.setattr(
        LlamaCppBackend,
        "_installed_ggml_backends",
        staticmethod(lambda binary = None: frozenset(libs[binary or selected[0]])),
    )
    assert LlamaCppBackend._sysmem_fallback_risk() is False
    selected[0] = "/cuda/llama-server"
    assert LlamaCppBackend._sysmem_fallback_risk() is True


def test_the_check_runs_before_the_load_returns(tmp_path, monkeypatch):
    """A verdict recorded after the load response is never shown, so it must come first."""
    from test_llama_cpp_placement import _backend as _placement_backend, _launch

    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "home"))
    backend, gguf = _placement_backend(tmp_path, vulkan = False, memory = [(0, 24_576, 24_576)])
    backend._verify_vram_residency = lambda: backend._record_load_warning("SPILL")
    _launch(backend, gguf, n_ctx = 2048)
    assert "SPILL" in (backend.last_load_warning or "")


def test_shared_growth_with_room_left_on_the_card_is_not_a_spill(on_windows):
    """WDDM spills only once dedicated VRAM is exhausted; 4 GiB free means someone else."""
    used_mib = QWEN3_VL_8B_FLOOR / MIB - 300
    backend = _Backend(
        [(0, 4096.0, 20480)],
        baseline_rows = ((0, used_mib + 4096.0, 20480),),
        baseline_shared = _shared(120),
        shared = _shared(620, nvidia_dedicated_mib = 12000),
    )
    _arm(backend)
    assert backend._verify_vram_residency() is None


def test_a_retry_baseline_waits_for_the_killed_child_to_release(on_windows):
    backend = _Backend([(0, 16384.0 - 3000.0, 16384)])
    backend._last_kill_monotonic = 1234.5
    _arm(backend)
    assert backend._settle_calls == [1234.5]


def test_a_small_card_is_full_only_below_its_own_reserve(on_windows):
    """A 4 GiB card keeps a 512 MiB reserve, so 800 MiB free is not a full card."""
    backend = _Backend(
        [(0, 800.0, 4096)],
        baseline_rows = ((0, 3800.0, 4096),),
        baseline_shared = _shared(120),
        shared = _shared(170, nvidia_dedicated_mib = 3000),
    )
    backend._arm_residency_check(3000 * MIB, [0])
    backend._sample_residency_baseline(PIN_ARGV, {})
    assert backend._verify_vram_residency() is None


def test_a_release_on_one_pinned_adapter_does_not_cancel_a_spill_on_another():
    second = "luid_0x00000000_0x0000beef_phys_0"
    before = {
        NVIDIA_LUID: 0,
        second: 100 * MIB,
        "dedicated|" + NVIDIA_LUID: 0,
        "dedicated|" + second: 0,
    }
    after = {
        NVIDIA_LUID: 74 * MIB,
        second: 0,
        "dedicated|" + NVIDIA_LUID: 9000 * MIB,
        "dedicated|" + second: 9000 * MIB,
    }
    assert LlamaCppBackend._shared_usage_growth_bytes(before, after, 2) == 74 * MIB


def test_fallback_risk_follows_an_in_place_binary_update(monkeypatch):
    monkeypatch.setattr(mod.sys, "platform", "win32")
    monkeypatch.setattr(LlamaCppBackend, "_SYSMEM_FALLBACK_RISK", {})
    build = ["vulkan"]
    monkeypatch.setattr(
        LlamaCppBackend, "_binary_revision", staticmethod(lambda binary: (binary, build[0]))
    )
    monkeypatch.setattr(
        LlamaCppBackend,
        "_installed_ggml_backends",
        staticmethod(lambda binary = None: frozenset({build[0]})),
    )
    assert LlamaCppBackend._sysmem_fallback_risk("/llama-server") is False
    build[0] = "cuda"
    assert LlamaCppBackend._sysmem_fallback_risk("/llama-server") is True


def test_a_retry_that_changes_the_slot_layout_disarms(on_windows):
    """The floor priced 4 unified slots; a --parallel 1 respawn holds a different KV."""
    backend = _Backend([(0, 16384.0 - 3000.0, 16384)])
    backend._arm_residency_check(QWEN3_VL_8B_FLOOR, [0])
    backend._sample_residency_baseline(PIN_ARGV + ["--parallel", "4", "--kv-unified"], {})
    assert backend._pin_resident_floor_bytes is not None
    backend._sample_residency_baseline(PIN_ARGV + ["--parallel", "1"], {})
    assert backend._pin_resident_floor_bytes is None
    assert backend._verify_vram_residency() is None


def test_a_retry_with_the_same_layout_stays_armed(on_windows):
    backend = _Backend([(0, 16384.0 - 11469.0, 16384)])
    backend._arm_residency_check(QWEN3_VL_8B_FLOOR, [0])
    argv = PIN_ARGV + ["--parallel", "4", "--kv-unified"]
    backend._sample_residency_baseline(argv, {})
    backend._sampled = False  # the respawn reads a fresh pre-spawn baseline
    backend._sample_residency_baseline(argv, {})
    assert backend._verify_vram_residency() is not None


def test_an_empty_backend_scan_is_not_cached(monkeypatch):
    """An unreadable lib dir scans empty; that must not pin False for the revision."""
    monkeypatch.setattr(mod.sys, "platform", "win32")
    monkeypatch.setattr(LlamaCppBackend, "_SYSMEM_FALLBACK_RISK", {})
    monkeypatch.setattr(LlamaCppBackend, "_binary_revision", staticmethod(lambda b: (b, 1)))
    scans = [frozenset(), frozenset({"cuda"})]
    monkeypatch.setattr(
        LlamaCppBackend, "_installed_ggml_backends", staticmethod(lambda b = None: scans.pop(0))
    )
    assert LlamaCppBackend._sysmem_fallback_risk("/llama-server") is False
    assert LlamaCppBackend._sysmem_fallback_risk("/llama-server") is True


def test_the_advice_names_the_launched_executable(on_windows):
    backend = _Backend([(0, 16384.0 - 11469.0, 16384)])
    backend._arm_residency_check(QWEN3_VL_8B_FLOOR, [0], "/opt/rt/my-server.exe")
    backend._sample_residency_baseline(PIN_ARGV, {})
    assert "add my-server.exe," in backend._verify_vram_residency()


@pytest.mark.parametrize("tied, expect_gib", [(False, 7), (True, 8)])
def test_the_floor_leaves_out_a_host_pinned_input_embedding(
    tmp_path, monkeypatch, tied, expect_gib
):
    """An untied token_embd stays on the CPU at full offload, so it is not owed to the card."""
    from types import SimpleNamespace

    from test_llama_cpp_placement import _backend as _placement_backend, _launch

    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(LlamaCppBackend, "_sysmem_fallback_risk", staticmethod(lambda b = None: True))
    backend, gguf = _placement_backend(tmp_path, vulkan = False, memory = [(0, 24_576, 24_576)])
    backend._get_gguf_size_bytes = lambda _path: 8 * GIB
    backend._tensor_spill_layout = lambda *a, **k: SimpleNamespace(
        complete = True,
        token_embd_bytes = 1 * GIB,
        lm_head_bytes = 0 if tied else 1 * GIB,
        excluded_block_bytes = 0,
    )
    floors = []
    backend._arm_residency_check = lambda floor, *a, **k: floors.append(floor)
    _launch(backend, gguf, n_ctx = 2048)
    assert floors == [expect_gib * GIB]
