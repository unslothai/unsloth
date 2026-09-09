# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
"""Advice to enlarge an integrated GPU's dedicated memory.

Measured on a Ryzen AI Max+ PRO 395 (gfx1151, 128 GB): the same 42.90 GiB model runs
3-4x faster with the weights inside the GPU allocation than spilling out of it (decode
11.58 -> 46.70 t/s, prefill 150.88 -> 579.68 t/s on ROCm). Worth telling the user, but
only when raising the setting would help and only where the setting exists.
"""

import ast
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.inference.llama_cpp import LlamaCppBackend  # noqa: E402

_GB = 1024**3
_advice = LlamaCppBackend._igpu_carveout_advice
_message = LlamaCppBackend._igpu_carveout_advice_message


def gb(n: float) -> int:
    return int(n * _GB)


class TestGating:
    """Who must never see this."""

    def test_a_discrete_gpu_is_never_advised(self):
        # A discrete card's VRAM is fixed silicon, however badly the model fits.
        assert _advice(gb(40), gb(8), gb(64), is_igpu = False) is None

    def test_a_model_that_already_fits_says_nothing(self):
        # 20 GB of weights inside a 32 GB allocation: nothing to fix.
        assert _advice(gb(20), gb(32), gb(96), is_igpu = True) is None

    def test_a_model_too_large_for_the_machine_says_nothing(self):
        # 120 GB of weights on a 128 GB machine: no allocation here holds it.
        assert _advice(gb(120), gb(32), gb(95.78), is_igpu = True) is None

    def test_unknown_inputs_never_advise(self):
        assert _advice(None, gb(32), gb(96), is_igpu = True) is None
        assert _advice(gb(40), None, gb(96), is_igpu = True) is None
        assert _advice(gb(40), gb(32), None, is_igpu = True) is None
        assert _advice(gb(40), 0, gb(96), is_igpu = True) is None


class TestTheMeasuredMachine:
    """The Strix Halo host the 3-4x was measured on, at both carve-outs."""

    def test_the_slow_configuration_is_advised(self):
        # 32 GiB allocation, 95.78 GiB visible, a 42.90 GiB model: the 3-4x slower one.
        got = _advice(gb(42.90), gb(32), gb(95.78), is_igpu = True)
        assert got is not None
        assert got["current_gb"] == 32.0
        assert got["needed_gb"] == 42.9
        # Smallest rung that holds the weights: every GB suggested leaves the desktop.
        assert got["suggested_gb"] == 48
        assert got["machine_gb"] == 127.8

    def test_the_fast_configuration_is_silent(self):
        # Same model after raising it to 96 GiB: weights resident, nothing to say.
        assert _advice(gb(42.90), gb(96), gb(31.78), is_igpu = True) is None

    def test_a_bigger_model_on_the_raised_machine_is_advised_again(self):
        # 67.56 GiB (Qwen3.8-Flash-Next UD-IQ1_S) against a 64 GiB allocation.
        got = _advice(gb(67.56), gb(64), gb(63.78), is_igpu = True)
        assert got is not None and got["suggested_gb"] == 96


class TestGeneralisesToOtherMachines:
    """Nothing may be pinned to 128 GB or to one vendor's menu."""

    def test_a_small_laptop(self):
        # 16 GB machine, 2 GB allocated, an 8 GB model.
        got = _advice(gb(8), gb(2), gb(14), is_igpu = True)
        assert got is not None
        assert got["suggested_gb"] == 8
        assert got["host_left_gb"] == 8.0

    def test_a_large_workstation(self):
        # 512 GB machine, 32 GB allocated, a 300 GB model.
        got = _advice(gb(300), gb(32), gb(480), is_igpu = True)
        assert got is not None
        assert got["suggested_gb"] == 384
        assert got["machine_gb"] == 512.0

    def test_the_host_always_keeps_a_share(self):
        # A fifth of the machine, or 8 GB, whichever is larger, stays with the OS.
        for machine, carve, need in ((64, 8, 40), (128, 32, 80), (256, 16, 150)):
            got = _advice(gb(need), gb(carve), gb(machine - carve), is_igpu = True)
            assert got is not None, (machine, carve, need)
            assert got["suggested_gb"] <= machine - max(8, machine * 0.20)

    def test_a_suggestion_is_always_an_increase(self):
        # A rung at or below what is already set is not advice.
        got = _advice(gb(33), gb(32), gb(95.78), is_igpu = True)
        assert got is None or got["suggested_gb"] > got["current_gb"]

    def test_the_ladder_is_ascending_and_bounded(self):
        for cap in (7, 16, 100, 1000):
            rungs = LlamaCppBackend._igpu_carveout_ladder_gb(cap)
            assert rungs == sorted(rungs)
            assert all(r <= cap for r in rungs)


class TestRecordingItOnALoad:
    """The launch-site wiring: what actually reaches the client."""

    @staticmethod
    def _backend(
        monkeypatch,
        *,
        is_igpu = True,
        carve_bytes = 32 * _GB,
        total_mib = 95 * 1024,
    ):
        # __new__: no real server, config or filesystem. The method under test only
        # reads the stubs below and writes one attribute.
        backend = LlamaCppBackend.__new__(LlamaCppBackend)
        monkeypatch.setattr(
            LlamaCppBackend, "_amd_apu_wants_unified_memory", staticmethod(lambda _i = None: is_igpu)
        )
        monkeypatch.setattr(
            LlamaCppBackend, "_integrated_cuda_unified_memory", staticmethod(lambda _i = None: False)
        )
        monkeypatch.setattr(
            LlamaCppBackend,
            "_igpu_dedicated_memory_bytes",
            staticmethod(lambda _i = None, **_kwargs: carve_bytes),
        )
        monkeypatch.setattr(
            LlamaCppBackend, "_total_system_memory_mib", staticmethod(lambda: total_mib)
        )
        return backend

    def test_a_spilling_load_records_advice_with_prose(self, monkeypatch):
        backend = self._backend(monkeypatch)
        backend._record_carveout_advice(None, gb(42.90))
        advice = backend.last_carveout_advice
        assert advice is not None
        assert advice["suggested_gb"] == 48
        assert "48 GB" in advice["message"]

    def test_a_discrete_gpu_records_nothing(self, monkeypatch):
        backend = self._backend(monkeypatch, is_igpu = False)
        backend._record_carveout_advice(None, gb(42.90))
        assert backend.last_carveout_advice is None

    def test_an_unknown_model_size_records_nothing(self, monkeypatch):
        # _unified_need is None unless the launch forces a full offload.
        backend = self._backend(monkeypatch)
        backend._record_carveout_advice(None, None)
        assert backend.last_carveout_advice is None

    def test_an_unreadable_allocation_records_nothing(self, monkeypatch):
        backend = self._backend(monkeypatch, carve_bytes = None)
        backend._record_carveout_advice(None, gb(42.90))
        assert backend.last_carveout_advice is None

    def test_a_dismissed_notice_is_not_recorded_again(self, monkeypatch):
        from utils.igpu_carveout_notice_settings import dismiss_notice

        backend = self._backend(monkeypatch)
        dismiss_notice(32.0)
        backend._record_carveout_advice(None, gb(42.90))
        assert backend.last_carveout_advice is None

    def test_a_raised_allocation_speaks_again_after_dismissal(self, monkeypatch):
        # Dismissed at 32 GB, user raised it to 64 GB, still short: say it once more.
        from utils.igpu_carveout_notice_settings import dismiss_notice

        dismiss_notice(32.0)
        backend = self._backend(monkeypatch, carve_bytes = 64 * _GB, total_mib = int(63.78 * 1024))
        backend._record_carveout_advice(None, gb(67.56))
        assert backend.last_carveout_advice is not None

    def test_a_broken_reading_never_breaks_the_load(self, monkeypatch):
        def boom(_i = None):
            raise RuntimeError("driver went away")

        backend = self._backend(monkeypatch)
        monkeypatch.setattr(LlamaCppBackend, "_igpu_dedicated_memory_bytes", staticmethod(boom))
        backend._record_carveout_advice(None, gb(42.90))  # must not raise
        assert backend.last_carveout_advice is None


class TestTheMessage:
    """What the user reads."""

    def test_it_names_the_numbers_and_the_cost(self):
        msg = _message(_advice(gb(42.90), gb(32), gb(95.78), is_igpu = True))
        assert "43 GB" in msg and "32 GB" in msg and "48 GB" in msg
        assert "80 GB" in msg  # what the host keeps: the trade-off, stated

    def test_it_says_where_the_setting_lives_without_claiming_a_menu(self):
        # Firmware on one machine, a driver panel on the next, under different names.
        # Naming both and neither specifically is as far as this can honestly go.
        msg = _message(_advice(gb(42.90), gb(32), gb(95.78), is_igpu = True))
        assert "firmware" in msg and "control panel" in msg
        # No menu path, key or control name is claimed.
        for invented in ("F10", "UMA Frame Buffer", "Variable Graphics Memory", "Advanced >"):
            assert invented not in msg

    def test_it_stays_short_enough_for_a_toast(self):
        # A tall toast covers the controls under it for as long as it is up (see
        # xet-progress-notice.ts, #9293). Bounded at the widest plausible reading,
        # since a 4-digit machine renders longer than the development one.
        widest = _message(_advice(gb(400), gb(0.5), gb(1023.5), is_igpu = True))
        assert len(widest) <= 260, (len(widest), widest)
        # Two sentences, and no paragraph breaks: the dialog's three-paragraph body
        # is what this replaced.
        assert "\n" not in widest

    def test_it_names_no_vendor(self):
        msg = _message(_advice(gb(42.90), gb(32), gb(95.78), is_igpu = True))
        for vendor in ("AMD", "Adrenalin", "Intel", "NVIDIA", "HP", "Ryzen", "Radeon"):
            assert vendor not in msg


class TestPropertiesOverEveryPlausibleMachine:
    """Swept rather than exampled. These caught three defects the chosen examples did
    not: a negative allocation read as truthy, a sub-1 GB allocation printing as
    "0 GB", and the integrated-GPU probe running on loads that could never advise."""

    MACHINES_GB = [8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 512, 1024]
    ALLOCATIONS_GB = [0.125, 0.5, 1, 2, 4, 8, 16, 24, 32, 48, 64, 96]
    MODELS_GB = [0.5, 2, 7, 13, 20, 30, 43, 60, 80, 110, 200, 400]

    def _cases(self):
        for machine in self.MACHINES_GB:
            for carve in self.ALLOCATIONS_GB:
                if carve >= machine:
                    continue
                host = gb(machine - carve)
                for model in self.MODELS_GB:
                    yield machine, carve, model, host

    def test_following_the_advice_always_ends_it(self):
        # An advisory that survives being followed is a nag, and one that cannot be
        # satisfied is a bug.
        checked = 0
        for machine, carve, model, host in self._cases():
            first = _advice(gb(model), gb(carve), host, is_igpu = True)
            if first is None:
                continue
            applied = float(first["suggested_gb"])
            again = _advice(gb(model), gb(applied), gb(machine - applied), is_igpu = True)
            assert again is None, (machine, carve, model, first, again)
            checked += 1
        assert checked > 200, checked

    def test_the_host_always_keeps_a_workable_share(self):
        for machine, carve, model, host in self._cases():
            result = _advice(gb(model), gb(carve), host, is_igpu = True)
            if result is None:
                continue
            floor = max(8, result["machine_gb"] * 0.20)
            assert result["host_left_gb"] >= floor - 0.15, (result, floor)

    def test_the_suggestion_is_always_an_increase_that_covers_the_model(self):
        for machine, carve, model, host in self._cases():
            result = _advice(gb(model), gb(carve), host, is_igpu = True)
            if result is None:
                continue
            assert result["suggested_gb"] > result["current_gb"], result
            assert result["suggested_gb"] >= result["needed_gb"] - 0.05, result

    def test_a_discrete_gpu_is_never_advised_anywhere_in_the_sweep(self):
        for machine, carve, model, host in self._cases():
            assert _advice(gb(model), gb(carve), host, is_igpu = False) is None

    @pytest.mark.parametrize("bad", [-1, -(10**12), 0, None, float("nan"), float("inf")])
    def test_a_nonsense_reading_produces_no_advice(self, bad):
        # -1 is truthy, so a bare falsiness test produced confident wrong advice.
        assert _advice(bad, gb(32), gb(96), is_igpu = True) is None
        assert _advice(gb(43), bad, gb(96), is_igpu = True) is None
        assert _advice(gb(43), gb(32), bad, is_igpu = True) is None


class TestASmallAutomaticAllocation:
    """An APU left on its automatic setting reports a few hundred megabytes, not
    a round number of GB. The DirectX record really does read that way."""

    def test_it_is_described_rather_than_rounded_to_zero(self):
        result = _advice(gb(12), gb(0.5), gb(31.5), is_igpu = True)
        assert result is not None
        msg = _message(result)
        assert "0.5 GB" in msg, msg
        assert not re.search(r"(?<![\d.])0 GB", msg), msg

    def test_no_sweep_case_prints_a_zero_or_negative_quantity(self):
        for carve in (0.125, 0.25, 0.5, 0.75, 1, 2):
            for model in (1, 4, 12, 30):
                result = _advice(gb(model), gb(carve), gb(64 - carve), is_igpu = True)
                if result is None:
                    continue
                msg = _message(result)
                assert not re.search(r"(?<![\d.])0 GB", msg), msg
                assert not re.search(r"-\d", msg), msg


class TestTheSmallAllocationsTheLadderMustOffer:
    """An APU on its automatic setting reports a few hundred megabytes."""

    def test_the_low_rungs_exist(self):
        # Firmware offers 1 and 2 GB, so a ladder starting at 4 could only advise
        # past them.
        assert LlamaCppBackend._igpu_carveout_ladder_gb(8) == [1, 2, 3, 4, 6, 8]

    def test_a_small_model_is_advised_onto_the_smallest_rung_that_fits(self):
        # 1 GB allocated, 2 GB of weights, a 16 GB machine.
        advice = _advice(gb(2), gb(1), gb(15), is_igpu = True)
        assert advice is not None
        assert advice["suggested_gb"] == 2
        assert advice["host_left_gb"] == 14.0

    def test_the_measured_machine_is_unchanged(self):
        # The low rungs must not perturb the case this feature was built for.
        advice = _advice(gb(42.90), gb(32), gb(95.78), is_igpu = True)
        assert advice is not None and advice["suggested_gb"] == 48


class TestTheLadderTerminates:
    """It is a `while` loop on the model-load path, and the caller's try/except
    cannot rescue a hang."""

    @pytest.mark.parametrize("cap", [float("inf"), float("-inf"), float("nan"), 0, -5, 2**60])
    def test_it_returns_promptly_for_any_cap(self, cap):
        import threading

        done = threading.Event()

        def run():
            LlamaCppBackend._igpu_carveout_ladder_gb(cap)
            done.set()

        thread = threading.Thread(target = run, daemon = True)
        thread.start()
        thread.join(timeout = 5)
        assert done.is_set(), f"ladder({cap}) did not terminate"


class TestTheRungTheUserIsAlreadyOn:
    """A driver reports the pool it kept, not the number in the firmware menu."""

    def test_a_reading_just_under_its_own_rung_is_not_advised_back_to_it(self):
        # 95.83 GB is the 96 GB setting as the driver reports it, so the rung covering
        # a 95.9 GB model is 96: the setting already in force.
        assert _advice(gb(95.9), gb(95.83), gb(31.78), is_igpu = True) is None

    def test_the_next_real_rung_is_still_advised(self):
        # The slack must not swallow a genuine step up: a 32 GB reading still earns 48.
        assert _advice(gb(42.9), gb(32), gb(95.8), is_igpu = True)["suggested_gb"] == 48

    def test_the_slack_is_narrower_than_the_gap_between_rungs(self):
        # 0.5 GB of drift, against a ladder whose closest pair is 4 -> 6.
        assert _advice(gb(5), gb(4.4), gb(27.6), is_igpu = True)["suggested_gb"] == 6


class TestThePlacementItAdvisesAbout:
    """Which device the advice is about, in the index space that device is named in.

    A Vulkan launch numbers devices with VULKAN ORDINALS; the ROCm gate reads the same
    integers as physical HIP ids. On a mixed APU/dGPU host that is how a dGPU load
    earns advice to resize an integrated GPU it never touched.
    """

    @staticmethod
    def _backend(
        monkeypatch,
        *,
        probes,
        rocm_gate = True,
        carve_bytes = 32 * _GB,
    ):
        backend = LlamaCppBackend.__new__(LlamaCppBackend)

        def _read(_i = None, **_kwargs):
            probes.append(_i)
            return carve_bytes

        monkeypatch.setattr(LlamaCppBackend, "_igpu_dedicated_memory_bytes", staticmethod(_read))
        monkeypatch.setattr(
            LlamaCppBackend,
            "_amd_apu_wants_unified_memory",
            staticmethod(lambda _i = None: rocm_gate),
        )
        monkeypatch.setattr(
            LlamaCppBackend, "_total_system_memory_mib", staticmethod(lambda: 95 * 1024)
        )
        return backend

    def test_a_vulkan_launch_on_a_discrete_device_is_never_even_priced(self, monkeypatch):
        # Ordinal 1 is not in the planner's shared set, so this offloads to a discrete
        # card. No advice, and no allocation reading either: on Linux that falls
        # through to the ROCm pool, which imports torch for a load that cannot be
        # advised.
        probes = []
        backend = self._backend(monkeypatch, probes = probes)
        backend._record_carveout_advice(
            [1],
            gb(42.90),
            is_vulkan_backend = True,
            shared_gpu_ids = {0},
            detected_gpus = [(0, 0), (1, 0)],
        )
        assert backend.last_carveout_advice is None
        assert probes == [], "the allocation was read for a device that shares nothing"

    def test_a_vulkan_launch_on_the_shared_device_is_advised(self, monkeypatch):
        # The ROCm gate answers False, having read the Vulkan ordinal as a physical id.
        # The Vulkan classification applies, and it says this device shares memory.
        backend = self._backend(monkeypatch, probes = [], rocm_gate = False)
        backend._record_carveout_advice(
            [0],
            gb(42.90),
            is_vulkan_backend = True,
            shared_gpu_ids = {0},
            detected_gpus = [(0, 0)],
        )
        advice = backend.last_carveout_advice
        assert advice is not None and advice["suggested_gb"] == 48

    def test_an_unknown_vulkan_inventory_says_nothing(self, monkeypatch):
        # Fails closed like every other reading here: no shared set, no advice.
        probes = []
        backend = self._backend(monkeypatch, probes = probes)
        backend._record_carveout_advice(
            [0],
            gb(42.90),
            is_vulkan_backend = True,
            shared_gpu_ids = None,
            detected_gpus = [],
        )
        assert backend.last_carveout_advice is None
        assert probes == []

    def test_a_user_device_override_declines(self, monkeypatch):
        # With no gpu_ids a user --device wins last-wins over the generated pin, so the
        # placement this would describe is not the one that runs. The cache tuning
        # declines for the same reason and with the same test.
        probes = []
        backend = self._backend(monkeypatch, probes = probes)
        backend._record_carveout_advice([0], gb(42.90), target_unknown = True)
        assert backend.last_carveout_advice is None
        assert probes == []

    def test_a_non_vulkan_launch_still_uses_the_rocm_gate(self, monkeypatch):
        # The ROCm path is unchanged, gate still asked last, after the shortfall.
        backend = self._backend(monkeypatch, probes = [], rocm_gate = False)
        backend._record_carveout_advice([0], gb(42.90))
        assert backend.last_carveout_advice is None


class TestTheCpuOnlyReplay:
    """A Vulkan crash replays with --gpu-layers 0 --device none."""

    def test_the_advice_does_not_survive_it(self, monkeypatch):
        # No allocation holds any weights after this replay. Both CPU-fallback call
        # sites reach this one function, which is why the clear lives here.
        backend = LlamaCppBackend.__new__(LlamaCppBackend)
        backend._last_carveout_advice = {"current_gb": 32.0, "suggested_gb": 48}
        backend._last_load_warning = None
        monkeypatch.setattr(LlamaCppBackend, "_launch_host_shortfall_message", lambda *a, **k: None)
        monkeypatch.setattr(LlamaCppBackend, "_record_load_warning", lambda self, msg: None)
        backend._reprice_after_cpu_only_fallback(
            host_msg = None, cpu_cmd = ["llama-server"], env = {}, avail_mib = 1024
        )
        assert backend.last_carveout_advice is None


class TestTheResponseRechecksTheDismissal:
    """The already-resident path returns a cached payload, not a fresh launch."""

    @staticmethod
    def _backend(advice):
        backend = LlamaCppBackend.__new__(LlamaCppBackend)
        backend._last_carveout_advice = advice
        return backend

    def test_a_dismissed_notice_is_stripped_from_the_response(self):
        # The already-resident path never runs the launch-time gate, so the toast came
        # back on every later load of the same model.
        from routes.inference import _live_carveout_advice
        from utils.igpu_carveout_notice_settings import dismiss_notice

        advice = {"current_gb": 32.0, "suggested_gb": 48, "message": "..."}
        backend = self._backend(advice)
        assert _live_carveout_advice(backend) == advice
        dismiss_notice(32.0)
        assert _live_carveout_advice(backend) is None

    def test_a_larger_allocation_still_speaks(self):
        # Dismissed at 32, now running 64 and short again: the launch path's rule.
        from routes.inference import _live_carveout_advice
        from utils.igpu_carveout_notice_settings import dismiss_notice

        dismiss_notice(32.0)
        backend = self._backend({"current_gb": 64.0, "suggested_gb": 96, "message": "..."})
        assert _live_carveout_advice(backend) is not None

    def test_no_advice_stays_no_advice(self):
        from routes.inference import _live_carveout_advice
        assert _live_carveout_advice(self._backend(None)) is None


class TestTheArchitectureGatedCpuLaunch:
    """Every GPU unsupported by this llama.cpp build, so the child runs on the CPU."""

    def test_a_forced_cpu_launch_is_never_advised(self, monkeypatch):
        # Priced before the env block masks every device away, so the argv still reads
        # as a full GPU offload. Without the gate an unsupported APU's CPU-only load
        # returns a toast about a GPU this build cannot use at all.
        probes = []

        def _read(_i = None, **_kwargs):
            probes.append(_i)
            return 32 * _GB

        backend = LlamaCppBackend.__new__(LlamaCppBackend)
        monkeypatch.setattr(LlamaCppBackend, "_igpu_dedicated_memory_bytes", staticmethod(_read))
        monkeypatch.setattr(
            LlamaCppBackend, "_amd_apu_wants_unified_memory", staticmethod(lambda _i = None: True)
        )
        monkeypatch.setattr(
            LlamaCppBackend, "_total_system_memory_mib", staticmethod(lambda: 95 * 1024)
        )
        backend._record_carveout_advice([0], gb(42.90), forced_cpu = True)
        assert backend.last_carveout_advice is None
        assert probes == [], "the allocation was read for a launch that reaches no GPU"


class TestEveryCallSitePricesThePlacementItRuns:
    """The gate and the retry, pinned at the call sites rather than in prose.

    Both are one keyword argument, and both were missing at first: a launch reaching
    no GPU was advised to enlarge one, and a crash retry landing on a different GPU
    said nothing about it.
    """

    @staticmethod
    def _calls():
        source = Path(sys.modules[LlamaCppBackend.__module__].__file__)
        tree = ast.parse(source.read_text(encoding = "utf-8"))
        return [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "_record_carveout_advice"
        ]

    def test_the_launch_call_sites_pass_the_forced_cpu_gate(self):
        by_target = {
            ast.unparse(call.args[0]): {kw.arg for kw in call.keywords} for call in self._calls()
        }
        assert "forced_cpu" in by_target["gpu_indices"]
        assert "forced_cpu" in by_target["_unified_gpu_indices"]

    def test_the_proactive_arch_gate_reprices_against_the_survivors(self):
        # The gate narrows onto supported devices before the spawn, and
        # _rocm_selected_pool_mib declines the unnarrowed set on a mixed host, so a
        # model outgrowing the surviving APU's carve-out was never advised about.
        by_target = {
            ast.unparse(call.args[0]): {kw.arg for kw in call.keywords} for call in self._calls()
        }
        assert (
            "_survivors" in by_target
        ), "the proactive architecture gate does not re-price the carve-out advice"
        assert "forced_cpu" in by_target["_survivors"]

    def test_the_architecture_retry_reprices_against_the_surviving_gpus(self):
        # _begin_load_warnings() drops the advice priced for the crashed placement, but
        # the respawn can land on an APU the same weights outgrow, so the retry has to
        # price it again rather than only clear.
        by_target = {
            ast.unparse(call.args[0]): {kw.arg for kw in call.keywords} for call in self._calls()
        }
        assert (
            "_remaining" in by_target
        ), "the arch-crash retry does not re-price the carve-out advice"
        assert "shared_gpu_ids" in by_target["_remaining"]


class TestWhichAdapterTheAllocationBelongsTo:
    """A registry record is only the integrated GPU's when nothing else could be."""

    @staticmethod
    def _with_records(monkeypatch, by_vendor):
        import utils.hardware.hardware as hw

        def _records(vendor_id = hw._AMD_PCI_VENDOR_ID, *, distinguish_failure = False):
            answer = by_vendor.get(vendor_id, {})
            if answer is None and not distinguish_failure:
                return {}
            return answer

        monkeypatch.setattr(hw, "_windows_amd_adapter_records_by_luid", _records)
        monkeypatch.setattr(
            LlamaCppBackend, "_rocm_selected_pool_mib", staticmethod(lambda _i = None: None)
        )

    def test_one_adapter_is_the_one_being_advised_about(self, monkeypatch):
        import utils.hardware.hardware as hw
        self._with_records(
            monkeypatch, {hw._AMD_PCI_VENDOR_ID: {1: {"dedicated_memory_bytes": 32 * _GB}}}
        )
        assert LlamaCppBackend._igpu_dedicated_memory_bytes([0]) == 32 * _GB

    def test_an_intel_igpu_beside_a_discrete_radeon_declines(self, monkeypatch):
        # What the AMD-only query got wrong: the Intel iGPU is the shared one, but the
        # single AMD record is the discrete card, whose fixed VRAM was then quoted as
        # the integrated GPU's allocation.
        import utils.hardware.hardware as hw
        self._with_records(
            monkeypatch,
            {
                hw._AMD_PCI_VENDOR_ID: {1: {"dedicated_memory_bytes": 16 * _GB}},
                hw._INTEL_PCI_VENDOR_ID: {2: {"dedicated_memory_bytes": 2 * _GB}},
            },
        )
        assert LlamaCppBackend._igpu_dedicated_memory_bytes([0]) is None

    def test_an_intel_only_host_is_counted_but_never_quoted(self, monkeypatch):
        # Intel is read for the COUNT, which makes the attribution above safe, never
        # for the answer: on Intel UMA the DirectX value is a small dedicated block
        # beside memory the driver hands out dynamically.
        import utils.hardware.hardware as hw
        self._with_records(
            monkeypatch, {hw._INTEL_PCI_VENDOR_ID: {2: {"dedicated_memory_bytes": 8 * _GB}}}
        )
        assert LlamaCppBackend._igpu_dedicated_memory_bytes([0]) is None

    def test_an_adapter_with_no_readable_allocation_still_counts(self, monkeypatch):
        # The count IS the attribution test: filtering out an APU record with no
        # dedicated-memory value left the discrete Radeon beside it as the only
        # candidate, with its fixed 16 GB quoted as the APU's carve-out.
        import utils.hardware.hardware as hw
        self._with_records(
            monkeypatch,
            {
                hw._AMD_PCI_VENDOR_ID: {
                    1: {"gfx": "gfx1151"},
                    2: {"dedicated_memory_bytes": 16 * _GB},
                }
            },
        )
        assert LlamaCppBackend._igpu_dedicated_memory_bytes([0]) is None

    def test_a_vendor_the_registry_could_not_read_fails_closed(self, monkeypatch):
        # An incomplete inventory cannot say the adapter it did see is the only one.
        import utils.hardware.hardware as hw
        self._with_records(
            monkeypatch,
            {
                hw._AMD_PCI_VENDOR_ID: {1: {"dedicated_memory_bytes": 32 * _GB}},
                hw._INTEL_PCI_VENDOR_ID: None,
            },
        )
        assert LlamaCppBackend._igpu_dedicated_memory_bytes([0]) is None

    def test_no_registry_record_at_all_falls_through_to_rocm(self, monkeypatch):
        # The Linux path: no adapter records, so the ROCm pool is the reading, and the
        # vendor rule above must not swallow it.
        self._with_records(monkeypatch, {})
        monkeypatch.setattr(
            LlamaCppBackend, "_rocm_selected_pool_mib", staticmethod(lambda _i = None: 32 * 1024)
        )
        assert LlamaCppBackend._igpu_dedicated_memory_bytes([0]) == 32 * _GB


class TestTheRoutingIsDeterministic:
    """Every gate this advisory routes through, enumerated rather than sampled.

    Seven interacting inputs: a Vulkan launch is classified by the planner's shared
    set and a non-Vulkan one by the ROCm unified-memory ids, two gates decline before
    any reading is taken, and the dismissal is asked last. Each was added for a defect
    found one at a time, so the value is the whole product rather than the cases
    anyone thought to write.

    Nothing here touches the host -- both classifiers, the allocation reading and the
    total memory are pinned -- so the table is the same everywhere.
    """

    _NEED = gb(42.90)
    _HOST_MIB = 95 * 1024
    # None is "no reading", 64 GB holds the weights, 32 GB is the measured shortfall.
    _CARVE_OUTS = (None, 64 * _GB, 32 * _GB)

    @staticmethod
    def _expected(
        *, carve, is_vulkan, shared_ids, unified_ids, forced_cpu, target_unknown, dismissed
    ) -> bool:
        """The routing, written out independently of the code under test."""
        if forced_cpu or target_unknown:
            return False
        if is_vulkan:
            if not shared_ids or 0 not in shared_ids:
                return False
        elif 0 not in unified_ids:
            return False
        if carve is None or TestTheRoutingIsDeterministic._NEED <= carve:
            return False
        return not dismissed

    def _run(self, monkeypatch, **case):
        probes = []

        def _read(_i = None, **_kwargs):
            probes.append(_i)
            return case["carve"]

        monkeypatch.setattr(LlamaCppBackend, "_igpu_dedicated_memory_bytes", staticmethod(_read))
        monkeypatch.setattr(
            LlamaCppBackend,
            "_rocm_unified_memory_gpu_ids",
            staticmethod(lambda: set(case["unified_ids"])),
        )
        monkeypatch.setattr(
            LlamaCppBackend, "_total_system_memory_mib", staticmethod(lambda: self._HOST_MIB)
        )
        import utils.igpu_carveout_notice_settings as notice_settings

        monkeypatch.setattr(
            notice_settings, "notice_already_dismissed", lambda _gb: case["dismissed"]
        )

        backend = LlamaCppBackend.__new__(LlamaCppBackend)
        backend._record_carveout_advice(
            [0],
            self._NEED,
            is_vulkan_backend = case["is_vulkan"],
            shared_gpu_ids = case["shared_ids"],
            detected_gpus = [(0, 0)],
            target_unknown = case["target_unknown"],
            forced_cpu = case["forced_cpu"],
        )
        return backend.last_carveout_advice, probes

    @staticmethod
    def _cases():
        import itertools
        for (
            carve,
            is_vulkan,
            shared_ids,
            unified_ids,
            forced_cpu,
            target_unknown,
            dismissed,
        ) in itertools.product(
            TestTheRoutingIsDeterministic._CARVE_OUTS,
            (False, True),
            (None, frozenset(), frozenset({0}), frozenset({1})),
            (frozenset(), frozenset({0})),
            (False, True),
            (False, True),
            (False, True),
        ):
            yield {
                "carve": carve,
                "is_vulkan": is_vulkan,
                "shared_ids": shared_ids,
                "unified_ids": unified_ids,
                "forced_cpu": forced_cpu,
                "target_unknown": target_unknown,
                "dismissed": dismissed,
            }

    def test_every_combination_routes_the_way_the_table_says(self, monkeypatch):
        wrong = []
        spoke = 0
        for case in self._cases():
            advice, _probes = self._run(monkeypatch, **case)
            expected = self._expected(**case)
            spoke += bool(advice)
            if bool(advice) != expected:
                wrong.append((case, bool(advice), expected))
        assert not wrong, f"{len(wrong)} of the routing cases disagree: {wrong[:3]}"
        # A table where nothing ever speaks would pass every assertion above.
        assert spoke, "no combination produced advice, so this proves nothing"

    def test_the_same_inputs_always_reach_the_same_answer(self, monkeypatch):
        # Nothing here is time, order or host dependent, message included.
        first = [self._run(monkeypatch, **case)[0] for case in self._cases()]
        second = [self._run(monkeypatch, **case)[0] for case in self._cases()]
        assert first == second

    def test_a_declined_placement_is_never_priced(self, monkeypatch):
        # The cheap gates all sit BEFORE the allocation reading, which on Linux imports
        # torch. A load that could never be advised must not pay for it.
        for case in self._cases():
            declined_early = (
                case["forced_cpu"]
                or case["target_unknown"]
                or (case["is_vulkan"] and (not case["shared_ids"] or 0 not in case["shared_ids"]))
            )
            if not declined_early:
                continue
            _advice, probes = self._run(monkeypatch, **case)
            assert probes == [], f"the allocation was read for a declined load: {case}"


class TestTheAdvisoryCannotReachTheLaunch:
    """Static, because "it only sets a field" is a claim about every path at once.

    A test can only show that the paths it drives change nothing. These read the
    module instead: what the recorder may write, and what the launch may do with what
    it returns.
    """

    @staticmethod
    def _module_tree():
        source = Path(sys.modules[LlamaCppBackend.__module__].__file__)
        return ast.parse(source.read_text(encoding = "utf-8"))

    @staticmethod
    def _recorder(tree):
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "_record_carveout_advice":
                return node
        raise AssertionError("_record_carveout_advice is gone")

    def test_it_writes_one_attribute_and_no_other(self):
        # Any other `self.x = ...` here would be launch state written by an advisory.
        written = set()
        for node in ast.walk(self._recorder(self._module_tree())):
            targets = []
            if isinstance(node, ast.Assign):
                targets = node.targets
            elif isinstance(node, (ast.AugAssign, ast.AnnAssign)):
                targets = [node.target]
            for target in targets:
                if (
                    isinstance(target, ast.Attribute)
                    and isinstance(target.value, ast.Name)
                    and target.value.id == "self"
                ):
                    written.add(target.attr)
        assert written == {"_last_carveout_advice"}, written

    def test_no_call_site_computes_anything_of_its_own(self):
        # Arguments are evaluated OUTSIDE the recorder's try/except, so a call in the
        # argument list is a way for an advisory to raise into a launch.
        tree = self._module_tree()
        calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "_record_carveout_advice"
        ]
        assert calls

        def cannot_raise(node) -> bool:
            # A name, a constant, an attribute of one, or an `or` over those. Anything
            # else -- a call above all -- is work outside the recorder's guard.
            if isinstance(node, (ast.Name, ast.Constant)):
                return True
            if isinstance(node, ast.Attribute):
                return cannot_raise(node.value)
            if isinstance(node, ast.BoolOp):
                return all(cannot_raise(value) for value in node.values)
            return False

        for call in calls:
            for argument in [*call.args, *(kw.value for kw in call.keywords)]:
                assert cannot_raise(
                    argument
                ), f"an advisory call site evaluates {ast.unparse(argument)}"

    def test_no_caller_can_branch_on_what_it_returns(self):
        # Every call is a bare expression statement, so a launch cannot branch on it.
        tree = self._module_tree()
        statements = {
            id(node.value)
            for node in ast.walk(tree)
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
        }
        calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "_record_carveout_advice"
        ]
        assert calls, "the recorder is never called"
        assert all(
            id(call) in statements for call in calls
        ), "a call site uses the return value of an advisory"


class TestTheIndexSpaceTheAllocationIsReadIn:
    """The Linux fallback reads HIP ids, and a Vulkan launch does not name devices that way.

    Two reasons it must not run there. `_rocm_selected_pool_mib` compares its argument
    with physical HIP ids, so Vulkan ordinals would land on another device wherever
    the enumerations differ. And the reading creates a HIP primary context in the
    backend process -- about 800 MiB, out of the pool the advice would then call too
    small. The Windows registry answer is unaffected: a few winreg queries, no ordinal.
    """

    @staticmethod
    def _readers(
        monkeypatch,
        *,
        pool_mib = 32 * 1024,
        records = None,
    ):
        """Record what the ROCm pool reader is asked, and stub the Windows registry."""
        asked = []

        def _pool(indices = None):
            asked.append(indices)
            return pool_mib

        monkeypatch.setattr(LlamaCppBackend, "_rocm_selected_pool_mib", staticmethod(_pool))
        import utils.hardware.hardware as hw

        monkeypatch.setattr(
            hw,
            "_windows_amd_adapter_records_by_luid",
            lambda vendor_id = hw._AMD_PCI_VENDOR_ID, **_kw: (records or {}).get(vendor_id, {}),
        )
        return asked

    def test_a_vulkan_launch_never_reaches_the_hip_reader(self, monkeypatch):
        asked = self._readers(monkeypatch)
        got = LlamaCppBackend._igpu_dedicated_memory_bytes([1], ordinals_are_vulkan = True)
        assert got is None
        assert asked == [], f"a Vulkan launch paid for a HIP context: {asked}"

    def test_a_vulkan_launch_still_reads_the_windows_registry(self, monkeypatch):
        # The half that must keep working: per adapter, and never touches an ordinal.
        import utils.hardware.hardware as hw

        asked = self._readers(
            monkeypatch,
            records = {hw._AMD_PCI_VENDOR_ID: {1: {"dedicated_memory_bytes": 32 * _GB}}},
        )
        got = LlamaCppBackend._igpu_dedicated_memory_bytes([0], ordinals_are_vulkan = True)
        assert got == 32 * _GB
        assert asked == []

    def test_a_non_vulkan_launch_still_scopes_by_physical_id(self, monkeypatch):
        # Those integers ARE HIP ids, and narrowing keeps a dGPU out of the answer.
        asked = self._readers(monkeypatch)
        LlamaCppBackend._igpu_dedicated_memory_bytes([0])
        assert asked == [[0]]
