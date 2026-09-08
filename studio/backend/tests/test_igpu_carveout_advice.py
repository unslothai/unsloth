# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
"""Advice to enlarge an integrated GPU's dedicated memory.

Measured on a Ryzen AI Max+ PRO 395 (gfx1151, 128 GB): the same 42.90 GiB model
runs 3-4x faster with the weights inside the GPU allocation than spilling out of
it (decode 11.58 -> 46.70 t/s, prefill 150.88 -> 579.68 t/s on ROCm). That gap is
worth telling the user about, but only when raising the setting would actually
help, and only on hardware where the setting exists.
"""

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
        # A discrete card's VRAM is fixed silicon. Telling someone to enlarge it
        # would be nonsense, however badly the model fits.
        assert _advice(gb(40), gb(8), gb(64), is_igpu = False) is None

    def test_a_model_that_already_fits_says_nothing(self):
        # 20 GB of weights inside a 32 GB allocation: nothing to fix.
        assert _advice(gb(20), gb(32), gb(96), is_igpu = True) is None

    def test_a_model_too_large_for_the_machine_says_nothing(self):
        # 120 GB of weights on a 128 GB machine: no allocation this machine can
        # offer holds it, so advice would be something the user cannot act on.
        assert _advice(gb(120), gb(32), gb(95.78), is_igpu = True) is None

    def test_unknown_inputs_never_advise(self):
        assert _advice(None, gb(32), gb(96), is_igpu = True) is None
        assert _advice(gb(40), None, gb(96), is_igpu = True) is None
        assert _advice(gb(40), gb(32), None, is_igpu = True) is None
        assert _advice(gb(40), 0, gb(96), is_igpu = True) is None


class TestTheMeasuredMachine:
    """The Strix Halo host the 3-4x was measured on, at both carve-outs."""

    def test_the_slow_configuration_is_advised(self):
        # 32 GiB allocation, 95.78 GiB visible RAM, a 42.90 GiB model: the weights
        # spill, and this is the configuration that ran 3-4x slower.
        got = _advice(gb(42.90), gb(32), gb(95.78), is_igpu = True)
        assert got is not None
        assert got["current_gb"] == 32.0
        assert got["needed_gb"] == 42.9
        # Smallest rung that holds the weights, not the largest the machine allows:
        # every GB suggested is a GB taken from the desktop.
        assert got["suggested_gb"] == 48
        assert got["machine_gb"] == 127.8

    def test_the_fast_configuration_is_silent(self):
        # Same model and machine after raising it to 96 GiB: 31.78 GiB visible RAM,
        # weights resident, nothing to say.
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
        # Whatever is suggested, a fifth of the machine (or 8 GB, whichever is
        # larger) stays with the OS.
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
        # __new__ so no real server, config or filesystem is involved: the method
        # under test only reads the stubs below and writes one attribute.
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
            staticmethod(lambda _i = None: carve_bytes),
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
        # The control is firmware on one machine and a driver panel on the next,
        # under different names. Naming both and neither specifically is as far as
        # this can honestly go. (On the machine this was developed against the
        # setting is in firmware and absent from the vendor's control panel, which
        # is exactly the trap this avoids.)
        msg = _message(_advice(gb(42.90), gb(32), gb(95.78), is_igpu = True))
        assert "firmware" in msg and "control panel" in msg
        # No menu path, key or control name is claimed.
        for invented in ("F10", "UMA Frame Buffer", "Variable Graphics Memory", "Advanced >"):
            assert invented not in msg

    def test_it_stays_short_enough_for_a_toast(self):
        # A toast is only harmless while it is small: sonner grows downwards over
        # whatever is under it, and a tall one takes those controls away for as long
        # as it is up (studio/frontend .../xet-progress-notice.ts records the same
        # constraint after #9293). The widest plausible reading is the bound, since a
        # 4-digit machine renders longer than the development one.
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
    """Swept rather than exampled. These caught three defects that the chosen
    examples above did not: a negative allocation read as truthy, a sub-1 GB
    allocation printing as "0 GB", and the integrated-GPU probe running on loads
    that could never produce advice."""

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
        # The property the whole feature rests on. An advisory that survives
        # being followed is a nag, and one that cannot be satisfied is a bug.
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
        # -1 is truthy, so a bare falsiness test carried it into the arithmetic
        # and produced confident wrong advice.
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
