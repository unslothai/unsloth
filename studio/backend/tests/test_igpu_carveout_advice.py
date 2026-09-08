# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
"""Advice to enlarge an integrated GPU's dedicated memory.

Measured on a Ryzen AI Max+ PRO 395 (gfx1151, 128 GB): the same 42.90 GiB model
runs 3-4x faster with the weights inside the GPU allocation than spilling out of
it (decode 11.58 -> 46.70 t/s, prefill 150.88 -> 579.68 t/s on ROCm). That gap is
worth telling the user about, but only when raising the setting would actually
help, and only on hardware where the setting exists.
"""

import sys
from pathlib import Path

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
        assert "official documentation" in advice["message"]

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
        assert "128 GB" in msg  # the machine
        assert "80 GB" in msg  # what the host keeps: the trade-off, stated

    def test_it_sends_the_user_to_their_own_documentation(self):
        # The control is firmware on one machine and a driver panel on the next,
        # under different names. A confident wrong instruction costs more than a
        # pointer to the manufacturer. (On the machine this was developed against
        # the setting is in firmware and absent from the vendor's control panel,
        # which is exactly the trap this avoids.)
        msg = _message(_advice(gb(42.90), gb(32), gb(95.78), is_igpu = True))
        assert "official documentation" in msg
        assert "restart" in msg
        # No menu path, key or control name is claimed.
        for invented in ("F10", "UMA Frame Buffer", "Variable Graphics Memory", "Advanced >"):
            assert invented not in msg

    def test_it_names_no_vendor(self):
        msg = _message(_advice(gb(42.90), gb(32), gb(95.78), is_igpu = True))
        for vendor in ("AMD", "Adrenalin", "Intel", "NVIDIA", "HP", "Ryzen", "Radeon"):
            assert vendor not in msg
