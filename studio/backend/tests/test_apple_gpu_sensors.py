# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for Apple Silicon GPU sensors (SMC temperature + IOReport power)."""

import ctypes
import platform
import time

import pytest

from utils.hardware import apple

_IS_APPLE_SILICON = platform.system() == "Darwin" and platform.machine() == "arm64"


class TestFourcc:
    def test_roundtrip(self):
        for key in ("#KEY", "Tg0D", "flt "):
            assert apple._fourcc_str(apple._fourcc(key)) == key

    def test_known_value(self):
        # "flt " FourCC, same constant macmon uses.
        assert apple._fourcc("flt ") == 1718383648


class TestWatts:
    def test_millijoules(self):
        assert apple._watts(2000, "mJ", 2.0) == pytest.approx(1.0)

    def test_microjoules(self):
        assert apple._watts(5_000_000, "uJ", 1.0) == pytest.approx(5.0)

    def test_nanojoules(self):
        assert apple._watts(1_500_000_000, "nJ", 1.0) == pytest.approx(1.5)

    def test_unknown_unit_returns_none(self):
        assert apple._watts(1000, "J", 1.0) is None

    def test_zero_elapsed_returns_none(self):
        assert apple._watts(1000, "mJ", 0.0) is None


class TestAverageValidTemps:
    def test_averages_and_rounds(self):
        assert apple._average_valid_temps([40.0, 50.0, 60.05]) == 50.0

    def test_filters_invalid(self):
        assert apple._average_valid_temps([-1.0, 0.0, 151.0, 42.0]) == 42.0

    def test_empty_returns_none(self):
        assert apple._average_valid_temps([]) is None
        assert apple._average_valid_temps([0.0, 200.0]) is None


class TestSmcStructLayout:
    def test_key_data_matches_smc_protocol_size(self):
        # The AppleSMC user client rejects calls whose struct size differs.
        assert ctypes.sizeof(apple._SMCKeyData) == 80


@pytest.mark.skipif(not _IS_APPLE_SILICON, reason = "requires Apple Silicon")
class TestLiveSensors:
    def test_gpu_temperature_in_plausible_range(self):
        temp = apple.read_gpu_temperature_c()
        assert temp is not None
        assert 0.0 < temp <= 150.0

    def test_gpu_power_after_baseline(self):
        apple.read_gpu_power_w()  # first call only sets the baseline
        time.sleep(0.3)
        power = apple.read_gpu_power_w()
        assert power is not None
        assert power >= 0.0
