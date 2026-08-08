# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Unit tests for the ROCm OOM guard: device classification and fraction selection.

Classification paths: (1) canonical gcnArchName, (2) alternate-spelling attr,
(3) all arch attrs absent -> device-name substring match.

Fraction selection: the unified-Linux reserve crossover, the discrete cap, the
Windows budget-exact 1.0, and the UNSLOTH_ROCM_MEM_FRACTION override.

Regression: Strix Halo (gfx1151) was misclassified as discrete on Radeon wheels
that set props.name="Radeon 8060S Graphics" but no gcnArchName, applying the
wrong headroom factor on a 128 GiB unified-memory pool.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from core.training.worker import (
    _DISCRETE_MEM_FRACTION,
    _UNIFIED_MAX_RESERVE_FRACTION,
    _UNIFIED_OS_RESERVE_BYTES,
    _parse_mem_fraction_env,
    _rocm_classify_unified_memory,
    _rocm_memory_fraction,
)

GIB = 1024**3

# Derived so that tuning the reserve retunes the tests with it: below this pool size the
# percentage arm wins, above it the byte arm does, and the two are equal exactly here.
_CROSSOVER_BYTES = int(_UNIFIED_OS_RESERVE_BYTES / _UNIFIED_MAX_RESERVE_FRACTION)
_HISTORICAL_CAP = 1.0 - _UNIFIED_MAX_RESERVE_FRACTION


# Past this the byte reserve is a smaller share than the discrete cap allows, so the
# clamp takes over and the reserve stops being the flat constant.
_CLAMP_BYTES = int(_UNIFIED_OS_RESERVE_BYTES / (1.0 - _DISCRETE_MEM_FRACTION))


def _expected_unified_fraction(total_bytes: int) -> float:
    """The reserve policy, restated independently of the implementation."""
    reserve = min(_UNIFIED_MAX_RESERVE_FRACTION, _UNIFIED_OS_RESERVE_BYTES / total_bytes)
    return min(1.0 - reserve, _DISCRETE_MEM_FRACTION)


# ── helpers ──────────────────────────────────────────────────────────────────


def _props(**kwargs) -> SimpleNamespace:
    """Build a fake device-properties object with the given attributes."""
    return SimpleNamespace(**kwargs)


# ── Path 0: props.is_integrated (driver's own unified-memory answer) ─────────


class TestIsIntegratedSignal:
    """hipDeviceProp_t.integrated wins when truthy; 0/absent never downgrades.

    Same universal gate PR #5988's UMA safetensors fast-load uses -- keeps
    Unsloth's two unified-memory consumers on one signal."""

    def test_integrated_upgrades_unknown_apu(self) -> None:
        # gfx1103 Phoenix iGPU: outside the hardcoded arch set, but the
        # driver says integrated -> unified.
        props = _props(gcnArchName = "gfx1103", name = "Radeon 780M", is_integrated = 1)
        gcn, is_unified = _rocm_classify_unified_memory(props)
        assert gcn == "gfx1103"
        assert is_unified is True

    def test_integrated_wins_without_any_arch(self) -> None:
        props = _props(name = "Some Future APU", is_integrated = 1)
        gcn, is_unified = _rocm_classify_unified_memory(props)
        assert gcn == ""
        assert is_unified is True

    def test_zero_does_not_downgrade_known_apu(self) -> None:
        # A wheel that zeroes the field must not flip Strix Halo to discrete.
        props = _props(gcnArchName = "gfx1151", name = "x", is_integrated = 0)
        gcn, is_unified = _rocm_classify_unified_memory(props)
        assert is_unified is True

    def test_absent_keeps_existing_behavior(self) -> None:
        props = _props(gcnArchName = "gfx1201", name = "RX 9070 XT")
        gcn, is_unified = _rocm_classify_unified_memory(props)
        assert is_unified is False

    def test_discrete_with_zero_stays_discrete(self) -> None:
        props = _props(gcnArchName = "gfx1100", name = "RX 7900 XTX", is_integrated = 0)
        gcn, is_unified = _rocm_classify_unified_memory(props)
        assert is_unified is False


# ── Path 1: canonical gcnArchName ────────────────────────────────────────────


class TestCanonicalGcnArchName:
    """gcnArchName is present and populated."""

    @pytest.mark.parametrize(
        "arch, expected_unified",
        [
            ("gfx1150", True),  # Strix Point
            ("gfx1151", True),  # Strix Halo
            ("gfx1152", True),  # Krackan Point (Radeon 860M/840M)
            ("gfx1100", False),  # Navi 31 (RX 7900 XTX) — discrete
            ("gfx906", False),  # MI50 — discrete server GPU
            ("gfx1201", False),  # RX 9070 XT — discrete
        ],
    )
    def test_canonical_attr(self, arch: str, expected_unified: bool) -> None:
        props = _props(gcnArchName = arch, name = "irrelevant")
        gcn, is_unified = _rocm_classify_unified_memory(props)
        assert gcn == arch
        assert is_unified is expected_unified

    def test_arch_with_colon_suffix_stripped(self) -> None:
        """gcnArchName can carry xnack/sramecc suffix; only the base is kept."""
        props = _props(gcnArchName = "gfx1151:xnack-", name = "irrelevant")
        gcn, is_unified = _rocm_classify_unified_memory(props)
        assert gcn == "gfx1151"
        assert is_unified is True

    def test_canonical_attr_wins_over_name(self) -> None:
        """Arch attr takes priority; device name is ignored."""
        # Discrete arch, but name looks like a unified SKU — arch must win.
        props = _props(gcnArchName = "gfx1100", name = "Radeon 890M")
        gcn, is_unified = _rocm_classify_unified_memory(props)
        assert gcn == "gfx1100"
        assert is_unified is False


# ── Path 2: alternate-spelling fallback ──────────────────────────────────────


class TestAlternateSpellingFallback:
    """gcnArchName is missing but an alternate attr spelling is present."""

    @pytest.mark.parametrize(
        "attr_name",
        ["gcn_arch_name", "arch_name", "gfx_arch_name"],
    )
    def test_alternate_attr_unified(self, attr_name: str) -> None:
        props = _props(**{attr_name: "gfx1151"}, name = "Radeon 8060S Graphics")
        gcn, is_unified = _rocm_classify_unified_memory(props)
        assert gcn == "gfx1151"
        assert is_unified is True

    @pytest.mark.parametrize(
        "attr_name",
        ["gcn_arch_name", "arch_name", "gfx_arch_name"],
    )
    def test_alternate_attr_discrete(self, attr_name: str) -> None:
        props = _props(**{attr_name: "gfx1201"}, name = "Radeon RX 9070 XT")
        gcn, is_unified = _rocm_classify_unified_memory(props)
        assert gcn == "gfx1201"
        assert is_unified is False

    def test_first_non_empty_attr_wins(self) -> None:
        """With multiple alternate attrs, the first non-empty one wins."""
        props = _props(gcn_arch_name = "gfx1151", arch_name = "gfx1100", name = "irrelevant")
        gcn, is_unified = _rocm_classify_unified_memory(props)
        assert gcn == "gfx1151"
        assert is_unified is True


# ── Path 3: device-name fallback ─────────────────────────────────────────────


class TestDeviceNameFallback:
    """ALL arch attrs absent — classifier must rely solely on device name."""

    # --- unified-memory devices that MUST be detected ---

    @pytest.mark.parametrize(
        "device_name",
        [
            # gfx1150 Strix Point
            "Radeon 890M",
            "AMD Radeon 890M Graphics",
            "RADEON 890M",  # case-insensitive
            "Radeon 880M",
            "AMD Radeon 880M Graphics",
            # gfx1151 Strix Halo — the regression case from the review
            "Radeon 8060S Graphics",  # Ryzen AI MAX+ 395 (as returned by torch)
            "AMD Radeon 8060S",
            "Radeon 8050S Graphics",  # cut-down Strix Halo SKU
            "AMD Radeon 8050S",
            # gfx1151 Gorgon Halo (Ryzen AI Max 400 refresh)
            "Radeon 8065S Graphics",  # Ryzen AI Max+ 495
            "AMD Radeon 8065S",
            # gfx1152 Krackan Point (Ryzen AI 7 350 / AI 5 340)
            "Radeon 860M",
            "AMD Radeon 860M Graphics",
            "Radeon 840M",
            "AMD Radeon 840M Graphics",
            # case variants
            "RADEON 8060S GRAPHICS",
            "radeon 8050s",
            "RADEON 860M",
        ],
    )
    def test_unified_memory_detected(self, device_name: str) -> None:
        props = _props(name = device_name)
        gcn, is_unified = _rocm_classify_unified_memory(props)
        assert gcn == "", f"expected empty gcn_arch, got {gcn!r}"
        assert is_unified is True, f"device {device_name!r} should be classified as unified-memory"

    # --- discrete devices that must NOT be mis-classified ---

    @pytest.mark.parametrize(
        "device_name",
        [
            "Radeon RX 9070 XT",
            "AMD Radeon RX 7900 XTX",
            "Radeon RX 6900 XT",
            "Radeon Pro W7900",
            "AMD Instinct MI300X",
            # Superficially similar substrings but discrete
            "Radeon RX 580",
            "Radeon VII",
        ],
    )
    def test_discrete_not_misclassified(self, device_name: str) -> None:
        props = _props(name = device_name)
        gcn, is_unified = _rocm_classify_unified_memory(props)
        assert gcn == ""
        assert (
            is_unified is False
        ), f"discrete device {device_name!r} should NOT be classified as unified-memory"

    def test_empty_name_returns_false(self) -> None:
        """Absent name must not crash and must default to discrete."""
        props = _props()  # no 'name' attr at all
        gcn, is_unified = _rocm_classify_unified_memory(props)
        assert gcn == ""
        assert is_unified is False

    def test_none_name_returns_false(self) -> None:
        props = _props(name = None)
        gcn, is_unified = _rocm_classify_unified_memory(props)
        assert gcn == ""
        assert is_unified is False


# ── Fraction selection ───────────────────────────────────────────────────────


_WORKER_PY = Path(__file__).resolve().parents[1] / "core" / "training" / "worker.py"


class TestMemFractionSelection:
    """Pin the per-platform fraction policy (_rocm_memory_fraction).

    On native Windows, torch.cuda.mem_get_info's total is the WDDM budget
    the driver grants HIP -- the OS share of RAM is already outside it, so
    a sub-1.0 cap double-taxes (field report: 48.49 GiB budget -> '38.79 GiB
    allowed' OOM denying a 47.29 GiB load that fit in free memory). 1.0
    removes the double-tax; current AMD Windows wheels enforce only
    sub-1.0 fractions, so it behaves like torch's uncapped default with
    WDDM arbitrating residency (measured on gfx1151)."""

    def test_unified_win32_uses_budget_exact_fraction(self) -> None:
        assert _rocm_memory_fraction(128 * GIB, True, "win32") == 1.0

    def test_discrete_keeps_its_cap_at_every_size(self) -> None:
        assert _rocm_memory_fraction(24 * GIB, False, "linux") == _DISCRETE_MEM_FRACTION
        assert _rocm_memory_fraction(128 * GIB, False, "linux") == _DISCRETE_MEM_FRACTION

    def test_win32_unified_logs_vgm_hint(self) -> None:
        """Users must learn the WDDM budget is raisable (BIOS UMA / AMD
        Software Variable Graphics Memory) instead of assuming a bug."""
        source = _WORKER_PY.read_text(encoding = "utf-8")
        assert "Variable Graphics Memory" in source

    def test_guard_delegates_to_the_fraction_helper(self) -> None:
        """Section 1g only runs behind _hw.IS_ROCM, which no CI machine satisfies, so its
        wiring has no other coverage. Pins the three things this PR's review turned on:
        the guard calls the helper, sizes against the allocator's own total, and tags the
        log off the parsed override rather than the raw string."""
        source = _WORKER_PY.read_text(encoding = "utf-8")
        assert "_mem_fraction = _rocm_memory_fraction(" in source
        assert "_torch_mem.cuda.mem_get_info(0)[1]" in source
        assert "if _env_fraction is not None" in source


class TestUnifiedLinuxReserve:
    """Linux unified pools reserve a bounded amount, not a flat 20%.

    A flat 0.80 withheld ~25 GiB on a 128 GiB Strix Halo. The reserve is
    min(20% of total, 16 GiB), so the 20% arm still wins below the 80 GiB
    crossover and those hosts keep exactly the historical cap."""

    @pytest.mark.parametrize("share_of_crossover", [0.1, 0.2, 0.4, 0.8, 1.0])
    def test_at_or_below_crossover_is_unchanged(self, share_of_crossover: float) -> None:
        # Exact, not approx: the helper is solved in fraction space precisely so this
        # stays bit-identical to the historical cap, and approx(0.80) would accept the
        # 0.7999999999999999 that the bytes-space arithmetic would have produced.
        total = int(_CROSSOVER_BYTES * share_of_crossover)
        assert _rocm_memory_fraction(total, True, "linux") == _HISTORICAL_CAP

    @pytest.mark.parametrize("multiple_of_crossover", [1.2, 1.6, 2.0])
    def test_large_pool_reserves_the_constant_not_a_percentage(
        self, multiple_of_crossover: float
    ) -> None:
        # Between the crossover and the clamp the reserve is the flat byte constant.
        total = int(_CROSSOVER_BYTES * multiple_of_crossover)
        assert total <= _CLAMP_BYTES, "sized past the clamp; the flat reserve no longer holds"
        fraction = _rocm_memory_fraction(total, True, "linux")
        assert total - fraction * total == pytest.approx(_UNIFIED_OS_RESERVE_BYTES)
        assert fraction == pytest.approx(_expected_unified_fraction(total))

    @pytest.mark.parametrize("pool_gib", [96, 128, 156, 160, 161, 256, 1024])
    def test_never_exceeds_the_090_recorded_as_starving(self, pool_gib: int) -> None:
        # 0.90 stays a literal on purpose: it is a field measurement of where a 128 GiB
        # pool starved the OS, not a policy constant. Deriving it from the reserve would
        # make this pass by construction and stop it catching an over-tuned reserve.
        # Sizes past 160 GiB are the clamp's job: without it the byte reserve falls under
        # 10% of the pool and a unified host would outrank a discrete card.
        assert _rocm_memory_fraction(pool_gib * GIB, True, "linux") <= 0.90

    def test_huge_pools_clamp_to_the_discrete_cap(self) -> None:
        # Above the clamp the reserve is no longer the flat constant, by design.
        assert _rocm_memory_fraction(1024 * GIB, True, "linux") == _DISCRETE_MEM_FRACTION

    def test_fraction_is_monotonic_and_floored_at_the_historical_cap(self) -> None:
        seen = [_rocm_memory_fraction(g * GIB, True, "linux") for g in range(4, 260, 4)]
        assert all(f >= _HISTORICAL_CAP for f in seen)
        assert seen == sorted(seen)

    def test_missing_total_falls_back_to_historical_cap(self) -> None:
        # The guard defaults an absent total to 0; must not divide by zero. Exact for the
        # same reason as the crossover assertions.
        assert _rocm_memory_fraction(0, True, "linux") == _HISTORICAL_CAP
        assert _rocm_memory_fraction(-1, True, "linux") == _HISTORICAL_CAP


class TestMemFractionEnvOverride:
    """UNSLOTH_ROCM_MEM_FRACTION is the escape hatch for hosts the formula
    gets wrong. Bad values are ignored, never fatal -- a typo in an env var
    must not take down a training run."""

    def test_override_wins_over_computed(self) -> None:
        assert _rocm_memory_fraction(128 * GIB, True, "linux", "0.95") == 0.95

    def test_override_applies_to_discrete_and_win32(self) -> None:
        assert _rocm_memory_fraction(24 * GIB, False, "linux", "0.5") == 0.5
        assert _rocm_memory_fraction(128 * GIB, True, "win32", "0.75") == 0.75

    @pytest.mark.parametrize("bad", ["", "abc", "0", "0.0", "-0.5", "1.5", "  ", "nan", "inf"])
    def test_unusable_values_fall_through(self, bad: str) -> None:
        total = 8 * _CROSSOVER_BYTES
        assert _rocm_memory_fraction(total, True, "linux", bad) == pytest.approx(
            _expected_unified_fraction(total)
        )

    def test_one_is_accepted(self) -> None:
        assert _rocm_memory_fraction(128 * GIB, True, "linux", "1.0") == 1.0


class TestParseMemFractionEnv:
    """The guard's log line tags the fraction 'from <env>' off this parse, so a
    rejected value has to be indistinguishable from an unset one -- otherwise the
    log credits an override the user never got."""

    @pytest.mark.parametrize("raw, expected", [("0.95", 0.95), ("1.0", 1.0), (" 0.5 ", 0.5)])
    def test_usable_values_parse(self, raw: str, expected: float) -> None:
        assert _parse_mem_fraction_env(raw) == pytest.approx(expected)

    @pytest.mark.parametrize("raw", [None, "", "  ", "abc", "O.95", "0", "0.0", "-0.5", "1.5"])
    def test_unusable_values_are_none(self, raw: str | None) -> None:
        # None is what makes the log say "computed" instead of naming the env var.
        assert _parse_mem_fraction_env(raw) is None

    @pytest.mark.parametrize("raw", ["nan", "NaN", "inf", "-inf", "Infinity"])
    def test_non_finite_values_are_rejected(self, raw: str) -> None:
        # float() accepts all of these. They are rejected only because every comparison
        # against NaN is False and inf falls outside the range -- rewriting the bound as
        # `if override <= 0.0 or override > 1.0` would let NaN through into
        # set_per_process_memory_fraction(), so pin it.
        assert _parse_mem_fraction_env(raw) is None

    @pytest.mark.parametrize("raw", ["abc", "1.5", "0"])
    def test_rejected_value_matches_the_unset_fraction(self, raw: str) -> None:
        # The tag and the fraction must agree: both fall back to the computed path.
        assert _rocm_memory_fraction(128 * GIB, True, "linux", raw) == _rocm_memory_fraction(
            128 * GIB, True, "linux", None
        )
