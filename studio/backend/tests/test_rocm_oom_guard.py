# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Unit tests for _rocm_classify_unified_memory (ROCm OOM-guard classifier).

Three paths: (1) canonical gcnArchName, (2) alternate-spelling attr, (3) all
arch attrs absent -> device-name substring match.

Regression: Strix Halo (gfx1151) was misclassified as discrete on Radeon wheels
that set props.name="Radeon 8060S Graphics" but no gcnArchName, applying the
wrong headroom factor on a 128 GiB unified-memory pool.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from core.training.worker import _rocm_classify_unified_memory


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
            # case variants
            "RADEON 8060S GRAPHICS",
            "radeon 8050s",
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


# ── Fraction selection (source-pinned) ───────────────────────────────────────


_WORKER_PY = Path(__file__).resolve().parents[1] / "core" / "training" / "worker.py"


class TestMemFractionSelection:
    """Pin the per-platform fraction policy in worker.py section 1g.

    On native Windows, torch.cuda.mem_get_info's total is the WDDM budget
    the driver grants HIP -- the OS share of RAM is already outside it, so
    a 0.80 cap double-taxes (field report: 48.49 GiB budget -> '38.79 GiB
    allowed' OOM denying a 47.29 GiB load that fit in free memory). 1.0
    removes the double-tax; current AMD Windows wheels enforce only
    sub-1.0 fractions, so it behaves like torch's uncapped default with
    WDDM arbitrating residency (measured on gfx1151)."""

    def test_unified_win32_uses_budget_exact_fraction(self) -> None:
        source = _WORKER_PY.read_text(encoding = "utf-8")
        assert '1.0 if sys.platform == "win32" else 0.80' in source

    def test_discrete_keeps_090(self) -> None:
        source = _WORKER_PY.read_text(encoding = "utf-8")
        assert "_mem_fraction = 0.90" in source

    def test_win32_unified_logs_vgm_hint(self) -> None:
        """Users must learn the WDDM budget is raisable (BIOS UMA / AMD
        Software Variable Graphics Memory) instead of assuming a bug."""
        source = _WORKER_PY.read_text(encoding = "utf-8")
        assert "Variable Graphics Memory" in source
