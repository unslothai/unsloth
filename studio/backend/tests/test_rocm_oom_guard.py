# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Unit tests for _rocm_classify_unified_memory (ROCm OOM-guard classifier).

Covers the three classification paths:
  Path 1 – canonical gcnArchName attribute present.
  Path 2 – gcnArchName absent, alternate-spelling attribute present.
  Path 3 – ALL arch attrs absent; falls back to device-name substring match.

Regression for: Strix Halo (gfx1151) misclassified as discrete on AMD SDK /
Radeon wheels that populate props.name = "Radeon 8060S Graphics" but do NOT
set any gcnArchName attribute.  Without the 8060s/8050s name patterns the
fallback returned is_unified=False, applying the 0.90 fraction instead of
0.80 and leaving only ~12.8 GiB OS headroom on a 128 GiB unified-memory pool.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from core.training.worker import _rocm_classify_unified_memory


# ── helpers ──────────────────────────────────────────────────────────────────


def _props(**kwargs) -> SimpleNamespace:
    """Build a fake device-properties object with the given attributes."""
    return SimpleNamespace(**kwargs)


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
        """Arch attr takes priority; device name should be ignored."""
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
        """When multiple alternate attrs are present the first non-empty one wins."""
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
            # case variants
            "RADEON 8060S GRAPHICS",
            "radeon 8050s",
        ],
    )
    def test_unified_memory_detected(self, device_name: str) -> None:
        props = _props(name = device_name)
        gcn, is_unified = _rocm_classify_unified_memory(props)
        assert gcn == "", f"expected empty gcn_arch, got {gcn!r}"
        assert (
            is_unified is True
        ), f"device {device_name!r} should be classified as unified-memory"

    # --- discrete devices that must NOT be mis-classified ---

    @pytest.mark.parametrize(
        "device_name",
        [
            "Radeon RX 9070 XT",
            "AMD Radeon RX 7900 XTX",
            "Radeon RX 6900 XT",
            "Radeon Pro W7900",
            "AMD Instinct MI300X",
            # Names that contain superficially similar substrings but are discrete
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
        """Completely absent name must not crash and must default to discrete."""
        props = _props()  # no 'name' attr at all
        gcn, is_unified = _rocm_classify_unified_memory(props)
        assert gcn == ""
        assert is_unified is False

    def test_none_name_returns_false(self) -> None:
        props = _props(name = None)
        gcn, is_unified = _rocm_classify_unified_memory(props)
        assert gcn == ""
        assert is_unified is False
