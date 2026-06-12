# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for _nvidia_classify_spark_unified_memory (Spark OOM-guard classifier).

Two paths: (1) ``is_integrated`` property (authoritative on native Linux),
(2) name-token match -- needed because WSL2 GPU paravirtualization masks
``is_integrated`` to 0 and renames the device (N1X reports ``JMJWOA-Generic-GPU``;
verified live). Mirrors test_rocm_oom_guard.py, which the NVIDIA guard models.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from core.training.worker import _nvidia_classify_spark_unified_memory


def _props(**kwargs) -> SimpleNamespace:
    """Fake device-properties object with the given attributes."""
    return SimpleNamespace(**kwargs)


# ── Path 1: is_integrated property ───────────────────────────────────────────


class TestIsIntegratedProperty:
    """``is_integrated`` truthy means unified memory, regardless of name."""

    def test_integrated_native_spark(self) -> None:
        props = _props(is_integrated = 1, name = "NVIDIA GB10")
        marker, is_unified = _nvidia_classify_spark_unified_memory(props)
        assert marker == "is_integrated"
        assert is_unified is True

    def test_integrated_wins_even_with_unknown_name(self) -> None:
        props = _props(is_integrated = 1, name = "Some Future Unified Part")
        marker, is_unified = _nvidia_classify_spark_unified_memory(props)
        assert marker == "is_integrated"
        assert is_unified is True


# ── Path 2: device-name token fallback (WSL masks is_integrated) ────────────


class TestDeviceNameTokenFallback:
    """is_integrated == 0 (or absent) -> classify by Spark name tokens."""

    @pytest.mark.parametrize(
        "name, expected_marker",
        [
            ("JMJWOA-Generic-GPU", "JMJWOA"),  # N1X under WSL2 (verified live)
            ("NVIDIA GB10", "GB10"),  # native DGX Spark
            ("NVIDIA GB110", "GB110"),  # "GB10" is not a substring of "GB110"
            ("NVIDIA DGX Spark", "DGX SPARK"),
            ("nvidia n1x prototype", "N1X"),  # case-insensitive
        ],
    )
    def test_spark_names_unified(self, name: str, expected_marker: str) -> None:
        props = _props(is_integrated = 0, name = name)
        marker, is_unified = _nvidia_classify_spark_unified_memory(props)
        assert is_unified is True
        assert marker == expected_marker

    @pytest.mark.parametrize(
        "name",
        [
            "NVIDIA GeForce RTX 4090",
            "NVIDIA H100 80GB HBM3",
            "NVIDIA RTX 6000 Ada Generation",
            "Tesla T4",
        ],
    )
    def test_discrete_names_not_unified(self, name: str) -> None:
        props = _props(is_integrated = 0, name = name)
        marker, is_unified = _nvidia_classify_spark_unified_memory(props)
        assert is_unified is False
        assert marker == ""

    def test_missing_attrs_defaults_discrete(self) -> None:
        """No is_integrated, no name -> discrete (guard stays off)."""
        marker, is_unified = _nvidia_classify_spark_unified_memory(_props())
        assert is_unified is False
        assert marker == ""

    def test_none_name_defaults_discrete(self) -> None:
        props = _props(is_integrated = 0, name = None)
        marker, is_unified = _nvidia_classify_spark_unified_memory(props)
        assert is_unified is False
        assert marker == ""
