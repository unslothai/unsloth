# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Unit tests for the shared prebuilt-consumer selection core.

Pins the coverage-aware CUDA/ROCm selection primitives that both
install_whisper_prebuilt.py and (in Phase B) install_llama_prebuilt.py depend on:
SM membership + range coverage, tightest-range ranking, Blackwell runtime-line
override, torch preference, driver-based runtime lines, and exact ROCm matching.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_studio = Path(__file__).resolve().parent.parent.parent
if str(_studio) not in sys.path:
    sys.path.insert(0, str(_studio))

from backend.utils.prebuilt import selection as sel  # noqa: E402


def _art(
    asset,
    line = None,
    cls = None,
    sms = (),
    gfx = None,
    mapped = (),
):  # noqa: ANN001
    return sel.SelArtifact(
        asset = asset,
        os = "linux",
        arch = "x64",
        backend = "cuda" if line else "rocm",
        runtime_line = line,
        coverage_class = cls,
        supported_sms = tuple(sms),
        min_sm = int(sms[0]) if sms else None,
        max_sm = int(sms[-1]) if sms else None,
        gfx_target = gfx,
        mapped_targets = tuple(mapped),
    )


# ── compute-cap normalisation ──
def test_normalize_compute_caps():
    assert sel.normalize_compute_caps(["8.9", "10.0", "8.9"]) == ["89", "100"]
    assert sel.normalize_compute_caps(["9.0", "bogus", "7.5"]) == ["75", "90"]
    assert sel.normalize_compute_caps([]) == []
    # sorted ascending so caps[-1] is the highest SM
    assert sel.normalize_compute_caps(["12.0", "8.0"]) == ["80", "120"]


def test_normalize_compute_cap_edge_cases():
    assert sel.normalize_compute_cap("8.9") == "89"
    assert sel.normalize_compute_cap("90") == "90"
    assert sel.normalize_compute_cap("") is None
    assert sel.normalize_compute_cap("8.x") is None


# ── coverage primitives ──
def test_artifact_covers_sms():
    newer = _art("newer", "cuda13", "newer", ["86", "89", "90", "100", "103", "120"])
    assert sel.artifact_covers_sms(newer, ["100"]) is True
    assert sel.artifact_covers_sms(newer, ["80"]) is False  # not in supported_sms
    # missing metadata never covers
    assert sel.artifact_covers_sms(_art("x", "cuda13", "newer", []), ["90"]) is False


def test_sm_range_tighter_wins_and_missing_sorts_last():
    older = _art("older", "cuda13", "older", ["75", "80", "86", "89"])  # 75-89 -> 14
    newer = _art("newer", "cuda13", "newer", ["86", "89", "90", "100", "103", "120"])  # -> 34
    nometa = _art("nometa", "cuda13", "older", [])
    assert sel.sm_range(older) == 14
    assert sel.sm_range(newer) == 34
    assert sel.sm_range(nometa) == 9999


def test_host_is_blackwell_and_min_toolkit():
    assert sel.host_is_blackwell(["89"]) is False
    assert sel.host_is_blackwell(["100"]) is True
    assert sel.host_is_blackwell(["120"]) is True
    assert sel.blackwell_min_toolkit_for_caps(["100"]) == (12, 8)
    assert sel.blackwell_min_toolkit_for_caps(["103"]) == (12, 9)  # needs 12.9
    assert sel.blackwell_min_toolkit_for_caps(["89"]) == (12, 8)


def test_compatible_runtime_lines_for_driver():
    assert sel.compatible_runtime_lines_for_driver((13, 0)) == ["cuda13", "cuda12"]
    assert sel.compatible_runtime_lines_for_driver((12, 8)) == ["cuda12"]
    assert sel.compatible_runtime_lines_for_driver((11, 8)) == []  # below floor
    assert sel.compatible_runtime_lines_for_driver(None) == []


# ── ranking + selection ──
_ALL = [
    _art("cuda12-legacy", "cuda12", "legacy", ["50", "52", "60", "61"]),
    _art("cuda12-older", "cuda12", "older", ["70", "75", "80", "86", "89"]),
    _art("cuda12-newer", "cuda12", "newer", ["86", "89", "90", "100", "103", "120"]),
    _art(
        "cuda12-portable",
        "cuda12",
        "portable",
        ["70", "75", "80", "86", "89", "90", "100", "103", "120"],
    ),
    _art("cuda13-older", "cuda13", "older", ["75", "80", "86", "89"]),
    _art("cuda13-newer", "cuda13", "newer", ["86", "89", "90", "100", "103", "120"]),
    _art(
        "cuda13-portable", "cuda13", "portable", ["75", "80", "86", "89", "90", "100", "103", "120"]
    ),
]


def _pick(
    caps,
    driver,
    torch = None,
):  # noqa: ANN001
    log: list[str] = []
    attempts = sel.select_cuda_attempts(
        list(_ALL), sel.normalize_compute_caps(caps), driver, torch, log
    )
    return attempts[0].asset if attempts else None


def test_blackwell_override_prefers_native_cuda13():
    # B200 sm_100: Blackwell override puts cuda13 first even without torch.
    assert _pick(["10.0"], (13, 0)) == "cuda13-newer"


def test_torch_preference_moves_line_to_front_on_non_blackwell():
    # sm_89 with both lines available; torch=cuda12 should prefer a cuda12 bundle.
    assert _pick(["8.9"], (13, 0), torch = "cuda12") == "cuda12-older"
    # without torch, newest driver line wins -> cuda13-older (tightest covering 89)
    assert _pick(["8.9"], (13, 0)) == "cuda13-older"


def test_rank_prefers_tightest_then_keeps_portable_as_fallback():
    log: list[str] = []
    attempts = sel.select_cuda_attempts(
        list(_ALL), sel.normalize_compute_caps(["10.0"]), (13, 0), None, log
    )
    names = [a.asset for a in attempts]
    # best (newer, tight) first, portable as an explicit fallback, then cuda12 line
    assert names[0] == "cuda13-newer"
    assert "cuda13-portable" in names
    assert names.index("cuda13-newer") < names.index("cuda13-portable")


def test_targeted_artifact_missing_sm_metadata_is_rejected():
    bad = [_art("cuda13-x", "cuda13", "newer", [])]  # no supported_sms
    log: list[str] = []
    assert sel.select_cuda_attempts(bad, ["90"], (13, 0), None, log) == []


def test_unknown_caps_only_portable():
    log: list[str] = []
    attempts = sel.select_cuda_attempts(list(_ALL), [], (13, 0), None, log)
    assert attempts and all(a.coverage_class == "portable" for a in attempts)


# ── ROCm ──
def test_match_rocm_exact_mapped_and_family_token():
    cands = [
        _art("rocm-gfx110X", gfx = "gfx110X", mapped = ["gfx1100", "gfx1101", "gfx1102"]),
        _art("rocm-gfx90a", gfx = "gfx90a", mapped = ["gfx90a"]),
    ]
    log: list[str] = []
    assert sel.match_rocm_artifact(cands, "gfx1100", log).asset == "rocm-gfx110X"
    assert sel.match_rocm_artifact(cands, "gfx110X", log).asset == "rocm-gfx110X"  # family token
    assert sel.match_rocm_artifact(cands, "gfx90a", log).asset == "rocm-gfx90a"


def test_match_rocm_unbuilt_arch_returns_none():
    cands = [_art("rocm-gfx110X", gfx = "gfx110X", mapped = ["gfx1100", "gfx1101", "gfx1102"])]
    log: list[str] = []
    assert sel.match_rocm_artifact(cands, "gfx1103", log) is None
    assert sel.match_rocm_artifact(cands, None, log) is None
