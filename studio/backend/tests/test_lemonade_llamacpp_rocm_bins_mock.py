# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Validates that the installer correctly resolves lemonade ROCm prebuilt assets.

Uses a faked HostInfo so no AMD GPU is needed. Network calls to the lemonade
GitHub API are stubbed out so the suite runs without internet access and is
not subject to rate limits.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

_studio = Path(__file__).resolve().parent.parent.parent
if str(_studio) not in sys.path:
    sys.path.insert(0, str(_studio))

_mod = importlib.import_module("install_llama_prebuilt")
HostInfo = _mod.HostInfo
resolve_lemonade_rocm_choice = getattr(_mod, "resolve_lemonade_rocm_choice", None)
_LEMONADE_GFX_FAMILIES = getattr(_mod, "_LEMONADE_GFX_FAMILIES", None)

if resolve_lemonade_rocm_choice is None or _LEMONADE_GFX_FAMILIES is None:
    pytest.skip("PR symbols not present - check branch", allow_module_level = True)


@pytest.fixture(autouse = True)
def _clear_lemonade_release_cache():
    """Prevent cross-test pollution of the lemonade release lru_cache when
    future tests vary the fetch_json mock return value."""
    _cache = getattr(_mod, "_fetch_lemonade_release_cached", None)
    if _cache is not None and hasattr(_cache, "cache_clear"):
        _cache.cache_clear()
    yield
    if _cache is not None and hasattr(_cache, "cache_clear"):
        _cache.cache_clear()


_STUB_TAG = "b1262"
_STUB_OS_PREFIXES = ("ubuntu", "windows")
_STUB_FAMILIES = ("gfx1151", "gfx1150", "gfx120X", "gfx110X", "gfx103X")


def _stub_lemonade_release() -> dict:
    """Minimal lemonade release payload covering all supported GPU/OS combinations."""
    assets = [
        {
            "name": f"llama-{_STUB_TAG}-{prefix}-rocm-{family}-x64.zip",
            "browser_download_url": (
                f"https://github.com/lemonade-sdk/llamacpp-rocm/releases/download/"
                f"{_STUB_TAG}/llama-{_STUB_TAG}-{prefix}-rocm-{family}-x64.zip"
            ),
        }
        for prefix in _STUB_OS_PREFIXES
        for family in _STUB_FAMILIES
    ]
    return {"tag_name": _STUB_TAG, "assets": assets}


def _make_rocm_host(gfx_target: str, *, windows: bool = False) -> HostInfo:
    return HostInfo(
        system = "Windows" if windows else "Linux",
        machine = "amd64" if windows else "x86_64",
        is_windows = windows,
        is_linux = not windows,
        is_macos = False,
        is_x86_64 = True,
        is_arm64 = False,
        nvidia_smi = None,
        driver_cuda_version = None,
        compute_caps = [],
        visible_cuda_devices = None,
        has_physical_nvidia = False,
        has_usable_nvidia = False,
        has_rocm = True,
        rocm_gfx_target = gfx_target,
    )


def _lookup_family(gfx: str) -> str | None:
    for prefix, family in _LEMONADE_GFX_FAMILIES:
        if gfx.startswith(prefix):
            return family
    return None


# ---------------------------------------------------------------------------
# GPU family mapping
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "gfx,expected_family",
    [
        ("gfx1151", "gfx1151"),
        ("gfx1150", "gfx1150"),
        ("gfx1201", "gfx120X"),
        ("gfx1200", "gfx120X"),
        ("gfx1100", "gfx110X"),
        ("gfx1030", "gfx103X"),
    ],
)
def test_gpu_family_mapping(gfx, expected_family):
    assert _lookup_family(gfx) == expected_family


def test_unknown_gpu_not_in_families():
    assert _lookup_family("gfx999") is None


# ---------------------------------------------------------------------------
# Asset resolution - hits real lemonade GitHub API
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "gfx,os_prefix,windows",
    [
        ("gfx1151", "ubuntu", False),
        ("gfx1150", "ubuntu", False),
        ("gfx1201", "ubuntu", False),
        ("gfx1100", "ubuntu", False),
        ("gfx1030", "ubuntu", False),
        ("gfx1151", "windows", True),
        ("gfx1100", "windows", True),
    ],
)
def test_asset_resolves_for_known_gpu(gfx, os_prefix, windows):
    host = _make_rocm_host(gfx, windows = windows)
    with patch.object(_mod, "fetch_json", return_value = _stub_lemonade_release()):
        result = resolve_lemonade_rocm_choice(
            host, os_prefix, "default", llama_tag = "latest"
        )
    assert (
        result is not None
    ), f"Installer will NOT fetch lemonade binary for {gfx} ({os_prefix})"
    assert _lookup_family(gfx) in result.name
    assert result.url.startswith("https://github.com/lemonade-sdk/llamacpp-rocm")


def test_unknown_gpu_falls_through_to_upstream():
    host = _make_rocm_host("gfx999")
    result = resolve_lemonade_rocm_choice(host, "ubuntu", "default", llama_tag = "latest")
    assert result is None


# ---------------------------------------------------------------------------
# direct_linux_release_plan must plan a lemonade ROCm attempt for AMD-only hosts.
# This is the path setup.sh actually invokes, so the lemonade integration is
# useless if it isn't wired in here.
# ---------------------------------------------------------------------------

direct_linux_release_plan = getattr(_mod, "direct_linux_release_plan", None)
direct_upstream_release_plan = getattr(_mod, "direct_upstream_release_plan", None)


def _stub_unsloth_release(release_tag: str = "b9022") -> dict:
    # Minimal payload that parse_direct_linux_release_bundle accepts. It
    # requires at least one `app-{label}-linux-x64*.tar.gz` asset for the
    # bundle to be recognised; we ship a bare CPU one so the planner has a
    # baseline non-ROCm attempt to fall through to.
    asset_name = f"app-{release_tag}-linux-x64.tar.gz"
    return {
        "tag_name": release_tag,
        "name": release_tag,
        "assets": [
            {
                "name": asset_name,
                "browser_download_url": f"https://example.invalid/{asset_name}",
            },
        ],
    }


@pytest.mark.skipif(
    direct_linux_release_plan is None,
    reason = "direct release planners not present on this branch",
)
def test_direct_linux_plan_includes_lemonade_for_rocm_host():
    host = _make_rocm_host("gfx1151")
    with patch.object(_mod, "fetch_json", return_value = _stub_lemonade_release()):
        plan = direct_linux_release_plan(
            _stub_unsloth_release(),
            host,
            "unslothai/llama.cpp",
            "latest",
        )
    assert plan is not None, "ROCm host should not be skipped by the planner"
    kinds = [a.install_kind for a in plan.attempts]
    assert (
        "linux-rocm" in kinds
    ), f"planner did not include a lemonade ROCm attempt; got {kinds}"
    rocm_attempt = next(a for a in plan.attempts if a.install_kind == "linux-rocm")
    assert rocm_attempt.source_label == "lemonade"
    assert "gfx1151" in rocm_attempt.name


@pytest.mark.skipif(
    direct_upstream_release_plan is None,
    reason = "direct release planners not present on this branch",
)
def test_direct_upstream_plan_includes_lemonade_for_windows_hip_host():
    host = _make_rocm_host("gfx1151", windows = True)
    release = {
        "tag_name": "b9022",
        "name": "b9022",
        "assets": [],
    }
    with patch.object(_mod, "fetch_json", return_value = _stub_lemonade_release()):
        plan = direct_upstream_release_plan(
            release, host, "ggml-org/llama.cpp", "latest"
        )
    assert plan is not None, "Windows ROCm host should plan a lemonade HIP attempt"
    kinds = [a.install_kind for a in plan.attempts]
    assert (
        "windows-hip" in kinds
    ), f"planner did not include a lemonade HIP attempt; got {kinds}"


@pytest.mark.skipif(
    direct_upstream_release_plan is None,
    reason = "direct release planners not present on this branch",
)
def test_windows_hip_falls_back_to_upstream_when_lemonade_unavailable():
    """If lemonade returns None (e.g. gfx999 or transient API failure), the planner
    must still include the upstream HIP asset rather than silently downgrading to CPU."""
    host = _make_rocm_host("gfx999", windows = True)
    hip_asset = "llama-b9022-bin-win-hip-radeon-x64.zip"
    release = {
        "tag_name": "b9022",
        "name": "b9022",
        "assets": [
            {
                "name": hip_asset,
                "browser_download_url": f"https://example.invalid/{hip_asset}",
            },
        ],
    }
    plan = direct_upstream_release_plan(release, host, "ggml-org/llama.cpp", "latest")
    assert plan is not None
    kinds = [a.install_kind for a in plan.attempts]
    assert (
        "windows-hip" in kinds
    ), f"upstream HIP asset not included as fallback; got {kinds}"
    hip_attempt = next(a for a in plan.attempts if a.install_kind == "windows-hip")
    assert hip_attempt.source_label == "upstream"


# ── Follow-up: pinned-tag URL helper, URL trust pinning, opt-out env, autouse cache clear ──


def test_lemonade_release_api_url_pinned_tag():
    """A pinned llama_tag must produce the /releases/tags/<tag> URL."""
    assert _mod._lemonade_release_api_for("b1262").endswith("/releases/tags/b1262")
    assert _mod._lemonade_release_api_for("latest").endswith("/releases/latest")
    assert _mod._lemonade_release_api_for("").endswith("/releases/latest")


def test_lemonade_release_api_url_encodes_tag():
    """Unexpected slashes / hashes in the tag must be URL-encoded so the URL
    cannot be reshaped (defence in depth -- tags should already be sanitised
    upstream)."""
    url = _mod._lemonade_release_api_for("b1260/../latest")
    assert "/releases/tags/b1260%2F..%2Flatest" in url
    assert "//latest" not in url.split("/releases/tags/", 1)[1]


def test_lemonade_resolver_skipped_by_opt_out_env(monkeypatch):
    """UNSLOTH_DISABLE_LEMONADE_ROCM=1 must short-circuit the resolver."""
    monkeypatch.setenv("UNSLOTH_DISABLE_LEMONADE_ROCM", "1")
    host = _make_rocm_host("gfx1151")
    res = resolve_lemonade_rocm_choice(host, "ubuntu", "linux-rocm", llama_tag = "latest")
    assert res is None


def test_lemonade_resolver_rejects_non_github_url(monkeypatch):
    """If the GitHub API response somehow contained an off-host download URL,
    the resolver must refuse to use it (lemonade assets are not in the
    approved-hash manifest)."""
    bad_release = {
        "tag_name": _STUB_TAG,
        "assets": [
            {
                "name": f"llama-{_STUB_TAG}-ubuntu-rocm-gfx1151-x64.zip",
                "browser_download_url": "https://attacker.invalid/llama.zip",
            },
        ],
    }
    host = _make_rocm_host("gfx1151")
    with patch.object(_mod, "fetch_json", return_value = bad_release):
        res = resolve_lemonade_rocm_choice(
            host, "ubuntu", "linux-rocm", llama_tag = "latest"
        )
    assert res is None


def test_lemonade_resolver_rejects_http_scheme():
    assert not _mod._is_trusted_github_release_url(
        "http://github.com/lemonade-sdk/llamacpp-rocm/releases/download/x/y.zip",
        "lemonade-sdk/llamacpp-rocm",
    )


def test_lemonade_resolver_accepts_github_cdn():
    # Real GitHub release CDN URLs carry the /github-production-release-asset- prefix.
    assert _mod._is_trusted_github_release_url(
        "https://objects.githubusercontent.com/github-production-release-asset-abc123/456/789?token=x",
        "lemonade-sdk/llamacpp-rocm",
    )


def test_lemonade_resolver_rejects_arbitrary_cdn_path():
    # A CDN URL without the release-asset path prefix must be rejected.
    assert not _mod._is_trusted_github_release_url(
        "https://objects.githubusercontent.com/abc/def",
        "lemonade-sdk/llamacpp-rocm",
    )


def test_lemonade_resolver_accepts_release_path():
    url = "https://github.com/lemonade-sdk/llamacpp-rocm/releases/download/b1262/llama-b1262-ubuntu-rocm-gfx1151-x64.zip"
    assert _mod._is_trusted_github_release_url(url, "lemonade-sdk/llamacpp-rocm")


def test_lemonade_resolver_rejects_wrong_repo():
    """A github.com release URL for a different repo must be rejected."""
    assert not _mod._is_trusted_github_release_url(
        "https://github.com/attacker/llamacpp-rocm/releases/download/x/y.zip",
        "lemonade-sdk/llamacpp-rocm",
    )


def test_lemonade_resolver_rejects_empty_browser_download_url():
    """An asset entry with an empty browser_download_url must fall through."""
    release = {
        "tag_name": _STUB_TAG,
        "assets": [
            {
                "name": f"llama-{_STUB_TAG}-ubuntu-rocm-gfx1151-x64.zip",
                "browser_download_url": "",
            },
        ],
    }
    host = _make_rocm_host("gfx1151")
    with patch.object(_mod, "fetch_json", return_value = release):
        res = resolve_lemonade_rocm_choice(
            host, "ubuntu", "linux-rocm", llama_tag = "latest"
        )
    assert res is None


def test_lemonade_runtime_patterns_include_hip_runtime():
    """linux-rocm overlay must use a broad lib glob to catch all bundled .so files.

    Lemonade ZIPs carry transitive deps (libamd_comgr, libLLVM, libclang-cpp,
    ...) whose names change across ROCm releases. A broad ``lib*.so*`` glob
    avoids having to enumerate every transitive dependency by name.
    """
    from install_llama_prebuilt import runtime_patterns_for_choice, AssetChoice

    choice = AssetChoice(
        repo = "lemonade-sdk/llamacpp-rocm",
        tag = "b1262",
        name = "llama-b1262-ubuntu-rocm-gfx1151-x64.zip",
        url = "https://github.com/lemonade-sdk/llamacpp-rocm/releases/download/b1262/x.zip",
        source_label = "lemonade",
        install_kind = "linux-rocm",
    )
    pats = runtime_patterns_for_choice(choice)
    # The broad glob must be present so every .so in the lemonade bundle
    # (including transitive deps added in future ROCm releases) gets overlaid.
    assert "lib*.so*" in pats, f"'lib*.so*' missing from linux-rocm patterns: {pats}"


_pick_rocm_gfx_target = getattr(_mod, "_pick_rocm_gfx_target", None)


@pytest.mark.skipif(
    _pick_rocm_gfx_target is None,
    reason = "_pick_rocm_gfx_target not present on this branch",
)
def test_pick_rocm_gfx_target_honors_cuda_visible_devices(monkeypatch):
    """AMD HIP honours CUDA_VISIBLE_DEVICES identically to HIP_VISIBLE_DEVICES;
    on a gfx1151 + gfx1100 mixed host, CUDA_VISIBLE_DEVICES=1 must select gfx1100."""
    # Two GPUs; rocminfo reports each token twice (as in the real tool output).
    probe_out = "gfx1151\ngfx1151\ngfx1100\ngfx1100"
    monkeypatch.delenv("HIP_VISIBLE_DEVICES", raising = False)
    monkeypatch.delenv("ROCR_VISIBLE_DEVICES", raising = False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    assert _pick_rocm_gfx_target(probe_out) == "gfx1100"


@pytest.mark.skipif(
    _pick_rocm_gfx_target is None,
    reason = "_pick_rocm_gfx_target not present on this branch",
)
def test_pick_rocm_gfx_target_cuda_visible_devices_minus_one_returns_none(monkeypatch):
    """CUDA_VISIBLE_DEVICES=-1 means no GPU visible; resolver must return None."""
    probe_out = "gfx1151\ngfx1100"
    monkeypatch.delenv("HIP_VISIBLE_DEVICES", raising = False)
    monkeypatch.delenv("ROCR_VISIBLE_DEVICES", raising = False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")
    assert _pick_rocm_gfx_target(probe_out) is None


@pytest.mark.skipif(
    _pick_rocm_gfx_target is None,
    reason = "_pick_rocm_gfx_target not present on this branch",
)
def test_pick_rocm_gfx_target_same_arch_multi_gpu(monkeypatch):
    """Regression: [gfx1100, gfx1100, gfx1151] with HIP_VISIBLE_DEVICES=2 must
    return gfx1151, not fall back to GPU 0 due to dict.fromkeys collapsing the
    two gfx1100 entries into one and making index 2 out of range."""
    # Simulate rocminfo output for 3 GPUs (2x gfx1100 dGPU + 1x gfx1151 APU).
    # Each GPU gets its own Agent section with a few token mentions.
    probe_out = (
        "***\nAgent 1\n***\n  gfx1100 some info\n  gfx1100\n"
        "***\nAgent 2\n***\n  gfx1100 some info\n  gfx1100\n"
        "***\nAgent 3\n***\n  gfx1151 some info\n  gfx1151\n"
    )
    monkeypatch.delenv("ROCR_VISIBLE_DEVICES", raising = False)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising = False)
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "2")
    assert _pick_rocm_gfx_target(probe_out) == "gfx1151"
