# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for the llama.cpp prebuilt freshness check.

Pins the marker parser, disk+memory cache, stale-decision matrix, and
fail-open behaviour on missing data.
"""

from __future__ import annotations

import json
import os
import sys
import time
import types as _types
from datetime import datetime, timedelta, timezone
from pathlib import Path

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)


class _NoopLogger:
    """structlog-style logger: every method swallows positional + kwargs.

    A stdlib logging.Logger rejects structlog's keyword fields (e.g.
    ``logger.warning(msg, error=...)``), which leaked into the update module's
    error path and failed only when this file's stub loaded first.
    """

    def __getattr__(self, _name):
        return lambda *a, **k: None


_loggers_stub = _types.ModuleType("loggers")
_loggers_stub.get_logger = lambda *a, **k: _NoopLogger()
sys.modules.setdefault("loggers", _loggers_stub)

_structlog_stub = _types.ModuleType("structlog")
_structlog_stub.get_logger = lambda *a, **k: _NoopLogger()
sys.modules.setdefault("structlog", _structlog_stub)

import pytest

from utils import llama_cpp_freshness as fr


# Helpers.


def _write_marker(install_dir: Path, **overrides) -> Path:
    payload = {
        "requested_tag": "latest",
        "tag": "b9190",
        "release_tag": "b9190",
        "published_repo": "unslothai/llama.cpp",
        "asset": "app-b9190-linux-x64-cuda13-newer.tar.gz",
        "asset_sha256": None,
        "source": "published",
        "installed_at_utc": (datetime.now(tz = timezone.utc) - timedelta(days = 1))
        .isoformat()
        .replace("+00:00", "Z"),
    }
    payload.update(overrides)
    # The installer always writes `tag` and `release_tag` from the same release
    # (a normalized base vs the full release tag), so keep the pair consistent
    # when a test overrides only `tag`.
    if "tag" in overrides and "release_tag" not in overrides:
        payload["release_tag"] = overrides["tag"]
    install_dir.mkdir(parents = True, exist_ok = True)
    (install_dir / "UNSLOTH_PREBUILT_INFO.json").write_text(json.dumps(payload))
    return install_dir / "UNSLOTH_PREBUILT_INFO.json"


def _fake_binary(install_dir: Path, *, layout: str = "cmake") -> Path:
    """Stub llama-server under a supported install layout."""
    if layout == "cmake":
        bin_dir = install_dir / "build" / "bin"
        bin_name = "llama-server"
    elif layout == "root":
        bin_dir = install_dir
        bin_name = "llama-server"
    elif layout == "windows":
        bin_dir = install_dir / "build" / "bin" / "Release"
        bin_name = "llama-server.exe"
    else:
        raise ValueError(f"unknown layout {layout}")
    bin_dir.mkdir(parents = True, exist_ok = True)
    bin_path = bin_dir / bin_name
    bin_path.write_text("stub\n")
    return bin_path


@pytest.fixture(autouse = True)
def _reset(monkeypatch, tmp_path):
    # Isolate disk cache per-test; never touch the real cache.
    monkeypatch.setattr(fr, "_cache_dir", lambda: tmp_path / ".freshness")
    fr.reset_caches()
    yield
    fr.reset_caches()


# read_install_marker.


def test_read_install_marker_finds_cmake_layout(tmp_path):
    install_dir = tmp_path / "llama.cpp"
    _write_marker(install_dir, tag = "b9190")
    bin_path = _fake_binary(install_dir, layout = "cmake")
    marker = fr.read_install_marker(str(bin_path))
    assert marker is not None
    assert marker["tag"] == "b9190"
    assert marker["published_repo"] == "unslothai/llama.cpp"


def test_read_install_marker_finds_root_layout(tmp_path):
    install_dir = tmp_path / "llama.cpp"
    _write_marker(install_dir, tag = "b9999")
    bin_path = _fake_binary(install_dir, layout = "root")
    marker = fr.read_install_marker(str(bin_path))
    assert marker is not None
    assert marker["tag"] == "b9999"


def test_read_install_marker_finds_windows_cmake_layout(tmp_path):
    # Windows cmake puts the .exe under build/bin/Release/, so the marker
    # is four levels above the binary.
    install_dir = tmp_path / "llama.cpp"
    _write_marker(install_dir, tag = "b8888")
    bin_path = _fake_binary(install_dir, layout = "windows")
    marker = fr.read_install_marker(str(bin_path))
    assert marker is not None
    assert marker["tag"] == "b8888"


@pytest.mark.parametrize("repo", ["unslothai/llama.cpp", "ggml-org/llama.cpp"])
def test_read_install_marker_carries_published_repo_dynamically(tmp_path, repo):
    # The freshness check queries whichever release repo the marker records:
    # new installs record the fork, legacy CPU/macOS markers still say ggml-org,
    # and both must get the right "latest" tag.
    install_dir = tmp_path / "llama.cpp"
    _write_marker(install_dir, tag = "b9000", published_repo = repo)
    bin_path = _fake_binary(install_dir, layout = "cmake")
    marker = fr.read_install_marker(str(bin_path))
    assert marker is not None
    assert marker["published_repo"] == repo


def test_read_install_marker_missing_returns_none(tmp_path):
    bin_path = _fake_binary(tmp_path / "no_marker", layout = "root")
    assert fr.read_install_marker(str(bin_path)) is None


def test_read_install_marker_handles_invalid_json(tmp_path):
    install_dir = tmp_path / "llama.cpp"
    install_dir.mkdir(parents = True)
    (install_dir / "UNSLOTH_PREBUILT_INFO.json").write_text("not json")
    bin_path = _fake_binary(install_dir, layout = "root")
    assert fr.read_install_marker(str(bin_path)) is None


def test_read_install_marker_handles_none_path():
    assert fr.read_install_marker(None) is None


# latest_published_release (with monkeypatched fetcher).


def test_latest_published_release_uses_disk_cache(monkeypatch):
    calls = []

    def _fake_fetch(repo, timeout = 5.0):
        calls.append(repo)
        return "b9999"

    monkeypatch.setattr(fr, "_fetch_latest_release_tag", _fake_fetch)
    first = fr.latest_published_release("unslothai/llama.cpp")
    second = fr.latest_published_release("unslothai/llama.cpp")
    assert first == "b9999"
    assert second == "b9999"
    # Memo + disk cache -> only one fetch.
    assert len(calls) == 1


def test_latest_published_release_returns_none_on_network_failure(monkeypatch):
    monkeypatch.setattr(fr, "_fetch_latest_release_tag", lambda repo, timeout = 5.0: None)
    assert fr.latest_published_release("unslothai/llama.cpp") is None


def test_latest_published_release_keeps_old_cache_on_transient_failure(monkeypatch, tmp_path):
    # Disk entry older than TTL + network fail -> return cached value.
    cache_dir = tmp_path / ".freshness"
    cache_dir.mkdir()
    cache_file = cache_dir / "unslothai__llama.cpp.json"
    yesterday = time.time() - 25 * 60 * 60  # > 24h
    cache_file.write_text(json.dumps({"fetched_at": yesterday, "latest_tag": "b9000"}))
    monkeypatch.setattr(fr, "_fetch_latest_release_tag", lambda repo, timeout = 5.0: None)
    assert fr.latest_published_release("unslothai/llama.cpp") == "b9000"


# check_prebuilt_freshness end-to-end.


def test_check_prebuilt_freshness_reports_stale_when_old_and_behind(monkeypatch, tmp_path):
    install_dir = tmp_path / "llama.cpp"
    _write_marker(
        install_dir,
        tag = "b9190",
        installed_at_utc = (datetime.now(tz = timezone.utc) - timedelta(days = 5))
        .isoformat()
        .replace("+00:00", "Z"),
    )
    bin_path = _fake_binary(install_dir, layout = "root")
    monkeypatch.setattr(fr, "_fetch_latest_release_tag", lambda repo, timeout = 5.0: "b9300")
    info = fr.check_prebuilt_freshness(str(bin_path))
    assert info["has_marker"] is True
    assert info["stale"] is True
    assert info["installed_tag"] == "b9190"
    assert info["latest_tag"] == "b9300"
    assert info["age_days"] == 5
    assert info["published_repo"] == "unslothai/llama.cpp"


def test_check_prebuilt_freshness_not_stale_when_tag_matches(monkeypatch, tmp_path):
    install_dir = tmp_path / "llama.cpp"
    _write_marker(
        install_dir,
        tag = "b9300",
        installed_at_utc = (datetime.now(tz = timezone.utc) - timedelta(days = 30))
        .isoformat()
        .replace("+00:00", "Z"),
    )
    bin_path = _fake_binary(install_dir, layout = "root")
    monkeypatch.setattr(fr, "_fetch_latest_release_tag", lambda repo, timeout = 5.0: "b9300")
    info = fr.check_prebuilt_freshness(str(bin_path))
    assert info["stale"] is False
    assert info["installed_tag"] == "b9300"
    assert info["latest_tag"] == "b9300"


def test_check_prebuilt_freshness_not_stale_within_threshold(monkeypatch, tmp_path):
    # Behind by tag but within the 3-day grace window.
    install_dir = tmp_path / "llama.cpp"
    _write_marker(
        install_dir,
        tag = "b9190",
        installed_at_utc = (datetime.now(tz = timezone.utc) - timedelta(days = 1))
        .isoformat()
        .replace("+00:00", "Z"),
    )
    bin_path = _fake_binary(install_dir, layout = "root")
    monkeypatch.setattr(fr, "_fetch_latest_release_tag", lambda repo, timeout = 5.0: "b9300")
    info = fr.check_prebuilt_freshness(str(bin_path))
    assert info["stale"] is False
    assert info["age_days"] == 1


def test_check_prebuilt_freshness_fails_open_without_marker(tmp_path):
    bin_path = _fake_binary(tmp_path / "custom_build", layout = "root")
    info = fr.check_prebuilt_freshness(str(bin_path))
    assert info["has_marker"] is False
    assert info["stale"] is False


def test_check_prebuilt_freshness_fails_open_when_github_unreachable(monkeypatch, tmp_path):
    install_dir = tmp_path / "llama.cpp"
    _write_marker(
        install_dir,
        tag = "b9190",
        installed_at_utc = (datetime.now(tz = timezone.utc) - timedelta(days = 10))
        .isoformat()
        .replace("+00:00", "Z"),
    )
    bin_path = _fake_binary(install_dir, layout = "root")
    monkeypatch.setattr(fr, "_fetch_latest_release_tag", lambda repo, timeout = 5.0: None)
    info = fr.check_prebuilt_freshness(str(bin_path))
    assert info["has_marker"] is True
    assert info["stale"] is False
    assert info["latest_tag"] is None


def test_check_prebuilt_freshness_handles_unparseable_install_timestamp(monkeypatch, tmp_path):
    install_dir = tmp_path / "llama.cpp"
    _write_marker(install_dir, tag = "b9190", installed_at_utc = "not-a-date")
    bin_path = _fake_binary(install_dir, layout = "root")
    monkeypatch.setattr(fr, "_fetch_latest_release_tag", lambda repo, timeout = 5.0: "b9300")
    info = fr.check_prebuilt_freshness(str(bin_path))
    assert info["stale"] is False
    assert info["age_days"] is None


def test_check_prebuilt_freshness_respects_custom_threshold(monkeypatch, tmp_path):
    install_dir = tmp_path / "llama.cpp"
    _write_marker(
        install_dir,
        tag = "b9190",
        installed_at_utc = (datetime.now(tz = timezone.utc) - timedelta(days = 2))
        .isoformat()
        .replace("+00:00", "Z"),
    )
    bin_path = _fake_binary(install_dir, layout = "root")
    monkeypatch.setattr(fr, "_fetch_latest_release_tag", lambda repo, timeout = 5.0: "b9300")
    info = fr.check_prebuilt_freshness(str(bin_path), threshold_days = 1)
    assert info["stale"] is True


# format_stale_warning.


def test_format_stale_warning_contains_actionable_command():
    msg = fr.format_stale_warning({"installed_tag": "b9190", "latest_tag": "b9300", "age_days": 5})
    assert "b9190" in msg
    assert "b9300" in msg
    assert "5 days" in msg
    assert "unsloth studio update" in msg


def test_format_stale_warning_singular_day():
    msg = fr.format_stale_warning({"installed_tag": "b9190", "latest_tag": "b9300", "age_days": 1})
    assert "1 day" in msg
    assert "1 days" not in msg


# parse_base_build / is_behind.


def test_parse_base_build():
    assert fr.parse_base_build("b9596") == 9596
    assert fr.parse_base_build(" b9596 ") == 9596
    assert fr.parse_base_build("b9596-mix-e6f2453") == 9596  # mix suffix doesn't defeat it
    assert fr.parse_base_build("9596") is None
    assert fr.parse_base_build("master-abc") is None
    assert fr.parse_base_build("") is None
    assert fr.parse_base_build(None) is None


@pytest.mark.parametrize(
    "installed, latest, expected",
    [
        (
            "b9596-mix-e6f2453",
            "b9596-mix-e6f2453",
            False,
        ),  # already on the mix latest -> not behind
        ("b9596", "b9594", False),  # latest is an older build -> downgrade guard
        ("b9596", "b9594-mix-xxx", False),  # older mix latest -> still guarded
        ("b9500", "b9596-mix-e6f2453", True),  # newer base -> behind
        ("b9596-mix-aaa", "b9596-mix-bbb", True),  # new mix at same base -> behind
        ("b9596", "b9596-mix-bbb", True),  # clean -> mix at same base -> behind
        ("b9596-mix-aaa", "b9596", False),  # bare base never supersedes a mix install
        ("b9596", "b9596", False),  # identical -> not behind
        (" b9596 ", "b9596", False),  # whitespace-only diff -> not behind
        ("master-abc", "master-def", True),  # non-bNNNN both -> plain inequality
        ("master-abc", "master-abc", False),
        (None, "b9596", False),
        ("b9596", None, False),
    ],
)
def test_is_behind(installed, latest, expected):
    assert fr.is_behind(installed, latest) is expected


def test_check_prebuilt_freshness_not_behind_on_mix_latest(monkeypatch, tmp_path):
    # Installed the mix latest: marker base tag b9596, full release_tag with sha,
    # GitHub latest is that same full tag. Must not report behind (sticky bug).
    install_dir = tmp_path / "llama.cpp"
    _write_marker(install_dir, tag = "b9596", release_tag = "b9596-mix-e6f2453")
    bin_path = _fake_binary(install_dir, layout = "root")
    monkeypatch.setattr(
        fr, "_fetch_latest_release_tag", lambda repo, timeout = 5.0: "b9596-mix-e6f2453"
    )
    info = fr.check_prebuilt_freshness(str(bin_path))
    assert info["behind"] is False
    assert info["stale"] is False


def test_check_prebuilt_freshness_downgrade_guard(monkeypatch, tmp_path):
    # A lagging latest (older build than installed) must never read as behind/stale.
    install_dir = tmp_path / "llama.cpp"
    _write_marker(
        install_dir,
        tag = "b9585",
        installed_at_utc = (datetime.now(tz = timezone.utc) - timedelta(days = 30))
        .isoformat()
        .replace("+00:00", "Z"),
    )
    bin_path = _fake_binary(install_dir, layout = "root")
    monkeypatch.setattr(fr, "_fetch_latest_release_tag", lambda repo, timeout = 5.0: "b9518")
    info = fr.check_prebuilt_freshness(str(bin_path))
    assert info["behind"] is False
    assert info["stale"] is False


def test_fetch_latest_release_tag_uses_publish_time(monkeypatch):
    # Resolves newest by published_at (like the installer), skips drafts/prereleases,
    # and does NOT just take GitHub's first/`/releases/latest` item.
    import urllib.request

    class _Resp:
        def __init__(self, payload):
            self._p = json.dumps(payload).encode()

        def read(self):
            return self._p

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    payload = [
        {
            "tag_name": "b9518",
            "draft": False,
            "prerelease": False,
            "published_at": "2026-06-04T21:11:19Z",
        },
        {
            "tag_name": "b9596-mix-e6f2453",
            "draft": False,
            "prerelease": False,
            "published_at": "2026-06-11T22:50:41Z",
        },
        {
            "tag_name": "b9999-draft",
            "draft": True,
            "prerelease": False,
            "published_at": "2026-06-12T00:00:00Z",
        },
    ]
    monkeypatch.setattr(urllib.request, "urlopen", lambda req, timeout = 5.0: _Resp(payload))
    assert fr._fetch_latest_release_tag("unslothai/llama.cpp") == "b9596-mix-e6f2453"


# reset_caches(drop_disk=...) -- post-update stale same-base mix disk cache.


def _seed_disk_cache(tmp_path: Path, latest_tag: str) -> Path:
    # Matches _cache_path_for under the fixture's stubbed _cache_dir.
    cache_dir = tmp_path / ".freshness"
    cache_dir.mkdir(exist_ok = True)
    cache_file = cache_dir / "unslothai__llama.cpp.json"
    cache_file.write_text(json.dumps({"fetched_at": time.time(), "latest_tag": latest_tag}))
    return cache_file


def test_reset_caches_drop_disk_removes_disk_cache(tmp_path):
    cache_file = _seed_disk_cache(tmp_path, "b9596-mix-aaa")
    assert cache_file.exists()
    fr.reset_caches(drop_disk = True)
    assert not cache_file.exists()


def test_reset_caches_default_keeps_disk_cache(tmp_path):
    # The no-arg form is in-memory only (its existing test-only contract); it
    # must not delete the on-disk cache.
    cache_file = _seed_disk_cache(tmp_path, "b9596-mix-aaa")
    fr.reset_caches()
    assert cache_file.exists()


def test_reset_caches_drop_disk_on_missing_dir_is_noop(tmp_path):
    # Fresh machine, no cache dir yet: drop_disk must be a quiet no-op.
    assert not (tmp_path / ".freshness").exists()
    fr.reset_caches(drop_disk = True)  # must not raise


def test_drop_disk_lets_banner_fail_open_after_same_base_mix_swap(monkeypatch, tmp_path):
    # P2 #2: the disk cache holds a still-fresh same-base mix (b9596-mix-aaa)
    # from before an update to a *different* same-base mix (b9596-mix-bbb).
    # The post-install path drops the disk cache; if the forced refresh is then
    # offline, latest reads as None and the banner fails open -- instead of
    # replaying the stale b9596-mix-aaa and falsely reading "behind".
    _seed_disk_cache(tmp_path, "b9596-mix-aaa")
    install_dir = tmp_path / "llama.cpp"
    _write_marker(
        install_dir,
        tag = "b9596",
        release_tag = "b9596-mix-bbb",
        installed_at_utc = (datetime.now(tz = timezone.utc) - timedelta(days = 5))
        .isoformat()
        .replace("+00:00", "Z"),
    )
    bin_path = _fake_binary(install_dir, layout = "root")
    # GitHub unreachable for the rest of the test (the offline post-install
    # refresh, and the later status check).
    monkeypatch.setattr(fr, "_fetch_latest_release_tag", lambda repo, timeout = 5.0: None)

    fr.reset_caches(drop_disk = True)  # exactly what the apply path now does
    info = fr.check_prebuilt_freshness(str(bin_path))
    assert info["latest_tag"] is None
    assert info["behind"] is False
    assert info["stale"] is False


def test_in_memory_only_reset_replays_stale_same_base_mix(monkeypatch, tmp_path):
    # Contrast/guard for the case above: an in-memory-only reset leaves the
    # stale same-base mix on disk, so an offline check replays it and falsely
    # reads behind/stale. This is exactly the failure drop_disk removes; if a
    # future change makes the no-arg reset also clear disk, the apply-path call
    # and this guard should be revisited together.
    _seed_disk_cache(tmp_path, "b9596-mix-aaa")
    install_dir = tmp_path / "llama.cpp"
    _write_marker(
        install_dir,
        tag = "b9596",
        release_tag = "b9596-mix-bbb",
        installed_at_utc = (datetime.now(tz = timezone.utc) - timedelta(days = 5))
        .isoformat()
        .replace("+00:00", "Z"),
    )
    bin_path = _fake_binary(install_dir, layout = "root")
    monkeypatch.setattr(fr, "_fetch_latest_release_tag", lambda repo, timeout = 5.0: None)

    fr.reset_caches()  # in-memory only -> stale disk value survives
    info = fr.check_prebuilt_freshness(str(bin_path))
    assert info["latest_tag"] == "b9596-mix-aaa"
    assert info["behind"] is True
    assert info["stale"] is True


# update_download_size_bytes (banner download-size lookup).


def _patch_assets(monkeypatch, mapping):
    """Stub latest_release_assets with a per-repo {asset_name: size} lookup."""
    monkeypatch.setattr(
        fr,
        "latest_release_assets",
        lambda repo, *, force_refresh = False: mapping.get(repo),
    )


def test_update_size_unsloth_prebuilt_exact_match(monkeypatch):
    # The unsloth fork's own bundle (app-<tag>-<platform>): the want= exact match
    # on app-<latest>-<suffix> wins.
    marker = {
        "asset": "app-b9190-linux-x64-cuda13-newer.tar.gz",
        "published_repo": "unslothai/llama.cpp",
    }
    _patch_assets(
        monkeypatch,
        {
            "unslothai/llama.cpp": {
                "app-b9300-linux-x64-cuda13-newer.tar.gz": 123_456_789,
                "app-b9300-windows-x64-cuda13-newer.zip": 999,
            }
        },
    )
    assert fr.update_download_size_bytes(marker, "b9300", "unslothai/llama.cpp") == 123_456_789


def test_update_size_macos_fork_asset_suffix_fallback(monkeypatch):
    # macOS bundles use the upstream-style llama-<tag>-bin-macos-*, matched via the
    # endswith fallback in the publish repo.
    marker = {
        "asset": "llama-b9190-bin-macos-arm64.tar.gz",
        "published_repo": "unslothai/llama.cpp",
    }
    _patch_assets(
        monkeypatch,
        {"unslothai/llama.cpp": {"llama-b9300-bin-macos-arm64.tar.gz": 55_000_000}},
    )
    assert fr.update_download_size_bytes(marker, "b9300", "unslothai/llama.cpp") == 55_000_000


def test_update_size_upstream_ubuntu_uses_binary_repo(monkeypatch):
    # #6338 P2: ggml-org ubuntu-* prebuilt lives in binary_repo, not the fork
    # publish repo. The size must still resolve.
    marker = {
        "asset": "llama-b9190-bin-ubuntu-x64.tar.gz",
        "published_repo": "unslothai/llama.cpp",
        "binary_repo": "ggml-org/llama.cpp",
    }
    _patch_assets(
        monkeypatch,
        {
            "unslothai/llama.cpp": {"app-b9300-linux-x64-cuda13-newer.tar.gz": 1},
            "ggml-org/llama.cpp": {
                "llama-b9673-bin-ubuntu-x64.tar.gz": 42_000_000,
                "llama-b9673-bin-ubuntu-vulkan-x64.tar.gz": 7,
            },
        },
    )
    assert fr.update_download_size_bytes(marker, "b9300", "unslothai/llama.cpp") == 42_000_000


def test_update_size_upstream_windows_uses_binary_repo(monkeypatch):
    # Regression (#6338 P2): the Windows upstream CPU prebuilt uses a win-* token.
    marker = {
        "asset": "llama-b9190-bin-win-cpu-x64.zip",
        "published_repo": "unslothai/llama.cpp",
        "binary_repo": "ggml-org/llama.cpp",
    }
    _patch_assets(
        monkeypatch,
        {"ggml-org/llama.cpp": {"llama-b9673-bin-win-cpu-x64.zip": 33_000_000}},
    )
    assert fr.update_download_size_bytes(marker, "b9300", "unslothai/llama.cpp") == 33_000_000


def test_update_size_no_matching_asset_fails_open(monkeypatch):
    # A ROCm version drift (installed 6.4 vs latest 7.2) leaves no suffix match;
    # the helper fails open to None rather than guessing a wrong artifact.
    marker = {
        "asset": "llama-b9190-bin-ubuntu-rocm-6.4-x64.tar.gz",
        "published_repo": "unslothai/llama.cpp",
        "binary_repo": "ggml-org/llama.cpp",
    }
    _patch_assets(
        monkeypatch,
        {"ggml-org/llama.cpp": {"llama-b9673-bin-ubuntu-rocm-7.2-x64.tar.gz": 9}},
    )
    assert fr.update_download_size_bytes(marker, "b9300", "unslothai/llama.cpp") is None


def test_update_size_missing_inputs_fail_open(monkeypatch):
    _patch_assets(
        monkeypatch,
        {"unslothai/llama.cpp": {"app-b9300-linux-x64-cpu.tar.gz": 5}},
    )
    # No marker, no latest tag, or no asset string -> None (never raise).
    assert fr.update_download_size_bytes(None, "b9300", "unslothai/llama.cpp") is None
    assert (
        fr.update_download_size_bytes(
            {"asset": "app-b9190-linux-x64-cpu.tar.gz"}, None, "unslothai/llama.cpp"
        )
        is None
    )
    assert fr.update_download_size_bytes({"asset": None}, "b9300", "unslothai/llama.cpp") is None
