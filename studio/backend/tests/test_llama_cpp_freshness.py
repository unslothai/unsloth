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
from utils.prebuilt import freshness_flow


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
    # The installer writes tag and release_tag from one release; keep them paired.
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
    monkeypatch.setattr(fr, "_cache_dir", lambda: tmp_path / ".freshness")
    fr.reset_caches()
    yield
    fr.reset_caches()


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
    # Windows cmake puts the .exe under build/bin/Release/ (marker 4 levels up).
    install_dir = tmp_path / "llama.cpp"
    _write_marker(install_dir, tag = "b8888")
    bin_path = _fake_binary(install_dir, layout = "windows")
    marker = fr.read_install_marker(str(bin_path))
    assert marker is not None
    assert marker["tag"] == "b8888"


@pytest.mark.parametrize("repo", ["unslothai/llama.cpp", "ggml-org/llama.cpp"])
def test_read_install_marker_carries_published_repo_dynamically(tmp_path, repo):
    # Markers may record the fork or legacy ggml-org; both must resolve latest.
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


def test_read_install_marker_handles_non_utf8(tmp_path):
    install_dir = tmp_path / "llama.cpp"
    install_dir.mkdir(parents = True)
    (install_dir / "UNSLOTH_PREBUILT_INFO.json").write_bytes(b'{"runtime_line": "\xff\xfecuda13"}')
    bin_path = _fake_binary(install_dir, layout = "root")
    assert fr.read_install_marker(str(bin_path)) is None


@pytest.mark.parametrize(
    "payload",
    ["[]", '["cpu"]', '"cuda"', "123", "true", "null"],
    ids = ["empty-list", "list", "string", "int", "bool", "null"],
)
def test_read_install_marker_rejects_non_object_json(tmp_path, payload):
    """JSON that parses but is not an object must read as "no marker".

    Every caller treats a non-None return as a mapping -- the update planner, the
    backend picker and crash recovery all reach straight for ``.get`` -- so a marker
    holding ``["cpu"]`` used to raise AttributeError out of a plain status read
    instead of degrading to the source-build path a corrupt file deserves.
    """
    install_dir = tmp_path / "llama.cpp"
    install_dir.mkdir(parents = True)
    (install_dir / "UNSLOTH_PREBUILT_INFO.json").write_text(payload)
    bin_path = _fake_binary(install_dir, layout = "root")
    assert fr.read_install_marker(str(bin_path)) is None


def test_read_install_marker_handles_none_path():
    assert fr.read_install_marker(None) is None


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
    assert len(calls) == 1


def test_latest_published_release_returns_none_on_network_failure(monkeypatch):
    calls = []

    def _failed_fetch(repo, timeout = 5.0):
        calls.append(repo)
        return None

    monkeypatch.setattr(fr, "_fetch_latest_release_tag", _failed_fetch)
    assert fr.latest_published_release("unslothai/llama.cpp") is None
    assert fr.latest_published_release("unslothai/llama.cpp") is None
    assert calls == ["unslothai/llama.cpp"]


def test_latest_published_release_retries_after_failure_ttl(monkeypatch):
    wall_now = [1000.0]
    monotonic_now = [100.0]
    calls = []

    monkeypatch.setattr(fr._flow.time, "time", lambda: wall_now[0])
    monkeypatch.setattr(fr._flow.time, "monotonic", lambda: monotonic_now[0])

    def _fetch(repo, timeout = 5.0):
        calls.append(repo)
        return None if len(calls) == 1 else "b9999"

    monkeypatch.setattr(fr, "_fetch_latest_release_tag", _fetch)
    assert fr.latest_published_release("unslothai/llama.cpp") is None
    assert fr.latest_published_release("unslothai/llama.cpp") is None

    # A wall-clock rollback must not extend the in-process failure TTL.
    wall_now[0] -= 500
    monotonic_now[0] += fr._flow.RELEASE_FAILURE_CACHE_TTL_SECONDS + 1
    assert fr.latest_published_release("unslothai/llama.cpp") == "b9999"
    assert calls == ["unslothai/llama.cpp", "unslothai/llama.cpp"]


def test_latest_published_release_force_refresh_bypasses_failure_ttl(monkeypatch):
    calls = []

    def _fetch(repo, timeout = 5.0):
        calls.append(repo)
        return None if len(calls) == 1 else "b9999"

    monkeypatch.setattr(fr, "_fetch_latest_release_tag", _fetch)
    assert fr.latest_published_release("unslothai/llama.cpp") is None
    assert fr.latest_published_release("unslothai/llama.cpp", force_refresh = True) == "b9999"
    assert fr.latest_published_release("unslothai/llama.cpp") == "b9999"
    assert calls == ["unslothai/llama.cpp", "unslothai/llama.cpp"]


def test_reset_caches_clears_release_failure_memo(monkeypatch):
    calls = []

    def _failed_fetch(repo, timeout = 5.0):
        calls.append(repo)
        return None

    monkeypatch.setattr(fr, "_fetch_latest_release_tag", _failed_fetch)
    assert fr.latest_published_release("unslothai/llama.cpp") is None
    assert fr.latest_published_release("unslothai/llama.cpp") is None

    fr.reset_caches()

    assert fr.latest_published_release("unslothai/llama.cpp") is None
    assert calls == ["unslothai/llama.cpp", "unslothai/llama.cpp"]


def test_latest_published_release_keeps_old_cache_on_transient_failure(monkeypatch, tmp_path):
    cache_dir = tmp_path / ".freshness"
    cache_dir.mkdir()
    cache_file = cache_dir / "unslothai__llama.cpp.json"
    yesterday = time.time() - 25 * 60 * 60
    cache_file.write_text(json.dumps({"fetched_at": yesterday, "latest_tag": "b9000"}))
    calls = []

    def _failed_fetch(repo, timeout = 5.0):
        calls.append(repo)
        return None

    monkeypatch.setattr(fr, "_fetch_latest_release_tag", _failed_fetch)
    assert fr.latest_published_release("unslothai/llama.cpp") == "b9000"
    assert fr.latest_published_release("unslothai/llama.cpp") == "b9000"
    assert calls == ["unslothai/llama.cpp"]


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


def test_check_prebuilt_freshness_skips_github_when_update_checks_disabled(monkeypatch, tmp_path):
    install_dir = tmp_path / "llama.cpp"
    _write_marker(install_dir, tag = "b9190")
    bin_path = _fake_binary(install_dir)

    def _fetch(repo, timeout = 5.0):
        raise AssertionError("fetched a release despite UNSLOTH_DISABLE_UPDATE_CHECK=1")

    monkeypatch.setattr(fr, "_fetch_latest_release_tag", _fetch)
    monkeypatch.setenv("UNSLOTH_DISABLE_UPDATE_CHECK", "1")
    info = fr.check_prebuilt_freshness(str(bin_path))
    assert info["has_marker"] is True
    assert info["installed_tag"] == "b9190"
    assert info["latest_tag"] is None
    assert info["behind"] is False
    assert info["stale"] is False


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


def test_parse_base_build():
    assert fr.parse_base_build("b9596") == 9596
    assert fr.parse_base_build(" b9596 ") == 9596
    assert fr.parse_base_build("b9596-mix-e6f2453") == 9596
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
        ),
        ("b9596", "b9594", False),  # latest is an older build -> downgrade guard
        ("b9596", "b9594-mix-xxx", False),
        ("b9500", "b9596-mix-e6f2453", True),
        ("b9596-mix-aaa", "b9596-mix-bbb", True),
        ("b9596", "b9596-mix-bbb", True),
        ("b9596-mix-aaa", "b9596", False),  # bare base never supersedes a mix install
        ("b9596", "b9596", False),
        (" b9596 ", "b9596", False),
        ("master-abc", "master-def", True),
        ("master-abc", "master-abc", False),
        (None, "b9596", False),
        ("b9596", None, False),
    ],
)
def test_is_behind(installed, latest, expected):
    assert fr.is_behind(installed, latest) is expected


def test_check_prebuilt_freshness_not_behind_on_mix_latest(monkeypatch, tmp_path):
    # Mix latest installed (base tag + full release_tag): must not read as behind.
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
    # Newest by published_at, skipping drafts/prereleases, not /releases/latest.
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
    monkeypatch.setattr(freshness_flow, "auth_safe_open", lambda req, timeout = 5.0: _Resp(payload))
    assert fr._fetch_latest_release_tag("unslothai/llama.cpp") == "b9596-mix-e6f2453"


def _seed_disk_cache(tmp_path: Path, latest_tag: str) -> Path:
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
    # No-arg reset is in-memory only and must not delete the disk cache.
    cache_file = _seed_disk_cache(tmp_path, "b9596-mix-aaa")
    fr.reset_caches()
    assert cache_file.exists()


def test_reset_caches_drop_disk_on_missing_dir_is_noop(tmp_path):
    assert not (tmp_path / ".freshness").exists()
    fr.reset_caches(drop_disk = True)


def test_drop_disk_lets_banner_fail_open_after_same_base_mix_swap(monkeypatch, tmp_path):
    # Offline after dropping disk cache: latest is None, so the banner fails open.
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

    fr.reset_caches(drop_disk = True)
    info = fr.check_prebuilt_freshness(str(bin_path))
    assert info["latest_tag"] is None
    assert info["behind"] is False
    assert info["stale"] is False


def test_in_memory_only_reset_replays_stale_same_base_mix(monkeypatch, tmp_path):
    # In-memory reset keeps the stale same-base mix on disk; drop_disk removes it.
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

    fr.reset_caches()
    info = fr.check_prebuilt_freshness(str(bin_path))
    assert info["latest_tag"] == "b9596-mix-aaa"
    assert info["behind"] is True
    assert info["stale"] is True


def _patch_assets(monkeypatch, mapping):
    """Stub latest_release_assets with a per-repo {asset_name: size} lookup."""
    monkeypatch.setattr(
        fr,
        "latest_release_assets",
        lambda repo, *, force_refresh = False: mapping.get(repo),
    )


def test_update_size_unsloth_prebuilt_exact_match(monkeypatch):
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
    # ROCm version drift leaves no suffix match: fail open to None.
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
    assert fr.update_download_size_bytes(None, "b9300", "unslothai/llama.cpp") is None
    assert (
        fr.update_download_size_bytes(
            {"asset": "app-b9190-linux-x64-cpu.tar.gz"}, None, "unslothai/llama.cpp"
        )
        is None
    )
    assert fr.update_download_size_bytes({"asset": None}, "b9300", "unslothai/llama.cpp") is None


@pytest.mark.parametrize("fetch", ["_fetch_latest_release_tag", "_fetch_latest_release_assets"])
def test_release_fetch_cannot_outlive_its_deadline(monkeypatch, fetch):
    """urllib applies its timeout per address, so /api/inference/status inherits that
    multiplication without a wall-clock deadline; one stalled connect stands in for the
    walk. Both entry points, since they share the fetch."""

    def _stalls(req, timeout = 5.0):
        time.sleep(30)
        raise AssertionError("deadline did not cut the fetch short")

    monkeypatch.setattr(freshness_flow, "auth_safe_open", _stalls)
    started = time.monotonic()
    assert getattr(fr, fetch)("unslothai/llama.cpp", timeout = 0.25) is None
    # Pins the implemented timeout + 1, not merely "faster than the 30s stall".
    assert time.monotonic() - started < 2.0


def _refusal(
    code: int,
    headers: dict | None = None,
    body: bytes = b"",
):
    import email.message
    import io
    import urllib.error

    message = email.message.Message()
    for name, value in (headers or {}).items():
        message[name] = value
    return urllib.error.HTTPError(
        "https://api.github.com/repos/x/y/releases", code, "refused", message, io.BytesIO(body)
    )


@pytest.mark.parametrize(
    ("code", "headers", "body", "low", "high"),
    [
        (
            403,
            {"X-RateLimit-Remaining": "4998"},
            b'{"message": "Resource not accessible"}',
            None,
            None,
        ),
        (404, {"X-RateLimit-Remaining": "0"}, b"", None, None),
        (403, {"Retry-After": "60"}, b"", 60, 60),
        (403, {"X-RateLimit-Remaining": "0", "X-RateLimit-Reset": "+42"}, b"", 35, 42),
        (403, {}, b'{"message": "You have exceeded a secondary rate limit"}', 900, 900),
        (429, {"X-RateLimit-Remaining": "4998"}, b"", 900, 900),
        (403, {"X-RateLimit-Remaining": "0", "X-RateLimit-Reset": "+21600"}, b"", 3600, 3600),
    ],
)
def test_rate_limit_wait(code, headers, body, low, high):
    headers = {
        k: str(int(time.time()) + int(v[1:])) if v.startswith("+") else v
        for k, v in headers.items()
    }
    wait = fr._flow.rate_limit_wait(_refusal(code, headers, body))
    assert wait is None if low is None else low <= wait <= high


def test_a_rate_limit_parks_every_github_fetch_until_it_resets(monkeypatch):
    mono = [1000.0]
    monkeypatch.setattr(fr._flow.time, "monotonic", lambda: mono[0])
    calls = []

    def refuse(req, timeout = 5.0):
        calls.append(req.full_url)
        raise _refusal(403, {"Retry-After": "1800"})

    monkeypatch.setattr(fr._flow, "auth_safe_open", refuse)
    assert fr._fetch_latest_release_tag("unslothai/llama.cpp") is None
    assert fr._fetch_latest_release_tag("unslothai/llama.cpp") is None
    assert fr._fetch_latest_release_assets("unslothai/llama.cpp") is None
    assert len(calls) == 1
    fr._flow.hold_github_api(60)
    assert fr._flow.github_rate_limit_remaining() == 1800
    mono[0] += 1801
    assert fr._fetch_latest_release_tag("unslothai/llama.cpp") is None
    assert len(calls) == 2
