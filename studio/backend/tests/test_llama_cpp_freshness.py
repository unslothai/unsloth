# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for the llama.cpp prebuilt freshness check.

Pins the marker parser, the disk+memory cache, the stale decision
matrix, and fail-open behaviour on missing data.
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

_loggers_stub = _types.ModuleType("loggers")
_loggers_stub.get_logger = lambda name: __import__("logging").getLogger(name)
sys.modules.setdefault("loggers", _loggers_stub)

_structlog_stub = _types.ModuleType("structlog")
_structlog_stub.get_logger = lambda *a, **k: __import__("logging").getLogger("stub")
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
        "installed_at_utc": (
            datetime.now(tz = timezone.utc) - timedelta(days = 1)
        ).isoformat().replace("+00:00", "Z"),
    }
    payload.update(overrides)
    install_dir.mkdir(parents = True, exist_ok = True)
    (install_dir / "UNSLOTH_PREBUILT_INFO.json").write_text(json.dumps(payload))
    return install_dir / "UNSLOTH_PREBUILT_INFO.json"


def _fake_binary(install_dir: Path, *, layout: str = "cmake") -> Path:
    """Stub llama-server under one of the supported install layouts."""
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
    # Isolate disk cache per-test; never touch the user's real cache.
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
    # Windows cmake puts the .exe under build/bin/Release/, so the
    # marker is four levels above the binary.
    install_dir = tmp_path / "llama.cpp"
    _write_marker(install_dir, tag = "b8888")
    bin_path = _fake_binary(install_dir, layout = "windows")
    marker = fr.read_install_marker(str(bin_path))
    assert marker is not None
    assert marker["tag"] == "b8888"


@pytest.mark.parametrize("repo", ["unslothai/llama.cpp", "ggml-org/llama.cpp"])
def test_read_install_marker_carries_published_repo_dynamically(tmp_path, repo):
    # The freshness check queries whichever release repo the marker
    # records, so CUDA Linux (unslothai), CPU Linux x86_64 / macOS
    # (ggml-org), and ROCm source-build (unslothai upstream label)
    # all surface the right "latest" tag.
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
        installed_at_utc = (
            datetime.now(tz = timezone.utc) - timedelta(days = 5)
        ).isoformat().replace("+00:00", "Z"),
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
        installed_at_utc = (
            datetime.now(tz = timezone.utc) - timedelta(days = 30)
        ).isoformat().replace("+00:00", "Z"),
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
        installed_at_utc = (
            datetime.now(tz = timezone.utc) - timedelta(days = 1)
        ).isoformat().replace("+00:00", "Z"),
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
        installed_at_utc = (
            datetime.now(tz = timezone.utc) - timedelta(days = 10)
        ).isoformat().replace("+00:00", "Z"),
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
        installed_at_utc = (
            datetime.now(tz = timezone.utc) - timedelta(days = 2)
        ).isoformat().replace("+00:00", "Z"),
    )
    bin_path = _fake_binary(install_dir, layout = "root")
    monkeypatch.setattr(fr, "_fetch_latest_release_tag", lambda repo, timeout = 5.0: "b9300")
    info = fr.check_prebuilt_freshness(str(bin_path), threshold_days = 1)
    assert info["stale"] is True


# format_stale_warning.


def test_format_stale_warning_contains_actionable_command():
    msg = fr.format_stale_warning(
        {"installed_tag": "b9190", "latest_tag": "b9300", "age_days": 5}
    )
    assert "b9190" in msg
    assert "b9300" in msg
    assert "5 days" in msg
    assert "unsloth studio update" in msg


def test_format_stale_warning_singular_day():
    msg = fr.format_stale_warning(
        {"installed_tag": "b9190", "latest_tag": "b9300", "age_days": 1}
    )
    assert "1 day" in msg
    assert "1 days" not in msg
