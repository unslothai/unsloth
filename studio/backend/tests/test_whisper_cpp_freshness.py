# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for the whisper.cpp prebuilt freshness check.

Pins the whisper version parser, the is_behind decision matrix (with its
downgrade guard), the marker parser across install layouts, and the fail-open
behaviour on missing data.
"""

from __future__ import annotations

import json
import sys
import types as _types
from datetime import datetime, timedelta, timezone
from pathlib import Path

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)


class _NoopLogger:
    """structlog-style logger: every method swallows positional + kwargs."""

    def __getattr__(self, _name):
        return lambda *a, **k: None


_loggers_stub = _types.ModuleType("loggers")
_loggers_stub.get_logger = lambda *a, **k: _NoopLogger()
sys.modules.setdefault("loggers", _loggers_stub)

_structlog_stub = _types.ModuleType("structlog")
_structlog_stub.get_logger = lambda *a, **k: _NoopLogger()
sys.modules.setdefault("structlog", _structlog_stub)

import pytest

from utils import whisper_cpp_freshness as fr


# Helpers.


def _write_marker(install_dir: Path, **overrides) -> Path:
    payload = {
        "requested_tag": "latest",
        "release_tag": "v1.9.1-unsloth.1",
        "upstream_tag": "v1.9.1",
        "published_repo": "unslothai/whisper.cpp",
        "asset": "whisper-v1.9.1-unsloth.1-linux-x64-cpu.tar.gz",
        "asset_sha256": None,
        "source": "published",
        "installed_at_utc": (datetime.now(tz = timezone.utc) - timedelta(days = 1))
        .isoformat()
        .replace("+00:00", "Z"),
    }
    payload.update(overrides)
    install_dir.mkdir(parents = True, exist_ok = True)
    marker = install_dir / "UNSLOTH_WHISPER_PREBUILT_INFO.json"
    marker.write_text(json.dumps(payload))
    return marker


def _fake_binary(install_dir: Path, *, layout: str = "cmake") -> Path:
    """Stub whisper-server under a supported install layout."""
    if layout == "cmake":
        bin_dir = install_dir / "build" / "bin"
        bin_name = "whisper-server"
    elif layout == "root":
        bin_dir = install_dir
        bin_name = "whisper-server"
    elif layout == "windows":
        bin_dir = install_dir / "build" / "bin" / "Release"
        bin_name = "whisper-server.exe"
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


# parse_release_version.


def test_parse_release_version():
    assert fr.parse_release_version("v1.9.1-unsloth.2") == (1, 9, 1, 2)
    assert fr.parse_release_version("1.10.0") == (1, 10, 0, 0)  # no v, no serial
    assert fr.parse_release_version(" v2.0.0-unsloth.10 ") == (2, 0, 0, 10)
    assert fr.parse_release_version("v1.9") == (1, 9, 0, 0)  # padded
    assert fr.parse_release_version("nightly") is None
    assert fr.parse_release_version(None) is None
    assert fr.parse_release_version("") is None


# is_behind decision matrix + downgrade guard.


def test_is_behind_serial_bump():
    assert fr.is_behind("v1.9.1-unsloth.1", "v1.9.1-unsloth.2") is True


def test_is_behind_downgrade_guard():
    # A lower serial or version is never "behind".
    assert fr.is_behind("v1.9.1-unsloth.2", "v1.9.1-unsloth.1") is False
    assert fr.is_behind("v1.10.0-unsloth.1", "v1.9.1-unsloth.9") is False


def test_is_behind_upstream_bump():
    assert fr.is_behind("v1.9.1-unsloth.1", "v1.10.0-unsloth.1") is True


def test_is_behind_identical_is_false():
    assert fr.is_behind("v1.9.1-unsloth.1", "v1.9.1-unsloth.1") is False


def test_is_behind_unparseable_differs_is_behind():
    assert fr.is_behind("v1.9.1-unsloth.1", "nightly") is True


def test_is_behind_missing_side_fails_open():
    assert fr.is_behind(None, "v1.9.1-unsloth.2") is False
    assert fr.is_behind("v1.9.1-unsloth.1", None) is False


# read_install_marker across layouts.


def test_read_install_marker_finds_cmake_layout(tmp_path):
    _write_marker(tmp_path)
    bin_path = _fake_binary(tmp_path, layout = "cmake")
    marker = fr.read_install_marker(str(bin_path))
    assert marker is not None
    assert marker["release_tag"] == "v1.9.1-unsloth.1"


def test_read_install_marker_finds_windows_layout(tmp_path):
    _write_marker(tmp_path)
    bin_path = _fake_binary(tmp_path, layout = "windows")
    marker = fr.read_install_marker(str(bin_path))
    assert marker is not None
    assert marker["published_repo"] == "unslothai/whisper.cpp"


def test_read_install_marker_missing_returns_none(tmp_path):
    bin_path = _fake_binary(tmp_path, layout = "cmake")
    assert fr.read_install_marker(str(bin_path)) is None


def test_read_install_marker_handles_none_path():
    assert fr.read_install_marker(None) is None


# check_prebuilt_freshness end-to-end.


def test_check_prebuilt_freshness_reports_stale_when_old_and_behind(monkeypatch, tmp_path):
    _write_marker(
        tmp_path,
        release_tag = "v1.9.1-unsloth.1",
        installed_at_utc = (datetime.now(tz = timezone.utc) - timedelta(days = 10))
        .isoformat()
        .replace("+00:00", "Z"),
    )
    bin_path = _fake_binary(tmp_path, layout = "cmake")
    monkeypatch.setattr(fr, "latest_published_release", lambda *a, **k: "v1.9.1-unsloth.3")
    info = fr.check_prebuilt_freshness(str(bin_path))
    assert info["has_marker"] is True
    assert info["behind"] is True
    assert info["stale"] is True
    assert info["installed_tag"] == "v1.9.1-unsloth.1"
    assert info["latest_tag"] == "v1.9.1-unsloth.3"


def test_check_prebuilt_freshness_not_stale_when_tag_matches(monkeypatch, tmp_path):
    _write_marker(tmp_path, release_tag = "v1.9.1-unsloth.1")
    bin_path = _fake_binary(tmp_path, layout = "cmake")
    monkeypatch.setattr(fr, "latest_published_release", lambda *a, **k: "v1.9.1-unsloth.1")
    info = fr.check_prebuilt_freshness(str(bin_path))
    assert info["behind"] is False
    assert info["stale"] is False


def test_check_prebuilt_freshness_fails_open_without_marker(tmp_path):
    bin_path = _fake_binary(tmp_path, layout = "cmake")
    info = fr.check_prebuilt_freshness(str(bin_path))
    assert info["has_marker"] is False
    assert info["stale"] is False
    assert info["behind"] is False


def test_check_prebuilt_freshness_fails_open_when_github_unreachable(monkeypatch, tmp_path):
    _write_marker(tmp_path)
    bin_path = _fake_binary(tmp_path, layout = "cmake")
    monkeypatch.setattr(fr, "latest_published_release", lambda *a, **k: None)
    info = fr.check_prebuilt_freshness(str(bin_path))
    assert info["has_marker"] is True
    assert info["behind"] is False
    assert info["stale"] is False


def test_format_stale_warning_mentions_whisper():
    warning = fr.format_stale_warning(
        {"age_days": 4, "installed_tag": "v1.9.1-unsloth.1", "latest_tag": "v1.9.1-unsloth.3"}
    )
    assert "whisper.cpp" in warning
    assert "v1.9.1-unsloth.1" in warning
    assert "v1.9.1-unsloth.3" in warning
