# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""detect_install_source() memo: reused while the dist-info metadata is unchanged, recomputed after a
reinstall, never covering answers that depend on a Git checkout, and env overrides still read per call."""

from __future__ import annotations

import json
import os
from importlib.metadata import PackageNotFoundError, PathDistribution
from pathlib import Path

import pytest

from utils import update_status


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents = True, exist_ok = True)
    path.write_text(text, encoding = "utf-8")


def _bump(path: Path, text: str) -> None:
    """Rewrite with new content and a distinct mtime, as a reinstall would."""
    before = path.stat().st_mtime_ns if path.exists() else 0
    _write(path, text)
    os.utime(path, ns = (before + 10**9, before + 10**9))


def _make_install(root: Path, package_dir: str = "site-packages") -> Path:
    site = root / package_dir
    dist_info = site / "unsloth-2026.1.1.dist-info"
    files = ["unsloth/__init__.py", "unsloth_cli/__init__.py", "studio/backend/main.py"]
    for rel in files:
        _write(site / rel, "")
    _write(dist_info / "METADATA", "Name: unsloth\nVersion: 2026.1.1\n")
    _write(dist_info / "RECORD", "".join(f"{rel},,\n" for rel in files))
    return dist_info


@pytest.fixture
def install(tmp_path, monkeypatch):
    dist_info = _make_install(tmp_path)
    monkeypatch.setattr(update_status, "distribution", lambda name: PathDistribution(dist_info))
    update_status.reset_install_source_cache()
    yield dist_info
    update_status.reset_install_source_cache()


@pytest.fixture
def scans(monkeypatch):
    calls = []
    real = update_status._distribution_package_paths

    def counting(dist):
        calls.append(dist)
        return real(dist)

    monkeypatch.setattr(update_status, "_distribution_package_paths", counting)
    return calls


def test_pypi_answer_is_reused_without_rescanning(install, scans):
    assert update_status.detect_install_source() == "pypi"
    assert update_status.detect_install_source() == "pypi"
    assert update_status.detect_install_source() == "pypi"
    assert len(scans) == 1


def test_record_rewrite_recomputes(install, scans):
    assert update_status.detect_install_source() == "pypi"
    _bump(install / "RECORD", "unsloth/__init__.py,,\nunsloth/extra.py,,\n")
    assert update_status.detect_install_source() == "pypi"
    assert len(scans) == 2
    assert update_status.detect_install_source() == "pypi"
    assert len(scans) == 2


@pytest.mark.parametrize(
    "payload, expected",
    [
        ({"url": "file:///src/unsloth", "dir_info": {"editable": True}}, "editable"),
        (
            {
                "url": "https://github.com/unslothai/unsloth",
                "vcs_info": {"vcs": "git", "commit_id": "0" * 40},
            },
            "vcs",
        ),
        ({"url": "file:///tmp/unsloth.whl", "archive_info": {}}, "local_path"),
        ({"url": "https://example.invalid/unsloth.whl", "archive_info": {}}, "unknown"),
    ],
)
def test_direct_url_added_and_removed_is_picked_up(install, payload, expected):
    assert update_status.detect_install_source() == "pypi"
    _bump(install / "direct_url.json", json.dumps(payload))
    assert update_status.detect_install_source() == expected
    assert update_status.detect_install_source() == expected
    (install / "direct_url.json").unlink()
    assert update_status.detect_install_source() == "pypi"


def test_direct_url_rewrite_recomputes(install):
    _bump(
        install / "direct_url.json",
        json.dumps({"url": "file:///x", "dir_info": {"editable": True}}),
    )
    assert update_status.detect_install_source() == "editable"
    _bump(
        install / "direct_url.json",
        json.dumps({"url": "https://github.com/u/u", "vcs_info": {"vcs": "git"}}),
    )
    assert update_status.detect_install_source() == "vcs"


def test_direct_url_seen_only_mid_detection_is_not_reused(install, monkeypatch):
    direct_url = install / "direct_url.json"

    class Racing(PathDistribution):
        def read_text(self, filename):
            if filename != "direct_url.json":
                return super().read_text(filename)
            # The file appears after a stat of the metadata and is gone again before the next call.
            _write(direct_url, json.dumps({"url": "file:///x", "dir_info": {"editable": True}}))
            try:
                return super().read_text(filename)
            finally:
                direct_url.unlink()

    monkeypatch.setattr(update_status, "distribution", lambda name: Racing(install))
    assert update_status.detect_install_source() == "editable"
    monkeypatch.setattr(update_status, "distribution", lambda name: PathDistribution(install))
    assert update_status.detect_install_source() == "pypi"


def test_record_rewritten_mid_scan_is_not_memoized(install, monkeypatch, scans):
    real = update_status._distribution_package_paths

    def rewriting(dist):
        paths = real(dist)
        _bump(install / "RECORD", "unsloth/__init__.py,,\nunsloth/extra.py,,\n")
        return paths

    monkeypatch.setattr(update_status, "_distribution_package_paths", rewriting)
    assert update_status.detect_install_source() == "pypi"
    monkeypatch.setattr(update_status, "_distribution_package_paths", real)
    assert update_status._install_source_memo is None


def test_git_checkout_answer_is_not_memoized(tmp_path, monkeypatch, scans):
    dist_info = _make_install(tmp_path / "checkout", package_dir = "src")
    monkeypatch.setattr(update_status, "distribution", lambda name: PathDistribution(dist_info))
    update_status.reset_install_source_cache()
    git_dir = tmp_path / "checkout" / ".git"
    git_dir.mkdir()
    assert update_status.detect_install_source() == "local_repo"
    assert update_status.detect_install_source() == "local_repo"
    assert len(scans) == 2
    git_dir.rmdir()
    assert update_status.detect_install_source() == "pypi"
    git_dir.mkdir()
    assert update_status.detect_install_source() == "local_repo"
    update_status.reset_install_source_cache()


def test_missing_distribution_is_not_memoized(install, monkeypatch):
    assert update_status.detect_install_source() == "pypi"

    def missing(name):
        raise PackageNotFoundError(name)

    monkeypatch.setattr(update_status, "distribution", missing)
    monkeypatch.setattr(update_status, "_repo_root_from_this_file", lambda: install / "nowhere")
    monkeypatch.setattr(update_status, "_path_has_git_parent", lambda path: False)
    assert update_status.detect_install_source() == "unknown"
    monkeypatch.setattr(update_status, "_path_has_git_parent", lambda path: True)
    assert update_status.detect_install_source() == "local_repo"


def test_distribution_without_a_path_is_not_memoized(install, monkeypatch, scans):
    class Opaque:
        def __init__(self, inner):
            self._inner = inner
            self.files = inner.files

        def read_text(self, name):
            return self._inner.read_text(name)

        def locate_file(self, path):
            return self._inner.locate_file(path)

    monkeypatch.setattr(
        update_status, "distribution", lambda name: Opaque(PathDistribution(install))
    )
    assert update_status.detect_install_source() == "pypi"
    assert update_status.detect_install_source() == "pypi"
    assert len(scans) == 2


def test_env_overrides_are_read_per_call_after_a_memo_hit(install, monkeypatch):
    monkeypatch.delenv(update_status.DISABLE_ENV_VAR, raising = False)
    monkeypatch.delenv(update_status.FAKE_UPDATE_ENV_VAR, raising = False)
    assert update_status.get_studio_update_status("dev")["reason"] == "dev_build"

    monkeypatch.setenv(update_status.DISABLE_ENV_VAR, "1")
    status = update_status.get_studio_update_status("dev")
    assert status["reason"] == "disabled"
    assert status["install_source"] == "pypi"

    monkeypatch.delenv(update_status.DISABLE_ENV_VAR)
    monkeypatch.setenv(update_status.FAKE_UPDATE_ENV_VAR, "9999.1.1")
    status = update_status.get_studio_update_status("dev")
    assert status["latest_version"] == "9999.1.1"
    assert status["update_available"] is True

    monkeypatch.delenv(update_status.FAKE_UPDATE_ENV_VAR)
    assert update_status.get_studio_update_status("dev")["reason"] == "dev_build"
    assert update_status.get_studio_install_source_status("dev")["install_source"] == "pypi"
