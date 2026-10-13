# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The audio.cpp release policy: the tolerant version parser, is_behind with its downgrade guard, and the release caches, including the cache-only read the Audio page status relies on."""

from __future__ import annotations

import importlib.util
import json
import time
from pathlib import Path

import pytest

from utils import audio_cpp_freshness as fr
from utils.prebuilt import freshness_flow

_REPO = "unslothai/audio.cpp"

# The same table the installer's release_version_key is held to.
_VERSIONS = [
    ("v0.9.1-unsloth.1", (0, 9, 1, 1)),
    ("v0.9.0-unsloth.1", (0, 9, 0, 1)),
    ("v0.9.0", (0, 9, 0, 0)),
    ("v0.8.2-audio8-perf-hotfix-unsloth.1", (0, 8, 2, 1)),
    ("b123", None),
    ("", None),
    (None, None),
    ("latest", None),
    # Tags are lowercase-v and carry no spaces; both parsers refuse anything else alike.
    ("V0.9.1-unsloth.1", None),
    ("v0.9.1-unsloth.1 rc", None),
]


@pytest.fixture(autouse = True)
def _isolated(monkeypatch, tmp_path):
    monkeypatch.setattr(fr, "_cache_dir", lambda: tmp_path / "audio_cpp_freshness")
    fr.reset_caches()
    yield
    fr.reset_caches()


def _no_network(monkeypatch):
    calls: list = []

    def boom(*args, **kwargs):
        calls.append(args)
        raise AssertionError("network fetch")

    monkeypatch.setattr(freshness_flow, "_fetch_newest_published_release", boom)
    return calls


# --- version policy ---


@pytest.mark.parametrize("tag, key", _VERSIONS)
def test_parse_release_version(tag, key):
    assert fr.parse_release_version(tag) == key


def test_the_installer_parses_the_same_table():
    installer_path = Path(__file__).resolve().parents[2] / "install_audio_cpp_prebuilt.py"
    spec = importlib.util.spec_from_file_location("_audio_cpp_installer_parity", installer_path)
    installer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(installer)
    for tag, key in _VERSIONS:
        assert installer.release_version_key(tag) == key, tag
        assert installer.release_version_key(tag) == fr.parse_release_version(tag), tag


@pytest.mark.parametrize(
    "installed, latest, behind",
    [
        ("v0.9.0-unsloth.1", "v0.9.1-unsloth.1", True),
        # Upstream's tag carries no serial, so the fork's build of the same version is newer.
        ("v0.9.0", "v0.9.0-unsloth.1", True),
        # A hotfix of an older line published last is never offered over the newer install.
        ("v0.9.1-unsloth.1", "v0.8.2-audio8-perf-hotfix-unsloth.1", False),
        ("v0.9.1-unsloth.1", "v0.9.1-unsloth.1", False),
        ("v0.9.1-unsloth.1", "v0.9.0-unsloth.1", False),
        # No order, no proof the install is older.
        ("custom-build", "v0.9.1-unsloth.1", False),
        ("v0.9.0-unsloth.1", "nightly", False),
        (None, "v0.9.1-unsloth.1", False),
        ("v0.9.0-unsloth.1", None, False),
    ],
)
def test_is_behind(installed, latest, behind):
    assert fr.is_behind(installed, latest) is behind


# --- release caches ---


def test_latest_is_memoized_and_written_to_disk(monkeypatch, tmp_path):
    calls: list = []
    monkeypatch.setattr(
        fr, "_fetch_latest_release_tag", lambda repo: calls.append(repo) or "v0.9.1-unsloth.1"
    )
    assert fr.latest_published_release(_REPO) == "v0.9.1-unsloth.1"
    assert fr.latest_published_release(_REPO) == "v0.9.1-unsloth.1"
    assert calls == [_REPO]
    cached = json.loads(
        (tmp_path / "audio_cpp_freshness" / "unslothai__audio.cpp.json").read_text()
    )
    assert cached["latest_tag"] == "v0.9.1-unsloth.1"
    assert fr.latest_published_release(_REPO, force_refresh = True) == "v0.9.1-unsloth.1"
    assert calls == [_REPO, _REPO]


def test_a_failed_lookup_is_not_retried_on_every_poll(monkeypatch):
    calls: list = []
    monkeypatch.setattr(fr, "_fetch_latest_release_tag", lambda repo: calls.append(repo))
    assert fr.latest_published_release(_REPO) is None
    assert fr.latest_published_release(_REPO) is None
    assert calls == [_REPO]
    # An explicit check skips the failure memo.
    assert fr.latest_published_release(_REPO, force_refresh = True) is None
    assert calls == [_REPO, _REPO]


def test_a_failed_lookup_keeps_the_last_good_value(monkeypatch, tmp_path):
    freshness_flow.save_disk_cache(
        _REPO, "v0.9.0-unsloth.1", tmp_path / "audio_cpp_freshness", log_message = "test"
    )
    monkeypatch.setattr(fr, "_fetch_latest_release_tag", lambda repo: None)
    assert fr.latest_published_release(_REPO, force_refresh = True) == "v0.9.0-unsloth.1"


def test_the_lookup_goes_through_the_shared_fetch(monkeypatch):
    seen: list = []

    def newest(repo, timeout, *, log_message):
        seen.append(repo)
        return {"tag_name": "v0.9.1-unsloth.1", "assets": [{"name": "a.tar.gz", "size": 7}]}

    monkeypatch.setattr(freshness_flow, "_fetch_newest_published_release", newest)
    assert fr.latest_published_release(_REPO) == "v0.9.1-unsloth.1"
    assert fr.latest_release_assets(_REPO) == {"a.tar.gz": 7}
    assert seen == [_REPO, _REPO]


def test_cached_latest_reads_memory_without_the_network(monkeypatch):
    monkeypatch.setattr(fr, "_fetch_latest_release_tag", lambda repo: "v0.9.1-unsloth.1")
    fr.latest_published_release(_REPO)
    calls = _no_network(monkeypatch)
    assert fr.cached_latest_release(_REPO) == "v0.9.1-unsloth.1"
    assert calls == []


def test_cached_latest_reads_an_expired_disk_cache_without_the_network(monkeypatch, tmp_path):
    calls = _no_network(monkeypatch)
    path = freshness_flow.cache_path_for(_REPO, tmp_path / "audio_cpp_freshness")
    path.parent.mkdir(parents = True)
    stale = time.time() - 10 * freshness_flow.RELEASE_CACHE_TTL_SECONDS
    path.write_text(json.dumps({"fetched_at": stale, "latest_tag": "v0.9.1-unsloth.1"}))
    assert fr.cached_latest_release(_REPO) == "v0.9.1-unsloth.1"
    assert calls == []


def test_cached_latest_is_none_when_nothing_was_looked_up(monkeypatch):
    calls = _no_network(monkeypatch)
    assert fr.cached_latest_release(_REPO) is None
    assert fr.cached_latest_release("") is None
    assert calls == []


def test_reset_caches_can_drop_the_disk_cache(monkeypatch, tmp_path):
    monkeypatch.setattr(fr, "_fetch_latest_release_tag", lambda repo: "v0.9.1-unsloth.1")
    monkeypatch.setattr(fr, "_fetch_latest_release_assets", lambda repo: {"a.tar.gz": 1})
    fr.latest_published_release(_REPO)
    fr.latest_release_assets(_REPO)
    fr.reset_caches()
    # Memory only: the disk cache still answers.
    assert fr._release_memo == {} and fr._assets_memo == {}
    assert fr.cached_latest_release(_REPO) == "v0.9.1-unsloth.1"
    fr.reset_caches(drop_disk = True)
    assert not (tmp_path / "audio_cpp_freshness").exists()
    assert fr.cached_latest_release(_REPO) is None
