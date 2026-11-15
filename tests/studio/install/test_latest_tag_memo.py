# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""One HEAD on github.com/<repo>/releases/latest per process per update.

The "already current" check and the release resolve that follows it ask the same
redirect; llama.cpp and whisper.cpp share prebuilt_core's lookup, so the second ask
must come from the per-process memo, and only a resolved tag may be kept.
"""

import importlib.util
import sys
import urllib.error
from pathlib import Path

import pytest

_STUDIO_DIR = Path(__file__).resolve().parents[3] / "studio"
if str(_STUDIO_DIR) not in sys.path:
    sys.path.insert(0, str(_STUDIO_DIR))


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, _STUDIO_DIR / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ILP = _load("studio_install_llama_prebuilt_latest_memo", "install_llama_prebuilt.py")
WSP = _load("studio_install_whisper_prebuilt_latest_memo", "install_whisper_prebuilt.py")

LLAMA_REPO = ILP.DEFAULT_PUBLISHED_REPO
WHISPER_REPO = WSP.DEFAULT_PUBLISHED_REPO
LLAMA_TAG = "b9999-mix-abc1234"
WHISPER_TAG = "v1.9.9-mix-def5678"


class _Response:
    def __init__(self, url):
        self._url = url

    def __enter__(self):
        return self

    def __exit__(self, *_a):
        return False

    def geturl(self):
        return self._url


class _CountingOpener:
    """Answers /releases/latest HEADs from *tags*; *failures* raise first."""

    def __init__(
        self,
        tags,
        failures = 0,
    ):
        self.tags = tags
        self.failures = failures
        self.heads: list[str] = []

    def open(
        self,
        request,
        timeout = None,
    ):
        url = request.full_url
        assert request.get_method() == "HEAD" and url.endswith("/releases/latest"), url
        self.heads.append(url)
        if self.failures:
            self.failures -= 1
            raise urllib.error.URLError("simulated outage")
        repo = url[len("https://github.com/") : -len("/releases/latest")]
        return _Response(f"https://github.com/{repo}/releases/tag/{self.tags[repo]}")


def _no_checksum_asset(url):
    raise urllib.error.HTTPError(url, 404, "Not Found", {}, None)  # type: ignore[arg-type]


@pytest.fixture
def opener(monkeypatch):
    fake = _CountingOpener({LLAMA_REPO: LLAMA_TAG, WHISPER_REPO: WHISPER_TAG})
    for module in (ILP, WSP, WSP.llama):
        monkeypatch.setattr(module, "_URL_OPENER", fake)
    monkeypatch.delenv("UNSLOTH_LLAMA_DISABLE_DOWNLOAD_HOST_RESOLVE", raising = False)
    return fake


def _tags_asked(monkeypatch, module, attr):
    asked: list[str] = []
    real = getattr(module, attr)

    def spy(repo, tag, asset):
        asked.append(tag)
        return real(repo, tag, asset)

    monkeypatch.setattr(module, attr, spy)
    return asked


def test_llama_update_sends_one_latest_head(monkeypatch, opener):
    monkeypatch.setattr(ILP, "_fetch_download_host_json", _no_checksum_asset)
    asked = _tags_asked(monkeypatch, ILP, "_release_asset_download_url")
    # The currency check (existing_install_current_without_plan) ...
    expected = ILP._expected_release_tag_without_plan({}, "latest", LLAMA_REPO, "")
    # ... then _select -> iter_resolved_published_releases -> the download-host resolve.
    assert ILP._download_host_resolved_release(LLAMA_REPO) is None
    assert expected == LLAMA_TAG
    assert asked and set(asked) == {LLAMA_TAG}
    assert len(opener.heads) == 1, opener.heads


def test_whisper_update_sends_one_latest_head(monkeypatch, opener):
    monkeypatch.setattr(WSP, "_download_host_json", _no_checksum_asset)
    asked = _tags_asked(monkeypatch, WSP, "release_asset_download_url")
    # The whisper currency check asks through install_llama_prebuilt ...
    latest = WSP.llama._download_host_latest_release_tag(WHISPER_REPO)
    # ... and the resolve asks through whisper's own namespace.
    assert WSP._resolve_release_via_download_host(WHISPER_REPO, "") is None
    assert latest == WHISPER_TAG
    assert asked and set(asked) == {WHISPER_TAG}
    assert len(opener.heads) == 1, opener.heads


def test_memo_is_per_repo(opener):
    assert ILP._download_host_latest_release_tag(LLAMA_REPO) == LLAMA_TAG
    assert WSP._download_host_latest_release_tag(WHISPER_REPO) == WHISPER_TAG
    assert ILP._download_host_latest_release_tag(LLAMA_REPO) == LLAMA_TAG
    assert WSP._download_host_latest_release_tag(WHISPER_REPO) == WHISPER_TAG
    assert len(opener.heads) == 2, opener.heads


def test_failure_is_not_memoised(opener):
    opener.failures = 1
    with pytest.raises(urllib.error.URLError):
        ILP._download_host_latest_release_tag(LLAMA_REPO)
    assert ILP._download_host_latest_release_tag(LLAMA_REPO) == LLAMA_TAG
    assert ILP._download_host_latest_release_tag(LLAMA_REPO) == LLAMA_TAG
    assert len(opener.heads) == 2, opener.heads


def test_none_answer_is_not_memoised(monkeypatch, opener):
    opener.tags[LLAMA_REPO] = ""  # redirect lands on .../releases/tag/ -> None
    assert ILP._download_host_latest_release_tag(LLAMA_REPO) is None
    opener.tags[LLAMA_REPO] = LLAMA_TAG
    assert ILP._download_host_latest_release_tag(LLAMA_REPO) == LLAMA_TAG
    assert len(opener.heads) == 2, opener.heads


def test_memo_expires_so_a_new_release_is_seen(monkeypatch, opener):
    core = sys.modules["prebuilt_core"]
    now = [1000.0]
    monkeypatch.setattr(core.time, "monotonic", lambda: now[0])
    assert ILP._download_host_latest_release_tag(LLAMA_REPO) == LLAMA_TAG
    opener.tags[LLAMA_REPO] = "b10000-mix-new0000"
    now[0] += core._DOWNLOAD_HOST_LATEST_TAG_TTL_S - 1
    assert ILP._download_host_latest_release_tag(LLAMA_REPO) == LLAMA_TAG
    now[0] += 2
    assert ILP._download_host_latest_release_tag(LLAMA_REPO) == "b10000-mix-new0000"
    assert len(opener.heads) == 2, opener.heads


def test_pinned_tag_skips_the_head(monkeypatch, opener):
    monkeypatch.setattr(ILP, "_fetch_download_host_json", _no_checksum_asset)
    assert ILP._expected_release_tag_without_plan({}, "latest", LLAMA_REPO, "pin-1") == "pin-1"
    assert ILP._download_host_resolved_release(LLAMA_REPO, "pin-1") is None
    assert opener.heads == []


def test_disabled_download_host_skips_the_head(monkeypatch, opener):
    monkeypatch.setenv("UNSLOTH_LLAMA_DISABLE_DOWNLOAD_HOST_RESOLVE", "1")
    monkeypatch.setattr(ILP, "_api_newest_release_tag", lambda repo: "api-tag")
    assert ILP._expected_release_tag_without_plan({}, "latest", LLAMA_REPO, "") == "api-tag"
    assert opener.heads == []
