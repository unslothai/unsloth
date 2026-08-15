# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Trust-anchor tests for install_whisper_prebuilt.py.

Whisper verifies each download against the release's own
whisper-prebuilt-sha256.json checksum index (the same model as
install_llama_prebuilt.py), not a committed pins file. These pin the index
parser, the fail-closed behaviour when an asset is not covered, the
tampered-manifest guard, and the newest-release resolution.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

_studio = Path(__file__).resolve().parent.parent.parent
if str(_studio) not in sys.path:
    sys.path.insert(0, str(_studio))

iwp = importlib.import_module("install_whisper_prebuilt")

if not hasattr(iwp, "parse_release_checksums"):
    pytest.skip("checksum-model symbols not present - check branch", allow_module_level = True)

_A = "0" * 64
_B = "1" * 64
_TAG = "v1.9.1-unsloth.1"
_REPO = "unslothai/whisper.cpp"


def _index(**overrides) -> dict:
    payload = {
        "schema_version": 1,
        "component": "whisper.cpp",
        "release_tag": _TAG,
        "upstream_tag": "v1.9.1",
        "artifacts": {
            "whisper-v1.9.1-unsloth.1-linux-x64-cpu.tar.gz": {"sha256": _A},
            "whisper-v1.9.1-unsloth.1-linux-x64-cuda12-portable.tar.gz": {"sha256": _B},
        },
    }
    payload.update(overrides)
    return payload


# parse_release_checksums / expected_sha256_for are prebuilt_core re-exports;
# their valid/fail-closed matrix is asserted against the real whisper
# descriptor in tests/studio/install/test_prebuilt_core.py. The download-host
# fast-path tests below still route through this module's parse wrapper.

# release tag resolution.


def test_resolve_release_tag_explicit_override_passthrough():
    assert iwp.resolve_release_tag(_REPO, published_release_tag = "v1.9.1-unsloth.2") == (
        "v1.9.1-unsloth.2"
    )


def test_resolve_release_tag_resolves_newest_when_no_override(monkeypatch):
    monkeypatch.setattr(iwp, "resolve_newest_release_tag", lambda repo: "v9.9.9-unsloth.9")
    assert iwp.resolve_release_tag(_REPO, published_release_tag = None) == "v9.9.9-unsloth.9"


def test_resolve_newest_release_tag_picks_latest_published(monkeypatch):
    releases = [
        {"tag_name": "v1.9.1-unsloth.1", "published_at": "2026-01-01T00:00:00Z"},
        {"tag_name": "v1.9.1-unsloth.3", "published_at": "2026-03-01T00:00:00Z"},
        {"tag_name": "v1.9.1-unsloth.2", "published_at": "2026-02-01T00:00:00Z"},
        {"tag_name": "draft", "published_at": "2026-09-01T00:00:00Z", "draft": True},
        {"tag_name": "pre", "published_at": "2026-09-01T00:00:00Z", "prerelease": True},
    ]
    monkeypatch.setattr(iwp, "fetch_json", lambda url: releases)
    assert iwp.resolve_newest_release_tag(_REPO) == "v1.9.1-unsloth.3"


def test_resolve_newest_release_tag_none_published_fails_closed(monkeypatch):
    monkeypatch.setattr(iwp, "fetch_json", lambda url: [{"tag_name": "d", "draft": True}])
    with pytest.raises(iwp.PrebuiltFallback):
        iwp.resolve_newest_release_tag(_REPO)


def test_pins_symbols_are_gone():
    # The committed-pins trust model was removed in favour of llama's runtime index.
    for gone in ("load_pins", "pins_path", "resolve_expected_sha256", "PINS_FILENAME"):
        assert not hasattr(iwp, gone), f"{gone} should have been removed"


# Download-host fast path (resolve + fetch the JSON assets with no GitHub API).

_CPU_ASSET = "whisper-v1.9.1-unsloth.1-linux-x64-cpu.tar.gz"


def _manifest() -> dict:
    return {
        "schema_version": 1,
        "component": "whisper.cpp",
        "upstream_tag": "v1.9.1",
        "artifacts": [{"asset": _CPU_ASSET, "os": "linux", "arch": "x64", "backend": "cpu"}],
    }


def _no_api(monkeypatch):
    """Fail loudly if any code path touches api.github.com."""

    def _boom(*a, **k):
        raise AssertionError("api.github.com was used on the fast path")

    monkeypatch.setattr(iwp, "fetch_json", _boom)
    monkeypatch.setattr(iwp, "github_release", _boom)
    monkeypatch.setattr(iwp, "fetch_release_bundle", _boom)


def test_fetch_release_for_install_prefers_download_host(monkeypatch):
    _no_api(monkeypatch)
    monkeypatch.setattr(iwp, "_download_host_latest_release_tag", lambda repo: _TAG)

    def _dhj(url):
        if url.endswith(iwp.SHA256_ASSET_NAME):
            return _index()
        if url.endswith(iwp.MANIFEST_ASSET_NAME):
            return _manifest()
        raise AssertionError(f"unexpected url {url}")

    monkeypatch.setattr(iwp, "_download_host_json", _dhj)
    bundle, checks = iwp.fetch_release_for_install(_REPO, published_release_tag = None)
    assert bundle.release_tag == _TAG
    assert checks[_CPU_ASSET] == _A
    # asset_urls point at the download host (github.com), not the API.
    assert bundle.asset_urls[iwp.SHA256_ASSET_NAME].startswith(
        f"https://github.com/{_REPO}/releases/"
    )
    assert bundle.asset_urls[_CPU_ASSET].startswith(
        f"https://github.com/{_REPO}/releases/download/"
    )
    walked = iwp._fetch_release_candidate(_REPO, _TAG)
    assert iwp.SHA256_ASSET_NAME in walked.asset_urls
    assert _CPU_ASSET in walked.asset_urls


def test_fetch_release_for_install_explicit_tag_skips_the_head(monkeypatch):
    # An explicit tag needs no /releases/latest HEAD: resolving it must not call it.
    monkeypatch.setattr(
        iwp,
        "_download_host_latest_release_tag",
        lambda repo: (_ for _ in ()).throw(AssertionError("HEAD used for an explicit tag")),
    )
    monkeypatch.setattr(
        iwp,
        "_download_host_json",
        lambda url: _index() if url.endswith(iwp.SHA256_ASSET_NAME) else _manifest(),
    )
    _no_api(monkeypatch)
    bundle, checks = iwp.fetch_release_for_install(_REPO, published_release_tag = _TAG)
    assert bundle.release_tag == _TAG


def test_fetch_release_for_install_falls_back_to_api(monkeypatch):
    # Fast path returns None (e.g. a 404) -> the API path resolves the release.
    monkeypatch.setattr(iwp, "_resolve_release_via_download_host", lambda repo, tag: None)
    sentinel = iwp.ReleaseBundle(repo = _REPO, release_tag = _TAG, manifest = _manifest(), asset_urls = {})
    monkeypatch.setattr(iwp, "resolve_release_tag", lambda repo, *, published_release_tag: _TAG)
    monkeypatch.setattr(iwp, "fetch_release_bundle", lambda repo, tag: sentinel)
    monkeypatch.setattr(iwp, "fetch_release_checksums", lambda bundle: {_CPU_ASSET: _A})
    bundle, checks = iwp.fetch_release_for_install(_REPO, published_release_tag = None)
    assert bundle is sentinel
    assert checks == {_CPU_ASSET: _A}


def test_resolve_via_download_host_sha_404_returns_none(monkeypatch):
    import urllib.error

    monkeypatch.setattr(iwp, "_download_host_latest_release_tag", lambda repo: _TAG)

    def _dhj(url):
        raise urllib.error.HTTPError(url, 404, "not found", {}, None)

    monkeypatch.setattr(iwp, "_download_host_json", _dhj)
    assert iwp._resolve_release_via_download_host(_REPO, None) is None


def test_resolve_via_download_host_tag_mismatch_returns_none(monkeypatch):
    # A checksum index whose self-reported release_tag disagrees is rejected (None).
    monkeypatch.setattr(iwp, "_download_host_latest_release_tag", lambda repo: _TAG)
    monkeypatch.setattr(
        iwp, "_download_host_json", lambda url: _index(release_tag = "v1.9.1-unsloth.2")
    )
    assert iwp._resolve_release_via_download_host(_REPO, None) is None


def test_download_host_latest_release_tag_parses_redirect(monkeypatch):
    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def geturl(self):
            return f"https://github.com/{_REPO}/releases/tag/{_TAG}"

    class _Opener:
        def open(
            self,
            req,
            timeout = None,
        ):
            return _Resp()

    monkeypatch.setattr(iwp, "_URL_OPENER", _Opener())
    assert iwp._download_host_latest_release_tag(_REPO) == _TAG


def test_download_host_latest_release_tag_404_returns_none(monkeypatch):
    import urllib.error

    class _Opener:
        def open(
            self,
            req,
            timeout = None,
        ):
            raise urllib.error.HTTPError(req.full_url, 404, "nf", {}, None)

    monkeypatch.setattr(iwp, "_URL_OPENER", _Opener())
    assert iwp._download_host_latest_release_tag(_REPO) is None
