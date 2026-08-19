"""Routing of iter_resolved_published_releases between the download-host fast
path and the GitHub API enumeration; all I/O monkeypatched."""

import importlib.util
import json
import sys
import urllib.error
from pathlib import Path

import pytest


PACKAGE_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = PACKAGE_ROOT / "studio" / "install_llama_prebuilt.py"
SPEC = importlib.util.spec_from_file_location("studio_install_llama_prebuilt_dlhost", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
ILP = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = ILP
SPEC.loader.exec_module(ILP)

FORK_REPO = ILP.DEFAULT_PUBLISHED_REPO
PrebuiltFallback = ILP.PrebuiltFallback


def _api_raises(*_a, **_k):
    raise AssertionError("GitHub API enumeration was used")


def _fast_path_raises(_repo):
    raise AssertionError("download-host fast path was used")


def _empty_api(*_a, **_k):
    return iter(())


def _resolve(tag = "latest", **kw):
    return list(ILP.iter_resolved_published_releases(tag, FORK_REPO, "", **kw))


def test_fast_path_yields_latest_without_api(monkeypatch):
    sentinel = object()
    monkeypatch.delenv("UNSLOTH_LLAMA_DISABLE_DOWNLOAD_HOST_RESOLVE", raising = False)
    monkeypatch.setattr(ILP, "_download_host_resolved_release", lambda _repo: sentinel)
    monkeypatch.setattr(ILP, "iter_published_release_bundles", _api_raises)
    assert _resolve() == [sentinel]


def test_fast_path_disabled_by_caller_uses_api(monkeypatch):
    # macOS passes allow_download_host_fast_path = False to keep the walk-back.
    monkeypatch.delenv("UNSLOTH_LLAMA_DISABLE_DOWNLOAD_HOST_RESOLVE", raising = False)
    monkeypatch.setattr(ILP, "_download_host_resolved_release", _fast_path_raises)
    monkeypatch.setattr(ILP, "iter_published_release_bundles", _empty_api)
    with pytest.raises(PrebuiltFallback):
        _resolve(allow_download_host_fast_path = False)


def test_fast_path_disabled_by_env_uses_api(monkeypatch):
    monkeypatch.setenv("UNSLOTH_LLAMA_DISABLE_DOWNLOAD_HOST_RESOLVE", "1")
    monkeypatch.setattr(ILP, "_download_host_resolved_release", _fast_path_raises)
    monkeypatch.setattr(ILP, "iter_published_release_bundles", _empty_api)
    with pytest.raises(PrebuiltFallback):
        _resolve()


def test_fast_path_skipped_for_non_latest_request(monkeypatch):
    monkeypatch.delenv("UNSLOTH_LLAMA_DISABLE_DOWNLOAD_HOST_RESOLVE", raising = False)
    monkeypatch.setattr(ILP, "_download_host_resolved_release", _fast_path_raises)
    monkeypatch.setattr(ILP, "iter_published_release_bundles", _empty_api)
    with pytest.raises(PrebuiltFallback):
        _resolve(tag = "b9964")


def test_fast_path_none_falls_back_to_api(monkeypatch):
    monkeypatch.delenv("UNSLOTH_LLAMA_DISABLE_DOWNLOAD_HOST_RESOLVE", raising = False)
    monkeypatch.setattr(ILP, "_download_host_resolved_release", lambda _repo: None)
    used = {"api": False}

    def _api(*_a, **_k):
        used["api"] = True
        return iter(())

    monkeypatch.setattr(ILP, "iter_published_release_bundles", _api)
    with pytest.raises(PrebuiltFallback):
        _resolve()
    assert used["api"]


def test_fast_path_rejected_checksum_falls_back_to_api(monkeypatch):
    monkeypatch.delenv("UNSLOTH_LLAMA_DISABLE_DOWNLOAD_HOST_RESOLVE", raising = False)

    def _reject(_repo):
        raise PrebuiltFallback("checksum mismatch")

    monkeypatch.setattr(ILP, "_download_host_resolved_release", _reject)
    used = {"api": False}

    def _api(*_a, **_k):
        used["api"] = True
        return iter(())

    monkeypatch.setattr(ILP, "iter_published_release_bundles", _api)
    with pytest.raises(PrebuiltFallback):
        _resolve()
    assert used["api"]


# --- _download_host_resolved_release body (only download_bytes stubbed) --------

RELEASE_TAG = "b9964-mix-53618c5"
UPSTREAM_TAG = "b9964"
BINARY_ASSET = "app-b9964-mix-53618c5-windows-x64-rocm-gfx1151.zip"
MANIFEST_ASSET = ILP.DEFAULT_PUBLISHED_MANIFEST_ASSET
SHA256_ASSET = ILP.DEFAULT_PUBLISHED_SHA256_ASSET


def _manifest_bytes():
    return json.dumps(
        {
            "schema_version": 1,
            "component": "llama.cpp",
            "upstream_tag": UPSTREAM_TAG,
            "source_repo": "unslothai/llama.cpp",
            "source_repo_url": "https://github.com/unslothai/llama.cpp",
            "artifacts": [
                {
                    "asset_name": BINARY_ASSET,
                    "install_kind": "windows-rocm-app",
                    "supported_sms": [],
                    "gfx_target": "gfx1151",
                    "mapped_targets": ["gfx1151"],
                    "rank": 10,
                }
            ],
        }
    ).encode("utf-8")


def _sha_payload(*, manifest_sha256 = None):
    # BINARY_ASSET is deliberately absent: real releases key its hash under an
    # upstream-tag alias, so the manifest name must still get a tag-pinned URL.
    # The source archive entry keeps _validate_checksums_against_bundle happy.
    artifacts = {"llama.cpp-source-b9964.tar.gz": {"sha256": "a" * 64}}
    if manifest_sha256 is not None:
        artifacts[MANIFEST_ASSET] = {"sha256": manifest_sha256}
    return {
        "schema_version": 1,
        "component": "llama.cpp",
        "release_tag": RELEASE_TAG,
        "upstream_tag": UPSTREAM_TAG,
        "artifacts": artifacts,
    }


def _stub_downloads(
    monkeypatch,
    sha_payload,
    manifest_bytes,
    *,
    latest_tag = RELEASE_TAG,
):
    def _no_api(*_a, **_k):
        raise AssertionError("GitHub API was used")

    monkeypatch.setattr(ILP, "github_release", _no_api)
    monkeypatch.setattr(ILP, "fetch_json", _no_api)
    # The authoritative latest tag comes from the /releases/latest redirect
    # (github.com, no api.github.com); stub it so no real request is made.
    monkeypatch.setattr(ILP, "_download_host_latest_release_tag", lambda _repo: latest_tag)

    def _download_bytes(url, *_a, **_k):
        if SHA256_ASSET in url:
            return json.dumps(sha_payload).encode("utf-8")
        if MANIFEST_ASSET in url:
            if isinstance(manifest_bytes, Exception):
                raise manifest_bytes
            return manifest_bytes
        raise AssertionError(f"unexpected download: {url}")

    monkeypatch.setattr(ILP, "download_bytes", _download_bytes)


def test_resolved_release_adds_tag_pinned_url_for_manifest_only_asset(monkeypatch):
    _stub_downloads(monkeypatch, _sha_payload(), _manifest_bytes())
    resolved = ILP._download_host_resolved_release(FORK_REPO)
    assert resolved is not None
    assert resolved.bundle.release_tag == RELEASE_TAG
    # Binary is named only in the manifest, yet the fast path must expose a
    # tag-pinned CDN URL for it (the API path gets it from the real asset list).
    assert resolved.bundle.assets[BINARY_ASSET] == (
        f"https://github.com/{FORK_REPO}/releases/download/{RELEASE_TAG}/{BINARY_ASSET}"
    )


def test_resolved_release_rejects_manifest_checksum_mismatch(monkeypatch):
    # A wrong manifest hash in the checksum payload must fail closed, so the
    # router falls back to the API rather than trusting the fast path.
    _stub_downloads(monkeypatch, _sha_payload(manifest_sha256 = "b" * 64), _manifest_bytes())
    with pytest.raises(PrebuiltFallback, match = "manifest checksum"):
        ILP._download_host_resolved_release(FORK_REPO)


def test_resolved_release_rejects_release_tag_mismatch(monkeypatch):
    # The checksum asset self-reports RELEASE_TAG, but the authoritative
    # /releases/latest redirect resolves a different tag: the fast path must not
    # pin to the stale self-reported tag (it raises, so the router falls back).
    _stub_downloads(monkeypatch, _sha_payload(), _manifest_bytes(), latest_tag = "b9999-mix-other")
    with pytest.raises(RuntimeError, match = "did not match pinned release tag"):
        ILP._download_host_resolved_release(FORK_REPO)


def test_resolved_release_manifest_404_falls_back(monkeypatch):
    # An in-progress release can serve the checksum asset before the manifest; a
    # manifest 404 returns None so the router falls back to the API.
    not_found = urllib.error.HTTPError(
        f"https://github.com/{FORK_REPO}/releases/download/{RELEASE_TAG}/{MANIFEST_ASSET}",
        404,
        "Not Found",
        {},  # type: ignore[arg-type]
        None,
    )
    _stub_downloads(monkeypatch, _sha_payload(), not_found)
    assert ILP._download_host_resolved_release(FORK_REPO) is None


# --- _download_host_latest_release_tag (redirect resolution) -------------------


class _FakeResponse:
    def __init__(self, url):
        self._url = url

    def __enter__(self):
        return self

    def __exit__(self, *_a):
        return False

    def geturl(self):
        return self._url


class _FakeOpener:
    def __init__(
        self,
        *,
        url = None,
        exc = None,
    ):
        self._url = url
        self._exc = exc

    def open(
        self,
        _request,
        timeout = None,
    ):
        if self._exc is not None:
            raise self._exc
        return _FakeResponse(self._url)


def test_latest_release_tag_parses_redirect(monkeypatch):
    final = f"https://github.com/{FORK_REPO}/releases/tag/{RELEASE_TAG}"
    monkeypatch.setattr(ILP, "_URL_OPENER", _FakeOpener(url = final))
    assert ILP._download_host_latest_release_tag(FORK_REPO) == RELEASE_TAG


def test_latest_release_tag_none_on_404(monkeypatch):
    not_found = urllib.error.HTTPError(
        f"https://github.com/{FORK_REPO}/releases/latest",
        404,
        "Not Found",
        {},
        None,  # type: ignore[arg-type]
    )
    monkeypatch.setattr(ILP, "_URL_OPENER", _FakeOpener(exc = not_found))
    assert ILP._download_host_latest_release_tag(FORK_REPO) is None


def test_latest_release_tag_none_when_not_a_tag_url(monkeypatch):
    # No /releases/tag/ segment (e.g. redirected somewhere unexpected) -> None.
    monkeypatch.setattr(ILP, "_URL_OPENER", _FakeOpener(url = f"https://github.com/{FORK_REPO}"))
    assert ILP._download_host_latest_release_tag(FORK_REPO) is None
