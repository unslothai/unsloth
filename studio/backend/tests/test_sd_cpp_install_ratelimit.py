# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The pinned sd.cpp prebuilt installs when the GitHub releases API is rate limited or unreachable."""

from __future__ import annotations

import hashlib
import http.client
import io
import sys
import urllib.error
import zipfile
from email.message import Message
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import install_sd_cpp_prebuilt as sdmod  # noqa: E402

_CLI = "sd-cli.exe" if sys.platform == "win32" else "sd-cli"
_LINUX_CPU = f"sd-{sdmod.DEFAULT_TAG}-bin-Linux-Ubuntu-22.04-x86_64.zip"


def _zip_with_sd_cli() -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(f"build/bin/{_CLI}", b"#!/bin/sh\necho sd-cli\n")
    return buf.getvalue()


def _rate_limited(url: str, code: int = 403) -> urllib.error.HTTPError:
    headers = Message()
    headers["X-RateLimit-Remaining"] = "0"
    body = io.BytesIO(b'{"message": "API rate limit exceeded for 203.0.113.7."}')
    return urllib.error.HTTPError(url, code, "rate limit exceeded", headers, body)


@pytest.fixture
def host(monkeypatch):
    monkeypatch.delenv("UNSLOTH_SD_CPP_REPO", raising = False)
    monkeypatch.delenv("UNSLOTH_SD_CPP_TAG", raising = False)
    monkeypatch.delenv("GH_TOKEN", raising = False)
    monkeypatch.delenv("GITHUB_TOKEN", raising = False)
    monkeypatch.setenv("UNSLOTH_SD_CPP_VERIFY_INSTALL", "0")
    monkeypatch.setattr(sdmod.platform, "system", lambda: "Linux")
    monkeypatch.setattr(sdmod.platform, "machine", lambda: "x86_64")
    downloads: list[str] = []
    zb = _zip_with_sd_cli()

    def fake_download(url, dest, **_k):
        downloads.append(url)
        dest.write_bytes(zb)

    monkeypatch.setattr(sdmod, "_download", fake_download)
    return {"downloads": downloads, "zip": zb}


def _api_fails(monkeypatch, make_exc):
    calls: list[str] = []

    def fake_open(req, timeout = 30.0):
        url = getattr(req, "full_url", req)
        calls.append(url)
        raise make_exc(url)

    monkeypatch.setattr(sdmod, "auth_safe_open", fake_open)
    return calls


def _pin_digest(monkeypatch, data: bytes):
    pinned = dict(getattr(sdmod, "PINNED_RELEASE_ASSETS", {}))
    table = dict(pinned.get(sdmod.DEFAULT_TAG, {}))
    table[_LINUX_CPU] = hashlib.sha256(data).hexdigest()
    pinned[sdmod.DEFAULT_TAG] = table
    monkeypatch.setattr(sdmod, "PINNED_RELEASE_ASSETS", pinned, raising = False)


@pytest.mark.parametrize(
    "make_exc",
    [
        lambda url: _rate_limited(url, 403),
        lambda url: _rate_limited(url, 429),
        lambda url: urllib.error.URLError("[Errno 111] Connection refused"),
        lambda url: TimeoutError("timed out"),
        lambda url: http.client.IncompleteRead(b"{", 99),
        lambda url: http.client.RemoteDisconnected("Remote end closed connection without response"),
    ],
    ids = [
        "403-rate-limit",
        "429",
        "unreachable",
        "timeout",
        "incomplete-read",
        "remote-disconnected",
    ],
)
def test_failed_api_installs_pin_from_direct_url(tmp_path, monkeypatch, host, make_exc):
    calls = _api_fails(monkeypatch, make_exc)
    _pin_digest(monkeypatch, host["zip"])
    sd_cli = sdmod.install(install_dir = tmp_path / "sd")
    assert sd_cli.name == _CLI and sd_cli.is_file()
    assert any("api.github.com" in c and sdmod.DEFAULT_TAG in c for c in calls)
    assert host["downloads"] == [
        f"https://github.com/{sdmod.DEFAULT_REPO}/releases/download/{sdmod.DEFAULT_TAG}/{_LINUX_CPU}"
    ]
    rec = sdmod.read_install_record(tmp_path / "sd")
    assert rec.get("tag") == sdmod.DEFAULT_TAG and rec.get("repo") == sdmod.DEFAULT_REPO


def test_failed_api_with_sha_mismatch_still_refuses(tmp_path, monkeypatch, host):
    _api_fails(monkeypatch, lambda url: _rate_limited(url))
    # The shipped digest is the real asset's; the fake download does not match it.
    with pytest.raises(RuntimeError, match = "sha256 mismatch"):
        sdmod.install(install_dir = tmp_path / "sd")
    assert sdmod._locate_sd_cli(tmp_path / "sd") is None


def test_sha_mismatch_on_api_path_does_not_fall_back(tmp_path, monkeypatch, host):
    release = {
        "tag_name": sdmod.DEFAULT_TAG,
        "assets": [
            {
                "name": _LINUX_CPU,
                "browser_download_url": "https://example.invalid/from-api.zip",
                "digest": "sha256:" + "0" * 64,
            }
        ],
    }
    monkeypatch.setattr(sdmod, "_fetch_release", lambda *a, **k: release)
    _pin_digest(monkeypatch, host["zip"])
    with pytest.raises(RuntimeError, match = "sha256 mismatch"):
        sdmod.install(install_dir = tmp_path / "sd")
    assert host["downloads"] == ["https://example.invalid/from-api.zip"]


def test_working_api_path_is_unchanged(tmp_path, monkeypatch, host):
    seen = []
    release = {
        "tag_name": sdmod.DEFAULT_TAG,
        "assets": [
            {
                "name": _LINUX_CPU,
                "browser_download_url": "https://example.invalid/from-api.zip",
                "digest": "sha256:" + hashlib.sha256(host["zip"]).hexdigest(),
            }
        ],
    }

    def fake_fetch(
        tag = None,
        *,
        repo = None,
        **_k,
    ):
        seen.append((repo, tag))
        return release

    monkeypatch.setattr(sdmod, "_fetch_release", fake_fetch)
    assert sdmod.install(install_dir = tmp_path / "sd").is_file()
    assert seen == [(sdmod.DEFAULT_REPO, sdmod.DEFAULT_TAG)]
    assert host["downloads"] == ["https://example.invalid/from-api.zip"]


def test_404_pin_does_not_use_the_shipped_list(monkeypatch, host):
    """A pin the API says is gone keeps the existing 404 ladder (upstream, then latest)."""
    seen = []

    def fake_fetch(
        tag = None,
        *,
        repo = None,
        allow_latest = True,
        **_k,
    ):
        seen.append((repo, tag))
        if allow_latest:
            return {
                "tag_name": "latest",
                "assets": [{"name": "sd-x-bin-Linux-Ubuntu-22.04-x86_64.zip"}],
            }
        return None

    monkeypatch.setattr(sdmod, "_fetch_release", fake_fetch)
    _repo, release, _chosen = sdmod._resolve_with_fallback("cpu", None)
    assert release["tag_name"] == "latest"


def test_no_shipped_list_for_other_tags_or_repos(monkeypatch, host):
    _api_fails(monkeypatch, lambda url: _rate_limited(url))
    monkeypatch.setenv("UNSLOTH_SD_CPP_TAG", "master-1-deadbee-u1234567")
    with pytest.raises(sdmod.GitHubRateLimited):
        sdmod._resolve_with_fallback("cpu", None)
    monkeypatch.delenv("UNSLOTH_SD_CPP_TAG")
    monkeypatch.setenv("UNSLOTH_SD_CPP_REPO", "someone/fork")
    with pytest.raises(sdmod.GitHubRateLimited):
        sdmod._resolve_with_fallback("cpu", None)


def test_shipped_list_covers_the_pin():
    """Bumping DEFAULT_TAG without refreshing the shipped asset list fails here."""
    table = sdmod.PINNED_RELEASE_ASSETS[sdmod.DEFAULT_TAG]
    assert _LINUX_CPU in table and sdmod.SD_MANIFEST_ASSET in table
    for sys_name, machine, acc in [
        ("Darwin", "arm64", "auto"),
        ("Linux", "x86_64", "cpu"),
        ("Linux", "x86_64", "vulkan"),
        ("Windows", "AMD64", "cpu"),
    ]:
        assert sdmod.resolve_release_asset(
            list(table), system = sys_name, machine = machine, accelerator = acc
        )
    for digest in table.values():
        assert len(digest) == 64 and int(digest, 16) >= 0


def test_print_asset_stdout_is_only_the_asset_name(monkeypatch, capsys, host):
    """Scripts capture --print-asset stdout as the filename: the fallback note belongs on stderr."""
    _api_fails(monkeypatch, lambda url: _rate_limited(url, 429))
    assert sdmod.main(["--print-asset"]) == 0
    captured = capsys.readouterr()
    assert captured.out.splitlines() == [_LINUX_CPU]
    assert "shipped asset list" in captured.err
    assert "sha256 verified" not in captured.err


def test_missing_mirror_archive_continues_to_upstream(tmp_path, monkeypatch, host):
    """Mirror API 503 and its direct archive 404: the upstream rung still installs, as before the shipped list."""
    up_tag = sdmod.upstream_tag_for(sdmod.DEFAULT_TAG)
    up_name = f"sd-{up_tag}-bin-Linux-Ubuntu-22.04-x86_64.zip"
    seen = []

    def fake_fetch(
        tag = None,
        *,
        repo = None,
        **_k,
    ):
        seen.append((repo, tag))
        if repo == sdmod.DEFAULT_REPO:
            raise urllib.error.HTTPError(f"https://api/{repo}", 503, "unavailable", None, None)
        if repo == sdmod.UPSTREAM_FALLBACK_REPO and tag == up_tag:
            return {
                "tag_name": up_tag,
                "assets": [
                    {
                        "name": up_name,
                        "browser_download_url": f"https://example.invalid/{up_name}",
                        "digest": "sha256:" + hashlib.sha256(host["zip"]).hexdigest(),
                    }
                ],
            }
        return None

    def fake_download(url, dest, **_k):
        host["downloads"].append(url)
        if url.startswith(f"https://github.com/{sdmod.DEFAULT_REPO}/"):
            raise urllib.error.HTTPError(url, 404, "Not Found", None, None)
        dest.write_bytes(host["zip"])

    monkeypatch.setattr(sdmod, "_fetch_release", fake_fetch)
    monkeypatch.setattr(sdmod, "_download", fake_download)
    sd_cli = sdmod.install(install_dir = tmp_path / "sd")
    assert sd_cli.is_file()
    assert host["downloads"][-1] == f"https://example.invalid/{up_name}"
    assert sdmod.read_install_record(tmp_path / "sd").get("repo") == sdmod.UPSTREAM_FALLBACK_REPO
    assert not (tmp_path / "sd" / _LINUX_CPU).exists()


def test_missing_mirror_archive_with_no_other_rung_still_errors(tmp_path, monkeypatch, host):
    def fake_fetch(
        tag = None,
        *,
        repo = None,
        **_k,
    ):
        raise urllib.error.HTTPError(f"https://api/{repo}", 503, "unavailable", None, None)

    def fake_download(url, dest, **_k):
        raise urllib.error.URLError("[Errno 111] Connection refused")

    monkeypatch.setattr(sdmod, "_fetch_release", fake_fetch)
    monkeypatch.setattr(sdmod, "_download", fake_download)
    with pytest.raises(RuntimeError, match = "No prebuilt sd-cli"):
        sdmod.install(install_dir = tmp_path / "sd")
