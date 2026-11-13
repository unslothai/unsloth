# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Test the Lemonade and FastFlowLM pin updater against canned GitHub releases, offline."""

from __future__ import annotations

import hashlib
import io
import json
import sys
import tarfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
if str(REPO_ROOT / "studio") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "studio"))

import install_lemonade_prebuilt as lp  # noqa: E402
import update_lemonade_pins as up  # noqa: E402


def _blob(name: str) -> bytes:
    return f"bytes of {name}".encode()


def _release(tag: str, names: dict[str, str], **overrides) -> dict:
    assets = []
    for name in names.values():
        data = _blob(name)
        asset = {
            "name": name,
            "browser_download_url": f"https://example.invalid/{name}",
            "size": len(data),
            "digest": "sha256:" + hashlib.sha256(data).hexdigest(),
        }
        asset.update(overrides.get(name, {}))
        assets.append(asset)
    return {"tag_name": tag, "assets": assets}


def _hasher(url: str, needs = None) -> tuple:
    data = _blob(url.rsplit("/", 1)[1])
    return hashlib.sha256(data).hexdigest(), len(data), needs


def _newer(version: str) -> str:
    """A release above `version` in its own scheme: Lemonade moved from 11.9.0 to calendar
    versions (2026.42.1), so a fixed fake like 99.0.0 stops being newer once the pin crosses it."""
    major = up.version_key(version)[0]
    return f"v{major + 1}.0.1"


_PINS = lp.load_pins()
NEW_LEMONADE = _newer(_PINS["lemonade"]["version"])
NEW_FLM = _newer(_PINS["fastflowlm"]["version"])


def _releases(lemonade: str, flm: str, **overrides):
    by_repo = {
        "lemonade-sdk/lemonade": _release(
            lemonade, up._lemonade_assets(lemonade.lstrip("v")), **overrides
        ),
        "ROCm/FastFlowLM": _release(flm, up._fastflowlm_assets(flm.lstrip("v")), **overrides),
    }
    return lambda repo: [by_repo[repo]]


def test_newer_releases_move_both_pins_in_the_installers_format():
    pins = lp.load_pins()
    new, changes = up.update_pins(pins, _releases(NEW_LEMONADE, NEW_FLM), _hasher)
    lemonade, flm = NEW_LEMONADE[1:], NEW_FLM[1:]
    assert changes == [
        f"lemonade: {pins['lemonade']['version']} -> {lemonade}",
        f"fastflowlm: {pins['fastflowlm']['version']} -> {NEW_FLM}",
    ]
    assert new["lemonade"]["version"] == lemonade
    assert new["fastflowlm"]["version"] == NEW_FLM
    linux = new["lemonade"]["assets"]["linux-x64"]
    assert linux["name"] == f"lemonade-embeddable-{lemonade}-ubuntu-x64.tar.gz"
    assert linux["size"] == len(_blob(linux["name"]))
    windows = f"fastflowlm_{flm}_windows_amd64.zip"
    assert new["fastflowlm"]["assets"]["windows-x64"] == {
        "name": windows,
        "sha256": hashlib.sha256(_blob(windows)).hexdigest(),
    }
    assert pins == lp.load_pins(), "the input pins were modified"


def test_the_fake_releases_are_newer_than_the_pins():
    # Every "newer release" case below is vacuous if the fakes sort at or below the pins.
    for section, fake in (("lemonade", NEW_LEMONADE), ("fastflowlm", NEW_FLM)):
        assert up.version_key(fake) > up.version_key(_PINS[section]["version"])


def test_current_or_older_releases_change_nothing():
    pins = lp.load_pins()
    current = _releases("v" + pins["lemonade"]["version"], pins["fastflowlm"]["version"])
    assert up.update_pins(pins, current, _hasher) == (pins, [])
    assert up.update_pins(pins, _releases("v1.0.0", "v0.0.1"), _hasher) == (pins, [])


def test_calendar_versions_sort_after_the_old_scheme():
    assert up.version_key("v2026.41.1") > up.version_key("11.9.0")
    assert up.version_key("v1.0.10") > up.version_key("v1.0.7")
    with pytest.raises(ValueError):
        up.version_key("candidate-v2026.42.0")


def test_a_digest_mismatch_is_refused():
    name = f"fastflowlm_{NEW_FLM[1:]}_linux.tar.gz"
    releases = _releases("v1.0.0", NEW_FLM, **{name: {"digest": "sha256:" + "0" * 64}})
    with pytest.raises(RuntimeError, match = "GitHub publishes"):
        up.update_pins(lp.load_pins(), releases, _hasher)


def test_an_asset_without_a_published_digest_is_refused():
    name = f"fastflowlm_{NEW_FLM[1:]}_linux.tar.gz"
    releases = _releases("v1.0.0", NEW_FLM, **{name: {"digest": None}})
    with pytest.raises(RuntimeError, match = "publishes no digest"):
        up.update_pins(lp.load_pins(), releases, _hasher)


def test_a_release_missing_an_asset_is_refused():
    release = _release(NEW_FLM, {"linux-x64": f"fastflowlm_{NEW_FLM[1:]}_linux.tar.gz"})
    releases = {"lemonade-sdk/lemonade": [{"tag_name": "v1.0.0", "assets": []}]}
    releases["ROCm/FastFlowLM"] = [release]
    with pytest.raises(RuntimeError, match = "windows_amd64.zip"):
        up.update_pins(lp.load_pins(), releases.__getitem__, _hasher)


def test_write_is_idempotent_and_only_limits_the_sections(tmp_path, monkeypatch):
    path = tmp_path / "pins.json"
    path.write_text(json.dumps(lp.load_pins(), indent = 2) + "\n", encoding = "utf-8")
    monkeypatch.setattr(up, "fetch_stable_releases", _releases(NEW_LEMONADE, NEW_FLM))
    monkeypatch.setattr(up, "hash_asset", _hasher)
    assert up.main(["--pins", str(path), "--check"]) == 1
    assert up.main(["--pins", str(path), "--write", "--only", "fastflowlm"]) == 0
    written = json.loads(path.read_text(encoding = "utf-8"))
    assert written["fastflowlm"]["version"] == NEW_FLM
    assert written["lemonade"] == lp.load_pins()["lemonade"]
    lp.load_pins(path)
    before = path.read_text(encoding = "utf-8")
    assert up.main(["--pins", str(path), "--write", "--only", "fastflowlm"]) == 0
    assert path.read_text(encoding = "utf-8") == before


def _lemonade_only(*tags: str):
    releases = [_release(tag, up._lemonade_assets(tag.lstrip("v"))) for tag in tags]
    return lambda repo: releases if repo == "lemonade-sdk/lemonade" else []


def _needs_by_version(needs: dict[str, dict]):
    def hasher(url: str) -> tuple:
        name = url.rsplit("/", 1)[1]
        return _hasher(url, next((n for v, n in needs.items() if f"-{v}-ubuntu" in name), None))

    return hasher


def test_a_release_needing_a_newer_glibc_falls_back_to_the_newest_that_runs(capsys):
    pins = lp.load_pins()
    major = up.version_key(pins["lemonade"]["version"])[0]
    broken, good, older = f"{major + 2}.0.1", f"{major + 1}.0.2", f"{major + 1}.0.1"
    hasher = _needs_by_version(
        {broken: {"GLIBC": (2, 43)}, good: {"GLIBC": (2, 38)}, older: {"GLIBC": (2, 35)}}
    )
    releases = _lemonade_only(f"v{older}", f"v{broken}", f"v{good}")
    new, changes = up.update_pins(pins, releases, hasher, only = "lemonade")
    assert changes == [f"lemonade: {pins['lemonade']['version']} -> {good}"]
    assert new["lemonade"]["version"] == good
    assert "needs GLIBC 2.43 > 2.39." in capsys.readouterr().err


def test_only_releases_needing_a_newer_glibc_change_nothing():
    pins = lp.load_pins()
    broken = _newer(pins["lemonade"]["version"])
    hasher = _needs_by_version({broken[1:]: {"GLIBC": (2, 40)}})
    assert up.update_pins(pins, _lemonade_only(broken), hasher, only = "lemonade") == (pins, [])


def test_the_abi_limits_themselves_are_accepted():
    pins = lp.load_pins()
    tag = _newer(pins["lemonade"]["version"])
    hasher = _needs_by_version({tag[1:]: dict(up.LINUX_ABI_LIMITS)})
    _, changes = up.update_pins(pins, _lemonade_only(tag), hasher, only = "lemonade")
    assert changes == [f"lemonade: {pins['lemonade']['version']} -> {tag[1:]}"]


def test_candidate_tags_are_not_releases():
    pins = lp.load_pins()
    newer = _newer(pins["lemonade"]["version"])
    assert (
        up.newer_releases([{"tag_name": "candidate-" + newer}], pins["lemonade"]["version"]) == []
    )


def test_stable_releases_are_read_from_every_page(monkeypatch):
    first = f"{up._API}/repos/lemonade-sdk/lemonade/releases?per_page=100"
    second = f"{up._API}/repositories/1/releases?per_page=100&page=2"
    pages = {
        first: (
            [{"tag_name": f"candidate-v9.{n}.0", "prerelease": True} for n in range(100)],
            f'<{second}>; rel="next", <{second}>; rel="last"',
        ),
        second: ([{"tag_name": "v9.0.1"}, {"tag_name": "v9.0.0", "draft": True}], None),
    }

    class Page(io.BytesIO):
        def __init__(self, url):
            body, link = pages[url]
            super().__init__(json.dumps(body).encode())
            self.headers = {"Link": link} if link else {}

    seen = []
    monkeypatch.setattr(
        up.urllib.request,
        "urlopen",
        lambda request, timeout: seen.append(request.full_url) or Page(request.full_url),
    )
    assert up.fetch_stable_releases("lemonade-sdk/lemonade") == [{"tag_name": "v9.0.1"}]
    assert seen == [first, second]


def _tarball(files: dict[str, bytes]) -> io.BytesIO:
    buffer = io.BytesIO()
    with tarfile.open(fileobj = buffer, mode = "w:gz") as archive:
        for name, data in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
    buffer.seek(0)
    return buffer


def test_abi_needs_read_only_elf_files():
    elf = b"\x7fELF" + b"\0" * 60 + b"GLIBC_2.2.5\0GLIBC_2.38\0GLIBCXX_3.4.32\0OPENSSL_3.0.0\0"
    lib = b"\x7fELF" + b"\0" * 60 + b"GLIBC_2.43\0OPENSSL_3.2.0\0"
    script = b"#!/bin/sh\n# needs GLIBC_2.99\n"
    expected = {"GLIBC": (2, 38), "GLIBCXX": (3, 4, 32), "OPENSSL": (3, 0, 0)}
    assert up.tar_abi_needs(_tarball({"x/lemond": elf, "x/flm": script})) == expected
    both = up.tar_abi_needs(_tarball({"x/lemond": elf, "x/lib/libfoo.so": lib}))
    assert both == {**expected, "GLIBC": (2, 43), "OPENSSL": (3, 2, 0)}
    assert up.tar_abi_needs(_tarball({"x/README": b"GLIBC_2.50"})) == {}


def test_a_newer_openssl_alone_is_skipped(capsys):
    pins = lp.load_pins()
    tag = _newer(pins["lemonade"]["version"])
    hasher = _needs_by_version({tag[1:]: {"GLIBC": (2, 38), "OPENSSL": (3, 2, 0)}})
    assert up.update_pins(pins, _lemonade_only(tag), hasher, only = "lemonade") == (pins, [])
    assert "needs OPENSSL 3.2.0 > 3.0.9." in capsys.readouterr().err


def test_a_digest_mismatch_is_refused_even_when_glibc_would_skip_it():
    pins = lp.load_pins()
    tag = _newer(pins["lemonade"]["version"])
    name = f"lemonade-embeddable-{tag[1:]}-ubuntu-x64.tar.gz"
    release = _release(
        tag, up._lemonade_assets(tag[1:]), **{name: {"digest": "sha256:" + "0" * 64}}
    )
    hasher = _needs_by_version({tag[1:]: {"GLIBC": (2, 43)}})
    with pytest.raises(RuntimeError, match = "GitHub publishes"):
        up.update_pins(pins, lambda repo: [release], hasher, only = "lemonade")
