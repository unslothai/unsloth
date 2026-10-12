# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Test the Lemonade and FastFlowLM pin updater against canned GitHub releases, offline."""

from __future__ import annotations

import hashlib
import json
import sys
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


def _hasher(url: str) -> tuple[str, int]:
    data = _blob(url.rsplit("/", 1)[1])
    return hashlib.sha256(data).hexdigest(), len(data)


def _releases(lemonade: str, flm: str, **overrides):
    by_repo = {
        "lemonade-sdk/lemonade": _release(
            lemonade, up._lemonade_assets(lemonade.lstrip("v")), **overrides
        ),
        "ROCm/FastFlowLM": _release(flm, up._fastflowlm_assets(flm.lstrip("v")), **overrides),
    }
    return lambda repo: by_repo[repo]


def test_newer_releases_move_both_pins_in_the_installers_format():
    pins = lp.load_pins()
    new, changes = up.update_pins(pins, _releases("v99.0.0", "v99.0.1"), _hasher)
    assert changes == [
        f"lemonade: {pins['lemonade']['version']} -> 99.0.0",
        f"fastflowlm: {pins['fastflowlm']['version']} -> v99.0.1",
    ]
    assert new["lemonade"]["version"] == "99.0.0"
    assert new["fastflowlm"]["version"] == "v99.0.1"
    linux = new["lemonade"]["assets"]["linux-x64"]
    assert linux["name"] == "lemonade-embeddable-99.0.0-ubuntu-x64.tar.gz"
    assert linux["size"] == len(_blob(linux["name"]))
    assert new["fastflowlm"]["assets"]["windows-x64"] == {
        "name": "fastflowlm_99.0.1_windows_amd64.zip",
        "sha256": hashlib.sha256(_blob("fastflowlm_99.0.1_windows_amd64.zip")).hexdigest(),
    }
    assert pins == lp.load_pins(), "the input pins were modified"


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
    name = "fastflowlm_99.0.1_linux.tar.gz"
    releases = _releases("v1.0.0", "v99.0.1", **{name: {"digest": "sha256:" + "0" * 64}})
    with pytest.raises(RuntimeError, match = "GitHub publishes"):
        up.update_pins(lp.load_pins(), releases, _hasher)


def test_an_asset_without_a_published_digest_is_refused():
    name = "fastflowlm_99.0.1_linux.tar.gz"
    releases = _releases("v1.0.0", "v99.0.1", **{name: {"digest": None}})
    with pytest.raises(RuntimeError, match = "publishes no digest"):
        up.update_pins(lp.load_pins(), releases, _hasher)


def test_a_release_missing_an_asset_is_refused():
    release = _release("v99.0.1", {"linux-x64": "fastflowlm_99.0.1_linux.tar.gz"})
    releases = {"lemonade-sdk/lemonade": {"tag_name": "v1.0.0", "assets": []}}
    releases["ROCm/FastFlowLM"] = release
    with pytest.raises(RuntimeError, match = "windows_amd64.zip"):
        up.update_pins(lp.load_pins(), releases.__getitem__, _hasher)


def test_write_is_idempotent_and_only_limits_the_sections(tmp_path, monkeypatch):
    path = tmp_path / "pins.json"
    path.write_text(json.dumps(lp.load_pins(), indent = 2) + "\n", encoding = "utf-8")
    monkeypatch.setattr(up, "fetch_latest_release", _releases("v99.0.0", "v99.0.1"))
    monkeypatch.setattr(up, "hash_asset", _hasher)
    assert up.main(["--pins", str(path), "--check"]) == 1
    assert up.main(["--pins", str(path), "--write", "--only", "fastflowlm"]) == 0
    written = json.loads(path.read_text(encoding = "utf-8"))
    assert written["fastflowlm"]["version"] == "v99.0.1"
    assert written["lemonade"] == lp.load_pins()["lemonade"]
    lp.load_pins(path)
    before = path.read_text(encoding = "utf-8")
    assert up.main(["--pins", str(path), "--write", "--only", "fastflowlm"]) == 0
    assert path.read_text(encoding = "utf-8") == before
