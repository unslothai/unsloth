# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Test archive integrity, atomic installation, reuse and FastFlowLM pins with a local server.

Since #18554 the default install path downloads the pinned FastFlowLM directly, with no Lemonade;
UNSLOTH_NPU_LEMONADE=1 keeps the lemond path. Both are covered here, each pinned explicitly, so a
change of default can never leave one of them tested against the other's assets."""

from __future__ import annotations

import hashlib
import io
import json
import sys
import tarfile
import threading
import zipfile
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

STUDIO_DIR = Path(__file__).resolve().parents[3] / "studio"
if str(STUDIO_DIR) not in sys.path:
    sys.path.insert(0, str(STUDIO_DIR))

import install_lemonade_prebuilt as lp  # noqa: E402


def _archive(directory: Path, key: str) -> Path:
    top = "lemonade-embeddable-9.9.9"
    binary = "lemond.exe" if key == "windows-x64" else "lemond"
    versions = json.dumps({"flm": {"npu": "v0.0.1"}, "checksums": {"github": {"other/repo": {}}}})
    files = {
        f"{top}/{binary}": b"#!/bin/sh\n",
        f"{top}/resources/backend_versions.json": versions.encode(),
    }
    if key == "windows-x64":
        path = directory / "lemonade.zip"
        with zipfile.ZipFile(path, "w") as archive:
            for name, data in files.items():
                archive.writestr(name, data)
    else:
        path = directory / "lemonade.tar.gz"
        with tarfile.open(path, "w:gz") as archive:
            for name, data in files.items():
                info = tarfile.TarInfo(name)
                info.size = len(data)
                info.mode = 0o755
                archive.addfile(info, io.BytesIO(data))
    return path


def _flm_archive(directory: Path, key: str) -> Path:
    binary = "flm.exe" if key == "windows-x64" else "flm"
    if key == "windows-x64":
        path = directory / "fastflowlm.zip"
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr(binary, b"flm\n")
    else:
        path = directory / "fastflowlm.tar.gz"
        with tarfile.open(path, "w:gz") as archive:
            info = tarfile.TarInfo(binary)
            info.size = 4
            info.mode = 0o755
            archive.addfile(info, io.BytesIO(b"flm\n"))
    return path


@pytest.fixture
def lemond_path(monkeypatch):
    """Opt in to the lemond install path (the default runs FastFlowLM directly)."""
    monkeypatch.setenv("UNSLOTH_NPU_LEMONADE", "1")
    assert lp.direct_flm() is False


@pytest.fixture
def direct_path(monkeypatch):
    """The default install path: the pinned FastFlowLM, no Lemonade."""
    monkeypatch.delenv("UNSLOTH_NPU_LEMONADE", raising = False)
    assert lp.direct_flm() is True


@pytest.fixture
def served(tmp_path, monkeypatch):
    key = lp.host_asset_key()
    if key is None:
        pytest.skip("no Lemonade build for this platform")
    archive = _archive(tmp_path, key)
    sha = hashlib.sha256(archive.read_bytes()).hexdigest()
    flm_archive = _flm_archive(tmp_path, key)
    flm_sha = hashlib.sha256(flm_archive.read_bytes()).hexdigest()
    httpd = ThreadingHTTPServer(
        ("127.0.0.1", 0), partial(SimpleHTTPRequestHandler, directory = str(tmp_path))
    )
    httpd.requests = 0
    original = httpd.finish_request

    def counted(*args, **kwargs):
        httpd.requests += 1
        return original(*args, **kwargs)

    httpd.finish_request = counted
    thread = threading.Thread(target = httpd.serve_forever, daemon = True)
    thread.start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    monkeypatch.setattr(
        lp.core, "release_asset_download_url", lambda repo, tag, name: f"{base}/{name}"
    )

    def pins(
        digest: str = sha,
        flm_version: str = "v1.0.3",
        flm_digest: str = flm_sha,
    ) -> Path:
        path = tmp_path / "pins.json"
        path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "lemonade": {
                        "version": "9.9.9",
                        "repo": "lemonade-sdk/lemonade",
                        "assets": {key: {"name": archive.name, "sha256": digest}},
                    },
                    "fastflowlm": {
                        "version": flm_version,
                        "repo": "ROCm/FastFlowLM",
                        "assets": {key: {"name": flm_archive.name, "sha256": flm_digest}},
                    },
                }
            )
        )
        return path

    pins.flm_name, pins.flm_sha = flm_archive.name, flm_sha
    yield pins, httpd
    httpd.shutdown()
    httpd.server_close()


def test_install_verifies_and_pins_fastflowlm(tmp_path, served, lemond_path):
    pins, _ = served
    path = pins()
    root = tmp_path / "root"
    binary = lp.install(root, pins_path = path)
    assert binary == lp.installed_lemond(root, lp.load_pins(path))
    versions = json.loads((binary.parent / "resources" / "backend_versions.json").read_text())
    assert versions["flm"]["npu"] == "v1.0.3"
    assert versions["checksums"]["github"]["ROCm/FastFlowLM"]["v1.0.3"] == {
        pins.flm_name: "sha256:" + pins.flm_sha
    }
    assert "other/repo" in versions["checksums"]["github"]
    assert lp.installed_version(binary) == "9.9.9"
    assert lp.installed_version(tmp_path / "elsewhere" / "lemond") is None


def test_a_complete_install_is_not_downloaded_again(tmp_path, served, lemond_path):
    pins, httpd = served
    path = pins()
    root = tmp_path / "root"
    lp.install(root, pins_path = path)
    before = httpd.requests
    lp.install(root, pins_path = path)
    assert httpd.requests == before


def test_a_fastflowlm_only_pin_bump_repins_lemond(tmp_path, served, lemond_path):
    pins, _ = served
    root = tmp_path / "root"
    lp.install(root, pins_path = pins())
    bumped = pins(flm_version = "v1.0.5")
    assert lp.installed_lemond(root, lp.load_pins(bumped)) is None
    binary = lp.install(root, pins_path = bumped)
    versions = json.loads((binary.parent / "resources" / "backend_versions.json").read_text())
    assert versions["flm"]["npu"] == "v1.0.5"


def test_a_digest_mismatch_installs_nothing(tmp_path, served, lemond_path):
    pins, _ = served
    path = pins("00" * 32)
    root = tmp_path / "root"
    with pytest.raises(lp.core.ReleaseIntegrityError):
        lp.install(root, pins_path = path)
    assert lp.installed_lemond(root, lp.load_pins(path)) is None
    assert not (root / "9.9.9").exists()


def test_a_cancelled_download_installs_nothing(tmp_path, served, lemond_path):
    pins, _ = served
    path = pins()
    root = tmp_path / "root"
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(lp.LemonadeInstallCancelled):
        lp.install(root, pins_path = path, cancel = cancel)
    assert lp.installed_lemond(root, lp.load_pins(path)) is None


def test_the_shipped_pins_cover_both_npu_platforms():
    pins = lp.load_pins()
    for section in ("lemonade", "fastflowlm"):
        assert set(pins[section]["assets"]) == {"linux-x64", "windows-x64"}
        for asset in pins[section]["assets"].values():
            assert len(asset["sha256"]) == 64
    assert pins["fastflowlm"]["version"].startswith("v")


def test_the_default_installs_the_pinned_fastflowlm_without_lemonade(tmp_path, served, direct_path):
    pins, httpd = served
    path = pins()
    root = tmp_path / "root"
    host = lp.install(root, pins_path = path)
    assert host == lp.FLM_HOST == lp.installed_lemond(root, lp.load_pins(path))
    target = lp.flm_dir(root)
    marker = json.loads((target / lp.FLM_MARKER_NAME).read_text())
    assert marker == {"version": "v1.0.3", "asset": pins.flm_name, "sha256": pins.flm_sha}
    assert not (root / "9.9.9").exists(), "the direct path must not install Lemonade"
    before = httpd.requests
    lp.install(root, pins_path = path)
    assert httpd.requests == before, "a complete FastFlowLM install was downloaded again"


def test_a_fastflowlm_digest_mismatch_installs_nothing(tmp_path, served, direct_path):
    pins, _ = served
    path = pins(flm_digest = "00" * 32)
    root = tmp_path / "root"
    with pytest.raises(lp.core.ReleaseIntegrityError):
        lp.install(root, pins_path = path)
    assert lp.installed_flm_host(root, lp.load_pins(path)) is None
    assert not lp.flm_dir(root).exists()


def test_a_cancelled_fastflowlm_download_installs_nothing(tmp_path, served, direct_path):
    pins, _ = served
    path = pins()
    root = tmp_path / "root"
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(lp.LemonadeInstallCancelled):
        lp.install(root, pins_path = path, cancel = cancel)
    assert lp.installed_flm_host(root, lp.load_pins(path)) is None
