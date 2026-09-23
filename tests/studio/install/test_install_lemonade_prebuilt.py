# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Test archive integrity, atomic installation, reuse and FastFlowLM pins with a local server."""

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


@pytest.fixture
def served(tmp_path, monkeypatch):
    key = lp.host_asset_key()
    if key is None:
        pytest.skip("no Lemonade build for this platform")
    archive = _archive(tmp_path, key)
    sha = hashlib.sha256(archive.read_bytes()).hexdigest()
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

    def pins(digest: str = sha, flm_version: str = "v1.0.3") -> Path:
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
                        "assets": {key: {"name": "fastflowlm.tar.gz", "sha256": "ab" * 32}},
                    },
                }
            )
        )
        return path

    yield pins, httpd
    httpd.shutdown()
    httpd.server_close()


def test_install_verifies_and_pins_fastflowlm(tmp_path, served):
    pins, _ = served
    path = pins()
    root = tmp_path / "root"
    binary = lp.install(root, pins_path = path)
    assert binary == lp.installed_lemond(root, lp.load_pins(path))
    versions = json.loads((binary.parent / "resources" / "backend_versions.json").read_text())
    assert versions["flm"]["npu"] == "v1.0.3"
    assert versions["checksums"]["github"]["ROCm/FastFlowLM"]["v1.0.3"] == {
        "fastflowlm.tar.gz": "sha256:" + "ab" * 32
    }
    # What lemond already pinned stays.
    assert "other/repo" in versions["checksums"]["github"]


def test_a_complete_install_is_not_downloaded_again(tmp_path, served):
    pins, httpd = served
    path = pins()
    root = tmp_path / "root"
    lp.install(root, pins_path = path)
    before = httpd.requests
    lp.install(root, pins_path = path)
    assert httpd.requests == before


def test_a_fastflowlm_only_pin_bump_repins_lemond(tmp_path, served):
    pins, _ = served
    root = tmp_path / "root"
    lp.install(root, pins_path = pins())
    bumped = pins(flm_version = "v1.0.5")
    # Same Lemonade asset, so only the recorded FastFlowLM pin can tell the install is stale.
    assert lp.installed_lemond(root, lp.load_pins(bumped)) is None
    binary = lp.install(root, pins_path = bumped)
    versions = json.loads((binary.parent / "resources" / "backend_versions.json").read_text())
    assert versions["flm"]["npu"] == "v1.0.5"


def test_a_digest_mismatch_installs_nothing(tmp_path, served):
    pins, _ = served
    path = pins("00" * 32)
    root = tmp_path / "root"
    with pytest.raises(lp.core.ReleaseIntegrityError):
        lp.install(root, pins_path = path)
    assert lp.installed_lemond(root, lp.load_pins(path)) is None
    assert not (root / "9.9.9").exists()


def test_a_cancelled_download_installs_nothing(tmp_path, served):
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
