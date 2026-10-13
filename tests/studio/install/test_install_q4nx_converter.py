# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Test pin verification, atomic install and reuse of the Q4NX converter with a local server."""

from __future__ import annotations

import hashlib
import io
import json
import re
import sys
import tarfile
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

STUDIO_DIR = Path(__file__).resolve().parents[3] / "studio"
if str(STUDIO_DIR) not in sys.path:
    sys.path.insert(0, str(STUDIO_DIR))

import install_q4nx_converter as qc  # noqa: E402

COMMIT = "c" * 40


def _archive(directory: Path) -> Path:
    top = f"FLM_Q4NX_Converter-{COMMIT}"
    files = {f"{top}/convert.py": b"print('ok')\n", f"{top}/q4nx/__init__.py": b""}
    path = directory / "converter.tar.gz"
    with tarfile.open(path, "w:gz") as archive:
        for name, data in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
    return path


@pytest.fixture
def served(tmp_path):
    archive = _archive(tmp_path)
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
    url = f"http://127.0.0.1:{httpd.server_address[1]}/{archive.name}"

    def pins(digest: str = sha, commit: str = COMMIT) -> Path:
        path = tmp_path / "pins.json"
        pin = {"commit": commit, "archive_sha256": digest, "archive_size": archive.stat().st_size}
        path.write_text(
            json.dumps(
                {
                    "schema_version": 2,
                    "repo": "ROCm/FLM_Q4NX_Converter",
                    "converters": {"q4nx": pin, "q4k": {**pin, "commit": "e" * 40}},
                    "q4k_architectures": ["qwen35", "qwen35moe"],
                }
            )
        )
        return path

    yield pins, url, httpd
    httpd.shutdown()
    httpd.server_close()


def test_install_verifies_and_marks(tmp_path, served):
    pins, url, _ = served
    path = pins()
    script = qc.install(tmp_path / "root", pins_path = path, url = url)
    assert script == qc.installed_converter(tmp_path / "root", qc.load_pins(path))
    assert script.parent.name == COMMIT[:12]
    assert (script.parent / "q4nx" / "__init__.py").is_file()


def test_a_complete_install_is_not_downloaded_again(tmp_path, served):
    pins, url, httpd = served
    path = pins()
    qc.install(tmp_path / "root", pins_path = path, url = url)
    before = httpd.requests
    qc.install(tmp_path / "root", pins_path = path, url = url)
    assert httpd.requests == before


def test_digest_mismatch_installs_nothing(tmp_path, served):
    pins, url, _ = served
    path = pins(digest = "0" * 64)
    with pytest.raises(qc.core.ReleaseIntegrityError):
        qc.install(tmp_path / "root", pins_path = path, url = url)
    assert qc.installed_converter(tmp_path / "root", qc.load_pins(path)) is None


def test_a_new_pin_replaces_the_install(tmp_path, served):
    pins, url, _ = served
    qc.install(tmp_path / "root", pins_path = pins(), url = url)
    repinned = qc.load_pins(pins(commit = "d" * 40))
    assert qc.installed_converter(tmp_path / "root", repinned) is None


def test_each_named_converter_installs_into_its_own_commit_dir(tmp_path, served):
    pins, url, _ = served
    path = pins()
    q4nx = qc.install(tmp_path / "root", name = "q4nx", pins_path = path, url = url)
    q4k = qc.install(tmp_path / "root", name = "q4k", pins_path = path, url = url)
    assert (q4nx.parent.name, q4k.parent.name) == (COMMIT[:12], "e" * 12)


def test_only_qwen35_architectures_take_the_q4k_converter(tmp_path, served):
    pins, _, _ = served
    path = pins()
    assert qc.converter_for_architecture("qwen35", path) == "q4k"
    assert qc.converter_for_architecture("qwen35moe", path) == "q4k"
    for architecture in ("qwen3", "llama", "gemma3", "lfm2", "phi3", "qwen2", ""):
        assert qc.converter_for_architecture(architecture, path) == "q4nx"


@pytest.mark.parametrize("name", ["q4nx", "q4k"])
def test_shipped_pins_are_full_commits_and_digests(name):
    pins = qc.load_pins(name = name)
    assert pins["repo"] == "ROCm/FLM_Q4NX_Converter"
    assert re.fullmatch(r"[0-9a-f]{40}", pins["commit"])
    assert re.fullmatch(r"[0-9a-f]{64}", pins["archive_sha256"])
    assert qc.archive_url(pins).endswith(f"/tar.gz/{pins['commit']}")


def test_shipped_q4nx_pin_predates_the_q4_1_repack():
    # d1d5232 re-packed Q4_0 / Q4_1 in an order FastFlowLM's aie2p engines read as noise; the
    # q4nx pin must stay on the commit that reproduces Qwen3-0.6B-NPU2/model.q4nx.
    assert qc.load_pins(name = "q4nx")["commit"].startswith("dd0993c")
    assert qc.load_pins(name = "q4k")["commit"].startswith("d1d5232")
