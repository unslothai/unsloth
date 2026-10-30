# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The HTTP download worker overlaps independent files and still writes the same bytes; Xet stays serial."""

from __future__ import annotations

import ast
import hashlib
import json
import os
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from hub.workers import hf_download

_WORKER = _BACKEND / "hub" / "workers" / "hf_download.py"
_COMMIT = "c" * 40


def test_xet_and_unknown_transports_stay_serial():
    assert hf_download._file_workers("xet") == 1
    assert hf_download._file_workers("auto") == 1


@pytest.mark.parametrize(
    "version, restored, workers",
    [
        ((0, 36), False, 4),  # stock writer appends to the shared partial
        ((1, 17), False, 4),
        ((1, 33), True, 4),
        ((1, 33), False, 1),  # stock 1.18+ writer: a killed partial cannot resume
        ((), False, 1),  # unreadable version proves neither writer
    ],
)
def test_http_overlaps_only_when_partials_resume(monkeypatch, version, restored, workers):
    monkeypatch.setattr(hf_download, "_hub_version", lambda: version)
    monkeypatch.setattr(hf_download, "_PARTIALS_RESUMABLE", restored)
    assert hf_download._file_workers("http") == workers
    assert hf_download._file_workers("xet") == 1


def test_every_snapshot_download_call_uses_the_transport_budget():
    tree = ast.parse(_WORKER.read_text(encoding = "utf-8"))
    seen = 0
    for node in ast.walk(tree):
        values = []
        # _pinned_snapshot_download forwards its keywords to snapshot_download unchanged.
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) in (
            "snapshot_download",
            "_pinned_snapshot_download",
        ):
            values = [kw.value for kw in node.keywords if kw.arg == "max_workers"]
            if not values:
                # The dataset path passes a kwargs dict; checked below.
                assert any(kw.arg is None for kw in node.keywords)
                continue
        elif isinstance(node, ast.Dict):
            values = [
                v
                for k, v in zip(node.keys, node.values)
                if isinstance(k, ast.Constant) and k.value == "max_workers"
            ]
        for value in values:
            seen += 1
            assert ast.unparse(value) == "_file_workers(mode)"
    assert seen == 4


def _fixture() -> dict:
    files = {}
    for i in range(6):
        files[f"small_{i}.json"] = json.dumps({"i": i}).encode() * 20
    for i in range(3):
        files[f"model-0000{i + 1}-of-00003.safetensors"] = (
            hashlib.sha256(str(i).encode()).digest() * 4096
        )
    return files


def _serve(files: dict, stats: dict):
    def etag(path, data):
        if path.endswith(".safetensors"):
            return hashlib.sha256(data).hexdigest()
        return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):
            pass

        def _reply(self, head: bool):
            path = self.path.split("?", 1)[0]
            if path.startswith("/api/models/") and "/tree/" in path:
                # huggingface_hub >= 1.x lists the tree before fetching a snapshot.
                tree = []
                for name, data in files.items():
                    entry = {
                        "type": "file",
                        "path": name,
                        "size": len(data),
                        "oid": etag(name, data),
                    }
                    if name.endswith(".safetensors"):
                        entry["lfs"] = {
                            "oid": etag(name, data),
                            "size": len(data),
                            "pointerSize": 130,
                        }
                    tree.append(entry)
                body = json.dumps(tree).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                if not head:
                    self.wfile.write(body)
                return
            if path.startswith("/api/models/"):
                siblings = []
                for name, data in files.items():
                    entry = {"rfilename": name, "size": len(data), "blobId": etag(name, data)}
                    if name.endswith(".safetensors"):
                        entry["lfs"] = {
                            "sha256": etag(name, data),
                            "size": len(data),
                            "pointerSize": 130,
                        }
                    siblings.append(entry)
                body = json.dumps(
                    {"id": "fake/repo", "sha": _COMMIT, "siblings": siblings, "private": False}
                ).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                if not head:
                    self.wfile.write(body)
                return
            name = path.split("/resolve/", 1)[-1].split("/", 1)[-1]
            data = files.get(name)
            if data is None:
                self.send_response(404)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            tag = etag(name, data)
            self.send_response(200)
            self.send_header("X-Repo-Commit", _COMMIT)
            self.send_header("ETag", f'"{tag}"')
            if name.endswith(".safetensors"):
                self.send_header("X-Linked-Etag", f'"{tag}"')
                self.send_header("X-Linked-Size", str(len(data)))
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            if head:
                return
            with stats["lock"]:
                stats["gets"][name] = stats["gets"].get(name, 0) + 1
                stats["live"] += 1
                stats["peak"] = max(stats["peak"], stats["live"])
            try:
                # Hold the connection long enough that overlapping fetches are observable.
                time.sleep(0.3)
                self.wfile.write(data)
            finally:
                with stats["lock"]:
                    stats["live"] -= 1

        def do_GET(self):
            self._reply(False)

        def do_HEAD(self):
            self._reply(True)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    threading.Thread(target = server.serve_forever, daemon = True).start()
    return server


@pytest.mark.skipif(
    not __import__("hub.utils.hf_cache_state", fromlist = ["x"]).hf_partials_are_resumable(),
    reason = "partials do not resume on this cache filesystem, so the worker stays serial",
)
def test_http_worker_overlaps_files_and_writes_identical_bytes(tmp_path):
    files = _fixture()
    stats = {"lock": threading.Lock(), "gets": {}, "live": 0, "peak": 0}
    server = _serve(files, stats)
    try:
        env = dict(os.environ)
        env.pop("HF_TOKEN", None)
        env.update(
            HF_ENDPOINT = f"http://127.0.0.1:{server.server_address[1]}",
            HF_HOME = str(tmp_path / "hf"),
            HF_HUB_CACHE = str(tmp_path / "hf" / "hub"),
            UNSLOTH_STUDIO_HOME = str(tmp_path / "studio"),
            HF_HUB_DISABLE_XET = "1",
            HF_HUB_ENABLE_HF_TRANSFER = "0",
            HF_HUB_DISABLE_TELEMETRY = "1",
            HF_HUB_OFFLINE = "0",
        )
        proc = subprocess.run(
            [sys.executable, str(_WORKER), "--repo-id", "fake/repo", "--transport", "http"],
            cwd = str(_BACKEND),
            env = env,
            capture_output = True,
            text = True,
            timeout = 120,
        )
    finally:
        server.shutdown()
    assert proc.returncode == 0, proc.stderr[-2000:]
    snapshot = tmp_path / "hf" / "hub" / "models--fake--repo" / "snapshots" / _COMMIT
    for name, data in files.items():
        assert (snapshot / name).read_bytes() == data
    assert stats["gets"] == {name: 1 for name in files}
    assert stats["peak"] >= 2
    assert not list((tmp_path / "hf" / "hub" / "models--fake--repo" / "blobs").glob("*.incomplete"))
