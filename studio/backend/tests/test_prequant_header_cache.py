# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Hosted headers are cached per Hub ETag: a fresh process does one HEAD, not HEAD + two ranged GETs."""

from __future__ import annotations

import http.server
import json
import os
import struct
import threading
import types

import pytest

import core.inference.prequant_header_probe as probe


def _write_safetensors(
    path,
    meta,
    payload = 4 << 20,
):
    from core.inference.prequant_safetensors import UNSLOTH_FORMAT_KEY, UNSLOTH_METADATA_KEY

    header = {
        "__metadata__": {
            UNSLOTH_FORMAT_KEY: "unsloth-prequant-v1",
            UNSLOTH_METADATA_KEY: json.dumps(meta),
        },
        "w": {"dtype": "I8", "shape": [payload], "data_offsets": [0, payload]},
    }
    raw = json.dumps(header).encode("utf-8")
    with open(path, "wb") as handle:
        handle.write(struct.pack("<Q", len(raw)))
        handle.write(raw)
        handle.write(b"\0" * payload)
    return path


class _RangeServer:
    def __init__(self, root):
        self.root = root
        self.ranged_gets = 0
        owner = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                path = os.path.join(owner.root, self.path.lstrip("/").split("?")[0])
                size = os.path.getsize(path)
                a, _, b = self.headers.get("Range", "bytes=0-").replace("bytes=", "").partition("-")
                start, end = int(a), min(size - 1, int(b) if b else size - 1)
                owner.ranged_gets += 1
                with open(path, "rb") as handle:
                    handle.seek(start)
                    body = handle.read(end - start + 1)
                self.send_response(206)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
                self.end_headers()
                self.wfile.write(body)

        self.httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target = self.httpd.serve_forever, daemon = True)
        self.thread.start()
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture
def hub(tmp_path, monkeypatch):
    import huggingface_hub
    import utils.paths.storage_roots as roots

    root = tmp_path / "hub"
    (root / "org" / "repo").mkdir(parents = True)
    server = _RangeServer(str(root))
    state = types.SimpleNamespace(root = root, server = server, etag = "a" * 64, heads = 0)

    def hf_hub_url(
        repo_id,
        filename,
        revision = None,
        **_,
    ):
        return f"{server.url}/{repo_id}/{filename}"

    def get_hf_file_metadata(
        url,
        token = None,
        timeout = None,
        **_,
    ):
        state.heads += 1
        path = os.path.join(str(root), url[len(server.url) + 1 :])
        if not os.path.isfile(path):
            err = Exception("404")
            err.response = types.SimpleNamespace(status_code = 404)
            raise err
        return types.SimpleNamespace(
            size = os.path.getsize(path), commit_hash = "c0ffee", etag = state.etag
        )

    monkeypatch.setattr(huggingface_hub, "hf_hub_url", hf_hub_url)
    monkeypatch.setattr(huggingface_hub, "get_hf_file_metadata", get_hf_file_metadata)
    cache_dir = tmp_path / "studio_cache"
    monkeypatch.setattr(roots, "cache_root", lambda: cache_dir)
    monkeypatch.delenv(
        probe.__dict__.get("HEADER_CACHE_ENV", "UNSLOTH_DIFFUSION_HEADER_CACHE"), raising = False
    )
    _write_safetensors(
        root / "org" / "repo" / "model.safetensors", {"scheme": "fp8", "min_features": 512}
    )
    yield state
    server.close()


def _read(name = "model.safetensors"):
    return probe.read_hub_checkpoint_header("org/repo", name, timeout_seconds = 10.0)


def test_a_second_read_of_the_same_content_skips_the_ranged_gets(hub):
    first, exists = _read()
    assert exists is True and first is not None
    assert first["format"] == "unsloth-prequant-v1"
    reads = hub.server.ranged_gets
    assert reads >= 1 and first["bytes_read"] > 0

    second, exists = _read()
    assert exists is True
    # same header, one more HEAD, no range read
    assert hub.server.ranged_gets == reads
    assert hub.heads == 2
    assert second["format"] == first["format"]
    assert second["metadata"] == first["metadata"]
    assert second["container"] == first["container"]
    assert second["size"] == first["size"] and second["commit"] == "c0ffee"
    assert second["bytes_read"] == 0


def test_a_reupload_with_a_new_etag_reads_the_new_header(hub):
    _read()
    reads = hub.server.ranged_gets
    _write_safetensors(
        hub.root / "org" / "repo" / "model.safetensors", {"scheme": "fp8", "min_features": 1024}
    )
    hub.etag = "b" * 64
    header, _ = _read()
    assert hub.server.ranged_gets > reads
    assert header["metadata"]["min_features"] == 1024


def test_a_missing_file_is_still_a_definite_404(hub):
    _read()
    header, exists = _read("gone.safetensors")
    assert header is None and exists is False


def test_no_etag_never_caches(hub):
    hub.etag = None
    _read()
    reads = hub.server.ranged_gets
    _read()
    assert hub.server.ranged_gets > reads


def test_a_corrupt_cache_file_is_a_miss(hub, tmp_path):
    _read()
    (tmp_path / "studio_cache" / "prequant_header_cache.json").write_text(
        "{not json", encoding = "utf-8"
    )
    reads = hub.server.ranged_gets
    header, _ = _read()
    assert header is not None and hub.server.ranged_gets > reads


def test_the_kill_switch_reads_every_time(hub, monkeypatch):
    monkeypatch.setenv("UNSLOTH_DIFFUSION_HEADER_CACHE", "0")
    _read()
    reads = hub.server.ranged_gets
    _read()
    assert hub.server.ranged_gets > reads
    assert not (hub.root.parent / "studio_cache" / "prequant_header_cache.json").exists()
