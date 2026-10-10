# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Resumable FastFlowLM file downloads against a local server that can drop connections."""

from __future__ import annotations

import hashlib
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from core.inference import flm_files as ff

WEIGHTS = os.urandom(3 * (1 << 20) + 123)
CONFIG = b'{"flm_version": "1.0.3"}'


class _Files:
    """Serves WEIGHTS and CONFIG, honouring Range unless told not to."""

    def __init__(self) -> None:
        self.ranges: list[str | None] = []
        # Bytes each successive response sends before closing the connection; None sends it all.
        self.cut_after: list[int | None] = []
        self.ignore_range = False
        # Answers a Range request with 206 from byte 0, as a broken cache might.
        self.wrong_range = False
        self.status: int | None = None
        files = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args) -> None:
                return

            def do_GET(self) -> None:  # noqa: N802
                name = self.path.split("?")[0].rsplit("/", 1)[-1]
                body = {"model.q4nx": WEIGHTS, "config.json": CONFIG}.get(name)
                if files.status is not None or body is None:
                    self.send_response(files.status or 404)
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                requested = self.headers.get("Range")
                files.ranges.append(requested)
                start = 0
                if requested and not files.ignore_range:
                    start = (
                        0
                        if files.wrong_range
                        else int(requested.removeprefix("bytes=").rstrip("-"))
                    )
                    self.send_response(206)
                    self.send_header("Content-Range", f"bytes {start}-{len(body) - 1}/{len(body)}")
                else:
                    self.send_response(200)
                self.send_header("Content-Length", str(len(body) - start))
                self.end_headers()
                cut = files.cut_after.pop(0) if files.cut_after else None
                self.wfile.write(body[start:] if cut is None else body[start : start + cut])
                if cut is not None:
                    self.close_connection = True

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target = self.server.serve_forever, daemon = True).start()
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}/FastFlowLM/Tiny-NPU2"

    def model(
        self,
        folder: Path,
        weights_digest: str | None = None,
    ) -> ff.FlmModelFiles:
        blob = hashlib.sha1(f"blob {len(CONFIG)}\0".encode() + CONFIG).hexdigest()
        return ff.FlmModelFiles(
            folder = folder,
            files = (
                ff.FlmFile("config.json", f"{self.url}/config.json", len(CONFIG), blob, False),
                ff.FlmFile(
                    "model.q4nx",
                    f"{self.url}/model.q4nx",
                    len(WEIGHTS),
                    weights_digest or hashlib.sha256(WEIGHTS).hexdigest(),
                    True,
                ),
            ),
        )


@pytest.fixture
def files(monkeypatch):
    monkeypatch.setattr(ff, "_RETRY_DELAY_SECONDS", 0.0)
    server = _Files()
    yield server
    server.server.shutdown()
    server.server.server_close()


def _percents(events) -> list[int]:
    return [event["percent"] for event in events]


def test_files_land_whole_with_progress_over_the_model(files, tmp_path):
    events = list(ff.download_files(files.model(tmp_path)))
    assert (tmp_path / "model.q4nx").read_bytes() == WEIGHTS
    assert (tmp_path / "config.json").read_bytes() == CONFIG
    assert not list(tmp_path.glob("*.partial"))
    percents = _percents(events)
    assert percents == sorted(percents) and percents[-1] == 99
    # One event per percent, over the whole model; lemond's pull then brings the 100.
    assert len(events) <= 100 + 2
    assert {event["bytes_total"] for event in events} == {len(WEIGHTS) + len(CONFIG)}


def test_a_dropped_connection_continues_from_the_byte_it_reached(files, tmp_path):
    files.cut_after = [None, 1 << 20, 5000]  # config.json whole, then the weights in three goes
    list(ff.download_files(files.model(tmp_path)))
    assert (tmp_path / "model.q4nx").read_bytes() == WEIGHTS
    assert files.ranges == [None, None, f"bytes={1 << 20}-", f"bytes={(1 << 20) + 5000}-"]


def test_an_interrupted_download_resumes_in_the_next_one(files, tmp_path):
    model = files.model(tmp_path)
    downloading = ff.download_files(model)
    for event in downloading:
        if event["file"] == "model.q4nx" and event["bytes_downloaded"] > len(CONFIG) + (1 << 20):
            break
    downloading.close()
    kept = (tmp_path / "model.q4nx.partial").stat().st_size
    assert not (tmp_path / "model.q4nx").exists()
    assert model.resume_percent() == (len(CONFIG) + kept) * 100 // model.total

    events = list(ff.download_files(model))
    assert files.ranges[-1] == f"bytes={kept}-"
    assert events[0]["percent"] == (len(CONFIG) + kept) * 100 // model.total
    assert (tmp_path / "model.q4nx").read_bytes() == WEIGHTS


def test_a_prefix_fastflowlm_left_under_the_final_name_is_continued(files, tmp_path):
    # A killed `flm pull` leaves this, and FastFlowLM would then take the model as complete.
    (tmp_path / "config.json").write_bytes(CONFIG)
    (tmp_path / "model.q4nx").write_bytes(WEIGHTS[:4096])
    list(ff.download_files(files.model(tmp_path)))
    assert files.ranges == ["bytes=4096-"]
    assert (tmp_path / "model.q4nx").read_bytes() == WEIGHTS


def test_a_server_that_ignores_range_sends_the_file_again(files, tmp_path):
    (tmp_path / "model.q4nx.partial").write_bytes(WEIGHTS[:4096])
    files.ignore_range = True
    list(ff.download_files(files.model(tmp_path)))
    assert (tmp_path / "model.q4nx").read_bytes() == WEIGHTS


def test_refetching_the_same_prefix_is_not_progress(files, tmp_path):
    # A server that ignores Range and drops every connection early restarts the file each time,
    # never passing the byte already reached; that must count as stalled, not loop forever.
    (tmp_path / "config.json").write_bytes(CONFIG)
    (tmp_path / "model.q4nx.partial").write_bytes(WEIGHTS[:4096])
    files.ignore_range = True
    files.cut_after = [1000] * 20
    with pytest.raises(ff.FlmDownloadError, match = "model.q4nx failed"):
        list(ff.download_files(files.model(tmp_path)))
    assert len(files.ranges) == ff._MAX_STALLED_ATTEMPTS


def test_a_restart_that_passes_the_furthest_byte_is_progress(files, tmp_path):
    (tmp_path / "config.json").write_bytes(CONFIG)
    (tmp_path / "model.q4nx.partial").write_bytes(WEIGHTS[:4096])
    files.ignore_range = True
    # Four stalls, a restart that gets past byte 4096, then four more stalls before it lands.
    files.cut_after = [1000] * 4 + [6000] + [1000] * 4
    list(ff.download_files(files.model(tmp_path)))
    assert len(files.ranges) == 10
    assert (tmp_path / "model.q4nx").read_bytes() == WEIGHTS


def test_a_206_from_the_wrong_byte_is_not_appended(files, tmp_path):
    (tmp_path / "model.q4nx.partial").write_bytes(WEIGHTS[:4096])
    files.wrong_range = True
    with pytest.raises(ff.FlmDownloadError, match = "asked for byte 4096"):
        list(ff.download_files(files.model(tmp_path)))
    assert (tmp_path / "model.q4nx.partial").read_bytes() == WEIGHTS[:4096]


def test_an_oversized_partial_starts_over_and_is_counted_once(files, tmp_path):
    (tmp_path / "config.json").write_bytes(CONFIG)
    (tmp_path / "model.q4nx.partial").write_bytes(WEIGHTS + b"stale tail")
    events = list(ff.download_files(files.model(tmp_path)))
    assert files.ranges == [None]
    assert max(event["bytes_downloaded"] for event in events) <= events[0]["bytes_total"]
    assert (tmp_path / "model.q4nx").read_bytes() == WEIGHTS


def test_a_complete_file_drops_a_leftover_partial(files, tmp_path):
    (tmp_path / "config.json").write_bytes(CONFIG)
    (tmp_path / "model.q4nx").write_bytes(WEIGHTS)
    (tmp_path / "model.q4nx.partial").write_bytes(WEIGHTS[:4096])
    assert list(ff.download_files(files.model(tmp_path))) == []
    assert not (tmp_path / "model.q4nx.partial").exists()


def test_a_same_sized_file_with_other_content_is_downloaded_again(files, tmp_path):
    stale = bytes([CONFIG[0] ^ 1]) + CONFIG[1:]
    (tmp_path / "config.json").write_bytes(stale)
    (tmp_path / "model.q4nx").write_bytes(WEIGHTS)
    list(ff.download_files(files.model(tmp_path)))
    assert (tmp_path / "config.json").read_bytes() == CONFIG
    assert (tmp_path / "model.q4nx").read_bytes() == WEIGHTS


def test_a_file_that_fails_its_hash_is_removed(files, tmp_path):
    with pytest.raises(ff.FlmDownloadError, match = "did not match its published hash"):
        list(ff.download_files(files.model(tmp_path, weights_digest = "0" * 64)))
    assert not (tmp_path / "model.q4nx").exists()
    assert not (tmp_path / "model.q4nx.partial").exists()


def test_a_missing_file_fails_without_retrying(files, tmp_path):
    files.status = 404
    with pytest.raises(ff.FlmDownloadError, match = "HTTP 404"):
        list(ff.download_files(files.model(tmp_path)))
    assert files.ranges == []


def test_a_download_that_stops_progressing_gives_up(files, tmp_path):
    files.status = 503
    with pytest.raises(ff.FlmDownloadError, match = "503"):
        list(ff.download_files(files.model(tmp_path)))


def test_the_file_list_comes_from_fastflowlm_manifests(tmp_path):
    flm = tmp_path / "bin" / "flm"
    flm.parent.mkdir()
    (flm.parent / "model_list.json").write_text(
        json.dumps(
            {
                "model_path": "models",
                "models": {
                    "qwen3.6-moe": {
                        "35b-a3b": {
                            "name": "Qwen3.6-35B-A3B-NPU2",
                            "url": "https://huggingface.co/FastFlowLM/Qwen3.6-35B-A3B-NPU2/resolve/flm_q4k_high_precision",
                            "files": ["config.json", "model.q4nx", "unlisted.bin"],
                        }
                    },
                    "llama3.2": {
                        "1b": {
                            "name": "Llama-3.2-1B-NPU2",
                            "url": "https://huggingface.co/FastFlowLM/Llama-3.2-1B-NPU2",
                            "files": ["config.json"],
                        }
                    },
                },
            }
        )
    )
    (flm.parent / "model_info.json").write_text(
        json.dumps(
            {
                "qwen3.6-moe:35b-a3b": [
                    {"path": "config.json", "size": 4017, "oid": "663d"},
                    {"path": "model.q4nx", "size": 21, "oid": "db25", "lfs": {"oid": "a763"}},
                ],
                "llama3.2:1b": [{"path": "config.json", "size": 10, "oid": "aa"}],
            }
        )
    )
    model = ff.read_model_files(flm, tmp_path / "flm", "qwen3.6-moe:35b-a3b")
    assert model.folder == tmp_path / "flm" / "models" / "Qwen3.6-35B-A3B-NPU2"
    assert model.files == (
        ff.FlmFile(
            "config.json",
            "https://huggingface.co/FastFlowLM/Qwen3.6-35B-A3B-NPU2/resolve/flm_q4k_high_precision/config.json?download=true",
            4017,
            "663d",
            False,
        ),
        ff.FlmFile(
            "model.q4nx",
            "https://huggingface.co/FastFlowLM/Qwen3.6-35B-A3B-NPU2/resolve/flm_q4k_high_precision/model.q4nx?download=true",
            21,
            "a763",
            True,
        ),
    )
    llama = ff.read_model_files(flm, tmp_path / "flm", "llama3.2:1b")
    assert llama.files[0].url == (
        "https://huggingface.co/FastFlowLM/Llama-3.2-1B-NPU2/resolve/main/config.json?download=true"
    )
    assert ff.read_model_files(flm, tmp_path / "flm", "gemma3:4b") is None
    assert ff.read_model_files(None, tmp_path / "flm", "llama3.2:1b") is None
