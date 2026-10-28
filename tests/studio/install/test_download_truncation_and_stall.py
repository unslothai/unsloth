"""Installer downloads retry a body cut short or trickling, instead of hashing or waiting on it.

http.client's read(amt) returns b"" when the server closes before Content-Length, so the copy loop
used to accept a truncated archive and report it as a checksum mismatch (or, with no published
digest, hand it to the extractor). A real local server is used because that EOF behaviour is what
is under test; a BytesIO double cannot reproduce it.
"""

import importlib.util
import http.server
import sys
import threading
import time
from pathlib import Path

import pytest


PACKAGE_ROOT = Path(__file__).resolve().parents[3]


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, PACKAGE_ROOT / "studio" / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


NODE = _load("studio_install_node_prebuilt_truncation", "install_node_prebuilt.py")
LLAMA = _load("studio_install_llama_prebuilt_truncation", "install_llama_prebuilt.py")
CORE = sys.modules["prebuilt_core"]

PAYLOAD = bytes(range(256)) * 4096  # 1 MiB


class _Server:
    """Serves PAYLOAD with its full Content-Length, closing early for the first `short` requests."""

    def __init__(self, short: int) -> None:
        self.short = short
        self.requests = 0
        server = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):  # noqa: N802
                server.requests += 1
                self.send_response(200)
                self.send_header("Content-Length", str(len(PAYLOAD)))
                self.end_headers()
                cut = server.requests <= server.short
                self.wfile.write(PAYLOAD[: len(PAYLOAD) // 4] if cut else PAYLOAD)
                self.wfile.flush()
                self.close_connection = True

            def log_message(self, *args):
                pass

        self.httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}/node.tar.gz"
        self.thread = threading.Thread(target = self.httpd.serve_forever, daemon = True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture(autouse = True)
def _no_backoff(monkeypatch):
    monkeypatch.setattr(NODE, "sleep_backoff", lambda *a, **k: None)
    monkeypatch.setattr(CORE, "sleep_backoff", lambda *a, **k: None)


def test_a_truncated_body_is_retryable_not_accepted():
    with _Server(short = 1) as srv:
        import urllib.request
        with urllib.request.urlopen(srv.url, timeout = 10) as response:
            sink = bytearray()
            with pytest.raises(CORE.TruncatedDownload) as info:
                CORE.copy_response_body(
                    response, sink.extend, total_bytes = int(response.headers["Content-Length"])
                )
    assert info.value.received == len(PAYLOAD) // 4
    assert info.value.expected == len(PAYLOAD)
    assert NODE.is_retryable_url_error(info.value)
    assert CORE.is_retryable_url_error(info.value)


@pytest.mark.parametrize("which", ["node", "llama"])
def test_download_file_reconnects_after_a_truncated_body(tmp_path, which):
    download_file = NODE.download_file if which == "node" else LLAMA.download_file
    with _Server(short = 2) as srv:
        destination = tmp_path / "node.tar.gz"
        download_file(srv.url, destination)
        assert srv.requests == 3
    assert destination.read_bytes() == PAYLOAD


def test_download_bytes_reconnects_after_a_truncated_body():
    with _Server(short = 1) as srv:
        data = LLAMA.download_bytes(srv.url)
        assert srv.requests == 2
    assert data == PAYLOAD


def test_a_body_that_never_finishes_short_is_still_refused(tmp_path):
    """Every attempt truncated: the error surfaces, and no partial file is left behind."""
    with _Server(short = 99) as srv:
        destination = tmp_path / "node.tar.gz"
        with pytest.raises(CORE.TruncatedDownload):
            NODE.download_file(srv.url, destination)
        assert srv.requests == NODE.HTTP_FETCH_ATTEMPTS
    assert not destination.exists()
    assert list(tmp_path.iterdir()) == []


class _Trickle:
    """A response that keeps sending tiny chunks, so no single read ever times out."""

    def __init__(self, chunks: int, delay: float) -> None:
        self.left = chunks
        self.delay = delay

    def read(self, amt):
        if self.left == 0:
            return b""
        self.left -= 1
        time.sleep(self.delay)
        return b"x"


def test_a_trickling_body_is_abandoned(monkeypatch):
    monkeypatch.setattr(CORE, "DOWNLOAD_STALL_WINDOW_SECONDS", 0.05)
    with pytest.raises(CORE.StalledDownload) as info:
        CORE.copy_response_body(_Trickle(1000, 0.01), lambda _b: None, total_bytes = 10_000)
    assert CORE.is_retryable_url_error(info.value)
    assert NODE.is_retryable_url_error(info.value)


def test_a_slow_but_moving_body_is_not_abandoned(monkeypatch):
    monkeypatch.setattr(CORE, "DOWNLOAD_STALL_WINDOW_SECONDS", 0.05)
    monkeypatch.setattr(CORE, "DOWNLOAD_STALL_MIN_BYTES", 1)
    sink = bytearray()
    n = CORE.copy_response_body(_Trickle(20, 0.01), sink.extend, total_bytes = 20)
    assert n == 20 and bytes(sink) == b"x" * 20


def test_the_last_bytes_arriving_slowly_do_not_count_as_a_stall(monkeypatch):
    """A window that closes on the final byte has nothing left to wait for."""
    monkeypatch.setattr(CORE, "DOWNLOAD_STALL_WINDOW_SECONDS", 0.0)
    n = CORE.copy_response_body(_Trickle(1, 0.0), lambda _b: None, total_bytes = 1)
    assert n == 1
