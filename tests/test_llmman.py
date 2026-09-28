"""The ``llmman serve`` client behind oci:// model names, against a real
loopback HTTP server so the NDJSON contract is exercised without mocks."""

import http.server
import json
import os
import socketserver
import sys
import threading

import pytest
from unsloth import llmman


def _ndjson(*objs):
    return "".join(json.dumps(o) + "\n" for o in objs)


class _FakeDaemon:
    def __init__(self):
        self.version = {"version": "0.1.0", "pid": 1}
        self.pull_body = _ndjson({"status": "success"})
        self.pull_status = 200
        self.last_request = None
        daemon = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def _send(self, status, body):
                raw = body.encode()
                self.send_response(status)
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def do_GET(self):
                self._send(200, json.dumps(daemon.version))

            def do_POST(self):
                length = int(self.headers.get("Content-Length", 0))
                daemon.last_request = json.loads(self.rfile.read(length))
                self._send(daemon.pull_status, daemon.pull_body)

        self._server = socketserver.TCPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self._server.server_address[1]}"
        threading.Thread(target = self._server.serve_forever, daemon = True).start()

    def close(self):
        self._server.shutdown()
        self._server.server_close()


@pytest.fixture
def daemon():
    d = _FakeDaemon()
    yield d
    d.close()


def test_accepts_a_llmman_daemon(daemon):
    llmman.check_daemon(daemon.url)


@pytest.mark.parametrize("version", [{"hello": "world"}, ["not", "a", "dict"]])
def test_rejects_a_non_llmman_server(daemon, version):
    daemon.version = version
    with pytest.raises(RuntimeError, match = "not an llmman daemon"):
        llmman.check_daemon(daemon.url)


def test_reports_nothing_listening_actionably():
    with pytest.raises(RuntimeError, match = "llmman serve"):
        llmman.check_daemon("http://127.0.0.1:1")


def test_pull_succeeds_and_forwards_progress(daemon):
    daemon.pull_body = _ndjson(
        {"status": "pulling manifest"},
        {"status": "pulling blobs", "completed": 50, "total": 100},
        {"status": "success"},
    )
    seen = []
    llmman.pull(daemon.url, "ghcr.io/org/model:tag", lambda *a: seen.append(a))
    assert daemon.last_request == {"model": "ghcr.io/org/model:tag"}
    assert seen == [("pulling manifest", 0, 0), ("pulling blobs", 50, 100)]


def test_reports_an_in_band_error_at_http_200(daemon):
    daemon.pull_body = _ndjson({"status": "pulling"}, {"error": "unauthorized"})
    with pytest.raises(RuntimeError, match = "unauthorized"):
        llmman.pull(daemon.url, "ref")


def test_rejects_a_stream_that_ends_without_success(daemon):
    daemon.pull_body = _ndjson({"status": "pulling blobs"})
    with pytest.raises(RuntimeError, match = "without reporting success"):
        llmman.pull(daemon.url, "ref")


def test_reports_a_non_ok_status_with_its_body(daemon):
    daemon.pull_status = 400
    daemon.pull_body = '{"error":"bad request"}'
    with pytest.raises(RuntimeError, match = "HTTP 400.*bad request"):
        llmman.pull(daemon.url, "ref")


def test_tolerates_non_json_and_blank_lines(daemon):
    daemon.pull_body = "not json\n\n[1]\n" + _ndjson({"status": "success"})
    llmman.pull(daemon.url, "ref")


@pytest.mark.parametrize("value", ["oci://ghcr.io/org/model:tag", "OCI://ghcr.io/org/model:tag"])
def test_recognizes_the_oci_scheme(value):
    assert llmman.is_oci_ref(value)


@pytest.mark.parametrize(
    "value",
    [
        "unsloth/Llama-3.2-1B-Instruct",
        "ghcr.io/org/model:tag",
        "/local/path/to/model",
        "",
        None,
    ],
)
def test_leaves_every_other_model_name_alone(value):
    assert not llmman.is_oci_ref(value)
    assert llmman.maybe_resolve(value) == (value, False)


def test_strips_the_scheme_only_when_present():
    assert llmman.strip_scheme("oci://ghcr.io/org/model:tag") == "ghcr.io/org/model:tag"
    assert llmman.strip_scheme("unsloth/Llama-3.2-1B") == "unsloth/Llama-3.2-1B"


@pytest.mark.parametrize("ref", ["oci://", "oci://   "])
def test_rejects_an_empty_reference(ref):
    with pytest.raises(ValueError):
        llmman.resolve_model(ref)


@pytest.mark.parametrize(
    "host,want",
    [
        ("", "http://127.0.0.1:17434"),
        ("1.2.3.4:9999", "http://1.2.3.4:9999"),
        ("1.2.3.4", "http://1.2.3.4:17434"),
        ("http://example.com:8080/x", "http://example.com:8080"),
        ("'localhost'", "http://localhost:17434"),
        ("::1", "http://[::1]:17434"),
        ("[::1]", "http://[::1]:17434"),
        # A wildcard bind is meaningful to the server but not to a client.
        ("0.0.0.0:9999", "http://127.0.0.1:9999"),
        ("[::]:9999", "http://[::1]:9999"),
    ],
)
def test_endpoint_parsing(monkeypatch, host, want):
    monkeypatch.setenv(llmman.HOST_ENV, host)
    assert llmman.endpoint() == want


def test_parse_resolve_output(tmp_path):
    out = "noise\n" + json.dumps({"path": str(tmp_path), "format": "safetensors"})
    assert llmman.parse_resolve_output(out, "ref") == str(tmp_path)


@pytest.mark.parametrize(
    "out", ["", "not json", "[]", '{"path": ""}', '{"path": "/does/not/exist"}']
)
def test_parse_resolve_output_rejects_bad_output(out):
    with pytest.raises(RuntimeError, match = "llmman resolve"):
        llmman.parse_resolve_output(out, "ref")


def test_missing_binary_is_actionable(monkeypatch):
    monkeypatch.setenv(llmman.BIN_ENV, "definitely-not-llmman-xyz")
    with pytest.raises(RuntimeError, match = llmman.BIN_ENV):
        llmman.resolve("ref")


@pytest.fixture
def fake_llmman_bin(tmp_path, monkeypatch):
    """An executable standing in for ``llmman resolve --no-pull``."""
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    script = tmp_path / "fake_llmman.py"
    script.write_text(
        "import json, sys\n"
        "assert sys.argv[1:3] == ['resolve', '--no-pull'], sys.argv\n"
        f"print(json.dumps({{'reference': sys.argv[3], 'path': {str(model_dir)!r}}}))\n"
    )
    if os.name == "nt":
        launcher = tmp_path / "fake_llmman.bat"
        launcher.write_text(f'@"{sys.executable}" "{script}" %*\n')
    else:
        launcher = tmp_path / "fake_llmman"
        launcher.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{script}" "$@"\n')
        launcher.chmod(0o755)
    monkeypatch.setenv(llmman.BIN_ENV, str(launcher))
    return str(model_dir)


def test_maybe_resolve_end_to_end(daemon, fake_llmman_bin, monkeypatch):
    monkeypatch.setenv(llmman.HOST_ENV, daemon.url)
    got = llmman.maybe_resolve("oci://ghcr.io/org/model:tag")
    assert got == (fake_llmman_bin, True)
    assert daemon.last_request == {"model": "ghcr.io/org/model:tag"}
