# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Exercise authenticated release fetches through real loopback redirects."""

import json
import os
import shutil
import ssl
import subprocess
import sys
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

STUDIO = Path(__file__).resolve().parents[3] / "studio"
sys.path.insert(0, str(STUDIO))
sys.path.insert(0, str(STUDIO / "backend"))

import install_sd_cpp_prebuilt as sd
import prebuilt_core
from utils import llama_cpp_changelog
from utils.prebuilt import freshness_flow

TOKEN = "Bearer issue-11103-test-token"


@pytest.fixture(scope = "module")
def tls_certificate(tmp_path_factory):
    openssl = shutil.which("openssl")
    if openssl is None:
        pytest.skip("loopback TLS tests need openssl to generate a test certificate")
    root = tmp_path_factory.mktemp("redirect-tls")
    cert, key = root / "cert.pem", root / "key.pem"
    subprocess.run(
        [
            openssl,
            "req",
            "-x509",
            "-newkey",
            "rsa:2048",
            "-nodes",
            "-days",
            "1",
            "-subj",
            "/CN=localhost",
            "-addext",
            "subjectAltName=DNS:localhost,IP:127.0.0.1",
            "-keyout",
            str(key),
            "-out",
            str(cert),
        ],
        check = True,
        capture_output = True,
    )
    return cert, key


@pytest.fixture
def stdlib_ssl():
    """Run on the stdlib ``ssl``: importing ``prebuilt_core`` injects truststore on
    macOS and Windows, whose SSLContext ignores SSL_CERT_FILE and cannot wrap a
    server-side socket (``get_unverified_chain`` on None in ``_verify_peercerts``).
    """
    injected = ssl.SSLContext.__module__.startswith("truststore")
    if injected:
        import truststore
        truststore.extract_from_ssl()
    try:
        yield
    finally:
        if injected:
            truststore.inject_into_ssl()


@pytest.fixture
def servers(stdlib_ssl, tls_certificate, monkeypatch):
    cert, key = tls_certificate
    monkeypatch.setenv("SSL_CERT_FILE", str(cert))
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    monkeypatch.setenv("no_proxy", "127.0.0.1,localhost")
    for handler in prebuilt_core._URL_OPENER.handlers:
        if isinstance(handler, urllib.request.HTTPSHandler):
            monkeypatch.setattr(handler, "_context", ssl.create_default_context(cafile = str(cert)))
    running = []

    def start(*, tls = True, payload = None):
        class Handler(BaseHTTPRequestHandler):
            timeout = 3

            def do_GET(self):
                self.server.seen.append((self.path, self.headers.get("Authorization")))
                location = self.server.redirects.get(self.path)
                self.send_response(self.server.code if location else 200)
                if location:
                    self.send_header("Location", location)
                body = json.dumps(payload).encode()
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                if self.command != "HEAD":
                    self.wfile.write(body)

            do_HEAD = do_GET  # without it the stdlib 501s and HEAD tests the error path

            def log_message(self, *args):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        server.seen, server.redirects, server.code = [], {}, 302
        if tls:
            context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            context.load_cert_chain(cert, key)
            server.socket = context.wrap_socket(server.socket, server_side = True)
        server.url = f"{'https' if tls else 'http'}://127.0.0.1:{server.server_port}"
        thread = threading.Thread(target = server.serve_forever, kwargs = {"poll_interval": 0.01})
        thread.start()
        running.append((server, thread))
        return server

    yield start
    for server, thread in reversed(running):
        server.shutdown()
        server.server_close()
        thread.join(timeout = 5)
        assert not thread.is_alive()


def fetch(client, url, monkeypatch):
    monkeypatch.setenv("GH_TOKEN", TOKEN.removeprefix("Bearer "))
    monkeypatch.delenv("GITHUB_TOKEN", raising = False)
    original_request = urllib.request.Request

    class LocalRequest(original_request):
        def __init__(self, request_url, *args, **kwargs):
            if request_url.startswith("https://api.github.com/"):
                request_url = url
            super().__init__(request_url, *args, **kwargs)

    monkeypatch.setattr(urllib.request, "Request", LocalRequest)
    if client == "prebuilt":
        request = original_request(url, headers = {"Authorization": TOKEN})
        with prebuilt_core._URL_OPENER.open(request, timeout = 3) as response:
            return json.load(response)
    if client == "freshness":
        return freshness_flow._fetch_newest_published_release_blocking(
            "owner/repo", 3, log_message = "redirect test"
        )
    if client == "changelog":
        return llama_cpp_changelog._fetch_release_blocking("owner/repo", "v1", 3)
    return sd._fetch_release("v1", repo = "owner/repo", timeout = 3)


@pytest.mark.parametrize("client", ["prebuilt", "freshness", "changelog", "sd"])
@pytest.mark.parametrize("code", [301, 302, 303, 307, 308])
@pytest.mark.parametrize(
    "target", ["same_origin", "other_host", "other_port", "return", "downgrade"]
)
def test_release_redirect_credentials(client, code, target, servers, monkeypatch):
    release = {"tag_name": "v1", "published_at": "2026-01-01T00:00:00Z"}
    payload = [release] if client == "freshness" else release
    source = servers(payload = payload)
    source.code = code
    destination = (
        source if target == "same_origin" else servers(tls = target != "downgrade", payload = payload)
    )
    destination_url = destination.url
    if target == "other_host":
        destination_url = destination_url.replace("127.0.0.1", "localhost")
    source.redirects["/start"] = destination_url + "/final"
    if target == "return":
        destination.redirects["/final"] = source.url + "/back"
    if target == "downgrade" or (code == 308 and sys.version_info < (3, 11)):
        if client in ("prebuilt", "sd"):
            with pytest.raises(urllib.error.HTTPError) as error:
                fetch(client, source.url + "/start", monkeypatch)
            assert error.value.code == code
            error.value.close()
        else:
            assert fetch(client, source.url + "/start", monkeypatch) is None
        assert all(path != "/final" for path, _ in destination.seen)
    else:
        assert fetch(client, source.url + "/start", monkeypatch) == release
        expected_token = TOKEN if target == "same_origin" else None
        assert ("/final", expected_token) in destination.seen
        if target == "return":
            assert ("/back", None) in source.seen
    assert source.seen[0] == ("/start", TOKEN)


@pytest.mark.parametrize(
    "start,target,expected",
    [
        ("https://hub.example/a", "https://hub.example:443/b", TOKEN),
        ("http://hub.example/a", "http://hub.example:80/b", TOKEN),
        ("https://HUB.example/a", "https://hub.example/b", TOKEN),
        ("http://hub.example:8443/a", "https://hub.example:8443/b", None),
        ("https://hub.example/a", "https://hub.example:0/b", None),
    ],
)
def test_prebuilt_origin_comparison(start, target, expected):
    request = urllib.request.Request(
        start,
        headers = {
            "Authorization": TOKEN,
            "Accept": "application/json",
            "User-Agent": "redirect-test",
        },
    )
    result = prebuilt_core._CrossHostAuthStrippingRedirectHandler().redirect_request(
        request, None, 302, "Found", {}, target
    )
    assert result.get_header("Authorization") == expected
    assert request.get_header("Authorization") == TOKEN
    assert result.get_header("Accept") == "application/json"
    assert result.get_header("User-agent") == "redirect-test"


@pytest.mark.parametrize("name", ["prebuilt_core", "install_sd_cpp_prebuilt"])
@pytest.mark.parametrize("mode", ["script", "package"])
def test_installer_import_without_backend_dependencies(name, mode, tmp_path):
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["UNSLOTH_STUDIO_NATIVE_TLS"] = "0"
    if mode == "script":
        args = [str(STUDIO / f"{name}.py"), "--help"]
    else:
        args = [
            "-c",
            f"import sys; sys.path.insert(0, {str(STUDIO.parent)!r}); import studio.{name}",
        ]
    result = subprocess.run(
        [sys.executable, "-S", *args],
        cwd = tmp_path,
        env = env,
        capture_output = True,
        text = True,
        timeout = 10,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    "start,target",
    [
        ("https://hub.example/a", "https://hub.example:99999/b"),
        ("https://hub.example/a", "https://hub.example:abc/b"),
        # Both ends unreadable: the sentinel must not compare equal to itself.
        ("https://hub.example:abc/a", "https://hub.example:abc/b"),
    ],
)
def test_an_unparseable_redirect_port_strips_rather_than_raising(start, target):
    """No client catches ValueError, so raising turns a soft "no release info" into an
    uncaught exception on an update check. Unreadable target, other origin, strip."""
    request = urllib.request.Request(
        start, headers = {"Authorization": TOKEN, "Accept": "application/json"}
    )
    result = prebuilt_core._CrossHostAuthStrippingRedirectHandler().redirect_request(
        request, None, 302, "Found", {}, target
    )
    assert result.get_header("Authorization") is None
    assert result.get_header("Accept") == "application/json"
    assert request.get_header("Authorization") == TOKEN


@pytest.mark.parametrize("client", ["prebuilt", "freshness", "changelog", "sd"])
def test_an_unparseable_redirect_port_stays_soft_for_every_client(client, servers, monkeypatch):
    """The same case end to end: no ValueError reaches the caller."""
    release = {"tag_name": "v1", "published_at": "2026-01-01T00:00:00Z"}
    source = servers(payload = [release] if client == "freshness" else release)
    source.redirects["/start"] = "https://127.0.0.1:99999/final"
    # No ValueError arm: it must not be raised, so letting it escape fails the test.
    try:
        fetch(client, source.url + "/start", monkeypatch)
    except (urllib.error.HTTPError, urllib.error.URLError, OSError) as error:
        getattr(error, "close", lambda: None)()
    assert source.seen[0] == ("/start", TOKEN)


def test_a_downgrade_is_refused_part_way_down_a_chain(servers):
    """https -> https -> http. Every other case here is one hop from the request, so a
    policy comparing only the ORIGINAL request URL would pass them all and still leak."""
    source = servers(payload = {"tag_name": "v1"})
    plaintext = servers(tls = False, payload = {"tag_name": "v1"})
    source.redirects["/start"] = source.url + "/two"
    source.redirects["/two"] = plaintext.url + "/final"

    request = urllib.request.Request(source.url + "/start", headers = {"Authorization": TOKEN})
    with pytest.raises(urllib.error.HTTPError) as error:
        prebuilt_core._URL_OPENER.open(request, timeout = 3)
    error.value.close()
    assert [path for path, _ in source.seen] == ["/start", "/two"]
    assert plaintext.seen == []


@pytest.mark.parametrize("method", ["GET", "HEAD"])
@pytest.mark.parametrize("cross_origin", [False, True])
def test_the_method_and_a_signed_query_survive_the_hop(method, cross_origin, servers):
    """routes/training.py preflights with HEAD, and GitHub and Hugging Face redirect to
    a CDN URL whose credentials are in the query: mangling it reads as an auth failure."""
    source = servers(payload = {"tag_name": "v1"})
    destination = servers(payload = {"tag_name": "v1"}) if cross_origin else source
    target = destination.url
    if cross_origin:
        target = target.replace("127.0.0.1", "localhost")
    source.redirects["/start"] = f"{target}/final?sig=abc123&exp=99"

    request = urllib.request.Request(
        source.url + "/start", headers = {"Authorization": TOKEN}, method = method
    )
    with prebuilt_core._URL_OPENER.open(request, timeout = 3) as response:
        assert response.status == 200
    landed = [(path, token) for path, token in destination.seen if path.startswith("/final")]
    assert landed, "the redirect was not followed"
    path, token = landed[0]
    assert path == "/final?sig=abc123&exp=99"
    assert token == (None if cross_origin else TOKEN)


@pytest.mark.parametrize("unredirected", [False, True])
def test_cross_origin_does_not_copy_unredirected_auth(unredirected):
    request = urllib.request.Request("https://hub.example/a")
    if unredirected:
        request.add_unredirected_header("Authorization", TOKEN)
    result = prebuilt_core._CrossHostAuthStrippingRedirectHandler().redirect_request(
        request, None, 302, "Found", {}, "https://other.example/b"
    )
    assert result.get_header("Authorization") is None
    assert request.get_header("Authorization") == (TOKEN if unredirected else None)
