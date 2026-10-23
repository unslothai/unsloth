# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Real TLS proxy -> HTTP loopback socket -> packaged frontend and auth router."""

from contextlib import ExitStack, contextmanager
from datetime import datetime, timedelta, timezone
from http.client import HTTPConnection
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import ssl
from threading import Thread
import time

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID
from fastapi import FastAPI
import httpx
import pytest
import uvicorn

from utils.local_proxy import PROXY_ORIGIN_ENV


@contextmanager
def _backend_server(app):
    config = uvicorn.Config(app, host = "127.0.0.1", port = 0, log_level = "error", lifespan = "off")
    sock = config.bind_socket()
    server = uvicorn.Server(config)
    thread = Thread(target = server.run, kwargs = {"sockets": [sock]}, daemon = True)
    thread.start()
    try:
        deadline = time.monotonic() + 10
        while not server.started:
            if not thread.is_alive() or time.monotonic() >= deadline:
                pytest.fail("test backend did not start")
            time.sleep(0.01)
        yield sock.getsockname()[1]
    finally:
        server.should_exit = True
        thread.join(timeout = 10)
        sock.close()
        assert not thread.is_alive(), "test backend did not stop"


def _certificate(tmp_path):
    key = rsa.generate_private_key(public_exponent = 65537, key_size = 2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "localhost")])
    now = datetime.now(timezone.utc)
    certificate = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(minutes = 1))
        .not_valid_after(now + timedelta(days = 1))
        .add_extension(x509.SubjectAlternativeName([x509.DNSName("localhost")]), critical = False)
        .sign(key, hashes.SHA256())
    )
    cert_path, key_path = tmp_path / "test-cert.pem", tmp_path / "test-key.pem"
    cert_path.write_bytes(certificate.public_bytes(serialization.Encoding.PEM))
    key_path.write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    return cert_path, key_path


@pytest.mark.parametrize("root_dot", [False, True])
def test_verified_https_proxy_frontend_and_auth(tmp_path, monkeypatch, root_dot):
    import main
    from routes.auth import router

    # main enables client-only native TLS on macOS/Windows. This test proxy is
    # a TLS server; restore stdlib SSLContext just for its temporary listener.
    stdlib = next(c for c in ssl._SSLContext.__subclasses__() if c.__module__ == "ssl")
    monkeypatch.setattr(ssl, "SSLContext", stdlib)

    def forbidden(*args):
        pytest.fail("proxy HTML must not receive bootstrap credentials")

    monkeypatch.setattr(main, "_inject_bootstrap", forbidden)
    (tmp_path / "index.html").write_text("<html><head></head><body>Studio</body></html>")
    (tmp_path / "assets").mkdir()
    (tmp_path / "assets/app.js").write_text("export const ready = true;")
    app = FastAPI()
    app.include_router(router, prefix = "/api/auth")

    with ExitStack() as stack:
        backend_port = stack.enter_context(_backend_server(app))

        class Proxy(BaseHTTPRequestHandler):
            def do_GET(self):
                headers = dict(self.headers)
                headers["X-Forwarded-Host"] = self.headers["Host"]
                headers["X-Forwarded-Proto"] = "https"
                headers["X-Forwarded-For"] = "100.64.0.10"
                connection = HTTPConnection("127.0.0.1", backend_port, timeout = 5)
                try:
                    connection.request("GET", self.path, headers = headers)
                    response = connection.getresponse()
                    body = response.read()
                    self.send_response(response.status)
                    for name, value in response.getheaders():
                        if name.lower() not in {
                            "connection",
                            "transfer-encoding",
                            "content-length",
                        }:
                            self.send_header(name, value)
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                finally:
                    connection.close()

            def log_message(self, *args):
                pass

        proxy = ThreadingHTTPServer(("127.0.0.1", 0), Proxy)
        stack.callback(proxy.server_close)
        cert_path, key_path = _certificate(tmp_path)
        tls = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        tls.load_cert_chain(cert_path, key_path)
        proxy.socket = tls.wrap_socket(proxy.socket, server_side = True)
        origin = f"https://localhost:{proxy.server_port}"
        monkeypatch.setenv(PROXY_ORIGIN_ENV, origin)
        main.setup_frontend(app, tmp_path, tunnel_only = True)
        thread = Thread(target = proxy.serve_forever, daemon = True)
        thread.start()
        stack.callback(thread.join, 10)
        stack.callback(proxy.shutdown)

        verify = ssl.create_default_context(cafile = str(cert_path))
        authority = f"localhost{'.' if root_dot else ''}:{proxy.server_port}"
        with httpx.Client(base_url = origin, verify = verify, trust_env = False) as client:
            headers = {"Host": authority, "Origin": origin}
            for path in ("/", "/chat", "/index.html", "/assets/app.js"):
                response = client.get(path, headers = headers)
                assert response.status_code == 200
                assert "__UNSLOTH_BOOTSTRAP__" not in response.text
            assert client.get("/api/auth/api-keys", headers = headers).status_code == 401
            assert (
                client.get("/", headers = {**headers, "Origin": "https://evil.example"}).status_code
                == 404
            )
            assert client.get("/", headers = {**headers, "Host": "evil.example"}).status_code == 404
