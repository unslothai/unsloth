# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""An MCP server with nothing listening (the editor or app is closed) gets a plain answer, not a raw connect error.

Seen live with Unreal: the editor stopped mid-chat and the model got "Client failed to connect: All connection attempts
failed" on every call, then slept and retried for minutes."""

import socket

import pytest

from core.inference import mcp_client
from core.inference import tools
from storage import mcp_servers_db as db


def _closed_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.mark.parametrize("scope", [None, "s=chat:t=thread"])
def test_nothing_listening_is_reported_as_unreachable(scope):
    url = f"http://127.0.0.1:{_closed_port()}/mcp"
    result = mcp_client.call_tool_sync(url, None, "list_toolsets", {}, timeout = 20, scope = scope)
    assert result.startswith("Error: MCP tool 'list_toolsets' did not run: ")
    assert (
        mcp_client.UNREACHABLE_SERVER in result
        and "waiting or calling it again will not help" in result
    )


def test_unreachable_names_the_server_and_its_setup_step(monkeypatch):
    port = _closed_port()
    db.create_server(
        "ue0123456789abcd", "Unreal Engine", f"http://127.0.0.1:{port}/mcp", builtin_id = "unreal"
    )
    db.create_server("fedcba9876543210", "My server", f"http://127.0.0.1:{port}/mcp")
    monkeypatch.setattr(tools, "stdio_mcp_enabled", lambda: True)
    result = tools.execute_tool("mcp__unreal__list_toolsets", {}, timeout = 20)
    assert "'Unreal Engine' is not reachable" in result
    assert result.endswith("Open your project with the Unreal MCP plugin enabled.")
    other = tools.execute_tool("mcp__fedcba9876543210__anything", {}, timeout = 20)
    assert "'My server' is not reachable" in other and "Unreal" not in other


def test_other_failures_keep_their_message():
    class Boom(Exception):
        pass

    assert not mcp_client.server_unreachable(Boom("bad request"))
    wrapped = RuntimeError("Client failed to connect: All connection attempts failed")
    assert mcp_client.server_unreachable(wrapped)
    try:
        try:
            raise ConnectionRefusedError(111, "refused")
        except ConnectionRefusedError as inner:
            raise RuntimeError("outer") from inner
    except RuntimeError as outer:
        assert mcp_client.server_unreachable(outer)


def test_an_untrusted_certificate_is_not_reported_as_a_closed_server(tmp_path):
    """httpx raises ConnectError for a TLS verification failure too; that server is up, so keep the real cause."""
    import datetime
    import http.server
    import ssl
    import threading

    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.x509.oid import NameOID

    key = ec.generate_private_key(ec.SECP256R1())
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "127.0.0.1")])
    now = datetime.datetime.now(datetime.timezone.utc)
    cert = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(minutes = 1))
        .not_valid_after(now + datetime.timedelta(days = 1))
        .sign(key, hashes.SHA256())
    )
    cert_path, key_path = tmp_path / "cert.pem", tmp_path / "key.pem"
    cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    key_path.write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), http.server.BaseHTTPRequestHandler)
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(cert_path, key_path)
    server.socket = context.wrap_socket(server.socket, server_side = True)
    threading.Thread(target = server.serve_forever, daemon = True).start()
    try:
        url = f"https://127.0.0.1:{server.server_address[1]}/mcp"
        result = mcp_client.call_tool_sync(url, None, "list_toolsets", {}, timeout = 20)
    finally:
        server.shutdown()
    assert result.startswith("Error: MCP tool 'list_toolsets' failed: ")
    assert mcp_client.UNREACHABLE_SERVER not in result

    wrapped = RuntimeError("Client failed to connect")
    wrapped.__cause__ = ssl.SSLCertVerificationError(
        1, "[SSL: CERTIFICATE_VERIFY_FAILED] self-signed certificate"
    )
    assert not mcp_client.server_unreachable(wrapped)
