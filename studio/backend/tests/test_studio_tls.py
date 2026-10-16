# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""--ssl-certfile / --ssl-keyfile: validation, the HTTPS handshake under native TLS, and the URLs Studio prints."""

from __future__ import annotations

import datetime
import socket
import ssl
import subprocess
import sys
import textwrap
import types
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parent.parent


def _write_pair(tmp_path: Path, name: str = "studio.example.ts.net") -> tuple[Path, Path]:
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID

    key = ec.generate_private_key(ec.SECP256R1())
    subject = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, name)])
    now = datetime.datetime.now(datetime.timezone.utc)
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(subject)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(minutes = 5))
        .not_valid_after(now + datetime.timedelta(days = 1))
        .add_extension(x509.SubjectAlternativeName([x509.DNSName(name)]), critical = False)
        .add_extension(x509.BasicConstraints(ca = True, path_length = None), critical = True)
        # macOS's verifier (truststore in the client too) refuses a TLS server cert without serverAuth
        .add_extension(x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH]), critical = False)
        .add_extension(
            x509.KeyUsage(
                digital_signature = True,
                content_commitment = False,
                key_encipherment = False,
                data_encipherment = False,
                key_agreement = False,
                key_cert_sign = True,
                crl_sign = False,
                encipher_only = False,
                decipher_only = False,
            ),
            critical = True,
        )
        .sign(key, hashes.SHA256())
    )
    crt, key_path = tmp_path / f"{name}.crt", tmp_path / f"{name}.key"
    crt.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    key_path.write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    return crt, key_path


def test_no_flags_is_plain_http():
    from utils.studio_tls import resolve_studio_tls
    assert resolve_studio_tls(None, None) is None
    assert resolve_studio_tls("", "  ") is None


def test_valid_pair_resolves_to_absolute_paths(tmp_path, monkeypatch):
    from utils.studio_tls import resolve_studio_tls

    crt, key = _write_pair(tmp_path)
    monkeypatch.chdir(tmp_path)
    tls = resolve_studio_tls(crt.name, key.name)
    assert tls.certfile == str(crt) and tls.keyfile == str(key)


@pytest.mark.parametrize("which", ["cert", "key"])
def test_half_a_pair_is_refused(tmp_path, which):
    from utils.studio_tls import resolve_studio_tls
    crt, key = _write_pair(tmp_path)
    with pytest.raises(SystemExit, match = "must be passed together"):
        resolve_studio_tls(
            str(crt) if which == "cert" else None, str(key) if which == "key" else None
        )


def test_missing_and_mismatched_files_fail_before_binding(tmp_path):
    from utils.studio_tls import resolve_studio_tls

    crt, key = _write_pair(tmp_path)
    with pytest.raises(SystemExit, match = "--ssl-keyfile: file not found"):
        resolve_studio_tls(str(crt), str(tmp_path / "absent.key"))
    other = tmp_path / "other"
    other.mkdir()
    _, other_key = _write_pair(other, "other.example")
    with pytest.raises(SystemExit, match = "Could not load the TLS certificate and key"):
        resolve_studio_tls(str(crt), str(other_key))


_SERVE = textwrap.dedent(
    """
    import socket, sys, threading, time
    backend, crt, key, mode = sys.argv[1:5]
    sys.path[:0] = [backend]
    sys.path.append(backend + "/vendor")
    import truststore
    if sys.platform not in ("darwin", "win32"):
        # Linux truststore verifies inside OpenSSL; reproduce the macOS / Windows hook, which raises on an
        # empty peer chain (truststore/_windows.py), so this cell fails the same way they do.
        import ssl
        import truststore._api as api
        def _no_chain(ctx, chain, server_hostname = None):
            if not chain:
                raise ssl.SSLCertVerificationError("Peer sent no certificates to verify")
        api._verify_peercerts_impl = _no_chain
    truststore.inject_into_ssl()
    import uvicorn
    from utils.studio_tls import StudioTLS, StudioTLSConfig

    async def app(scope, receive, send):
        if scope["type"] != "http":
            return
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})

    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    kwargs = dict(log_level = "error", lifespan = "off")
    if mode == "studio":
        config = StudioTLSConfig(app, tls = StudioTLS(crt, key), **kwargs)
    else:
        config = uvicorn.Config(app, ssl_certfile = crt, ssl_keyfile = key, **kwargs)
    server = uvicorn.Server(config)
    thread = threading.Thread(target = server.run, kwargs = {"sockets": [sock]}, daemon = True)
    thread.start()
    while not server.started:
        if not thread.is_alive():
            sys.exit(1)
        time.sleep(0.05)
    print(sock.getsockname()[1], flush = True)
    sys.stdin.read()
    """
)


def _handshake(tmp_path, mode: str) -> "bytes | Exception":
    crt, key = _write_pair(tmp_path)
    proc = subprocess.Popen(
        [sys.executable, "-c", _SERVE, str(_BACKEND), str(crt), str(key), mode],
        stdin = subprocess.PIPE,
        stdout = subprocess.PIPE,
        text = True,
    )
    try:
        line = proc.stdout.readline()
        assert line.strip(), f"TLS test server exited during startup (rc={proc.poll()})"
        port = int(line)
        client = ssl.create_default_context(cafile = str(crt))
        try:
            with socket.create_connection(("127.0.0.1", port), timeout = 10) as raw:
                with client.wrap_socket(raw, server_hostname = "studio.example.ts.net") as tls:
                    tls.sendall(
                        b"GET / HTTP/1.1\r\nHost: studio.example.ts.net\r\nConnection: close\r\n\r\n"
                    )
                    data = b""
                    while chunk := tls.recv(4096):
                        data += chunk
                    return data
        except (OSError, ssl.SSLError) as exc:
            return exc
    finally:
        proc.stdin.close()
        proc.kill()
        proc.wait(timeout = 10)


def _served(reply) -> bool:
    return (
        isinstance(reply, bytes)
        and reply.startswith(b"HTTP/1.1 200")
        and b"\r\n2\r\nok\r\n" in reply
    )


def test_https_handshake_survives_native_tls_injection(tmp_path):
    reply = _handshake(tmp_path, "studio")
    assert _served(reply), reply


def test_plain_uvicorn_tls_fails_under_injection(tmp_path):
    """Negative control: the hook really breaks a context truststore owns. Under TLS 1.3 the client finishes first,
    so the server's rejection reads as a closed connection, not a client-side error."""
    reply = _handshake(tmp_path, "uvicorn")
    # the client accepting the certificate is what makes this the server's failure
    assert not isinstance(reply, ssl.SSLCertVerificationError), reply
    assert not _served(reply), reply


def test_banner_uses_https_only_when_asked(capsys):
    from startup_banner import print_studio_access_banner

    print_studio_access_banner(port = 8888, bind_host = "0.0.0.0", display_host = "192.168.1.5")
    plain = capsys.readouterr().out
    print_studio_access_banner(
        port = 8888, bind_host = "0.0.0.0", display_host = "192.168.1.5", scheme = "https"
    )
    tls = capsys.readouterr().out
    assert "http://127.0.0.1:8888" in plain and "https://" not in plain
    assert "https://127.0.0.1:8888" in tls and "https://192.168.1.5:8888" in tls
    assert "http://" not in tls


def test_lan_listener_serves_tls_when_the_launch_does(tmp_path):
    from lan_access import _listener_config
    from utils.studio_tls import StudioTLSConfig, resolve_studio_tls

    tls = resolve_studio_tls(*map(str, _write_pair(tmp_path)))
    plain_app = types.SimpleNamespace(state = types.SimpleNamespace())
    tls_app = types.SimpleNamespace(state = types.SimpleNamespace(studio_tls = tls))
    assert type(_listener_config(plain_app, "192.168.1.5", 8888)) is not StudioTLSConfig
    config = _listener_config(tls_app, "192.168.1.5", 8888)
    assert isinstance(config, StudioTLSConfig) and config.is_ssl


def test_self_calls_use_the_loopback_listener():
    from utils.studio_tls import loopback_http_port

    assert loopback_http_port(types.SimpleNamespace()) is None
    assert loopback_http_port(types.SimpleNamespace(loopback_http_port = None)) is None
    assert loopback_http_port(types.SimpleNamespace(loopback_http_port = 41234)) == 41234


def test_tls_launch_publishes_no_ip_server_url(monkeypatch):
    """A Tailscale / Let's Encrypt cert names a DNS host, so an https://<LAN IP> base would fail verification."""
    import run

    monkeypatch.setattr(run, "_network_share_host_for_bind", lambda host: "192.168.1.5")
    monkeypatch.setattr(run, "_url_scheme", "http")
    assert run._direct_server_url("0.0.0.0", 8888) == "http://192.168.1.5:8888"
    monkeypatch.setattr(run, "_url_scheme", "https")
    assert run._direct_server_url("0.0.0.0", 8888) is None


@pytest.mark.parametrize("bind_host", ["192.168.1.5", "studio.example.ts.net"])
def test_tls_launch_on_a_specific_host_still_lists_its_lan_url(tmp_path, bind_host):
    from utils.lan_access_settings import _launch_urls
    from utils.studio_tls import resolve_studio_tls

    tls = resolve_studio_tls(*map(str, _write_pair(tmp_path)))
    state = types.SimpleNamespace(
        lan_access_wildcard_bind = False,
        server_url = None,
        studio_tls = tls,
        lan_access_bind_host = bind_host,
        lan_access_port = 8888,
    )
    assert _launch_urls(state) == [f"https://{bind_host}:8888"]
    state.studio_tls = None
    assert _launch_urls(state) == []
