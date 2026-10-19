# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Pairing and identity for cluster_link. The master runs in this process; every worker runs in a
subprocess with its own state directory, because the state directory is process-wide."""

from __future__ import annotations

import asyncio
import base64
import json
import os
import ssl
import stat
import struct
import sys
import textwrap
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from studio.cluster_link import framing, identity, pairing, tls, tokens  # noqa: E402


@pytest.fixture
def master_dir(tmp_path, monkeypatch):
    d = tmp_path / "master"
    monkeypatch.setenv("UNSLOTH_CLUSTER_STATE_DIR", str(d))
    return d


WORKER_JOIN = textwrap.dedent(
    """
    import asyncio, json, os, sys
    sys.path.insert(0, {repo!r})
    if os.environ.get("INJECT") == "1":
        sys.path.insert(0, os.path.join({repo!r}, "studio", "backend", "vendor"))
        import truststore
        truststore.inject_into_ssl()
    from studio.cluster_link import pairing
    try:
        ident = asyncio.run(pairing.join(sys.argv[1], name = "w-" + sys.argv[2]))
        print(json.dumps({{"ok": True, "node_id": ident.node_id, "fp": ident.fingerprint}}))
    except Exception as exc:
        print(json.dumps({{"ok": False, "error": type(exc).__name__ + ": " + str(exc)}}))
    """
).format(repo = str(REPO))

WORKER_CONNECT = textwrap.dedent(
    """
    import asyncio, json, sys
    sys.path.insert(0, {repo!r})
    from studio.cluster_link import identity, tls
    async def main():
        me = identity.worker_identity()
        ctx = tls.mutual_context(server = False, certfile = me.certfile, keyfile = me.keyfile, rootfile = me.rootfile)
        reader, writer = await asyncio.open_connection("127.0.0.1", int(sys.argv[1]), ssl = ctx)
        pid = tls.peer_identity(writer.get_extra_info("ssl_object"))
        peer = identity.Registry.open().authorize(pid, "master")
        writer.write(b"ping")
        await writer.drain()
        echo = await reader.readexactly(4)
        writer.close()
        print(json.dumps({{"master_ok": peer is not None, "echo": echo.decode()}}))
    asyncio.run(main())
    """
).format(repo = str(REPO))


async def _worker(
    script,
    state_dir: Path,
    *args,
    inject = False,
):
    env = dict(os.environ, UNSLOTH_CLUSTER_STATE_DIR = str(state_dir), INJECT = "1" if inject else "0")
    proc = await asyncio.create_subprocess_exec(
        sys.executable,
        "-c",
        script,
        *args,
        env = env,
        stdout = asyncio.subprocess.PIPE,
        stderr = asyncio.subprocess.PIPE,
    )
    out, err = await asyncio.wait_for(proc.communicate(), 60)
    lines = out.decode().strip().splitlines()
    assert lines, err.decode()[-2000:]
    return json.loads(lines[-1])


def _window(**kw):
    master = identity.master_identity()
    kw.setdefault("host", "127.0.0.1")
    kw.setdefault("port", 0)
    kw.setdefault("addrs", ["127.0.0.1"])
    return master, pairing.PairingWindow(master, **kw)


def test_happy_path_then_mutual_tls(master_dir, tmp_path):
    async def main():
        master, window = _window()
        token = await window.start()
        result = await _worker(WORKER_JOIN, tmp_path / "w1", token, "1")
        assert result["ok"], result
        peer = await window.wait(5)
        assert peer.node_id == result["node_id"] and peer.fingerprint == result["fp"]
        assert peer.agent_port == 50070
        reg = identity.Registry.open()
        assert [p.node_id for p in reg.workers()] == [peer.node_id]

        seen = {}
        ctx = tls.mutual_context(
            server = True, certfile = master.certfile, keyfile = master.keyfile, rootfile = master.rootfile
        )

        async def handle(reader, writer):
            pid = tls.peer_identity(writer.get_extra_info("ssl_object"))
            seen["worker"] = identity.Registry.open().authorize(pid, "worker")
            writer.write(await reader.readexactly(4))
            await writer.drain()
            writer.close()

        server = await asyncio.start_server(handle, "127.0.0.1", 0, ssl = ctx)
        port = server.sockets[0].getsockname()[1]
        out = await _worker(WORKER_CONNECT, tmp_path / "w1", str(port))
        server.close()
        assert out == {"master_ok": True, "echo": "ping"}
        assert seen["worker"] is not None and seen["worker"].node_id == peer.node_id

    asyncio.run(main())


def test_keys_are_owner_only(master_dir):
    identity.master_identity()
    if os.name == "nt":
        pytest.skip("POSIX modes")
    for path in (master_dir / "identity" / "key.pem", master_dir / "ca" / "root_key.pem"):
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert stat.S_IMODE((master_dir / "identity").stat().st_mode) == 0o700


def test_master_identity_is_stable(master_dir):
    a = identity.master_identity()
    b = identity.master_identity()
    assert a == b
    root = (master_dir / "ca" / "root.pem").read_bytes()
    identity.master_identity()
    assert (master_dir / "ca" / "root.pem").read_bytes() == root


def test_wrong_master_fingerprint_sends_nothing(master_dir, tmp_path):
    async def main():
        master, window = _window()
        token = await window.start()
        invite = tokens.decode(token)
        invite.master_fp = "0" * 64
        result = await _worker(WORKER_JOIN, tmp_path / "w", tokens.encode(invite), "x")
        assert not result["ok"] and "certificate mismatch" in result["error"]
        assert window._attempts == 0  # no hello frame reached the master
        assert identity.Registry.open().workers() == []
        window.close()

    asyncio.run(main())


def test_wrong_secret_is_refused(master_dir, tmp_path):
    async def main():
        _, window = _window()
        invite = tokens.decode(await window.start())
        invite.secret = bytes(32)
        result = await _worker(WORKER_JOIN, tmp_path / "w", tokens.encode(invite), "x")
        assert not result["ok"] and "refused" in result["error"]
        assert identity.Registry.open().workers() == []
        assert not (tmp_path / "w" / "identity" / "identity.json").exists()
        window.close()

    asyncio.run(main())


def test_expired_token(master_dir, tmp_path):
    async def main():
        _, window = _window()
        invite = tokens.decode(await window.start())
        invite.exp = time.time() - 1
        result = await _worker(WORKER_JOIN, tmp_path / "w", tokens.encode(invite), "x")
        assert not result["ok"] and "expired" in result["error"]
        window.close()

    asyncio.run(main())


def test_window_expiry_refuses_on_master_side(master_dir, tmp_path):
    async def main():
        clock = [time.time()]
        _, window = _window(now = lambda: clock[0])
        token = await window.start()
        clock[0] += 10_000
        result = await _worker(WORKER_JOIN, tmp_path / "w", token, "x")
        assert not result["ok"]
        window.close()

    asyncio.run(main())


def test_replayed_token_fails(master_dir, tmp_path):
    async def main():
        _, window = _window()
        token = await window.start()
        first = await _worker(WORKER_JOIN, tmp_path / "w1", token, "1")
        second = await _worker(WORKER_JOIN, tmp_path / "w2", token, "2")
        assert first["ok"] and not second["ok"]
        assert len(identity.Registry.open().workers()) == 1

    asyncio.run(main())


def test_two_concurrent_joins_exactly_one_wins(master_dir, tmp_path):
    async def main():
        _, window = _window()
        token = await window.start()
        results = await asyncio.gather(
            *(_worker(WORKER_JOIN, tmp_path / f"w{i}", token, str(i)) for i in range(3))
        )
        assert sum(r["ok"] for r in results) == 1, results
        assert len(identity.Registry.open().workers()) == 1

    asyncio.run(main())


def test_truststore_injected_worker_pairs(master_dir, tmp_path):
    async def main():
        _, window = _window()
        token = await window.start()
        result = await _worker(WORKER_JOIN, tmp_path / "w", token, "t", inject = True)
        assert result["ok"], result

    asyncio.run(main())


def test_truststore_injected_master_pairs(master_dir, tmp_path):
    sys.path.insert(0, str(REPO / "studio" / "backend" / "vendor"))
    import truststore

    truststore.inject_into_ssl()
    try:

        async def main():
            _, window = _window()
            token = await window.start()
            result = await _worker(WORKER_JOIN, tmp_path / "w", token, "t")
            assert result["ok"], result

        asyncio.run(main())
    finally:
        truststore.extract_from_ssl()


def _fields():
    return dict(
        cluster_id = "c" * 32,
        invite_id = "i" * 32,
        session = b"s" * 16,
        worker_nonce = b"w" * 32,
        master_nonce = b"m" * 32,
        master_fp = "f" * 64,
        csr_digest = b"d" * 32,
    )


def test_transcript_binds_every_field():
    secret = b"k" * 32
    base = pairing._proof(secret, pairing.transcript(**_fields()))
    for name, value in _fields().items():
        changed = _fields()
        changed[name] = value[:-1] + (b"X" if isinstance(value, bytes) else "X")
        assert pairing._proof(secret, pairing.transcript(**changed)) != base, name
    # length prefixes keep a byte moved between neighbouring fields from colliding
    moved = _fields()
    moved["cluster_id"] = "c" * 31
    moved["invite_id"] = "c" + "i" * 32
    assert pairing._proof(secret, pairing.transcript(**moved)) != base


def test_csr_san_is_overridden(master_dir):
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec

    identity.master_identity()
    key = ec.generate_private_key(ec.SECP256R1())
    csr = (
        x509.CertificateSigningRequestBuilder()
        .subject_name(x509.Name([]))
        .add_extension(
            x509.SubjectAlternativeName(
                [
                    x509.UniformResourceIdentifier("unsloth-node:evil"),
                    x509.UniformResourceIdentifier("unsloth-role:master"),
                ]
            ),
            critical = False,
        )
        .sign(key, hashes.SHA256())
    )
    cert = identity.issue_worker_cert(csr.public_bytes(serialization.Encoding.PEM), "abc123")
    pid = tls.identity_from_der(identity._der_of_pem(cert))
    assert pid.node_id == "abc123" and pid.role == "worker"
    loaded = x509.load_pem_x509_certificate(cert)
    assert loaded.extensions.get_extension_for_class(x509.BasicConstraints).value.ca is False


def test_bad_csr_signature_rejected(master_dir):
    identity.master_identity()
    _, csr_pem = identity.new_worker_key_and_csr()
    _, other_pem = identity.new_worker_key_and_csr()
    with pytest.raises(ValueError):
        identity.issue_worker_cert(b"not a csr", "x")
    from cryptography import x509
    from cryptography.hazmat.primitives import serialization

    # A well-formed CSR whose signature no longer matches: flip one byte inside the signature.
    der = bytearray(x509.load_pem_x509_csr(csr_pem).public_bytes(serialization.Encoding.DER))
    der[-5] ^= 0x01
    tampered = (
        b"-----BEGIN CERTIFICATE REQUEST-----\n"
        + base64.encodebytes(bytes(der))
        + b"-----END CERTIFICATE REQUEST-----\n"
    )
    assert x509.load_pem_x509_csr(tampered) is not None
    with pytest.raises(ValueError):
        identity.issue_worker_cert(tampered, "x")
    # and a valid CSR still issues, with the master's node id, not one the CSR asked for
    assert identity.issue_worker_cert(other_pem, "abc")


def _peer(node_id, role, fp):
    from studio.cluster_link.model import Peer
    return Peer(node_id = node_id, name = node_id, role = role, fingerprint = fp, cert_pem = "")


def test_registry_authorize_rules(master_dir):
    reg = identity.Registry()
    with reg.transaction():
        reg.add(_peer("w1", "worker", "a" * 64))
        reg.add(_peer("m1", "master", "b" * 64))
    reg = identity.Registry.open()
    good = tls.PeerIdentity(fingerprint = "a" * 64, node_id = "w1", role = "worker")
    assert reg.authorize(good, "worker") is not None
    assert reg.authorize(tls.PeerIdentity("c" * 64, "w1", "worker"), "worker") is None
    assert (
        reg.authorize(tls.PeerIdentity("a" * 64, "w1", "master"), "master") is None
    )  # worker as master
    assert reg.authorize(good, "master") is None
    assert reg.authorize(tls.PeerIdentity("a" * 64, "unknown", "worker"), "worker") is None
    assert reg.authorize(None, "worker") is None
    with reg.transaction():
        assert reg.remove("w1")
    reg = identity.Registry.open()
    assert reg.authorize(good, "worker") is None
    assert reg.peers["w1"].removed  # tombstone kept
    assert reg.workers() == [] and reg.master().node_id == "m1"


def test_registry_record_address(master_dir):
    reg = identity.Registry()
    with reg.transaction():
        reg.add(_peer("w1", "worker", "a" * 64))
        reg.record_address("w1", "10.0.0.5", 50070)
        reg.record_address("w1", "10.0.0.6")
        reg.record_address("w1", "10.0.0.5")
    peer = identity.Registry.open().peers["w1"]
    assert peer.addrs == ["10.0.0.5", "10.0.0.6"] and peer.agent_port == 50070


def test_token_round_trip_and_rejections():
    invite = tokens.Invite(
        cluster_id = "a" * 32,
        invite_id = "b" * 32,
        addrs = ["192.168.1.5", "10.0.0.2"],
        port = 50071,
        master_fp = "c" * 64,
        secret = os.urandom(32),
        exp = 2_000_000_000,
    )
    token = tokens.encode(invite)
    assert token.startswith(tokens.PREFIX) and invite.secret.hex() not in token
    assert tokens.decode(token) == invite
    assert "secret" not in repr(invite) or invite.secret.hex() not in repr(invite)
    for bad in ("", "nope", tokens.PREFIX + "!!!", tokens.PREFIX + "e30", token[:-4] + "AAAA"):
        with pytest.raises(tokens.TokenError):
            tokens.decode(bad)


def _raw_frame(body: bytes) -> bytes:
    return struct.pack(">I", len(body)) + body


@pytest.mark.parametrize(
    "payloads",
    [
        [struct.pack(">I", framing.MAX_FRAME + 1)],
        [_raw_frame(b"not json")],
        [_raw_frame(b"[1,2]")],
        [_raw_frame(json.dumps({"type": "pair_hello", "v": "1"}).encode())],
        [_raw_frame(json.dumps({"type": "pair_proof", "proof": "00"}).encode())],
        [struct.pack(">I", 100) + b"short"],
        [
            _raw_frame(
                json.dumps(
                    {
                        "type": "pair_hello",
                        "v": 1,
                        "invite_id": "0" * 32,
                        "worker_nonce": "AAAA",
                        "name": "x",
                        "csr_pem": "x",
                    }
                ).encode()
            )
        ],
    ],
)
def test_malformed_frames_fail_closed(master_dir, payloads):
    async def main():
        _, window = _window()
        token = await window.start()
        invite = tokens.decode(token)
        reader, writer = await asyncio.open_connection(
            "127.0.0.1", invite.port, ssl = tls.bootstrap_client_context()
        )
        for chunk in payloads:
            writer.write(chunk)
        await writer.drain()
        if payloads[0].endswith(b"short"):
            writer.write_eof() if writer.can_write_eof() else None
        try:
            reply = await asyncio.wait_for(reader.read(65536), 15)
        except (ConnectionError, ssl.SSLError):
            reply = b""
        writer.close()
        assert b"pair_issued" not in reply
        assert identity.Registry.open().workers() == []
        assert not window._used  # a malformed attempt does not burn the invitation
        window.close()

    asyncio.run(main())


def test_real_hello_then_wrong_order_fails_closed(master_dir):
    async def main():
        _, window = _window()
        token = await window.start()
        invite = tokens.decode(token)
        _, csr = identity.new_worker_key_and_csr()
        reader, writer = await asyncio.open_connection(
            "127.0.0.1", invite.port, ssl = tls.bootstrap_client_context()
        )
        hello = {
            "type": "pair_hello",
            "v": 1,
            "invite_id": invite.invite_id,
            "worker_nonce": pairing._b64e(os.urandom(32)),
            "name": "x",
            "csr_pem": csr.decode(),
        }
        await framing.write_frame(writer, hello)
        challenge = await framing.read_frame(reader)
        assert challenge["type"] == "pair_challenge"
        await framing.write_frame(writer, hello)  # hello again instead of a proof
        reply = await framing.read_frame(reader)
        assert reply["type"] == "pair_error"
        writer.close()
        assert identity.Registry.open().workers() == []
        window.close()

    asyncio.run(main())
