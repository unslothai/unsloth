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


def test_a_worker_whose_clock_is_ahead_still_pairs_while_the_window_is_open(master_dir, tmp_path):
    # A new or offline machine's clock may be off: the token's expiry as the worker sees it must not
    # refuse a pairing the master (which this connection pins) still accepts.
    async def main():
        _, window = _window()
        invite = tokens.decode(await window.start())
        invite.exp = time.time() - 3600  # what a worker an hour ahead would see
        result = await _worker(WORKER_JOIN, tmp_path / "w", tokens.encode(invite), "x")
        assert result["ok"], result
        await window.wait(5)

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


@pytest.mark.skipif(sys.version_info < (3, 10), reason = "the vendored truststore needs Python 3.10+")
def test_truststore_injected_worker_pairs(master_dir, tmp_path):
    async def main():
        _, window = _window()
        token = await window.start()
        result = await _worker(WORKER_JOIN, tmp_path / "w", token, "t", inject = True)
        assert result["ok"], result

    asyncio.run(main())


@pytest.mark.skipif(sys.version_info < (3, 10), reason = "the vendored truststore needs Python 3.10+")
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


def test_hellos_without_the_token_cannot_close_the_window(master_dir, tmp_path):
    """A LAN attacker who never saw the token must not be able to burn the window's attempt budget."""

    async def main():
        _, window = _window()
        token = await window.start()
        invite = tokens.decode(token)
        _, csr = identity.new_worker_key_and_csr()
        for _ in range(pairing.MAX_ATTEMPTS + 5):
            reader, writer = await asyncio.open_connection(
                "127.0.0.1", invite.port, ssl = tls.bootstrap_client_context()
            )
            await framing.write_frame(
                writer,
                {
                    "type": "pair_hello",
                    "v": 1,
                    "invite_id": "0" * 32,
                    "worker_nonce": pairing._b64e(os.urandom(32)),
                    "name": "attacker",
                    "csr_pem": csr.decode(),
                },
            )
            reply = await framing.read_frame(reader)
            assert reply["type"] == "pair_error"
            writer.close()
        assert not window._used
        result = await _worker(WORKER_JOIN, tmp_path / "w1", token, "1")
        assert result["ok"], result
        await window.wait(5)

    asyncio.run(main())


def test_a_failure_after_the_proof_reaches_the_waiting_master(master_dir, tmp_path):
    """If recording the new worker fails, `pair` must say so now, not hang until the window expires."""

    async def main():
        _, window = _window(ttl = 120)
        token = await window.start()
        (master_dir / "peers.json").write_text("{not json")
        await _worker(WORKER_JOIN, tmp_path / "w1", token, "1")
        with pytest.raises(pairing.PairingError):
            await window.wait(10)

    asyncio.run(main())


def test_registry_keeps_peers_with_fields_from_a_newer_version(master_dir):
    path = master_dir / "peers.json"
    master_dir.mkdir(parents = True)
    row = dict(node_id = "w1", name = "w1", role = "worker", fingerprint = "a" * 64, cert_pem = "")
    path.write_text(json.dumps({"peers": [{**row, "removed": True, "future": {"x": 1}}]}))
    reg = identity.Registry()
    with reg.transaction():
        assert reg.peers["w1"].removed  # the tombstone survives an unknown field
        reg.record_address("w1", "10.0.0.2")
    saved = json.loads(path.read_text())["peers"][0]
    assert saved["future"] == {"x": 1} and saved["removed"] is True


def test_a_lock_that_cannot_be_taken_fails_loudly(master_dir, monkeypatch):
    if os.name == "nt":
        pytest.skip("flock path")
    import fcntl

    def refuse(fd, op):
        raise OSError(37, "No locks available")

    monkeypatch.setattr(fcntl, "flock", refuse)
    with pytest.raises(RuntimeError, match = "could not lock"):
        with identity.Registry().transaction():
            pass


def test_concurrent_master_creation_makes_one_cluster_root(master_dir):
    import threading

    made, barrier = [], threading.Barrier(4)

    def make():
        barrier.wait()
        made.append(identity.master_identity(create = True))

    threads = [threading.Thread(target = make) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len({(m.cluster_id, m.node_id, m.fingerprint) for m in made}) == 1
    assert identity.master_identity(create = False).fingerprint == made[0].fingerprint


WORKER_JOIN_ON_PORT = textwrap.dedent(
    """
    import asyncio, json, sys
    sys.path.insert(0, {repo!r})
    from studio.cluster_link import identity, pairing
    ident = asyncio.run(pairing.join(sys.argv[1], name = "w-port", agent_port = int(sys.argv[2])))
    print(json.dumps({{"node_id": ident.node_id, "port": identity.worker_agent_port()}}))
    """
).format(repo = str(REPO))


def test_the_port_a_worker_pairs_with_is_what_both_sides_remember(master_dir, tmp_path):
    async def main():
        _, window = _window()
        token = await window.start()
        result = await _worker(WORKER_JOIN_ON_PORT, tmp_path / "w", token, "50111")
        peer = await window.wait(5)
        assert peer.agent_port == 50111 and result == {"node_id": peer.node_id, "port": 50111}

    asyncio.run(main())


def test_forget_master_leaves_nothing_that_blocks_joining_elsewhere(master_dir):
    first = identity.master_identity()
    identity.forget_master()
    assert identity.master_identity(create = False) is None
    assert not list((master_dir / "ca").glob("*.pem"))
    # A later master on this machine is a different cluster, not the old root.
    assert identity.master_identity().cluster_id != first.cluster_id


WORKER_JOIN_DROPS_ACK = textwrap.dedent(
    """
    import asyncio, json, sys
    sys.path.insert(0, {repo!r})
    from studio.cluster_link import framing, identity, pairing
    plain = framing.write_frame
    async def write_frame(writer, obj):
        if obj.get("type") == "pair_ack":
            writer.transport.abort()  # the link drops after the master issued, before it hears back
            raise ConnectionResetError("dropped")
        await plain(writer, obj)
    framing.write_frame = write_frame
    try:
        asyncio.run(pairing.join(sys.argv[1], name = "w-drop"))
        print(json.dumps({{"ok": True}}))
    except Exception as exc:
        print(json.dumps({{"ok": False, "paired": identity.worker_identity() is not None, "error": type(exc).__name__}}))
    """
).format(repo = str(REPO))


def test_a_pairing_cut_before_the_ack_records_nothing_on_the_master(master_dir, tmp_path):
    async def main():
        _, window = _window()
        token = await window.start()
        result = await _worker(WORKER_JOIN_DROPS_ACK, tmp_path / "w", token)
        # The worker keeps what it saved and is told how to check; the master never heard the ack, so
        # it recorded nothing: no ghost on the master, and `cluster leave` + a new token recover.
        assert result == {"ok": False, "paired": True, "error": "PairingUnconfirmed"}
        with pytest.raises(pairing.PairingError):
            await window.wait(5)
        assert identity.Registry.open().workers() == []

    asyncio.run(main())


def test_a_paired_worker_cannot_join_again_without_leaving(master_dir, tmp_path):
    async def main():
        _, window = _window()
        token = await window.start()
        assert (await _worker(WORKER_JOIN, tmp_path / "w", token, "1"))["ok"]
        await window.wait(5)
        _, again = _window()
        token2 = await again.start()
        second = await _worker(WORKER_JOIN, tmp_path / "w", token2, "1")
        again.close()
        assert not second["ok"] and "cluster leave" in second["error"]
        assert len(identity.Registry.open().workers()) == 1  # no dead duplicate on the master

    asyncio.run(main())


WORKER_JOIN_STATUS = textwrap.dedent(
    """
    import asyncio, json, sys
    sys.path.insert(0, {repo!r})
    from studio.cluster_link import identity, pairing
    try:
        asyncio.run(pairing.join(sys.argv[1], name = "w-status"))
        print(json.dumps({{"ok": True}}))
    except Exception as exc:
        me = identity.worker_identity()
        print(json.dumps({{"ok": False, "paired": me is not None, "node_id": me.node_id if me else None,
                          "error": type(exc).__name__}}))
    """
).format(repo = str(REPO))


def test_a_lost_final_confirmation_keeps_both_sides_paired(master_dir, tmp_path, monkeypatch):
    # The master recorded the worker and only its pair_ok was lost: the worker keeps its identity, or
    # the master is left with a worker that no longer exists.
    plain = pairing.framing.write_frame

    async def write_frame(writer, obj):
        if obj.get("type") == "pair_ok":
            writer.transport.abort()
            return
        await plain(writer, obj)

    monkeypatch.setattr(pairing.framing, "write_frame", write_frame)

    async def main():
        _, window = _window()
        token = await window.start()
        result = await _worker(WORKER_JOIN_STATUS, tmp_path / "w", token)
        peer = await window.wait(5)
        assert result == {
            "ok": False,
            "paired": True,
            "node_id": peer.node_id,
            "error": "PairingUnconfirmed",
        }
        assert [p.node_id for p in identity.Registry.open().workers()] == [peer.node_id]

    asyncio.run(main())


def test_a_master_reset_during_pairing_records_nothing(master_dir, tmp_path, monkeypatch):
    # `cluster leave` resets the master between the certificate and the commit: the commit sees no
    # master (or another one) under the shared lock and refuses.
    async def main():
        _, window = _window()
        token = await window.start()
        monkeypatch.setattr(pairing.identity, "master_identity", lambda create = True: None)
        result = await _worker(WORKER_JOIN_STATUS, tmp_path / "w", token)
        with pytest.raises(pairing.PairingError):
            await window.wait(5)
        assert result["ok"] is False

    asyncio.run(main())
    monkeypatch.undo()
    assert identity.Registry.open().workers() == []


def test_pair_rejects_addresses_no_worker_could_decode(master_dir):
    async def main():
        for bad in (["10.0.0.%d" % i for i in range(17)], ["a" * 254], ["has space"]):
            _, window = _window(addrs = bad)
            with pytest.raises(pairing.PairingError, match = "--addr"):
                await window.start()
            assert window._server is None  # refused before listening

    asyncio.run(main())


def test_two_joins_at_once_on_one_machine_do_not_interleave(master_dir, tmp_path):
    # Both pass an unlocked "not paired yet" check, then overwrite each other's key and cert.
    async def main():
        _, w1 = _window()
        _, w2 = _window()
        t1, t2 = await w1.start(), await w2.start()
        state = tmp_path / "w"
        r1, r2 = await asyncio.gather(
            _worker(WORKER_JOIN, state, t1, "a"), _worker(WORKER_JOIN, state, t2, "b")
        )
        assert sorted([r1["ok"], r2["ok"]]) == [False, True], (r1, r2)
        loser = r1 if not r1["ok"] else r2
        assert "already paired" in loser["error"]
        w1.close()
        w2.close()

    asyncio.run(main())


WORKER_JOIN_SHORT_TIMEOUT = textwrap.dedent(
    """
    import asyncio, json, sys
    sys.path.insert(0, {repo!r})
    from studio.cluster_link import identity, pairing
    try:
        asyncio.run(pairing.join(sys.argv[1], name = "w-slow", timeout = 2.0))
        print(json.dumps({{"ok": True}}))
    except BaseException as exc:
        print(json.dumps({{"ok": False, "paired": identity.worker_identity() is not None, "error": type(exc).__name__}}))
    """
).format(repo = str(REPO))


def test_the_join_deadline_does_not_cut_the_final_confirmation(master_dir, tmp_path, monkeypatch):
    # The deadline covers the handshake; once the worker has saved its identity, a slow pair_ok must
    # still be waited for (or end as unconfirmed), never as a bare timeout that leaves it half paired.
    plain = pairing.framing.write_frame

    async def write_frame(writer, obj):
        if obj.get("type") == "pair_ok":
            await asyncio.sleep(3.5)  # past the worker's 2 s join timeout
        await plain(writer, obj)

    monkeypatch.setattr(pairing.framing, "write_frame", write_frame)

    async def main():
        _, window = _window()
        token = await window.start()
        result = await _worker(WORKER_JOIN_SHORT_TIMEOUT, tmp_path / "w", token)
        await window.wait(10)
        assert result == {"ok": True}, result

    asyncio.run(main())
