# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Pairing and identity for cluster_link. The master runs in this process; every worker runs in a
subprocess with its own state directory, because the state directory is process-wide."""

from __future__ import annotations

import asyncio
import base64
import functools
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
from studio.cluster_link.model import Peer  # noqa: E402


@pytest.fixture
def master_dir(tmp_path, monkeypatch):
    d = tmp_path / "master"
    monkeypatch.setenv("UNSLOTH_CLUSTER_STATE_DIR", str(d))
    return d


# `connect PORT` echoes over mutual TLS; else joins, with MODE dropping (`drop_ack`) or holding
# until RELEASE exists (`slow_ack`) the pair_ack.
WORKER = textwrap.dedent(
    """
    import asyncio, json, os, sys
    sys.path.insert(0, {repo!r})
    if os.environ.get("INJECT") == "1":
        sys.path.insert(0, os.path.join({repo!r}, "studio", "backend", "vendor"))
        import truststore
        truststore.inject_into_ssl()
    from studio.cluster_link import framing, identity, pairing, tls
    async def connect(port):
        me = identity.worker_identity()
        ctx = tls.mutual_context(server = False, certfile = me.certfile, keyfile = me.keyfile, rootfile = me.rootfile)
        reader, writer = await asyncio.open_connection("127.0.0.1", port, ssl = ctx)
        pid = tls.peer_identity(writer.get_extra_info("ssl_object"))
        peer = identity.Registry().load().authorize(pid, "master")
        writer.write(b"ping")
        await writer.drain()
        echo = await reader.readexactly(4)
        writer.close()
        return {{"master_ok": peer is not None, "echo": echo.decode()}}
    if sys.argv[1] == "connect":
        print(json.dumps(asyncio.run(connect(int(sys.argv[2])))))
        sys.exit()
    mode, plain = os.environ.get("MODE"), framing.write_frame
    async def write_frame(writer, obj):
        if obj.get("type") == "pair_ack" and mode == "drop_ack":
            writer.transport.abort()  # the link drops after the master issued, before it hears back
            raise ConnectionResetError("dropped")
        if obj.get("type") == "pair_ack" and mode == "slow_ack":
            for _ in range(600):
                if os.path.exists(os.environ["RELEASE"]):
                    break
                await asyncio.sleep(0.05)
        await plain(writer, obj)
    framing.write_frame = write_frame
    try:
        kw = json.loads(os.environ.get("JOIN_KW", "{{}}"))
        ident = asyncio.run(pairing.join(sys.argv[1], name = "w-" + sys.argv[2], **kw))
        print(json.dumps({{"ok": True, "node_id": ident.node_id, "fp": ident.fingerprint,
                          "port": identity.worker_agent_port()}}))
    except BaseException as exc:
        me = identity.worker_identity()
        print(json.dumps({{"ok": False, "paired": me is not None, "node_id": me.node_id if me else None,
                          "type": type(exc).__name__, "error": type(exc).__name__ + ": " + str(exc)}}))
    """
).format(repo = str(REPO))


async def _run(state_dir: Path, *args, **env):
    env = dict(os.environ, UNSLOTH_CLUSTER_STATE_DIR = str(state_dir), **env)
    pipe = asyncio.subprocess.PIPE
    proc = await asyncio.create_subprocess_exec(
        sys.executable, "-c", WORKER, *args, env = env, stdout = pipe, stderr = pipe
    )
    out, err = await asyncio.wait_for(proc.communicate(), 60)
    lines = out.decode().strip().splitlines()
    assert lines, err.decode()[-2000:]
    return json.loads(lines[-1])


def _join(
    state_dir,
    token,
    tag = "1",
    **env,
):
    return _run(state_dir, token, tag, **env)


async def _open(**kw):
    """Master identity, an open pairing window and its token."""
    master = identity.master_identity()
    window = pairing.PairingWindow(
        master, **dict(dict(host = "127.0.0.1", port = 0, addrs = ["127.0.0.1"]), **kw)
    )
    return master, window, await window.start()


def _async(fn):
    @functools.wraps(fn)
    def run(*args, **kwargs):
        asyncio.run(fn(*args, **kwargs))

    return run


def _workers():
    return [p.node_id for p in identity.Registry().load().workers()]


@_async
async def test_happy_path_then_mutual_tls(master_dir, tmp_path):
    master, window, token = await _open()
    result = await _join(tmp_path / "w1", token)
    assert result["ok"], result
    peer = await window.wait(5)
    assert peer.node_id == result["node_id"] and peer.fingerprint == result["fp"]
    assert peer.agent_port == 50070
    assert _workers() == [peer.node_id]

    seen = {}
    files = dict(certfile = master.certfile, keyfile = master.keyfile, rootfile = master.rootfile)
    ctx = tls.mutual_context(server = True, **files)

    async def handle(reader, writer):
        pid = tls.peer_identity(writer.get_extra_info("ssl_object"))
        seen["worker"] = identity.Registry().load().authorize(pid, "worker")
        writer.write(await reader.readexactly(4))
        await writer.drain()
        writer.close()

    server = await asyncio.start_server(handle, "127.0.0.1", 0, ssl = ctx)
    out = await _run(tmp_path / "w1", "connect", str(server.sockets[0].getsockname()[1]))
    server.close()
    assert out == {"master_ok": True, "echo": "ping"}
    assert seen["worker"] is not None and seen["worker"].node_id == peer.node_id


def test_master_identity_is_stable_and_its_keys_owner_only(master_dir):
    assert identity.master_identity() == identity.master_identity()
    root = (master_dir / "ca" / "root.pem").read_bytes()
    identity.master_identity()
    assert (master_dir / "ca" / "root.pem").read_bytes() == root
    if os.name == "nt":
        pytest.skip("POSIX modes")
    for path in (master_dir / "identity" / "key.pem", master_dir / "ca" / "root_key.pem"):
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert stat.S_IMODE((master_dir / "identity").stat().st_mode) == 0o700


def test_an_interrupted_first_root_write_is_recovered(master_dir, monkeypatch):
    # The first `cluster pair` dies between the two cluster-root writes: the next one must still work.
    real, calls = identity.write_private, []

    def dies_after_one(path, data):
        calls.append(path)
        real(path, data)
        if len(calls) == 1:
            raise KeyboardInterrupt

    monkeypatch.setattr(identity, "write_private", dies_after_one)
    with pytest.raises(KeyboardInterrupt):
        identity.master_identity()
    monkeypatch.setattr(identity, "write_private", real)
    me = identity.master_identity()
    assert me is not None and identity.master_identity() == me


def test_a_root_key_left_without_its_certificate_is_replaced(master_dir):
    identity.master_identity()
    (master_dir / "ca" / "root.pem").unlink()
    (master_dir / "identity" / "identity.json").unlink()
    me = identity.master_identity()
    assert me is not None and (master_dir / "ca" / "root.pem").is_file()
    if os.name != "nt":
        assert stat.S_IMODE((master_dir / "ca" / "root_key.pem").stat().st_mode) == 0o600


@pytest.mark.parametrize(
    "field, value, error",
    [
        pytest.param(
            "master_fp",
            "0" * 64,
            "certificate mismatch",
            id = "wrong_master_fingerprint_sends_nothing",
        ),
        pytest.param("secret", bytes(32), "refused", id = "wrong_secret"),
        pytest.param("now", 10_000, "refused", id = "window_expiry_refuses_on_master_side"),
        pytest.param(
            "exp", -3600, None, id = "worker_clock_ahead_still_pairs_while_the_window_is_open"
        ),
    ],
)
@_async
async def test_invite_checks(master_dir, tmp_path, field, value, error):
    clock = [time.time()]
    _, window, token = await _open(now = lambda: clock[0])
    invite = tokens.decode(token)
    if field == "now":
        clock[0] += value
    elif field == "exp":
        invite.exp = time.time() + value  # what a worker an hour ahead would see
    else:
        setattr(invite, field, value)
    result = await _join(tmp_path / "w", tokens.encode(invite), "x")
    if error is None:
        assert result["ok"], result
        await window.wait(5)
        return
    assert not result["ok"] and error in result["error"]
    if field == "master_fp":
        assert window._attempts == 0  # no hello frame reached the master
    assert identity.Registry().load().workers() == []
    assert not (tmp_path / "w" / "identity" / "identity.json").exists()
    window.close()


@pytest.mark.parametrize("concurrent", [False, True], ids = ["replayed_token", "concurrent_joins"])
@_async
async def test_one_token_pairs_exactly_one_worker(master_dir, tmp_path, concurrent):
    _, _, token = await _open()
    if concurrent:
        results = await asyncio.gather(
            *(_join(tmp_path / f"w{i}", token, str(i)) for i in range(3))
        )
    else:
        results = [await _join(tmp_path / f"w{i}", token, str(i)) for i in range(2)]
        assert results[0]["ok"]
    assert sum(r["ok"] for r in results) == 1, results
    assert len(_workers()) == 1


@_async
async def test_a_spent_invitation_refuses_a_replayed_hello_before_any_challenge(
    master_dir, tmp_path
):
    _, window, token = await _open()
    assert (await _join(tmp_path / "w", token))["ok"]
    await window.wait(5)
    hello = dict(type = "pair_hello", v = pairing.PROTOCOL_VERSION, invite_id = window.invite_id)
    reader = asyncio.StreamReader()
    reader.feed_data(framing.encode(dict(hello, worker_nonce = "", name = "replay", csr_pem = "")))
    reader.feed_eof()
    with pytest.raises(pairing.PairingError, match = "window closed"):
        await window._pair(reader, None)  # a connection accepted before the window closed
    assert window._attempts == 1


@pytest.mark.skipif(sys.version_info < (3, 10), reason = "the vendored truststore needs Python 3.10+")
@pytest.mark.parametrize("side", ["worker", "master"])
@_async
async def test_truststore_injected_side_pairs(master_dir, tmp_path, side):
    sys.path.insert(0, str(REPO / "studio" / "backend" / "vendor"))
    import truststore

    if side == "master":
        truststore.inject_into_ssl()
    try:
        _, _, token = await _open()
        result = await _join(tmp_path / "w", token, "t", INJECT = "1" if side == "worker" else "0")
        assert result["ok"], result
    finally:
        if side == "master":
            truststore.extract_from_ssl()


def test_transcript_binds_every_field():
    secret, fields = b"k" * 32, dict(cluster_id = "c" * 32, invite_id = "i" * 32, session = b"s" * 16)
    fields.update(
        worker_nonce = b"w" * 32, master_nonce = b"m" * 32, master_fp = "f" * 64, csr_digest = b"d" * 32
    )
    base = pairing._proof(secret, pairing.transcript(**fields))
    for name, value in fields.items():
        changed = dict(fields, **{name: value[:-1] + (b"X" if isinstance(value, bytes) else "X")})
        assert pairing._proof(secret, pairing.transcript(**changed)) != base, name
    # length prefixes keep a byte moved between neighbouring fields from colliding
    moved = dict(fields, cluster_id = "c" * 31, invite_id = "c" + "i" * 32)
    assert pairing._proof(secret, pairing.transcript(**moved)) != base


def test_csr_san_is_overridden(master_dir):
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec

    identity.master_identity()
    uris = ("unsloth-node:evil", "unsloth-role:master")
    san = x509.SubjectAlternativeName([x509.UniformResourceIdentifier(u) for u in uris])
    csr = (
        x509.CertificateSigningRequestBuilder()
        .subject_name(x509.Name([]))
        .add_extension(san, critical = False)
        .sign(ec.generate_private_key(ec.SECP256R1()), hashes.SHA256())
    )
    cert = identity.issue_worker_cert(csr.public_bytes(serialization.Encoding.PEM), "abc123")
    pid = tls.identity_from_der(identity._der_of_pem(cert))
    assert pid.node_id == "abc123" and pid.role == "worker"
    loaded = x509.load_pem_x509_certificate(cert)
    assert loaded.extensions.get_extension_for_class(x509.BasicConstraints).value.ca is False


def test_bad_csr_signature_rejected(master_dir):
    from cryptography import x509
    from cryptography.hazmat.primitives import serialization

    identity.master_identity()
    _, csr_pem = identity.new_worker_key_and_csr()
    _, other_pem = identity.new_worker_key_and_csr()
    with pytest.raises(ValueError):
        identity.issue_worker_cert(b"not a csr", "x")
    # A well-formed CSR whose signature no longer matches: flip one byte inside the signature.
    der = bytearray(x509.load_pem_x509_csr(csr_pem).public_bytes(serialization.Encoding.DER))
    der[-5] ^= 0x01
    tampered = b"-----BEGIN CERTIFICATE REQUEST-----\n" + base64.encodebytes(bytes(der))
    tampered += b"-----END CERTIFICATE REQUEST-----\n"
    assert x509.load_pem_x509_csr(tampered) is not None
    with pytest.raises(ValueError):
        identity.issue_worker_cert(tampered, "x")
    assert identity.issue_worker_cert(other_pem, "abc")  # a valid CSR still issues


def _peer(
    node_id,
    role = "worker",
    fp = "a" * 64,
):
    return Peer(node_id = node_id, name = node_id, role = role, fingerprint = fp, cert_pem = "")


def test_registry_authorize_rules(master_dir):
    reg = identity.Registry()
    with reg.transaction():
        reg.add(_peer("w1"))
        reg.add(_peer("m1", "master", "b" * 64))
    reg = identity.Registry().load()
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
    reg = identity.Registry().load()
    assert reg.authorize(good, "worker") is None
    assert reg.peers["w1"].removed  # tombstone kept
    assert reg.workers() == [] and reg.master().node_id == "m1"


@pytest.mark.parametrize(
    "start, calls, addrs, port",
    [
        (
            None,
            [("10.0.0.5", 50070), ("10.0.0.6",), ("10.0.0.5",)],
            ["10.0.0.5:50070", "10.0.0.6:50070"],
            50070,
        ),
        # Seen once on a temporary port, back on its paired port: history keeps the usual pair.
        ("10.0.0.5", [("10.0.0.5", 50099)], ["10.0.0.5:50099", "10.0.0.5:50070"], 50099),
    ],
    ids = ["record_address", "a_temporary_port_keeps_the_usual_pair_in_the_history"],
)
def test_registry_record_address(master_dir, start, calls, addrs, port):
    reg = identity.Registry()
    with reg.transaction():
        peer = _peer("w1")
        if start:
            peer.addrs, peer.agent_port = [start], 50070  # as pairing records it
        reg.add(peer)
        for call in calls:
            reg.record_address("w1", *call)
    peer = identity.Registry().load().peers["w1"]
    assert peer.addrs == addrs and peer.agent_port == port


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


def _raw_frame(obj) -> bytes:
    body = obj if isinstance(obj, bytes) else json.dumps(obj).encode()
    return struct.pack(">I", len(body)) + body


def _hello(
    invite_id,
    csr = "x",
    name = "x",
    worker_nonce = None,
):
    worker_nonce = worker_nonce or pairing._b64e(os.urandom(32))
    hello = dict(type = "pair_hello", v = 1, invite_id = invite_id, worker_nonce = worker_nonce)
    return dict(hello, name = name, csr_pem = csr)


async def _bootstrap_connect(invite):
    return await asyncio.open_connection(
        "127.0.0.1", invite.port, ssl = tls.bootstrap_client_context()
    )


@pytest.mark.parametrize(
    "payload",
    [
        struct.pack(">I", framing.MAX_FRAME + 1),
        _raw_frame(b"not json"),
        _raw_frame(b"[1,2]"),
        _raw_frame({"type": "pair_hello", "v": "1"}),
        _raw_frame({"type": "pair_proof", "proof": "00"}),
        struct.pack(">I", 100) + b"short",
        _raw_frame(_hello("0" * 32, worker_nonce = "AAAA")),
    ],
    ids = [
        "oversized",
        "not_json",
        "not_object",
        "string_version",
        "proof_first",
        "truncated",
        "short_nonce",
    ],
)
@_async
async def test_malformed_frames_fail_closed(master_dir, payload):
    _, window, token = await _open()
    reader, writer = await _bootstrap_connect(tokens.decode(token))
    writer.write(payload)
    await writer.drain()
    if payload.endswith(b"short"):
        writer.write_eof() if writer.can_write_eof() else None
    try:
        reply = await asyncio.wait_for(reader.read(65536), 15)
    except (ConnectionError, ssl.SSLError):
        reply = b""
    writer.close()
    assert b"pair_issued" not in reply
    assert identity.Registry().load().workers() == []
    assert not window._used  # a malformed attempt does not burn the invitation
    window.close()


@_async
async def test_real_hello_then_wrong_order_fails_closed(master_dir):
    _, window, token = await _open()
    invite = tokens.decode(token)
    _, csr = identity.new_worker_key_and_csr()
    reader, writer = await _bootstrap_connect(invite)
    hello = _hello(invite.invite_id, csr.decode())
    await framing.write_frame(writer, hello)
    assert (await framing.read_frame(reader))["type"] == "pair_challenge"
    await framing.write_frame(writer, hello)  # hello again instead of a proof
    assert (await framing.read_frame(reader))["type"] == "pair_error"
    writer.close()
    assert identity.Registry().load().workers() == []
    window.close()


@_async
async def test_hellos_without_the_token_cannot_close_the_window(master_dir, tmp_path):
    # A LAN attacker who never saw the token must not burn the window's attempt budget.
    _, window, token = await _open()
    invite = tokens.decode(token)
    _, csr = identity.new_worker_key_and_csr()
    for _ in range(pairing.MAX_ATTEMPTS + 5):
        reader, writer = await _bootstrap_connect(invite)
        await framing.write_frame(writer, _hello("0" * 32, csr.decode(), name = "attacker"))
        assert (await framing.read_frame(reader))["type"] == "pair_error"
        writer.close()
    assert not window._used
    result = await _join(tmp_path / "w1", token)
    assert result["ok"], result
    await window.wait(5)


@_async
async def test_a_failure_after_the_proof_reaches_the_waiting_master(master_dir, tmp_path):
    # Recording the worker fails: `pair` must say so now, not hang until the window expires.
    _, window, token = await _open(ttl = 120)
    (master_dir / "peers.json").write_text("{not json")
    await _join(tmp_path / "w1", token)
    with pytest.raises(pairing.PairingError):
        await window.wait(10)


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


@pytest.mark.skipif(os.name == "nt", reason = "flock path")
def test_a_lock_that_cannot_be_taken_fails_loudly(master_dir, monkeypatch):
    import fcntl

    def refuse(fd, op):
        raise OSError(37, "No locks available")

    monkeypatch.setattr(fcntl, "flock", refuse)
    with pytest.raises(RuntimeError, match = "could not lock"):
        with identity.Registry().transaction():
            pass


def test_concurrent_master_creation_makes_one_cluster_root(master_dir):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    barrier = Barrier(4)

    def make(_):
        barrier.wait()
        return identity.master_identity(create = True)

    with ThreadPoolExecutor(4) as pool:
        made = list(pool.map(make, range(4)))
    assert len({(m.cluster_id, m.node_id, m.fingerprint) for m in made}) == 1
    assert identity.master_identity(create = False).fingerprint == made[0].fingerprint


@_async
async def test_the_port_a_worker_pairs_with_is_what_both_sides_remember(master_dir, tmp_path):
    _, window, token = await _open()
    result = await _join(tmp_path / "w", token, JOIN_KW = '{"agent_port": 50111}')
    peer = await window.wait(5)
    assert (
        peer.agent_port == 50111 and result["port"] == 50111 and result["node_id"] == peer.node_id
    )


def test_forget_master_leaves_nothing_that_blocks_joining_elsewhere(master_dir):
    first = identity.master_identity()
    identity.forget_master()
    assert identity.master_identity(create = False) is None
    assert not list((master_dir / "ca").glob("*.pem"))
    assert (
        identity.master_identity().cluster_id != first.cluster_id
    )  # a new cluster, not the old root


@pytest.mark.parametrize(
    "concurrent", [False, True], ids = ["join_again_without_leaving", "two_at_once"]
)
@_async
async def test_a_paired_machine_cannot_join_twice(master_dir, tmp_path, concurrent):
    # At once: both would pass an unlocked "not paired yet" check, then overwrite each other's key.
    (_, w1, t1), (_, w2, t2) = await _open(), await _open()
    state = tmp_path / "w"
    if concurrent:
        r1, r2 = await asyncio.gather(_join(state, t1, "a"), _join(state, t2, "b"))
    else:
        r1 = await _join(state, t1)
        assert r1["ok"], r1
        await w1.wait(5)
        r2 = await _join(state, t2)
    w1.close()
    w2.close()
    assert sorted([r1["ok"], r2["ok"]]) == [False, True], (r1, r2)
    loser = r1 if not r1["ok"] else r2
    assert "already paired" in loser["error"] and "cluster leave" in loser["error"]
    assert len(_workers()) == 1  # no dead duplicate on the master


@pytest.mark.parametrize("pair_ok", ["lost", "slow"])
@_async
async def test_a_lost_or_slow_final_confirmation_keeps_both_sides_paired(
    master_dir, tmp_path, monkeypatch, pair_ok
):
    # Lost: the worker keeps its identity (else the master has a ghost). Slow: past the worker's 2 s
    # join timeout once the identity is saved, it is waited for, never a bare timeout.
    plain = pairing.framing.write_frame

    async def write_frame(writer, obj):
        if obj.get("type") == "pair_ok" and pair_ok == "lost":
            return writer.transport.abort()
        if obj.get("type") == "pair_ok":
            await asyncio.sleep(3.5)
        await plain(writer, obj)

    monkeypatch.setattr(pairing.framing, "write_frame", write_frame)
    _, window, token = await _open()
    result = await _join(
        tmp_path / "w", token, JOIN_KW = '{"timeout": 2.0}' if pair_ok == "slow" else "{}"
    )
    peer = await window.wait(10)
    if pair_ok == "slow":
        assert result["ok"], result
    else:
        assert (result["ok"], result["paired"], result["type"]) == (
            False,
            True,
            "PairingUnconfirmed",
        )
        assert result["node_id"] == peer.node_id
    assert _workers() == [peer.node_id]


@pytest.mark.parametrize("cut", ["before_the_ack", "master_reset"])
@_async
async def test_an_interrupted_pairing_records_nothing_on_the_master(
    master_dir, tmp_path, monkeypatch, cut
):
    _, window, token = await _open()
    real = identity.master_identity
    if cut == "master_reset":  # `cluster leave` between certificate and commit: the commit refuses
        monkeypatch.setattr(pairing.identity, "master_identity", lambda create = True: None)
    result = await _join(tmp_path / "w", token, MODE = "drop_ack" if cut == "before_the_ack" else "")
    if cut == "before_the_ack":  # the worker keeps its identity and is told how to check
        assert (result["ok"], result["paired"], result["type"]) == (
            False,
            True,
            "PairingUnconfirmed",
        )
    assert result["ok"] is False
    with pytest.raises(pairing.PairingError):
        await window.wait(5)
    monkeypatch.setattr(pairing.identity, "master_identity", real)
    assert identity.Registry().load().workers() == []  # no ghost


@pytest.mark.parametrize(
    "addrs",
    [["10.0.0.%d" % i for i in range(17)], ["a" * 254], ["has space"], ["fe80::1"], ["::1"]],
    ids = ["too_many", "too_long", "space", "ipv6_link_local", "ipv6_loopback"],
)
@_async
async def test_pair_rejects_addresses_no_worker_could_decode(master_dir, addrs):
    # IPv6 literals too: the pairing listener and the agent bind IPv4 only.
    window = pairing.PairingWindow(
        identity.master_identity(), host = "127.0.0.1", port = 0, addrs = addrs
    )
    with pytest.raises(pairing.PairingError, match = "--addr"):
        await window.start()
    assert window._server is None  # refused before listening


@pytest.mark.parametrize("deadline", ["ttl_timer", "caller_wait"])
@_async
async def test_a_deadline_after_the_proof_does_not_fail_a_pairing_that_completes(
    master_dir, tmp_path, deadline
):
    # Token proved, then the TTL timer fires or `cluster pair`'s own wait deadline passes mid-ack.
    _, window, token = await _open()
    release = tmp_path / "release_ack"
    worker = asyncio.ensure_future(
        _join(tmp_path / "w", token, MODE = "slow_ack", RELEASE = str(release))
    )
    end = time.monotonic() + 60
    while not window._used and time.monotonic() < end:
        await asyncio.sleep(0.05)
    assert window._used, "the worker never proved the token"
    waiting = None
    if deadline == "ttl_timer":
        window._expire()
    else:
        waiting = asyncio.ensure_future(window.wait(0.2))
        await asyncio.sleep(0.6)  # past the caller's deadline
    release.write_text("go")
    peer = await waiting if waiting else None
    result = await worker
    assert result["ok"], result
    peer = peer or await window.wait(5)
    assert peer.node_id == result["node_id"]
    assert _workers() == [peer.node_id]


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason = "needs 127.0.0.2 on loopback")
@_async
async def test_a_mismatched_first_address_falls_through_to_the_master(
    master_dir, tmp_path, monkeypatch
):
    # Multihomed token whose first address is another machine: nothing sent, try the next address.
    master, window, token = await _open(addrs = ["127.0.0.2", "127.0.0.1"])
    port = window.bound_port
    monkeypatch.setenv("UNSLOTH_CLUSTER_STATE_DIR", str(tmp_path / "decoy"))
    other = (
        identity.master_identity()
    )  # another cluster's master: a certificate that does not match
    monkeypatch.setenv("UNSLOTH_CLUSTER_STATE_DIR", str(master_dir))

    async def decoy(reader, writer):
        writer.close()

    ctx = tls.bootstrap_server_context(certfile = other.certfile, keyfile = other.keyfile)
    decoy_server = await asyncio.start_server(decoy, "127.0.0.2", port, ssl = ctx)
    try:
        addr, _, writer = await pairing._connect(
            ["127.0.0.2", "127.0.0.1"], port, master.fingerprint
        )
        writer.close()
        assert addr == "127.0.0.1"
        result = await _join(tmp_path / "w", token)
        assert result["ok"], result
        assert (await window.wait(5)).node_id == result["node_id"]
        with pytest.raises(pairing.PairingError, match = "certificate mismatch"):
            await pairing._connect(["127.0.0.2"], port, master.fingerprint)
    finally:
        decoy_server.close()
        window.close()


@_async
async def test_waiting_for_the_identity_lock_does_not_block_the_event_loop(master_dir):
    # Two joins in one process: the second waits for the lock without freezing the loop.
    holding, release = asyncio.Event(), asyncio.Event()
    order = []

    async def first():
        async with pairing._identity_lock():
            holding.set()
            await release.wait()
            order.append("first")

    async def second():
        await holding.wait()
        async with pairing._identity_lock():
            order.append("second")

    async def ticker():
        await holding.wait()
        await asyncio.sleep(0.2)  # runs only if the loop is free while `second` waits
        order.append("tick")
        release.set()

    await asyncio.wait_for(asyncio.gather(first(), second(), ticker()), 20)
    assert order == ["tick", "first", "second"]

    # Cancelled while waiting: the lock is still handed back once the thread gets it.
    holding.clear()
    release.clear()
    holder = asyncio.ensure_future(first())
    await holding.wait()
    waiter = asyncio.ensure_future(second())
    await asyncio.sleep(0.2)
    waiter.cancel()
    release.set()
    await holder
    with pytest.raises(asyncio.CancelledError):
        await waiter
    await asyncio.wait_for(second(), 20)
    assert order[-1] == "second"


@pytest.mark.skipif(os.name == "nt", reason = "POSIX shared temp directory")
def test_a_run_dir_another_user_could_have_planted_is_refused(tmp_path, monkeypatch):
    from studio.cluster_link import paths

    monkeypatch.delenv(paths.ENV_RUN_DIR, raising = False)
    monkeypatch.setattr(paths.tempfile, "gettempdir", lambda: str(tmp_path))
    run = tmp_path / f"unsloth-cluster-{os.getuid()}"
    assert paths.run_dir() == run and stat.S_IMODE(run.stat().st_mode) == 0o700
    run.rmdir()
    victim = tmp_path / "victim"
    victim.mkdir(mode = 0o755)
    run.symlink_to(victim)
    with pytest.raises(RuntimeError, match = "not a directory owned by this user"):
        paths.run_dir()
    assert stat.S_IMODE(victim.stat().st_mode) == 0o755  # never chmod-ed through the link
    run.unlink()
    real_uid = os.getuid()
    run.mkdir(mode = 0o777)
    monkeypatch.setattr(paths.os, "getuid", lambda: real_uid + 1)
    with pytest.raises(RuntimeError, match = "not a directory owned by this user"):
        paths._own_shared_tmp_dir(run)


def test_the_state_dir_defaults_under_the_studio_root(tmp_path, monkeypatch):
    import types

    from studio.cluster_link import paths

    for var in (paths.ENV_STATE_DIR, "UNSLOTH_STUDIO_HOME", "STUDIO_HOME"):
        monkeypatch.delenv(var, raising = False)
    roots = types.SimpleNamespace(studio_root = lambda: tmp_path / "root")
    monkeypatch.setitem(sys.modules, "utils.paths.storage_roots", roots)
    assert paths.state_dir() == tmp_path / "root" / "cluster"
    monkeypatch.setitem(sys.modules, "utils.paths.storage_roots", None)  # no Studio backend
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    assert paths.state_dir() == tmp_path / ".unsloth" / "studio" / "cluster"
    monkeypatch.setenv("STUDIO_HOME", str(tmp_path / "alt"))
    assert paths.state_dir() == tmp_path / "alt" / "cluster"
