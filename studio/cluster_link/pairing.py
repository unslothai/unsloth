# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Pairing: the master opens a short window and prints a token; the worker joins with it."""

from __future__ import annotations

import asyncio
import base64
import contextlib
import hashlib
import hmac
import logging
import secrets
import socket
import ssl
import time
from typing import Callable, List, Optional, Sequence

from . import (
    AGENT_PORT_DEFAULT,
    PAIR_PORT_DEFAULT,
    PROTOCOL_VERSION,
    framing,
    identity,
    tls,
    tokens,
)
from .model import LocalIdentity, Peer

logger = logging.getLogger(__name__)

TRANSCRIPT_LABEL = b"unsloth-pair-v1"
CONFIRM_LABEL = b"unsloth-pair-confirm-v1"
FRAME_TIMEOUT_S = 10.0
CONNECT_TIMEOUT_S = 5.0
CLAIMED_GRACE_S = 3 * FRAME_TIMEOUT_S
MAX_ATTEMPTS = 20
MAX_NAME = 64


class PairingUnconfirmed(RuntimeError):
    """The worker saved its identity and acknowledged, but never heard the master record it."""


class PairingError(RuntimeError):
    """Pairing did not complete; the message is safe to show (it never contains the token)."""


def _b64e(raw: bytes) -> str:
    return base64.b64encode(raw).decode("ascii")


def _b64d(text, length: int, name: str) -> bytes:
    try:
        raw = base64.b64decode(text, validate = True) if isinstance(text, str) else b""
    except (ValueError, TypeError):
        raw = b""
    if len(raw) != length:
        raise framing.FrameError(f"bad {name}")
    return raw


def transcript(
    *,
    cluster_id: str,
    invite_id: str,
    session: bytes,
    worker_nonce: bytes,
    master_nonce: bytes,
    master_fp: str,
    csr_digest: bytes,
) -> bytes:
    out = bytearray()
    for part in (
        TRANSCRIPT_LABEL,
        cluster_id.encode("ascii"),
        invite_id.encode("ascii"),
        session,
        worker_nonce,
        master_nonce,
        master_fp.encode("ascii"),
        csr_digest,
    ):
        out += len(part).to_bytes(4, "big") + part
    return bytes(out)


def _proof(secret: bytes, data: bytes) -> str:
    return hmac.new(secret, data, hashlib.sha256).hexdigest()


def _confirm(secret: bytes, data: bytes) -> str:
    return hmac.new(secret, CONFIRM_LABEL + data, hashlib.sha256).hexdigest()


def local_ipv4_addresses() -> List[str]:
    """Non-loopback IPv4 addresses this host can be reached at, best first."""
    found: List[str] = []
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("192.0.2.1", 9))
            found.append(s.getsockname()[0])
    except OSError:
        pass
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            found.append(info[4][0])
    except OSError:
        pass
    out = []
    for addr in found:
        if addr not in out and not addr.startswith("127."):
            out.append(addr)
    return out


def _clean_name(name) -> str:
    text = "".join(ch for ch in str(name or "") if ch.isprintable()).strip()
    return text[:MAX_NAME] or "worker"


class PairingWindow:
    """One-use invitation served by the master until it is used, expires, or is closed."""

    def __init__(
        self,
        master: LocalIdentity,
        *,
        host: str = "0.0.0.0",
        port: int = PAIR_PORT_DEFAULT,
        ttl: float = 600.0,
        addrs: Optional[Sequence[str]] = None,
        registry: Optional[identity.Registry] = None,
        now: Callable[[], float] = time.time,
    ):
        if master.role != "master":
            raise ValueError("only a master opens a pairing window")
        self.master = master
        self.host = host
        self.port = port
        self.ttl = ttl
        self.addrs = list(addrs) if addrs else None
        self.registry = registry
        self._now = now
        self._secret = secrets.token_bytes(tokens.SECRET_BYTES)
        self.invite_id = secrets.token_hex(16)
        self.expires_at = 0.0
        self._lock = asyncio.Lock()
        self._used = False
        self._claimed = False
        self._attempts = 0
        self._server: Optional[asyncio.AbstractServer] = None
        self._done: Optional[asyncio.Future] = None
        self._expiry: Optional[asyncio.TimerHandle] = None

    @property
    def bound_port(self) -> int:
        if self._server is None or not self._server.sockets:
            return self.port
        return self._server.sockets[0].getsockname()[1]

    async def start(self) -> str:
        addrs = self.addrs or local_ipv4_addresses() or ["127.0.0.1"]
        try:
            tokens.check_addrs(addrs)
        except tokens.TokenError as exc:
            raise PairingError(f"--addr: {exc}") from None
        ctx = tls.bootstrap_server_context(
            certfile = self.master.certfile, keyfile = self.master.keyfile
        )
        loop = asyncio.get_running_loop()
        self._done = loop.create_future()
        self._server = await asyncio.start_server(
            self._handle, self.host, self.port, ssl = ctx, ssl_handshake_timeout = CONNECT_TIMEOUT_S
        )
        self.expires_at = self._now() + self.ttl
        self._expiry = loop.call_later(self.ttl, self._expire)
        return tokens.encode(
            tokens.Invite(
                cluster_id = self.master.cluster_id,
                invite_id = self.invite_id,
                addrs = addrs,
                port = self.bound_port,
                master_fp = self.master.fingerprint,
                secret = self._secret,
                exp = self.expires_at,
            )
        )

    def _fail(self, exc: BaseException) -> None:
        self._done.set_exception(exc)
        self._done.exception()  # wait() still raises it; nobody waiting is not an error

    def _expire(self) -> None:
        if self._claimed:
            return  # a worker proved the token in time: its handler settles the window
        if self._done is not None and not self._done.done():
            self._fail(PairingError("the pairing window expired"))
        self.close()

    def close(self) -> None:
        self._used = True
        if self._expiry is not None:
            self._expiry.cancel()
        if self._server is not None:
            self._server.close()
        if self._done is not None and not self._done.done():
            self._fail(PairingError("the pairing window was closed"))

    async def wait(self, timeout: Optional[float] = None) -> Peer:
        if self._done is None:
            raise RuntimeError("start() first")
        try:
            return await asyncio.wait_for(asyncio.shield(self._done), timeout)
        except asyncio.TimeoutError:
            if not self._claimed:
                raise
        return await asyncio.wait_for(asyncio.shield(self._done), CLAIMED_GRACE_S)

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            await self._pair(reader, writer)
        except (framing.FrameError, PairingError, ValueError) as exc:
            logger.info("pairing attempt refused: %s", exc)
            try:
                await framing.write_frame(
                    writer, {"type": "pair_error", "reason": "pairing refused"}
                )
            except Exception:
                pass
        except Exception as exc:
            logger.info("pairing attempt failed: %s", type(exc).__name__)
        finally:
            writer.close()

    async def _pair(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        if self._used or self._now() > self.expires_at:
            raise PairingError("window closed")
        hello = framing.expect(
            await framing.read_frame(reader, FRAME_TIMEOUT_S),
            "pair_hello",
            v = int,
            invite_id = str,
            worker_nonce = str,
            name = str,
            csr_pem = str,
        )
        if hello["v"] != PROTOCOL_VERSION or not hmac.compare_digest(
            hello["invite_id"].encode("utf-8"), self.invite_id.encode("ascii")
        ):
            raise PairingError("unknown invitation")
        # Counted only for hellos naming this invitation, so LAN strangers cannot exhaust the budget.
        self._attempts += 1
        if self._attempts > MAX_ATTEMPTS:
            self.close()
            raise PairingError("too many pairing attempts")
        worker_nonce = _b64d(hello["worker_nonce"], 32, "worker nonce")
        csr_pem = hello["csr_pem"].encode("ascii", "replace")
        der = identity.csr_der(csr_pem)
        session, master_nonce = secrets.token_bytes(16), secrets.token_bytes(32)
        await framing.write_frame(
            writer,
            {
                "type": "pair_challenge",
                "session": _b64e(session),
                "master_nonce": _b64e(master_nonce),
            },
        )
        proof = framing.expect(
            await framing.read_frame(reader, FRAME_TIMEOUT_S), "pair_proof", proof = str
        )
        data = transcript(
            cluster_id = self.master.cluster_id,
            invite_id = self.invite_id,
            session = session,
            worker_nonce = worker_nonce,
            master_nonce = master_nonce,
            master_fp = self.master.fingerprint,
            csr_digest = hashlib.sha256(der).digest(),
        )
        if not hmac.compare_digest(proof["proof"], _proof(self._secret, data)):
            raise PairingError("bad proof")
        async with self._lock:
            if self._used:
                raise PairingError("invitation already used")
            self._used = self._claimed = True
        if self._expiry is not None:
            self._expiry.cancel()
        try:
            node_id = secrets.token_hex(8)
            cert_pem = identity.issue_worker_cert(csr_pem, node_id)
            peername = writer.get_extra_info("peername")
            agent_port = hello.get("agent_port")
            if (
                isinstance(agent_port, bool)
                or not isinstance(agent_port, int)
                or not 0 < agent_port < 65536
            ):
                agent_port = AGENT_PORT_DEFAULT
            peer = Peer(
                node_id = node_id,
                name = _clean_name(hello["name"]),
                role = "worker",
                fingerprint = identity.pem_fingerprint(cert_pem),
                cert_pem = cert_pem.decode("ascii"),
                addrs = [peername[0]] if peername else [],
                agent_port = agent_port,
                added_at = self._now(),
            )
            await framing.write_frame(
                writer,
                {
                    "type": "pair_issued",
                    "node_id": node_id,
                    "worker_cert": cert_pem.decode("ascii"),
                    "root_cert": identity.root_pem().decode("ascii"),
                    "master_node_id": self.master.node_id,
                    "master_cert": identity.own_cert_pem().decode("ascii"),
                    "agent_hint": None,
                    "confirm": _confirm(self._secret, data),
                },
            )
            # Recorded only after the worker saved its identity: never a worker unaware it is paired.
            framing.expect(
                await framing.read_frame(reader, FRAME_TIMEOUT_S), "pair_ack", node_id = str
            )
            reg = self.registry or identity.Registry()
            async with _identity_lock():
                now_master = identity.master_identity(create = False)
                if now_master is None or now_master.fingerprint != self.master.fingerprint:
                    raise PairingError("this master was reset while pairing; pair again")
                with reg.transaction():
                    reg.add(peer)
            await framing.write_frame(writer, {"type": "pair_ok"})
            if self._done is not None and not self._done.done():
                self._done.set_result(peer)
            if self._expiry is not None:
                self._expiry.cancel()
            if self._server is not None:
                self._server.close()
        except Exception as exc:
            if self._done is not None and not self._done.done():
                self._fail(
                    PairingError(f"pairing failed after the worker's proof: {type(exc).__name__}")
                )
            self.close()
            raise


@contextlib.asynccontextmanager
async def _identity_lock():
    cm = identity.identity_lock()
    entered = asyncio.ensure_future(asyncio.to_thread(cm.__enter__))
    try:
        await asyncio.shield(entered)
    except asyncio.CancelledError:
        entered.add_done_callback(
            lambda f: f.cancelled() or f.exception() is not None or cm.__exit__(None, None, None)
        )
        raise
    try:
        yield
    finally:
        cm.__exit__(None, None, None)


async def _connect(addrs: Sequence[str], port: int, master_fp: str):
    last: Optional[BaseException] = None
    mismatched: List[str] = []
    for addr in addrs:
        try:
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(addr, port, ssl = tls.bootstrap_client_context()),
                CONNECT_TIMEOUT_S,
            )
        except (OSError, asyncio.TimeoutError) as exc:
            last = exc
            continue
        seen = tls.peer_fingerprint(writer.get_extra_info("ssl_object"))
        if seen is None or not hmac.compare_digest(seen, master_fp):
            writer.close()
            mismatched.append(f"{addr}:{port}")
            continue
        return addr, reader, writer
    if mismatched:
        raise PairingError(
            f"{', '.join(mismatched)} is not the master this token was made for (certificate mismatch)"
        )
    raise PairingError(f"could not reach the master at {', '.join(addrs)} port {port}: {last}")


async def join(
    token: str,
    *,
    name: str,
    devices: Optional[list] = None,
    timeout: float = 15.0,
    now: Callable[[], float] = time.time,
    replace: bool = False,
    agent_port: int = AGENT_PORT_DEFAULT,
) -> LocalIdentity:
    """Worker side: pair with the master named by ``token`` and persist the result."""
    invite = tokens.decode(token)
    async with _identity_lock():
        return await _join_locked(invite, name, devices, timeout, now, replace, agent_port)


async def _join_locked(invite, name, devices, timeout, now, replace, agent_port) -> LocalIdentity:
    current = identity.worker_identity()
    if current is not None and not replace:
        raise PairingError(
            "this machine is already paired as a worker; run `unsloth cluster leave` first"
        )
    if identity.master_identity(create = False) is not None:
        raise PairingError(
            "this machine is a cluster master and cannot also join as a worker; if no worker is paired "
            "to it, `unsloth cluster leave` resets it"
        )

    key_pem, csr_pem = identity.new_worker_key_and_csr()
    der = identity.csr_der(csr_pem)
    worker_nonce = secrets.token_bytes(32)

    async def answer(reader) -> dict:
        frame = await framing.read_frame(reader, FRAME_TIMEOUT_S)
        if frame.get("type") == "pair_error":
            raise PairingError("the master refused this pairing (token used, expired or wrong)")
        return frame

    async def handshake(reader, writer) -> dict:
        hello = {
            "type": "pair_hello",
            "v": PROTOCOL_VERSION,
            "invite_id": invite.invite_id,
            "worker_nonce": _b64e(worker_nonce),
            "name": _clean_name(name),
            "csr_pem": csr_pem.decode("ascii"),
            "agent_port": int(agent_port),
        }
        if devices is not None:
            hello["devices"] = list(devices)[:64]
        await framing.write_frame(writer, hello)
        challenge = framing.expect(
            await answer(reader), "pair_challenge", session = str, master_nonce = str
        )
        data = transcript(
            cluster_id = invite.cluster_id,
            invite_id = invite.invite_id,
            session = _b64d(challenge["session"], 16, "session"),
            worker_nonce = worker_nonce,
            master_nonce = _b64d(challenge["master_nonce"], 32, "master nonce"),
            master_fp = invite.master_fp,
            csr_digest = hashlib.sha256(der).digest(),
        )
        await framing.write_frame(
            writer, {"type": "pair_proof", "proof": _proof(invite.secret, data)}
        )
        issued = framing.expect(
            await answer(reader),
            "pair_issued",
            node_id = str,
            worker_cert = str,
            root_cert = str,
            master_node_id = str,
            master_cert = str,
            confirm = str,
        )
        if not hmac.compare_digest(issued["confirm"], _confirm(invite.secret, data)):
            raise PairingError("the master could not prove it holds the token secret")
        return issued

    async def run():
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        addr, reader, writer = await asyncio.wait_for(
            _connect(invite.addrs, invite.port, invite.master_fp), timeout
        )
        try:
            issued = await asyncio.wait_for(
                handshake(reader, writer), max(0.1, deadline - loop.time())
            )
            local = _save(addr, issued)
            try:
                await framing.write_frame(writer, {"type": "pair_ack", "node_id": local.node_id})
                framing.expect(await framing.read_frame(reader, FRAME_TIMEOUT_S), "pair_ok")
            except (
                framing.FrameError,
                OSError,
                ssl.SSLError,
                asyncio.TimeoutError,
                asyncio.CancelledError,
            ) as exc:
                raise PairingUnconfirmed(
                    f"saved this worker's identity ({local.node_id[:12]}) but the master's final "
                    f"confirmation did not arrive ({type(exc).__name__}). If `unsloth cluster list` on "
                    "the master does not show it, run `unsloth cluster leave` here and pair again."
                ) from None
            return local
        finally:
            writer.close()

    def _save(addr: str, issued: dict) -> LocalIdentity:
        worker_cert = issued["worker_cert"].encode("ascii")
        master_cert = issued["master_cert"].encode("ascii")
        root_cert = issued["root_cert"].encode("ascii")
        pid = tls.identity_from_der(identity._der_of_pem(worker_cert))
        master_pid = tls.identity_from_der(identity._der_of_pem(master_cert))
        if pid is None or pid.node_id != issued["node_id"] or pid.role != "worker":
            raise PairingError("the master issued a certificate that does not name this worker")
        if not identity.cert_matches_key(worker_cert, key_pem):
            raise PairingError("the master issued a certificate for a different key")
        if (
            master_pid is None
            or master_pid.role != "master"
            or master_pid.node_id != issued["master_node_id"]
        ):
            raise PairingError("the master's certificate does not name a master")
        if not hmac.compare_digest(master_pid.fingerprint, invite.master_fp):
            raise PairingError("the master's certificate changed during pairing")
        if current is not None:
            identity.forget_identity()
        local = identity.save_worker_identity(
            node_id = issued["node_id"],
            cluster_id = invite.cluster_id,
            key_pem = key_pem,
            cert_pem = worker_cert,
            root_pem = root_cert,
            agent_port = agent_port,
        )
        reg = identity.Registry()
        with reg.transaction():
            for old in [p for p in reg.peers.values() if p.role == "master"]:
                reg.remove(old.node_id)
            reg.add(
                Peer(
                    node_id = issued["master_node_id"],
                    name = "master",
                    role = "master",
                    fingerprint = master_pid.fingerprint,
                    cert_pem = master_cert.decode("ascii"),
                    addrs = [addr] + [a for a in invite.addrs if a != addr],
                    added_at = now(),
                )
            )
        return local

    return await run()
