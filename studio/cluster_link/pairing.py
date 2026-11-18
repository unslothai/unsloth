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
from typing import Any, Awaitable, Callable, List, Optional, Sequence, Union

from . import (
    AGENT_PORT_DEFAULT,
    PAIR_PORT_DEFAULT,
    PROTOCOL_VERSION,
    framing,
    identity,
    remote,
    tls,
    tokens,
)
from .model import LocalIdentity, Peer

logger = logging.getLogger(__name__)

TRANSCRIPT_LABEL = b"unsloth-pair-v1"
CONFIRM_LABEL = b"unsloth-pair-confirm-v1"
# Pairing v2 (both sides say "pair": 2): tailcat fields bound into the proof, approval words.
PAIR_V2 = 2
TRANSCRIPT_LABEL_V2 = b"unsloth-pair-v2"
CONFIRM_LABEL_V2 = b"unsloth-pair-confirm-v2"
WORDS_LABEL = b"unsloth-pair-words-v1"
FRAME_TIMEOUT_S = 10.0
CONNECT_TIMEOUT_S = 5.0
APPROVAL_TIMEOUT_S = 300.0
CLAIMED_GRACE_S = APPROVAL_TIMEOUT_S + 3 * FRAME_TIMEOUT_S
MAX_ATTEMPTS = 20
MAX_NAME = 64
MAX_TC_PUB = 200
NOT_APPROVED = "not approved on the master"
OLD_WORKER = "runs an older Unsloth that cannot show approval words: update it, then pair again"

WORDS = (
    "acid",
    "acorn",
    "actor",
    "adobe",
    "agent",
    "album",
    "alien",
    "alpha",
    "amber",
    "anchor",
    "angle",
    "apple",
    "apron",
    "arena",
    "armor",
    "arrow",
    "aspen",
    "atlas",
    "attic",
    "audio",
    "autumn",
    "award",
    "bacon",
    "badge",
    "bagel",
    "baker",
    "bamboo",
    "banjo",
    "barley",
    "basil",
    "basin",
    "beach",
    "beacon",
    "bean",
    "bear",
    "beetle",
    "bench",
    "berry",
    "bison",
    "blade",
    "blanket",
    "bloom",
    "board",
    "bonus",
    "boot",
    "bottle",
    "bracket",
    "brain",
    "bread",
    "breeze",
    "brick",
    "bridge",
    "brook",
    "brush",
    "bubble",
    "bucket",
    "buffalo",
    "bugle",
    "cabin",
    "cable",
    "cactus",
    "camel",
    "camera",
    "candle",
    "canoe",
    "canvas",
    "canyon",
    "carbon",
    "cargo",
    "carpet",
    "carrot",
    "castle",
    "cedar",
    "cello",
    "chalk",
    "cheese",
    "cherry",
    "chess",
    "circle",
    "clay",
    "cliff",
    "clock",
    "cloud",
    "clover",
    "coast",
    "cobalt",
    "cocoa",
    "comet",
    "copper",
    "coral",
    "cotton",
    "cradle",
    "crane",
    "crater",
    "crayon",
    "cricket",
    "daisy",
    "delta",
    "denim",
    "desert",
    "diamond",
    "dolphin",
    "donkey",
    "dragon",
    "drum",
    "eagle",
    "ember",
    "engine",
    "falcon",
    "feather",
    "fern",
    "fiddle",
    "field",
    "flame",
    "flute",
    "forest",
    "fossil",
    "fox",
    "garden",
    "garlic",
    "gecko",
    "ginger",
    "glacier",
    "globe",
    "goose",
    "granite",
    "grape",
    "gravel",
    "guitar",
    "hammer",
    "harbor",
    "hazel",
    "helmet",
    "heron",
    "honey",
    "iceberg",
    "igloo",
    "island",
    "ivory",
    "jacket",
    "jaguar",
    "jasmine",
    "jelly",
    "jungle",
    "kayak",
    "kettle",
    "kitten",
    "koala",
    "ladder",
    "lagoon",
    "lantern",
    "lemon",
    "lily",
    "lizard",
    "llama",
    "magnet",
    "mango",
    "maple",
    "marble",
    "meadow",
    "melon",
    "meteor",
    "mint",
    "mirror",
    "mitten",
    "monkey",
    "moose",
    "mosaic",
    "moth",
    "muffin",
    "nectar",
    "needle",
    "nickel",
    "noodle",
    "oasis",
    "ocean",
    "olive",
    "onion",
    "orange",
    "orbit",
    "orchid",
    "otter",
    "oyster",
    "paddle",
    "panda",
    "papaya",
    "parrot",
    "peach",
    "pebble",
    "pencil",
    "pepper",
    "piano",
    "pickle",
    "pigeon",
    "pillow",
    "pine",
    "planet",
    "plum",
    "pocket",
    "pony",
    "poppy",
    "potato",
    "prism",
    "pumpkin",
    "puzzle",
    "quartz",
    "quill",
    "rabbit",
    "radar",
    "radish",
    "raven",
    "reef",
    "ribbon",
    "river",
    "robin",
    "rocket",
    "saddle",
    "salmon",
    "sandal",
    "satin",
    "scarf",
    "shell",
    "sierra",
    "silver",
    "sketch",
    "spider",
    "spruce",
    "squash",
    "staple",
    "summit",
    "sunset",
    "swan",
    "tablet",
    "tango",
    "teapot",
    "thunder",
    "tiger",
    "timber",
    "tomato",
    "topaz",
    "torch",
    "tulip",
    "turtle",
    "valley",
    "velvet",
    "violin",
    "walnut",
    "walrus",
    "wave",
    "willow",
    "window",
    "winter",
    "wizard",
    "yogurt",
    "zebra",
    "zephyr",
)

Approve = Callable[[str, Optional[List[str]]], Union[bool, Awaitable[bool]]]


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
    v2: Optional[Sequence[str]] = None,
) -> bytes:
    """``v2``: the worker's tailcat address, its key, how it reached the master and its LAN addresses."""
    out = bytearray()
    parts = [
        TRANSCRIPT_LABEL if v2 is None else TRANSCRIPT_LABEL_V2,
        cluster_id.encode("ascii"),
        invite_id.encode("ascii"),
        session,
        worker_nonce,
        master_nonce,
        master_fp.encode("ascii"),
        csr_digest,
    ]
    parts += [str(x).encode("utf-8") for x in v2 or ()]
    for part in parts:
        out += len(part).to_bytes(4, "big") + part
    return bytes(out)


def _proof(secret: bytes, data: bytes) -> str:
    return hmac.new(secret, data, hashlib.sha256).hexdigest()


def _confirm(
    secret: bytes,
    data: bytes,
    master_tc_pub: Optional[str] = None,
) -> str:
    if master_tc_pub is None:
        return hmac.new(secret, CONFIRM_LABEL + data, hashlib.sha256).hexdigest()
    tail = master_tc_pub.encode("utf-8")
    body = CONFIRM_LABEL_V2 + data + len(tail).to_bytes(4, "big") + tail
    return hmac.new(secret, body, hashlib.sha256).hexdigest()


def approval_words(secret: bytes, data: bytes) -> List[str]:
    """Three words both sides derive from the transcript: matching words mean the same pairing."""
    digest = hmac.new(secret, WORDS_LABEL + data, hashlib.sha256).digest()
    return [WORDS[b] for b in digest[:3]]


def _v2_fields(hello: dict) -> List[str]:
    tc_addr, tc_pub = hello.get("tc_addr", ""), hello.get("tc_pub", "")
    via, lan = hello.get("via", "lan"), hello.get("lan", [])
    if tc_addr and not tokens.is_tc_address(tc_addr):
        raise PairingError("bad tailcat address")
    if not isinstance(tc_pub, str) or len(tc_pub) > MAX_TC_PUB or not tc_pub.isprintable():
        raise PairingError("bad tailcat key")
    if bool(tc_addr) != bool(tc_pub) or via not in ("lan", "tailnet"):
        raise PairingError("bad tailcat fields")
    if not isinstance(lan, list) or (lan and not tokens._addrs_ok(lan)):
        raise PairingError("bad LAN addresses")
    return [tc_addr, tc_pub, via, ",".join(lan)]


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
        approve: Optional[Approve] = None,
        tailnet: Optional[bool] = None,
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
        self._cancelled = False
        self.paired: Optional[Peer] = None
        self._attempts = 0
        self._server: Optional[asyncio.AbstractServer] = None
        self._done: Optional[asyncio.Future] = None
        self._expiry: Optional[asyncio.TimerHandle] = None
        self._approve = approve
        self._use_tailnet = remote.enabled() if tailnet is None else tailnet
        self._net: Any = None
        self.tc: Optional[str] = None

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
        if self._use_tailnet:
            try:
                served = await remote.owned(
                    remote.serve, [self.bound_port], None, None, close = lambda r: r[0].close()
                )
            except Exception as exc:
                logger.info("pairing: no tailcat address (%s); LAN only", type(exc).__name__)
                served = None
            if served is not None and self._cancelled:
                _close_net(served[0])
            elif served is not None:
                self._net, self.tc = served
        return tokens.encode(
            tokens.Invite(
                cluster_id = self.master.cluster_id,
                invite_id = self.invite_id,
                addrs = addrs,
                port = self.bound_port,
                master_fp = self.master.fingerprint,
                secret = self._secret,
                exp = self.expires_at,
                tc = self.tc,
            )
        )

    async def _approved(self, name: str, words: Optional[List[str]]) -> bool:
        if self._approve is None:
            return True
        try:
            return await asyncio.wait_for(_ask(self._approve, name, words), APPROVAL_TIMEOUT_S)
        except Exception as exc:
            logger.info("pairing: approval failed (%s)", type(exc).__name__)
            return False

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
        """Cancel: no worker is recorded after this, unless it already was (``paired``)."""
        self._used = self._cancelled = True
        if self._expiry is not None:
            self._expiry.cancel()
        if self._server is not None:
            self._server.close()
        net, self._net = self._net, None
        if net is not None:
            try:
                asyncio.get_running_loop().run_in_executor(None, _close_net, net)
            except RuntimeError:
                _close_net(net)
        if self._done is not None and not self._done.done():
            if self.paired is not None:
                self._done.set_result(self.paired)
            else:
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
            reason = str(exc) if str(exc) in (NOT_APPROVED, OLD_WORKER) else "pairing refused"
            try:
                await framing.write_frame(writer, {"type": "pair_error", "reason": reason})
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
        v2 = _v2_fields(hello) if hello.get("pair") == PAIR_V2 else None
        session, master_nonce = secrets.token_bytes(16), secrets.token_bytes(32)
        challenge = {
            "type": "pair_challenge",
            "session": _b64e(session),
            "master_nonce": _b64e(master_nonce),
        }
        if v2 is not None:
            challenge["pair"] = PAIR_V2
        await framing.write_frame(writer, challenge)
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
            v2 = v2,
        )
        if not hmac.compare_digest(proof["proof"], _proof(self._secret, data)):
            raise PairingError("bad proof")
        name = _clean_name(hello["name"])
        if v2 is None and self._approve is not None:
            # A pre-v2 worker shows no words to compare, and skipping approval would let anyone
            # holding the code downgrade around it.
            if self._done is not None and not self._done.done():
                self._fail(PairingError(f"{name} {OLD_WORKER}"))
            self.close()
            raise PairingError(OLD_WORKER)
        async with self._lock:
            if self._used:
                raise PairingError("invitation already used")
            self._used = self._claimed = True
        if self._expiry is not None:
            self._expiry.cancel()
        if not await self._approved(name, approval_words(self._secret, data) if v2 else None):
            if self._done is not None and not self._done.done():
                self._fail(PairingError(f"{name} was not approved"))
            self.close()
            raise PairingError(NOT_APPROVED)
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
            tc_addr, tc_pub, via, lan = v2 or ("", "", "lan", "")
            # Through tailcat the peer address is the local forward: record the LAN addresses it sent.
            addrs = (
                [a for a in lan.split(",") if a]
                if via == "tailnet"
                else [peername[0]]
                if peername
                else []
            )
            peer = Peer(
                node_id = node_id,
                name = name,
                role = "worker",
                fingerprint = identity.pem_fingerprint(cert_pem),
                cert_pem = cert_pem.decode("ascii"),
                addrs = addrs,
                agent_port = agent_port,
                added_at = self._now(),
                tc_addr = tc_addr,
                tc_pub = tc_pub,
            )
            issued = {
                "type": "pair_issued",
                "node_id": node_id,
                "worker_cert": cert_pem.decode("ascii"),
                "root_cert": identity.root_pem().decode("ascii"),
                "master_node_id": self.master.node_id,
                "master_cert": identity.own_cert_pem().decode("ascii"),
                "agent_hint": None,
                "confirm": _confirm(self._secret, data),
            }
            if v2 is not None:
                master_tc_pub = (
                    (await asyncio.to_thread(remote.own_pub, "client") or "") if tc_addr else ""
                )
                issued["tc_pub"] = master_tc_pub
                issued["confirm"] = _confirm(self._secret, data, master_tc_pub)
            await framing.write_frame(writer, issued)
            # Recorded only after the worker saved its identity: never a worker unaware it is paired.
            framing.expect(
                await framing.read_frame(reader, FRAME_TIMEOUT_S), "pair_ack", node_id = str
            )
            reg = self.registry or identity.Registry()
            async with _identity_lock():
                if self._cancelled:
                    raise PairingError("the pairing was cancelled")
                now_master = identity.master_identity(create = False)
                if now_master is None or now_master.fingerprint != self.master.fingerprint:
                    raise PairingError("this master was reset while pairing; pair again")
                with reg.transaction():
                    reg.add(peer)
                self.paired = peer
            await framing.write_frame(writer, {"type": "pair_ok"})
            if self._net is not None:
                # Closing the window stops its tunnel: wait for the worker to hang up after pair_ok.
                with contextlib.suppress(Exception):
                    await asyncio.wait_for(reader.read(1), FRAME_TIMEOUT_S)
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


def _close_net(net: Any) -> None:
    try:
        net.close()
    except Exception:
        pass


async def _ask(approve: Approve, name: str, words: Optional[List[str]]) -> bool:
    answer = approve(name, words)
    if asyncio.iscoroutine(answer) or isinstance(answer, asyncio.Future):
        answer = await answer
    return bool(answer)


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
    on_words: Optional[Callable[[List[str]], None]] = None,
) -> LocalIdentity:
    """Worker side: pair with the master named by ``token`` and persist the result.

    ``on_words`` gets the approval words to show while the master's owner compares them."""
    invite = tokens.decode(token)
    async with _identity_lock():
        return await _join_locked(
            invite, name, devices, timeout, now, replace, agent_port, on_words
        )


async def _connect_any(invite, timeout: float):
    """LAN addresses first, then the master's tailcat address: (addr, reader, writer, net or None)."""
    try:
        addr, reader, writer = await asyncio.wait_for(
            _connect(invite.addrs, invite.port, invite.master_fp), timeout
        )
        return addr, reader, writer, None
    except (PairingError, asyncio.TimeoutError) as exc:
        if not invite.tc or not remote.enabled():
            raise
        lan_error = exc
    net = None
    try:
        net, port = await remote.owned(
            remote.dial, invite.tc, invite.port, close = lambda r: r[0].close()
        )
        addr, reader, writer = await asyncio.wait_for(
            _connect(["127.0.0.1"], port, invite.master_fp), timeout
        )
    except BaseException as exc:
        if net is not None:
            asyncio.get_running_loop().run_in_executor(None, _close_net, net)
        if not isinstance(exc, Exception):
            raise
        if isinstance(lan_error, asyncio.TimeoutError):
            raise PairingError(f"could not reach the master on the LAN or through tailcat ({exc})")
        raise PairingError(
            f"{_clause(lan_error)}, and not through tailcat either ({exc})"
        ) from None
    return addr, reader, writer, net


def _clause(exc: BaseException) -> str:
    return str(exc).strip().rstrip(".") or type(exc).__name__


async def _join_locked(
    invite, name, devices, timeout, now, replace, agent_port, on_words
) -> LocalIdentity:
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
    own = None
    if remote.enabled():
        try:
            own = await asyncio.to_thread(remote.worker_address, agent_port, held = True)
        except Exception as exc:
            logger.info("pairing: no tailcat address for this worker (%s)", type(exc).__name__)

    async def answer(reader, wait: float = FRAME_TIMEOUT_S) -> dict:
        frame = await framing.read_frame(reader, wait)
        if frame.get("type") == "pair_error":
            if frame.get("reason") == NOT_APPROVED:
                raise PairingError("the master's owner did not approve this machine")
            raise PairingError("the master refused this pairing (token used, expired or wrong)")
        return frame

    async def handshake(reader, writer, via: str) -> dict:
        tc_addr, tc_pub = own if own else ("", "")
        lan = local_ipv4_addresses()[:16] if via == "tailnet" else []
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
        hello.update(pair = PAIR_V2, tc_addr = tc_addr, tc_pub = tc_pub, via = via, lan = lan)
        await framing.write_frame(writer, hello)
        challenge = framing.expect(
            await answer(reader), "pair_challenge", session = str, master_nonce = str
        )
        v2 = challenge.get("pair") == PAIR_V2
        data = transcript(
            cluster_id = invite.cluster_id,
            invite_id = invite.invite_id,
            session = _b64d(challenge["session"], 16, "session"),
            worker_nonce = worker_nonce,
            master_nonce = _b64d(challenge["master_nonce"], 32, "master nonce"),
            master_fp = invite.master_fp,
            csr_digest = hashlib.sha256(der).digest(),
            v2 = [tc_addr, tc_pub, via, ",".join(lan)] if v2 else None,
        )
        await framing.write_frame(
            writer, {"type": "pair_proof", "proof": _proof(invite.secret, data)}
        )
        if v2 and on_words is not None:
            on_words(approval_words(invite.secret, data))
        issued = framing.expect(
            await answer(reader, APPROVAL_TIMEOUT_S + FRAME_TIMEOUT_S if v2 else FRAME_TIMEOUT_S),
            "pair_issued",
            node_id = str,
            worker_cert = str,
            root_cert = str,
            master_node_id = str,
            master_cert = str,
            confirm = str,
        )
        master_tc_pub = issued.get("tc_pub") if v2 else None
        if v2 and not isinstance(master_tc_pub, str):
            raise PairingError("the master sent no tailcat key")
        if not hmac.compare_digest(issued["confirm"], _confirm(invite.secret, data, master_tc_pub)):
            raise PairingError("the master could not prove it holds the token secret")
        issued["_tc_pub"] = master_tc_pub or ""
        return issued

    async def run():
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        addr, reader, writer, net = await _connect_any(invite, timeout)
        via = "lan" if net is None else "tailnet"
        try:
            # The approval wait is the master owner's, not this deadline's: each frame has its own.
            issued = (
                await handshake(reader, writer, via)
                if on_words
                else await asyncio.wait_for(
                    handshake(reader, writer, via), max(0.1, deadline - loop.time())
                )
            )
            local = _save(None if net is not None else addr, issued)
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
            if net is not None:
                net.close()

    def _save(addr: Optional[str], issued: dict) -> LocalIdentity:
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
                    addrs = ([addr] if addr else []) + [a for a in invite.addrs if a != addr],
                    added_at = now(),
                    tc_pub = issued.get("_tc_pub", ""),
                )
            )
        return local

    return await run()
