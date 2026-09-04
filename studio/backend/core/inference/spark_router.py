# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Asynchronous request router for llama-server replicas on a DGX Spark pair.

Two Sparks cabled together are two hosts, so the fastest way to serve a model that fits
on one of them is one llama-server per node and a front door that spreads requests over
both. That front door is this module. It runs inside the Studio backend's event loop and
listens on loopback; ``LlamaCppBackend.base_url`` points at it while replicas are active,
so every existing Studio code path (the threaded chat generator, the OpenAI pass-through,
tokenize, slots, health) goes through it without knowing it exists.

Design, and what it does and does not borrow from vLLM
------------------------------------------------------
vLLM's V1 ``AsyncLLM`` (vllm/v1/engine/async_llm.py) keeps HTTP handling non-blocking by
never running the engine in the request task: ``generate()`` makes a per-request
``AsyncStream``, adds the request to the ``EngineCore`` ("separate process"), and "a
separate output_handler loop runs in a background AsyncIO task, pulling outputs from
EngineCore and putting them into the per-request AsyncStream". The engine step loop and
the handlers only ever meet through queues, so a slow step never stalls the loop
(https://docs.vllm.ai/en/v0.9.1/api/vllm/v1/engine/async_llm.html).

Mirrored here: request handlers never block. Every upstream exchange is an ``httpx``
async stream awaited chunk by chunk, health probing lives in its own background task
(``_health_loop``), admission is an ``asyncio.Condition`` per backend rather than a
thread lock, and the listener is a plain ``asyncio.start_server`` on the same loop.

Not mirrored: there is no single engine core and no output queue to demultiplex. The
engines are separate llama-server processes with their own HTTP servers, own schedulers
and own KV caches, so a request is a connection, not a queue entry, and the response
bytes go straight from the upstream socket to the client socket. That is also why prefix
caching needs care (below): vLLM's one engine has one prefix cache, two llama-servers
have two.

Routing and prefix caching
--------------------------
llama-server keeps its KV cache and prompt cache per process. A conversation whose turns
alternate between replicas re-prefills its whole history on every turn, so routing is
sticky: a request with a conversation key (the chat thread id, a session id, or a hash of
the prompt prefix as a fallback) is mapped to a backend by consistent hashing over the
healthy set. The trade-off is uneven load: a burst of turns in one conversation cannot
spread across both nodes. Keyless requests fall back to least-outstanding-requests. When
the sticky backend is at capacity and its queue is full the request overflows to the
other node, paying one re-prefill rather than being refused.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Awaitable, Callable, Dict, List, Optional, Tuple

import httpx

logger = logging.getLogger(__name__)

# The chat path puts the thread id in the JSON body under this key (see
# spark_serving.tag_conversation); the router pops it before forwarding so llama-server
# never sees a field it does not know. Header form for external clients.
CONVERSATION_FIELD = "unsloth_conversation"
CONVERSATION_HEADER = "x-unsloth-conversation"

# Paths that generate and therefore benefit from KV locality and need admission control.
# Everything else (health, props, slots, tokenize, apply-template, metrics) is answered
# by the primary node, whose process the rest of Studio already manages.
GENERATION_PATHS = frozenset(
    {
        "/v1/chat/completions",
        "/chat/completions",
        "/completion",
        "/completions",
        "/v1/completions",
        "/infill",
        "/v1/embeddings",
        "/embeddings",
        "/embedding",
        "/rerank",
        "/reranking",
        "/v1/rerank",
    }
)

_HOP_BY_HOP = frozenset(
    {
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailer",
        "transfer-encoding",
        "upgrade",
        "host",
        "content-length",
    }
)

# Characters of prompt used for the fallback key. A conversation's prefix (system prompt
# plus first user turn) is stable across turns, which is exactly the part llama-server's
# prompt cache can reuse. Characters rather than tokens: the router does not tokenize.
PREFIX_KEY_CHARS = 1024

_VIRTUAL_NODES = 64
_HEAD_LIMIT = 64 * 1024
_BODY_LIMIT = 256 * 1024 * 1024
_READ_CHUNK = 64 * 1024

_REASONS = {
    200: "OK",
    400: "Bad Request",
    404: "Not Found",
    413: "Payload Too Large",
    500: "Internal Server Error",
    502: "Bad Gateway",
    503: "Service Unavailable",
    504: "Gateway Timeout",
}


class RouterError(Exception):
    """A request the router could not place; carries the HTTP status to answer with."""

    def __init__(
        self,
        status: int,
        message: str,
        retry_after: Optional[int] = None,
    ):
        super().__init__(message)
        self.status = status
        self.message = message
        self.retry_after = retry_after


class UpstreamUnreachable(RouterError):
    """No backend accepted the connection.

    The listener answers this by closing the client connection without a response, so
    an ``httpx`` caller sees the same ``RemoteProtocolError`` a dead llama-server
    produces and ``LlamaCppBackend._respawn_if_dead`` keeps working unchanged.
    """

    def __init__(self, message: str):
        super().__init__(502, message)


@dataclass
class Backend:
    """One llama-server process and the bookkeeping the router keeps on it."""

    name: str
    host: str
    port: int
    slots: int
    queue_limit: int
    primary: bool = False
    healthy: bool = False
    in_flight: int = 0
    queued: int = 0
    served: int = 0
    failures: int = 0
    consecutive_failures: int = 0
    last_error: str = ""
    last_check: float = 0.0
    slots_busy: Optional[int] = None
    client: Optional[httpx.AsyncClient] = None
    _cond: Optional[asyncio.Condition] = field(default = None, repr = False)

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    @property
    def capacity(self) -> int:
        return max(1, int(self.slots))

    def cond(self) -> asyncio.Condition:
        if self._cond is None:
            self._cond = asyncio.Condition()
        return self._cond

    def snapshot(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "host": self.host,
            "port": self.port,
            "primary": self.primary,
            "healthy": self.healthy,
            "slots": self.slots,
            "queue_limit": self.queue_limit,
            "in_flight": self.in_flight,
            "queued": self.queued,
            "slots_busy": self.slots_busy,
            "served": self.served,
            "failures": self.failures,
            "last_error": self.last_error,
            "last_check": self.last_check,
        }


@dataclass
class Routed:
    """An upstream response ready to relay: status, filtered headers, body iterator."""

    status: int
    headers: List[Tuple[str, str]]
    body: AsyncIterator[bytes]
    backend: Backend
    close: Callable[[], Awaitable[None]]


def _stable_hash(text: str) -> int:
    return int.from_bytes(hashlib.blake2b(text.encode("utf-8"), digest_size = 8).digest(), "big")


def _prompt_prefix(body: Dict[str, Any]) -> str:
    """The stable prefix of a request: system prompt plus the first user turn, or the
    raw prompt. Truncated so a huge first message does not cost a hash of megabytes."""
    messages = body.get("messages")
    if isinstance(messages, list) and messages:
        parts: List[str] = []
        first_user_seen = False
        for message in messages:
            if not isinstance(message, dict):
                continue
            role = str(message.get("role", ""))
            content = message.get("content")
            if isinstance(content, list):
                content = " ".join(
                    str(part.get("text", ""))
                    for part in content
                    if isinstance(part, dict) and part.get("type", "text") == "text"
                )
            text = str(content or "")
            if role == "system":
                parts.append("system:" + text)
                continue
            if role == "user" and not first_user_seen:
                parts.append("user:" + text)
                first_user_seen = True
                break
        return "\n".join(parts)[:PREFIX_KEY_CHARS]
    prompt = body.get("prompt")
    if isinstance(prompt, list):
        prompt = " ".join(str(p) for p in prompt)
    if isinstance(prompt, str):
        return prompt[:PREFIX_KEY_CHARS]
    inp = body.get("input")
    if isinstance(inp, str):
        return inp[:PREFIX_KEY_CHARS]
    if isinstance(inp, list):
        return " ".join(str(p) for p in inp)[:PREFIX_KEY_CHARS]
    return ""


def conversation_key(headers: Dict[str, str], body: Optional[Dict[str, Any]]) -> Optional[str]:
    """The stickiness key for a request, or None when nothing identifies a conversation.

    Precedence: the explicit header, then explicit body fields (Studio's tag, then the
    session and user ids OpenAI-style clients send), then a hash of the prompt prefix.
    """
    header = (headers.get(CONVERSATION_HEADER) or "").strip()
    if header:
        return header
    if not isinstance(body, dict):
        return None
    for name in (CONVERSATION_FIELD, "conversation_id", "thread_id", "session_id", "user"):
        value = body.get(name)
        if isinstance(value, (str, int)) and str(value).strip():
            return f"{name}:{value}"
    prefix = _prompt_prefix(body)
    if prefix:
        return "prefix:" + hashlib.sha1(prefix.encode("utf-8")).hexdigest()
    return None


class SparkRouter:
    """Spreads requests over llama-server backends without blocking the event loop."""

    def __init__(
        self,
        *,
        listen_host: str = "127.0.0.1",
        listen_port: int = 0,
        health_interval: float = 2.0,
        health_timeout: float = 2.0,
        unhealthy_after: int = 2,
        queue_wait_s: float = 120.0,
        connect_timeout: float = 5.0,
        on_backend_down: Optional[Callable[[Backend], Awaitable[None]]] = None,
        on_backend_up: Optional[Callable[[Backend], Awaitable[None]]] = None,
    ):
        self.listen_host = listen_host
        self._requested_port = listen_port
        self.listen_port: Optional[int] = None
        self.health_interval = health_interval
        self.health_timeout = health_timeout
        self.unhealthy_after = max(1, unhealthy_after)
        self.queue_wait_s = queue_wait_s
        self.connect_timeout = connect_timeout
        self.on_backend_down = on_backend_down
        self.on_backend_up = on_backend_up
        self.backends: List[Backend] = []
        self._server: Optional[asyncio.AbstractServer] = None
        self._health_task: Optional[asyncio.Task] = None
        self._started = False
        self.started_at: Optional[float] = None
        self.routed_sticky = 0
        self.routed_keyless = 0
        self.rejected = 0

    # ── Backends ──────────────────────────────────────────────────────────────

    def add_backend(
        self,
        name: str,
        host: str,
        port: int,
        slots: int,
        *,
        primary: bool = False,
        queue_limit: Optional[int] = None,
    ) -> Backend:
        if queue_limit is None:
            # A small queue: enough to absorb a burst between two slot releases, not
            # enough to hide a node that is saturated.
            queue_limit = max(2, min(8, int(slots) // 2))
        backend = Backend(
            name = name,
            host = host,
            port = port,
            slots = max(1, int(slots)),
            queue_limit = max(0, int(queue_limit)),
            primary = primary,
        )
        self.backends.append(backend)
        if self._started:
            backend.client = self._new_client(backend)
        return backend

    async def remove_backend(self, name: str) -> None:
        keep = []
        for backend in self.backends:
            if backend.name == name:
                await self._close_client(backend)
            else:
                keep.append(backend)
        self.backends = keep

    def get_backend(self, name: str) -> Optional[Backend]:
        for backend in self.backends:
            if backend.name == name:
                return backend
        return None

    async def set_backend_address(self, name: str, host: str, port: int) -> None:
        """Re-point a backend after its process was respawned on a new port."""
        backend = self.get_backend(name)
        if backend is None or (backend.host == host and backend.port == port):
            return
        await self._close_client(backend)
        backend.host, backend.port = host, port
        backend.healthy = False
        backend.consecutive_failures = 0
        if self._started:
            backend.client = self._new_client(backend)

    def _new_client(self, backend: Backend) -> httpx.AsyncClient:
        # One pool per backend. Generation streams are long, so the pool must hold
        # every slot plus the queue; reads have no timeout because a slot can wait on
        # other slots' prefill for a long time and the Studio side already enforces
        # its own first-token and stall deadlines.
        pool = backend.capacity + backend.queue_limit + 4
        return httpx.AsyncClient(
            base_url = backend.base_url,
            limits = httpx.Limits(max_connections = pool, max_keepalive_connections = pool),
            timeout = httpx.Timeout(connect = self.connect_timeout, read = None, write = 30.0, pool = None),
            trust_env = False,
        )

    async def _close_client(self, backend: Backend) -> None:
        client, backend.client = backend.client, None
        if client is not None:
            try:
                await client.aclose()
            except Exception:
                pass

    @property
    def primary(self) -> Optional[Backend]:
        for backend in self.backends:
            if backend.primary:
                return backend
        return self.backends[0] if self.backends else None

    def healthy_backends(self) -> List[Backend]:
        return [b for b in self.backends if b.healthy]

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def start(self, *, listen: bool = True) -> None:
        if self._started:
            return
        self._started = True
        self.started_at = time.time()
        for backend in self.backends:
            if backend.client is None:
                backend.client = self._new_client(backend)
        await self.check_health()
        self._health_task = asyncio.create_task(self._health_loop())
        if listen:
            self._server = await asyncio.start_server(
                self._serve_connection, self.listen_host, self._requested_port
            )
            sockets = self._server.sockets or ()
            if sockets:
                self.listen_port = sockets[0].getsockname()[1]
            logger.info(
                "spark router listening on %s:%s for %s",
                self.listen_host,
                self.listen_port,
                ", ".join(f"{b.name}={b.base_url}" for b in self.backends),
            )

    async def stop(self) -> None:
        if not self._started:
            return
        self._started = False
        task, self._health_task = self._health_task, None
        if task is not None:
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
        server, self._server = self._server, None
        if server is not None:
            server.close()
            try:
                await asyncio.wait_for(server.wait_closed(), timeout = 2.0)
            except (asyncio.TimeoutError, Exception):
                pass
        for backend in self.backends:
            await self._close_client(backend)
            async with backend.cond():
                backend.cond().notify_all()
        self.listen_port = None

    @property
    def running(self) -> bool:
        return self._started

    @property
    def base_url(self) -> Optional[str]:
        if self.listen_port is None:
            return None
        return f"http://{self.listen_host}:{self.listen_port}"

    # ── Health ────────────────────────────────────────────────────────────────

    async def _health_loop(self) -> None:
        while self._started:
            try:
                await asyncio.sleep(self.health_interval)
                await self.check_health()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.debug("spark router health loop error", exc_info = True)

    async def check_health(self) -> None:
        await asyncio.gather(*(self._check_one(b) for b in list(self.backends)))

    async def _check_one(self, backend: Backend) -> None:
        client = backend.client
        if client is None:
            return
        ok, error = False, ""
        try:
            resp = await client.get("/health", timeout = self.health_timeout)
            if resp.status_code == 200:
                ok = True
            else:
                error = f"/health returned {resp.status_code}"
        except httpx.HTTPError as exc:
            error = f"{type(exc).__name__}: {exc}"[:200]
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"[:200]
        if ok:
            try:
                resp = await client.get("/slots", timeout = self.health_timeout)
                if resp.status_code == 200:
                    data = resp.json()
                    if isinstance(data, list):
                        backend.slots_busy = sum(
                            1 for s in data if isinstance(s, dict) and s.get("is_processing")
                        )
                # A build with --no-slots answers 501: not a health signal.
            except Exception:
                backend.slots_busy = None
        backend.last_check = time.time()
        await self._record_probe(backend, ok, error)

    async def _record_probe(self, backend: Backend, ok: bool, error: str) -> None:
        if ok:
            backend.consecutive_failures = 0
            backend.last_error = ""
            if not backend.healthy:
                backend.healthy = True
                logger.info(
                    "spark router: backend %s is healthy (%s)", backend.name, backend.base_url
                )
                await self._notify(self.on_backend_up, backend)
                async with backend.cond():
                    backend.cond().notify_all()
            return
        backend.consecutive_failures += 1
        backend.last_error = error
        if backend.healthy and backend.consecutive_failures >= self.unhealthy_after:
            await self.mark_down(backend, error)

    async def mark_down(self, backend: Backend, error: str) -> None:
        """Take a backend out of rotation now; the health loop puts it back."""
        backend.last_error = error
        backend.failures += 1
        was_healthy = backend.healthy
        backend.healthy = False
        backend.consecutive_failures = max(backend.consecutive_failures, self.unhealthy_after)
        async with backend.cond():
            backend.cond().notify_all()
        if was_healthy:
            logger.warning("spark router: backend %s out of rotation: %s", backend.name, error)
            await self._notify(self.on_backend_down, backend)

    async def _notify(self, callback, backend: Backend) -> None:
        if callback is None:
            return
        try:
            await callback(backend)
        except Exception:
            logger.warning("spark router: backend callback failed", exc_info = True)

    # ── Selection ─────────────────────────────────────────────────────────────

    def _ring(self, candidates: List[Backend]) -> List[Tuple[int, Backend]]:
        points = []
        for backend in candidates:
            for i in range(_VIRTUAL_NODES):
                points.append((_stable_hash(f"{backend.name}#{i}"), backend))
        points.sort(key = lambda p: p[0])
        return points

    def pick(
        self,
        key: Optional[str],
        candidates: Optional[List[Backend]] = None,
    ) -> Optional[Backend]:
        """Consistent hashing when there is a key, least outstanding otherwise."""
        candidates = self.healthy_backends() if candidates is None else candidates
        if not candidates:
            return None
        if len(candidates) == 1:
            return candidates[0]
        if key:
            point = _stable_hash(key)
            ring = self._ring(candidates)
            for h, backend in ring:
                if h >= point:
                    return backend
            return ring[0][1]
        return min(candidates, key = lambda b: (b.in_flight + b.queued, b.name))

    # ── Admission ─────────────────────────────────────────────────────────────

    def _has_room(self, backend: Backend) -> bool:
        return backend.in_flight < backend.capacity

    async def _acquire(self, backend: Backend) -> None:
        """Take a slot on ``backend``, waiting in its bounded queue when full."""
        cond = backend.cond()
        async with cond:
            if self._has_room(backend):
                backend.in_flight += 1
                return
            if backend.queued >= backend.queue_limit:
                raise RouterError(503, f"backend {backend.name} is at capacity", retry_after = 1)
            backend.queued += 1
            try:
                deadline = time.monotonic() + self.queue_wait_s
                while not self._has_room(backend):
                    if not backend.healthy or not self._started:
                        raise UpstreamUnreachable(f"backend {backend.name} went away while queued")
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise RouterError(
                            503, f"backend {backend.name} queue wait timed out", retry_after = 2
                        )
                    try:
                        await asyncio.wait_for(cond.wait(), timeout = remaining)
                    except asyncio.TimeoutError:
                        pass
                backend.in_flight += 1
            finally:
                backend.queued -= 1

    async def _release(self, backend: Backend) -> None:
        cond = backend.cond()
        async with cond:
            backend.in_flight = max(0, backend.in_flight - 1)
            cond.notify()

    def _choose(self, key: Optional[str]) -> Backend:
        healthy = self.healthy_backends()
        if not healthy:
            raise UpstreamUnreachable("no healthy llama-server backend")
        target = self.pick(key, healthy)
        assert target is not None
        if key and not self._has_room(target) and target.queued >= target.queue_limit:
            # Sticky target saturated and its queue full: overflow to the emptiest other
            # node. Costs one re-prefill on that node; refusing would cost the request.
            others = [
                b
                for b in healthy
                if b is not target and (self._has_room(b) or b.queued < b.queue_limit)
            ]
            if others:
                target = min(others, key = lambda b: (b.in_flight + b.queued, b.name))
        return target

    # ── Dispatch ──────────────────────────────────────────────────────────────

    async def dispatch(
        self, method: str, path: str, headers: Dict[str, str], body: bytes
    ) -> Routed:
        """Forward one request and return the upstream response for relaying.

        Raises ``RouterError`` for a request that cannot be placed and
        ``UpstreamUnreachable`` when the chosen backend refuses the connection.
        """
        if not self._started:
            raise UpstreamUnreachable("router stopped")
        route_path = path.split("?", 1)[0]
        is_generation = route_path in GENERATION_PATHS
        parsed: Optional[Dict[str, Any]] = None
        if is_generation and body:
            try:
                candidate = json.loads(body)
                if isinstance(candidate, dict):
                    parsed = candidate
            except (ValueError, UnicodeDecodeError):
                parsed = None
        key = conversation_key(headers, parsed) if is_generation else None
        if parsed is not None and CONVERSATION_FIELD in parsed:
            parsed.pop(CONVERSATION_FIELD, None)
            body = json.dumps(parsed, ensure_ascii = False).encode("utf-8")

        if is_generation:
            backend = self._choose(key)
            if key:
                self.routed_sticky += 1
            else:
                self.routed_keyless += 1
            try:
                await self._acquire(backend)
            except RouterError:
                self.rejected += 1
                raise
            admitted = True
        else:
            backend = self.primary
            if backend is None:
                raise UpstreamUnreachable("no backend configured")
            admitted = False

        client = backend.client
        if client is None:
            if admitted:
                await self._release(backend)
            raise UpstreamUnreachable(f"backend {backend.name} has no client")

        upstream_headers = [(k, v) for k, v in headers.items() if k.lower() not in _HOP_BY_HOP]
        request = client.build_request(method, path, headers = upstream_headers, content = body)
        try:
            response = await client.send(request, stream = True)
        except (httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout) as exc:
            if admitted:
                await self._release(backend)
            await self.mark_down(backend, f"{type(exc).__name__}: {exc}"[:200])
            raise UpstreamUnreachable(f"backend {backend.name} unreachable: {exc}") from exc
        except httpx.HTTPError as exc:
            if admitted:
                await self._release(backend)
            backend.failures += 1
            backend.last_error = f"{type(exc).__name__}: {exc}"[:200]
            raise RouterError(502, f"backend {backend.name} failed: {exc}") from exc

        released = False

        async def _close() -> None:
            nonlocal released
            try:
                await response.aclose()
            except Exception:
                pass
            if admitted and not released:
                released = True
                await self._release(backend)

        async def _body() -> AsyncIterator[bytes]:
            try:
                async for chunk in response.aiter_raw():
                    if chunk:
                        yield chunk
                backend.served += 1
            except httpx.HTTPError as exc:
                backend.failures += 1
                backend.last_error = f"{type(exc).__name__}: {exc}"[:200]
                # A transport failure mid-body means the process is gone or wedged. Take
                # it out of rotation now rather than after two more probes; the health
                # loop restores it when /health answers again.
                await self.mark_down(backend, backend.last_error)
                raise

        out_headers = [
            (k, v) for k, v in response.headers.multi_items() if k.lower() not in _HOP_BY_HOP
        ]
        return Routed(
            status = response.status_code,
            headers = out_headers,
            body = _body(),
            backend = backend,
            close = _close,
        )

    # ── Status ────────────────────────────────────────────────────────────────

    def status(self) -> Dict[str, Any]:
        backends = [b.snapshot() for b in self.backends]
        return {
            "running": self._started,
            "listen": self.base_url,
            "started_at": self.started_at,
            "queue_depth": sum(b.queued for b in self.backends),
            "in_flight": sum(b.in_flight for b in self.backends),
            "healthy_backends": sum(1 for b in self.backends if b.healthy),
            "routed_sticky": self.routed_sticky,
            "routed_keyless": self.routed_keyless,
            "rejected": self.rejected,
            "backends": backends,
        }

    # ── Loopback HTTP/1.1 listener ────────────────────────────────────────────
    # Hand-rolled on purpose: the Studio process already runs uvicorn for its own app
    # and a second uvicorn.Server in the same loop wants the signal handlers; a raw
    # asyncio server needs nothing, parses only what httpx sends (Content-Length or
    # chunked bodies), and relays response bodies as chunked transfer encoding so a
    # token stream goes out the moment it comes in.

    async def _serve_connection(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        try:
            while True:
                keep_alive = await self._serve_one(reader, writer)
                if not keep_alive:
                    break
        except (asyncio.IncompleteReadError, ConnectionError, asyncio.LimitOverrunError):
            pass
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.debug("spark router connection error", exc_info = True)
        finally:
            try:
                writer.close()
            except Exception:
                pass

    async def _serve_one(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> bool:
        try:
            head = await reader.readuntil(b"\r\n\r\n")
        except asyncio.IncompleteReadError:
            return False
        except asyncio.LimitOverrunError:
            await self._write_error(writer, 413, "request head too large")
            return False
        if len(head) > _HEAD_LIMIT:
            await self._write_error(writer, 413, "request head too large")
            return False
        lines = head.decode("latin-1").split("\r\n")
        request_line = lines[0].split(" ")
        if len(request_line) < 3:
            await self._write_error(writer, 400, "malformed request line")
            return False
        method, path, version = request_line[0], request_line[1], request_line[2]
        headers: Dict[str, str] = {}
        for line in lines[1:]:
            if not line:
                continue
            name, sep, value = line.partition(":")
            if sep:
                headers[name.strip().lower()] = value.strip()
        client_wants_close = (
            headers.get("connection", "").lower() == "close" or version == "HTTP/1.0"
        )

        if headers.get("expect", "").lower() == "100-continue":
            writer.write(b"HTTP/1.1 100 Continue\r\n\r\n")
            await writer.drain()
        body = b""
        if "content-length" in headers:
            try:
                length = int(headers["content-length"])
            except ValueError:
                await self._write_error(writer, 400, "bad content-length")
                return False
            if length > _BODY_LIMIT:
                await self._write_error(writer, 413, "request body too large")
                return False
            body = await reader.readexactly(length) if length else b""
        elif "chunked" in headers.get("transfer-encoding", "").lower():
            body = await self._read_chunked(reader)

        try:
            routed = await self.dispatch(method, path, headers, body)
        except UpstreamUnreachable as exc:
            logger.warning("spark router: %s", exc.message)
            # No response at all: see UpstreamUnreachable.
            return False
        except RouterError as exc:
            await self._write_error(writer, exc.status, exc.message, retry_after = exc.retry_after)
            return not client_wants_close

        # Watch the client socket while relaying so a disconnect during a long prefill
        # (no bytes flowing yet) tears the upstream stream down promptly, which is what
        # makes llama-server stop decoding for a request nobody is reading.
        disconnected = asyncio.Event()
        pipelined = asyncio.Event()

        async def _watch() -> None:
            try:
                data = await reader.read(1)
            except Exception:
                data = b""
            if not data:
                disconnected.set()
            else:
                pipelined.set()

        watcher = asyncio.create_task(_watch())
        try:
            await self._relay(routed, writer, method, disconnected)
        finally:
            watcher.cancel()
            try:
                await watcher
            except (asyncio.CancelledError, Exception):
                pass
            await routed.close()
        if disconnected.is_set() or pipelined.is_set():
            # A byte that arrived mid-response belongs to a pipelined request whose
            # first byte the watcher consumed; close so the client resends it on a
            # fresh connection (httpx never pipelines, so this is a guard, not a path).
            return False
        # Responses are relayed chunked and the request is fully consumed, so the
        # connection can carry another request unless the client asked otherwise.
        return not client_wants_close

    async def _relay(
        self, routed: Routed, writer: asyncio.StreamWriter, method: str, disconnected: asyncio.Event
    ) -> None:
        reason = _REASONS.get(routed.status, "OK")
        no_body = method == "HEAD" or routed.status in (204, 304) or 100 <= routed.status < 200
        head = [f"HTTP/1.1 {routed.status} {reason}"]
        for name, value in routed.headers:
            head.append(f"{name}: {value}")
        if not no_body:
            head.append("Transfer-Encoding: chunked")
        head.append("Connection: keep-alive")
        writer.write(("\r\n".join(head) + "\r\n\r\n").encode("latin-1"))
        await writer.drain()
        if no_body:
            return
        disconnect_wait = asyncio.create_task(disconnected.wait())
        try:
            body_iter = routed.body
            while True:
                next_chunk = asyncio.ensure_future(body_iter.__anext__())
                done, _ = await asyncio.wait(
                    {next_chunk, disconnect_wait}, return_when = asyncio.FIRST_COMPLETED
                )
                if disconnect_wait in done and next_chunk not in done:
                    next_chunk.cancel()
                    try:
                        await next_chunk
                    except (asyncio.CancelledError, StopAsyncIteration, Exception):
                        pass
                    return
                try:
                    chunk = next_chunk.result()
                except StopAsyncIteration:
                    break
                except httpx.HTTPError as exc:
                    # Upstream died after the headers went out. Tell the client in-band
                    # in the same shape llama-server uses for its own mid-stream errors,
                    # then end the chunked body cleanly so nothing waits on a hang.
                    message = (
                        f"Lost connection to llama-server on {routed.backend.name} mid-response "
                        f"({type(exc).__name__}); the request cannot be resumed."
                    )
                    frame = (
                        "data: "
                        + json.dumps(
                            {"error": {"code": 502, "message": message, "type": "server_error"}}
                        )
                        + "\n\n"
                    )
                    await self._write_chunk(writer, frame.encode("utf-8"))
                    break
                await self._write_chunk(writer, chunk)
            writer.write(b"0\r\n\r\n")
            await writer.drain()
        finally:
            disconnect_wait.cancel()
            try:
                await disconnect_wait
            except (asyncio.CancelledError, Exception):
                pass

    @staticmethod
    async def _write_chunk(writer: asyncio.StreamWriter, chunk: bytes) -> None:
        writer.write(f"{len(chunk):x}\r\n".encode("ascii") + chunk + b"\r\n")
        await writer.drain()

    @staticmethod
    async def _write_error(
        writer: asyncio.StreamWriter,
        status: int,
        message: str,
        retry_after: Optional[int] = None,
    ) -> None:
        payload = json.dumps(
            {"error": {"code": status, "message": message, "type": "spark_router"}}
        ).encode("utf-8")
        head = [
            f"HTTP/1.1 {status} {_REASONS.get(status, 'Error')}",
            "Content-Type: application/json",
            f"Content-Length: {len(payload)}",
            "Connection: keep-alive",
        ]
        if retry_after is not None:
            head.append(f"Retry-After: {retry_after}")
        try:
            writer.write(("\r\n".join(head) + "\r\n\r\n").encode("latin-1") + payload)
            await writer.drain()
        except Exception:
            pass

    @staticmethod
    async def _read_chunked(reader: asyncio.StreamReader) -> bytes:
        parts: List[bytes] = []
        total = 0
        while True:
            size_line = await reader.readuntil(b"\r\n")
            size = int(size_line.split(b";", 1)[0].strip() or b"0", 16)
            if size == 0:
                # Trailers, then the blank line.
                while True:
                    line = await reader.readuntil(b"\r\n")
                    if line == b"\r\n":
                        break
                return b"".join(parts)
            total += size
            if total > _BODY_LIMIT:
                raise asyncio.LimitOverrunError("request body too large", total)
            parts.append(await reader.readexactly(size))
            await reader.readexactly(2)
