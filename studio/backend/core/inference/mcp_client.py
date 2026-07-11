# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import asyncio
import atexit
import concurrent.futures
import hashlib
import json
import os
import shlex
import sys
import threading
import time
import uuid
from typing import Any, Optional

from loggers import get_logger

logger = get_logger(__name__)

MCP_TOOL_PREFIX = "mcp__"

# A failed probe isn't cached (a recovered server must come back), but it's
# recorded so a down server isn't re-probed -- and the chat send re-hung for
# the full timeout -- on every message. Cool off for this long after a failure;
# much longer for OAuth, whose probe can hang up to _OAUTH_PROBE_TIMEOUT,
# so that hang doesn't recur every minute.
FAILED_PROBE_COOLOFF_SECONDS = 60.0
OAUTH_FAILED_PROBE_COOLOFF_SECONDS = 300.0

_oauth_token_store = None


def is_stdio(address: str) -> bool:
    """A non-HTTP address is a local stdio command, e.g.
    'npx -y @modelcontextprotocol/server-filesystem /path'."""
    return not address.strip().lower().startswith(("http://", "https://"))


def _split_windows_command_line(address: str) -> list[str]:
    """Parse a Windows command line using the same backslash/quote rules that
    subprocess.list2cmdline() writes. This keeps trailing backslashes before a
    closing quote from being doubled in the resulting argv."""
    parts: list[str] = []
    current: list[str] = []
    in_quotes = False
    backslashes = 0
    arg_started = False
    i = 0

    while i < len(address):
        ch = address[i]
        if ch == "\\":
            backslashes += 1
            i += 1
            continue
        if ch == '"':
            current.extend("\\" * (backslashes // 2))
            if backslashes % 2:
                current.append('"')
            else:
                in_quotes = not in_quotes
            arg_started = True
            backslashes = 0
            i += 1
            continue
        if ch.isspace() and not in_quotes:
            if backslashes:
                current.extend("\\" * backslashes)
                arg_started = True
                backslashes = 0
            if arg_started or current:
                parts.append("".join(current))
                current = []
                arg_started = False
            i += 1
            while i < len(address) and address[i].isspace():
                i += 1
            continue
        if backslashes:
            current.extend("\\" * backslashes)
            arg_started = True
            backslashes = 0
        current.append(ch)
        arg_started = True
        i += 1

    if backslashes:
        current.extend("\\" * backslashes)
        arg_started = True
    if in_quotes:
        raise ValueError("No closing quotation")
    if arg_started or current:
        parts.append("".join(current))
    return parts


def parse_stdio_command(address: str) -> list[str]:
    """Split a stdio command line into argv. Shared by route validation and the
    transport so both agree on quoting (notably Windows backslash paths)."""
    posix = sys.platform != "win32"
    if posix:
        return shlex.split(address, posix = posix)
    if address.lstrip().startswith("'"):
        raise ValueError("Single-quoted executables are not supported on Windows")
    return _split_windows_command_line(address)


def join_stdio_command(parts: list[str]) -> str:
    """Inverse of parse_stdio_command: join argv into a single command string
    that parse_stdio_command() splits back into ``parts`` on this platform.
    Config files (issue #5936) carry structured command + args; storage holds
    one string in the url field. Windows uses list2cmdline so spaced/backslash
    paths round-trip through the posix=False quote-strip; posix uses shlex."""
    if sys.platform == "win32":
        import subprocess
        return subprocess.list2cmdline(parts)
    return shlex.join(parts)


def stdio_mcp_enabled() -> bool:
    """stdio MCP servers spawn local processes as the backend user (bypassing the
    sandbox), so allowed only when the host is the user's own machine. On startup
    a loopback bind defaults UNSLOTH_STUDIO_ALLOW_STDIO_MCP=1 (see
    utils.host_policy.apply_stdio_mcp_loopback_default, called from run.py); the
    Tauri app does the same. Off for Colab and any network (0.0.0.0) bind unless
    an operator sets the var out-of-band; set it to 0 to force-disable.

    When stdio is on only because of that loopback auto-default, an explicit
    `unsloth studio run --disable-tools` turns it back off (a local stdio command
    is server-side code execution). An explicit operator opt-in via the env var
    still wins -- including the documented `=1` network opt-in, where the process
    tool policy is False merely by the external-host default, not by choice."""
    if os.environ.get("UNSLOTH_STUDIO_ALLOW_STDIO_MCP") != "1":
        return False
    from state.tool_policy import get_tool_policy
    from utils.host_policy import loopback_default_active

    if loopback_default_active() and get_tool_policy() is False:
        return False
    return True


# Probe timeouts for discovering a server's tool list. OAuth needs minutes for
# first-connect/expired-token browser sign-in; stdio allows for first-run
# package download (e.g. `npx -y ...`); HTTP fails fast.
_HTTP_PROBE_TIMEOUT = 8.0
_OAUTH_PROBE_TIMEOUT = 305.0
_STDIO_PROBE_TIMEOUT = 60.0


def probe_timeout(address: str, use_oauth: bool) -> float:
    if use_oauth:
        return _OAUTH_PROBE_TIMEOUT
    return _STDIO_PROBE_TIMEOUT if is_stdio(address) else _HTTP_PROBE_TIMEOUT


def parse_server_headers(server: dict) -> Optional[dict]:
    """Parsed headers_json. For stdio servers this dict is the process env
    instead of HTTP headers (see _client)."""
    raw = server.get("headers_json")
    if not raw:
        return None
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _oauth_store():
    global _oauth_token_store
    if _oauth_token_store is None:
        from key_value.aio._utils.sanitization import AlwaysHashStrategy
        from key_value.aio.stores.filetree import FileTreeStore
        from utils.paths.storage_roots import ensure_dir, studio_root

        # Hash keys/collections — fastmcp uses raw URLs as keys, and FileTreeStore
        # would treat the "://" as nested directories.
        _oauth_token_store = FileTreeStore(
            data_directory = ensure_dir(studio_root() / "mcp-oauth-tokens"),
            key_sanitization_strategy = AlwaysHashStrategy(),
            collection_sanitization_strategy = AlwaysHashStrategy(),
        )
    return _oauth_token_store


async def clear_oauth_tokens_async(url: str) -> None:
    """Drop any persisted OAuth tokens for ``url``. fastmcp keys tokens by MCP
    URL, so on server delete / URL change / OAuth disable we must clear them, else
    re-registering the same URL reuses the old account's token. Best-effort: store
    / OAuth failures must not 500 the delete / update route."""
    try:
        from fastmcp.client.auth import OAuth
        auth = OAuth(mcp_url = url, token_storage = _oauth_store())
        await auth.token_storage_adapter.clear()
    except Exception as exc:  # noqa: BLE001
        # Cleanup is best-effort; the row delete still wins.
        logger.warning("Failed to clear OAuth tokens for %s: %s", url, exc)


def _client(
    url: str,
    headers: Optional[dict],
    use_oauth: bool = False,
):
    from fastmcp import Client

    if is_stdio(url):
        # Belt-and-suspenders: never spawn unless stdio is enabled on this host.
        if not stdio_mcp_enabled():
            raise PermissionError("stdio MCP servers are disabled on this host")
        from fastmcp.client.transports import StdioTransport

        parts = parse_stdio_command(url)
        if not parts:
            raise ValueError(f"Empty stdio command: {url!r}")
        # env vars ride the headers field (merged over the SDK default env).
        # keep_alive=False tears the subprocess down so a one-shot call leaves no orphan.
        return Client(
            StdioTransport(
                command = parts[0],
                args = parts[1:],
                env = headers or None,
                keep_alive = False,
            )
        )

    from fastmcp.client.transports import SSETransport, StreamableHttpTransport
    from fastmcp.mcp_config import infer_transport_type_from_url

    auth = None
    if use_oauth:
        from fastmcp.client.auth import OAuth
        auth = OAuth(mcp_url = url, token_storage = _oauth_store())

    transport_cls = (
        SSETransport if infer_transport_type_from_url(url) == "sse" else StreamableHttpTransport
    )
    return Client(transport_cls(url = url, headers = headers or None, auth = auth))


# Persistent stdio sessions: a stdio MCP server owns live state (a browser, a
# DB handle), so keep one connected client per (command, env, chat session) on
# a dedicated event-loop thread instead of respawning per call.

_STDIO_SESSION_IDLE_TTL = 300.0
_STDIO_SESSION_REAP_INTERVAL = 30.0
_STDIO_CONNECT_TIMEOUT = 60.0  # allows first-run `npx -y ...` package download
_STDIO_CLOSE_TIMEOUT = 10.0
_STDIO_WEDGE_MARGIN = 15.0


class _SessionWedged(Exception):
    pass


class _SessionClosed(Exception):
    """The session was closed (server update/delete/shutdown) mid-call."""


def _abort_future(future) -> None:
    # Let the cancelled coroutine unwind before its loop is stopped.
    future.cancel()
    try:
        future.result(1.0)
    except BaseException:  # noqa: BLE001
        pass


class _StdioSession:
    def __init__(self, url: str, headers: Optional[dict]):
        self.url = url
        self.headers = headers
        self.client = None
        self.closed = threading.Event()
        self.defunct = False  # discarded; close once in_flight drains (see _retire)
        self._close_lock = threading.Lock()
        self.last_used = time.monotonic()
        self.in_flight = 0  # guarded by _stdio_sessions_lock
        self.loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target = self._run_loop, name = "mcp-stdio-session", daemon = True
        )
        self._thread.start()

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self.loop)
        try:
            self.loop.run_forever()
        finally:
            self.loop.close()

    def connect(self, timeout: Optional[float], cancel_event) -> None:
        async def _open():
            client = _client(self.url, self.headers)
            await client.__aenter__()
            # Publish on the loop thread with no await in between: if an abort
            # races a just-completed connect, close() still sees the client and
            # __aexit__s it instead of orphaning the subprocess.
            self.client = client
            return client

        future = asyncio.run_coroutine_threadsafe(_open(), self.loop)
        window = _STDIO_CONNECT_TIMEOUT if timeout is None else min(timeout, _STDIO_CONNECT_TIMEOUT)
        deadline = time.monotonic() + window
        while True:
            if cancel_event is not None and cancel_event.is_set():
                _abort_future(future)
                raise _MCPCancelled
            try:
                future.result(0.05)
                return
            except (concurrent.futures.TimeoutError, asyncio.TimeoutError):
                if time.monotonic() >= deadline:
                    _abort_future(future)
                    raise asyncio.TimeoutError

    def is_connected(self) -> bool:
        client = self.client
        if client is None:
            return False
        probe = getattr(client, "is_connected", None)
        try:
            return bool(probe()) if callable(probe) else True
        except Exception:
            return False

    def run(self, coro, timeout: Optional[float]):
        self.last_used = time.monotonic()
        future = asyncio.run_coroutine_threadsafe(coro, self.loop)
        # The coroutine enforces the tool timeout; the margin only catches a
        # wedged loop. No deadline at all when the caller set none -- but poll
        # so a session closed under us (server update/delete) can't hang the
        # request thread forever on a stopped loop.
        deadline = None if timeout is None else time.monotonic() + timeout + _STDIO_WEDGE_MARGIN
        try:
            while True:
                try:
                    return future.result(0.25)
                except concurrent.futures.CancelledError:
                    # Only close() cancels in-flight tasks (in _shutdown).
                    raise _SessionClosed
                except (concurrent.futures.TimeoutError, asyncio.TimeoutError):
                    if future.done():
                        raise  # the call's own timeout; the session stays usable
                    if self.closed.is_set():
                        future.cancel()
                        raise _SessionClosed
                    if deadline is not None and time.monotonic() >= deadline:
                        future.cancel()
                        raise _SessionWedged
        finally:
            self.last_used = time.monotonic()

    def close(self) -> None:
        # Idempotent: a discard racing close_stdio_sessions() may close twice.
        # Setting `closed` first also unblocks run() waiters (they poll it).
        with self._close_lock:
            if self.closed.is_set():
                return
            self.closed.set()
        loop = getattr(self, "loop", None)
        loop_alive = loop is not None and not loop.is_closed()
        if loop_alive:

            async def _shutdown() -> None:
                # Runs on the loop thread, so it serializes with an aborted
                # connect() that finished anyway and just published its client.
                client, self.client = self.client, None
                if client is not None:
                    await client.__aexit__(None, None, None)
                # Cancel in-flight calls so they unwind before loop.stop
                # (their run() waiters have already been released via `closed`).
                for task in asyncio.all_tasks():
                    if task is not asyncio.current_task():
                        task.cancel()

            try:
                asyncio.run_coroutine_threadsafe(_shutdown(), loop).result(_STDIO_CLOSE_TIMEOUT)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "MCP stdio session close failed for %r: %s", getattr(self, "url", ""), exc
                )
            try:
                loop.call_soon_threadsafe(loop.stop)
            except RuntimeError:
                pass
        else:
            self.client = None
        thread = getattr(self, "_thread", None)
        if thread is not None:
            thread.join(timeout = 5.0)


_stdio_sessions: dict[tuple, _StdioSession] = {}


# Per-key locks so a slow connect/close never blocks unrelated servers; the
# global lock only guards the dicts.
class _StdioKeyLock:
    """A per-key lock that can be removed once nobody references it."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.users = 0  # guarded by _stdio_sessions_lock


_stdio_key_locks: dict[tuple, _StdioKeyLock] = {}
_stdio_sessions_lock = threading.Lock()
_stdio_reaper_started = False
# close_stdio_sessions() can only close sessions already published in
# _stdio_sessions; one still inside connect() would be missed and cached
# stale. Bump a generation on every close so that connect discards its
# session instead of publishing it. Guarded by _stdio_sessions_lock.
_stdio_close_all_gen = 0
_stdio_url_close_gen: dict[str, int] = {}
_stdio_cfg_close_gen: dict[tuple, int] = {}

# close_stdio_sessions(url): match any env for that command.
_ANY_HEADERS = object()


def _headers_key(headers: Optional[dict]) -> tuple:
    return tuple(sorted((headers or {}).items()))


def _cfg_close_key(url: str, headers: Optional[dict]) -> tuple:
    # Env values can hold secrets and this map is never pruned; key by digest
    # so closed/edited configs don't retain them in memory forever.
    digest = hashlib.sha256(repr(_headers_key(headers)).encode()).hexdigest()
    return (url, digest)


def _stdio_close_generation(url: str, headers: Optional[dict]) -> tuple[int, int, int]:
    return (
        _stdio_close_all_gen,
        _stdio_url_close_gen.get(url, 0),
        _stdio_cfg_close_gen.get(_cfg_close_key(url, headers), 0),
    )


def _session_key(url: str, headers: Optional[dict], scope: Optional[str]) -> tuple:
    return (url, _headers_key(headers), scope or "")


def _checkout_stdio_session(key: tuple) -> Optional[_StdioSession]:
    session = _stdio_sessions.get(key)
    if session is not None and session.is_connected():
        session.last_used = time.monotonic()
        session.in_flight += 1
        return session
    return None


def _borrow_stdio_key_lock(key: tuple) -> _StdioKeyLock:
    """Return a stable per-key lock while a caller waits for/connects it."""
    key_lock = _stdio_key_locks.setdefault(key, _StdioKeyLock())
    key_lock.users += 1
    return key_lock


def _discard_stdio_key_lock(key: tuple) -> None:
    key_lock = _stdio_key_locks.get(key)
    if key_lock is not None and key_lock.users == 0 and key not in _stdio_sessions:
        _stdio_key_locks.pop(key, None)


def _return_stdio_key_lock(key: tuple, key_lock: _StdioKeyLock) -> None:
    with _stdio_sessions_lock:
        key_lock.users -= 1
        _discard_stdio_key_lock(key)


def _get_stdio_session(
    url: str, headers: Optional[dict], scope: Optional[str], timeout, cancel_event, config_check
) -> _StdioSession:
    global _stdio_reaper_started
    key = _session_key(url, headers, scope)
    with _stdio_sessions_lock:
        session = _checkout_stdio_session(key)
        if session is not None:
            return session
        key_lock = _borrow_stdio_key_lock(key)
    try:
        with key_lock.lock:
            stale = None
            with _stdio_sessions_lock:
                session = _checkout_stdio_session(key)
                if session is not None:
                    return session
                if key in _stdio_sessions:
                    stale = _stdio_sessions.pop(key)
                generation = _stdio_close_generation(url, headers)
            if stale is not None:
                _retire_stdio_session(stale)
            session = _StdioSession(url, headers)
            try:
                session.connect(timeout, cancel_event)
            except Exception:
                session.close()
                raise
            # A caller can read the server row, then lose to an update/delete
            # whose close ran before our generation snapshot. Re-verify the row
            # after connect; the generation check below covers a close landing
            # between this check and publish.
            if config_check is not None:
                try:
                    current = bool(config_check())
                except Exception:  # noqa: BLE001
                    current = False
                if not current:
                    session.close()
                    raise RuntimeError("MCP server was updated or removed while connecting")
            with _stdio_sessions_lock:
                closed_while_connecting = _stdio_close_generation(url, headers) != generation
                if not closed_while_connecting:
                    session.in_flight = 1
                    _stdio_sessions[key] = session
                    if not _stdio_reaper_started:
                        _stdio_reaper_started = True
                        threading.Thread(
                            target = _stdio_session_reaper, name = "mcp-stdio-reaper", daemon = True
                        ).start()
                        atexit.register(close_stdio_sessions)
            if closed_while_connecting:
                session.close()
                raise RuntimeError("MCP server was updated or removed while connecting")
            return session
    finally:
        _return_stdio_key_lock(key, key_lock)


def _release_stdio_session(session: _StdioSession) -> None:
    with _stdio_sessions_lock:
        session.in_flight = max(0, session.in_flight - 1)
        session.last_used = time.monotonic()
        close_now = session.defunct and session.in_flight == 0
    if close_now:
        session.close()


def _retire_stdio_session(session: _StdioSession) -> None:
    """Close a discarded session, but only once no other borrower is mid-call
    on it -- overlapping same-scope calls share one client, and one call's
    timeout must not kill another's in-flight request. The last borrower's
    _release_stdio_session() performs the deferred close."""
    with _stdio_sessions_lock:
        session.defunct = True
        busy = session.in_flight > 0
    if not busy:
        session.close()


def _drop_stdio_session(key: tuple, session: _StdioSession) -> None:
    with _stdio_sessions_lock:
        if _stdio_sessions.get(key) is session:
            _stdio_sessions.pop(key)
        _discard_stdio_key_lock(key)
    _retire_stdio_session(session)


def close_stdio_sessions(url: Optional[str] = None, headers = _ANY_HEADERS) -> None:
    """Close persistent stdio sessions: all of them (``url`` None), every env
    for one command (``headers`` omitted), or one server config (url + headers).
    Two server rows can share a command with different envs; editing one must
    not kill the other's live state, so the routes pass the edited row's env."""
    global _stdio_close_all_gen
    hk = None if headers is _ANY_HEADERS else _headers_key(headers)
    with _stdio_sessions_lock:
        if url is None:
            _stdio_close_all_gen += 1
        elif hk is None:
            _stdio_url_close_gen[url] = _stdio_url_close_gen.get(url, 0) + 1
        else:
            cfg = _cfg_close_key(url, headers)
            _stdio_cfg_close_gen[cfg] = _stdio_cfg_close_gen.get(cfg, 0) + 1
        keys = [
            k
            for k in _stdio_sessions
            if (url is None or k[0] == url) and (hk is None or k[1] == hk)
        ]
        sessions = [_stdio_sessions.pop(k) for k in keys]
        for key in keys:
            _discard_stdio_key_lock(key)
    for session in sessions:
        session.close()


def _reap_idle_stdio_sessions(now: Optional[float] = None) -> None:
    now = time.monotonic() if now is None else now
    with _stdio_sessions_lock:
        expired = [
            key
            for key, session in _stdio_sessions.items()
            if session.in_flight == 0 and now - session.last_used >= _STDIO_SESSION_IDLE_TTL
        ]
        sessions = [_stdio_sessions.pop(key) for key in expired]
        for key in expired:
            _discard_stdio_key_lock(key)
    for session in sessions:
        logger.info("Closing idle stdio MCP session: %s", session.url)
        session.close()


def _stdio_session_reaper() -> None:
    while True:
        time.sleep(_STDIO_SESSION_REAP_INTERVAL)
        try:
            _reap_idle_stdio_sessions()
        except Exception as exc:  # noqa: BLE001
            logger.debug("stdio session reaper iteration failed: %s", exc)


async def list_tools_async(
    url: str,
    headers: Optional[dict] = None,
    timeout: float = 5.0,
    use_oauth: bool = False,
) -> list[dict]:
    async def _fetch() -> list[dict]:
        async with _client(url, headers, use_oauth) as client:
            tools = await client.list_tools()
        return [t.model_dump(exclude_none = True) for t in tools]

    return await asyncio.wait_for(_fetch(), timeout = timeout)


# Discovered-tool cache, keyed by MCP server id. get_enabled_mcp_tools()
# probes a server only on a cache miss, keeping MCP discovery off the chat
# send's critical path -- tool schemas are stable within a session. The
# /refresh route warms it; a URL/header/OAuth change or a delete evicts it.
# Successful probes are cached indefinitely.
_tool_cache: dict[str, list[dict]] = {}

# server_id -> monotonic time before which a failed server must not be
# re-probed (see record_probe_failure). Cleared on a successful probe or
# eviction.
_probe_cooloff_until: dict[str, float] = {}

# MCP server fields whose change invalidates a server's discovered tools: the
# endpoint/auth used to probe it (url, headers, oauth) or whether it's used at
# all (is_enabled). A rename does not. The update route's eviction and
# get_enabled_mcp_tools' mid-probe guard both key off this so they can't drift.
TOOL_CACHE_INVALIDATING_FIELDS = frozenset({"url", "headers_json", "use_oauth", "is_enabled"})


def get_cached_tools(server_id: str) -> Optional[list[dict]]:
    return _tool_cache.get(server_id)


def cache_tools(server_id: str, tools: list[dict]) -> None:
    _tool_cache[server_id] = tools
    _probe_cooloff_until.pop(server_id, None)


def record_probe_failure(server_id: str, use_oauth: bool = False) -> None:
    cooloff = OAUTH_FAILED_PROBE_COOLOFF_SECONDS if use_oauth else FAILED_PROBE_COOLOFF_SECONDS
    _probe_cooloff_until[server_id] = time.monotonic() + cooloff


def in_failure_cooloff(server_id: str) -> bool:
    return _probe_cooloff_until.get(server_id, 0.0) > time.monotonic()


def invalidate_tool_cache(server_id: Optional[str] = None) -> None:
    """Evict one server's cached tools, or every entry when server_id is None."""
    if server_id is None:
        _tool_cache.clear()
        _probe_cooloff_until.clear()
    else:
        _tool_cache.pop(server_id, None)
        _probe_cooloff_until.pop(server_id, None)


def _flatten_result(result: Any) -> str:
    parts = []
    for block in getattr(result, "content", None) or []:
        text = getattr(block, "text", None)
        if text:
            parts.append(str(text))
    body = "\n".join(parts)
    if not body:
        structured = getattr(result, "structured_content", None)
        body = str(structured) if structured is not None else ""

    if getattr(result, "is_error", False):
        # "Error: " prefix triggers tool_call_parser's TOOL_ERROR_PREFIXES nudge.
        return f"Error: {body}" if body else "Error: tool returned no content"
    return body


async def _race_tool_call(call_coro, timeout: Optional[float], cancel_event) -> Any:
    """Await ``call_coro`` under ``timeout``, polling ``cancel_event`` so a
    /cancel POST interrupts even mid-network-read."""

    async def _watch_cancel() -> None:
        while cancel_event is not None and not cancel_event.is_set():
            await asyncio.sleep(0.05)

    if cancel_event is not None and cancel_event.is_set():
        call_coro.close()
        raise _MCPCancelled
    call_task = asyncio.create_task(call_coro)
    if cancel_event is None:
        return await asyncio.wait_for(call_task, timeout = timeout)
    watch_task = asyncio.create_task(_watch_cancel())
    try:
        done, pending = await asyncio.wait(
            {call_task, watch_task},
            timeout = timeout,
            return_when = asyncio.FIRST_COMPLETED,
        )
    finally:
        for t in (call_task, watch_task):
            if not t.done():
                t.cancel()
    if not done:
        raise asyncio.TimeoutError
    if call_task in done:
        return call_task.result()
    raise _MCPCancelled


def _call_stdio_tool(
    url: str,
    headers: Optional[dict],
    name: str,
    args: dict,
    timeout,
    cancel_event,
    scope: Optional[str],
    config_check,
) -> Any:
    if cancel_event is not None and cancel_event.is_set():
        raise _MCPCancelled
    # Callers without a Studio session id must retain the former one-shot
    # behavior: no browser/cookie/tool state can leak into another request.
    # Use an ephemeral key (and close it below) rather than the shared empty
    # scope that the persistent-session cache used previously.
    ephemeral = not scope
    if ephemeral:
        scope = f"request-{uuid.uuid4().hex}"
    key = _session_key(url, headers, scope)
    for fresh in (False, True):
        session = _get_stdio_session(url, headers, scope, timeout, cancel_event, config_check)
        discard_session = ephemeral
        try:
            coro = _race_tool_call(session.client.call_tool(name, args), timeout, cancel_event)
            return session.run(coro, timeout)
        except (_MCPCancelled, asyncio.TimeoutError):
            # _race_tool_call cancels the pending call but cancellation is
            # cooperative. Never return this client to the cache while the
            # timed-out/cancelled operation might still run on its transport.
            discard_session = True
            raise
        except _SessionWedged:
            discard_session = True
            raise asyncio.TimeoutError
        except _SessionClosed:
            # close_stdio_sessions() shut this session mid-call (server
            # update/delete/shutdown); don't retry on the stale config.
            discard_session = True
            raise RuntimeError("MCP server was updated or removed during the call")
        except Exception:
            if session.closed.is_set():
                # An intentional close (server update/delete) can surface as a
                # plain transport error or AttributeError instead of
                # _SessionClosed; don't mistake it for a crash and retry.
                discard_session = True
                raise RuntimeError("MCP server was updated or removed during the call")
            # A dead subprocess surfaces as a transport error; retry once on a
            # fresh session. Tool-level failures leave the session connected.
            if fresh or session.is_connected():
                raise
            _drop_stdio_session(key, session)
        finally:
            _release_stdio_session(session)
            if discard_session:
                _drop_stdio_session(key, session)
    raise RuntimeError("unreachable")


def call_tool_sync(
    url: str,
    headers: Optional[dict],
    name: str,
    args: dict,
    timeout: Optional[float] = 300.0,
    use_oauth: bool = False,
    cancel_event = None,
    scope: Optional[str] = None,
    config_check = None,
) -> str:
    """Synchronously call an MCP tool. stdio servers reuse a persistent session
    keyed by (command, env, scope) only when ``scope`` is provided; calls
    without one stay one-shot. HTTP servers always stay one-shot.
    ``cancel_event`` (threading.Event) cancels the in-flight call when set.
    ``config_check`` (callable -> bool) re-validates the caller's server config
    before a fresh stdio session is cached; False fails the call."""

    async def _one_shot() -> Any:
        async with _client(url, headers, use_oauth) as client:
            return await client.call_tool(name, args)

    try:
        if is_stdio(url):
            result = _call_stdio_tool(
                url, headers, name, args, timeout, cancel_event, scope, config_check
            )
        else:
            result = asyncio.run(_race_tool_call(_one_shot(), timeout, cancel_event))
    except _MCPCancelled:
        return f"Error: MCP tool '{name}' cancelled"
    except asyncio.TimeoutError:
        suffix = f" after {timeout:g}s" if timeout is not None else ""
        return f"Error: MCP tool '{name}' timed out{suffix}"
    except Exception as exc:
        logger.exception("MCP call_tool failed for %s: %s", name, exc)
        return f"Error: MCP tool '{name}' failed: {exc}"

    return _flatten_result(result)


class _MCPCancelled(Exception):
    """Internal sentinel raised when cancel_event fires before the tool returns."""
