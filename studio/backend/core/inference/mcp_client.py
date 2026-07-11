# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import asyncio
import atexit
import concurrent.futures
import json
import os
import shlex
import sys
import threading
import time
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


# ── Persistent stdio sessions ────────────────────────────────────────────────
# A stdio MCP server owns live state (a playwright browser, a DB handle); a
# one-shot client per call resets that state, so e.g. browser_navigate +
# browser_take_screenshot screenshotted a fresh about:blank tab. Keep one
# connected client per (command, env) on a dedicated event-loop thread and
# reap it after idling.

_STDIO_SESSION_IDLE_TTL = 300.0
_STDIO_SESSION_REAP_INTERVAL = 30.0
# Generous: the first call may download the package (`npx -y ...`).
_STDIO_CONNECT_TIMEOUT = 60.0
_STDIO_CLOSE_TIMEOUT = 10.0


class _SessionWedged(Exception):
    pass


class _StdioSession:
    def __init__(self, url: str, headers: Optional[dict]):
        self.url = url
        self.headers = headers
        self.client = None
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

    def connect(self) -> None:
        async def _open():
            client = _client(self.url, self.headers)
            await client.__aenter__()
            return client

        future = asyncio.run_coroutine_threadsafe(_open(), self.loop)
        self.client = future.result(_STDIO_CONNECT_TIMEOUT)

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
        try:
            # The coroutine enforces the tool timeout itself; the margin only
            # guards a wedged loop.
            return future.result((timeout or 300.0) + 15.0)
        except (concurrent.futures.TimeoutError, asyncio.TimeoutError):
            if future.done():
                raise  # the call's own timeout; the session stays usable
            future.cancel()
            raise _SessionWedged
        finally:
            self.last_used = time.monotonic()

    def close(self) -> None:
        client, self.client = self.client, None
        if client is not None:
            try:
                future = asyncio.run_coroutine_threadsafe(
                    client.__aexit__(None, None, None), self.loop
                )
                future.result(_STDIO_CLOSE_TIMEOUT)
            except Exception as exc:  # noqa: BLE001
                logger.warning("MCP stdio session close failed for %r: %s", self.url, exc)
        self.loop.call_soon_threadsafe(self.loop.stop)
        self._thread.join(timeout = 5.0)


_stdio_sessions: dict[tuple, _StdioSession] = {}
_stdio_sessions_lock = threading.Lock()
_stdio_reaper_started = False


def _session_key(url: str, headers: Optional[dict]) -> tuple:
    return (url, tuple(sorted((headers or {}).items())))


def _get_stdio_session(url: str, headers: Optional[dict]) -> _StdioSession:
    global _stdio_reaper_started
    key = _session_key(url, headers)
    with _stdio_sessions_lock:
        stale = None
        session = _stdio_sessions.get(key)
        if session is not None and session.is_connected():
            session.last_used = time.monotonic()
            session.in_flight += 1
            return session
        if session is not None:
            stale = _stdio_sessions.pop(key)
        if stale is not None:
            stale.close()
        session = _StdioSession(url, headers)
        try:
            session.connect()
        except Exception:
            session.close()
            raise
        session.in_flight = 1
        _stdio_sessions[key] = session
        if not _stdio_reaper_started:
            _stdio_reaper_started = True
            threading.Thread(
                target = _stdio_session_reaper, name = "mcp-stdio-reaper", daemon = True
            ).start()
            atexit.register(close_stdio_sessions)
        return session


def _release_stdio_session(session: _StdioSession) -> None:
    with _stdio_sessions_lock:
        session.in_flight = max(0, session.in_flight - 1)
        session.last_used = time.monotonic()


def _drop_stdio_session(url: str, headers: Optional[dict], session: _StdioSession) -> None:
    key = _session_key(url, headers)
    with _stdio_sessions_lock:
        if _stdio_sessions.get(key) is session:
            _stdio_sessions.pop(key)
    session.close()


def close_stdio_sessions(url: Optional[str] = None) -> None:
    """Close every persistent stdio session, or only those for ``url`` (used
    when a server is deleted or its endpoint/env/enabled state changes)."""
    with _stdio_sessions_lock:
        keys = [k for k in _stdio_sessions if url is None or k[0] == url]
        sessions = [_stdio_sessions.pop(k) for k in keys]
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
    """Await ``call_coro`` under ``timeout``, polling ``cancel_event`` (50 ms
    cadence, matching routes/inference.py's cancel watcher) so a /cancel POST
    interrupts even mid-network-read."""

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
    url: str, headers: Optional[dict], name: str, args: dict, timeout, cancel_event
) -> Any:
    if cancel_event is not None and cancel_event.is_set():
        raise _MCPCancelled
    for fresh in (False, True):
        session = _get_stdio_session(url, headers)
        try:
            coro = _race_tool_call(session.client.call_tool(name, args), timeout, cancel_event)
            return session.run(coro, timeout)
        except (_MCPCancelled, asyncio.TimeoutError):
            raise
        except _SessionWedged:
            _drop_stdio_session(url, headers, session)
            raise asyncio.TimeoutError
        except Exception:
            # A dead subprocess surfaces as a transport error; retry once on a
            # fresh session. Tool-level failures leave the session connected.
            if fresh or session.is_connected():
                raise
            _drop_stdio_session(url, headers, session)
        finally:
            _release_stdio_session(session)
    raise RuntimeError("unreachable")


def call_tool_sync(
    url: str,
    headers: Optional[dict],
    name: str,
    args: dict,
    timeout: Optional[float] = 300.0,
    use_oauth: bool = False,
    cancel_event = None,
) -> str:
    """Synchronously call an MCP tool.

    stdio servers go through a persistent session so server-side state (e.g. a
    browser) survives across calls; HTTP servers stay one-shot per call.

    ``cancel_event``: optional ``threading.Event``. When set, the in-flight call
    is cancelled and a cancellation Error returned.
    """

    async def _one_shot() -> Any:
        async with _client(url, headers, use_oauth) as client:
            return await client.call_tool(name, args)

    try:
        if is_stdio(url):
            result = _call_stdio_tool(url, headers, name, args, timeout, cancel_event)
        else:
            result = asyncio.run(_race_tool_call(_one_shot(), timeout, cancel_event))
    except _MCPCancelled:
        return f"Error: MCP tool '{name}' cancelled"
    except asyncio.TimeoutError:
        return f"Error: MCP tool '{name}' timed out after {timeout:g}s"
    except Exception as exc:
        logger.exception("MCP call_tool failed for %s: %s", name, exc)
        return f"Error: MCP tool '{name}' failed: {exc}"

    return _flatten_result(result)


class _MCPCancelled(Exception):
    """Internal sentinel raised when cancel_event fires before the tool returns."""
