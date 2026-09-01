# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import asyncio
import atexit
import concurrent.futures
import hashlib
import json
import mimetypes
import os
import re
import shlex
import shutil
import subprocess
import sys
import threading
import time
import uuid
from contextlib import contextmanager
from functools import wraps
from typing import Any, Optional
from urllib.parse import urlsplit, urlunsplit
from weakref import WeakKeyDictionary

from loggers import get_logger
from utils.workspace_context import current_workspace_subject

logger = get_logger(__name__)

MCP_TOOL_PREFIX = "mcp__"
_WINDOWS_BATCH_ALWAYS_UNSAFE_ARGUMENT_CHARS = frozenset('%!"\r\n')
_WINDOWS_BATCH_UNQUOTED_UNSAFE_ARGUMENT_CHARS = frozenset("&|<>^()")

# A failed probe isn't cached (a recovered server must come back), but it's
# recorded so a down server isn't re-probed -- and the chat send re-hung for
# the full timeout -- on every message. Cool off for this long after a failure;
# much longer for OAuth, whose probe can hang up to _OAUTH_PROBE_TIMEOUT,
# so that hang doesn't recur every minute.
FAILED_PROBE_COOLOFF_SECONDS = 60.0
OAUTH_FAILED_PROBE_COOLOFF_SECONDS = 300.0

_oauth_token_store = None
_oauth_token_store_lock = threading.Lock()


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
        # subprocess.list2cmdline() implements the MS C runtime grammar: only
        # space and tab delimit arguments. Other Unicode/control whitespace is
        # ordinary argument data and must not be split here.
        if ch in (" ", "\t") and not in_quotes:
            if backslashes:
                current.extend("\\" * backslashes)
                arg_started = True
                backslashes = 0
            if arg_started or current:
                parts.append("".join(current))
                current = []
                arg_started = False
            i += 1
            while i < len(address) and address[i] in (" ", "\t"):
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
        return subprocess.list2cmdline(parts)
    return shlex.join(parts)


def _windows_batch_argument_is_unsafe(argument: str) -> bool:
    if _WINDOWS_BATCH_ALWAYS_UNSAFE_ARGUMENT_CHARS.intersection(argument):
        return True
    serialized = subprocess.list2cmdline([argument])
    is_quoted = serialized.startswith('"') and serialized.endswith('"')
    return not is_quoted and bool(
        _WINDOWS_BATCH_UNQUOTED_UNSAFE_ARGUMENT_CHARS.intersection(argument)
    )


def _session_log_id(url: str) -> str:
    """A non-secret label for logs. stdio commands can embed credentials in argv
    (e.g. ``npx server --token sk-...``) and HTTP URLs in their query string, so
    never log the raw address; use the executable basename (or the host) plus a
    short digest of the full address instead."""
    digest = hashlib.sha256(url.encode()).hexdigest()[:12]
    if not is_stdio(url):
        try:
            label = urlsplit(url).hostname or "<url>"
        except Exception:  # noqa: BLE001
            label = "<invalid>"
        return f"{label}#{digest}"
    try:
        parts = parse_stdio_command(url)
        exe = os.path.basename(parts[0]) if parts else "<empty>"
    except Exception:  # noqa: BLE001
        exe = "<invalid>"
    return f"{exe}#{digest}"


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
    from utils.host_policy import loopback_default_active, remote_connector_active

    if loopback_default_active() and (remote_connector_active() or get_tool_policy() is False):
        return False
    return True


def stdio_mcp_disabled_reason() -> str:
    """User-facing reason local commands are off, mirroring stdio_mcp_enabled().

    Telling a user whose gate is suspended by an active tunnel to set
    UNSLOTH_STUDIO_ALLOW_STDIO_MCP=1 would re-enable local command execution on
    a published API, so the suspended cases must name their actual cause."""
    from state.tool_policy import get_tool_policy
    from utils.host_policy import loopback_default_active, remote_connector_active

    if os.environ.get("UNSLOTH_STUDIO_ALLOW_STDIO_MCP") == "1" and loopback_default_active():
        if remote_connector_active():
            return (
                "Local commands are disabled while Remote Access is on, because the "
                "server is reachable from outside this machine. Turn off Remote Access "
                "to use local MCP servers, or use an http:// or https:// URL instead."
            )
        if get_tool_policy() is False:
            return (
                "Local commands are disabled because tools are disabled for this "
                "server. Restart without --disable-tools, or use an http:// or "
                "https:// URL instead."
            )
    return (
        "Local commands aren't enabled on this server. To allow them, set "
        "UNSLOTH_STUDIO_ALLOW_STDIO_MCP=1 and restart Unsloth, or use an "
        "http:// or https:// URL instead."
    )


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
    from key_value.aio._utils.sanitization import AlwaysHashStrategy
    from key_value.aio.stores.filetree import FileTreeStore
    from utils.paths.storage_roots import ensure_dir, workspace_root

    directory = ensure_dir(workspace_root() / "mcp-oauth-tokens")
    key = os.path.normcase(os.path.realpath(str(directory)))
    with _oauth_token_store_lock:
        # Keep the historical single-cache variable so tests and hot reloads can
        # clear it, but key it by workspace. A caller retains the returned store,
        # so another account replacing this cache cannot redirect an in-flight OAuth flow.
        if _oauth_token_store is None or _oauth_token_store[0] != key:
            # Hash keys/collections — fastmcp uses raw URLs as keys, and FileTreeStore
            # would treat the "://" as nested directories.
            _oauth_token_store = (
                key,
                FileTreeStore(
                    data_directory = directory,
                    key_sanitization_strategy = AlwaysHashStrategy(),
                    collection_sanitization_strategy = AlwaysHashStrategy(),
                ),
            )
        return _oauth_token_store[1]


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
        logger.warning("Failed to clear OAuth tokens for %s: %s", url, exc)


_IS_WINDOWS = os.name == "nt"
_NODE_COMMANDS = frozenset({"node", "npm", "npx"})
_WINDOWS_LAUNCHER_SUFFIXES = (".cmd", ".exe", ".bat", ".ps1")


def _launcher_name(command: str) -> str:
    """argv[0] reduced to its bare launcher name, Windows suffix stripped."""
    name = os.path.basename(command).lower()
    for suffix in _WINDOWS_LAUNCHER_SUFFIXES:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def _is_node_command(command: str) -> bool:
    """Whether argv[0] is a Node launcher. Only those need the managed runtime, so a
    Python or other stdio server keeps the toolchain its own env pinned."""
    return _launcher_name(command) in _NODE_COMMANDS


def _command_selects_runtime(command: Optional[str]) -> bool:
    """A path to a ``node`` launcher picks the runtime explicitly and runs regardless of
    PATH, so handing its children a different Node would only split the two."""
    return (
        command is not None and bool(os.path.dirname(command)) and _launcher_name(command) == "node"
    )


def _runtime_requirements(command: Optional[str]) -> tuple[bool, bool]:
    """``(needs npm, needs npx)`` for argv[0]. Each launcher asks only for what it runs,
    so an unrelated missing launcher cannot shadow a good runtime: node needs neither, npm
    needs npm, and npx needs npx alone -- npx never shells out to an ``npm`` executable,
    its npx-cli.js delegates in-process to the npm library it ships with, so a PATH
    exposing node and npx without a separate npm runs it fine and must be left alone.
    A pathed npm/npx is already located and only needs a node for its shebang, so it
    does not require a second copy of itself on PATH either."""
    name = _launcher_name(command) if command is not None else None
    if name == "node":
        return False, False
    if command is not None and os.path.dirname(command):
        return False, False
    if name == "npm":
        return True, False
    if name == "npx":
        return False, True
    return True, True


def _path_key(env: dict) -> str:
    """The key holding PATH. Windows env names are case-insensitive, so a config may
    spell it ``Path``; on POSIX only the exact name counts."""
    if _IS_WINDOWS:
        for key in env:
            if key.upper() == "PATH":
                return key
    return "PATH"


def _stdio_env(headers: Optional[dict], command: Optional[str] = None) -> Optional[dict]:
    """Process env for a stdio server: its own vars, plus the managed Node bin dir
    on PATH so ``npx ...`` servers spawn on hosts with no usable system Node."""
    env = dict(headers or {})
    for key, value in env.items():
        if not isinstance(key, str) or not isinstance(value, str):
            raise ValueError("stdio environment names and values must be strings")
        if "\x00" in key or "\x00" in value:
            raise ValueError("stdio environment must not contain NUL characters")
        if "=" in key:
            raise ValueError("stdio environment variable names must not contain '='")
    key = _path_key(env)
    base = env.get(key)
    if isinstance(base, str) and not base:
        # An explicitly empty PATH is a deliberate sandbox: hand it over untouched.
        return env
    if command is not None and not _is_node_command(command):
        return env or None
    if _command_selects_runtime(command):
        return env or None
    if not isinstance(base, str):
        base = os.environ.get("PATH", "")
    try:
        from utils.node_runtime import path_with_managed_node
        require_npm, require_npx = _runtime_requirements(command)
        patched = path_with_managed_node(base, require_npm = require_npm, require_npx = require_npx)
    except (ImportError, OSError, ValueError):
        patched = base
    if patched and patched != env.get(key):
        env[key] = patched
    return env or None


def _stdio_argv(parts: list, env: Optional[dict]) -> list:
    """argv with argv[0] resolved against the child's PATH. Windows resolves the
    command against the parent environment before ``env`` applies, so a managed-only
    ``npx`` has to be handed over as a full path."""
    path_key = _path_key(env or {})
    explicit_path = env is not None and path_key in env
    path = (env or {}).get(path_key)
    if not isinstance(path, str):
        path = os.environ.get("PATH", "")
    try:
        resolved = shutil.which(parts[0], path = path)
    except OSError:
        resolved = None
    if _IS_WINDOWS and resolved is None and explicit_path and not os.path.dirname(parts[0]):
        raise ValueError(f"Cannot find {parts[0]!r} on the MCP server's configured PATH")
    executable = resolved or parts[0]
    if _IS_WINDOWS:
        suffix = os.path.splitext(executable)[1].lower()
        if suffix in {".cmd", ".bat"} and _launcher_name(executable) in {"npm", "npx"}:
            launcher_dir = os.path.dirname(executable)
            cli = os.path.join(
                launcher_dir,
                "node_modules",
                "npm",
                "bin",
                f"{_launcher_name(executable)}-cli.js",
            )
            sibling_node = os.path.join(launcher_dir, "node.exe")
            try:
                node = (
                    sibling_node
                    if os.path.isfile(sibling_node)
                    else shutil.which("node", path = path)
                )
                cli_exists = os.path.isfile(cli)
            except OSError:
                node = None
                cli_exists = False
            if node and cli_exists:
                # bypass cmd.exe so shell metacharacters remain literal argv.
                return [node, cli, *parts[1:]]
            raise ValueError(
                f"Cannot launch {executable!r} without its Node executable and npm CLI script"
            )
        if suffix in {".cmd", ".bat"} and any(
            _windows_batch_argument_is_unsafe(argument) for argument in parts[1:]
        ):
            raise ValueError(
                "Windows batch launchers cannot safely preserve these MCP command arguments; "
                "invoke the executable directly, or use node.exe with the JavaScript entry point"
            )
    return [executable, *parts[1:]]


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
        env = _stdio_env(headers, parts[0])
        argv = _stdio_argv(parts, env)
        return Client(
            StdioTransport(
                command = argv[0],
                args = argv[1:],
                env = env,
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


_SESSION_IDLE_TTL = 300.0
_SESSION_REAP_INTERVAL = 30.0
_STDIO_CONNECT_TIMEOUT = 60.0  # allows first-run `npx -y ...` package download
_SESSION_CLOSE_TIMEOUT = 10.0
_SESSION_WEDGE_MARGIN = 15.0
_SESSION_LIVENESS_TIMEOUT = 5.0
_CANCEL_UNWIND_TIMEOUT = 2.0
# An HTTP session idle this long is re-proved with tools/list before the next
# dispatch: the server may have expired it (MCP says a server MAY terminate a
# session at any time), and no HTTP transport exposes a liveness probe we can
# ask instead -- see _transport_dead.
_HTTP_IDLE_RECHECK = 30.0
_DEFAULT_MAX_SESSIONS = 32


def _max_sessions_from_env() -> int:
    # The cap covers stdio and HTTP sessions alike now, so the stdio-specific
    # name is wrong; keep honouring it so deployments that already set it do not
    # silently jump back to the default.
    raw = os.environ.get("UNSLOTH_STUDIO_MAX_MCP_SESSIONS")
    if raw is None:
        raw = os.environ.get("UNSLOTH_STUDIO_MAX_STDIO_MCP_SESSIONS")
    if raw is None:
        return _DEFAULT_MAX_SESSIONS
    try:
        return max(1, int(raw))
    except ValueError:
        return _DEFAULT_MAX_SESSIONS


_MAX_SESSIONS = _max_sessions_from_env()


def _connect_window(url: str, timeout: Optional[float]) -> Optional[float]:
    """How long connecting may take, out of the caller's remaining budget.

    stdio keeps the cold-start cap: a first run may download a package before the
    server says anything. HTTP has no such phase, and capping it would reject
    connections the caller explicitly allowed time for -- which is what the
    one-shot path it replaced always did."""
    if timeout is None or not is_stdio(url):
        return timeout
    return min(timeout, _STDIO_CONNECT_TIMEOUT)


class _ConnectTimeout(asyncio.TimeoutError):
    """Ran out of time before the transport was up. Carries the window that
    actually expired, which is not the caller's timeout when stdio's cold-start
    cap is the tighter bound."""

    def __init__(self, window: Optional[float]):
        super().__init__()
        self.window = window


def _is_tool_error(exc: BaseException) -> bool:
    """A tool-level failure (the tool ran and errored) leaves the transport alive,
    so the session is kept; fastmcp raises ToolError for these. Anything else from
    call_tool is transport-level. Version-safe (fastmcp 3.0.2 has no dead probe)."""
    try:
        from fastmcp.exceptions import ToolError
    except Exception:  # noqa: BLE001
        return False
    return isinstance(exc, ToolError)


def _is_protocol_error(exc: BaseException) -> bool:
    """A JSON-RPC error response, as opposed to a broken connection.

    A FastMCP server answers an unknown tool or bad arguments with a result
    carrying is_error, but the MCP spec also lets a server report those as a
    protocol error, and plenty of non-FastMCP servers do. fastmcp surfaces that
    as MCPError (McpError before the rename) carrying the ErrorData the server
    sent. Receiving it proves the connection is working, so retiring the session
    over it would throw away the chat's server-side state for a mistyped tool
    name. The caller still marks the session for a probe before reuse."""
    for module, name in (
        ("mcp.shared.exceptions", "MCPError"),
        ("mcp.shared.exceptions", "McpError"),
    ):
        try:
            cls = getattr(__import__(module, fromlist = [name]), name, None)
        except Exception:  # noqa: BLE001
            continue
        if cls is not None and isinstance(exc, cls):
            # Only when the server actually sent an error object; a synthetic
            # MCPError with nothing behind it stays transport-level.
            return getattr(exc, "error", None) is not None
    return False


def _transport_dead(session) -> bool:
    """Best-effort, version-adaptive liveness probe for a cached client.
    ``Client.is_connected()`` only checks a session object exists, not that the
    subprocess (or the server's HTTP session) is alive, so it is never used here.
    Returns True only when the transport is positively gone; unknown returns
    False (the call surfaces it).

    Only the stdio transport answers: ``_is_session_dead``/``_connect_task`` are
    ``StdioTransport`` internals, and neither ``StreamableHttpTransport`` nor
    ``SSETransport`` has ever carried them (checked on fastmcp 3.0.2 and 4.0.0).
    For HTTP this returns "unknown" every time, which is why an idle HTTP session
    is re-proved with tools/list instead -- see _needs_idle_recheck."""
    client = getattr(session, "client", None)
    if client is None:
        return True
    transport = getattr(client, "transport", None)
    probe = getattr(transport, "_is_session_dead", None)
    if callable(probe):
        try:
            if probe():
                return True
        except Exception:  # noqa: BLE001
            pass
    connect_task = getattr(transport, "_connect_task", None)
    if connect_task is not None:
        try:
            if connect_task.done():
                return True
        except Exception:  # noqa: BLE001
            pass
    return False


def _needs_idle_recheck(session, idle_for: float, remaining: Optional[float]) -> bool:
    """Whether an idle HTTP session must prove itself before the next dispatch.

    A server MAY drop an HTTP session whenever it likes, and the client only
    learns on the next request -- which would surface as a failed tool call the
    user has to retry by hand. stdio is exempt: _transport_dead answers there, and
    a live subprocess does not expire on its own.

    Skipped when the caller cannot afford it. The probe exists to save someone a
    failed call, so spending their whole budget on it (tool_call_timeout goes down
    to 1s) would cause the very failure it is meant to avoid; dispatching straight
    away is the better bet with that little time left."""
    if is_stdio(session.url):
        return False
    # Negative means this borrower connected the session itself, so the handshake
    # it just completed is proof enough.
    if idle_for < 0.0 or idle_for < _HTTP_IDLE_RECHECK:
        return False
    return remaining is None or remaining > _SESSION_LIVENESS_TIMEOUT * 2


def _session_responsive(
    session,
    budget: Optional[float] = None,
    cancel_event = None,
    timeout_is_fatal: bool = True,
) -> bool:
    """Whether a session left dirty by an abandoned call can be reused: the
    server must answer inside ``budget`` (the caller's remaining deadline).
    Proves the server is alive, not that the abandoned call finished -- MCP
    requests are concurrent. Probes with a raw single-page tools/list: ping
    answers "Method not found" on a modern-era connection, and list_tools()
    auto-paginates up to 250 pages.

    ``timeout_is_fatal`` separates the two callers. A dirty session is under
    suspicion, so silence within the window condemns it. An idle one is only
    being spot-checked: a slow answer says nothing about whether the transport
    is gone, and retiring it there would throw away the very state this cache
    exists to keep. Only a definite failure retires that one."""
    client = session.client
    if client is None:
        return False
    window = _SESSION_LIVENESS_TIMEOUT if budget is None else min(_SESSION_LIVENESS_TIMEOUT, budget)
    if window <= 0:
        # No budget left to ask in; that is not evidence either way.
        return not timeout_is_fatal
    probe = getattr(client, "list_tools_mcp", None) or client.list_tools
    try:
        # margin=0: a wedged loop must fail inside the window, not 15s past it.
        session.run(_race_tool_call(probe(), window, cancel_event), window, margin = 0.0)
    except _MCPCancelled:
        raise
    except (asyncio.TimeoutError, _SessionWedged):
        return not timeout_is_fatal
    except Exception as exc:  # noqa: BLE001
        # A JSON-RPC error (a rate limit, a permission rule on tools/list) is the
        # server answering on this very session, exactly as it is for call_tool.
        # Reconnecting would throw away the chat's state over a reply that proves
        # the transport works.
        if not _is_protocol_error(exc):
            return False
    session.dirty = False
    session.proved_at = time.monotonic()
    return True


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


class _McpSession:
    def __init__(
        self,
        url: str,
        headers: Optional[dict],
        use_oauth: bool = False,
    ):
        # A cached session is built by _client(url, headers) with no auth, so an
        # OAuth server must never reach here. call_tool_sync already routes it to
        # the one-shot path; this makes a future routing slip fail loudly rather
        # than quietly talk to an OAuth server unauthenticated.
        if use_oauth and not is_stdio(url):
            raise ValueError("OAuth MCP servers cannot use a shared session")
        self.url = url
        self.headers = headers
        self.client = None
        self.closed = threading.Event()
        self.defunct = False  # discarded; close once in_flight drains (see _retire)
        self.dirty = False  # a call was abandoned on it; ping before reuse
        self._close_lock = threading.Lock()
        self.call_lock = threading.Lock()
        # One stdio subprocess is one ordered byte stream, so overlapping calls
        # must not interleave on it. HTTP has no such constraint: every JSON-RPC
        # message is its own POST and the spec lets a client keep several streams
        # open at once, so serializing it would only undo the parallelism the
        # one-shot path had.
        self.serialize_calls = is_stdio(url)
        self.last_used = time.monotonic()
        # When the transport was last shown to be alive, as opposed to merely
        # borrowed. Checkout refreshes last_used immediately, so the idle gap the
        # recheck needs has to be measured from here or a second borrower arriving
        # during the first one's probe would see no gap at all.
        self.proved_at = self.last_used
        self.in_flight = 0  # guarded by _mcp_sessions_lock
        # On Windows a bare new_event_loop() can be a SelectorEventLoop (if any
        # component set that policy), which cannot spawn subprocesses natively;
        # force a ProactorEventLoop so the stdio transport always works.
        if sys.platform == "win32":
            self.loop = asyncio.ProactorEventLoop()
        else:
            self.loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target = self._run_loop, name = "mcp-session", daemon = True)
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
        # timeout=None means unlimited (no connect deadline); a finite caller
        # timeout bounds connect by _connect_window(), which caps stdio at its
        # cold-start limit and hands HTTP the whole remaining budget.
        window = _connect_window(self.url, timeout)
        deadline = None if window is None else time.monotonic() + window
        while True:
            if cancel_event is not None and cancel_event.is_set():
                _abort_future(future)
                raise _MCPCancelled
            try:
                future.result(0.05)
                return
            except (concurrent.futures.TimeoutError, asyncio.TimeoutError):
                if future.done():
                    raise  # the connect itself failed fast; don't wait out the window
                if deadline is not None and time.monotonic() >= deadline:
                    _abort_future(future)
                    raise _ConnectTimeout(window)

    def is_connected(self) -> bool:
        client = self.client
        if client is None:
            return False
        probe = getattr(client, "is_connected", None)
        try:
            return bool(probe()) if callable(probe) else True
        except Exception:
            return False

    def run(
        self,
        coro,
        timeout: Optional[float],
        margin: float = _SESSION_WEDGE_MARGIN,
    ):
        self.last_used = time.monotonic()
        future = asyncio.run_coroutine_threadsafe(coro, self.loop)
        # The coroutine enforces the tool timeout; the margin only catches a
        # wedged loop. No deadline at all when the caller set none -- but poll
        # so a session closed under us (server update/delete) can't hang the
        # request thread forever on a stopped loop. Callers whose whole budget is
        # the timeout (the liveness probe) pass margin=0.
        deadline = None if timeout is None else time.monotonic() + timeout + margin
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
                asyncio.run_coroutine_threadsafe(_shutdown(), loop).result(_SESSION_CLOSE_TIMEOUT)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "MCP session close failed for %s: %s",
                    _session_log_id(getattr(self, "url", "")),
                    exc,
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


_mcp_sessions: dict[tuple, _McpSession] = {}


# Per-key locks so a slow connect/close never blocks unrelated servers; the
# global lock only guards the dicts.
class _McpKeyLock:
    """A per-key lock that can be removed once nobody references it."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.users = 0  # guarded by _mcp_sessions_lock


_mcp_key_locks: dict[tuple, _McpKeyLock] = {}
_mcp_sessions_lock = threading.Lock()
_mcp_reaper_started = False
# Sessions discarded while somebody else was mid-call, closed by one worker off
# the request path. See _close_detached.
_mcp_cleanup_lock = threading.Lock()
_mcp_cleanup_queue: list = []
# Depth past which an evicting caller closes the overflow itself. See
# _close_detached.
_MAX_PENDING_CLOSES = 8
# How wide close_mcp_sessions fans out. See _close_all.
_MAX_CLOSE_THREADS = 16
_mcp_cleanup_worker: Optional[threading.Thread] = None
# close_mcp_sessions() can only close sessions already published in _mcp_sessions;
# one still inside connect() would be missed and then cached already
# stale. Bump a generation on every close so that connect discards its
# session instead of publishing it. Guarded by _mcp_sessions_lock.
_mcp_close_all_gen = 0
# "close everything in one workspace", kept apart from the shutdown counter above
# so one account's bulk close cannot reject another account's live connection.
_mcp_subject_close_gen: dict[str, int] = {}
_mcp_url_close_gen: dict[str, int] = {}
_mcp_cfg_close_gen: dict[tuple, int] = {}
_mcp_connects_in_flight = 0

_ANY_HEADERS = object()


def _headers_key(headers: Optional[dict]) -> tuple:
    return tuple(sorted((headers or {}).items()))


def _url_close_key(url: str, subject: Optional[str] = None) -> str:
    # Commands/URLs (token args, embedded credentials) and env values can hold
    # secrets and these maps are never pruned; key by digest so closed/edited
    # configs don't retain them in memory forever. Scoped by workspace so editing
    # one account's row does not invalidate another's identical command.
    return hashlib.sha256(repr((subject or current_workspace_subject(), url)).encode()).hexdigest()


def _cfg_close_key(
    url: str,
    headers: Optional[dict],
    subject: Optional[str] = None,
) -> str:
    return hashlib.sha256(
        repr((subject or current_workspace_subject(), url, _headers_key(headers))).encode()
    ).hexdigest()


def _mcp_close_generation(url: str, headers: Optional[dict]) -> tuple[int, int, int, int]:
    subject = current_workspace_subject()
    return (
        _mcp_close_all_gen,
        _mcp_subject_close_gen.get(subject, 0),
        _mcp_url_close_gen.get(_url_close_key(url, subject), 0),
        _mcp_cfg_close_gen.get(_cfg_close_key(url, headers, subject), 0),
    )


def _session_key(url: str, headers: Optional[dict], scope: Optional[str]) -> tuple:
    # Workspace last: scope carries client-chosen session/thread ids, so two
    # accounts can present the same one and would otherwise share a live stdio
    # child and whatever browser or REPL state it holds. Appended rather than
    # prepended because close_stdio_sessions matches on k[0]/k[1].
    return (url, _headers_key(headers), scope or "", current_workspace_subject())


def _checkout_session(key: tuple) -> tuple[Optional[_McpSession], float]:
    """Returns the session and how long it had been unused, measured before
    last_used is refreshed.

    The idle gap is returned rather than stored on the session because HTTP
    borrowers run concurrently: a second checkout would otherwise overwrite the
    first one's gap with a near-zero value and talk it out of proving a session
    that really had gone stale."""
    session = _mcp_sessions.get(key)
    if session is not None and session.is_connected():
        now = time.monotonic()
        idle_for = now - session.proved_at
        session.last_used = now
        session.in_flight += 1
        return session, idle_for
    return None, -1.0


def _borrow_key_lock(key: tuple) -> _McpKeyLock:
    """Return a stable per-key lock while a caller waits for/connects it."""
    key_lock = _mcp_key_locks.setdefault(key, _McpKeyLock())
    key_lock.users += 1
    return key_lock


def _discard_key_lock(key: tuple) -> None:
    key_lock = _mcp_key_locks.get(key)
    if key_lock is not None and key_lock.users == 0 and key not in _mcp_sessions:
        _mcp_key_locks.pop(key, None)


def _return_key_lock(key: tuple, key_lock: _McpKeyLock) -> None:
    with _mcp_sessions_lock:
        key_lock.users -= 1
        _discard_key_lock(key)


@contextmanager
def _connect_slot(url: str, headers: Optional[dict]):
    global _mcp_connects_in_flight
    with _mcp_sessions_lock:
        generation = _mcp_close_generation(url, headers)
        _mcp_connects_in_flight += 1
    try:
        yield generation
    finally:
        with _mcp_sessions_lock:
            _mcp_connects_in_flight -= 1


def _get_session(
    url: str,
    headers: Optional[dict],
    scope: Optional[str],
    deadline,
    cancel_event,
    config_check,
    use_oauth: bool = False,
) -> tuple[_McpSession, float]:
    """``deadline`` is the caller's absolute monotonic budget (None = no limit):
    the key-lock wait and the connect share it, so a slow startup can't stack
    full timeout windows (see _call_session_tool).

    Returns the session and this borrower's idle gap (negative when we connected
    it ourselves), which only the borrower may act on -- see _checkout_session."""
    global _mcp_reaper_started
    key = _session_key(url, headers, scope)
    with _mcp_sessions_lock:
        session, idle_for = _checkout_session(key)
        if session is not None:
            return session, idle_for
        key_lock = _borrow_key_lock(key)
    try:
        # Poll the acquire with connect()'s deadline/cancel semantics: a second
        # same-scope call must not block uncancellably behind another caller's
        # slow startup (e.g. a first-run npx download).
        remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
        # timeout=None means no key-lock deadline (only cancel unblocks it). The
        # wait is bounded exactly like the connect it is queueing behind, so an
        # HTTP caller is not cut off at stdio's cold-start cap here either.
        window = _connect_window(url, remaining)
        lock_deadline = None if window is None else time.monotonic() + window
        while not key_lock.lock.acquire(timeout = 0.05):
            if cancel_event is not None and cancel_event.is_set():
                raise _MCPCancelled
            if lock_deadline is not None and time.monotonic() >= lock_deadline:
                raise _ConnectTimeout(window)
        try:
            stale = None
            with _mcp_sessions_lock:
                session, idle_for = _checkout_session(key)
                if session is not None:
                    return session, idle_for
                if key in _mcp_sessions:
                    stale = _mcp_sessions.pop(key)
            if stale is not None:
                _retire_session(stale)
            with _connect_slot(url, headers) as generation:
                session = _McpSession(url, headers, use_oauth)
                try:
                    session.connect(
                        None if deadline is None else max(0.0, deadline - time.monotonic()),
                        cancel_event,
                    )
                except Exception:
                    session.close()
                    raise
                if config_check is not None:
                    try:
                        current = bool(config_check())
                    except Exception:  # noqa: BLE001
                        current = False
                    if not current:
                        session.close()
                        raise RuntimeError("MCP server was updated or removed while connecting")
                evicted: list = []
                with _mcp_sessions_lock:
                    closed_while_connecting = _mcp_close_generation(url, headers) != generation
                    if not closed_while_connecting:
                        session.in_flight = 1
                        evicted = _evict_lru_locked()  # bound the cache (LRU idle)
                        _mcp_sessions[key] = session
                        if not _mcp_reaper_started:
                            _mcp_reaper_started = True
                            threading.Thread(
                                target = _session_reaper, name = "mcp-session-reaper", daemon = True
                            ).start()
                            atexit.register(_close_sessions_at_exit)
                for victim in evicted:
                    logger.info("Evicting LRU idle MCP session: %s", _session_log_id(victim.url))
                if evicted:
                    # Detached: these belong to other scopes and nobody is waiting
                    # on them, but an unresponsive transport costs
                    # _SESSION_CLOSE_TIMEOUT each. Charging that to this caller
                    # would spend the deadline meant for their tool call.
                    _close_detached(evicted)
                if closed_while_connecting:
                    session.close()
                    raise RuntimeError("MCP server was updated or removed while connecting")
                return session, -1.0
        finally:
            key_lock.lock.release()
    finally:
        _return_key_lock(key, key_lock)


def _release_session(session: _McpSession, defer_close: bool = False) -> None:
    victims: list = []
    with _mcp_sessions_lock:
        session.in_flight = max(0, session.in_flight - 1)
        session.last_used = time.monotonic()
        close_now = session.defunct and session.in_flight == 0
        # Re-enforce the cap once a burst's sessions go idle. Insert-time eviction
        # only trims idle sessions, so it can overshoot while every cached session
        # is busy; reclaim that overshoot here instead of waiting for the idle
        # reaper. Never evict the session we just used (its last_used is newest).
        while len(_mcp_sessions) > _MAX_SESSIONS:
            idle = [
                (s.last_used, k)
                for k, s in _mcp_sessions.items()
                if s.in_flight == 0 and s is not session
            ]
            if not idle:
                break
            _, oldest = min(idle, key = lambda item: item[0])
            victims.append(_mcp_sessions.pop(oldest))
            _discard_key_lock(oldest)
    if close_now and defer_close:
        # This borrower was the last one on a session that has been discarded,
        # either by its own failure or by a sibling's. Either way the caller is
        # mid-request -- it may still have a reconnect and retry to do on the
        # same deadline -- and a transport that is being discarded because it
        # stopped answering is exactly the one whose close runs long. Only the
        # unscoped one-shot path closes inline (defer_close is False there),
        # because there the teardown is the call's own work and the caller
        # expects the subprocess gone by the time it returns.
        victims.append(session)
    elif close_now:
        session.close()
    if victims:
        _close_detached(victims)


def _retire_session(session: _McpSession) -> None:
    """Close a discarded session, but only once no other borrower is mid-call
    on it -- overlapping same-scope calls share one client, and one call's
    timeout must not kill another's in-flight request. The last borrower's
    _release_session() performs the deferred close."""
    with _mcp_sessions_lock:
        session.defunct = True
        busy = session.in_flight > 0
    if not busy:
        session.close()


def _drop_session(key: tuple, session: _McpSession) -> None:
    with _mcp_sessions_lock:
        if _mcp_sessions.get(key) is session:
            _mcp_sessions.pop(key)
        _discard_key_lock(key)
    _retire_session(session)


def _evict_lru_locked() -> list:
    """Caller holds _mcp_sessions_lock. Evict least-recently-used *idle*
    sessions until the cache is under the cap. Returns the evicted sessions so
    the caller can close them OUTSIDE the lock. If every session is busy the
    cache may transiently overshoot rather than kill an in-flight call."""
    victims: list = []
    while len(_mcp_sessions) >= _MAX_SESSIONS:
        idle = [(s.last_used, k) for k, s in _mcp_sessions.items() if s.in_flight == 0]
        if not idle:
            break
        _, oldest = min(idle, key = lambda item: item[0])
        victims.append(_mcp_sessions.pop(oldest))
        _discard_key_lock(oldest)
    return victims


def close_mcp_sessions(
    url: Optional[str] = None,
    headers = _ANY_HEADERS,
    *,
    all_workspaces: bool = False,
) -> None:
    """Close cached sessions: all of them (``url`` None), every env for one
    address (``headers`` omitted), or one server config (url + headers). Two
    server rows can share an address with different envs; editing one must not
    kill the other's live state, so the routes pass the edited row's env.

    Confined to the calling workspace, for the same reason: two accounts can
    configure the same server, and editing one account's row must not kill the
    other's live child. ``all_workspaces`` is the process-shutdown path.
    """
    global _mcp_close_all_gen
    hk = None if headers is _ANY_HEADERS else _headers_key(headers)
    subject = current_workspace_subject()
    with _mcp_sessions_lock:
        keys = [
            k
            for k in _mcp_sessions
            if (url is None or k[0] == url)
            and (hk is None or k[1] == hk)
            and (all_workspaces or k[3] == subject)
        ]
        sessions = [_mcp_sessions.pop(k) for k in keys]
        for key in keys:
            _discard_key_lock(key)
        if sessions or _mcp_connects_in_flight:
            if url is None and all_workspaces:
                _mcp_close_all_gen += 1
            elif url is None:
                _mcp_subject_close_gen[subject] = _mcp_subject_close_gen.get(subject, 0) + 1
            elif hk is None:
                uk = _url_close_key(url, subject)
                _mcp_url_close_gen[uk] = _mcp_url_close_gen.get(uk, 0) + 1
            else:
                cfg = _cfg_close_key(url, headers, subject)
                _mcp_cfg_close_gen[cfg] = _mcp_cfg_close_gen.get(cfg, 0) + 1
    pending, worker = _drain_cleanup_queue()
    _close_all(sessions + pending)
    if worker is not None and worker is not threading.current_thread():
        # Draining the queue does not recall the session the worker had already
        # popped, and this function promises its caller (a server edit, or
        # atexit) that the teardown has happened. The worker stops as soon as the
        # queue is empty, so this waits for that one close and no longer.
        worker.join(_SESSION_CLOSE_TIMEOUT + 5.0)


def _close_sessions_at_exit() -> None:
    """Process teardown: every account's sessions, not only the exiting thread's.

    atexit runs on the main thread, which carries the default workspace, so the
    workspace-confined default would leave every managed account's stdio child
    running after the server stops.
    """
    close_mcp_sessions(all_workspaces = True)


def _close_all(sessions: list) -> None:
    """Close sessions in parallel.

    Serially, each unresponsive transport can burn _SESSION_CLOSE_TIMEOUT plus
    the thread join before the next one starts. This runs on the request thread
    when a server is edited or deleted, and a popular HTTP server now holds a
    session per chat rather than one overall, so a serial close could stall that
    route for minutes.

    Fanned out _MAX_CLOSE_THREADS wide rather than one thread per session. The
    cache is allowed to overshoot _MAX_SESSIONS while every session in it is
    busy, so the list handed here has no fixed length, and a shutdown is the
    worst moment to ask the process for an unbounded number of threads."""
    if not sessions:
        return
    if len(sessions) == 1:
        sessions[0].close()
        return

    pending = list(sessions)
    pending_lock = threading.Lock()

    def _drain() -> None:
        while True:
            with pending_lock:
                if not pending:
                    return
                session = pending.pop()
            _close_quietly(session)

    # Bare threads, not a ThreadPoolExecutor: this also runs as the atexit
    # handler, and Python shuts the executor machinery down before normal atexit
    # callbacks, so submitting there raises ("can't register atexit after
    # shutdown") and the whole cleanup aborts with stdio subprocesses still up.
    width = min(len(pending), _MAX_CLOSE_THREADS)
    threads = [threading.Thread(target = _drain, name = "mcp-close", daemon = True) for _ in range(width)]
    for thread in threads:
        thread.start()
    # Each worker may take several sessions in turn, so the wait scales with the
    # rounds it has to make rather than with a single close.
    rounds = -(-len(sessions) // width)
    for thread in threads:
        thread.join(rounds * (_SESSION_CLOSE_TIMEOUT + 5.0))


def _close_detached(sessions: list) -> None:
    """Hand sessions nobody is waiting on to the cleanup worker.

    Used for LRU victims and for the deferred close of a retired session: those
    belong to another scope or to a call that has already ended, while the thread
    holding them is in the middle of serving a tool call on its own deadline and
    an unresponsive transport costs _SESSION_CLOSE_TIMEOUT to shut down.

    One worker rather than a thread per session: nobody is waiting on these, so
    closing them one at a time is fine, and a run of new chat scopes against a
    server that hangs on shutdown then cannot spawn threads without bound.
    close_mcp_sessions stays synchronous and drains this queue, because its caller
    (a server edit, or atexit) does need the teardown to have happened.

    Past _MAX_PENDING_CLOSES the overflow is closed on the caller instead. A queue
    that keeps growing means the server is shutting down slower than chats are
    opening, and every waiting session still holds its own loop thread and
    connection, so the queue has to be bounded as well as the cache. Making the
    caller wait is the backpressure that stops it: unpleasant, but the same thing
    that happened before any of this was deferred, and only once the deferral has
    already failed to keep up."""
    global _mcp_cleanup_worker
    if not sessions:
        return
    with _mcp_cleanup_lock:
        room = max(0, _MAX_PENDING_CLOSES - len(_mcp_cleanup_queue))
        _mcp_cleanup_queue.extend(sessions[:room])
        overflow = sessions[room:]
        if _mcp_cleanup_queue and _mcp_cleanup_worker is None:
            _mcp_cleanup_worker = threading.Thread(
                target = _cleanup_worker, name = "mcp-cleanup", daemon = True
            )
            _mcp_cleanup_worker.start()
    _close_all(overflow)


def _cleanup_worker() -> None:
    global _mcp_cleanup_worker
    while True:
        with _mcp_cleanup_lock:
            if not _mcp_cleanup_queue:
                _mcp_cleanup_worker = None  # _close_detached starts the next one
                return
            session = _mcp_cleanup_queue.pop(0)
        _close_quietly(session)


def _drain_cleanup_queue() -> tuple[list, Optional[threading.Thread]]:
    """Take back whatever the worker has not started on yet, plus the worker
    itself so the caller can wait out the one close already under way."""
    with _mcp_cleanup_lock:
        pending = list(_mcp_cleanup_queue)
        _mcp_cleanup_queue.clear()
        return pending, _mcp_cleanup_worker


def _close_quietly(session) -> None:
    try:
        session.close()
    except Exception:  # noqa: BLE001
        logger.exception("Closing a discarded MCP session failed")


# The cache stopped being stdio-only, but an in-place upgrade can leave a caller
# holding the old name; it costs two lines to keep it working.
close_stdio_sessions = close_mcp_sessions


def _reset_after_fork() -> None:
    """Drop everything the child inherited from the parent's cache.

    Only the forking thread survives a fork, so every session's loop thread is
    gone while its client still reports connected. A child that checked one out
    would wait on a loop that will never run, and _transport_dead cannot see it
    for HTTP. Nothing here is closed: those objects belong to the parent, which
    is still using them."""
    global _mcp_reaper_started, _mcp_connects_in_flight, _mcp_sessions_lock
    global _mcp_cleanup_lock, _mcp_cleanup_worker
    # Replaced, not just cleared: a lock the fork caught held belongs to a thread
    # that no longer exists here, so the child would block on it forever.
    _mcp_sessions_lock = threading.Lock()
    _mcp_sessions.clear()
    _mcp_key_locks.clear()
    _mcp_connects_in_flight = 0
    _mcp_reaper_started = False
    # The cleanup worker did not survive the fork either, and its queue holds the
    # parent's sessions.
    _mcp_cleanup_lock = threading.Lock()
    _mcp_cleanup_queue.clear()
    _mcp_cleanup_worker = None


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child = _reset_after_fork)


def _reap_idle_sessions(now: Optional[float] = None) -> None:
    now = time.monotonic() if now is None else now
    with _mcp_sessions_lock:
        expired = [
            key
            for key, session in _mcp_sessions.items()
            if session.in_flight == 0 and now - session.last_used >= _SESSION_IDLE_TTL
        ]
        sessions = [_mcp_sessions.pop(key) for key in expired]
        for key in expired:
            _discard_key_lock(key)
    for session in sessions:
        logger.info("Closing idle MCP session: %s", _session_log_id(session.url))
        session.close()


def _session_reaper() -> None:
    while True:
        time.sleep(_SESSION_REAP_INTERVAL)
        try:
            _reap_idle_sessions()
        except Exception as exc:  # noqa: BLE001
            logger.debug("MCP session reaper iteration failed: %s", exc)


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


# Discovered-tool cache, keyed by MCP server id. get_enabled_mcp_tools() probes a server only
# on a cache miss, keeping MCP discovery off the chat send's critical path -- tool schemas are
# stable within a session. The /refresh route warms it; a URL/header/OAuth change or a delete
# evicts it. Successful probes are cached indefinitely.
_tool_cache: dict[str, list[dict]] = {}

# server_id -> monotonic time before which a failed server must not be
# re-probed (see record_probe_failure). Cleared on a successful probe or
# eviction.
_probe_cooloff_until: dict[str, float] = {}

# Coordinate off-loop token-count snapshots with row and schema-cache mutations.
_mcp_server_snapshot_locks: WeakKeyDictionary[asyncio.AbstractEventLoop, asyncio.Lock] = (
    WeakKeyDictionary()
)


def mcp_server_snapshot_guard() -> asyncio.Lock:
    loop = asyncio.get_running_loop()
    return _mcp_server_snapshot_locks.setdefault(loop, asyncio.Lock())


def serialize_mcp_server_mutation(handler):
    """Run an MCP mutation from validation through row/cache commit as one snapshot."""

    @wraps(handler)
    async def _serialized(*args, **kwargs):
        async with mcp_server_snapshot_guard():
            return await handler(*args, **kwargs)

    return _serialized


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


MCP_IMAGES_SENTINEL = "__MCP_IMAGES__:"
MAX_IMAGE_PAYLOAD_CHARS = 12_000_000


def _block_text(block: Any) -> Optional[str]:
    text = getattr(block, "text", None)
    if text:
        return str(text)
    resource = getattr(block, "resource", None)
    if resource is not None:
        text = getattr(resource, "text", None)
        return str(text) if text else None
    return None


def _block_link(block: Any) -> Optional[str]:
    # keep host-generated link text from suppressing structured_content
    uri = getattr(block, "uri", None)
    if uri and getattr(block, "type", None) == "resource_link":
        name = getattr(block, "name", None)
        return f"[resource: {name} <{uri}>]" if name else f"[resource: <{uri}>]"
    return None


# fastmcp File(data=..., format=...) labels payloads as application/<format>
_IMAGE_SUBTYPES = {
    "apng": "image/apng",
    "png": "image/png",
    "jpeg": "image/jpeg",
    "jpg": "image/jpeg",
    "gif": "image/gif",
    "webp": "image/webp",
    "bmp": "image/bmp",
    "avif": "image/avif",
    "tif": "image/tiff",
    "tiff": "image/tiff",
    "ico": "image/vnd.microsoft.icon",
    "svg": "image/svg+xml",
    "svg+xml": "image/svg+xml",
}


# What tool-fallback.tsx may interpolate into data:<type>;base64,... : an RFC 9110
# 8.3.1 token subtype, minus "*", which names a range and never a payload.
_MEDIA_TYPE = re.compile(r"^image/[a-z0-9][a-z0-9!#$%&'^_`|~.+-]*$")


def _uri_mime(uri: Any) -> Optional[str]:
    """Guess a media type from the part of a URI that names the resource.

    mimetypes only stopped reading the query and fragment in 3.11.9 / 3.12.3 / 3.13
    (CPython gh-117217), and on older supported interpreters 'gen.png?download=1'
    guessed nothing while 'download?name=gen.png' guessed image/png. Dropping both
    keeps every interpreter in agreement. The scheme stays so a data: URI still
    resolves; a bare host goes, since a host name is not a file name."""
    split = urlsplit(str(uri))
    cleaned = urlunsplit((split.scheme, split.netloc if split.path else "", split.path, "", ""))
    return mimetypes.guess_type(cleaned, strict = False)[0]


def _image_mime(mime: Any) -> Optional[str]:
    if not isinstance(mime, str):
        return None
    # media type names are case-insensitive; data urls need only the essence
    essence = mime.partition(";")[0].strip().lower()
    if essence.startswith("image/"):
        resolved = essence
    else:
        subtype = essence[len("application/") :] if essence.startswith("application/") else ""
        resolved = _IMAGE_SUBTYPES.get(subtype) or _uri_mime(f"file:///image.{subtype}")
    # one gate for every branch. Lowercased again because a registry answer carries the
    # host's spelling: Windows returns image/JXL for .jxl, Linux and macOS image/jxl.
    resolved = resolved.lower() if resolved else ""
    return resolved if _MEDIA_TYPE.match(resolved) else None


def _resource_mime(obj: Any) -> Any:
    # mcp 2.x renames mimeType to mime_type, keeping camelCase only as an alias
    mime = getattr(obj, "mimeType", None)
    return mime if mime is not None else getattr(obj, "mime_type", None)


def _block_image(block: Any) -> Optional[tuple[str, str]]:
    # embedded resources keep binary data on resource.blob
    data = getattr(block, "data", None)
    mime = _resource_mime(block)
    if not data:
        resource = getattr(block, "resource", None)
        if resource is None:
            return None
        data = getattr(resource, "blob", None)
        mime = _resource_mime(resource)
        if not mime:
            uri = getattr(resource, "uri", None)
            mime = _uri_mime(uri) if uri else None
    mime = _image_mime(mime)
    if data and mime:
        return str(data), mime
    return None


def _flatten_result(result: Any) -> str:
    parts = []
    images = []
    omitted = 0
    has_text = False
    budget = MAX_IMAGE_PAYLOAD_CHARS
    for block in getattr(result, "content", None) or []:
        text = _block_text(block)
        if text:
            parts.append(text)
            has_text = True
            continue
        link = _block_link(block)
        if link:
            parts.append(link)
            continue
        image = _block_image(block)
        if image is not None:
            data, mime = image
            if len(data) > budget:
                omitted += 1
                continue
            budget -= len(data)
            images.append({"data": data, "mimeType": mime})
    body = "\n".join(parts)
    if not has_text:
        structured = getattr(result, "structured_content", None)
        if structured is not None:
            body = f"{structured}\n{body}" if body else str(structured)
    if images or omitted:
        notes = []
        if images:
            n = len(images)
            notes.append(f"{n} image{'s' if n > 1 else ''} attached; displayed to the user")
        if omitted:
            notes.append(f"{omitted} image{'s' if omitted > 1 else ''} omitted (too large)")
        note = f"[{'; '.join(notes)}]"
        body = f"{body}\n{note}" if body else note

    if getattr(result, "is_error", False):
        # "Error: " prefix triggers tool_call_parser's TOOL_ERROR_PREFIXES nudge.
        body = f"Error: {body}" if body else "Error: tool returned no content"
    if images:
        body += "\n" + MCP_IMAGES_SENTINEL + json.dumps(images)
    return body


def _unwind_budget(unwind_timeout: float, timeout: Optional[float], elapsed: float) -> float:
    """How long a cancelled call may take to unwind: whatever is left of the
    caller's window, so the wait is never charged on top of an expired deadline."""
    if not unwind_timeout:
        return 0.0
    if timeout is None:
        return unwind_timeout
    return min(unwind_timeout, max(0.0, timeout - elapsed))


async def _race_tool_call(
    call_coro,
    timeout: Optional[float],
    cancel_event,
    unwind_timeout: float = 0.0,
) -> Any:
    """Await ``call_coro`` under ``timeout``, polling ``cancel_event`` so a
    /cancel POST interrupts even mid-network-read. ``unwind_timeout`` waits up to
    that long for a cancelled call to finish unwinding; only callers that hand the
    client back to a cache need it (one-shot clients are discarded anyway)."""

    async def _watch_cancel() -> None:
        while cancel_event is not None and not cancel_event.is_set():
            await asyncio.sleep(0.05)

    if cancel_event is not None and cancel_event.is_set():
        call_coro.close()
        raise _MCPCancelled
    started = time.monotonic()
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
        # Let a cancelled call unwind before its session is reused, out of the
        # caller's remaining budget. Outlasting it just leaves the session dirty.
        left = _unwind_budget(unwind_timeout, timeout, time.monotonic() - started)
        if left:
            await asyncio.wait({call_task, watch_task}, timeout = left)
    if not done:
        raise asyncio.TimeoutError
    if call_task in done:
        return call_task.result()
    raise _MCPCancelled


def _call_session_tool(
    url: str,
    headers: Optional[dict],
    name: str,
    args: dict,
    timeout,
    cancel_event,
    scope: Optional[str],
    config_check,
    use_oauth: bool = False,
) -> Any:
    if cancel_event is not None and cancel_event.is_set():
        raise _MCPCancelled
    # One deadline covers the key-lock wait, connect, call-lock wait, and the
    # call itself, matching the one-shot path where the caller's timeout wrapped
    # connect plus call in a single window.
    deadline = None if timeout is None else time.monotonic() + timeout

    def _remaining() -> Optional[float]:
        return None if deadline is None else max(0.0, deadline - time.monotonic())

    # Callers without an Unsloth session id must retain the former one-shot
    # behavior: no browser/cookie/tool state can leak into another request.
    # Use an ephemeral key (and close it below) rather than the shared empty
    # scope that the persistent-session cache used previously.
    def _config_ok() -> bool:
        if config_check is None:
            return True
        try:
            return bool(config_check())
        except Exception:  # noqa: BLE001
            return False

    ephemeral = not scope
    if ephemeral:
        scope = f"request-{uuid.uuid4().hex}"
    key = _session_key(url, headers, scope)
    # attempt 0 may find the cached session stale/dead *before* dispatch and
    # reconnect once (safe); attempt 1 is a freshly connected session.
    for attempt in (0, 1):
        session, idle_for = _get_session(
            url, headers, scope, deadline, cancel_event, config_check, use_oauth
        )
        locked = False
        try:
            # Serialize calls per session where the transport demands it:
            # overlapping same-scope calls must not interleave operations on one
            # stateful stdio server (browser, REPL). HTTP multiplexes by request
            # id, so its calls run in parallel as they did one-shot.
            if session.serialize_calls:
                while not session.call_lock.acquire(timeout = 0.05):
                    if cancel_event is not None and cancel_event.is_set():
                        raise _MCPCancelled
                    rem = _remaining()
                    if rem is not None and rem <= 0:
                        raise asyncio.TimeoutError
                locked = True
        except BaseException:
            # Never touched the transport: keep the session for its borrower.
            _release_session(session)
            if ephemeral:
                _drop_session(key, session)
            raise
        discard_session = ephemeral
        retry = False
        try:
            # We may have waited on the call lock while another caller's timeout retired this
            # session, a server update/delete invalidated it, or a reused subprocess died. Re-check
            # all three before dispatch so we never run on a retired/dead client or a stale config.
            if session.closed.is_set():
                # Intentional close (server update/delete/shutdown): don't retry on stale config.
                discard_session = True
                raise RuntimeError("MCP server was updated or removed during the call")
            elif session.defunct:
                # A concurrent same-scope caller's timeout retired this session;
                # move to a fresh one instead of reusing the retired client.
                discard_session = True
                if attempt == 0:
                    retry = True
                else:
                    raise RuntimeError("MCP server session was retired during the call")
            elif not _config_ok():
                discard_session = True
                raise RuntimeError("MCP server was updated or removed during the call")
            elif _transport_dead(session):
                # Dead BEFORE dispatch: no request was sent, so reconnect + retry.
                discard_session = True
                if attempt == 0:
                    retry = True
                else:
                    raise RuntimeError("MCP server connection is not available")
            elif (
                session.dirty or _needs_idle_recheck(session, idle_for, _remaining())
            ) and not _session_responsive(
                session, _remaining(), cancel_event, timeout_is_fatal = session.dirty
            ):
                # Dirty: still stuck on the abandoned call. Idle HTTP: the server
                # may have expired the session while nothing was using it, and no
                # HTTP transport lets us ask. Either way it failed to answer, so
                # reconnect BEFORE dispatch rather than losing the user's call.
                discard_session = True
                if attempt == 0:
                    retry = True
                else:
                    raise RuntimeError("MCP server is not responding")
            else:
                rem = _remaining()
                # raise_on_error=False for the same reason as the one-shot path.
                coro = _race_tool_call(
                    session.client.call_tool(name, args, raise_on_error = False),
                    rem,
                    cancel_event,
                    # Only a cached session is worth waiting on.
                    0.0 if ephemeral else _CANCEL_UNWIND_TIMEOUT,
                )
                out = session.run(coro, rem)
                # A completed round trip proves the transport better than any
                # probe could, so it resets the idle clock the recheck reads.
                session.proved_at = time.monotonic()
                return out
        except (_MCPCancelled, asyncio.TimeoutError):
            # Keep the session so a Stop doesn't destroy the server's state; the
            # SDK drops the abandoned reply, and reuse is gated on a live probe.
            session.dirty = True
            raise
        except _SessionWedged:
            discard_session = True
            raise asyncio.TimeoutError
        except _SessionClosed:
            # close_mcp_sessions() shut this session mid-call (server
            # update/delete/shutdown); don't retry on the stale config.
            discard_session = True
            raise RuntimeError("MCP server was updated or removed during the call")
        except Exception as exc:
            if session.closed.is_set():
                # An intentional close (server update/delete) can surface as a plain transport
                # error or AttributeError instead of _SessionClosed; don't mistake it for a crash.
                discard_session = True
                raise RuntimeError("MCP server was updated or removed during the call")
            # A ToolError or a JSON-RPC error response means the server answered,
            # so the transport is alive -> keep the session and its state.
            # Anything else is transport-level (dead subprocess, broken pipe,
            # dropped HTTP stream): evict so it can't poison the scope, but DO
            # NOT replay (the tool may already have run); the next call opens a
            # fresh session.
            if _is_protocol_error(exc):
                # The server replied, so the transport is fine and the session's
                # state is worth keeping. Probe it before the next call anyway,
                # in case the error was the server telling us the session is no
                # longer one it recognises.
                session.dirty = True
            elif not _is_tool_error(exc):
                discard_session = True
            raise
        finally:
            # Remove from the cache and mark defunct BEFORE giving up the borrow,
            # so no other caller can check this session out after it failed. The
            # order matters more now that HTTP callers do not queue on call_lock:
            # released first, a concurrent same-scope call could check out the
            # broken transport while it was still cached, and _release_session can
            # sit closing LRU victims for seconds first. in_flight is still held
            # here, so the close defers to the release below.
            if discard_session:
                _drop_session(key, session)
            _release_session(session, defer_close = not ephemeral)
            if locked:
                session.call_lock.release()
        if not retry:
            break
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
    """Call one MCP tool and return its flattened text/image result.

    Never raises: every failure comes back as an "Error: ..." string for the model.

    Which transport path runs depends on ``scope`` (an opaque per-chat key) and
    ``use_oauth``:

    * stdio always goes through the session machinery. Without a scope it still
      gets a private ephemeral session that is closed afterwards, so no browser
      or cookie state leaks between requests.
    * HTTP reuses a cached session only when it has a scope AND OAuth is off, so
      a server that keeps state between calls keeps it for the whole chat.
    * OAuth HTTP, and HTTP without a scope, connect once and disconnect, exactly
      as before sessions were shared. Refreshing a token on a long-lived shared
      connection is not something this code can do safely yet.

    ``timeout`` is one budget covering connect and call together. ``cancel_event``
    aborts an in-flight call. ``config_check`` re-reads the server row so a call
    that raced an edit or delete cannot dispatch on the stale configuration."""

    async def _one_shot() -> Any:
        async with _client(url, headers, use_oauth) as client:
            # raise_on_error=False lets an is_error result (which may still carry
            # image content) reach _flatten_result instead of FastMCP raising ToolError
            # and dropping the images. Transport failures still raise (handled below).
            return await client.call_tool(name, args, raise_on_error = False)

    try:
        if is_stdio(url) or (scope and not use_oauth):
            result = _call_session_tool(
                url, headers, name, args, timeout, cancel_event, scope, config_check, use_oauth
            )
        else:
            result = asyncio.run(_race_tool_call(_one_shot(), timeout, cancel_event))
    except _MCPCancelled:
        return f"Error: MCP tool '{name}' cancelled"
    except _ConnectTimeout as exc:
        # Report the window that actually expired: for stdio that is the
        # cold-start cap, not the (larger) caller timeout.
        suffix = f" after {round(exc.window, 1):g}s" if exc.window is not None else ""
        return f"Error: MCP tool '{name}' timed out connecting{suffix}"
    except asyncio.TimeoutError:
        suffix = f" after {timeout:g}s" if timeout is not None else ""
        return f"Error: MCP tool '{name}' timed out{suffix}"
    except Exception as exc:
        logger.exception("MCP call_tool failed for %s: %s", name, exc)
        return f"Error: MCP tool '{name}' failed: {exc}"

    return _flatten_result(result)


class _MCPCancelled(Exception):
    """Internal sentinel raised when cancel_event fires before the tool returns."""
