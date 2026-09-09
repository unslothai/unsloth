# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""End-to-end MCP session behaviour against a real server over Streamable HTTP.

Everything else in the MCP suite runs against a fake client that takes a url and
discards the headers, so nothing there observes a real HTTP request. These start
a real MCP server on 127.0.0.1 and ask it what it saw.

Oracle: fastmcp negotiates sessionless Streamable HTTP here, so there is no
Mcp-Session-Id to follow and Context.session_id is a fresh UUID per request
(checked with a held-open client). The server therefore keys its state on the
client's TCP connection, which is what a stateful server's per-connection state
behaves like: a held-open client sees its own notes, separate clients do not.
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path

import pytest

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

pytest.importorskip("fastmcp")
pytest.importorskip("uvicorn")

from core.inference import mcp_client
from core.inference.mcp_client import call_tool_sync, close_mcp_sessions

# Matches the scope tools.py builds: "s={session_id}:t={thread_id}".
SCOPE = "s=sess1:t=threadA"
SCOPE_B = "s=sess1:t=threadB"

_SERVER = '''
import contextvars, sys, threading
import uvicorn
from fastmcp import FastMCP
from fastmcp.server.dependencies import get_http_headers
from fastmcp.server.middleware import Middleware

mcp = FastMCP("notes")
_peer = contextvars.ContextVar("peer", default="unknown")
_lock = threading.Lock()
_notes, _peers = {}, []
_live = _max_live = 0
_expired = set()


class Observe:
    """Raw ASGI: BaseHTTPMiddleware buffers the response and deadlocks the SSE
    stream Streamable HTTP replies on."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        client = scope.get("client")
        peer = f"{client[0]}:{client[1]}" if client else "unknown"
        _peer.set(peer)
        with _lock:
            if peer not in _peers:
                _peers.append(peer)
            expired = bool(_expired)
            _expired.clear()
        if expired:
            # What MCP requires of a server that terminated a session: the next
            # request gets 404 and the client must start a new one. One-shot and
            # keyed on nothing, because which socket a fastmcp client uses for a
            # given request is a pooling detail that varies by version.
            await send({
                "type": "http.response.start",
                "status": 404,
                "headers": [(b"content-type", b"text/plain")],
            })
            await send({"type": "http.response.body", "body": b"Session not found"})
            return
        await self.app(scope, receive, send)


def _protocol_error(code, message):
    """mcp<2 names the class McpError and takes an ErrorData; mcp 2 renamed it
    MCPError and takes the fields. constraints.txt allows both."""
    import mcp.shared.exceptions as mcp_exceptions
    from mcp.types import ErrorData

    cls = getattr(mcp_exceptions, "MCPError", None) or mcp_exceptions.McpError
    try:
        return cls(code=code, message=message)
    except TypeError:
        return cls(ErrorData(code=code, message=message))


class ProtocolErrors(Middleware):
    """Answer one tool with a JSON-RPC error instead of a result, the way the
    spec lets a server report an unknown tool. FastMCP itself returns a result
    carrying is_error for that, so this stands in for the servers that do not;
    raising from inside a tool would not reach the client as MCPError."""

    async def on_call_tool(self, context, call_next):
        if context.message.name == "protocol_error":
            raise _protocol_error(-32602, "Unknown tool: protocol_error")
        return await call_next(context)


mcp.add_middleware(ProtocolErrors())


@mcp.tool
def protocol_error() -> str:
    return "never reached"


@mcp.tool
def save_note(text: str) -> str:
    peer = _peer.get()
    with _lock:
        _notes.setdefault(peer, []).append(text)
    return f"saved on connection {peer}"


@mcp.tool
def list_notes() -> str:
    peer = _peer.get()
    with _lock:
        return f"connection={peer} notes={list(_notes.get(peer, []))}"


@mcp.tool
def whoami() -> str:
    headers = get_http_headers()
    return f"connection={_peer.get()} credential={headers.get('x-test-credential', '<none>')}"


@mcp.tool
def expire_me() -> str:
    """Make the next request 404, the way a server that dropped the session does."""
    with _lock:
        _expired.add("next")
    return "expired"


@mcp.tool
async def delayed_call(delay: float) -> str:
    global _live, _max_live
    import asyncio
    peer = _peer.get()
    with _lock:
        _live += 1
        _max_live = max(_max_live, _live)
    try:
        await asyncio.sleep(delay)
    finally:
        with _lock:
            _live -= 1
    return f"connection={peer}"


@mcp.tool
def stats() -> str:
    with _lock:
        return f"connections={len(_peers)} max_concurrency={_max_live}"


@mcp.tool
def reset_stats() -> str:
    global _live, _max_live
    with _lock:
        _max_live = 0
        _peers.clear()
        _notes.clear()
        _expired.clear()
    return "reset"


if __name__ == "__main__":
    uvicorn.run(
        Observe(mcp.http_app()), host=sys.argv[2], port=int(sys.argv[1]), log_level="warning"
    )
'''


def _free_port(host: str) -> int:
    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    with socket.socket(family) as s:
        s.bind((host, 0))
        return s.getsockname()[1]


def _start(tmp_path: Path, host: str):
    script = tmp_path / "mcp_notes_server.py"
    script.write_text(textwrap.dedent(_SERVER))
    port = _free_port(host)
    proc = subprocess.Popen(
        [sys.executable, str(script), str(port), host],
        env = dict(os.environ, PYTHONUNBUFFERED = "1"),
        stdout = subprocess.PIPE,
        stderr = subprocess.STDOUT,
        text = True,
    )
    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    deadline = time.monotonic() + 90
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"server died:\n{proc.stdout.read()}")
        try:
            with socket.socket(family) as s:
                s.settimeout(0.5)
                s.connect((host, port))
            break
        except OSError:
            time.sleep(0.2)
    else:
        proc.kill()
        raise RuntimeError("server never came up")
    label = f"[{host}]" if ":" in host else host
    return proc, f"http://{label}:{port}/mcp/"


@pytest.fixture(scope = "module")
def server(tmp_path_factory):
    proc, url = _start(tmp_path_factory.mktemp("mcp"), "127.0.0.1")
    try:
        yield url
    finally:
        close_mcp_sessions()
        proc.terminate()
        try:
            proc.wait(15)
        except subprocess.TimeoutExpired:
            proc.kill()


@pytest.fixture(autouse = True)
def clean_cache():
    yield
    close_mcp_sessions()


def _call(
    url,
    name,
    args = None,
    *,
    scope = None,
    headers = None,
    use_oauth = False,
):
    return call_tool_sync(
        url,
        headers,
        name,
        args or {},
        timeout = 60.0,
        use_oauth = use_oauth,
        scope = scope,
    )


def _conn(text: str) -> str:
    return text.split("connection=")[1].split(" ")[0]


def test_a_note_survives_to_the_next_tool_call_in_one_chat(server):
    """The behaviour the shared session exists for. Reconnecting per call loses
    whatever the server kept."""
    saved = _call(server, "save_note", {"text": "buy milk"}, scope = SCOPE)
    listed = _call(server, "list_notes", scope = SCOPE)
    assert "buy milk" in listed, listed
    assert saved.split("connection ")[-1].strip() == _conn(listed)


def test_a_second_chat_gets_its_own_connection(server):
    _call(server, "save_note", {"text": "chat-A-note"}, scope = SCOPE)
    other = _call(server, "list_notes", scope = SCOPE_B)
    assert "chat-A-note" not in other, other


def test_unscoped_calls_keep_the_old_one_shot_isolation(server):
    _call(server, "save_note", {"text": "unscoped"}, scope = None)
    listed = _call(server, "list_notes", scope = None)
    assert "unscoped" not in listed, listed


def test_credentials_reach_the_server_and_stay_apart(server):
    a = _call(server, "whoami", scope = SCOPE, headers = {"X-Test-Credential": "cred-A"})
    b = _call(server, "whoami", scope = SCOPE, headers = {"X-Test-Credential": "cred-B"})
    assert "cred-A" in a and "cred-B" in b
    assert "cred-A" not in b, f"credential A leaked into the B session: {b}"
    assert _conn(a) != _conn(b)


def test_an_oauth_server_is_never_cached(server):
    _call(server, "list_notes", scope = SCOPE, use_oauth = True)
    assert mcp_client._mcp_sessions == {}


def test_state_survives_a_long_run_of_calls_in_one_chat(server):
    """The property the shared session actually buys, held over many calls.

    Deliberately not asserted as a socket count: how many TCP connections a
    fastmcp client keeps open is a pooling detail that differs by version (3.0.2
    opens far more than 4.0.0 for the same work), and the MCP spec makes every
    JSON-RPC message its own POST regardless. Server-side state is the invariant."""
    _call(server, "save_note", {"text": "note-0"}, scope = SCOPE)
    for i in range(1, 10):
        _call(server, "save_note", {"text": f"note-{i}"}, scope = SCOPE)
    listed = _call(server, "list_notes", scope = SCOPE)
    for i in range(10):
        assert f"note-{i}" in listed, f"note-{i} was lost: {listed}"


def test_parallel_calls_in_one_chat_are_not_serialized(server):
    """Regression: sharing a session must not cost the parallelism the one-shot
    path had. MCP Streamable HTTP posts each message separately, so there is
    nothing to interleave."""
    _call(server, "reset_stats", scope = "s=admin:t=admin")
    results: list[str] = []
    lock = threading.Lock()

    def run():
        r = _call(server, "delayed_call", {"delay": 1.0}, scope = SCOPE)
        with lock:
            results.append(r)

    started = time.monotonic()
    threads = [threading.Thread(target = run) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(60)
    elapsed = time.monotonic() - started
    stats = _call(server, "stats", scope = "s=admin:t=admin")
    assert len(results) == 2
    assert "max_concurrency=2" in stats, stats
    assert elapsed < 1.8, f"the two calls were serialized: {elapsed:.2f}s ({stats})"


def test_an_expired_session_is_replaced_before_the_users_call_fails(monkeypatch, server):
    """A server may drop an HTTP session at any time and no HTTP transport
    exposes a liveness probe, so without the idle recheck the user's next tool
    call is the thing that discovers it."""
    monkeypatch.setattr(mcp_client, "_HTTP_IDLE_RECHECK", 0.0)
    _call(server, "list_notes", scope = SCOPE)
    _call(server, "expire_me", scope = SCOPE)
    # The next request on that session gets 404, which is what the spec requires
    # of a server that terminated it. The recheck must absorb that and reconnect
    # rather than letting the user's tool call be the thing that discovers it.
    second = _call(server, "list_notes", scope = SCOPE)
    assert not second.startswith("Error:"), second


def test_ipv6_loopback_works(tmp_path):
    if not socket.has_ipv6:
        pytest.skip("no IPv6 on this host")
    try:
        proc, url = _start(tmp_path, "::1")
    except OSError:
        pytest.skip("IPv6 loopback unavailable")
    try:
        saved = _call(url, "save_note", {"text": "v6"}, scope = SCOPE)
        assert "saved on connection" in saved, saved
        assert "v6" in _call(url, "list_notes", scope = SCOPE)
    finally:
        close_mcp_sessions()
        proc.terminate()
        try:
            proc.wait(15)
        except subprocess.TimeoutExpired:
            proc.kill()


def test_a_json_rpc_error_does_not_cost_the_chat_its_state(server):
    """The server replied, so the connection is fine and the notes it is holding
    for this chat must survive."""
    _call(server, "save_note", {"text": "before-the-error"}, scope = SCOPE)
    failed = _call(server, "protocol_error", scope = SCOPE)
    assert failed.startswith("Error:"), failed
    listed = _call(server, "list_notes", scope = SCOPE)
    assert "before-the-error" in listed, f"the session was discarded: {listed}"
