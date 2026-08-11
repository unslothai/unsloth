# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Issue #8404: no h11 traceback when a poll lands after we close the connection.

On Windows, every clean shutdown printed an unhandled asyncio traceback ending in
``h11._util.LocalProtocolError: can't handle event type Response when role=SERVER
and state=CLOSED``. The frontend keeps polling ``/api/inference/status`` while
uvicorn is closing connections, so a request is already in the socket when
``H11Protocol.shutdown()`` sends ``h11.ConnectionClosed()`` and closes the
transport. The proactor transport hands that read to the protocol anyway (the
selector transport used on Linux and macOS removes the reader inside
``close()``, which is why this only shows up on Windows), h11 then rejects the
bytes and uvicorn's unguarded ``send_400_response()`` blows up.

Two things in these fixtures are shaped by the platform, not by convenience:

* the post-close ``data_received()`` call is made directly, exactly as
  ``_ProactorReadPipeTransport._loop_reading()``'s ``finally:`` clause does it,
  because Linux's selector transport will never make that call;
* ``uvicorn.protocols.http.auto.AutoHTTPProtocol`` is forced to ``H11Protocol``,
  because a dev box with httptools installed resolves to httptools while the
  Studio requirements pin plain uvicorn, which is the h11 path the report hit.

Everything downstream of the injected read -- the h11 state machine, the
``RemoteProtocolError``, uvicorn's 400 path -- is the real code.
"""

from __future__ import annotations

import ast
import asyncio
import logging
from pathlib import Path

import h11
import pytest
from uvicorn.config import Config
from uvicorn.protocols.http.h11_impl import H11Protocol
from uvicorn.server import ServerState

from utils.uvicorn_h11_shutdown import uvicorn_http_protocol


REQUEST = b"GET /api/inference/status HTTP/1.1\r\nHost: 127.0.0.1\r\n\r\n"


@pytest.fixture
def patched_protocol_class(monkeypatch):
    """The protocol Studio installs when uvicorn would otherwise use plain h11."""
    import uvicorn.protocols.http.auto as auto

    monkeypatch.setattr(auto, "AutoHTTPProtocol", H11Protocol)
    protocol_class = uvicorn_http_protocol()
    assert protocol_class is not "auto"  # noqa: F632 - a class is expected here
    assert issubclass(protocol_class, H11Protocol)
    return protocol_class


class _FakeTransport(asyncio.Transport):
    """Enough of a transport for H11Protocol; records writes, owns no socket."""

    def __init__(self):
        super().__init__()
        self.chunks = []
        self.closing = False

    def write(self, data):
        self.chunks.append(bytes(data))

    def close(self):
        self.closing = True

    def is_closing(self):
        return self.closing

    def get_extra_info(
        self,
        name,
        default = None,
    ):
        if name == "sockname":
            return ("127.0.0.1", 8000)
        if name == "peername":
            return ("127.0.0.1", 54321)
        return default

    def pause_reading(self):
        pass

    def resume_reading(self):
        pass


async def _app(scope, receive, send):
    assert scope["type"] == "http"
    await send(
        {
            "type": "http.response.start",
            "status": 200,
            "headers": [(b"content-type", b"text/plain")],
        }
    )
    await send({"type": "http.response.body", "body": b"ok"})


def _build_protocol(protocol_class):
    config = Config(app = _app, access_log = False)
    config.load()
    return protocol_class(
        config = config,
        server_state = ServerState(),
        app_state = {},
    )


async def _serve_one_request(protocol_class):
    protocol = _build_protocol(protocol_class)
    transport = _FakeTransport()
    protocol.connection_made(transport)

    protocol.data_received(REQUEST)
    for _ in range(100):
        await asyncio.sleep(0)
        if protocol.cycle is not None and protocol.cycle.response_complete:
            break
    assert protocol.cycle is not None and protocol.cycle.response_complete
    return protocol, transport


async def _serve_then_shutdown_then_poll(protocol_class):
    """One keep-alive request, then a graceful shutdown, then a late poll."""
    protocol, transport = await _serve_one_request(protocol_class)

    # uvicorn.Server.shutdown() calls this on every live connection.
    protocol.shutdown()
    assert protocol.conn.our_state is h11.CLOSED

    # The poll that was already in the socket when we closed.
    protocol.data_received(REQUEST)
    return protocol, transport


def test_late_poll_after_shutdown_is_dropped(patched_protocol_class, caplog):
    """The patched protocol must ignore the post-close read instead of answering it."""
    with caplog.at_level(logging.WARNING, logger = "uvicorn.error"):
        protocol, transport = asyncio.run(_serve_then_shutdown_then_poll(patched_protocol_class))

    assert protocol.conn.our_state is h11.CLOSED
    assert b"400" not in b"".join(transport.chunks)
    assert "Invalid HTTP request received." not in caplog.text


def test_unpatched_h11_protocol_still_raises():
    """Pin the upstream behaviour the fix works around, so the test cannot go vacuous."""
    with pytest.raises(h11.LocalProtocolError) as excinfo:
        asyncio.run(_serve_then_shutdown_then_poll(H11Protocol))
    assert "state=CLOSED" in str(excinfo.value)


def test_live_connection_still_parses_normally(patched_protocol_class):
    """The guard must only fire on a closed connection, never on a live one."""
    _protocol, transport = asyncio.run(_serve_one_request(patched_protocol_class))
    assert b"200 OK" in b"".join(transport.chunks)


def test_httptools_choice_is_left_alone(monkeypatch):
    """uvicorn picks httptools when it is installed; that path needs no patching."""
    import uvicorn.protocols.http.auto as auto

    class _NotH11:
        pass

    monkeypatch.setattr(auto, "AutoHTTPProtocol", _NotH11)
    assert uvicorn_http_protocol() == "auto"


def test_run_server_passes_the_protocol_to_uvicorn_config():
    """AST-only check, because importing run.py drags in torch and the whole app."""
    source = (Path(__file__).resolve().parents[1] / "run.py").read_text(encoding = "utf-8")
    tree = ast.parse(source)

    wired = False
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
        if name != "dict":
            continue
        for keyword in node.keywords:
            if keyword.arg != "http":
                continue
            value = keyword.value
            if (
                isinstance(value, ast.Call)
                and getattr(value.func, "id", "") == "uvicorn_http_protocol"
            ):
                wired = True
    assert wired, "run_server must build uvicorn.Config with http = uvicorn_http_protocol()"
