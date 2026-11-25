# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A 127.0.0.1 Studio also holds [::1]:<port>, so http://localhost cannot reach another local listener."""

import asyncio
import socket

import pytest
import uvicorn

import run


def _ipv6_loopback_available() -> bool:
    try:
        with socket.socket(socket.AF_INET6, socket.SOCK_STREAM) as probe:
            probe.bind(("::1", 0))
        return True
    except OSError:
        return False


async def _app(scope, receive, send):
    if scope["type"] != "http":
        return
    await send({"type": "http.response.start", "status": 200, "headers": []})
    await send({"type": "http.response.body", "body": b"studio"})


def _factory(server):
    config = server.config

    def _protocol(_loop = None):
        return config.http_protocol_class(
            config = config,
            server_state = server.server_state,
            app_state = server.lifespan.state,
            _loop = _loop,
        )

    return _protocol


async def _get(host: str, port: int) -> bytes:
    reader, writer = await asyncio.open_connection(host, port)
    writer.write(b"GET / HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n")
    await writer.drain()
    data = await reader.read()
    writer.close()
    return data


@pytest.mark.skipif(not _ipv6_loopback_available(), reason = "no IPv6 loopback on this host")
def test_studio_answers_on_ipv6_loopback_too():
    async def scenario():
        server = uvicorn.Server(
            uvicorn.Config(_app, host = "127.0.0.1", port = 0, log_level = "error", lifespan = "off")
        )
        config = server.config
        config.load()
        server.lifespan = config.lifespan_class(config)
        await server.startup()
        held = await run._hold_ipv6_loopback(_factory(server), server, "127.0.0.1", 16)
        try:
            port = run._final_bound_port(server, 0)
            assert held is not None
            # Nobody else can take [::1]:<port> now.
            with socket.socket(socket.AF_INET6, socket.SOCK_STREAM) as squatter:
                with pytest.raises(OSError):
                    squatter.bind(("::1", port))
            assert b"studio" in await _get("::1", port)
            assert b"studio" in await _get("127.0.0.1", port)
        finally:
            if held is not None:
                held.close()
            await server.shutdown()

    asyncio.run(scenario())


def test_only_a_plain_ipv4_loopback_bind_is_mirrored():
    async def scenario():
        return await run._hold_ipv6_loopback(None, None, "0.0.0.0", 16)

    assert asyncio.run(scenario()) is None
