# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import asyncio

from mcp_server import BearerTokenMiddleware, _dump, create_studio_mcp


def test_studio_mcp_registers_control_plane_tools():
    tools = asyncio.run(create_studio_mcp().list_tools())

    assert {tool.name for tool in tools} == {
        "studio_status",
        "list_local_models",
        "get_training_status",
        "start_training",
        "stop_training",
        "list_training_runs",
        "validate_recipe",
        "get_recipe_job_status",
        "get_recipe_job_dataset",
        "load_checkpoint",
        "export_gguf",
    }


def test_dump_serializes_pydantic_values():
    class Response:
        def model_dump(self, *, mode):
            assert mode == "json"
            return {"ok": True}

    assert _dump(Response()) == {"ok": True}
    assert _dump({"already": "json"}) == {"already": "json"}


def test_bearer_token_middleware_rejects_wrong_token():
    events = []

    async def app(scope, receive, send):
        events.append("app")

    async def send(message):
        events.append(message)

    middleware = BearerTokenMiddleware(app, "secret")
    asyncio.run(
        middleware(
            {"type": "http", "headers": [(b"authorization", b"Bearer wrong")]},
            None,
            send,
        )
    )

    assert events[0]["status"] == 401
    assert "app" not in events


def test_bearer_token_middleware_closes_unauthorized_websocket():
    events = []

    async def app(scope, receive, send):
        events.append("app")

    async def send(message):
        events.append(message)

    middleware = BearerTokenMiddleware(app, "secret")
    asyncio.run(
        middleware(
            {"type": "websocket", "headers": []},
            None,
            send,
        )
    )

    assert events == [{"type": "websocket.close", "code": 4401}]