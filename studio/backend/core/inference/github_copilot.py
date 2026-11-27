# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Copilot subscription transport through the official SDK, never private HTTP endpoints."""

import asyncio
import contextlib
import json
import tempfile
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any


BASE_URL = "https://github.com"


@asynccontextmanager
async def sdk_client(token: str):
    if not token:
        raise ValueError("Connect your GitHub Copilot subscription first.")
    try:
        from copilot import CopilotClient
        from copilot.client import RuntimeConnection
    except ImportError:
        raise RuntimeError(
            "Copilot requires Python 3.11+ and microsoft-copilot-sdk==2.0.3. "
            "Install the Studio dependencies and restart Studio."
        ) from None
    # Separate runtime state for every request/account. Never fall back to the host's login,
    # instructions, plugins, or workspace. Empty mode also disables environment discovery.
    try:
        with tempfile.TemporaryDirectory(prefix = "unsloth-copilot-") as directory:
            async with CopilotClient(
                mode = "empty",
                github = {"token": token, "use_logged_in_user": False},
                runtime = RuntimeConnection.for_spawned_stdio(
                    base_directory = directory,
                    working_directory = directory,
                ),
            ) as client:
                yield client
    except Exception:
        # Runtime/transport errors may quote request credentials. Only return a curated error.
        raise RuntimeError(
            "GitHub Copilot is unavailable. Check your subscription, usage allowance and "
            "organization policies, or reconnect GitHub. The Copilot runtime also requires "
            "network access on first use."
        ) from None


async def list_models(token: str) -> list[dict[str, Any]]:
    async with sdk_client(token) as client:
        async with asyncio.timeout(60):
            models = await client.list_models()
            if not models:
                # Some accounts can use Copilot Auto even when the SDK returns no named models.
                # Confirm Auto in the authenticated session without sending a prompt.
                from copilot.rpc import PermissionDecisionReject

                async with await client.create_session(
                    model = "auto",
                    tools = {"available_tools": []},
                    permissions = {"on_request": lambda *_: PermissionDecisionReject()},
                ) as session:
                    catalog = await session.rpc.model.list()
                if catalog.auto is not None:
                    return [{"id": "auto", "name": "Copilot Auto", "owned_by": "github_copilot"}]
        return [
            {"id": model.id, "name": model.name, "owned_by": "github_copilot"}
            for model in models
            if model.policy is None
            or getattr(model.policy.state, "value", model.policy.state) != "disabled"
        ]


def _prompts(messages: list[dict]) -> tuple[str, str]:
    system, conversation = [], []
    for message in messages:
        content = message.get("content") or ""
        if isinstance(content, list):
            if any(part.get("type") != "text" for part in content):
                raise ValueError(
                    "The Copilot subscription connection currently supports text chat."
                )
            content = "\n".join(part.get("text", "") for part in content)
        if message.get("role") in ("system", "developer"):
            system.append(content)
        elif message.get("role") in ("user", "assistant") and not message.get("tool_calls"):
            conversation.append({"role": message["role"], "content": content})
        else:
            raise ValueError("Copilot subscription chat does not yet support Studio tool calls.")
    if not conversation or conversation[-1]["role"] != "user":
        raise ValueError("Copilot requires a final user message.")
    if len(conversation) == 1:
        prompt = conversation[0]["content"]
    else:
        # The SDK owns its conversation engine and accepts prompts, not Chat Completions
        # messages. Recreate a request from Studio's authoritative, editable transcript;
        # replaying old user turns would generate and bill for their answers again.
        prompt = (
            "Continue the following conversation. Reply only to the final user message.\n"
            + json.dumps(conversation, ensure_ascii = False)
        )
    return "\n\n".join(system) or "You are a helpful assistant.", prompt


async def stream_chat(
    token: str,
    messages: list[dict],
    model: str,
    *,
    tools = None,
    enabled_tools = None,
    response_format = None,
):
    if tools or enabled_tools or response_format:
        raise ValueError(
            "Copilot subscription chat currently supports text without tools or JSON mode."
        )
    system, prompt = _prompts(messages)
    identifier = "chatcmpl-" + uuid.uuid4().hex

    def chunk(
        delta: dict,
        finish_reason = None,
        usage = None,
    ):
        result = {
            "id": identifier,
            "object": "chat.completion.chunk",
            "model": model,
            "created": int(time.time()),
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
        }
        if usage:
            result["usage"] = usage
        return "data: " + json.dumps(result) + "\n\n"

    async with sdk_client(token) as client:
        from copilot.rpc import PermissionDecisionReject
        async with await client.create_session(
            model = model,
            events = {"streaming": True},
            system_message = {"mode": "replace", "content": system},
            tools = {"available_tools": []},
            permissions = {"on_request": lambda *_: PermissionDecisionReject()},
        ) as session:
            queue = asyncio.Queue()
            session.on(queue.put_nowait)
            task = asyncio.create_task(session.send_and_wait(prompt, timeout = 300))
            # Wake the reader even if the SDK fails before emitting an event.
            task.add_done_callback(lambda _: queue.put_nowait(None))
            emitted = False
            usage = None
            try:
                yield chunk({"role": "assistant"})
                while True:
                    event = await queue.get()
                    if event is None:
                        await task
                        break
                    if getattr(event, "agent_id", None):
                        continue
                    kind = getattr(event.type, "value", event.type)
                    if kind == "assistant.message_delta":
                        content = event.data.delta_content
                        emitted = emitted or bool(content)
                        yield chunk({"content": content})
                    elif kind == "assistant.message" and not emitted:
                        yield chunk({"content": event.data.content})
                        emitted = True
                    elif kind == "assistant.usage":
                        inputs = int(event.data.input_tokens or 0)
                        outputs = int(event.data.output_tokens or 0)
                        usage = {
                            "prompt_tokens": inputs,
                            "completion_tokens": outputs,
                            "total_tokens": inputs + outputs,
                        }
                    elif kind == "session.error":
                        # SDK error messages can contain request details or credentials.
                        raise RuntimeError(
                            "Copilot could not complete this request. Check your subscription, "
                            "usage allowance and organization policies, or reconnect GitHub."
                        )
                yield chunk({}, "stop", usage)
                yield "data: [DONE]\n\n"
            finally:
                if not task.done():
                    task.cancel()
                    with contextlib.suppress(Exception):
                        await session.abort()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await task
