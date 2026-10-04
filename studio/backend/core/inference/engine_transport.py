# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Bounded async HTTP bridge for the orchestrator's synchronous generator API."""

import asyncio
import json
import queue
import re
import threading

import httpx

from .http_stream import closing_response_lines


class EngineHTTPError(RuntimeError):
    """Keep a native rejection distinct from a transport or worker failure."""

    def __init__(self, status_code, body):
        self.status_code = status_code
        self.body = body[:500]
        super().__init__(f"Engine rejected generation (HTTP {status_code}): {self.body}")


_BASE64 = re.compile(r"[A-Za-z0-9+/=\s]+")


def engine_messages(messages):
    """Send engines data-URL images only: SGLang opens a bare "/..." image string as a host file."""
    out = []
    for message in messages:
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, list):
            out.append(message)
            continue
        parts = []
        for part in content:
            image = part.get("image_url") if isinstance(part, dict) else None
            url = image.get("url") if isinstance(image, dict) else image
            if isinstance(url, str) and not url.lstrip().startswith("data:"):
                if not _BASE64.fullmatch(url):
                    raise ValueError("Send images over https or as base64 data URLs.")
                image = image if isinstance(image, dict) else {}
                url = "data:image/png;base64," + "".join(url.split())
                part = {**part, "image_url": {**image, "url": url}}
            parts.append(part)
        out.append({**message, "content": parts})
    return out


def engine_request_timeout():
    """The llama-server first-token deadline: long prefill, queueing or a non-streamed
    reply routinely outlast a fixed short read timeout."""
    from routes.inference import _first_token_timeout_s
    return httpx.Timeout(_first_token_timeout_s(), connect = 5)


def stream_chat_events(base_url, headers, payload, cancelled):
    timeout = engine_request_timeout()
    items = queue.Queue(maxsize = 64)
    done = threading.Event()
    stopped = threading.Event()
    control = {}
    errors = []

    async def consume():
        control["loop"] = asyncio.get_running_loop()
        control["task"] = asyncio.current_task()
        if stopped.is_set() or cancelled():
            return
        async with httpx.AsyncClient(trust_env = False, timeout = timeout) as client:
            async with client.stream(
                "POST", base_url + "/v1/chat/completions", headers = headers, json = payload
            ) as response:
                if response.status_code != 200:
                    await response.aread()
                    raise EngineHTTPError(response.status_code, response.text)
                lines = closing_response_lines(response)
                try:
                    async for line in lines:
                        if not line.startswith("data:"):
                            continue
                        data = line[5:].strip()
                        if data == "[DONE]":
                            break
                        if not data:
                            continue
                        event = json.loads(data)
                        if not isinstance(event, dict):
                            raise RuntimeError("Engine returned an invalid stream event.")
                        while items.full():
                            if stopped.is_set() or cancelled():
                                return
                            await asyncio.sleep(0.02)
                        items.put_nowait(event)
                finally:
                    await lines.aclose()

    def run():
        try:
            asyncio.run(consume())
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            errors.append(exc)
        finally:
            done.set()

    reader = threading.Thread(target = run, daemon = True)
    reader.start()
    try:
        while not done.is_set() or not items.empty():
            if cancelled():
                return
            try:
                yield items.get(timeout = 0.1)
            except queue.Empty:
                continue
        if errors:
            raise errors[0]
    finally:
        stopped.set()
        loop, task = control.get("loop"), control.get("task")
        if loop and task and not loop.is_closed():
            try:
                loop.call_soon_threadsafe(task.cancel)
            except RuntimeError:
                pass
        reader.join(timeout = 3)
