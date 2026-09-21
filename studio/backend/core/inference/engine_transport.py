# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Bounded async HTTP bridge for the orchestrator's synchronous generator API."""

import asyncio
import json
import queue
import threading

import httpx

from .http_stream import closing_response_lines


def stream_chat_events(base_url, headers, payload, cancelled):
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
        async with httpx.AsyncClient(
            trust_env = False, timeout = httpx.Timeout(120, connect = 5)
        ) as client:
            async with client.stream(
                "POST", base_url + "/v1/chat/completions", headers = headers, json = payload
            ) as response:
                if response.status_code != 200:
                    raise RuntimeError(
                        f"Engine rejected generation (HTTP {response.status_code}). Try a smaller context or token limit."
                    )
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
