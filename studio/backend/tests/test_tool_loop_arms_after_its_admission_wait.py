# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A tool-loop chat that queued at admission is armed once it is granted."""

from __future__ import annotations

import asyncio
import json

import pytest
from fastapi import FastAPI

from auth.authentication import get_current_subject
from core.inference import llama_admission
import routes.inference as inference_route

from .asgi_stream_helpers import wait_for_frame
from .llama_backend_double import FakeLlamaCppBackend


@pytest.fixture(autouse = True)
def _fresh_queues():
    llama_admission.reset_llama_admission_queues()
    yield
    llama_admission.reset_llama_admission_queues()


_ONE_SLOT = llama_admission.LlamaAdmissionConfig(max_queue = 4)


class _OneSlotToolBackend(FakeLlamaCppBackend):
    base_url = "http://llama.test"
    effective_parallel_slots = 1
    supports_tools = True
    context_length = 8192

    def generate_chat_completion_with_tools(self, **kwargs):
        yield {"type": "content", "text": "hi"}
        yield {
            "type": "metadata",
            "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
            "timings": {"prompt_n": 3, "predicted_n": 1},
            "finish_reason": "stop",
        }


def _scope(app, body: bytes) -> dict:
    return {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/chat/completions",
        "raw_path": b"/chat/completions",
        "query_string": b"",
        "root_path": "",
        "headers": [
            (b"host", b"testserver"),
            (b"content-type", b"application/json"),
            (b"content-length", str(len(body)).encode()),
            # The Studio UI's opt-in: tools with confirmation are refused without it.
            (b"x-unsloth-events", b"1"),
        ],
        "client": ("127.0.0.1", 12345),
        "server": ("testserver", 80),
        "app": app,
    }


def test_a_queued_tool_chat_is_armed_once_granted(monkeypatch):
    monkeypatch.setattr(inference_route, "get_llama_cpp_backend", lambda: _OneSlotToolBackend())
    monkeypatch.setattr(inference_route, "_effective_enable_tools", lambda payload: True)

    async def _fake_select(payload, **_kwargs):
        return [{"type": "function", "function": {"name": "python"}}]

    monkeypatch.setattr(inference_route, "_select_request_tools", _fake_select)

    arms: list[tuple[str, bool]] = []
    real_arm = inference_route._openai_llama_preemption_arm

    def _recording_arm(
        *,
        request,
        llama_backend,
        reservation,
        gen_id,
        signal,
        loop = None,
    ):
        held = reservation is not None and reservation.lease_nowait() is not None
        arms.append((gen_id, held))
        return real_arm(
            request = request,
            llama_backend = llama_backend,
            reservation = reservation,
            gen_id = gen_id,
            signal = signal,
            loop = loop,
        )

    monkeypatch.setattr(inference_route, "_openai_llama_preemption_arm", _recording_arm)

    app = FastAPI()
    app.include_router(inference_route.router)
    app.dependency_overrides[get_current_subject] = lambda: "test-user"

    async def _drive():
        # Somebody else holds the only slot, so the request below must queue.
        queue = llama_admission.get_llama_admission_queue("http://llama.test")
        holder = queue.reserve(capacity = 1, config = _ONE_SLOT).lease_nowait()
        assert holder is not None

        body = json.dumps(
            {
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
                "enable_tools": True,
            }
        ).encode()
        queued = asyncio.Event()
        done = asyncio.Event()
        frames: list[str] = []

        async def receive():
            if not frames:
                return {"type": "http.request", "body": body, "more_body": False}
            await asyncio.Event().wait()

        async def send(message):
            if message.get("type") == "http.response.body":
                chunk = message.get("body", b"").decode()
                frames.append(chunk)
                if "admission-wait" in chunk:
                    queued.set()
                if chunk == inference_route._SSE_DONE_CHUNK:
                    done.set()

        task = asyncio.create_task(app(_scope(app, body), receive, send))
        try:
            await wait_for_frame(queued, task, what = "the admission-wait comment")
            assert (
                arms and not arms[0][1]
            ), "the arm beside the reservation should have found no lease yet"
            holder.release()
            await wait_for_frame(done, task, what = "the [DONE] frame")
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions = True)

    asyncio.run(_drive())
    assert any(held for _gen, held in arms), (
        "the chat was granted after waiting and never armed with its lease: it decodes "
        "outside the preemptor's ledger, which is the three-of-four-dead case"
    )
