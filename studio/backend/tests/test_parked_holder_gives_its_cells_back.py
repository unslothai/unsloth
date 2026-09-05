# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A chat stopped on a tool approval must not keep waiting chats out of an empty cache.

WHAT WENT WRONG

Four tool-enabled API chats on the 35B model at -c 8192, 2026-09-05. The leader parsed a
tool call that needed confirmation and stopped. The preemptor, seeing two chats waiting
for room, erased the leader's idle slot ("reclaimed-idle-early: freed=4676"), which is
right: its cells are the cheapest room there is. Then nothing happened for three minutes.
`resident=0` -- the cache was EMPTY -- while `committed=5931`, because the leader was
still DECODING in the ledger with 3847 tokens charged, and a waiter wanting 3092 could
not fit under a 6136 ceiling beside a charge for cells that no longer existed. Both
waiters gave up at the 90 second stall bound ("nothing decoding" was, after all, true)
and ended their turns with nothing.

The states for this, PARKED_ON_TOOL and TOOLS_RUNNING, had been defined with the
controller and never set by anything.

THE RULES THIS PINS

1. A holder reported parked or in a tool stops counting once an idle-slot reclaim has
   erased its cells, and counts again when it decodes.
2. Its admission lease hands its commitment back the same way, so the pool ledger agrees.
3. The tool-loop route reports the transitions: approval prompt -> parked, answered ->
   tools running, first content of the next round -> decoding.
4. A waiter's stall clock treats a holder in a tool as work in progress.
"""

from __future__ import annotations

import asyncio
import json

import pytest
from fastapi import FastAPI

from auth.authentication import get_current_subject
from core.inference import llama_admission
from core.inference.llama_admission import LlamaAdmissionConfig
from core.inference.llama_preemption import (
    ParticipantState,
    PreemptionController,
    get_preemption_controller,
)
import routes.inference as inference_route

from .asgi_stream_helpers import wait_for_frame
from .llama_backend_double import FakeLlamaCppBackend


@pytest.fixture(autouse = True)
def _fresh_queues():
    llama_admission.reset_llama_admission_queues()
    yield
    llama_admission.reset_llama_admission_queues()


def _controller() -> PreemptionController:
    made = PreemptionController("test://parked")
    # 8192 cells, the buffer this configuration produces is well under 2100, so a
    # 6000-plus ceiling: the shape of the run that failed.
    made.configure(budget = 8192, kv_unified = True, slots = 4, draft_tokens = 2, batch_tokens = 2048)
    return made


class TestTheLedger:
    def test_a_parked_holder_whose_cells_were_reclaimed_stops_counting(self):
        c = _controller()
        leader = c.register("leader", tokens = 3847)
        c.note_tokens("leader", 3847)
        waiter = c.register("waiter", tokens = 3092, state = ParticipantState.PAUSED)
        assert waiter.state == ParticipantState.PAUSED
        c.note_resident(4676, 0)

        # As it stood: the leader decoding, no room for the waiter beside it.
        assert c.try_grant_resume("waiter", 3092) is False

        # The leader stops on an approval prompt, and its idle slot is erased.
        assert c.note_state("leader", ParticipantState.PARKED_ON_TOOL) is True
        assert c.snapshot().parked == 1
        assert c.note_cells_reclaimed() == 1
        c.note_resident(0, 0)

        assert leader.holds_kv is False
        assert c.snapshot().committed == 0, (
            "the cache is empty and the ledger must say so; charging cells that were "
            "erased is what kept two waiters out for three minutes"
        )
        assert c.try_grant_resume("waiter", 3092) is True

    def test_the_charge_comes_back_when_the_holder_decodes_again(self):
        c = _controller()
        leader = c.register("leader", tokens = 3847)
        c.note_tokens("leader", 3847)
        c.note_state("leader", ParticipantState.TOOLS_RUNNING)
        c.note_cells_reclaimed()
        assert leader.holds_kv is False

        # Back at the model: llama-server prefills the prompt in before the next token.
        c.note_state("leader", ParticipantState.DECODING)
        assert leader.holds_kv is True
        assert leader.tokens == 3847

    def test_a_reclaim_leaves_decoding_holders_alone(self):
        c = _controller()
        a = c.register("a", tokens = 1000)
        c.note_tokens("a", 1000)
        assert c.note_cells_reclaimed() == 0
        assert a.holds_kv is True

    def test_only_live_states_move(self):
        c = _controller()
        c.register("p", tokens = 10, state = ParticipantState.PAUSED)
        assert c.note_state("p", ParticipantState.TOOLS_RUNNING) is False
        c.register("q", tokens = 10)
        assert (
            c.note_state("q", ParticipantState.PAUSED) is False
        ), "PAUSED is the preemptor's to set"
        assert c.note_state("q", ParticipantState.TOOLS_RUNNING) is True
        assert c.note_state("q", ParticipantState.TOOLS_RUNNING) is False, "no change, no report"

    def test_a_parked_winner_gives_up_its_epoch(self):
        c = _controller()
        c.register("w", tokens = 5000)
        c.note_tokens("w", 5000)
        c.register("v", tokens = 3000)
        c.note_tokens("v", 3000)
        # Crowned as the leader of the current epoch.
        c._epoch_winner = "w"
        assert c.snapshot().winner == "w"
        c.note_state("w", ParticipantState.PARKED_ON_TOOL)
        assert c.snapshot().winner != "w"

    def test_the_reclaim_also_hands_back_the_admission_commitment(self):
        class _Lease:
            def __init__(self):
                self.yielded = 0

            def yield_parked_commitment(self):
                self.yielded += 1
                return 3847

        lease = _Lease()
        c = _controller()
        c.register("leader", tokens = 3847, lease = lease)
        c.note_state("leader", ParticipantState.PARKED_ON_TOOL)
        assert c.note_cells_reclaimed() == 1
        assert lease.yielded == 1
        assert c.note_cells_reclaimed() == 0, "once per park, not once per sweep"
        assert lease.yielded == 1


class TestTheLease:
    @pytest.mark.asyncio
    async def test_yield_parked_commitment_frees_the_pool_and_recost_takes_it_back(self):
        # park() wants a running loop for the waiters it may wake.
        queue = llama_admission.get_llama_admission_queue("http://lease.test")
        reservation = queue.reserve(
            capacity = 4, config = LlamaAdmissionConfig(), tokens = 3847, budget = 8192
        )
        lease = reservation.lease_nowait()
        assert lease is not None
        assert queue.committed_now() == 3847
        assert lease.park() is True

        assert lease.yield_parked_commitment() == 3847
        assert queue.committed_now() == 0
        assert lease.yield_parked_commitment() == 0, "nothing left to hand back"

        # Somebody else can take the room now.
        other = queue.reserve(
            capacity = 4, config = LlamaAdmissionConfig(), tokens = 3092, budget = 8192
        ).lease_nowait()
        assert other is not None
        assert queue.committed_now() == 3092

        # The parked chat's next round re-states its size and is charged again.
        assert lease.recost(3847) is True
        assert queue.committed_now() == 3092 + 3847
        lease.release()
        other.release()
        assert queue.committed_now() == 0


class _ApprovalToolBackend(FakeLlamaCppBackend):
    base_url = "http://llama.test"
    effective_parallel_slots = 2
    supports_tools = True
    context_length = 8192

    def _maybe_recover_from_mtp_crash(self, exc):
        return None

    def generate_chat_completion_with_tools(self, **kwargs):
        yield {"type": "content", "text": "Let me check."}
        yield {
            "type": "tool_start",
            "tool_call_id": "call_1",
            "name": "write_file",
            "arguments": {},
            "awaiting_confirmation": True,
        }
        yield {
            "type": "tool_result",
            "tool_call_id": "call_1",
            "name": "write_file",
            "result": "ok",
        }
        yield {"type": "content", "text": "Let me check. Done."}
        yield {
            "type": "metadata",
            "usage": {"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7},
            "timings": {"prompt_n": 3, "predicted_n": 4},
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
        ],
        "client": ("127.0.0.1", 12345),
        "server": ("testserver", 80),
        "app": app,
    }


class TestTheRouteReports:
    def test_the_approval_prompt_and_the_next_round_are_reported(self, monkeypatch):
        monkeypatch.setattr(
            inference_route, "get_llama_cpp_backend", lambda: _ApprovalToolBackend()
        )
        monkeypatch.setattr(inference_route, "_effective_enable_tools", lambda payload: True)

        async def _fake_select(payload, **_kwargs):
            return [{"type": "function", "function": {"name": "write_file"}}]

        monkeypatch.setattr(inference_route, "_select_request_tools", _fake_select)

        reported: list[str] = []
        real_note_state = PreemptionController.note_state

        def _recording(self, gen_id, state):
            reported.append(state)
            return real_note_state(self, gen_id, state)

        monkeypatch.setattr(PreemptionController, "note_state", _recording)

        app = FastAPI()
        app.include_router(inference_route.router)
        app.dependency_overrides[get_current_subject] = lambda: "test-user"

        async def _drive():
            body = json.dumps(
                {
                    "messages": [{"role": "user", "content": "save it"}],
                    "stream": True,
                    "enable_tools": True,
                }
            ).encode()
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
                    if chunk == inference_route._SSE_DONE_CHUNK:
                        done.set()

            task = asyncio.create_task(app(_scope(app, body), receive, send))
            try:
                await wait_for_frame(done, task, what = "the [DONE] frame")
            finally:
                task.cancel()
                await asyncio.gather(task, return_exceptions = True)
            return "".join(frames)

        body = asyncio.run(_drive())
        assert "Done." in body
        assert reported[:3] == [
            ParticipantState.PARKED_ON_TOOL,
            ParticipantState.TOOLS_RUNNING,
            ParticipantState.DECODING,
        ], reported
        # And the chat is gone from the ledger when it ends.
        assert get_preemption_controller("http://llama.test").snapshot().parked == 0
