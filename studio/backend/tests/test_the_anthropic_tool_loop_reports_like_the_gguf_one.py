# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The Anthropic tool loop is armed, so it owes the ledger what the GGUF one gives it.

Both surfaces drive `generate_chat_completion_with_tools` against the same cache, but the
Anthropic copy published neither its round charge nor its tool states. A round grew the
prompt with the tool history and the ledger kept the opening figure, so the watermark fired
late by that whole history; and a tool that may run for the configured 300 seconds left the
holder marked DECODING, which is the one state `await_resume` does not extend its 90 second
stall deadline for.
"""

from __future__ import annotations

import asyncio
import pathlib
import threading
from types import SimpleNamespace

import pytest

from core.inference.llama_preemption import (
    ParticipantState,
    PreemptionController,
)
import routes.inference as inference_route

pytestmark = pytest.mark.usefixtures("preemption_opted_in")


ROUTES = pathlib.Path(inference_route.__file__)


def _controller(key: str = "test://anthropic") -> PreemptionController:
    made = PreemptionController(key)
    made.configure(budget = 8192, kv_unified = True, slots = 4, draft_tokens = 2, batch_tokens = 2048)
    return made


def _backend():
    return SimpleNamespace(base_url = "http://anthropic.test", admission_key = "test://anthropic")


class TestTheRoundChargeReachesTheLedger:
    def test_publishing_re_baselines_the_participant_and_sweeps(self, monkeypatch):
        controller = _controller()
        monkeypatch.setattr(inference_route, "get_preemption_controller", lambda key: controller)
        backend = _backend()
        controller.register("gen", tokens = 1000, prompt_tokens = 800)

        conversation = [
            {"role": "user", "content": "what is in the file"},
            {"role": "assistant", "content": "calling a tool"},
            {"role": "tool", "content": "x" * 4000},
        ]
        payload = SimpleNamespace()
        expected_prompt = inference_route._openai_llama_admission_charged_prompt_tokens(
            payload,
            conversation = conversation,
            image_tokens = inference_route._openai_llama_admission_image_tokens(backend),
            injected_tools = None,
        )

        swept = []
        inference_route._openai_llama_publish_round_charge(
            llama_backend = backend,
            gen_id = "gen",
            reservation = SimpleNamespace(lease_nowait = lambda: SimpleNamespace(tokens = 6000)),
            payload = payload,
            conversation = conversation,
            rendered_tools = None,
            observe_tokens = swept.append,
        )

        held = controller.participant("gen")
        assert held.base_tokens == 6000, "the round's charge is what admission now holds"
        assert held.prompt_tokens == expected_prompt
        assert held.prompt_tokens > 800, "the tool history grew the prompt the cache holds"
        # The sweep runs on the new figure; recording it and not sweeping is what let three
        # chats prefill past the cache together.
        assert swept == [0]

    def test_a_reservation_without_a_lease_publishes_nothing(self, monkeypatch):
        controller = _controller()
        monkeypatch.setattr(inference_route, "get_preemption_controller", lambda key: controller)
        controller.register("gen", tokens = 1000, prompt_tokens = 800)
        swept = []
        inference_route._openai_llama_publish_round_charge(
            llama_backend = _backend(),
            gen_id = "gen",
            reservation = SimpleNamespace(lease_nowait = lambda: None),
            payload = SimpleNamespace(),
            conversation = [{"role": "user", "content": "hi"}],
            rendered_tools = None,
            observe_tokens = swept.append,
        )
        assert swept == []
        assert controller.participant("gen").base_tokens == 1000


class TestBothRecostsPublish:
    """The two closures live in different route handlers; only the helper is shared."""

    @pytest.mark.parametrize(
        ("opener", "closer"),
        [
            (
                "def _gguf_recost(conversation, round_tools = None)",
                "# Active tool names gating the bare-rehearsal strip",
            ),
            (
                "def _anthropic_recost(conversation, round_tools = None)",
                "async def _admitted_anthropic(",
            ),
        ],
    )
    def test_the_recost_runs_before_the_publish(self, opener, closer):
        source = inference_route.__file__
        with open(source, encoding = "utf-8") as handle:
            text = handle.read()
        body = text[text.index(opener) :]
        body = body[: body.index(closer)]
        recost = body.index("_openai_llama_admission_recost(")
        publish = body.index("_openai_llama_publish_round_charge(")
        assert recost < publish, (
            "the publish reads `lease.tokens` and must read it after the re-cost that grew "
            "it for this round, not before"
        )


def _drive_tool_stream(events, note_state, **kwargs):
    """Run `_anthropic_tool_stream` over canned events and return its SSE chunks.

    ``events`` is a factory, not a list: the state a test reads between two events has to
    be read while the stream is consuming them, not while they are being built.
    """

    def run_gen():
        def gen():
            yield from events()

        return gen()

    async def _drive():
        async def _is_disconnected():
            return False

        response = await inference_route._anthropic_tool_stream(
            SimpleNamespace(is_disconnected = _is_disconnected),
            threading.Event(),
            run_gen,
            "msg_state",
            "m",
            note_state = note_state,
            **kwargs,
        )
        return [chunk async for chunk in response.body_iterator]

    return asyncio.run(_drive())


class TestTheToolStatesReachTheController:
    def _observer(self, monkeypatch, controller):
        monkeypatch.setattr(inference_route, "get_preemption_controller", lambda key: controller)
        _refresh, _observe, note_state = inference_route._openai_llama_residency_observer(
            llama_backend = _backend(), completion_id = "msg_state"
        )
        return note_state

    def test_a_running_tool_is_reported_as_tools_running(self, monkeypatch):
        controller = _controller()
        note_state = self._observer(monkeypatch, controller)
        held = controller.register("msg_state", tokens = 1000)

        seen = []

        def _gen_events():
            yield {
                "type": "tool_start",
                "tool_name": "python",
                "tool_call_id": "c0",
                "arguments": {},
            }
            # Where the tool actually runs: the holder must not read as DECODING here, or a
            # chat paused behind it gives up after 90 seconds of a working backend.
            seen.append(held.state)
            yield {
                "type": "tool_end",
                "tool_name": "python",
                "tool_call_id": "c0",
                "result": "done",
            }
            yield {"type": "content", "text": "the answer"}

        _drive_tool_stream(_gen_events, note_state)

        assert seen == [ParticipantState.TOOLS_RUNNING]
        assert controller.snapshot().tools_running == 0, "the answer ended the tool"
        assert held.state == ParticipantState.DECODING

    def test_an_approval_parks_the_holder_until_it_is_answered(self, monkeypatch):
        controller = _controller()
        note_state = self._observer(monkeypatch, controller)
        held = controller.register("msg_state", tokens = 1000)

        seen = []

        def _gen_events():
            yield {
                "type": "tool_start",
                "tool_name": "python",
                "tool_call_id": "c0",
                "arguments": {},
                "awaiting_confirmation": True,
            }
            seen.append(held.state)
            yield {
                "type": "tool_end",
                "tool_name": "python",
                "tool_call_id": "c0",
                "result": "done",
            }
            seen.append(held.state)
            yield {"type": "content", "text": "the answer"}

        _drive_tool_stream(_gen_events, note_state)

        # Parked while the user is asked, then running once they answer: a parked holder is
        # the sweep's first victim, since it holds cells and consumes no compute.
        assert seen == [ParticipantState.PARKED_ON_TOOL, ParticipantState.TOOLS_RUNNING]
        assert held.state == ParticipantState.DECODING

    def test_streamed_tool_arguments_are_decoding(self, monkeypatch):
        """`tool_args` are decoded tokens; a holder left TOOLS_RUNNING through them is out
        of `_PREEMPTABLE` while it grows, so nothing can be paused and the pool overflows.
        """
        controller = _controller()
        note_state = self._observer(monkeypatch, controller)
        held = controller.register("msg_state", tokens = 1000)

        seen = []

        def _gen_events():
            yield {
                "type": "tool_start",
                "tool_name": "python",
                "tool_call_id": "c0",
                "arguments": {},
            }
            yield {
                "type": "tool_args",
                "tool_name": "python",
                "tool_call_id": "c1",
                "text": '{"path":',
            }
            seen.append(held.state)
            yield {"type": "content", "text": "done"}

        _drive_tool_stream(_gen_events, note_state)

        assert seen == [ParticipantState.DECODING]
        assert held.preemptable is True

    def test_a_dropped_parallel_call_still_reports_its_tool(self, monkeypatch):
        """`disable_parallel_tool_use` drops the second call from the wire, but the tool
        still runs server-side and still holds its cells.
        """
        controller = _controller()
        note_state = self._observer(monkeypatch, controller)
        held = controller.register("msg_state", tokens = 1000)

        seen = []

        def _gen_events():
            yield {
                "type": "tool_start",
                "tool_name": "python",
                "tool_call_id": "c0",
                "arguments": {},
            }
            yield {"type": "tool_end", "tool_name": "python", "tool_call_id": "c0", "result": "r1"}
            yield {
                "type": "tool_start",
                "tool_name": "python",
                "tool_call_id": "c1",
                "arguments": {},
            }
            seen.append(held.state)
            yield {"type": "tool_end", "tool_name": "python", "tool_call_id": "c1", "result": "r2"}
            yield {"type": "content", "text": "the answer"}

        _drive_tool_stream(_gen_events, note_state, disable_parallel_tool_use = True)

        assert seen == [ParticipantState.TOOLS_RUNNING]

    def test_the_stream_runs_unchanged_without_a_reporter(self):
        """Every other caller passes nothing, and a missing reporter is not an error."""
        chunks = _drive_tool_stream(
            lambda: [
                {
                    "type": "tool_start",
                    "tool_name": "python",
                    "tool_call_id": "c0",
                    "arguments": {},
                },
                {"type": "tool_end", "tool_name": "python", "tool_call_id": "c0", "result": "done"},
                {"type": "content", "text": "the answer"},
            ],
            None,
        )
        assert any("the answer" in chunk for chunk in chunks)


class TestTheAnthropicToolLoopIsHandedTheReporter:
    def test_the_streaming_branch_passes_note_state(self):
        with open(inference_route.__file__, encoding = "utf-8") as handle:
            text = handle.read()
        call = text[text.index("_anthropic_tool_stream(\n                    request,") :]
        call = call[: call.index("tool_loop = True,")]
        assert "note_state = _anthropic_note_state," in call, (
            "the observer's third callback is the only way the controller hears about a "
            "tool that may run for the whole 300 second timeout"
        )


class TestTheFinalPassIsNotChargedForACatalogueItDoesNotSend:
    """`on_conversation_grew(messages, None)`: the synthesis pass renders no tools."""

    def test_the_resident_figure_drops_the_catalogue_on_the_final_pass(self, monkeypatch):
        controller = _controller("test://rendered")
        monkeypatch.setattr(inference_route, "get_preemption_controller", lambda key: controller)
        backend = _backend()
        catalogue = [
            {"type": "function", "function": {"name": f"t{i}", "description": "x" * 400}}
            for i in range(4)
        ]
        conversation = [{"role": "user", "content": "hi"}]

        def _publish(rendered):
            controller._participants.clear()
            controller.register("gen", tokens = 1000, prompt_tokens = 800)
            inference_route._openai_llama_publish_round_charge(
                llama_backend = backend,
                gen_id = "gen",
                reservation = SimpleNamespace(lease_nowait = lambda: SimpleNamespace(tokens = 20000)),
                payload = SimpleNamespace(),
                conversation = conversation,
                rendered_tools = rendered,
                observe_tokens = lambda _n: None,
            )
            return controller.participant("gen").prompt_tokens

        with_tools = _publish(catalogue)
        without = _publish(None)
        assert with_tools > without, "the catalogue has to be worth measuring here"
        # A round that sends none must not carry them as cells the cache holds: on a small
        # context that difference is enough to preempt a healthy chat.
        assert without < 100

    @pytest.mark.parametrize(
        ("opener", "closer"),
        [
            (
                "def _gguf_recost(conversation, round_tools = None)",
                "# Active tool names gating the bare-rehearsal strip",
            ),
            (
                "def _anthropic_recost(conversation, round_tools = None)",
                "async def _admitted_anthropic(",
            ),
        ],
    )
    def test_both_recosts_publish_the_rounds_own_tools(self, opener, closer):
        text = ROUTES.read_text(encoding = "utf-8")
        body = text[text.index(opener) :]
        body = body[: body.index(closer)]
        assert "rendered_tools = round_tools," in body, (
            "the publish must use what the round SENDS; the lease keeps charging the "
            "whole catalogue, which is a separate and deliberate thing"
        )


class TestEveryDrainReportsItsToolStates:
    @pytest.mark.parametrize(
        ("opener", "closer"),
        [
            # The GGUF streaming loop and its non-streaming drain.
            (
                "async def produce_openai_chat_completions(",
                "\ndef _openai_messages_for_passthrough",
            ),
            # The Anthropic streaming loop.
            ("async def _anthropic_tool_stream(", "\nasync def _anthropic_plain_stream("),
            # The Anthropic non-streaming drain.
            ("def _collect_anthropic_events(", "\ndef _anthropic_tool_response_from_events("),
        ],
    )
    def test_the_shared_reader_is_called(self, opener, closer):
        text = ROUTES.read_text(encoding = "utf-8")
        body = text[text.index(opener) :]
        body = body[: body.index(closer)]
        assert "_note_tool_loop_state(" in body, (
            "an armed drain that reports no state leaves the participant DECODING through "
            "a tool that may run for the whole 300 second timeout"
        )

    def test_the_reader_is_the_only_implementation(self):
        text = ROUTES.read_text(encoding = "utf-8")
        # Written out at a call site, a later fix lands in one copy and not the others,
        # which is how the non-streaming drains came to be three rounds behind.
        assert text.count("ParticipantState.PARKED_ON_TOOL\n") <= 2, (
            "the parked/tools-running transition is written out somewhere other than "
            "_note_tool_loop_state"
        )


class TestAPausedAnthropicStreamKeepsTalking:
    def test_the_pause_events_reach_the_wire(self):
        chunks = _drive_tool_stream(
            lambda: [
                {"type": "preempt", "state": "paused"},
                {"type": "preempt", "state": "keepalive"},
                {"type": "preempt", "state": "keepalive"},
                {"type": "preempt", "state": "resumed"},
                {"type": "content", "text": "the answer"},
            ],
            None,
        )
        # Dropped, the connection is silent for the whole pause: the wait's two-second
        # keepalive completes next(gen) before the stall timer can fire, so an
        # intermediary that drops an idle connection at ~100s cancels a resumable answer.
        assert inference_route._OPENAI_PREEMPT_SSE_PAUSED in chunks, "the pause was not forwarded"
        assert chunks.count(inference_route._OPENAI_PREEMPT_SSE_KEEPALIVE) == 2
        assert inference_route._OPENAI_PREEMPT_SSE_RESUMED in chunks
        assert any("the answer" in chunk for chunk in chunks)

    def test_the_plain_stream_forwards_them_too(self):
        text = ROUTES.read_text(encoding = "utf-8")
        body = text[text.index("async def _anthropic_plain_stream(") :]
        body = body[: body.index("\nasync def _anthropic_plain_non_streaming(")]
        assert (
            "_OPENAI_PREEMPT_SSE_BY_STATE" in body
        ), "the no-tool Anthropic stream pauses too, and its emitter ignores the event"
