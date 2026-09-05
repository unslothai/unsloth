# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A chat waiting for KV room is queued, not stuck, for as long as anything is moving.

The live failure, four chats on the 35B at -c 8192 with the GPU shared
(logs/studio_gpu0_swap_20260905_154407.log, outputs/swap_c/four_studio_35b.json):

    llama preemption awaiting-room: gen_id=chatcmpl-ef6143032791 want=2032
    llama preemption gave-up: gen_id=chatcmpl-ef6143032791 want=2032 (no progress for 90.0s)

Across those 90 seconds the other three chats decoded to completion, so the backend was
never stuck; what was frozen was the thing the waiter looked at. `progress_signature` was
`(committed, holders)`, and `committed` is `max(resident, measured) + pending`: while
llama-server's resident reading is the larger of the two, every token the ledger adds is
invisible. The same log shows resident 3254 against measured 2343 a second after the
give-up, so the maximum was pinned on a figure the decoders were not moving.

The chat had generated nothing yet, so its client got `tokens: 0, chars: 0, error: None`
and the GUI a blank turn. The product rule is that an evicted chat waits however long the
others need; giving up is for a genuine hang, which is now defined as nothing moving at
all: no token anywhere, no tool call, no room returned, no holder leaving.
"""

import threading
import time

import pytest

from core.inference.llama_preemption import (
    DEFAULT_RESUME_WAIT_TIMEOUT_S,
    MAX_RESUME_WAIT_MULTIPLE,
    ControllerPreemptionPolicy,
    ParticipantState,
    PreemptSignal,
    PreemptionController,
    reset_preemption_controllers,
)


@pytest.fixture(autouse = True)
def _clean_registry():
    reset_preemption_controllers()
    yield
    reset_preemption_controllers()


# The blocker's cells as llama-server reports them. Two things ride on this figure: it is
# far above the ledger's own sum, which is what pins `committed` at a number the decoders
# do not move, and it is past the ceiling, which is what keeps the waiter refused. Both
# are the live shape rather than a contrivance -- a full cache with other chats decoding
# in it is exactly when a chat is evicted and made to wait.
_BUDGET = 16384
_RESIDENT = _BUDGET
_WANT = 4000


def _pinned_controller(key: str) -> PreemptionController:
    """One holder whose resident figure hides every token it goes on to generate."""
    controller = PreemptionController(key)
    controller.configure(budget = _BUDGET, kv_unified = True)
    controller.register("holder", tokens = 1000, signal = PreemptSignal())
    controller.note_tokens("holder", 1000)
    controller.note_resident(_RESIDENT)
    return controller


def _waiting_policy(controller: PreemptionController, gen_id: str, tokens: int):
    """A policy that reaches the wait loop, with its participant PAUSED as a real one is.

    `await_resume` short-circuits before the loop without a lease and without an event
    loop to take it back on, so both are stubbed.
    """
    import asyncio

    class _Lease:
        is_released = False
        tokens = 0

        async def resume_async(self, want, **kwargs):
            return True

    loop = asyncio.new_event_loop()
    thread = threading.Thread(target = loop.run_forever, daemon = True)
    thread.start()
    controller.register(gen_id, lease = _Lease(), tokens = tokens, signal = PreemptSignal())
    controller.set_state(gen_id, ParticipantState.PAUSED)
    policy = ControllerPreemptionPolicy(controller, gen_id, PreemptSignal(), loop = loop)
    return policy, loop


def _shutdown(loop):
    loop.call_soon_threadsafe(loop.stop)


class TestTheSignatureSeesADecodingBackend:
    def test_a_generated_token_moves_the_signature_though_committed_does_not(self):
        """The exact blindness that abandoned a live turn, in three lines."""

        controller = _pinned_controller("pinned")
        before = controller.progress_signature()
        controller.observe("holder", 512)
        after = controller.progress_signature()

        # The two terms the signature used to consist of, both unmoved: the maximum is
        # still the resident reading, and the same holder is still there.
        assert after[0] == before[0] == _RESIDENT, "committed must be pinned for this test"
        assert after[1] == before[1], "and the same holder must still hold"
        # And yet the backend plainly decoded 512 tokens.
        assert after != before, (
            "a signature that cannot see 512 generated tokens is the one that reported "
            "'no progress for 90.0s' about three chats decoding at full rate"
        )
        assert after[2] > before[2], "the token total is the term that moves"

    def test_a_resumed_attempt_restarting_its_count_is_not_read_as_lost_tokens(self):
        """`observe` is given a per-attempt cumulative count, and a resume starts a new
        one at zero. A fall is a new attempt, never tokens being taken back, so the total
        must not go backwards or a resume would look like a stall in reverse."""

        controller = _pinned_controller("restarts")
        controller.observe("holder", 512)
        mid = controller.progress_signature()[2]
        controller.observe("holder", 32)  # a fresh attempt, first report
        after = controller.progress_signature()[2]
        assert after > mid, "the fresh attempt's own tokens still count as progress"

    def test_a_tool_call_starting_moves_the_signature(self):
        """A holder that stops decoding to run a tool moves nothing else. It is still
        work, and the count of holders inside one is in the signature for that reason."""

        controller = _pinned_controller("tools")
        before = controller.progress_signature()
        assert controller.note_state("holder", ParticipantState.TOOLS_RUNNING) is True
        after = controller.progress_signature()
        assert after[3] == before[3] + 1
        assert after != before

    def test_a_round_boundary_that_folds_a_tool_result_in_counts(self):
        controller = _pinned_controller("round-boundary")
        before = controller.progress_signature()[2]
        controller.note_tokens("holder", 1400)
        assert controller.progress_signature()[2] > before


class TestTheWaiterHoldsOnWhileAnythingMoves:
    def test_it_is_still_waiting_after_the_timeout_and_resumes_when_room_appears(self):
        """Committed never falls, no holder ever leaves, and tokens keep arriving.

        Under `(committed, holders)` this waiter gives up at `timeout` with an empty
        turn. It must instead outlast the timeout several times over and then take the
        room the moment it is real.
        """

        timeout = 0.4
        controller = _pinned_controller("still-decoding")
        policy, loop = _waiting_policy(controller, "waiter", _WANT)
        decoded = {"tokens": 0}
        stop = threading.Event()

        def decode_then_free():
            # Long enough that a wall-clock or committed-only bound has certainly fired.
            deadline = time.monotonic() + timeout * 3
            generated = 0
            while time.monotonic() < deadline and not stop.is_set():
                time.sleep(0.02)
                generated += 32
                controller.observe("holder", generated)
                decoded["tokens"] = generated
            # The blocker finishes: its cells come back and the waiter fits at last.
            controller.note_resident(1000)
            controller.note_tokens("holder", 1000)

        worker = threading.Thread(target = decode_then_free, daemon = True)
        worker.start()
        try:
            started = time.monotonic()
            resumed = policy.await_resume(timeout = timeout)
            elapsed = time.monotonic() - started
        finally:
            stop.set()
            worker.join(timeout = 5)
            _shutdown(loop)

        assert decoded["tokens"] > 0, "the fixture never decoded; the test proves nothing"
        assert controller.committed_tokens() <= _RESIDENT, (
            "committed must never have risen above the pinned figure, or the waiter "
            "could have been kept alive by the wrong term"
        )
        assert elapsed > timeout * 2, (
            f"gave up after {elapsed}s of a backend generating tokens throughout; a chat "
            "queued behind live answers waits, however long that takes"
        )
        assert resumed is True, "and it takes the room once the room is real"

    def test_a_backend_where_nothing_moves_at_all_still_gives_up(self):
        """The hang this bound exists for, unchanged: a full cache nobody is draining.

        No token, no tool, no room returned, no holder leaving. The turn finishes with
        what it has rather than waiting on a backend that has stopped.
        """

        timeout = 0.5
        controller = _pinned_controller("frozen")
        policy, loop = _waiting_policy(controller, "waiter", _WANT)
        try:
            started = time.monotonic()
            resumed = policy.await_resume(timeout = timeout)
            elapsed = time.monotonic() - started
        finally:
            _shutdown(loop)

        assert resumed is False
        assert (
            timeout * 0.8 < elapsed < 5.0
        ), f"a stall must still end near the timeout, took {elapsed}s"


class TestTheBackstopOutlastsARealAnswer:
    def test_the_hard_bound_is_longer_than_the_slowest_answer_measured(self):
        """`hard_deadline` is the one failure the stall clock cannot see: a cache that
        churns forever while THIS chat is never quite fitted. It is not a second
        give-up timer, so it must not fire inside a legitimate answer.

        The 2026-09-05 run decoded at 2.3 tok/s on its slowest chat, so one 8192-token
        answer is about an hour on its own, and a waiter can be behind more than one.
        The old 20x (30 minutes) was shorter than a single such answer.
        """

        bound = DEFAULT_RESUME_WAIT_TIMEOUT_S * MAX_RESUME_WAIT_MULTIPLE
        assert bound >= 2 * (
            8192 / 2.3
        ), f"the backstop is {bound}s, shorter than two answers at the rate measured"
