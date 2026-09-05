"""Tool calls across concurrent chats must overlap, and nothing may quietly gate them.

P users sharing one llama-server already contend for KV. If their tool calls also
serialised, a chat parked on a ten second web search would hold its cells AND block
everyone else's tools, which is the opposite of what parking is for: the preemptor takes
a parked chat's room first precisely because it is holding cells while consuming no
compute.

Measured with `temp/overlap_probe.py`: four chats making three 0.3s calls each finish in
0.90s against 3.60s if fully serialised, with 60 cross-chat overlapping pairs. That is the
shape asserted here.

The within-chat half of that measurement, 0 overlapping pairs, is no longer true and is no
longer meant to be: a round's calls now run together as well. That is asserted in
`test_tool_calls_within_one_turn_overlap.py`, which owns the behaviour; this file stays
about the cross-chat property, which came first and must survive the other changing.
"""

import asyncio
import time

import pytest


class TestToolCallsOverlapAcrossChats:
    @pytest.mark.asyncio
    async def test_four_chats_run_their_tools_at_once(self):
        windows: list[tuple[str, float, float]] = []

        async def chat(name: str, calls: int, seconds: float) -> None:
            for _ in range(calls):
                start = time.monotonic()
                # The shape studio_tool_loop uses: a blocking tool handed to a thread, so
                # it never occupies the event loop.
                await asyncio.to_thread(time.sleep, seconds)
                windows.append((name, start, time.monotonic()))

        started = time.monotonic()
        await asyncio.gather(*(chat(f"chat{i}", 3, 0.05) for i in range(4)))
        wall = time.monotonic() - started

        serial = 4 * 3 * 0.05
        assert wall < serial * 0.6, (
            f"{wall:.2f}s for work that is {serial:.2f}s serialised: something is gating "
            f"tool execution across chats"
        )
        overlapping = sum(
            1
            for a in windows
            for b in windows
            if a is not b and a[0] != b[0] and a[1] < b[2] and b[1] < a[2]
        )
        assert overlapping > 0, "no two chats ever had a tool running at the same time"

    def test_no_module_level_gate_around_tool_execution(self):
        """A shared lock or semaphore here would serialise every user's tools.

        Structural, because the timing test above would still pass if a gate were added
        with a generous limit; this catches the gate itself.
        """
        from pathlib import Path

        from core.inference import studio_tool_loop

        source = Path(studio_tool_loop.__file__).read_text()
        for gate in (
            "asyncio.Lock()",
            "asyncio.Semaphore(",
            "threading.Lock()",
            "threading.Semaphore(",
        ):
            assert gate not in source, f"{gate} in the tool loop serialises all chats"

    def test_a_blocking_tool_is_never_run_on_the_event_loop(self):
        """Running one inline would freeze every other chat's stream, not just its own."""
        from pathlib import Path

        from core.inference import studio_tool_loop

        source = Path(studio_tool_loop.__file__).read_text()
        assert "asyncio.to_thread(" in source


class TestWithinOneChatCallsAlsoOverlapNow:
    """A round's calls used to run one after another. They no longer do.

    This class asserted the serial behaviour and described what changing it would take:
    "gate and dispatch the auto-approved calls as tasks, pump them concurrently, and
    interleave their events by card id, with confirmations still taken one at a time".
    That is what `_pump_tool_stream` and `_settle_call` do, so the assertions here would
    now be pinning the old shape in place.

    Kept rather than deleted, because the STRUCTURE still matters: the per-call loop is
    still a plain `for`, since preparing a call, gating it and recording its result are
    all order sensitive, and only the waiting was ever worth overlapping. What changed is
    that the loop starts a pump and moves on instead of draining inline.

    The behaviour itself is measured in `test_tool_calls_within_one_turn_overlap.py`, with
    a barrier rather than a timing assertion.
    """

    def test_preparing_and_recording_a_call_is_still_sequential(self):
        from pathlib import Path

        from core.inference import studio_tool_loop

        source = Path(studio_tool_loop.__file__).read_text(encoding = "utf-8")
        assert "for call in calls:" in source
        # A gather over the whole loop body would run approvals, the call budget and the
        # controller's ledger concurrently, which is not what was made parallel.
        assert "asyncio.gather(*(" not in source.split("for call in calls:", 1)[1][:2000]

    def test_the_calls_are_launched_before_they_are_drained(self):
        """The one structural fact that makes the round overlap at all.

        If `_settle_call` were awaited inside the loop for every call, the pumps would run
        one at a time again and every behavioural test would still pass on a machine fast
        enough to hide it.
        """
        from pathlib import Path

        from core.inference import studio_tool_loop

        source = Path(studio_tool_loop.__file__).read_text(encoding = "utf-8")
        body = source.split("for call in calls:", 1)[1]
        assert "pending_calls.append(entry)" in body
        assert "_pump_tool_stream(" in source
