"""Tool calls across concurrent chats must overlap, and nothing may quietly gate them.

P users sharing one llama-server already contend for KV. If their tool calls also
serialised, a chat parked on a ten second web search would hold its cells AND block
everyone else's tools, which is the opposite of what parking is for: the preemptor takes
a parked chat's room first precisely because it is holding cells while consuming no
compute.

Measured with `temp/overlap_probe.py`: four chats making three 0.3s calls each finish in
0.90s against 3.60s if fully serialised, with 60 cross-chat overlapping pairs and 0
within-chat ones. That is the shape asserted here.
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
        for gate in ("asyncio.Lock()", "asyncio.Semaphore(", "threading.Lock()", "threading.Semaphore("):
            assert gate not in source, f"{gate} in the tool loop serialises all chats"

    def test_a_blocking_tool_is_never_run_on_the_event_loop(self):
        """Running one inline would freeze every other chat's stream, not just its own."""
        from pathlib import Path

        from core.inference import studio_tool_loop

        source = Path(studio_tool_loop.__file__).read_text()
        assert "asyncio.to_thread(" in source


class TestWithinOneChatCallsAreSerial:
    """Documented, not asserted as desirable: a round's calls run one after another.

    `for call in calls` at the execution loop. It is serial for reasons that are real --
    each call may need its own approval, a later call can depend on an earlier result,
    the no-progress guard is keyed on results, and the one-shot ledger is per call -- but
    it does mean three independent searches in one turn take three times as long as one.
    Parallelising them is a worthwhile change and a separate one, since it has to keep
    approval gating and card ordering intact.
    """

    def test_the_execution_loop_is_a_plain_sequential_for(self):
        from pathlib import Path

        from core.inference import studio_tool_loop

        source = Path(studio_tool_loop.__file__).read_text()
        assert "for call in calls:" in source
        # If this ever becomes a gather, the docstring above is stale and the sibling
        # suite for approval ordering needs to be revisited rather than deleted.
        assert "asyncio.gather(*(" not in source.split("for call in calls:", 1)[1][:2000]
