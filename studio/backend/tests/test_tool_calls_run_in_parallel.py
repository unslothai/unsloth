# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tool calls across concurrent chats must overlap, and nothing may quietly gate them."""

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
        """A shared lock or semaphore here would serialise every user's tools."""
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
    """A round's calls used to run one after another."""

    def test_preparing_and_recording_a_call_is_still_sequential(self):
        from pathlib import Path

        from core.inference import studio_tool_loop

        source = Path(studio_tool_loop.__file__).read_text(encoding = "utf-8")
        assert "for call in calls:" in source
        # A gather over the whole loop body would run approvals, the call budget and the
        # controller's ledger concurrently, which is not what was made parallel.
        assert "asyncio.gather(*(" not in source.split("for call in calls:", 1)[1][:2000]

    def test_the_calls_are_launched_before_they_are_drained(self):
        """The one structural fact that makes the round overlap at all."""
        from pathlib import Path

        from core.inference import studio_tool_loop

        source = Path(studio_tool_loop.__file__).read_text(encoding = "utf-8")
        body = source.split("for call in calls:", 1)[1]
        assert "pending_calls.append(entry)" in body
        assert "_pump_tool_stream(" in source
