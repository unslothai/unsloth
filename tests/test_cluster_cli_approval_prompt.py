# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""An approval prompt nobody answers must not keep `unsloth cluster pair` from exiting."""

from __future__ import annotations

import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

_SCRIPT = textwrap.dedent(
    """
    import asyncio, sys, threading
    sys.path.insert(0, sys.argv[1])
    from unsloth_cli.commands.cluster import prompt_in_background

    never = threading.Event()

    def unanswered_prompt(name):
        never.wait()  # the owner walked away from the terminal
        return True

    async def main():
        await asyncio.wait_for(prompt_in_background(unanswered_prompt, "desk"), 0.3)

    try:
        asyncio.run(main())
    except asyncio.TimeoutError:
        print("timed out")
    """
)


def test_a_timed_out_prompt_does_not_hold_the_command_open() -> None:
    started = time.monotonic()
    result = subprocess.run(
        [sys.executable, "-c", _SCRIPT, str(REPO)],
        capture_output = True,
        text = True,
        timeout = 60,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "timed out"
    assert time.monotonic() - started < 30


@pytest.mark.parametrize(
    ("answer", "expected"), [(lambda: True, True), (lambda: False, False)], ids = ["yes", "no"]
)
def test_the_owners_answer_reaches_the_pairing_window(answer, expected) -> None:
    import asyncio

    from unsloth_cli.commands.cluster import prompt_in_background

    async def main():
        return await prompt_in_background(answer)

    assert asyncio.run(main()) is expected


def test_a_prompt_that_raises_is_a_refusal_not_a_hang() -> None:
    import asyncio

    from studio.cluster_link import pairing
    from unsloth_cli.commands.cluster import prompt_in_background

    def broken():
        raise EOFError  # stdin closed

    async def main():
        with pytest.raises(EOFError):
            await pairing._ask(lambda name, words: prompt_in_background(broken), "desk", None)

    asyncio.run(main())
