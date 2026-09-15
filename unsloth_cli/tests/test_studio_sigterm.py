# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""SIGTERM (docker stop through supervisord, `unsloth studio stop`) takes the same path as Ctrl+C.

The `unsloth studio` and `unsloth start` commands run the server in-process and only
caught KeyboardInterrupt, so a SIGTERM killed the process with Python's default action:
no `_graceful_shutdown`, no stop-and-save for a running training job, no child cleanup.
"""

import re
import signal
from pathlib import Path

import pytest

from unsloth_cli.commands import studio as studio_mod

STUDIO_SRC = Path(studio_mod.__file__).read_text(encoding = "utf-8")


@pytest.fixture
def restore_sigterm():
    previous = signal.getsignal(signal.SIGTERM)
    yield
    signal.signal(signal.SIGTERM, previous)


def test_sigterm_becomes_a_keyboard_interrupt_once(restore_sigterm):
    studio_mod._graceful_shutdown_on_sigterm()
    handler = signal.getsignal(signal.SIGTERM)
    assert callable(handler) and handler is not signal.SIG_DFL
    with pytest.raises(KeyboardInterrupt):
        handler(signal.SIGTERM, None)
    assert signal.getsignal(signal.SIGTERM) is signal.SIG_DFL


def test_both_server_wait_loops_install_the_handler():
    loops = [
        m.start() for m in re.finditer(r"run_mod\._shutdown_event\.wait\(timeout = 1\)", STUDIO_SRC)
    ]
    assert len(loops) == 2, loops
    for loop in loops:
        before = STUDIO_SRC[:loop]
        assert (
            "_graceful_shutdown_on_sigterm()" in before[before.rindex("run_server(**run_kwargs)") :]
        )
