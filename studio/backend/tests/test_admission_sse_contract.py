# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Every admission signal the frontend can read, the backend must actually send."""

import pathlib
import re

import pytest

BACKEND = pathlib.Path(__file__).resolve().parent.parent
FRONTEND = BACKEND.parent / "frontend"
ADMISSION_TS = FRONTEND / "src" / "features" / "chat" / "utils" / "admission-status.ts"
ROUTES = BACKEND / "routes" / "inference.py"


def _frontend_comments() -> set[str]:
    """The comment payloads the client will act on, read from the source of truth."""
    text = ADMISSION_TS.read_text(encoding = "utf-8")
    return set(re.findall(r'^export const ADMISSION_COMMENT_\w+ = "([^"]+)";', text, re.M))


@pytest.mark.skipif(not ADMISSION_TS.exists(), reason = "frontend not present in this tree")
class TestTheSignalsLineUp:
    def test_the_frontend_declares_the_four_we_expect(self):
        """A guard on the guard: if this file is renamed or restructured, the test below would
        silently start asserting nothing at all.
        """
        assert _frontend_comments() == {
            "admission-wait",
            "admission-done",
            "preempt-paused",
            "preempt-resumed",
        }

    def test_every_readable_signal_is_emitted_somewhere(self):
        routes = ROUTES.read_text(encoding = "utf-8")
        missing = sorted(c for c in _frontend_comments() if f": {c}" not in routes)
        assert not missing, (
            f"the client understands {missing} and the server never sends them. That is "
            "not a dormant feature, it is a user staring at a half-written answer that "
            "stopped for no stated reason."
        )

    def test_the_emitted_constants_are_actually_yielded(self):
        """Defining the string is not sending it."""
        routes = ROUTES.read_text(encoding = "utf-8")
        for name in (
            "_OPENAI_ADMISSION_SSE_WAIT",
            "_OPENAI_ADMISSION_SSE_DONE",
            "_OPENAI_PREEMPT_SSE_PAUSED",
            "_OPENAI_PREEMPT_SSE_RESUMED",
        ):
            assert (
                routes.count(name) >= 2
            ), f"{name} is defined but never yielded, so the client is told nothing"
            # Yielded directly, or through the state table every consumer reads.
            assert f"yield {name}" in routes or f"{name}\n" in routes or f"{name}," in routes

    def test_the_pause_signal_has_a_producer_in_the_generator(self):
        """The route can only forward what the generator hands it."""
        llama_cpp = (BACKEND / "core" / "inference" / "llama_cpp.py").read_text(encoding = "utf-8")
        assert '{"type": "preempt", "state": "paused"}' in llama_cpp
        assert '{"type": "preempt", "state": "resumed"}' in llama_cpp
        assert '"type") == "preempt"' in ROUTES.read_text(encoding = "utf-8")
