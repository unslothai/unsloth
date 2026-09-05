# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A continuation the backend declines must not be retried by the client.

Observed in the browser on 2026-09-05, Qwen3.6-35B-A3B at a 8192-token window: one chat
filled the window on its own, the loop declined its own continuation ("the retry prompt
would not be served"), and the turn ended at `finish_reason` "length" holding the partial.
That is exactly the shape the client resumes automatically, so it sent the retry the
backend had just refused to send, the count preflight rejected it, and the user got
"Response interrupted" plus a red error box next to a Continue button that could never
work.

The client already refuses on `contextTruncation.fits === false`. The decline knows that
answer -- it priced the retry and tried evicting for it -- so it says so on that channel,
once, and the "raise Context Length" bar is shown instead of a doomed round.
"""

from __future__ import annotations

import sys
from pathlib import Path

_TESTS_DIR = str(Path(__file__).resolve().parent)
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from test_truncated_answer_continuation import (  # noqa: E402
    _HALF_AN_ANSWER,
    _cut_off_then,
    _done,
    _make_backend,
    _metadata,
    _run,
    _run_no_tools,
    _sse,
    _texts,
)


def _refusals(events) -> list[dict]:
    return [event for event in events if event.get("type") == "context_truncated"]


def _declining_backend(monkeypatch, payloads):
    """A server whose continuation preflight cannot admit the retry.

    The window is 4096 and every count comes back 4096, which is the real case in
    miniature: the answer consumed the physical context, so replaying it leaves no room
    to answer in and a single user turn has nothing older to evict.
    """
    backend = _make_backend(
        monkeypatch,
        _cut_off_then([_sse({"content": " never sent"}), _done()]),
        payloads,
    )
    monkeypatch.setattr(backend, "count_chat_tokens", lambda *_a, **_k: 4096)
    return backend


def test_the_in_loop_decline_tells_the_client_the_retry_would_not_fit(monkeypatch):
    payloads: list[dict] = []
    events = _run(_declining_backend(monkeypatch, payloads))

    assert len(payloads) == 1, "the continuation was sent after all"
    refusals = _refusals(events)
    # Once. `mergeContextTruncation` on the client SUMS the counters across a turn, so a
    # second copy of the same refusal is not a no-op.
    assert len(refusals) == 1, refusals
    refusal = refusals[0]
    assert refusal["fits"] is False
    # Nothing was evicted, and a non-zero count here raises "This conversation was
    # compacted" for a compaction that never happened.
    assert refusal["dropped_messages"] == 0
    assert refusal["context_length"] == 4096
    # What the client compares its own estimate of the partial against.
    assert refusal["prompt_target"] > 0
    assert refusal["prompt_target"] < refusal["context_length"]


def test_the_declined_turn_still_ends_with_length_and_the_whole_partial(monkeypatch):
    """The signal is added to the existing outcome, not in place of it. The partial is
    still the answer and Continue is still offered; only the AUTOMATIC round stops."""

    payloads: list[dict] = []
    events = _run(_declining_backend(monkeypatch, payloads))

    assert _metadata(events)["finish_reason"] == "length"
    content = "".join(_texts(events, "content"))
    assert "<!DOCTYPE html>" in content
    assert content.endswith("ctx.arc(6, -5, 5, 0")


def test_the_final_pass_decline_says_the_same_thing(monkeypatch):
    """The tool-free path has the same twin, and the client cannot tell them apart."""

    payloads: list[dict] = []
    events = _run_no_tools(_declining_backend(monkeypatch, payloads))

    assert len(payloads) == 1
    refusals = _refusals(events)
    assert len(refusals) == 1, refusals
    assert refusals[0]["fits"] is False
    assert _metadata(events)["finish_reason"] == "length"
    assert "<!DOCTYPE html>" in "".join(_texts(events, "content"))


def test_a_spent_output_cap_is_not_reported_as_a_context_refusal(monkeypatch):
    """The other reason a continuation is declined, and it is not the window.

    `max_tokens` belongs to THIS request; the client's next one carries its own, so that
    continuation is servable. Saying the prompt does not fit would hide a Continue that
    works, and send the user to the one setting that was never the constraint.
    """

    payloads: list[dict] = []
    backend = _make_backend(
        monkeypatch,
        _cut_off_then([_sse({"content": " never sent"}), _done()]),
        payloads,
    )
    # Room to spare in the window; it is the caller's cap that is gone.
    monkeypatch.setattr(backend, "count_chat_tokens", lambda *_a, **_k: 100)

    events = _run(backend, max_tokens = 64)

    assert len(payloads) == 1, "the continuation ran past the caller's cap"
    assert _refusals(events) == []


def test_a_continuation_that_is_sent_announces_no_refusal(monkeypatch):
    """The ordinary case still has to be silent, or every long answer would stop
    resuming itself."""

    payloads: list[dict] = []
    backend = _make_backend(
        monkeypatch,
        _cut_off_then([_sse({"content": ", 0, 6.28);\n</script>\n</html>"}), _done()]),
        payloads,
    )
    monkeypatch.setattr(backend, "count_chat_tokens", lambda *_a, **_k: 3800)

    events = _run(backend)

    assert len(payloads) == 2
    assert _refusals(events) == []
