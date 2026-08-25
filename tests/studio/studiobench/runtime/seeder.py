# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Seed a thread's bulk mass over REST, and check that seeding is equivalent to streaming.

WHY SEED AT ALL. At the field's own cadence -- 24 characters every 73 milliseconds -- a million
tokens is three and a half hours of streaming. A benchmark nobody can run measures nothing, so all
but the last turn is written straight into the store with
`PUT /api/chat/threads/{id}/messages`, and only the last reply streams.

WHY THE EQUIVALENCE IS CHECKED AND NOT ASSUMED. Seeding takes a different path into the app, and
reading the shipped code says it is a MATERIALLY different one. A streamed reply arrives as
`delta.reasoning_content`, is wrapped into `<think>...</think>`, appended to a cumulative buffer,
and `parseAssistantContent(cumulativeText)` re-parses the whole growing buffer on every delta. Only
at the end is the parsed parts array persisted. A seeded reply skips all of that: it is written as
the finished parts array and loaded straight into the runtime, and `<think>` in a stored text part
is NOT re-parsed on load, because parsing happens only during streaming.

So the two paths should converge on the same DOM and may not. The check is run at the 10K rung,
where both are affordable, and it compares what the app actually built: the message count, the
assistant character count, the highlight span count, the reasoning pane count. Rungs above 10K are
labelled `fidelity: seeded_only` when it fails. That is a FINDING, printed, not a bug to hide --
it says exactly which of this tool's numbers are about the streaming path and which are about a
thread that was put there.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from ..fixture.corpus import RungPlan, Unit
from .lifecycle import StudioAuth, auth_request_json

# How close the two paths must land to count as equivalent. Not zero: a streamed reply carries a
# usage record and a duration the seeded one does not, and the composer state differs, so a
# handful of elements always differ. 2% on the quantities that scale with content.
EQUIVALENCE_TOLERANCE = 0.02


def _now_ms() -> int:
    return int(time.time() * 1000)


def _assistant_content(unit: Unit) -> list[dict]:
    """The stored parts array for an assistant turn.

    A `{"type": "reasoning"}` PART, not `reasoning_content` and not `<think>` inside a text part.
    There is no reasoning_content column on a stored message, and a text part containing `<think>`
    is not re-parsed when the thread is loaded, so it would render as literal angle brackets in
    the visible answer -- a thread that looks wrong and measures the wrong DOM.
    """
    parts: list[dict] = []
    if unit.reasoning:
        parts.append({"type": "reasoning", "text": unit.reasoning})
    # Tool calls sit BETWEEN the reasoning and the answer, which is where a real turn puts them:
    # the model thinks, calls tools, then answers. reasoning.tsx groups adjacent tool-call parts
    # with the reasoning above them, so the order decides whether a tool group renders inside the
    # collapsible pane or as its own block, and those are different components with different
    # costs.
    for call in unit.tool_calls:
        parts.append(dict(call))
    if unit.content:
        parts.append({"type": "text", "text": unit.content})
    return parts


def turn_marker(index: int, unit_index: int) -> str:
    """The exact plain text this harness writes into the user turn at `index`.

    ONE function rather than an f-string in two places, because the readiness gate matches on this
    string in the DOM. A marker that the seeder writes and the gate looks for in slightly different
    words is a gate that never passes, and the symptom would be a timeout that looks like a slow
    app.
    """
    return f"studiobench turn {index}: continue with unit {unit_index}"


@dataclass
class SeededThread:
    thread_id: str
    messages: int
    seeded_chars: int
    seconds: float
    turns: int
    # The markers on the FIRST and LAST user turns. The readiness gate uses `last_marker` to prove
    # the end of the thread is mounted, and the completeness probe uses `first_marker` to prove a
    # windowed arm still holds the head of the conversation. Plain text written by this harness,
    # so neither is a guess about what a markdown renderer will do to the corpus.
    first_marker: Optional[str] = None
    last_marker: Optional[str] = None


@dataclass
class Seeder:
    base_url: str
    auth: StudioAuth
    model_id: str
    log: Callable[[str], None] = print
    # Messages per PUT. The route replaces the whole message list in one SQLite transaction, so a
    # 1M-token thread is one enormous request; it is sent whole because a partial PUT with
    # pruneMissing would delete everything not in the batch.
    batch_note: str = field(default = "one transaction, pruneMissing", init = False)

    def _url(self, path: str) -> str:
        return f"{self.base_url.rstrip('/')}{path}"

    def create_thread(self, title: str = "studiobench") -> str:
        thread_id = str(uuid.uuid4())
        # `auth_request_json`, not `request_json`, and that is not a spelling preference: the
        # seeder is asked for a thread once per cell for as long as the run lasts, and an access
        # token is good for 60 minutes. See `StudioAuth`: the token is re-minted before it expires
        # and once more if a 401 arrives anyway.
        auth_request_json(
            self.auth,
            self._url("/api/chat/threads"),
            method = "POST",
            timeout = 60,
            body = {
                "id": thread_id,
                "title": title,
                "modelType": "base",
                "modelId": self.model_id,
                "createdAt": _now_ms(),
            },
        )
        return thread_id

    def seed(
        self,
        plan: RungPlan,
        thread_id: Optional[str] = None,
    ) -> SeededThread:
        """Write every unit except the streamed one into the thread, as user/assistant pairs."""
        thread_id = thread_id or self.create_thread()
        messages: list[dict] = []
        created = _now_ms() - len(plan.seeded_units) * 2000
        parent: Optional[str] = None
        for i, unit in enumerate(plan.seeded_units):
            user_id = str(uuid.uuid4())
            messages.append(
                {
                    "id": user_id,
                    "threadId": thread_id,
                    "parentId": parent,
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": turn_marker(i, unit.index),
                        }
                    ],
                    "attachments": None,
                    "metadata": None,
                    "createdAt": created + i * 2000,
                }
            )
            assistant_id = str(uuid.uuid4())
            messages.append(
                {
                    "id": assistant_id,
                    "threadId": thread_id,
                    "parentId": user_id,
                    "role": "assistant",
                    "content": _assistant_content(unit),
                    "attachments": None,
                    "metadata": None,
                    "createdAt": created + i * 2000 + 1000,
                }
            )
            parent = assistant_id
        started = time.monotonic()
        if messages:
            # pruneMissing so this REPLACES the thread rather than merging into whatever a
            # previous cell left behind. A merge would make every rung after the first cumulative.
            auth_request_json(
                self.auth,
                self._url(f"/api/chat/threads/{thread_id}/messages"),
                method = "PUT",
                timeout = 900,
                body = {"messages": messages, "pruneMissing": True},
            )
        seconds = time.monotonic() - started
        self.log(
            f"  seeded {len(messages)} messages ({plan.seeded_chars:,} chars) " f"in {seconds:.1f}s"
        )
        units = list(plan.seeded_units)
        return SeededThread(
            thread_id = thread_id,
            messages = len(messages),
            seeded_chars = plan.seeded_chars,
            seconds = seconds,
            turns = len(units),
            first_marker = turn_marker(0, units[0].index) if units else None,
            last_marker = turn_marker(len(units) - 1, units[-1].index) if units else None,
        )

    def read_back(self, thread_id: str) -> list[dict]:
        got = auth_request_json(
            self.auth,
            self._url(f"/api/chat/threads/{thread_id}/messages"),
            timeout = 300,
        )
        if isinstance(got, dict):
            return got.get("messages", [])
        return got or []


# ── the equivalence check ───────────────────────────────────────────


def dom_signature(page) -> dict:
    """What the app BUILT, read from the DOM. The only fair comparison between the two paths."""
    return page.evaluate("() => window.__sb.dom.counts()")


def compare_signatures(
    streamed: dict,
    seeded: dict,
    tolerance: float = EQUIVALENCE_TOLERANCE,
) -> dict:
    """Are the two paths equivalent on the quantities that scale with content?

    Element count is compared too but is NOT a gate on its own: a streamed reply leaves a usage
    record and a "thought for N seconds" label a seeded one has no source for, so a handful of
    elements legitimately differ and gating on exact equality would fail every time for a reason
    that has nothing to do with fidelity.
    """
    # GATED ON CONTENT, REPORTED ON REASONING.
    #
    # A collapsed reasoning pane in a SEEDED thread does not mount its children, while a streamed
    # one does: it was open while the text arrived, and collapsing afterwards leaves the subtree
    # in place. Measured directly, the same text carried 1,485 reasoning spans one way and 0 the
    # other, and open-then-close on the seeded side unmounted them again, so this is not something
    # seeding can be made to reproduce -- it is a property of how the app builds a thread.
    #
    # Gating on total `highlight_spans` therefore asked a question seeding can never pass, and the
    # answer moved with whatever pane state the film happened to leave behind: two runs of the
    # same rung reported 2.1% and 36.4% drift. The question worth asking is whether the same text
    # renders the same CONTENT, which is what these keys are. The reasoning difference is measured
    # and reported below rather than swept into a pass or a fail.
    keys = ("assistant_messages", "content_code_blocks", "content_spans", "reasoning_panes")
    fields: dict = {}
    equivalent = True
    for key in keys:
        a, b = streamed.get(key), seeded.get(key)
        if a is None or b is None:
            fields[key] = {
                "streamed": a,
                "seeded": b,
                "within_tolerance": None,
                "reason": "one side did not report this quantity",
            }
            equivalent = False
            continue
        biggest = max(abs(a), abs(b), 1)
        drift = abs(a - b) / biggest
        ok = drift <= tolerance
        fields[key] = {"streamed": a, "seeded": b, "drift": round(drift, 4), "within_tolerance": ok}
        equivalent = equivalent and ok
    fields["elements"] = {
        "streamed": streamed.get("elements"),
        "seeded": seeded.get("elements"),
        "gating": False,
        "note": "reported, not gated: a streamed reply carries a usage record "
        "and a reasoning duration label a seeded one has no source for",
    }
    for key, note in (
        (
            "reasoning_spans",
            "reported, not gated: a collapsed reasoning pane mounts its children when the text was "
            "STREAMED into it and does not when the thread was seeded, so this difference is a "
            "property of the app and not of the fixture",
        ),
        (
            "highlight_spans",
            "reported, not gated: the total includes reasoning spans, which the two paths cannot "
            "agree on; content_spans is the gated quantity",
        ),
        (
            "assistant_chars",
            "reported, not gated: textContent counts hidden-but-mounted reasoning text, so it "
            "carries the same asymmetry as reasoning_spans",
        ),
    ):
        a, b = streamed.get(key), seeded.get(key)
        entry = {"streamed": a, "seeded": b, "gating": False, "note": note}
        if a is not None and b is not None:
            entry["drift"] = round(abs(a - b) / max(abs(a), abs(b), 1), 4)
        fields[key] = entry
    return {
        "equivalent": equivalent,
        "tolerance": tolerance,
        "fields": fields,
        "checked_attempted": True,
    }


# ── chars per token ─────────────────────────────────────────────────


def measure_chars_per_token(
    text: str, base_url: str, auth: Optional[StudioAuth], model_id: str
) -> dict:
    """The MEASURED characters-per-token of this corpus, never an assumed 4.0.

    The rungs are named in tokens and the corpus is built in characters, so the ratio is the thing
    that makes the two the same claim. It is measured, in this order, from whatever is available,
    and the SOURCE is reported with the number so a reader can see which one answered. A run that
    can only fall back to the whitespace estimate says so, rather than printing a ratio that looks
    like every other run's.
    """
    sample = text[:200_000]
    if not sample:
        return {
            "chars_per_token": None,
            "source": None,
            "chars_per_token_attempted": False,
            "reason": "no text to measure",
        }
    try:
        import tiktoken  # type: ignore[import]

        enc = tiktoken.get_encoding("cl100k_base")
        n = len(enc.encode(sample))
        return {
            "chars_per_token": round(len(sample) / max(1, n), 3),
            "source": "tiktoken/cl100k",
            "tokens": n,
            "sample_chars": len(sample),
            "chars_per_token_attempted": True,
        }
    except Exception:  # noqa: BLE001
        pass
    if auth is not None:
        try:
            got = auth_request_json(
                auth,
                f"{base_url.rstrip('/')}/api/inference/chat/count_tokens",
                method = "POST",
                timeout = 120,
                body = {"model": model_id, "messages": [{"role": "user", "content": sample}]},
            )
            n = (got or {}).get("total_tokens") or (got or {}).get("tokens")
            if n:
                return {
                    "chars_per_token": round(len(sample) / n, 3),
                    "source": "studio /api/inference/chat/count_tokens",
                    "tokens": n,
                    "sample_chars": len(sample),
                    "chars_per_token_attempted": True,
                }
        except Exception:  # noqa: BLE001
            pass
    # Last resort, and LABELLED. Counting whitespace-delimited words plus punctuation is a rough
    # stand-in for a BPE tokeniser and is off by tens of percent on dense code, which is most of
    # this corpus.
    words = len(sample.split())
    punct = sum(1 for c in sample if not c.isalnum() and not c.isspace())
    est = max(1, words + punct // 2)
    return {
        "chars_per_token": round(len(sample) / est, 3),
        "source": "whitespace-and-punctuation estimate",
        "tokens": est,
        "sample_chars": len(sample),
        "chars_per_token_attempted": True,
        "reason": "no tokeniser was available; this ratio is an estimate and is off by tens "
        "of percent on dense code",
    }
