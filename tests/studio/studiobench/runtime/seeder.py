# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Seeds old turns over REST, skipping the streaming path; equivalence to streaming is checked."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from ..fixture.corpus import RungPlan, Unit
from .lifecycle import StudioAuth, auth_request_json

# Not zero: a streamed reply carries usage and duration the seeded one lacks.
EQUIVALENCE_TOLERANCE = 0.02


def _now_ms() -> int:
    return int(time.time() * 1000)


def _assistant_content(unit: Unit) -> list[dict]:
    """Reasoning must be a `reasoning` part: a `<think>` inside a text part is not re-parsed on load."""
    parts: list[dict] = []
    if unit.reasoning:
        parts.append({"type": "reasoning", "text": unit.reasoning})
    # Tool calls go between reasoning and answer; order decides which component renders them.
    for call in unit.tool_calls:
        parts.append(dict(call))
    if unit.content:
        parts.append({"type": "text", "text": unit.content})
    return parts


def turn_marker(index: int, unit_index: int) -> str:
    """Single source for the user-turn marker: the readiness gate matches this exact string in the DOM."""
    return f"studiobench turn {index}: continue with unit {unit_index}"


@dataclass
class SeededThread:
    thread_id: str
    messages: int
    seeded_chars: int
    seconds: float
    turns: int
    first_marker: Optional[str] = None
    last_marker: Optional[str] = None


@dataclass
class Seeder:
    base_url: str
    auth: StudioAuth
    model_id: str
    log: Callable[[str], None] = print
    # Sent whole: a partial PUT with pruneMissing would delete everything not in the batch.
    batch_note: str = field(default = "one transaction, pruneMissing", init = False)

    def _url(self, path: str) -> str:
        return f"{self.base_url.rstrip('/')}{path}"

    def create_thread(self, title: str = "studiobench") -> str:
        thread_id = str(uuid.uuid4())
        # Authenticated helper because the run outlives the 60-minute access token.
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
            # pruneMissing replaces the thread, else every rung after the first would be cumulative.
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


def dom_signature(page) -> dict:
    """What the app BUILT, read from the DOM. The only fair comparison between the two paths."""
    return page.evaluate("() => window.__sb.dom.counts()")


def compare_signatures(
    streamed: dict,
    seeded: dict,
    tolerance: float = EQUIVALENCE_TOLERANCE,
) -> dict:
    """Element count is not gated: a streamed reply has usage and timing labels a seeded one lacks."""
    # Gate on content only: collapsed reasoning panes in seeded threads do not mount their spans.
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


def measure_chars_per_token(
    text: str, base_url: str, auth: Optional[StudioAuth], model_id: str
) -> dict:
    """Measured chars per token, never an assumed 4.0; the answering source is reported with it."""
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
    # Last resort: word counting is off by tens of percent on code, so the result is labelled.
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
