# Copyright 2026-present the Unforgettable contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""End-of-episode extract: naive, drift, and bounded LLM drafts."""

from __future__ import annotations

import json
from typing import Any, Optional, TYPE_CHECKING

from unforgettable.eyes.gate import LogGateEyes
from unforgettable.host import EXTRACT_MAX_TOKENS, ToolTrace

if TYPE_CHECKING:
    from unforgettable.host import Host
    from unforgettable.loop.context import EpisodeState

TWIN_NOTE_TITLE = "World/sim disagreement"
TWIN_NOTE_BODY_CAP = 800
EPISODE_USER_TEXT_CLIP = 200
EPISODE_BODY_CAP = 2000
EPISODE_TITLE_ID_CHARS = 8
EPISODE_RIM_CONTACTS = frozenset({"world", "sim"})
EPISODE_FALLBACK_PROVENANCE = "infer"

EXTRACT_TRACE_WINDOW = 24
EXTRACT_TRACE_CHAR_BUDGET = 8000
EXTRACT_MAX_DRAFTS = 8
EXTRACT_TITLE_CAP = 80
EXTRACT_BODY_CAP = 1200
EXTRACT_MIN_NON_MEMORY_TRACES = 2
EXTRACT_ALLOWED_KINDS = frozenset({"claim", "procedure", "error_fix", "entity", "twin_note"})
EXTRACT_FORBIDDEN_KINDS = frozenset({"directive", "episode"})
EXTRACT_PROVENANCE = "infer"
EXTRACT_SYSTEM = (
    "Extract durable memories from this episode. "
    "Reply with a JSON array of objects, each with keys kind, title, body. "
    f"Allowed kinds: {', '.join(sorted(EXTRACT_ALLOWED_KINDS))}. "
    f"Do not emit {', '.join(sorted(EXTRACT_FORBIDDEN_KINDS))}. "
    "Do not include provenance. "
    f"At most {EXTRACT_MAX_DRAFTS} objects. "
    f"Title at most {EXTRACT_TITLE_CAP} characters; "
    f"body at most {EXTRACT_BODY_CAP}."
)


def _is_memory_tool(name: str) -> bool:
    return name.startswith("memory.") or name.startswith("memory_")


def _non_memory_traces(state: "EpisodeState") -> list[ToolTrace]:
    return [trace for trace in state.traces if not _is_memory_tool(trace.name)]


def _should_llm_extract(state: "EpisodeState") -> bool:
    if len(_non_memory_traces(state)) >= EXTRACT_MIN_NON_MEMORY_TRACES:
        return True
    return any(event.get("kind") == "failure" for event in state.trace_events)


EXTRACT_RESULT_CHARS = 240


def _render_trace(trace: ToolTrace) -> str:
    # Name + clipped result only. Arguments often hold secrets / env dumps.
    result = (trace.result or "").replace("\n", " ").strip()
    if len(result) > EXTRACT_RESULT_CHARS:
        result = result[:EXTRACT_RESULT_CHARS].rstrip() + "..."
    return f"{trace.name} {result}".strip()


def _windowed_trace_text(traces: list[ToolTrace]) -> str:
    selected = [t for t in traces if not _is_memory_tool(t.name)][-EXTRACT_TRACE_WINDOW:]
    chunks = [_render_trace(t) for t in selected]
    total = sum(len(chunk) for chunk in chunks)
    while chunks and total > EXTRACT_TRACE_CHAR_BUDGET:
        oldest = chunks[0]
        overflow = total - EXTRACT_TRACE_CHAR_BUDGET
        if len(oldest) <= overflow:
            chunks.pop(0)
            total -= len(oldest)
        else:
            chunks[0] = oldest[: len(oldest) - overflow]
            total = EXTRACT_TRACE_CHAR_BUDGET
    return "\n".join(chunks)


def _events_text(events: list[dict[str, Any]]) -> str:
    lines = []
    for event in events:
        contact = event.get("contact") or ""
        kind = event.get("kind") or ""
        summary = event.get("summary") or ""
        lines.append(f"{contact}/{kind}: {summary}")
    return "\n".join(lines)


def _extract_messages(state: "EpisodeState") -> list[dict[str, Any]]:
    parts = [_windowed_trace_text(state.traces)]
    events = _events_text(state.trace_events)
    if events:
        parts.append(events)
    return [
        {"role": "system", "content": EXTRACT_SYSTEM},
        {"role": "user", "content": "\n\n".join(part for part in parts if part)},
    ]


def _strip_markdown_fences(raw: str) -> str:
    text = (raw or "").strip()
    if not text.startswith("```"):
        return text
    lines = text.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _parse_extract(raw: str) -> list[dict[str, Any]]:
    data = json.loads(_strip_markdown_fences(raw))
    if not isinstance(data, list):
        raise ValueError("extract is not a JSON array")
    drafts: list[dict[str, Any]] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        kind = item.get("kind")
        if kind not in EXTRACT_ALLOWED_KINDS:
            continue
        title = str(item.get("title") or "").strip()
        if not title:
            continue
        title = title[:EXTRACT_TITLE_CAP]
        body = item.get("body")
        if body is None:
            body = ""
        elif not isinstance(body, str):
            body = str(body)
        drafts.append(
            {
                "kind": kind,
                "title": title,
                "body": body[:EXTRACT_BODY_CAP],
                "provenance": EXTRACT_PROVENANCE,
                "explicit": False,
                "speaker": "model",
            }
        )
        if len(drafts) >= EXTRACT_MAX_DRAFTS:
            break
    return drafts


def _log_extract_failure(host: "Host", message: str) -> None:
    try:
        LogGateEyes().note(message, db_path = host.memory_db_path())
    except Exception:
        return


async def llm_extract(state: "EpisodeState", host: "Host") -> list[dict[str, Any]]:
    """Model-proposed infer drafts from traces. Never raises."""
    if not _should_llm_extract(state):
        return []
    try:
        raw = await host.complete(_extract_messages(state), max_tokens = EXTRACT_MAX_TOKENS)
        return _parse_extract(raw)
    except Exception as exc:
        _log_extract_failure(host, f"llm_extract failed: {exc}")
        return []


def from_drift(state: "EpisodeState") -> list[dict[str, Any]]:
    """If sim succeeded and a later world retry failed, write one twin_note."""
    saw_sim_success = False
    sim_ok = None
    world_fail = None
    for event in state.trace_events:
        if event.get("kind") == "success" and event.get("contact") == "sim":
            saw_sim_success = True
            sim_ok = event
        elif saw_sim_success and event.get("kind") == "failure" and event.get("contact") == "world":
            world_fail = event
            break
    if sim_ok is None or world_fail is None:
        return []
    body = (
        f"Sim: {sim_ok.get('summary') or 'sim succeeded'}\n"
        f"World retry: {world_fail.get('summary') or 'world failed'}"
    )
    if len(body) > TWIN_NOTE_BODY_CAP:
        body = body[:TWIN_NOTE_BODY_CAP]
    return [
        {
            "kind": "twin_note",
            "title": TWIN_NOTE_TITLE,
            "body": body,
            "provenance": "mixed",
            "explicit": False,
            "bookkeeping": True,
            "speaker": "world",
            "warrant": body,
        }
    ]


def from_episode(state: "EpisodeState") -> list[dict[str, Any]]:
    """If a world/sim failure was later followed by success, propose one error_fix."""
    fail = None
    success = None
    world_success = None
    for event in state.trace_events:
        if event.get("kind") == "failure" and fail is None:
            fail = event
        if event.get("kind") == "success" and fail is not None:
            if success is None:
                success = event
            if event.get("contact") == "world":
                world_success = event
    if world_success is not None:
        success = world_success
    if fail is None or success is None:
        return []
    fail_contact = fail.get("contact") or "world"
    ok_contact = success.get("contact") or "world"
    if fail_contact == "sim" and ok_contact == "sim":
        provenance = "sim"
    elif fail_contact != ok_contact:
        provenance = "mixed"
    else:
        provenance = fail_contact
    fail_summary = (fail.get("summary") or "failed action").strip()
    title = f"Error then fix: {fail_summary}"[:EXTRACT_TITLE_CAP]
    body = (f"Tried: {fail_summary}\n" f"Then: {success.get('summary') or 'later succeeded'}")[
        :EXTRACT_BODY_CAP
    ]
    return [
        {
            "kind": "error_fix",
            "title": title,
            "body": body,
            "provenance": provenance,
            "explicit": False,
            "speaker": fail_contact if fail_contact in EPISODE_RIM_CONTACTS else "model",
            "warrant": body,
        }
    ]


def _clip(text: str, cap: int) -> str:
    if len(text) <= cap:
        return text
    return text[:cap]


def _episode_provenance(state: "EpisodeState") -> str:
    contacts = {
        event.get("contact")
        for event in state.trace_events
        if event.get("contact") in EPISODE_RIM_CONTACTS
    }
    if contacts == EPISODE_RIM_CONTACTS:
        return "mixed"
    if len(contacts) == 1:
        return next(iter(contacts))
    return EPISODE_FALLBACK_PROVENANCE


def episode_summary(
    state: "EpisodeState",
    *,
    last_user: str,
    draft_ids: list[str],
    actions: Optional[list[str]] = None,
) -> dict[str, Any]:
    """Build the bookkeeping episode draft. Pointer, not a transcript."""
    sections = [f"## User\n{_clip(last_user or '', EPISODE_USER_TEXT_CLIP)}"]
    action_list = list(actions or [])
    if action_list:
        sections.append("## Actions\n" + "\n".join(f"- {action}" for action in action_list))
    if state.trace_events:
        lines = []
        for event in state.trace_events:
            contact = event.get("contact") or EPISODE_FALLBACK_PROVENANCE
            kind = event.get("kind") or ""
            summary = event.get("summary") or ""
            lines.append(f"- {contact}/{kind}: {summary}")
        sections.append("## Events\n" + "\n".join(lines))
    if draft_ids:
        sections.append("## Drafts\n" + "\n".join(f"- {rid}" for rid in draft_ids))
    return {
        "kind": "episode",
        "title": f"Episode {state.episode_id[:EPISODE_TITLE_ID_CHARS]}",
        "body": _clip("\n\n".join(sections), EPISODE_BODY_CAP),
        "provenance": _episode_provenance(state),
        "explicit": False,
        "bookkeeping": True,
        "speaker": "model",
    }
