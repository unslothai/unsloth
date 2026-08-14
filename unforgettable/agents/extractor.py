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

"""End-of-episode extract. LLM extract is a named gap."""

from __future__ import annotations

from typing import Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from unforgettable.loop.context import EpisodeState

TWIN_NOTE_TITLE = "World/sim disagreement"
TWIN_NOTE_BODY_CAP = 800
EPISODE_USER_TEXT_CLIP = 200
EPISODE_BODY_CAP = 2000
EPISODE_TITLE_ID_CHARS = 8
EPISODE_RIM_CONTACTS = frozenset({"world", "sim"})
EPISODE_FALLBACK_PROVENANCE = "infer"


def llm_extract(_state: "EpisodeState") -> list[dict[str, Any]]:
    """Gap: model-proposed records from traces. Phase 1 returns nothing."""
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
        }
    ]


def from_episode(state: "EpisodeState") -> list[dict[str, Any]]:
    """If a world/sim failure was later followed by success, propose one error_fix."""
    fail = None
    success = None
    for event in state.trace_events:
        if event.get("kind") == "failure" and fail is None:
            fail = event
        if event.get("kind") == "success" and fail is not None and success is None:
            success = event
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
    title = "Error then fix"
    body = (
        f"Tried: {fail.get('summary') or 'failed action'}\n"
        f"Then: {success.get('summary') or 'later succeeded'}"
    )
    return [
        {
            "kind": "error_fix",
            "title": title,
            "body": body,
            "provenance": provenance,
            "explicit": False,
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
    }
