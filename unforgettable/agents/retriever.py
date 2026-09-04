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

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional

from unforgettable.constants import is_what, is_who, speaker_of
from unforgettable.store.search import search_records
from unforgettable.store.titles import normalize_title

# Episode summaries are last-run logs, not standing knowledge.
DEFAULT_RETRIEVE_KINDS = frozenset(
    {"claim", "procedure", "error_fix", "entity", "directive", "twin_note"}
)

DEFAULT_MAX_RECORDS = 6
DEFAULT_MAX_CHARS = 2400
DEFAULT_SNIPPET_CHARS = 280
DEFAULT_MAX_TWIN_NOTES = 1
STALE_AGE_DAYS = 30

HIGH_STAKES_PROVENANCE = ("world", "mixed", "human")
_SIM_LESSON_KINDS = frozenset({"error_fix", "twin_note"})
_SIM_LESSON_PROVENANCE = frozenset({"sim", "mixed"})

_INJECT_HEADER = "Durable memories relevant to this task:"
_ELLIPSIS = "..."


@dataclass(frozen = True)
class RetrievePolicy:
    max_records: int = DEFAULT_MAX_RECORDS
    max_chars: int = DEFAULT_MAX_CHARS
    snippet_chars: int = DEFAULT_SNIPPET_CHARS
    high_stakes: bool = False
    max_twin_notes: int = DEFAULT_MAX_TWIN_NOTES
    contact: str = "world"  # world | sim
    exclude_ids: frozenset[str] = frozenset()


def retrieve(
    query: str,
    *,
    policy: Optional[RetrievePolicy] = None,
    db_path = None,
    namespace_id: Optional[str] = None,
) -> list[dict[str, Any]]:
    if not (query or "").strip():
        return []
    policy = policy or RetrievePolicy()
    # Rehearsal needs sim lessons; high-stakes provenance drop is world-acting only.
    provenances = (
        HIGH_STAKES_PROVENANCE if policy.high_stakes and policy.contact == "world" else None
    )
    overfetch = policy.max_records + len(policy.exclude_ids)
    hits = search_records(
        query,
        top_k = overfetch,
        kinds = DEFAULT_RETRIEVE_KINDS,
        provenances = provenances,
        namespace_id = namespace_id,
        db_path = db_path,
    )
    if policy.exclude_ids:
        hits = [rec for rec in hits if rec.get("id") not in policy.exclude_ids]
    hits = hits[: policy.max_records]
    capped = _cap_twin_notes(hits, policy.max_twin_notes)
    if policy.contact == "sim":
        capped = _prefer_sim_lessons(capped)
    return _drop_colliding_who(capped)


def _drop_colliding_who(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    what_titles = {normalize_title(rec.get("title") or "") for rec in records if is_what(rec)}
    if not what_titles:
        return records
    return [
        rec
        for rec in records
        if not (is_who(rec) and normalize_title(rec.get("title") or "") in what_titles)
    ]


def _is_sim_lesson(rec: dict[str, Any]) -> bool:
    return rec.get("kind") in _SIM_LESSON_KINDS and rec.get("provenance") in _SIM_LESSON_PROVENANCE


def _prefer_sim_lessons(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    # Stable: sim/mixed error_fix and twin_note first; world procedures keep FTS order.
    return sorted(records, key = lambda rec: 0 if _is_sim_lesson(rec) else 1)


def _cap_twin_notes(records: list[dict[str, Any]], max_twin_notes: int) -> list[dict[str, Any]]:
    twins = [rec for rec in records if rec.get("kind") == "twin_note"]
    if len(twins) <= max_twin_notes:
        return records
    keep_ids = {
        rec["id"]
        for rec in sorted(twins, key = lambda rec: rec.get("updated_at") or "", reverse = True)[
            :max_twin_notes
        ]
    }
    return [rec for rec in records if rec.get("kind") != "twin_note" or rec["id"] in keep_ids]


def _age_note(updated_at: Optional[str]) -> str:
    if not updated_at:
        return ""
    try:
        when = datetime.fromisoformat(updated_at)
        if when.tzinfo is None:
            when = when.replace(tzinfo = timezone.utc)
        days = (datetime.now(timezone.utc) - when).days
    except ValueError:
        return ""
    if days >= STALE_AGE_DAYS:
        return f" (last updated {days}d ago — verify)"
    return ""


def _clip_text(text: str, limit: int) -> str:
    if limit <= 0:
        return ""
    if len(text) <= limit:
        return text
    if limit <= len(_ELLIPSIS):
        return text[:limit]
    return text[: limit - len(_ELLIPSIS)] + _ELLIPSIS


def _record_block(rec: dict[str, Any], snippet_chars: int) -> str:
    age = _age_note(rec.get("updated_at"))
    speaker = speaker_of(rec)
    line = (
        f"- [{rec['id'][:8]}] ({rec['kind']}, {rec['provenance']}, {speaker}) "
        f"{rec['title']}{age}"
    )
    body = (rec.get("body") or "").strip()
    if not body:
        return line
    return f"{line}\n  {_clip_text(body, snippet_chars)}"


def format_inject(records: list[dict[str, Any]], *, policy: Optional[RetrievePolicy] = None) -> str:
    policy = policy or RetrievePolicy()
    if not records:
        return ""
    blocks: list[str] = []
    used = 0
    for index, rec in enumerate(records):
        block = _record_block(rec, policy.snippet_chars)
        if index == 0:
            # Keep the first hit even when it exceeds the budget; clip it.
            if len(block) > policy.max_chars:
                block = _clip_text(block, policy.max_chars)
            blocks.append(block)
            used = len(block)
            continue
        extra = 1 + len(block)
        if used + extra > policy.max_chars:
            break
        blocks.append(block)
        used += extra
    return _INJECT_HEADER + "\n" + "\n".join(blocks)
