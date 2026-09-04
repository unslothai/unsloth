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

"""Graded rollout retrieve. Episode FTS + join; never injects episode bodies."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

from unforgettable.store.records import get_record, list_records, list_rollouts
from unforgettable.store.search import search_records

TRAJECTORY_MAX_ROWS = 2
TRAJECTORY_MAX_CHARS = 400
TRAJECTORY_OVERFETCH = 8
TRAJECTORY_HEADER = "Prior rollouts:"
_ELLIPSIS = "..."


def _clip_text(text: str, limit: int) -> str:
    if limit <= 0:
        return ""
    if len(text) <= limit:
        return text
    if limit <= len(_ELLIPSIS):
        return text[:limit]
    return text[: limit - len(_ELLIPSIS)] + _ELLIPSIS


def _created_ts(row: dict[str, Any]) -> float:
    raw = row.get("created_at") or ""
    try:
        when = datetime.fromisoformat(raw)
    except ValueError:
        return 0.0
    if when.tzinfo is None:
        when = when.replace(tzinfo = timezone.utc)
    return when.timestamp()


def _outcome_rank(row: dict[str, Any], contact: str) -> int:
    preferred = "pass" if contact == "world" else "fail"
    return 0 if row.get("outcome") == preferred else 1


def _sort_key(row: dict[str, Any], contact: str) -> tuple:
    return (
        0 if row.get("contact") == contact else 1,
        _outcome_rank(row, contact),
        -_created_ts(row),
    )


def _flatten(rollout: dict[str, Any], episode: Optional[dict[str, Any]]) -> dict[str, Any]:
    return {
        **rollout,
        "episode_title": (episode or {}).get("title") or "",
        "episode_record_id": (episode or {}).get("id"),
        "episode_provenance": (episode or {}).get("provenance") or "",
    }


def _episode_index(db_path) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for rec in list_records(kinds = ["episode"], statuses = ["active"], db_path = db_path):
        index.setdefault(rec["id"], rec)
        src = rec.get("source_episode_id")
        if src:
            index.setdefault(src, rec)
    return index


def retrieve_trajectories(
    query: str,
    *,
    contact: str = "world",
    high_stakes: bool = False,
    max_rows: int = TRAJECTORY_MAX_ROWS,
    db_path = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if (query or "").strip():
        hits = search_records(
            query,
            top_k = TRAJECTORY_OVERFETCH,
            kinds = ["episode"],
            statuses = ["active"],
            db_path = db_path,
        )
        for hit in hits:
            episode_id = hit.get("source_episode_id") or hit["id"]
            for rollout in list_rollouts(episode_id = episode_id, db_path = db_path):
                rows.append(_flatten(rollout, hit))
    else:
        episodes = _episode_index(db_path)
        for rollout in list_rollouts(limit = TRAJECTORY_OVERFETCH, db_path = db_path):
            ep = episodes.get(rollout.get("episode_id") or "")
            if ep is None:
                ep = get_record(rollout.get("episode_id") or "", db_path = db_path)
                if ep is not None and ep.get("kind") != "episode":
                    ep = None
            rows.append(_flatten(rollout, ep))
    if high_stakes:
        rows = [row for row in rows if row.get("contact") != "sim"]
    rows.sort(key = lambda row: _sort_key(row, contact))
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for row in rows:
        rid = row.get("id")
        if not rid or rid in seen:
            continue
        seen.add(rid)
        out.append(row)
        if len(out) >= max_rows:
            break
    return out


def format_trajectories(
    rows: list[dict[str, Any]], *, max_chars: int = TRAJECTORY_MAX_CHARS
) -> str:
    if not rows:
        return ""
    lines = [TRAJECTORY_HEADER]
    for row in rows:
        ep = (row.get("episode_id") or "")[:8]
        contact = row.get("contact") or ""
        outcome = row.get("outcome") or ""
        summary = (row.get("summary") or "").replace("\n", " ").strip()
        lines.append(f"- [{ep}] {contact}/{outcome}: {summary}")
    text = "\n".join(lines)
    if len(text) <= max_chars:
        return text
    if max_chars <= len(TRAJECTORY_HEADER):
        return TRAJECTORY_HEADER
    return _clip_text(text, max_chars)
