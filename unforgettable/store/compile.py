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

"""Membership cache for compiled procedures. Standing text is formatted live from B."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

from unforgettable.eyes.probes import is_probe_title
from unforgettable.rims.detect import TEST_COMMAND_TITLE
from unforgettable.store.db import get_connection
from unforgettable.store.records import get_record, list_records
from unforgettable.store.titles import normalize_title

COMPILE_MIN_HITS = 2
COMPILE_PROVENANCE = frozenset({"world", "mixed", "human"})
COMPILE_BODY_CHARS = 800
STANDING_MAX_RECORDS = 4
STANDING_MAX_CHARS = 1600
STANDING_HEADER = "Standing procedures (compiled from B; Source: is the record id):"
_ELLIPSIS = "..."


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _row_to_dict(row) -> dict[str, Any]:
    return dict(row)


def procedure_hits(record_id: str, *, db_path = None) -> int:
    conn = get_connection(db_path)
    try:
        row = conn.execute(
            """
            SELECT COUNT(DISTINCT ru.episode_id)
            FROM retrieve_uses ru
            JOIN rollouts ro ON ro.episode_id = ru.episode_id
            WHERE ru.record_id = ?
              AND ru.contact = 'world'
              AND ro.contact = 'world'
              AND ro.outcome = 'pass'
            """,
            (record_id,),
        ).fetchone()
        return int(row[0]) if row else 0
    finally:
        conn.close()


def _refusal_reason(rec: Optional[dict[str, Any]], *, hits: int, explicit: bool) -> Optional[str]:
    if rec is None:
        return "unknown record"
    if rec.get("kind") != "procedure":
        return "not a procedure"
    if rec.get("status") != "active":
        return "not active"
    if rec.get("provenance") not in COMPILE_PROVENANCE:
        return f"untrusted provenance: {rec.get('provenance')}"
    if is_probe_title(rec.get("title") or ""):
        return "probe procedures cannot compile"
    if normalize_title(rec.get("title") or "") == TEST_COMMAND_TITLE:
        return "test command cannot compile"
    if not explicit and hits < COMPILE_MIN_HITS:
        return f"not enough hits ({hits} < {COMPILE_MIN_HITS})"
    return None


def is_compile_candidate(rec: Optional[dict], *, hits: int, explicit: bool) -> bool:
    return _refusal_reason(rec, hits = hits, explicit = explicit) is None


def get_compiled(record_id: str, *, db_path = None) -> Optional[dict[str, Any]]:
    conn = get_connection(db_path)
    try:
        row = conn.execute(
            "SELECT * FROM compiled WHERE source_record_id = ?", (record_id,)
        ).fetchone()
        return _row_to_dict(row) if row else None
    finally:
        conn.close()


def refresh_compiled(db_path = None) -> list[str]:
    conn = get_connection(db_path)
    try:
        rows = [_row_to_dict(r) for r in conn.execute("SELECT * FROM compiled").fetchall()]
    finally:
        conn.close()
    dropped: list[str] = []
    for row in rows:
        rid = row["source_record_id"]
        rec = get_record(rid, db_path = db_path)
        explicit = bool(row["explicit"])
        hits = procedure_hits(rid, db_path = db_path)
        if not is_compile_candidate(rec, hits = hits, explicit = explicit):
            unpin_compiled(rid, db_path = db_path)
            dropped.append(rid)
    return dropped


def _is_blocked(record_id: str, *, db_path = None) -> bool:
    conn = get_connection(db_path)
    try:
        row = conn.execute(
            "SELECT 1 FROM compiled_blocked WHERE source_record_id = ?",
            (record_id,),
        ).fetchone()
        return row is not None
    finally:
        conn.close()


def maybe_compile(db_path = None) -> list[str]:
    refresh_compiled(db_path)
    pinned: list[str] = []
    for rec in list_records(kinds = ["procedure"], statuses = ["active"], db_path = db_path):
        rid = rec["id"]
        if get_compiled(rid, db_path = db_path) is not None:
            continue
        if _is_blocked(rid, db_path = db_path):
            continue
        hits = procedure_hits(rid, db_path = db_path)
        if not is_compile_candidate(rec, hits = hits, explicit = False):
            continue
        pin_compiled(rid, explicit = False, db_path = db_path)
        pinned.append(rid)
    return pinned


def pin_compiled(
    record_id: str,
    *,
    explicit: bool = False,
    db_path = None,
) -> dict[str, Any]:
    rec = get_record(record_id, db_path = db_path)
    hits = procedure_hits(record_id, db_path = db_path)
    reason = _refusal_reason(rec, hits = hits, explicit = explicit)
    if reason:
        raise ValueError(f"cannot compile {record_id}: {reason}")
    existing = get_compiled(record_id, db_path = db_path)
    if existing is not None:
        if explicit and not existing["explicit"]:
            conn = get_connection(db_path)
            try:
                conn.execute(
                    "UPDATE compiled SET explicit = 1 WHERE source_record_id = ?",
                    (record_id,),
                )
                conn.commit()
            finally:
                conn.close()
        found = get_compiled(record_id, db_path = db_path)
        if found is None:
            raise RuntimeError("compiled pin did not persist")
        return found
    conn = get_connection(db_path)
    try:
        conn.execute("DELETE FROM compiled_blocked WHERE source_record_id = ?", (record_id,))
        conn.execute(
            "INSERT INTO compiled(source_record_id, explicit, compiled_at) VALUES(?,?,?)",
            (record_id, 1 if explicit else 0, _now()),
        )
        conn.commit()
    finally:
        conn.close()
    found = get_compiled(record_id, db_path = db_path)
    if found is None:
        raise RuntimeError("compiled pin did not persist")
    return found


def unpin_compiled(record_id: str, *, db_path = None) -> None:
    conn = get_connection(db_path)
    try:
        conn.execute("DELETE FROM compiled WHERE source_record_id = ?", (record_id,))
        conn.execute(
            "INSERT OR IGNORE INTO compiled_blocked(source_record_id) VALUES(?)",
            (record_id,),
        )
        conn.commit()
    finally:
        conn.close()


def _load_members(db_path = None) -> list[dict[str, Any]]:
    conn = get_connection(db_path)
    try:
        rows = [_row_to_dict(r) for r in conn.execute("SELECT * FROM compiled").fetchall()]
    finally:
        conn.close()
    members: list[dict[str, Any]] = []
    for row in rows:
        rec = get_record(row["source_record_id"], db_path = db_path)
        if rec is None:
            continue
        members.append(
            {
                **rec,
                "hits": procedure_hits(rec["id"], db_path = db_path),
                "compiled_at": row["compiled_at"],
                "explicit": row["explicit"],
            }
        )
    members.sort(
        key = lambda rec: (int(rec.get("hits") or 0), rec.get("compiled_at") or ""),
        reverse = True,
    )
    return members


def count_compiled(db_path = None) -> int:
    """Membership count only. Does not refresh or unpin."""
    conn = get_connection(db_path)
    try:
        row = conn.execute("SELECT COUNT(*) FROM compiled").fetchone()
        return int(row[0]) if row else 0
    finally:
        conn.close()


def list_compiled(db_path = None) -> list[dict[str, Any]]:
    refresh_compiled(db_path)
    return _load_members(db_path)


def list_standing(db_path = None) -> list[dict[str, Any]]:
    return list_compiled(db_path)[:STANDING_MAX_RECORDS]


def _clip_text(text: str, limit: int) -> str:
    if limit <= 0:
        return ""
    if len(text) <= limit:
        return text
    if limit <= len(_ELLIPSIS):
        return text[:limit]
    return text[: limit - len(_ELLIPSIS)] + _ELLIPSIS


def _section_parts(rec: dict[str, Any], *, body_chars: int) -> tuple[str, str, str]:
    rid = rec["id"]
    heading = f"### [{rid[:8]}] {rec.get('title') or ''}"
    body = _clip_text((rec.get("body") or "").strip(), body_chars)
    return heading, body, f"Source: {rid}"


def _join_section(heading: str, body: str, source: str) -> str:
    lines = [heading]
    if body:
        lines.append(body)
    lines.append("")
    lines.append(source)
    return "\n".join(lines)


def _standing_section(rec: dict[str, Any]) -> str:
    heading, body, source = _section_parts(rec, body_chars = COMPILE_BODY_CHARS)
    return _join_section(heading, body, source)


def pack_standing(
    rows: list[dict[str, Any]], *, max_chars: int = STANDING_MAX_CHARS
) -> tuple[str, list[dict[str, Any]]]:
    if not rows:
        return "", []
    heading, _, source = _section_parts(rows[0], body_chars = COMPILE_BODY_CHARS)
    leftover = max_chars - len(STANDING_HEADER) - 1
    reserved = len(heading) + 3 + len(source)
    if leftover > reserved:
        body = _clip_text(
            (rows[0].get("body") or "").strip(),
            min(COMPILE_BODY_CHARS, leftover - reserved),
        )
    else:
        body = ""
    first = _join_section(heading, body, source)
    blocks = [first]
    kept = [rows[0]]
    used = len(STANDING_HEADER) + 1 + len(first)
    for rec in rows[1:]:
        section = _standing_section(rec)
        extra = 2 + len(section)
        if used + extra > max_chars:
            break
        blocks.append(section)
        kept.append(rec)
        used += extra
    return STANDING_HEADER + "\n" + "\n\n".join(blocks), kept


def format_standing(rows: list[dict[str, Any]], *, max_chars: int = STANDING_MAX_CHARS) -> str:
    text, _ = pack_standing(rows, max_chars = max_chars)
    return text
