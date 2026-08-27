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

import uuid
from datetime import datetime, timezone
from typing import Any, Iterable, Optional

from unforgettable.constants import (
    ADMISSION_MODES,
    DEFAULT_NAMESPACE_ID,
    DEFAULT_NAMESPACE_NAME,
    KINDS,
    PROVENANCES,
    RECORD_BODY_CHARS,
    RECORD_TITLE_CHARS,
    ROLLOUT_SUMMARY_CHARS,
    SPEAKER_LABEL_CHARS,
    SPEAKERS,
    STATUSES,
    WARRANT_CHARS,
    coerce_unbacked_user_provenance,
    resolve_speaker,
)

from .db import get_connection

DEFAULT_ADMISSIONS_LIMIT = 50
ROLLOUT_CONTACTS = frozenset({"world", "sim"})
ROLLOUT_OUTCOMES = frozenset({"pass", "fail"})


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _clip(text: str, limit: int) -> str:
    body = "" if text is None else str(text)
    if limit <= 0 or len(body) <= limit:
        return body
    return body[:limit]


def _row_to_dict(row) -> dict[str, Any]:
    return dict(row)


def ensure_default_namespace(db_path = None) -> dict[str, Any]:
    existing = get_namespace(DEFAULT_NAMESPACE_ID, db_path = db_path)
    if existing:
        return existing
    return create_namespace(
        namespace_id = DEFAULT_NAMESPACE_ID,
        name = DEFAULT_NAMESPACE_NAME,
        admission = "auto",
        db_path = db_path,
    )


def create_namespace(
    *,
    name: str,
    admission: str = "auto",
    namespace_id: Optional[str] = None,
    db_path = None,
) -> dict[str, Any]:
    if admission not in ADMISSION_MODES:
        raise ValueError(f"unknown admission mode: {admission}")
    ns_id = namespace_id or str(uuid.uuid4())
    conn = get_connection(db_path)
    try:
        conn.execute(
            "INSERT INTO namespaces(id, name, admission, created_at) VALUES(?,?,?,?)",
            (ns_id, name, admission, _now()),
        )
        conn.commit()
    finally:
        conn.close()
    found = get_namespace(ns_id, db_path = db_path)
    if found is None:
        raise RuntimeError("namespace insert did not persist")
    return found


def get_namespace(namespace_id: str, db_path = None) -> Optional[dict[str, Any]]:
    conn = get_connection(db_path)
    try:
        row = conn.execute("SELECT * FROM namespaces WHERE id = ?", (namespace_id,)).fetchone()
        return _row_to_dict(row) if row else None
    finally:
        conn.close()


def insert_record(
    *,
    kind: str,
    title: str,
    body: str,
    provenance: str,
    status: str = "active",
    namespace_id: Optional[str] = None,
    confidence: Optional[float] = None,
    supersedes_id: Optional[str] = None,
    source_episode_id: Optional[str] = None,
    contact_tag: Optional[str] = None,
    speaker: Optional[str] = None,
    speaker_label: Optional[str] = None,
    warrant: Optional[str] = None,
    record_id: Optional[str] = None,
    db_path = None,
) -> dict[str, Any]:
    if kind not in KINDS:
        raise ValueError(f"unknown kind: {kind}")
    if status not in STATUSES:
        raise ValueError(f"unknown status: {status}")
    if provenance not in PROVENANCES:
        raise ValueError(f"unknown provenance: {provenance}")
    title = _clip(title, RECORD_TITLE_CHARS)
    body = _clip(body, RECORD_BODY_CHARS)
    contact = contact_tag or provenance
    resolved_speaker = resolve_speaker(
        speaker = speaker,
        provenance = provenance,
        kind = kind,
        contact_tag = contact,
    )
    if resolved_speaker not in SPEAKERS:
        raise ValueError(f"unknown speaker: {resolved_speaker}")
    warrant_text = _clip(warrant or "", WARRANT_CHARS)
    provenance = coerce_unbacked_user_provenance(
        provenance, speaker = resolved_speaker, warrant = warrant_text
    )
    label = _clip(speaker_label or "", SPEAKER_LABEL_CHARS) or None
    ensure_default_namespace(db_path = db_path)
    ns = namespace_id or DEFAULT_NAMESPACE_ID
    if get_namespace(ns, db_path = db_path) is None:
        raise ValueError(f"unknown namespace: {ns}")
    rid = record_id or str(uuid.uuid4())
    now = _now()
    conn = get_connection(db_path)
    try:
        conn.execute(
            """
            INSERT INTO records(
                id, namespace_id, kind, status, title, body, provenance,
                confidence, supersedes_id, source_episode_id, contact_tag,
                speaker, speaker_label, warrant, created_at, updated_at
            ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                rid,
                ns,
                kind,
                status,
                title,
                body,
                provenance,
                confidence,
                supersedes_id,
                source_episode_id,
                contact,
                resolved_speaker,
                label,
                warrant_text,
                now,
                now,
            ),
        )
        conn.execute(
            "INSERT INTO record_fts(title, body, record_id) VALUES(?,?,?)",
            (title, body, rid),
        )
        conn.commit()
    finally:
        conn.close()
    found = get_record(rid, db_path = db_path)
    if found is None:
        raise RuntimeError("record insert did not persist")
    return found


def get_record(record_id: str, db_path = None) -> Optional[dict[str, Any]]:
    conn = get_connection(db_path)
    try:
        row = conn.execute("SELECT * FROM records WHERE id = ?", (record_id,)).fetchone()
        return _row_to_dict(row) if row else None
    finally:
        conn.close()


def list_records(
    *,
    namespace_id: Optional[str] = None,
    statuses: Optional[Iterable[str]] = None,
    kinds: Optional[Iterable[str]] = None,
    limit: Optional[int] = None,
    offset: Optional[int] = None,
    db_path = None,
) -> list[dict[str, Any]]:
    clauses = []
    args: list[Any] = []
    if namespace_id:
        clauses.append("namespace_id = ?")
        args.append(namespace_id)
    if statuses:
        status_list = list(statuses)
        clauses.append(f"status IN ({','.join('?' * len(status_list))})")
        args.extend(status_list)
    if kinds:
        kind_list = list(kinds)
        clauses.append(f"kind IN ({','.join('?' * len(kind_list))})")
        args.extend(kind_list)
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    sql = f"SELECT * FROM records {where} ORDER BY updated_at DESC"
    skip = int(offset or 0)
    if skip < 0:
        skip = 0
    if limit is not None:
        sql += " LIMIT ?"
        args.append(limit)
        if skip:
            sql += " OFFSET ?"
            args.append(skip)
    elif skip:
        sql += " LIMIT -1 OFFSET ?"
        args.append(skip)
    conn = get_connection(db_path)
    try:
        rows = conn.execute(sql, args).fetchall()
        return [_row_to_dict(r) for r in rows]
    finally:
        conn.close()


def summarize_records(*, namespace_id: Optional[str] = None, db_path = None) -> dict[str, Any]:
    """Count records by status, kind, and provenance. Empty buckets stay 0."""
    clauses = []
    args: list[Any] = []
    if namespace_id:
        clauses.append("namespace_id = ?")
        args.append(namespace_id)
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    conn = get_connection(db_path)
    try:
        rows = conn.execute(
            f"SELECT status, kind, provenance, COUNT(*) AS n "
            f"FROM records {where} GROUP BY status, kind, provenance",
            args,
        ).fetchall()
        total_row = conn.execute(f"SELECT COUNT(*) FROM records {where}", args).fetchone()
    finally:
        conn.close()
    by_status = {name: 0 for name in STATUSES}
    by_kind = {name: 0 for name in KINDS}
    by_provenance = {name: 0 for name in PROVENANCES}
    cells: list[dict[str, Any]] = []
    for row in rows:
        count = int(row["n"])
        status = row["status"]
        kind = row["kind"]
        provenance = row["provenance"]
        by_status[status] = by_status.get(status, 0) + count
        by_kind[kind] = by_kind.get(kind, 0) + count
        by_provenance[provenance] = by_provenance.get(provenance, 0) + count
        cells.append(
            {
                "status": status,
                "kind": kind,
                "provenance": provenance,
                "count": count,
            }
        )
    return {
        "total": int(total_row[0]) if total_row else 0,
        "by_status": by_status,
        "by_kind": by_kind,
        "by_provenance": by_provenance,
        "cells": cells,
    }


def update_proposed_record(
    record_id: str,
    *,
    title: Optional[str] = None,
    body: Optional[str] = None,
    db_path = None,
) -> dict[str, Any]:
    """In-place title/body edit. Refuses anything that is not proposed."""
    rec = get_record(record_id, db_path = db_path)
    if rec is None:
        raise KeyError(record_id)
    if rec["status"] != "proposed":
        raise ValueError("only proposed records can be edited in place")
    new_title = _clip(title if title is not None else rec["title"], RECORD_TITLE_CHARS)
    new_body = _clip(body if body is not None else rec["body"], RECORD_BODY_CHARS)
    conn = get_connection(db_path)
    try:
        conn.execute(
            "UPDATE records SET title = ?, body = ?, updated_at = ? WHERE id = ?",
            (new_title, new_body, _now(), record_id),
        )
        _rewrite_fts(conn, record_id, new_title, new_body)
        conn.commit()
    finally:
        conn.close()
    found = get_record(record_id, db_path = db_path)
    if found is None:
        raise RuntimeError("proposed update did not persist")
    return found


def _rewrite_fts(conn, record_id: str, title: str, body: str) -> None:
    conn.execute("DELETE FROM record_fts WHERE record_id = ?", (record_id,))
    conn.execute(
        "INSERT INTO record_fts(title, body, record_id) VALUES(?,?,?)",
        (title, body, record_id),
    )


def supersede_record(
    record_id: str,
    *,
    body: str,
    title: Optional[str] = None,
    provenance: Optional[str] = None,
    source_episode_id: Optional[str] = None,
    status: str = "active",
    new_id: Optional[str] = None,
    contact_tag: Optional[str] = None,
    speaker: Optional[str] = None,
    speaker_label: Optional[str] = None,
    warrant: Optional[str] = None,
    db_path = None,
) -> dict[str, Any]:
    old = get_record(record_id, db_path = db_path)
    if old is None:
        raise KeyError(record_id)
    if status not in STATUSES:
        raise ValueError(f"unknown status: {status}")
    new_prov = provenance or old["provenance"]
    if new_prov not in PROVENANCES:
        raise ValueError(f"unknown provenance: {new_prov}")
    contact = contact_tag if contact_tag is not None else new_prov
    resolved_speaker = resolve_speaker(
        speaker = speaker if speaker is not None else old.get("speaker"),
        provenance = new_prov,
        kind = old["kind"],
        contact_tag = contact,
    )
    if resolved_speaker not in SPEAKERS:
        raise ValueError(f"unknown speaker: {resolved_speaker}")
    warrant_text = _clip(
        warrant if warrant is not None else (old.get("warrant") or ""),
        WARRANT_CHARS,
    )
    new_prov = coerce_unbacked_user_provenance(
        new_prov, speaker = resolved_speaker, warrant = warrant_text
    )
    label = speaker_label if speaker_label is not None else old.get("speaker_label")
    label = _clip(label or "", SPEAKER_LABEL_CHARS) or None
    rid = new_id or str(uuid.uuid4())
    now = _now()
    new_title = _clip(title if title is not None else old["title"], RECORD_TITLE_CHARS)
    body = _clip(body, RECORD_BODY_CHARS)
    conn = get_connection(db_path)
    try:
        conn.execute(
            "UPDATE records SET status = 'superseded', updated_at = ? WHERE id = ?",
            (now, record_id),
        )
        conn.execute(
            """
            INSERT INTO records(
                id, namespace_id, kind, status, title, body, provenance,
                confidence, supersedes_id, source_episode_id, contact_tag,
                speaker, speaker_label, warrant, created_at, updated_at
            ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                rid,
                old["namespace_id"],
                old["kind"],
                status,
                new_title,
                body,
                new_prov,
                old.get("confidence"),
                record_id,
                source_episode_id or old.get("source_episode_id"),
                contact,
                resolved_speaker,
                label,
                warrant_text,
                now,
                now,
            ),
        )
        conn.execute(
            "INSERT INTO record_fts(title, body, record_id) VALUES(?,?,?)",
            (new_title, body, rid),
        )
        conn.commit()
    finally:
        conn.close()
    found = get_record(rid, db_path = db_path)
    if found is None:
        raise RuntimeError("supersede did not persist")
    return found


def deprecate_record(
    record_id: str,
    *,
    reason: Optional[str] = None,
    db_path = None,
) -> dict[str, Any]:
    rec = get_record(record_id, db_path = db_path)
    if rec is None:
        raise KeyError(record_id)
    now = _now()
    body = rec["body"]
    if reason:
        body = f"{body}\n\n[deprecated] {reason}"
    conn = get_connection(db_path)
    try:
        conn.execute(
            "UPDATE records SET status = 'deprecated', body = ?, updated_at = ? WHERE id = ?",
            (body, now, record_id),
        )
        _rewrite_fts(conn, record_id, rec["title"], body)
        conn.commit()
    finally:
        conn.close()
    found = get_record(record_id, db_path = db_path)
    if found is None:
        raise RuntimeError("deprecate did not persist")
    return found


def log_admission(
    *,
    record_id: Optional[str],
    decision: str,
    reason: str,
    db_path = None,
) -> None:
    conn = get_connection(db_path)
    try:
        conn.execute(
            "INSERT INTO admissions_log(record_id, decision, reason, created_at) VALUES(?,?,?,?)",
            (record_id, decision, reason, _now()),
        )
        conn.commit()
    finally:
        conn.close()


def list_admissions(
    *,
    limit: int = DEFAULT_ADMISSIONS_LIMIT,
    decision: Optional[str] = None,
    db_path = None,
) -> list[dict[str, Any]]:
    clauses = []
    args: list[Any] = []
    if decision:
        clauses.append("decision = ?")
        args.append(decision)
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    args.append(limit)
    conn = get_connection(db_path)
    try:
        rows = conn.execute(
            f"SELECT * FROM admissions_log {where} ORDER BY created_at DESC, id DESC LIMIT ?",
            args,
        ).fetchall()
        return [_row_to_dict(r) for r in rows]
    finally:
        conn.close()


def set_record_status(
    record_id: str,
    status: str,
    *,
    reason: Optional[str] = None,
    db_path = None,
) -> dict[str, Any]:
    if status not in STATUSES:
        raise ValueError(f"unknown status: {status}")
    rec = get_record(record_id, db_path = db_path)
    if rec is None:
        raise KeyError(record_id)
    now = _now()
    body = rec["body"]
    if status == "active":
        while "\n\n[deprecated]" in body:
            body = body.rsplit("\n\n[deprecated]", 1)[0]
    if reason and status == "deprecated":
        body = f"{body}\n\n[deprecated] {reason}"
    conn = get_connection(db_path)
    try:
        conn.execute(
            "UPDATE records SET status = ?, body = ?, updated_at = ? WHERE id = ?",
            (status, body, now, record_id),
        )
        if body != rec["body"]:
            _rewrite_fts(conn, record_id, rec["title"], body)
        conn.commit()
    finally:
        conn.close()
    if reason is not None:
        log_admission(
            record_id = record_id,
            decision = status,
            reason = reason,
            db_path = db_path,
        )
    found = get_record(record_id, db_path = db_path)
    if found is None:
        raise RuntimeError("status update did not persist")
    return found


def insert_rollout(
    *,
    episode_id: str,
    contact: str,
    outcome: str,
    summary: str,
    source_record_id: Optional[str] = None,
    rollout_id: Optional[str] = None,
    db_path = None,
) -> dict[str, Any]:
    if contact not in ROLLOUT_CONTACTS:
        raise ValueError(f"unknown rollout contact: {contact}")
    if outcome not in ROLLOUT_OUTCOMES:
        raise ValueError(f"unknown rollout outcome: {outcome}")
    summary = _clip(summary, ROLLOUT_SUMMARY_CHARS)
    rid = rollout_id or str(uuid.uuid4())
    now = _now()
    conn = get_connection(db_path)
    try:
        conn.execute(
            """
            INSERT INTO rollouts(
                id, episode_id, contact, outcome, summary, source_record_id, created_at
            ) VALUES(?,?,?,?,?,?,?)
            """,
            (rid, episode_id, contact, outcome, summary, source_record_id, now),
        )
        conn.commit()
    finally:
        conn.close()
    found = _get_rollout(rid, db_path = db_path)
    if found is None:
        raise RuntimeError("rollout insert did not persist")
    return found


def _get_rollout(rollout_id: str, db_path = None) -> Optional[dict[str, Any]]:
    conn = get_connection(db_path)
    try:
        row = conn.execute("SELECT * FROM rollouts WHERE id = ?", (rollout_id,)).fetchone()
        return _row_to_dict(row) if row else None
    finally:
        conn.close()


def list_rollouts(
    *,
    episode_id: Optional[str] = None,
    contact: Optional[str] = None,
    outcome: Optional[str] = None,
    limit: Optional[int] = None,
    db_path = None,
) -> list[dict[str, Any]]:
    clauses: list[str] = []
    args: list[Any] = []
    if episode_id is not None:
        clauses.append("episode_id = ?")
        args.append(episode_id)
    if contact is not None:
        clauses.append("contact = ?")
        args.append(contact)
    if outcome is not None:
        clauses.append("outcome = ?")
        args.append(outcome)
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    # Episode inspect stays chronological; the library path wants newest first.
    order = "ASC" if episode_id is not None else "DESC"
    sql = f"SELECT * FROM rollouts {where} ORDER BY created_at {order}"
    if limit is not None:
        sql += " LIMIT ?"
        args.append(limit)
    conn = get_connection(db_path)
    try:
        rows = conn.execute(sql, args).fetchall()
        return [_row_to_dict(r) for r in rows]
    finally:
        conn.close()


def insert_retrieve_use(
    *,
    episode_id: str,
    record_id: str,
    contact: str,
    use_id: Optional[str] = None,
    db_path = None,
) -> dict[str, Any]:
    if contact not in ROLLOUT_CONTACTS:
        raise ValueError(f"unknown retrieve contact: {contact}")
    rid = use_id or str(uuid.uuid4())
    now = _now()
    conn = get_connection(db_path)
    try:
        conn.execute(
            """
            INSERT INTO retrieve_uses(
                id, episode_id, record_id, contact, created_at
            ) VALUES(?,?,?,?,?)
            """,
            (rid, episode_id, record_id, contact, now),
        )
        conn.commit()
    finally:
        conn.close()
    found = _get_retrieve_use(rid, db_path = db_path)
    if found is None:
        raise RuntimeError("retrieve_use insert did not persist")
    return found


def _get_retrieve_use(use_id: str, db_path = None) -> Optional[dict[str, Any]]:
    conn = get_connection(db_path)
    try:
        row = conn.execute("SELECT * FROM retrieve_uses WHERE id = ?", (use_id,)).fetchone()
        return _row_to_dict(row) if row else None
    finally:
        conn.close()


def insert_inject_stats(
    *,
    episode_id: str,
    contact: str,
    standing_chars: int,
    retrieve_chars: int,
    trajectory_chars: int,
    total_chars: int,
    compiled_ids: str,
    retrieved_ids: str,
    stats_id: Optional[str] = None,
    db_path = None,
) -> dict[str, Any]:
    if contact not in ROLLOUT_CONTACTS:
        raise ValueError(f"unknown inject contact: {contact}")
    rid = stats_id or str(uuid.uuid4())
    now = _now()
    conn = get_connection(db_path)
    try:
        conn.execute(
            """
            INSERT INTO inject_stats(
                id, episode_id, contact, standing_chars, retrieve_chars,
                trajectory_chars, total_chars, compiled_ids, retrieved_ids,
                created_at
            ) VALUES(?,?,?,?,?,?,?,?,?,?)
            """,
            (
                rid,
                episode_id,
                contact,
                standing_chars,
                retrieve_chars,
                trajectory_chars,
                total_chars,
                compiled_ids,
                retrieved_ids,
                now,
            ),
        )
        conn.commit()
    finally:
        conn.close()
    found = _get_inject_stats(rid, db_path = db_path)
    if found is None:
        raise RuntimeError("inject_stats insert did not persist")
    return found


def _get_inject_stats(stats_id: str, db_path = None) -> Optional[dict[str, Any]]:
    conn = get_connection(db_path)
    try:
        row = conn.execute("SELECT * FROM inject_stats WHERE id = ?", (stats_id,)).fetchone()
        return _row_to_dict(row) if row else None
    finally:
        conn.close()


def list_inject_stats(*, limit: int = 20, db_path = None) -> list[dict[str, Any]]:
    conn = get_connection(db_path)
    try:
        rows = conn.execute(
            "SELECT * FROM inject_stats ORDER BY created_at DESC, id DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [_row_to_dict(r) for r in rows]
    finally:
        conn.close()


def list_retrieve_uses(
    *,
    episode_id: Optional[str] = None,
    limit: int = 50,
    db_path = None,
) -> list[dict[str, Any]]:
    clauses = []
    args: list[Any] = []
    if episode_id:
        clauses.append("episode_id = ?")
        args.append(episode_id)
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    args.append(limit)
    conn = get_connection(db_path)
    try:
        rows = conn.execute(
            f"SELECT * FROM retrieve_uses {where} ORDER BY created_at ASC, id ASC LIMIT ?",
            args,
        ).fetchall()
        return [_row_to_dict(r) for r in rows]
    finally:
        conn.close()
