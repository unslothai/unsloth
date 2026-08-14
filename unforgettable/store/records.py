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
    STATUSES,
)

from .db import get_connection

DEFAULT_ADMISSIONS_LIMIT = 50


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _row_to_dict(row) -> dict[str, Any]:
    return dict(row)


def ensure_default_namespace(db_path=None) -> dict[str, Any]:
    existing = get_namespace(DEFAULT_NAMESPACE_ID, db_path=db_path)
    if existing:
        return existing
    return create_namespace(
        namespace_id=DEFAULT_NAMESPACE_ID,
        name=DEFAULT_NAMESPACE_NAME,
        admission="auto",
        db_path=db_path,
    )


def create_namespace(
    *,
    name: str,
    admission: str = "auto",
    namespace_id: Optional[str] = None,
    db_path=None,
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
    found = get_namespace(ns_id, db_path=db_path)
    if found is None:
        raise RuntimeError("namespace insert did not persist")
    return found


def get_namespace(namespace_id: str, db_path=None) -> Optional[dict[str, Any]]:
    conn = get_connection(db_path)
    try:
        row = conn.execute(
            "SELECT * FROM namespaces WHERE id = ?", (namespace_id,)
        ).fetchone()
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
    record_id: Optional[str] = None,
    db_path=None,
) -> dict[str, Any]:
    if kind not in KINDS:
        raise ValueError(f"unknown kind: {kind}")
    if status not in STATUSES:
        raise ValueError(f"unknown status: {status}")
    if provenance not in PROVENANCES:
        raise ValueError(f"unknown provenance: {provenance}")
    ensure_default_namespace(db_path=db_path)
    ns = namespace_id or DEFAULT_NAMESPACE_ID
    if get_namespace(ns, db_path=db_path) is None:
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
                created_at, updated_at
            ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)
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
                contact_tag or provenance,
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
    found = get_record(rid, db_path=db_path)
    if found is None:
        raise RuntimeError("record insert did not persist")
    return found


def get_record(record_id: str, db_path=None) -> Optional[dict[str, Any]]:
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
    db_path=None,
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
    if limit is not None:
        sql += " LIMIT ?"
        args.append(limit)
    conn = get_connection(db_path)
    try:
        rows = conn.execute(sql, args).fetchall()
        return [_row_to_dict(r) for r in rows]
    finally:
        conn.close()


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
    db_path=None,
) -> dict[str, Any]:
    old = get_record(record_id, db_path=db_path)
    if old is None:
        raise KeyError(record_id)
    conn = get_connection(db_path)
    try:
        conn.execute(
            "UPDATE records SET status = 'superseded', updated_at = ? WHERE id = ?",
            (_now(), record_id),
        )
        conn.commit()
    finally:
        conn.close()
    return insert_record(
        kind=old["kind"],
        title=title if title is not None else old["title"],
        body=body,
        provenance=provenance or old["provenance"],
        status=status,
        namespace_id=old["namespace_id"],
        confidence=old.get("confidence"),
        supersedes_id=record_id,
        source_episode_id=source_episode_id or old.get("source_episode_id"),
        contact_tag=provenance or old.get("contact_tag"),
        db_path=db_path,
    )


def deprecate_record(record_id: str, *, reason: Optional[str] = None, db_path=None) -> dict[str, Any]:
    rec = get_record(record_id, db_path=db_path)
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
    found = get_record(record_id, db_path=db_path)
    if found is None:
        raise RuntimeError("deprecate did not persist")
    return found


def log_admission(
    *,
    record_id: Optional[str],
    decision: str,
    reason: str,
    db_path=None,
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
    db_path=None,
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
    db_path=None,
) -> dict[str, Any]:
    if status not in STATUSES:
        raise ValueError(f"unknown status: {status}")
    rec = get_record(record_id, db_path=db_path)
    if rec is None:
        raise KeyError(record_id)
    now = _now()
    body = rec["body"]
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
            record_id=record_id,
            decision=status,
            reason=reason,
            db_path=db_path,
        )
    found = get_record(record_id, db_path=db_path)
    if found is None:
        raise RuntimeError("status update did not persist")
    return found
