# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Flows, their runs, replayable run events, node results, the output cache and player saves in the account's studio.db."""

from __future__ import annotations

import json
import sqlite3
import time
from typing import Any, Optional

from storage.studio_db import get_connection

ACTIVE_RUN_STATUSES = ("queued", "running", "cancelling")
TERMINAL_RUN_STATUSES = ("cancelled", "completed", "failed", "interrupted")


def now_ms() -> int:
    return int(time.time() * 1000)


class FlowDeleted(Exception):
    pass


class FlowConflict(Exception):
    pass


def _flow_from_row(row: sqlite3.Row) -> dict:
    return {
        "id": row["id"],
        "name": row["name"],
        "doc": json.loads(row["doc_json"]),
        "createdAt": row["created_at"],
        "updatedAt": row["updated_at"],
    }


def _summary_from_row(row: sqlite3.Row) -> dict:
    doc = json.loads(row["doc_json"])
    meta = doc.get("meta") if isinstance(doc.get("meta"), dict) else {}
    cover = meta.get("cover")
    if isinstance(cover, dict):
        cover = cover.get("url")
    return {
        "id": row["id"],
        "name": row["name"],
        "kind": row["kind"],
        "description": doc.get("description") or "",
        "createdAt": row["created_at"],
        "updatedAt": row["updated_at"],
        "cover": cover if isinstance(cover, str) else None,
        "nodeCount": len(doc.get("nodes") or []),
    }


_SUMMARY_SQL = """
    SELECT id, name, kind, created_at, updated_at,
           json_extract(doc_json, '$.description') AS description,
           json_extract(doc_json, '$.meta.cover') AS cover,
           json_type(doc_json, '$.meta.cover') AS cover_type,
           json_array_length(doc_json, '$.nodes') AS node_count
    FROM flows ORDER BY updated_at DESC
"""


def list_flows() -> list[dict]:
    conn = get_connection()
    try:
        try:
            # The gallery needs four fields: read them in SQLite instead of parsing every document (baked outputs included).
            rows = conn.execute(_SUMMARY_SQL).fetchall()
        except sqlite3.OperationalError:
            rows = conn.execute("SELECT * FROM flows ORDER BY updated_at DESC").fetchall()
            return [_summary_from_row(r) for r in rows]
        return [_summary_from_columns(r) for r in rows]
    finally:
        conn.close()


def _summary_from_columns(row: sqlite3.Row) -> dict:
    cover = row["cover"]
    if row["cover_type"] == "object":
        cover = json.loads(cover).get("url")
    elif row["cover_type"] != "text":
        cover = None
    description = row["description"]
    return {
        "id": row["id"],
        "name": row["name"],
        "kind": row["kind"],
        "description": description if isinstance(description, str) and description else "",
        "createdAt": row["created_at"],
        "updatedAt": row["updated_at"],
        "cover": cover if isinstance(cover, str) else None,
        "nodeCount": row["node_count"] or 0,
    }


def recipe_flows() -> dict[str, str]:
    """Saved Data Recipe id -> the flow it was brought into (`meta.from_recipe.id`)."""
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT id, json_extract(doc_json, '$.meta.from_recipe.id') AS recipe FROM flows"
            " WHERE json_extract(doc_json, '$.meta.from_recipe.id') IS NOT NULL"
            " ORDER BY created_at"
        ).fetchall()
        found: dict[str, str] = {}
        for row in rows:
            found.setdefault(str(row["recipe"]), row["id"])
        return found
    finally:
        conn.close()


def get_flow(flow_id: str) -> Optional[dict]:
    conn = get_connection()
    try:
        row = conn.execute("SELECT * FROM flows WHERE id = ?", (flow_id,)).fetchone()
        return _flow_from_row(row) if row else None
    finally:
        conn.close()


def flow_deleted(flow_id: str) -> bool:
    conn = get_connection()
    try:
        return (
            conn.execute("SELECT 1 FROM flow_tombstones WHERE id = ?", (flow_id,)).fetchone()
            is not None
        )
    finally:
        conn.close()


def save_flow(doc: dict, base_updated_at: Optional[int] = None) -> dict:
    """Insert or update ``doc``. ``base_updated_at`` is the version the caller edited: a newer stored row is a conflict."""
    now = now_ms()
    conn = get_connection()
    try:
        prior = conn.execute("SELECT updated_at FROM flows WHERE id = ?", (doc["id"],)).fetchone()
        updated_at = max(now, prior["updated_at"] + 1) if prior else now
        cur = conn.execute(
            """
            INSERT INTO flows (id, name, kind, doc_json, created_at, updated_at)
            SELECT ?, ?, ?, ?, ?, ?
            WHERE NOT EXISTS (SELECT 1 FROM flow_tombstones WHERE id = ?1)
            ON CONFLICT(id) DO UPDATE SET
                name = excluded.name,
                kind = excluded.kind,
                doc_json = excluded.doc_json,
                updated_at = excluded.updated_at
            WHERE ?7 IS NULL OR flows.updated_at = ?7
            """,
            (
                doc["id"],
                doc.get("name") or "Untitled flow",
                doc.get("kind") or "pipeline",
                json.dumps(doc),
                now,
                updated_at,
                base_updated_at,
            ),
        )
        row = conn.execute("SELECT * FROM flows WHERE id = ?", (doc["id"],)).fetchone()
        conn.commit()
        if cur.rowcount:
            return _flow_from_row(row)
        if row is None:
            raise FlowDeleted(doc["id"])
        raise FlowConflict(doc["id"])
    finally:
        conn.close()


def delete_flow(flow_id: str) -> None:
    conn = get_connection()
    try:
        conn.execute("DELETE FROM flows WHERE id = ?", (flow_id,))
        conn.execute("DELETE FROM flow_sessions WHERE flow_id = ?", (flow_id,))
        conn.execute("DELETE FROM flow_capabilities WHERE flow_id = ?", (flow_id,))
        conn.execute("DELETE FROM flow_triggers WHERE flow_id = ?", (flow_id,))
        conn.execute("DELETE FROM flow_runs WHERE flow_id = ?", (flow_id,))
        conn.execute(
            "INSERT OR IGNORE INTO flow_tombstones (id, deleted_at) VALUES (?, ?)",
            (flow_id, now_ms()),
        )
        conn.commit()
    finally:
        conn.close()


def _run_from_row(row: sqlite3.Row) -> dict:
    return {
        "id": row["id"],
        "flow_id": row["flow_id"],
        "status": row["status"],
        "cancel_requested": bool(row["cancel_requested"]),
        "options": json.loads(row["options_json"] or "{}"),
        "outputs": json.loads(row["outputs_json"]) if row["outputs_json"] else {},
        "error": row["error"],
        "last_seq": row["last_event_seq"],
        "created_at": row["created_at"],
        "started_at": row["started_at"],
        "finished_at": row["finished_at"],
    }


def create_run(run_id: str, flow_id: str, options: dict) -> dict:
    conn = get_connection()
    try:
        conn.execute(
            "INSERT INTO flow_runs (id, flow_id, status, options_json, created_at) VALUES (?, ?, 'queued', ?, ?)",
            (run_id, flow_id, json.dumps(options), now_ms()),
        )
        conn.commit()
        row = conn.execute("SELECT * FROM flow_runs WHERE id = ?", (run_id,)).fetchone()
        return _run_from_row(row)
    finally:
        conn.close()


def get_run(run_id: str) -> Optional[dict]:
    conn = get_connection()
    try:
        row = conn.execute("SELECT * FROM flow_runs WHERE id = ?", (run_id,)).fetchone()
        return _run_from_row(row) if row else None
    finally:
        conn.close()


def list_runs(flow_id: str, limit: int = 50) -> list[dict]:
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT * FROM flow_runs WHERE flow_id = ? ORDER BY created_at DESC LIMIT ?",
            (flow_id, limit),
        ).fetchall()
        return [_run_from_row(r) for r in rows]
    finally:
        conn.close()


def latest_run(flow_id: str) -> Optional[dict]:
    runs = list_runs(flow_id, limit = 1)
    return runs[0] if runs else None


def update_run(run_id: str, **fields: Any) -> None:
    columns = {
        "status": "status",
        "cancel_requested": "cancel_requested",
        "outputs": "outputs_json",
        "error": "error",
        "started_at": "started_at",
        "finished_at": "finished_at",
    }
    sets, values = [], []
    for key, value in fields.items():
        column = columns[key]
        if key == "outputs":
            value = json.dumps(value)
        elif key == "cancel_requested":
            value = 1 if value else 0
        sets.append(f"{column} = ?")
        values.append(value)
    if not sets:
        return
    conn = get_connection()
    try:
        # A terminal status is final: a late writer never reopens it.
        conn.execute(
            f"UPDATE flow_runs SET {', '.join(sets)} WHERE id = ? AND status NOT IN (?, ?, ?, ?)",
            (*values, run_id, *TERMINAL_RUN_STATUSES),
        )
        conn.commit()
    finally:
        conn.close()


def append_event(run_id: str, seq: int, event: dict) -> None:
    conn = get_connection()
    try:
        conn.execute(
            "INSERT OR IGNORE INTO flow_run_events (run_id, seq, event_json) VALUES (?, ?, ?)",
            (run_id, seq, json.dumps(event)),
        )
        conn.execute(
            "UPDATE flow_runs SET last_event_seq = MAX(last_event_seq, ?) WHERE id = ?",
            (seq, run_id),
        )
        conn.commit()
    finally:
        conn.close()


def write_batch(
    run_id: str,
    events: list[tuple[int, dict]],
    results: list[tuple[str, str, Optional[dict], Optional[str]]],
    cache_rows: list[tuple[str, str, dict]],
    touched: tuple[str, ...] = (),
) -> None:
    """A run's buffered events, node results and cache entries in one transaction (one per flush, not per write)."""
    if not (events or results or cache_rows or touched):
        return
    now = now_ms()
    conn = get_connection()
    try:
        if events:
            conn.executemany(
                "INSERT OR IGNORE INTO flow_run_events (run_id, seq, event_json) VALUES (?, ?, ?)",
                [(run_id, seq, json.dumps(event)) for seq, event in events],
            )
            conn.execute(
                "UPDATE flow_runs SET last_event_seq = MAX(last_event_seq, ?) WHERE id = ?",
                (max(seq for seq, _ in events), run_id),
            )
        if results:
            conn.executemany(
                """
                INSERT INTO flow_node_results (run_id, node_id, status, outputs_json, error, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id, node_id) DO UPDATE SET
                    status = excluded.status, outputs_json = excluded.outputs_json,
                    error = excluded.error, updated_at = excluded.updated_at
                """,
                [
                    (
                        run_id,
                        nid,
                        status,
                        json.dumps(outputs) if outputs is not None else None,
                        error,
                        now,
                    )
                    for nid, status, outputs, error in results
                ],
            )
        if cache_rows:
            conn.executemany(
                """
                INSERT INTO flow_cache (key, node_type, outputs_json, created_at, last_used_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET outputs_json = excluded.outputs_json, last_used_at = excluded.last_used_at
                """,
                [
                    (key, node_type, json.dumps(outputs), now, now)
                    for key, node_type, outputs in cache_rows
                ],
            )
        conn.commit()
    finally:
        conn.close()


def events_after(
    run_id: str,
    after_seq: int,
    limit: int = 5000,
) -> list[dict]:
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT event_json FROM flow_run_events WHERE run_id = ? AND seq > ? ORDER BY seq LIMIT ?",
            (run_id, after_seq, limit),
        ).fetchall()
        return [json.loads(r["event_json"]) for r in rows]
    finally:
        conn.close()


def save_node_result(
    run_id: str, node_id: str, status: str, outputs: Optional[dict], error: Optional[str]
) -> None:
    conn = get_connection()
    try:
        conn.execute(
            """
            INSERT INTO flow_node_results (run_id, node_id, status, outputs_json, error, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(run_id, node_id) DO UPDATE SET
                status = excluded.status, outputs_json = excluded.outputs_json,
                error = excluded.error, updated_at = excluded.updated_at
            """,
            (
                run_id,
                node_id,
                status,
                json.dumps(outputs) if outputs is not None else None,
                error,
                now_ms(),
            ),
        )
        conn.commit()
    finally:
        conn.close()


def node_results(run_id: str) -> dict[str, dict]:
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT node_id, status, outputs_json, error FROM flow_node_results WHERE run_id = ?",
            (run_id,),
        ).fetchall()
        return {
            r["node_id"]: {
                "status": r["status"],
                "outputs": json.loads(r["outputs_json"]) if r["outputs_json"] else None,
                "error": r["error"],
            }
            for r in rows
        }
    finally:
        conn.close()


def latest_node_outputs(flow_id: str) -> dict[str, dict]:
    """For each node, its newest successful output in any of the flow's runs (a side branch run keeps the rest)."""
    conn = get_connection()
    try:
        rows = conn.execute(
            """
            SELECT node_id, status, outputs_json FROM (
                SELECT r.node_id, r.status, r.outputs_json, ROW_NUMBER() OVER (
                    PARTITION BY r.node_id ORDER BY f.created_at DESC, r.updated_at DESC
                ) AS newest
                FROM flow_node_results r JOIN flow_runs f ON f.id = r.run_id
                WHERE f.flow_id = ? AND r.status IN ('done', 'cached') AND r.outputs_json IS NOT NULL
            ) WHERE newest = 1
            """,
            (flow_id,),
        ).fetchall()
        return {
            r["node_id"]: {"status": r["status"], "outputs": json.loads(r["outputs_json"])}
            for r in rows
        }
    finally:
        conn.close()


def node_output_history(
    flow_id: str,
    node_id: str,
    limit: int = 10,
) -> list[dict]:
    """A node's last ``limit`` distinct successful outputs across the flow's runs, newest first.

    Outputs hold media refs, never bytes, so the history costs no storage of its own."""
    conn = get_connection()
    seen: set[str] = set()
    entries: list[dict] = []
    try:
        # Rows stream newest first until ``limit`` distinct outputs are found: cached reruns repeat one output many times.
        rows = conn.execute(
            """
            SELECT r.run_id, r.status, r.outputs_json, r.updated_at
            FROM flow_node_results r JOIN flow_runs f ON f.id = r.run_id
            WHERE f.flow_id = ? AND r.node_id = ? AND r.status IN ('done', 'cached')
                AND r.outputs_json IS NOT NULL
            ORDER BY f.created_at DESC, f.rowid DESC, r.updated_at DESC
            """,
            (flow_id, node_id),
        )
        for r in rows:
            if r["outputs_json"] in seen:
                continue
            seen.add(r["outputs_json"])
            entries.append(
                {
                    "run_id": r["run_id"],
                    "updated_at": r["updated_at"],
                    "outputs": json.loads(r["outputs_json"]),
                }
            )
            if len(entries) >= limit:
                break
    finally:
        conn.close()
    return entries


def reconcile_orphaned_runs(
    error: str = "Unsloth Studio restarted while this flow was running",
) -> int:
    """Runs a previous process left active can never finish: mark them interrupted (never re-run them)."""
    conn = get_connection()
    try:
        cur = conn.execute(
            f"UPDATE flow_runs SET status = 'interrupted', error = ?, finished_at = ?"
            f" WHERE status IN ({', '.join('?' * len(ACTIVE_RUN_STATUSES))})",
            (error, now_ms(), *ACTIVE_RUN_STATUSES),
        )
        conn.commit()
        return cur.rowcount
    finally:
        conn.close()


def cache_get(key: str, touch: bool = True) -> Optional[dict]:
    conn = get_connection()
    try:
        row = conn.execute("SELECT outputs_json FROM flow_cache WHERE key = ?", (key,)).fetchone()
        if row is None:
            return None
        if touch:
            conn.execute("UPDATE flow_cache SET last_used_at = ? WHERE key = ?", (now_ms(), key))
            conn.commit()
        return json.loads(row["outputs_json"])
    finally:
        conn.close()


def cache_put(key: str, node_type: str, outputs: dict) -> None:
    now = now_ms()
    conn = get_connection()
    try:
        conn.execute(
            """
            INSERT INTO flow_cache (key, node_type, outputs_json, created_at, last_used_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(key) DO UPDATE SET outputs_json = excluded.outputs_json, last_used_at = excluded.last_used_at
            """,
            (key, node_type, json.dumps(outputs), now, now),
        )
        conn.commit()
    finally:
        conn.close()


def cache_clear() -> int:
    conn = get_connection()
    try:
        cur = conn.execute("DELETE FROM flow_cache")
        conn.commit()
        return cur.rowcount
    finally:
        conn.close()


def list_sessions(flow_id: str) -> list[dict]:
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT slot, label, updated_at FROM flow_sessions WHERE flow_id = ? ORDER BY updated_at DESC",
            (flow_id,),
        ).fetchall()
        return [
            {"slot": r["slot"], "label": r["label"], "updatedAt": r["updated_at"]} for r in rows
        ]
    finally:
        conn.close()


def get_session(flow_id: str, slot: str) -> Optional[dict]:
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT * FROM flow_sessions WHERE flow_id = ? AND slot = ?", (flow_id, slot)
        ).fetchone()
        if row is None:
            return None
        return {
            "slot": row["slot"],
            "label": row["label"],
            "state": json.loads(row["state_json"]),
            "updatedAt": row["updated_at"],
        }
    finally:
        conn.close()


def put_session(flow_id: str, slot: str, state: Any, label: Optional[str]) -> dict:
    now = now_ms()
    conn = get_connection()
    try:
        conn.execute(
            """
            INSERT INTO flow_sessions (flow_id, slot, label, state_json, updated_at) VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(flow_id, slot) DO UPDATE SET
                label = excluded.label, state_json = excluded.state_json, updated_at = excluded.updated_at
            """,
            (flow_id, slot, label, json.dumps(state), now),
        )
        conn.commit()
        return {"slot": slot, "label": label, "state": state, "updatedAt": now}
    finally:
        conn.close()


def delete_session(flow_id: str, slot: str) -> None:
    conn = get_connection()
    try:
        conn.execute("DELETE FROM flow_sessions WHERE flow_id = ? AND slot = ?", (flow_id, slot))
        conn.commit()
    finally:
        conn.close()


def get_capabilities(flow_id: str) -> dict:
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT approved_json FROM flow_capabilities WHERE flow_id = ?", (flow_id,)
        ).fetchone()
        return json.loads(row["approved_json"]) if row else {}
    finally:
        conn.close()


def set_capabilities(flow_id: str, approved: dict) -> None:
    conn = get_connection()
    try:
        conn.execute(
            "INSERT INTO flow_capabilities (flow_id, approved_json, updated_at) VALUES (?, ?, ?) "
            "ON CONFLICT(flow_id) DO UPDATE SET approved_json = excluded.approved_json, "
            "updated_at = excluded.updated_at",
            (flow_id, json.dumps(approved), now_ms()),
        )
        conn.commit()
    finally:
        conn.close()


def _trigger_from_row(row: sqlite3.Row) -> dict:
    return {
        "flow_id": row["flow_id"],
        "node_id": row["node_id"],
        "kind": row["kind"],
        "config": json.loads(row["config_json"]),
        "token": row["token"],
        "last_fired_at": row["last_fired_at"],
    }


def list_triggers(flow_id: Optional[str] = None) -> list[dict]:
    conn = get_connection()
    try:
        if flow_id is None:
            rows = conn.execute("SELECT * FROM flow_triggers").fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM flow_triggers WHERE flow_id = ?", (flow_id,)
            ).fetchall()
        return [_trigger_from_row(r) for r in rows]
    finally:
        conn.close()


def replace_triggers(flow_id: str, wanted: list[dict], new_token: Any) -> None:
    """Make the flow's trigger rows match ``wanted`` ({node_id, kind, config}); kept rows keep their secret and last
    fire time, new webhook rows get ``new_token()``."""
    conn = get_connection()
    try:
        old = {
            r["node_id"]: r
            for r in conn.execute("SELECT * FROM flow_triggers WHERE flow_id = ?", (flow_id,))
        }
        keep = {w["node_id"] for w in wanted}
        for node_id in set(old) - keep:
            conn.execute(
                "DELETE FROM flow_triggers WHERE flow_id = ? AND node_id = ?", (flow_id, node_id)
            )
        for w in wanted:
            prev = old.get(w["node_id"])
            token = prev["token"] if prev is not None and prev["kind"] == w["kind"] else None
            if w["kind"] == "webhook" and not token:
                token = new_token()
            if w["kind"] != "webhook":
                token = None
            conn.execute(
                "INSERT INTO flow_triggers (flow_id, node_id, kind, config_json, token, token_hash, last_fired_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?) ON CONFLICT(flow_id, node_id) DO UPDATE SET kind = excluded.kind, "
                "config_json = excluded.config_json, token = excluded.token, token_hash = excluded.token_hash",
                (
                    flow_id,
                    w["node_id"],
                    w["kind"],
                    json.dumps(w["config"]),
                    token,
                    _token_hash(token) if token else None,
                    prev["last_fired_at"] if prev is not None else None,
                ),
            )
        conn.commit()
    finally:
        conn.close()


def _token_hash(token: str) -> str:
    import hashlib
    return hashlib.sha256(token.encode()).hexdigest()


def trigger_by_token(token: str) -> Optional[dict]:
    import hmac
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT * FROM flow_triggers WHERE token_hash = ? AND kind = 'webhook'",
            (_token_hash(token),),
        ).fetchone()
        if row is None or not hmac.compare_digest(str(row["token"] or ""), token):
            return None
        return _trigger_from_row(row)
    finally:
        conn.close()


def rotate_trigger_token(flow_id: str, node_id: str, token: str) -> bool:
    conn = get_connection()
    try:
        cur = conn.execute(
            "UPDATE flow_triggers SET token = ?, token_hash = ? WHERE flow_id = ? AND node_id = ? "
            "AND kind = 'webhook'",
            (token, _token_hash(token), flow_id, node_id),
        )
        conn.commit()
        return cur.rowcount > 0
    finally:
        conn.close()


def mark_trigger_fired(flow_id: str, node_id: str, at_ms: int) -> None:
    conn = get_connection()
    try:
        conn.execute(
            "UPDATE flow_triggers SET last_fired_at = ? WHERE flow_id = ? AND node_id = ?",
            (at_ms, flow_id, node_id),
        )
        conn.commit()
    finally:
        conn.close()
