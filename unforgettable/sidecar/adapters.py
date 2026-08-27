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

"""Adapter registry: shadow → promote / discard. Files are never deleted here."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from unforgettable.store.db import get_connection

ADAPTER_STATUSES = frozenset({"shadow", "promoted", "discarded"})
STATUS_SHADOW = "shadow"
STATUS_PROMOTED = "promoted"
STATUS_DISCARDED = "discarded"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _row_to_dict(row) -> dict[str, Any]:
    return dict(row)


def _parse_metrics(raw: Any) -> Optional[dict[str, Any]]:
    if raw is None or raw == "":
        return None
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            loaded = json.loads(raw)
        except json.JSONDecodeError:
            return None
        return loaded if isinstance(loaded, dict) else None
    return None


def insert_adapter(
    *,
    pack_id: str,
    backend: str,
    base_model: str,
    recipe: str,
    path: str,
    status: str = STATUS_SHADOW,
    metrics: Any = None,
    adapter_id: Optional[str] = None,
    gguf_path: Optional[str] = None,
    db_path = None,
) -> dict[str, Any]:
    if status not in ADAPTER_STATUSES:
        raise ValueError(f"unknown adapter status: {status}")
    aid = adapter_id or str(uuid.uuid4())
    if metrics is None or metrics == "":
        metrics_text = None
    elif isinstance(metrics, str):
        metrics_text = metrics
    else:
        metrics_text = json.dumps(metrics)
    conn = get_connection(db_path)
    try:
        conn.execute(
            """
            INSERT INTO adapters(
                id, pack_id, status, backend, base_model, recipe,
                path, gguf_path, metrics, created_at, promoted_at
            ) VALUES(?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                aid,
                pack_id,
                status,
                backend,
                base_model,
                recipe,
                path,
                gguf_path,
                metrics_text,
                _now(),
                None,
            ),
        )
        conn.commit()
    finally:
        conn.close()
    found = get_adapter(aid, db_path = db_path)
    if found is None:
        raise RuntimeError("adapter insert did not persist")
    return found


def set_adapter_gguf_path(
    adapter_id: str,
    gguf_path: Optional[str],
    *,
    db_path = None,
) -> dict[str, Any]:
    if get_adapter(adapter_id, db_path = db_path) is None:
        raise KeyError(adapter_id)
    conn = get_connection(db_path)
    try:
        conn.execute(
            "UPDATE adapters SET gguf_path = ? WHERE id = ?",
            (gguf_path, adapter_id),
        )
        conn.commit()
    finally:
        conn.close()
    found = get_adapter(adapter_id, db_path = db_path)
    if found is None:
        raise RuntimeError("adapter gguf_path did not persist")
    return found


def get_adapter(adapter_id: str, *, db_path = None) -> Optional[dict[str, Any]]:
    conn = get_connection(db_path)
    try:
        row = conn.execute("SELECT * FROM adapters WHERE id = ?", (adapter_id,)).fetchone()
        return _row_to_dict(row) if row else None
    finally:
        conn.close()


def list_adapters(*, status: Optional[str] = None, db_path = None) -> list[dict[str, Any]]:
    if status is not None and status not in ADAPTER_STATUSES:
        raise ValueError(f"unknown adapter status: {status}")
    sql = "SELECT * FROM adapters"
    args: list[Any] = []
    if status is not None:
        sql += " WHERE status = ?"
        args.append(status)
    sql += " ORDER BY created_at DESC, id DESC"
    conn = get_connection(db_path)
    try:
        return [_row_to_dict(row) for row in conn.execute(sql, args).fetchall()]
    finally:
        conn.close()


def set_adapter_metrics(
    adapter_id: str,
    metrics: dict,
    *,
    db_path = None,
) -> dict[str, Any]:
    if get_adapter(adapter_id, db_path = db_path) is None:
        raise KeyError(adapter_id)
    if isinstance(metrics, str):
        metrics_text = metrics
    else:
        metrics_text = json.dumps(metrics)
    conn = get_connection(db_path)
    try:
        conn.execute(
            "UPDATE adapters SET metrics = ? WHERE id = ?",
            (metrics_text, adapter_id),
        )
        conn.commit()
    finally:
        conn.close()
    found = get_adapter(adapter_id, db_path = db_path)
    if found is None:
        raise RuntimeError("adapter metrics did not persist")
    return found


def get_promoted_adapter(*, db_path = None) -> Optional[dict[str, Any]]:
    conn = get_connection(db_path)
    try:
        row = conn.execute(
            """
            SELECT * FROM adapters
            WHERE status = ?
            ORDER BY promoted_at DESC, id DESC
            LIMIT 1
            """,
            (STATUS_PROMOTED,),
        ).fetchone()
        return _row_to_dict(row) if row else None
    finally:
        conn.close()


def _set_status(
    adapter_id: str,
    status: str,
    *,
    promoted_at: Optional[str] = None,
    db_path = None,
) -> dict[str, Any]:
    conn = get_connection(db_path)
    try:
        conn.execute(
            "UPDATE adapters SET status = ?, promoted_at = ? WHERE id = ?",
            (status, promoted_at, adapter_id),
        )
        conn.commit()
    finally:
        conn.close()
    found = get_adapter(adapter_id, db_path = db_path)
    if found is None:
        raise RuntimeError("adapter update did not persist")
    return found


def discard_adapter(adapter_id: str, *, db_path = None) -> dict[str, Any]:
    row = get_adapter(adapter_id, db_path = db_path)
    if row is None:
        raise KeyError(adapter_id)
    if row["status"] == STATUS_DISCARDED:
        return row
    return _set_status(
        adapter_id, STATUS_DISCARDED, promoted_at = row.get("promoted_at"), db_path = db_path
    )


def promote_adapter(
    adapter_id: str,
    *,
    force: bool = False,
    db_path = None,
) -> dict[str, Any]:
    row = get_adapter(adapter_id, db_path = db_path)
    if row is None:
        raise KeyError(adapter_id)
    if row["status"] == STATUS_DISCARDED and not force:
        raise ValueError("discarded adapter cannot be promoted without force")
    if not force:
        metrics = _parse_metrics(row.get("metrics"))
        if metrics is None or "adapter_lean" not in metrics or "base_lean" not in metrics:
            raise ValueError("promote refused: no eval metrics")
        if metrics.get("passed") is not True:
            raise ValueError("promote refused: eval did not pass")
        if metrics["adapter_lean"] < metrics["base_lean"]:
            raise ValueError("promote refused: adapter_lean < base_lean")
    now = _now()
    conn = get_connection(db_path)
    try:
        conn.execute(
            """
            UPDATE adapters SET status = ?
            WHERE status = ? AND id != ?
            """,
            (STATUS_DISCARDED, STATUS_PROMOTED, adapter_id),
        )
        conn.execute(
            "UPDATE adapters SET status = ?, promoted_at = ? WHERE id = ?",
            (STATUS_PROMOTED, now, adapter_id),
        )
        conn.commit()
    finally:
        conn.close()
    found = get_adapter(adapter_id, db_path = db_path)
    if found is None:
        raise RuntimeError("adapter promote did not persist")
    return found


def rollback_adapter(*, db_path = None) -> Optional[dict[str, Any]]:
    current = get_promoted_adapter(db_path = db_path)
    if current is None:
        return None
    return discard_adapter(current["id"], db_path = db_path)
