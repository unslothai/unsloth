# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Atomic sidebar edits in the current account's SQLite database."""

import json
from contextlib import closing
from datetime import datetime, timezone

from storage.studio_db import get_app_setting, get_connection

_KEY = "sidebarSections"


def _merge_order(primary: list[str], other: list[str]) -> list[str]:
    """Keep primary's order. Place additional IDs before their next shared neighbour."""
    shared = set(primary)
    anchor = None
    before = {}
    for key in reversed(other):
        if key in shared:
            anchor = key
        else:
            before.setdefault(anchor, []).append(key)
    result = []
    for key in [*primary, None]:
        result.extend(reversed(before.get(key, ())))
        if key is not None:
            result.append(key)
    return list(dict.fromkeys(result))


def _edited_order(saved, base, layout, importing, moves):
    if moves and base != layout:
        if saved == base and set(saved).issubset(layout):
            return _merge_order(layout, saved)
        order = _merge_order(saved, layout)
        for move in moves:
            row, target = move["id"], move["targetId"]
            if row != target and row in order and target in order:
                order.remove(row)
                order.insert(order.index(target) + (move["edge"] == "bottom"), row)
        return order
    if not importing and base != layout:
        # Insertions and deletions alone leave a concurrent reorder intact.
        common = set(base).intersection(layout)
        if [key for key in base if key in common] != [key for key in layout if key in common]:
            return _merge_order(layout, saved)
    return _merge_order(saved, layout)


def _merge(
    stored,
    base,
    layout,
    *,
    importing,
    order_moves = (),
):
    # Null values are tombstones: a late browser import cannot restore a deletion or unfiling.
    sections = dict(stored.get("sections", {}))
    for key in base["sections"].keys() | layout["sections"].keys():
        if key in ("pinned", "projects"):
            continue
        previous = base["sections"].get(key)
        edit = layout["sections"].get(key)
        if edit == previous or (importing and key in sections):
            continue
        current = sections.get(key)
        if edit is None or previous is None:
            sections[key] = edit
        elif current is not None:
            sections[key] = {
                **current,
                **{field: value for field, value in edit.items() if previous.get(field) != value},
            }

    live = {key for key, value in sections.items() if value is not None}
    result = {"sections": sections}
    imported_rows: dict[str, set[str]] = {}
    for field in ("sectionByChatId", "sectionByProjectId", "sectionByPageId"):
        assignments = dict(stored.get(field, {}))
        for key in base[field].keys() | layout[field].keys():
            target = layout[field].get(key)
            if target == base[field].get(key) or (importing and key in assignments):
                continue
            # Removing a section unfiles its rows; another client may have moved one first.
            if target is None and assignments.get(key) != base[field].get(key):
                continue
            if target is None or target in live:
                assignments[key] = target
                if importing and target is not None:
                    imported_rows.setdefault(target, set()).add(key)
        result[field] = {
            key: value if value in live else None for key, value in assignments.items()
        }

    moves_by_section = {}
    for move in order_moves:
        moves_by_section.setdefault(move["sectionId"], []).append(move)
    saved_order = stored.get("sectionOrder", [])
    order = _edited_order(
        saved_order,
        base["sectionOrder"],
        layout["sectionOrder"],
        importing,
        moves_by_section.get(None, ()),
    )
    result["sectionOrder"] = [key for key in order if key in live or key in ("pinned", "projects")]
    row_order = dict(stored.get("rowOrder", {}))
    for key, ids in layout["rowOrder"].items():
        previous = base["rowOrder"].get(key, [])
        moves = moves_by_section.get(key, ())
        if importing or ids != previous or key not in row_order or moves:
            saved = row_order.get(key, [])
            if importing and key in row_order:
                # Only newly imported rows receive a rank from this browser. Existing rows
                # may since have been dragged, unranked, unfiled, or filed elsewhere.
                eligible = set(saved) | imported_rows.get(key, set())
                ids = [row for row in ids if row in eligible]
            if not importing:
                # Browser-local pins can be absent from the editing browser's list.
                pages = result["sectionByPageId"]
                removed = {row for row in set(previous).difference(ids) if pages.get(row) != key}
                saved = [row for row in saved if row not in removed]
                previous = [row for row in previous if row not in removed]
            row_order[key] = _edited_order(saved, previous, ids, importing, moves)
    result["rowOrder"] = {key: ids for key, ids in row_order.items() if key in live}
    return result


def _visible(stored):
    return {
        key: {k: v for k, v in value.items() if v is not None} if isinstance(value, dict) else value
        for key, value in stored.items()
    }


def get_sidebar_layout():
    return _visible(get_app_setting(_KEY, {}))


def save_sidebar_layout(
    operation_id,
    base,
    layout,
    *,
    importing,
    order_moves = (),
):
    with closing(get_connection()) as conn, conn:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute("SELECT value_json FROM app_settings WHERE key = ?", (_KEY,)).fetchone()
        stored = json.loads(row[0]) if row else {}
        # The reply may be lost, or another tab may have sent this browser's pending edit.
        inserted = conn.execute(
            "INSERT OR IGNORE INTO sidebar_section_edits (id) VALUES (?)", (operation_id,)
        )
        if inserted.rowcount:
            merged = _merge(stored, base, layout, importing = importing, order_moves = order_moves)
            conn.execute(
                """INSERT INTO app_settings (key, value_json, updated_at) VALUES (?, ?, ?)
                   ON CONFLICT(key) DO UPDATE SET
                       value_json = excluded.value_json, updated_at = excluded.updated_at""",
                (_KEY, json.dumps(merged), datetime.now(timezone.utc).isoformat()),
            )
            stored = merged
        return _visible(stored)
