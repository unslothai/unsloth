# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Atomic sidebar edits in the current account's SQLite database."""

import json
from contextlib import closing
from datetime import datetime, timezone
from itertools import groupby

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


def _unchanged_rows(base, layout):
    positions = {row: index for index, row in enumerate(layout)}
    unchanged = set()
    furthest = -1
    for index, row in enumerate(base):
        position = positions[row]
        # Equal positions only preserve order when the same rows precede them.
        if index == position and furthest < index:
            unchanged.add(row)
        furthest = max(furthest, position)
    return unchanged


def _apply_placement(order, move):
    row, target = move["id"], move["targetId"]
    if row != target and row in order and target in order:
        order.remove(row)
        order.insert(order.index(target) + (move["edge"] == "bottom"), row)


def _discard_undone_runs(base, layout, moves, common):
    runs = [list(group) for _, group in groupby(moves, key = lambda move: move["id"])]
    if all(len(run) == 1 for run in runs):
        return moves

    order = _merge_order(base, layout)
    kept = []
    for run in runs:
        row = run[0]["id"]
        can_undo = (
            len(run) > 1
            and row in common
            and all(move["targetId"] in common and not move.get("afterUnpin") for move in run)
        )
        original_position = order.index(row) if can_undo else None
        for move in run:
            _apply_placement(order, move)
        # No other row could have observed a temporary placement inside this run.
        if not can_undo or order.index(row) != original_position:
            kept.extend(run)
    return kept


def _edited_order(
    saved,
    base,
    layout,
    importing,
    moves,
    *,
    switched_to_manual = False,
):
    if base == layout and not switched_to_manual:
        moves = [move for move in moves if move.get("afterUnpin")]
    elif not importing and not switched_to_manual:
        common = set(base).intersection(layout)
        common_base = [key for key in base if key in common]
        common_layout = [key for key in layout if key in common]
        if moves:
            moves = _discard_undone_runs(base, layout, moves, common)
            unchanged = _unchanged_rows(common_base, common_layout)
            moves = [
                move
                for move in moves
                if move.get("afterUnpin")
                or move["id"] not in unchanged
                or move["targetId"] not in common
            ]
        elif common_base != common_layout:
            return _merge_order(layout, saved)
    if moves:
        if saved == base and (switched_to_manual or set(saved).issubset(layout)):
            return _merge_order(layout, saved)
        order = _merge_order(saved, layout)
        for move in moves:
            _apply_placement(order, move)
        if saved == base:
            ranked = set(saved)
            return _merge_order(layout, [row for row in order if row in ranked])
        return order
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
    assigned_rows: dict[str, set[str]] = {}
    for field in ("sectionByChatId", "sectionByProjectId", "sectionByPageId"):
        assignments = dict(stored.get(field, {}))
        for key in base[field].keys() | layout[field].keys():
            previous = base[field].get(key)
            target = layout[field].get(key)
            if target == previous or (importing and key in assignments):
                continue
            # Deleting a section must preserve rows another client moved elsewhere.
            if (
                target is None
                and previous not in layout["sections"]
                and assignments.get(key) != previous
            ):
                continue
            if target is None or target in live:
                assignments[key] = target
                if target is not None:
                    assigned_rows.setdefault(target, set()).add(key)
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
            switched_to_manual = (
                base["sections"].get(key, {}).get("sort") == "updated"
                and layout["sections"].get(key, {}).get("sort") == "manual"
            )
            if importing and key in row_order:
                # Only newly imported rows receive a rank from this browser. Existing rows
                # may since have been dragged, unranked, unfiled, or filed elsewhere.
                eligible = set(saved) | assigned_rows.get(key, set())
                ids = [row for row in ids if row in eligible]
            if not importing:
                # Pages and locally pinned projects can be absent from the editing browser's list.
                pages = result["sectionByPageId"]
                projects = result["sectionByProjectId"]
                removed = {
                    row
                    for row in set(previous).difference(ids)
                    if pages.get(row) != key and projects.get(row) != key
                }
                # A sort switch compares the complete saved ranks. An ordinary refile inserts the row anew.
                if not switched_to_manual:
                    removed.update(assigned_rows.get(key, ()))
                saved = [row for row in saved if row not in removed]
                previous = [row for row in previous if row not in removed]
            row_order[key] = _edited_order(
                saved, previous, ids, importing, moves, switched_to_manual = switched_to_manual
            )
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
