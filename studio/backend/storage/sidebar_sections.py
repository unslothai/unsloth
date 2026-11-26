# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Atomic sidebar edits in the current account's SQLite database."""

import json
from collections import Counter
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


def _replay_placements(order, moves):
    for move in moves:
        _apply_placement(order, move)
    return order


def _mark_later_unpin_placements(moves):
    unpinned = set()
    for move in moves:
        if move.get("afterUnpin"):
            unpinned.add(move["id"])
        elif move["id"] in unpinned:
            move = {**move, "afterUnpin": True}
        yield move


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
    unranked_first = False,
):
    if any(move.get("afterUnpin") for move in moves):
        moves = list(_mark_later_unpin_placements(moves))
    if base == layout and not switched_to_manual:
        moves = [move for move in moves if move.get("afterUnpin")]
    elif not importing and not switched_to_manual:
        common = set(base).intersection(layout)
        common_base = [key for key in base if key in common]
        common_layout = [key for key in layout if key in common]
        if moves:
            moves = _discard_undone_runs(base, layout, moves, common)
            unchanged = _unchanged_rows(common_base, common_layout)

            def essential(move):
                return (
                    move.get("afterUnpin")
                    or move["id"] not in unchanged
                    or move["targetId"] not in common
                )

            candidates = [move for move in moves if essential(move)]
            if len(candidates) < len(moves):
                # Another placement can depend on a row that kept its final rank.
                initial = _merge_order(base, layout)
                expected = _replay_placements(initial.copy(), moves)
                if _replay_placements(initial.copy(), candidates) == expected:
                    moves = candidates
                else:
                    counts = Counter(move["id"] for move in moves)
                    with_single_moves = [
                        move for move in moves if essential(move) or counts[move["id"]] == 1
                    ]
                    if (
                        len(with_single_moves) < len(moves)
                        and _replay_placements(initial, with_single_moves) == expected
                    ):
                        moves = with_single_moves
        elif common_base != common_layout:
            return _merge_order(layout, saved), False
    if moves:
        visible = set(layout)
        placed = any(
            move["id"] != move["targetId"] and move["id"] in visible and move["targetId"] in visible
            for move in moves
        )
        if saved == base and (switched_to_manual or set(saved).issubset(visible)):
            return _merge_order(layout, saved), placed
        primary = saved
        if unranked_first and saved != base:
            ranked = set(saved).union(base)
            # Manual sections display unranked rows before their saved order.
            unranked = [row for row in layout if row not in ranked]
            primary = unranked + saved
        order = _replay_placements(_merge_order(primary, layout), moves)
        if saved == base:
            ranked = set(saved)
            return _merge_order(layout, [row for row in order if row in ranked]), placed
        return order, placed
    return _merge_order(saved, layout), False


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
    unpin_filings = {
        (move.get("unpinKind"), move["id"], move["sectionId"])
        for move in order_moves
        if move.get("afterUnpin") and not importing
    }
    for kind in ("chat", "project", "page"):
        field = f"sectionBy{kind.capitalize()}Id"
        assignments = dict(stored.get(field, {}))
        for key in base[field].keys() | layout[field].keys():
            previous = base[field].get(key)
            target = layout[field].get(key)
            returned_after_unpin = (
                target is not None
                and (kind, key, target) in unpin_filings
                and assignments.get(key) != target
            )
            if (target == previous and not returned_after_unpin) or (
                importing and key in assignments
            ):
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
    order, _ = _edited_order(
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
            previous_sort = base["sections"].get(key, {}).get("sort")
            sort = layout["sections"].get(key, {}).get("sort")
            switched_to_manual = previous_sort == "updated" and sort == "manual"
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
            row_order[key], placed = _edited_order(
                saved,
                previous,
                ids,
                importing,
                moves,
                switched_to_manual = switched_to_manual,
                unranked_first = previous_sort == sort == "manual" and not importing,
            )
            if (
                placed
                and not importing
                and key in live
                and sort == "manual"
                and sections[key]["sort"] != sort
            ):
                sections[key] = {**sections[key], "sort": sort}
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
