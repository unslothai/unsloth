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

"""Deterministic hygiene pass. No LLM, no hard-delete of admitted rows."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from unforgettable.constants import PROVENANCE_WEIGHT, is_who

from .records import list_records, set_record_status
from .titles import normalize_title

EMPTY_PROPOSED_AGE_DAYS = 7
STALE_PROPOSED_AGE_DAYS = 30
KEEP_SUPERSEDED_ANCESTORS = 2
COMPACT_DEDUPE_KINDS = frozenset({"claim", "procedure", "entity"})
EMPTY_PROPOSED_BODIES = frozenset({"", "todo", "(empty)"})
UNKNOWN_PROVENANCE_WEIGHT = 99
COMPACT_EMPTY_REASON = "compact: empty proposed"
COMPACT_STALE_REASON = "compact: stale proposed"
KEEP_STALE_ERROR_FIX_PROVENANCE = frozenset({"world", "mixed"})


@dataclass(frozen = True)
class CompactReport:
    emptied: list[str]
    deduped: list[tuple[str, str]]
    folded: list[str]
    dry_run: bool


def run_compact(
    db_path = None,
    *,
    dry_run: bool = False,
    older_than_days: Optional[int] = None,
) -> CompactReport:
    records = list_records(db_path = db_path)
    now = datetime.now(timezone.utc)
    empty_ids = _empty_proposed_ids(records, now = now)
    stale_ids = _stale_proposed_ids(records, now = now, older_than_days = older_than_days)
    emptied = list(dict.fromkeys([*empty_ids, *stale_ids]))
    empty_set = set(empty_ids)
    deduped = _dedupe_pairs(records)
    folded = _fold_ids(records)
    if not dry_run:
        for rid in emptied:
            reason = COMPACT_EMPTY_REASON if rid in empty_set else COMPACT_STALE_REASON
            set_record_status(rid, "rejected", reason = reason, db_path = db_path)
        for loser_id, winner_id in deduped:
            set_record_status(
                loser_id,
                "deprecated",
                reason = _duplicate_reason(winner_id),
                db_path = db_path,
            )
        for rid in folded:
            set_record_status(rid, "deprecated", db_path = db_path)
    return CompactReport(
        emptied = emptied,
        deduped = deduped,
        folded = folded,
        dry_run = dry_run,
    )


def _duplicate_reason(winner_id: str) -> str:
    return f"compact: duplicate of {winner_id}"


def _empty_proposed_ids(records: list[dict[str, Any]], *, now: datetime) -> list[str]:
    ids: list[str] = []
    for rec in records:
        if rec["status"] != "proposed":
            continue
        if not _is_empty_proposed_body(rec.get("body")):
            continue
        if _is_at_least_days_old(rec.get("created_at"), EMPTY_PROPOSED_AGE_DAYS, now = now):
            ids.append(rec["id"])
    return ids


def _is_empty_proposed_body(body: Optional[str]) -> bool:
    return (body or "").strip() in EMPTY_PROPOSED_BODIES


def _stale_proposed_ids(
    records: list[dict[str, Any]], *, now: datetime, older_than_days: Optional[int]
) -> list[str]:
    days = STALE_PROPOSED_AGE_DAYS if older_than_days is None else int(older_than_days)
    if days < 1:
        days = STALE_PROPOSED_AGE_DAYS
    ids: list[str] = []
    for rec in records:
        if rec["status"] != "proposed":
            continue
        if _is_empty_proposed_body(rec.get("body")):
            continue
        if (
            rec.get("kind") == "error_fix"
            and rec.get("provenance") in KEEP_STALE_ERROR_FIX_PROVENANCE
        ):
            continue
        if not (is_who(rec) or rec.get("provenance") == "infer"):
            continue
        if _is_at_least_days_old(rec.get("created_at"), days, now = now):
            ids.append(rec["id"])
    return ids


def _dedupe_pairs(records: list[dict[str, Any]]) -> list[tuple[str, str]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for rec in records:
        if rec["status"] != "active":
            continue
        if rec["kind"] not in COMPACT_DEDUPE_KINDS:
            continue
        key = (
            rec.get("namespace_id") or "",
            rec["kind"],
            normalize_title(rec.get("title") or ""),
        )
        groups.setdefault(key, []).append(rec)
    pairs: list[tuple[str, str]] = []
    for group in groups.values():
        if len(group) < 2:
            continue
        winner = min(group, key = _dedupe_rank)
        for rec in group:
            if rec["id"] != winner["id"]:
                pairs.append((rec["id"], winner["id"]))
    return pairs


def _dedupe_rank(rec: dict[str, Any]) -> tuple[int, float, str]:
    weight = PROVENANCE_WEIGHT.get(rec.get("provenance"), UNKNOWN_PROVENANCE_WEIGHT)
    ts = _parse_dt(rec.get("updated_at"))
    stamp = ts.timestamp() if ts is not None else 0.0
    return (weight, -stamp, rec["id"])


def _fold_ids(records: list[dict[str, Any]]) -> list[str]:
    by_id = {rec["id"]: rec for rec in records}
    referenced = {rec.get("supersedes_id") for rec in records if rec.get("supersedes_id")}
    folded: list[str] = []
    seen: set[str] = set()
    for head in records:
        if head["id"] in referenced:
            continue
        for rec in _ancestors(head, by_id)[KEEP_SUPERSEDED_ANCESTORS:]:
            if rec["status"] != "superseded" or rec["id"] in seen:
                continue
            seen.add(rec["id"])
            folded.append(rec["id"])
    return folded


def _ancestors(head: dict[str, Any], by_id: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    ancestors: list[dict[str, Any]] = []
    walked = {head["id"]}
    current_id = head.get("supersedes_id")
    while current_id and current_id not in walked:
        walked.add(current_id)
        rec = by_id.get(current_id)
        if rec is None:
            break
        ancestors.append(rec)
        current_id = rec.get("supersedes_id")
    return ancestors


def _is_at_least_days_old(created_at: Optional[str], days: int, *, now: datetime) -> bool:
    ts = _parse_dt(created_at)
    if ts is None:
        return False
    return now - ts >= timedelta(days = days)


def _parse_dt(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        ts = datetime.fromisoformat(value)
    except ValueError:
        return None
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo = timezone.utc)
    return ts.astimezone(timezone.utc)
