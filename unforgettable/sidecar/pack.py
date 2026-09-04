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

"""Eligibility, world-pass votes, holdout split, and pack persist."""

from __future__ import annotations

import json
import math
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Optional

from unforgettable.eyes.probes import is_probe_title
from unforgettable.rims.detect import TEST_COMMAND_TITLE
from unforgettable.sidecar.format import PACK_BODY_CHARS, format_sft_item
from unforgettable.store.db import get_connection
from unforgettable.store.records import list_inject_stats, list_records, list_rollouts
from unforgettable.store.titles import normalize_title

PACK_KINDS = frozenset({"procedure", "error_fix"})
PACK_PROVENANCE = frozenset({"world", "mixed", "human"})
PACK_MIN_TRAIN = 4
HOLDOUT_FRACTION = 0.2
HOLDOUT_MIN_EPISODES = 5
DISTILL_CHAR_THRESHOLD = 2000
DISTILL_MIN_COMPILED = 3
DISTILL_STATS_WINDOW = 20

PACK_SOURCE = "record"
ROLE_TRAIN = "train"
ROLE_HOLDOUT = "holdout"

REASON_NOT_PACK_KIND = "not a pack kind"
REASON_NOT_ACTIVE = "not active"
REASON_UNTRUSTED = "untrusted provenance"
REASON_PROBE = "probe"
REASON_TEST_COMMAND = "test command"
REASON_EMPTY_TITLE = "empty title"
REASON_EMPTY_BODY = "empty body"
REASON_NO_WORLD_PASS = "no world-pass vote"
REASON_SIM_NO_WORLD = "sim vote without world-pass"
REASON_SIM_TWIN = "sim vote has twin_note"


@dataclass(frozen = True)
class PackReport:
    pack_id: Optional[str]
    n_train: int
    n_holdout: int
    dropped: list[tuple[str, str]]
    include_sim: bool
    dry_run: bool


@dataclass
class _Candidate:
    rec: dict[str, Any]
    vote_eps: set[str]
    contact: str
    role: str = ROLE_TRAIN


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _row_to_dict(row) -> dict[str, Any]:
    return dict(row)


def _refusal_reason(rec: Optional[dict[str, Any]]) -> Optional[str]:
    if rec is None:
        return REASON_NOT_PACK_KIND
    if rec.get("kind") not in PACK_KINDS:
        return REASON_NOT_PACK_KIND
    if rec.get("status") != "active":
        return REASON_NOT_ACTIVE
    if rec.get("provenance") not in PACK_PROVENANCE:
        return REASON_UNTRUSTED
    if rec.get("kind") == "procedure":
        if is_probe_title(rec.get("title") or ""):
            return REASON_PROBE
        if normalize_title(rec.get("title") or "") == TEST_COMMAND_TITLE:
            return REASON_TEST_COMMAND
    if not (rec.get("title") or "").strip():
        return REASON_EMPTY_TITLE
    if not (rec.get("body") or "").strip():
        return REASON_EMPTY_BODY
    return None


def is_pack_record(rec: dict) -> bool:
    return _refusal_reason(rec) is None


def pack_is_retrieval_heavy(db_path = None) -> bool:
    # Count compiled rows only. list_compiled() refreshes/unpins membership.
    if len(_compiled_ids(db_path = db_path)) >= DISTILL_MIN_COMPILED:
        return True
    world_rows = [
        row
        for row in list_inject_stats(limit = DISTILL_STATS_WINDOW * 10, db_path = db_path)
        if row.get("contact") == "world"
    ][:DISTILL_STATS_WINDOW]
    if not world_rows:
        return False
    total = 0
    for row in world_rows:
        total += int(row.get("standing_chars") or 0) + int(row.get("retrieve_chars") or 0)
    return (total / len(world_rows)) >= DISTILL_CHAR_THRESHOLD


def _compiled_ids(*, db_path = None) -> set[str]:
    conn = get_connection(db_path)
    try:
        rows = conn.execute("SELECT source_record_id FROM compiled").fetchall()
        return {row[0] for row in rows}
    finally:
        conn.close()


def _all_retrieve_uses(*, db_path = None) -> list[dict[str, Any]]:
    conn = get_connection(db_path)
    try:
        rows = conn.execute("SELECT * FROM retrieve_uses").fetchall()
        return [_row_to_dict(row) for row in rows]
    finally:
        conn.close()


def _pass_episodes(rollouts: list[dict[str, Any]], contact: str) -> set[str]:
    return {
        row["episode_id"]
        for row in rollouts
        if row.get("contact") == contact and row.get("outcome") == "pass"
    }


def _twin_note_episodes(*, db_path = None) -> set[str]:
    eps: set[str] = set()
    for rec in list_records(kinds = ["twin_note"], statuses = ["active"], db_path = db_path):
        eid = rec.get("source_episode_id")
        if eid:
            eps.add(eid)
    return eps


def _vote_for_record(
    rec: dict[str, Any],
    *,
    include_sim: bool,
    compiled_ids: set[str],
    uses_by_record: dict[str, list[dict[str, Any]]],
    world_pass: set[str],
    sim_pass: set[str],
    twin_eps: set[str],
) -> tuple[Optional[set[str]], str, Optional[str]]:
    """Return (vote episodes, contact, drop reason). Reason is set when unvoted."""
    rid = rec["id"]
    world_vote: set[str] = set()
    sim_no_world = False
    sim_twin = False
    sim_vote: set[str] = set()
    for use in uses_by_record.get(rid, ()):
        eid = use.get("episode_id")
        if not eid:
            continue
        contact = use.get("contact")
        if contact == "world" and eid in world_pass:
            world_vote.add(eid)
        if contact == "sim" and eid in sim_pass:
            if eid not in world_pass:
                sim_no_world = True
            elif eid in twin_eps:
                sim_twin = True
            else:
                sim_vote.add(eid)
    vote_eps = set(world_vote)
    if include_sim:
        vote_eps |= sim_vote
    if rid in compiled_ids or vote_eps:
        contact = "sim" if not world_vote and sim_vote else "world"
        return vote_eps, contact, None
    if include_sim and (sim_no_world or sim_twin or sim_vote):
        if sim_twin and not sim_no_world:
            return None, "world", REASON_SIM_TWIN
        if sim_no_world:
            return None, "world", REASON_SIM_NO_WORLD
        return None, "world", REASON_NO_WORLD_PASS
    return None, "world", REASON_NO_WORLD_PASS


def _assign_roles(candidates: list[_Candidate]) -> None:
    vote_eps = sorted({eid for cand in candidates for eid in cand.vote_eps})
    n = len(vote_eps)
    holdout_n = math.ceil(n * HOLDOUT_FRACTION) if n >= HOLDOUT_MIN_EPISODES else 0
    holdout_eps = set(vote_eps[-holdout_n:]) if holdout_n else set()
    for cand in candidates:
        if cand.vote_eps and cand.vote_eps <= holdout_eps:
            cand.role = ROLE_HOLDOUT
        else:
            cand.role = ROLE_TRAIN
    train = [c for c in candidates if c.role == ROLE_TRAIN]
    holdout = [c for c in candidates if c.role == ROLE_HOLDOUT]
    if len(train) >= PACK_MIN_TRAIN or not holdout:
        return
    holdout.sort(key = lambda c: (c.rec.get("created_at") or "", c.rec["id"]), reverse = True)
    while len(train) < PACK_MIN_TRAIN and holdout:
        moved = holdout.pop(0)
        moved.role = ROLE_TRAIN
        train.append(moved)


def _persist(
    pack_id: str,
    report: PackReport,
    candidates: list[_Candidate],
    *,
    db_path = None,
) -> None:
    now = _now()
    conn = get_connection(db_path)
    try:
        conn.execute(
            """
            INSERT INTO packs(
                id, created_at, n_train, n_holdout, include_sim, report
            ) VALUES(?,?,?,?,?,?)
            """,
            (
                pack_id,
                now,
                report.n_train,
                report.n_holdout,
                1 if report.include_sim else 0,
                json.dumps(asdict(report)),
            ),
        )
        for cand in candidates:
            rec = cand.rec
            episode_id = sorted(cand.vote_eps)[0] if cand.vote_eps else None
            conn.execute(
                """
                INSERT INTO pack_items(
                    id, pack_id, role, source, source_id, episode_id,
                    kind, provenance, contact, messages, created_at
                ) VALUES(?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    str(uuid.uuid4()),
                    pack_id,
                    cand.role,
                    PACK_SOURCE,
                    rec["id"],
                    episode_id,
                    rec.get("kind") or "",
                    rec.get("provenance") or "",
                    cand.contact,
                    json.dumps(format_sft_item(rec)),
                    now,
                ),
            )
        conn.commit()
    finally:
        conn.close()


def pack_from_admitted_b(
    *,
    include_sim: bool = False,
    dry_run: bool = False,
    db_path = None,
) -> PackReport:
    records = list_records(db_path = db_path)
    compiled_ids = _compiled_ids(db_path = db_path)
    uses: list[dict[str, Any]] = _all_retrieve_uses(db_path = db_path)
    uses_by_record: dict[str, list[dict[str, Any]]] = {}
    for use in uses:
        uses_by_record.setdefault(use["record_id"], []).append(use)
    rollouts = list_rollouts(db_path = db_path)
    world_pass = _pass_episodes(rollouts, "world")
    sim_pass = _pass_episodes(rollouts, "sim")
    twin_eps = _twin_note_episodes(db_path = db_path)

    dropped: list[tuple[str, str]] = []
    candidates: list[_Candidate] = []
    for rec in records:
        reason = _refusal_reason(rec)
        if reason:
            dropped.append((rec.get("id") or rec.get("kind") or "", reason))
            continue
        vote_eps, contact, vote_reason = _vote_for_record(
            rec,
            include_sim = include_sim,
            compiled_ids = compiled_ids,
            uses_by_record = uses_by_record,
            world_pass = world_pass,
            sim_pass = sim_pass,
            twin_eps = twin_eps,
        )
        if vote_reason:
            dropped.append((rec["id"], vote_reason))
            continue
        candidates.append(_Candidate(rec = rec, vote_eps = vote_eps or set(), contact = contact))

    _assign_roles(candidates)
    n_train = sum(1 for c in candidates if c.role == ROLE_TRAIN)
    n_holdout = sum(1 for c in candidates if c.role == ROLE_HOLDOUT)
    persist = (not dry_run) and bool(candidates)
    pack_id = str(uuid.uuid4()) if persist else None
    report = PackReport(
        pack_id = pack_id,
        n_train = n_train,
        n_holdout = n_holdout,
        dropped = dropped,
        include_sim = include_sim,
        dry_run = dry_run,
    )
    if persist:
        _persist(pack_id, report, candidates, db_path = db_path)
    return report


def get_pack(pack_id: str, *, db_path = None) -> Optional[dict[str, Any]]:
    conn = get_connection(db_path)
    try:
        row = conn.execute("SELECT * FROM packs WHERE id = ?", (pack_id,)).fetchone()
        rec = _row_to_dict(row) if row else None
    finally:
        conn.close()
    if rec is None:
        return None
    rec["include_sim"] = bool(rec.get("include_sim"))
    return rec


def list_packs(*, limit: Optional[int] = None, db_path = None) -> list[dict[str, Any]]:
    sql = "SELECT * FROM packs ORDER BY created_at DESC, id DESC"
    args: list[Any] = []
    if limit is not None:
        sql += " LIMIT ?"
        args.append(limit)
    conn = get_connection(db_path)
    try:
        rows = [_row_to_dict(row) for row in conn.execute(sql, args).fetchall()]
    finally:
        conn.close()
    for row in rows:
        row["include_sim"] = bool(row.get("include_sim"))
    return rows


def list_pack_items(pack_id: str, *, db_path = None) -> list[dict[str, Any]]:
    conn = get_connection(db_path)
    try:
        rows = [
            _row_to_dict(row)
            for row in conn.execute(
                "SELECT * FROM pack_items WHERE pack_id = ? ORDER BY role ASC, source_id ASC",
                (pack_id,),
            ).fetchall()
        ]
    finally:
        conn.close()
    for row in rows:
        raw = row.get("messages")
        if isinstance(raw, str):
            row["messages"] = json.loads(raw)
    return rows
