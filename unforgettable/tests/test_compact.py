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

from datetime import datetime, timedelta, timezone

from unforgettable.store.compact import (
    COMPACT_DEDUPE_KINDS,
    COMPACT_STALE_REASON,
    EMPTY_PROPOSED_AGE_DAYS,
    KEEP_SUPERSEDED_ANCESTORS,
    STALE_PROPOSED_AGE_DAYS,
    run_compact,
)
from unforgettable.store.db import get_connection
from unforgettable.store.records import (
    get_record,
    insert_record,
    list_admissions,
    list_records,
    supersede_record,
)


def _age_created_at(record_id: str, db_path, *, days: int) -> None:
    past = (datetime.now(timezone.utc) - timedelta(days = days)).isoformat()
    conn = get_connection(db_path)
    try:
        conn.execute(
            "UPDATE records SET created_at = ? WHERE id = ?",
            (past, record_id),
        )
        conn.commit()
    finally:
        conn.close()


def test_compact_dedupe_kinds_are_locked():
    assert COMPACT_DEDUPE_KINDS == frozenset({"claim", "procedure", "entity"})


def _insert_pair(
    db_path,
    *,
    kind,
    title,
    winner_body,
    loser_body,
    loser_provenance = "infer",
):
    winner = insert_record(
        kind = kind,
        title = title,
        body = winner_body,
        provenance = "world",
        db_path = db_path,
    )
    loser = insert_record(
        kind = kind,
        title = title,
        body = loser_body,
        provenance = loser_provenance,
        db_path = db_path,
    )
    return winner, loser


def _assert_deduped(db_path, winner, loser, report):
    assert (loser["id"], winner["id"]) in report.deduped
    loaded_loser = get_record(loser["id"], db_path = db_path)
    loaded_winner = get_record(winner["id"], db_path = db_path)
    assert loaded_loser["status"] == "deprecated"
    assert f"duplicate of {winner['id']}" in loaded_loser["body"]
    assert loaded_winner["status"] == "active"


def _assert_both_active(db_path, first, second):
    assert get_record(first["id"], db_path = db_path)["status"] == "active"
    assert get_record(second["id"], db_path = db_path)["status"] == "active"


def test_compact_does_not_dedupe_across_namespaces(db_path):
    from unforgettable.store.records import create_namespace

    create_namespace(name = "other", namespace_id = "other", db_path = db_path)
    first = insert_record(
        kind = "claim",
        title = "Shared title",
        body = "default ns",
        provenance = "world",
        db_path = db_path,
    )
    second = insert_record(
        kind = "claim",
        title = "Shared title",
        body = "other ns",
        provenance = "infer",
        namespace_id = "other",
        db_path = db_path,
    )
    report = run_compact(db_path = db_path)
    assert report.deduped == []
    assert get_record(first["id"], db_path = db_path)["status"] == "active"
    assert get_record(second["id"], db_path = db_path)["status"] == "active"


def test_duplicate_claims_deprecate_loser(db_path):
    world, infer = _insert_pair(
        db_path,
        kind = "claim",
        title = "Friction",
        winner_body = "surface friction is high on steel in the world",
        loser_body = "surface friction is high on steel",
    )
    report = run_compact(db_path)
    assert report.dry_run is False
    _assert_deduped(db_path, world, infer, report)


def test_duplicate_procedures_deprecate_loser(db_path):
    world, infer = _insert_pair(
        db_path,
        kind = "procedure",
        title = "Bleed the line",
        winner_body = "close valve A then open B",
        loser_body = "maybe close a valve",
    )
    report = run_compact(db_path)
    _assert_deduped(db_path, world, infer, report)


def test_duplicate_entities_deprecate_loser(db_path):
    world, infer = _insert_pair(
        db_path,
        kind = "entity",
        title = "Pump X",
        winner_body = "Pump X is the north-loop booster",
        loser_body = "some pump named X",
    )
    report = run_compact(db_path)
    _assert_deduped(db_path, world, infer, report)


def test_same_title_twin_notes_stay_active(db_path):
    first, second = _insert_pair(
        db_path,
        kind = "twin_note",
        title = "World/sim disagreement",
        winner_body = "sim said yes; world said no",
        loser_body = "another drifted episode",
        loser_provenance = "mixed",
    )
    run_compact(db_path)
    _assert_both_active(db_path, first, second)


def test_same_title_error_fixes_stay_active(db_path):
    first, second = _insert_pair(
        db_path,
        kind = "error_fix",
        title = "Error then fix",
        winner_body = "first failure then success",
        loser_body = "another failure then success",
        loser_provenance = "mixed",
    )
    run_compact(db_path)
    _assert_both_active(db_path, first, second)


def test_same_title_episodes_stay_active(db_path):
    first, second = _insert_pair(
        db_path,
        kind = "episode",
        title = "Episode abcdef12",
        winner_body = "ran the pump checklist",
        loser_body = "ran it again later",
        loser_provenance = "mixed",
    )
    run_compact(db_path)
    _assert_both_active(db_path, first, second)


def test_same_title_directives_stay_active(db_path):
    first, second = _insert_pair(
        db_path,
        kind = "directive",
        title = "Always cite ids",
        winner_body = "Ground answers in returned memory ids.",
        loser_body = "Cite ids when recalling facts.",
        loser_provenance = "human",
    )
    run_compact(db_path)
    _assert_both_active(db_path, first, second)


def test_compact_stale_proposed_infer(db_path):
    stale = insert_record(
        kind = "claim",
        title = "Old infer noise",
        body = "a long unadmitted guess about the pump",
        provenance = "infer",
        speaker = "model",
        status = "proposed",
        db_path = db_path,
    )
    user_who = insert_record(
        kind = "claim",
        title = "User guess",
        body = "I think the rate is 12",
        provenance = "infer",
        speaker = "user",
        status = "proposed",
        db_path = db_path,
    )
    keep_fix = insert_record(
        kind = "error_fix",
        title = "Error then fix: traceback",
        body = "Tried: traceback\nThen: tests passed",
        provenance = "world",
        status = "proposed",
        db_path = db_path,
    )
    fresh = insert_record(
        kind = "claim",
        title = "Fresh infer",
        body = "yesterday's draft",
        provenance = "infer",
        status = "proposed",
        db_path = db_path,
    )
    _age_created_at(stale["id"], db_path, days = STALE_PROPOSED_AGE_DAYS + 1)
    _age_created_at(user_who["id"], db_path, days = STALE_PROPOSED_AGE_DAYS + 1)
    _age_created_at(keep_fix["id"], db_path, days = STALE_PROPOSED_AGE_DAYS + 1)
    report = run_compact(db_path)
    assert stale["id"] in report.emptied
    assert user_who["id"] in report.emptied
    assert keep_fix["id"] not in report.emptied
    assert fresh["id"] not in report.emptied
    assert get_record(stale["id"], db_path = db_path)["status"] == "rejected"
    assert get_record(keep_fix["id"], db_path = db_path)["status"] == "proposed"
    reasons = [row.get("reason") or "" for row in list_admissions(db_path = db_path)]
    assert any(COMPACT_STALE_REASON in reason for reason in reasons)
    preview = run_compact(db_path, dry_run = True, older_than_days = 1)
    assert preview.dry_run is True


def test_old_empty_proposed_is_rejected(db_path):
    old = insert_record(
        kind = "claim",
        title = "Empty leftover",
        body = "",
        provenance = "infer",
        status = "proposed",
        db_path = db_path,
    )
    fresh = insert_record(
        kind = "claim",
        title = "Fresh empty",
        body = "todo",
        provenance = "infer",
        status = "proposed",
        db_path = db_path,
    )
    _age_created_at(old["id"], db_path, days = EMPTY_PROPOSED_AGE_DAYS + 1)
    report = run_compact(db_path)
    assert old["id"] in report.emptied
    assert fresh["id"] not in report.emptied
    assert get_record(old["id"], db_path = db_path)["status"] == "rejected"
    assert get_record(fresh["id"], db_path = db_path)["status"] == "proposed"


def test_supersede_chain_keeps_head_and_two_ancestors(db_path):
    rec = insert_record(
        kind = "claim",
        title = "Rate",
        body = "v0",
        provenance = "human",
        db_path = db_path,
    )
    chain = [rec]
    for step in range(4):
        rec = supersede_record(rec["id"], body = f"v{step + 1}", db_path = db_path)
        chain.append(rec)
    report = run_compact(db_path)
    head = chain[-1]
    kept = chain[-(KEEP_SUPERSEDED_ANCESTORS + 1) : -1]
    folded = chain[: -(KEEP_SUPERSEDED_ANCESTORS + 1)]
    assert get_record(head["id"], db_path = db_path)["status"] == "active"
    for rec in kept:
        assert get_record(rec["id"], db_path = db_path)["status"] == "superseded"
    for rec in folded:
        assert get_record(rec["id"], db_path = db_path)["status"] == "deprecated"
        assert rec["id"] in report.folded


def test_dry_run_changes_nothing(db_path):
    world = insert_record(
        kind = "claim",
        title = "Dup",
        body = "world copy",
        provenance = "world",
        db_path = db_path,
    )
    infer = insert_record(
        kind = "claim",
        title = "Dup",
        body = "infer copy",
        provenance = "infer",
        db_path = db_path,
    )
    empty = insert_record(
        kind = "claim",
        title = "Empty leftover",
        body = "(empty)",
        provenance = "infer",
        status = "proposed",
        db_path = db_path,
    )
    _age_created_at(empty["id"], db_path, days = EMPTY_PROPOSED_AGE_DAYS + 1)
    rec = insert_record(
        kind = "procedure",
        title = "Steps",
        body = "v0",
        provenance = "human",
        db_path = db_path,
    )
    for step in range(4):
        rec = supersede_record(rec["id"], body = f"v{step + 1}", db_path = db_path)
    before = {row["id"]: (row["status"], row["body"]) for row in list_records(db_path = db_path)}
    report = run_compact(db_path, dry_run = True)
    after = {row["id"]: (row["status"], row["body"]) for row in list_records(db_path = db_path)}
    assert report.dry_run is True
    assert (infer["id"], world["id"]) in report.deduped
    assert empty["id"] in report.emptied
    assert report.folded
    assert after == before
