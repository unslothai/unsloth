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
    EMPTY_PROPOSED_AGE_DAYS,
    KEEP_SUPERSEDED_ANCESTORS,
    run_compact,
)
from unforgettable.store.db import get_connection
from unforgettable.store.records import get_record, insert_record, list_records, supersede_record


def _age_created_at(record_id: str, db_path, *, days: int) -> None:
    past = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    conn = get_connection(db_path)
    try:
        conn.execute(
            "UPDATE records SET created_at = ? WHERE id = ?",
            (past, record_id),
        )
        conn.commit()
    finally:
        conn.close()


def test_duplicate_claims_deprecate_loser(db_path):
    world = insert_record(
        kind="claim",
        title="Friction",
        body="surface friction is high on steel in the world",
        provenance="world",
        db_path=db_path,
    )
    infer = insert_record(
        kind="claim",
        title="Friction",
        body="surface friction is high on steel",
        provenance="infer",
        db_path=db_path,
    )
    report = run_compact(db_path)
    assert report.dry_run is False
    assert (infer["id"], world["id"]) in report.deduped
    loser = get_record(infer["id"], db_path=db_path)
    winner = get_record(world["id"], db_path=db_path)
    assert loser["status"] == "deprecated"
    assert f"duplicate of {world['id']}" in loser["body"]
    assert winner["status"] == "active"


def test_same_title_twin_notes_stay_active(db_path):
    first = insert_record(
        kind="twin_note",
        title="World/sim disagreement",
        body="sim said yes; world said no",
        provenance="mixed",
        db_path=db_path,
    )
    second = insert_record(
        kind="twin_note",
        title="World/sim disagreement",
        body="another drifted episode",
        provenance="mixed",
        db_path=db_path,
    )
    run_compact(db_path)
    assert get_record(first["id"], db_path=db_path)["status"] == "active"
    assert get_record(second["id"], db_path=db_path)["status"] == "active"


def test_same_title_error_fixes_stay_active(db_path):
    first = insert_record(
        kind="error_fix",
        title="Error then fix",
        body="first failure then success",
        provenance="mixed",
        db_path=db_path,
    )
    second = insert_record(
        kind="error_fix",
        title="Error then fix",
        body="another failure then success",
        provenance="mixed",
        db_path=db_path,
    )
    run_compact(db_path)
    assert get_record(first["id"], db_path=db_path)["status"] == "active"
    assert get_record(second["id"], db_path=db_path)["status"] == "active"


def test_old_empty_proposed_is_rejected(db_path):
    old = insert_record(
        kind="claim",
        title="Empty leftover",
        body="",
        provenance="infer",
        status="proposed",
        db_path=db_path,
    )
    fresh = insert_record(
        kind="claim",
        title="Fresh empty",
        body="todo",
        provenance="infer",
        status="proposed",
        db_path=db_path,
    )
    _age_created_at(old["id"], db_path, days=EMPTY_PROPOSED_AGE_DAYS + 1)
    report = run_compact(db_path)
    assert old["id"] in report.emptied
    assert fresh["id"] not in report.emptied
    assert get_record(old["id"], db_path=db_path)["status"] == "rejected"
    assert get_record(fresh["id"], db_path=db_path)["status"] == "proposed"


def test_supersede_chain_keeps_head_and_two_ancestors(db_path):
    rec = insert_record(
        kind="claim",
        title="Rate",
        body="v0",
        provenance="human",
        db_path=db_path,
    )
    chain = [rec]
    for step in range(4):
        rec = supersede_record(rec["id"], body=f"v{step + 1}", db_path=db_path)
        chain.append(rec)
    report = run_compact(db_path)
    head = chain[-1]
    kept = chain[-(KEEP_SUPERSEDED_ANCESTORS + 1) : -1]
    folded = chain[: -(KEEP_SUPERSEDED_ANCESTORS + 1)]
    assert get_record(head["id"], db_path=db_path)["status"] == "active"
    for rec in kept:
        assert get_record(rec["id"], db_path=db_path)["status"] == "superseded"
    for rec in folded:
        assert get_record(rec["id"], db_path=db_path)["status"] == "deprecated"
        assert rec["id"] in report.folded


def test_dry_run_changes_nothing(db_path):
    world = insert_record(
        kind="claim",
        title="Dup",
        body="world copy",
        provenance="world",
        db_path=db_path,
    )
    infer = insert_record(
        kind="claim",
        title="Dup",
        body="infer copy",
        provenance="infer",
        db_path=db_path,
    )
    empty = insert_record(
        kind="claim",
        title="Empty leftover",
        body="(empty)",
        provenance="infer",
        status="proposed",
        db_path=db_path,
    )
    _age_created_at(empty["id"], db_path, days=EMPTY_PROPOSED_AGE_DAYS + 1)
    rec = insert_record(
        kind="procedure",
        title="Steps",
        body="v0",
        provenance="human",
        db_path=db_path,
    )
    for step in range(4):
        rec = supersede_record(rec["id"], body=f"v{step + 1}", db_path=db_path)
    before = {
        row["id"]: (row["status"], row["body"])
        for row in list_records(db_path=db_path)
    }
    report = run_compact(db_path, dry_run=True)
    after = {
        row["id"]: (row["status"], row["body"])
        for row in list_records(db_path=db_path)
    }
    assert report.dry_run is True
    assert (infer["id"], world["id"]) in report.deduped
    assert empty["id"] in report.emptied
    assert report.folded
    assert after == before
