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

from unforgettable.store.records import (
    deprecate_record,
    ensure_default_namespace,
    get_record,
    insert_record,
    list_records,
    summarize_records,
    supersede_record,
    update_proposed_record,
)
from unforgettable.store.search import search_records


def test_schema_and_crud(db_path):
    ensure_default_namespace(db_path = db_path)
    rec = insert_record(
        kind = "claim",
        title = "Pump max rate",
        body = "Pump X max rate is 12 L/min",
        provenance = "world",
        db_path = db_path,
    )
    loaded = get_record(rec["id"], db_path = db_path)
    assert loaded is not None
    assert loaded["kind"] == "claim"
    assert loaded["status"] == "active"
    assert loaded["speaker"] == "world"
    assert loaded["warrant"] == ""


def test_unbacked_user_claim_cannot_mint_world(db_path):
    rec = insert_record(
        kind = "claim",
        title = "Rate is twelve",
        body = "the rate is 12",
        provenance = "world",
        speaker = "user",
        db_path = db_path,
    )
    loaded = get_record(rec["id"], db_path = db_path)
    assert loaded["provenance"] == "infer"
    assert loaded["speaker"] == "user"


def test_supersede_keeps_history(db_path):
    old = insert_record(
        kind = "claim",
        title = "Rate",
        body = "rate is 10",
        provenance = "human",
        db_path = db_path,
    )
    new = supersede_record(old["id"], body = "rate is 12", provenance = "world", db_path = db_path)
    assert get_record(old["id"], db_path = db_path)["status"] == "superseded"
    assert new["supersedes_id"] == old["id"]
    assert new["body"] == "rate is 12"
    assert new["status"] == "active"


def test_deprecate_hidden_from_default_search(db_path):
    rec = insert_record(
        kind = "claim",
        title = "Secret convention",
        body = "use snake_case for tests",
        provenance = "human",
        db_path = db_path,
    )
    hits = search_records("snake_case", db_path = db_path)
    assert any(h["id"] == rec["id"] for h in hits)
    deprecate_record(rec["id"], reason = "obsolete", db_path = db_path)
    hits = search_records("snake_case", db_path = db_path)
    assert hits == []
    still = get_record(rec["id"], db_path = db_path)
    assert still["status"] == "deprecated"


def test_search_prefers_world_over_infer(db_path):
    insert_record(
        kind = "claim",
        title = "Friction",
        body = "surface friction is high on steel",
        provenance = "infer",
        db_path = db_path,
    )
    world = insert_record(
        kind = "claim",
        title = "Friction",
        body = "surface friction is high on steel in the world",
        provenance = "world",
        db_path = db_path,
    )
    hits = search_records("friction steel", db_path = db_path)
    assert hits
    assert hits[0]["id"] == world["id"]


def test_summarize_records_groups_status_kind_provenance(db_path):
    insert_record(
        kind = "claim",
        title = "A",
        body = "a",
        provenance = "world",
        db_path = db_path,
    )
    insert_record(
        kind = "procedure",
        title = "B",
        body = "b",
        provenance = "infer",
        status = "proposed",
        db_path = db_path,
    )
    insert_record(
        kind = "claim",
        title = "C",
        body = "c",
        provenance = "world",
        status = "proposed",
        db_path = db_path,
    )
    summary = summarize_records(db_path = db_path)
    assert summary["total"] == 3
    assert summary["by_status"]["active"] == 1
    assert summary["by_status"]["proposed"] == 2
    assert summary["by_kind"]["claim"] == 2
    assert summary["by_kind"]["procedure"] == 1
    assert summary["by_provenance"]["world"] == 2
    assert summary["by_provenance"]["infer"] == 1
    assert summary["by_status"]["rejected"] == 0


def test_list_records_offset(db_path):
    for index in range(3):
        insert_record(
            kind = "claim",
            title = f"Row {index}",
            body = f"body {index}",
            provenance = "world",
            db_path = db_path,
        )
    first = list_records(limit = 1, db_path = db_path)
    second = list_records(limit = 1, offset = 1, db_path = db_path)
    assert len(first) == 1
    assert len(second) == 1
    assert first[0]["id"] != second[0]["id"]


def test_update_proposed_record_edits_in_place(db_path):
    rec = insert_record(
        kind = "claim",
        title = "Draft",
        body = "old",
        provenance = "infer",
        status = "proposed",
        db_path = db_path,
    )
    updated = update_proposed_record(rec["id"], title = "Draft v2", body = "new", db_path = db_path)
    assert updated["id"] == rec["id"]
    assert updated["status"] == "proposed"
    assert updated["title"] == "Draft v2"
    assert updated["body"] == "new"
    hits = search_records("new", statuses = ["proposed"], db_path = db_path)
    assert any(hit["id"] == rec["id"] for hit in hits)


def test_update_proposed_record_refuses_active(db_path):
    rec = insert_record(
        kind = "claim",
        title = "Live",
        body = "stays",
        provenance = "world",
        db_path = db_path,
    )
    try:
        update_proposed_record(rec["id"], body = "nope", db_path = db_path)
    except ValueError as exc:
        assert "proposed" in str(exc)
    else:
        raise AssertionError("active rows must not edit in place")
    assert get_record(rec["id"], db_path = db_path)["body"] == "stays"
