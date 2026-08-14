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
    supersede_record,
)
from unforgettable.store.search import search_records


def test_schema_and_crud(db_path):
    ensure_default_namespace(db_path=db_path)
    rec = insert_record(
        kind="claim",
        title="Pump max rate",
        body="Pump X max rate is 12 L/min",
        provenance="world",
        db_path=db_path,
    )
    loaded = get_record(rec["id"], db_path=db_path)
    assert loaded is not None
    assert loaded["kind"] == "claim"
    assert loaded["status"] == "active"


def test_supersede_keeps_history(db_path):
    old = insert_record(
        kind="claim",
        title="Rate",
        body="rate is 10",
        provenance="human",
        db_path=db_path,
    )
    new = supersede_record(old["id"], body="rate is 12", provenance="world", db_path=db_path)
    assert get_record(old["id"], db_path=db_path)["status"] == "superseded"
    assert new["supersedes_id"] == old["id"]
    assert new["body"] == "rate is 12"
    assert new["status"] == "active"


def test_deprecate_hidden_from_default_search(db_path):
    rec = insert_record(
        kind="claim",
        title="Secret convention",
        body="use snake_case for tests",
        provenance="human",
        db_path=db_path,
    )
    hits = search_records("snake_case", db_path=db_path)
    assert any(h["id"] == rec["id"] for h in hits)
    deprecate_record(rec["id"], reason="obsolete", db_path=db_path)
    hits = search_records("snake_case", db_path=db_path)
    assert hits == []
    still = get_record(rec["id"], db_path=db_path)
    assert still["status"] == "deprecated"


def test_search_prefers_world_over_infer(db_path):
    insert_record(
        kind="claim",
        title="Friction",
        body="surface friction is high on steel",
        provenance="infer",
        db_path=db_path,
    )
    world = insert_record(
        kind="claim",
        title="Friction",
        body="surface friction is high on steel in the world",
        provenance="world",
        db_path=db_path,
    )
    hits = search_records("friction steel", db_path=db_path)
    assert hits
    assert hits[0]["id"] == world["id"]
