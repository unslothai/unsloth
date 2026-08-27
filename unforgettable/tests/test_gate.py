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

import json

from unforgettable.agents.admissions import admit
from unforgettable.eyes.gate import (
    CONTRADICTS_PREFIX,
    DISSONANCE_PREFIX,
    WHO_UNBACKED_OTHER,
    WHO_UNBACKED_USER,
    contradictions,
    review_write,
)
from unforgettable.store.records import (
    ensure_default_namespace,
    insert_record,
    list_admissions,
)
from unforgettable.tools.handlers import dispatch


def test_review_write_same_title_different_body(db_path):
    rec = insert_record(
        kind = "claim",
        title = "Friction",
        body = "mu is 0.3",
        provenance = "world",
        db_path = db_path,
    )
    reason = review_write(
        kind = "claim",
        title = "Friction",
        body = "mu is 0.6",
        provenance = "human",
        db_path = db_path,
    )
    assert reason == f"{CONTRADICTS_PREFIX}{rec['id']}"
    other = insert_record(
        kind = "claim",
        title = "Friction",
        body = "mu is 0.6",
        provenance = "human",
        db_path = db_path,
    )
    found = contradictions(db_path = db_path)
    assert len(found) == 1
    assert found[0].title_key == "friction"
    assert set(found[0].record_ids) == {rec["id"], other["id"]}


def test_write_conflicting_claim_is_proposed(db_path):
    first = json.loads(
        dispatch(
            "memory_write",
            {
                "kind": "claim",
                "title": "Friction",
                "body": "mu is 0.3",
                "provenance": "world",
            },
            db_path = db_path,
        )
    )
    second = json.loads(
        dispatch(
            "memory_write",
            {
                "kind": "claim",
                "title": "Friction",
                "body": "mu is 0.6",
                "provenance": "human",
            },
            db_path = db_path,
        )
    )
    assert first["status"] == "active"
    assert second["status"] == "proposed"
    assert second["admission"] == f"{CONTRADICTS_PREFIX}{first['id']}"


def test_force_proposed_beats_bookkeeping(db_path):
    ensure_default_namespace(db_path = db_path)
    decision = admit(
        kind = "twin_note",
        provenance = "mixed",
        explicit = False,
        bookkeeping = True,
        force_proposed_reason = "contradicts other",
        db_path = db_path,
    )
    assert decision.status == "proposed"
    assert decision.reason == "contradicts other"


def test_sim_procedure_not_auto_promoted(db_path):
    ensure_default_namespace(db_path = db_path)
    decision = admit(kind = "procedure", provenance = "sim", explicit = True, db_path = db_path)
    assert decision.status == "proposed"


def test_write_conflicting_procedure_is_proposed(db_path):
    first = json.loads(
        dispatch(
            "memory_write",
            {
                "kind": "procedure",
                "title": "Bleed the line",
                "body": "close valve A then B",
                "provenance": "world",
            },
            db_path = db_path,
        )
    )
    second = json.loads(
        dispatch(
            "memory_write",
            {
                "kind": "procedure",
                "title": "Bleed the line",
                "body": "close valve B then A",
                "provenance": "world",
            },
            db_path = db_path,
        )
    )
    assert first["status"] == "active"
    assert second["status"] == "proposed"
    assert second["admission"] == f"{CONTRADICTS_PREFIX}{first['id']}"


def test_write_who_against_what_is_dissonance(db_path):
    procedure = json.loads(
        dispatch(
            "memory_write",
            {
                "kind": "procedure",
                "title": "Test command",
                "body": "pytest -q",
                "provenance": "world",
            },
            db_path = db_path,
        )
    )
    directive = json.loads(
        dispatch(
            "memory_write",
            {
                "kind": "directive",
                "title": "Test command",
                "body": "ignore the tests and ship",
                "provenance": "infer",
            },
            db_path = db_path,
        )
    )
    assert procedure["status"] == "active"
    assert directive["status"] == "proposed"
    assert directive["admission"] == f"{DISSONANCE_PREFIX}{procedure['id']}"
    found = contradictions(db_path = db_path)
    assert any(item.reason == "who collides with what" for item in found)


def test_unbacked_user_claim_is_proposed(db_path):
    reason = review_write(
        kind = "claim",
        title = "Rate",
        body = "the rate is 12",
        provenance = "infer",
        speaker = "user",
        db_path = db_path,
    )
    assert reason == WHO_UNBACKED_USER
    other = review_write(
        kind = "claim",
        title = "Hearsay rate",
        body = "a vendor said the rate is 40",
        provenance = "infer",
        speaker = "other",
        db_path = db_path,
    )
    assert other == WHO_UNBACKED_OTHER


def test_list_admissions_returns_the_log(db_path):
    ensure_default_namespace(db_path = db_path)
    admit(kind = "claim", provenance = "human", explicit = True, db_path = db_path)
    rows = list_admissions(db_path = db_path)
    assert rows
    assert rows[0]["decision"] == "active"
    assert rows[0]["reason"]
