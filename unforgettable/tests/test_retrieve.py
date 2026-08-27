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

from unforgettable.agents.retriever import RetrievePolicy, format_inject, retrieve
from unforgettable.constants import (
    TYPOLOGY_WHAT_BACKED,
    TYPOLOGY_WHAT_WORLD,
    TYPOLOGY_WHO_OTHER,
    TYPOLOGY_WHO_USER,
    typology_class,
)
from unforgettable.store.db import get_connection
from unforgettable.store.records import insert_record
from unforgettable.store.search import search_records

_INJECT_HEADER = "Durable memories relevant to this task:"


def _set_updated_at(record_id: str, updated_at: str, db_path) -> None:
    conn = get_connection(db_path)
    try:
        conn.execute("UPDATE records SET updated_at = ? WHERE id = ?", (updated_at, record_id))
        conn.commit()
    finally:
        conn.close()


def test_max_chars_keeps_first_hit(db_path):
    insert_record(
        kind = "procedure",
        title = "Long procedure",
        body = "x" * 500,
        provenance = "world",
        db_path = db_path,
    )
    policy = RetrievePolicy(max_chars = 100)
    hits = retrieve("procedure", policy = policy, db_path = db_path)
    assert len(hits) >= 1
    text = format_inject(hits, policy = policy)
    assert _INJECT_HEADER in text
    assert len(text) <= 100 + len(_INJECT_HEADER) + 1
    assert "Long procedure" in text


def test_high_stakes_excludes_sim_in_favor_of_world(db_path):
    sim = insert_record(
        kind = "claim",
        title = "Friction",
        body = "surface friction is high on steel from sim",
        provenance = "sim",
        db_path = db_path,
    )
    world = insert_record(
        kind = "claim",
        title = "Friction",
        body = "surface friction is high on steel in the world",
        provenance = "world",
        db_path = db_path,
    )
    hits = retrieve("friction steel", policy = RetrievePolicy(high_stakes = True), db_path = db_path)
    ids = {hit["id"] for hit in hits}
    assert world["id"] in ids
    assert sim["id"] not in ids


def test_episode_row_is_not_injected(db_path):
    insert_record(
        kind = "episode",
        title = "Last pytest run",
        body = "We ran pytest and it passed last episode.",
        provenance = "world",
        db_path = db_path,
    )
    claim = insert_record(
        kind = "claim",
        title = "pytest is the runner",
        body = "Use pytest to run tests.",
        provenance = "world",
        db_path = db_path,
    )
    hits = retrieve("pytest", db_path = db_path)
    assert all(hit["kind"] != "episode" for hit in hits)
    assert any(hit["id"] == claim["id"] for hit in hits)
    text = format_inject(hits)
    assert "Last pytest run" not in text
    assert "pytest is the runner" in text


def test_second_twin_note_dropped(db_path):
    older = insert_record(
        kind = "twin_note",
        title = "Twin old",
        body = "older twin note about deploy",
        provenance = "world",
        db_path = db_path,
    )
    newer = insert_record(
        kind = "twin_note",
        title = "Twin new",
        body = "newer twin note about deploy",
        provenance = "world",
        db_path = db_path,
    )
    insert_record(
        kind = "claim",
        title = "Deploy window",
        body = "deploy on Tuesday",
        provenance = "human",
        db_path = db_path,
    )
    _set_updated_at(older["id"], "2020-01-01T00:00:00+00:00", db_path)
    _set_updated_at(newer["id"], "2024-06-01T00:00:00+00:00", db_path)
    hits = retrieve("deploy", policy = RetrievePolicy(max_twin_notes = 1), db_path = db_path)
    twins = [hit for hit in hits if hit["kind"] == "twin_note"]
    assert [twin["id"] for twin in twins] == [newer["id"]]
    assert any(hit["title"] == "Deploy window" for hit in hits)


def test_sim_contact_high_stakes_keeps_sim_error_fix(db_path):
    sim = insert_record(
        kind = "error_fix",
        title = "Sim traceback lesson",
        body = "clone pytest failed on import of app",
        provenance = "sim",
        db_path = db_path,
    )
    world = insert_record(
        kind = "claim",
        title = "World traceback note",
        body = "world pytest failed on import of app",
        provenance = "world",
        db_path = db_path,
    )
    hits = retrieve(
        "pytest failed import",
        policy = RetrievePolicy(contact = "sim", high_stakes = True),
        db_path = db_path,
    )
    ids = {hit["id"] for hit in hits}
    assert sim["id"] in ids
    assert world["id"] in ids
    assert hits[0]["id"] == sim["id"]


def test_world_contact_high_stakes_drops_sim(db_path):
    sim = insert_record(
        kind = "error_fix",
        title = "Sim traceback lesson",
        body = "clone pytest failed on import of app",
        provenance = "sim",
        db_path = db_path,
    )
    world = insert_record(
        kind = "claim",
        title = "World traceback note",
        body = "world pytest failed on import of app",
        provenance = "world",
        db_path = db_path,
    )
    hits = retrieve(
        "pytest failed import",
        policy = RetrievePolicy(contact = "world", high_stakes = True),
        db_path = db_path,
    )
    ids = {hit["id"] for hit in hits}
    assert world["id"] in ids
    assert sim["id"] not in ids


def test_sim_policy_keeps_three_twin_notes(db_path):
    twins = []
    for index, when in enumerate(
        (
            "2020-01-01T00:00:00+00:00",
            "2022-01-01T00:00:00+00:00",
            "2024-06-01T00:00:00+00:00",
        )
    ):
        rec = insert_record(
            kind = "twin_note",
            title = f"Twin {index}",
            body = "twin note about deploy window",
            provenance = "world",
            db_path = db_path,
        )
        _set_updated_at(rec["id"], when, db_path)
        twins.append(rec)
    one = retrieve("deploy", policy = RetrievePolicy(max_twin_notes = 1), db_path = db_path)
    one_twins = [hit for hit in one if hit["kind"] == "twin_note"]
    assert [twin["id"] for twin in one_twins] == [twins[-1]["id"]]
    three = retrieve(
        "deploy",
        policy = RetrievePolicy(contact = "sim", max_twin_notes = 3),
        db_path = db_path,
    )
    three_ids = {hit["id"] for hit in three if hit["kind"] == "twin_note"}
    assert three_ids == {twin["id"] for twin in twins}


def test_exclude_ids_drops_from_inject_not_search(db_path):
    compiled = insert_record(
        kind = "procedure",
        title = "How we run the formatter",
        body = "Always run ruff then pytest.",
        provenance = "world",
        db_path = db_path,
    )
    other = insert_record(
        kind = "claim",
        title = "Formatter config",
        body = "ruff settings live in pyproject.",
        provenance = "world",
        db_path = db_path,
    )
    hits = retrieve(
        "formatter",
        policy = RetrievePolicy(exclude_ids = frozenset({compiled["id"]})),
        db_path = db_path,
    )
    text = format_inject(hits)
    assert compiled["title"] not in text
    assert other["title"] in text
    found = search_records("formatter", db_path = db_path)
    assert any(hit["id"] == compiled["id"] for hit in found)


def test_retrieve_drops_colliding_who(db_path):
    procedure = insert_record(
        kind = "procedure",
        title = "Friction",
        body = "measure mu on the steel coupon",
        provenance = "world",
        db_path = db_path,
    )
    directive = insert_record(
        kind = "directive",
        title = "Friction",
        body = "the executive is always right about friction",
        provenance = "human",
        speaker = "user",
        db_path = db_path,
    )
    assert typology_class(procedure) == TYPOLOGY_WHAT_WORLD
    assert typology_class(directive) == TYPOLOGY_WHO_USER
    hits = retrieve("friction", db_path = db_path)
    ids = {hit["id"] for hit in hits}
    assert procedure["id"] in ids
    assert directive["id"] not in ids
    text = format_inject(hits)
    assert "the executive is always right" not in text
    assert "(procedure, world, world)" in text


def test_backed_user_text_outranks_other_hearsay(db_path):
    backed = insert_record(
        kind = "claim",
        title = "Pump rate",
        body = "Pump X max rate is 12 L/min because the nameplate says so.",
        provenance = "human",
        speaker = "user",
        warrant = "nameplate photo in docs/pump-x.md",
        db_path = db_path,
    )
    hearsay = insert_record(
        kind = "claim",
        title = "Pump rumor",
        body = "someone said Pump X max rate is 40 L/min",
        provenance = "infer",
        speaker = "other",
        speaker_label = "slack thread",
        db_path = db_path,
    )
    assert typology_class(backed) == TYPOLOGY_WHAT_BACKED
    assert typology_class(hearsay) == TYPOLOGY_WHO_OTHER
    hits = search_records("Pump rate L/min", db_path = db_path)
    assert hits[0]["id"] == backed["id"]
