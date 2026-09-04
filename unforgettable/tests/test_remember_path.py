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

"""Locks the context → B → pack path against the holes public review will poke."""

from __future__ import annotations

import json

from unforgettable.constants import EVENT_SUMMARY_CHARS, RECORD_BODY_CHARS
from unforgettable.loop.context import EpisodeState
from unforgettable.loop.runtime import bind_episode, reset_episode, set_contact
from unforgettable.store.records import get_record, insert_record, list_records
from unforgettable.tools.handlers import UNBOUND_DB_ERROR, dispatch


def test_unbound_dispatch_does_not_create_a_home_db():
    result = dispatch(
        "memory_write",
        {
            "kind": "claim",
            "title": "orphan",
            "body": "should not land in ~/.unforgettable",
            "provenance": "world",
        },
    )
    assert result == UNBOUND_DB_ERROR


def test_tools_cannot_claim_human_or_write_episode(db_path):
    human = json.loads(
        dispatch(
            "memory_write",
            {
                "kind": "directive",
                "title": "Always cite ids",
                "body": "the model claimed this was human",
                "provenance": "human",
            },
            db_path = db_path,
        )
    )
    rec = get_record(human["id"], db_path = db_path)
    assert rec["provenance"] == "infer"
    assert rec["status"] == "proposed"
    refused = dispatch(
        "memory_write",
        {
            "kind": "episode",
            "title": "Episode leaked",
            "body": "chat transcript must not be B",
            "provenance": "world",
        },
        db_path = db_path,
    )
    assert refused.startswith("Error:")
    assert not any(row["kind"] == "episode" for row in list_records(db_path = db_path))


def test_sim_contact_cannot_mint_world_provenance(db_path):
    tokens, _ = bind_episode(db_path = str(db_path), episode_id = "ep-sim")
    try:
        set_contact("sim")
        written = json.loads(
            dispatch(
                "memory_write",
                {
                    "kind": "claim",
                    "title": "World rate",
                    "body": "laundered from sim",
                    "provenance": "world",
                },
                db_path = db_path,
            )
        )
    finally:
        reset_episode(tokens)
    rec = get_record(written["id"], db_path = db_path)
    assert rec["provenance"] == "sim"
    assert rec["contact_tag"] == "sim"
    assert rec["status"] == "proposed"


def test_supersede_does_not_promote_proposed(db_path):
    first = json.loads(
        dispatch(
            "memory_write",
            {
                "kind": "error_fix",
                "title": "Error then fix",
                "body": "Tried: boom\nThen: maybe",
                "provenance": "infer",
            },
            db_path = db_path,
        )
    )
    assert first["status"] == "proposed"
    second = json.loads(
        dispatch(
            "memory_supersede",
            {
                "id": first["id"],
                "body": "Tried: boom\nThen: world looks fine",
                "provenance": "world",
            },
            db_path = db_path,
        )
    )
    assert second["status"] == "proposed"
    assert get_record(first["id"], db_path = db_path)["status"] == "superseded"


def test_rejected_cannot_be_superseded(db_path):
    rec = insert_record(
        kind = "claim",
        title = "Dead",
        body = "no",
        provenance = "infer",
        status = "rejected",
        db_path = db_path,
    )
    out = dispatch(
        "memory_supersede",
        {"id": rec["id"], "body": "revived"},
        db_path = db_path,
    )
    assert out.startswith("Error:")
    assert get_record(rec["id"], db_path = db_path)["status"] == "rejected"


def test_generate_text_is_clipped_before_it_becomes_an_event():
    state = EpisodeState(episode_id = "ep-clip", world_session = "world")
    state.note_success("x" * 4000, "world")
    assert len(state.trace_events[0]["summary"]) <= EVENT_SUMMARY_CHARS + 3


def test_insert_record_clips_unbounded_bodies(db_path):
    rec = insert_record(
        kind = "procedure",
        title = "Huge",
        body = "B" * 20000,
        provenance = "world",
        db_path = db_path,
    )
    assert len(rec["body"]) == RECORD_BODY_CHARS


def test_memory_search_strips_episode_kind(db_path):
    insert_record(
        kind = "episode",
        title = "Episode secret",
        body = "EPISODE_SECRET last user said ship it",
        provenance = "mixed",
        db_path = db_path,
    )
    insert_record(
        kind = "claim",
        title = "Ship it",
        body = "the real claim",
        provenance = "world",
        db_path = db_path,
    )
    out = dispatch(
        "memory_search",
        {"query": "ship", "kinds": "episode,claim"},
        db_path = db_path,
    )
    assert "EPISODE_SECRET" not in out
    hits = json.loads(out)
    assert hits[0]["kind"] == "claim"
