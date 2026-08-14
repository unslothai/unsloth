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

from unforgettable.store.compile import get_compiled, pin_compiled
from unforgettable.store.records import (
    get_record,
    insert_record,
    insert_retrieve_use,
    insert_rollout,
)
from unforgettable.tools.handlers import dispatch
from unforgettable.tools.specs import (
    CONTACT_TOOL_NAMES,
    MEMORY_COMPILE,
    MEMORY_TOOL_NAMES,
)


def test_tool_names_are_stable():
    assert MEMORY_TOOL_NAMES == {
        "memory_write",
        "memory_search",
        "memory_get",
        "memory_supersede",
        "memory_deprecate",
        "memory_compact",
        "memory_compile",
    }


def test_write_search_get_supersede_deprecate(db_path):
    written = json.loads(
        dispatch(
            "memory_write",
            {
                "kind": "directive",
                "title": "Always cite ids",
                "body": "Ground answers in returned memory ids.",
                "provenance": "human",
            },
            db_path=db_path,
        )
    )
    assert written["status"] == "active"
    search = json.loads(dispatch("memory_search", {"query": "cite ids"}, db_path=db_path))
    assert search[0]["id"] == written["id"]
    got = json.loads(dispatch("memory_get", {"id": written["id"]}, db_path=db_path))
    assert got["title"] == "Always cite ids"
    superseded = json.loads(
        dispatch(
            "memory_supersede",
            {"id": written["id"], "body": "Cite ids and provenance."},
            db_path=db_path,
        )
    )
    assert get_record(written["id"], db_path=db_path)["status"] == "superseded"
    deprecated = json.loads(
        dispatch(
            "memory_deprecate",
            {"id": superseded["id"], "reason": "replaced later"},
            db_path=db_path,
        )
    )
    assert deprecated["status"] == "deprecated"
    assert "No matching" in dispatch("memory_search", {"query": "cite ids"}, db_path=db_path)


def test_memory_compact_default_is_dry_run(db_path):
    first = insert_record(
        kind="claim",
        title="Dup",
        body="world copy",
        provenance="world",
        db_path=db_path,
    )
    second = insert_record(
        kind="claim",
        title="Dup",
        body="infer copy",
        provenance="infer",
        db_path=db_path,
    )
    payload = json.loads(dispatch("memory_compact", None, db_path=db_path))
    assert payload["dry_run"] is True
    assert get_record(first["id"], db_path=db_path)["status"] == "active"
    assert get_record(second["id"], db_path=db_path)["status"] == "active"


def test_memory_compact_wet_mutates(db_path):
    first = insert_record(
        kind="claim",
        title="Dup",
        body="world copy",
        provenance="world",
        db_path=db_path,
    )
    second = insert_record(
        kind="claim",
        title="Dup",
        body="infer copy",
        provenance="infer",
        db_path=db_path,
    )
    payload = json.loads(dispatch("memory_compact", {"dry_run": False}, db_path=db_path))
    assert payload["dry_run"] is False
    statuses = {
        get_record(first["id"], db_path=db_path)["status"],
        get_record(second["id"], db_path=db_path)["status"],
    }
    assert statuses == {"active", "deprecated"}


def test_rims_enter_sim_is_contact_tool():
    assert "rims_enter_sim" in CONTACT_TOOL_NAMES
    assert "rims_enter_sim" not in MEMORY_TOOL_NAMES
    assert dispatch("rims_enter_sim", {"reason": "rehearse"}) == "enter_sim requested"
    assert dispatch("rims.enter_sim", None) == "enter_sim requested"
    assert dispatch("rims_nope", {}).startswith("Error:")


def _world_pass_use(record_id: str, episode_id: str, db_path) -> None:
    insert_retrieve_use(
        episode_id=episode_id,
        record_id=record_id,
        contact="world",
        db_path=db_path,
    )
    insert_rollout(
        episode_id=episode_id,
        contact="world",
        outcome="pass",
        summary="ok",
        db_path=db_path,
    )


def _trusted_procedure(db_path):
    return insert_record(
        kind="procedure",
        title="How we run the formatter",
        body="Always run ruff, then pytest.",
        provenance="world",
        db_path=db_path,
    )


def test_memory_compile_is_a_memory_tool():
    assert "memory_compile" in MEMORY_TOOL_NAMES
    assert MEMORY_COMPILE["function"]["name"] == "memory_compile"
    assert "unpin" not in MEMORY_COMPILE["function"]["parameters"]["properties"]


def test_memory_compile_default_is_dry_run(db_path):
    rec = _trusted_procedure(db_path)
    _world_pass_use(rec["id"], "ep-1", db_path)
    _world_pass_use(rec["id"], "ep-2", db_path)
    payload = json.loads(dispatch("memory_compile", None, db_path=db_path))
    assert payload["dry_run"] is True
    assert get_compiled(rec["id"], db_path=db_path) is None
    none_payload = json.loads(
        dispatch("memory_compile", {"dry_run": None}, db_path=db_path)
    )
    assert none_payload["dry_run"] is True
    assert get_compiled(rec["id"], db_path=db_path) is None


def test_memory_compile_with_id_default_is_dry_run(db_path):
    rec = _trusted_procedure(db_path)
    _world_pass_use(rec["id"], "ep-1", db_path)
    _world_pass_use(rec["id"], "ep-2", db_path)
    payload = json.loads(dispatch("memory_compile", {"id": rec["id"]}, db_path=db_path))
    assert payload["dry_run"] is True
    assert payload["hits"] == 2
    assert payload["eligible"] is True
    assert get_compiled(rec["id"], db_path=db_path) is None
    none_payload = json.loads(
        dispatch("memory_compile", {"id": rec["id"], "dry_run": None}, db_path=db_path)
    )
    assert none_payload["dry_run"] is True
    assert none_payload["hits"] == 2
    assert none_payload["eligible"] is True
    assert get_compiled(rec["id"], db_path=db_path) is None
    probe = insert_record(
        kind="procedure",
        title="Probe: old login",
        body="echo login\n",
        provenance="human",
        db_path=db_path,
    )
    probe_payload = json.loads(
        dispatch("memory_compile", {"id": probe["id"]}, db_path=db_path)
    )
    assert probe_payload["dry_run"] is True
    assert probe_payload["eligible"] is False
    assert "hits" in probe_payload
    assert get_compiled(probe["id"], db_path=db_path) is None


def test_memory_compile_wet_without_id_runs_maybe_compile(db_path):
    rec = _trusted_procedure(db_path)
    _world_pass_use(rec["id"], "ep-1", db_path)
    _world_pass_use(rec["id"], "ep-2", db_path)
    payload = json.loads(
        dispatch("memory_compile", {"dry_run": False}, db_path=db_path)
    )
    assert payload["dry_run"] is False
    row = get_compiled(rec["id"], db_path=db_path)
    assert row is not None
    assert not row["explicit"]
    assert rec["id"] in payload["pinned"]


def test_memory_compile_wet_with_id_pins_explicit(db_path):
    rec = _trusted_procedure(db_path)
    payload = json.loads(
        dispatch(
            "memory_compile",
            {"id": rec["id"], "dry_run": False},
            db_path=db_path,
        )
    )
    assert payload["dry_run"] is False
    row = get_compiled(rec["id"], db_path=db_path)
    assert row is not None
    assert row["explicit"] == 1
    assert payload["pinned"] == rec["id"]


def test_memory_compile_cannot_unpin(db_path):
    rec = _trusted_procedure(db_path)
    pin_compiled(rec["id"], explicit=True, db_path=db_path)
    assert "memory_uncompile" not in MEMORY_TOOL_NAMES
    dispatch(
        "memory_compile",
        {"id": rec["id"], "dry_run": False, "unpin": True},
        db_path=db_path,
    )
    assert get_compiled(rec["id"], db_path=db_path) is not None
    dispatch("memory_uncompile", {"id": rec["id"], "dry_run": False}, db_path=db_path)
    assert get_compiled(rec["id"], db_path=db_path) is not None
