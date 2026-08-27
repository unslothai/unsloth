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

import pytest

from unforgettable.agents.retriever import RetrievePolicy, retrieve
from unforgettable.store.compile import (
    format_standing,
    get_compiled,
    is_compile_candidate,
    list_standing,
    maybe_compile,
    pack_standing,
    pin_compiled,
    procedure_hits,
    refresh_compiled,
    unpin_compiled,
)
from unforgettable.store.records import (
    deprecate_record,
    insert_record,
    insert_retrieve_use,
    insert_rollout,
)


def _world_pass_use(record_id: str, episode_id: str, db_path) -> None:
    insert_retrieve_use(
        episode_id = episode_id,
        record_id = record_id,
        contact = "world",
        db_path = db_path,
    )
    insert_rollout(
        episode_id = episode_id,
        contact = "world",
        outcome = "pass",
        summary = "ok",
        db_path = db_path,
    )


def test_zero_hits_trusted_procedure_is_not_candidate(db_path):
    rec = insert_record(
        kind = "procedure",
        title = "How we run the formatter",
        body = "Always run ruff, then pytest.",
        provenance = "world",
        db_path = db_path,
    )
    assert procedure_hits(rec["id"], db_path = db_path) == 0
    assert not is_compile_candidate(rec, hits = 0, explicit = False)
    assert maybe_compile(db_path) == []
    assert get_compiled(rec["id"], db_path = db_path) is None


def test_maybe_compile_pins_after_two_world_pass_hits(db_path):
    rec = insert_record(
        kind = "procedure",
        title = "How we run the formatter",
        body = "Always run ruff, then pytest.",
        provenance = "world",
        db_path = db_path,
    )
    _world_pass_use(rec["id"], "ep-1", db_path)
    assert procedure_hits(rec["id"], db_path = db_path) == 1
    assert maybe_compile(db_path) == []
    assert get_compiled(rec["id"], db_path = db_path) is None
    _world_pass_use(rec["id"], "ep-2", db_path)
    assert procedure_hits(rec["id"], db_path = db_path) == 2
    pinned = maybe_compile(db_path)
    assert rec["id"] in pinned
    row = get_compiled(rec["id"], db_path = db_path)
    assert row is not None
    assert not row["explicit"]
    assert maybe_compile(db_path) == []


def test_sim_only_retrieve_uses_do_not_count(db_path):
    rec = insert_record(
        kind = "procedure",
        title = "How we run the formatter",
        body = "Always run ruff, then pytest.",
        provenance = "world",
        db_path = db_path,
    )
    insert_retrieve_use(
        episode_id = "ep-sim-1",
        record_id = rec["id"],
        contact = "sim",
        db_path = db_path,
    )
    insert_rollout(
        episode_id = "ep-sim-1",
        contact = "world",
        outcome = "pass",
        summary = "ok",
        db_path = db_path,
    )
    insert_retrieve_use(
        episode_id = "ep-sim-2",
        record_id = rec["id"],
        contact = "sim",
        db_path = db_path,
    )
    insert_rollout(
        episode_id = "ep-sim-2",
        contact = "world",
        outcome = "pass",
        summary = "ok",
        db_path = db_path,
    )
    insert_retrieve_use(
        episode_id = "ep-sim-3",
        record_id = rec["id"],
        contact = "world",
        db_path = db_path,
    )
    insert_rollout(
        episode_id = "ep-sim-3",
        contact = "sim",
        outcome = "pass",
        summary = "ok",
        db_path = db_path,
    )
    assert procedure_hits(rec["id"], db_path = db_path) == 0
    assert maybe_compile(db_path) == []
    assert get_compiled(rec["id"], db_path = db_path) is None


def test_probe_test_command_sim_proposed_cannot_auto_pin_with_hits(db_path):
    probe = insert_record(
        kind = "procedure",
        title = "Probe: old login",
        body = "echo login\n",
        provenance = "human",
        db_path = db_path,
    )
    command = insert_record(
        kind = "procedure",
        title = "Test Command",
        body = "pytest\n",
        provenance = "human",
        db_path = db_path,
    )
    sim = insert_record(
        kind = "procedure",
        title = "Sim only playbook",
        body = "rehearse in the clone",
        provenance = "sim",
        db_path = db_path,
    )
    proposed = insert_record(
        kind = "procedure",
        title = "Proposed playbook",
        body = "not admitted yet",
        provenance = "world",
        status = "proposed",
        db_path = db_path,
    )
    refused = (probe, command, sim, proposed)
    for rec in refused:
        _world_pass_use(rec["id"], f"ep-a-{rec['id'][:8]}", db_path)
        _world_pass_use(rec["id"], f"ep-b-{rec['id'][:8]}", db_path)
        assert procedure_hits(rec["id"], db_path = db_path) == 2
    assert maybe_compile(db_path) == []
    for rec in refused:
        assert get_compiled(rec["id"], db_path = db_path) is None
        with pytest.raises(ValueError):
            pin_compiled(rec["id"], explicit = False, db_path = db_path)
        with pytest.raises(ValueError):
            pin_compiled(rec["id"], explicit = True, db_path = db_path)


def test_maybe_compile_leaves_explicit_pin(db_path):
    rec = insert_record(
        kind = "procedure",
        title = "How we run the formatter",
        body = "Always run ruff, then pytest.",
        provenance = "world",
        db_path = db_path,
    )
    pin_compiled(rec["id"], explicit = True, db_path = db_path)
    assert maybe_compile(db_path) == []
    row = get_compiled(rec["id"], db_path = db_path)
    assert row is not None
    assert row["explicit"] == 1


def test_explicit_pin_trusted_world_procedure(db_path):
    rec = insert_record(
        kind = "procedure",
        title = "How we run the formatter",
        body = "Always run ruff, then pytest.",
        provenance = "world",
        db_path = db_path,
    )
    pin_compiled(rec["id"], explicit = True, db_path = db_path)
    text = format_standing(list_standing(db_path))
    assert rec["title"] in text
    assert f"Source: {rec['id']}" in text


def test_probe_and_test_command_cannot_pin(db_path):
    probe = insert_record(
        kind = "procedure",
        title = "Probe: old login",
        body = "echo login\n",
        provenance = "human",
        db_path = db_path,
    )
    command = insert_record(
        kind = "procedure",
        title = "Test Command",
        body = "pytest\n",
        provenance = "human",
        db_path = db_path,
    )
    with pytest.raises(ValueError):
        pin_compiled(probe["id"], explicit = True, db_path = db_path)
    with pytest.raises(ValueError):
        pin_compiled(command["id"], explicit = True, db_path = db_path)


def test_sim_and_proposed_never_pin(db_path):
    sim = insert_record(
        kind = "procedure",
        title = "Sim only playbook",
        body = "rehearse in the clone",
        provenance = "sim",
        db_path = db_path,
    )
    proposed = insert_record(
        kind = "procedure",
        title = "Proposed playbook",
        body = "not admitted yet",
        provenance = "world",
        status = "proposed",
        db_path = db_path,
    )
    with pytest.raises(ValueError):
        pin_compiled(sim["id"], explicit = True, db_path = db_path)
    with pytest.raises(ValueError):
        pin_compiled(proposed["id"], explicit = True, db_path = db_path)


def test_uncompile_blocks_auto_recompile(db_path):
    rec = insert_record(
        kind = "procedure",
        title = "How we format",
        body = "ruff then pytest",
        provenance = "world",
        db_path = db_path,
    )
    _world_pass_use(rec["id"], "ep-1", db_path)
    _world_pass_use(rec["id"], "ep-2", db_path)
    assert rec["id"] in maybe_compile(db_path)
    unpin_compiled(rec["id"], db_path = db_path)
    assert get_compiled(rec["id"], db_path = db_path) is None
    assert maybe_compile(db_path) == []
    pin_compiled(rec["id"], explicit = True, db_path = db_path)
    assert get_compiled(rec["id"], db_path = db_path) is not None


def test_refresh_drops_pin_after_deprecate(db_path):
    rec = insert_record(
        kind = "procedure",
        title = "How we ship",
        body = "tag then push",
        provenance = "world",
        db_path = db_path,
    )
    pin_compiled(rec["id"], explicit = True, db_path = db_path)
    deprecate_record(rec["id"], reason = "obsolete", db_path = db_path)
    dropped = refresh_compiled(db_path)
    assert rec["id"] in dropped
    assert get_compiled(rec["id"], db_path = db_path) is None


def test_standing_max_chars_keeps_first_section(db_path):
    rec = insert_record(
        kind = "procedure",
        title = "Long standing playbook",
        body = "step " * 400,
        provenance = "mixed",
        db_path = db_path,
    )
    pin_compiled(rec["id"], explicit = True, db_path = db_path)
    rows = list_standing(db_path)
    full = format_standing(rows)
    text = format_standing(rows, max_chars = 200)
    assert len(full) > 200
    assert len(text) <= 200
    assert "###" in text
    assert rec["title"] in text
    assert f"Source: {rec['id']}" in text


def test_standing_overflow_stays_fts_eligible(db_path):
    older = insert_record(
        kind = "procedure",
        title = "Older compiled playbook",
        body = "O" * 800,
        provenance = "world",
        db_path = db_path,
    )
    pin_compiled(older["id"], explicit = True, db_path = db_path)
    newer = insert_record(
        kind = "procedure",
        title = "Newer compiled playbook",
        body = "N" * 800,
        provenance = "world",
        db_path = db_path,
    )
    pin_compiled(newer["id"], explicit = True, db_path = db_path)
    text, kept = pack_standing(list_standing(db_path))
    kept_ids = {row["id"] for row in kept}
    assert newer["id"] in kept_ids
    assert older["id"] not in kept_ids
    assert f"Source: {newer['id']}" in text
    assert older["title"] not in text
    hits = retrieve(
        older["title"],
        policy = RetrievePolicy(exclude_ids = frozenset(kept_ids)),
        db_path = db_path,
    )
    assert any(hit["id"] == older["id"] for hit in hits)
