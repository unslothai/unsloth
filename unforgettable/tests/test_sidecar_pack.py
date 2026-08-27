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

import asyncio
import json

from unforgettable.loop.context import EpisodeState
from unforgettable.loop.episode import _extract
from unforgettable.sidecar import pack_from_admitted_b
from unforgettable.sidecar.format import preference_pairs
from unforgettable.sidecar.pack import (
    list_pack_items,
    list_packs,
    pack_is_retrieval_heavy,
)
from unforgettable.store.compile import pin_compiled
from unforgettable.store.records import (
    insert_inject_stats,
    insert_record,
    insert_retrieve_use,
    insert_rollout,
    list_records,
    list_rollouts,
    set_record_status,
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


def _sim_pass_use(record_id: str, episode_id: str, db_path) -> None:
    insert_retrieve_use(
        episode_id = episode_id,
        record_id = record_id,
        contact = "sim",
        db_path = db_path,
    )
    insert_rollout(
        episode_id = episode_id,
        contact = "sim",
        outcome = "pass",
        summary = "ok in sim",
        db_path = db_path,
    )


def _procedure(*, title: str, body: str, db_path, **kwargs):
    return insert_record(
        kind = "procedure",
        title = title,
        body = body,
        provenance = "world",
        db_path = db_path,
        **kwargs,
    )


def test_world_procedure_with_world_pass_is_packed(db_path):
    rec = _procedure(
        title = "How we run the formatter",
        body = "Always run ruff, then pytest.",
        db_path = db_path,
    )
    _world_pass_use(rec["id"], "ep-1", db_path)
    report = pack_from_admitted_b(db_path = db_path)
    assert report.n_train == 1
    assert report.n_holdout == 0
    assert report.pack_id
    items = list_pack_items(report.pack_id, db_path = db_path)
    assert len(items) == 1
    assert items[0]["source"] == "record"
    assert items[0]["source_id"] == rec["id"]
    assert items[0]["role"] == "train"
    assert items[0]["messages"] == [
        {"role": "user", "content": rec["title"]},
        {"role": "assistant", "content": rec["body"]},
    ]


def test_dropped_reasons_are_locked_strings(db_path):
    proposed = _procedure(
        title = "Proposed playbook",
        body = "not admitted yet",
        db_path = db_path,
        status = "proposed",
    )
    infer = insert_record(
        kind = "procedure",
        title = "Inferred playbook",
        body = "guessed steps",
        provenance = "infer",
        db_path = db_path,
    )
    sim_prov = insert_record(
        kind = "procedure",
        title = "Sim only playbook",
        body = "rehearse in the clone",
        provenance = "sim",
        db_path = db_path,
    )
    claim = insert_record(
        kind = "claim",
        title = "Pump max rate",
        body = "Pump X max rate is 12 L/min",
        provenance = "world",
        db_path = db_path,
    )
    episode = insert_record(
        kind = "episode",
        title = "Episode leaked",
        body = "EPISODE_SECRET_TEXT_SHOULD_NOT_LEAK last user said ship it",
        provenance = "mixed",
        source_episode_id = "ep-secret",
        db_path = db_path,
    )
    probe = _procedure(
        title = "Probe: old login",
        body = "echo login\n",
        db_path = db_path,
    )
    command = _procedure(
        title = "Test Command",
        body = "pytest\n",
        db_path = db_path,
    )
    empty = _procedure(title = "Empty body playbook", body = "   ", db_path = db_path)
    untitled = _procedure(title = "   ", body = "has a body", db_path = db_path)
    report = pack_from_admitted_b(db_path = db_path)
    dropped = dict(report.dropped)
    assert dropped[proposed["id"]] == "not active"
    assert dropped[infer["id"]] == "untrusted provenance"
    assert dropped[sim_prov["id"]] == "untrusted provenance"
    assert dropped[claim["id"]] == "not a pack kind"
    assert dropped[episode["id"]] == "not a pack kind"
    assert dropped[probe["id"]] == "probe"
    assert dropped[command["id"]] == "test command"
    assert dropped[empty["id"]] == "empty body"
    assert dropped[untitled["id"]] == "empty title"
    assert report.n_train == 0


def test_episode_body_does_not_appear_in_messages(db_path):
    secret = "EPISODE_SECRET_TEXT_SHOULD_NOT_LEAK"
    insert_record(
        kind = "episode",
        title = "Episode leaked",
        body = f"{secret} last user said ship it",
        provenance = "mixed",
        source_episode_id = "ep-1",
        db_path = db_path,
    )
    rec = _procedure(
        title = "How we ship",
        body = "tag then push",
        db_path = db_path,
    )
    _world_pass_use(rec["id"], "ep-1", db_path)
    report = pack_from_admitted_b(db_path = db_path)
    assert report.n_train == 1
    blob = json.dumps(list_pack_items(report.pack_id, db_path = db_path))
    assert secret not in blob
    assert "last user said" not in blob
    assert rec["body"] in blob


def test_include_sim_false_ignores_sim_pass_only(db_path):
    rec = _procedure(
        title = "How we rehearse",
        body = "run the clone tests",
        db_path = db_path,
    )
    _sim_pass_use(rec["id"], "ep-sim-only", db_path)
    report = pack_from_admitted_b(include_sim = False, db_path = db_path)
    assert report.n_train == 0
    assert (rec["id"], "no world-pass vote") in report.dropped


def test_include_sim_true_drops_without_world_pass(db_path):
    rec = _procedure(
        title = "How we rehearse",
        body = "run the clone tests",
        db_path = db_path,
    )
    _sim_pass_use(rec["id"], "ep-sim-fail-world", db_path)
    insert_rollout(
        episode_id = "ep-sim-fail-world",
        contact = "world",
        outcome = "fail",
        summary = "still broken in world",
        db_path = db_path,
    )
    report = pack_from_admitted_b(include_sim = True, db_path = db_path)
    assert report.n_train == 0
    assert (rec["id"], "sim vote without world-pass") in report.dropped


def test_rejected_twin_note_does_not_veto_sim_vote(db_path):
    rec = _procedure(
        title = "How we rehearse",
        body = "run the clone tests",
        db_path = db_path,
    )
    _sim_pass_use(rec["id"], "ep-rejected-twin", db_path)
    insert_rollout(
        episode_id = "ep-rejected-twin",
        contact = "world",
        outcome = "pass",
        summary = "world also passed",
        db_path = db_path,
    )
    insert_record(
        kind = "twin_note",
        title = "Twin: world vs sim",
        body = "operator rejected this",
        provenance = "mixed",
        status = "rejected",
        source_episode_id = "ep-rejected-twin",
        db_path = db_path,
    )
    report = pack_from_admitted_b(include_sim = True, db_path = db_path)
    assert report.n_train == 1
    assert rec["id"] not in dict(report.dropped)


def test_include_sim_true_drops_twin_note(db_path):
    rec = _procedure(
        title = "How we rehearse",
        body = "run the clone tests",
        db_path = db_path,
    )
    _sim_pass_use(rec["id"], "ep-twin", db_path)
    insert_rollout(
        episode_id = "ep-twin",
        contact = "world",
        outcome = "pass",
        summary = "world also passed",
        db_path = db_path,
    )
    insert_record(
        kind = "twin_note",
        title = "Twin: world vs sim",
        body = "disagreement on ep-twin",
        provenance = "mixed",
        source_episode_id = "ep-twin",
        db_path = db_path,
    )
    report = pack_from_admitted_b(include_sim = True, db_path = db_path)
    assert report.n_train == 0
    assert (rec["id"], "sim vote has twin_note") in report.dropped


def test_include_sim_true_accepts_world_pass_without_twin(db_path):
    rec = _procedure(
        title = "How we rehearse",
        body = "run the clone tests",
        db_path = db_path,
    )
    _sim_pass_use(rec["id"], "ep-ok", db_path)
    insert_rollout(
        episode_id = "ep-ok",
        contact = "world",
        outcome = "pass",
        summary = "world also passed",
        db_path = db_path,
    )
    report = pack_from_admitted_b(include_sim = True, db_path = db_path)
    assert report.n_train == 1
    assert rec["id"] not in dict(report.dropped)
    items = list_pack_items(report.pack_id, db_path = db_path)
    assert items[0]["messages"][1]["content"] == rec["body"]


def test_holdout_is_by_episode(db_path, monkeypatch):
    monkeypatch.setattr("unforgettable.sidecar.pack.HOLDOUT_MIN_EPISODES", 2)
    monkeypatch.setattr("unforgettable.sidecar.pack.PACK_MIN_TRAIN", 1)
    first = _procedure(title = "Playbook A", body = "steps a", db_path = db_path)
    second = _procedure(title = "Playbook B", body = "steps b", db_path = db_path)
    _world_pass_use(first["id"], "ep-aaa", db_path)
    _world_pass_use(second["id"], "ep-zzz", db_path)
    report = pack_from_admitted_b(db_path = db_path)
    assert report.n_train == 1
    assert report.n_holdout == 1
    roles = {
        item["source_id"]: item["role"] for item in list_pack_items(report.pack_id, db_path = db_path)
    }
    assert roles[first["id"]] == "train"
    assert roles[second["id"]] == "holdout"


def test_dry_run_inserts_nothing(db_path):
    rec = _procedure(
        title = "How we ship",
        body = "tag then push",
        db_path = db_path,
    )
    _world_pass_use(rec["id"], "ep-1", db_path)
    report = pack_from_admitted_b(dry_run = True, db_path = db_path)
    assert report.dry_run is True
    assert report.pack_id is None
    assert report.n_train == 1
    assert list_packs(db_path = db_path) == []


def test_pack_from_admitted_b_name_still_exists():
    from unforgettable.sidecar import pack_from_admitted_b as imported
    assert imported is pack_from_admitted_b
    assert callable(imported)


def test_compiled_membership_is_a_vote(db_path):
    rec = _procedure(
        title = "How we run the formatter",
        body = "Always run ruff, then pytest.",
        db_path = db_path,
    )
    pin_compiled(rec["id"], explicit = True, db_path = db_path)
    report = pack_from_admitted_b(db_path = db_path)
    assert report.n_train == 1
    items = list_pack_items(report.pack_id, db_path = db_path)
    assert items[0]["source_id"] == rec["id"]
    assert items[0]["messages"][0]["content"] == rec["title"]


def test_pack_is_retrieval_heavy_when_compiled_count_meets_min(db_path):
    assert pack_is_retrieval_heavy(db_path) is False
    for i in range(3):
        rec = _procedure(
            title = f"Standing playbook {i}",
            body = f"compiled body {i}",
            db_path = db_path,
        )
        pin_compiled(rec["id"], explicit = True, db_path = db_path)
    assert pack_is_retrieval_heavy(db_path) is True


def test_pack_is_retrieval_heavy_when_inject_stats_mean_high(db_path):
    rec = _procedure(
        title = "Standing playbook 0",
        body = "compiled body 0",
        db_path = db_path,
    )
    pin_compiled(rec["id"], explicit = True, db_path = db_path)
    assert pack_is_retrieval_heavy(db_path) is False
    insert_inject_stats(
        episode_id = "ep-sim-heavy",
        contact = "sim",
        standing_chars = 9000,
        retrieve_chars = 9000,
        trajectory_chars = 0,
        total_chars = 18000,
        compiled_ids = "",
        retrieved_ids = "",
        db_path = db_path,
    )
    assert pack_is_retrieval_heavy(db_path) is False
    insert_inject_stats(
        episode_id = "ep-world-heavy",
        contact = "world",
        standing_chars = 1800,
        retrieve_chars = 300,
        trajectory_chars = 0,
        total_chars = 2100,
        compiled_ids = "",
        retrieved_ids = "",
        db_path = db_path,
    )
    assert pack_is_retrieval_heavy(db_path) is True


def _admitted_error_fix(
    db_path,
    *,
    episode_id: str,
    body: str,
    provenance: str = "mixed",
    title: str = "Error then fix",
) -> dict:
    return insert_record(
        kind = "error_fix",
        title = title,
        body = body,
        provenance = provenance,
        source_episode_id = episode_id,
        db_path = db_path,
    )


class _NoCompleteHost:
    def __init__(self, db):
        self.db = db

    def memory_db_path(self):
        return self.db


def test_preference_pairs_world_pass_and_admitted_error_fix(db_path):
    insert_rollout(
        episode_id = "ep-pref",
        contact = "world",
        outcome = "pass",
        summary = "works in world",
        db_path = db_path,
    )
    _admitted_error_fix(
        db_path,
        episode_id = "ep-pref",
        body = "Tried: broke in world\nThen: fixed in world",
    )
    pairs = preference_pairs(db_path = db_path)
    assert len(pairs) == 1
    assert pairs[0]["prompt"] == [{"role": "user", "content": "broke in world"}]
    assert pairs[0]["chosen"] == "Tried: broke in world\nThen: fixed in world"
    assert pairs[0]["rejected"] == "broke in world"
    assert pairs[0]["episode_id"] == "ep-pref"


def test_preference_pairs_skips_twin_note_episode(db_path):
    insert_rollout(
        episode_id = "ep-twin",
        contact = "world",
        outcome = "pass",
        summary = "works in world",
        db_path = db_path,
    )
    _admitted_error_fix(
        db_path,
        episode_id = "ep-twin",
        body = "Tried: broke in world\nThen: later passed",
    )
    insert_record(
        kind = "twin_note",
        title = "Twin: world vs sim",
        body = "disagreement on ep-twin",
        provenance = "mixed",
        source_episode_id = "ep-twin",
        db_path = db_path,
    )
    assert preference_pairs(db_path = db_path) == []


def test_preference_pairs_sim_only_pass_not_chosen(db_path):
    insert_rollout(
        episode_id = "ep-sim",
        contact = "world",
        outcome = "fail",
        summary = "broke in world",
        db_path = db_path,
    )
    insert_rollout(
        episode_id = "ep-sim",
        contact = "sim",
        outcome = "pass",
        summary = "sim only glory",
        db_path = db_path,
    )
    _admitted_error_fix(
        db_path,
        episode_id = "ep-sim",
        body = "Tried: sim fail\nThen: sim only glory",
        provenance = "sim",
    )
    assert preference_pairs(db_path = db_path) == []
    insert_rollout(
        episode_id = "ep-sim",
        contact = "world",
        outcome = "pass",
        summary = "world fixed",
        db_path = db_path,
    )
    assert preference_pairs(db_path = db_path) == []
    _admitted_error_fix(
        db_path,
        episode_id = "ep-sim",
        body = "Tried: broke in world\nThen: world fixed",
        provenance = "mixed",
    )
    pairs = preference_pairs(db_path = db_path)
    assert len(pairs) == 1
    assert pairs[0]["chosen"] == "Tried: broke in world\nThen: world fixed"
    assert pairs[0]["rejected"] == "broke in world"
    assert "sim only glory" not in json.dumps(pairs)


def test_preference_pairs_prefers_admitted_error_fix(db_path):
    insert_rollout(
        episode_id = "ep-fix",
        contact = "world",
        outcome = "pass",
        summary = "fixed in world",
        db_path = db_path,
    )
    _admitted_error_fix(
        db_path,
        episode_id = "ep-fix",
        body = "the admitted fix body",
    )
    pairs = preference_pairs(db_path = db_path)
    assert len(pairs) == 1
    assert pairs[0]["chosen"] == "the admitted fix body"
    assert pairs[0]["rejected"] == "Error then fix"


def test_preference_pairs_from_episode_writer_world_pass(db_path):
    state = EpisodeState(episode_id = "ep-live-01", world_session = "world")
    state.note_failure("exit code 1", "world")
    state.note_success("ok in sim", "sim")
    state.note_success("works in world", "world")
    asyncio.run(
        _extract(
            state,
            str(db_path),
            last_user = "run the tests",
            actions = ["act"],
            host = _NoCompleteHost(db_path),
        )
    )
    grades = {
        (row["contact"], row["outcome"])
        for row in list_rollouts(episode_id = state.episode_id, db_path = db_path)
    }
    assert grades == {("world", "fail"), ("world", "pass"), ("sim", "pass")}
    assert preference_pairs(db_path = db_path) == []
    fixes = list_records(kinds = ["error_fix"], db_path = db_path)
    assert len(fixes) == 1
    assert fixes[0]["status"] == "proposed"
    set_record_status(fixes[0]["id"], "active", reason = "cli admit", db_path = db_path)
    pairs = preference_pairs(db_path = db_path)
    assert len(pairs) == 1
    assert pairs[0]["episode_id"] == state.episode_id
    assert pairs[0]["chosen"] == fixes[0]["body"]
    assert "works in world" in pairs[0]["chosen"]
    assert "ok in sim" not in pairs[0]["chosen"]
    assert pairs[0]["rejected"] == "exit code 1"
    assert pairs[0]["prompt"] == [{"role": "user", "content": "exit code 1"}]
