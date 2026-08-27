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
from pathlib import Path

from unforgettable.sidecar.adapters import get_adapter
from unforgettable.sidecar.eval import completion_score, eval_adapter, scored_completion
from unforgettable.supervisor import SupervisorConfig
from unforgettable.sidecar.pack import ROLE_HOLDOUT, list_pack_items, pack_from_admitted_b
from unforgettable.sidecar.train import FakeTrainBackend, train_pack
from unforgettable.store.records import insert_record, insert_retrieve_use, insert_rollout


def _voted_procedure(db_path, *, title: str, body: str, episode_id: str) -> dict:
    rec = insert_record(
        kind = "procedure",
        title = title,
        body = body,
        provenance = "world",
        db_path = db_path,
    )
    insert_retrieve_use(
        episode_id = episode_id,
        record_id = rec["id"],
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
    return rec


def _seed_holdout_gold(adapter_path: str, pack_id: str, db_path) -> None:
    dest = Path(adapter_path) / "fake_gold.json"
    gold = json.loads(dest.read_text(encoding = "utf-8")) if dest.is_file() else {}
    for item in list_pack_items(pack_id, db_path = db_path):
        if item.get("role") != ROLE_HOLDOUT:
            continue
        user = ""
        assistant = ""
        for msg in item.get("messages") or []:
            if msg.get("role") == "user":
                user = msg.get("content") or ""
            elif msg.get("role") == "assistant":
                assistant = msg.get("content") or ""
        if user:
            gold[user] = assistant
    dest.write_text(json.dumps(gold), encoding = "utf-8")


def test_eval_holdout_gold_beats_base(db_path, monkeypatch):
    monkeypatch.setattr("unforgettable.sidecar.pack.HOLDOUT_MIN_EPISODES", 1)
    for i in range(5):
        _voted_procedure(
            db_path,
            title = f"Playbook {i}",
            body = f"steps {i}",
            episode_id = f"ep-{i}",
        )
    packed = pack_from_admitted_b(db_path = db_path)
    assert packed.n_holdout >= 1
    result = train_pack(
        packed.pack_id,
        backend = FakeTrainBackend(),
        base_model = "fake",
        db_path = db_path,
    )
    _seed_holdout_gold(result.path, packed.pack_id, db_path)
    report = eval_adapter(result.adapter_id, backend = FakeTrainBackend(), db_path = db_path)
    assert report.n_holdout >= 1
    assert report.adapter_lean == 1.0
    assert report.base_lean == 0.0
    assert report.passed is True
    row = get_adapter(result.adapter_id, db_path = db_path)
    metrics = json.loads(row["metrics"])
    assert metrics["adapter_lean"] == 1.0
    assert metrics["passed"] is True


def test_eval_without_holdout_gold_fails(db_path, monkeypatch):
    monkeypatch.setattr("unforgettable.sidecar.pack.HOLDOUT_MIN_EPISODES", 1)
    for i in range(5):
        _voted_procedure(
            db_path,
            title = f"Playbook {i}",
            body = f"steps {i}",
            episode_id = f"ep-{i}",
        )
    packed = pack_from_admitted_b(db_path = db_path)
    assert packed.n_holdout >= 1
    result = train_pack(
        packed.pack_id,
        backend = FakeTrainBackend(),
        base_model = "fake",
        db_path = db_path,
    )
    report = eval_adapter(result.adapter_id, backend = FakeTrainBackend(), db_path = db_path)
    assert report.n_holdout >= 1
    assert report.adapter_lean == 0.0
    assert report.base_lean == 0.0
    assert report.passed is False


def test_eval_empty_holdout_no_world_fails(db_path):
    for i in range(4):
        _voted_procedure(
            db_path,
            title = f"Playbook {i}",
            body = f"steps {i}",
            episode_id = f"ep-{i}",
        )
    packed = pack_from_admitted_b(db_path = db_path)
    assert packed.n_holdout == 0
    result = train_pack(
        packed.pack_id,
        backend = FakeTrainBackend(),
        base_model = "fake",
        db_path = db_path,
    )
    report = eval_adapter(result.adapter_id, backend = FakeTrainBackend(), db_path = db_path)
    assert report.n_holdout == 0
    assert report.adapter_lean == 0.0
    assert report.base_lean == 0.0
    assert report.passed is False


def test_completion_score_gold_prefix_in_output():
    assert completion_score("hello world extra", "hello world") == 1.0


def test_completion_score_empty_output():
    assert completion_score("", "hello world") == 0.0


class _JudgeHost:
    def __init__(self, text: str):
        self.text = text
        self.calls = []

    async def supervise(
        self,
        purpose,
        messages,
        *,
        model = None,
        max_tokens = 400,
    ):
        self.calls.append({"purpose": purpose, "model": model})
        return self.text


def test_scored_completion_uses_judge_then_falls_back():
    host = _JudgeHost('{"score": 0.8}')
    assert scored_completion("nope", "hello world", host = host, model = "judge") == 0.8
    assert host.calls[0]["purpose"] == "judge"
    assert host.calls[0]["model"] == "judge"
    bad = _JudgeHost("not json")
    assert scored_completion("hello world extra", "hello world", host = bad, model = "j") == 1.0
    assert scored_completion("hello world extra", "hello world") == 1.0


def test_eval_adapter_uses_judge_scores(db_path, monkeypatch):
    monkeypatch.setattr("unforgettable.sidecar.pack.HOLDOUT_MIN_EPISODES", 1)
    for i in range(5):
        _voted_procedure(
            db_path,
            title = f"Playbook {i}",
            body = f"steps {i}",
            episode_id = f"ep-{i}",
        )
    packed = pack_from_admitted_b(db_path = db_path)
    result = train_pack(
        packed.pack_id,
        backend = FakeTrainBackend(),
        base_model = "fake",
        db_path = db_path,
    )
    report = eval_adapter(
        result.adapter_id,
        backend = FakeTrainBackend(),
        db_path = db_path,
        host = _JudgeHost('{"score": 0.8}'),
        config = SupervisorConfig(judge_model = "judge"),
    )
    assert report.n_holdout >= 1
    assert report.adapter_lean == 0.8
    assert report.base_lean == 0.8
    assert report.passed is True
