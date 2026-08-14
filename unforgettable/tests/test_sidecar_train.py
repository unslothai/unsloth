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

import pytest

from unforgettable.sidecar.adapters import get_adapter
from unforgettable.sidecar.pack import pack_from_admitted_b
from unforgettable.sidecar.train import (
    FakeTrainBackend,
    UnslothTrainBackend,
    train_pack,
)
from unforgettable.store.records import insert_record, insert_retrieve_use, insert_rollout


def _voted_procedure(db_path, *, title: str, body: str, episode_id: str) -> dict:
    rec = insert_record(
        kind="procedure",
        title=title,
        body=body,
        provenance="world",
        db_path=db_path,
    )
    insert_retrieve_use(
        episode_id=episode_id,
        record_id=rec["id"],
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
    return rec


def _voted_pack(db_path, n: int):
    for i in range(n):
        _voted_procedure(
            db_path,
            title=f"Playbook {i}",
            body=f"steps {i}",
            episode_id=f"ep-{i}",
        )
    return pack_from_admitted_b(db_path=db_path)


def test_train_pack_unknown_pack_raises(db_path):
    with pytest.raises(KeyError):
        train_pack(
            "missing-pack",
            backend=FakeTrainBackend(),
            base_model="fake",
            db_path=db_path,
        )


def test_train_pack_refuses_below_min(db_path):
    report = _voted_pack(db_path, 3)
    assert report.n_train == 3
    with pytest.raises(ValueError, match="train items"):
        train_pack(
            report.pack_id,
            backend=FakeTrainBackend(),
            base_model="fake",
            db_path=db_path,
        )


def test_train_pack_fake_writes_shadow_adapter(db_path):
    report = _voted_pack(db_path, 4)
    result = train_pack(
        report.pack_id,
        backend=FakeTrainBackend(),
        base_model="fake",
        db_path=db_path,
    )
    assert result.backend == "fake"
    assert result.recipe == "sft"
    assert result.n_examples == 4
    dest = Path(result.path)
    config = json.loads((dest / "adapter_config.json").read_text(encoding="utf-8"))
    assert config == {"fake": True, "recipe": "sft", "n": 4}
    gold = json.loads((dest / "fake_gold.json").read_text(encoding="utf-8"))
    assert gold["Playbook 0"] == "steps 0"
    row = get_adapter(result.adapter_id, db_path=db_path)
    assert row is not None
    assert row["status"] == "shadow"
    assert row["base_model"] == "fake"
    assert row["pack_id"] == report.pack_id
    assert row["path"] == result.path
    backend = FakeTrainBackend()
    assert (
        backend.complete(
            [{"role": "user", "content": "Playbook 1"}],
            adapter_path=result.path,
        )
        == "steps 1"
    )
    assert backend.complete(
        [{"role": "user", "content": "Playbook 1"}],
        adapter_path=None,
    ) == ""


def test_unsloth_backend_refuses_full_finetune(monkeypatch, tmp_path):
    import sys

    monkeypatch.setenv("UNSLOTH_ENABLE_FULL_FINETUNING", "1")
    backend = UnslothTrainBackend()
    with pytest.raises(RuntimeError, match="full fine-tune"):
        backend.train(
            [{"messages": [{"role": "user", "content": "u"}]}],
            output_dir=tmp_path / "out",
            base_model="dummy",
        )
    assert "unsloth" not in sys.modules
    assert "torch" not in sys.modules


def test_unsloth_backend_preference_raises_before_import(tmp_path):
    import sys

    backend = UnslothTrainBackend()
    with pytest.raises((RuntimeError, NotImplementedError), match="preference|DPO|wired"):
        backend.train(
            [{"messages": [{"role": "user", "content": "u"}]}],
            output_dir=tmp_path / "out",
            base_model="dummy",
            recipe="preference",
        )
    assert "unsloth" not in sys.modules
    assert "torch" not in sys.modules


def test_train_pack_preference_writes_pairs_jsonl(db_path):
    report = _voted_pack(db_path, 4)
    insert_rollout(
        episode_id="ep-pref",
        contact="world",
        outcome="pass",
        summary="works in world",
        db_path=db_path,
    )
    insert_record(
        kind="error_fix",
        title="Error then fix",
        body="Tried: broke in world\nThen: fixed in world",
        provenance="mixed",
        source_episode_id="ep-pref",
        db_path=db_path,
    )
    result = train_pack(
        report.pack_id,
        backend=FakeTrainBackend(),
        base_model="fake",
        recipe="preference",
        db_path=db_path,
    )
    assert result.recipe == "preference"
    assert result.n_examples == 1
    dest = Path(result.path)
    config = json.loads((dest / "adapter_config.json").read_text(encoding="utf-8"))
    assert config["recipe"] == "preference"
    assert config["n"] == 1
    lines = (dest / "pairs.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    pair = json.loads(lines[0])
    assert pair["chosen"] == "Tried: broke in world\nThen: fixed in world"
    assert pair["rejected"] == "broke in world"
    assert pair["episode_id"] == "ep-pref"
    row = get_adapter(result.adapter_id, db_path=db_path)
    assert row is not None
    assert row["status"] == "shadow"
    assert row["recipe"] == "preference"


def test_train_pack_preference_without_pairs_raises(db_path):
    report = _voted_pack(db_path, 4)
    with pytest.raises(ValueError, match="preference pairs"):
        train_pack(
            report.pack_id,
            backend=FakeTrainBackend(),
            base_model="fake",
            recipe="preference",
            db_path=db_path,
        )
