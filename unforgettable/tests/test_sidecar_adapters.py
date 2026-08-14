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

from pathlib import Path

import pytest

from unforgettable.sidecar.adapters import (
    get_adapter,
    get_promoted_adapter,
    promote_adapter,
    rollback_adapter,
)
from unforgettable.sidecar.pack import pack_from_admitted_b
from unforgettable.sidecar.train import FakeTrainBackend, train_pack
from unforgettable.store.records import insert_record, insert_retrieve_use, insert_rollout


def _voted_pack(db_path, n: int = 4):
    for i in range(n):
        rec = insert_record(
            kind="procedure",
            title=f"Playbook {i}",
            body=f"steps {i}",
            provenance="world",
            db_path=db_path,
        )
        insert_retrieve_use(
            episode_id=f"ep-{i}",
            record_id=rec["id"],
            contact="world",
            db_path=db_path,
        )
        insert_rollout(
            episode_id=f"ep-{i}",
            contact="world",
            outcome="pass",
            summary="ok",
            db_path=db_path,
        )
    return pack_from_admitted_b(db_path=db_path)


def _train_shadow(db_path):
    report = _voted_pack(db_path)
    return train_pack(
        report.pack_id,
        backend=FakeTrainBackend(),
        base_model="fake",
        db_path=db_path,
    )


def test_rollback_without_promote_is_noop(db_path):
    assert rollback_adapter(db_path=db_path) is None
    assert get_promoted_adapter(db_path=db_path) is None


def test_force_promote_then_rollback_clears_promoted_keeps_files(db_path):
    result = _train_shadow(db_path)
    promote_adapter(result.adapter_id, force=True, db_path=db_path)
    discarded = rollback_adapter(db_path=db_path)
    assert discarded is not None
    assert discarded["id"] == result.adapter_id
    assert discarded["status"] == "discarded"
    assert get_promoted_adapter(db_path=db_path) is None
    dest = Path(result.path)
    assert (dest / "adapter_config.json").is_file()
    assert (dest / "fake_gold.json").is_file()


def test_second_force_promote_discards_the_first(db_path):
    first = _train_shadow(db_path)
    pack_id = get_adapter(first.adapter_id, db_path=db_path)["pack_id"]
    second = train_pack(
        pack_id,
        backend=FakeTrainBackend(),
        base_model="fake",
        db_path=db_path,
    )
    promote_adapter(first.adapter_id, force=True, db_path=db_path)
    promote_adapter(second.adapter_id, force=True, db_path=db_path)
    promoted = get_promoted_adapter(db_path=db_path)
    assert promoted is not None
    assert promoted["id"] == second.adapter_id
    assert promoted["status"] == "promoted"
    previous = get_adapter(first.adapter_id, db_path=db_path)
    assert previous is not None
    assert previous["status"] == "discarded"


def test_promote_shadow_without_metrics_refuses(db_path):
    result = _train_shadow(db_path)
    with pytest.raises(ValueError, match="no eval metrics"):
        promote_adapter(result.adapter_id, db_path=db_path)
    assert get_promoted_adapter(db_path=db_path) is None
    assert get_adapter(result.adapter_id, db_path=db_path)["status"] == "shadow"


def test_discarded_promote_without_force_raises(db_path):
    result = _train_shadow(db_path)
    promote_adapter(result.adapter_id, force=True, db_path=db_path)
    rollback_adapter(db_path=db_path)
    with pytest.raises(ValueError):
        promote_adapter(result.adapter_id, db_path=db_path)
    assert get_promoted_adapter(db_path=db_path) is None
