# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The media resolver index is one account's answer, not whoever warmed it first: a cache keyed only by task hands the next account the previous one's picks for the whole TTL."""

from __future__ import annotations

import pytest

import core.inference.media_auto_switch as mas
import core.inference.media_model_index as index
import routes.models as models_route
from auth import policy
from storage import studio_db
from utils.account_context import OWNER, AccountContext, run_as

ALICE = AccountContext("a" * 32, "alice")
BOB = AccountContext("b" * 32, "bob")


@pytest.fixture
def home(monkeypatch, tmp_path):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    monkeypatch.setattr(policy, "installation_is_multi_user", lambda: True)
    monkeypatch.setattr(models_route, "_local_model_task", lambda info: mas.IMAGE_TASK)
    mas.invalidate_index()
    yield tmp_path
    mas.invalidate_index()


def _private_pipeline(account, home, name):
    root = home / "accounts" / account.account_id / "private_models"
    pipeline = root / name
    pipeline.mkdir(parents = True)
    (pipeline / "model_index.json").write_text("{}")

    def _register():
        connection = studio_db.get_connection()
        with connection:
            connection.execute(
                "INSERT INTO scan_folders (path, created_at) VALUES (?, datetime('now'))",
                (str(root),),
            )
        connection.close()

    run_as(account, _register)
    return pipeline


def test_a_warm_index_does_not_answer_for_the_next_account(home):
    alice_model = _private_pipeline(ALICE, home, "alice-secret-flux")
    bob_model = _private_pipeline(BOB, home, "bob-own-flux")

    assert run_as(ALICE, mas.available_media_model_ids, mas.IMAGE_TASK) == ["alice-secret-flux"]
    assert run_as(BOB, mas.available_media_model_ids, mas.IMAGE_TASK) == ["bob-own-flux"]

    bob_pick = run_as(
        BOB, lambda: mas.resolve_local_media_model("bob-own-flux", task = mas.IMAGE_TASK)
    )
    assert bob_pick is not None and bob_pick.model_path == str(bob_model)
    assert (
        run_as(BOB, lambda: mas.resolve_local_media_model("alice-secret-flux", task = mas.IMAGE_TASK))
        is None
    )
    alice_pick = run_as(
        ALICE, lambda: mas.resolve_local_media_model("alice-secret-flux", task = mas.IMAGE_TASK)
    )
    assert alice_pick is not None and alice_pick.model_path == str(alice_model)


def test_the_owner_still_gets_one_cached_scan(home, monkeypatch):
    _private_pipeline(OWNER, home, "owner-flux")
    scans = []
    real = models_route.collect_local_models
    monkeypatch.setattr(
        models_route,
        "collect_local_models",
        lambda root: (scans.append(root), real(root))[1],
    )

    assert mas.available_media_model_ids(mas.IMAGE_TASK) == ["owner-flux"]
    assert mas.available_media_model_ids(mas.IMAGE_TASK) == ["owner-flux"]
    assert len(scans) == 1
    assert list(index._index) == [(OWNER.account_id, mas.IMAGE_TASK)]
