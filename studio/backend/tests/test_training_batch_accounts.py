# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A multi-LoRA batch's stats land only in the account that started it; another account can neither see its
live status nor stop it."""

import asyncio
import sys
from pathlib import Path

import pytest
from fastapi import HTTPException

import core.training.account_jobs as account_jobs
import routes.training as training_routes
from auth import policy
from utils.account_context import OWNER, AccountContext, arun_as, run_as

_TESTS_DIR = str(Path(__file__).resolve().parent)
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

from test_training_batch_backend import BATCH, MEMBERS, _batch_backend, _exported, _progress  # noqa: E402

ALICE = AccountContext("a" * 32, "alice")
BOB = AccountContext("b" * 32, "bob")


@pytest.fixture
def db(monkeypatch, tmp_path):
    from storage import studio_db

    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    monkeypatch.setattr(policy, "installation_is_multi_user", lambda: True)
    monkeypatch.setattr(account_jobs, "_has_managed_accounts", lambda: True)
    monkeypatch.setattr(studio_db, "_schema_ready", set())
    return studio_db


def test_batch_stats_and_controls_stay_with_the_owning_account(db, tmp_path, monkeypatch):
    backend = _batch_backend(tmp_path)
    backend._result_account = ALICE
    outs = {run_id: _exported(tmp_path, run_id) for run_id in MEMBERS}

    def pump():
        # The pump thread runs bound to the account that started the batch.
        for run_id in MEMBERS:
            backend._handle_event(
                {"type": "output_dir", "member_run_id": run_id, "output_dir": outs[run_id]}
            )
        for step in range(1, 13):
            for i, run_id in enumerate(MEMBERS):
                backend._handle_event(_progress(run_id, step, 1.0 + i + step / 10))
        backend._handle_event(
            {"type": "complete", "member_run_id": "job_a", "output_dir": outs["job_a"]}
        )
        backend._handle_event(
            {
                "type": "stopped",
                "member_run_id": "job_b",
                "output_dir": outs["job_b"],
                "saved": True,
            }
        )
        backend._handle_event(
            {"type": "complete", "member_run_id": "job_c", "output_dir": outs["job_c"]}
        )

    run_as(ALICE, pump)

    for run_id in MEMBERS:
        assert run_as(ALICE, db.get_run, run_id) is not None
        assert run_as(ALICE, db.get_run_metrics, run_id)["step_history"] == list(range(1, 13))
        assert run_as(BOB, db.get_run, run_id) is None
        assert run_as(OWNER, db.get_run, run_id) is None
    assert run_as(ALICE, db.get_run, BATCH) is None

    monkeypatch.setattr(training_routes, "get_training_backend", lambda: backend)
    monkeypatch.setattr(training_routes, "_run_active", lambda _backend: True)

    def status(account):
        return asyncio.run(
            arun_as(
                account, training_routes.get_training_status(current_subject = "x", via_api_key = False)
            )
        )

    assert status(ALICE).job_id == BATCH
    bob = status(BOB)
    assert bob.job_id == "" and bob.phase == "idle"

    with pytest.raises(HTTPException) as refused:
        run_as(BOB, account_jobs.require_job_owner, backend)
    assert refused.value.status_code == 404
    run_as(ALICE, account_jobs.require_job_owner, backend)
