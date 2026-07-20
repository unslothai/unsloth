# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import pytest

from core.training.training import TrainingBackend
from storage import studio_db


@pytest.fixture(autouse = True)
def _isolated_db(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    monkeypatch.setattr(studio_db, "_schema_ready", False)


def _finalizable_backend() -> TrainingBackend:
    backend = TrainingBackend()
    backend.current_job_id = "job_notify_test"
    backend._db_run_created = True
    return backend


def test_notify_fires_exactly_once_per_finalize():
    backend = _finalizable_backend()
    calls = []
    backend.on_job_finished = lambda: calls.append(1)

    backend._finalize_run_in_db("completed")
    assert len(calls) == 1

    # Re-finalizing an already-finalized run early-returns without notifying.
    backend._finalize_run_in_db("completed")
    assert len(calls) == 1


def test_notify_skipped_when_run_never_created():
    backend = TrainingBackend()  # no current_job_id / _db_run_created
    calls = []
    backend.on_job_finished = lambda: calls.append(1)

    backend._finalize_run_in_db("error")
    # Early return: the poll loop is the recovery path, not the callback.
    assert calls == []


def test_raising_callback_is_swallowed_and_run_still_finalized():
    backend = _finalizable_backend()

    def _boom():
        raise RuntimeError("queue runner exploded")

    backend.on_job_finished = _boom
    backend._finalize_run_in_db("completed")
    assert backend._run_finalized is True


def test_no_callback_is_fine():
    backend = _finalizable_backend()
    assert backend.on_job_finished is None
    backend._finalize_run_in_db("stopped")
    assert backend._run_finalized is True
