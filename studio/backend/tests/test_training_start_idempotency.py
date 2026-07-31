# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from core.training.training import TrainingBackend


def test_start_request_reservation_is_idempotent_and_serialized():
    backend = TrainingBackend()

    outcome, pending = backend.reserve_start_request("request-1", "job-1")
    assert outcome == "reserved"
    assert pending.state == "pending"

    outcome, duplicate = backend.reserve_start_request("request-1", "job-other")
    assert outcome == "existing"
    assert duplicate.job_id == "job-1"

    outcome, conflict = backend.reserve_start_request("request-2", "job-2")
    assert outcome == "conflict"
    assert conflict.start_request_id == "request-2"
    assert conflict.state == "rejected"

    rejected = backend.resolve_start_request(
        "request-1",
        state = "rejected",
        message = "Model unavailable",
        error = "Model unavailable",
    )
    assert rejected is not None
    assert rejected.state == "rejected"
    assert backend.status_start_request() == rejected
    assert backend.acknowledge_start_request("request-1") is True
    assert backend.status_start_request() is None
    assert backend.get_start_request("request-1") == rejected

    outcome, next_request = backend.reserve_start_request("request-3", "job-3")
    assert outcome == "reserved"
    assert next_request.job_id == "job-3"


def test_accepted_start_request_remains_queryable():
    backend = TrainingBackend()
    backend.reserve_start_request("request-1", "job-1")
    backend.resolve_start_request(
        "request-1",
        state = "accepted",
        message = "Training queued",
    )

    record = backend.get_start_request("request-1")
    assert record is not None
    assert record.state == "accepted"
    assert record.job_id == "job-1"
