# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""/activations without a job_id must not name another account's active run, as /metrics doesn't."""

from types import SimpleNamespace

import routes.training as tr


def _call(monkeypatch, foreign):
    asked = []
    monkeypatch.setattr(
        tr, "get_training_backend", lambda: SimpleNamespace(current_job_id = "run-other")
    )
    monkeypatch.setattr(tr, "job_is_foreign", lambda backend: foreign)
    monkeypatch.setattr(tr, "get_activation_metadata", lambda job_id: asked.append(job_id) or {})
    monkeypatch.setattr(tr, "get_activation_records", lambda job_id, since: [])
    result = tr.get_activations(job_id = None, output_dir = None, since_step = None, current_subject = "u")
    return result, asked


def test_a_foreign_active_run_is_not_resolved(monkeypatch):
    result, asked = _call(monkeypatch, foreign = True)
    assert asked == []
    assert result == {"metadata": None, "records": []}


def test_the_callers_own_active_run_still_resolves(monkeypatch):
    result, asked = _call(monkeypatch, foreign = False)
    assert asked == ["run-other"]
