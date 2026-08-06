# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The provenance refusal has to reach the user, not just the start route.

300fe6321 made ``POST /api/train/start`` report the real reason instead of blaming the
checkpoint, but the History UI never gets that far: ``can_resume: false`` hides the
Resume button outright, and ``resume-training-run.ts`` throws its own
"Only stopped or errored runs with a saved checkpoint can be resumed" *before* issuing
any request. For a run whose checkpoint is intact and whose pinned snapshot was evicted,
that sentence is exactly the wrong diagnosis.

So the summary carries the reason, and the client prefers it over its generic string.
"""

from models.training import TrainingRunSummary
from routes import training_history


_REASON = "The exact model snapshot for this run is no longer available."


def _row(**overrides):
    row = {
        "id": "run-1",
        "status": "stopped",
        "model_name": "unsloth/Llama-3.2-1B-Instruct",
        "project_name": None,
        "dataset_name": "yahma/alpaca-cleaned",
        "started_at": "2026-08-05T00:00:00Z",
        "output_dir": "/runs/run-1",
        "config_json": "{}",
    }
    row.update(overrides)
    return row


def test_the_field_is_optional_so_old_clients_are_unaffected():
    field = TrainingRunSummary.model_fields["resume_blocked_reason"]
    assert field.default is None
    assert (
        TrainingRunSummary(
            id = "r",
            status = "stopped",
            model_name = "m",
            dataset_name = "d",
            started_at = "t",
        ).resume_blocked_reason
        is None
    )


def test_a_provenance_refusal_is_reported_on_the_summary(monkeypatch):
    # An intact checkpoint is the precondition: with the trainer state missing the
    # checkpoint is the real cause and the reason is deliberately None. See
    # test_resume_reason_matches_cause.py.
    from core.training import resume as resume_mod

    monkeypatch.setattr(resume_mod, "has_resume_state", lambda output_dir: True)
    monkeypatch.setattr(training_history, "can_resume_run", lambda *a, **k: False)
    monkeypatch.setattr(training_history, "artifacts_present", lambda *a, **k: True)
    monkeypatch.setattr(
        training_history, "_preview_fields", lambda *a, **k: {"has_preview_model": False}
    )
    from core.training import provenance as provenance_mod

    monkeypatch.setattr(
        provenance_mod, "resource_provenance_resume_blocker", lambda config: _REASON
    )

    summary = training_history._summary_from_row(_row(), False)

    assert summary.can_resume is False
    assert summary.resume_blocked_reason == _REASON
    assert "checkpoint" not in (summary.resume_blocked_reason or "").lower()


def test_a_resumable_run_carries_no_reason(monkeypatch):
    monkeypatch.setattr(training_history, "can_resume_run", lambda *a, **k: True)
    monkeypatch.setattr(training_history, "artifacts_present", lambda *a, **k: True)
    monkeypatch.setattr(
        training_history, "_preview_fields", lambda *a, **k: {"has_preview_model": False}
    )

    summary = training_history._summary_from_row(_row(), False)

    assert summary.can_resume is True
    assert summary.resume_blocked_reason is None


def test_a_missing_checkpoint_keeps_the_clients_own_wording(monkeypatch):
    """No provenance cause means None, so the client falls back to its message."""
    monkeypatch.setattr(training_history, "can_resume_run", lambda *a, **k: False)
    monkeypatch.setattr(training_history, "artifacts_present", lambda *a, **k: False)
    monkeypatch.setattr(
        training_history, "_preview_fields", lambda *a, **k: {"has_preview_model": False}
    )
    from core.training import provenance as provenance_mod

    monkeypatch.setattr(provenance_mod, "resource_provenance_resume_blocker", lambda config: None)

    summary = training_history._summary_from_row(_row(), False)

    assert summary.can_resume is False
    assert summary.resume_blocked_reason is None


def test_a_failure_computing_the_reason_is_not_fatal(monkeypatch):
    """History must render even if the gate raises; the row simply carries no reason.

    Narrower than it first appears: ``can_resume_run`` calls the same gate without a
    guard one line earlier, so if the gate raises on a row that reaches it, History
    fails there instead. What this ``except`` genuinely protects is the path where
    ``can_resume_run`` short-circuits before touching the gate and
    ``_resume_blocked_reason`` is its first caller.
    """
    from core.training import resume as resume_mod

    monkeypatch.setattr(resume_mod, "has_resume_state", lambda output_dir: True)
    monkeypatch.setattr(training_history, "can_resume_run", lambda *a, **k: False)
    monkeypatch.setattr(training_history, "artifacts_present", lambda *a, **k: True)
    monkeypatch.setattr(
        training_history, "_preview_fields", lambda *a, **k: {"has_preview_model": False}
    )
    from core.training import provenance as provenance_mod

    def boom(config):
        raise RuntimeError("gate exploded")

    monkeypatch.setattr(provenance_mod, "resource_provenance_resume_blocker", boom)

    summary = training_history._summary_from_row(_row(), False)

    assert summary.resume_blocked_reason is None


def test_the_client_prefers_the_server_reason():
    """Wiring contract: the pre-request guard must not hardcode its own diagnosis."""
    from pathlib import Path

    source = (
        Path(__file__).resolve().parent.parent.parent
        / "frontend"
        / "src"
        / "features"
        / "training"
        / "lib"
        / "resume-training-run.ts"
    ).read_text(encoding = "utf-8")

    guard = source.split("if (!(detail.run.can_resume && outputDir))", 1)[1][:400]
    assert "detail.run.resume_blocked_reason" in guard
    assert guard.index("resume_blocked_reason") < guard.index(
        "RESUME_UNAVAILABLE_ERROR"
    ), "the server's reason must take precedence over the generic string"
