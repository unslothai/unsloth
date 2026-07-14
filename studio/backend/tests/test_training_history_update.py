# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

import asyncio
import os
import sys

import pytest
from pydantic import ValidationError

_backend = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, _backend)

from models.training import TrainingRunUpdateRequest
from routes import training_history


BASE_RUN = {
    "id": "run-1",
    "status": "stopped",
    "model_name": "unsloth/test-model",
    "dataset_name": "test-dataset",
    "display_name": "Existing name",
    "started_at": "2026-01-01T00:00:00Z",
    "ended_at": "2026-01-01T00:01:00Z",
    "total_steps": 10,
    "final_step": 5,
    "output_dir": "/tmp/run-1",
    "resumed_later": False,
}


def _patch_run(monkeypatch: pytest.MonkeyPatch, payload: TrainingRunUpdateRequest):
    stored = dict(BASE_RUN)
    calls: list[str | None] = []

    def fake_get_run(run_id: str):
        assert run_id == "run-1"
        return dict(stored)

    def fake_update_run_display_name(run_id: str, display_name: str | None):
        assert run_id == "run-1"
        calls.append(display_name)
        stored["display_name"] = display_name

    monkeypatch.setattr(training_history, "get_run", fake_get_run)
    monkeypatch.setattr(
        training_history,
        "update_run_display_name",
        fake_update_run_display_name,
    )
    monkeypatch.setattr(training_history, "can_resume_run", lambda run: True)

    result = asyncio.run(
        training_history.update_training_run(
            "run-1",
            payload,
            current_subject = "test-user",
        )
    )
    return result, calls


def test_update_run_omitted_display_name_is_noop(monkeypatch: pytest.MonkeyPatch):
    result, calls = _patch_run(monkeypatch, TrainingRunUpdateRequest.model_validate({}))

    assert calls == []
    assert result.display_name == "Existing name"
    assert result.can_resume is True


def test_update_run_explicit_null_clears_display_name(monkeypatch: pytest.MonkeyPatch):
    result, calls = _patch_run(
        monkeypatch,
        TrainingRunUpdateRequest.model_validate({"display_name": None}),
    )

    assert calls == [None]
    assert result.display_name is None
    assert result.can_resume is True


def test_update_run_whitespace_clears_display_name(monkeypatch: pytest.MonkeyPatch):
    result, calls = _patch_run(
        monkeypatch,
        TrainingRunUpdateRequest.model_validate({"display_name": "   "}),
    )

    assert calls == [None]
    assert result.display_name is None


def test_get_run_detail_includes_preview_fields(monkeypatch: pytest.MonkeyPatch):
    # Regression: detail/update must pass the sharing flag into _preview_fields;
    # a missing arg used to surface as a 500 TypeError after get_run succeeded.
    monkeypatch.setattr(training_history, "get_run", lambda run_id: dict(BASE_RUN))
    monkeypatch.setattr(training_history, "get_run_metrics", lambda run_id: {})
    monkeypatch.setattr(training_history, "can_resume_run", lambda run: False)
    monkeypatch.setattr(training_history, "get_preview_sharing_enabled", lambda: True)

    detail = asyncio.run(
        training_history.get_training_run_detail("run-1", current_subject = "test-user")
    )

    assert detail.run.id == "run-1"
    # Not a previewable dir, so no signed ref - but the field is built without error.
    assert detail.run.preview_sig is None


def test_update_run_rejects_unknown_fields():
    with pytest.raises(ValidationError):
        TrainingRunUpdateRequest.model_validate({"unknown": "value"})


def test_update_run_rejects_overlong_display_name():
    with pytest.raises(ValidationError):
        TrainingRunUpdateRequest.model_validate({"display_name": "x" * 121})


def test_sanitize_db_config_strips_subject_and_secrets():
    # config_json is returned by run-history GET to any authenticated user, so the run
    # owner's subject (username / API-key id) and secrets must never be persisted.
    from core.training.training import _sanitize_db_config

    db = _sanitize_db_config(
        {
            "model_name": "unsloth/test-model",
            "subject": "alice@example.com",
            "hf_token": "hf_secret",
            "wandb_token": "wb_secret",
            "lora_r": 16,
        }
    )
    assert "subject" not in db
    assert "hf_token" not in db and "wandb_token" not in db
    assert db["model_name"] == "unsloth/test-model" and db["lora_r"] == 16
