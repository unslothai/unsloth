# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The history listing reads a run's checkpoint state only when the answer can depend on it."""

import json
from pathlib import Path

import pytest

from core.training import provenance as provenance_mod
from core.training import resume as resume_mod
from routes import training_history

_REASON = "The exact model snapshot for this run is no longer available."


def _row(output_dir, **overrides):
    row = {
        "id": "run-1",
        "status": "completed",
        "model_name": "m",
        "dataset_name": "d",
        "started_at": "t",
        "output_dir": str(output_dir),
        "config_json": json.dumps({}),
        "final_step": 100,
        "total_steps": 100,
    }
    row.update(overrides)
    return row


@pytest.fixture
def reads(monkeypatch, tmp_path):
    monkeypatch.setattr(resume_mod, "outputs_root", lambda: tmp_path)
    monkeypatch.setattr(resume_mod, "resolve_output_dir", lambda value: Path(value))
    calls: list = []
    real = resume_mod.get_resume_checkpoint_path

    def counted(path_value, expected_step = None):
        calls.append(path_value)
        return real(path_value, expected_step)

    monkeypatch.setattr(resume_mod, "get_resume_checkpoint_path", counted)
    return calls


def test_a_completed_run_the_gate_does_not_refuse_reads_no_checkpoint(reads, tmp_path, monkeypatch):
    monkeypatch.setattr(provenance_mod, "resource_provenance_resume_blocker", lambda config: None)
    (tmp_path / "run-1").mkdir()

    summary = training_history._summaries_from_rows([_row(tmp_path / "run-1")], False)[0]

    assert summary.can_resume is False
    assert summary.resume_blocked_reason is None
    assert reads == []


@pytest.mark.parametrize("has_state", [True, False])
def test_a_refused_run_reads_its_checkpoint_once(reads, tmp_path, monkeypatch, has_state):
    monkeypatch.setattr(
        provenance_mod, "resource_provenance_resume_blocker", lambda config: _REASON
    )
    run = tmp_path / "run-1"
    run.mkdir()
    monkeypatch.setattr(
        resume_mod, "is_resume_checkpoint_valid", lambda path, step = None, backend = None: has_state
    )

    summary = training_history._summaries_from_rows(
        [_row(run, status = "stopped", final_step = 50)], False
    )[0]

    assert summary.can_resume is False
    assert summary.resume_blocked_reason == (_REASON if has_state else None)
    assert reads == [str(run)]


def test_state_files_are_not_parsed_without_a_valid_trainer_step(tmp_path, monkeypatch):
    parsed: list = []
    monkeypatch.setattr(
        resume_mod, "_valid_state_file", lambda path, require_tensor = True: parsed.append(path)
    )
    monkeypatch.setattr(resume_mod, "_has_model_state", lambda path: parsed.append(path) or True)
    checkpoint = tmp_path / "checkpoint-30"
    checkpoint.mkdir()

    assert resume_mod.is_resume_checkpoint_valid(checkpoint) is False
    (checkpoint / "trainer_state.json").write_text(json.dumps({"global_step": 31}))
    assert resume_mod.is_resume_checkpoint_valid(checkpoint) is False
    (checkpoint / "trainer_state.json").write_text(json.dumps({"global_step": 30}))
    assert resume_mod.is_resume_checkpoint_valid(checkpoint, expected_step = 29) is False
    assert parsed == []


def test_no_preview_model_skips_the_preview_ref_lookup(monkeypatch, tmp_path):
    monkeypatch.setattr(training_history, "has_preview_model", lambda output_dir: False)

    def unexpected(output_dir):
        raise AssertionError("preview_ref is None whenever there is no preview model")

    monkeypatch.setattr(training_history, "preview_ref", unexpected)

    assert training_history._preview_fields(str(tmp_path), True) == {
        "has_preview_model": False,
        "preview_ref": None,
        "preview_sig": None,
    }
