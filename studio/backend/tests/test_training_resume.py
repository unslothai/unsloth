# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Regression tests for resumable training run eligibility."""

import importlib.util
import json
from pathlib import Path


_BACKEND = Path(__file__).resolve().parents[1]


def _load_resume_module():
    spec = importlib.util.spec_from_file_location(
        "training_resume_under_test",
        _BACKEND / "core" / "training" / "resume.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


resume = _load_resume_module()


def _stopped_run(**overrides):
    run = {
        "status": "stopped",
        "final_step": 5,
        "total_steps": 10,
        "output_dir": "/tmp/unsloth-output",
        "resumed_later": False,
        "config_json": json.dumps({"hf_dataset": "org/dataset"}),
    }
    run.update(overrides)
    return run


def test_can_resume_run_allows_checkpointed_non_s3_run(monkeypatch):
    monkeypatch.setattr(resume, "has_resume_state", lambda _path: True)

    assert resume.can_resume_run(_stopped_run()) is True


def test_can_resume_run_rejects_s3_dataset_source(monkeypatch):
    monkeypatch.setattr(resume, "has_resume_state", lambda _path: True)

    run = _stopped_run(
        config_json = json.dumps(
            {
                "dataset_source": "s3",
                "s3_dataset": {
                    "bucket": "training-data",
                    "prefix": "datasets/",
                    "region": "us-east-1",
                    "use_iam_role": True,
                },
            }
        )
    )

    assert resume.can_resume_run(run) is False


def test_can_resume_run_rejects_s3_metadata_marker(monkeypatch):
    monkeypatch.setattr(resume, "has_resume_state", lambda _path: True)

    run = _stopped_run(
        config_json = json.dumps({"s3_dataset": {"bucket": "training-data"}})
    )

    assert resume.can_resume_run(run) is False


def test_list_runs_includes_config_json_for_resume_policy(monkeypatch, tmp_path):
    from storage import studio_db

    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    monkeypatch.setattr(studio_db, "_schema_ready", False)
    config_json = json.dumps(
        {"dataset_source": "s3", "s3_dataset": {"bucket": "training-data"}}
    )

    studio_db.create_run(
        id = "run-s3",
        model_name = "unsloth/test-model",
        dataset_name = "s3://training-data",
        config_json = config_json,
        started_at = "2026-01-01T00:00:00Z",
        total_steps = 10,
    )

    result = studio_db.list_runs()

    assert result["runs"][0]["config_json"] == config_json
