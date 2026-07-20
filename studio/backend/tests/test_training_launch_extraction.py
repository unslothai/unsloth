# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import threading
import time
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from fastapi import HTTPException

from core.training import launch
from models import TrainingStartRequest


def _request(**overrides) -> TrainingStartRequest:
    base = dict(
        model_name = "unsloth/test",
        training_type = "LoRA/QLoRA",
        format_type = "alpaca",
    )
    base.update(overrides)
    return TrainingStartRequest(**base)


def test_generate_job_id_format():
    job_id = launch.generate_job_id()
    assert job_id.startswith("job_")
    # job_{YYYYMMDD_HHMMSS}_{uuid8}
    parts = job_id.split("_")
    assert len(parts) == 4
    assert len(parts[3]) == 8


def _isolated_datasets_root(tmp_path, monkeypatch):
    """Point UNSLOTH_STUDIO_HOME at tmp_path and return the datasets root."""
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    from utils.paths.storage_roots import datasets_root

    root = datasets_root()
    root.mkdir(parents = True, exist_ok = True)
    return root


def test_validate_s3_support_rejects_without_boto3():
    request = _request(s3_config = {"bucket": "my-bucket", "use_iam_role": True})
    with patch("core.training.s3_dataset.boto3_available", return_value = False):
        with pytest.raises(HTTPException) as exc_info:
            launch.validate_s3_support(request)
    assert exc_info.value.status_code == 501
    assert "boto3" in exc_info.value.detail


def test_validate_training_request_rejects_missing_local_dataset(tmp_path, monkeypatch):
    root = _isolated_datasets_root(tmp_path, monkeypatch)
    request = _request(local_datasets = [str(root / "missing.jsonl")])
    with pytest.raises(HTTPException) as exc_info:
        launch.validate_training_request(request)
    assert exc_info.value.status_code == 400
    assert "not found" in exc_info.value.detail


def test_validate_training_request_resolves_local_dataset(tmp_path, monkeypatch):
    root = _isolated_datasets_root(tmp_path, monkeypatch)
    dataset = root / "data.jsonl"
    dataset.write_text('{"text": "hi"}\n')
    request = _request(local_datasets = [str(dataset)])
    resume_output_dir = launch.validate_training_request(request)
    assert resume_output_dir is None
    assert request.local_datasets == [str(dataset)]


def test_validate_training_request_rejects_streaming_without_hf_dataset():
    request = _request(dataset_streaming = True, max_steps = 10)
    with pytest.raises(HTTPException) as exc_info:
        launch.validate_training_request(request)
    assert exc_info.value.status_code == 400
    assert "hf_dataset" in exc_info.value.detail


def test_launch_passes_before_spawn_hook_and_kwargs():
    captured = {}

    def _start_training(**kwargs):
        captured.update(kwargs)
        return True

    backend = SimpleNamespace(start_training = _start_training)
    ok = launch.launch_training(
        job_id = "job_x",
        request = _request(),
        resume_output_dir = None,
        subject = "test-user",
        backend = backend,
        model_defaults_loader = lambda _name: {},
    )
    assert ok is True
    assert captured["job_id"] == "job_x"
    assert callable(captured["before_spawn"])
    assert captured["model_name"] == "unsloth/test"
    assert captured["subject"] == "test-user"
    # Defaulted the same way the route always did.
    assert captured["gradient_checkpointing"] == "unsloth"
    assert captured["hf_token"] == ""


def test_launch_propagates_value_error():
    def _start_training(**kwargs):
        raise ValueError("Invalid gpu_ids [99]")

    backend = SimpleNamespace(start_training = _start_training)
    with pytest.raises(ValueError, match = "gpu_ids"):
        launch.launch_training(
            job_id = "job_x",
            request = _request(gpu_ids = [99]),
            resume_output_dir = None,
            subject = "test-user",
            backend = backend,
            model_defaults_loader = lambda _name: {},
        )


def test_start_lock_serializes_concurrent_launches():
    max_concurrent = [0]
    active = [0]
    lock = threading.Lock()

    def _start_training(**kwargs):
        with lock:
            active[0] += 1
            max_concurrent[0] = max(max_concurrent[0], active[0])
        time.sleep(0.05)
        with lock:
            active[0] -= 1
        return True

    backend = SimpleNamespace(start_training = _start_training)

    def _launch(job_id: str) -> None:
        launch.launch_training(
            job_id = job_id,
            request = _request(),
            resume_output_dir = None,
            subject = "test-user",
            backend = backend,
            model_defaults_loader = lambda _name: {},
        )

    threads = [threading.Thread(target = _launch, args = (f"job_{i}",)) for i in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert max_concurrent[0] == 1


def test_yaml_trust_applies_only_for_trusted_org():
    captured = {}
    backend = SimpleNamespace(start_training = lambda **kw: captured.update(kw) or True)

    with patch("utils.security.trusted_org.is_trusted_org_repo", return_value = True) as trusted:
        launch.launch_training(
            job_id = "job_x",
            request = _request(),
            resume_output_dir = None,
            subject = "test-user",
            backend = backend,
            model_defaults_loader = lambda _name: {"training": {"trust_remote_code": True}},
        )
    trusted.assert_called_once()
    assert captured["trust_remote_code"] is True

    captured.clear()
    with patch("utils.security.trusted_org.is_trusted_org_repo", return_value = False):
        launch.launch_training(
            job_id = "job_y",
            request = _request(),
            resume_output_dir = None,
            subject = "test-user",
            backend = backend,
            model_defaults_loader = lambda _name: {"training": {"trust_remote_code": True}},
        )
    assert captured["trust_remote_code"] is False
