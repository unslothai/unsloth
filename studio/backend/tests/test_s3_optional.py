# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""S3 is an opt-in dependency, including when the full API is imported."""

import importlib.machinery
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


_AWS_MODULES = ("boto3", "botocore", "s3transfer", "jmespath")
_INSTALL_COMMAND = 'python -m pip install "unsloth[s3]"'


@pytest.fixture
def without_aws(monkeypatch):
    monkeypatch.setenv("UNSLOTH_S3_AUTO_INSTALL", "0")
    # None blocks imports even with the SDK installed, and find_spec reports it as absent.
    for name in _AWS_MODULES:
        monkeypatch.setitem(sys.modules, name, None)


def test_backend_and_training_routes_import_without_aws(tmp_path):
    code = f"""
import sys
for name in {_AWS_MODULES!r}:
    sys.modules[name] = None
import main
import routes.training
assert '/api/train/start' in main.app.openapi()['paths']
assert all(sys.modules[name] is None for name in {_AWS_MODULES!r})
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd = Path(__file__).resolve().parents[1],
        env = {
            **os.environ,
            "UNSLOTH_STUDIO_HOME": str(tmp_path),
            "UNSLOTH_STUDIO_DISABLE_TORCH_WARM": "1",
        },
        capture_output = True,
        text = True,
        timeout = 60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_local_dataset_loading_without_aws(without_aws, tmp_path):
    from utils.datasets.cache_safe import load_dataset_cache_safe
    from utils.datasets.raw_text import prepare_raw_text_dataset

    source = tmp_path / "train.jsonl"
    source.write_text('{"text": "hello"}\n{"text": "world"}\n', encoding = "utf-8")
    dataset = load_dataset_cache_safe(
        "json", data_files = str(source), split = "train", cache_dir = str(tmp_path / "cache")
    )
    prepared = prepare_raw_text_dataset(dataset, eos_token = "<eos>", append_eos = True)
    assert list(prepared.dataset["text"]) == ["hello<eos>", "world<eos>"]
    assert all(sys.modules[name] is None for name in _AWS_MODULES)


def test_s3_request_without_aws_returns_actionable_501(without_aws, monkeypatch):
    from routes import training
    from core.training import token_preview

    backend = Mock()
    monkeypatch.setattr(training, "get_training_backend", lambda: backend)
    monkeypatch.setattr(token_preview, "stop_token_preview_worker", lambda: None)
    app = FastAPI()
    app.include_router(training.router, prefix = "/api/train")
    app.dependency_overrides[training.get_current_subject] = lambda: "test-user"
    app.dependency_overrides[training.authenticated_via_api_key] = lambda: False
    with TestClient(app) as client:
        response = client.post(
            "/api/train/start",
            json = {
                "model_name": "unsloth/test",
                "training_type": "LoRA/QLoRA",
                "format_type": "alpaca",
                "s3_config": {"bucket": "test-bucket", "use_iam_role": True},
            },
        )
    assert response.status_code == 501
    assert _INSTALL_COMMAND in response.json()["detail"]
    backend.start_training.assert_not_called()


def test_s3_download_uses_optional_sdk(monkeypatch, tmp_path):
    from core.training import s3_dataset

    def download(bucket, key, filename, **kwargs):
        assert bucket == "test-bucket"
        assert key == "train.jsonl"
        Path(filename).write_text('{"text": "hello"}\n', encoding = "utf-8")

    client = Mock()
    client.get_paginator.return_value.paginate.return_value = [
        {"Contents": [{"Key": "train.jsonl"}]}
    ]
    client.download_file.side_effect = download
    sdk = ModuleType("boto3")
    sdk.__spec__ = importlib.machinery.ModuleSpec("boto3", loader = None)
    sdk.client = Mock(return_value = client)
    monkeypatch.setitem(sys.modules, "boto3", sdk)

    files = s3_dataset.download_s3_dataset(
        {
            "bucket": "test-bucket",
            "region": "us-west-2",
            "access_key_id": "test-key",
            "secret_access_key": "test-secret",
        },
        dest_dir = str(tmp_path),
    )
    assert len(files) == 1
    assert Path(files[0]).read_text(encoding = "utf-8") == '{"text": "hello"}\n'
    sdk.client.assert_called_once_with(
        "s3",
        region_name = "us-west-2",
        aws_access_key_id = "test-key",
        aws_secret_access_key = "test-secret",
    )


@pytest.mark.parametrize("s3_requested", [False, True])
def test_training_installs_sdk_only_for_s3_then_continues(monkeypatch, s3_requested):
    import asyncio
    from routes import training
    from core.training import s3_dataset, token_preview
    from utils import s3_runtime

    installed = []

    def install():
        # The installer must run off the async route's event loop.
        with pytest.raises(RuntimeError, match = "no running event loop"):
            asyncio.get_running_loop()
        installed.append(True)

    backend = Mock()
    backend.is_training_active.return_value = True
    backend.current_job_id = "existing-job"
    monkeypatch.setattr(training, "get_training_backend", lambda: backend)
    monkeypatch.setattr(token_preview, "stop_token_preview_worker", lambda: None)
    monkeypatch.setattr(s3_dataset, "boto3_available", lambda: False)
    monkeypatch.setattr(s3_runtime, "ensure_boto3", install)
    app = FastAPI()
    app.include_router(training.router, prefix = "/api/train")
    app.dependency_overrides[training.get_current_subject] = lambda: "test-user"
    app.dependency_overrides[training.authenticated_via_api_key] = lambda: False
    payload = {"model_name": "unsloth/test", "training_type": "LoRA/QLoRA", "format_type": "alpaca"}
    if s3_requested:
        payload["s3_config"] = {"bucket": "test-bucket", "use_iam_role": True}
    with TestClient(app) as client:
        response = client.post("/api/train/start", json = payload)
    assert response.status_code == 200
    assert response.json()["message"].startswith("Training is already in progress")
    assert installed == ([True] if s3_requested else [])
