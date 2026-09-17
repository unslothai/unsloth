# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi import HTTPException

from auth import policy
from core.training import account_jobs as jobs
from hub.services.models import account_access
from utils import hf_cache_settings
from utils.account_context import AccountContext, run_as

ALICE = AccountContext("a" * 32, "alice")


@pytest.fixture
def shared_cache(monkeypatch, tmp_path):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "studio"))
    monkeypatch.setattr(policy, "installation_is_multi_user", lambda: True)
    cache = tmp_path / "hub-cache"
    (cache / "models--public--model" / "snapshots" / "rev").mkdir(parents = True)
    (cache / "models--private--model" / "snapshots" / "rev").mkdir(parents = True)
    (cache / "datasets--public--set" / "snapshots" / "rev").mkdir(parents = True)
    monkeypatch.setattr(hf_cache_settings, "active_hf_hub_cache", lambda: str(cache))
    monkeypatch.setattr(hf_cache_settings, "known_hf_hub_caches", lambda: [cache])
    monkeypatch.setattr(account_access, "model_grants", lambda: set())
    monkeypatch.setattr(
        account_access,
        "repo_is_public",
        lambda repo_id, repo_type = "model": not repo_id.startswith("private/"),
    )
    return cache


def _snapshot(cache: Path, prefix: str, repo: str) -> str:
    owner, name = repo.split("/")
    return str(cache / f"{prefix}--{owner}--{name}" / "snapshots" / "rev")


def test_a_visible_cached_model_and_dataset_start_a_managed_training_run(shared_cache):
    """The picker sends the cache path of the row the inventory showed; refusing it left a managed account unable to train on any cached selection at all."""
    config = {
        "model_name": "public/model",
        "model_local_path": _snapshot(shared_cache, "models", "public/model"),
        "hf_dataset": "public/set",
        "dataset_local_path": _snapshot(shared_cache, "datasets", "public/set"),
        "hf_token": "alice-token",
    }
    run_as(ALICE, jobs.validate_job_paths, config)


def test_a_cached_model_the_account_cannot_see_is_still_refused(shared_cache):
    with pytest.raises(HTTPException) as exc:
        run_as(
            ALICE,
            jobs.validate_job_paths,
            {
                "model_name": "private/model",
                "model_local_path": _snapshot(shared_cache, "models", "private/model"),
                "hf_token": "alice-token",
            },
        )
    assert exc.value.status_code == 403


def test_a_path_outside_the_shared_cache_is_still_refused(shared_cache, tmp_path):
    with pytest.raises(HTTPException) as exc:
        run_as(
            ALICE,
            jobs.validate_job_paths,
            {"model_local_path": str(tmp_path / "elsewhere" / "model"), "hf_token": "t"},
        )
    assert exc.value.status_code == 403


def test_the_dataset_picker_can_read_a_visible_cached_dataset(shared_cache):
    from hub.schemas.datasets import LocalDatasetOptionsRequest
    from hub.services.datasets import local_options

    visible = LocalDatasetOptionsRequest(
        dataset_name = "public/set",
        local_path = _snapshot(shared_cache, "datasets", "public/set"),
    )
    run_as(ALICE, local_options.local_dataset_options, visible)

    hidden = LocalDatasetOptionsRequest(
        dataset_name = "private/set",
        local_path = _snapshot(shared_cache, "datasets", "private/set"),
    )
    with pytest.raises(HTTPException) as exc:
        run_as(ALICE, local_options.local_dataset_options, hidden)
    assert exc.value.status_code == 403


def test_a_cached_claim_on_an_invisible_repo_is_refused(shared_cache):
    """``model_known_cached`` makes the preflight pin the shared cache copy locally, so the caller's own Hub token never authorizes the repo: check it here or nothing does."""
    with pytest.raises(HTTPException) as exc:
        run_as(
            ALICE,
            jobs.validate_job_paths,
            {
                "model_name": "private/model",
                "model_known_cached": True,
                "hf_token": "alice-token",
            },
        )
    assert exc.value.status_code == 404


def test_a_cached_dataset_claim_on_an_invisible_repo_is_refused(shared_cache):
    with pytest.raises(HTTPException) as exc:
        run_as(
            ALICE,
            jobs.validate_job_paths,
            {"hf_dataset": "private/set", "dataset_known_cached": True, "hf_token": "t"},
        )
    assert exc.value.status_code == 404


def test_a_cached_claim_on_a_visible_repo_still_starts(shared_cache):
    run_as(
        ALICE,
        jobs.validate_job_paths,
        {
            "model_name": "public/model",
            "model_known_cached": True,
            "hf_dataset": "public/set",
            "dataset_known_cached": True,
            "hf_token": "alice-token",
        },
    )


@pytest.mark.parametrize("repo,expected", [("private/model", 404), ("public/model", None)])
def test_a_cache_fallback_after_a_refused_remote_probe_needs_a_grant(
    shared_cache, monkeypatch, repo, expected
):
    """The remote probe refused the caller's token, so the shared cache copy another account left behind is not a grant; a bare repo id must clear require_model_access."""
    from routes import training
    from models.training import TrainingStartRequest

    def refused(model_name, hf_token):
        raise HTTPException(status_code = 422, detail = {"code": "hf_model_access_denied"})

    monkeypatch.setattr(training, "_remote_untrainable_model_format", refused)
    from core.training import training as core_training

    monkeypatch.setattr(
        core_training,
        "_resolve_model_snapshot",
        lambda model_name, local_path: _snapshot(shared_cache, "models", model_name),
    )
    monkeypatch.setattr(
        training, "_has_trainable_local_weights", lambda *a, **k: True, raising = False
    )
    request = TrainingStartRequest(
        model_name = repo,
        dataset_name = "public/set",
        hf_token = "x",
        training_type = "LoRA/QLoRA",
        format_type = "alpaca",
    )
    if expected is None:
        result = run_as(ALICE, training._reject_untrainable_model_request, request)
        assert result.cached_model_pin[0] == repo
    else:
        with pytest.raises(HTTPException) as exc:
            run_as(ALICE, training._reject_untrainable_model_request, request)
        assert exc.value.status_code == expected
