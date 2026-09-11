# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The offline dataset fallback must not hand another account's cached private dataset away."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi import HTTPException

from auth import policy
from hub.services.models import account_access
from hub.utils import dataset_cache
from models.training import TrainingStartRequest
from routes import training
from utils import hf_cache_settings
from utils.account_context import AccountContext, run_as

ALICE = AccountContext("a" * 32, "alice")


@pytest.fixture
def shared_cache(monkeypatch, tmp_path):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "studio"))
    monkeypatch.setattr(policy, "installation_is_multi_user", lambda: True)
    cache = tmp_path / "hub-cache"
    for repo in ("public--set", "private--set"):
        (cache / f"datasets--{repo}" / "snapshots" / "rev").mkdir(parents = True)
    monkeypatch.setattr(hf_cache_settings, "active_hf_hub_cache", lambda: str(cache))
    monkeypatch.setattr(hf_cache_settings, "known_hf_hub_caches", lambda: [cache])
    monkeypatch.setattr(account_access, "model_grants", lambda: set())
    monkeypatch.setattr(
        account_access,
        "repo_is_public",
        lambda repo_id, repo_type = "model": not repo_id.startswith("private/"),
    )
    # Bob already downloaded both; the cache lookup finds them for anyone.
    monkeypatch.setattr(
        dataset_cache,
        "training_dataset_cache_pin",
        lambda repo_id, local_path = None: (
            cache / f"datasets--{repo_id.replace('/', '--')}" / "snapshots" / "rev",
            "rev",
        ),
    )
    # No Hub: only the offline fallback can answer.
    monkeypatch.setattr(training, "hf_env_offline", lambda: True)
    monkeypatch.setattr(training, "_hub_unreachable", lambda: True)
    return cache


def _request(dataset: str) -> TrainingStartRequest:
    return TrainingStartRequest(
        model_name = "public/model",
        hf_dataset = dataset,
        hf_token = "not-a-real-token",
        training_type = "LoRA/QLoRA",
        format_type = "alpaca",
    )


def test_the_offline_cache_fallback_refuses_an_invisible_dataset(shared_cache):
    """No cached claim is advertised, so nothing else in the start path checks the grant."""
    with pytest.raises(HTTPException) as exc:
        run_as(ALICE, training._preflight_hf_dataset_request, _request("private/set"))
    assert exc.value.status_code == 404


def test_the_offline_cache_fallback_still_accepts_a_visible_dataset(shared_cache):
    assert run_as(ALICE, training._preflight_hf_dataset_request, _request("public/set")) is None


def test_the_owner_offline_cache_fallback_is_unchanged(shared_cache):
    assert training._preflight_hf_dataset_request(_request("private/set")) is None
