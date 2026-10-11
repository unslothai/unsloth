# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi import HTTPException

from auth import policy
from hub.schemas.datasets import CheckFormatRequest
from hub.services.datasets import formatting
from hub.services.models import account_access
from utils import hf_cache_settings
from utils.account_context import AccountContext, run_as

ALICE = AccountContext("a" * 32, "alice")

SECRET = "bob-private-row"


def _cache_dataset(cache: Path, repo: str, text: str) -> Path:
    owner, name = repo.split("/")
    snapshot = cache / f"datasets--{owner}--{name}" / "snapshots" / "rev"
    snapshot.mkdir(parents = True)
    (snapshot / "train.jsonl").write_text(
        json.dumps({"instruction": "q", "output": text}) + "\n",
        encoding = "utf-8",
    )
    return snapshot


@pytest.fixture
def shared_cache(monkeypatch, tmp_path):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "studio"))
    monkeypatch.setattr(policy, "installation_is_multi_user", lambda: True)
    monkeypatch.setattr(formatting, "ensure_audio_decoding", lambda: True)
    cache = tmp_path / "hub-cache"
    cache.mkdir()
    _cache_dataset(cache, "private/set", SECRET)
    _cache_dataset(cache, "public/set", "public-row")
    monkeypatch.setattr(hf_cache_settings, "active_hf_hub_cache", lambda: str(cache))
    monkeypatch.setattr(hf_cache_settings, "known_hf_hub_caches", lambda: [cache])
    monkeypatch.setattr(account_access, "model_grants", lambda: set())
    monkeypatch.setattr(
        account_access,
        "repo_is_public",
        lambda repo_id, repo_type = "model": not repo_id.startswith("private/"),
    )
    # The shared cache-read gate answers for the credential; the account gate is under test.
    from hub.utils import hf_tokens

    monkeypatch.setattr(
        hf_tokens,
        "cache_reads_authorized",
        lambda hf_token, **kwargs: True,
    )
    return cache


def _preview(dataset_name: str, prefer_local_cache: bool = True):
    return run_as(
        ALICE,
        formatting.check_format_response,
        CheckFormatRequest(
            dataset_name = dataset_name,
            prefer_local_cache = prefer_local_cache,
        ),
        "bogus-token",
    )


@pytest.fixture
def offline_hub(monkeypatch):
    class _NoHubApi:
        def __init__(self, *args, **kwargs):
            pass

        def list_repo_files(self, *args, **kwargs):
            raise ConnectionError("offline")

    import datasets

    real_load_dataset = datasets.load_dataset

    def _offline_load_dataset(*args, **kwargs):
        if "path" in kwargs:
            raise ConnectionError("offline")
        return real_load_dataset(*args, **kwargs)

    monkeypatch.setattr("huggingface_hub.HfApi", _NoHubApi)
    monkeypatch.setattr(datasets, "load_dataset", _offline_load_dataset)


def test_managed_account_cannot_preview_another_accounts_cached_private_dataset(shared_cache):
    with pytest.raises(HTTPException) as exc:
        _preview("private/set")
    assert exc.value.status_code == 404
    assert SECRET not in json.dumps(exc.value.detail)


def test_managed_account_still_previews_a_public_cached_dataset(shared_cache):
    response = _preview("public/set")
    assert SECRET not in json.dumps(response.preview_samples)
    assert response.preview_samples


def test_managed_account_still_previews_its_own_granted_cached_dataset(shared_cache, monkeypatch):
    monkeypatch.setattr(account_access, "model_grants", lambda: {"dataset:private/set"})
    response = _preview("private/set")
    assert response.preview_samples


def test_owner_preview_is_unchanged(shared_cache):
    response = formatting.check_format_response(
        CheckFormatRequest(dataset_name = "private/set", prefer_local_cache = True),
        "bogus-token",
    )
    assert response.preview_samples


def test_online_fallback_cannot_reach_another_accounts_cached_private_dataset(
    shared_cache, offline_hub
):
    with pytest.raises(Exception) as exc:
        _preview("private/set", prefer_local_cache = False)
    assert SECRET not in json.dumps(getattr(exc.value, "detail", str(exc.value)))


def test_online_fallback_still_serves_a_public_cached_dataset(shared_cache, offline_hub):
    assert _preview("public/set", prefer_local_cache = False).preview_samples
