# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

from pathlib import Path

import pytest

from auth import policy
from hub.services.models import account_access
from utils import hf_cache_settings
from utils.account_context import AccountContext, OWNER, run_as

ALICE = AccountContext("a" * 32, "alice")

_CARD = """---
configs:
- config_name: secret_config
  data_files:
  - split: secret_split
    path: data.csv
---
"""


@pytest.fixture
def shared_cache(monkeypatch, tmp_path):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "studio"))
    monkeypatch.setattr(policy, "installation_is_multi_user", lambda: True)
    cache = tmp_path / "hub-cache"
    for repo in ("public--set", "private--set"):
        snapshot = cache / f"datasets--{repo}" / "snapshots" / "rev"
        snapshot.mkdir(parents = True)
        (snapshot / "README.md").write_text(_CARD, encoding = "utf-8")
    monkeypatch.setattr(hf_cache_settings, "active_hf_hub_cache", lambda: str(cache))
    monkeypatch.setattr(hf_cache_settings, "known_hf_hub_caches", lambda: [cache])
    monkeypatch.setattr(account_access, "model_grants", lambda: set())
    monkeypatch.setattr(
        account_access,
        "repo_is_public",
        lambda repo_id, repo_type = "model": not repo_id.startswith("private/"),
    )
    return cache


def _options(dataset_name: str, local_path: str | None = None):
    from hub.schemas.datasets import LocalDatasetOptionsRequest
    from hub.services.datasets import local_options
    return local_options.local_dataset_options(
        LocalDatasetOptionsRequest(dataset_name = dataset_name, local_path = local_path)
    )


def test_a_repo_only_request_cannot_read_another_accounts_private_cached_dataset(shared_cache):
    """Omitting local_path skipped every check and resolved the installation-wide cache."""
    response = run_as(ALICE, _options, "private/set")
    assert response.cache_available is False
    assert response.splits == []


def test_a_repo_only_request_still_reads_a_visible_cached_dataset(shared_cache):
    response = run_as(ALICE, _options, "public/set")
    assert response.cache_available is True
    assert [(item.config, item.split) for item in response.splits] == [
        ("secret_config", "secret_split")
    ]


def test_the_owner_still_reads_any_cached_dataset(shared_cache):
    response = run_as(OWNER, _options, "private/set")
    assert response.cache_available is True
    assert response.splits


def test_a_single_user_install_still_reads_any_cached_dataset(shared_cache, monkeypatch):
    monkeypatch.setattr(policy, "installation_is_multi_user", lambda: False)
    response = _options("private/set")
    assert response.cache_available is True
    assert response.splits
