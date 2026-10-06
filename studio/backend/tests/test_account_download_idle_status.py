# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A managed account must be able to observe an idle download it does not own a job for."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from auth import policy
from hub.services import download_lifecycle
from hub.services.models import account_access as access
from hub.services.models import downloads as model_downloads
from hub.utils import download_registry
from utils.account_context import AccountContext, arun_as, run_as

ALICE = AccountContext("a" * 32, "alice")
BOB = AccountContext("b" * 32, "bob")


@pytest.fixture(autouse = True)
def isolated(monkeypatch, tmp_path):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    monkeypatch.setattr(policy, "installation_is_multi_user", lambda: True)
    monkeypatch.setattr(access, "_public_repos", {})
    monkeypatch.setattr(download_lifecycle, "_job_accounts", {})
    monkeypatch.setattr(
        access,
        "HfApi",
        lambda: SimpleNamespace(
            repo_info = lambda *a, **k: (_ for _ in ()).throw(OSError("offline"))
        ),
    )


def test_status_for_a_repo_with_no_job_is_idle_for_a_managed_account():
    registry = download_registry.DownloadRegistry()
    key = "org/model::"
    state, error, generation = run_as(
        BOB,
        lambda: download_lifecycle.idle_status(
            registry, key, repo_type = "model", repo_id = "org/model", variant = None
        ),
    )
    assert (state, error, generation) == ("idle", None, 0)


def test_hydrating_after_a_restart_settles_instead_of_404ing_forever():
    """The registry and the ownership map are in-memory, so a restart leaves the client's
    persisted active download with no entry at all; its status poll must settle."""
    before = download_registry.DownloadRegistry()
    key = "org/model::"
    before.claim(key, "http", repo_type = "model", repo_id = "org/model")
    run_as(ALICE, download_lifecycle.record_download_account, before, key)

    after_restart = download_registry.DownloadRegistry()
    state, _, _ = run_as(
        ALICE,
        lambda: download_lifecycle.idle_status(
            after_restart, key, repo_type = "model", repo_id = "org/model", variant = None
        ),
    )
    assert state == "idle"


def test_status_route_reports_idle_for_a_managed_account(monkeypatch):
    registry = download_registry.DownloadRegistry()
    monkeypatch.setattr(model_downloads, "_registry", registry)
    monkeypatch.setattr(
        model_downloads, "resolve_cached_repo_id_case", lambda repo_id, **k: repo_id
    )
    status = asyncio.run(
        arun_as(BOB, model_downloads.get_download_status_response("org/model", ""))
    )
    assert status.state == "idle"


def test_another_accounts_live_job_stays_hidden():
    registry = download_registry.DownloadRegistry()
    key = "org/secret::"
    registry.set_job(key, "running")
    download_lifecycle._job_accounts[(id(registry), key)] = ALICE.account_id
    with pytest.raises(HTTPException) as exc:
        run_as(
            BOB,
            lambda: download_lifecycle.idle_status(
                registry, key, repo_type = "model", repo_id = "org/secret", variant = None
            ),
        )
    assert exc.value.status_code == 404
    assert run_as(ALICE, download_lifecycle.download_belongs_to_account, registry, key)
