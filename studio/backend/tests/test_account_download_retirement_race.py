# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A download request caught mid-flight by retirement must not reach spawn.

Retirement scans the download registries for jobs to cancel. A request that is still inside a
pre-claim await owns no registry entry yet, so the scan reports clean retirement and the request
then claims a key and launches a worker holding the deleted account's captured token."""

from __future__ import annotations

import asyncio
import subprocess
import threading

import pytest
from fastapi import HTTPException

from auth import policy
from core.training import account_jobs as jobs
from hub.schemas.downloads import DownloadDatasetRequest, DownloadModelRequest
from hub.services import download_lifecycle
from hub.services.datasets import downloads as dataset_downloads
from hub.services.models import account_access as access, downloads
from hub.utils import download_manifest, download_registry
from utils.account_context import AccountContext, run_as

ALICE = AccountContext("a" * 32, "alice")


@pytest.fixture(autouse = True)
def isolated(monkeypatch, tmp_path):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    monkeypatch.setattr(policy, "installation_is_multi_user", lambda: True)
    monkeypatch.setattr(policy, "installation_has_managed_accounts", lambda: True)
    monkeypatch.setattr(download_lifecycle, "_job_accounts", {})
    monkeypatch.setattr(downloads, "_registry", download_registry.DownloadRegistry())
    monkeypatch.setattr(dataset_downloads, "_account_registries", {})
    monkeypatch.setattr(jobs, "_services", [])
    monkeypatch.setattr(jobs, "_retired", set())
    monkeypatch.setattr(
        download_lifecycle, "resolve_requested_use_xet", lambda mode, use_xet: (False, "HTTP")
    )
    from core import research_runs
    from core.rag import folder_sync, ingestion

    monkeypatch.setattr(ingestion, "retire_account_ingestions", lambda: None)
    monkeypatch.setattr(folder_sync, "retire_account_sync", lambda: None)
    monkeypatch.setattr(research_runs, "retire_account_research", lambda account: None)


def _retire_while_blocked(blocked, release, start_request):
    """Run *start_request* until it parks on *blocked*, retire ALICE, then let it finish."""
    result = {}

    def request_thread():
        try:
            result["value"] = run_as(ALICE, lambda: asyncio.run(start_request()))
        except HTTPException as exc:
            result["error"] = (exc.status_code, exc.detail)
        except Exception as exc:  # noqa: BLE001
            result["error"] = (type(exc).__name__, str(exc))

    thread = threading.Thread(target = request_thread)
    thread.start()
    assert blocked.wait(timeout = 30), "the request never reached its pre-claim await"
    jobs.retire_account_jobs(ALICE)
    release.set()
    thread.join(timeout = 30)
    assert not thread.is_alive()
    return result


def test_model_download_cannot_launch_after_retirement_reported_success(monkeypatch):
    blocked, release, spawned = threading.Event(), threading.Event(), []
    monkeypatch.setattr(dataset_downloads, "retire_account_downloads", lambda: None)

    def authorize(repo_id, repo_type, hf_token):
        blocked.set()
        release.wait(timeout = 30)

    monkeypatch.setattr(access, "authorize_download", authorize)
    monkeypatch.setattr(downloads, "resolve_cached_repo_id_case", lambda repo_id, **k: repo_id)
    monkeypatch.setattr(downloads, "_load_in_flight", lambda repo_id: False)

    def fake_spawn(repo_id, variant, hf_token, **kwargs):
        proc = subprocess.Popen(["sleep", "3"])
        spawned.append((proc, hf_token))
        return proc

    monkeypatch.setattr(downloads, "_spawn_download_worker", fake_spawn)

    result = _retire_while_blocked(
        blocked,
        release,
        lambda: downloads.download_model_response(
            DownloadModelRequest(repo_id = "org/private", use_xet = False),
            hf_token = "alice-secret-token",
        ),
    )
    for proc, _ in spawned:
        proc.kill()
    assert not spawned, f"a worker was launched for a retired account: {spawned}"
    assert result.get("error") == (403, "Account is retired"), result


def test_dataset_download_cannot_launch_after_retirement_reported_success(monkeypatch):
    blocked, release, spawned = threading.Event(), threading.Event(), []
    monkeypatch.setattr(downloads, "retire_account_downloads", lambda: None)

    def blocking_case(repo_id, **kwargs):
        blocked.set()
        release.wait(timeout = 30)
        return repo_id

    monkeypatch.setattr(dataset_downloads, "resolve_cached_repo_id_case", blocking_case)
    monkeypatch.setattr(access, "authorize_download", lambda *a, **k: None)

    def fake_spawn(args, hf_token, **kwargs):
        proc = subprocess.Popen(["sleep", "3"])
        spawned.append((proc, hf_token))
        return proc

    monkeypatch.setattr(download_lifecycle, "spawn_worker", fake_spawn)
    monkeypatch.setattr(download_lifecycle, "register_worker", lambda *a, **k: True)

    result = _retire_while_blocked(
        blocked,
        release,
        lambda: dataset_downloads.download_dataset_response(
            DownloadDatasetRequest(repo_id = "org/private-ds", use_xet = False),
            hf_token = "alice-secret-token",
        ),
    )
    for proc, _ in spawned:
        proc.kill()
    assert not spawned, f"a dataset worker was launched for a retired account: {spawned}"
    assert result.get("error") == (403, "Account is retired"), result


def test_dataset_retirement_cancels_a_job_claimed_but_not_yet_launched(monkeypatch):
    """Ownership must be recorded with the claim: retirement scans the per-account registry, and an
    unattributed job makes its cancel raise "Download not found" and abort the whole deletion."""
    blocked, release, spawned = threading.Event(), threading.Event(), []
    monkeypatch.setattr(downloads, "retire_account_downloads", lambda: None)
    monkeypatch.setattr(dataset_downloads, "_registry", download_registry.DownloadRegistry())
    monkeypatch.setattr(dataset_downloads, "resolve_cached_repo_id_case", lambda r, **k: r)
    monkeypatch.setattr(access, "authorize_download", lambda *a, **k: None)

    def blocking_clear(*args, **kwargs):
        # Between the claim and launch_worker: the route clears the cancel marker here.
        blocked.set()
        release.wait(timeout = 30)

    monkeypatch.setattr(download_manifest, "clear_cancel_marker", blocking_clear)

    def fake_spawn(args, hf_token, **kwargs):
        proc = subprocess.Popen(["sleep", "3"])
        spawned.append((proc, hf_token))
        return proc

    monkeypatch.setattr(download_lifecycle, "spawn_worker", fake_spawn)
    monkeypatch.setattr(download_lifecycle, "register_worker", lambda *a, **k: True)

    result = _retire_while_blocked(
        blocked,
        release,
        lambda: dataset_downloads.download_dataset_response(
            DownloadDatasetRequest(repo_id = "org/private-ds", use_xet = False),
            hf_token = "alice-secret-token",
        ),
    )
    for proc, _ in spawned:
        proc.kill()
    assert not spawned, f"a dataset worker was launched for a retired account: {spawned}"
    assert result.get("error") == (403, "Account is retired"), result
