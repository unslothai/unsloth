# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Private shared-cache content needs anonymous Hub proof or a durable account grant."""

from __future__ import annotations

import asyncio
import io
import json
import logging
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from auth import policy
from hub.services import download_lifecycle
from hub.services.models import account_access as access, cache_inventory, local_inventory
from hub.utils import download_registry
from utils.account_context import OWNER, AccountContext, arun_as, run_as
from utils.paths.storage_roots import studio_db_path

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


@pytest.mark.parametrize("repo_type", ["model", "dataset"])
@pytest.mark.parametrize(
    "answer,visible",
    [
        ("public", True),
        ("private", False),
        ("unreachable", False),
        ("unknown", False),
        ("gated", False),
    ],
)
def test_public_proof_is_anonymous_cached_and_fail_closed(monkeypatch, answer, visible, repo_type):
    calls = []

    def info(repo, **kwargs):
        calls.append((repo, kwargs))
        if answer == "unreachable":
            raise OSError("Hub unavailable")
        return SimpleNamespace(
            private = {"public": False, "private": True, "unknown": None, "gated": False}[answer],
            gated = answer == "gated",
        )

    monkeypatch.setattr(access, "HfApi", lambda: SimpleNamespace(repo_info = info))
    assert run_as(ALICE, access.repo_visible, "Org/Secret", repo_type) is visible
    assert run_as(BOB, access.repo_visible, "Org/Secret", repo_type) is visible
    assert calls == [("Org/Secret", {"repo_type": repo_type, "token": False, "timeout": 5.0})]
    assert run_as(OWNER, access.repo_visible, "Org/Secret", repo_type)
    assert len(calls) == 1


@pytest.mark.parametrize("failure", ["unreachable", "forced_offline"])
def test_a_proven_public_repo_stays_visible_when_the_hub_cannot_be_asked(
    monkeypatch, tmp_path, failure
):
    answers = {"mode": "public"}

    def info(repo, **kwargs):
        if answers["mode"] == "public":
            return SimpleNamespace(private = False, gated = False)
        if answers["mode"] == "forced_offline":
            from huggingface_hub.errors import OfflineModeIsEnabled
            raise OfflineModeIsEnabled("HF_HUB_OFFLINE=1")
        if answers["mode"] == "private":
            raise type("RepositoryNotFoundError", (Exception,), {})(
                "private",
            )
        raise OSError("Hub unavailable")

    monkeypatch.setattr(access, "HfApi", lambda: SimpleNamespace(repo_info = info))
    assert run_as(ALICE, access.repo_visible, "Org/Public")
    assert json.loads(access._public_verdicts_path().read_text()).keys() == {"model:org/public"}
    assert access._public_verdicts_path().is_relative_to(tmp_path / "cache")

    access._public_repos.clear()
    answers["mode"] = failure
    assert run_as(BOB, access.repo_visible, "org/public"), "the proof on disk carries the answer"
    assert not run_as(BOB, access.repo_visible, "org/never-proven")

    access._public_repos.clear()
    answers["mode"] = "private"
    error = Exception("gone")
    error.response = SimpleNamespace(status_code = 404)

    def definitive(repo, **kwargs):
        raise error

    monkeypatch.setattr(access, "HfApi", lambda: SimpleNamespace(repo_info = definitive))
    assert not run_as(ALICE, access.repo_visible, "org/public")
    assert json.loads(access._public_verdicts_path().read_text()) == {}


def test_grants_survive_restart_and_username_reuse_inherits_nothing():
    run_as(ALICE, access.record_model_grant, "Org/Secret")
    path = run_as(ALICE, studio_db_path)
    with sqlite3.connect(path) as conn:
        raw = conn.execute(
            "SELECT value_json FROM app_settings WHERE key = 'model_grants'"
        ).fetchone()[0]
    assert json.loads(raw) == ["model:org/secret"]
    access._public_repos.clear()
    assert run_as(ALICE, access.repo_visible, "org/secret")
    assert not run_as(BOB, access.repo_visible, "org/secret")
    assert not run_as(AccountContext("c" * 32, "alice"), access.repo_visible, "org/secret")
    assert not run_as(ALICE, access.repo_visible, "org/secret", "dataset")
    run_as(BOB, access.record_model_grant, "Org/Secret")
    assert run_as(BOB, access.repo_visible, "org/secret")


def test_simultaneous_download_completions_preserve_all_grants():
    with ThreadPoolExecutor(max_workers = 4) as pool:
        futures = [
            pool.submit(run_as, ALICE, access.record_model_grant, f"org/model-{i}")
            for i in range(12)
        ]
        for future in futures:
            future.result()
    assert run_as(ALICE, access.model_grants) == {f"model:org/model-{i}" for i in range(12)}


def test_snapshot_paths_and_symlinks_cannot_bypass_repo_grants(monkeypatch, tmp_path):
    from utils import hf_cache_settings

    cache = tmp_path / "cache"
    monkeypatch.setattr(hf_cache_settings, "known_hf_hub_caches", lambda: [cache])
    snapshot = cache / "models--org--secret" / "snapshots" / "commit"
    snapshot.mkdir(parents = True)
    weights = snapshot / "model.gguf"
    weights.write_bytes(b"weights")
    assert not run_as(BOB, access.model_visible, str(weights))
    run_as(ALICE, access.record_model_grant, "org/secret")
    assert run_as(ALICE, access.model_visible, str(weights))
    assert not run_as(BOB, access.model_visible, str(weights))
    bob_root = tmp_path / "accounts" / BOB.account_id
    bob_root.mkdir(parents = True)
    link = bob_root / "stolen.gguf"
    link.symlink_to(weights)
    assert not run_as(BOB, access.model_visible, str(link))
    # A directory merely named like a cache repository is not a shared cache.
    fake = tmp_path / "accounts" / ALICE.account_id / "models--org--public"
    fake.mkdir(parents = True)
    monkeypatch.setattr(access, "repo_is_public", lambda *a: True)
    assert not run_as(BOB, access.model_visible, str(fake))


@pytest.mark.parametrize("kind", ["models", "gguf"])
def test_shared_catalog_is_filtered_after_each_account_reads_it(monkeypatch, kind):
    rows = [{"repo_id": "org/secret", "path": None}, {"repo_id": "org/public", "path": None}]
    run_as(ALICE, access.record_model_grant, "org/secret")
    monkeypatch.setattr(access, "repo_is_public", lambda repo, *a: repo == "org/public")

    async def scan(*args):
        return cache_inventory._CachedInventoryScan(rows, True)

    monkeypatch.setattr(cache_inventory, "_shared_cached_inventory_scan", scan)
    fn = (
        cache_inventory.list_cached_models_response
        if kind == "models"
        else cache_inventory.list_cached_gguf_response
    )
    assert len(asyncio.run(arun_as(ALICE, fn()))["cached"]) == 2
    assert asyncio.run(arun_as(BOB, fn()))["cached"] == [rows[1]]
    assert len(asyncio.run(arun_as(OWNER, fn()))["cached"]) == 2
    assert len(rows) == 2


def test_local_inventory_does_not_mutate_shared_scan_objects():
    class Response:
        models = [SimpleNamespace(path = "org/secret", id = "same-id")]

        def model_copy(self, *, update):
            return SimpleNamespace(**update)

    response = Response()
    assert asyncio.run(arun_as(BOB, local_inventory._account_local_response(response))).models == []
    assert len(response.models) == 1


@pytest.mark.parametrize("repo_type", ["model", "dataset"])
def test_only_successful_downloads_record_a_grant(monkeypatch, repo_type):
    from core.inference import local_model_resolver

    monkeypatch.setattr(local_model_resolver, "note_downloaded", lambda *a: None)
    monkeypatch.setattr(local_model_resolver, "invalidate_index", lambda **k: None)
    monkeypatch.setattr(local_model_resolver, "warm_index_soon", lambda: None)
    monkeypatch.setattr(
        download_lifecycle.download_manifest, "clear_cancel_marker", lambda *a, **k: None
    )
    registry = SimpleNamespace(
        cancel_requested = lambda key: False,
        drop_process = lambda *a: True,
        get_job_metadata = lambda key: None,
        set_job = lambda *a: None,
        update_job_transport = lambda *a: None,
    )
    for rc in [1, 0]:
        proc = SimpleNamespace(stderr = io.BytesIO(), wait = lambda: rc)
        state = run_as(
            ALICE,
            download_lifecycle.finalize_worker_exit,
            registry,
            "org/secret::",
            proc,
            hf_token = "alice-token",
            label = "org/secret",
            log_prefix = "Download",
            logger = logging.getLogger(__name__),
            repo_type = repo_type,
            repo_id = "org/secret",
        )
        assert state == ("complete" if rc == 0 else "error")
        assert run_as(ALICE, access.repo_visible, "org/secret", repo_type) is (rc == 0)
        assert not run_as(BOB, access.repo_visible, "org/secret", repo_type)


def test_private_cache_hit_cannot_turn_into_a_download_grant_without_hub_authorization():
    with pytest.raises(HTTPException) as exc:
        run_as(BOB, access.authorize_download, "org/secret", "model", None)
    assert exc.value.status_code == 404
    assert run_as(BOB, access.model_grants) == set()


def test_download_cancel_and_status_are_owned_by_the_initiating_account():
    registry = download_registry.DownloadRegistry()
    key = "org/secret::"
    registry.set_job(key, "downloading")
    download_lifecycle._job_accounts[(id(registry), key)] = ALICE.account_id
    for fn in [
        lambda: download_lifecycle.cancel_worker(
            registry, key, generation = None, label = "secret", logger = logging.getLogger(__name__)
        ),
        lambda: download_lifecycle.idle_status(
            registry, key, repo_type = "model", repo_id = "org/secret", variant = None
        ),
    ]:
        with pytest.raises(HTTPException) as exc:
            run_as(BOB, fn)
        assert exc.value.status_code == 404
    assert run_as(ALICE, download_lifecycle.download_belongs_to_account, registry, key)


def test_single_owner_does_no_hub_or_grant_io(monkeypatch, tmp_path):
    def unexpected(*args, **kwargs):
        raise AssertionError("owner must not probe private-model policy")

    monkeypatch.setattr(access, "HfApi", unexpected)
    monkeypatch.setattr(access, "model_grants", unexpected)
    monkeypatch.setattr(policy, "installation_is_multi_user", lambda: False)
    rows = [{"repo_id": "org/secret"}]
    assert access.filter_model_rows(rows) is rows
    assert access.model_visible("/any/legacy/path")
    access.record_model_grant("org/secret")
    assert not (tmp_path / "studio.db").exists()
    monkeypatch.setenv("HF_TOKEN", "owner-token")
    assert access.ambient_hf_token() == "owner-token"


def test_managed_directory_scans_and_media_companions_are_private(tmp_path):
    from hub.services.models import folder_browser

    owner_output = str(tmp_path / "outputs")
    alice_output = str(tmp_path / "accounts" / ALICE.account_id / "outputs")
    assert run_as(ALICE, access.private_directory, owner_output, "outputs") == alice_output
    assert run_as(ALICE, folder_browser._build_browse_allowlist) == [
        tmp_path / "accounts" / ALICE.account_id
    ]
    with pytest.raises(HTTPException):
        run_as(BOB, access.private_directory, alice_output, "outputs")
    for reference in [alice_output + "/model.gguf", "../../private/model.gguf"]:
        with pytest.raises(HTTPException):
            run_as(BOB, access.require_media_references, SimpleNamespace(gguf_filename = reference))
    with pytest.raises(HTTPException) as exc:
        run_as(BOB, access.require_installation_owner)
    assert exc.value.status_code == 403


def test_managed_download_process_never_inherits_ambient_hf_tokens(monkeypatch):
    from utils import hf_cache_settings

    captured = []
    monkeypatch.setattr(
        hf_cache_settings,
        "get_hf_cache_paths",
        lambda: SimpleNamespace(
            child_env = lambda: {
                "HF_TOKEN": "owner-secret",
                "HUGGING_FACE_HUB_TOKEN": "legacy-owner-secret",
            }
        ),
    )
    monkeypatch.setattr(
        download_lifecycle.subprocess,
        "Popen",
        lambda *args, **kwargs: captured.append(kwargs["env"]) or SimpleNamespace(pid = 123),
    )
    run_as(
        ALICE, download_lifecycle.spawn_worker, ["--repo-id", "org/private"], None, use_xet = False
    )
    assert "HF_TOKEN" not in captured[0]
    assert "HUGGING_FACE_HUB_TOKEN" not in captured[0]
    assert captured[0]["HF_HUB_DISABLE_IMPLICIT_TOKEN"] == "1"
    run_as(
        BOB,
        download_lifecycle.spawn_worker,
        ["--repo-id", "org/private"],
        "bob-token",
        use_xet = False,
    )
    assert captured[1]["HF_TOKEN"] == "bob-token"
    assert "HUGGING_FACE_HUB_TOKEN" not in captured[1]


@pytest.mark.parametrize("authorized", [False, True])
def test_gated_metadata_alone_is_not_download_authorization(monkeypatch, authorized):
    calls = []

    def check(repo, **kwargs):
        calls.append((repo, kwargs))
        if not authorized:
            raise OSError("license not accepted")

    monkeypatch.setattr(
        access,
        "HfApi",
        lambda: SimpleNamespace(
            repo_info = lambda *a, **k: SimpleNamespace(gated = True), auth_check = check
        ),
    )
    if authorized:
        run_as(ALICE, access.authorize_download, "org/gated", "model", "alice-token")
    else:
        with pytest.raises(HTTPException):
            run_as(ALICE, access.authorize_download, "org/gated", "model", "alice-token")
    assert calls == [("org/gated", {"repo_type": "model", "token": "alice-token"})]
    assert run_as(ALICE, access.model_grants) == set()
