# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Private shared-cache content needs anonymous Hub proof or a durable account grant."""

from __future__ import annotations

import asyncio
import io
import json
import logging
import sqlite3
import threading
import time
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


def test_concurrent_misses_ask_the_hub_once_per_repo(monkeypatch):
    calls = []
    barrier = threading.Barrier(8)

    def answer(repo_id, repo_type):
        calls.append(repo_id)
        barrier.wait(timeout = 30)
        time.sleep(0.02)
        return True

    monkeypatch.setattr(access, "_hub_public_answer", answer)
    rows = [{"repo_id": f"org/repo-{index}"} for index in range(8)]
    with ThreadPoolExecutor(max_workers = 8) as pool:
        listings = [
            pool.submit(run_as, ALICE, access.filter_model_rows, list(rows)) for _ in range(8)
        ]
        results = [listing.result(timeout = 60) for listing in listings]
    assert all(len(result) == 8 for result in results)
    assert sorted(calls) == sorted(row["repo_id"] for row in rows)


def test_a_listing_probes_its_unknown_repos_together(monkeypatch):
    monkeypatch.setattr(access, "_hub_public_answer", lambda *a: time.sleep(0.1) or True)
    rows = [{"repo_id": f"org/slow-{index}"} for index in range(8)]
    start = time.perf_counter()
    assert len(run_as(ALICE, access.filter_model_rows, rows)) == 8
    assert time.perf_counter() - start < 0.4


def test_a_warm_listing_asks_the_hub_nothing(monkeypatch):
    rows = [{"repo_id": f"org/warm-{index}"} for index in range(4)]
    monkeypatch.setattr(access, "_hub_public_answer", lambda *a: True)
    assert len(run_as(ALICE, access.filter_model_rows, rows)) == 4

    def unexpected(*args):
        raise AssertionError("a cached verdict must not be probed again")

    monkeypatch.setattr(access, "_hub_public_answer", unexpected)
    assert len(run_as(ALICE, access.filter_model_rows, rows)) == 4


def test_shared_dataset_catalog_is_filtered_for_each_account(monkeypatch):
    from hub.services.datasets import cache_inventory as dataset_inventory

    rows = [{"repo_id": "org/secret"}, {"repo_id": "org/public"}]
    run_as(ALICE, access.record_model_grant, "org/secret", "dataset")
    monkeypatch.setattr(access, "repo_is_public", lambda repo, *a: repo == "org/public")
    monkeypatch.setattr(dataset_inventory, "_scan_hf_dataset_caches", lambda: rows)

    fn = dataset_inventory.list_cached_datasets_response
    assert len(asyncio.run(arun_as(ALICE, fn()))["cached"]) == 2
    assert asyncio.run(arun_as(BOB, fn()))["cached"] == [rows[1]]
    assert len(asyncio.run(arun_as(OWNER, fn()))["cached"]) == 2
    assert len(rows) == 2


def test_a_blank_managed_token_is_anonymous_not_the_installations(monkeypatch):
    assert run_as(ALICE, access.account_hf_token, "   ") is False
    assert run_as(ALICE, access.account_hf_token, "") is False
    assert run_as(ALICE, access.account_hf_token, None) is False
    assert run_as(ALICE, access.account_hf_token, " hf_real ") == " hf_real "
    monkeypatch.setattr(policy, "installation_is_multi_user", lambda: False)
    assert access.account_hf_token("   ") == "   "


def test_the_shared_chat_template_read_needs_model_access(monkeypatch):
    from picker.routes import templates

    monkeypatch.setattr(access, "repo_is_public", lambda *a, **k: False)
    monkeypatch.setattr(templates, "read_default_chat_template", lambda *a, **k: "{{ raw }}")
    run_as(ALICE, access.record_model_grant, "org/secret")

    async def read(account):
        return await arun_as(
            account,
            templates.get_default_chat_template_route(
                model_name = "org/secret",
                gguf_variant = None,
                hf_token = None,
                current_subject = account.username,
            ),
        )

    assert asyncio.run(read(OWNER)).chat_template == "{{ raw }}"
    assert asyncio.run(read(ALICE)).chat_template == "{{ raw }}"
    with pytest.raises(HTTPException) as refused:
        asyncio.run(read(BOB))
    assert refused.value.status_code == 404


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


def test_download_ownership_replaces_the_previous_downloader_of_the_same_key():
    registry = download_registry.DownloadRegistry()
    key = "org/secret::"
    download_lifecycle._job_accounts[(id(registry), key)] = BOB.account_id
    run_as(ALICE, download_lifecycle.record_download_account, registry, key)
    assert run_as(ALICE, download_lifecycle.download_belongs_to_account, registry, key)
    assert not run_as(BOB, download_lifecycle.download_belongs_to_account, registry, key)


def test_download_ownership_lands_before_the_hub_authorization(monkeypatch):
    registry = download_registry.DownloadRegistry()
    key = "org/secret::"
    download_lifecycle._job_accounts[(id(registry), key)] = BOB.account_id
    registry.claim(key, "http", repo_type = "model", repo_id = "org/secret")
    seen = {}

    def repo_info(*args, **kwargs):
        seen["bob"] = run_as(BOB, download_lifecycle.download_belongs_to_account, registry, key)
        seen["alice"] = run_as(ALICE, download_lifecycle.download_belongs_to_account, registry, key)
        return SimpleNamespace(gated = False)

    monkeypatch.setattr(access, "HfApi", lambda: SimpleNamespace(repo_info = repo_info))

    def spawn():
        raise OSError("no worker in this test")

    with pytest.raises(HTTPException):
        run_as(
            ALICE,
            lambda: download_lifecycle.launch_worker(
                registry,
                key,
                spawn = spawn,
                hf_token = "alice-token",
                label = "org/secret",
                log_prefix = "Download",
                logger = logging.getLogger(__name__),
                repo_type = "model",
                repo_id = "org/secret",
                transport = "http",
                watch_name = "watch",
            ),
        )
    assert seen == {"bob": False, "alice": True}


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


def test_a_persisted_public_proof_expires_rather_than_outliving_a_privacy_change(monkeypatch):
    monkeypatch.setattr(
        access,
        "HfApi",
        lambda: SimpleNamespace(
            repo_info = lambda *a, **k: SimpleNamespace(private = False, gated = False)
        ),
    )
    assert run_as(ALICE, access.repo_visible, "Org/Public")
    path = access._public_verdicts_path()
    assert json.loads(path.read_text()).keys() == {"model:org/public"}

    monkeypatch.setattr(
        access,
        "HfApi",
        lambda: SimpleNamespace(
            repo_info = lambda *a, **k: (_ for _ in ()).throw(OSError("Hub unavailable"))
        ),
    )
    access._public_repos.clear()
    path.write_text(json.dumps({"model:org/public": time.time() - 3600}))
    assert run_as(BOB, access.repo_visible, "org/public"), "a proof inside the bound still carries"

    access._public_repos.clear()
    path.write_text(json.dumps({"model:org/public": time.time() - 30 * 24 * 3600}))
    assert not run_as(BOB, access.repo_visible, "org/public")
    assert not run_as(ALICE, access.repo_visible, "org/public")


def test_a_checkpoint_in_the_accounts_projects_tree_is_loadable_by_that_account(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("UNSLOTH_STUDIO_PROJECTS_HOME", str(tmp_path / "projects"))
    from core.training.account_jobs import validate_job_paths
    from utils.paths.storage_roots import project_workspaces_root

    checkpoint = run_as(ALICE, project_workspaces_root) / "demo" / "outputs" / "checkpoint-10"
    checkpoint.mkdir(parents = True)
    run_as(ALICE, validate_job_paths, {"output_dir": str(checkpoint)})
    assert run_as(ALICE, access.model_visible, str(checkpoint))
    assert run_as(ALICE, access.require_model_access, str(checkpoint)) is None
    assert not run_as(BOB, access.model_visible, str(checkpoint))
    with pytest.raises(HTTPException) as refused:
        run_as(BOB, access.require_model_access, str(checkpoint))
    assert refused.value.status_code == 404
    assert run_as(OWNER, project_workspaces_root) != run_as(ALICE, project_workspaces_root)


@pytest.mark.parametrize("repo_id,folder", [("gpt2", "models--gpt2"), ("org/m", "models--org--m")])
def test_one_segment_hub_ids_follow_their_grant(monkeypatch, tmp_path, repo_id, folder):
    """The download validator accepts ``repo_name`` alone, so visibility must too."""
    from utils import hf_cache_settings

    cache = tmp_path / "cache"
    monkeypatch.setattr(hf_cache_settings, "known_hf_hub_caches", lambda: [cache])
    monkeypatch.setattr(access, "repo_is_public", lambda *a: False)
    weights = cache / folder / "snapshots" / "commit" / "model.safetensors"
    weights.parent.mkdir(parents = True)
    weights.write_bytes(b"weights")
    assert not run_as(BOB, access.model_visible, repo_id)
    assert not run_as(BOB, access.model_visible, str(weights))
    run_as(ALICE, access.record_model_grant, repo_id)
    assert run_as(ALICE, access.model_visible, repo_id)
    assert run_as(ALICE, access.model_visible, str(weights))
    assert not run_as(BOB, access.model_visible, repo_id)
    assert not run_as(BOB, access.model_visible, str(weights))


def test_cache_path_rows_join_the_concurrent_hub_warmup():
    """Inventory rows carry snapshot paths; skipping them serialised one Hub call per repo."""
    path = "/cache/models--org--m/snapshots/abc/model.gguf"
    assert access._hub_probe_targets([path], "model", set()) == {"org/m"}
    assert (
        access._hub_probe_targets([path], "model", {access._grant_key("org/m", "model")}) == set()
    )
    assert access._hub_probe_targets([path], "dataset", set()) == set()


def test_a_grant_racing_retirement_does_not_recreate_the_workspace(monkeypatch, tmp_path):
    from core.training import account_jobs
    from utils.paths import storage_roots

    root = run_as(ALICE, storage_roots.workspace_root)
    root.mkdir(parents = True)
    real = account_jobs.account_is_retired
    fired = []

    def retire_after_check():
        # The retirement lands between the tombstone check and the directory creation.
        if not fired:
            fired.append(True)
            account_jobs._retired.add(ALICE.account_id)
            root.rename(root.with_name(root.name + "-deleted-1"))
            return False
        return real()

    monkeypatch.setattr(account_jobs, "account_is_retired", retire_after_check)
    try:
        run_as(ALICE, access.record_model_grant, "org/secret")
    finally:
        account_jobs._retired.discard(ALICE.account_id)
    assert not root.exists()
