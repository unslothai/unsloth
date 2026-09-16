# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A managed account's paths resolve inside its own roots, and monitor rows are visible only to the owner and the loading account."""

import json
import os
import sys

import pytest
from fastapi import HTTPException

from auth import policy
from core.inference import api_monitor
from hub.storage import scan_folders
from utils.account_context import OWNER, AccountContext, run_as
from utils.models import checkpoints, model_config
from utils.paths import storage_roots

ALICE = AccountContext("alice-id", "alice")
BOB = AccountContext("bob-id", "bob")


@pytest.fixture(autouse = True)
def studio(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "studio"))
    monkeypatch.setattr(policy, "installation_is_multi_user", lambda: True)
    return tmp_path / "studio"


def _adapter_run(root, name, base):
    run = root / name
    run.mkdir(parents = True, exist_ok = True)
    (run / "adapter_config.json").write_text(json.dumps({"base_model_name_or_path": base}))
    return run


@pytest.mark.skipif(sys.platform == "win32", reason = "symlinks need privileges on Windows")
def test_managed_dataset_path_does_not_follow_a_link_out_of_the_account():
    secret = run_as(ALICE, storage_roots.dataset_uploads_root) / "private.jsonl"
    secret.parent.mkdir(parents = True)
    secret.write_text('{"text": "ALICE-SECRET"}')
    bob_uploads = run_as(BOB, storage_roots.dataset_uploads_root)
    bob_uploads.mkdir(parents = True)
    (bob_uploads / "linked.jsonl").symlink_to(secret)
    for spelling in ("uploads/linked.jsonl", "linked.jsonl", str(bob_uploads / "linked.jsonl")):
        with pytest.raises(ValueError, match = "escapes the account workspace|under a dataset root"):
            run_as(BOB, storage_roots.resolve_dataset_path, spelling)
    own = bob_uploads / "mine.jsonl"
    own.write_text("{}")
    assert run_as(BOB, storage_roots.resolve_dataset_path, "uploads/mine.jsonl") == own


@pytest.mark.skipif(sys.platform == "win32", reason = "symlinks need privileges on Windows")
def test_owner_keeps_following_links_in_its_own_install(tmp_path):
    elsewhere = tmp_path / "other-drive" / "data.jsonl"
    elsewhere.parent.mkdir(parents = True)
    elsewhere.write_text("{}")
    uploads = run_as(OWNER, storage_roots.dataset_uploads_root)
    uploads.mkdir(parents = True)
    (uploads / "linked.jsonl").symlink_to(elsewhere)
    assert (
        run_as(OWNER, storage_roots.resolve_dataset_path, "uploads/linked.jsonl")
        == uploads / "linked.jsonl"
    )


@pytest.mark.skipif(sys.platform == "win32", reason = "symlinks need privileges on Windows")
def test_scanners_skip_a_linked_run_that_belongs_to_another_account():
    alice_run = _adapter_run(
        run_as(ALICE, storage_roots.outputs_root), "alice-run", "ALICE-PRIVATE-BASE"
    )
    alice_export = run_as(ALICE, storage_roots.exports_root) / "alice-export" / "checkpoint-1"
    _adapter_run(alice_export.parent, "checkpoint-1", "ALICE-PRIVATE-BASE")
    bob_outputs = run_as(BOB, storage_roots.outputs_root)
    bob_outputs.mkdir(parents = True)
    _adapter_run(bob_outputs, "bob-run", "BOB-BASE")
    (bob_outputs / "linked-run").symlink_to(alice_run, target_is_directory = True)
    bob_exports = run_as(BOB, storage_roots.exports_root)
    bob_exports.mkdir(parents = True)
    (bob_exports / "linked-export").symlink_to(alice_export.parent, target_is_directory = True)

    names = [name for name, _cps, _meta in run_as(BOB, checkpoints.scan_checkpoints)]
    assert names == ["bob-run"]
    trained = [name for name, _path, _kind in run_as(BOB, model_config.scan_trained_models)]
    assert trained == ["bob-run"]
    exported = run_as(BOB, model_config.scan_exported_models)
    assert all("ALICE" not in json.dumps(row) for row in exported)
    owner_outputs = run_as(OWNER, storage_roots.outputs_root)
    owner_outputs.mkdir(parents = True, exist_ok = True)
    (owner_outputs / "linked-run").symlink_to(alice_run, target_is_directory = True)
    assert "linked-run" in [name for name, _c, _m in run_as(OWNER, checkpoints.scan_checkpoints)]


def test_managed_export_write_dir_stays_inside_its_roots():
    foreign = run_as(ALICE, storage_roots.exports_root) / "victim"
    foreign.mkdir(parents = True)
    with pytest.raises(ValueError, match = "escapes the account workspace"):
        run_as(BOB, storage_roots.resolve_export_write_dir, str(foreign))
    own = run_as(BOB, storage_roots.exports_root) / "mine"
    assert run_as(BOB, storage_roots.resolve_export_write_dir, str(own)) == own
    assert run_as(OWNER, storage_roots.resolve_export_write_dir, str(foreign)) == foreign


@pytest.mark.parametrize("module_name", ["hub.storage.scan_folders", "storage.studio_db"])
def test_scan_folder_storage_refuses_a_foreign_directory(monkeypatch, module_name):
    import importlib

    module = importlib.import_module(module_name)
    foreign = run_as(ALICE, storage_roots.outputs_root)
    foreign.mkdir(parents = True, exist_ok = True)
    own = run_as(BOB, storage_roots.outputs_root)
    own.mkdir(parents = True, exist_ok = True)
    assert run_as(OWNER, module.add_scan_folder_with_status, str(foreign))[1]
    assert run_as(BOB, module.add_scan_folder_with_status, str(own))[1]
    monkeypatch.setattr(
        module,
        "get_connection",
        lambda: (_ for _ in ()).throw(AssertionError("opened the database")),
    )
    with pytest.raises(ValueError, match = "outside this account's workspace"):
        run_as(BOB, module.add_scan_folder_with_status, str(foreign))


def test_image_training_dataset_name_resolves_before_the_account_check():
    from routes import training

    for account in (OWNER, ALICE):
        root = run_as(account, storage_roots.datasets_root)
        (root / "my-images").mkdir(parents = True, exist_ok = True)
        assert run_as(account, training._resolve_diffusion_data_dir, "my-images") == (
            root / "my-images"
        )
    foreign = run_as(ALICE, storage_roots.datasets_root) / "my-images"
    with pytest.raises((HTTPException, ValueError)):
        run_as(BOB, training._resolve_diffusion_data_dir, str(foreign))


def test_monitor_hides_a_foreign_load_row_from_managed_accounts():
    monitor = (
        api_monitor.ApiMonitor() if hasattr(api_monitor, "ApiMonitor") else api_monitor.api_monitor
    )
    entry = run_as(
        ALICE,
        api_monitor.ApiMonitorEntry,
        id = "load-1",
        endpoint = "",
        method = "",
        model = "/private/alice/outputs/SECRET.gguf",
        prompt = "",
        status = "completed",
        started_at = 0.0,
        updated_at = 0.0,
        started_monotonic = 0.0,
        finished_at = 0.0,
        finished_monotonic = 0.0,
        kind = "lifecycle",
        event = "load",
        reason = "",
        shared = True,
        subject = "alice",
    )
    assert run_as(ALICE, monitor._visible, entry, "alice")
    assert run_as(OWNER, monitor._visible, entry, "unsloth")
    assert not run_as(BOB, monitor._visible, entry, "bob")
    monkeypatch_single = policy.installation_is_multi_user
    policy.installation_is_multi_user = lambda: False
    try:
        assert run_as(OWNER, monitor._visible, entry, "unsloth")
        assert not run_as(BOB, monitor._visible, entry, "bob")
    finally:
        policy.installation_is_multi_user = monkeypatch_single


def test_a_recreated_username_cannot_read_the_previous_accounts_monitor_rows():
    old_alice = AccountContext("alice-first-id", "alice")
    new_alice = AccountContext("alice-second-id", "alice")
    monitor = api_monitor.ApiMonitor()

    request_id = run_as(
        old_alice,
        lambda: monitor.start(
            endpoint = "/v1/chat/completions",
            method = "POST",
            model = "org/Repo",
            prompt = "ALICE-PRIVATE-PROMPT",
            subject = "alice",
        ),
    )
    run_as(
        old_alice,
        monitor.record_lifecycle,
        event = "load",
        model = "/alice/secret.gguf",
        subject = "alice",
    )

    assert run_as(new_alice, monitor.get, request_id, subject = "alice") is None
    assert run_as(new_alice, monitor.snapshot, subject = "alice") == []
    assert run_as(new_alice, monitor.active_count, subject = "alice") == 0
    run_as(new_alice, monitor.clear, subject = "alice")
    assert len(monitor._entries) == 2

    assert run_as(old_alice, monitor.get, request_id, subject = "alice") is not None
    assert run_as(old_alice, monitor.active_count, subject = "alice") == 1
    assert len(run_as(old_alice, monitor.snapshot, subject = "alice")) == 2
    run_as(old_alice, monitor.clear, subject = "alice")
    assert run_as(old_alice, monitor.get, request_id, subject = "alice") is not None


def test_unload_rows_without_a_subject_stay_inside_the_account_that_recorded_them():
    monitor = api_monitor.ApiMonitor()
    run_as(ALICE, monitor.record_lifecycle, event = "unload", model = "acme/private", reason = "manual")
    run_as(ALICE, monitor.record_lifecycle, event = "unload", model = "alice-run-merged", reason = "idle")
    assert run_as(BOB, monitor.snapshot, subject = "bob") == []
    assert len(run_as(ALICE, monitor.snapshot, subject = "alice")) == 2
    assert len(run_as(OWNER, monitor.snapshot, subject = "unsloth")) == 2


def test_one_account_installs_keep_every_monitor_row(monkeypatch):
    monkeypatch.setattr(policy, "installation_is_multi_user", lambda: False)
    monitor = api_monitor.ApiMonitor()
    request_id = monitor.start(
        endpoint = "/v1/chat/completions",
        method = "POST",
        model = "org/Repo",
        prompt = "hello",
        subject = "unsloth",
    )
    monitor.finish(request_id)
    assert monitor.get(request_id, subject = "unsloth") is not None
    assert len(monitor.snapshot(subject = "unsloth")) == 1
    monitor.clear(subject = "unsloth")
    assert monitor.snapshot(subject = "unsloth") == []


@pytest.mark.skipif(sys.platform == "win32", reason = "symlinks need privileges on Windows")
def test_a_managed_account_directory_replaced_by_a_link_is_refused(tmp_path):
    from core.inference import audio_gallery, image_gallery, video_gallery

    alice_images = run_as(ALICE, image_gallery.gallery_dir)
    (alice_images / "secret.png").write_bytes(b"ALICE-PNG")
    bob_root = run_as(BOB, storage_roots.workspace_root)
    bob_root.mkdir(parents = True, exist_ok = True)
    (bob_root / "images").symlink_to(alice_images, target_is_directory = True)
    (bob_root / "outputs").symlink_to(
        run_as(ALICE, storage_roots.outputs_root), target_is_directory = True
    )
    (bob_root / "studio.db").symlink_to(run_as(ALICE, storage_roots.studio_db_path))
    with pytest.raises(ValueError, match = "escapes the account workspace"):
        run_as(BOB, image_gallery.gallery_dir)
    with pytest.raises(ValueError, match = "escapes the account workspace"):
        run_as(BOB, image_gallery.list_images)
    with pytest.raises(ValueError, match = "escapes the account workspace"):
        run_as(BOB, storage_roots.outputs_root)
    with pytest.raises(ValueError, match = "escapes the account workspace"):
        run_as(BOB, storage_roots.studio_db_path)
    assert run_as(BOB, audio_gallery.gallery_dir) == bob_root / "audio"
    assert run_as(BOB, video_gallery.gallery_dir) == bob_root / "videos"
    assert run_as(BOB, storage_roots.exports_root) == bob_root / "exports"
    owner_root = run_as(OWNER, storage_roots.workspace_root)
    owner_root.mkdir(parents = True, exist_ok = True)
    (owner_root / "images").symlink_to(tmp_path / "other-drive-images", target_is_directory = True)
    (tmp_path / "other-drive-images").mkdir()
    assert run_as(OWNER, image_gallery.gallery_dir) == owner_root / "images"


def test_tool_stream_worker_runs_as_the_calling_account():
    from core.inference.tool_stream_exec import stream_tool_execution
    from utils.account_context import current_account

    def consume(fn):
        gen = stream_tool_execution(lambda callback: fn(), tool_name = "python")
        try:
            while True:
                next(gen)
        except StopIteration as stop:
            return stop.value

    assert run_as(BOB, consume, lambda: current_account().account_id) == "bob-id"
    assert run_as(BOB, consume, lambda: str(storage_roots.studio_db_path())) == str(
        run_as(BOB, storage_roots.studio_db_path)
    )
    assert run_as(OWNER, consume, lambda: current_account().account_id) == OWNER.account_id


def test_download_watcher_is_pinned_to_the_requesting_account(monkeypatch):
    from hub.services import download_lifecycle
    from utils import account_context

    captured = {}

    class _Thread:
        def __init__(
            self,
            *,
            target,
            args = (),
            kwargs = None,
            **_kw,
        ):
            captured["target"], captured["args"] = target, args

        def start(self):
            pass

    monkeypatch.setattr(account_context.threading, "Thread", _Thread)
    monkeypatch.setattr(policy, "installation_is_multi_user", lambda: False)
    run_as(BOB, download_lifecycle.account_thread, target = lambda: None, name = "watch", daemon = True)
    assert captured["target"] is account_context.run_as
    assert captured["args"][0] == BOB


def test_external_provider_cancel_keys_are_scoped_like_every_lookup():
    import inspect

    import routes.inference as inference

    source = inspect.getsource(inference._proxy_to_external_provider)
    registration = source.index("cancel_keys = tuple(")
    assert "_account_cancel_key(key)" in source[registration : registration + 200]
    assert "_PENDING_CANCELS.pop(_account_cancel_key(payload.cancel_id), None)" in source


def test_export_size_cache_is_keyed_per_managed_account(monkeypatch):
    import routes.models as models
    from hub.services.models import account_access

    monkeypatch.setattr(models, "_EXPORT_SIZE_CACHE", {})
    monkeypatch.setattr(account_access, "managed_account", lambda: True)
    monkeypatch.setattr(models, "is_local_path", lambda model: False)
    import utils.hardware.hardware as hardware

    monkeypatch.setattr(
        hardware,
        "estimate_fp16_model_size_bytes",
        lambda model, hf_token = None: (1000, "ALICE-SOURCE"),
    )
    assert run_as(ALICE, models._export_size_cached, "my-finetune", None) == (
        1000,
        500,
        "ALICE-SOURCE",
    )
    monkeypatch.setattr(
        hardware,
        "estimate_fp16_model_size_bytes",
        lambda model, hf_token = None: (2000, "BOB-SOURCE"),
    )
    assert run_as(BOB, models._export_size_cached, "my-finetune", None) == (
        2000,
        1000,
        "BOB-SOURCE",
    )
    assert run_as(ALICE, models._export_size_cached, "my-finetune", None) == (
        1000,
        500,
        "ALICE-SOURCE",
    )


def test_a_managed_account_may_export_into_its_own_project_workspace(tmp_path, monkeypatch):
    from core.training import account_jobs

    monkeypatch.setenv("UNSLOTH_STUDIO_PROJECTS_HOME", str(tmp_path / "documents"))
    own = run_as(BOB, storage_roots.project_workspaces_root) / "demo" / "exports"
    own.mkdir(parents = True)
    run_as(BOB, account_jobs.validate_job_paths, {"save_directory": str(own)})
    assert run_as(BOB, storage_roots.resolve_export_write_dir, str(own)) == own
    assert run_as(BOB, scan_folders.add_scan_folder_with_status, str(own))[1]
    foreign = run_as(ALICE, storage_roots.project_workspaces_root) / "alice-demo"
    foreign.mkdir(parents = True)
    with pytest.raises(ValueError, match = "escapes the account workspace"):
        run_as(BOB, storage_roots.resolve_export_write_dir, str(foreign))
    with pytest.raises(ValueError, match = "outside this account"):
        run_as(BOB, scan_folders.add_scan_folder_with_status, str(foreign))
