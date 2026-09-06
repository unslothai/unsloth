# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A managed account's paths resolve inside its own roots even through a link,
its storage helpers refuse foreign absolute paths, and the API monitor shows a
model load row only to the owner and to the account that loaded it."""

import json
import os
import sys

import pytest

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
    # The owner's scanners are unchanged: a link to another drive is still a run.
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


def test_scan_folder_storage_refuses_a_foreign_directory(monkeypatch):
    foreign = run_as(ALICE, storage_roots.outputs_root)
    foreign.mkdir(parents = True)
    monkeypatch.setattr(
        scan_folders,
        "get_connection",
        lambda: (_ for _ in ()).throw(AssertionError("opened the database")),
    )
    with pytest.raises(ValueError, match = "outside this account's workspace"):
        run_as(BOB, scan_folders.add_scan_folder_with_status, str(foreign))


def test_monitor_hides_a_foreign_load_row_from_managed_accounts():
    monitor = (
        api_monitor.ApiMonitor() if hasattr(api_monitor, "ApiMonitor") else api_monitor.api_monitor
    )
    entry = api_monitor.ApiMonitorEntry(
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
    # Single-account installs are unchanged.
    monkeypatch_single = policy.installation_is_multi_user
    policy.installation_is_multi_user = lambda: False
    try:
        assert run_as(BOB, monitor._visible, entry, "bob")
    finally:
        policy.installation_is_multi_user = monkeypatch_single


@pytest.mark.skipif(sys.platform == "win32", reason = "symlinks need privileges on Windows")
def test_a_managed_account_directory_replaced_by_a_link_is_refused(tmp_path):
    from core.inference import audio_gallery, image_gallery, video_gallery

    alice_images = run_as(ALICE, image_gallery.gallery_dir)
    (alice_images / "secret.png").write_bytes(b"ALICE-PNG")
    bob_root = run_as(BOB, storage_roots.workspace_root)
    bob_root.mkdir(parents = True, exist_ok = True)
    (bob_root / "images").symlink_to(alice_images, target_is_directory = True)
    (bob_root / "outputs").symlink_to(run_as(ALICE, storage_roots.outputs_root), target_is_directory = True)
    (bob_root / "studio.db").symlink_to(run_as(ALICE, storage_roots.studio_db_path))
    with pytest.raises(ValueError, match = "escapes the account workspace"):
        run_as(BOB, image_gallery.gallery_dir)
    with pytest.raises(ValueError, match = "escapes the account workspace"):
        run_as(BOB, image_gallery.list_images)
    with pytest.raises(ValueError, match = "escapes the account workspace"):
        run_as(BOB, storage_roots.outputs_root)
    with pytest.raises(ValueError, match = "escapes the account workspace"):
        run_as(BOB, storage_roots.studio_db_path)
    # Untouched entries of the same account still resolve.
    assert run_as(BOB, audio_gallery.gallery_dir) == bob_root / "audio"
    assert run_as(BOB, video_gallery.gallery_dir) == bob_root / "videos"
    assert run_as(BOB, storage_roots.exports_root) == bob_root / "exports"
    # The owner's install may point its directories anywhere, as before.
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
    """The account travels with the watcher thread on every install, not only
    when the multi-user policy happens to be readable at that instant."""
    from hub.services import download_lifecycle
    from utils import account_context

    captured = {}

    class _Thread:
        def __init__(self, *, target, args = (), kwargs = None, **_kw):
            captured["target"], captured["args"] = target, args

        def start(self):
            pass

    monkeypatch.setattr(account_context.threading, "Thread", _Thread)
    monkeypatch.setattr(policy, "installation_is_multi_user", lambda: False)
    run_as(BOB, download_lifecycle.account_thread, target = lambda: None, name = "watch", daemon = True)
    assert captured["target"] is account_context.run_as
    assert captured["args"][0] == BOB
