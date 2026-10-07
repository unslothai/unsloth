# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import asyncio
from types import SimpleNamespace

import pytest

from hub.services import snapshot_progress
from hub.services.models import downloads
from hub.utils import download_manifest


@pytest.fixture
def scoped_progress(monkeypatch, tmp_path):
    entry = tmp_path / "models--Org--Model"
    blobs = entry / "blobs"
    blobs.mkdir(parents = True)
    snapshot = entry / "snapshots" / "revision"
    snapshot.mkdir(parents = True)
    state = SimpleNamespace(manifest = None, job_state = "running")
    metadata = SimpleNamespace(
        scoped_files = ("new.gguf",),
        progress_blob_hashes = frozenset({"new"}),
        completed_baseline_bytes = 0,
        hub_cache = str(tmp_path),
        revision = None,
    )
    monkeypatch.setattr(downloads.account_access, "managed_account", lambda: False)
    monkeypatch.setattr(
        downloads,
        "_registry",
        SimpleNamespace(
            get_job = lambda _key: SimpleNamespace(state = state.job_state),
            get_job_metadata = lambda _key: metadata,
        ),
    )
    for module in (downloads, snapshot_progress):
        monkeypatch.setattr(module, "preferred_repo_cache_dirs", lambda *a, **k: [entry])
    monkeypatch.setattr(download_manifest, "read_manifest", lambda *a, **k: state.manifest)
    monkeypatch.setattr(download_manifest, "has_cancel_marker", lambda *a, **k: False)
    monkeypatch.setattr(downloads.gguf_variants, "gguf_variant_requirements", lambda *a, **k: None)
    monkeypatch.setattr(
        downloads.gguf_variants, "gguf_variant_blob_hashes", lambda *a, **k: frozenset()
    )

    def manifest(
        size,
        path = "new.gguf",
        sha = "new",
    ):
        state.manifest = download_manifest.Manifest(
            repo_type = "model",
            repo_id = "Org/Model",
            variant = "@diffusion",
            started_at = "",
            expected_files = (download_manifest.ExpectedFile(path = path, size = size, sha256 = sha),),
        )

    def poll(hint, variant = "@diffusion"):
        return asyncio.run(
            downloads.get_gguf_download_progress_response(
                "Org/Model",
                variant = variant,
                expected_bytes = hint,
            )
        )

    return SimpleNamespace(
        blobs = blobs, snapshot = snapshot, state = state, metadata = metadata, manifest = manifest, poll = poll
    )


@pytest.mark.parametrize("scope", ["@diffusion", "@hub-assets"])
@pytest.mark.parametrize("old_size,new_size", [(670, 627), (450, 627), (627, 627)])
def test_scope_file_change_rejects_old_manifest(scoped_progress, scope, old_size, new_size):
    case = scoped_progress
    case.manifest(old_size, "old.gguf", "old")
    (case.blobs / "old").write_bytes(b"o" * old_size)
    (case.snapshot / "old.gguf").write_bytes(b"o" * old_size)
    (case.blobs / "new.incomplete").write_bytes(b"n" * 20)
    first = case.poll(new_size, scope)
    assert first["expected_bytes"] == new_size
    assert first["downloaded_bytes"] == 20
    assert first["complete_on_disk"] is False
    case.manifest(new_size)
    second = case.poll(first["expected_bytes"], scope)
    assert second["expected_bytes"] == new_size
    assert second["downloaded_bytes"] == 20


@pytest.mark.parametrize("job_state", ["running", "cancelling", "idle"])
def test_scoped_metadata_corrects_stale_hint(scoped_progress, job_state):
    case = scoped_progress
    case.state.job_state = job_state
    case.manifest(627)
    (case.blobs / "new.incomplete").write_bytes(b"n" * 20)
    assert case.poll(670)["expected_bytes"] == 627


def test_scoped_retry_keeps_current_file_progress(scoped_progress):
    case = scoped_progress
    case.manifest(627)
    (case.blobs / "new.incomplete").write_bytes(b"n" * 20)
    result = case.poll(627)
    assert result["expected_bytes"] == 627
    assert result["downloaded_bytes"] == 20


def test_same_filename_new_revision_rejects_old_hash(scoped_progress):
    case = scoped_progress
    case.manifest(670, sha = "old")
    (case.snapshot / "new.gguf").write_bytes(b"o" * 670)
    (case.blobs / "old").write_bytes(b"o" * 670)
    (case.blobs / "new.incomplete").write_bytes(b"n" * 20)
    result = case.poll(627)
    assert result["expected_bytes"] == 627
    assert result["downloaded_bytes"] == 20
    assert result["complete_on_disk"] is False


def test_pinned_scope_rejects_a_manifest_for_another_revision(scoped_progress):
    case = scoped_progress
    case.metadata.revision = "a" * 40
    case.state.manifest = download_manifest.Manifest(
        repo_type = "model",
        repo_id = "Org/Model",
        variant = "@diffusion",
        started_at = "",
        expected_files = (download_manifest.ExpectedFile(path = "new.gguf", size = 627, sha256 = "new"),),
        commit_hash = "b" * 40,
        metadata_derived = True,
    )
    (case.snapshot / "new.gguf").write_bytes(b"n" * 627)
    (case.blobs / "new").write_bytes(b"n" * 627)

    result = case.poll(627)

    assert result["expected_bytes"] == 627
    assert result["complete_on_disk"] is False


def test_missing_manifest_uses_current_hashes_and_hint(scoped_progress):
    case = scoped_progress
    (case.blobs / "new.incomplete").write_bytes(b"n" * 20)
    (case.blobs / "old").write_bytes(b"o" * 670)
    result = case.poll(627)
    assert result["expected_bytes"] == 627
    assert result["downloaded_bytes"] == 20
    assert result["complete_on_disk"] is False


def test_unknown_hashes_do_not_count_old_scope_snapshot(scoped_progress):
    case = scoped_progress
    case.metadata.progress_blob_hashes = frozenset()
    case.manifest(670, "old.gguf", "old")
    (case.snapshot / "old.gguf").write_bytes(b"o" * 670)
    result = case.poll(627)
    assert result["expected_bytes"] == 627
    assert result["downloaded_bytes"] == 0
    assert result["complete_on_disk"] is False


def test_current_scope_can_complete_after_total_shrinks(scoped_progress):
    case = scoped_progress
    case.manifest(627)
    (case.blobs / "new").write_bytes(b"n" * 627)
    (case.snapshot / "new.gguf").write_bytes(b"n" * 627)
    result = case.poll(670)
    assert result["expected_bytes"] == 627
    assert result["downloaded_bytes"] == 627
    assert result["complete_on_disk"] is True
    assert result["progress"] == 1


def test_restored_scope_without_registry_metadata_uses_manifest(scoped_progress):
    case = scoped_progress
    case.metadata.scoped_files = ()
    case.metadata.progress_blob_hashes = frozenset()
    case.manifest(627)
    (case.blobs / "new.incomplete").write_bytes(b"n" * 20)
    result = case.poll(670)
    assert result["expected_bytes"] == 627
    assert result["downloaded_bytes"] == 20


def test_regular_gguf_total_includes_companions(monkeypatch, scoped_progress):
    case = scoped_progress
    case.metadata.scoped_files = ()
    monkeypatch.setattr(
        downloads.gguf_variants,
        "gguf_variant_requirements",
        lambda *a, **k: SimpleNamespace(
            download_size_bytes = 657,
            required_hashes = frozenset({"new", "companion"}),
            expected_files = (),
        ),
    )
    (case.blobs / "new.incomplete").write_bytes(b"n" * 20)
    (case.blobs / "companion").write_bytes(b"c" * 30)
    result = case.poll(700, "Q6_K")
    assert result["expected_bytes"] == 657
    assert result["downloaded_bytes"] == 50


def test_unresolved_variant_keeps_size_hint(scoped_progress):
    case = scoped_progress
    case.metadata.scoped_files = ()
    case.metadata.progress_blob_hashes = frozenset()
    result = case.poll(627, "Q6_K")
    assert result["expected_bytes"] == 627
    assert result["complete_on_disk"] is False


def test_full_snapshot_retains_conservative_total(scoped_progress):
    case = scoped_progress
    case.metadata.scoped_files = ()
    result = snapshot_progress.compute_snapshot_progress(
        repo_type = "model",
        repo_id = "Org/Model",
        job_key = "Org/Model",
        expected_bytes = 670,
        hf_token = None,
        registry = downloads._registry,
        metadata_resolver = lambda *a: (627, frozenset({"new"})),
    )
    assert result["expected_bytes"] == 670
    assert result["complete_on_disk"] is False


def test_offline_variant_manifest_does_not_shrink_hint(scoped_progress):
    case = scoped_progress
    case.metadata.scoped_files = ()
    case.metadata.progress_blob_hashes = frozenset()
    case.manifest(627, "model-Q6_K.gguf", "old")
    (case.blobs / "old").write_bytes(b"o" * 627)
    (case.snapshot / "model-Q6_K.gguf").write_bytes(b"o" * 627)
    result = case.poll(670, "Q6_K")
    assert result["expected_bytes"] == 670
    assert result["complete_on_disk"] is False
