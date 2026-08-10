# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Focused reproduction for Hugging Face's process-unique partial filenames."""

from types import SimpleNamespace

from hub.services import snapshot_progress
from hub.utils import download_manifest, download_registry
from hub.utils.hf_cache_state import incomplete_blob_hash


_BLOB_HASH = "a" * 64


def _running_registry():
    return SimpleNamespace(
        get_job = lambda _key: SimpleNamespace(state = "running"),
        get_job_metadata = lambda _key: SimpleNamespace(completed_baseline_bytes = 0),
    )


def test_incomplete_blob_hash_supports_legacy_and_process_unique_names():
    assert incomplete_blob_hash(f"{_BLOB_HASH}.incomplete") == _BLOB_HASH
    assert incomplete_blob_hash(f"{_BLOB_HASH}.deadbeef.incomplete") == _BLOB_HASH
    assert incomplete_blob_hash(_BLOB_HASH) is None


def test_registry_groups_duplicate_process_unique_writers_by_blob(monkeypatch, tmp_path):
    """Parallel partial attempts are one logical blob, not additive progress."""
    entry = tmp_path / "models--Org--Model"
    blobs = entry / "blobs"
    blobs.mkdir(parents = True)
    (blobs / f"{_BLOB_HASH}.11111111.incomplete").write_bytes(b"x" * 3)
    (blobs / f"{_BLOB_HASH}.22222222.incomplete").write_bytes(b"x" * 5)

    monkeypatch.setattr(
        download_registry,
        "iter_repo_cache_dirs",
        lambda *_args, **_kwargs: [entry],
    )
    monkeypatch.setattr(
        download_registry,
        "iter_active_repo_cache_dirs",
        lambda *_args, **_kwargs: [entry],
    )

    assert download_registry.incomplete_blob_hashes("model", "Org/Model") == {_BLOB_HASH}
    assert download_registry.existing_blob_bytes(
        "model",
        "Org/Model",
        frozenset({_BLOB_HASH}),
    ) == 5


def test_registry_purges_process_unique_partial(tmp_path):
    entry = tmp_path / "models--Org--Model"
    blobs = entry / "blobs"
    blobs.mkdir(parents = True)
    partial = blobs / f"{_BLOB_HASH}.deadbeef.incomplete"
    partial.write_bytes(b"x" * 5)

    outcome = download_registry._purge_incomplete_blobs(
        entry,
        only_hashes = frozenset({_BLOB_HASH}),
    )

    assert outcome == (1, 0)
    assert not partial.exists()


def test_progress_counts_process_unique_incomplete_blob(monkeypatch, tmp_path):
    """An active ``<etag>.<uuid>.incomplete`` target must contribute bytes."""
    entry = tmp_path / "models--Org--Model-GGUF"
    blobs = entry / "blobs"
    blobs.mkdir(parents = True)
    (blobs / f"{_BLOB_HASH}.deadbeef.incomplete").write_bytes(b"x" * 5)

    monkeypatch.setattr(
        snapshot_progress,
        "preferred_repo_cache_dirs",
        lambda *_args, **_kwargs: [entry],
    )
    result = snapshot_progress.compute_snapshot_progress(
        repo_type = "model",
        repo_id = "Org/Model-GGUF",
        job_key = "model:org/model-gguf#q4_k_m",
        expected_bytes = 100,
        hf_token = None,
        registry = _running_registry(),
        metadata_resolver = lambda *_args: (100, frozenset({_BLOB_HASH})),
        variant = "Q4_K_M",
    )

    assert result["completed_bytes"] == 0
    assert result["downloaded_bytes"] == 5
    assert result["progress"] == 0.05


def test_progress_counts_completed_materialized_snapshot_file(monkeypatch, tmp_path):
    """A Windows copy-layout snapshot must count without a finalized blob file."""
    entry = tmp_path / "models--Org--Model"
    snapshot = entry / "snapshots" / "revision"
    (entry / "blobs").mkdir(parents = True)
    snapshot.mkdir(parents = True)
    (snapshot / "model.safetensors").write_bytes(b"x" * 5)
    manifest = download_manifest.Manifest(
        repo_type = "model",
        repo_id = "Org/Model",
        variant = "@diffusion",
        started_at = "",
        expected_files = (
            download_manifest.ExpectedFile(
                path = "model.safetensors",
                size = 5,
                sha256 = _BLOB_HASH,
            ),
        ),
    )

    monkeypatch.setattr(
        snapshot_progress,
        "preferred_repo_cache_dirs",
        lambda *_args, **_kwargs: [entry],
    )
    monkeypatch.setattr(
        snapshot_progress.download_manifest,
        "read_manifest",
        lambda *_args, **_kwargs: manifest,
    )

    result = snapshot_progress.compute_snapshot_progress(
        repo_type = "model",
        repo_id = "Org/Model",
        job_key = "model:org/model#@diffusion",
        expected_bytes = 100,
        hf_token = None,
        registry = _running_registry(),
        metadata_resolver = lambda *_args: (100, frozenset({_BLOB_HASH})),
        variant = "@diffusion",
        variant_file_matcher = lambda path, **_kwargs: path == "model.safetensors",
    )

    assert result["completed_bytes"] == 5
    assert result["downloaded_bytes"] == 5
    assert result["progress"] == 0.05
