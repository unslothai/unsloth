# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import os
import time
from pathlib import Path

import pytest

from hub.utils import hf_cache_state

latest_snapshot_from_cache_path = hf_cache_state.latest_snapshot_from_cache_path


@pytest.fixture(autouse = True)
def _known_cache_root(monkeypatch, tmp_path):
    monkeypatch.setattr(hf_cache_state, "hf_cache_roots", lambda: [tmp_path])


def _model_repo(root: Path, repo_id: str) -> Path:
    repo_root = root / f"models--{repo_id.replace('/', '--')}"
    (repo_root / "snapshots").mkdir(parents = True)
    return repo_root


def _snapshot(
    repo_root: Path,
    name: str,
    files: tuple[str, ...] = (),
) -> Path:
    snap = repo_root / "snapshots" / name
    snap.mkdir()
    for filename in files:
        (snap / filename).write_text("{}")
    return snap


def test_returns_newest_snapshot_with_metadata(tmp_path):
    repo_root = _model_repo(tmp_path, "Org/Model")
    old = _snapshot(repo_root, "old", ("config.json",))
    new = _snapshot(repo_root, "new", ("config.json",))
    past = time.time() - 3600
    os.utime(old, (past, past))

    resolved = latest_snapshot_from_cache_path(
        str(repo_root), "model", "Org/Model", ("config.json",)
    )
    assert resolved == str(new.resolve())


def test_requires_metadata_filenames_when_given(tmp_path):
    repo_root = _model_repo(tmp_path, "Org/Model")
    _snapshot(repo_root, "rev")

    assert (
        latest_snapshot_from_cache_path(str(repo_root), "model", "Org/Model", ("config.json",))
        is None
    )


def test_accepts_adapter_metadata(tmp_path):
    repo_root = _model_repo(tmp_path, "Org/Model")
    snap = _snapshot(repo_root, "rev", ("adapter_config.json",))

    resolved = latest_snapshot_from_cache_path(
        str(repo_root), "model", "Org/Model", ("config.json", "adapter_config.json")
    )
    assert resolved == str(snap.resolve())


def test_rejects_paths_outside_the_repo_cache_dir(tmp_path):
    foreign = tmp_path / "somewhere-else"
    foreign.mkdir()
    (foreign / "config.json").write_text("{}")

    assert (
        latest_snapshot_from_cache_path(str(foreign), "model", "Org/Model", ("config.json",))
        is None
    )


def test_rejects_mismatched_repo_id(tmp_path):
    repo_root = _model_repo(tmp_path, "Org/Model")
    _snapshot(repo_root, "rev", ("config.json",))

    assert (
        latest_snapshot_from_cache_path(str(repo_root), "model", "Other/Repo", ("config.json",))
        is None
    )


def test_accepts_snapshot_dir_directly(tmp_path):
    repo_root = _model_repo(tmp_path, "Org/Model")
    snap = _snapshot(repo_root, "rev", ("config.json",))

    resolved = latest_snapshot_from_cache_path(str(snap), "model", "Org/Model", ("config.json",))
    assert resolved == str(snap.resolve())


def test_none_inputs_return_none(tmp_path):
    assert latest_snapshot_from_cache_path(None, "model", "Org/Model") is None
    assert latest_snapshot_from_cache_path(str(tmp_path), "model", "") is None


def test_refs_main_preferred_over_newer_mtime(tmp_path):
    repo_root = _model_repo(tmp_path, "Org/Model")
    old = _snapshot(repo_root, "commit-old", ("config.json",))
    new = _snapshot(repo_root, "commit-new", ("config.json",))
    past = time.time() - 3600
    os.utime(old, (past, past))
    refs = repo_root / "refs"
    refs.mkdir()
    (refs / "main").write_text("commit-old")

    resolved = latest_snapshot_from_cache_path(
        str(repo_root), "model", "Org/Model", ("config.json",)
    )
    assert resolved == str(old.resolve())
    assert resolved != str(new.resolve())


def test_refs_main_skipped_without_metadata_or_missing_target(tmp_path):
    repo_root = _model_repo(tmp_path, "Org/Model")
    _snapshot(repo_root, "commit-pinned")
    fallback = _snapshot(repo_root, "commit-fallback", ("config.json",))
    refs = repo_root / "refs"
    refs.mkdir()
    (refs / "main").write_text("commit-pinned")

    resolved = latest_snapshot_from_cache_path(
        str(repo_root), "model", "Org/Model", ("config.json",)
    )
    assert resolved == str(fallback.resolve())


def test_rejects_lookalike_repo_outside_known_cache(monkeypatch, tmp_path):
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    repo_root = _model_repo(tmp_path / "outside", "Org/Model")
    _snapshot(repo_root, "rev", ("config.json",))
    monkeypatch.setattr(hf_cache_state, "hf_cache_roots", lambda: [allowed])

    assert (
        latest_snapshot_from_cache_path(str(repo_root), "model", "Org/Model", ("config.json",))
        is None
    )


def test_refs_main_cannot_escape_snapshots(tmp_path):
    repo_root = _model_repo(tmp_path, "Org/Model")
    fallback = _snapshot(repo_root, "rev", ("config.json",))
    escaped = tmp_path / "escaped"
    escaped.mkdir()
    (escaped / "config.json").write_text("{}")
    refs = repo_root / "refs"
    refs.mkdir()
    (refs / "main").write_text("../../escaped")

    resolved = latest_snapshot_from_cache_path(
        str(repo_root), "model", "Org/Model", ("config.json",)
    )

    assert resolved == str(fallback.resolve())

    (refs / "main").write_text("commit-missing")
    resolved = latest_snapshot_from_cache_path(
        str(repo_root), "model", "Org/Model", ("config.json",)
    )
    assert resolved == str(fallback.resolve())


def test_snapshot_symlink_cannot_escape_cache(tmp_path):
    repo_root = _model_repo(tmp_path, "Org/Model")
    fallback = _snapshot(repo_root, "rev", ("config.json",))
    escaped = tmp_path / "escaped-snapshot"
    escaped.mkdir()
    (escaped / "config.json").write_text("{}")
    link = repo_root / "snapshots" / "linked"
    link.symlink_to(escaped, target_is_directory = True)
    future = time.time() + 3600
    os.utime(escaped, (future, future))

    resolved = latest_snapshot_from_cache_path(
        str(repo_root), "model", "Org/Model", ("config.json",)
    )

    assert resolved == str(fallback.resolve())


def test_snapshots_directory_symlink_cannot_escape_cache(tmp_path):
    repo_root = _model_repo(tmp_path, "Org/Model")
    external = tmp_path / "external-snapshots"
    snapshot = external / "rev"
    snapshot.mkdir(parents = True)
    (snapshot / "config.json").write_text("{}")
    (repo_root / "snapshots").rmdir()
    (repo_root / "snapshots").symlink_to(external, target_is_directory = True)

    resolved = latest_snapshot_from_cache_path(
        str(repo_root), "model", "Org/Model", ("config.json",)
    )

    assert resolved is None


def test_ref_file_symlink_cannot_escape_cache(tmp_path):
    repo_root = _model_repo(tmp_path, "Org/Model")
    fallback = _snapshot(repo_root, "rev", ("config.json",))
    external_ref = tmp_path / "external-ref"
    external_ref.write_text("rev")
    refs = repo_root / "refs"
    refs.mkdir()
    (refs / "main").symlink_to(external_ref)

    resolved = latest_snapshot_from_cache_path(
        str(repo_root), "model", "Org/Model", ("config.json",)
    )

    assert resolved == str(fallback.resolve())


def test_refs_directory_symlink_cannot_escape_cache(tmp_path):
    repo_root = _model_repo(tmp_path, "Org/Model")
    fallback = _snapshot(repo_root, "rev", ("config.json",))
    external_refs = tmp_path / "external-refs"
    external_refs.mkdir()
    (external_refs / "main").write_text("rev")
    (repo_root / "refs").symlink_to(external_refs, target_is_directory = True)

    resolved = latest_snapshot_from_cache_path(
        str(repo_root), "model", "Org/Model", ("config.json",)
    )

    assert resolved == str(fallback.resolve())


def test_training_pin_prefers_a_snapshot_that_has_weights(tmp_path):
    # refs/main can point at a metadata-only revision while a complete snapshot sits
    # beside it. Pinning the metadata-only one fails the start with "no trainable weights".
    from core.training.training import _resolve_model_snapshot

    repo_root = _model_repo(tmp_path, "Org/Model")
    metadata_only = _snapshot(repo_root, "commit-metadata", ("config.json",))
    complete = _snapshot(repo_root, "commit-complete", ("config.json", "model.safetensors"))
    refs = repo_root / "refs"
    refs.mkdir()
    (refs / "main").write_text("commit-metadata")

    resolved = _resolve_model_snapshot("Org/Model", str(repo_root))

    assert resolved == str(complete.resolve())
    assert resolved != str(metadata_only.resolve())


def test_training_pin_still_falls_back_to_metadata_only_snapshots(tmp_path):
    # With no weights anywhere the pin is unchanged, so the worker's Hub retry still runs.
    from core.training.training import _resolve_model_snapshot

    repo_root = _model_repo(tmp_path, "Org/Model")
    metadata_only = _snapshot(repo_root, "commit-metadata", ("config.json",))

    assert _resolve_model_snapshot("Org/Model", str(repo_root)) == str(metadata_only.resolve())
