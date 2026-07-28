# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A dangling ``refs/<branch>`` must not hide an intact repo from the scan.

``scan_cache_dir`` raises CorruptedCacheException for a repo whose ref names a
commit with no ``snapshots/<commit>/`` directory and omits it from ``.repos``,
so the model stays visible in the model picker (a plain directory walk) while
disappearing from every Hub inventory endpoint that feeds chat auto-load.
"""

from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

from hub.utils import inventory_scan

SNAPSHOT = "a" * 40
UPSTREAM_HEAD = "b" * 40


def _build_repo(
    cache_root: Path,
    *,
    ref: str,
    extra_refs: dict[str, str] | None = None,
    incomplete: bool = False,
    name: str = "models--Org--Model",
) -> Path:
    """A cache repo holding one snapshot at ``SNAPSHOT``, shaped like HF's."""
    repo_dir = cache_root / name
    blobs = repo_dir / "blobs"
    snapshot = repo_dir / "snapshots" / SNAPSHOT
    refs = repo_dir / "refs"
    for directory in (blobs, snapshot, refs):
        directory.mkdir(parents = True, exist_ok = True)
    blob = blobs / ("c" * 40)
    blob.write_bytes(b"\0")
    os.symlink(os.path.relpath(blob, snapshot), snapshot / "Q4_K_M.gguf")
    (refs / "main").write_text(ref, encoding = "utf-8")
    for name, commit in (extra_refs or {}).items():
        (refs / name).write_text(commit, encoding = "utf-8")
    if incomplete:
        (blobs / "d0d0d0d0.incomplete").write_bytes(b"partial")
    return repo_dir


def _ref_names(repo_dir: Path) -> list[str]:
    return sorted(entry.name for entry in (repo_dir / "refs").rglob("*") if entry.is_file())


def _scanned_repo_ids(cache_root: Path, monkeypatch) -> list[str]:
    monkeypatch.setattr(inventory_scan, "hf_cache_roots", lambda: [cache_root])
    return [
        repo.repo_id for scan in inventory_scan._compute_all_hf_cache_scans() for repo in scan.repos
    ]


def test_dangling_ref_no_longer_hides_an_intact_repo(tmp_path, monkeypatch):
    repo_dir = _build_repo(tmp_path, ref = UPSTREAM_HEAD)

    assert _scanned_repo_ids(tmp_path, monkeypatch) == ["Org/Model"]
    assert _ref_names(repo_dir) == []


def test_healthy_cache_is_neither_pruned_nor_rescanned(tmp_path, monkeypatch):
    import huggingface_hub

    repo_dir = _build_repo(tmp_path, ref = SNAPSHOT, extra_refs = {"v1.0": SNAPSHOT})
    calls: list[str] = []
    real_scan = huggingface_hub.scan_cache_dir

    def counting_scan(cache_dir = None):
        calls.append(str(cache_dir))
        return real_scan(cache_dir = cache_dir)

    monkeypatch.setattr(huggingface_hub, "scan_cache_dir", counting_scan)

    assert _scanned_repo_ids(tmp_path, monkeypatch) == ["Org/Model"]
    assert len(calls) == 1
    assert _ref_names(repo_dir) == ["main", "v1.0"]


def test_in_flight_download_keeps_its_dangling_ref(tmp_path, monkeypatch):
    """A download writes its ref before the snapshot lands, so it owns it."""
    repo_dir = _build_repo(tmp_path, ref = UPSTREAM_HEAD, incomplete = True)

    assert _scanned_repo_ids(tmp_path, monkeypatch) == []
    assert _ref_names(repo_dir) == ["main"]


def test_only_the_dangling_ref_of_a_mixed_repo_is_pruned(tmp_path, monkeypatch):
    repo_dir = _build_repo(tmp_path, ref = SNAPSHOT, extra_refs = {"stale": UPSTREAM_HEAD})

    assert _scanned_repo_ids(tmp_path, monkeypatch) == ["Org/Model"]
    assert _ref_names(repo_dir) == ["main"]


def test_an_unreadable_repo_does_not_abort_the_prune_sweep(tmp_path):
    """is_dir() propagates a permission error instead of returning False, so an
    unguarded probe would escape to the caller's whole-scan except and drop
    every model in the cache. The sweep must skip that repo and carry on.

    Scoped to the sweep because scan_cache_dir itself already raises on an
    unreadable repo dir, which is upstream of this code and unchanged here.
    """
    dangling = _build_repo(tmp_path, ref = UPSTREAM_HEAD)
    locked = _build_repo(tmp_path, ref = SNAPSHOT, name = "models--Org--Locked")
    locked.chmod(0)
    if os.access(locked / "refs", os.R_OK):
        pytest.skip("filesystem does not enforce directory permissions")
    try:
        assert inventory_scan._prune_dangling_hf_cache_refs(tmp_path) == 1
        assert _ref_names(dangling) == []
    finally:
        locked.chmod(stat.S_IRWXU)


def test_unwritable_refs_dir_degrades_instead_of_raising(tmp_path, monkeypatch):
    repo_dir = _build_repo(tmp_path, ref = UPSTREAM_HEAD)
    refs_dir = repo_dir / "refs"
    refs_dir.chmod(stat.S_IRUSR | stat.S_IXUSR)
    if os.access(refs_dir, os.W_OK):
        pytest.skip("filesystem does not enforce directory write permissions")
    try:
        assert _scanned_repo_ids(tmp_path, monkeypatch) == []
        assert _ref_names(repo_dir) == ["main"]
    finally:
        refs_dir.chmod(stat.S_IRWXU)
