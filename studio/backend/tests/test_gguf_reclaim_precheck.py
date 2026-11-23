# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Post-download GGUF reclaim skips the all-roots cache scan when only protected snapshots exist."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest

from hub.services.models import deletion
from hub.utils import inventory_scan

REPO_ID = "Org/Repo-GGUF"
OLD = "a" * 40
NEW = "b" * 40
OLD_BLOB = "1" * 64
NEW_BLOB = "2" * 64
Q8_BLOB = "3" * 64


def _write_snapshot(repo_dir: Path, commit: str, files: dict[str, str], layout: str) -> None:
    snap = repo_dir / "snapshots" / commit
    snap.mkdir(parents = True, exist_ok = True)
    for name, blob_name in files.items():
        blob = repo_dir / "blobs" / blob_name
        if not blob.exists():
            blob.parent.mkdir(parents = True, exist_ok = True)
            blob.write_bytes(blob_name.encode() * 4)
        link = snap / name
        if layout == "symlink":
            link.symlink_to(os.path.relpath(blob, snap))
        else:
            # Windows without developer mode stores plain copies in the snapshot.
            shutil.copyfile(blob, link)
    refs = repo_dir / "refs"
    refs.mkdir(parents = True, exist_ok = True)
    (refs / "main").write_text(commit)


def _repo(hub: Path, dir_name: str = "models--Org--Repo-GGUF") -> Path:
    return hub / dir_name


@pytest.fixture
def hub(monkeypatch, tmp_path):
    hub = tmp_path / "hub"
    hub.mkdir()
    monkeypatch.setattr(inventory_scan, "hf_cache_roots", lambda: [hub])
    monkeypatch.setattr(deletion.download_manifest, "purge_state", lambda *_args, **_kwargs: False)
    inventory_scan.invalidate_hf_cache_scans()
    yield hub
    inventory_scan.invalidate_hf_cache_scans()


def _forbid_scan(monkeypatch) -> list:
    calls: list = []

    def boom(*_args, **_kwargs):
        calls.append(1)
        raise AssertionError("full cache scan must not run")

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "scan_cache_dir", boom)
    monkeypatch.setattr(deletion.cache_inventory, "all_hf_cache_scans", boom)
    return calls


def _reclaim(hub: Path, keep: str, protected: set[str]) -> dict:
    return deletion.reclaim_replaced_gguf_variant(
        REPO_ID,
        "Q4_K_M",
        frozenset({keep}),
        None,
        hub_cache = hub,
        protected_revisions = frozenset(protected),
    )


def _tree(root: Path) -> set[str]:
    return {str(p.relative_to(root)) for p in root.rglob("*")}


@pytest.mark.parametrize("layout", ["symlink", "copy"])
@pytest.mark.parametrize("dir_name", ["models--Org--Repo-GGUF", "models--org--repo-gguf"])
def test_first_download_skips_scan(monkeypatch, hub, layout, dir_name):
    repo_dir = _repo(hub, dir_name)
    _write_snapshot(repo_dir, NEW, {"Model-Q4_K_M.gguf": NEW_BLOB}, layout)
    before = _tree(hub)
    calls = _forbid_scan(monkeypatch)

    result = _reclaim(hub, NEW_BLOB, {NEW})

    assert result["status"] == "skipped"
    assert result["reason"] == "nothing_to_reclaim"
    assert calls == []
    assert _tree(hub) == before


@pytest.mark.parametrize("layout", ["symlink", "copy"])
@pytest.mark.parametrize("dir_name", ["models--Org--Repo-GGUF", "models--org--repo-gguf"])
def test_second_quant_same_commit_skips_scan(monkeypatch, hub, layout, dir_name):
    repo_dir = _repo(hub, dir_name)
    _write_snapshot(
        repo_dir,
        NEW,
        {"Model-Q8_0.gguf": Q8_BLOB, "Model-Q4_K_M.gguf": NEW_BLOB},
        layout,
    )
    before = _tree(hub)
    calls = _forbid_scan(monkeypatch)

    result = _reclaim(hub, NEW_BLOB, {NEW, "c" * 40})

    assert result["reason"] == "nothing_to_reclaim"
    assert calls == []
    assert _tree(hub) == before


def test_missing_hub_cache_skips_scan(monkeypatch, tmp_path):
    calls = _forbid_scan(monkeypatch)
    result = _reclaim(tmp_path / "absent", NEW_BLOB, {NEW})
    assert result["reason"] == "nothing_to_reclaim"
    assert calls == []


def test_other_repo_snapshots_do_not_force_scan(monkeypatch, hub):
    _write_snapshot(_repo(hub), NEW, {"Model-Q4_K_M.gguf": NEW_BLOB}, "symlink")
    other = _repo(hub, "models--Org--Repo-GGUF-v2")
    _write_snapshot(other, OLD, {"Model-Q4_K_M.gguf": OLD_BLOB}, "symlink")
    _write_snapshot(_repo(hub, "datasets--Org--Repo-GGUF"), OLD, {"x.gguf": OLD_BLOB}, "symlink")
    calls = _forbid_scan(monkeypatch)

    assert _reclaim(hub, NEW_BLOB, {NEW})["reason"] == "nothing_to_reclaim"
    assert calls == []


def test_unprotected_snapshot_in_case_variant_dir_runs_scan(monkeypatch, hub):
    _write_snapshot(_repo(hub), NEW, {"Model-Q4_K_M.gguf": NEW_BLOB}, "symlink")
    _write_snapshot(
        _repo(hub, "models--org--repo-gguf"), OLD, {"Model-Q4_K_M.gguf": OLD_BLOB}, "symlink"
    )
    calls = _forbid_scan(monkeypatch)
    monkeypatch.setattr(
        deletion.cache_inventory, "all_hf_cache_scans", lambda: calls.append(1) or []
    )

    result = _reclaim(hub, NEW_BLOB, {NEW})

    assert calls == [1]
    assert result["status"] == "reclaimed"


def test_unreadable_snapshots_fall_back_to_scan(hub):
    repo_dir = _repo(hub)
    repo_dir.mkdir()
    (repo_dir / "snapshots").write_text("not a directory")
    assert deletion._has_unprotected_snapshot(hub, REPO_ID, frozenset({NEW}))


def _replaced_variant_cache(hub: Path, dir_name: str, layout: str) -> None:
    repo_dir = _repo(hub, dir_name)
    _write_snapshot(
        repo_dir,
        OLD,
        {"Model-Q4_K_M.gguf": OLD_BLOB, "Model-Q8_0.gguf": Q8_BLOB},
        layout,
    )
    _write_snapshot(repo_dir, NEW, {"Model-Q4_K_M.gguf": NEW_BLOB}, layout)


@pytest.mark.parametrize("layout", ["symlink", "copy"])
@pytest.mark.parametrize("dir_name", ["models--Org--Repo-GGUF", "models--org--repo-gguf"])
def test_replaced_variant_reclaims_like_full_scan(monkeypatch, tmp_path, layout, dir_name):
    monkeypatch.setattr(deletion.download_manifest, "purge_state", lambda *_args, **_kwargs: False)
    results = {}
    trees = {}
    for arm in ("base", "head"):
        hub = tmp_path / arm / "hub"
        hub.mkdir(parents = True)
        _replaced_variant_cache(hub, dir_name, layout)
        monkeypatch.setattr(inventory_scan, "hf_cache_roots", lambda hub = hub: [hub])
        inventory_scan.invalidate_hf_cache_scans()
        with monkeypatch.context() as m:
            if arm == "base":
                # the pre-change behaviour: always run the full scan.
                m.setattr(deletion, "_has_unprotected_snapshot", lambda *_a: True)
            results[arm] = _reclaim(hub, NEW_BLOB, {NEW})
        trees[arm] = _tree(hub)
    inventory_scan.invalidate_hf_cache_scans()

    assert results["head"] == results["base"]
    assert trees["head"] == trees["base"]
    repo_rel = os.path.join(dir_name, "blobs", OLD_BLOB)
    if layout == "symlink":
        assert results["head"]["deleted_blobs"] == 1
        assert repo_rel not in trees["head"]
    assert os.path.join(dir_name, "blobs", NEW_BLOB) in trees["head"]
    assert os.path.join(dir_name, "blobs", Q8_BLOB) in trees["head"]
