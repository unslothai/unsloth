# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The startup manifest downgrade migration records a clean pass and skips the scan on later starts."""

import json
import os
import shutil
import time
import types
from pathlib import Path

import pytest

from hub.utils import download_manifest, hf_cache_state, state_dir


@pytest.fixture
def home(monkeypatch, tmp_path):
    active = tmp_path / "active-hub"
    active.mkdir()
    monkeypatch.setattr(hf_cache_state, "hf_cache_roots", lambda **kw: [tmp_path])
    monkeypatch.setattr("utils.paths.storage_roots.cache_root", lambda: tmp_path / "app-cache")
    monkeypatch.setattr(state_dir, "cache_root", lambda: tmp_path / "state")
    settings = types.SimpleNamespace(hub_cache = active)
    monkeypatch.setattr("utils.hf_cache_settings.get_hf_cache_paths", lambda: settings)
    return types.SimpleNamespace(root = tmp_path, active = active, settings = settings)


def _write(
    repo_id,
    version,
    *,
    hub_cache,
    repo_type = "model",
    variant = None,
    **kw,
):
    assert download_manifest.write_manifest(
        repo_type,
        repo_id,
        variant,
        [download_manifest.ExpectedFile("model.safetensors", 8)],
        "http",
        hub_cache = hub_cache,
        _schema_version = version,
        **kw,
    )


def _populate(home):
    for i in range(6):
        _write(f"Org/V1-{i}", 1, hub_cache = home.active)
    _write("Org/Ordinary", 2, hub_cache = home.active, variant = "Q4_K_M")
    _write("Org/Unscoped", 2, hub_cache = None)
    other = home.root / "other-hub"
    other.mkdir(exist_ok = True)
    _write(
        "Org/Data",
        2,
        hub_cache = other,
        repo_type = "dataset",
        commit_hash = "commit-a",
        metadata_derived = True,
    )
    assert download_manifest.write_dataset_completion(
        "Org/Done",
        "c" * 40,
        [download_manifest.ExpectedFile("train.parquet", 4)],
        "http",
        hub_cache = home.active,
    )


def _settle(path: Path, age_s: float = 30.0) -> None:
    """Age every manifest directory past the racy window, as a restart a while after the last write sees it."""
    old = time.time_ns() - int(age_s * 1e9)
    for dirpath, _dirs, _files in os.walk(path):
        os.utime(dirpath, ns = (old, old))


def _snapshot(path: Path) -> dict:
    return {
        str(p.relative_to(path)): p.read_bytes() for p in sorted(path.rglob("*")) if p.is_file()
    }


def _count_reads(monkeypatch):
    calls = []
    real = download_manifest._read_migration_payload

    def counted(path):
        calls.append(path)
        return real(path)

    monkeypatch.setattr(download_manifest, "_read_migration_payload", counted)
    return calls


def _marker():
    return state_dir.state_root() / download_manifest._MANIFEST_MIGRATION_MARKER


def test_result_matches_the_unmarked_scan(home, monkeypatch, tmp_path):
    _populate(home)
    manifests = state_dir.manifests_dir()
    reference = tmp_path / "reference"
    shutil.copytree(manifests, reference, symlinks = True)

    migrated = download_manifest.migrate_ordinary_v2_manifests_for_downgrade()
    head_state = _snapshot(manifests)

    # The same pass with the marker hooks disabled is the pre-change algorithm.
    shutil.rmtree(manifests)
    shutil.copytree(reference, manifests, symlinks = True)
    with monkeypatch.context() as m:
        m.setattr(download_manifest, "_migration_already_clean", lambda parent: False)
        m.setattr(download_manifest, "_record_clean_migration", lambda *a: None)
        assert download_manifest.migrate_ordinary_v2_manifests_for_downgrade() == migrated
    assert _snapshot(manifests) == head_state
    assert migrated >= 3
    for path in manifests.rglob("*.json"):
        payload = json.loads(path.read_text(encoding = "utf-8"))
        if not str(payload.get("variant") or "").startswith("_studio-dataset-complete-"):
            assert payload["version"] == 1, path


def test_fresh_home_without_manifests_dir(home):
    assert state_dir.manifests_dir() is not None
    assert not state_dir.manifests_dir().exists()
    assert download_manifest.migrate_ordinary_v2_manifests_for_downgrade() == 0
    assert download_manifest.migrate_ordinary_v2_manifests_for_downgrade() == 0
    assert not _marker().exists()


def test_second_start_skips_the_scan(home, monkeypatch):
    _populate(home)
    assert download_manifest.migrate_ordinary_v2_manifests_for_downgrade() >= 3
    manifests = state_dir.manifests_dir()
    _settle(manifests)
    before = _snapshot(manifests)

    calls = _count_reads(monkeypatch)
    assert download_manifest.migrate_ordinary_v2_manifests_for_downgrade() == 0
    assert len(calls) >= 8

    calls.clear()
    assert download_manifest.migrate_ordinary_v2_manifests_for_downgrade() == 0
    assert calls == []
    assert _snapshot(manifests) == before
    assert _marker().is_file()


def _clean_marked(home, monkeypatch):
    _populate(home)
    download_manifest.migrate_ordinary_v2_manifests_for_downgrade()
    _settle(state_dir.manifests_dir())
    download_manifest.migrate_ordinary_v2_manifests_for_downgrade()
    assert _marker().is_file()
    return _count_reads(monkeypatch)


def test_new_ordinary_v2_manifest_reruns_the_migration(home, monkeypatch):
    calls = _clean_marked(home, monkeypatch)
    _write("Org/Later", 2, hub_cache = home.active)
    assert download_manifest.migrate_ordinary_v2_manifests_for_downgrade() == 1
    assert calls
    path = download_manifest.manifest_path("model", "Org/Later", None, hub_cache = home.active)
    assert json.loads(path.read_text(encoding = "utf-8"))["version"] == 1


def test_replaced_manifest_in_scope_dir_reruns(home, monkeypatch):
    calls = _clean_marked(home, monkeypatch)
    # Overwrite an existing name: the set of names is unchanged, only the scope dir mtime moves.
    _write("Org/V1-0", 2, hub_cache = home.active)
    assert download_manifest.migrate_ordinary_v2_manifests_for_downgrade() == 1
    assert calls


def test_new_cache_scope_reruns(home, monkeypatch):
    calls = _clean_marked(home, monkeypatch)
    fresh = home.root / "fresh-hub"
    fresh.mkdir()
    _write("Org/Fresh", 2, hub_cache = fresh)
    assert download_manifest.migrate_ordinary_v2_manifests_for_downgrade() == 1
    assert calls


def test_migration_id_change_reruns(home, monkeypatch):
    calls = _clean_marked(home, monkeypatch)
    monkeypatch.setattr(download_manifest, "_MANIFEST_MIGRATION_ID", "ordinary-v2-to-v1:2")
    assert download_manifest.migrate_ordinary_v2_manifests_for_downgrade() == 0
    assert calls


def test_manifest_version_change_reruns(home, monkeypatch):
    calls = _clean_marked(home, monkeypatch)
    monkeypatch.setattr(download_manifest, "_MANIFEST_VERSION", 3)
    download_manifest.migrate_ordinary_v2_manifests_for_downgrade()
    assert calls


def test_active_cache_change_reruns(home, monkeypatch):
    calls = _clean_marked(home, monkeypatch)
    moved = home.root / "moved-hub"
    moved.mkdir()
    home.settings.hub_cache = moved
    download_manifest.migrate_ordinary_v2_manifests_for_downgrade()
    assert calls


def test_corrupt_marker_reruns(home, monkeypatch):
    calls = _clean_marked(home, monkeypatch)
    _marker().write_text("{not json", encoding = "utf-8")
    assert download_manifest.migrate_ordinary_v2_manifests_for_downgrade() == 0
    assert calls


def test_recently_changed_dirs_are_not_recorded(home):
    _populate(home)
    download_manifest.migrate_ordinary_v2_manifests_for_downgrade()
    # No settling: the dirs were written within the racy window.
    assert download_manifest.migrate_ordinary_v2_manifests_for_downgrade() == 0
    assert not _marker().exists()


def test_pass_that_migrates_is_not_recorded(home):
    _populate(home)
    _settle(state_dir.manifests_dir())
    assert download_manifest.migrate_ordinary_v2_manifests_for_downgrade() >= 3
    assert not _marker().exists()


def test_failed_write_is_not_recorded_and_retries(home, monkeypatch):
    _populate(home)
    _settle(state_dir.manifests_dir())
    with monkeypatch.context() as m:
        m.setattr(download_manifest, "_atomic_write_json", lambda *a, **k: False)
        assert download_manifest.migrate_ordinary_v2_manifests_for_downgrade() == 0
    assert not _marker().exists()
    assert download_manifest.migrate_ordinary_v2_manifests_for_downgrade() >= 3


def test_file_cap_is_not_recorded(home, monkeypatch):
    _populate(home)
    _settle(state_dir.manifests_dir())
    download_manifest.migrate_ordinary_v2_manifests_for_downgrade()
    _settle(state_dir.manifests_dir())
    monkeypatch.setattr(download_manifest, "_MANIFEST_MIGRATION_MAX_FILES", 2)
    download_manifest.migrate_ordinary_v2_manifests_for_downgrade()
    assert not _marker().exists()


def test_marker_lives_outside_the_manifest_dirs(home):
    _populate(home)
    download_manifest.migrate_ordinary_v2_manifests_for_downgrade()
    _settle(state_dir.manifests_dir())
    download_manifest.migrate_ordinary_v2_manifests_for_downgrade()
    manifests = state_dir.manifests_dir()
    assert _marker().parent != manifests
    assert manifests not in _marker().parents


def test_deeply_nested_marker_still_migrates(home):
    # json.loads raises RecursionError (not ValueError) on this; it must not abort the migration.
    _populate(home)
    _settle(state_dir.manifests_dir())
    _marker().parent.mkdir(parents = True, exist_ok = True)
    _marker().write_text("[" * 60_000, encoding = "utf-8")
    assert download_manifest.migrate_ordinary_v2_manifests_for_downgrade() >= 3


def test_marker_check_error_falls_back_to_the_full_scan(home, monkeypatch):
    _populate(home)
    _settle(state_dir.manifests_dir())

    def boom(*a, **k):
        raise RuntimeError("unexpected")

    monkeypatch.setattr(download_manifest, "_migration_already_clean", boom)
    assert download_manifest.migrate_ordinary_v2_manifests_for_downgrade() >= 3
    assert not _marker().exists()
