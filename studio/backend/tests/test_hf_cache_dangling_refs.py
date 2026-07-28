# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A dangling ``refs/<branch>`` must not hide an intact repo from the scan.

``scan_cache_dir`` raises CorruptedCacheException for a repo whose ref names a
commit with no ``snapshots/<commit>/`` directory and omits it from ``.repos``,
so the model stays visible in the model picker (a plain directory walk) while
disappearing from every Hub inventory endpoint that feeds chat auto-load.

The repair is read-only: the hidden repo is rebuilt from the same directories
huggingface_hub reads and the ref file is left exactly as it is, because
``_cache_commit_hash_for_specific_revision`` writes refs with an unlocked
in-place ``write_text``, so no external process can delete one race-free.
"""

from __future__ import annotations

import dataclasses
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
    ref: str | None = UPSTREAM_HEAD,
    extra_refs: dict[str, str] | None = None,
    name: str = "models--Org--Model",
    payload: bytes = b"\0" * 11,
    snapshots: tuple[str, ...] = (SNAPSHOT,),
) -> Path:
    """A cache repo shaped like HF's, using regular files rather than symlinks.

    ``_scan_cached_repo`` resolves each snapshot entry to its blob, and a
    regular file resolves to itself, so this exercises the real scanner while
    staying runnable on Windows without the symlink privilege.
    """
    repo_dir = cache_root / name
    refs = repo_dir / "refs"
    refs.mkdir(parents = True, exist_ok = True)
    (repo_dir / "blobs").mkdir(parents = True, exist_ok = True)
    for commit in snapshots:
        snapshot = repo_dir / "snapshots" / commit
        snapshot.mkdir(parents = True, exist_ok = True)
        (snapshot / "model.safetensors").write_bytes(payload)
    if ref is not None:
        (refs / "main").write_text(ref, encoding = "utf-8")
    for ref_name, commit in (extra_refs or {}).items():
        (refs / ref_name).write_text(commit, encoding = "utf-8")
    return repo_dir


def _ref_names(repo_dir: Path) -> list[str]:
    return sorted(entry.name for entry in (repo_dir / "refs").rglob("*") if entry.is_file())


def _scan(cache_root: Path, monkeypatch) -> list:
    monkeypatch.setattr(inventory_scan, "hf_cache_roots", lambda: [cache_root])
    return inventory_scan._compute_all_hf_cache_scans()


def _scanned_repo_ids(cache_root: Path, monkeypatch) -> list[str]:
    return sorted(repo.repo_id for scan in _scan(cache_root, monkeypatch) for repo in scan.repos)


def _only_repo(cache_root: Path, monkeypatch):
    repos = [repo for scan in _scan(cache_root, monkeypatch) for repo in scan.repos]
    assert [repo.repo_id for repo in repos] == ["Org/Model"], "repo hidden from the scan"
    return repos[0]


# --- the #7374 symptom -------------------------------------------------------


def test_huggingface_hub_really_hides_a_repo_behind_a_dangling_ref(tmp_path):
    """Baseline for the bug: the snapshot is intact, yet the repo is dropped."""
    from huggingface_hub import scan_cache_dir

    repo_dir = _build_repo(tmp_path)

    raw = scan_cache_dir(cache_dir = str(tmp_path))

    assert [repo.repo_id for repo in raw.repos] == []
    assert raw.warnings
    assert (repo_dir / "snapshots" / SNAPSHOT / "model.safetensors").is_file()


def test_dangling_ref_no_longer_hides_an_intact_repo(tmp_path, monkeypatch):
    repo_dir = _build_repo(tmp_path)

    repo = _only_repo(tmp_path, monkeypatch)

    assert repo.repo_id == "Org/Model"
    assert repo.repo_type == "model"
    # routes/models.py does repo_path.parent, so this must stay a real Path.
    assert isinstance(repo.repo_path, Path) and repo.repo_path == repo_dir
    assert repo.size_on_disk == 11
    # The recovered revision keeps the identity the snapshot is loadable by.
    revision = next(iter(repo.revisions))
    assert revision.commit_hash == SNAPSHOT
    assert revision.snapshot_path == repo_dir / "snapshots" / SNAPSHOT
    assert {f.file_name for f in revision.files} == {"model.safetensors"}
    # The dangling ref resolves to nothing, so it maps to no revision...
    assert revision.refs == frozenset()
    # ...and, crucially, is still on disk: the repair never writes to the cache.
    assert _ref_names(repo_dir) == ["main"]
    assert (repo_dir / "refs" / "main").read_text(encoding = "utf-8") == UPSTREAM_HEAD


def test_recovery_reads_a_multi_file_multi_revision_repo(tmp_path, monkeypatch):
    other = "c" * 40
    repo_dir = _build_repo(tmp_path, snapshots = (SNAPSHOT, other))
    (repo_dir / "snapshots" / SNAPSHOT / "nested").mkdir()
    (repo_dir / "snapshots" / SNAPSHOT / "nested" / "extra.json").write_bytes(b"{}")

    repo = _only_repo(tmp_path, monkeypatch)

    assert {rev.commit_hash for rev in repo.revisions} == {SNAPSHOT, other}
    files = {f.file_path for rev in repo.revisions for f in rev.files}
    assert repo_dir / "snapshots" / SNAPSHOT / "nested" / "extra.json" in files
    # _resolve_cached_model_path relativises file_path against snapshot_path.
    for rev in repo.revisions:
        for f in rev.files:
            assert f.file_path.relative_to(rev.snapshot_path)


def test_a_still_resolvable_ref_keeps_its_revision_mapping(tmp_path, monkeypatch):
    """One good ref plus one stale ref: hub drops the repo, we keep the mapping."""
    repo_dir = _build_repo(tmp_path, ref = SNAPSHOT, extra_refs = {"stale": UPSTREAM_HEAD})

    repo = _only_repo(tmp_path, monkeypatch)

    assert next(iter(repo.revisions)).refs == frozenset({"main"})
    assert _ref_names(repo_dir) == ["main", "stale"]


# --- the repair must not widen past the leftover-refs assertion --------------


def test_a_healthy_cache_is_returned_untouched(tmp_path, monkeypatch):
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


def test_a_download_that_has_only_written_its_ref_is_not_invented(tmp_path, monkeypatch):
    """snapshot_download writes refs/<revision> before fetching the first file.

    There is no snapshot to recover yet, so nothing must be reported as
    downloaded -- that would be the very "already have it" lie #7374 is about.
    """
    repo_dir = tmp_path / "models--Org--Model"
    (repo_dir / "snapshots").mkdir(parents = True)
    (repo_dir / "refs").mkdir(parents = True)
    (repo_dir / "refs" / "main").write_text(UPSTREAM_HEAD, encoding = "utf-8")

    assert _scanned_repo_ids(tmp_path, monkeypatch) == []
    assert _ref_names(repo_dir) == ["main"]


def test_a_repo_corrupted_beyond_a_dangling_ref_stays_omitted(tmp_path, monkeypatch):
    """A broken snapshot symlink is corruption hub rejects for its own reasons."""
    repo_dir = _build_repo(tmp_path)
    broken = repo_dir / "snapshots" / SNAPSHOT / "weights.bin"
    try:
        os.symlink(repo_dir / "blobs" / "missing", broken)
    except (NotImplementedError, OSError):
        pytest.skip("symlinks unavailable (Windows without developer mode)")

    assert _scanned_repo_ids(tmp_path, monkeypatch) == []


def test_an_unrelated_repo_is_never_disturbed(tmp_path, monkeypatch):
    hidden = _build_repo(tmp_path)
    healthy = _build_repo(tmp_path, ref = SNAPSHOT, name = "models--Org--Healthy")

    assert _scanned_repo_ids(tmp_path, monkeypatch) == ["Org/Healthy", "Org/Model"]
    assert _ref_names(hidden) == ["main"]
    assert _ref_names(healthy) == ["main"]


def test_an_unreadable_repo_does_not_abort_the_recovery(tmp_path):
    """One unreadable repo must not stop the others being recovered.

    Scoped to the recovery pass because scan_cache_dir itself raises on an
    unreadable repo dir, which is upstream of this code and unchanged here.
    """
    from huggingface_hub import HFCacheInfo

    hidden = _build_repo(tmp_path)
    locked = _build_repo(tmp_path, ref = SNAPSHOT, name = "models--Org--Locked")
    locked.chmod(0)
    if os.access(locked / "refs", os.R_OK):
        pytest.skip("filesystem does not enforce directory permissions")
    try:
        scan = HFCacheInfo(size_on_disk = 0, repos = frozenset(), warnings = [])
        merged = inventory_scan._with_repos_hidden_by_dangling_refs(scan, tmp_path)
        assert sorted(repo.repo_id for repo in merged.repos) == ["Org/Model"]
        assert _ref_names(hidden) == ["main"]
    finally:
        locked.chmod(stat.S_IRWXU)


def test_a_non_repo_directory_is_ignored(tmp_path, monkeypatch):
    _build_repo(tmp_path)
    (tmp_path / ".locks").mkdir()
    (tmp_path / "notarepo").mkdir()
    (tmp_path / "spaces--Org--Thing").mkdir()

    assert _scanned_repo_ids(tmp_path, monkeypatch) == ["Org/Model"]


def test_a_scan_object_without_warnings_still_reaches_the_caller(tmp_path, monkeypatch):
    """The gate must read .warnings defensively: an AttributeError here lands in
    the per-root ``except Exception`` and silently blanks the whole cache."""
    from types import SimpleNamespace

    import huggingface_hub

    monkeypatch.setattr(
        huggingface_hub,
        "scan_cache_dir",
        lambda cache_dir = None: SimpleNamespace(cache_dir = cache_dir),
    )

    assert len(_scan(tmp_path, monkeypatch)) == 1


# --- version robustness ------------------------------------------------------


def test_recovered_entries_match_the_huggingface_hub_field_surface():
    """The recovered entries are duck-typed rather than built with hub's own
    constructors, so nothing breaks when a field is added or removed upstream.
    This is the tripwire that says the surfaces have drifted."""
    from huggingface_hub import CachedFileInfo, CachedRepoInfo, CachedRevisionInfo

    pairs = (
        (inventory_scan._RecoveredFileInfo, CachedFileInfo),
        (inventory_scan._RecoveredRevisionInfo, CachedRevisionInfo),
        (inventory_scan._RecoveredRepoInfo, CachedRepoInfo),
    )
    for ours, theirs in pairs:
        missing = {f.name for f in dataclasses.fields(theirs)} - {
            f.name for f in dataclasses.fields(ours)
        }
        assert not missing, f"{ours.__name__} is missing {sorted(missing)}"


def test_a_recovered_repo_survives_delete_revisions(tmp_path, monkeypatch):
    """Deleting a recovered model routes through HFCacheInfo.delete_revisions,
    which keys a dict by repo and takes a set difference over .revisions."""
    repo_dir = _build_repo(tmp_path)
    scan = _scan(tmp_path, monkeypatch)[0]

    strategy = scan.delete_revisions(SNAPSHOT)

    assert strategy.repos == frozenset({repo_dir})
    assert strategy.expected_freed_size == 11


# --- load identity for a recovered snapshot ----------------------------------


def _autoload_rows(cache_root: Path, monkeypatch) -> list[dict]:
    """What chat auto-load sees: GET /api/hub/cached-models."""
    from hub.services.models import cache_inventory
    from types import SimpleNamespace

    monkeypatch.setattr(inventory_scan, "hf_cache_roots", lambda: [cache_root])
    monkeypatch.setattr(
        "utils.hf_cache_settings.get_hf_cache_paths",
        lambda: SimpleNamespace(hub_cache = cache_root),
    )
    inventory_scan.invalidate_hf_cache_scans()
    try:
        return cache_inventory._scan_cached_models()
    finally:
        inventory_scan.invalidate_hf_cache_scans()


def test_auto_load_sees_a_model_hidden_behind_a_dangling_ref(tmp_path, monkeypatch):
    """The #7374 symptom end to end: the snapshot is on disk and the picker
    lists it, but auto-load's inventory reported nothing downloaded and the app
    fell through to a fresh download."""
    repo_dir = _build_repo(tmp_path)
    snapshot = repo_dir / "snapshots" / SNAPSHOT
    (snapshot / "config.json").write_text("{}", encoding = "utf-8")

    rows = _autoload_rows(tmp_path, monkeypatch)

    assert [row["repo_id"] for row in rows] == ["Org/Model"]
    # Recovered rows must carry the snapshot as their load identity: refs/main
    # dangles, so from_pretrained("Org/Model") would fail offline and download
    # the current upstream HEAD online instead of using what is already here.
    assert rows[0]["load_id"] == str(snapshot)
    assert rows[0]["active_cache"] is True


def test_a_resolvable_repo_still_loads_by_repo_id(tmp_path, monkeypatch):
    repo_dir = _build_repo(tmp_path, ref = SNAPSHOT)
    (repo_dir / "snapshots" / SNAPSHOT / "config.json").write_text("{}", encoding = "utf-8")

    rows = _autoload_rows(tmp_path, monkeypatch)

    assert [row["repo_id"] for row in rows] == ["Org/Model"]
    assert rows[0]["load_id"] == "Org/Model"


def test_default_ref_resolves_only_when_main_names_a_snapshot(tmp_path):
    dangling = _build_repo(tmp_path, name = "models--Org--Dangling")
    resolved = _build_repo(tmp_path, ref = SNAPSHOT, name = "models--Org--Resolved")
    detached = _build_repo(tmp_path, ref = None, name = "models--Org--Detached")

    assert inventory_scan.default_ref_resolves_on_disk(dangling) is False
    assert inventory_scan.default_ref_resolves_on_disk(resolved) is True
    assert inventory_scan.default_ref_resolves_on_disk(detached) is False


# --- the load id must name a snapshot that holds the advertised payload ------

OLDER = "d" * 40
NEWER = "e" * 40


def _age(path: Path, seconds: float) -> None:
    """Backdate a snapshot dir; snapshot selection orders by directory mtime."""
    stamp = os.stat(path).st_mtime - seconds
    os.utime(path, (stamp, stamp))


def _autoload_gguf_rows(cache_root: Path, monkeypatch) -> list[dict]:
    """What chat auto-load sees for GGUF: GET /api/hub/cached-gguf."""
    from hub.services.models import cache_inventory
    from types import SimpleNamespace

    monkeypatch.setattr(inventory_scan, "hf_cache_roots", lambda: [cache_root])
    monkeypatch.setattr(
        "utils.hf_cache_settings.get_hf_cache_paths",
        lambda: SimpleNamespace(hub_cache = cache_root),
    )
    inventory_scan.invalidate_hf_cache_scans()
    try:
        return cache_inventory._scan_cached_gguf()
    finally:
        inventory_scan.invalidate_hf_cache_scans()


def _two_snapshot_repo(cache_root: Path, older_files: dict, newer_files: dict) -> Path:
    """A repo whose payload sits in the older of two snapshots.

    Realistic because a metadata probe (config.json only) against a commit that
    has moved on materialises a newer, weightless snapshot beside the download.
    """
    repo_dir = cache_root / "models--Org--Model"
    (repo_dir / "blobs").mkdir(parents = True, exist_ok = True)
    (repo_dir / "refs").mkdir(parents = True, exist_ok = True)
    (repo_dir / "refs" / "main").write_text(UPSTREAM_HEAD, encoding = "utf-8")
    for commit, files in ((OLDER, older_files), (NEWER, newer_files)):
        snapshot = repo_dir / "snapshots" / commit
        snapshot.mkdir(parents = True, exist_ok = True)
        for name, payload in files.items():
            (snapshot / name).write_bytes(payload)
    _age(repo_dir / "snapshots" / OLDER, 600)
    return repo_dir


def test_load_id_names_the_snapshot_holding_the_safetensors_payload(tmp_path, monkeypatch):
    """The row aggregates weights over every revision, so the payload it
    advertises can live in an older snapshot than the newest directory."""
    repo_dir = _two_snapshot_repo(
        tmp_path,
        older_files = {"config.json": b"{}", "model.safetensors": b"\0" * 11},
        newer_files = {"config.json": b"{}"},
    )

    rows = _autoload_rows(tmp_path, monkeypatch)

    assert [row["repo_id"] for row in rows] == ["Org/Model"]
    assert rows[0]["model_format"] == "safetensors"
    load_dir = Path(rows[0]["load_id"])
    # Behavioural: from_pretrained(load_id) has to find the weights the row
    # advertised, otherwise auto-load fails on a model that is fully cached.
    assert any(entry.suffix == ".safetensors" for entry in load_dir.iterdir()), (
        f"load_id {load_dir.name} holds no weights; payload is in {OLDER[:8]}"
    )
    assert load_dir == repo_dir / "snapshots" / OLDER


def test_load_id_names_the_snapshot_holding_the_advertised_gguf_quant(tmp_path, monkeypatch):
    """Same for GGUF: the row's size sums quants across revisions, while local
    variant resolution only ever reads the one directory in ``load_id``."""
    from hub.utils.gguf import list_local_gguf_variants

    repo_dir = _two_snapshot_repo(
        tmp_path,
        older_files = {"config.json": b"{}", "Model-Q4_K_M.gguf": b"\0" * 32},
        newer_files = {"config.json": b"{}"},
    )

    rows = _autoload_gguf_rows(tmp_path, monkeypatch)

    assert [row["repo_id"] for row in rows] == ["Org/Model"]
    assert rows[0]["size_bytes"] == 32
    load_dir = Path(rows[0]["load_id"])
    variants, _has_vision = list_local_gguf_variants(str(load_dir))
    assert [v.quant for v in variants] == ["Q4_K_M"], (
        f"no variant resolves under load_id {load_dir.name}; the quant is in {OLDER[:8]}"
    )
    assert load_dir == repo_dir / "snapshots" / OLDER


def test_load_id_still_prefers_the_newest_snapshot_that_holds_the_payload(tmp_path, monkeypatch):
    """The payload rule must not pin loads to stale revisions: when both
    snapshots are runnable, the newest one still wins."""
    repo_dir = _two_snapshot_repo(
        tmp_path,
        older_files = {"config.json": b"{}", "model.safetensors": b"\0" * 11},
        newer_files = {"config.json": b"{}", "model.safetensors": b"\0" * 13},
    )

    rows = _autoload_rows(tmp_path, monkeypatch)

    assert Path(rows[0]["load_id"]) == repo_dir / "snapshots" / NEWER
