# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A dangling ``refs/<branch>`` must not hide an intact repo from the scan.

``scan_cache_dir`` raises CorruptedCacheException for a repo whose ref names a commit with no
``snapshots/<commit>/`` directory and omits it from ``.repos``, so the model stays visible in the
picker (a plain directory walk) while disappearing from every Hub inventory endpoint that feeds chat
auto-load. The repair is read-only: ``_cache_commit_hash_for_specific_revision`` writes refs with an
unlocked in-place ``write_text``, so no external process can delete one race-free.
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

    ``_scan_cached_repo`` resolves each snapshot entry to its blob and a regular file resolves to
    itself, so this exercises the real scanner while still running on Windows without the symlink
    privilege.
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


def _empty_cache_info(cls):
    """An empty ``HFCacheInfo`` across huggingface_hub versions.

    Fields are read off the dataclass rather than named, so a release adding or dropping one cannot
    fail this test on a signature it never exercises.
    """
    known = {"size_on_disk": 0, "repos": frozenset(), "warnings": []}
    return cls(**{f.name: known.get(f.name, frozenset()) for f in dataclasses.fields(cls)})


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
    """snapshot_download writes refs/<revision> before fetching the first file, so there is no
    snapshot to recover yet and nothing may be reported as downloaded."""
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

    Scoped to the recovery pass: scan_cache_dir itself raises on an unreadable repo dir, which is
    upstream of this code and unchanged here.
    """
    from huggingface_hub import HFCacheInfo

    hidden = _build_repo(tmp_path)
    locked = _build_repo(tmp_path, ref = SNAPSHOT, name = "models--Org--Locked")
    locked.chmod(0)
    if os.access(locked / "refs", os.R_OK):
        pytest.skip("filesystem does not enforce directory permissions")
    try:
        scan = _empty_cache_info(HFCacheInfo)
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
    """The gate must read .warnings defensively: an AttributeError here lands in the per-root
    ``except Exception`` and silently blanks the whole cache."""
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
    """The recovered entries are duck-typed rather than built with hub's own constructors; this is
    the tripwire for upstream field drift."""
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
    """Deleting a recovered model routes through HFCacheInfo.delete_revisions, which keys a dict by
    repo and takes a set difference over .revisions."""
    repo_dir = _build_repo(tmp_path)
    scan = _scan(tmp_path, monkeypatch)[0]

    strategy = scan.delete_revisions(SNAPSHOT)

    assert strategy.repos == frozenset({repo_dir})
    assert strategy.expected_freed_size == 11


# --- load identity for a recovered snapshot ----------------------------------


def _autoload_rows(
    cache_root: Path,
    monkeypatch,
    *,
    gguf: bool = False,
) -> list[dict]:
    """What chat auto-load sees: GET /api/hub/cached-models, or /cached-gguf with *gguf* set."""
    from hub.services.models import cache_inventory
    from types import SimpleNamespace

    monkeypatch.setattr(inventory_scan, "hf_cache_roots", lambda: [cache_root])
    monkeypatch.setattr(
        "utils.hf_cache_settings.get_hf_cache_paths",
        lambda: SimpleNamespace(hub_cache = cache_root),
    )
    inventory_scan.invalidate_hf_cache_scans()
    try:
        if gguf:
            return cache_inventory._scan_cached_gguf()
        return cache_inventory._scan_cached_models()
    finally:
        inventory_scan.invalidate_hf_cache_scans()


def test_auto_load_sees_a_model_hidden_behind_a_dangling_ref(tmp_path, monkeypatch):
    """End to end: the snapshot is on disk and the picker lists it, but the inventory reported
    nothing downloaded and the app re-downloaded."""
    repo_dir = _build_repo(tmp_path)
    snapshot = repo_dir / "snapshots" / SNAPSHOT
    (snapshot / "config.json").write_text("{}", encoding = "utf-8")

    rows = _autoload_rows(tmp_path, monkeypatch)

    assert [row["repo_id"] for row in rows] == ["Org/Model"]
    # Recovered rows carry the snapshot as their load identity: refs/main dangles, so
    # from_pretrained("Org/Model") ignores what is already here.
    assert rows[0]["load_id"] == str(snapshot)
    assert rows[0]["active_cache"] is True


def test_a_resolvable_repo_still_loads_by_repo_id(tmp_path, monkeypatch):
    repo_dir = _build_repo(tmp_path, ref = SNAPSHOT)
    (repo_dir / "snapshots" / SNAPSHOT / "config.json").write_text("{}", encoding = "utf-8")

    rows = _autoload_rows(tmp_path, monkeypatch)

    assert [row["repo_id"] for row in rows] == ["Org/Model"]
    assert rows[0]["load_id"] == "Org/Model"


def test_default_ref_resolves_only_when_main_names_a_snapshot(tmp_path):
    """The pin only defers to refs/main when it names a directory actually on disk; a dangling or
    absent ref has to fall through."""
    dangling = _build_repo(tmp_path, name = "models--Org--Dangling")
    resolved = _build_repo(tmp_path, ref = SNAPSHOT, name = "models--Org--Resolved")
    detached = _build_repo(tmp_path, ref = None, name = "models--Org--Detached")

    assert inventory_scan.default_ref_snapshot(dangling) is None
    assert inventory_scan.default_ref_snapshot(resolved) is not None
    assert inventory_scan.default_ref_snapshot(detached) is None


# --- the load id must name a snapshot that holds the advertised payload ------

OLDER = "d" * 40
NEWER = "e" * 40


def _age(path: Path, seconds: float) -> None:
    """Backdate a snapshot dir; snapshot selection orders by directory mtime."""
    stamp = os.stat(path).st_mtime - seconds
    os.utime(path, (stamp, stamp))


def _autoload_gguf_rows(cache_root: Path, monkeypatch) -> list[dict]:
    """What chat auto-load sees for GGUF: GET /api/hub/cached-gguf."""
    return _autoload_rows(cache_root, monkeypatch, gguf = True)


def _two_snapshot_repo(
    cache_root: Path,
    older_files: dict,
    newer_files: dict,
    *,
    ref: str | None = UPSTREAM_HEAD,
) -> Path:
    """A repo whose payload sits in the older of two snapshots.

    Realistic because a metadata probe (config.json only) against a commit that has moved on
    materialises a newer, weightless snapshot beside the download. ``ref = None`` is what a
    commit-pinned fetch leaves: ``snapshot_download`` only writes a ref for a branch or tag.
    """
    repo_dir = cache_root / "models--Org--Model"
    (repo_dir / "blobs").mkdir(parents = True, exist_ok = True)
    (repo_dir / "refs").mkdir(parents = True, exist_ok = True)
    if ref is not None:
        (repo_dir / "refs" / "main").write_text(ref, encoding = "utf-8")
    for commit, files in ((OLDER, older_files), (NEWER, newer_files)):
        snapshot = repo_dir / "snapshots" / commit
        snapshot.mkdir(parents = True, exist_ok = True)
        for name, payload in files.items():
            # Keys may name a subdir ("MTP/...") the way real GGUF repos ship companions.
            (snapshot / name).parent.mkdir(parents = True, exist_ok = True)
            (snapshot / name).write_bytes(payload)
    _age(repo_dir / "snapshots" / OLDER, 600)
    return repo_dir


def test_load_id_names_the_snapshot_holding_the_safetensors_payload(tmp_path, monkeypatch):
    """The row aggregates weights over every revision, so the payload it advertises can live in an
    older snapshot than the newest directory."""
    repo_dir = _two_snapshot_repo(
        tmp_path,
        older_files = {"config.json": b"{}", "model.safetensors": b"\0" * 11},
        newer_files = {"config.json": b"{}"},
    )

    rows = _autoload_rows(tmp_path, monkeypatch)

    assert [row["repo_id"] for row in rows] == ["Org/Model"]
    assert rows[0]["model_format"] == "safetensors"
    load_dir = Path(rows[0]["load_id"])
    # from_pretrained(load_id) must find the weights the row advertised.
    assert any(
        entry.suffix == ".safetensors" for entry in load_dir.iterdir()
    ), f"load_id {load_dir.name} holds no weights; payload is in {OLDER[:8]}"
    assert load_dir == repo_dir / "snapshots" / OLDER


def test_load_id_names_the_snapshot_holding_the_advertised_gguf_quant(tmp_path, monkeypatch):
    """Same for GGUF: the row's size sums quants across revisions, while local variant resolution
    only ever reads the one directory in ``load_id``."""
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
    assert [v.quant for v in variants] == [
        "Q4_K_M"
    ], f"no variant resolves under load_id {load_dir.name}; the quant is in {OLDER[:8]}"
    assert load_dir == repo_dir / "snapshots" / OLDER


def test_load_id_still_prefers_the_newest_snapshot_that_holds_the_payload(tmp_path, monkeypatch):
    """The payload rule must not pin loads to stale revisions: with both snapshots runnable, the
    newest still wins."""
    repo_dir = _two_snapshot_repo(
        tmp_path,
        older_files = {"config.json": b"{}", "model.safetensors": b"\0" * 11},
        newer_files = {"config.json": b"{}", "model.safetensors": b"\0" * 13},
    )

    rows = _autoload_rows(tmp_path, monkeypatch)

    assert Path(rows[0]["load_id"]) == repo_dir / "snapshots" / NEWER


def test_load_id_leaves_the_payload_snapshot_when_main_resolves_elsewhere(tmp_path, monkeypatch):
    """A resolving ``refs/main`` is not enough: the metadata probe that strands the weights in an
    older snapshot repoints it at the weightless one, so loading by repo id finds no weights."""
    repo_dir = _two_snapshot_repo(
        tmp_path,
        older_files = {"config.json": b"{}", "model.safetensors": b"\0" * 11},
        newer_files = {"config.json": b"{}"},
        ref = NEWER,
    )

    rows = _autoload_rows(tmp_path, monkeypatch)

    assert [row["repo_id"] for row in rows] == ["Org/Model"]
    load_id = rows[0]["load_id"]
    # Follow the load identity the way a load would (repo id via refs/main).
    resolved = (
        repo_dir / "snapshots" / (repo_dir / "refs" / "main").read_text(encoding = "utf-8")
        if load_id == "Org/Model"
        else Path(load_id)
    )
    assert any(
        entry.suffix == ".safetensors" for entry in resolved.iterdir()
    ), f"load_id {load_id} resolves to {resolved.name}, which holds no weights"
    assert Path(load_id) == repo_dir / "snapshots" / OLDER


def test_load_id_stays_the_repo_id_when_main_resolves_onto_the_payload(tmp_path, monkeypatch):
    """The rule must stay narrow: when ``refs/main`` names a snapshot that does hold the payload,
    the pinned revision keeps winning and the row keeps the repo id."""
    _two_snapshot_repo(
        tmp_path,
        older_files = {"config.json": b"{}", "model.safetensors": b"\0" * 11},
        newer_files = {"config.json": b"{}", "model.safetensors": b"\0" * 13},
        ref = OLDER,
    )

    rows = _autoload_rows(tmp_path, monkeypatch)

    assert rows[0]["load_id"] == "Org/Model"


# --- everything the row advertises must resolve under its load id ------------


def _local_gguf_variants_for_autoload(row: dict, cache_root: Path) -> list[str]:
    """The quants chat auto-load is offered: GET /api/models/gguf-variants with ``preferLocalCache``
    and the row's ``cache_path``, exactly as chat-adapter calls it before /load."""
    import asyncio

    from hub.services.models import gguf_variants

    response = asyncio.run(
        gguf_variants.get_gguf_variants_response(
            row["repo_id"],
            prefer_local_cache = True,
            local_path = row["cache_path"],
        )
    )
    return [variant.quant for variant in response.variants if variant.downloaded]


def _listed_gguf_variants(row: dict, cache_root: Path) -> list[str]:
    """Every quant the same call reports, ready or not: what Settings and the Hub cards show, so a
    torn download can still be resumed or deleted."""
    import asyncio

    from hub.services.models import gguf_variants

    response = asyncio.run(
        gguf_variants.get_gguf_variants_response(
            row["repo_id"],
            prefer_local_cache = True,
            local_path = row["cache_path"],
        )
    )
    return sorted(variant.quant for variant in response.variants)


def test_a_half_split_quant_shadows_neither_the_load_id_nor_the_variants(tmp_path, monkeypatch):
    """The newest snapshot can hold shard 1 of an interrupted split download while a complete quant
    sits in an older one. Both ends are asserted because they are only correct together: the load id
    and the quants offered under it must name one directory."""
    from hub.utils.gguf import list_local_gguf_variants

    repo_dir = _two_snapshot_repo(
        tmp_path,
        older_files = {"Model-Q4_K_M.gguf": b"\0" * 32},
        newer_files = {"Model-Q8_0-00001-of-00002.gguf": b"\0" * 16},
    )

    rows = _autoload_gguf_rows(tmp_path, monkeypatch)

    assert [row["repo_id"] for row in rows] == ["Org/Model"]
    load_dir = Path(rows[0]["load_id"])
    held = {v.quant for v in list_local_gguf_variants(str(load_dir))[0] if v.quant}
    complete = inventory_scan._completed_gguf_variants(load_dir)
    assert held and held <= complete, (
        f"load_id {load_dir.name} offers {sorted(held)} with only "
        f"{sorted(complete)} complete; the usable quant is in {OLDER[:8]}"
    )
    assert load_dir == repo_dir / "snapshots" / OLDER
    offered = _local_gguf_variants_for_autoload(rows[0], tmp_path)
    resolvable = {v.quant for v in list_local_gguf_variants(str(load_dir))[0]}
    # Every quant offered as downloaded must resolve under the load id, unshadowed by the broken
    # one.
    assert set(offered) <= resolvable, (
        f"auto-load is offered {sorted(offered)} but load_id {load_dir.name[:8]} "
        f"resolves only {sorted(resolvable)}"
    )
    assert offered == ["Q4_K_M"]


@pytest.mark.parametrize(
    "newer_files, listed, offered",
    [
        # With nothing complete anywhere the newest snapshot holding quants is still reported so it
        # can be resumed or deleted, but a quant short a shard cannot be handed to /load.
        pytest.param(
            {"Model-Q8_0-00001-of-00002.gguf": b"\0" * 16},
            ["Q8_0"],
            [],
            id = "nothing-complete-anywhere",
        ),
        # When that snapshot holds a whole quant beside the half-downloaded one, offering both
        # shadows the usable one: auto-load takes only the smallest.
        pytest.param(
            {
                "Model-Q8_0.gguf": b"\0" * 64,
                "Model-Q4_K_M-00001-of-00002.gguf": b"\0" * 16,
            },
            ["Q4_K_M", "Q8_0"],
            ["Q8_0"],
            id = "one-whole-quant-beside-a-half-one",
        ),
    ],
)
def test_gguf_variants_still_list_when_no_snapshot_is_complete(
    newer_files, listed, offered, tmp_path, monkeypatch
):
    """The completeness preference must not empty the list, nor offer a quant short a shard while a
    whole one sits beside it."""
    repo_dir = _two_snapshot_repo(
        tmp_path,
        older_files = {"Model-Q4_K_M-00001-of-00002.gguf": b"\0" * 32},
        newer_files = newer_files,
    )

    rows = _autoload_gguf_rows(tmp_path, monkeypatch)

    load_dir = Path(rows[0]["load_id"])
    assert load_dir == repo_dir / "snapshots" / NEWER
    assert _listed_gguf_variants(rows[0], tmp_path) == listed
    assert _local_gguf_variants_for_autoload(rows[0], tmp_path) == offered


def test_a_whole_quant_in_a_mixed_newest_snapshot_beats_an_older_larger_one(tmp_path, monkeypatch):
    """A whole small quant can sit in the newest snapshot beside an interrupted split one while an
    older snapshot holds only a whole larger quant.

    Auto-load takes the smallest quant offered, so skipping that newest snapshot spends the attempt
    on the larger one. The snapshot counts as usable and only its completed subset is offered, so
    both ends still name one directory."""
    from hub.utils.gguf import list_local_gguf_variants

    repo_dir = _two_snapshot_repo(
        tmp_path,
        older_files = {"Model-Q6_K.gguf": b"\0" * 96},
        newer_files = {
            "Model-Q4_K_M.gguf": b"\0" * 16,
            "Model-Q8_0-00001-of-00002.gguf": b"\0" * 8,
        },
    )

    rows = _autoload_gguf_rows(tmp_path, monkeypatch)

    assert [row["repo_id"] for row in rows] == ["Org/Model"]
    load_dir = Path(rows[0]["load_id"])
    assert load_dir == repo_dir / "snapshots" / NEWER
    offered = _local_gguf_variants_for_autoload(rows[0], tmp_path)
    assert offered == ["Q4_K_M"]
    # The pair still has to agree on one directory, without advertising the interrupted split quant.
    resolvable = {v.quant for v in list_local_gguf_variants(str(load_dir))[0]}
    assert set(offered) <= resolvable, (
        f"auto-load is offered {sorted(offered)} but load_id {load_dir.name[:8]} "
        f"resolves only {sorted(resolvable)}"
    )
    assert "Q8_0" not in offered
    # Pinning the snapshot holding the interrupted download must not flip the row partial: a whole
    # quant is loadable from it.
    assert rows[0].get("partial") is False
    assert rows[0].get("capabilities", {}).get("can_chat") is True


def test_load_id_is_not_pinned_to_a_snapshot_that_has_no_config(tmp_path, monkeypatch):
    """The repo-level format may rest on transformer-named weights alone, but a pinned load id names
    one directory and from_pretrained needs ``config.json`` inside it. Keep the repo id, which can
    still fill the config in from the hub, rather than pin a weight-only snapshot."""
    _two_snapshot_repo(
        tmp_path,
        older_files = {"model.safetensors": b"\0" * 11},
        newer_files = {"config.json": b"{}"},
        ref = NEWER,
    )

    rows = _autoload_rows(tmp_path, monkeypatch)

    assert [row["repo_id"] for row in rows] == ["Org/Model"]
    assert rows[0]["model_format"] == "safetensors"
    load_id = rows[0]["load_id"]
    assert load_id == "Org/Model", (
        f"load_id {Path(load_id).name[:8]} holds no config.json, so a local "
        "from_pretrained on it cannot resolve the architecture"
    )


def test_no_snapshot_holds_the_payload_so_a_dangling_ref_pins_nothing(tmp_path, monkeypatch):
    """Same repo, but with the dangling ``refs/main`` this branch exists for.

    The dangling-ref arm must not pin the fallback newest snapshot, already known not to hold the
    payload: a directory the load cannot use is worse than the repo id, which can still complete the
    config."""
    _two_snapshot_repo(
        tmp_path,
        older_files = {"model.safetensors": b"\0" * 11},
        newer_files = {"config.json": b"{}"},
        ref = UPSTREAM_HEAD,
    )

    rows = _autoload_rows(tmp_path, monkeypatch)

    assert [row["repo_id"] for row in rows] == ["Org/Model"]
    load_id = rows[0]["load_id"]
    if load_id != "Org/Model":
        # Behavioural: a pinned directory is only useful if from_pretrained can read weights out of
        # it.
        pinned = Path(load_id)
        held = sorted(entry.name for entry in pinned.iterdir())
        assert (pinned / "config.json").is_file() and any(
            entry.endswith(".safetensors") for entry in held
        ), f"load_id pins {pinned.name[:8]}, which holds only {held}"
    assert load_id == "Org/Model"


# --- the metadata must describe the snapshot the row hands out ---------------

QUANTIZED_CONFIG = b'{"quantization_config": {"quant_method": "bitsandbytes"}}'
MODEL_CARD = b"---\npipeline_tag: text-generation\nlibrary_name: transformers\n---\n"


@pytest.mark.parametrize(
    "older_files, newer_files, ref, pinned",
    [
        # Reading the newest snapshot while the load id names the payload one describes a directory
        # the row does not hand out, so the quant chip and the type filter judge the model on absent
        # data.
        pytest.param(
            {
                "config.json": QUANTIZED_CONFIG,
                "model.safetensors": b"\0" * 11,
                "README.md": MODEL_CARD,
            },
            {"config.json": b"{}"},
            NEWER,
            True,
            id = "payload-snapshot-supplies-the-row",
        ),
        # The rule stays narrow: with no self-contained payload snapshot there is nothing to scope
        # to, so the newest snapshot still supplies the row.
        pytest.param(
            {"model.safetensors": b"\0" * 11},
            {"config.json": QUANTIZED_CONFIG, "README.md": MODEL_CARD},
            NEWER,
            False,
            id = "newest-snapshot-fallback",
        ),
        # Both revisions are self-contained and ``refs/main`` resolves onto the OLDER one, so the
        # load id stays the repo id and ``from_pretrained`` reads the OLDER directory: describe the
        # row from that one.
        pytest.param(
            {
                "config.json": QUANTIZED_CONFIG,
                "model.safetensors": b"\0" * 11,
                "README.md": MODEL_CARD,
            },
            {"config.json": b"{}", "model.safetensors": b"\0" * 13},
            OLDER,
            False,
            id = "repo-id-resolves-onto-the-older-payload",
        ),
    ],
)
def test_metadata_describes_the_snapshot_the_row_hands_out(
    older_files, newer_files, ref, pinned, tmp_path, monkeypatch
):
    repo_dir = _two_snapshot_repo(
        tmp_path,
        older_files = older_files,
        newer_files = newer_files,
        ref = ref,
    )

    rows = _autoload_rows(tmp_path, monkeypatch)

    expected_load_id = str(repo_dir / "snapshots" / OLDER) if pinned else "Org/Model"
    assert rows[0]["load_id"] == expected_load_id
    assert rows[0].get("quant_method") == "bitsandbytes"
    assert rows[0].get("pipeline_tag") == "text-generation"
    assert rows[0].get("library_name") == "transformers"


# --- the signals paired with the pinned snapshot ------------------------------


def test_a_companion_only_snapshot_is_not_a_gguf_payload(tmp_path, monkeypatch):
    """``MTP/`` drafters are recognisable only from the snapshot-relative path (``huggingface_hub``
    sets ``file_name`` to the bare name for nested files). Matching bare names let a drafter-only
    snapshot win the load id, where the variant lister offers nothing."""
    from hub.utils.gguf import list_local_gguf_variants

    repo_dir = _two_snapshot_repo(
        tmp_path,
        older_files = {"Model-Q4_K_M-00001-of-00002.gguf": b"\0" * 32},
        newer_files = {"MTP/Model-Q8_0-MTP.gguf": b"\0" * 64},
    )

    rows = _autoload_gguf_rows(tmp_path, monkeypatch)

    assert [row["repo_id"] for row in rows] == ["Org/Model"]
    load_dir = Path(rows[0]["load_id"])
    variants, _has_vision = list_local_gguf_variants(str(load_dir))
    assert [v.quant for v in variants], (
        f"load_id {load_dir.name[:8]} offers no quant at all; it holds only a " "companion drafter"
    )
    assert load_dir == repo_dir / "snapshots" / OLDER


@pytest.mark.parametrize(
    "older_files, newer_files, has_vision",
    [
        # A projector fetched on its own lands in a newer snapshot with no main quant, so the row
        # pins the older one and must not OR the flag over.
        pytest.param(
            {"Model-Q4_K_M.gguf": b"\0" * 32},
            {"mmproj-F16.gguf": b"\0" * 64},
            False,
            id = "projector-stranded-in-another-snapshot",
        ),
        # Negative side: colocated with the pinned quant it is reachable, so the flag must survive.
        pytest.param(
            {"Model-Q4_K_M.gguf": b"\0" * 32, "mmproj-F16.gguf": b"\0" * 64},
            {"config.json": b"{}"},
            True,
            id = "projector-beside-the-pinned-quant",
        ),
    ],
)
def test_vision_is_reported_only_from_the_snapshot_the_row_pins(
    older_files, newer_files, has_vision, tmp_path, monkeypatch
):
    """The load path looks for companions no higher than the selected snapshot directory, so a
    projector in any other snapshot is unreachable and must not be advertised."""
    from hub.utils.gguf import list_gguf_variants_from_hf_cache

    repo_dir = _two_snapshot_repo(
        tmp_path,
        older_files = older_files,
        newer_files = newer_files,
    )

    rows = _autoload_gguf_rows(tmp_path, monkeypatch)
    load_dir = Path(rows[0]["load_id"])
    assert load_dir == repo_dir / "snapshots" / OLDER

    listed = list_gguf_variants_from_hf_cache("Org/Model", root = tmp_path)
    assert listed is not None
    variants, reported_vision, _complete = listed
    # The quant on disk stays offered either way; only the flag may move.
    assert [v.quant for v in variants] == ["Q4_K_M"]
    assert reported_vision is has_vision
    # The picker prefers the row's own copy of the flag over the one above.
    assert rows[0]["capabilities"].get("supports_vision") is has_vision
    # The loader's companion search stops at the pinned snapshot.
    assert (load_dir / "mmproj-F16.gguf").is_file() is has_vision


def test_a_repo_root_drafter_still_leaves_a_real_quant_selectable(tmp_path, monkeypatch):
    """The rule must stay narrow: a snapshot holding a drafter *and* a real quant is still a
    payload snapshot."""
    repo_dir = _two_snapshot_repo(
        tmp_path,
        older_files = {"Model-Q4_K_M.gguf": b"\0" * 32},
        newer_files = {"mtp-Model-Q8_0.gguf": b"\0" * 64, "Model-Q8_0.gguf": b"\0" * 128},
    )

    rows = _autoload_gguf_rows(tmp_path, monkeypatch)

    assert Path(rows[0]["load_id"]) == repo_dir / "snapshots" / NEWER


def _write_repo_wide_signal(kind: str, hub_cache: Path) -> None:
    """Either repo-wide partial signal, neither of which records a revision.

    The manifest names a file the pinned older snapshot holds at a different size, as a revision
    that renamed or resized its weights leaves behind.
    """
    from hub.utils import download_manifest

    if kind == "marker":
        download_manifest.write_cancel_marker(
            "model", "Org/Model", None, "http", hub_cache = hub_cache
        )
        return
    download_manifest.write_manifest(
        "model",
        "Org/Model",
        None,
        [download_manifest.ExpectedFile("model.safetensors", 99)],
        "http",
        hub_cache = hub_cache,
    )


@pytest.mark.parametrize("signal", ["marker", "manifest"])
@pytest.mark.parametrize(
    "newer_files, ref, advertised, partial",
    [
        # The signal belongs to the newest snapshot while the row advertises an older, complete one,
        # so inheriting it turns ``can_chat`` off wrongly.
        pytest.param({"config.json": b"{}"}, NEWER, OLDER, False, id = "pinned-older-snapshot"),
        # Negative side: the row advertises the newest snapshot, which the signal does describe. No
        # ``refs/main`` (a commit-pinned fetch) carries no evidence either way.
        pytest.param(
            {"config.json": b"{}", "model.safetensors": b"\0" * 13},
            None,
            NEWER,
            True,
            id = "advertised-snapshot",
        ),
        # A ``refs/main`` naming a commit with no directory does carry evidence: it is rewritten
        # before the first file lands, so that attempt left none.
        pytest.param(
            {"config.json": b"{}", "model.safetensors": b"\0" * 13},
            UPSTREAM_HEAD,
            NEWER,
            False,
            id = "unmaterialised-attempt",
        ),
        # ``refs/main`` resolves onto the OLDER payload snapshot while the newer one is
        # self-contained too, so the load reads the OLDER directory while the signal belongs to the
        # newest snapshot.
        pytest.param(
            {"config.json": b"{}", "model.safetensors": b"\0" * 13},
            OLDER,
            "Org/Model",
            False,
            id = "repo-id-resolves-onto-the-older-payload",
        ),
    ],
)
def test_repo_wide_partial_signals_are_charged_to_the_newest_snapshot(
    signal, newer_files, ref, advertised, partial, tmp_path, monkeypatch
):
    """A cancel marker and a repo-wide manifest both record only the *last* attempt (the marker is
    cleared at every download start and on success; the manifest is overwritten), so both belong to
    the newest snapshot and neither may be verified against an older revision's payload."""
    repo_dir = _two_snapshot_repo(
        tmp_path,
        older_files = {"config.json": b"{}", "model.safetensors": b"\0" * 11},
        newer_files = newer_files,
        ref = ref,
    )
    _write_repo_wide_signal(signal, tmp_path)

    rows = _autoload_rows(tmp_path, monkeypatch)

    # *advertised* is a commit whose snapshot the row pins, or the repo id when the row keeps it and
    # lets ``refs/main`` resolve the directory.
    expected_load_id = (
        advertised if advertised == "Org/Model" else str(repo_dir / "snapshots" / advertised)
    )
    assert rows[0]["load_id"] == expected_load_id
    assert rows[0].get("partial") is partial
    assert rows[0]["capabilities"].get("can_chat") is not partial


def test_gguf_partial_is_judged_against_the_snapshot_the_row_advertises(tmp_path, monkeypatch):
    """The GGUF row picked its payload snapshot after computing ``partial``, a walk over the repo's
    blobs plus the newest snapshot, so an interrupted re-download flipped ``can_chat`` off for the
    older complete quant."""
    from hub.utils.gguf import list_local_gguf_variants

    repo_dir = _two_snapshot_repo(
        tmp_path,
        older_files = {"Model-Q4_K_M.gguf": b"\0" * 32},
        newer_files = {"Model-Q8_0-00001-of-00002.gguf": b"\0" * 16},
    )
    (repo_dir / "blobs" / ("a" * 40 + ".incomplete")).write_bytes(b"\0" * 3)

    rows = _autoload_gguf_rows(tmp_path, monkeypatch)

    load_dir = Path(rows[0]["load_id"])
    assert load_dir == repo_dir / "snapshots" / OLDER
    variants, _has_vision = list_local_gguf_variants(str(load_dir))
    assert [v.quant for v in variants] == ["Q4_K_M"]
    assert rows[0].get("partial") is False
    assert rows[0]["capabilities"].get("can_chat") is True


def test_a_gguf_download_interrupted_in_its_own_snapshot_is_still_partial(tmp_path, monkeypatch):
    """Negative side of the same rule: with no complete quant anywhere the row falls back to the
    newest snapshot, the ``.incomplete`` blob does belong to it, and the row must still be
    partial."""
    repo_dir = _two_snapshot_repo(
        tmp_path,
        older_files = {"Model-Q4_K_M-00001-of-00002.gguf": b"\0" * 32},
        newer_files = {"Model-Q8_0-00001-of-00002.gguf": b"\0" * 16},
        ref = None,
    )
    (repo_dir / "blobs" / ("a" * 40 + ".incomplete")).write_bytes(b"\0" * 3)

    rows = _autoload_gguf_rows(tmp_path, monkeypatch)

    assert Path(rows[0]["load_id"]) == repo_dir / "snapshots" / NEWER
    assert rows[0].get("partial") is True
    assert rows[0]["capabilities"].get("can_chat") is False


@pytest.mark.parametrize(
    "older_files, partial",
    [
        # Half a split quant and no manifest or marker: the ``.incomplete`` blob is the only
        # evidence, so the dangling ref must not clear it.
        pytest.param(
            {"Model-Q4_K_M-00001-of-00002.gguf": b"\0" * 32},
            True,
            id = "pinned-snapshot-holds-half-a-split-quant",
        ),
        # Negative side: the pinned snapshot serves the whole quant, so the unmaterialised attempt
        # is charged to nothing and the row stays chattable.
        pytest.param(
            {
                "Model-Q4_K_M-00001-of-00002.gguf": b"\0" * 32,
                "Model-Q4_K_M-00002-of-00002.gguf": b"\0" * 32,
            },
            False,
            id = "pinned-snapshot-holds-the-whole-quant",
        ),
    ],
)
def test_a_dangling_ref_keeps_a_legacy_partial_signal_for_a_broken_snapshot(
    older_files, partial, tmp_path, monkeypatch
):
    """A legacy interrupted GGUF download predates the manifest, so its only trace is an
    ``.incomplete`` blob. A later update rewrote ``refs/main`` and fetched no file, making the ref
    dangle; suppressing every repo-wide signal on sight then reported the row chattable."""
    repo_dir = _two_snapshot_repo(
        tmp_path,
        older_files = older_files,
        newer_files = {"config.json": b"{}"},
        ref = UPSTREAM_HEAD,
    )
    (repo_dir / "blobs" / ("a" * 40 + ".incomplete")).write_bytes(b"\0" * 3)

    rows = _autoload_gguf_rows(tmp_path, monkeypatch)

    assert Path(rows[0]["load_id"]) == repo_dir / "snapshots" / OLDER
    assert rows[0].get("partial") is partial
    assert rows[0]["capabilities"].get("can_chat") is not partial


@pytest.mark.parametrize(
    "older_files, partial",
    [
        # The safetensors half of the case above: a shard names the set's total, so half a set is
        # provable from the directory alone.
        pytest.param(
            {"config.json": b"{}", "model-00001-of-00002.safetensors": b"\0" * 32},
            True,
            id = "pinned-snapshot-holds-half-a-sharded-set",
        ),
        # Negative side: the whole set is here, so the unmaterialised attempt is charged to nothing
        # and the row stays chattable.
        pytest.param(
            {
                "config.json": b"{}",
                "model-00001-of-00002.safetensors": b"\0" * 32,
                "model-00002-of-00002.safetensors": b"\0" * 32,
            },
            False,
            id = "pinned-snapshot-holds-the-whole-sharded-set",
        ),
        # Nothing names a total, so there is no proof of breakage: this is the #7374 shape and it
        # must keep loading from disk.
        pytest.param(
            {"config.json": b"{}", "model.safetensors": b"\0" * 32},
            False,
            id = "pinned-snapshot-holds-an-unsharded-payload",
        ),
        # from_pretrained loads one family, so an interrupted alternative checkpoint family beside a
        # complete one must not take auto-load away.
        pytest.param(
            {
                "config.json": b"{}",
                "model.safetensors": b"\0" * 32,
                "pytorch_model-00001-of-00002.bin": b"\0" * 32,
            },
            False,
            id = "pinned-snapshot-holds-a-whole-family-beside-a-broken-one",
        ),
        pytest.param(
            {
                "config.json": b"{}",
                "model-00001-of-00002.safetensors": b"\0" * 32,
                "model-00002-of-00002.safetensors": b"\0" * 32,
                "pytorch_model-00001-of-00003.bin": b"\0" * 32,
            },
            False,
            id = "pinned-snapshot-holds-a-whole-sharded-family-beside-a-broken-one",
        ),
        # The half-fetched set above stays proof of breakage: neither a training artefact nor an
        # adapter is a runnable base weight family.
        pytest.param(
            {
                "config.json": b"{}",
                "model-00001-of-00002.safetensors": b"\0" * 32,
                "training_args.bin": b"\0" * 8,
                "optimizer.bin": b"\0" * 8,
            },
            True,
            id = "half-a-sharded-set-beside-training-artefacts",
        ),
        pytest.param(
            {
                "config.json": b"{}",
                "model-00001-of-00002.safetensors": b"\0" * 32,
                "adapter_model.safetensors": b"\0" * 8,
            },
            True,
            id = "half-a-sharded-set-beside-an-adapter",
        ),
        # A COMPLETE auxiliary set is not a runnable base family either, so it cannot stand in for
        # the torn base shards beside it.
        pytest.param(
            {
                "config.json": b"{}",
                "model-00001-of-00002.safetensors": b"\0" * 32,
                "adapter_model-00001-of-00002.safetensors": b"\0" * 8,
                "adapter_model-00002-of-00002.safetensors": b"\0" * 8,
            },
            True,
            id = "half-a-sharded-set-beside-a-whole-sharded-adapter",
        ),
        pytest.param(
            {
                "config.json": b"{}",
                "model-00001-of-00002.safetensors": b"\0" * 32,
                "optimizer-00001-of-00002.bin": b"\0" * 8,
                "optimizer-00002-of-00002.bin": b"\0" * 8,
            },
            True,
            id = "half-a-sharded-set-beside-a-whole-sharded-optimizer",
        ),
    ],
)
def test_a_dangling_ref_keeps_a_legacy_partial_signal_for_a_half_fetched_snapshot(
    older_files, partial, tmp_path, monkeypatch
):
    """The same bytes on disk went partial while ``refs/main`` resolved, then chattable once a later
    attempt rewrote the ref and materialised no directory, so the row offered unfetched shards."""
    repo_dir = _two_snapshot_repo(
        tmp_path,
        older_files = older_files,
        newer_files = {"config.json": b"{}"},
        ref = UPSTREAM_HEAD,
    )
    (repo_dir / "blobs" / ("a" * 40 + ".incomplete")).write_bytes(b"\0" * 3)

    rows = _autoload_rows(tmp_path, monkeypatch)

    assert Path(rows[0]["load_id"]) == repo_dir / "snapshots" / OLDER
    assert rows[0].get("partial") is partial
    assert (rows[0].get("capabilities") or {}).get("can_chat") is not partial


@pytest.mark.parametrize(
    "older_files, partial",
    [
        # Half a sharded set and NO other trace: no manifest, marker, ``.incomplete`` blob or broken
        # symlink, as a cancelled fetch that cleaned its blobs up, or a copied cache, leaves.
        pytest.param(
            {"config.json": b"{}", "model-00001-of-00002.safetensors": b"\0" * 32},
            True,
            id = "half-a-sharded-set-and-no-other-trace",
        ),
        # Negative side, and #7374's own shape: the payload is whole, so the recovered row must load
        # from disk rather than refetch.
        pytest.param(
            {
                "config.json": b"{}",
                "model-00001-of-00002.safetensors": b"\0" * 32,
                "model-00002-of-00002.safetensors": b"\0" * 32,
            },
            False,
            id = "a-whole-sharded-set-and-no-other-trace",
        ),
        # Nothing names a total, so nothing proves breakage and the row loads.
        pytest.param(
            {"config.json": b"{}", "model.safetensors": b"\0" * 32},
            False,
            id = "an-unsharded-payload-and-no-other-trace",
        ),
        # One whole family beside a torn one still loads, exactly as it does when an ``.incomplete``
        # blob is present.
        pytest.param(
            {
                "config.json": b"{}",
                "model.safetensors": b"\0" * 32,
                "pytorch_model-00001-of-00002.bin": b"\0" * 32,
            },
            False,
            id = "a-whole-family-beside-a-torn-one-and-no-other-trace",
        ),
        # A complete auxiliary set does not vouch for torn base shards either.
        pytest.param(
            {
                "config.json": b"{}",
                "model-00001-of-00002.safetensors": b"\0" * 32,
                "adapter_model-00001-of-00002.safetensors": b"\0" * 8,
                "adapter_model-00002-of-00002.safetensors": b"\0" * 8,
            },
            True,
            id = "half-a-base-set-beside-a-whole-adapter-and-no-other-trace",
        ),
    ],
)
def test_a_recovered_snapshot_short_a_shard_is_partial_with_no_other_signal(
    older_files, partial, tmp_path, monkeypatch
):
    """The recovery is what puts this row on screen at all (``scan_cache_dir`` drops the repo), so a
    snapshot it restores must be judged on its own contents when the interrupted attempt left
    nothing else behind, or the row advertises ``can_chat`` for a payload short a shard."""
    repo_dir = _two_snapshot_repo(
        tmp_path,
        older_files = older_files,
        newer_files = {"config.json": b"{}"},
        ref = UPSTREAM_HEAD,
    )
    # No .incomplete blob, no marker, no manifest: the point of this case.
    assert not any((repo_dir / "blobs").iterdir())

    rows = _autoload_rows(tmp_path, monkeypatch)

    assert Path(rows[0]["load_id"]) == repo_dir / "snapshots" / OLDER
    assert rows[0].get("partial") is partial
    assert (rows[0].get("capabilities") or {}).get("can_chat") is not partial


def test_a_resolving_ref_is_not_judged_on_the_recovery_walk(tmp_path, monkeypatch):
    """The new signal is scoped to the rows the recovery adds. With ``refs/main`` naming a snapshot
    on disk the repo is one ``scan_cache_dir`` already returns, and its answer is left alone."""
    repo_dir = _two_snapshot_repo(
        tmp_path,
        older_files = {
            "config.json": b"{}",
            "model-00001-of-00002.safetensors": b"\0" * 32,
        },
        newer_files = {"config.json": b"{}"},
        ref = OLDER,
    )

    assert inventory_scan.default_ref_snapshot(repo_dir) is not None
    rows = _autoload_rows(tmp_path, monkeypatch)

    # The ref resolves, so the load id stays the repo id and the row keeps the answer it had before
    # the recovery branch existed.
    assert rows[0]["load_id"] == "Org/Model"
    assert rows[0].get("partial") is False


@pytest.mark.parametrize("signal", ["marker", "manifest"])
def test_an_update_that_never_materialised_leaves_the_cached_payload_chattable(
    signal, tmp_path, monkeypatch
):
    """The recovered row's own scenario, and why it must not arrive partial.

    ``snapshot_download`` rewrites ``refs/main`` before fetching a byte and the manifest earlier
    still, so an update interrupted before the first file leaves the previous complete snapshot as
    the only payload under a ref that resolves nowhere: the state this branch recovers rows from."""
    repo_dir = _build_repo(tmp_path, ref = UPSTREAM_HEAD)
    snapshot = repo_dir / "snapshots" / SNAPSHOT
    (snapshot / "config.json").write_text("{}", encoding = "utf-8")
    _write_repo_wide_signal(signal, tmp_path)

    rows = _autoload_rows(tmp_path, monkeypatch)

    assert Path(rows[0]["load_id"]) == snapshot
    assert rows[0].get("partial") is False
    assert rows[0]["capabilities"].get("can_chat") is True


@pytest.mark.parametrize(
    "older_files, newer_files, ref, advertised, partial",
    [
        # A re-download that stops before materialising its snapshot leaves a manifest of the NEW
        # revision's files, which no rename or resize survives.
        pytest.param(
            {"Model-Q4_K_M.gguf": b"\0" * 32},
            {"config.json": b"{}"},
            NEWER,
            OLDER,
            False,
            id = "pinned-older-snapshot",
        ),
        # Negative side: the quant the manifest names is not complete under the pinned snapshot, so
        # the manifest is all that can judge it.
        pytest.param(
            {"config.json": b"{}"},
            {"Model-Q4_K_M.gguf": b"\0" * 32},
            None,
            NEWER,
            True,
            id = "advertised-snapshot",
        ),
    ],
)
def test_a_gguf_variant_manifest_is_scoped_to_the_snapshot_the_row_pins(
    older_files, newer_files, ref, advertised, partial, tmp_path, monkeypatch
):
    from hub.utils import download_manifest

    repo_dir = _two_snapshot_repo(
        tmp_path,
        older_files = older_files,
        newer_files = newer_files,
        ref = ref,
    )
    download_manifest.write_manifest(
        "model",
        "Org/Model",
        "Q4_K_M",
        [download_manifest.ExpectedFile("Model-Q4_K_M.gguf", 999)],
        "http",
        hub_cache = tmp_path,
    )

    rows = _autoload_gguf_rows(tmp_path, monkeypatch)

    assert Path(rows[0]["load_id"]) == repo_dir / "snapshots" / advertised
    assert rows[0].get("partial") is partial
    assert rows[0]["capabilities"].get("can_chat") is not partial


def test_a_gguf_variant_marker_from_a_newer_attempt_does_not_disable_the_pinned_quant(
    tmp_path, monkeypatch
):
    """The GGUF twin of the repo-wide marker rule. A cancel marker is keyed by (repo, variant) with
    no revision, so cancelling a re-download of a held quant marked the complete copy broken."""
    from hub.utils import download_manifest
    from hub.utils.gguf import list_local_gguf_variants

    repo_dir = _two_snapshot_repo(
        tmp_path,
        older_files = {"Model-Q4_K_M.gguf": b"\0" * 32},
        newer_files = {"config.json": b"{}"},
        ref = NEWER,
    )
    download_manifest.write_cancel_marker(
        "model", "Org/Model", "Q4_K_M", "http", hub_cache = tmp_path
    )

    rows = _autoload_gguf_rows(tmp_path, monkeypatch)

    load_dir = Path(rows[0]["load_id"])
    assert load_dir == repo_dir / "snapshots" / OLDER
    # The quant the marker names resolves under the load id.
    assert [v.quant for v in list_local_gguf_variants(str(load_dir))[0]] == ["Q4_K_M"]
    assert rows[0].get("partial") is False
    assert rows[0]["capabilities"].get("can_chat") is True


def test_a_gguf_variant_marker_against_the_advertised_snapshot_is_still_partial(
    tmp_path, monkeypatch
):
    """Negative side of the same rule: when the row advertises the newest snapshot the marker does
    describe the attempt that wrote it, so the quant stays broken and the row keeps its resume."""
    from hub.utils import download_manifest

    repo_dir = _two_snapshot_repo(
        tmp_path,
        older_files = {"config.json": b"{}"},
        newer_files = {"Model-Q4_K_M.gguf": b"\0" * 32},
        ref = None,
    )
    download_manifest.write_cancel_marker(
        "model", "Org/Model", "Q4_K_M", "http", hub_cache = tmp_path
    )

    rows = _autoload_gguf_rows(tmp_path, monkeypatch)

    assert Path(rows[0]["load_id"]) == repo_dir / "snapshots" / NEWER
    assert rows[0].get("partial") is True
    assert rows[0]["capabilities"].get("can_chat") is False


def test_a_marker_for_another_quant_still_leaves_the_pinned_one_chattable(tmp_path, monkeypatch):
    """The Q8+Q4 mixed-state rule holds across snapshots: a cancelled quant the load id does not
    resolve must not veto the clean one."""
    from hub.utils import download_manifest

    repo_dir = _two_snapshot_repo(
        tmp_path,
        older_files = {"Model-Q4_K_M.gguf": b"\0" * 32},
        newer_files = {"config.json": b"{}"},
        ref = NEWER,
    )
    download_manifest.write_cancel_marker("model", "Org/Model", "Q8_0", "http", hub_cache = tmp_path)

    rows = _autoload_gguf_rows(tmp_path, monkeypatch)

    assert Path(rows[0]["load_id"]) == repo_dir / "snapshots" / OLDER
    assert rows[0].get("partial") is False
    assert rows[0]["capabilities"].get("can_chat") is True


# --- one snapshot ordering, so the row and the picker cannot disagree ---------


def test_equal_mtime_snapshots_order_the_same_way_whatever_the_iteration_order(tmp_path):
    """Selection ran off directory mtime alone, which is not an order.

    Candidates reach the row through a ``frozenset`` and the variant walk through ``iterdir()``, so
    on equal mtimes the two picked different directories."""
    from hub.services.models import cache_inventory

    first = tmp_path / "snapshots" / OLDER
    second = tmp_path / "snapshots" / NEWER
    for snapshot in (first, second):
        snapshot.mkdir(parents = True)
        os.utime(snapshot, (1_700_000_000, 1_700_000_000))

    forward = cache_inventory._newest_snapshot_dir([first, second])
    backward = cache_inventory._newest_snapshot_dir([second, first])

    assert forward == backward, "the pick moved with the order the candidates arrived in"
    assert forward == second.resolve()


def test_the_row_and_the_picker_agree_on_equal_mtime_snapshots(tmp_path, monkeypatch):
    """End to end at the tie: whichever snapshot wins, the quants offered as downloaded must resolve
    under the load id the row hands out. On a coarse timestamp filesystem the row pinned one
    snapshot while the picker offered the other one's quant."""
    from hub.utils.gguf import iter_hf_cache_snapshots, list_local_gguf_variants

    repo_dir = _two_snapshot_repo(
        tmp_path,
        older_files = {"Model-Q4_K_M.gguf": b"\0" * 32},
        newer_files = {"Model-Q8_0.gguf": b"\0" * 64},
    )
    for commit in (OLDER, NEWER):
        os.utime(repo_dir / "snapshots" / commit, (1_700_000_000, 1_700_000_000))

    rows = _autoload_gguf_rows(tmp_path, monkeypatch)

    load_dir = Path(rows[0]["load_id"])
    offered = _local_gguf_variants_for_autoload(rows[0], tmp_path)
    resolvable = {v.quant for v in list_local_gguf_variants(str(load_dir))[0] if v.quant}
    assert set(offered) <= resolvable, (
        f"auto-load is offered {sorted(offered)} but load_id {load_dir.name[:8]} "
        f"resolves only {sorted(resolvable)}"
    )
    # Both selectors head the same list, so the tie cannot split them again.
    assert load_dir == next(iter(iter_hf_cache_snapshots("Org/Model", root = tmp_path)))


def test_vision_does_not_travel_between_two_cache_roots(tmp_path, monkeypatch):
    """The same repo can sit in the active hub cache and in a previous one.

    One row survives the merge and only its directory is loaded, so carrying the loser's projector
    flag over put a vision badge on a text-only load."""
    from types import SimpleNamespace

    from hub.services.models import cache_inventory

    active = tmp_path / "active"
    previous = tmp_path / "previous"
    for root, files in (
        (active, {"Model-Q4_K_M.gguf": b"\0" * 32}),
        (previous, {"Model-Q4_K_M.gguf": b"\0" * 32, "mmproj-F16.gguf": b"\0" * 64}),
    ):
        snapshot = root / "models--Org--Model" / "snapshots" / SNAPSHOT
        snapshot.mkdir(parents = True)
        (root / "models--Org--Model" / "refs").mkdir(parents = True)
        (root / "models--Org--Model" / "refs" / "main").write_text(SNAPSHOT, encoding = "utf-8")
        for name, payload in files.items():
            (snapshot / name).write_bytes(payload)

    monkeypatch.setattr(inventory_scan, "hf_cache_roots", lambda: [active, previous])
    monkeypatch.setattr(
        "utils.hf_cache_settings.get_hf_cache_paths",
        lambda: SimpleNamespace(hub_cache = active),
    )
    inventory_scan.invalidate_hf_cache_scans()
    try:
        rows = cache_inventory._scan_cached_gguf()
    finally:
        inventory_scan.invalidate_hf_cache_scans()

    assert [row["repo_id"] for row in rows] == ["Org/Model"]
    assert rows[0]["active_cache"] is True
    # Behavioural anchor: the row the picker shows loads out of the active root, and there is no
    # projector there.
    assert not (active / "models--Org--Model" / "snapshots" / SNAPSHOT / "mmproj-F16.gguf").exists()
    assert rows[0]["capabilities"].get("supports_vision") is False


# --- the chokepoints, so a new signal cannot pick its own snapshot ------------

_BACKEND = Path(__file__).resolve().parents[1]
# Every helper the per-repo scan may hand the whole repo to. Each aggregates across revisions on
# purpose (bytes, mtimes, the payload snapshot set), so a new name here is a new repo-wide signal on
# a row that loads out of one directory, and has to be argued for.
_REPO_WIDE_HELPERS = frozenset(
    {
        "_cache_inventory_fields",
        "_repo_gguf_last_modified",
        "_repo_gguf_payload_snapshots",
        "_repo_gguf_size_bytes",
        "_repo_has_gguf_files",
        "_repo_non_gguf_model_payload",
        "getattr",
    }
)
# Only the shared ordering key may read a snapshot directory's mtime; _blob_mtime reads a blob's,
# which orders nothing.
_MTIME_READERS = {
    "hub/utils/hf_cache_state.py": frozenset({"snapshot_selection_key"}),
    "hub/utils/gguf.py": frozenset(),
    "hub/services/models/cache_inventory.py": frozenset({"_blob_mtime"}),
    # Mirrors what huggingface_hub records per revision; it selects nothing.
    "hub/utils/inventory_scan.py": frozenset({"_recover_repo_hidden_by_dangling_refs"}),
    # The compatibility routes, listed so the two snapshot selectors here cannot reintroduce their
    # own mtime reads. The names left rank plain directories (./models, LM Studio, Ollama) or read a
    # repo dir's or blob's mtime; none picks a snapshot.
    "routes/models.py": frozenset(
        {
            "_blob_mtime",
            "_scan_hf_cache",
            "_scan_lmstudio_dir",
            "_scan_models_dir",
            "_scan_ollama_dir",
        }
    ),
}


def _function_defs(path: Path) -> dict:
    import ast
    tree = ast.parse(path.read_text(encoding = "utf-8"))
    return {
        node.name: node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


@pytest.mark.parametrize("scan", ["_scan_cached_gguf", "_scan_cached_models"])
def test_the_scan_loop_cannot_advertise_a_signal_it_did_not_scope(scan):
    """A row's flags come from ``_cache_inventory_fields``, the sole producer, which has exactly one
    snapshot in scope. Setting one afterwards, or walking ``repo_info.revisions`` in the loop, puts
    the whole repo back in scope."""
    import ast

    node = _function_defs(_BACKEND / "hub/services/models/cache_inventory.py")[scan]

    mutations = [
        ast.unparse(target)
        for stmt in ast.walk(node)
        if isinstance(stmt, ast.Assign)
        for target in stmt.targets
        if isinstance(target, ast.Subscript) and "capabilities" in ast.unparse(target)
    ]
    assert mutations == [], f"{scan} sets a capability outside _cache_inventory_fields: {mutations}"

    revision_reads = [
        ast.unparse(sub)
        for sub in ast.walk(node)
        if isinstance(sub, ast.Attribute)
        and sub.attr == "revisions"
        and isinstance(sub.value, ast.Name)
        and sub.value.id == "repo_info"
    ]
    assert revision_reads == [], f"{scan} walks every revision itself: {revision_reads}"

    handed_off = sorted(
        {
            ast.unparse(call.func)
            for call in ast.walk(node)
            if isinstance(call, ast.Call)
            and any(
                isinstance(arg, ast.Name) and arg.id == "repo_info"
                for arg in [*call.args, *(kw.value for kw in call.keywords)]
            )
        }
    )
    assert (
        set(handed_off) <= _REPO_WIDE_HELPERS
    ), f"{scan} hands the whole repo to {sorted(set(handed_off) - _REPO_WIDE_HELPERS)}"


@pytest.mark.parametrize("module, allowed", sorted(_MTIME_READERS.items()))
def test_only_the_shared_key_orders_snapshots_by_mtime(module, allowed):
    """Two selectors with their own mtime read disagree on equal timestamps."""
    import ast

    readers = {
        name
        for name, node in _function_defs(_BACKEND / module).items()
        if any(isinstance(sub, ast.Attribute) and sub.attr == "st_mtime" for sub in ast.walk(node))
    }
    assert readers == set(
        allowed
    ), f"{module} reads a snapshot mtime outside snapshot_selection_key: {sorted(readers)}"


# --- review-round regressions -------------------------------------------------


def _repo_with(
    cache_root: Path,
    snapshots: dict,
    refs: dict,
    name: str = "models--Org--Model",
):
    """A cache repo with arbitrary per-snapshot contents and ref targets."""
    repo_dir = cache_root / name
    (repo_dir / "blobs").mkdir(parents = True, exist_ok = True)
    (repo_dir / "refs").mkdir(parents = True, exist_ok = True)
    for commit, files in snapshots.items():
        snapshot = repo_dir / "snapshots" / commit
        snapshot.mkdir(parents = True, exist_ok = True)
        for rel, payload in files.items():
            (snapshot / rel).parent.mkdir(parents = True, exist_ok = True)
            (snapshot / rel).write_bytes(payload)
    for ref_name, commit in refs.items():
        (repo_dir / "refs" / ref_name).write_text(commit, encoding = "utf-8")
    return repo_dir


def test_a_secondary_dangling_ref_still_judges_the_recovered_snapshot(tmp_path, monkeypatch):
    """refs/main resolves while refs/stale dangles, so recovery fires but the default-ref test does
    not. The pinned snapshot is short a shard with no marker, manifest or .incomplete blob, so its
    contents are the only evidence: the guard has to key on ANY dangling ref, not just refs/main."""
    _repo_with(
        tmp_path,
        snapshots = {
            SNAPSHOT: {
                "config.json": b'{"model_type":"llama"}',
                "model-00001-of-00002.safetensors": b"\0" * 256,
            }
        },
        refs = {"main": SNAPSHOT, "stale": UPSTREAM_HEAD},
    )
    rows = _autoload_rows(tmp_path, monkeypatch)
    assert [r["repo_id"] for r in rows] == ["Org/Model"]
    assert rows[0]["partial"] is True
    assert rows[0]["capabilities"]["can_chat"] is False


def test_a_torn_payload_stays_partial_when_the_ref_resolves(tmp_path, monkeypatch):
    """Excusing a non-newest snapshot from the repo-wide signals is the point of the attribution,
    but it only holds while that snapshot can serve the row. A torn one has no other evidence it is
    unfinished, so an interrupted attempt on another revision must not make it chattable."""
    repo_dir = _two_snapshot_repo(
        tmp_path,
        older_files = {
            "config.json": b'{"model_type":"llama"}',
            "model-00001-of-00002.safetensors": b"\0" * 256,
        },
        newer_files = {"README.md": b"probe"},
        ref = OLDER,
    )
    (repo_dir / "blobs" / "deadbeef.incomplete").write_bytes(b"\0" * 8)
    rows = _autoload_rows(tmp_path, monkeypatch)
    assert rows[0]["partial"] is True
    assert rows[0]["capabilities"]["can_chat"] is False


def test_a_whole_payload_under_a_resolving_ref_is_still_chattable(tmp_path, monkeypatch):
    """Negative control for the test above: same shape, but the pinned payload is whole."""
    repo_dir = _two_snapshot_repo(
        tmp_path,
        older_files = {
            "config.json": b'{"model_type":"llama"}',
            "model-00001-of-00002.safetensors": b"\0" * 256,
            "model-00002-of-00002.safetensors": b"\0" * 256,
        },
        newer_files = {"README.md": b"probe"},
        ref = OLDER,
    )
    (repo_dir / "blobs" / "deadbeef.incomplete").write_bytes(b"\0" * 8)
    rows = _autoload_rows(tmp_path, monkeypatch)
    assert rows[0]["partial"] is False
    assert rows[0]["capabilities"]["can_chat"] is True


def test_a_payload_split_across_snapshots_is_not_advertised_runnable(tmp_path, monkeypatch):
    """The payload flags are OR-ed over every revision, so config.json in one snapshot and the
    weights in another look runnable while no single directory can serve the row. With no
    self-contained snapshot and no refs/main, from_pretrained(repo_id) resolves to nothing."""
    _repo_with(
        tmp_path,
        snapshots = {
            SNAPSHOT: {"model.safetensors": b"\0" * 256},
            NEWER: {"config.json": b'{"model_type":"llama"}'},
        },
        refs = {"main": UPSTREAM_HEAD},
        name = "models--Org--Split",
    )
    rows = _autoload_rows(tmp_path, monkeypatch)
    assert [r["repo_id"] for r in rows] == ["Org/Split"]
    assert rows[0]["partial"] is True
    assert rows[0]["capabilities"]["can_chat"] is False


def test_a_complete_quant_family_does_not_vouch_for_a_torn_sibling(tmp_path):
    """One quant label can cover several shard families and the lister offers only the
    lexicographically first file under it, so grouping on the shard total alone let the complete B
    family mark the torn A file downloadable."""
    snapshot = tmp_path / "snap"
    snapshot.mkdir()
    for name in (
        "A-Q4_K_M-00001-of-00002.gguf",
        "B-Q4_K_M-00001-of-00002.gguf",
        "B-Q4_K_M-00002-of-00002.gguf",
    ):
        (snapshot / name).write_bytes(b"\0" * 64)
    from hub.utils.gguf import list_local_gguf_variants

    advertised = {v.filename for v in list_local_gguf_variants(str(snapshot))[0]}
    assert advertised == {"A-Q4_K_M-00001-of-00002.gguf"}
    assert inventory_scan._completed_gguf_variants(snapshot) == set()


def test_a_big_endian_build_does_not_vouch_for_a_torn_little_endian_quant(tmp_path):
    """The lister never offers a big-endian build, so it must not make the little-endian quant of
    the same name look complete either."""
    snapshot = tmp_path / "snap"
    snapshot.mkdir()
    (snapshot / "Model-Q8_0-BE.gguf").write_bytes(b"\0" * 64)
    (snapshot / "Model-Q8_0-00001-of-00002.gguf").write_bytes(b"\0" * 64)
    assert inventory_scan._completed_gguf_variants(snapshot) == set()


def _local_offer(snapshot: Path) -> list:
    """What GET /api/models/gguf-variants reports for a directory path, the shape of a snapshot load
    id."""
    import asyncio

    from hub.services.models import gguf_variants

    response = asyncio.run(
        gguf_variants.get_gguf_variants_response(str(snapshot), prefer_local_cache = True)
    )
    return sorted((v.quant, bool(v.downloaded)) for v in response.variants)


def test_a_torn_quant_is_not_reported_downloaded_under_a_snapshot_load_id(tmp_path):
    """The load id is an absolute snapshot path and get_gguf_variants_response short-circuits on
    is_local_path, so the completeness check has to live at that shared offer site. The torn quant
    stays listed to resume or delete; it just is not offered as ready."""
    snapshot = tmp_path / "snap"
    snapshot.mkdir()
    (snapshot / "Model-Q8_0.gguf").write_bytes(b"\0" * 64)
    (snapshot / "Model-Q4_K_M-00001-of-00002.gguf").write_bytes(b"\0" * 16)
    assert _local_offer(snapshot) == [("Q4_K_M", False), ("Q8_0", True)]


def test_a_big_endian_sibling_does_not_make_a_torn_quant_downloadable(tmp_path):
    """End to end for the completion walk: the lister drops the big-endian build and offers the torn
    little-endian file under the same label, so counting the big-endian one complete marked it
    ready."""
    snapshot = tmp_path / "snap"
    snapshot.mkdir()
    (snapshot / "model-Q4_K_M-be.gguf").write_bytes(b"\0" * 64)
    (snapshot / "model-Q4_K_M-00001-of-00002.gguf").write_bytes(b"\0" * 16)
    assert _local_offer(snapshot) == [("Q4_K_M", False)]


def test_a_resume_only_folder_still_lists_when_nothing_is_complete(tmp_path):
    """A folder holding only a half-fetched download still shows up so it can be resumed or deleted,
    rather than vanishing from the picker."""
    snapshot = tmp_path / "snap"
    snapshot.mkdir()
    (snapshot / "Model-Q4_K_M-00001-of-00002.gguf").write_bytes(b"\0" * 16)
    assert _local_offer(snapshot) == [("Q4_K_M", False)]


def test_a_whole_folder_is_still_reported_downloaded(tmp_path):
    """Negative control: completeness must not make a good folder look broken."""
    snapshot = tmp_path / "snap"
    snapshot.mkdir()
    (snapshot / "Model-Q8_0.gguf").write_bytes(b"\0" * 64)
    (snapshot / "Model-Q4_K_M-00001-of-00002.gguf").write_bytes(b"\0" * 16)
    (snapshot / "Model-Q4_K_M-00002-of-00002.gguf").write_bytes(b"\0" * 16)
    assert _local_offer(snapshot) == [("Q4_K_M", True), ("Q8_0", True)]


def test_the_ignored_cache_entries_track_huggingface_hub(tmp_path):
    """huggingface_hub 1.x skips Thumbs.db and desktop.ini as well as .DS_Store. Hardcoding the
    older set meant an Explorer artefact in snapshots/ made the recovery decline a repo a newer hub
    had dropped for the dangling ref alone."""
    from huggingface_hub.utils import _cache_manager

    upstream = getattr(_cache_manager, "FILES_TO_IGNORE", None)
    if not upstream:
        pytest.skip("this huggingface_hub does not export FILES_TO_IGNORE")
    assert set(inventory_scan._CACHE_ENTRIES_TO_IGNORE) == set(upstream)


def _primary_gguf_predicates(rel_paths: list[str]) -> dict:
    """Both copies of the primary-GGUF classification, fed one revision whose files sit at
    *rel_paths* under the snapshot. huggingface_hub records only the bare file_name, so a companion
    is identifiable from file_path alone."""
    from types import SimpleNamespace

    from hub.services.models import cache_inventory
    from routes import models as models_route

    files = [
        SimpleNamespace(
            file_path = f"/cache/models--Org--Model/snapshots/{SNAPSHOT}/{rel}",
            file_name = rel.rsplit("/", 1)[-1],
            size_on_disk = 64,
            blob_path = None,
            blob_last_modified = 1.0,
        )
        for rel in rel_paths
    ]
    repo_info = SimpleNamespace(
        repo_path = None,
        revisions = [SimpleNamespace(commit_hash = SNAPSHOT, snapshot_path = "/s", files = files)],
    )
    return {
        "inventory": cache_inventory._repo_gguf_size_bytes(repo_info),
        "route": models_route._repo_gguf_size_bytes(repo_info),
    }


@pytest.mark.parametrize(
    "rel_paths",
    [
        ["MTP/drafter-Q4_K_M.gguf"],
        ["mtp-model-Q8_0.gguf"],
        ["mmproj-F16.gguf"],
    ],
)
def test_a_companion_only_repo_is_not_a_gguf_model(rel_paths):
    """An MTP drafter or a vision projector is a companion, never a loadable weight. Counting one
    made the repo a chattable GGUF row with no selectable variant, and hid a real Transformers model
    sharing the repo behind a GGUF that does not exist."""
    assert _primary_gguf_predicates(rel_paths) == {"inventory": 0, "route": 0}


def test_a_companion_beside_a_real_quant_is_not_counted_twice():
    """Negative control: the companion drops out of the size, the primary weight still makes the
    repo a GGUF model."""
    assert _primary_gguf_predicates(["Model-Q4_K_M.gguf", "MTP/drafter-Q4_K_M.gguf"]) == {
        "inventory": 64,
        "route": 64,
    }


def test_the_two_primary_gguf_predicates_agree():
    """routes.models keeps its own copy for the compatibility route. They drifted once already; a
    disagreement means one endpoint lists a repo the other hides."""
    from routes import models as models_route
    from hub.services.models import common

    names = [
        "Model-Q4_K_M.gguf",
        "MTP/drafter-Q4_K_M.gguf",
        "mtp-model-Q8_0.gguf",
        "mmproj-F16.gguf",
        "config.json",
    ]
    assert [models_route._is_main_gguf_filename(n) for n in names] == [
        common._is_main_gguf_filename(n) for n in names
    ]


def test_an_mtp_only_recovered_repo_does_not_become_a_chattable_gguf_row(tmp_path, monkeypatch):
    """End to end for the recovery path. Un-hiding a repo behind a dangling ref must not classify it
    more loosely than a healthy one: with only a drafter in the snapshot there is nothing to
    load."""
    _repo_with(
        tmp_path,
        snapshots = {SNAPSHOT: {"MTP/drafter-Q4_K_M.gguf": b"\0" * 64}},
        refs = {"main": UPSTREAM_HEAD},
    )
    assert _autoload_rows(tmp_path, monkeypatch, gguf = True) == []


def test_an_all_incomplete_repo_is_offered_by_repo_id_as_partial(tmp_path, monkeypatch):
    """The lister returns its untrimmed offer when no snapshot holds a whole quant, so the repo
    stays manageable. That fallback carried no completeness set, so every torn quant reached the
    repo-id caller marked ready, and Settings/Agents filters on the per-variant flag."""
    import asyncio
    from types import SimpleNamespace

    from hub.services.models import gguf_variants

    _repo_with(
        tmp_path,
        snapshots = {SNAPSHOT: {"Model-Q4_K_M-00001-of-00002.gguf": b"\0" * 16}},
        refs = {"main": SNAPSHOT},
    )
    monkeypatch.setattr(inventory_scan, "hf_cache_roots", lambda: [tmp_path])
    monkeypatch.setattr(
        "utils.hf_cache_settings.get_hf_cache_paths",
        lambda: SimpleNamespace(hub_cache = tmp_path),
    )
    inventory_scan.invalidate_hf_cache_scans()
    try:
        response = asyncio.run(
            gguf_variants.get_gguf_variants_response("Org/Model", prefer_local_cache = True)
        )
    finally:
        inventory_scan.invalidate_hf_cache_scans()
    assert [(v.quant, bool(v.downloaded), bool(v.partial)) for v in response.variants] == [
        ("Q4_K_M", False, True)
    ]


def test_a_whole_repo_is_still_offered_by_repo_id_as_downloaded(tmp_path, monkeypatch):
    """Negative control for the fallback: passing the completeness set through must not demote a
    repo whose quant is actually whole."""
    import asyncio
    from types import SimpleNamespace

    from hub.services.models import gguf_variants

    _repo_with(
        tmp_path,
        snapshots = {SNAPSHOT: {"Model-Q4_K_M.gguf": b"\0" * 64}},
        refs = {"main": SNAPSHOT},
    )
    monkeypatch.setattr(inventory_scan, "hf_cache_roots", lambda: [tmp_path])
    monkeypatch.setattr(
        "utils.hf_cache_settings.get_hf_cache_paths",
        lambda: SimpleNamespace(hub_cache = tmp_path),
    )
    inventory_scan.invalidate_hf_cache_scans()
    try:
        response = asyncio.run(
            gguf_variants.get_gguf_variants_response("Org/Model", prefer_local_cache = True)
        )
    finally:
        inventory_scan.invalidate_hf_cache_scans()
    assert [(v.quant, bool(v.downloaded), bool(v.partial)) for v in response.variants] == [
        ("Q4_K_M", True, False)
    ]


def _split_payload_rows(tmp_path, monkeypatch, *, where: str, refs: dict) -> list[dict]:
    """A repo whose config.json and weights sit in DIFFERENT snapshots, so no single directory can
    serve a load, placed in the active or in a legacy cache."""
    from types import SimpleNamespace

    from hub.services.models import cache_inventory

    active = tmp_path / "active"
    legacy = tmp_path / "legacy"
    active.mkdir()
    legacy.mkdir()
    _repo_with(
        active if where == "active" else legacy,
        snapshots = {
            SNAPSHOT: {"config.json": b'{"model_type":"llama"}'},
            OLDER: {"model.safetensors": b"\0" * 256},
        },
        refs = refs,
    )
    monkeypatch.setattr(inventory_scan, "hf_cache_roots", lambda: [active, legacy])
    monkeypatch.setattr(
        "utils.hf_cache_settings.get_hf_cache_paths",
        lambda: SimpleNamespace(hub_cache = active),
    )
    inventory_scan.invalidate_hf_cache_scans()
    try:
        return cache_inventory._scan_cached_models()
    finally:
        inventory_scan.invalidate_hf_cache_scans()


@pytest.mark.parametrize("where", ["active", "legacy"])
@pytest.mark.parametrize(
    "ref_label, refs",
    [
        ("dangling", {"main": UPSTREAM_HEAD}),
        ("resolves-to-the-config-half", {"main": SNAPSHOT}),
        ("resolves-to-the-weights-half", {"main": OLDER}),
        ("no-refs", {}),
    ],
)
def test_a_split_payload_is_partial_wherever_it_is_cached(
    where, ref_label, refs, tmp_path, monkeypatch
):
    """The payload flags are OR-ed over revisions, so a repo holding config.json in one snapshot and
    the weights in another reads as runnable while nothing on disk can serve it.

    Neither the cache it sits in nor the state of refs/main changes that. A repo outside the active
    cache is always pinned to an absolute snapshot path, and a resolving ref only ever lands on one
    of the two halves: a directory that could serve the payload would be a payload snapshot."""
    rows = _split_payload_rows(tmp_path, monkeypatch, where = where, refs = refs)
    assert [row["repo_id"] for row in rows] == ["Org/Model"]
    assert rows[0]["partial"] is True
    assert rows[0]["capabilities"]["can_chat"] is False


@pytest.mark.parametrize("where", ["active", "legacy"])
@pytest.mark.parametrize(
    "ref_label, refs", [("dangling", {"main": UPSTREAM_HEAD}), ("resolving", {"main": SNAPSHOT})]
)
def test_a_self_contained_snapshot_is_not_made_partial_by_a_second_one(
    where, ref_label, refs, tmp_path, monkeypatch
):
    """The negative control that matters most: one snapshot holds a whole payload and a second holds
    only weights, so a payload snapshot exists and the row stays chattable."""
    from types import SimpleNamespace

    from hub.services.models import cache_inventory

    active = tmp_path / "active"
    legacy = tmp_path / "legacy"
    active.mkdir()
    legacy.mkdir()
    _repo_with(
        active if where == "active" else legacy,
        snapshots = {
            SNAPSHOT: {
                "config.json": b'{"model_type":"llama"}',
                "model.safetensors": b"\0" * 256,
            },
            OLDER: {"model.safetensors": b"\0" * 256},
        },
        refs = refs,
    )
    monkeypatch.setattr(inventory_scan, "hf_cache_roots", lambda: [active, legacy])
    monkeypatch.setattr(
        "utils.hf_cache_settings.get_hf_cache_paths",
        lambda: SimpleNamespace(hub_cache = active),
    )
    inventory_scan.invalidate_hf_cache_scans()
    try:
        rows = cache_inventory._scan_cached_models()
    finally:
        inventory_scan.invalidate_hf_cache_scans()
    assert [row["repo_id"] for row in rows] == ["Org/Model"]
    assert rows[0]["partial"] is False
    assert rows[0]["capabilities"]["can_chat"] is True
