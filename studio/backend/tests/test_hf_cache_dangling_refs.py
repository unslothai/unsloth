# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A dangling ``refs/<branch>`` must not hide an intact repo from the scan.

``scan_cache_dir`` raises CorruptedCacheException for a repo whose ref names a commit with no
``snapshots/<commit>/`` directory and omits it from ``.repos``, so the model stays visible in the
picker (a plain directory walk) but vanishes from every Hub inventory endpoint chat auto-load reads.
The repair is read-only: refs are written with an unlocked in-place ``write_text``, so no external
process can delete one race-free.
"""

from __future__ import annotations

import dataclasses
import json
import os
import shutil
import stat
import tracemalloc
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
    """A cache repo shaped like HF's, using regular files rather than symlinks: a regular file
    resolves to itself, so the real scanner runs without the Windows symlink privilege."""
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
    """An empty ``HFCacheInfo``: fields are read off the dataclass rather than named, so a release
    adding or dropping one cannot fail this test on a signature it never exercises."""
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
    # routes/models.py does repo_path.parent, so this stays a real Path.
    assert isinstance(repo.repo_path, Path) and repo.repo_path == repo_dir
    assert repo.size_on_disk == 11
    # The recovered revision keeps the identity the snapshot is loadable by.
    revision = next(iter(repo.revisions))
    assert revision.commit_hash == SNAPSHOT
    assert revision.snapshot_path == repo_dir / "snapshots" / SNAPSHOT
    assert {f.file_name for f in revision.files} == {"model.safetensors"}
    # The dangling ref maps to no revision...
    assert revision.refs == frozenset()
    # ...and is still on disk: the repair never writes to the cache.
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
    """One unreadable repo must not stop the others being recovered. Scoped to the recovery pass:
    scan_cache_dir itself raises on an unreadable repo dir, upstream of this code."""
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
    # Recovered rows pin the snapshot: refs/main dangles, so from_pretrained(repo id) misses it.
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


# from_pretrained opens every name this map lists, so they must be real.
def _shard_index(*shards: str) -> bytes:
    return json.dumps(
        {"metadata": {}, "weight_map": {f"w{i}": name for i, name in enumerate(shards)}}
    ).encode()


_SHARD_INDEX = _shard_index("model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors")
_BIN_SHARD_INDEX = _shard_index(
    "pytorch_model-00001-of-00002.bin", "pytorch_model-00002-of-00002.bin"
)
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
    """A repo whose payload sits in the older of two snapshots: a metadata probe against a moved-on
    commit materialises a newer, weightless snapshot beside the download. ``ref = None`` is what a
    commit-pinned fetch leaves, since only a branch or tag gets a ref."""
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
    # Every quant offered as downloaded resolves under the load id, unshadowed by the broken one.
    assert set(offered) <= resolvable, (
        f"auto-load is offered {sorted(offered)} but load_id {load_dir.name[:8]} "
        f"resolves only {sorted(resolvable)}"
    )
    assert offered == ["Q4_K_M"]


@pytest.mark.parametrize(
    "newer_files, listed, offered",
    [
        # With nothing complete the newest snapshot holding quants is reported but not loadable.
        pytest.param(
            {"Model-Q8_0-00001-of-00002.gguf": b"\0" * 16},
            ["Q8_0"],
            [],
            id = "nothing-complete-anywhere",
        ),
        # When that snapshot holds a whole quant too, offering both shadows it.
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
    older snapshot holds only a whole larger quant. Auto-load takes the smallest offered, so skipping
    the newest snapshot spends the attempt on the larger one; only its completed subset is offered,
    so both ends still name one directory."""
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
    # The pair still has to agree on one directory, without offering the interrupted split quant.
    resolvable = {v.quant for v in list_local_gguf_variants(str(load_dir))[0]}
    assert set(offered) <= resolvable, (
        f"auto-load is offered {sorted(offered)} but load_id {load_dir.name[:8]} "
        f"resolves only {sorted(resolvable)}"
    )
    assert "Q8_0" not in offered
    # Pinning the snapshot holding the interrupted download must not flip the row partial.
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
    """Same repo, but with the dangling ``refs/main`` this branch exists for. That arm must not pin
    the fallback newest snapshot, known not to hold the payload: a directory the load cannot use is
    worse than the repo id, which can still complete the config."""
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
        # A pinned directory is only useful if from_pretrained can read weights out of it.
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
        # Reading the newest snapshot while the load id names the payload one judges absent data.
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
        # The rule stays narrow: with no self-contained snapshot the newest still supplies the row.
        pytest.param(
            {"model.safetensors": b"\0" * 11},
            {"config.json": QUANTIZED_CONFIG, "README.md": MODEL_CARD},
            NEWER,
            False,
            id = "newest-snapshot-fallback",
        ),
        # Both revisions are self-contained and refs/main resolves onto the OLDER one.
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
        # A lone projector lands in a newer quantless snapshot, so the row pins the older one.
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


@pytest.mark.parametrize(
    ("projector", "has_vision"),
    [
        pytest.param(b"", False, id = "a-projector-with-nothing-behind-the-name"),
        pytest.param(b"\0" * 128, True, id = "a-projector-with-content"),
    ],
)
def test_an_empty_projector_is_not_vision_support(tmp_path, monkeypatch, projector, has_vision):
    """has_vision came off the filename alone, so an interrupted companion download advertised a
    projector llama.cpp cannot open while the row's own quant stayed whole."""
    from hub.utils.gguf import list_local_gguf_variants

    _repo_with(
        tmp_path,
        snapshots = {
            SNAPSHOT: {"Model-Q4_K_M.gguf": b"\0" * 256, "mmproj-F16.gguf": projector},
        },
        refs = {"main": UPSTREAM_HEAD},
    )
    rows = _autoload_gguf_rows(tmp_path, monkeypatch)
    assert rows[0]["partial"] is False
    assert rows[0]["capabilities"]["can_chat"] is True
    assert rows[0]["capabilities"].get("supports_vision") is has_vision
    # The lister and the row capability read the same walk, so they cannot drift apart.
    snapshot = tmp_path / "models--Org--Model" / "snapshots" / SNAPSHOT
    assert list_local_gguf_variants(str(snapshot))[1] is has_vision


@pytest.mark.parametrize(
    ("files", "torn"),
    [
        pytest.param(
            {"Model-Q4_K_M-00001-of-00002.gguf": b"\0" * 256},
            True,
            id = "the-only-quant-is-half-a-split",
        ),
        pytest.param(
            {
                "Model-Q4_K_M-00001-of-00002.gguf": b"\0" * 256,
                "Model-Q4_K_M-00002-of-00002.gguf": b"\0" * 256,
            },
            False,
            id = "the-split-is-whole",
        ),
        pytest.param(
            {
                "Model-Q8_0.gguf": b"\0" * 256,
                "Model-Q4_K_M-00001-of-00002.gguf": b"\0" * 256,
            },
            False,
            id = "a-whole-quant-beside-the-torn-one",
        ),
        pytest.param({"Model-Q4_K_M.gguf": b"\0" * 256}, False, id = "one-unsplit-quant"),
    ],
)
def test_a_manifestless_torn_quant_is_still_reported(tmp_path, monkeypatch, files, torn):
    """Every other quant signal comes from a manifest, a marker or the completed set, and an
    interrupted attempt leaves none of those, so the shards are the only evidence of a torn split."""
    _repo_with(tmp_path, snapshots = {SNAPSHOT: files}, refs = {"main": UPSTREAM_HEAD})
    rows = _autoload_gguf_rows(tmp_path, monkeypatch)
    assert rows[0]["partial"] is torn
    assert rows[0]["capabilities"]["can_chat"] is not torn


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
    """Either repo-wide partial signal, neither of which records a revision. The manifest names a
    file the pinned older snapshot holds at a different size, as a renamed revision leaves behind."""
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
        # The signal belongs to the newest snapshot while the row advertises an older, complete one.
        pytest.param({"config.json": b"{}"}, NEWER, OLDER, False, id = "pinned-older-snapshot"),
        # Negative side: the signal does describe the snapshot the row advertises. No refs/main
        # (a commit-pinned fetch) carries no evidence either way.
        pytest.param(
            {"config.json": b"{}", "model.safetensors": b"\0" * 13},
            None,
            NEWER,
            True,
            id = "advertised-snapshot",
        ),
        # A refs/main naming no directory does carry evidence: that attempt landed no file.
        pytest.param(
            {"config.json": b"{}", "model.safetensors": b"\0" * 13},
            UPSTREAM_HEAD,
            NEWER,
            False,
            id = "unmaterialised-attempt",
        ),
        # refs/main resolves onto the OLDER payload snapshot though the newer one is self-contained too.
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

    # *advertised* is a commit whose snapshot the row pins, or the repo id when refs/main resolves.
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
        # Half a split quant and no manifest or marker: the .incomplete blob is the only evidence.
        pytest.param(
            {"Model-Q4_K_M-00001-of-00002.gguf": b"\0" * 32},
            True,
            id = "pinned-snapshot-holds-half-a-split-quant",
        ),
        # Negative side: the pinned snapshot serves the whole quant, so the row stays chattable.
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
        # The safetensors half of the case above: a shard names the total, so half a set is provable.
        pytest.param(
            {"config.json": b"{}", "model-00001-of-00002.safetensors": b"\0" * 32},
            True,
            id = "pinned-snapshot-holds-half-a-sharded-set",
        ),
        # Negative side: the whole set is here, so the row stays chattable.
        pytest.param(
            {
                "config.json": b"{}",
                "model-00001-of-00002.safetensors": b"\0" * 32,
                "model-00002-of-00002.safetensors": b"\0" * 32,
                "model.safetensors.index.json": _SHARD_INDEX,
            },
            False,
            id = "pinned-snapshot-holds-the-whole-sharded-set",
        ),
        # Nothing names a total, so nothing proves breakage: the #7374 shape must keep loading.
        pytest.param(
            {"config.json": b"{}", "model.safetensors": b"\0" * 32},
            False,
            id = "pinned-snapshot-holds-an-unsharded-payload",
        ),
        # from_pretrained loads one family, so a torn set beside a complete one keeps auto-load.
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
                "model.safetensors.index.json": _SHARD_INDEX,
                "pytorch_model-00001-of-00003.bin": b"\0" * 32,
            },
            False,
            id = "pinned-snapshot-holds-a-whole-sharded-family-beside-a-broken-one",
        ),
        # The half-fetched set stays proof: no training artefact or adapter is a base family.
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
        # A COMPLETE auxiliary set is not a base family either, so it cannot stand in for one.
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
        # Half a sharded set and NO other trace, as a cleaned-up fetch or a copied cache leaves.
        pytest.param(
            {"config.json": b"{}", "model-00001-of-00002.safetensors": b"\0" * 32},
            True,
            id = "half-a-sharded-set-and-no-other-trace",
        ),
        # Negative side, and #7374's own shape: the payload is whole, so the row loads from disk.
        pytest.param(
            {
                "config.json": b"{}",
                "model-00001-of-00002.safetensors": b"\0" * 32,
                "model-00002-of-00002.safetensors": b"\0" * 32,
                "model.safetensors.index.json": _SHARD_INDEX,
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
        # One whole family beside a torn one still loads, as with an .incomplete blob present.
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

    # The ref resolves, so the load id stays the repo id and the row keeps its pre-recovery answer.
    assert rows[0]["load_id"] == "Org/Model"
    assert rows[0].get("partial") is False


@pytest.mark.parametrize("signal", ["marker", "manifest"])
def test_an_update_that_never_materialised_leaves_the_cached_payload_chattable(
    signal, tmp_path, monkeypatch
):
    """The recovered row's own scenario. ``refs/main`` is rewritten before the first byte and the
    manifest earlier still, so an update interrupted that early leaves the previous complete snapshot
    as the only payload under a ref that resolves nowhere: it must not arrive partial."""
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
        # A re-download that stops before materialising its snapshot leaves a NEW-revision manifest.
        pytest.param(
            {"Model-Q4_K_M.gguf": b"\0" * 32},
            {"config.json": b"{}"},
            NEWER,
            OLDER,
            False,
            id = "pinned-older-snapshot",
        ),
        # Negative side: the manifest's quant is incomplete under the pinned snapshot, which judges.
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
    """Selection ran off directory mtime alone, which is not a total order: candidates reach the row
    through a ``frozenset`` and the variant walk through ``iterdir()``, so equal mtimes picked
    different directories."""
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
    """The same repo can sit in the active hub cache and in a previous one. One row survives the
    merge and only its directory loads, so the loser's projector flag must not carry over."""
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
    # The row the picker shows loads out of the active root, which holds no projector.
    assert not (active / "models--Org--Model" / "snapshots" / SNAPSHOT / "mmproj-F16.gguf").exists()
    assert rows[0]["capabilities"].get("supports_vision") is False


# --- the chokepoints, so a new signal cannot pick its own snapshot ------------

_BACKEND = Path(__file__).resolve().parents[1]
# Every helper the per-repo scan may hand the whole repo to. Each aggregates across revisions on
# purpose, so a new name here is a new repo-wide signal on a one-directory row.
_REPO_WIDE_HELPERS = frozenset(
    {
        "_cache_inventory_fields",
        # The row's pipeline task. Repo-wide on purpose and NOT a snapshot signal: it answers "which
        # model is this", which every revision of a repo agrees on. The non-GGUF classifier returns
        # non-None only when detect_family(repo_id) does, and _repo_is_diffusers is True whenever
        # that holds, so its newest-revision _repo_has_pipeline_index branch cannot change the
        # answer; the GGUF one reads general.architecture, identical in every cached quant. Scoping
        # it would only lose rows -- an unreadable header in the one pinned snapshot would drop the
        # repo from the Images/Video pickers entirely.
        "_cached_row_task",
        "_repo_gguf_last_modified",
        "_repo_gguf_payload_snapshots",
        "_repo_gguf_size_bytes",
        "_repo_has_gguf_files",
        "_repo_non_gguf_model_payload",
        "getattr",
    }
)
# Only the shared ordering key may read a snapshot dir's mtime; _blob_mtime orders nothing.
_MTIME_READERS = {
    "hub/utils/hf_cache_state.py": frozenset({"snapshot_selection_key"}),
    "hub/utils/gguf.py": frozenset(),
    "hub/services/models/cache_inventory.py": frozenset({"_blob_mtime"}),
    # Mirrors what huggingface_hub records per revision; it selects nothing.
    "hub/utils/inventory_scan.py": frozenset({"_recover_repo_hidden_by_dangling_refs"}),
    # The compatibility routes, listed so the two snapshot selectors cannot reintroduce their own
    # mtime reads. The names left rank directories or repo/blob mtimes, never snapshots.
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


@pytest.mark.parametrize(
    "weight",
    ["model.safetensors", "adapter_model.safetensors"],
    ids = ["base", "adapter"],
)
def test_selection_passes_over_a_newer_snapshot_whose_weight_file_is_empty(
    tmp_path, monkeypatch, weight
):
    """A zero-byte weight used to be skipped outright, so the newer revision read as holding nothing
    broken and beat a whole older one. The loader picks the name by existence, then cannot open."""
    config = "adapter_config.json" if weight.startswith("adapter") else "config.json"
    body = b'{"peft_type":"LORA"}' if config == "adapter_config.json" else b'{"model_type":"llama"}'
    repo_dir = _repo_with(
        tmp_path,
        snapshots = {
            OLDER: {config: body, weight: b"\0" * 256},
            NEWER: {config: body, weight: b""},
        },
        refs = {"main": UPSTREAM_HEAD},
    )
    _age(repo_dir / "snapshots" / OLDER, 600)

    rows = _autoload_rows(tmp_path, monkeypatch)

    assert rows[0]["partial"] is False
    assert rows[0]["capabilities"]["can_chat"] is True
    assert rows[0]["load_id"] == str(repo_dir / "snapshots" / OLDER)


def test_a_snapshot_whose_every_shard_is_empty_names_no_family_to_miss(tmp_path, monkeypatch):
    """An empty numbered shard reads as absent from its family rather than unreadable, so a
    snapshot holding nothing but one named no family at all and read as having nothing missing."""
    repo_dir = _repo_with(
        tmp_path,
        snapshots = {
            OLDER: {
                "config.json": b'{"model_type":"llama"}',
                "model-00001-of-00002.safetensors": b"\0" * 256,
                "model-00002-of-00002.safetensors": b"\0" * 256,
                "model.safetensors.index.json": _SHARD_INDEX,
            },
            NEWER: {
                "config.json": b'{"model_type":"llama"}',
                "model-00001-of-00002.safetensors": b"",
            },
        },
        refs = {"main": UPSTREAM_HEAD},
    )
    _age(repo_dir / "snapshots" / OLDER, 600)

    rows = _autoload_rows(tmp_path, monkeypatch)

    assert rows[0]["partial"] is False
    assert rows[0]["capabilities"]["can_chat"] is True
    assert rows[0]["load_id"] == str(repo_dir / "snapshots" / OLDER)


@pytest.mark.parametrize(
    "config_body",
    [b'{"model_type": "llam', b"[1,2,3]", b""],
    ids = ["truncated", "not-an-object", "empty"],
)
def test_a_required_config_has_to_parse_before_it_proves_a_payload(
    tmp_path, monkeypatch, config_body
):
    """from_pretrained parses config.json before it looks at a single weight, so one that does not
    parse fails the load as surely as a zero-byte one and cannot mark the snapshot ready."""
    _repo_with(
        tmp_path,
        snapshots = {
            OLDER: {
                "config.json": config_body,
                "model-00001-of-00002.safetensors": b"\0" * 256,
                "model-00002-of-00002.safetensors": b"\0" * 256,
                "model.safetensors.index.json": _SHARD_INDEX,
            }
        },
        refs = {"main": UPSTREAM_HEAD},
    )

    rows = _autoload_rows(tmp_path, monkeypatch)

    assert rows[0]["partial"] is True
    assert rows[0]["capabilities"]["can_chat"] is False


@pytest.mark.parametrize(
    "artefact",
    ["optimizer.safetensors", "scheduler.safetensors", "trainer_state.safetensors"],
)
def test_trainer_state_is_not_the_model_it_was_saved_beside(tmp_path, monkeypatch, artefact):
    """A trainer writes these next to the weights. They end in .safetensors and from_pretrained
    will not load one, so a snapshot holding nothing else has no payload to offer."""
    _repo_with(
        tmp_path,
        snapshots = {OLDER: {"config.json": b'{"model_type":"llama"}', artefact: b"\0" * 256}},
        refs = {"main": UPSTREAM_HEAD},
    )

    assert _autoload_rows(tmp_path, monkeypatch) == []


def test_trainer_state_beside_real_weights_changes_nothing(tmp_path, monkeypatch):
    """Negative control for the test above: the artefact is ignored, not held against the row."""
    _repo_with(
        tmp_path,
        snapshots = {
            OLDER: {
                "config.json": b'{"model_type":"llama"}',
                "model.safetensors": b"\0" * 256,
                "optimizer.safetensors": b"\0" * 256,
            }
        },
        refs = {"main": UPSTREAM_HEAD},
    )

    rows = _autoload_rows(tmp_path, monkeypatch)

    assert rows[0]["partial"] is False
    assert rows[0]["capabilities"]["can_chat"] is True


def test_an_empty_stray_adapter_does_not_veto_a_checkpoint(tmp_path, monkeypatch):
    """A zero-byte file from the other family is not what this row loads. The non-empty case is
    already exempted when the row's own payload names no family, and this is the same shape."""
    _repo_with(
        tmp_path,
        snapshots = {
            OLDER: {
                "config.json": b'{"model_type":"llama"}',
                "model.ckpt": b"\0" * 256,
                "adapter_model.safetensors": b"",
            }
        },
        refs = {"main": UPSTREAM_HEAD},
    )

    rows = _autoload_rows(tmp_path, monkeypatch)

    assert rows[0]["partial"] is False
    assert rows[0]["capabilities"]["can_chat"] is True


def test_an_empty_weight_of_the_rows_own_kind_still_vetoes(tmp_path, monkeypatch):
    """Negative control for the test above: the loader stops on a name of the kind it wants."""
    _repo_with(
        tmp_path,
        snapshots = {
            OLDER: {
                "adapter_config.json": b'{"peft_type":"LORA"}',
                "adapter_model.safetensors": b"",
                "adapter_model.bin": b"",
            }
        },
        refs = {"main": UPSTREAM_HEAD},
    )

    assert _autoload_rows(tmp_path, monkeypatch) == []


@pytest.mark.parametrize(
    "weight",
    [
        "model-00000-of-00002.safetensors",
        "model-00003-of-00002.safetensors",
        "model-00001-of-00000.safetensors",
    ],
    ids = ["index-zero", "index-past-total", "total-zero"],
)
def test_numbering_no_set_of_shards_can_satisfy_is_a_family_short(tmp_path, monkeypatch, weight):
    """The name still classifies the snapshot, so dropping the family left nothing to be short of
    and it read as complete. Neither transformers nor peft can assemble one of these."""
    _repo_with(
        tmp_path,
        snapshots = {OLDER: {"config.json": b'{"model_type":"llama"}', weight: b"\0" * 256}},
        refs = {"main": UPSTREAM_HEAD},
    )

    rows = _autoload_rows(tmp_path, monkeypatch)

    assert rows[0]["partial"] is True
    assert rows[0]["capabilities"]["can_chat"] is False


@pytest.mark.parametrize("artefact", ["optimizer.pt", "scheduler.pth", "rng_state.npz"])
def test_trainer_state_in_checkpoint_format_is_not_a_payload_either(
    tmp_path, monkeypatch, artefact
):
    """The checkpoint extensions are a suffix test, so trainer state saved as .pt read as weights
    the same way optimizer.safetensors did before it was excluded."""
    _repo_with(
        tmp_path,
        snapshots = {OLDER: {"config.json": b'{"model_type":"llama"}', artefact: b"\0" * 256}},
        refs = {"main": UPSTREAM_HEAD},
    )

    assert _autoload_rows(tmp_path, monkeypatch) == []


def test_an_empty_checkpoint_is_not_a_payload_this_walk_can_excuse(tmp_path, monkeypatch):
    """A .ckpt names no family, so absence of one cannot be held against it. An empty one is
    different: the file is there, the loader opens it, and there is nothing inside."""
    repo_dir = _repo_with(
        tmp_path,
        snapshots = {
            OLDER: {"config.json": b'{"model_type":"llama"}', "model.ckpt": b"\0" * 256},
            NEWER: {"config.json": b'{"model_type":"llama"}', "model.ckpt": b""},
        },
        refs = {"main": UPSTREAM_HEAD},
    )
    _age(repo_dir / "snapshots" / OLDER, 600)

    rows = _autoload_rows(tmp_path, monkeypatch)

    assert rows[0]["partial"] is False
    assert rows[0]["load_id"] == str(repo_dir / "snapshots" / OLDER)


def test_an_empty_diffusion_weight_is_judged_like_an_empty_checkpoint(tmp_path, monkeypatch):
    """A diffusion .safetensors names no family either, and the walk counts a non-empty one as
    payload. An empty one has to count the same way, or the newer snapshot holding it wins the
    selection and the pinned load opens a zero-byte weight."""
    repo_dir = _repo_with(
        tmp_path,
        snapshots = {
            OLDER: {
                "config.json": b'{"model_type":"llama"}',
                "diffusion_pytorch_model.safetensors": b"\0" * 256,
            },
            NEWER: {
                "config.json": b'{"model_type":"llama"}',
                "diffusion_pytorch_model.safetensors": b"",
            },
        },
        refs = {"main": UPSTREAM_HEAD},
    )
    _age(repo_dir / "snapshots" / OLDER, 600)

    rows = _autoload_rows(tmp_path, monkeypatch)

    assert rows[0]["partial"] is False
    assert rows[0]["load_id"] == str(repo_dir / "snapshots" / OLDER)


def test_a_family_in_a_subdirectory_does_not_stand_in_for_the_root_one(tmp_path, monkeypatch):
    """from_pretrained reads the snapshot root and fails on the index it finds there. It does not
    go looking for an unrelated set under backup/, so that set cannot vouch for the row."""
    _repo_with(
        tmp_path,
        snapshots = {
            OLDER: {
                "config.json": b'{"model_type":"llama"}',
                "model-00001-of-00002.safetensors": b"\0" * 64,
                "model.safetensors.index.json": _SHARD_INDEX,
                "backup/model-00001-of-00002.safetensors": b"\0" * 64,
                "backup/model-00002-of-00002.safetensors": b"\0" * 64,
                "backup/model.safetensors.index.json": _SHARD_INDEX,
            }
        },
        refs = {"main": UPSTREAM_HEAD},
    )

    rows = _autoload_rows(tmp_path, monkeypatch)

    assert rows[0]["partial"] is True
    assert rows[0]["capabilities"]["can_chat"] is False


def test_a_layout_that_keeps_every_weight_in_a_subdirectory_cannot_serve_the_root(
    tmp_path, monkeypatch
):
    """Companion to the test above: from_pretrained opens the names it finds at the snapshot root,
    so a set that lives only under backup/ is not one the pinned load can reach either. A layout
    that genuinely keeps its weights in subdirectories names no family this walk groups, so it is
    carried by the ungrouped payload rather than by this one."""
    _repo_with(
        tmp_path,
        snapshots = {
            OLDER: {
                "config.json": b'{"model_type":"llama"}',
                "backup/model-00001-of-00002.safetensors": b"\0" * 64,
                "backup/model-00002-of-00002.safetensors": b"\0" * 64,
                "backup/model.safetensors.index.json": _SHARD_INDEX,
            }
        },
        refs = {"main": UPSTREAM_HEAD},
    )

    rows = _autoload_rows(tmp_path, monkeypatch)

    assert rows[0]["partial"] is True
    assert rows[0]["capabilities"]["can_chat"] is False


def test_an_unsharded_weight_under_a_subdirectory_does_not_serve_the_root(tmp_path, monkeypatch):
    """The unsharded twin of the test above. A lone backup/adapter_model.safetensors is not the
    name peft opens at the snapshot root, so it cannot make the pinned snapshot loadable."""
    _repo_with(
        tmp_path,
        snapshots = {
            OLDER: {
                "adapter_config.json": b'{"peft_type":"LORA"}',
                "backup/adapter_model.safetensors": b"\0" * 256,
            }
        },
        refs = {"main": UPSTREAM_HEAD},
    )

    rows = _autoload_rows(tmp_path, monkeypatch)

    assert rows[0]["partial"] is True
    assert rows[0]["capabilities"]["can_chat"] is False


def test_a_config_the_loader_cannot_open_by_name_does_not_classify(tmp_path, monkeypatch):
    """The loaders open config.json by its exact path, so on a case-sensitive volume a Config.json
    is not the file they find. The filesystem answers that, not a lowercased basename."""
    _repo_with(
        tmp_path,
        snapshots = {
            OLDER: {"Config.json": b'{"model_type":"llama"}', "model.safetensors": b"\0" * 256}
        },
        refs = {"main": UPSTREAM_HEAD},
    )

    probe = tmp_path / "case-probe"
    probe.mkdir()
    (probe / "A").write_bytes(b"")
    case_sensitive = not (probe / "a").is_file()

    rows = _autoload_rows(tmp_path, monkeypatch)

    if case_sensitive:
        assert rows[0]["partial"] is True
        assert rows[0]["capabilities"]["can_chat"] is False
    else:
        assert rows[0]["partial"] is False


def test_a_weight_the_loader_cannot_open_by_name_does_not_serve(tmp_path, monkeypatch):
    """Same rule as the config above, for the weight itself: the loader opens these names by exact
    path, so on a case-sensitive volume MODEL.SAFETENSORS is not the file it finds."""
    probe = tmp_path / "case-probe"
    probe.mkdir()
    (probe / "A").write_bytes(b"")
    case_sensitive = not (probe / "a").is_file()

    for config, weight in (
        ('{"model_type":"llama"}', "MODEL.SAFETENSORS"),
        ('{"model_type":"llama"}', "PyTorch_Model.bin"),
        ('{"peft_type":"LORA"}', "ADAPTER_MODEL.SAFETENSORS"),
    ):
        root = tmp_path / weight.replace(".", "_")
        config_name = "adapter_config.json" if "peft_type" in config else "config.json"
        _repo_with(
            root,
            snapshots = {OLDER: {config_name: config.encode(), weight: b"\0" * 256}},
            refs = {"main": UPSTREAM_HEAD},
        )

        rows = _autoload_rows(root, monkeypatch)

        assert rows[0]["partial"] is case_sensitive
        assert rows[0]["capabilities"]["can_chat"] is not case_sensitive


def test_a_weight_under_the_wrong_case_does_not_hide_one_under_the_right_case(
    tmp_path, monkeypatch
):
    """Control: the name the loader opens decides, beside a miscased copy or later in the chain."""
    for extra in ({"model.safetensors": b"\0" * 256}, {"pytorch_model.bin": b"\0" * 256}):
        root = tmp_path / next(iter(extra)).replace(".", "_")
        _repo_with(
            root,
            snapshots = {
                OLDER: {
                    "config.json": b'{"model_type":"llama"}',
                    "MODEL.SAFETENSORS": b"\0" * 256,
                    **extra,
                }
            },
            refs = {"main": UPSTREAM_HEAD},
        )

        rows = _autoload_rows(root, monkeypatch)

        assert rows[0]["partial"] is False
        assert rows[0]["capabilities"]["can_chat"] is True


def test_a_root_index_naming_shards_in_a_subdirectory_still_serves(tmp_path, monkeypatch):
    """The loader resolves every weight_map entry against the index it selected, so a canonical root
    index pointing into weights/ is one it can load. Only the index's own paths are judged."""
    _repo_with(
        tmp_path,
        snapshots = {
            OLDER: {
                "config.json": b'{"model_type":"llama"}',
                "model.safetensors.index.json": _shard_index(
                    "weights/model-00001-of-00001.safetensors"
                ),
                "weights/model-00001-of-00001.safetensors": b"\0" * 64,
            }
        },
        refs = {"main": UPSTREAM_HEAD},
    )

    rows = _autoload_rows(tmp_path, monkeypatch)

    assert rows[0]["partial"] is False
    assert rows[0]["capabilities"]["can_chat"] is True


def test_a_canonical_index_with_no_shards_is_not_skipped_for_the_bin(tmp_path, monkeypatch):
    """_get_resolved_checkpoint_files tries model.safetensors.index.json before pytorch_model.bin,
    and the branches are exclusive, so a whole .bin beside an index whose shards were never
    recovered is not the file the loader reaches."""
    _repo_with(
        tmp_path,
        snapshots = {
            OLDER: {
                "config.json": b'{"model_type":"llama"}',
                "pytorch_model.bin": b"\0" * 256,
                "model.safetensors.index.json": _SHARD_INDEX,
            }
        },
        refs = {"main": UPSTREAM_HEAD},
    )

    rows = _autoload_rows(tmp_path, monkeypatch)

    assert rows[0]["partial"] is True
    assert rows[0]["capabilities"]["can_chat"] is False


def test_a_bin_with_no_safetensors_index_beside_it_still_serves(tmp_path, monkeypatch):
    """Control for the test above: with no safetensors name of any kind, pytorch_model.bin is the
    file the loader reaches."""
    _repo_with(
        tmp_path,
        snapshots = {
            OLDER: {
                "config.json": b'{"model_type":"llama"}',
                "pytorch_model.bin": b"\0" * 256,
            }
        },
        refs = {"main": UPSTREAM_HEAD},
    )

    rows = _autoload_rows(tmp_path, monkeypatch)

    assert rows[0]["partial"] is False
    assert rows[0]["capabilities"]["can_chat"] is True


def test_a_family_behind_an_alternate_index_name_is_not_visible(tmp_path, monkeypatch):
    """transformers probes model.safetensors.index.json, never model-copy.safetensors.index.json, so
    a set behind the alternate name is one the pinned load never opens."""
    _repo_with(
        tmp_path,
        snapshots = {
            OLDER: {
                "config.json": b'{"model_type":"llama"}',
                "model-copy-00001-of-00001.safetensors": b"\0" * 64,
                "model-copy.safetensors.index.json": _shard_index(
                    "model-copy-00001-of-00001.safetensors"
                ),
            }
        },
        refs = {"main": UPSTREAM_HEAD},
    )

    rows = _autoload_rows(tmp_path, monkeypatch)

    assert rows[0]["partial"] is True
    assert rows[0]["capabilities"]["can_chat"] is False


def test_an_alternate_index_family_does_not_rescue_a_broken_canonical_one(tmp_path, monkeypatch):
    """Same reason from the other side: the loader opens the canonical index, fails on the shard it
    is short, and never reaches the alternate set sitting beside it."""
    _repo_with(
        tmp_path,
        snapshots = {
            OLDER: {
                "config.json": b'{"model_type":"llama"}',
                "model-00001-of-00002.safetensors": b"\0" * 64,
                "model.safetensors.index.json": _SHARD_INDEX,
                "model-copy-00001-of-00001.safetensors": b"\0" * 64,
                "model-copy.safetensors.index.json": _shard_index(
                    "model-copy-00001-of-00001.safetensors"
                ),
            }
        },
        refs = {"main": UPSTREAM_HEAD},
    )

    rows = _autoload_rows(tmp_path, monkeypatch)

    assert rows[0]["partial"] is True
    assert rows[0]["capabilities"]["can_chat"] is False


def test_an_empty_canonical_weight_is_not_excused_by_another_root_file(tmp_path, monkeypatch):
    """_get_resolved_checkpoint_files opens model.safetensors first and pytorch_model.bin next, and
    falls back to neither once the name it picked exists. A whole consolidated.safetensors beside an
    empty model.safetensors therefore vouches for nothing."""
    for empty, whole in (
        ("model.safetensors", "consolidated.safetensors"),
        ("pytorch_model.bin", "consolidated.bin"),
    ):
        root = tmp_path / empty.replace(".", "_")
        _repo_with(
            root,
            snapshots = {
                OLDER: {
                    "config.json": b'{"model_type":"llama"}',
                    empty: b"",
                    whole: b"\0" * 256,
                }
            },
            refs = {"main": UPSTREAM_HEAD},
        )

        rows = _autoload_rows(root, monkeypatch)

        assert rows[0]["partial"] is True
        assert rows[0]["capabilities"]["can_chat"] is False


def test_a_whole_canonical_weight_beside_another_root_file_still_serves(tmp_path, monkeypatch):
    """Control for the test above: with the name the loader picks whole, the second root file is
    beside the point."""
    _repo_with(
        tmp_path,
        snapshots = {
            OLDER: {
                "config.json": b'{"model_type":"llama"}',
                "model.safetensors": b"\0" * 256,
                "consolidated.safetensors": b"\0" * 256,
            }
        },
        refs = {"main": UPSTREAM_HEAD},
    )

    rows = _autoload_rows(tmp_path, monkeypatch)

    assert rows[0]["partial"] is False
    assert rows[0]["capabilities"]["can_chat"] is True


def test_a_torn_quant_does_not_veto_the_weights_row_beside_it(tmp_path, monkeypatch):
    """The weights row loads model.safetensors and never opens a .gguf, so an interrupted quant
    download beside it says nothing about the row."""
    _repo_with(
        tmp_path,
        snapshots = {
            OLDER: {
                "config.json": b'{"model_type":"llama"}',
                "model.safetensors": b"\0" * 256,
                "Model-Q4_K_M-00001-of-00002.gguf": b"GGUF" + b"\0" * 252,
            }
        },
        refs = {"main": UPSTREAM_HEAD},
    )

    rows = _autoload_rows(tmp_path, monkeypatch)

    assert rows[0]["partial"] is False
    assert rows[0]["capabilities"]["can_chat"] is True


def test_a_whole_quant_does_not_vouch_for_the_weights_row_beside_it(tmp_path, monkeypatch):
    """The converse: a complete quant is not evidence that a shard-short safetensors set loads."""
    _repo_with(
        tmp_path,
        snapshots = {
            OLDER: {
                "config.json": b'{"model_type":"llama"}',
                "model.safetensors.index.json": _shard_index(
                    "model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"
                ),
                "model-00001-of-00002.safetensors": b"\0" * 256,
                "Model-Q8_0.gguf": b"GGUF" + b"\0" * 252,
            }
        },
        refs = {"main": UPSTREAM_HEAD},
    )

    rows = _autoload_rows(tmp_path, monkeypatch)

    assert rows[0]["partial"] is True
    assert rows[0]["capabilities"]["can_chat"] is False


def test_a_root_weight_under_a_name_the_loader_never_opens_does_not_serve(tmp_path, monkeypatch):
    """The loader probes model.safetensors, its index, pytorch_model.bin and its index, and peft
    only the singular adapter_model.*; every other root name is one nothing opens, however whole."""
    for config, weight in (
        ('{"model_type":"llama"}', "consolidated.safetensors"),
        ('{"model_type":"llama"}', "consolidated.bin"),
        ('{"model_type":"llama"}', "model.fp16.safetensors"),
        ('{"peft_type":"LORA"}', "adapter_model_v2.safetensors"),
    ):
        root = tmp_path / weight.replace(".", "_")
        config_name = "adapter_config.json" if "peft_type" in config else "config.json"
        _repo_with(
            root,
            snapshots = {OLDER: {config_name: config.encode(), weight: b"\0" * 256}},
            refs = {"main": UPSTREAM_HEAD},
        )

        rows = _autoload_rows(root, monkeypatch)

        assert rows[0]["partial"] is True
        assert rows[0]["capabilities"]["can_chat"] is False


def test_a_root_weight_the_loader_never_opens_does_not_veto_the_name_it_does(tmp_path, monkeypatch):
    """Control: an unopened name is no evidence, so the next name in the chain still decides."""
    for extra in ({"pytorch_model.bin": b"\0" * 256}, {"model.safetensors": b"\0" * 256}):
        root = tmp_path / next(iter(extra)).replace(".", "_")
        _repo_with(
            root,
            snapshots = {
                OLDER: {
                    "config.json": b'{"model_type":"llama"}',
                    "consolidated.safetensors": b"\0" * 256,
                    **extra,
                }
            },
            refs = {"main": UPSTREAM_HEAD},
        )

        rows = _autoload_rows(root, monkeypatch)

        assert rows[0]["partial"] is False
        assert rows[0]["capabilities"]["can_chat"] is True


def test_an_ungroupable_payload_under_a_subdirectory_does_not_serve_the_root(tmp_path, monkeypatch):
    """A .ckpt or diffusion weight names no family this walk groups, which exempts it from the
    family checks. That exemption only holds at the snapshot root: the loader cannot discover one
    under backup/ any more than it can a nested shard set."""
    for weights in ("backup/model.ckpt", "backup/diffusion_pytorch_model.safetensors"):
        root = tmp_path / weights.replace("/", "_")
        _repo_with(
            root,
            snapshots = {OLDER: {"config.json": b'{"model_type":"llama"}', weights: b"\0" * 256}},
            refs = {"main": UPSTREAM_HEAD},
        )

        rows = _autoload_rows(root, monkeypatch)

        assert rows[0]["partial"] is True
        assert rows[0]["capabilities"]["can_chat"] is False


def test_a_root_ungroupable_payload_is_not_vetoed_by_a_nested_copy(tmp_path, monkeypatch):
    """Control for the test above: the root holds one the loader opens, so a second copy
    underneath it is beside the point."""
    _repo_with(
        tmp_path,
        snapshots = {
            OLDER: {
                "config.json": b'{"model_type":"llama"}',
                "model.ckpt": b"\0" * 256,
                "backup/model.ckpt": b"\0" * 256,
            }
        },
        refs = {"main": UPSTREAM_HEAD},
    )

    rows = _autoload_rows(tmp_path, monkeypatch)

    assert rows[0]["partial"] is False
    assert rows[0]["capabilities"]["can_chat"] is True


def test_an_unsharded_weight_at_the_root_beside_a_nested_copy_still_serves(tmp_path, monkeypatch):
    """Control for the test above: the root holds the name the loader opens, so a second copy
    underneath it changes nothing."""
    _repo_with(
        tmp_path,
        snapshots = {
            OLDER: {
                "config.json": b'{"model_type":"llama"}',
                "model.safetensors": b"\0" * 256,
                "backup/model.safetensors": b"\0" * 256,
            }
        },
        refs = {"main": UPSTREAM_HEAD},
    )

    rows = _autoload_rows(tmp_path, monkeypatch)

    assert rows[0]["partial"] is False
    assert rows[0]["capabilities"]["can_chat"] is True


def test_a_config_in_a_subdirectory_does_not_serve_the_snapshot_root(tmp_path, monkeypatch):
    """The pin names the snapshot root, and from_pretrained opens the config there. A copy under
    backup/ is not the one that load finds, so it cannot classify the snapshot as loadable."""
    _repo_with(
        tmp_path,
        snapshots = {
            OLDER: {
                "backup/config.json": b'{"model_type":"llama"}',
                "model.safetensors": b"\0" * 256,
            }
        },
        refs = {"main": UPSTREAM_HEAD},
    )

    rows = _autoload_rows(tmp_path, monkeypatch)

    assert rows[0]["partial"] is True
    assert rows[0]["capabilities"]["can_chat"] is False


def test_a_nested_config_beside_the_root_one_leaves_the_row_alone(tmp_path, monkeypatch):
    """Negative control: the root config is what the loader opens, so a second copy underneath it
    changes nothing."""
    repo_dir = _repo_with(
        tmp_path,
        snapshots = {
            OLDER: {
                "config.json": b'{"model_type":"llama"}',
                "backup/config.json": b'{"model_type":"llama"}',
                "model.safetensors": b"\0" * 256,
            }
        },
        refs = {"main": UPSTREAM_HEAD},
    )

    rows = _autoload_rows(tmp_path, monkeypatch)

    assert rows[0]["partial"] is False
    assert rows[0]["capabilities"]["can_chat"] is True
    assert rows[0]["load_id"] == str(repo_dir / "snapshots" / OLDER)


def test_a_nested_adapter_config_does_not_serve_the_snapshot_root(tmp_path, monkeypatch):
    """peft resolves adapter_config.json at the directory it is handed, the same as the base
    loader, so a nested copy cannot classify the snapshot as an adapter either."""
    _repo_with(
        tmp_path,
        snapshots = {
            OLDER: {
                "backup/adapter_config.json": b'{"peft_type":"LORA"}',
                "adapter_model.safetensors": b"\0" * 256,
            }
        },
        refs = {"main": UPSTREAM_HEAD},
    )

    rows = _autoload_rows(tmp_path, monkeypatch)

    assert rows == [] or rows[0]["partial"] is True


def test_a_broken_active_copy_withholds_the_compatibility_row(tmp_path, monkeypatch):
    """The compatibility model list carries no path, so a client loads by id out of the active
    cache. With a broken copy there, publishing another cache's copy under the same id offers a
    load that follows the broken one. The Hub inventory still lists it, with a path."""
    import asyncio

    import routes.models as models_route

    active, legacy = tmp_path / "active", tmp_path / "legacy"
    whole = {
        "config.json": b'{"model_type":"llama"}',
        "model-00001-of-00002.safetensors": b"\0" * 64,
        "model-00002-of-00002.safetensors": b"\0" * 64,
        "model.safetensors.index.json": _SHARD_INDEX,
    }
    torn = dict(whole)
    del torn["model-00002-of-00002.safetensors"]
    # Active: half a set behind a dangling ref. Legacy: the same repo, whole.
    _repo_with(active, snapshots = {OLDER: torn}, refs = {"main": UPSTREAM_HEAD})
    _repo_with(legacy, snapshots = {NEWER: whole}, refs = {"main": NEWER})

    monkeypatch.setattr(inventory_scan, "hf_cache_roots", lambda: [active, legacy])
    monkeypatch.setattr(
        "utils.hf_cache_settings.get_hf_cache_paths",
        lambda: SimpleNamespace(hub_cache = active),
    )
    monkeypatch.setattr(models_route, "_resolve_hf_cache_dir", lambda: active)
    inventory_scan.invalidate_hf_cache_scans()

    response = asyncio.run(
        models_route.list_cached_models(current_subject = "test-user", hf_token = None)
    )
    cached = response["cached"] if isinstance(response, dict) else response.cached
    assert cached == []

    # Control: the broken copy in the other cache leaves the active one publishable.
    monkeypatch.setattr(inventory_scan, "hf_cache_roots", lambda: [legacy, active])
    monkeypatch.setattr(
        "utils.hf_cache_settings.get_hf_cache_paths",
        lambda: SimpleNamespace(hub_cache = legacy),
    )
    monkeypatch.setattr(models_route, "_resolve_hf_cache_dir", lambda: legacy)
    inventory_scan.invalidate_hf_cache_scans()

    response = asyncio.run(
        models_route.list_cached_models(current_subject = "test-user", hf_token = None)
    )
    cached = response["cached"] if isinstance(response, dict) else response.cached
    assert len(cached) == 1


def test_a_config_that_parses_still_proves_a_payload(tmp_path, monkeypatch):
    """Negative control for the test above: the same shape with a readable config."""
    _repo_with(
        tmp_path,
        snapshots = {
            OLDER: {
                "config.json": b'{"model_type":"llama"}',
                "model-00001-of-00002.safetensors": b"\0" * 256,
                "model-00002-of-00002.safetensors": b"\0" * 256,
                "model.safetensors.index.json": _SHARD_INDEX,
            }
        },
        refs = {"main": UPSTREAM_HEAD},
    )

    rows = _autoload_rows(tmp_path, monkeypatch)

    assert rows[0]["partial"] is False
    assert rows[0]["capabilities"]["can_chat"] is True


def test_a_whole_payload_under_a_resolving_ref_is_still_chattable(tmp_path, monkeypatch):
    """Negative control for the test above: same shape, but the pinned payload is whole."""
    repo_dir = _two_snapshot_repo(
        tmp_path,
        older_files = {
            "config.json": b'{"model_type":"llama"}',
            "model-00001-of-00002.safetensors": b"\0" * 256,
            "model-00002-of-00002.safetensors": b"\0" * 256,
            "model.safetensors.index.json": _SHARD_INDEX,
        },
        newer_files = {"README.md": b"probe"},
        ref = OLDER,
    )
    (repo_dir / "blobs" / "deadbeef.incomplete").write_bytes(b"\0" * 8)
    rows = _autoload_rows(tmp_path, monkeypatch)
    assert rows[0]["partial"] is False
    assert rows[0]["capabilities"]["can_chat"] is True


_WHOLE_SHARDS = {
    "config.json": b'{"model_type":"llama"}',
    "model-00001-of-00002.safetensors": b"\0" * 256,
    "model-00002-of-00002.safetensors": b"\0" * 256,
}


@pytest.mark.parametrize(
    ("index", "torn"),
    [
        pytest.param(_SHARD_INDEX, False, id = "a-map-naming-both-shards"),
        pytest.param(
            b'{"metadata": {}, "weight_map": {"w0": "model-00001-of-0',
            True,
            id = "an-index-truncated-mid-write",
        ),
        pytest.param(
            _shard_index("model-00001-of-00003.safetensors"),
            True,
            id = "a-map-naming-a-shard-that-is-not-here",
        ),
        pytest.param(b'{"metadata": {}}', True, id = "an-index-with-no-weight-map"),
        pytest.param(
            _shard_index("model-00001-of-00002.safetensors"),
            True,
            id = "a-map-covering-only-part-of-the-numbered-family",
        ),
        pytest.param(_shard_index(), True, id = "a-map-naming-nothing"),
        pytest.param(
            _shard_index("../../elsewhere.safetensors"),
            True,
            id = "a-map-reaching-outside-the-snapshot",
        ),
    ],
)
def test_a_shard_index_has_to_resolve_before_the_family_counts(tmp_path, monkeypatch, index, torn):
    """Present and non-empty is not enough: the index must parse and every weight_map name must
    resolve, or the numbered files read as a whole family that nothing can load."""
    _repo_with(
        tmp_path,
        snapshots = {SNAPSHOT: {**_WHOLE_SHARDS, "model.safetensors.index.json": index}},
        refs = {"main": UPSTREAM_HEAD},
    )
    rows = _autoload_rows(tmp_path, monkeypatch)
    assert rows[0]["partial"] is torn
    assert rows[0]["capabilities"]["can_chat"] is not torn


@pytest.mark.parametrize(
    ("files", "torn"),
    [
        pytest.param(
            {
                "config.json": b'{"model_type":"llama"}',
                "pytorch_model.bin": b"\0" * 256,
                "model-00001-of-00002.safetensors": b"\0" * 64,
                "model-00002-of-00002.safetensors": b"\0" * 64,
                "model.safetensors.index.json": _shard_index(
                    "model-00001-of-00002.safetensors", "model-00009-of-00009.safetensors"
                ),
            },
            True,
            id = "a-whole-bin-cannot-vouch-for-a-broken-safetensors-index",
        ),
        pytest.param(
            {
                "config.json": b'{"model_type":"llama"}',
                "pytorch_model.bin": b"\0" * 256,
                "model-00001-of-00002.safetensors": b"\0" * 64,
                "model-00002-of-00002.safetensors": b"\0" * 64,
                "model.safetensors.index.json": _SHARD_INDEX,
            },
            False,
            id = "the-same-shape-with-an-index-that-resolves",
        ),
        pytest.param(
            {
                "config.json": b'{"model_type":"llama"}',
                "model.safetensors": b"\0" * 256,
                "model-00001-of-00002.safetensors": b"\0" * 64,
                "model.safetensors.index.json": _shard_index(
                    "model-00001-of-00002.safetensors", "model-00009-of-00009.safetensors"
                ),
            },
            False,
            id = "a-whole-model-safetensors-is-picked-before-any-index",
        ),
        pytest.param(
            {"config.json": b'{"model_type":"llama"}', "pytorch_model.bin": b"\0" * 256},
            False,
            id = "a-whole-bin-with-nothing-safetensors-beside-it",
        ),
        pytest.param(
            {
                "adapter_config.json": b'{"peft_type":"LORA"}',
                "adapter_model-00001-of-00002.safetensors": b"\0" * 64,
                "adapter_model-00002-of-00002.safetensors": b"\0" * 64,
            },
            True,
            id = "a-numbered-adapter-set-with-every-file-present",
        ),
        pytest.param(
            {
                "config.json": b'{"model_type":"llama"}',
                "model.ckpt": b"\0" * 256,
                "adapter_model.bin": b"\0" * 128,
            },
            False,
            id = "a-whole-ckpt-payload-beside-a-stray-adapter-file",
        ),
        pytest.param(
            {
                "config.json": b'{"model_type":"llama"}',
                "diffusion_pytorch_model.safetensors": b"\0" * 256,
                "adapter_model.bin": b"\0" * 128,
            },
            False,
            id = "a-diffusion-payload-beside-a-stray-adapter-file",
        ),
        pytest.param(
            {"config.json": b'{"model_type":"llama"}', "adapter_model.bin": b"\0" * 128},
            True,
            id = "an-adapter-file-standing-in-for-base-weights-that-are-not-here",
        ),
        pytest.param(
            {
                "config.json": b'{"model_type":"llama"}',
                "model-00001-of-00002.safetensors": b"\0" * 64,
                "model-00002-of-00002.safetensors": b"\0" * 64,
                "pytorch_model.bin": b"\0" * 256,
            },
            False,
            id = "index-less-shards-are-looked-past-to-the-whole-bin",
        ),
        pytest.param(
            {
                "config.json": b'{"model_type":"llama"}',
                "model-00001-of-00002.safetensors": b"\0" * 64,
                "model-00002-of-00002.safetensors": b"\0" * 64,
            },
            True,
            id = "index-less-shards-with-nothing-else-to-fall-back-to",
        ),
        pytest.param(
            {
                "config.json": b'{"model_type":"llama"}',
                "optimizer.bin": b"\0" * 256,
                "adapter_model.bin": b"\0" * 128,
            },
            True,
            id = "a-training-artefact-is-not-a-payload-either",
        ),
        pytest.param(
            {
                "adapter_config.json": b'{"peft_type":"LORA"}',
                "adapter_model.safetensors": b"\0" * 256,
            },
            False,
            id = "the-singular-name-peft-resolves",
        ),
        pytest.param(
            {"adapter_config.json": b'{"peft_type":"LORA"}', "adapter_model.bin": b"\0" * 256},
            False,
            id = "the-singular-pickle-name",
        ),
        pytest.param(
            {
                "adapter_config.json": b'{"peft_type":"LORA"}',
                "adapter_model.safetensors": b"\0" * 256,
                "adapter_model-00001-of-00002.safetensors": b"\0" * 64,
            },
            False,
            id = "a-whole-singular-adapter-beside-a-numbered-set",
        ),
    ],
)
def test_the_family_a_loader_would_pick_is_the_one_judged(tmp_path, monkeypatch, files, torn):
    """transformers takes model.safetensors, then its index, then pytorch_model.bin, and never falls
    back once one matches, so a whole .bin cannot vouch for a broken safetensors index. peft
    resolves only the singular adapter names and has no shard path, so numbered sets never load."""
    _repo_with(tmp_path, snapshots = {SNAPSHOT: files}, refs = {"main": UPSTREAM_HEAD})
    rows = _autoload_rows(tmp_path, monkeypatch)
    assert rows[0]["partial"] is torn
    assert rows[0]["capabilities"]["can_chat"] is not torn


def test_the_load_id_pins_the_whole_snapshot_when_the_default_ref_is_torn(tmp_path, monkeypatch):
    """Recovery fires off a secondary dangling ref while refs/main resolves to a torn revision, and
    that revision classifies, so membership alone would hand the row back to the half download."""
    repo_dir = _repo_with(
        tmp_path,
        snapshots = {
            OLDER: {**_WHOLE_SHARDS, "model.safetensors.index.json": _SHARD_INDEX},
            NEWER: {
                "config.json": b'{"model_type":"llama"}',
                "model-00001-of-00002.safetensors": b"\0" * 256,
                "model.safetensors.index.json": _SHARD_INDEX,
            },
        },
        refs = {"main": NEWER, "stale": UPSTREAM_HEAD},
    )
    _age(repo_dir / "snapshots" / OLDER, 600)
    rows = _autoload_rows(tmp_path, monkeypatch)
    assert rows[0]["partial"] is False
    assert rows[0]["capabilities"]["can_chat"] is True
    assert rows[0]["load_id"] == str(repo_dir / "snapshots" / OLDER)


def test_the_load_id_stays_the_repo_id_when_the_default_ref_is_whole(tmp_path, monkeypatch):
    """Control: where refs/main lands on the complete payload the repo id is what loads."""
    repo_dir = _repo_with(
        tmp_path,
        snapshots = {
            OLDER: {
                "config.json": b'{"model_type":"llama"}',
                "model-00001-of-00002.safetensors": b"\0" * 256,
                "model.safetensors.index.json": _SHARD_INDEX,
            },
            NEWER: {**_WHOLE_SHARDS, "model.safetensors.index.json": _SHARD_INDEX},
        },
        refs = {"main": NEWER, "stale": UPSTREAM_HEAD},
    )
    _age(repo_dir / "snapshots" / OLDER, 600)
    rows = _autoload_rows(tmp_path, monkeypatch)
    assert rows[0]["partial"] is False
    assert rows[0]["capabilities"]["can_chat"] is True
    assert rows[0]["load_id"] == "Org/Model"


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
    the weights in another reads as runnable while nothing on disk can serve it. Neither the cache it
    sits in nor the state of refs/main changes that: a resolving ref only ever lands on one half,
    since a directory that could serve the payload would be a payload snapshot."""
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


def test_a_newer_companion_only_snapshot_does_not_make_the_ref_snapshot_partial(
    tmp_path, monkeypatch
):
    """The pipeline signals must read the snapshot the row loads, not the newest one.

    A GGUF image load leaves a companion-only prefetch (root ``model_index.json`` + VAE /
    text-encoder, no ``transformer/``) in a NEW revision beside the complete pipeline that
    ``refs/main`` still resolves to. huggingface_hub reads ``refs/main`` to turn ``main`` into a
    commit, so ``from_pretrained(repo_id)`` opens the OLD, complete directory -- judging the row on
    the newest revision instead marks a fully downloaded model partial and unchattable."""
    from types import SimpleNamespace

    from hub.services.models import cache_inventory

    # No root config.json: real diffusers pipelines ship model_index.json and per-component
    # configs only, and a fixture that adds one would pin a layout that does not occur.
    pipeline = {
        "model_index.json": b'{"_class_name":"FluxPipeline"}',
        "transformer/config.json": b'{"_class_name":"FluxTransformer2DModel"}',
        "vae/config.json": b'{"_class_name":"AutoencoderKL"}',
        "text_encoder/model.safetensors": b"\0" * 256,
        "transformer/diffusion_pytorch_model.safetensors": b"\0" * 256,
        "vae/diffusion_pytorch_model.safetensors": b"\0" * 256,
    }
    companion_only = {name: blob for name, blob in pipeline.items() if "transformer/" not in name}
    repo_dir = _repo_with(
        tmp_path,
        # SNAPSHOT is the companion-only prefetch, OLDER the complete pipeline refs/main names.
        snapshots = {SNAPSHOT: companion_only, OLDER: pipeline},
        refs = {"main": OLDER},
    )
    # The companion-only revision is unambiguously the newest, the shape the repo-wide check picks.
    os.utime(repo_dir / "snapshots" / OLDER, (1_000_000, 1_000_000))
    os.utime(repo_dir / "snapshots" / SNAPSHOT, (2_000_000, 2_000_000))

    monkeypatch.setattr(inventory_scan, "hf_cache_roots", lambda: [tmp_path])
    monkeypatch.setattr(
        "utils.hf_cache_settings.get_hf_cache_paths",
        lambda: SimpleNamespace(hub_cache = tmp_path),
    )
    inventory_scan.invalidate_hf_cache_scans()
    try:
        rows = cache_inventory._scan_cached_models()
    finally:
        inventory_scan.invalidate_hf_cache_scans()

    assert [row["repo_id"] for row in rows] == ["Org/Model"]
    # The row loads by repo id, so refs/main decides: the complete pipeline.
    assert rows[0]["load_id"] == "Org/Model"
    assert rows[0]["partial"] is False
    assert rows[0]["capabilities"]["can_chat"] is True
    # ... and that same directory carries the manifest, so it is not a single-file checkpoint.
    assert rows[0]["single_file"] is False


def _pipeline_snapshot(tmp_path, manifest: dict, files: dict) -> Path:
    """A diffusers pipeline snapshot: root ``model_index.json`` plus component subdirs."""
    snapshot = tmp_path / "snap"
    snapshot.mkdir()
    (snapshot / "model_index.json").write_bytes(json.dumps(manifest).encode())
    for name, blob in files.items():
        target = snapshot / name
        target.parent.mkdir(parents = True, exist_ok = True)
        target.write_bytes(blob)
    return snapshot


_FLUX_INDEX = {
    "_class_name": "FluxPipeline",
    "transformer": ["diffusers", "FluxTransformer2DModel"],
    "vae": ["diffusers", "AutoencoderKL"],
    "safety_checker": [None, None],
}
# Trimmed from the real manifests of CalamitousFelicitousness/Ideogram-4-bf16-Diffusers and
# Wan-AI/Wan2.2-T2V-A14B-Diffusers, which each ship two sharded denoiser directories.
_IDEOGRAM_INDEX = {
    "_class_name": "Ideogram4Pipeline",
    "transformer": ["diffusers", "Ideogram4Transformer2DModel"],
    "unconditional_transformer": ["diffusers", "Ideogram4Transformer2DModel"],
    "vae": ["diffusers", "AutoencoderKLFlux2"],
}
_WAN_INDEX = {
    "_class_name": "WanPipeline",
    "transformer": ["diffusers", "WanTransformer3DModel"],
    "transformer_2": ["diffusers", "WanTransformer3DModel"],
    "vae": ["diffusers", "AutoencoderKLWan"],
}
# Wan-AI/Wan2.2-TI2V-5B-Diffusers: the single-expert sibling declares transformer_2 as
# [null, null] and ships no such directory.
_WAN_SINGLE_EXPERT_INDEX = dict(_WAN_INDEX, transformer_2 = [None, None])
# Stable Cascade, Wuerstchen, Kandinsky and Shap-E call theirs "decoder"/"prior", so no key here
# matches either fixed name.
_CASCADE_INDEX = {
    "_class_name": "StableCascadeDecoderPipeline",
    "decoder": ["diffusers", "StableCascadeUNet"],
    "text_encoder": ["transformers", "CLIPTextModelWithProjection"],
    "vqgan": ["wuerstchen", "PaellaVQModel"],
}


def test_a_denoiser_missing_half_its_shards_is_not_a_present_denoiser(tmp_path):
    """Shard 1 of 2 alone is not a present denoiser, and the last shard completes it."""
    snapshot = _pipeline_snapshot(
        tmp_path,
        _FLUX_INDEX,
        {
            "transformer/diffusion_pytorch_model.safetensors.index.json": json.dumps(
                {
                    "weight_map": {
                        "a": "diffusion_pytorch_model-00001-of-00002.safetensors",
                        "b": "diffusion_pytorch_model-00002-of-00002.safetensors",
                    }
                }
            ).encode(),
            "transformer/diffusion_pytorch_model-00001-of-00002.safetensors": b"\0" * 256,
            "vae/diffusion_pytorch_model.safetensors": b"\0" * 256,
        },
    )
    assert inventory_scan.snapshot_pipeline_missing_denoiser(snapshot) is True

    (snapshot / "transformer" / "diffusion_pytorch_model-00002-of-00002.safetensors").write_bytes(
        b"\0" * 256
    )
    assert inventory_scan.snapshot_pipeline_missing_denoiser(snapshot) is False


def _dual_format_shard_index(fmt: str) -> bytes:
    return json.dumps(
        {
            "weight_map": {
                "a": f"diffusion_pytorch_model-00001-of-00002{fmt}",
                "b": f"diffusion_pytorch_model-00002-of-00002{fmt}",
            }
        }
    ).encode()


def test_an_unused_alternate_format_index_does_not_tear_a_whole_snapshot(tmp_path):
    """A whole safetensors set stays whole beside the orphan ``.bin.index.json`` a dual-format
    repo leaves behind, because each index is judged on its own shard set."""
    snapshot = _pipeline_snapshot(
        tmp_path,
        _FLUX_INDEX,
        {
            "transformer/diffusion_pytorch_model.safetensors.index.json": _dual_format_shard_index(
                ".safetensors"
            ),
            "transformer/diffusion_pytorch_model.bin.index.json": _dual_format_shard_index(".bin"),
            "transformer/diffusion_pytorch_model-00001-of-00002.safetensors": b"\0" * 256,
            "transformer/diffusion_pytorch_model-00002-of-00002.safetensors": b"\0" * 256,
        },
    )
    assert inventory_scan.snapshot_pipeline_missing_denoiser(snapshot) is False


def test_neither_format_being_whole_is_still_torn(tmp_path):
    """The other direction: any-index-SATISFIED, not any-index-present, so both formats half
    landed is still torn."""
    snapshot = _pipeline_snapshot(
        tmp_path,
        _FLUX_INDEX,
        {
            "transformer/diffusion_pytorch_model.safetensors.index.json": _dual_format_shard_index(
                ".safetensors"
            ),
            "transformer/diffusion_pytorch_model.bin.index.json": _dual_format_shard_index(".bin"),
            "transformer/diffusion_pytorch_model-00001-of-00002.safetensors": b"\0" * 256,
            "transformer/diffusion_pytorch_model-00001-of-00002.bin": b"\0" * 256,
        },
    )
    assert inventory_scan.snapshot_pipeline_missing_denoiser(snapshot) is True


# diffusers' _add_variant inserts the variant before the LAST extension, so a bf16 shard index is
# "...safetensors.index.bf16.json", not "...bf16.safetensors.index.json".
_VARIANT_INDEX_NAME = "diffusion_pytorch_model.safetensors.index.bf16.json"


@pytest.mark.parametrize(
    "orphan_index, orphan_suffix",
    [
        ("diffusion_pytorch_model.bin.index.json", ".bin"),
        (_VARIANT_INDEX_NAME, ".bf16.safetensors"),
    ],
)
def test_an_orphan_variant_index_does_not_veto_the_default_weight(
    orphan_index, orphan_suffix, tmp_path
):
    """An orphan variant index must not hide the unsharded default weight beside it."""
    snapshot = _pipeline_snapshot(
        tmp_path,
        _FLUX_INDEX,
        {
            "transformer/diffusion_pytorch_model.safetensors": b"\0" * 256,
            f"transformer/{orphan_index}": _dual_format_shard_index(orphan_suffix),
        },
    )
    assert inventory_scan.snapshot_pipeline_missing_denoiser(snapshot) is False


def test_a_variant_only_component_is_missing_its_denoiser_whole_or_not(tmp_path):
    """A dtype twin is not the weight a default load asks for, so a bf16 set does not make the
    component readable -- not half landed, and not even whole.

    ``from_pretrained`` without ``variant`` resolves the plain name and has no fallback to the
    twin: against a directory holding only ``diffusion_pytorch_model.fp16.safetensors`` diffusers
    raises ``Error no file named diffusion_pytorch_model.safetensors``. The download plan skips
    those files for that reason, so a cache holding only them cannot serve this row.
    """
    snapshot = _pipeline_snapshot(
        tmp_path,
        _FLUX_INDEX,
        {
            f"transformer/{_VARIANT_INDEX_NAME}": _dual_format_shard_index(".bf16.safetensors"),
            "transformer/diffusion_pytorch_model-00001-of-00002.bf16.safetensors": b"\0" * 256,
        },
    )
    assert inventory_scan.snapshot_pipeline_missing_denoiser(snapshot) is True

    (
        snapshot / "transformer" / "diffusion_pytorch_model-00002-of-00002.bf16.safetensors"
    ).write_bytes(b"\0" * 256)
    assert inventory_scan.snapshot_pipeline_missing_denoiser(snapshot) is True


@pytest.mark.parametrize("whole_suffix", [".safetensors", ".bin"])
def test_a_corrupt_selected_index_hides_the_whole_weight_beside_it(whole_suffix, tmp_path):
    """An unreadable selected index is the failure, not an absence of evidence.

    ``is_sharded`` is set from that file merely existing, so ``_get_checkpoint_shard_files`` then
    parses it and raises, and neither the ``except IOError`` branch nor the pickle fallback under
    it is reachable once sharded. The complete weight sitting beside it is never opened.
    """
    snapshot = _pipeline_snapshot(
        tmp_path,
        _FLUX_INDEX,
        {
            "transformer/diffusion_pytorch_model.safetensors.index.json": b"{not json",
            f"transformer/diffusion_pytorch_model{whole_suffix}": b"\0" * 256,
        },
    )
    assert inventory_scan.snapshot_pipeline_missing_denoiser(snapshot) is True


def test_a_sharded_bin_set_is_not_what_a_default_load_resolves(tmp_path):
    """``.bin`` shards behind their own index answer no default load.

    ``use_safetensors`` unset coerces to True, so ``_fetch_index_file`` builds the safetensors
    index name and nothing else; with no index found the loader asks for the UNSHARDED
    ``diffusion_pytorch_model.bin``, which a sharded set does not provide.
    """
    snapshot = _pipeline_snapshot(
        tmp_path,
        _FLUX_INDEX,
        {
            "transformer/diffusion_pytorch_model.bin.index.json": _dual_format_shard_index(".bin"),
            "transformer/diffusion_pytorch_model-00001-of-00002.bin": b"\0" * 256,
            "transformer/diffusion_pytorch_model-00002-of-00002.bin": b"\0" * 256,
        },
    )
    assert inventory_scan.snapshot_pipeline_missing_denoiser(snapshot) is True


def test_the_legacy_variant_index_spelling_does_not_vouch_either(tmp_path):
    """``_fetch_index_file_legacy`` spells the variant BEFORE ``.index``, so a deprecated fp16 set
    is ``diffusion_pytorch_model.safetensors.fp16.index.json`` -- a name ending in ``.index.json``
    that a load passing no ``variant`` still never resolves."""
    snapshot = _pipeline_snapshot(
        tmp_path,
        _FLUX_INDEX,
        {
            "transformer/diffusion_pytorch_model.safetensors.fp16.index.json": (
                _dual_format_shard_index(".fp16.safetensors")
            ),
            "transformer/diffusion_pytorch_model-00001-of-00002.fp16.safetensors": b"\0" * 256,
            "transformer/diffusion_pytorch_model-00002-of-00002.fp16.safetensors": b"\0" * 256,
        },
    )
    assert inventory_scan.snapshot_pipeline_missing_denoiser(snapshot) is True


def test_shards_whose_index_never_landed_do_not_stand_in_for_the_whole_weight(tmp_path):
    """A numbered shard is only ever reached THROUGH an index: with ``is_sharded`` false the loader
    asks for the unsharded name and nothing else. So a complete shard set whose index is missing --
    or, as here, present in the snapshot only as a blob symlink the cache already collected -- is
    wreckage, not a loose weight."""
    snapshot = _pipeline_snapshot(
        tmp_path,
        _FLUX_INDEX,
        {
            "transformer/diffusion_pytorch_model-00001-of-00002.safetensors": b"\0" * 256,
            "transformer/diffusion_pytorch_model-00002-of-00002.safetensors": b"\0" * 256,
        },
    )
    assert inventory_scan.snapshot_pipeline_missing_denoiser(snapshot) is True

    dangling = snapshot / "transformer" / "diffusion_pytorch_model.safetensors.index.json"
    dangling.symlink_to(tmp_path / "blobs" / "collected")
    assert inventory_scan.snapshot_pipeline_missing_denoiser(snapshot) is True


def test_an_unsharded_dtype_twin_alone_is_not_the_default_weight(tmp_path):
    """The same rule without an index in sight: the twin a ``variant = "fp16"`` load left in the
    cache is the only weight here, and the default load this app issues cannot open it."""
    snapshot = _pipeline_snapshot(
        tmp_path,
        _FLUX_INDEX,
        {"transformer/diffusion_pytorch_model.fp16.safetensors": b"\0" * 256},
    )
    assert inventory_scan.snapshot_pipeline_missing_denoiser(snapshot) is True

    (snapshot / "transformer" / "diffusion_pytorch_model.safetensors").write_bytes(b"\0" * 256)
    assert inventory_scan.snapshot_pipeline_missing_denoiser(snapshot) is False


def test_a_pipeline_declaring_no_denoiser_key_is_not_hunted_for_one(tmp_path):
    """Stable Cascade names its denoiser ``decoder``, so neither fixed name is in the manifest.
    A readable manifest that declares no transformer/unet has nothing this check can prove
    absent, and the fully downloaded pipeline must not be hidden as partial."""
    snapshot = _pipeline_snapshot(
        tmp_path,
        _CASCADE_INDEX,
        {
            "decoder/diffusion_pytorch_model.safetensors": b"\0" * 256,
            "vqgan/diffusion_pytorch_model.safetensors": b"\0" * 256,
        },
    )
    assert inventory_scan.snapshot_pipeline_missing_denoiser(snapshot) is False


def test_a_manifest_entry_that_is_not_a_component_pair_is_left_to_the_loader(tmp_path):
    """JaiDalmotra/ACE-STEP-Stereo-Finetuned maps "transformer" to a dict pointing at
    ace_step_transformer/, and ships no transformer/ at all. A custom manifest that does not
    follow the [library, class] pair convention does not name a directory, so demanding one would
    hide the whole repo."""
    snapshot = _pipeline_snapshot(
        tmp_path,
        {
            "_class_name": "ACEStepPipeline",
            "transformer": {
                "_class_name": "ACEStepTransformer2DModel",
                "config": "ace_step_transformer/config.json",
            },
        },
        {"ace_step_transformer/diffusion_pytorch_model.safetensors": b"\0" * 256},
    )
    assert inventory_scan.snapshot_pipeline_missing_denoiser(snapshot) is False


@pytest.mark.parametrize(
    "denoisers",
    [
        # weizhou03/HunyuanVideo-1.5-Diffusers-1080p-2SR runs three.
        ("transformer", "transformer_2", "transformer_3"),
        # BoyuanJiang/FitDiT has no "transformer" key at all.
        ("transformer_garm", "transformer_vton"),
    ],
)
def test_the_manifest_keys_generalise_past_the_names_we_knew(denoisers, tmp_path):
    """Reading the names off the manifest is what keeps this right for layouts no hardcoded list
    anticipated, without another edit here."""
    manifest = {"_class_name": "SomePipeline", "vae": ["diffusers", "AutoencoderKL"]}
    manifest.update({name: ["diffusers", "SomeTransformer2DModel"] for name in denoisers})
    files = {f"{name}/diffusion_pytorch_model.safetensors": b"\0" * 256 for name in denoisers}
    snapshot = _pipeline_snapshot(tmp_path, manifest, files)
    assert inventory_scan.snapshot_pipeline_missing_denoiser(snapshot) is False

    shutil.rmtree(snapshot / denoisers[-1])
    assert inventory_scan.snapshot_pipeline_missing_denoiser(snapshot) is True


def test_a_declared_but_null_second_expert_is_not_a_missing_denoiser(tmp_path):
    """Wan 2.2's 5B sibling declares ``transformer_2`` as [null, null] and ships no such
    directory. That is the manifest saying the slot is deliberately empty, not a torn download."""
    snapshot = _pipeline_snapshot(
        tmp_path,
        _WAN_SINGLE_EXPERT_INDEX,
        {
            "transformer/diffusion_pytorch_model.safetensors": b"\0" * 256,
            "vae/diffusion_pytorch_model.safetensors": b"\0" * 256,
        },
    )
    assert inventory_scan.snapshot_pipeline_missing_denoiser(snapshot) is False


def test_a_half_landed_shard_set_is_not_rescued_by_the_loose_scan(tmp_path):
    """The loose fallback skips claimed names, so shard 1 of 2, itself a weight file, cannot pose
    as the whole set and undo the check above it."""
    snapshot = _pipeline_snapshot(
        tmp_path,
        _FLUX_INDEX,
        {
            "transformer/diffusion_pytorch_model.safetensors.index.json": (
                _dual_format_shard_index(".safetensors")
            ),
            "transformer/diffusion_pytorch_model-00001-of-00002.safetensors": b"\0" * 256,
        },
    )
    assert inventory_scan.snapshot_pipeline_missing_denoiser(snapshot) is True


@pytest.mark.parametrize("escape", ["../vae/diffusion_pytorch_model.safetensors", "/absolute"])
def test_a_denoiser_index_naming_a_shard_outside_the_component_is_not_a_denoiser(escape, tmp_path):
    """``component / shard`` follows ``..`` out to a sibling and drops the component entirely for
    an absolute name, so a corrupt map could be satisfied by the vae next door. Same rule the
    root-weight scanner applies to its own indexes."""
    outside = tmp_path / "outside.safetensors"
    outside.write_bytes(b"\0" * 256)
    shard = str(outside) if escape == "/absolute" else escape
    snapshot = _pipeline_snapshot(
        tmp_path,
        _FLUX_INDEX,
        {
            "transformer/diffusion_pytorch_model.safetensors.index.json": json.dumps(
                {"weight_map": {"a": shard}}
            ).encode(),
            "vae/diffusion_pytorch_model.safetensors": b"\0" * 256,
        },
    )
    assert inventory_scan.snapshot_pipeline_missing_denoiser(snapshot) is True


@pytest.mark.skipif(os.name == "nt", reason = "a directory called C: cannot exist on Windows")
def test_a_drive_qualified_shard_name_is_outside_the_component_everywhere(tmp_path):
    """The escape above, spelled the way a Windows-written index spells it.

    ``PurePosixPath`` reads ``C:/pipe/...`` as a subdirectory literally called ``C:``, so without a
    drive check the name resolves to a real file here while on Windows the same join discards the
    component and reaches the drive root. The verdict has to be the escape on both.
    """
    snapshot = _pipeline_snapshot(
        tmp_path,
        _FLUX_INDEX,
        {
            "transformer/diffusion_pytorch_model.safetensors.index.json": json.dumps(
                {"weight_map": {"a": "C:/pipe/diffusion_pytorch_model.safetensors"}}
            ).encode(),
            "transformer/C:/pipe/diffusion_pytorch_model.safetensors": b"\0" * 256,
        },
    )
    assert inventory_scan.snapshot_pipeline_missing_denoiser(snapshot) is True


def test_a_denoiser_index_short_of_the_total_its_shard_names_declare_is_torn(tmp_path):
    """An index truncated to shard 1 of 2 satisfies every name it maps, but the loader opens the
    map and nothing else, so the omitted half is silently dropped."""
    snapshot = _pipeline_snapshot(
        tmp_path,
        _FLUX_INDEX,
        {
            "transformer/diffusion_pytorch_model.safetensors.index.json": json.dumps(
                {"weight_map": {"a": "diffusion_pytorch_model-00001-of-00002.safetensors"}}
            ).encode(),
            "transformer/diffusion_pytorch_model-00001-of-00002.safetensors": b"\0" * 256,
        },
    )
    assert inventory_scan.snapshot_pipeline_missing_denoiser(snapshot) is True

    (snapshot / "transformer" / "diffusion_pytorch_model-00002-of-00002.safetensors").write_bytes(
        b"\0" * 256
    )
    (snapshot / "transformer" / "diffusion_pytorch_model.safetensors.index.json").write_bytes(
        _dual_format_shard_index(".safetensors")
    )
    assert inventory_scan.snapshot_pipeline_missing_denoiser(snapshot) is False


def test_a_whole_bin_set_does_not_stand_in_for_the_selected_safetensors_index(tmp_path):
    """diffusers coerces an unset use_safetensors to True, so it resolves only
    diffusion_pytorch_model.safetensors.index.json here; finding it makes the component sharded,
    and both the IOError handler and the pickle fallback below it are gated on not is_sharded. The
    whole .bin set beside it is never opened, so it cannot vouch for the component."""
    snapshot = _pipeline_snapshot(
        tmp_path,
        _FLUX_INDEX,
        {
            "transformer/diffusion_pytorch_model.safetensors.index.json": (
                _dual_format_shard_index(".safetensors")
            ),
            "transformer/diffusion_pytorch_model-00001-of-00002.safetensors": b"\0" * 256,
            "transformer/diffusion_pytorch_model.bin.index.json": _dual_format_shard_index(".bin"),
            "transformer/diffusion_pytorch_model-00001-of-00002.bin": b"\0" * 256,
            "transformer/diffusion_pytorch_model-00002-of-00002.bin": b"\0" * 256,
        },
    )
    assert inventory_scan.snapshot_pipeline_missing_denoiser(snapshot) is True


def test_an_unsharded_denoiser_still_passes_on_presence_alone(tmp_path):
    """No index, so nothing to be incomplete against: presence alone stands."""
    snapshot = _pipeline_snapshot(
        tmp_path,
        _FLUX_INDEX,
        {"transformer/diffusion_pytorch_model.safetensors": b"\0" * 256},
    )
    assert inventory_scan.snapshot_pipeline_missing_denoiser(snapshot) is False


@pytest.mark.parametrize(
    "manifest, second",
    [
        (_IDEOGRAM_INDEX, "unconditional_transformer"),
        (_WAN_INDEX, "transformer_2"),
    ],
)
def test_a_multi_denoiser_pipeline_needs_every_denoiser_it_declares(manifest, second, tmp_path):
    """Ideogram 4 and the dual-expert video pipelines declare two denoisers, and the manifest is
    what says so, so both must be on disk before the snapshot reads as complete."""
    files = {
        "transformer/diffusion_pytorch_model.safetensors": b"\0" * 256,
        "vae/diffusion_pytorch_model.safetensors": b"\0" * 256,
    }
    snapshot = _pipeline_snapshot(tmp_path, manifest, files)
    assert inventory_scan.snapshot_pipeline_missing_denoiser(snapshot) is True

    (snapshot / second).mkdir()
    (snapshot / second / "diffusion_pytorch_model.safetensors").write_bytes(b"\0" * 256)
    assert inventory_scan.snapshot_pipeline_missing_denoiser(snapshot) is False


@pytest.mark.parametrize(
    "target, missing",
    [
        ("model_index.json", False),
        ("transformer/diffusion_pytorch_model.safetensors.index.json", True),
    ],
)
def test_json_too_deep_to_parse_is_contained_rather_than_raising(target, missing, tmp_path):
    """json.load raises RecursionError, which is neither a ValueError nor an OSError, so an
    unguarded parse would escape past the caller and drop the row from the scan entirely.

    Contained is not the same as ignored, and which one it is depends on the file. An unreadable
    MANIFEST proves nothing about the denoiser, so the row stays. An unreadable SELECTED INDEX is
    the failure itself: diffusers marks the component sharded on that file existing and then
    parses it with the same json module, so the whole weight lying beside it is never opened.
    """
    snapshot = _pipeline_snapshot(
        tmp_path,
        _FLUX_INDEX,
        {
            "transformer/diffusion_pytorch_model.safetensors": b"\0" * 256,
            target: b"[" * 20000 + b"]" * 20000,
        },
    )
    assert inventory_scan.snapshot_pipeline_missing_denoiser(snapshot) is missing


def test_an_unreadable_manifest_keeps_the_fixed_denoiser_pair(tmp_path):
    """A corrupt manifest falls back to the fixed pair, rather than reading as a pipeline that
    declares no denoiser and is therefore complete."""
    snapshot = tmp_path / "snap"
    snapshot.mkdir()
    (snapshot / "model_index.json").write_bytes(b"{not json")
    (snapshot / "vae").mkdir()
    (snapshot / "vae" / "diffusion_pytorch_model.safetensors").write_bytes(b"\0" * 256)
    assert inventory_scan.snapshot_pipeline_missing_denoiser(snapshot) is True

    (snapshot / "unet").mkdir()
    (snapshot / "unet" / "diffusion_pytorch_model.safetensors").write_bytes(b"\0" * 256)
    assert inventory_scan.snapshot_pipeline_missing_denoiser(snapshot) is False


def _snapshot_with(tmp_path, files: dict) -> Path:
    snapshot = tmp_path / "snap"
    snapshot.mkdir()
    for name, size in files.items():
        (snapshot / name).write_bytes(b"\0" * size)
    return snapshot


def test_a_torn_sibling_family_does_not_veto_the_quant_that_loads(tmp_path):
    """Two families can share one quant label. The lister offers the lexicographically first file
    and the loader loads that family alone, so the torn z- set beside a whole a-Q4_K_M.gguf is
    never selected and must not make the label unavailable."""
    snapshot = _snapshot_with(
        tmp_path,
        {"a-Q4_K_M.gguf": 64, "z-Q4_K_M-00001-of-00002.gguf": 16},
    )
    assert inventory_scan._completed_gguf_variants(snapshot) == {"Q4_K_M"}
    assert _local_offer(snapshot) == [("Q4_K_M", True)]


def test_a_whole_sibling_family_does_not_vouch_for_the_torn_one_that_loads(tmp_path):
    """The other direction, which is why this keys on the selected family rather than on any
    complete one: the torn set sorts first, so that is what the loader picks."""
    snapshot = _snapshot_with(
        tmp_path,
        {"a-Q4_K_M-00001-of-00002.gguf": 16, "z-Q4_K_M.gguf": 64},
    )
    assert inventory_scan._completed_gguf_variants(snapshot) == set()
    assert _local_offer(snapshot) == [("Q4_K_M", False)]


def test_the_selected_family_is_judged_on_its_own_shards(tmp_path):
    """A complete split set selected ahead of a torn sibling stays complete: the missing shard
    belongs to a family nothing reads."""
    snapshot = _snapshot_with(
        tmp_path,
        {
            "a-Q4_K_M-00001-of-00002.gguf": 16,
            "a-Q4_K_M-00002-of-00002.gguf": 16,
            "z-Q4_K_M-00001-of-00003.gguf": 16,
        },
    )
    assert inventory_scan._completed_gguf_variants(snapshot) == {"Q4_K_M"}


def test_two_torn_families_under_one_label_stay_incomplete(tmp_path):
    """Control: selecting one family must not turn two broken sets into a load."""
    snapshot = _snapshot_with(
        tmp_path,
        {"a-Q4_K_M-00001-of-00002.gguf": 16, "z-Q4_K_M-00001-of-00002.gguf": 16},
    )
    assert inventory_scan._completed_gguf_variants(snapshot) == set()


def test_a_nonsensical_shard_spec_is_never_complete(tmp_path):
    """A shard numbered past its own total cannot be loaded, and an empty index set must not read
    as a satisfied range."""
    snapshot = _snapshot_with(tmp_path, {"Model-Q4_K_M-00003-of-00002.gguf": 16})
    assert inventory_scan._completed_gguf_variants(snapshot) == set()


def test_the_loader_and_the_inventory_break_an_mtime_tie_the_same_way(tmp_path):
    """Two snapshots can carry the same mtime, and mtime alone is not a total order. The inventory
    pins by (mtime, resolved path); the loader's walk let filesystem order settle the rest, so the
    row could advertise one revision while /load read weights from the other."""
    import os

    from hub.utils.hf_cache_state import latest_snapshot_dir
    from utils.models.model_config import _iter_hf_cache_snapshots

    cache = tmp_path / "hub"
    repo = _repo_with(
        cache,
        snapshots = {
            SNAPSHOT: {"Model-Q4_K_M.gguf": b"\0" * 64},
            OLDER: {"Model-Q4_K_M.gguf": b"\0" * 64},
        },
        refs = {"main": SNAPSHOT},
    )
    stamp = 1_700_000_000
    for commit in (SNAPSHOT, OLDER):
        os.utime(repo / "snapshots" / commit, (stamp, stamp))
    assert len({(repo / "snapshots" / c).stat().st_mtime for c in (SNAPSHOT, OLDER)}) == 1

    loader_order = list(_iter_hf_cache_snapshots("Org/Model", cache_dir = cache))
    assert loader_order, "the loader must still find the snapshots"
    assert loader_order[0] == latest_snapshot_dir(repo)


def test_the_two_snapshot_orderings_agree_on_every_permutation(tmp_path):
    """The keys must be one ordering, not merely agree on the winner: a caller taking the
    second-newest has to see the same sequence."""
    import os

    from hub.utils.hf_cache_state import snapshot_selection_key
    from utils.models.model_config import _snapshot_selection_key

    cache = tmp_path / "hub"
    commits = [c * 40 for c in "abcd"]
    repo = _repo_with(
        cache,
        snapshots = {c: {"Model-Q4_K_M.gguf": b"\0" * 64} for c in commits},
        refs = {"main": commits[0]},
    )
    stamp = 1_700_000_000
    for index, commit in enumerate(commits):
        # Two pairs at equal mtimes, so ties decide half the ordering.
        moment = stamp + (index // 2)
        os.utime(repo / "snapshots" / commit, (moment, moment))

    snapshots = sorted((repo / "snapshots").iterdir())
    assert [snapshot_selection_key(s) for s in snapshots] == [
        _snapshot_selection_key(s) for s in snapshots
    ]
    assert list(_iter_hf_cache_snapshots_names(cache)) == [
        s.name for s in sorted(snapshots, key = snapshot_selection_key, reverse = True)
    ]


def _iter_hf_cache_snapshots_names(cache: Path) -> list[str]:
    from utils.models.model_config import _iter_hf_cache_snapshots
    return [s.name for s in _iter_hf_cache_snapshots("Org/Model", cache_dir = cache)]


@pytest.mark.parametrize(
    "ref_label, refs", [("dangling", {"main": UPSTREAM_HEAD}), ("resolving", {"main": SNAPSHOT})]
)
def test_a_stray_base_shard_does_not_veto_a_complete_adapter(
    ref_label, refs, tmp_path, monkeypatch
):
    """A LoRA snapshot can carry an unrelated interrupted base family. The row classifies as an
    adapter and the adapter is whole, so it loads; judging base first regardless of format let that
    stray shard veto it. No config.json is what says this snapshot can only be an adapter."""
    _repo_with(
        tmp_path,
        snapshots = {
            SNAPSHOT: {
                "adapter_config.json": b'{"peft_type":"LORA"}',
                "adapter_model.safetensors": b"\0" * 128,
                "pytorch_model-00001-of-00002.bin": b"\0" * 64,
            }
        },
        refs = refs,
    )
    rows = _autoload_rows(tmp_path, monkeypatch)
    assert [row["repo_id"] for row in rows] == ["Org/Model"]
    assert rows[0]["model_format"] == "adapter"
    assert rows[0]["partial"] is False
    assert rows[0]["capabilities"]["can_chat"] is True


@pytest.mark.parametrize(
    "files, cannot_serve",
    [
        # A real transformers file beside a config.json outranks the adapter, so the torn base
        # decides and the whole adapter cannot stand in.
        (
            {
                "config.json": b"{}",
                "model-00001-of-00002.safetensors": b"\0" * 64,
                "adapter_config.json": b"{}",
                "adapter_model.safetensors": b"\0" * 64,
            },
            True,
        ),
        # Same two configs, but the only real weights are the adapter's, so this loads as adapter.
        (
            {
                "config.json": b"{}",
                "pytorch_model-00001-of-00002.bin": b"\0" * 64,
                "adapter_config.json": b"{}",
                "adapter_model.safetensors": b"\0" * 64,
            },
            False,
        ),
        # A config.json that exists but is empty still classifies by name, and nothing can parse it.
        ({"config.json": b"", "model.safetensors": b"\0" * 256}, True),
        # Mixed extensions never form one family: two half sets are not one whole set.
        (
            {
                "config.json": b"{}",
                "model-00001-of-00002.safetensors": b"\0" * 64,
                "model-00002-of-00002.bin": b"\0" * 64,
            },
            True,
        ),
        # No config.json, so the row is the torn adapter and a whole base family cannot stand in.
        (
            {
                "adapter_config.json": b"{}",
                "adapter_model-00001-of-00002.safetensors": b"\0" * 64,
                "model.safetensors": b"\0" * 256,
            },
            True,
        ),
        ({"config.json": b"{}", "model.safetensors": b"\0" * 256}, False),
    ],
)
def test_the_judged_weight_family_follows_the_row_format(files, cannot_serve, tmp_path):
    """Both directions, so the fix cannot be read as "an adapter always rescues the snapshot"."""
    snapshot = tmp_path / "snap"
    snapshot.mkdir()
    for name, payload in files.items():
        (snapshot / name).write_bytes(payload)
    assert inventory_scan._snapshot_lacks_a_complete_weight_family(snapshot) is cannot_serve


def _compat_cached_models(cache_root: Path, monkeypatch) -> list[str]:
    """GET /api/models/cached-models. Its schema has no partial and no load_id, so it can only
    describe a repo that loads by id."""
    import asyncio
    from types import SimpleNamespace

    import routes.models as models_route

    monkeypatch.setattr(inventory_scan, "hf_cache_roots", lambda: [cache_root])
    monkeypatch.setattr(
        "utils.hf_cache_settings.get_hf_cache_paths",
        lambda: SimpleNamespace(hub_cache = cache_root),
    )
    inventory_scan.invalidate_hf_cache_scans()
    try:
        response = asyncio.run(
            models_route.list_cached_models(current_subject = "tester", hf_token = None)
        )
    finally:
        inventory_scan.invalidate_hf_cache_scans()
    return [row["repo_id"] for row in response["cached"]]


@pytest.mark.parametrize(
    "snapshot_files",
    [
        {"config.json": b"{}", "model-00001-of-00002.safetensors": b"\0" * 256},
        {"config.json": b"{}", "model.safetensors": b"\0" * 256},
    ],
    ids = ["short-a-shard", "whole-but-only-loadable-by-path"],
)
def test_the_compatibility_route_withholds_a_recovery_it_cannot_describe(
    snapshot_files, tmp_path, monkeypatch
):
    """Un-hiding a repo must not smuggle it into a response that cannot say what is wrong with it:
    with neither partial nor a load id, a torn recovery reads as a plain cached model and a whole
    one is offered under a repo id that does not resolve."""
    _repo_with(tmp_path, snapshots = {SNAPSHOT: snapshot_files}, refs = {"main": UPSTREAM_HEAD})
    assert _compat_cached_models(tmp_path, monkeypatch) == []
    # The Hub inventory still lists it, with the fields to describe it.
    assert [row["repo_id"] for row in _autoload_rows(tmp_path, monkeypatch)] == ["Org/Model"]


def test_the_compatibility_route_still_lists_what_upstream_returns(tmp_path, monkeypatch):
    """The control that bounds the gate: a repo whose refs/main resolves is one upstream already
    returned, so it is unaffected."""
    _repo_with(
        tmp_path,
        snapshots = {SNAPSHOT: {"config.json": b"{}", "model.safetensors": b"\0" * 256}},
        refs = {"main": SNAPSHOT},
    )
    assert _compat_cached_models(tmp_path, monkeypatch) == ["Org/Model"]


def test_a_recovery_whose_default_ref_resolves_is_still_listed(tmp_path, monkeypatch):
    """Recovery also fires when a secondary ref dangles while refs/main resolves.
    Those load by id perfectly well, so the gate must not swallow them."""
    _repo_with(
        tmp_path,
        snapshots = {SNAPSHOT: {"config.json": b"{}", "model.safetensors": b"\0" * 256}},
        refs = {"main": SNAPSHOT, "stale": UPSTREAM_HEAD},
    )
    assert _compat_cached_models(tmp_path, monkeypatch) == ["Org/Model"]


@pytest.mark.parametrize(
    "refs, on_disk, partial",
    [
        # The case this fixes: refs/main resolves, so the manifest describes what a repo-id load
        # reads and a stale ref elsewhere must not suppress it.
        ({"main": SNAPSHOT, "stale": UPSTREAM_HEAD}, 13, True),
        ({"main": SNAPSHOT}, 13, True),
        # refs/main dangling is the exemption: that attempt is pinned to a revision not on disk.
        ({"main": UPSTREAM_HEAD}, 13, False),
        # Negative control: a stale ref must not flag a snapshot that matches.
        ({"main": SNAPSHOT, "stale": UPSTREAM_HEAD}, 999, False),
    ],
    ids = ["stale-ref-beside-a-resolving-main", "no-stale-ref", "main-dangles", "matches-manifest"],
)
def test_a_stale_ref_does_not_suppress_the_manifest_on_the_loaded_snapshot(
    refs, on_disk, partial, tmp_path
):
    """Repo-wide signals are excused only when the load target itself is absent. Keying that on any
    dangling ref let a leftover tag hide a manifest mismatch on the very snapshot refs/main
    resolves to, so a truncated download went out ready. Only attribution narrowed; the recovery
    guard still keys on any ref."""
    from hub.utils import download_manifest

    repo = _repo_with(
        tmp_path,
        snapshots = {SNAPSHOT: {"config.json": b"{}", "model.safetensors": b"\0" * on_disk}},
        refs = refs,
    )
    download_manifest.write_manifest(
        "model",
        "Org/Model",
        None,
        [download_manifest.ExpectedFile("model.safetensors", 999)],
        "http",
        hub_cache = tmp_path,
    )
    snapshot = repo / "snapshots" / SNAPSHOT
    assert (
        inventory_scan.is_snapshot_partial("model", "Org/Model", repo, snapshot_dir = snapshot)
        is partial
    )


def test_the_compatibility_route_withholds_a_half_payload_landing(tmp_path, monkeypatch):
    """Weights pool across revisions, so a repo can look runnable while the snapshot refs/main
    lands on holds only the config. That directory is what an id-only caller loads, so it has to
    classify on its own: the shard check says nothing about a snapshot carrying no weights."""
    _repo_with(
        tmp_path,
        snapshots = {SNAPSHOT: {"config.json": b"{}"}, OLDER: {"model.safetensors": b"\0" * 256}},
        refs = {"main": SNAPSHOT, "stale": UPSTREAM_HEAD},
    )
    assert _compat_cached_models(tmp_path, monkeypatch) == []


def test_the_compatibility_route_lists_a_self_contained_landing(tmp_path, monkeypatch):
    """Control for the above: the same shape with the config and the weights together in the
    snapshot refs/main names is loadable by id and stays listed."""
    _repo_with(
        tmp_path,
        snapshots = {
            SNAPSHOT: {"config.json": b"{}", "model.safetensors": b"\0" * 256},
            OLDER: {"model.safetensors": b"\0" * 256},
        },
        refs = {"main": SNAPSHOT, "stale": UPSTREAM_HEAD},
    )
    assert _compat_cached_models(tmp_path, monkeypatch) == ["Org/Model"]


def test_an_adapter_beside_a_config_json_is_still_an_adapter(tmp_path, monkeypatch):
    """A LoRA snapshot legitimately ships the base model's config.json next to
    adapter_config.json. Reading the config alone as "this is a base row" made a whole adapter
    unusable whenever an unrelated torn base shard sat beside it."""
    _repo_with(
        tmp_path,
        snapshots = {
            SNAPSHOT: {
                "config.json": b'{"model_type":"llama"}',
                "adapter_config.json": b'{"peft_type":"LORA"}',
                "adapter_model.safetensors": b"\0" * 128,
                "pytorch_model-00001-of-00002.bin": b"\0" * 64,
            }
        },
        refs = {"main": UPSTREAM_HEAD},
    )
    rows = _autoload_rows(tmp_path, monkeypatch)
    assert rows[0]["model_format"] == "adapter"
    assert rows[0]["partial"] is False
    assert rows[0]["capabilities"]["can_chat"] is True


@pytest.mark.parametrize(
    "files, partial",
    [
        # Neither transformers family is whole; merging them made one look complete.
        (
            {
                "config.json": b'{"model_type":"llama"}',
                "model-00001-of-00002.safetensors": b"\0" * 64,
                "model-00002-of-00002.bin": b"\0" * 64,
            },
            True,
        ),
        # Control: one family whole in its own extension still loads.
        (
            {
                "config.json": b'{"model_type":"llama"}',
                "model-00001-of-00002.safetensors": b"\0" * 64,
                "model-00002-of-00002.safetensors": b"\0" * 64,
                "model.safetensors.index.json": _SHARD_INDEX,
            },
            False,
        ),
        # A zero-byte config.json fails to parse, whole weights or not.
        ({"config.json": b"", "model.safetensors": b"\0" * 256}, True),
    ],
    ids = ["shards-split-across-extensions", "one-whole-family", "empty-config"],
)
def test_a_recovered_snapshot_that_cannot_load_is_not_chattable(
    files, partial, tmp_path, monkeypatch
):
    """Three ways a recovered snapshot passes a shard count yet cannot be loaded."""
    _repo_with(tmp_path, snapshots = {SNAPSHOT: files}, refs = {"main": UPSTREAM_HEAD})
    rows = _autoload_rows(tmp_path, monkeypatch)
    assert [row["repo_id"] for row in rows] == ["Org/Model"]
    assert rows[0]["partial"] is partial
    assert rows[0]["capabilities"]["can_chat"] is not partial


def test_a_complete_older_snapshot_beats_a_torn_newer_one(tmp_path, monkeypatch):
    """The payload classification is by filename, so a revision interrupted mid-download still
    qualifies. Pinning the newest that merely classifies hid a complete older payload and made the
    row resume-only, with can_chat false, for a model that loads."""
    import os

    repo = _repo_with(
        tmp_path,
        snapshots = {
            OLDER: {"config.json": b'{"model_type":"llama"}', "model.safetensors": b"\0" * 256},
            SNAPSHOT: {
                "config.json": b'{"model_type":"llama"}',
                "model-00001-of-00002.safetensors": b"\0" * 64,
            },
        },
        refs = {"main": UPSTREAM_HEAD},
    )
    os.utime(repo / "snapshots" / OLDER, (1_700_000_000, 1_700_000_000))
    os.utime(repo / "snapshots" / SNAPSHOT, (1_700_009_999, 1_700_009_999))

    rows = _autoload_rows(tmp_path, monkeypatch)
    assert [row["repo_id"] for row in rows] == ["Org/Model"]
    assert Path(rows[0]["load_id"]) == repo / "snapshots" / OLDER
    assert rows[0]["partial"] is False
    assert rows[0]["capabilities"]["can_chat"] is True


def test_the_newest_snapshot_still_wins_when_both_are_complete(tmp_path, monkeypatch):
    """Control: preferring a complete payload must not turn into preferring an older one."""
    import os

    repo = _repo_with(
        tmp_path,
        snapshots = {
            OLDER: {"config.json": b'{"model_type":"llama"}', "model.safetensors": b"\0" * 256},
            SNAPSHOT: {"config.json": b'{"model_type":"llama"}', "model.safetensors": b"\0" * 256},
        },
        refs = {"main": UPSTREAM_HEAD},
    )
    os.utime(repo / "snapshots" / OLDER, (1_700_000_000, 1_700_000_000))
    os.utime(repo / "snapshots" / SNAPSHOT, (1_700_009_999, 1_700_009_999))

    rows = _autoload_rows(tmp_path, monkeypatch)
    assert Path(rows[0]["load_id"]) == repo / "snapshots" / SNAPSHOT
    assert rows[0]["partial"] is False


def test_an_empty_adapter_config_is_not_loadable(tmp_path, monkeypatch):
    """The adapter's required config gets the same treatment as config.json: recognised by name so
    the snapshot classifies, but peft cannot parse an empty one, so whole weights are not enough."""
    _repo_with(
        tmp_path,
        snapshots = {
            SNAPSHOT: {"adapter_config.json": b"", "adapter_model.safetensors": b"\0" * 128}
        },
        refs = {"main": UPSTREAM_HEAD},
    )
    rows = _autoload_rows(tmp_path, monkeypatch)
    assert rows[0]["model_format"] == "adapter"
    assert rows[0]["partial"] is True
    assert rows[0]["capabilities"]["can_chat"] is False


def test_an_empty_base_config_does_not_block_an_adapter_row(tmp_path, monkeypatch):
    """The veto is scoped to the format that has to parse the file. An adapter loads through
    adapter_config.json, so a stray empty config.json beside it is not its problem."""
    _repo_with(
        tmp_path,
        snapshots = {
            SNAPSHOT: {
                "config.json": b"",
                "adapter_config.json": b'{"peft_type":"LORA"}',
                "adapter_model.safetensors": b"\0" * 128,
            }
        },
        refs = {"main": UPSTREAM_HEAD},
    )
    rows = _autoload_rows(tmp_path, monkeypatch)
    assert rows[0]["model_format"] == "adapter"
    assert rows[0]["partial"] is False
    assert rows[0]["capabilities"]["can_chat"] is True


def test_a_local_gguf_file_path_is_judged_on_its_directory(tmp_path):
    """A load id can name the .gguf file itself. The lister resolves that to the parent, so the
    completion walk has to as well: walking a regular file finds no quants, and every variant of a
    whole local model came back partial and unofferable."""
    folder = tmp_path / "local_model"
    folder.mkdir()
    (folder / "config.json").write_bytes(b'{"model_type":"llama"}')
    (folder / "Model-Q4_K_M.gguf").write_bytes(b"\0" * 64)

    assert inventory_scan._completed_gguf_variants(folder / "Model-Q4_K_M.gguf") == {"Q4_K_M"}
    assert _local_offer(folder / "Model-Q4_K_M.gguf") == [("Q4_K_M", True)]


def test_a_bare_gguf_file_with_no_marker_is_not_resolved(tmp_path):
    """The resolve only fires where the lister's does, on a parent holding one of its marker files.
    Without one the lister offers nothing, so there is nothing to call complete either."""
    folder = tmp_path / "loose"
    folder.mkdir()
    (folder / "Model-Q4_K_M.gguf").write_bytes(b"\0" * 64)

    assert inventory_scan._completed_gguf_variants(folder / "Model-Q4_K_M.gguf") == set()


def test_an_adapter_file_does_not_stand_in_for_a_checkpoint_row(tmp_path, monkeypatch):
    """adapter_model.bin reads as checkpoint-like, so a snapshot holding it beside a config.json
    and no adapter_config.json classifies checkpoint. The completeness walk then accepted the
    adapter file as that row's payload, though a checkpoint load finds no base weights."""
    _repo_with(
        tmp_path,
        snapshots = {
            SNAPSHOT: {"config.json": b'{"model_type":"llama"}', "adapter_model.bin": b"\0" * 256}
        },
        refs = {"main": UPSTREAM_HEAD},
    )
    rows = _autoload_rows(tmp_path, monkeypatch)
    assert rows[0]["model_format"] == "checkpoint"
    assert rows[0]["partial"] is True
    assert rows[0]["capabilities"]["can_chat"] is False


@pytest.mark.parametrize(
    "files",
    [
        {"config.json": b"{}", "model.safetensors": b"\0" * 256},
        {"adapter_config.json": b'{"peft_type":"LORA"}', "adapter_model.bin": b"\0" * 256},
        # Classifies from the suffix while naming no family, and absence of one is not evidence.
        {"config.json": b"{}", "diffusion_pytorch_model.safetensors": b"\0" * 256},
        {"config.json": b"{}", "model.ckpt": b"\0" * 256},
    ],
    ids = ["safetensors", "adapter", "diffusion", "ckpt-file"],
)
def test_a_payload_whose_own_kind_is_present_stays_chattable(files, tmp_path, monkeypatch):
    """The controls that bound the rule above: only weights of the OTHER kind standing in is
    evidence of a mismatch, and finding no family at all is not."""
    _repo_with(tmp_path, snapshots = {SNAPSHOT: files}, refs = {"main": UPSTREAM_HEAD})
    rows = _autoload_rows(tmp_path, monkeypatch)
    assert [row["repo_id"] for row in rows] == ["Org/Model"]
    assert rows[0]["partial"] is False
    assert rows[0]["capabilities"]["can_chat"] is True


@pytest.mark.parametrize(
    "files, partial",
    [
        # from_pretrained never globs, so without an index the set is unreachable.
        (
            {
                "config.json": b'{"model_type":"llama"}',
                "model-00001-of-00002.safetensors": b"\0" * 64,
                "model-00002-of-00002.safetensors": b"\0" * 64,
            },
            True,
        ),
        # An empty index parses no map, so it is no better than an absent one.
        (
            {
                "config.json": b'{"model_type":"llama"}',
                "model-00001-of-00002.safetensors": b"\0" * 64,
                "model-00002-of-00002.safetensors": b"\0" * 64,
                "model.safetensors.index.json": b"",
            },
            True,
        ),
        (
            {
                "config.json": b'{"model_type":"llama"}',
                "model-00001-of-00002.safetensors": b"\0" * 64,
                "model-00002-of-00002.safetensors": b"\0" * 64,
                "model.safetensors.index.json": _SHARD_INDEX,
            },
            False,
        ),
        # The index is named for its family: a .bin set wants pytorch_model.bin.index.json.
        (
            {
                "config.json": b'{"model_type":"llama"}',
                "pytorch_model-00001-of-00002.bin": b"\0" * 64,
                "pytorch_model-00002-of-00002.bin": b"\0" * 64,
                "pytorch_model.bin.index.json": _BIN_SHARD_INDEX,
            },
            False,
        ),
        # No index rescues a numbered adapter set: peft resolves only the singular name.
        (
            {
                "adapter_config.json": b'{"peft_type":"LORA"}',
                "adapter_model-00001-of-00002.safetensors": b"\0" * 64,
                "adapter_model-00002-of-00002.safetensors": b"\0" * 64,
                "adapter_model.safetensors.index.json": _shard_index(
                    "adapter_model-00001-of-00002.safetensors",
                    "adapter_model-00002-of-00002.safetensors",
                ),
            },
            True,
        ),
    ],
    ids = ["no-index", "empty-index", "with-index", "bin-index", "adapter-shards-never-load"],
)
def test_a_sharded_base_family_needs_its_index(files, partial, tmp_path, monkeypatch):
    """A complete shard set is not a loadable payload on its own."""
    _repo_with(tmp_path, snapshots = {SNAPSHOT: files}, refs = {"main": UPSTREAM_HEAD})
    rows = _autoload_rows(tmp_path, monkeypatch)
    assert [row["repo_id"] for row in rows] == ["Org/Model"]
    assert rows[0]["partial"] is partial
    assert rows[0]["capabilities"]["can_chat"] is not partial


def test_an_index_less_shard_set_does_not_veto_a_whole_family(tmp_path, monkeypatch):
    """The index makes a family uncountable, not the snapshot unusable. A whole model.safetensors
    beside an index-less .bin set still serves the row, the same as beside a torn one."""
    _repo_with(
        tmp_path,
        snapshots = {
            SNAPSHOT: {
                "config.json": b'{"model_type":"llama"}',
                "model.safetensors": b"\0" * 256,
                "pytorch_model-00001-of-00002.bin": b"\0" * 64,
                "pytorch_model-00002-of-00002.bin": b"\0" * 64,
            }
        },
        refs = {"main": UPSTREAM_HEAD},
    )
    rows = _autoload_rows(tmp_path, monkeypatch)
    assert rows[0]["partial"] is False
    assert rows[0]["capabilities"]["can_chat"] is True


def test_a_root_weight_no_runtime_discovers_by_name_does_not_serve(tmp_path, monkeypatch):
    """An arbitrary root .safetensors names no family and is no diffusers component, so nothing
    opens it. A .ckpt or diffusion weight is discovered by name and still serves."""
    for weights, serves in (
        ({"foo.safetensors": b"\0" * 256}, False),
        ({"foo.safetensors": b"\0" * 256, "bar.safetensors": b"\0" * 256}, False),
        ({"model.ckpt": b"\0" * 256}, True),
        ({"diffusion_pytorch_model.safetensors": b"\0" * 256}, True),
    ):
        root = tmp_path / "-".join(sorted(weights)).replace(".", "_")
        _repo_with(
            root,
            snapshots = {OLDER: {"config.json": b'{"model_type":"llama"}', **weights}},
            refs = {"main": UPSTREAM_HEAD},
        )

        rows = _autoload_rows(root, monkeypatch)

        assert rows[0]["partial"] is not serves
        assert rows[0]["capabilities"]["can_chat"] is serves


def test_a_root_weight_no_runtime_discovers_does_not_veto_one_it_does(tmp_path, monkeypatch):
    """Control for the test above: an unopened name is no evidence; a canonical weight decides."""
    _repo_with(
        tmp_path,
        snapshots = {
            OLDER: {
                "config.json": b'{"model_type":"llama"}',
                "foo.safetensors": b"\0" * 256,
                "model.safetensors": b"\0" * 256,
            }
        },
        refs = {"main": UPSTREAM_HEAD},
    )

    rows = _autoload_rows(tmp_path, monkeypatch)

    assert rows[0]["partial"] is False
    assert rows[0]["capabilities"]["can_chat"] is True


def test_a_torn_quant_does_not_charge_a_stale_incomplete_blob_to_the_weights_row(
    tmp_path, monkeypatch
):
    """The legacy .incomplete walk attributes a repo-wide signal, and that question is per row: a
    weights row never opens a .gguf, so a torn quant must not charge it the leftover blob."""
    repo_dir = _repo_with(
        tmp_path,
        snapshots = {
            OLDER: {
                "config.json": b'{"model_type":"llama"}',
                "model.safetensors": b"\0" * 256,
                "Model-Q4_K_M-00001-of-00002.gguf": b"GGUF" + b"\0" * 252,
            }
        },
        refs = {"main": UPSTREAM_HEAD},
    )
    (repo_dir / "blobs" / "abc123.incomplete").write_bytes(b"\0" * 8)

    rows = _autoload_rows(tmp_path, monkeypatch)

    assert rows[0]["partial"] is False
    assert rows[0]["capabilities"]["can_chat"] is True


def test_a_stale_incomplete_blob_still_reaches_the_snapshot_it_describes(tmp_path, monkeypatch):
    """Control for the test above: with no quant to mis-attribute, the blob charges the row."""
    repo_dir = _repo_with(
        tmp_path,
        snapshots = {
            OLDER: {
                "config.json": b'{"model_type":"llama"}',
                "model.safetensors.index.json": _shard_index(
                    "model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"
                ),
                "model-00001-of-00002.safetensors": b"\0" * 256,
            }
        },
        refs = {"main": UPSTREAM_HEAD},
    )
    (repo_dir / "blobs" / "abc123.incomplete").write_bytes(b"\0" * 8)

    rows = _autoload_rows(tmp_path, monkeypatch)

    assert rows[0]["partial"] is True
    assert rows[0]["capabilities"]["can_chat"] is False


def test_an_index_whose_map_holds_a_non_string_does_not_hide_the_repo(tmp_path, monkeypatch):
    """A weight_map value is whatever the file says, not necessarily a path. Reading one has to
    survive it: a raised TypeError escapes the scan and drops every row for the repo, so a hand
    edited index next to a loadable checkpoint would make the whole model disappear."""
    _repo_with(
        tmp_path,
        snapshots = {
            SNAPSHOT: {
                "config.json": b'{"model_type":"llama"}',
                "pytorch_model.bin": b"\0" * 256,
                "model.safetensors.index.json": json.dumps(
                    {"metadata": {}, "weight_map": {"w0": []}}
                ).encode(),
            }
        },
        refs = {"main": SNAPSHOT},
    )

    rows = _autoload_rows(tmp_path, monkeypatch)

    assert [row["repo_id"] for row in rows] == ["Org/Model"]
    assert rows[0]["partial"] is False
    assert rows[0]["capabilities"]["can_chat"] is True


def test_a_stale_root_shard_beside_a_whole_index_does_not_tear_the_snapshot(tmp_path, monkeypatch):
    """The index names nested shards and every one of them is there, so from_pretrained reads a
    whole model. A leftover root shard the map names nothing of is never opened, and judging the
    walk's families ahead of the index that was selected charges the load for it anyway."""
    _repo_with(
        tmp_path,
        snapshots = {
            SNAPSHOT: {
                "config.json": b'{"model_type":"llama"}',
                "model.safetensors.index.json": _shard_index(
                    "weights/model-00001-of-00001.safetensors"
                ),
                "weights/model-00001-of-00001.safetensors": b"\0" * 256,
                "model-00001-of-00002.safetensors": b"\0" * 256,
            }
        },
        refs = {"main": UPSTREAM_HEAD},
    )

    rows = _autoload_rows(tmp_path, monkeypatch)

    assert rows[0]["partial"] is False
    assert rows[0]["capabilities"]["can_chat"] is True


def test_an_index_naming_half_of_the_family_it_describes_still_reads_as_torn(tmp_path, monkeypatch):
    """Control for the test above. A shard names its own total, so an index listing one of a set
    has to list the whole set: the loader opens exactly what is mapped and silently drops the
    rest, which is an interrupted download rather than stale content beside a whole one."""
    _repo_with(
        tmp_path,
        snapshots = {
            SNAPSHOT: {
                "config.json": b'{"model_type":"llama"}',
                "model.safetensors.index.json": _shard_index("model-00001-of-00002.safetensors"),
                "model-00001-of-00002.safetensors": b"\0" * 256,
            }
        },
        refs = {"main": UPSTREAM_HEAD},
    )

    rows = _autoload_rows(tmp_path, monkeypatch)

    assert rows[0]["partial"] is True
    assert rows[0]["capabilities"]["can_chat"] is False


def test_a_torn_revision_does_not_choose_the_format_over_a_whole_one(tmp_path, monkeypatch):
    """The repo-wide format flags are OR-ed across revisions, so they can name a format whose every
    revision is torn while another format has a whole revision sitting right there. Load what
    loads: an interrupted safetensors attempt must not hide the checkpoint beside it."""
    repo_dir = _repo_with(
        tmp_path,
        snapshots = {
            OLDER: {
                "config.json": b'{"model_type":"llama"}',
                "model.safetensors.index.json": _SHARD_INDEX,
                "model-00001-of-00002.safetensors": b"\0" * 256,
            },
            NEWER: {
                "config.json": b'{"model_type":"llama"}',
                "pytorch_model.bin": b"\0" * 256,
            },
        },
        refs = {"main": UPSTREAM_HEAD},
    )
    _age(repo_dir / "snapshots" / OLDER, 600)

    rows = _autoload_rows(tmp_path, monkeypatch)

    assert rows[0]["model_format"] == "checkpoint"
    assert Path(rows[0]["load_id"]).name == NEWER
    assert rows[0]["partial"] is False
    assert rows[0]["capabilities"]["can_chat"] is True


def test_a_repo_whose_only_format_is_torn_still_reads_as_torn(tmp_path, monkeypatch):
    """Control for the test above: with no whole revision in any other format there is nothing to
    fall back to, so the torn safetensors revision stays the answer."""
    _repo_with(
        tmp_path,
        snapshots = {
            OLDER: {
                "config.json": b'{"model_type":"llama"}',
                "model.safetensors.index.json": _SHARD_INDEX,
                "model-00001-of-00002.safetensors": b"\0" * 256,
            }
        },
        refs = {"main": UPSTREAM_HEAD},
    )

    rows = _autoload_rows(tmp_path, monkeypatch)

    assert rows[0]["model_format"] == "safetensors"
    assert rows[0]["partial"] is True
    assert rows[0]["capabilities"]["can_chat"] is False


def test_a_shard_total_from_a_filename_is_not_materialised(tmp_path):
    """The total comes out of a repo-controlled filename, so it can name a set far larger than
    anything on disk. Recovery runs this check over repos scan_cache_dir used to drop, so listing
    cached models must not allocate a set per declared shard: -of-999999999 costs gigabytes."""
    snapshot = tmp_path / "snapshots" / SNAPSHOT
    snapshot.mkdir(parents = True)
    (snapshot / "Model-Q4_K_M-00001-of-1000000.gguf").write_bytes(b"GGUF" + b"\0" * 252)

    tracemalloc.start()
    try:
        complete = inventory_scan._completed_gguf_variants(snapshot)
        _current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert complete == set()
    assert peak < 8 * 1024 * 1024, f"allocated {peak} bytes for a declared set of 1000000 shards"


def test_a_whole_split_quant_is_still_complete(tmp_path):
    """Control for the test above: a set whose shards are all present still reads complete."""
    snapshot = tmp_path / "snapshots" / SNAPSHOT
    snapshot.mkdir(parents = True)
    for index in (1, 2):
        (snapshot / f"Model-Q4_K_M-0000{index}-of-00002.gguf").write_bytes(b"GGUF" + b"\0" * 252)

    assert inventory_scan._completed_gguf_variants(snapshot) == {"Q4_K_M"}
