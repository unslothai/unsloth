# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import asyncio
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from hub.schemas.datasets import CheckFormatRequest, LocalDatasetItem
from hub.services.datasets import (
    cache_inventory,
    downloads,
    formatting,
    local,
    local_options,
)
from hub.utils import (
    dataset_processed_cache,
    download_manifest,
    download_registry,
    hf_cache_state,
    state_dir,
)


@pytest.fixture(autouse = True)
def _app_dataset_cache_root(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "utils.paths.storage_roots.cache_root",
        lambda: tmp_path / "app-cache",
    )


class _Upload:
    def __init__(self, filename: str, payload: bytes):
        self.filename = filename
        self._payload = payload
        self._offset = 0

    async def read(self, size: int) -> bytes:
        if self._offset >= len(self._payload):
            return b""
        chunk = self._payload[self._offset : self._offset + size]
        self._offset += len(chunk)
        return chunk


def test_dataset_cache_scan_merges_raw_and_processed_rows(monkeypatch):
    raw_repo = SimpleNamespace(
        repo_id = "Org/Data",
        repo_type = "dataset",
        repo_path = "/cache/datasets--Org--Data",
        size_on_disk = 100,
        revisions = [SimpleNamespace(files = [], commit_hash = "abc")],
    )
    monkeypatch.setattr(
        cache_inventory,
        "_collect_hf_cache_scans",
        lambda: ([SimpleNamespace(repos = [raw_repo])], {"/cache"}),
    )
    monkeypatch.setattr(
        cache_inventory.hf_cache_scan,
        "is_snapshot_partial",
        lambda _repo_type, _repo_id, _cache_dir: False,
    )
    monkeypatch.setattr(cache_inventory, "_raw_dataset_cache_has_data", lambda *_args: True)
    monkeypatch.setattr(
        cache_inventory,
        "_scan_hub_dataset_cache_dirs",
        lambda: [],
    )
    monkeypatch.setattr(
        cache_inventory,
        "_scan_processed_dataset_caches",
        lambda: [
            {
                "repo_id": "org/data",
                "size_bytes": 250,
                "cache_path": "/processed/org___data",
                "processed_cache": True,
                "partial": False,
            }
        ],
    )

    rows = cache_inventory._scan_hf_dataset_caches()

    assert len(rows) == 1
    assert rows[0]["repo_id"] == "Org/Data"
    assert rows[0]["size_bytes"] == 250
    assert rows[0]["cache_path"] == "/cache/datasets--Org--Data"
    assert rows[0]["load_cache_path"] == "/processed/org___data"
    assert rows[0]["partial"] is False


def test_dataset_cache_scan_attaches_app_bytes_without_replacing_raw_path(monkeypatch):
    raw_repo = SimpleNamespace(
        repo_id = "Org/Data",
        repo_type = "dataset",
        repo_path = "/cache/hub/datasets--Org--Data",
        size_on_disk = 100,
        revisions = [SimpleNamespace(files = [], commit_hash = "abc")],
    )
    monkeypatch.setattr(
        cache_inventory,
        "_collect_hf_cache_scans",
        lambda: ([SimpleNamespace(repos = [raw_repo])], {"/cache/hub"}),
    )
    monkeypatch.setattr(
        cache_inventory.hf_cache_scan,
        "is_snapshot_partial",
        lambda *_args, **_kwargs: False,
    )
    monkeypatch.setattr(cache_inventory, "_raw_dataset_cache_has_data", lambda *_args: True)
    monkeypatch.setattr(cache_inventory, "_scan_hub_dataset_cache_dirs", lambda: [])
    monkeypatch.setattr(cache_inventory, "_scan_processed_dataset_caches", lambda: [])
    monkeypatch.setattr(
        cache_inventory,
        "_scan_app_processed_dataset_caches",
        lambda: [
            {
                "repo_id": "org/data",
                "size_bytes": 40,
                "cache_path": "/app/entry",
                "processed_cache": True,
                "app_processed_cache": True,
                "app_processed_hub_cache": "/cache/hub",
                "partial": True,
            }
        ],
    )

    rows = cache_inventory._scan_hf_dataset_caches()

    assert rows == [
        {
            "repo_id": "Org/Data",
            "size_bytes": 140,
            "cache_path": "/cache/hub/datasets--Org--Data",
            "partial": False,
            "partial_transport": None,
            "partial_resumable": False,
            "processed_cache": True,
            "app_processed_cache": True,
        }
    ]


def _dataset_snapshot(monkeypatch, tmp_path: Path, filenames: tuple[str, ...]) -> Path:
    hub_cache = tmp_path / "hub"
    repo_root = hub_cache / "datasets--Org--Data"
    snapshot = repo_root / "snapshots" / "abc"
    snapshot.mkdir(parents = True)
    for filename in filenames:
        target = snapshot / filename
        target.parent.mkdir(parents = True, exist_ok = True)
        target.write_bytes(b"x")
    monkeypatch.setattr(hf_cache_state, "hf_cache_roots", lambda **kw: [hub_cache])
    return repo_root


def test_raw_dataset_cache_has_data_rejects_a_metadata_only_snapshot(monkeypatch, tmp_path):
    repo_root = _dataset_snapshot(
        monkeypatch,
        tmp_path,
        (".gitattributes", "README.md", "dataset_infos.json", "LICENSE"),
    )

    assert cache_inventory._raw_dataset_cache_has_data("Org/Data", repo_root) is False


def test_raw_dataset_cache_has_data_ignores_os_clutter(monkeypatch, tmp_path):
    """Opening the cache dir in Finder or Explorer drops a `.DS_Store`/`Thumbs.db` beside the
    card, which must not read as payload."""
    repo_root = _dataset_snapshot(
        monkeypatch,
        tmp_path,
        ("README.md", ".DS_Store", "Thumbs.db"),
    )

    assert cache_inventory._raw_dataset_cache_has_data("Org/Data", repo_root) is False


def test_raw_dataset_cache_has_data_finds_nested_payload_of_any_format(monkeypatch, tmp_path):
    """Image and audio repos ship no extension the app keeps a format list for, so the check
    asks whether anything beyond metadata is present rather than matching known formats."""
    repo_root = _dataset_snapshot(
        monkeypatch,
        tmp_path,
        ("README.md", "data/train-00000-of-00001.parquet", "data/train/0001.png"),
    )

    assert cache_inventory._raw_dataset_cache_has_data("Org/Data", repo_root) is True


def test_raw_dataset_cache_has_data_ignores_appledouble_sidecars(monkeypatch, tmp_path):
    """A snapshot carried through a Mac zip picks up `._name` sidecars and a `__MACOSX`
    tree. `datasets` skips dotted names and `__`-prefixed dirs when it resolves data files,
    so counting them as payload offered a card-only snapshot On Device again."""
    repo_root = _dataset_snapshot(
        monkeypatch,
        tmp_path,
        ("README.md", "._README.md", "__MACOSX/._README.md"),
    )

    assert cache_inventory._raw_dataset_cache_has_data("Org/Data", repo_root) is False


def test_raw_dataset_cache_has_data_ignores_a_citation_and_a_loading_script(monkeypatch, tmp_path):
    """Neither suffix is in `datasets`' extension map, and a script also needs
    `trust_remote_code`, which no load path here passes and `datasets>=4` removed."""
    repo_root = _dataset_snapshot(
        monkeypatch,
        tmp_path,
        ("README.md", "CITATION.cff", "data.py", "docs/usage.md"),
    )

    assert cache_inventory._raw_dataset_cache_has_data("Org/Data", repo_root) is False


def test_raw_dataset_cache_has_data_counts_payload_beside_a_loading_script(monkeypatch, tmp_path):
    """The suffix rule is for files only: a script beside real data is still payload."""
    repo_root = _dataset_snapshot(
        monkeypatch,
        tmp_path,
        ("README.md", "data.py", "data/train.parquet"),
    )

    assert cache_inventory._raw_dataset_cache_has_data("Org/Data", repo_root) is True


def test_raw_dataset_cache_has_data_counts_payload_under_a_metadata_named_dir(
    monkeypatch, tmp_path
):
    """`datasets` excludes only hidden and `__`-prefixed dirs, so `license/train.parquet`
    resolves; applying the file list to the directory hid the dataset from On Device."""
    repo_root = _dataset_snapshot(
        monkeypatch,
        tmp_path,
        ("README.md", "license/train.parquet"),
    )

    assert cache_inventory._raw_dataset_cache_has_data("Org/Data", repo_root) is True


def test_raw_dataset_cache_has_data_does_not_loop_on_a_link_to_an_ancestor(monkeypatch, tmp_path):
    """A link back inside the snapshot passes containment, so only the visited set stops the
    descent. Linux `os.walk` declines a symlink on its own, which would make this pass either
    way, so the walk here descends it the way Windows descends a junction."""
    repo_root = _dataset_snapshot(monkeypatch, tmp_path, ("README.md", "data/notes.md"))
    data = next((repo_root / "snapshots").iterdir()) / "data"
    (data / "loop").symlink_to(data, target_is_directory = True)
    # a junction resolves like a link while reporting is_symlink() False, and Linux cannot create one,
    # so this is a symlink that answers the way a junction does
    real_is_symlink = Path.is_symlink
    monkeypatch.setattr(
        Path,
        "is_symlink",
        lambda self: False if self.name == "loop" else real_is_symlink(self),
    )
    depth = 0

    def _walk_following_junctions(
        top,
        followlinks = False,
        onerror = None,
    ):
        nonlocal depth
        stack = [Path(top)]
        while stack:
            base = stack.pop()
            depth = max(depth, len(base.parts))
            assert depth < 40, "the walk never pruned the link back to an ancestor"
            names = sorted(entry.name for entry in base.iterdir())
            dirnames = [name for name in names if (base / name).is_dir()]
            filenames = [name for name in names if name not in dirnames]
            yield str(base), dirnames, filenames
            stack.extend(base / name for name in dirnames)

    monkeypatch.setattr(cache_inventory.os, "walk", _walk_following_junctions)

    assert cache_inventory._raw_dataset_cache_has_data("Org/Data", repo_root) is False


def test_raw_dataset_cache_has_data_rejects_an_empty_payload_file(monkeypatch, tmp_path):
    """A zero-byte file looks like payload and yields nothing, so this asks
    `local_options._empty_payload` rather than settling for the file existing."""
    repo_root = _dataset_snapshot(monkeypatch, tmp_path, ("README.md",))
    snapshot = next((repo_root / "snapshots").iterdir())
    (snapshot / "train.jsonl").write_bytes(b"")

    assert cache_inventory._raw_dataset_cache_has_data("Org/Data", repo_root) is False

    (snapshot / "train.jsonl").write_bytes(b'{"text": "x"}\n')

    assert cache_inventory._raw_dataset_cache_has_data("Org/Data", repo_root) is True


def test_raw_dataset_cache_has_data_rejects_a_payload_file_with_no_rows(monkeypatch, tmp_path):
    """Bytes are not rows: `_rowless` probes a header-only csv and a `[]` json, from which the
    picker resolves no split. A rowless file is dropped, but it does not condemn its siblings."""
    repo_root = _dataset_snapshot(monkeypatch, tmp_path, ("README.md",))
    snapshot = next((repo_root / "snapshots").iterdir())
    (snapshot / "train.csv").write_bytes(b"text,label\n")
    (snapshot / "extra.json").write_bytes(b"[]")

    assert cache_inventory._raw_dataset_cache_has_data("Org/Data", repo_root) is False

    (snapshot / "train.csv").write_bytes(b"text,label\nhello,1\n")

    assert cache_inventory._raw_dataset_cache_has_data("Org/Data", repo_root) is True


def test_raw_dataset_cache_has_data_walks_a_redirect_instead_of_trusting_it(monkeypatch, tmp_path):
    """Pruning a redirect hid migrated caches; taking one as proof is the opposite error, since
    a stale redirect holds no rows. The target is walked, not assumed."""
    stale = tmp_path / "elsewhere" / "stale"
    stale.mkdir(parents = True)
    (stale / "README.md").write_bytes(b"x")
    repo_root = _dataset_snapshot(monkeypatch, tmp_path, ("README.md",))
    snapshot = next((repo_root / "snapshots").iterdir())
    (snapshot / "data").symlink_to(stale, target_is_directory = True)

    assert cache_inventory._raw_dataset_cache_has_data("Org/Data", repo_root) is False

    (stale / "train.parquet").write_bytes(b"PAR1" + b"\0" * 64)

    assert cache_inventory._raw_dataset_cache_has_data("Org/Data", repo_root) is True


def test_raw_dataset_cache_has_data_keeps_the_real_dir_when_an_alias_precedes_it(
    monkeypatch, tmp_path
):
    """The walk never descends the alias on POSIX, so booking its target pruned the real
    directory and lost the payload under it, on nothing but enumeration order."""
    repo_root = _dataset_snapshot(monkeypatch, tmp_path, ("README.md", "data/train.parquet"))
    snapshot = next((repo_root / "snapshots").iterdir())
    (snapshot / "alias").symlink_to(snapshot / "data", target_is_directory = True)

    def _walk_alias_first(
        top,
        followlinks = False,
        onerror = None,
    ):
        stack = [Path(top)]
        while stack:
            base = stack.pop()
            names = sorted(entry.name for entry in base.iterdir())
            dirnames = [name for name in names if (base / name).is_dir()]
            filenames = [name for name in names if name not in dirnames]
            yield str(base), dirnames, filenames
            # POSIX: a symlinked directory is listed but not descended.
            stack.extend(base / name for name in dirnames if not (base / name).is_symlink())

    monkeypatch.setattr(cache_inventory.os, "walk", _walk_alias_first)

    assert cache_inventory._raw_dataset_cache_has_data("Org/Data", repo_root) is True


def test_raw_dataset_cache_has_data_reads_the_suffix_chain_like_the_resolver(monkeypatch, tmp_path):
    """`_data_suffix` drops a trailing compression suffix then walks the rest. Reading only the
    last suffix got both directions backwards."""
    metadata_only = _dataset_snapshot(
        monkeypatch,
        tmp_path / "meta",
        ("README.md", "CITATION.cff.zip", "loader.py.gz"),
    )
    compressed_payload = _dataset_snapshot(
        monkeypatch,
        tmp_path / "data",
        ("README.md", "train.parquet.gz", "records.parquet.backup"),
    )

    assert cache_inventory._raw_dataset_cache_has_data("Org/Data", metadata_only) is False
    assert cache_inventory._raw_dataset_cache_has_data("Org/Data", compressed_payload) is True


def test_raw_dataset_cache_has_data_ignores_payload_links_whose_blob_is_gone(monkeypatch, tmp_path):
    """`is_snapshot_partial` catches this on the newest snapshot, but this check judges the
    pinned revision, so it must not depend on the two agreeing."""
    repo_root = _dataset_snapshot(monkeypatch, tmp_path, ("README.md",))
    snapshot = next((repo_root / "snapshots").iterdir())
    (snapshot / "train.parquet").symlink_to("../../blobs/pruned")

    assert cache_inventory._raw_dataset_cache_has_data("Org/Data", repo_root) is False


def test_raw_dataset_cache_has_data_counts_a_linked_payload_directory(monkeypatch, tmp_path):
    """A migrated or shared cache can keep its data behind a directory link. The walk must
    not descend into one -- it can point anywhere -- but pruning it silently made the repo
    read as metadata-only, and On Device drops every partial row."""
    payload = tmp_path / "elsewhere"
    payload.mkdir()
    (payload / "train-00000-of-00001.parquet").write_bytes(b"PAR1")
    repo_root = _dataset_snapshot(monkeypatch, tmp_path, ("README.md",))
    (repo_root / "snapshots" / "abc" / "data").symlink_to(payload, target_is_directory = True)

    assert cache_inventory._raw_dataset_cache_has_data("Org/Data", repo_root) is True


def test_raw_dataset_cache_has_data_ignores_payload_in_an_unpinned_revision(monkeypatch, tmp_path):
    """A payload-bearing sibling revision does not make the row usable.

    `training_dataset_cache_pin` resolves through `dataset_snapshot_from_cache_path`, which
    prefers the revision `refs/main` names. Clearing `partial` because some other revision
    holds data would offer a row whose pinned load path is the metadata-only revision, so the
    run falls back to the network and fails outright offline."""
    repo_root = _dataset_snapshot(monkeypatch, tmp_path, ("README.md",))
    (repo_root / "refs").mkdir(parents = True, exist_ok = True)
    (repo_root / "refs" / "main").write_text("abc")
    other = repo_root / "snapshots" / "def"
    other.mkdir(parents = True)
    (other / "train-00000-of-00001.parquet").write_bytes(b"PAR1")

    assert cache_inventory._raw_dataset_cache_has_data("Org/Data", repo_root) is False


def test_raw_dataset_cache_has_data_ignores_any_dotfile(monkeypatch, tmp_path):
    """`.gitignore` and friends are not in the enumerated list, but `datasets` skips every
    dotted name when resolving data files, so none of them can supply rows."""
    repo_root = _dataset_snapshot(
        monkeypatch,
        tmp_path,
        ("README.md", ".gitignore", ".gitattributes", ".hidden/notes.txt"),
    )

    assert cache_inventory._raw_dataset_cache_has_data("Org/Data", repo_root) is False


def test_raw_dataset_cache_has_data_never_walks_outside_the_snapshot(monkeypatch, tmp_path):
    """The prune is a containment test, not a link-type test.

    `Path.is_symlink()` is false for a Windows junction and `Path.is_junction()` does not
    exist before 3.12, which this package still supports, so a link-type test would let
    `os.walk` descend a junction into an arbitrary external tree. A directory that resolves
    outside the snapshot is left unvisited whatever kind of redirect it is."""
    outside = tmp_path / "outside"
    (outside / "deep").mkdir(parents = True)
    (outside / "deep" / "huge.parquet").write_bytes(b"PAR1")
    repo_root = _dataset_snapshot(monkeypatch, tmp_path, ("README.md",))
    # Named like clutter, so it is not payload evidence either, and the tree behind it is never entered,
    # which is what stops a scan from stalling on someone else's disk.
    (repo_root / "snapshots" / "abc" / ".hidden_link").symlink_to(outside, target_is_directory = True)

    assert cache_inventory._raw_dataset_cache_has_data("Org/Data", repo_root) is False


def test_raw_dataset_cache_has_data_does_not_read_an_unreadable_tree_as_empty(
    monkeypatch, tmp_path
):
    """`os.walk` swallows `scandir` errors unless `onerror` is given, so a cache written by
    another user answered "no payload" rather than "could not look", and a dataset that was
    merely uninspectable vanished from On Device."""
    repo_root = _dataset_snapshot(monkeypatch, tmp_path, ("README.md", "data/train.parquet"))
    locked = repo_root / "snapshots" / "abc" / "data"
    os.chmod(locked, 0o000)
    try:
        assert cache_inventory._raw_dataset_cache_has_data("Org/Data", repo_root) is True
    finally:
        os.chmod(locked, 0o755)


def _metadata_only_raw_repo() -> SimpleNamespace:
    return SimpleNamespace(
        repo_id = "Org/Data",
        repo_type = "dataset",
        repo_path = "/cache/hub/datasets--Org--Data",
        size_on_disk = 100,
        revisions = [SimpleNamespace(files = [], commit_hash = "abc")],
    )


def test_dataset_cache_without_data_files_is_partial(monkeypatch):
    """A snapshot holding only the dataset card passes every structural check but
    cannot be loaded, so it must not be offered as usable On Device."""
    monkeypatch.setattr(
        cache_inventory,
        "_collect_hf_cache_scans",
        lambda: ([SimpleNamespace(repos = [_metadata_only_raw_repo()])], {"/cache/hub"}),
    )
    monkeypatch.setattr(
        cache_inventory.hf_cache_scan,
        "is_snapshot_partial",
        lambda *_args, **_kwargs: False,
    )
    monkeypatch.setattr(cache_inventory, "_raw_dataset_cache_has_data", lambda *_args: False)
    monkeypatch.setattr(cache_inventory, "_scan_hub_dataset_cache_dirs", lambda: [])
    monkeypatch.setattr(cache_inventory, "_scan_processed_dataset_caches", lambda: [])
    monkeypatch.setattr(cache_inventory, "_scan_app_processed_dataset_caches", lambda: [])

    rows = cache_inventory._scan_hf_dataset_caches()

    assert len(rows) == 1
    assert rows[0]["partial"] is True


def test_processed_cache_settles_a_partial_raw_row_without_losing_its_path(monkeypatch):
    """The Arrow cache loads on its own, so it clears `partial`. The row must keep the hub
    `cache_path`, which is the only handle `delete_cached_dataset_response` can scope a
    hub-dir purge to."""
    monkeypatch.setattr(
        cache_inventory,
        "_collect_hf_cache_scans",
        lambda: ([SimpleNamespace(repos = [_metadata_only_raw_repo()])], {"/cache/hub"}),
    )
    monkeypatch.setattr(
        cache_inventory.hf_cache_scan,
        "is_snapshot_partial",
        lambda *_args, **_kwargs: False,
    )
    monkeypatch.setattr(cache_inventory, "_raw_dataset_cache_has_data", lambda *_args: False)
    monkeypatch.setattr(cache_inventory, "_scan_hub_dataset_cache_dirs", lambda: [])
    monkeypatch.setattr(
        cache_inventory,
        "_scan_processed_dataset_caches",
        lambda: [
            {
                "repo_id": "org/data",
                "size_bytes": 250,
                "cache_path": "/processed/org___data",
                "processed_cache": True,
                "partial": False,
            }
        ],
    )
    monkeypatch.setattr(cache_inventory, "_scan_app_processed_dataset_caches", lambda: [])

    rows = cache_inventory._scan_hf_dataset_caches()

    assert len(rows) == 1
    assert rows[0]["cache_path"] == "/cache/hub/datasets--Org--Data"
    assert rows[0]["load_cache_path"] == "/processed/org___data"
    assert rows[0]["partial"] is False
    assert rows[0]["partial_transport"] is None


def test_app_processed_cache_never_settles_a_partial_raw_row(monkeypatch):
    """App caches are written per snapshot commit but grouped without one, so a finished
    cache proves nothing about the snapshot the loader will resolve."""
    monkeypatch.setattr(
        cache_inventory,
        "_collect_hf_cache_scans",
        lambda: ([SimpleNamespace(repos = [_metadata_only_raw_repo()])], {"/cache/hub"}),
    )
    monkeypatch.setattr(
        cache_inventory.hf_cache_scan,
        "is_snapshot_partial",
        lambda *_args, **_kwargs: False,
    )
    monkeypatch.setattr(cache_inventory, "_raw_dataset_cache_has_data", lambda *_args: False)
    monkeypatch.setattr(cache_inventory, "_scan_hub_dataset_cache_dirs", lambda: [])
    monkeypatch.setattr(cache_inventory, "_scan_processed_dataset_caches", lambda: [])
    monkeypatch.setattr(
        cache_inventory,
        "_scan_app_processed_dataset_caches",
        lambda: [
            {
                "repo_id": "org/data",
                "size_bytes": 40,
                "cache_path": "/app/entry",
                "processed_cache": True,
                "app_processed_cache": True,
                "app_processed_hub_cache": "/cache/hub",
                "partial": True,
            }
        ],
    )

    rows = cache_inventory._scan_hf_dataset_caches()

    assert len(rows) == 1
    assert rows[0]["partial"] is True


def test_app_processed_cache_without_raw_snapshot_is_partial(monkeypatch):
    monkeypatch.setattr(cache_inventory, "_collect_hf_cache_scans", lambda: ([], set()))
    monkeypatch.setattr(cache_inventory, "_scan_hub_dataset_cache_dirs", lambda: [])
    monkeypatch.setattr(cache_inventory, "_scan_processed_dataset_caches", lambda: [])
    app_row = {
        "repo_id": "Org/Data",
        "size_bytes": 40,
        "cache_path": "/app/entry",
        "processed_cache": True,
        "app_processed_cache": True,
        "app_processed_hub_cache": "/cache/hub",
        "partial": True,
    }
    monkeypatch.setattr(
        cache_inventory,
        "_scan_app_processed_dataset_caches",
        lambda: [app_row],
    )

    assert cache_inventory._scan_hf_dataset_caches() == [app_row]


def test_delete_cached_dataset_scopes_delete_to_selected_root(monkeypatch, tmp_path):
    """A dataset present in the active cache and a previously selected cache is
    deleted only from the selected root, so the other cache's copy survives."""
    calls = []
    target_hub = tmp_path / "active" / "hub"
    other_hub = tmp_path / "previous" / "hub"
    for hub in (target_hub, other_hub):
        (hub / "datasets--Org--Data").mkdir(parents = True)

    class _DeleteStrategy:
        def __init__(self, label: str):
            self.label = label

        def execute(self):
            calls.append(self.label)

    def _cache(label: str, hub):
        return SimpleNamespace(
            cache_dir = label,
            repos = [
                SimpleNamespace(
                    repo_type = "dataset",
                    repo_id = "Org/Data",
                    repo_path = str(hub / "datasets--Org--Data"),
                    revisions = [SimpleNamespace(commit_hash = f"{label}-rev")],
                )
            ],
            delete_revisions = lambda *_revs, _label = label: _DeleteStrategy(_label),
        )

    monkeypatch.setattr(
        cache_inventory,
        "_collect_hf_cache_scans",
        lambda: ([_cache("active", target_hub), _cache("previous", other_hub)], set()),
    )
    monkeypatch.setattr(
        cache_inventory,
        "_delete_processed_dataset_cache",
        lambda _repo_id, **_kwargs: (False, []),
    )
    monkeypatch.setattr(
        cache_inventory.download_manifest,
        "purge_all_state_for_repo",
        lambda *_args, **_kwargs: 0,
    )
    monkeypatch.setattr(
        "utils.hf_cache_settings.get_hf_cache_paths",
        lambda: SimpleNamespace(hub_cache = target_hub),
    )
    monkeypatch.setattr(
        "hub.utils.hf_cache_state.hf_cache_roots",
        lambda: [target_hub, other_hub],
    )

    result = cache_inventory._delete_cached_dataset_blocking("Org/Data")

    assert result == {"status": "deleted", "repo_id": "Org/Data"}
    # Only the selected cache's revision is deleted; the other cache's copy stays.
    assert calls == ["active"]
    assert not (target_hub / "datasets--Org--Data").exists()
    assert (other_hub / "datasets--Org--Data").exists()


def test_delete_processed_only_dataset_accepts_processed_cache_path(monkeypatch, tmp_path):
    """A processed-only dataset row sends its Arrow cache path (<owner>___<repo>
    under HF_DATASETS_CACHE), which is not a Hub datasets-- dir. The delete must
    accept it and run the processed-cache delete instead of raising 400."""
    datasets_root = tmp_path / "datasets"
    processed_dir = datasets_root / "Org___Data"
    processed_dir.mkdir(parents = True)

    # No Hub-cache copy exists; only the processed Arrow cache holds this repo.
    monkeypatch.setattr(cache_inventory, "_collect_hf_cache_scans", lambda: ([], set()))
    monkeypatch.setattr(cache_inventory, "_hf_datasets_cache_roots", lambda: [datasets_root])
    processed_calls: list[str] = []
    monkeypatch.setattr(
        cache_inventory,
        "_delete_processed_dataset_cache",
        lambda repo_id, **_kwargs: (processed_calls.append(repo_id) or True, []),
    )

    result = cache_inventory._delete_cached_dataset_blocking("Org/Data", str(processed_dir))

    assert result == {"status": "deleted", "repo_id": "Org/Data"}
    assert processed_calls == ["Org/Data"]


def test_delete_processed_dataset_scopes_to_selected_root(monkeypatch, tmp_path):
    """A dataset processed under two HF_DATASETS_CACHE roots is deleted only from
    the selected root; the copy under the other cache home survives (real delete,
    not stubbed)."""
    selected_root = tmp_path / "selected" / "datasets"
    other_root = tmp_path / "other" / "datasets"
    for root in (selected_root, other_root):
        (root / "Org___Data").mkdir(parents = True)

    monkeypatch.setattr(cache_inventory, "_collect_hf_cache_scans", lambda: ([], set()))
    monkeypatch.setattr(
        cache_inventory, "_hf_datasets_cache_roots", lambda: [selected_root, other_root]
    )

    result = cache_inventory._delete_cached_dataset_blocking(
        "Org/Data", str(selected_root / "Org___Data")
    )

    assert result == {"status": "deleted", "repo_id": "Org/Data"}
    assert not (selected_root / "Org___Data").exists()
    assert (other_root / "Org___Data").exists()


def _app_cache_entry(monkeypatch, hub_cache: Path, repo_id: str, commit_hash: str):
    repo_root = hub_cache / f"datasets--{repo_id.replace('/', '--')}"
    snapshot = repo_root / "snapshots" / commit_hash
    snapshot.mkdir(parents = True)
    (snapshot / "train.parquet").write_bytes(b"rows")
    roots = getattr(monkeypatch, "_dataset_hub_roots", [])
    roots.append(hub_cache)
    monkeypatch._dataset_hub_roots = roots
    monkeypatch.setattr(hf_cache_state, "hf_cache_roots", lambda **kw: roots)
    entry = dataset_processed_cache.prepare_app_processed_dataset_cache(
        repo_id,
        snapshot,
    )
    dataset_processed_cache.mark_app_processed_dataset_cache_complete(entry)
    return entry


def test_delete_app_processed_cache_isolated_by_hub_root(monkeypatch, tmp_path):
    repo_id = "Org/Data"
    first = _app_cache_entry(
        monkeypatch,
        tmp_path / "first" / "hub",
        repo_id,
        "commit-a",
    )
    second = _app_cache_entry(
        monkeypatch,
        tmp_path / "second" / "hub",
        repo_id,
        "commit-b",
    )
    external = tmp_path / "external"
    external.mkdir()
    (external / "keep.txt").write_text("keep")
    (first.cache_dir / "external").symlink_to(external, target_is_directory = True)

    deleted, failures = cache_inventory._delete_app_processed_dataset_cache(
        repo_id,
        hub_cache = first.hub_cache,
    )

    assert deleted is True
    assert failures == []
    assert not first.path.exists()
    assert second.path.exists()
    assert (external / "keep.txt").read_text() == "keep"


def test_delete_raw_scope_purges_corrupt_app_cache_entry(monkeypatch, tmp_path):
    repo_id = "Org/Data"
    entry = _app_cache_entry(
        monkeypatch,
        tmp_path / "hub",
        repo_id,
        "commit-a",
    )
    (entry.path / "metadata.json").write_text("{")

    assert list(dataset_processed_cache.iter_app_processed_dataset_caches()) == []

    deleted, failures = cache_inventory._delete_app_processed_dataset_cache(
        repo_id,
        hub_cache = entry.hub_cache,
    )

    assert deleted is True
    assert failures == []
    assert not entry.path.exists()


def test_delete_app_only_cache_path_isolated_by_hub_root(monkeypatch, tmp_path):
    repo_id = "Org/Data"
    first = _app_cache_entry(
        monkeypatch,
        tmp_path / "first" / "hub",
        repo_id,
        "commit-a",
    )
    second = _app_cache_entry(
        monkeypatch,
        tmp_path / "second" / "hub",
        repo_id,
        "commit-b",
    )
    monkeypatch.setattr(cache_inventory, "_collect_hf_cache_scans", lambda: ([], set()))
    monkeypatch.setattr(
        cache_inventory,
        "_delete_processed_dataset_cache",
        lambda *_args, **_kwargs: (False, []),
    )

    result = cache_inventory._delete_cached_dataset_blocking(
        repo_id,
        str(first.path),
    )

    assert result == {"status": "deleted", "repo_id": repo_id}
    assert not first.path.exists()
    assert second.path.exists()
    assert (
        first.hub_cache / "datasets--Org--Data" / "snapshots" / "commit-a" / "train.parquet"
    ).exists()


def test_delete_raw_path_removes_only_same_scope_app_cache(monkeypatch, tmp_path):
    repo_id = "Org/Data"
    first = _app_cache_entry(
        monkeypatch,
        tmp_path / "first" / "hub",
        repo_id,
        "commit-a",
    )
    second = _app_cache_entry(
        monkeypatch,
        tmp_path / "second" / "hub",
        repo_id,
        "commit-b",
    )

    class _Strategy:
        def execute(self):
            return None

    scans = []
    for entry in (first, second):
        repo_path = entry.hub_cache / "datasets--Org--Data"
        scans.append(
            SimpleNamespace(
                repos = [
                    SimpleNamespace(
                        repo_type = "dataset",
                        repo_id = repo_id,
                        repo_path = str(repo_path),
                        revisions = [SimpleNamespace(commit_hash = entry.commit_hash)],
                    )
                ],
                delete_revisions = lambda *_args: _Strategy(),
            )
        )
    monkeypatch.setattr(
        cache_inventory,
        "_collect_hf_cache_scans",
        lambda: (scans, set()),
    )
    monkeypatch.setattr(
        cache_inventory,
        "_delete_processed_dataset_cache",
        lambda *_args, **_kwargs: (False, []),
    )
    monkeypatch.setattr(
        cache_inventory,
        "purge_repo_cache_dirs",
        lambda *_args, **_kwargs: False,
    )
    monkeypatch.setattr(
        cache_inventory,
        "purge_partial_repo",
        lambda *_args, **_kwargs: False,
    )
    monkeypatch.setattr(
        cache_inventory.download_manifest,
        "purge_all_state_for_repo",
        lambda *_args, **_kwargs: 0,
    )

    result = cache_inventory._delete_cached_dataset_blocking(
        repo_id,
        str(first.hub_cache / "datasets--Org--Data"),
    )

    assert result == {"status": "deleted", "repo_id": repo_id}
    assert not first.path.exists()
    assert second.path.exists()


def test_app_cache_symlinked_root_is_not_scanned_or_deleted(monkeypatch, tmp_path):
    external = tmp_path / "external"
    external.mkdir()
    (external / "keep.txt").write_text("keep")
    root = tmp_path / "app-processed"
    root.symlink_to(external, target_is_directory = True)

    assert list(dataset_processed_cache.iter_app_processed_dataset_caches()) == []
    assert cache_inventory._delete_app_processed_dataset_cache("Org/Data") == (False, [])
    assert (external / "keep.txt").read_text() == "keep"


def test_app_cache_symlinked_entry_is_not_scanned_or_deleted(monkeypatch, tmp_path):
    entry = _app_cache_entry(
        monkeypatch,
        tmp_path / "hub",
        "Org/Data",
        "commit-a",
    )
    external = tmp_path / "external"
    external.mkdir()
    (external / "keep.txt").write_text("keep")
    import shutil

    shutil.rmtree(entry.path)
    entry.path.symlink_to(external, target_is_directory = True)

    assert list(dataset_processed_cache.iter_app_processed_dataset_caches()) == []
    deleted, failures = cache_inventory._delete_app_processed_dataset_cache("Org/Data")
    assert deleted is False
    assert failures
    assert (external / "keep.txt").read_text() == "keep"


def test_delete_cached_dataset_purges_blob_only_repo_dir(monkeypatch):
    """A blob-only ``datasets--owner--repo`` dir (no usable snapshot/refs) is
    fully removable: purge_partial_repo alone clears only ``.incomplete`` files
    and would leave the complete blobs and the row."""
    purged_dirs: list[str] = []

    monkeypatch.setattr(
        cache_inventory,
        "_collect_hf_cache_scans",
        lambda: ([], set()),
    )
    monkeypatch.setattr(
        cache_inventory,
        "_delete_processed_dataset_cache",
        lambda _repo_id, **_kwargs: (False, []),
    )
    monkeypatch.setattr(
        cache_inventory,
        "purge_repo_cache_dirs",
        lambda _repo_type, repo_id, **_kwargs: purged_dirs.append(repo_id) or True,
    )
    monkeypatch.setattr(
        cache_inventory,
        "purge_partial_repo",
        lambda *_args, **_kwargs: False,
    )
    monkeypatch.setattr(
        cache_inventory.download_manifest,
        "purge_all_state_for_repo",
        lambda *_args, **_kwargs: 0,
    )

    result = cache_inventory._delete_cached_dataset_blocking("Org/Data")

    assert result == {"status": "deleted", "repo_id": "Org/Data"}
    assert purged_dirs == ["Org/Data"]


def test_delete_cached_dataset_absent_everywhere_raises_404(monkeypatch):
    monkeypatch.setattr(
        cache_inventory,
        "_collect_hf_cache_scans",
        lambda: ([], set()),
    )
    monkeypatch.setattr(
        cache_inventory,
        "_delete_processed_dataset_cache",
        lambda _repo_id, **_kwargs: (False, []),
    )
    monkeypatch.setattr(
        cache_inventory,
        "purge_repo_cache_dirs",
        lambda *_args, **_kwargs: False,
    )
    monkeypatch.setattr(
        cache_inventory,
        "purge_partial_repo",
        lambda *_args, **_kwargs: False,
    )
    monkeypatch.setattr(
        cache_inventory.download_manifest,
        "purge_all_state_for_repo",
        lambda *_args, **_kwargs: 0,
    )

    with pytest.raises(HTTPException) as exc_info:
        cache_inventory._delete_cached_dataset_blocking("Org/Missing")

    assert exc_info.value.status_code == 404


def test_check_format_rejects_invalid_path_as_400():
    with pytest.raises(HTTPException) as exc_info:
        formatting.check_format_response(CheckFormatRequest(dataset_name = "../../etc/passwd"))

    assert exc_info.value.status_code == 400


def test_legacy_tier1_fallback_preserves_single_source_file_order():
    files = ["README.md", "alpaca_data_cleaned.json"]

    assert (
        formatting._select_tier1_repo_file(
            files,
            subset = None,
            train_split = "train",
        )
        is None
    )
    assert (
        formatting._select_tier1_repo_file(
            files,
            subset = None,
            train_split = "train",
            allow_unlabeled_fallback = True,
        )
        == "alpaca_data_cleaned.json"
    )


def test_legacy_tier1_fallback_does_not_guess_between_multiple_files():
    files = ["part-a.json", "part-b.json"]

    assert (
        formatting._select_tier1_repo_file(
            files,
            subset = None,
            train_split = "train",
            allow_unlabeled_fallback = True,
        )
        is None
    )


def test_legacy_tier1_fallback_does_not_use_a_different_split():
    assert (
        formatting._select_tier1_repo_file(
            ["validation.json"],
            subset = None,
            train_split = "train",
            allow_unlabeled_fallback = True,
        )
        is None
    )


@pytest.mark.parametrize(
    "spelling",
    [
        pytest.param("{path}", id = "absolute"),
        pytest.param("  {path}  ", id = "surrounding-whitespace"),
        pytest.param(
            "~/.unsloth/studio/assets/datasets/uploads/deleted.jsonl",
            id = "tilde",
        ),
    ],
)
def test_check_format_missing_local_path_never_reaches_hub(monkeypatch, tmp_path, spelling):
    studio_home = tmp_path / ".unsloth" / "studio"
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(studio_home))
    missing = studio_home / "assets" / "datasets" / "uploads" / "deleted.jsonl"
    dataset_name = spelling.format(path = missing)
    attempts = []

    class _NoHubApi:
        def __init__(self, *args, **kwargs):
            attempts.append("HfApi")
            raise AssertionError("a missing local path must not reach Hugging Face")

    def _no_load_dataset(*args, **kwargs):
        attempts.append("load_dataset")
        raise AssertionError("a missing local path must not reach datasets.load_dataset")

    monkeypatch.setattr("huggingface_hub.HfApi", _NoHubApi)
    monkeypatch.setattr("datasets.load_dataset", _no_load_dataset)

    with pytest.raises(HTTPException) as exc_info:
        formatting.check_format_response(CheckFormatRequest(dataset_name = dataset_name))

    assert exc_info.value.status_code == 404
    assert exc_info.value.detail == formatting._MISSING_DATASET_DETAIL
    assert attempts == []


def test_check_format_returns_stable_code_for_missing_selected_cache(monkeypatch, tmp_path):
    monkeypatch.setattr(
        formatting,
        "resolve_dataset_path",
        lambda _dataset_name: tmp_path / "missing",
    )
    monkeypatch.setattr(
        formatting,
        "_load_any_cached_hf_preview_slice",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setitem(
        sys.modules,
        "datasets",
        SimpleNamespace(Dataset = object, load_dataset = object),
    )

    with pytest.raises(HTTPException) as exc_info:
        formatting.check_format_response(
            CheckFormatRequest(
                dataset_name = "Org/Data",
                prefer_local_cache = True,
            )
        )

    assert exc_info.value.status_code == 404
    assert exc_info.value.detail == {
        "code": "dataset_local_cache_miss",
        "message": "Dataset is not available in the local cache.",
    }


def test_cached_preview_does_not_fallback_from_selected_path(monkeypatch):
    monkeypatch.setattr(
        formatting,
        "_shared_dataset_snapshot_from_cache_path",
        lambda _local_path, _repo_id: None,
    )
    monkeypatch.setattr(
        formatting,
        "_shared_latest_cached_dataset_snapshot",
        lambda *_args, **_kwargs: pytest.fail("selected cache path must remain strict"),
    )

    assert formatting._latest_cached_dataset_snapshot("Org/Data", "/cache/selected") is None


def test_processed_preview_loads_selected_cache_path(monkeypatch):
    calls = []

    class _PreviewDataset:
        def __len__(self):
            return 3

        def select(self, indices):
            assert list(indices) == [0, 1]
            return [{"text": "selected"}]

    def load_cached(repo_id, local_path, **kwargs):
        calls.append((repo_id, local_path, kwargs))
        return _PreviewDataset()

    monkeypatch.setattr(formatting, "_shared_load_cached_hf_dataset", load_cached)
    monkeypatch.setattr(
        formatting,
        "_shared_latest_cached_dataset_path",
        lambda *_args, **_kwargs: pytest.fail("selected cache path must remain strict"),
    )
    request = CheckFormatRequest(
        dataset_name = "Org/Data",
        subset = "english",
        train_split = "validation",
        prefer_local_cache = True,
        local_path = "/cache/selected",
    )

    preview, total_rows = formatting._load_processed_hf_preview_slice(
        request,
        2,
        "hf_test",
    )

    assert preview == [{"text": "selected"}]
    assert total_rows == 3
    assert calls == [
        (
            "Org/Data",
            "/cache/selected",
            {"subset": "english", "split": "validation", "token": "hf_test"},
        )
    ]


def test_dataset_download_status_preserves_idle_shape():
    status = downloads._dataset_status("Org/Data")

    assert status.state == "idle"
    assert status.error is None


def test_dataset_download_registry_key_is_case_insensitive():
    registry = download_registry.DownloadRegistry()

    claimed, state = registry.claim(
        "Org/Data",
        download_registry.TRANSPORT_HTTP,
        repo_type = "dataset",
        repo_id = "Org/Data",
    )
    duplicate_claimed, duplicate_state = registry.claim(
        "org/data",
        download_registry.TRANSPORT_HTTP,
        repo_type = "dataset",
        repo_id = "org/data",
    )

    assert claimed is True
    assert state == "running"
    assert duplicate_claimed is False
    assert duplicate_state == "running"
    assert registry.active_jobs("ORG/DATA") == {"org/data": "running"}


def test_dataset_idle_status_uses_cancel_marker_after_restart(monkeypatch, tmp_path):
    monkeypatch.setattr(state_dir, "cache_root", lambda: tmp_path)
    monkeypatch.setattr(downloads, "_registry", download_registry.DownloadRegistry())
    assert download_manifest.write_cancel_marker("dataset", "Owner/Data", None, "http")

    status = asyncio.run(downloads.get_dataset_download_status_response("owner/data"))

    assert status.state == "cancelled"
    assert status.error is None


def test_dataset_claim_register_cancel_uses_registry_marker_owner(monkeypatch):
    killed = []

    class _Registry:
        def claim(self, *_args, **_kwargs):
            return True, "running"

        def current_generation(self, _key):
            return 1

        def register_process(self, _key, _proc):
            return False

        def persist_cancel_for_key(self, *_args, **_kwargs):
            raise AssertionError("register_process owns pending-cancel markers")

        def get_job(self, _key):
            return SimpleNamespace(state = "cancelled", error = None)

    monkeypatch.setattr(downloads, "_registry", _Registry())
    monkeypatch.setattr(
        downloads,
        "resolve_cached_repo_id_case",
        lambda repo_id, **_kwargs: repo_id,
    )
    monkeypatch.setattr(
        downloads.download_registry,
        "download_transport_unavailable_reason",
        lambda _transport: None,
    )
    monkeypatch.setattr(
        downloads.download_lifecycle,
        "spawn_worker",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(
        downloads.download_lifecycle,
        "kill_and_reap_process",
        lambda proc, **_kwargs: killed.append(proc),
    )

    result = asyncio.run(
        downloads.download_dataset_response(SimpleNamespace(repo_id = "Org/Data", use_xet = False))
    )

    assert result["state"] == "cancelled"
    assert killed


def test_dataset_cancel_pending_spawn_arms_pending_cancel(monkeypatch):
    events = []

    class _Registry:
        def get_process(self, _key):
            return None

        def mark_pending_cancel(self, key, generation):
            events.append(("pending", key, generation))
            return True

        def get_job(self, _key):
            return SimpleNamespace(state = "running")

    monkeypatch.setattr(downloads, "_registry", _Registry())
    monkeypatch.setattr(
        downloads,
        "resolve_cached_repo_id_case",
        lambda repo_id, **_kwargs: repo_id,
    )

    result = asyncio.run(
        downloads.cancel_dataset_download_response(
            SimpleNamespace(repo_id = "Org/Data", generation = 5)
        )
    )

    assert result == {"repo_id": "Org/Data", "state": "cancelling"}
    assert events == [("pending", "org/data", 5)]


def test_upload_dataset_response_writes_non_empty_file(monkeypatch, tmp_path):
    payload = b'{"text":"hello"}\n'
    offloaded = []

    async def run_offloaded(function, *args):
        offloaded.append(function)
        return function(*args)

    monkeypatch.setattr(local, "DATASET_UPLOAD_DIR", tmp_path)
    monkeypatch.setattr(local.asyncio, "to_thread", run_offloaded)

    response = asyncio.run(local.upload_dataset_response(_Upload("../train.jsonl", payload)))

    stored_path = Path(response.stored_path)
    assert response.filename == "train.jsonl"
    assert stored_path.parent == tmp_path
    assert stored_path.name.endswith("_train.jsonl")
    assert stored_path.read_bytes() == payload
    assert any(getattr(function, "__name__", "") == "write" for function in offloaded)


def test_local_dataset_items_expose_recipe_and_upload_source(monkeypatch, tmp_path):
    recipe_root = tmp_path / "recipes"
    upload_root = tmp_path / "uploads"
    parquet_dir = recipe_root / "recipe_alpha" / "parquet-files"
    parquet_dir.mkdir(parents = True)
    (parquet_dir / "part.parquet").write_bytes(b"parquet")
    upload_root.mkdir()
    (upload_root / "manual.jsonl").write_text('{"text":"hello"}\n', encoding = "utf-8")
    monkeypatch.setattr(local, "LOCAL_DATASETS_ROOT", recipe_root)
    monkeypatch.setattr(local, "DATASET_UPLOAD_DIR", upload_root)

    response = local.list_local_datasets_response()

    assert "source" in LocalDatasetItem.__annotations__
    by_id = {item.id: item for item in response.datasets}
    assert by_id["recipe_alpha"].source == "recipe"
    assert by_id["manual.jsonl"].source == "upload"


def test_raw_dataset_cache_has_data_ignores_every_datasets_metadata_name(monkeypatch, tmp_path):
    """The non-payload set reuses `local_options._IGNORED_DATA_FILENAMES` rather than copying
    it, so the check and the resolver that decides whether a cache yields rows cannot drift.
    A snapshot holding only these has no rows to load however complete it looks."""
    repo_root = _dataset_snapshot(
        monkeypatch,
        tmp_path,
        tuple(sorted(local_options._IGNORED_DATA_FILENAMES)),
    )

    assert cache_inventory._raw_dataset_cache_has_data("Org/Data", repo_root) is False
