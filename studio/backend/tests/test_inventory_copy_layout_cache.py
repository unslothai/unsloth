# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Downloaded models in a Hugging Face cache written without symlinks must stay listed.

Where symlinks are unavailable (Windows without Developer Mode, some network shares and
container mounts), huggingface_hub moves each finished blob into ``snapshots/<rev>/`` instead of
linking it, so ``blobs/`` ends up empty or absent. The Hub page inventory used to require a file
under ``blobs/`` and dropped these repos, while the older ``/api/models/local`` scan listed them.

No GPU/network: the cache layouts are built by hand, matching what huggingface_hub leaves.
"""

from __future__ import annotations

import os
import sys
import types
from pathlib import Path

import pytest

if "structlog" not in sys.modules:

    class _DummyLogger:
        def __getattr__(self, _name):
            return lambda *args, **kwargs: None

    sys.modules["structlog"] = types.SimpleNamespace(
        BoundLogger = _DummyLogger,
        get_logger = lambda *args, **kwargs: _DummyLogger(),
    )

import routes.models as models_route
from hub.services.models import local_inventory

REPO = "unsloth/Tiny-GGUF"
REV = "0123456789abcdef0123456789abcdef01234567"
GGUF = "Tiny-UD-Q4_K_XL.gguf"


def _repo_dir(cache: Path, repo_id: str = REPO) -> Path:
    return cache / ("models--" + repo_id.replace("/", "--"))


def _write(path: Path, data: bytes = b"GGUF\0\0\0\0") -> Path:
    path.parent.mkdir(parents = True, exist_ok = True)
    path.write_bytes(data)
    return path


def _copy_layout(
    cache: Path,
    *,
    keep_blobs_dir: bool = True,
    files = (GGUF,),
) -> Path:
    """What huggingface_hub leaves when it cannot symlink: real files under snapshots/."""
    repo = _repo_dir(cache)
    _write(repo / "refs" / "main", REV.encode())
    for name in files:
        _write(repo / "snapshots" / REV / name)
    if keep_blobs_dir:
        (repo / "blobs").mkdir(parents = True, exist_ok = True)
    return repo


def _symlinks_available(tmp_path: Path) -> bool:
    try:
        (tmp_path / "probe_link").symlink_to(tmp_path)
        return True
    except OSError:
        return False


def _discovered_ids(cache: Path) -> list[str]:
    return [model_id for _repo, model_id, _updated in local_inventory._discover_hf_cache(cache)]


@pytest.mark.parametrize("keep_blobs_dir", [True, False], ids = ["empty_blobs", "no_blobs"])
def test_copy_layout_repo_is_discovered(tmp_path, keep_blobs_dir):
    _copy_layout(tmp_path, keep_blobs_dir = keep_blobs_dir)
    assert _discovered_ids(tmp_path) == [REPO]


def test_copy_layout_gguf_is_listed_as_a_loadable_gguf_row(tmp_path):
    _copy_layout(tmp_path)
    rows = local_inventory._scan_hf_cache(tmp_path)
    assert [row.id for row in rows] == [REPO]
    assert rows[0].model_format == "gguf"
    assert rows[0].source == "hf_cache"
    assert rows[0].capabilities.can_chat
    assert not rows[0].partial
    assert rows[0].size_bytes > 0


def test_copy_layout_subdirectory_weights_are_discovered(tmp_path):
    # Split quants and diffusers components live in per-quant / per-component subdirectories.
    _copy_layout(tmp_path, files = ("UD-Q4_K_XL/Tiny-UD-Q4_K_XL-00001-of-00002.gguf",))
    assert _discovered_ids(tmp_path) == [REPO]


def test_both_inventories_agree_on_a_copy_layout_repo(tmp_path):
    _copy_layout(tmp_path)
    hub_page = {row.id for row in local_inventory._scan_hf_cache(tmp_path)}
    models_page = {row.id for row in models_route._scan_hf_cache(tmp_path)}
    assert hub_page == models_page == {REPO}


def test_symlink_layout_is_still_discovered(tmp_path):
    if not _symlinks_available(tmp_path):
        pytest.skip(reason = "no symlink support here (Windows without Developer Mode)")
    repo = _repo_dir(tmp_path)
    blob = _write(repo / "blobs" / "a")
    link = repo / "snapshots" / REV / GGUF
    link.parent.mkdir(parents = True)
    link.symlink_to(os.path.relpath(blob, link.parent))
    assert _discovered_ids(tmp_path) == [REPO]


def test_repo_with_an_in_flight_blob_is_still_discovered(tmp_path):
    # A copy-layout download writes the blob under blobs/ first and moves it when it finishes.
    repo = _repo_dir(tmp_path)
    _write(repo / "blobs" / "deadbeef.incomplete")
    assert _discovered_ids(tmp_path) == [REPO]


@pytest.mark.parametrize(
    "build",
    [
        pytest.param(lambda repo: (repo / "blobs").mkdir(parents = True), id = "empty_blobs_only"),
        pytest.param(
            lambda repo: (repo / "snapshots" / REV).mkdir(parents = True), id = "empty_snapshot"
        ),
        pytest.param(lambda repo: _write(repo / "refs" / "main", REV.encode()), id = "refs_only"),
    ],
)
def test_repo_without_any_downloaded_file_stays_hidden(tmp_path, build):
    build(_repo_dir(tmp_path))
    assert _discovered_ids(tmp_path) == []


def test_dangling_snapshot_link_does_not_count_as_content(tmp_path):
    if not _symlinks_available(tmp_path):
        pytest.skip(reason = "no symlink support here (Windows without Developer Mode)")
    repo = _repo_dir(tmp_path)
    (repo / "blobs").mkdir(parents = True)
    link = repo / "snapshots" / REV / GGUF
    link.parent.mkdir(parents = True)
    link.symlink_to("../../blobs/missing")
    assert _discovered_ids(tmp_path) == []


def test_unreadable_snapshot_tree_does_not_break_the_scan(tmp_path, monkeypatch):
    _copy_layout(tmp_path)
    other = _repo_dir(tmp_path, "unsloth/Other-GGUF")
    _write(other / "blobs" / "b")
    real_iterdir = Path.iterdir

    def iterdir(self):
        if self.name == "snapshots" and self.parent == _repo_dir(tmp_path):
            raise PermissionError("denied")
        return real_iterdir(self)

    monkeypatch.setattr(Path, "iterdir", iterdir)
    assert _discovered_ids(tmp_path) == ["unsloth/Other-GGUF"]


def test_finder_metadata_alone_does_not_count_as_content(tmp_path):
    # A failed copy to a macOS share can leave only the AppleDouble companion behind.
    repo = _repo_dir(tmp_path)
    (repo / "blobs").mkdir(parents = True)
    _write(
        repo / "snapshots" / REV / ("._" + GGUF), b"\x00\x05\x16\x07\x00\x02\x00\x00" + b"\0" * 24
    )
    assert _discovered_ids(tmp_path) == []


def test_the_snapshot_probe_is_bounded(tmp_path, monkeypatch):
    repo = _repo_dir(tmp_path)
    for index in range(5):
        (repo / "snapshots" / REV / f"empty{index}").mkdir(parents = True)
    _write(repo / "snapshots" / REV / "zz" / GGUF)
    monkeypatch.setattr(local_inventory.model_common, "_HF_CACHE_MODEL_FILE_PROBE_LIMIT", 3)
    assert _discovered_ids(tmp_path) == []
    monkeypatch.setattr(local_inventory.model_common, "_HF_CACHE_MODEL_FILE_PROBE_LIMIT", 2000)
    assert _discovered_ids(tmp_path) == [REPO]


def test_only_the_snapshot_that_is_classified_is_probed(tmp_path):
    # An older populated revision beside a newer empty one: the row would be classified from
    # the newer one and show as an unknown, unloadable model, so the repo stays hidden.
    repo = _copy_layout(tmp_path)
    newer = repo / "snapshots" / ("f" * 40)
    newer.mkdir()
    os.utime(repo / "snapshots" / REV, (1, 1))
    assert _discovered_ids(tmp_path) == []


def test_a_huge_snapshot_directory_is_read_only_up_to_the_limit(tmp_path, monkeypatch):
    # rglob lists a whole directory before yielding its first entry; on a share holding a
    # million entries that stalls the inventory, so the probe must stop reading at the limit.
    repo = _repo_dir(tmp_path)
    snapshot = repo / "snapshots" / REV
    snapshot.mkdir(parents = True)
    (repo / "refs").mkdir()
    (repo / "refs" / "main").write_text(REV)
    read = {"entries": 0}
    real_scandir = os.scandir

    class _Empty:
        def __init__(self, index):
            self.name = f"d{index}"
            self.path = str(snapshot / self.name)

        def is_dir(self, follow_symlinks = True):
            return False

        def is_file(self, follow_symlinks = True):
            return False

        def is_symlink(self):
            return False

    class _Huge:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def __iter__(self):
            for index in range(1_000_000):
                read["entries"] += 1
                yield _Empty(index)

        def close(self):
            pass

    def scandir(path = "."):
        return _Huge() if Path(path) == snapshot else real_scandir(path)

    monkeypatch.setattr(os, "scandir", scandir)
    # pathlib's glob binds scandir at import on 3.13+, so the negative control reaches it too.
    import glob

    if hasattr(glob, "_StringGlobber"):
        monkeypatch.setattr(glob._StringGlobber, "scandir", staticmethod(scandir))
    monkeypatch.setattr(local_inventory.model_common, "_HF_CACHE_MODEL_FILE_PROBE_LIMIT", 50)
    assert local_inventory._hf_snapshots_hold_files(repo) is False
    assert read["entries"] <= 51


@pytest.mark.parametrize("name", [".DS_Store", "Thumbs.db"])
def test_os_metadata_alone_does_not_count_as_content(tmp_path, name):
    # Explorer and Finder drop these into folders on a share; they are not a download.
    if name not in local_inventory.hf_cache_scan._CACHE_ENTRIES_TO_IGNORE:
        pytest.skip(reason = f"this huggingface_hub does not ignore {name}")
    repo = _repo_dir(tmp_path)
    (repo / "blobs").mkdir(parents = True)
    _write(repo / "snapshots" / REV / name, b"\0" * 16)
    assert _discovered_ids(tmp_path) == []


class _SortedListing:
    """``os.scandir`` in name order, so a test decides which entry the probe meets first."""

    def __init__(
        self,
        path,
        real_scandir,
        wrap = lambda entry: entry,
    ):
        self._inner = real_scandir(path)
        self._entries = sorted((wrap(e) for e in self._inner), key = lambda e: e.name)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self._inner.close()
        return False

    def __iter__(self):
        return iter(self._entries)


def test_an_unreadable_directory_does_not_hide_a_readable_sibling(tmp_path, monkeypatch):
    # A share that refuses one folder: the model in another is still found. The locked folder
    # is listed last, so a depth-first walk meets it first.
    repo = _repo_dir(tmp_path)
    (repo / "blobs").mkdir(parents = True)
    locked = repo / "snapshots" / REV / "z-locked"
    locked.mkdir(parents = True)
    _write(repo / "snapshots" / REV / "a-weights" / GGUF)
    real_scandir = os.scandir

    def scandir(path = "."):
        if Path(path) == locked:
            raise PermissionError("denied")
        return _SortedListing(path, real_scandir)

    monkeypatch.setattr(os, "scandir", scandir)
    assert _discovered_ids(tmp_path) == [REPO]


def test_an_entry_that_cannot_be_statted_is_skipped(tmp_path, monkeypatch):
    repo = _repo_dir(tmp_path)
    (repo / "blobs").mkdir(parents = True)
    snapshot = repo / "snapshots" / REV
    _write(snapshot / "0-broken.bin")
    _write(snapshot / GGUF)
    real_scandir = os.scandir

    class _Entry:
        def __init__(self, entry):
            self._entry = entry
            self.name = entry.name
            self.path = entry.path

        def is_dir(self, follow_symlinks = True):
            if self.name == "0-broken.bin":
                raise OSError("stale handle")
            return self._entry.is_dir(follow_symlinks = follow_symlinks)

        def is_file(self, follow_symlinks = True):
            return self._entry.is_file(follow_symlinks = follow_symlinks)

    # The broken entry sorts first, so skipping it is what finds the model.
    monkeypatch.setattr(os, "scandir", lambda path = ".": _SortedListing(path, real_scandir, _Entry))
    assert _discovered_ids(tmp_path) == [REPO]
