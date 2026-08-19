# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""macOS Finder metadata companions, against the predicates and the shared walks.

A ``._<name>`` companion carries the described file's extension, so it answers every name-shaped
question the way the real file does, and ``Path.glob`` matches it. Consumers are covered through
what they share -- the predicates, the cache walk, the dataset walk -- rather than one case each.
Every fixture also holds a real file a user named ``._something``, which must survive: nothing
may be refused for its name alone.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from utils.paths.path_utils import (  # noqa: E402
    drop_appledouble_metadata,
    drop_shadowed_appledouble_names,
    is_appledouble_metadata,
)

_AD = b"\x00\x05\x16\x07\x00\x02\x00\x00"


def test_only_the_magic_bytes_settle_it(tmp_path):
    (tmp_path / "model.safetensors").write_bytes(b"real")
    (tmp_path / "._model.safetensors").write_bytes(_AD)
    (tmp_path / "._mine.safetensors").write_bytes(b"real")

    assert is_appledouble_metadata(tmp_path / "._model.safetensors") is True
    assert is_appledouble_metadata(tmp_path / "._mine.safetensors") is False
    assert [p.name for p in drop_appledouble_metadata(sorted(tmp_path.iterdir()))] == [
        "._mine.safetensors",
        "model.safetensors",
    ]


def test_a_read_that_fails_settles_nothing(tmp_path):
    """An unreadable ``._`` file is not proven to be metadata, so it is kept."""
    locked = tmp_path / "._locked.safetensors"
    locked.write_bytes(_AD)
    locked.chmod(0o000)
    if os.access(locked, os.R_OK):
        locked.chmod(0o600)
        pytest.skip("this user can read a mode-000 file, so the failed read cannot be staged")
    try:
        metadata = tmp_path / "._readable.safetensors"
        metadata.write_bytes(_AD)
        assert is_appledouble_metadata(locked) is False
        assert drop_appledouble_metadata([locked, metadata]) == [locked]
    finally:
        locked.chmod(0o600)


def test_a_listing_with_no_bytes_pairs_by_subject():
    """Remote listings carry no bytes, so a ``._x`` goes only when its subject is listed beside
    it, in the same directory -- a sole candidate survives whatever it is called."""
    assert drop_shadowed_appledouble_names(["._solo.py"]) == ["._solo.py"]
    assert drop_shadowed_appledouble_names(["._a.py", "a.py"]) == ["a.py"]
    assert drop_shadowed_appledouble_names(["sub/._a.py", "a.py"]) == ["sub/._a.py", "a.py"]
    assert drop_shadowed_appledouble_names(["sub/._a.py", "sub/a.py"]) == ["sub/a.py"]


def test_a_split_quant_pairs_by_shard_family():
    """A split quant's shards need not all be shadowed, so the subject is the family."""
    from hub.utils.gguf import drop_shadowed_appledouble_names as by_family

    assert by_family(["._m-00001-of-00002.gguf", "m-00002-of-00002.gguf"]) == [
        "m-00002-of-00002.gguf"
    ]
    assert by_family(["._mine.gguf"]) == ["._mine.gguf"]


@dataclass(frozen = True)
class _CachedFile:
    file_name: str
    file_path: Path
    size_on_disk: int = 0


def _cache_repo(tmp_path, entries):
    """A repo scan shaped like a real cache: the snapshot entry is a symlink, and the bytes it
    names live in a content-addressed blob whose own name carries no prefix."""
    blobs = tmp_path / "blobs"
    snapshot = tmp_path / "snapshots" / "rev0"
    blobs.mkdir()
    snapshot.mkdir(parents = True)
    files = []
    for i, (name, payload, size) in enumerate(entries):
        blob = blobs / f"blob{i}"
        blob.write_bytes(payload)
        entry = snapshot / name
        entry.parent.mkdir(parents = True, exist_ok = True)
        entry.symlink_to(blob)
        files.append(_CachedFile(Path(name).name, entry, size))
    revision = SimpleNamespace(files = frozenset(files), snapshot_path = snapshot, refs = {"main"})
    return SimpleNamespace(revisions = [revision])


def test_cached_repo_files_follows_the_snapshot_entry_to_its_blob(tmp_path):
    """The ``._`` name is on the snapshot entry and the bytes are in the blob, but the entry
    links to it, so one open observes both. Every cache and model-route scan reads this list."""
    from hub.services.models.cache_inventory import cached_repo_files

    repo = _cache_repo(
        tmp_path,
        (
            ("transformer/model.safetensors", b"real", 0),
            ("transformer/._model.safetensors", _AD, 0),
            ("._mine.safetensors", b"real", 0),
            ("model-Q4_K_M.gguf", b"GGUF", 0),
            ("._model-Q4_K_M.gguf", _AD, 0),
            ("._mine.gguf", b"GGUF", 0),
        ),
    )
    assert sorted(f.file_name for f in cached_repo_files(repo.revisions[0])) == [
        "._mine.gguf",
        "._mine.safetensors",
        "model-Q4_K_M.gguf",
        "model.safetensors",
    ]


def test_a_drafter_budget_prices_the_largest_file_of_a_shared_basename(tmp_path, monkeypatch):
    """Quant subdirectories share a basename, so the budget maxes over every cached file that
    carries it: keeping one entry per name would charge whichever quant a frozenset yielded."""
    import huggingface_hub

    from routes.inference import _cached_repo_gguf_bytes
    from utils.models import drafters

    # Largest in the middle, so keeping the first or the last entry for a name both miss it.
    repo = _cache_repo(
        tmp_path,
        (
            ("Q4_K_M/model.gguf", b"GGUF", 4),
            ("F16/model.gguf", b"GGUF", 9),
            ("Q8_0/model.gguf", b"GGUF", 1),
            ("._model.gguf", _AD, 2),
        ),
    )
    repo.repo_id = "org/d"
    monkeypatch.setattr(
        huggingface_hub, "scan_cache_dir", lambda **kw: SimpleNamespace(repos = [repo])
    )
    seen: dict = {}
    monkeypatch.setattr(
        drafters, "dflash_budget_bytes", lambda sizes, *a, **k: seen.update(sizes) or 0
    )

    _cached_repo_gguf_bytes("org/d")
    assert seen == {"model.gguf": 9}


def test_a_dataset_directory_prefers_its_parquet_export(tmp_path):
    """The GPU, MLX and embedding trainers share this walk: the export outranks the directory's
    own files, a genuine prefixed file inside it survives, and holding neither is an error."""
    from utils.paths import dataset_files_in_dir

    (tmp_path / "train.jsonl").write_bytes(b'{"text": "hi"}\n')
    export = tmp_path / "parquet-files"
    export.mkdir()
    (export / "part.parquet").write_bytes(b"PAR1")
    (export / "._part.parquet").write_bytes(_AD)
    (export / "._mine.parquet").write_bytes(b"PAR1")
    assert [p.name for p in dataset_files_in_dir(tmp_path)] == ["._mine.parquet", "part.parquet"]

    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError):
        dataset_files_in_dir(empty)


def test_a_folder_holding_only_metadata_is_not_a_model(tmp_path):
    """The browse and downloaded-model probes walk the filesystem themselves. A folder whose
    only weight is the user's own prefixed file is a model; one holding only metadata is not."""
    from routes.models import _dir_has_downloaded_model, _has_direct_model_signal

    mine = tmp_path / "mine"
    mine.mkdir()
    (mine / "._mine.safetensors").write_bytes(b"real")
    assert _has_direct_model_signal(mine) is True
    assert _dir_has_downloaded_model(mine) is True

    metadata = tmp_path / "metadata"
    metadata.mkdir()
    (metadata / "._model.safetensors").write_bytes(_AD)
    assert _has_direct_model_signal(metadata) is False
    assert _dir_has_downloaded_model(metadata) is False


def test_consolidated_weights_are_not_hidden_by_their_companion():
    """The companion does not start with "consolidated", so it answered the weight test and
    every consolidated* file was then stripped from the download."""
    from hub.utils.snapshot_filters import repo_ships_transformers_weights

    assert (
        repo_ships_transformers_weights(["._consolidated.safetensors", "consolidated.safetensors"])
        is False
    )
    assert repo_ships_transformers_weights(["model.safetensors"]) is True
