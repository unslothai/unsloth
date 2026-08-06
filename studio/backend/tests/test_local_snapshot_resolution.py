# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Local-only snapshot resolution for background auto-loads.

A cache populated outside Studio (no download manifest) passes the partial
check while missing shard files, and ``from_pretrained`` on a repo id would
download the gaps. Background loads therefore rewrite the load path to the
LOCALLY resolved snapshot: resolution never touches the network, an uncached
repo resolves to None (409 upstream), and an incomplete snapshot still
resolves so the weight load fails on the missing files instead of fetching
them. No GPU or network required.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

pytest.importorskip("huggingface_hub")

from hub.utils.local_snapshot import resolve_local_snapshot_path


_REV = "0123456789abcdef0123456789abcdef01234567"


def _build_cached_repo(
    cache_dir: Path,
    repo_id: str,
    files: dict[str, str],
    with_refs: bool = True,
    rev: str = _REV,
) -> Path:
    """Lay out a minimal HF hub cache entry the way huggingface_hub expects:
    ``models--org--name/refs/main`` pointing at a snapshot directory.
    ``with_refs = False`` builds the revision-only layout (pruned or foreign
    caches) the inventory scanner accepts."""
    repo_dir = cache_dir / f"models--{repo_id.replace('/', '--')}"
    snapshot = repo_dir / "snapshots" / rev
    snapshot.mkdir(parents = True)
    if with_refs:
        (repo_dir / "refs").mkdir(exist_ok = True)
        (repo_dir / "refs" / "main").write_text(rev)
    for name, content in files.items():
        (snapshot / name).write_text(content)
    return snapshot


def test_cached_repo_resolves_to_its_snapshot_dir(tmp_path):
    snapshot = _build_cached_repo(
        tmp_path,
        "org/tiny-model",
        {"config.json": "{}", "model.safetensors": "weights"},
    )
    resolved = resolve_local_snapshot_path("org/tiny-model", cache_dir = str(tmp_path))
    assert resolved is not None
    assert Path(resolved).resolve() == snapshot.resolve()


def test_incomplete_snapshot_still_resolves_locally(tmp_path):
    """Missing shards must not block resolution: the local path is what makes
    the subsequent weight load fail closed instead of downloading."""
    snapshot = _build_cached_repo(
        tmp_path,
        "org/half-downloaded",
        {
            "config.json": "{}",
            "model-00001-of-00002.safetensors": "first shard only",
        },
    )
    resolved = resolve_local_snapshot_path("org/half-downloaded", cache_dir = str(tmp_path))
    assert resolved is not None
    assert Path(resolved).resolve() == snapshot.resolve()


def test_uncached_repo_resolves_to_none(tmp_path):
    assert resolve_local_snapshot_path("org/never-downloaded", cache_dir = str(tmp_path)) is None


def test_newest_snapshot_preferred_over_refs_main(tmp_path):
    """A newer snapshot downloaded at an explicit revision outranks the older
    refs/main target: the inventory surfaces the newest snapshot by mtime, so
    the load must resolve the same one instead of an older (possibly
    incomplete) main revision."""
    import os
    import time

    old_main = _build_cached_repo(
        tmp_path,
        "org/newer-rev",
        {"config.json": "{}"},
        rev = "a" * 40,
    )
    stale = time.time() - 1000
    os.utime(old_main, (stale, stale))
    newer = _build_cached_repo(
        tmp_path,
        "org/newer-rev",
        {"config.json": "{}", "model.safetensors": "weights"},
        with_refs = False,
        rev = "b" * 40,
    )
    resolved = resolve_local_snapshot_path("org/newer-rev", cache_dir = str(tmp_path))
    assert resolved is not None
    assert Path(resolved).resolve() == newer.resolve()


def test_revision_only_snapshot_resolves_without_refs(tmp_path):
    """snapshot_download(local_files_only = True) needs refs/main, but the
    inventory scanner accepts revision-only layouts (pruned refs), so the
    resolver must fall back to the snapshot directory itself."""
    snapshot = _build_cached_repo(
        tmp_path,
        "org/no-refs",
        {"config.json": "{}", "model.safetensors": "weights"},
        with_refs = False,
    )
    resolved = resolve_local_snapshot_path("org/no-refs", cache_dir = str(tmp_path))
    assert resolved is not None
    assert Path(resolved).resolve() == snapshot.resolve()


def test_refless_fallback_picks_newest_snapshot_with_config(tmp_path):
    """With several revision dirs, the fallback must pick the newest one that
    actually holds a config.json, skipping empty or partial revisions."""
    import os
    import time

    old = _build_cached_repo(
        tmp_path,
        "org/multi-rev",
        {"config.json": "{}"},
        with_refs = False,
        rev = "a" * 40,
    )
    stale = time.time() - 1000
    os.utime(old, (stale, stale))
    new = _build_cached_repo(
        tmp_path,
        "org/multi-rev",
        {"config.json": "{}"},
        with_refs = False,
        rev = "b" * 40,
    )
    configless = _build_cached_repo(
        tmp_path,
        "org/multi-rev",
        {"tokenizer.json": "{}"},
        with_refs = False,
        rev = "c" * 40,
    )
    assert configless.exists()
    resolved = resolve_local_snapshot_path("org/multi-rev", cache_dir = str(tmp_path))
    assert resolved is not None
    assert Path(resolved).resolve() == new.resolve()


def test_gguf_rows_select_gguf_bearing_snapshot_in_mixed_repos(tmp_path):
    """A mixed repo caching a newer safetensors revision beside an older GGUF
    revision: the GGUF row's snapshot selection must return the GGUF-bearing
    revision, not the safetensors one the model row prefers."""
    import os
    import sys
    import time

    backend_dir = str(Path(__file__).resolve().parent.parent)
    if backend_dir not in sys.path:
        sys.path.insert(0, backend_dir)
    from hub.services.models.cache_inventory import (
        _cached_gguf_repo_snapshot_path,
        _cached_model_snapshot_path,
    )

    repo_dir = tmp_path / "models--org--mixed"
    gguf_rev = repo_dir / "snapshots" / ("a" * 40)
    gguf_rev.mkdir(parents = True)
    (gguf_rev / "mixed-Q4_K_M.gguf").write_text("gguf-bytes")
    stale = time.time() - 1000
    os.utime(gguf_rev, (stale, stale))
    st_rev = repo_dir / "snapshots" / ("b" * 40)
    st_rev.mkdir(parents = True)
    (st_rev / "config.json").write_text("{}")
    (st_rev / "model.safetensors").write_text("weights")

    gguf_pick = _cached_gguf_repo_snapshot_path(repo_dir)
    assert gguf_pick is not None
    assert Path(gguf_pick).resolve() == gguf_rev.resolve()
    model_pick = _cached_model_snapshot_path(repo_dir)
    assert model_pick is not None
    assert Path(model_pick).resolve() == st_rev.resolve()


def test_weightless_newest_snapshot_does_not_shadow_complete_older_one(tmp_path):
    """A newest metadata-only revision (config.json, no weights) must not win
    over an older revision holding the inventoried safetensors weights: the
    inventory made the row eligible from the weightful revision, so the load
    must resolve that one instead of failing on the weightless dir."""
    import os
    import time

    complete = _build_cached_repo(
        tmp_path,
        "org/meta-newest",
        {"config.json": "{}", "model.safetensors": "weights"},
        with_refs = False,
        rev = "a" * 40,
    )
    stale = time.time() - 1000
    os.utime(complete, (stale, stale))
    _build_cached_repo(
        tmp_path,
        "org/meta-newest",
        {"config.json": "{}"},
        with_refs = False,
        rev = "b" * 40,
    )
    resolved = resolve_local_snapshot_path("org/meta-newest", cache_dir = str(tmp_path))
    assert resolved is not None
    assert Path(resolved).resolve() == complete.resolve()


def test_refless_fallback_without_config_resolves_to_none(tmp_path):
    """A snapshots dir with no config.json anywhere is not a loadable text
    model cache; resolution must stay None (409 upstream), not guess."""
    _build_cached_repo(
        tmp_path,
        "org/no-config",
        {"tokenizer.json": "{}"},
        with_refs = False,
    )
    assert resolve_local_snapshot_path("org/no-config", cache_dir = str(tmp_path)) is None


def test_resolution_never_uses_the_network(tmp_path, monkeypatch):
    """local_files_only resolution must not open any connection even when the
    repo is absent (the tempting fallback would be a Hub metadata call)."""
    import socket

    def _no_network(*_args, **_kwargs):
        raise AssertionError("network access attempted during local resolution")

    monkeypatch.setattr(socket.socket, "connect", _no_network)
    _build_cached_repo(tmp_path, "org/offline-ok", {"config.json": "{}"})
    assert resolve_local_snapshot_path("org/offline-ok", cache_dir = str(tmp_path))
    assert resolve_local_snapshot_path("org/absent", cache_dir = str(tmp_path)) is None


def test_worker_rebuilds_metadata_from_selected_snapshot(tmp_path, monkeypatch):
    """The worker must not pair an older refs/main config with weights from the
    newer snapshot selected by inventory. Only the external registry identity
    is restored to the Hub repo id after local metadata resolution."""
    from core.inference.worker import _build_model_config
    from utils.models import ModelConfig

    snapshot = tmp_path / "snapshots" / ("b" * 40)
    snapshot.mkdir(parents = True)
    seen = {}
    snapshot_config = SimpleNamespace(
        identifier = str(snapshot),
        display_name = snapshot.name,
        path = str(snapshot),
        is_local = True,
        is_cached = True,
        is_vision = True,
        is_lora = False,
        is_gguf = False,
        is_audio = True,
        audio_type = "audio_vlm",
        has_audio_input = True,
        base_model = "org/new-base",
    )

    def _from_identifier(cls, **kwargs):
        seen.update(kwargs)
        return snapshot_config

    monkeypatch.setattr(ModelConfig, "from_identifier", classmethod(_from_identifier))
    result = _build_model_config(
        {
            "model_name": "org/model",
            "local_snapshot_path": str(snapshot),
            "hf_token": "",
        }
    )

    assert seen["model_id"] == str(snapshot)
    assert result.path == str(snapshot)
    assert result.identifier == "org/model"
    assert result.display_name == "model"
    assert result.is_vision is True
    assert result.base_model == "org/new-base"
