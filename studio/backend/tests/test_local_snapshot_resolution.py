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
) -> Path:
    """Lay out a minimal HF hub cache entry the way huggingface_hub expects:
    ``models--org--name/refs/main`` pointing at a snapshot directory."""
    repo_dir = cache_dir / f"models--{repo_id.replace('/', '--')}"
    snapshot = repo_dir / "snapshots" / _REV
    snapshot.mkdir(parents = True)
    (repo_dir / "refs").mkdir()
    (repo_dir / "refs" / "main").write_text(_REV)
    for name, content in files.items():
        (snapshot / name).write_text(content)
    return snapshot


def test_cached_repo_resolves_to_its_snapshot_dir(tmp_path):
    snapshot = _build_cached_repo(
        tmp_path,
        "org/tiny-model",
        {"config.json": "{}", "model.safetensors": "weights"},
    )
    resolved = resolve_local_snapshot_path(
        "org/tiny-model", cache_dir = str(tmp_path)
    )
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
    resolved = resolve_local_snapshot_path(
        "org/half-downloaded", cache_dir = str(tmp_path)
    )
    assert resolved is not None
    assert Path(resolved).resolve() == snapshot.resolve()


def test_uncached_repo_resolves_to_none(tmp_path):
    assert (
        resolve_local_snapshot_path("org/never-downloaded", cache_dir = str(tmp_path))
        is None
    )


def test_resolution_never_uses_the_network(tmp_path, monkeypatch):
    """local_files_only resolution must not open any connection even when the
    repo is absent (the tempting fallback would be a Hub metadata call)."""
    import socket

    def _no_network(*_args, **_kwargs):
        raise AssertionError("network access attempted during local resolution")

    monkeypatch.setattr(socket.socket, "connect", _no_network)
    _build_cached_repo(tmp_path, "org/offline-ok", {"config.json": "{}"})
    assert resolve_local_snapshot_path("org/offline-ok", cache_dir = str(tmp_path))
    assert (
        resolve_local_snapshot_path("org/absent", cache_dir = str(tmp_path)) is None
    )
