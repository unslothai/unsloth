# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Variant-file selection guards for the cached-model-path endpoint.

The Copy path / Reveal endpoint must resolve a quant label to the same file
the variant menus offer: MTP drafters, mmproj vision adapters, and big-endian
builds are excluded, and directory layouts (``BF16/model-00001-of-....gguf``)
resolve their label from the snapshot-relative path, not the basename.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


def _find_repo_root() -> Path | None:
    env = os.environ.get("UNSLOTH_REPO_ROOT")
    if env:
        p = Path(env).resolve()
        if (p / "studio" / "backend").is_dir():
            return p
    here = Path(__file__).resolve()
    for parent in (here, *here.parents):
        if (parent / "studio" / "backend").is_dir():
            return parent
    return None


_REPO_ROOT = _find_repo_root()
if _REPO_ROOT is None:
    pytest.skip(
        "Could not locate studio/backend. Set UNSLOTH_REPO_ROOT or run from "
        "the repository checkout.",
        allow_module_level = True,
    )

_STUDIO_BACKEND = _REPO_ROOT / "studio" / "backend"
if str(_STUDIO_BACKEND) not in sys.path:
    sys.path.insert(0, str(_STUDIO_BACKEND))

pytest.importorskip("fastapi")
pytest.importorskip("huggingface_hub")

try:
    from routes import models as routes_models
except Exception as exc:
    pytest.skip(f"studio backend import unavailable: {exc}", allow_module_level = True)

from fastapi import HTTPException


def test_plain_quant_label_resolves():
    assert routes_models._main_variant_gguf_label("Model-Q8_0.gguf") == "Q8_0"


def test_mtp_drafter_in_subdir_is_excluded():
    assert routes_models._main_variant_gguf_label("MTP/Model-Q8_0-MTP.gguf") is None


def test_mtp_drafter_root_prefix_is_excluded():
    assert routes_models._main_variant_gguf_label("mtp-Model-Q8_0.gguf") is None


def test_mmproj_adapter_is_excluded():
    assert routes_models._main_variant_gguf_label("mmproj-Model-F16.gguf") is None


def test_directory_layout_quant_resolves_from_parent_dir():
    assert routes_models._main_variant_gguf_label("BF16/Model-00001-of-00002.gguf") == "BF16"


def test_big_endian_build_is_excluded():
    assert routes_models._main_variant_gguf_label("Model-Q8_0-BE.gguf") is None


def test_non_gguf_file_is_excluded():
    assert routes_models._main_variant_gguf_label("config.json") is None


def test_normalized_quant_label_ignores_separators():
    assert routes_models._normalized_quant_label("UD-Q4_K_XL") == "udq4kxl"
    assert routes_models._normalized_quant_label("Q8-0") == routes_models._normalized_quant_label(
        "Q8_0"
    )


def _revision(
    snapshot: Path,
    last_modified: float,
    names: list[str],
    size_on_disk: int = 4,
) -> SimpleNamespace:
    files = []
    for name in names:
        path = snapshot / name
        path.parent.mkdir(parents = True, exist_ok = True)
        path.write_bytes(b"x" * size_on_disk)
        files.append(
            SimpleNamespace(
                file_name = name,
                file_path = path,
                blob_path = path,
                size_on_disk = size_on_disk,
            )
        )
    return SimpleNamespace(snapshot_path = snapshot, last_modified = last_modified, files = files)


def _patch_cache(monkeypatch, tmp_path: Path, revisions: list[SimpleNamespace]) -> None:
    repo = SimpleNamespace(
        repo_id = "Org/Repo",
        repo_type = "model",
        repo_path = tmp_path,
        revisions = revisions,
    )
    monkeypatch.setattr(
        routes_models, "_all_hf_cache_scans", lambda: [SimpleNamespace(repos = [repo])]
    )


def _repo(root: Path, revisions: list[SimpleNamespace]) -> SimpleNamespace:
    return SimpleNamespace(
        repo_id = "Org/Repo",
        repo_type = "model",
        repo_path = root,
        revisions = revisions,
    )


def _patch_caches(monkeypatch, repos: list[SimpleNamespace]) -> None:
    monkeypatch.setattr(
        routes_models,
        "_all_hf_cache_scans",
        lambda: [SimpleNamespace(repos = [repo]) for repo in repos],
    )


@pytest.mark.parametrize("newest_first", [True, False])
def test_variant_resolves_from_newest_revision(monkeypatch, tmp_path, newest_first):
    old = _revision(tmp_path / "snapshots" / "aaa", 1_000.0, ["Model-Q4_K_M.gguf"])
    new = _revision(tmp_path / "snapshots" / "bbb", 2_000.0, ["Model-Q4_K_M.gguf"])
    _patch_cache(monkeypatch, tmp_path, [new, old] if newest_first else [old, new])
    resolved = routes_models._resolve_cached_model_path("Org/Repo", "Q4_K_M")
    assert resolved == tmp_path / "snapshots" / "bbb" / "Model-Q4_K_M.gguf"


def test_sharded_variant_resolves_first_split(monkeypatch, tmp_path):
    rev = _revision(
        tmp_path / "snapshots" / "aaa",
        1_000.0,
        ["Model-Q4_K_M-00002-of-00002.gguf", "Model-Q4_K_M-00001-of-00002.gguf"],
    )
    _patch_cache(monkeypatch, tmp_path, [rev])
    resolved = routes_models._resolve_cached_model_path("Org/Repo", "Q4_K_M")
    assert resolved.name == "Model-Q4_K_M-00001-of-00002.gguf"


def test_variant_only_in_older_revision_resolves(monkeypatch, tmp_path):
    old = _revision(tmp_path / "snapshots" / "aaa", 1_000.0, ["Model-Q4_K_M.gguf"])
    new = _revision(tmp_path / "snapshots" / "bbb", 2_000.0, ["Model-Q8_0.gguf"])
    _patch_cache(monkeypatch, tmp_path, [new, old])
    resolved = routes_models._resolve_cached_model_path("Org/Repo", "Q4_K_M")
    assert resolved == tmp_path / "snapshots" / "aaa" / "Model-Q4_K_M.gguf"


def test_missing_newest_file_falls_back_to_older_revision(monkeypatch, tmp_path):
    old = _revision(tmp_path / "snapshots" / "aaa", 1_000.0, ["Model-Q4_K_M.gguf"])
    new_snapshot = tmp_path / "snapshots" / "bbb"
    new_snapshot.mkdir(parents = True)
    new = SimpleNamespace(
        snapshot_path = new_snapshot,
        last_modified = 2_000.0,
        files = [
            SimpleNamespace(
                file_name = "Model-Q4_K_M.gguf",
                file_path = new_snapshot / "Model-Q4_K_M.gguf",
            )
        ],
    )
    _patch_cache(monkeypatch, tmp_path, [new, old])
    resolved = routes_models._resolve_cached_model_path("Org/Repo", "Q4_K_M")
    assert resolved == tmp_path / "snapshots" / "aaa" / "Model-Q4_K_M.gguf"


def test_variant_resolves_across_all_cache_roots(monkeypatch, tmp_path):
    first_root = tmp_path / "active"
    second_root = tmp_path / "default"
    old = _revision(
        first_root / "snapshots" / "aaa",
        1_000.0,
        ["Model-Q4_K_M.gguf"],
    )
    new = _revision(
        second_root / "snapshots" / "bbb",
        2_000.0,
        ["Model-Q4_K_M.gguf"],
    )
    _patch_caches(
        monkeypatch,
        [_repo(first_root, [old]), _repo(second_root, [new])],
    )
    resolved = routes_models._resolve_cached_model_path("Org/Repo", "Q4_K_M")
    assert resolved == second_root / "snapshots" / "bbb" / "Model-Q4_K_M.gguf"


def test_repo_path_matches_largest_visible_cache_entry(monkeypatch, tmp_path):
    first_root = tmp_path / "active"
    second_root = tmp_path / "default"
    small = _revision(
        first_root / "snapshots" / "aaa",
        2_000.0,
        ["Model-Q8_0.gguf"],
        size_on_disk = 4,
    )
    large = _revision(
        second_root / "snapshots" / "bbb",
        1_000.0,
        ["Model-Q8_0.gguf"],
        size_on_disk = 8,
    )
    _patch_caches(
        monkeypatch,
        [_repo(first_root, [small]), _repo(second_root, [large])],
    )
    resolved = routes_models._resolve_cached_model_path("Org/Repo", None)
    assert resolved == second_root / "snapshots" / "bbb"


def test_unknown_variant_raises_404(monkeypatch, tmp_path):
    rev = _revision(tmp_path / "snapshots" / "aaa", 1_000.0, ["Model-Q8_0.gguf"])
    _patch_cache(monkeypatch, tmp_path, [rev])
    with pytest.raises(HTTPException) as excinfo:
        routes_models._resolve_cached_model_path("Org/Repo", "Q4_K_M")
    assert excinfo.value.status_code == 404
    assert "Q4_K_M" in excinfo.value.detail
