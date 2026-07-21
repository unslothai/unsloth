# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Cleanup of empty leftover quant folders from interrupted split downloads."""

import errno
from pathlib import Path
from types import SimpleNamespace

from hub.schemas.inventory import GgufVariantDetail, GgufVariantsResponse
from hub.services.models import deletion, gguf_variants
from hub.utils import gguf


def _make_snapshot(root: Path) -> Path:
    snap = root / "snapshots" / "rev0"
    (snap / "UD-IQ1_M").mkdir(parents = True)
    (snap / "UD-IQ1_M" / "GLM-UD-IQ1_M-00001-of-00002.gguf").write_bytes(b"x")
    (snap / "UD-IQ1_M" / "GLM-UD-IQ1_M-00002-of-00002.gguf").write_bytes(b"y")
    (snap / "UD-IQ1_S").mkdir(parents = True)  # empty leftover
    return snap


def test_list_empty_gguf_variant_dirs_finds_empty_leftover(tmp_path, monkeypatch):
    snap = _make_snapshot(tmp_path)
    monkeypatch.setattr(gguf, "iter_hf_cache_snapshots", lambda repo_id: iter([snap]))
    assert gguf.list_empty_gguf_variant_dirs("org/Repo-GGUF") == {"UD-IQ1_S"}


def test_list_empty_excludes_quant_with_files_in_another_snapshot(tmp_path, monkeypatch):
    snap1 = tmp_path / "s1" / "snapshots" / "rev"
    (snap1 / "UD-IQ1_S").mkdir(parents = True)  # empty here
    snap2 = tmp_path / "s2" / "snapshots" / "rev"
    (snap2 / "UD-IQ1_S").mkdir(parents = True)
    (snap2 / "UD-IQ1_S" / "m-UD-IQ1_S-00001-of-00001.gguf").write_bytes(b"z")  # has shards
    monkeypatch.setattr(gguf, "iter_hf_cache_snapshots", lambda repo_id: iter([snap1, snap2]))
    assert gguf.list_empty_gguf_variant_dirs("org/Repo-GGUF") == set()


def test_list_empty_ignores_non_quant_dirs(tmp_path, monkeypatch):
    snap = tmp_path / "snapshots" / "rev"
    (snap / "not-a-quant").mkdir(parents = True)  # empty but not a quant label
    monkeypatch.setattr(gguf, "iter_hf_cache_snapshots", lambda repo_id: iter([snap]))
    assert gguf.list_empty_gguf_variant_dirs("org/Repo-GGUF") == set()


def test_remove_empty_variant_dirs_removes_only_empty_match(tmp_path):
    snap = _make_snapshot(tmp_path)
    repo = SimpleNamespace(repo_path = str(tmp_path))
    removed, failures = deletion._remove_empty_variant_dirs([repo], "UD-IQ1_S")
    assert removed == 1
    assert failures == []
    assert not (snap / "UD-IQ1_S").exists()
    assert (snap / "UD-IQ1_M").is_dir()


def test_remove_empty_variant_dirs_never_touches_populated_folder(tmp_path):
    snap = _make_snapshot(tmp_path)
    repo = SimpleNamespace(repo_path = str(tmp_path))
    removed, failures = deletion._remove_empty_variant_dirs([repo], "UD-IQ1_M")
    assert removed == 0
    assert failures == []
    assert len(list((snap / "UD-IQ1_M").iterdir())) == 2


def test_remove_empty_variant_dirs_surfaces_real_failure(tmp_path, monkeypatch):
    _make_snapshot(tmp_path)
    repo = SimpleNamespace(repo_path = str(tmp_path))

    def _denied(self):
        raise OSError(errno.EACCES, "permission denied")

    monkeypatch.setattr(Path, "rmdir", _denied)
    removed, failures = deletion._remove_empty_variant_dirs([repo], "UD-IQ1_S")
    assert removed == 0
    assert len(failures) == 1


def test_remove_empty_variant_dirs_ignores_concurrent_refill(tmp_path, monkeypatch):
    _make_snapshot(tmp_path)
    repo = SimpleNamespace(repo_path = str(tmp_path))

    def _refilled(self):
        raise OSError(errno.ENOTEMPTY, "directory not empty")

    monkeypatch.setattr(Path, "rmdir", _refilled)
    removed, failures = deletion._remove_empty_variant_dirs([repo], "UD-IQ1_S")
    assert removed == 0
    assert failures == []


def test_mark_empty_dir_cleanables_appends_unlisted(monkeypatch):
    monkeypatch.setattr(gguf_variants, "list_empty_gguf_variant_dirs", lambda repo_id: {"UD-IQ1_S"})
    resp = GgufVariantsResponse(
        repo_id = "org/Repo-GGUF",
        variants = [GgufVariantDetail(filename = "m-UD-IQ1_M.gguf", quant = "UD-IQ1_M", downloaded = True)],
    )
    out = gguf_variants._mark_empty_dir_cleanables("org/Repo-GGUF", resp)
    by_q = {v.quant: v for v in out.variants}
    assert by_q["UD-IQ1_M"].downloaded is True
    assert by_q["UD-IQ1_S"].partial is True and by_q["UD-IQ1_S"].downloaded is False


def test_mark_empty_dir_cleanables_flips_listed_variant(monkeypatch):
    monkeypatch.setattr(gguf_variants, "list_empty_gguf_variant_dirs", lambda repo_id: {"UD-IQ1_S"})
    resp = GgufVariantsResponse(
        repo_id = "org/Repo-GGUF",
        variants = [GgufVariantDetail(filename = "m-UD-IQ1_S.gguf", quant = "UD-IQ1_S")],
    )
    out = gguf_variants._mark_empty_dir_cleanables("org/Repo-GGUF", resp)
    assert len(out.variants) == 1
    assert out.variants[0].partial is True


def _force_compute_to_raise(monkeypatch):
    # Drive _compute() down its remote path, fail metadata, and have both cache
    # fallbacks miss so the original error re-raises.
    def _boom(*a, **k):
        raise RuntimeError("offline")

    monkeypatch.setattr(gguf_variants, "list_gguf_variants", _boom, raising = False)
    monkeypatch.setattr(
        gguf_variants,
        "list_gguf_variants_from_hf_cache",
        lambda repo_id, hf_token = None: None,
        raising = False,
    )
    monkeypatch.setattr(
        gguf_variants, "list_partial_gguf_variants_from_state", lambda repo_id: None, raising = False
    )


def test_get_variants_surfaces_cleanable_when_metadata_fails(monkeypatch):
    # Offline / model_info fails and only an empty leftover folder is cached:
    # the cleanable must still be returned instead of the error propagating.
    import asyncio

    _force_compute_to_raise(monkeypatch)
    monkeypatch.setattr(gguf_variants, "list_empty_gguf_variant_dirs", lambda repo_id: {"UD-IQ1_S"})

    resp = asyncio.run(
        gguf_variants.get_gguf_variants_response(
            "org/Repo-GGUF", prefer_local_cache = False, hf_token = None
        )
    )
    by_q = {v.quant: v for v in resp.variants}
    assert "UD-IQ1_S" in by_q
    assert by_q["UD-IQ1_S"].partial is True and by_q["UD-IQ1_S"].downloaded is False


def test_get_variants_reraises_when_no_cleanable(monkeypatch):
    # Offline with nothing cleanable: original error must propagate (as HTTP).
    import asyncio

    from fastapi import HTTPException

    _force_compute_to_raise(monkeypatch)
    monkeypatch.setattr(gguf_variants, "list_empty_gguf_variant_dirs", lambda repo_id: set())

    try:
        asyncio.run(
            gguf_variants.get_gguf_variants_response(
                "org/Repo-GGUF", prefer_local_cache = False, hf_token = None
            )
        )
        raised = False
    except (HTTPException, RuntimeError):
        raised = True
    assert raised
