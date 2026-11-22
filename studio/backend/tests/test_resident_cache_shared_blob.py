# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A resident GGUF in Hub >= 1.32's cache-wide shared blob store keeps its repo identity."""

import os

import pytest

from hub.services.models import account_access as access

HASH = "ab" * 32
ETAG = "cd" * 32


def _cache(monkeypatch, tmp_path):
    from utils import hf_cache_settings

    cache = tmp_path / "hub"
    (cache / "blobs" / "ab").mkdir(parents = True)
    (cache / "blobs" / ".huggingface-shared-blobs").write_text("1")
    (cache / "blobs" / "ab" / HASH).write_bytes(b"")
    monkeypatch.setattr(hf_cache_settings, "known_hf_hub_caches", lambda: [cache])
    return cache


def _repo(cache, name, blob_target):
    repo = cache / f"models--org--{name}"
    (repo / "blobs").mkdir(parents = True)
    (repo / "snapshots" / "revision").mkdir(parents = True)
    (repo / "blobs" / ETAG).symlink_to(blob_target)
    pointer = repo / "snapshots" / "revision" / "weights.gguf"
    pointer.symlink_to(f"../../blobs/{ETAG}")
    return pointer


@pytest.mark.skipif(os.name == "nt", reason = "shared blobs need symlinks")
def test_shared_blob_resident_keeps_its_repo(monkeypatch, tmp_path):
    cache = _cache(monkeypatch, tmp_path)
    pointer = _repo(cache, "private", f"../../blobs/ab/{HASH}")
    assert pointer.resolve() == cache / "blobs" / "ab" / HASH
    assert access.cached_model_repo(str(pointer)) == "org/private"


@pytest.mark.skipif(os.name == "nt", reason = "shared blobs need symlinks")
def test_a_pointer_into_another_repo_is_still_refused(monkeypatch, tmp_path):
    cache = _cache(monkeypatch, tmp_path)
    other = cache / "models--org--other" / "blobs"
    other.mkdir(parents = True)
    (other / "real").write_bytes(b"")
    pointer = _repo(cache, "private", "../../models--org--other/blobs/real")
    assert access.cached_model_repo(str(pointer)) is None


@pytest.mark.skipif(os.name == "nt", reason = "shared blobs need symlinks")
def test_a_shared_store_outside_the_known_cache_is_still_refused(monkeypatch, tmp_path):
    from utils import hf_cache_settings

    cache = _cache(monkeypatch, tmp_path)
    pointer = _repo(cache, "private", f"../../blobs/ab/{HASH}")
    monkeypatch.setattr(hf_cache_settings, "known_hf_hub_caches", lambda: [tmp_path / "elsewhere"])
    assert access.cached_model_repo(str(pointer)) is None


@pytest.mark.skipif(os.name == "nt", reason = "shared blobs need symlinks")
def test_a_repo_blob_pointing_outside_the_shared_store_is_refused(monkeypatch, tmp_path):
    cache = _cache(monkeypatch, tmp_path)
    (tmp_path / "loose").write_bytes(b"")
    pointer = _repo(cache, "private", "../../../loose")
    assert access.cached_model_repo(str(pointer)) is None


@pytest.mark.skipif(os.name == "nt", reason = "shared blobs need symlinks")
@pytest.mark.parametrize("store_entry", ["missing", "directory", "not-a-hash"])
def test_a_store_entry_huggingface_hub_would_not_use_is_refused(monkeypatch, tmp_path, store_entry):
    cache = _cache(monkeypatch, tmp_path)
    if store_entry == "missing":
        target = f"../../blobs/ab/{'ef' * 32}"
    elif store_entry == "directory":
        (cache / "blobs" / "ab" / ("ab" * 31 + "ef")).mkdir()
        target = f"../../blobs/ab/{'ab' * 31 + 'ef'}"
    else:
        (cache / "blobs" / "ab" / "weights.gguf").write_bytes(b"")
        target = "../../blobs/ab/weights.gguf"
    pointer = _repo(cache, "private", target)
    assert access.cached_model_repo(str(pointer)) is None


@pytest.mark.skipif(os.name == "nt", reason = "shared blobs need symlinks")
def test_a_granted_shared_blob_resident_is_visible(monkeypatch, tmp_path):
    cache = _cache(monkeypatch, tmp_path)
    pointer = _repo(cache, "private", f"../../blobs/ab/{HASH}")
    monkeypatch.setattr(access, "managed_account", lambda: True)
    monkeypatch.setattr(access, "model_allowed", lambda repo_id: True)
    monkeypatch.setattr(access, "repo_is_public", lambda *args, **kwargs: False)
    granted = {access._grant_key("org/private", "model")}
    assert access.model_visible(str(pointer), grants = granted)
    assert not access.model_visible(str(pointer), grants = set())
    assert not access.model_visible(str(pointer), grants = {access._grant_key("org/other", "model")})
