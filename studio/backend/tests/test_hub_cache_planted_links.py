# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tool sandboxes can write the Hub cache, so a snapshot entry may be a link to any host file.
Server-side readers that hand cache contents to an account must not follow such a link out of the cache."""

import json
import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(os.name == "nt", reason = "creates POSIX symlinks")

SECRET = "host-secret-not-in-the-cache"


def _snapshot(root: Path, kind: str, repo: str) -> Path:
    snap = root / f"{kind}s--{repo.replace('/', '--')}" / "snapshots" / ("a" * 40)
    snap.mkdir(parents = True)
    return snap


@pytest.fixture
def host_secret(tmp_path):
    secret = tmp_path / "outside" / "secret.txt"
    secret.parent.mkdir()
    secret.write_text(SECRET, encoding = "utf-8")
    return secret


def test_dataset_preview_skips_a_snapshot_with_a_link_out_of_the_cache(tmp_path, host_secret):
    from hub.utils import dataset_cache

    snap = _snapshot(tmp_path / "hub", "dataset", "org/data")
    (snap / "train.txt").write_text("row\n", encoding = "utf-8")
    assert dataset_cache.cached_dataset_candidates(
        snap, subset = None, train_split = "train", extensions = (".txt",)
    )
    (snap / "notes.txt").symlink_to(host_secret)
    assert (
        dataset_cache.cached_dataset_candidates(
            snap, subset = None, train_split = "train", extensions = (".txt",)
        )
        == []
    )
    assert not dataset_cache._snapshot_links_stay_in_cache(snap)


def test_dataset_snapshot_links_into_blobs_still_read(tmp_path):
    from hub.utils import dataset_cache

    snap = _snapshot(tmp_path / "hub", "dataset", "org/data")
    blob = snap.parent.parent / "blobs" / ("b" * 40)
    blob.parent.mkdir()
    blob.write_text("row\n", encoding = "utf-8")
    (snap / "train.txt").symlink_to(os.path.relpath(blob, snap))
    assert dataset_cache.cached_dataset_candidates(
        snap, subset = None, train_split = "train", extensions = (".txt",)
    ) == [snap / "train.txt"]


def test_example_import_does_not_copy_a_linked_host_file(tmp_path, host_secret, monkeypatch):
    import huggingface_hub
    from routes import training

    snap = _snapshot(tmp_path / "hub", "dataset", "org/tarot")
    (snap / "real.png").write_bytes(b"\x89PNG real")
    (snap / "zz_planted.png").symlink_to(host_secret)
    (snap / "captions.jsonl").write_text(
        json.dumps({"file_name": "real.png", "text": "a card"}) + "\n", encoding = "utf-8"
    )
    (snap / "zz_planted.jsonl").symlink_to(host_secret)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda *a, **k: str(snap))

    dest = tmp_path / "dest"
    dest.mkdir()
    written = training._materialize_imagefolder_jsonl({"repo": "org/tarot"}, dest, cap = 10)

    assert written == 1
    copied = b"".join(p.read_bytes() for p in dest.iterdir())
    assert SECRET.encode() not in copied
    assert (dest / "img_0000.png").read_bytes() == b"\x89PNG real"
    assert (dest / "img_0000.txt").read_text(encoding = "utf-8") == "a card"


def test_chat_template_walk_does_not_read_a_linked_host_file(tmp_path, monkeypatch):
    from picker import service

    secret = tmp_path / "outside" / "tokenizer_config.json"
    secret.parent.mkdir()
    secret.write_text(json.dumps({"chat_template": SECRET}), encoding = "utf-8")
    snap = _snapshot(tmp_path / "hub", "model", "org/model")
    (snap / "tokenizer_config.json").symlink_to(secret)
    monkeypatch.setattr(service, "resolve_cached_repo_id_case", lambda name: name)
    monkeypatch.setattr(
        service, "iter_snapshots_preferring_whole", lambda resolved, variant: [snap]
    )
    monkeypatch.setattr(service, "cache_reads_authorized", lambda *a, **k: True)
    monkeypatch.setattr(service, "hf_env_offline", lambda: True)

    assert service.read_default_chat_template("org/model") != SECRET

    # The same file kept inside the snapshot is still read.
    (snap / "tokenizer_config.json").unlink()
    (snap / "tokenizer_config.json").write_text(
        json.dumps({"chat_template": "IN_CACHE"}), encoding = "utf-8"
    )
    assert service.read_default_chat_template("org/model") == "IN_CACHE"


@pytest.mark.skipif(os.name == "nt", reason = "symlinked blob layout")
def test_a_recovered_shared_store_blob_maps_back_to_the_repos_own_link(tmp_path):
    # A repo recovered from a shared blob store has scan paths resolved into the cache-wide store; its
    # own blobs/<hash> link must still be found so the delete can unlink it and sweep the shared payload.
    from hub.services.models import deletion

    digest = "a" * 64
    shared = tmp_path / "blobs" / digest
    shared.parent.mkdir()
    shared.write_bytes(b"payload")
    repo = tmp_path / "models--org--m"
    (repo / "blobs").mkdir(parents = True)
    (repo / "blobs" / digest).symlink_to(shared)
    assert deletion._own_blob_link(shared, repo) == repo / "blobs" / digest
    assert deletion._own_blob_link(repo / "blobs" / digest, repo) == repo / "blobs" / digest
    # Same name elsewhere, not where the repo's link leads: not the repo's to free.
    stray = tmp_path / "elsewhere" / digest
    stray.parent.mkdir()
    stray.write_bytes(b"host file")
    assert deletion._own_blob_link(stray, repo) is None
