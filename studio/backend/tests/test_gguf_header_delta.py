# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Rebuilding a republished GGUF from the cached older copy when only its header changed."""

from __future__ import annotations

import hashlib
import os
import struct
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from hub.utils import gguf_header_delta as delta

COMMIT_OLD = "a" * 40
COMMIT_NEW = "b" * 40
NAME = "model-Q4_K_M.gguf"


def _s(text: str) -> bytes:
    raw = text.encode()
    return struct.pack("<Q", len(raw)) + raw


def make_gguf(kvs: dict, tensors: list, *, alignment: int = 32) -> bytes:
    """``tensors``: (name, dims, ggml type, payload bytes). String KVs only, plus an optional alignment KV."""
    head = b"GGUF" + struct.pack("<I", 3) + struct.pack("<QQ", len(tensors), len(kvs))
    for key, value in kvs.items():
        if key == "general.alignment":
            head += _s(key) + struct.pack("<I", 4) + struct.pack("<I", value)
        else:
            head += _s(key) + struct.pack("<I", 8) + _s(value)
    offset, blobs = 0, b""
    for name, dims, ggml_type, payload in tensors:
        head += _s(name) + struct.pack("<I", len(dims)) + b"".join(struct.pack("<Q", d) for d in dims)
        head += struct.pack("<I", ggml_type) + struct.pack("<Q", offset)
        pad = -len(payload) % alignment
        blobs += payload + b"\0" * pad
        offset += len(payload) + pad
    head += b"\0" * (-len(head) % alignment)
    return head + blobs


TENSORS = [
    ("blk.0.attn_q.weight", (8, 4), 0, os.urandom(4096)),
    ("blk.0.ffn_up.weight", (16, 4), 1, os.urandom(4096)),
]
OLD = make_gguf({"general.architecture": "llama"}, TENSORS)
NEW = make_gguf(
    {"general.architecture": "llama", "tokenizer.chat_template": "{{ fixed }}" * 50}, TENSORS
)


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class Fetcher:
    def __init__(self, data: bytes, *, fail: bool = False):
        self.data, self.fail, self.calls = data, fail, []

    def __call__(self, start: int, end: int) -> bytes:
        self.calls.append((start, end))
        if self.fail:
            raise ValueError("range request answered 200")
        return self.data[start : end + 1]


def cache_with_old(tmp_path: Path, *, symlinks: bool, data: bytes = OLD) -> Path:
    repo_dir = tmp_path / "models--unsloth--Model-GGUF"
    snap = repo_dir / "snapshots" / COMMIT_OLD
    snap.mkdir(parents = True)
    (repo_dir / "refs").mkdir()
    (repo_dir / "refs" / "main").write_text(COMMIT_OLD)
    if symlinks:
        blobs = repo_dir / "blobs"
        blobs.mkdir()
        (blobs / sha(data)).write_bytes(data)
        os.symlink(f"../../blobs/{sha(data)}", snap / NAME)
    else:
        (snap / NAME).write_bytes(data)
    return repo_dir


@pytest.fixture(autouse = True)
def _small_ranges(monkeypatch):
    monkeypatch.delenv(delta.DELTA_ENV, raising = False)
    monkeypatch.setattr(delta, "_FIRST_RANGE", 64)
    delta.reset_for_tests()


def rebuild(repo_dir: Path, new: bytes = NEW, fetch = None, **kw):
    return delta.rebuild_from_older_snapshot(
        repo_dir, COMMIT_NEW, NAME, len(new), sha(new), fetch or Fetcher(new), **kw
    )


def test_parse_layout_matches_writer():
    layout = delta.parse_gguf_layout(NEW)
    assert [t[0] for t in layout.tensors] == [b"blk.0.attn_q.weight", b"blk.0.ffn_up.weight"]
    assert layout.data_offset % 32 == 0
    assert NEW[layout.data_offset :] == OLD[delta.parse_gguf_layout(OLD).data_offset :]


def test_parse_layout_honours_alignment_kv():
    data = make_gguf({"general.alignment": 64}, TENSORS, alignment = 64)
    assert delta.parse_gguf_layout(data).data_offset % 64 == 0


def test_parse_truncated_asks_for_more():
    with pytest.raises(delta._NeedMore):
        delta.parse_gguf_layout(NEW[:40])


def test_header_only_change_rebuilds_in_blob_layout(tmp_path):
    repo_dir = cache_with_old(tmp_path, symlinks = True)
    fetch = Fetcher(NEW)
    result = rebuild(repo_dir, fetch = fetch)
    assert result.placed, result.reason
    pointer = repo_dir / "snapshots" / COMMIT_NEW / NAME
    assert pointer.is_symlink()
    assert pointer.read_bytes() == NEW
    assert (repo_dir / "blobs" / sha(NEW)).read_bytes() == NEW
    # Only the header crossed the network: ranges grow by doubling, so at most twice its size.
    assert result.fetched_bytes <= 2 * delta.parse_gguf_layout(NEW).data_offset < len(NEW)
    # The old revision is untouched.
    assert (repo_dir / "snapshots" / COMMIT_OLD / NAME).read_bytes() == OLD


def test_header_only_change_rebuilds_in_no_symlink_layout(tmp_path):
    repo_dir = cache_with_old(tmp_path, symlinks = False)
    result = rebuild(repo_dir)
    assert result.placed, result.reason
    pointer = repo_dir / "snapshots" / COMMIT_NEW / NAME
    assert not pointer.is_symlink() and pointer.read_bytes() == NEW
    assert not (repo_dir / "blobs").exists()


def test_huggingface_hub_then_serves_the_rebuilt_file(tmp_path):
    repo_dir = cache_with_old(tmp_path, symlinks = True)
    assert rebuild(repo_dir).placed
    from huggingface_hub import try_to_load_from_cache

    (repo_dir / "refs" / "main").write_text(COMMIT_NEW)
    got = try_to_load_from_cache("unsloth/Model-GGUF", NAME, cache_dir = str(tmp_path))
    assert Path(got).read_bytes() == NEW


@pytest.mark.parametrize(
    "new",
    [
        # A tensor's bytes changed: same table, the sha256 check refuses it.
        make_gguf(
            {"general.architecture": "llama", "x": "y"},
            [TENSORS[0], (TENSORS[1][0], TENSORS[1][1], TENSORS[1][2], os.urandom(4096))],
        ),
        # A tensor's size changed.
        make_gguf(
            {"general.architecture": "llama"},
            [TENSORS[0], (TENSORS[1][0], (16, 8), 1, os.urandom(8192))],
        ),
        # Same tensors, reordered.
        make_gguf({"general.architecture": "llama", "x": "y"}, [TENSORS[1], TENSORS[0]]),
        # Requantized type.
        make_gguf(
            {"general.architecture": "llama"},
            [TENSORS[0], (TENSORS[1][0], TENSORS[1][1], 2, TENSORS[1][3])],
        ),
    ],
    ids = ["tensor-bytes", "tensor-size", "reordered", "type"],
)
def test_changed_tensors_fall_back(tmp_path, new):
    repo_dir = cache_with_old(tmp_path, symlinks = True)
    result = rebuild(repo_dir, new = new)
    assert not result.placed
    assert not (repo_dir / "snapshots" / COMMIT_NEW).exists()
    assert not (repo_dir / "blobs" / sha(new)).exists()
    assert not list((repo_dir / "blobs").glob(".*delta-*"))


def test_hash_mismatch_falls_back_and_cleans_up(tmp_path):
    repo_dir = cache_with_old(tmp_path, symlinks = True)
    result = delta.rebuild_from_older_snapshot(
        repo_dir, COMMIT_NEW, NAME, len(NEW), "f" * 64, Fetcher(NEW)
    )
    assert not result.placed and result.reason == "sha256 mismatch"
    assert sorted(p.name for p in (repo_dir / "blobs").iterdir()) == [sha(OLD)]


def test_corrupt_cached_copy_falls_back(tmp_path):
    bad = bytearray(OLD)
    bad[-5] ^= 0xFF
    repo_dir = cache_with_old(tmp_path, symlinks = False, data = bytes(bad))
    result = rebuild(repo_dir)
    assert not result.placed and result.reason == "sha256 mismatch"


def test_range_failure_falls_back(tmp_path):
    repo_dir = cache_with_old(tmp_path, symlinks = True)
    result = rebuild(repo_dir, fetch = Fetcher(NEW, fail = True))
    assert not result.placed and result.reason.startswith("error")


def test_short_range_response_falls_back(tmp_path):
    repo_dir = cache_with_old(tmp_path, symlinks = True)
    result = rebuild(repo_dir, fetch = lambda start, end: NEW[start : start + 3])
    assert not result.placed


def test_header_past_cap_falls_back(tmp_path, monkeypatch):
    monkeypatch.setattr(delta, "_MAX_HEADER_BYTES", 80)
    repo_dir = cache_with_old(tmp_path, symlinks = True)
    assert not rebuild(repo_dir).placed


def test_kill_switch(tmp_path, monkeypatch):
    monkeypatch.setenv(delta.DELTA_ENV, "0")
    repo_dir = cache_with_old(tmp_path, symlinks = True)
    fetch = Fetcher(NEW)
    assert rebuild(repo_dir, fetch = fetch).reason == "disabled"
    assert fetch.calls == []


def test_no_older_copy_asks_nothing(tmp_path):
    repo_dir = tmp_path / "models--unsloth--Model-GGUF"
    fetch = Fetcher(NEW)
    assert rebuild(repo_dir, fetch = fetch).reason == "no older copy"
    assert fetch.calls == []


def test_protected_blob_is_left_alone(tmp_path):
    repo_dir = cache_with_old(tmp_path, symlinks = True)
    fetch = Fetcher(NEW)
    result = rebuild(repo_dir, fetch = fetch, protected_blob_hashes = frozenset({sha(NEW)}))
    assert result.reason == "blob protected" and fetch.calls == []


def test_locked_blob_is_left_alone(tmp_path):
    repo_dir = cache_with_old(tmp_path, symlinks = True)
    from filelock import FileLock

    lock = delta._lock_path(repo_dir, sha(NEW))
    lock.parent.mkdir(parents = True, exist_ok = True)
    held, release = threading.Event(), threading.Event()

    def hold():
        with FileLock(str(lock)):
            held.set()
            release.wait(5)

    t = threading.Thread(target = hold)
    t.start()
    held.wait(5)
    try:
        result = rebuild(repo_dir)
    finally:
        release.set()
        t.join()
    assert result.reason == "blob locked"
    assert not (repo_dir / "blobs" / sha(NEW)).exists()


def test_already_cached_target_is_left_alone(tmp_path):
    repo_dir = cache_with_old(tmp_path, symlinks = True)
    (repo_dir / "blobs" / sha(NEW)).write_bytes(NEW)
    fetch = Fetcher(NEW)
    assert rebuild(repo_dir, fetch = fetch).reason == "already cached"
    assert fetch.calls == []


def test_not_enough_disk_falls_back(tmp_path, monkeypatch):
    repo_dir = cache_with_old(tmp_path, symlinks = True)
    monkeypatch.setattr(delta.shutil, "disk_usage", lambda p: SimpleNamespace(free = 10))
    assert rebuild(repo_dir).reason == "not enough disk"


def test_cancel_falls_back(tmp_path):
    repo_dir = cache_with_old(tmp_path, symlinks = True)
    cancel = threading.Event()
    cancel.set()
    assert not rebuild(repo_dir, cancel_event = cancel).placed
    assert not list((repo_dir / "blobs").glob(".*delta-*"))


def _meta(data: bytes, commit: str = COMMIT_NEW):
    return SimpleNamespace(commit_hash = commit, etag = sha(data), size = len(data))


def test_hub_entry_rebuilds_and_remembers_up_to_date(tmp_path):
    repo_dir = cache_with_old(tmp_path, symlinks = True)
    calls = []

    def metadata():
        calls.append(1)
        return _meta(NEW)

    kw = dict(
        cache_dir = str(tmp_path),
        metadata_fn = metadata,
        fetcher_fn = lambda *a, **k: Fetcher(NEW),
    )
    first = delta.reuse_for_hub_download("unsloth/Model-GGUF", NAME, None, **kw)
    assert first.placed, first.reason
    assert (repo_dir / "snapshots" / COMMIT_NEW / NAME).read_bytes() == NEW
    second = delta.reuse_for_hub_download("unsloth/Model-GGUF", NAME, None, **kw)
    assert second.reason == "already cached"
    third = delta.reuse_for_hub_download("unsloth/Model-GGUF", NAME, None, **kw)
    assert third.reason == "recently up to date" and len(calls) == 2


def test_hub_entry_skips_without_older_copy_or_for_non_gguf(tmp_path):
    def boom():
        raise AssertionError("no HEAD expected")

    kw = dict(cache_dir = str(tmp_path), metadata_fn = boom)
    assert delta.reuse_for_hub_download("u/m", NAME, None, **kw).reason == "no older copy"
    assert (
        delta.reuse_for_hub_download("u/m", "model.safetensors", None, **kw).reason
        == "not applicable"
    )


def test_hub_entry_never_raises(tmp_path):
    cache_with_old(tmp_path, symlinks = True)

    def boom():
        raise OSError("network down")

    result = delta.reuse_for_hub_download(
        "unsloth/Model-GGUF", NAME, None, cache_dir = str(tmp_path), metadata_fn = boom
    )
    assert not result.placed and result.reason.startswith("error")


def test_download_wrapper_skips_force_after_a_rebuild(tmp_path, monkeypatch):
    from utils import hf_xet_fallback as fallback

    seen = {}
    monkeypatch.setattr(
        delta, "reuse_for_hub_download", lambda *a, **k: delta.DeltaResult(placed = True)
    )

    def fake_download(repo_id, filename, token, **kwargs):
        seen.update(kwargs)
        return "path"

    monkeypatch.setattr(fallback, "_shared_hf_hub_download_with_xet_fallback", fake_download)
    assert (
        fallback.hf_hub_download_with_xet_fallback(
            "unsloth/Model-GGUF", NAME, None, force_download = True, cache_dir = str(tmp_path)
        )
        == "path"
    )
    assert seen["force_download"] is False


def test_download_wrapper_keeps_force_without_a_rebuild(tmp_path, monkeypatch):
    from utils import hf_xet_fallback as fallback

    seen = {}
    monkeypatch.setattr(delta, "reuse_for_hub_download", lambda *a, **k: delta.DeltaResult())

    def fake_download(repo_id, filename, token, **kwargs):
        seen.update(kwargs)
        return "path"

    monkeypatch.setattr(fallback, "_shared_hf_hub_download_with_xet_fallback", fake_download)
    fallback.hf_hub_download_with_xet_fallback(
        "unsloth/Model-GGUF", NAME, None, force_download = True, cache_dir = str(tmp_path)
    )
    assert seen["force_download"] is True


def test_download_worker_rebuilds_pending_ggufs(tmp_path, monkeypatch):
    from hub.workers import hf_download

    repo_dir = cache_with_old(tmp_path, symlinks = True)
    monkeypatch.setattr("huggingface_hub.constants.HF_HUB_CACHE", str(tmp_path))
    monkeypatch.setattr(delta, "hub_range_fetcher", lambda *a, **k: Fetcher(NEW))
    files = [SimpleNamespace(path = NAME, size = len(NEW), sha256 = sha(NEW))]
    left = hf_download._reuse_unchanged_files("model", "unsloth/Model-GGUF", COMMIT_NEW, files, None)
    assert left == []
    assert (repo_dir / "snapshots" / COMMIT_NEW / NAME).read_bytes() == NEW
