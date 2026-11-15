# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""/api/models/local keeps each repo's partial verdict across requests, keyed on the stats of the
directories and files the verdict reads. The listing must equal an uncached one after every
download-shaped mutation, and a repeat request must not rerun the partial checks."""

from __future__ import annotations

import asyncio
import errno
import hashlib
import json
import os
import shutil
import sys
import threading
import time
from pathlib import Path

import pytest

import routes.models as models_route
from hub.utils import download_manifest, inventory_scan, partial_verdict_cache

pytestmark = pytest.mark.skipif(
    not sys.platform.startswith("linux"), reason = "the verdict cache is Linux only"
)

_REAL_TRUSTED_DEVICES = partial_verdict_cache._trusted_devices
_CONFIG = json.dumps({"architectures": ["LlamaForCausalLM"], "model_type": "llama"})


@pytest.fixture(autouse = True)
def _fresh_cache(monkeypatch):
    partial_verdict_cache.invalidate()
    # Fixture files are seconds old at most; production refuses to vouch for those.
    monkeypatch.setattr(partial_verdict_cache, "_RACY_NS", 0)
    # Whatever filesystem the test tmp dir sits on.
    monkeypatch.setattr(partial_verdict_cache, "_trusted_devices", lambda devices: True)
    yield
    partial_verdict_cache.invalidate()


def _settle() -> None:
    """Let the next write land on a later timestamp than the last stored signature."""
    time.sleep(0.03)


def _blob(repo: Path, body: bytes) -> str:
    digest = hashlib.sha256(body).hexdigest()
    (repo / "blobs").mkdir(exist_ok = True)
    (repo / "blobs" / digest).write_bytes(body)
    return digest


def _link(repo: Path, rev: str, name: str, digest: str) -> None:
    target = repo / "snapshots" / rev / name
    target.parent.mkdir(parents = True, exist_ok = True)
    depth = len(Path(name).parts)
    target.symlink_to(Path(*([".."] * (depth + 1)), "blobs", digest))


def _hf_repo(hub: Path, repo_id: str, rev: str, files: dict[str, bytes]) -> Path:
    repo = hub / f"models--{repo_id.replace('/', '--')}"
    (repo / "refs").mkdir(parents = True, exist_ok = True)
    (repo / "refs" / "main").write_text(rev, encoding = "utf-8")
    (repo / "snapshots" / rev).mkdir(parents = True, exist_ok = True)
    for name, body in files.items():
        _link(repo, rev, name, _blob(repo, body))
    return repo


_REV1 = "a" * 40
_REV2 = "b" * 40


@pytest.fixture
def hub(monkeypatch, tmp_path):
    active = tmp_path / "hub"
    active.mkdir()
    monkeypatch.setattr(models_route, "_resolve_hf_cache_dir", lambda: active)
    monkeypatch.setattr("utils.paths.legacy_hf_cache_dir", lambda: tmp_path / "legacy")
    monkeypatch.setattr("utils.paths.hf_default_cache_dir", lambda: tmp_path / "default")
    monkeypatch.setattr("utils.paths.lmstudio_model_dirs", lambda: [])
    monkeypatch.setattr("utils.paths.hermes_model_dirs", lambda: [])
    monkeypatch.setattr("utils.paths.ollama_model_dirs", lambda: [])
    monkeypatch.setattr("utils.paths.omlx_model_dirs", lambda: [])
    monkeypatch.setattr("utils.hf_cache_settings.known_hf_hub_caches", lambda: [active])
    monkeypatch.setattr("storage.studio_db.list_scan_folders", lambda: [])
    _hf_repo(
        active,
        "org/chat",
        _REV1,
        {"config.json": _CONFIG.encode(), "model.safetensors": b"\0" * 64},
    )
    _hf_repo(active, "org/tiny-GGUF", _REV1, {"tiny-Q8_0.gguf": b"GGUF" + b"\0" * 60})
    _hf_repo(
        active,
        "org/other",
        _REV1,
        {"config.json": _CONFIG.encode(), "model.safetensors": b"\1" * 64},
    )
    return active


def _scan(models_root: Path) -> list[dict]:
    rows = asyncio.run(models_route._shared_compat_local_inventory_scan(models_root))
    return [row.model_dump(mode = "json") for row in rows]


def _uncached_scan(models_root: Path, monkeypatch) -> list[dict]:
    with monkeypatch.context() as m:
        m.setattr(partial_verdict_cache, "enabled", lambda: False)
        return _scan(models_root)


def _partial(rows: list[dict], model_id: str) -> bool:
    return next(row for row in rows if row["model_id"] == model_id)["partial"]


def _check(models_root: Path, monkeypatch) -> list[dict]:
    """A cached listing (filled by a first scan, then served) equals an uncached one."""
    first = _scan(models_root)
    second = _scan(models_root)
    reference = _uncached_scan(models_root, monkeypatch)
    assert first == second == reference
    return second


def test_listing_matches_an_uncached_listing_through_a_download(hub, tmp_path, monkeypatch):
    models = tmp_path / "models"
    repo = hub / "models--org--chat"
    flips: list[bool] = []

    def step(expect: bool) -> None:
        _settle()
        rows = _check(models, monkeypatch)
        assert _partial(rows, "org/chat") is expect
        assert _partial(rows, "org/other") is False
        flips.append(expect)

    step(False)
    # A download starts: its partial blob appears.
    body = b"\2" * 128
    digest = hashlib.sha256(body).hexdigest()
    partial = repo / "blobs" / f"{digest}.incomplete"
    partial.write_bytes(body[:16])
    step(True)
    # It grows in place (no directory entry changes).
    with partial.open("ab") as handle:
        handle.write(body[16:64])
    step(True)
    # It completes: renamed into place and linked from the snapshot.
    with partial.open("ab") as handle:
        handle.write(body[64:])
    os.replace(partial, repo / "blobs" / digest)
    _link(repo, _REV1, "extra.safetensors", digest)
    step(False)
    # Cancel marker written, then cleared.
    assert download_manifest.write_cancel_marker("model", "org/chat", None, hub_cache = hub)
    step(True)
    download_manifest.clear_cancel_marker("model", "org/chat", None, hub_cache = hub)
    step(False)
    # A link repointed at a blob that is not there.
    link = repo / "snapshots" / _REV1 / "model.safetensors"
    link.unlink()
    link.symlink_to(Path("..", "..", "blobs", "0" * 64))
    step(True)
    link.unlink()
    _link(repo, _REV1, "model.safetensors", _blob(repo, b"\0" * 64))
    step(False)
    # A repo manifest whose shard is short on disk, then satisfied.
    expected = [
        download_manifest.ExpectedFile(path = "config.json", size = len(_CONFIG)),
        download_manifest.ExpectedFile(path = "model-2.safetensors", size = 32),
    ]
    assert download_manifest.write_manifest(
        "model", "org/chat", None, expected, "http", hub_cache = hub
    )
    step(True)
    _link(repo, _REV1, "model-2.safetensors", _blob(repo, b"\3" * 32))
    step(False)
    assert download_manifest.delete_manifest("model", "org/chat", None, hub_cache = hub) is not False
    step(False)
    # refs/main rewritten in place (same size) to a commit with no snapshot yet.
    (repo / "refs" / "main").write_text(_REV2, encoding = "utf-8")
    _check(models, monkeypatch)
    # The new snapshot lands.
    _link(repo, _REV2, "config.json", _blob(repo, _CONFIG.encode()))
    _link(repo, _REV2, "model.safetensors", _blob(repo, b"\0" * 64))
    step(False)
    # The repo is removed.
    _settle()
    shutil.rmtree(repo)
    rows = _check(models, monkeypatch)
    assert "org/chat" not in {row["model_id"] for row in rows}
    assert True in flips and False in flips


def test_gguf_variant_partial_then_complete(hub, tmp_path, monkeypatch):
    models = tmp_path / "models"
    repo = hub / "models--org--tiny-GGUF"
    _settle()
    assert _partial(_check(models, monkeypatch), "org/tiny-GGUF") is False
    # Studio starts a second quant and is cancelled before its file lands.
    expected = [download_manifest.ExpectedFile(path = "tiny-Q4_K_M.gguf", size = 40)]
    assert download_manifest.write_manifest(
        "model", "org/tiny-GGUF", "Q4_K_M", expected, "xet", hub_cache = hub
    )
    assert download_manifest.write_cancel_marker("model", "org/tiny-GGUF", "Q4_K_M", hub_cache = hub)
    _settle()
    rows = _check(models, monkeypatch)
    # Q8_0 is whole, so the row stays chattable.
    assert _partial(rows, "org/tiny-GGUF") is False
    # The whole quant goes away: only the broken one is left.
    (repo / "snapshots" / _REV1 / "tiny-Q8_0.gguf").unlink()
    _settle()
    assert _partial(_check(models, monkeypatch), "org/tiny-GGUF") is True
    # The quant completes.
    download_manifest.clear_cancel_marker("model", "org/tiny-GGUF", "Q4_K_M", hub_cache = hub)
    _link(repo, _REV1, "tiny-Q4_K_M.gguf", _blob(repo, b"GGUF" + b"\4" * 36))
    _settle()
    assert _partial(_check(models, monkeypatch), "org/tiny-GGUF") is False


def test_repeat_scan_does_not_rerun_the_partial_checks(hub, tmp_path, monkeypatch):
    calls: list[str] = []
    real_snapshot = inventory_scan.is_snapshot_partial
    real_gguf = inventory_scan.is_gguf_repo_partial

    def snapshot(*args, **kwargs):
        calls.append("snapshot")
        return real_snapshot(*args, **kwargs)

    def gguf(*args, **kwargs):
        calls.append("gguf")
        return real_gguf(*args, **kwargs)

    monkeypatch.setattr(inventory_scan, "is_snapshot_partial", snapshot)
    monkeypatch.setattr(inventory_scan, "is_gguf_repo_partial", gguf)
    _settle()
    first = _scan(tmp_path / "models")
    assert calls.count("snapshot") == 3
    calls.clear()
    assert _scan(tmp_path / "models") == first
    assert calls == []
    # One repo changes: only it is judged again.
    _settle()
    (hub / "models--org--chat" / "blobs" / "f.incomplete").write_bytes(b"x")
    third = _scan(tmp_path / "models")
    assert calls.count("snapshot") == 1
    assert _partial(third, "org/chat") is True


def test_concurrent_listings_and_writers_settle_on_the_uncached_answer(hub, tmp_path, monkeypatch):
    models = tmp_path / "models"
    repo = hub / "models--org--chat"
    stop = threading.Event()
    errors: list[BaseException] = []

    def writer() -> None:
        partial = repo / "blobs" / "c0ffee.incomplete"
        index = 0
        while not stop.is_set():
            try:
                if index % 2:
                    partial.write_bytes(b"x")
                    download_manifest.write_cancel_marker("model", "org/other", None, hub_cache = hub)
                else:
                    partial.unlink(missing_ok = True)
                    download_manifest.clear_cancel_marker("model", "org/other", None, hub_cache = hub)
            except BaseException as exc:  # noqa: BLE001 - surfaced below
                errors.append(exc)
                return
            index += 1
            time.sleep(0.002)

    def lister() -> None:
        try:
            for _ in range(6):
                _scan(models)
        except BaseException as exc:  # noqa: BLE001 - surfaced below
            errors.append(exc)

    threads = [threading.Thread(target = writer)] + [
        threading.Thread(target = lister) for _ in range(4)
    ]
    for thread in threads:
        thread.start()
    for thread in threads[1:]:
        thread.join()
    stop.set()
    threads[0].join()
    assert not errors
    _settle()
    assert _scan(models) == _uncached_scan(models, monkeypatch)
    for state in (True, False):
        _settle()
        if state:
            (repo / "blobs" / "c0ffee.incomplete").write_bytes(b"x")
        else:
            (repo / "blobs" / "c0ffee.incomplete").unlink(missing_ok = True)
        rows = _check(models, monkeypatch)
        assert _partial(rows, "org/chat") is state


def test_fresh_timestamps_are_not_cached(hub, tmp_path, monkeypatch):
    monkeypatch.setattr(partial_verdict_cache, "_RACY_NS", 3_000_000_000)
    _scan(tmp_path / "models")
    assert len(partial_verdict_cache._entries) == 0


def test_a_failed_stat_is_never_cached_or_served(hub, tmp_path, monkeypatch):
    _settle()
    _scan(tmp_path / "models")
    assert len(partial_verdict_cache._entries) == 3
    loop = tmp_path / "loop"
    loop.symlink_to(loop)
    with pytest.raises(partial_verdict_cache._Unstable):
        partial_verdict_cache._stat(os.fspath(loop))
    assert partial_verdict_cache._stat(os.fspath(tmp_path / "absent")) is None
    blobs = os.fspath(hub / "models--org--chat" / "blobs")
    real_stat = partial_verdict_cache._stat

    def failing_stat(path):
        if path == blobs:
            # What _stat makes of an EIO / EACCES / EMFILE: anything but "absent".
            raise partial_verdict_cache._Unstable from OSError(errno.EIO, "I/O error", path)
        return real_stat(path)

    monkeypatch.setattr(partial_verdict_cache, "_stat", failing_stat)
    calls: list[str] = []
    real_snapshot = inventory_scan.is_snapshot_partial

    def snapshot(*args, **kwargs):
        calls.append(args[1])
        return real_snapshot(*args, **kwargs)

    monkeypatch.setattr(inventory_scan, "is_snapshot_partial", snapshot)
    partial_verdict_cache.invalidate()
    _scan(tmp_path / "models")
    # The unreadable repo is judged and not stored; the others are stored.
    assert "org/chat" in calls
    assert len(partial_verdict_cache._entries) == 2
    calls.clear()
    _scan(tmp_path / "models")
    assert calls == ["org/chat"]


def test_invalidation_drops_every_verdict(hub, tmp_path):
    _settle()
    _scan(tmp_path / "models")
    assert partial_verdict_cache._entries
    inventory_scan.invalidate_hf_cache_scans()
    assert not partial_verdict_cache._entries
    _scan(tmp_path / "models")
    from core.inference.local_model_resolver import invalidate_index

    invalidate_index()
    assert not partial_verdict_cache._entries


def test_a_linked_directory_in_a_snapshot_is_not_cached(hub, tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (hub / "models--org--chat" / "snapshots" / _REV1 / "extra").symlink_to(
        outside, target_is_directory = True
    )
    _settle()
    _scan(tmp_path / "models")
    keys = {key[1] for key in partial_verdict_cache._entries}
    assert os.fspath(hub / "models--org--chat") not in keys
    assert os.fspath(hub / "models--org--other") in keys


def test_a_link_to_a_directory_in_blobs_is_not_cached(hub, tmp_path):
    repo = hub / "models--org--other"
    (repo / "blobs" / "d").mkdir()
    _link(repo, _REV1, "nested", "d")
    _settle()
    _scan(tmp_path / "models")
    keys = {key[1] for key in partial_verdict_cache._entries}
    assert os.fspath(repo) not in keys
    assert os.fspath(hub / "models--org--chat") in keys


def test_a_linked_snapshots_dir_tracks_the_blobs_its_links_reach(hub, tmp_path, monkeypatch):
    """Relative snapshot links resolve from where snapshots/ really is, not from the repo dir."""
    models = tmp_path / "models"
    elsewhere = _hf_repo(
        tmp_path / "elsewhere",
        "org/moved",
        _REV1,
        {"config.json": _CONFIG.encode(), "model.safetensors": b"\5" * 64},
    )
    repo = hub / "models--org--moved"
    (repo / "refs").mkdir(parents = True)
    (repo / "refs" / "main").write_text(_REV1, encoding = "utf-8")
    (repo / "blobs").mkdir()
    (repo / "snapshots").symlink_to(elsewhere / "snapshots", target_is_directory = True)
    _settle()
    rows = _check(models, monkeypatch)
    assert _partial(rows, "org/moved") is False
    # The blob the links really reach goes away; nothing under the repo dir itself changes.
    _settle()
    (elsewhere / "blobs" / hashlib.sha256(b"\5" * 64).hexdigest()).unlink()
    rows = _check(models, monkeypatch)
    assert _partial(rows, "org/moved") is True


def _share_blobs(hub: Path, repo: Path) -> list[Path]:
    """huggingface_hub's shared store: each repo blob becomes a relative link to <hub>/blobs/<xx>/<hash>."""
    stored = []
    for blob in sorted((repo / "blobs").iterdir()):
        target = hub / "blobs" / blob.name[:2] / blob.name
        target.parent.mkdir(parents = True, exist_ok = True)
        os.replace(blob, target)
        blob.symlink_to(Path("..", "..", "blobs", blob.name[:2], blob.name))
        stored.append(target)
    return stored


def test_shared_store_blobs_are_cached_and_tracked(hub, tmp_path, monkeypatch):
    models = tmp_path / "models"
    stored = _share_blobs(hub, hub / "models--org--chat")
    _settle()
    rows = _check(models, monkeypatch)
    assert _partial(rows, "org/chat") is False
    keys = {key[1] for key in partial_verdict_cache._entries}
    assert os.fspath(hub / "models--org--chat") in keys
    # Another repo's delete sweeps a store file: the link chain now dangles.
    _settle()
    weight = next(p for p in stored if p.name == hashlib.sha256(b"\0" * 64).hexdigest())
    weight.unlink()
    rows = _check(models, monkeypatch)
    assert _partial(rows, "org/chat") is True
    # Published again in place.
    _settle()
    weight.write_bytes(b"\0" * 64)
    rows = _check(models, monkeypatch)
    assert _partial(rows, "org/chat") is False


def test_shared_store_blobs_resolve_from_the_real_repo_dir(hub, tmp_path, monkeypatch):
    """A repo dir linked in from another disk: its blob links reach that disk's store, not the hub's."""
    models = tmp_path / "models"
    disk = tmp_path / "disk2"
    disk.mkdir()
    real_repo = disk / "models--org--chat"
    os.replace(hub / "models--org--chat", real_repo)
    (hub / "models--org--chat").symlink_to(real_repo, target_is_directory = True)
    stored = _share_blobs(disk, real_repo)
    # A same-named decoy in the hub's own store, which the links do not reach.
    for path in stored:
        decoy = hub / "blobs" / path.parent.name / path.name
        decoy.parent.mkdir(parents = True, exist_ok = True)
        decoy.write_bytes(path.read_bytes())
    _settle()
    rows = _check(models, monkeypatch)
    assert _partial(rows, "org/chat") is False
    _settle()
    next(p for p in stored if p.name == hashlib.sha256(b"\0" * 64).hexdigest()).unlink()
    rows = _check(models, monkeypatch)
    assert _partial(rows, "org/chat") is True


def test_cache_is_bounded(hub, tmp_path, monkeypatch):
    monkeypatch.setattr(partial_verdict_cache, "_MAX_ENTRIES", 2)
    _settle()
    _scan(tmp_path / "models")
    assert len(partial_verdict_cache._entries) == 2


def test_only_trusted_filesystems_are_cached(hub, tmp_path, monkeypatch):
    monkeypatch.setattr(partial_verdict_cache, "_trusted_devices", _REAL_TRUSTED_DEVICES)
    real = partial_verdict_cache._mount_fs_types
    dev = os.stat(hub).st_dev
    key = (os.major(dev), os.minor(dev))
    for fs_type, cached in (
        ("vfat", False),
        ("fuse.rclone", False),
        ("nfs4", False),
        ("ext4", True),
    ):
        partial_verdict_cache.invalidate()

        def types(fs_type = fs_type):
            found = real()
            found[key] = {fs_type}
            return found

        with monkeypatch.context() as m:
            m.setattr(partial_verdict_cache, "_mount_fs_types", types)
            _settle()
            _scan(tmp_path / "models")
        assert bool(partial_verdict_cache._entries) is cached, fs_type
