# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A partial no writer can reopen is litter, but only once nothing is writing it."""

import json
import os
import time

import pytest

from hub.utils import download_registry, hf_cache_state


_MAIN = "a" * 64
_PEER = "b" * 64
_LEGACY_PARTIAL = f"{_MAIN}{hf_cache_state.INCOMPLETE_SUFFIX}"
_NONCE_PARTIAL = f"{_MAIN}.deadbeef{hf_cache_state.INCOMPLETE_SUFFIX}"


@pytest.fixture
def blobs(monkeypatch, tmp_path):
    root = tmp_path / "hub"
    blobs_dir = root / "models--Org--Model" / "blobs"
    blobs_dir.mkdir(parents = True)
    monkeypatch.setenv("HF_HUB_CACHE", str(root))
    monkeypatch.setattr(download_registry, "hf_cache_root", lambda **_kwargs: root)
    # iter_active_repo_cache_dirs resolves the root through hf_cache_state, not the caller.
    monkeypatch.setattr(hf_cache_state, "hf_cache_root", lambda **_kwargs: root)
    return blobs_dir


def _abandon(path):
    """Backdate a partial past the grace, as an abandoned one would be."""
    old = time.time() - download_registry.ABANDONED_PARTIAL_SECONDS - 60
    os.utime(path, (old, old))
    return path


def _prepare(**kwargs):
    return download_registry.prepare_cache_for_transport(
        "model",
        "Org/Model",
        download_registry.TRANSPORT_HTTP,
        "Q4_K_M",
        only_blob_hashes = frozenset({_MAIN}),
        **kwargs,
    )


@pytest.mark.parametrize(
    "hf_version, resumable",
    [
        ("0.36.2", True),
        ("1.17.0", True),
        ("1.18.0", False),
        ("1.23.0", False),
        ("1.27.0", False),
        ("2.0.0.dev0", False),
        ("not-a-version", True),
    ],
)
def test_writer_resumability_tracks_the_installed_version(monkeypatch, hf_version, resumable):
    """1.18 is the line: before it a partial is appended to, after it a new file is written."""
    monkeypatch.setattr("huggingface_hub.__version__", hf_version, raising = False)
    hf_cache_state.hf_partials_are_resumable.cache_clear()
    try:
        assert hf_cache_state.hf_partials_are_resumable() is resumable
    finally:
        hf_cache_state.hf_partials_are_resumable.cache_clear()


def test_a_nonce_partial_is_unresumable_even_under_a_legacy_writer(monkeypatch):
    """The nonce path is private to the process that made it; nothing reopens it by name."""
    monkeypatch.setattr(hf_cache_state, "hf_partials_are_resumable", lambda: True)

    assert hf_cache_state.partial_is_resumable(_LEGACY_PARTIAL) is True
    assert hf_cache_state.partial_is_resumable(_NONCE_PARTIAL) is False


def test_unresumable_partial_is_purged_despite_a_matching_marker(monkeypatch, blobs):
    """The marker vouches for provenance, which is worth nothing with no resumer left."""
    monkeypatch.setattr(download_registry, "partial_is_resumable", lambda _name: False)
    _prepare()  # writes the http marker
    partial = blobs / _NONCE_PARTIAL
    partial.write_bytes(b"x" * 25)
    _abandon(partial)

    assert _prepare() == 1
    assert not partial.exists()


def test_resumable_partial_survives_a_matching_marker(monkeypatch, blobs):
    """Older hubs still append to it, so deleting it would throw away real bytes."""
    monkeypatch.setattr(download_registry, "partial_is_resumable", lambda _name: True)
    _prepare()
    partial = blobs / _LEGACY_PARTIAL
    partial.write_bytes(b"x" * 25)
    _abandon(partial)

    assert _prepare() == 0
    assert partial.exists()


def test_a_partial_still_being_written_is_left_alone(monkeypatch, blobs):
    """It may belong to a client this backend's peer registry cannot see."""
    monkeypatch.setattr(download_registry, "partial_is_resumable", lambda _name: False)
    _prepare()
    partial = blobs / _NONCE_PARTIAL
    partial.write_bytes(b"x" * 25)  # mtime is now, as a live writer's would be

    assert _prepare() == 0
    assert partial.exists()


def test_a_mismatched_marker_still_purges_without_waiting(monkeypatch, blobs):
    """That purge stops a corrupt append, so it cannot defer to a grace period."""
    monkeypatch.setattr(download_registry, "partial_is_resumable", lambda _name: True)
    download_registry.prepare_cache_for_transport(
        "model",
        "Org/Model",
        download_registry.TRANSPORT_XET,
        "Q4_K_M",
        only_blob_hashes = frozenset({_MAIN}),
    )
    partial = blobs / _LEGACY_PARTIAL
    partial.write_bytes(b"x" * 25)

    assert _prepare() == 1
    assert not partial.exists()


def test_a_peer_being_written_is_still_protected(monkeypatch, blobs):
    """Unresumable is not a licence to delete a blob another download is writing now."""
    monkeypatch.setattr(download_registry, "partial_is_resumable", lambda _name: False)
    _prepare()
    mine = blobs / _NONCE_PARTIAL
    mine.write_bytes(b"x" * 25)
    _abandon(mine)
    peer = blobs / f"{_PEER}.feedface{hf_cache_state.INCOMPLETE_SUFFIX}"
    peer.write_bytes(b"x" * 25)
    _abandon(peer)

    purged = download_registry.prepare_cache_for_transport(
        "model",
        "Org/Model",
        download_registry.TRANSPORT_HTTP,
        "Q4_K_M",
        only_blob_hashes = frozenset({_MAIN, _PEER}),
        protected_blob_hashes = frozenset({_PEER}),
    )

    assert purged == 1
    assert not mine.exists()
    assert peer.exists()


def test_transport_status_does_not_promise_a_resume_it_cannot_keep(monkeypatch, blobs):
    """``resumable`` drives a dialog offering to keep existing progress."""
    monkeypatch.setattr(download_registry, "read_active_transport_marker", lambda *_a, **_k: "http")
    (blobs / _NONCE_PARTIAL).write_bytes(b"x" * 25)

    monkeypatch.setattr(download_registry, "partial_is_resumable", lambda _name: True)
    assert download_registry.is_resumable_partial("model", "Org/Model") is True

    monkeypatch.setattr(download_registry, "partial_is_resumable", lambda _name: False)
    assert download_registry.is_resumable_partial("model", "Org/Model") is False


def test_a_skipped_partial_is_swept_once_it_ages_out(monkeypatch, blobs):
    """The start-of-download skip is not the last word on an orphan."""
    monkeypatch.setattr(download_registry, "partial_is_resumable", lambda _name: False)
    _prepare()
    partial = blobs / _NONCE_PARTIAL
    partial.write_bytes(b"x" * 25)

    # Too fresh at download start, so prepare leaves it alone.
    assert _prepare() == 0
    assert partial.exists()

    # By the time that download reaches a terminal state the grace has elapsed.
    _abandon(partial)
    assert download_registry.sweep_abandoned_partials("model", "Org/Model") == 1
    assert not partial.exists()


def test_the_sweep_still_spares_a_live_writer_and_a_peer(monkeypatch, blobs):
    """A terminal state for one job says nothing about what another is writing."""
    monkeypatch.setattr(download_registry, "partial_is_resumable", lambda _name: False)
    live = blobs / _NONCE_PARTIAL
    live.write_bytes(b"x" * 25)
    peer = blobs / f"{_PEER}.feedface{hf_cache_state.INCOMPLETE_SUFFIX}"
    peer.write_bytes(b"x" * 25)
    _abandon(peer)

    swept = download_registry.sweep_abandoned_partials(
        "model",
        "Org/Model",
        protected_blob_hashes = frozenset({_PEER}),
    )

    assert swept == 0
    assert live.exists()
    assert peer.exists()


def test_a_locked_blob_is_spared_however_stale_it_looks(monkeypatch, blobs):
    """A writer stalled past the grace still holds the lock, and still owns the file."""
    monkeypatch.setattr(download_registry, "partial_is_resumable", lambda _name: False)
    _prepare()
    partial = blobs / _NONCE_PARTIAL
    partial.write_bytes(b"x" * 25)
    _abandon(partial)

    monkeypatch.setattr(download_registry, "blob_download_lock_held", lambda *_a: True)
    assert _prepare() == 0
    assert partial.exists()

    monkeypatch.setattr(download_registry, "blob_download_lock_held", lambda *_a: False)
    assert _prepare() == 1
    assert not partial.exists()


def test_the_lock_probe_reads_the_layout_hf_writes(tmp_path):
    """<hub cache>/.locks/<repo dir>/<etag>.lock, and no lock file means nobody is writing."""
    from filelock import FileLock

    entry = tmp_path / "models--Org--Model"
    lock_path = tmp_path / ".locks" / "models--Org--Model" / f"{_MAIN}.lock"
    lock_path.parent.mkdir(parents = True)

    assert hf_cache_state.blob_download_lock_held(entry, _MAIN) is False

    lock_path.touch()
    assert hf_cache_state.blob_download_lock_held(entry, _MAIN) is False

    with FileLock(str(lock_path), timeout = 0):
        assert hf_cache_state.blob_download_lock_held(entry, _MAIN) is True


def test_unresumable_bytes_are_not_credited_against_the_disk_check(monkeypatch, blobs):
    """_preflight_disk_space subtracts this, so crediting a refetch can approve a full disk."""
    (blobs / _NONCE_PARTIAL).write_bytes(b"x" * 25)
    monkeypatch.setattr(
        download_registry,
        "iter_active_repo_cache_dirs",
        lambda *_a, **_k: [blobs.parent],
    )

    monkeypatch.setattr(download_registry, "partial_is_resumable", lambda _name: False)
    assert download_registry.existing_blob_bytes("model", "Org/Model", frozenset({_MAIN})) == 0

    monkeypatch.setattr(download_registry, "partial_is_resumable", lambda _name: True)
    assert download_registry.existing_blob_bytes("model", "Org/Model", frozenset({_MAIN})) == 25


def test_a_finalized_blob_still_counts_against_the_disk_check(monkeypatch, blobs):
    """Only partials are in question; a finished blob is bytes nobody refetches."""
    (blobs / _MAIN).write_bytes(b"x" * 25)
    monkeypatch.setattr(
        download_registry,
        "iter_active_repo_cache_dirs",
        lambda *_a, **_k: [blobs.parent],
    )
    monkeypatch.setattr(download_registry, "partial_is_resumable", lambda _name: False)

    assert download_registry.existing_blob_bytes("model", "Org/Model", frozenset({_MAIN})) == 25


def test_startup_sweep_does_not_depend_on_a_breadcrumb(monkeypatch, tmp_path, blobs):
    """finalize_worker_exit drops the breadcrumb, so the boot sweep cannot be driven off one."""
    workers = tmp_path / "workers"
    workers.mkdir()  # deliberately empty, as it is once drop_process has run
    partial = blobs / _NONCE_PARTIAL
    partial.write_bytes(b"x" * 25)
    _abandon(partial)

    monkeypatch.setattr(download_registry.state_dir, "workers_dir", lambda: workers)
    monkeypatch.setattr(download_registry, "partial_is_resumable", lambda _name: False)
    monkeypatch.setattr(
        download_registry, "hf_cache_roots", lambda *_a, **_k: [blobs.parent.parent]
    )

    download_registry.reap_orphan_workers()

    assert not partial.exists()


def test_startup_sweep_leaves_a_resumable_partial_alone(monkeypatch, tmp_path, blobs):
    """Walking every cache at boot is not a licence to widen what gets deleted."""
    workers = tmp_path / "workers"
    workers.mkdir()
    partial = blobs / _LEGACY_PARTIAL
    partial.write_bytes(b"x" * 25)
    _abandon(partial)

    monkeypatch.setattr(download_registry.state_dir, "workers_dir", lambda: workers)
    monkeypatch.setattr(download_registry, "partial_is_resumable", lambda _name: True)
    monkeypatch.setattr(
        download_registry, "hf_cache_roots", lambda *_a, **_k: [blobs.parent.parent]
    )

    download_registry.reap_orphan_workers()

    assert partial.exists()


def test_a_reaped_job_does_not_wait_out_the_grace_on_its_own_blobs(monkeypatch, blobs):
    """Cancelling writes the partial seconds before the sweep, so waiting strands it."""
    monkeypatch.setattr(download_registry, "partial_is_resumable", lambda _name: False)
    partial = blobs / _NONCE_PARTIAL
    partial.write_bytes(b"x" * 25)  # freshly written, as a just-cancelled download's would be
    monkeypatch.setattr(
        download_registry,
        "iter_active_repo_cache_dirs",
        lambda *_a, **_k: [blobs.parent],
    )

    # Without the ownership claim it has to wait, which is what stranded it for the session.
    assert download_registry.sweep_abandoned_partials("model", "Org/Model") == 0
    assert partial.exists()

    assert (
        download_registry.sweep_abandoned_partials(
            "model",
            "Org/Model",
            owned_blob_hashes = frozenset({_MAIN}),
        )
        == 1
    )
    assert not partial.exists()


def test_ownership_never_overrides_the_lock(monkeypatch, blobs):
    """hf locks before it creates the temp file, so a locked blob has a live writer."""
    monkeypatch.setattr(download_registry, "partial_is_resumable", lambda _name: False)
    monkeypatch.setattr(download_registry, "blob_download_lock_held", lambda *_a: True)
    partial = blobs / _NONCE_PARTIAL
    partial.write_bytes(b"x" * 25)
    _abandon(partial)
    monkeypatch.setattr(
        download_registry,
        "iter_active_repo_cache_dirs",
        lambda *_a, **_k: [blobs.parent],
    )

    swept = download_registry.sweep_abandoned_partials(
        "model",
        "Org/Model",
        owned_blob_hashes = frozenset({_MAIN}),
    )

    assert swept == 0
    assert partial.exists()


def test_ownership_never_overrides_peer_protection(monkeypatch, blobs):
    """A shared companion a sibling variant is writing stays out of reach."""
    monkeypatch.setattr(download_registry, "partial_is_resumable", lambda _name: False)
    partial = blobs / _NONCE_PARTIAL
    partial.write_bytes(b"x" * 25)
    monkeypatch.setattr(
        download_registry,
        "iter_active_repo_cache_dirs",
        lambda *_a, **_k: [blobs.parent],
    )

    swept = download_registry.sweep_abandoned_partials(
        "model",
        "Org/Model",
        protected_blob_hashes = frozenset({_MAIN}),
        owned_blob_hashes = frozenset({_MAIN}),
    )

    assert swept == 0
    assert partial.exists()
