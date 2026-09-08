# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A partial no writer can reopen is litter, but only once nothing is writing it."""

import json
import os
import threading
import time

import pytest

from hub.utils import download_registry, hf_cache_state, resumable_partials


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
    # The cache-dir iterators resolve the root through hf_cache_state, not the caller.
    monkeypatch.setattr(hf_cache_state, "hf_cache_root", lambda **_kwargs: root)
    monkeypatch.setattr(hf_cache_state, "hf_cache_roots", lambda *_a, **_k: [root])
    return blobs_dir


def _join_background_sweep():
    """The all-caches pass is threaded so it cannot delay startup; wait for it here."""
    for thread in threading.enumerate():
        if thread.name == "hf-abandoned-partial-sweep":
            thread.join(10)


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
    """1.18 is the line: before it a partial is appended to, after it a new file is written.

    Pinned with the restoration in :mod:`hub.utils.resumable_partials` unavailable, which is what
    a machine whose filesystem cannot prove ``flock`` excludes a second writer sees.
    """
    monkeypatch.setattr(resumable_partials, "can_restore_partials", lambda _c = None: False)
    monkeypatch.setattr("huggingface_hub.__version__", hf_version, raising = False)
    hf_cache_state.invalidate_partial_resumability()
    try:
        assert hf_cache_state.hf_partials_are_resumable() is resumable
    finally:
        hf_cache_state.invalidate_partial_resumability()


@pytest.mark.parametrize("hf_version", ["1.18.0", "1.23.0", "1.27.0"])
def test_restoring_the_1_17_writer_makes_partials_resumable_again(monkeypatch, hf_version):
    """Where the worker puts the append-mode writer back, a partial is worth keeping again."""
    monkeypatch.setattr(resumable_partials, "can_restore_partials", lambda _c = None: True)
    monkeypatch.setattr("huggingface_hub.__version__", hf_version, raising = False)
    hf_cache_state.invalidate_partial_resumability()
    try:
        assert hf_cache_state.hf_partials_are_resumable() is True
    finally:
        hf_cache_state.invalidate_partial_resumability()


def test_a_nonce_partial_stays_unresumable_after_the_writer_is_restored(monkeypatch):
    """Bytes already on disk under a nonce name are still litter: the restored writer opens the
    stable name, so it never finds them. Only what it writes from here is reusable."""
    monkeypatch.setattr(resumable_partials, "can_restore_partials", lambda _c = None: True)
    monkeypatch.setattr("huggingface_hub.__version__", "1.28.0", raising = False)
    hf_cache_state.invalidate_partial_resumability()
    try:
        assert hf_cache_state.partial_is_resumable(_NONCE_PARTIAL) is False
        assert hf_cache_state.partial_is_resumable(_LEGACY_PARTIAL) is True
    finally:
        hf_cache_state.invalidate_partial_resumability()


def test_a_nonce_partial_is_unresumable_even_under_a_legacy_writer(monkeypatch):
    """The nonce path is private to the process that made it; nothing reopens it by name."""
    monkeypatch.setattr(hf_cache_state, "hf_partials_are_resumable", lambda _root = None: True)

    assert hf_cache_state.partial_is_resumable(_LEGACY_PARTIAL) is True
    assert hf_cache_state.partial_is_resumable(_NONCE_PARTIAL) is False


def test_unresumable_partial_is_purged_despite_a_matching_marker(monkeypatch, blobs):
    """The marker vouches for provenance, which is worth nothing with no resumer left."""
    monkeypatch.setattr(download_registry, "partial_is_resumable", lambda _name, _root = None: False)
    _prepare()
    partial = blobs / _NONCE_PARTIAL
    partial.write_bytes(b"x" * 25)
    _abandon(partial)

    assert _prepare() == 1
    assert not partial.exists()


def test_resumable_partial_survives_a_matching_marker(monkeypatch, blobs):
    """Older hubs still append to it, so deleting it would throw away real bytes."""
    monkeypatch.setattr(download_registry, "partial_is_resumable", lambda _name, _root = None: True)
    _prepare()
    partial = blobs / _LEGACY_PARTIAL
    partial.write_bytes(b"x" * 25)
    _abandon(partial)

    # Too fresh at download start, so prepare leaves it alone.
    assert _prepare() == 0
    assert partial.exists()


def test_a_partial_still_being_written_is_left_alone(monkeypatch, blobs):
    """It may belong to a client this backend's peer registry cannot see."""
    monkeypatch.setattr(download_registry, "partial_is_resumable", lambda _name, _root = None: False)
    _prepare()
    partial = blobs / _NONCE_PARTIAL
    partial.write_bytes(b"x" * 25)

    assert _prepare() == 0
    assert partial.exists()


def test_a_mismatched_marker_still_purges_without_waiting(monkeypatch, blobs):
    """That purge stops a corrupt append, so it cannot defer to a grace period."""
    monkeypatch.setattr(download_registry, "partial_is_resumable", lambda _name, _root = None: True)
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
    monkeypatch.setattr(download_registry, "partial_is_resumable", lambda _name, _root = None: False)
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
    """``resumable`` drives a dialog offering to keep existing progress.

    The marker is written rather than stubbed: the verdict reads each cache entry's own
    marker, since one repo can own several and only the one beside a partial vouches for it.
    """
    download_registry._write_marker(blobs.parent, download_registry.TRANSPORT_HTTP)
    (blobs / _NONCE_PARTIAL).write_bytes(b"x" * 25)

    monkeypatch.setattr(download_registry, "partial_is_resumable", lambda _name, _root = None: True)
    assert download_registry.is_resumable_partial("model", "Org/Model") is True

    monkeypatch.setattr(download_registry, "partial_is_resumable", lambda _name, _root = None: False)
    assert download_registry.is_resumable_partial("model", "Org/Model") is False


def test_a_skipped_partial_is_swept_once_it_ages_out(monkeypatch, blobs):
    """The start-of-download skip is not the last word on an orphan."""
    monkeypatch.setattr(download_registry, "partial_is_resumable", lambda _name, _root = None: False)
    _prepare()
    partial = blobs / _NONCE_PARTIAL
    partial.write_bytes(b"x" * 25)

    assert _prepare() == 0
    assert partial.exists()

    # By the time that download reaches a terminal state the grace has elapsed.
    _abandon(partial)
    assert download_registry.sweep_abandoned_partials("model", "Org/Model") == 1
    assert not partial.exists()


def test_the_sweep_still_spares_a_live_writer_and_a_peer(monkeypatch, blobs):
    """A terminal state for one job says nothing about what another is writing."""
    monkeypatch.setattr(download_registry, "partial_is_resumable", lambda _name, _root = None: False)
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
    monkeypatch.setattr(download_registry, "partial_is_resumable", lambda _name, _root = None: False)
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
        "iter_destructive_repo_cache_dirs",
        lambda *_a, **_k: [blobs.parent],
    )

    monkeypatch.setattr(download_registry, "partial_is_resumable", lambda _name, _root = None: False)
    assert download_registry.existing_blob_bytes("model", "Org/Model", frozenset({_MAIN})) == 0

    monkeypatch.setattr(download_registry, "partial_is_resumable", lambda _name, _root = None: True)
    assert download_registry.existing_blob_bytes("model", "Org/Model", frozenset({_MAIN})) == 25


def test_a_finalized_blob_still_counts_against_the_disk_check(monkeypatch, blobs):
    """Only partials are in question; a finished blob is bytes nobody refetches."""
    (blobs / _MAIN).write_bytes(b"x" * 25)
    monkeypatch.setattr(
        download_registry,
        "iter_destructive_repo_cache_dirs",
        lambda *_a, **_k: [blobs.parent],
    )
    monkeypatch.setattr(download_registry, "partial_is_resumable", lambda _name, _root = None: False)

    assert download_registry.existing_blob_bytes("model", "Org/Model", frozenset({_MAIN})) == 25


def test_startup_sweep_does_not_depend_on_a_breadcrumb(monkeypatch, tmp_path, blobs):
    """finalize_worker_exit drops the breadcrumb, so the boot sweep cannot be driven off one."""
    workers = tmp_path / "workers"
    workers.mkdir()
    partial = blobs / _NONCE_PARTIAL
    partial.write_bytes(b"x" * 25)
    _abandon(partial)

    monkeypatch.setattr(download_registry.state_dir, "workers_dir", lambda: workers)
    monkeypatch.setattr(download_registry, "partial_is_resumable", lambda _name, _root = None: False)
    monkeypatch.setattr(
        download_registry, "hf_cache_roots", lambda *_a, **_k: [blobs.parent.parent]
    )

    download_registry.reap_orphan_workers()
    _join_background_sweep()

    assert not partial.exists()


def test_startup_sweep_leaves_a_resumable_partial_alone(monkeypatch, tmp_path, blobs):
    """Walking every cache at boot is not a licence to widen what gets deleted."""
    workers = tmp_path / "workers"
    workers.mkdir()
    partial = blobs / _LEGACY_PARTIAL
    partial.write_bytes(b"x" * 25)
    _abandon(partial)

    monkeypatch.setattr(download_registry.state_dir, "workers_dir", lambda: workers)
    monkeypatch.setattr(download_registry, "partial_is_resumable", lambda _name, _root = None: True)
    monkeypatch.setattr(
        download_registry, "hf_cache_roots", lambda *_a, **_k: [blobs.parent.parent]
    )

    download_registry.reap_orphan_workers()
    _join_background_sweep()

    assert partial.exists()


def test_a_reaped_job_does_not_wait_out_the_grace_on_its_own_blobs(monkeypatch, blobs):
    """Cancelling writes the partial seconds before the sweep, so waiting strands it."""
    monkeypatch.setattr(download_registry, "partial_is_resumable", lambda _name, _root = None: False)
    partial = blobs / _NONCE_PARTIAL
    partial.write_bytes(b"x" * 25)
    monkeypatch.setattr(
        download_registry,
        "iter_destructive_repo_cache_dirs",
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
    monkeypatch.setattr(download_registry, "partial_is_resumable", lambda _name, _root = None: False)
    monkeypatch.setattr(download_registry, "blob_download_lock_held", lambda *_a: True)
    partial = blobs / _NONCE_PARTIAL
    partial.write_bytes(b"x" * 25)
    _abandon(partial)
    monkeypatch.setattr(
        download_registry,
        "iter_destructive_repo_cache_dirs",
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
    monkeypatch.setattr(download_registry, "partial_is_resumable", lambda _name, _root = None: False)
    partial = blobs / _NONCE_PARTIAL
    partial.write_bytes(b"x" * 25)
    monkeypatch.setattr(
        download_registry,
        "iter_destructive_repo_cache_dirs",
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


def test_the_sweep_accepts_the_string_root_the_metadata_holds(monkeypatch, tmp_path):
    """DownloadMetadata.hub_cache is a str, and the caller hands it straight through.

    Deliberately not using the ``blobs`` fixture: patching hf_cache_root would hand the
    resolver a Path and hide the very conversion under test.
    """
    monkeypatch.setattr(download_registry, "partial_is_resumable", lambda _name, _root = None: False)
    blobs = tmp_path / "hub" / "models--Org--Model" / "blobs"
    blobs.mkdir(parents = True)
    partial = blobs / _NONCE_PARTIAL
    partial.write_bytes(b"x" * 25)
    _abandon(partial)

    # A Path-only signature raised AttributeError here and the caller's broad except swallowed it, so
    # the terminal sweep silently did nothing for every real download.
    swept = download_registry.sweep_abandoned_partials(
        "model",
        "Org/Model",
        root = str(blobs.parent.parent),
    )

    assert swept == 1
    assert not partial.exists()


def test_a_job_owning_its_whole_repo_needs_no_hash_list(monkeypatch, blobs):
    """A download with no variant resolves no blob hashes, and claim() gives it the repo."""
    monkeypatch.setattr(download_registry, "partial_is_resumable", lambda _name, _root = None: False)
    partial = blobs / _NONCE_PARTIAL
    partial.write_bytes(b"x" * 25)
    monkeypatch.setattr(
        download_registry,
        "iter_destructive_repo_cache_dirs",
        lambda *_a, **_k: [blobs.parent],
    )

    assert download_registry.sweep_abandoned_partials("model", "Org/Model") == 0
    assert (
        download_registry.sweep_abandoned_partials("model", "Org/Model", owns_all_blobs = True) == 1
    )
    assert not partial.exists()


def test_the_boot_sweep_runs_after_the_orphan_is_killed(monkeypatch, tmp_path, blobs):
    """Sweeping first reads the doomed worker's still-held lock and spares its partial."""
    workers = tmp_path / "workers"
    workers.mkdir()
    (workers / "job.json").write_text(
        json.dumps(
            {
                "pid": 4242,
                "repo_type": "model",
                "repo_id": "Org/Model",
                "variant": None,
                "transport": "http",
                "hub_cache": str(blobs.parent.parent),
            }
        ),
        encoding = "utf-8",
    )
    partial = blobs / _NONCE_PARTIAL
    partial.write_bytes(b"x" * 25)

    order = []
    locked = {"held": True}
    monkeypatch.setattr(download_registry.state_dir, "workers_dir", lambda: workers)
    monkeypatch.setattr(download_registry, "partial_is_resumable", lambda _name, _root = None: False)
    monkeypatch.setattr(download_registry, "_process_alive", lambda _pid: True)
    monkeypatch.setattr(download_registry, "_is_our_worker", lambda *_a: True)
    monkeypatch.setattr(download_registry, "_settle_orphaned_download", lambda *_a, **_k: None)
    monkeypatch.setattr(
        download_registry,
        "hf_cache_roots",
        lambda *_a, **_k: [blobs.parent.parent],
    )
    monkeypatch.setattr(
        download_registry,
        "iter_destructive_repo_cache_dirs",
        lambda *_a, **_k: [blobs.parent],
    )

    def _kill(_pid):
        order.append("kill")
        locked["held"] = False
        return True

    monkeypatch.setattr(download_registry, "_kill_orphan", _kill)
    monkeypatch.setattr(
        download_registry,
        "blob_download_lock_held",
        lambda *_a: order.append("sweep") or locked["held"],
    )

    download_registry.reap_orphan_workers()
    _join_background_sweep()

    assert order[0] == "kill"
    assert not partial.exists()


def test_a_companion_the_dead_worker_was_writing_is_owned_too(monkeypatch, blobs):
    """A shared mmproj lives in progress_blob_hashes, never in the main blob_hashes set."""
    monkeypatch.setattr(download_registry, "partial_is_resumable", lambda _name, _root = None: False)
    companion = blobs / f"{_PEER}.feedface{hf_cache_state.INCOMPLETE_SUFFIX}"
    companion.write_bytes(b"x" * 25)
    monkeypatch.setattr(
        download_registry,
        "iter_destructive_repo_cache_dirs",
        lambda *_a, **_k: [blobs.parent],
    )

    # Ownership limited to the variant's own quant leaves the companion waiting out the grace.
    assert (
        download_registry.sweep_abandoned_partials(
            "model",
            "Org/Model",
            owned_blob_hashes = frozenset({_MAIN}),
        )
        == 0
    )
    assert companion.exists()

    assert (
        download_registry.sweep_abandoned_partials(
            "model",
            "Org/Model",
            owned_blob_hashes = frozenset({_MAIN, _PEER}),
        )
        == 1
    )
    assert not companion.exists()


def test_the_reaper_waits_for_the_worker_to_actually_die(monkeypatch):
    """SIGKILL only schedules the death; the lock outlives the signal by a moment."""
    alive = {"n": 3}

    def _still_alive(_pid):
        alive["n"] -= 1
        return alive["n"] > 0

    monkeypatch.setattr(download_registry.os, "kill", lambda *_a: None)
    monkeypatch.setattr(download_registry, "_process_alive", _still_alive)

    download_registry._kill_orphan(4242)

    assert alive["n"] == 0


def test_a_worker_that_will_not_die_keeps_its_breadcrumb_and_its_partial(
    monkeypatch, tmp_path, blobs
):
    """An unreapable worker is still running, so nothing about it is ours to claim."""
    workers = tmp_path / "workers"
    workers.mkdir()
    crumb = workers / "job.json"
    crumb.write_text(
        json.dumps(
            {
                "pid": 4242,
                "repo_type": "model",
                "repo_id": "Org/Model",
                "variant": None,
                "transport": "http",
                "hub_cache": str(blobs.parent.parent),
            }
        ),
        encoding = "utf-8",
    )
    partial = blobs / _NONCE_PARTIAL
    partial.write_bytes(b"x" * 25)

    monkeypatch.setattr(download_registry.state_dir, "workers_dir", lambda: workers)
    monkeypatch.setattr(download_registry, "partial_is_resumable", lambda _name, _root = None: False)
    monkeypatch.setattr(download_registry, "_process_alive", lambda _pid: True)
    monkeypatch.setattr(download_registry, "_is_our_worker", lambda *_a: True)
    monkeypatch.setattr(download_registry, "_kill_orphan", lambda _pid: False)
    monkeypatch.setattr(download_registry, "hf_cache_roots", lambda *_a, **_k: [tmp_path / "none"])

    download_registry.reap_orphan_workers()
    _join_background_sweep()

    assert partial.exists()
    assert crumb.exists()


def test_a_locked_peer_partial_still_counts_against_the_disk_check(monkeypatch, blobs):
    """A sibling variant is finishing the shared companion, so we need no room for it."""
    monkeypatch.setattr(download_registry, "partial_is_resumable", lambda _name, _root = None: False)
    (blobs / _NONCE_PARTIAL).write_bytes(b"x" * 25)
    monkeypatch.setattr(
        download_registry,
        "iter_active_repo_cache_dirs",
        lambda *_a, **_k: [blobs.parent],
    )

    monkeypatch.setattr(download_registry, "blob_download_lock_held", lambda *_a: False)
    assert download_registry.existing_blob_bytes("model", "Org/Model", frozenset({_MAIN})) == 0

    monkeypatch.setattr(download_registry, "blob_download_lock_held", lambda *_a: True)
    assert download_registry.existing_blob_bytes("model", "Org/Model", frozenset({_MAIN})) == 25


def test_the_sweep_will_not_cross_a_case_variant_directory(monkeypatch, tmp_path):
    """owns_all_blobs plus a case-insensitive collision could otherwise reach a neighbour."""
    monkeypatch.setattr(download_registry, "partial_is_resumable", lambda _name, _root = None: False)
    root = tmp_path / "hub"
    mine = root / "models--Org--Model" / "blobs"
    other = root / "models--org--model" / "blobs"
    mine.mkdir(parents = True)
    other.mkdir(parents = True)
    for blobs_dir in (mine, other):
        partial = blobs_dir / _NONCE_PARTIAL
        partial.write_bytes(b"x" * 25)
        _abandon(partial)

    download_registry.sweep_abandoned_partials(
        "model",
        "Org/Model",
        owns_all_blobs = True,
        root = str(root),
    )

    assert not (mine / _NONCE_PARTIAL).exists()
    assert (other / _NONCE_PARTIAL).exists()


def test_ownership_is_recovered_from_the_manifest_when_hashes_never_resolved(monkeypatch):
    """A variant job whose API-side pre-resolution failed carries an EMPTY hash set."""
    from types import SimpleNamespace

    from hub.services import download_lifecycle
    from hub.utils import download_manifest

    metadata = SimpleNamespace(
        variant = "Q4_K_M",
        hub_cache = None,
        progress_blob_hashes = frozenset(),
    )
    manifest = download_manifest.Manifest(
        repo_type = "model",
        repo_id = "Org/Model",
        variant = "Q4_K_M",
        started_at = "",
        expected_files = (download_manifest.ExpectedFile(path = "m.gguf", size = 5, sha256 = _MAIN),),
    )
    monkeypatch.setattr(download_manifest, "read_manifest", lambda *_a, **_k: manifest)

    owned, owns_all = download_lifecycle._sweep_ownership(
        metadata, frozenset(), frozenset(), "model", "Org/Model"
    )

    assert owns_all is False
    assert owned == frozenset({_MAIN})


def test_a_filesystem_without_flock_does_not_escape_the_probe(monkeypatch, tmp_path):
    """NotImplementedError used to travel out and fail the download on every retry."""
    import filelock

    entry = tmp_path / "models--Org--Model"
    lock_path = tmp_path / ".locks" / "models--Org--Model" / f"{_MAIN}.lock"
    lock_path.parent.mkdir(parents = True)

    class _NoFlock:
        def __init__(self, *_a, **_k):
            pass

        def __enter__(self):
            raise NotImplementedError(
                "FileSystem does not appear to support flock; use SoftFileLock instead"
            )

        def __exit__(self, *_a):
            return False

    monkeypatch.setattr(filelock, "FileLock", _NoFlock)

    # No lock file: nobody has locked this blob, whatever the filesystem supports.
    assert hf_cache_state.blob_download_lock_held(entry, _MAIN) is False

    # With one, the answer is "held" rather than an exception, which is also what a SoftFileLock would
    # say, since its file IS the lock.
    lock_path.touch()
    assert hf_cache_state.blob_download_lock_held(entry, _MAIN) is True


def test_an_unprobeable_lock_reads_as_held(monkeypatch, tmp_path):
    """Ownership can skip the staleness gate, so a wrong 'free' deletes a live writer's file."""
    import filelock

    entry = tmp_path / "models--Org--Model"
    lock_path = tmp_path / ".locks" / "models--Org--Model" / f"{_MAIN}.lock"
    lock_path.parent.mkdir(parents = True)
    lock_path.touch()

    class _Broken:
        def __init__(self, *_a, **_k):
            pass

        def __enter__(self):
            raise RuntimeError("something unforeseen")

        def __exit__(self, *_a):
            return False

    monkeypatch.setattr(filelock, "FileLock", _Broken)

    assert hf_cache_state.blob_download_lock_held(entry, _MAIN) is True


def test_unreadable_breadcrumbs_do_not_cancel_the_cache_sweep(monkeypatch, tmp_path, blobs):
    """The workers dir and the HF caches are separate trees; one failing is not the other."""
    workers = tmp_path / "workers"
    workers.mkdir()
    partial = blobs / _NONCE_PARTIAL
    partial.write_bytes(b"x" * 25)
    _abandon(partial)

    class _UnreadableDir:
        def iterdir(self):
            raise OSError("permission denied")

    monkeypatch.setattr(download_registry.state_dir, "workers_dir", lambda: _UnreadableDir())
    monkeypatch.setattr(download_registry, "partial_is_resumable", lambda _name, _root = None: False)
    monkeypatch.setattr(
        download_registry, "hf_cache_roots", lambda *_a, **_k: [blobs.parent.parent]
    )

    download_registry.reap_orphan_workers()
    _join_background_sweep()

    assert not partial.exists()


def test_an_owned_partial_that_is_still_growing_is_spared(monkeypatch, blobs):
    """Ownership proves OUR writer died, never that no other process shares the cache."""
    monkeypatch.setattr(download_registry, "partial_is_resumable", lambda _name, _root = None: False)
    monkeypatch.setattr(download_registry, "blob_download_lock_held", lambda *_a: False)
    monkeypatch.setattr(download_registry, "_STILLNESS_PROBE_SECONDS", 0.05)
    monkeypatch.setattr(
        download_registry,
        "iter_destructive_repo_cache_dirs",
        lambda *_a, **_k: [blobs.parent],
    )
    partial = blobs / _NONCE_PARTIAL
    partial.write_bytes(b"x" * 25)

    real_sleep = time.sleep

    def _write_while_we_watch(_seconds):
        real_sleep(_seconds)
        with partial.open("ab") as handle:
            handle.write(b"y" * 10)

    monkeypatch.setattr(download_registry.time, "sleep", _write_while_we_watch)

    swept = download_registry.sweep_abandoned_partials(
        "model",
        "Org/Model",
        owns_all_blobs = True,
    )

    assert swept == 0
    assert partial.exists()


def test_an_owned_partial_that_never_moves_is_swept_without_the_full_grace(monkeypatch, blobs):
    """The corpse of a cancelled download must not outlive the retry that follows it."""
    monkeypatch.setattr(download_registry, "partial_is_resumable", lambda _name, _root = None: False)
    monkeypatch.setattr(download_registry, "_STILLNESS_PROBE_SECONDS", 0.05)
    monkeypatch.setattr(
        download_registry,
        "iter_destructive_repo_cache_dirs",
        lambda *_a, **_k: [blobs.parent],
    )
    partial = blobs / _NONCE_PARTIAL
    partial.write_bytes(b"x" * 25)

    swept = download_registry.sweep_abandoned_partials(
        "model",
        "Org/Model",
        owns_all_blobs = True,
    )

    assert swept == 1
    assert not partial.exists()


def test_a_breadcrumb_whose_worker_already_exited_is_claimed(monkeypatch, tmp_path, blobs):
    """A container restart leaves the crumb behind and the pid long gone."""
    workers = tmp_path / "workers"
    workers.mkdir()
    (workers / "job.json").write_text(
        json.dumps(
            {
                "pid": 4242,
                "repo_type": "model",
                "repo_id": "Org/Model",
                "variant": None,
                "transport": "http",
                "hub_cache": str(blobs.parent.parent),
            }
        ),
        encoding = "utf-8",
    )
    partial = blobs / _NONCE_PARTIAL
    partial.write_bytes(b"x" * 25)

    monkeypatch.setattr(download_registry.state_dir, "workers_dir", lambda: workers)
    monkeypatch.setattr(download_registry, "partial_is_resumable", lambda _name, _root = None: False)
    monkeypatch.setattr(download_registry, "_process_alive", lambda _pid: False)
    monkeypatch.setattr(download_registry, "_STILLNESS_PROBE_SECONDS", 0.05)
    monkeypatch.setattr(download_registry, "_settle_orphaned_download", lambda *_a, **_k: None)
    monkeypatch.setattr(download_registry, "hf_cache_roots", lambda *_a, **_k: [tmp_path / "none"])
    monkeypatch.setattr(
        download_registry,
        "iter_destructive_repo_cache_dirs",
        lambda *_a, **_k: [blobs.parent],
    )

    download_registry.reap_orphan_workers()
    _join_background_sweep()

    assert not partial.exists()
