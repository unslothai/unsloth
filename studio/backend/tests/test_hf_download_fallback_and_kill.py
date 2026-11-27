# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Metadata fallback and hard cancellation must use the publication protocol."""

import hashlib
import json
import os
import signal
import subprocess
import sys
import time

import pytest

from .test_hf_download_pinned_revision import (
    _BACKEND,
    _FakeHub,
    _NEW,
    _OLD,
    _WORKER,
    _repo_dir,
    _run,
)
from .test_hf_download_reclaim_race import _PAUSED_WORKER, _worker_env


def _wait_for_transfer(child, tmp_path):
    deadline = time.monotonic() + 20
    while not (tmp_path / "ready").exists():
        assert child.poll() is None, child.communicate()
        assert time.monotonic() < deadline, "worker did not reach the transfer barrier"
        time.sleep(0.01)


_FALLBACK_WORKER = r"""
import hashlib, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from hub.workers import hf_download as w
from hub.utils import download_manifest as dm
repo, commit, name = 'fake/repo', 'b' * 40, 'model-Q8_0.gguf'
data = b'GGUF' + b'q8' * 1024
assert dm.write_manifest('model', repo, 'Q8_0',
    [dm.ExpectedFile(name, len(data), hashlib.sha256(data).hexdigest())],
    'http', commit_hash=commit, metadata_derived=True)
original = w._model_info_with_retry
calls = 0
def fail_once(*args, **kwargs):
    global calls
    calls += 1
    if calls == 1:
        raise OSError('transient metadata failure')
    return original(*args, **kwargs)
w._model_info_with_retry = fail_once
w._download_gguf_variant(repo, 'Q8_0', None, 'http')
from huggingface_hub import hf_hub_download
path = hf_hub_download(repo, name, local_files_only=True)
assert Path(path).read_bytes() == data
"""


def test_metadata_fallback_completion_survives_older_pinned_worker(tmp_path):
    """Real GGUF workers: A starts, fallback B completes, then A completes."""
    hub = _FakeHub()
    child = None
    try:
        hub.commits[_NEW]["model-Q8_0.gguf"] = b"GGUF" + b"q8" * 1024
        child = subprocess.Popen(
            [sys.executable, "-c", _PAUSED_WORKER, str(_WORKER), "before_transfer", str(tmp_path)],
            env = _worker_env(hub, tmp_path),
            stdout = subprocess.PIPE,
            stderr = subprocess.PIPE,
            text = True,
        )
        _wait_for_transfer(child, tmp_path)
        hub.head = _NEW
        resumed = subprocess.run(
            [sys.executable, "-c", _FALLBACK_WORKER, str(_BACKEND)],
            env = _worker_env(hub, tmp_path),
            capture_output = True,
            text = True,
            timeout = 40,
        )
        assert resumed.returncode == 0, resumed.stdout + resumed.stderr
        repo = _repo_dir(tmp_path)
        assert (repo / "refs" / "main").read_text() == _NEW
        (tmp_path / "release").touch()
        out, err = child.communicate(timeout = 40)
        assert child.returncode == 0, out + err
        offline = subprocess.run(
            [
                sys.executable,
                "-c",
                "from huggingface_hub import hf_hub_download; from pathlib import Path; "
                "p = hf_hub_download('fake/repo', 'model-Q8_0.gguf', local_files_only=True); "
                "assert Path(p).read_bytes() == b'GGUF' + b'q8' * 1024",
            ],
            env = _worker_env(hub, tmp_path),
            capture_output = True,
            text = True,
            timeout = 20,
        )
        assert offline.returncode == 0, offline.stdout + offline.stderr
        assert (repo / "refs" / "main").read_text() == _NEW
    finally:
        (tmp_path / "release").touch()
        if child is not None and child.poll() is None:
            child.kill()
            child.communicate(timeout = 20)
        hub.close()


@pytest.mark.parametrize("commit", [None, "main", "A" * 40])
def test_unresolved_revision_fails_before_transfer(monkeypatch, tmp_path, commit):
    from huggingface_hub import constants
    from hub.workers import hf_download as w

    monkeypatch.setattr(constants, "HF_HUB_CACHE", str(tmp_path))

    def transfer(**kwargs):
        pytest.fail("unresolved revision must not run an unpinned transfer")

    with pytest.raises(OSError, match = "Cannot resolve an immutable revision"):
        w._pinned_snapshot_download(transfer, commit, repo_id = "fake/repo")
    assert not (tmp_path / "models--fake--repo" / "refs").exists()


@pytest.mark.parametrize("active", [False, True])
def test_legacy_journal_requires_owners_only_for_active_attempts(tmp_path, active):
    from hub.workers import hf_download as w

    journal = tmp_path / "studio-main.json"
    journal.write_text(
        json.dumps(
            {
                "next_sequence": 2,
                "published_sequence": 0,
                "published_commit": _OLD,
                "attempts": [[1, _NEW]] if active else [],
            }
        )
    )
    if active:
        # Old workers have no provable lifetime; do not guess they are dead.
        with pytest.raises(ValueError, match = "Invalid refs/main publication state"):
            w._load_main_ref_state(journal, _OLD)
    else:
        state = w._load_main_ref_state(journal, _OLD)
        assert state.published_commit == _OLD
        assert not state.attempts and not state.owners


def test_soft_owner_lock_fails_before_transfer(monkeypatch, tmp_path):
    import filelock
    from huggingface_hub import constants
    from hub.workers import hf_download as w

    monkeypatch.setattr(constants, "HF_HUB_CACHE", str(tmp_path))
    monkeypatch.setattr(filelock, "FileLock", filelock.SoftFileLock)

    def transfer(**kwargs):
        pytest.fail("a soft lock cannot prove owner death")

    with pytest.raises(OSError, match = "requires native file locks"):
        w._pinned_snapshot_download(transfer, _NEW, repo_id = "fake/repo")


def test_owner_dying_during_transfer_is_pruned_before_publish(monkeypatch, tmp_path):
    from huggingface_hub import constants
    from hub.workers import hf_download as w

    monkeypatch.setattr(constants, "HF_HUB_CACHE", str(tmp_path))
    repo = tmp_path / "models--fake--repo"
    lock_dir = tmp_path / ".locks" / repo.name
    published = []

    def transfer(**kwargs):
        # Simulate another attempt's death after this transfer has started.
        owner = "c" * 32
        lock = w._main_ref_owner_lock(lock_dir, owner)
        try:
            w._update_main_ref(repo, commit = _OLD, owner = owner)
        finally:
            lock.release()
        target = repo / "snapshots" / _NEW
        target.mkdir(parents = True)
        return str(target)

    w._pinned_snapshot_download(transfer, _NEW, repo_id = "fake/repo", on_publish = published.append)
    assert len(published) == 1
    assert published[0].published_commit == _NEW
    assert not published[0].attempts and not published[0].owners


@pytest.mark.skipif(os.name == "nt", reason = "requires POSIX SIGKILL")
def test_sigkill_attempt_is_pruned_by_a_fresh_worker_before_reclaim(tmp_path):
    """No parent cleanup: the new process recovers solely from the on-disk ledger."""
    hub = _FakeHub()
    child = None
    name = "model-Q4_K_M.gguf"
    try:
        child = subprocess.Popen(
            [sys.executable, "-c", _PAUSED_WORKER, str(_WORKER), "after_transfer", str(tmp_path)],
            env = _worker_env(hub, tmp_path),
            stdout = subprocess.PIPE,
            stderr = subprocess.PIPE,
            text = True,
        )
        _wait_for_transfer(child, tmp_path)
        repo = _repo_dir(tmp_path)
        stale = repo / "snapshots" / _OLD / name
        blob = repo / "blobs" / hashlib.sha256(hub.commits[_OLD][name]).hexdigest()
        assert stale.is_file() and blob.is_file()
        journal = repo.parent / ".locks" / repo.name / "studio-main.json"
        assert json.loads(journal.read_text())["attempts"]
        os.kill(child.pid, signal.SIGKILL)
        child.communicate(timeout = 20)
        assert child.returncode == -signal.SIGKILL
        assert json.loads(journal.read_text())["attempts"]  # SIGKILL bypassed finish
        hub.head = _NEW
        newer = _run(hub, tmp_path, "--variant", "Q4_K_M")
        assert newer.returncode == 0, newer.stderr
        assert (repo / "refs" / "main").read_text() == _NEW
        assert not stale.exists(), "dead attempt still protects obsolete snapshot"
        assert not blob.exists(), "dead attempt still protects obsolete blob"
        assert not json.loads(journal.read_text())["attempts"]
        assert (repo / "snapshots" / _NEW / name).read_bytes() == hub.commits[_NEW][name]
    finally:
        if child is not None and child.poll() is None:
            child.kill()
            child.communicate(timeout = 20)
        hub.close()
