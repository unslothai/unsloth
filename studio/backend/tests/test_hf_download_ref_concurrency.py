# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A later publication or completion wins over an older worker's ref changes."""

import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from .test_hf_download_pinned_revision import _NEW, _OLD, _REPO, _BACKEND


_CHILD = """
import sys, time
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from hub.workers import hf_download as w
root, mode = Path(sys.argv[2]), sys.argv[3]
repo = root / 'models--fake--repo'
def transfer(**kwargs):
    (root / 'ready').touch()
    deadline = time.monotonic() + 15
    while not (root / 'release').exists():
        if time.monotonic() > deadline:
            raise RuntimeError('peer did not release child')
        time.sleep(0.01)
    if mode == 'cancel':
        raise KeyboardInterrupt
    target = repo / 'snapshots' / ('a' * 40)
    target.mkdir(parents=True, exist_ok=True)
    (target / 'model-Q4_K_M.gguf').write_bytes(b'GGUF-old')
    return str(target)
try:
    w._pinned_snapshot_download(transfer, ('b' if mode == 'cancel' else 'a') * 40,
                                repo_id='fake/repo')
except KeyboardInterrupt:
    pass
"""


@pytest.mark.parametrize("mode", ["older_success", "cancel"])
def test_peer_completion_survives_an_older_process(monkeypatch, tmp_path, mode):
    from hub.workers import hf_download as w
    from huggingface_hub import constants, snapshot_download

    monkeypatch.setattr(constants, "HF_HUB_CACHE", str(tmp_path))
    repo = tmp_path / "models--fake--repo"
    ref = repo / "refs" / "main"
    ref.parent.mkdir(parents = True)
    ref.write_text(_OLD)
    target = repo / "snapshots" / _NEW
    target.mkdir(parents = True)
    # A no-op successful peer is still a completion. Snapshot state alone cannot detect it.
    (target / "model-Q8_0.gguf").write_bytes(b"GGUF-new")
    env = dict(os.environ, HF_HUB_CACHE = str(tmp_path), CUDA_VISIBLE_DEVICES = "")
    child = subprocess.Popen(
        [sys.executable, "-c", _CHILD, str(_BACKEND), str(tmp_path), mode],
        env = env,
        stdout = subprocess.PIPE,
        stderr = subprocess.PIPE,
        text = True,
    )
    try:
        deadline = time.monotonic() + 10
        while not (tmp_path / "ready").exists():
            assert child.poll() is None, child.communicate()
            assert time.monotonic() < deadline, "child did not start its transfer"
            time.sleep(0.01)
        w._pinned_snapshot_download(lambda **kw: str(target), _NEW, repo_id = _REPO)
        assert ref.read_text() == _NEW
        (tmp_path / "release").touch()
        out, err = child.communicate(timeout = 20)
        assert child.returncode == 0, out + err
        resolved = Path(snapshot_download(_REPO, cache_dir = str(tmp_path), local_files_only = True))
        assert resolved.name == _NEW
        assert (resolved / "model-Q8_0.gguf").read_bytes() == b"GGUF-new"
    finally:
        if child.poll() is None:
            child.kill()
            child.communicate()


def test_cancel_final_read_cannot_race_a_successful_peer(monkeypatch, tmp_path):
    from hub.workers import hf_download as w
    from huggingface_hub import constants

    monkeypatch.setattr(constants, "HF_HUB_CACHE", str(tmp_path))
    repo = tmp_path / "models--fake--repo"
    ref = repo / "refs" / "main"
    ref.parent.mkdir(parents = True)
    ref.write_text(_OLD)
    at_read, peer_started, peer_done = (threading.Event() for _ in range(3))
    original_read = w._read_ref
    cancelling = False
    paused = False
    failures = []

    def read(path):
        nonlocal paused
        captured = original_read(path)
        if (
            path == ref
            and threading.current_thread().name == "cancel"
            and cancelling
            and not paused
        ):
            paused = True
            at_read.set()
            assert peer_started.wait(5)
            peer_done.wait(0.5)  # A locked peer must wait until this critical section ends.
        return captured

    monkeypatch.setattr(w, "_read_ref", read)

    def transfer(**kw):
        nonlocal cancelling
        cancelling = True
        raise KeyboardInterrupt

    def cancel():
        try:
            w._pinned_snapshot_download(transfer, _NEW, repo_id = _REPO)
        except KeyboardInterrupt:
            pass
        except BaseException as exc:
            failures.append(exc)

    def peer_transfer(**kw):
        target = repo / "snapshots" / _NEW
        target.mkdir(parents = True)
        (target / "model-Q8_0.gguf").write_bytes(b"GGUF-new")
        return str(target)

    def peer():
        peer_started.set()
        try:
            w._pinned_snapshot_download(peer_transfer, _NEW, repo_id = _REPO)
        except BaseException as exc:
            failures.append(exc)
        finally:
            peer_done.set()

    a = threading.Thread(target = cancel, name = "cancel")
    b = threading.Thread(target = peer, name = "peer")
    a.start()
    try:
        assert at_read.wait(5)
        b.start()
    finally:
        peer_started.set()
        a.join(10)
        if b.ident is not None:
            b.join(10)
    assert not a.is_alive() and not b.is_alive()
    assert not failures, failures
    assert ref.read_text() == _NEW
