# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Exercise reclamation with real workers and a local HTTP Hub, including offline resolution."""

import hashlib
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from .test_hf_download_pinned_revision import _FakeHub, _NEW, _OLD, _REPO, _WORKER, _repo_dir, _run


def _worker_env(hub, tmp_path):
    env = dict(os.environ)
    env.pop("HF_TOKEN", None)
    env.update(
        HF_ENDPOINT = hub.endpoint,
        HF_HOME = str(tmp_path / "hf"),
        HF_HUB_CACHE = str(tmp_path / "hf" / "hub"),
        UNSLOTH_STUDIO_HOME = str(tmp_path / "studio"),
        HF_HUB_DISABLE_XET = "1",
        HF_HUB_ENABLE_HF_TRANSFER = "0",
        HF_HUB_DISABLE_TELEMETRY = "1",
        HF_HUB_DISABLE_IMPLICIT_TOKEN = "1",
        HF_HUB_OFFLINE = "0",
        CUDA_VISIBLE_DEVICES = "",
    )
    return env


_PAUSED_WORKER = """
import runpy, sys, time
from pathlib import Path
import huggingface_hub as hf
worker, phase, root = sys.argv[1:]
root = Path(root)
original = hf.snapshot_download
def pause():
    (root / 'ready').touch()
    deadline = time.monotonic() + 30
    while not (root / 'release').exists():
        if time.monotonic() > deadline:
            raise RuntimeError('test did not release transfer')
        time.sleep(0.01)
def transfer(**kwargs):
    if phase == 'before_transfer':
        pause()
    result = original(**kwargs)
    if phase == 'after_transfer':
        pause()
    return result
hf.snapshot_download = transfer
sys.argv = [worker, '--repo-id', 'fake/repo', '--variant', 'Q4_K_M', '--transport', 'http']
runpy.run_path(worker, run_name='__main__')
"""


@pytest.mark.parametrize("phase", ["before_transfer", "after_transfer"])
def test_gguf_reclaim_preserves_publication_and_active_peer(tmp_path, phase):
    hub = _FakeHub()
    child = None
    name = "model-Q4_K_M.gguf"
    try:
        # Same filename and size, different hashes: size verification cannot detect a mix-up.
        for commit, byte in ((_OLD, b"a"), (_NEW, b"b")):
            hub.commits[commit][name] = b"GGUF" + byte * 1024
        repo = _repo_dir(tmp_path)
        stale_commit = "c" * 40
        stale_data = b"GGUF" + b"c" * 1024
        stale_blob = repo / "blobs" / hashlib.sha256(stale_data).hexdigest()
        stale_blob.parent.mkdir(parents = True)
        stale_blob.write_bytes(stale_data)
        stale = repo / "snapshots" / stale_commit / name
        stale.parent.mkdir(parents = True)
        stale.symlink_to(stale_blob)
        ref = repo / "refs" / "main"
        ref.parent.mkdir()
        ref.write_text(stale_commit)

        child = subprocess.Popen(
            [sys.executable, "-c", _PAUSED_WORKER, str(_WORKER), phase, str(tmp_path)],
            env = _worker_env(hub, tmp_path),
            stdout = subprocess.PIPE,
            stderr = subprocess.PIPE,
            text = True,
        )
        deadline = time.monotonic() + 20
        while not (tmp_path / "ready").exists():
            assert child.poll() is None, child.communicate()
            assert time.monotonic() < deadline, "older worker did not reach the transfer barrier"
            time.sleep(0.01)
        assert ref.read_text() == stale_commit
        hub.head = _NEW
        newer = _run(hub, tmp_path, "--variant", "Q4_K_M")
        assert newer.returncode == 0, newer.stderr
        assert ref.read_text() == _NEW
        # Reclamation must still do useful work while protecting the active peer.
        assert not stale.exists() and not stale_blob.exists(), newer.stderr
        if phase == "after_transfer":
            assert (repo / "snapshots" / _OLD / name).read_bytes() == hub.commits[_OLD][name]
        (tmp_path / "release").touch()
        out, err = child.communicate(timeout = 30)
        assert child.returncode == 0, out + err
        assert ref.read_text() == _NEW

        # The real offline resolver must still be able to load the published winner.
        offline = subprocess.run(
            [
                sys.executable,
                "-c",
                "from huggingface_hub import hf_hub_download; from pathlib import Path; "
                "import sys; p = hf_hub_download('fake/repo', sys.argv[2], "
                "cache_dir=sys.argv[1], local_files_only=True); "
                "assert Path(p).read_bytes() == b'GGUF' + b'b' * 1024",
                str(repo.parent),
                name,
            ],
            env = _worker_env(hub, tmp_path),
            capture_output = True,
            text = True,
            timeout = 20,
        )
        assert offline.returncode == 0, offline.stderr
    finally:
        (tmp_path / "release").touch()
        if child is not None and child.poll() is None:
            child.kill()
            child.communicate()
        hub.close()
