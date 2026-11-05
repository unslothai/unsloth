# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A manifest dates copied snapshot files while refs/main still names the last verified success."""

import json
import subprocess
import sys

import pytest

from .test_hf_download_pinned_revision import _FakeHub, _FULL, _NEW, _OLD, _WORKER, _repo_dir
from .test_hf_download_reclaim_race import _worker_env


_COPY_WORKER = """
import json, runpy, sys
from pathlib import Path
from types import SimpleNamespace
import huggingface_hub as hf
from huggingface_hub import file_download
worker, kind, root = sys.argv[1:]
root = Path(root)
sys.path.insert(0, str(Path(worker).parents[2]))
from hub.services import snapshot_progress as sp
from hub.utils import download_manifest as dm
repo = root / 'hf' / 'hub' / 'models--fake--repo'
variant = {'full': None, 'scoped': 'test-scope', 'gguf': 'Q4_K_M'}[kind]
file_download.are_symlinks_supported = lambda *a, **k: False
sp.preferred_repo_cache_dirs = lambda *a, **k: [repo]
original = hf.snapshot_download
observations = []
def poll():
    manifest = dm.read_manifest('model', 'fake/repo', variant, hub_cache=repo.parent)
    assert manifest is not None
    hashes = frozenset(f.sha256 for f in manifest.expected_files if f.sha256)
    total = sum(f.size for f in manifest.expected_files)
    reading = sp.compute_snapshot_progress(
        repo_type='model', repo_id='fake/repo', job_key='fake/repo',
        expected_bytes=total, hf_token=None, variant=variant,
        variant_file_matcher=lambda name, **kw: name in {f.path for f in manifest.expected_files},
        registry=SimpleNamespace(get_job=lambda key: SimpleNamespace(state='running')),
        metadata_resolver=lambda *args: (total, hashes),
    )
    observations.append(dict(
        reading=reading, commit=manifest.commit_hash, attested=manifest.metadata_derived,
        ref=(repo / 'refs' / 'main').read_text(),
        old_is_stale=sp._snapshot_is_stale_copy(repo / 'snapshots' / ('a' * 40), manifest),
        new_exists=any((repo / 'snapshots' / ('b' * 40) / f.path).exists()
                       for f in manifest.expected_files),
    ))
def transfer(**kwargs):
    poll()  # Metadata and manifest exist; no B bytes have been transferred.
    result = original(**kwargs)
    poll()  # B now exists as regular files, but publication must still be deferred.
    return result
hf.snapshot_download = transfer
extra = []
if kind == 'gguf':
    extra = ['--variant', 'Q4_K_M']
elif kind == 'scoped':
    files = root / 'files.json'
    files.write_text(json.dumps(['config.json', 'model.safetensors']))
    extra = ['--variant', 'test-scope', '--files-json', str(files)]
sys.argv = [worker, '--repo-id', 'fake/repo', '--transport', 'http', *extra]
try:
    runpy.run_path(worker, run_name='__main__')
except SystemExit as exc:
    if exc.code:
        raise
(root / 'progress.json').write_text(json.dumps(observations))
"""


@pytest.mark.parametrize("kind", ["full", "scoped", "gguf"])
def test_copy_layout_progress_uses_attested_download_commit(tmp_path, kind):
    hub = _FakeHub()
    try:
        # Every old file has the expected new size and different content/hash.
        for name, data in hub.commits[_OLD].items():
            hub.commits[_NEW][name] = hub.commits[_NEW][name][: len(data)]
            assert len(hub.commits[_NEW][name]) == len(data)
            assert hub.commits[_NEW][name] != data
        repo = _repo_dir(tmp_path)
        old = repo / "snapshots" / _OLD
        old.mkdir(parents = True)
        for name, data in hub.commits[_OLD].items():
            (old / name).write_bytes(data)
        ref = repo / "refs" / "main"
        ref.parent.mkdir()
        ref.write_text(_OLD)
        hub.head = _NEW
        result = subprocess.run(
            [sys.executable, "-c", _COPY_WORKER, str(_WORKER), kind, str(tmp_path)],
            env = _worker_env(hub, tmp_path),
            capture_output = True,
            text = True,
            timeout = 60,
        )
        assert result.returncode == 0, result.stderr
        before, transferred = json.loads((tmp_path / "progress.json").read_text())
        assert before["ref"] == transferred["ref"] == _OLD
        assert before["new_exists"] is False
        assert transferred["new_exists"] is True
        # Full copy-layout progress already has no snapshot-byte fallback. All
        # three workers must nevertheless attest the revision that dates copies.
        assert before["reading"]["complete_on_disk"] is False, before
        assert before["reading"]["completed_bytes"] == 0, before
        assert before["reading"]["downloaded_bytes"] == 0, before
        assert before["reading"]["progress"] == 0, before
        assert before["commit"] == transferred["commit"] == _NEW, before
        assert before["attested"] and transferred["attested"], before
        assert before["old_is_stale"] and transferred["old_is_stale"], before
        if kind != "full":
            assert transferred["reading"]["complete_on_disk"] is True, transferred
            assert transferred["reading"]["progress"] == 1.0, transferred
        assert ref.read_text() == _NEW
        names = ["model-Q4_K_M.gguf"] if kind == "gguf" else _FULL
        for name in names:
            target = repo / "snapshots" / _NEW / name
            assert not target.is_symlink()
            assert target.read_bytes() == hub.commits[_NEW][name]
    finally:
        hub.close()
