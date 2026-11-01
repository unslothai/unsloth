# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Offline full/scoped retries must verify before becoming publication winners."""

import subprocess
import sys

import pytest

from .test_hf_download_pinned_revision import _BACKEND, _FakeHub, _NEW
from .test_hf_download_reclaim_race import _worker_env


_CHILD = r"""
import hashlib, json, os, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import huggingface_hub as hf
from hub.workers import hf_download as w
from hub.utils import download_manifest as dm
from hub.utils.snapshot_reuse import repo_cache_dir

kind, scenario = sys.argv[2:]
repo_id, old, new = 'fake/repo', 'a' * 40, 'b' * 40
files = ['config.json', 'model.safetensors']
scope = 'test-scope' if kind == 'scoped' else None
repo = repo_cache_dir('model', repo_id)
ref = repo / 'refs' / 'main'
journal = repo.parent / '.locks' / repo.name / 'studio-main.json'
original = hf.snapshot_download

def download():
    if scope is None:
        w._download_snapshot(repo_id, None, 'http')
    else:
        w._download_scoped_snapshot(repo_id, scope, files, None, 'http')

def unavailable(*args, **kwargs):
    raise OSError('test metadata unavailable')

def cancel_after_transfer(**kwargs):
    original(**kwargs)
    raise KeyboardInterrupt('cancelled after requested files landed')

def offline_transfer(**kwargs):
    transfers.append(kwargs['revision'])
    return original(**kwargs, local_files_only=True)

if scenario in ('resume', 'peer'):
    if scenario == 'peer':
        # Publish A, then leave an attested manifest for a cancelled B download.
        original(repo_id=repo_id, revision=old, allow_patterns=files, token=False)
        ref.parent.mkdir(parents=True, exist_ok=True)
        ref.write_text(old)
    hf.snapshot_download = cancel_after_transfer
    try:
        download()
    except KeyboardInterrupt:
        pass
    else:
        raise AssertionError('first worker was not cancelled')
    assert w._read_ref(ref) == (old if scenario == 'peer' else None)
    saved = dm.read_manifest('model', repo_id, scope)
    assert saved.metadata_derived and saved.commit_hash == new, saved
    assert {item.path for item in saved.expected_files} == set(files)
    # A real immutable local-only Hub lookup succeeds despite refs/main being absent/A.
    cached = original(repo_id=repo_id, revision=new, allow_patterns=files,
                      token=False, local_files_only=True)
    assert dm.verify_against_disk(saved, Path(cached)).ok
    assert dm.write_cancel_marker('model', repo_id, scope, 'http')
    w._model_info_with_retry = unavailable
    transfers = []
    hf.snapshot_download = offline_transfer
    if scenario == 'resume':
        download()
        assert transfers == [new], transfers
        assert ref.read_text() == new
        assert dm.verify_against_disk(saved, repo / 'snapshots' / new).ok
    else:
        # B is now incomplete. A later fallback must fail without outranking an
        # earlier B peer whose actual transfer completes after that failure.
        (repo / 'snapshots' / new / files[1]).unlink()
        failed = []
        def peer_transfer(**kwargs):
            try:
                download()
            except (OSError, SystemExit) as exc:
                failed.append(type(exc).__name__)
            else:
                raise AssertionError('incomplete metadata fallback succeeded')
            return original(**kwargs)
        w._pinned_snapshot_download(peer_transfer, new, repo_id=repo_id,
            token=False, allow_patterns=files, expected_files=saved.expected_files)
        assert failed, 'the fallback was not exercised'
        loaded = hf.hf_hub_download(repo_id, files[1], local_files_only=True)
        assert new in loaded, ('failed fallback suppressed verified peer', loaded,
                               json.loads(journal.read_text()))
        assert transfers == [new], transfers
        assert ref.read_text() == new
        state = w._load_main_ref_state(journal, old)
        assert state.published_sequence == 2, state  # cancelled first=1, peer=2, fallback=3
        assert not state.attempts and not state.owners
elif scenario in ('no_manifest', 'unattested'):
    # Legacy fallbacks still use the published ref. A manifest without metadata
    # attestation may supply expected files, but must not supply a revision.
    original(repo_id=repo_id, revision=old, allow_patterns=files, token=False)
    ref.parent.mkdir(parents=True, exist_ok=True)
    ref.write_text(old)
    if scenario == 'unattested':
        expected = [dm.ExpectedFile(name, (repo / 'snapshots' / old / name).stat().st_size)
                    for name in files]
        assert dm.write_manifest('model', repo_id, scope, expected, 'http',
                                 commit_hash=new, metadata_derived=False)
    else:
        assert dm.read_manifest('model', repo_id, scope) is None
    w._model_info_with_retry = unavailable
    transfers = []
    hf.snapshot_download = offline_transfer
    download()
    assert transfers == [old], transfers
    loaded = hf.hf_hub_download(repo_id, files[1], local_files_only=True)
    assert old in loaded, loaded
    assert Path(loaded).read_bytes() == hashlib.sha256(b'old').digest() * 2048
else:
    # The saved scope may describe fewer files than the current request. Even
    # without a manifest, its existing requested-file check must precede publish.
    assert scenario in ('requested_saved', 'requested_no_manifest')
    original(repo_id=repo_id, revision=new, allow_patterns=[files[0]], token=False)
    ref.parent.mkdir(parents=True, exist_ok=True)
    ref.write_text(old)
    if scenario == 'requested_saved':
        original(repo_id=repo_id, revision=old, allow_patterns=[files[0]], token=False)
        data = (repo / 'snapshots' / new / files[0]).read_bytes()
        assert dm.write_manifest('model', repo_id, scope,
            [dm.ExpectedFile(files[0], len(data))], 'http',
            commit_hash=new, metadata_derived=True)
    else:
        # refs/main is the only available pin when no manifest exists.
        ref.write_text(new)
    w._model_info_with_retry = unavailable
    transfers = []
    hf.snapshot_download = offline_transfer
    try:
        download()
    except (OSError, SystemExit):
        pass
    else:
        raise AssertionError('missing requested file was accepted')
    state = w._load_main_ref_state(journal, w._read_ref(ref))
    assert state.published_sequence == 0, ('failed scoped request was published', state)
    assert w._read_ref(ref) == (old if scenario == 'requested_saved' else new)
    assert not state.attempts and not state.owners

# Resolve by repo ID through the real Hub API and check the exact fake B bytes.
if scenario in ('resume', 'peer'):
    loaded = hf.hf_hub_download(repo_id, files[1], local_files_only=True)
    assert new in loaded, loaded
    assert Path(loaded).read_bytes() == hashlib.sha256(b'new').digest() * 4096
print('verified', kind, scenario, 'on Hub', hf.__version__)
"""


def _run_case(tmp_path, kind, scenario):
    hub = _FakeHub()
    try:
        hub.head = _NEW
        result = subprocess.run(
            [sys.executable, "-c", _CHILD, str(_BACKEND), kind, scenario],
            env = _worker_env(hub, tmp_path),
            capture_output = True,
            text = True,
            timeout = 40,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        # All online transfers and metadata reads use the loopback fake Hub.
        assert hub.requests
        print(result.stdout.strip())
    finally:
        hub.close()


@pytest.mark.parametrize("kind", ["full", "scoped"])
def test_cancelled_first_download_resumes_from_saved_manifest_without_metadata(tmp_path, kind):
    _run_case(tmp_path, kind, "resume")


@pytest.mark.parametrize("kind", ["full", "scoped"])
def test_failed_metadata_fallback_cannot_suppress_verified_peer(tmp_path, kind):
    _run_case(tmp_path, kind, "peer")


@pytest.mark.parametrize("scenario", ["requested_saved", "requested_no_manifest"])
def test_scoped_requested_files_checked_before_publication(tmp_path, scenario):
    _run_case(tmp_path, "scoped", scenario)


@pytest.mark.parametrize("kind", ["full", "scoped"])
@pytest.mark.parametrize("scenario", ["no_manifest", "unattested"])
def test_legacy_metadata_fallback_uses_published_revision(tmp_path, kind, scenario):
    _run_case(tmp_path, kind, scenario)
