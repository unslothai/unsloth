# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Follow-ups to the STT HTTP download work.

Pinning downloads to a commit stopped huggingface_hub writing refs/main, and
reserving the repository made a cancelled or wedged run everyone else's problem.
These cover what that changed.
"""

import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from core.inference import stt_download_worker as worker_mod
from core.inference import stt_ggml_sidecar as ggml_mod
from core.inference import stt_mtmd_sidecar as mtmd_mod
from core.inference import stt_sidecar as snapshot_mod


class _FakeMeta:
    """One Hub file's metadata, pinned to an immutable commit."""

    commit_hash = "a" * 40
    etag = "etag"
    size = 1


class _FakeInfo:
    """One repo's metadata, pinned to an immutable commit."""

    sha = "a" * 40


def _write_snapshot(hub_cache: Path, repo: str, revision: str, filename: str) -> Path:
    """Lay out one finished pinned download: a blob plus its snapshot entry."""
    repo_cache = hub_cache / f"models--{repo.replace('/', '--')}"
    blob = repo_cache / "blobs" / "etag"
    blob.parent.mkdir(parents = True, exist_ok = True)
    blob.write_bytes(b"weights")
    entry = repo_cache / "snapshots" / revision / filename
    entry.parent.mkdir(parents = True, exist_ok = True)
    entry.write_bytes(b"weights")
    return entry


def test_a_lost_revision_record_still_finds_the_downloaded_model(tmp_path, monkeypatch):
    """A pinned download writes no refs/main, so the record is the only pointer.

    Its write swallows OSError, so without a snapshot fallback an unwritable
    profile would re-download the model on every launch.
    """
    revision = "a" * 40
    repo = ggml_mod.GGML_STT_REPOS["tiny"]
    _write_snapshot(tmp_path, repo, revision, ggml_mod.GGML_STT_MODELS["tiny"])
    monkeypatch.setattr(ggml_mod, "_read_revision_record", lambda repo_id: None)
    recorded = []
    monkeypatch.setattr(
        ggml_mod, "_write_revision_record", lambda repo_id, rev: recorded.append((repo_id, rev))
    )

    assert snapshot_mod._fallback_revisions(repo, hub_cache = tmp_path) == [revision]
    assert ggml_mod._cached_model_path("tiny", hub_cache = tmp_path) is not None
    assert recorded == [(repo, revision)], "the record was not rebuilt from the cache"


def test_fallback_revisions_prefers_refs_main_and_skips_junk(tmp_path):
    """refs/main first when it exists, and only 40-hex snapshot dirs qualify."""
    repo = "owner/repo"
    repo_cache = tmp_path / "models--owner--repo"
    (repo_cache / "snapshots" / ("b" * 40)).mkdir(parents = True)
    (repo_cache / "snapshots" / "not-a-sha").mkdir(parents = True)
    (repo_cache / "refs").mkdir(parents = True)
    (repo_cache / "refs" / "main").write_text("c" * 40, encoding = "utf-8")

    assert snapshot_mod._fallback_revisions(repo, hub_cache = tmp_path) == ["c" * 40, "b" * 40]


@pytest.mark.parametrize(
    "module, state_factory, model_id",
    [
        (snapshot_mod, snapshot_mod._SnapshotDownloadState, "unsloth/whisper-tiny"),
        (ggml_mod, ggml_mod._GgmlDownloadState, "tiny"),
        (mtmd_mod, mtmd_mod._MtmdDownloadState, "qwen3-asr-0.6b"),
    ],
)
def test_a_cancel_during_metadata_leaves_the_shared_cache_alone(
    module, state_factory, model_id, monkeypatch, tmp_path
):
    """cancel() has no child to stop while metadata resolves.

    Claiming the repository would lock out the Model Hub, and preparing the
    cache would purge partials, both after the user was told it stopped.
    """
    claims, prepares, spawns = [], [], []
    # Stub the Hub metadata call. Unstubbed it reaches huggingface.co, and when that
    # fails (offline CI) the exception lands in _run's handler before the guard under
    # test, so every assertion below passes without the guard existing at all.
    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "get_hf_file_metadata", lambda *a, **k: _FakeMeta())
    monkeypatch.setattr(
        huggingface_hub,
        "HfApi",
        lambda *a, **k: SimpleNamespace(model_info = lambda *a, **k: _FakeInfo()),
    )
    monkeypatch.setattr(
        module, "_claim_stt_repository", lambda repo: claims.append(repo) or (None, None)
    )
    monkeypatch.setattr(
        module, "_prepare_stt_cache_for_http", lambda repo, cache: prepares.append(repo)
    )
    monkeypatch.setattr(
        worker_mod, "spawn_download", lambda *a, **k: spawns.append(a) or pytest.fail("spawned")
    )

    state = state_factory()
    state._cancelled = True  # as cancel() leaves it during metadata
    state._run(model_id, None, hub_cache = tmp_path)

    assert claims == [], "a cancelled run still reserved the repository"
    assert prepares == [], "a cancelled run still rewrote the shared cache"
    assert spawns == []


@pytest.mark.parametrize(
    "state_factory, model_id",
    [
        (snapshot_mod._SnapshotDownloadState, "unsloth/whisper-tiny"),
        (ggml_mod._GgmlDownloadState, "tiny"),
        (mtmd_mod._MtmdDownloadState, "qwen3-asr-0.6b"),
    ],
)
def test_restarting_a_cancelling_download_is_not_a_silent_no_op(state_factory, model_id):
    """The same-model early return would join a run that will download nothing."""
    state = state_factory()
    release = threading.Event()
    state._model_id = model_id
    state._cancelled = True
    state._thread = threading.Thread(target = release.wait, daemon = True)
    state._thread.start()
    try:
        with pytest.raises(snapshot_mod.SttModelIdError, match = "cancelling"):
            state.start(model_id)
    finally:
        release.set()
        state._thread.join(timeout = 5)


def test_a_worker_that_ignores_sigterm_is_killed():
    """The canceller holds the repository reservation until the reap returns."""
    process = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(30)",
        ]
    )
    try:
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(worker_mod, "TERMINATE_GRACE_SECONDS", 0.2)
            started = time.monotonic()
            worker_mod.terminate_download(process)
            assert time.monotonic() - started < 0.2, "cancel() waited on the worker"
            assert process.wait(timeout = 10) != 0
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout = 10)


@pytest.mark.parametrize(
    "state_factory, model_id",
    [
        (snapshot_mod._SnapshotDownloadState, "unsloth/whisper-tiny"),
        (ggml_mod._GgmlDownloadState, "tiny"),
        (mtmd_mod._MtmdDownloadState, "qwen3-asr-0.6b"),
    ],
)
def test_status_never_stats_the_cache_under_the_download_lock(state_factory, model_id):
    """Progress is now three stats per selected file; a cancel must not queue."""
    state = state_factory()
    observed = []

    def probe(*args, **kwargs):
        observed.append(state._lock.acquire(blocking = False))
        if observed[-1]:
            state._lock.release()
        return 1

    state._downloaded_bytes = probe
    state._model_id = model_id
    state._thread = threading.Thread(target = lambda: time.sleep(0.3), daemon = True)
    state._thread.start()
    try:
        assert state.status()["bytes_done"] == 1
    finally:
        state._thread.join(timeout = 5)
    assert observed == [True], "progress was computed while holding the download lock"


@pytest.mark.parametrize(
    "state_factory, model_id, restart",
    [
        (
            snapshot_mod._SnapshotDownloadState,
            "unsloth/whisper-tiny",
            lambda s: setattr(s, "_repo", "org/next") or setattr(s, "_selected_files", ()),
        ),
        (ggml_mod._GgmlDownloadState, "tiny", lambda s: setattr(s, "_etag", None)),
        (mtmd_mod._MtmdDownloadState, "qwen3-asr-0.6b", lambda s: setattr(s, "_selected_files", ())),
    ],
)
def test_progress_cannot_mix_a_new_runs_bytes_with_the_old_runs_total(
    state_factory, model_id, restart, tmp_path
):
    """status() reports bytes_total from under the lock and probes the cache after.

    A run that starts in that gap replaces the fields the probe reads, so reading
    them there paired one run's total with another's bytes, or blanked it.
    """
    state = state_factory()
    state._model_id = model_id
    state._repo = "org/repo"
    state._hub_cache = tmp_path
    state._total_bytes = 1000
    state._etag = "etag"
    state._selected_files = (snapshot_mod._SelectedHubFile(path = "f", size = 1000, blob_key = None),)
    release = threading.Event()
    state._thread = threading.Thread(target = release.wait, daemon = True)
    state._thread.start()

    real = state._downloaded_bytes

    def probe_after_a_restart(*args, **kwargs):
        restart(state)  # the next run lands between the lock release and the probe
        return real(*args, **kwargs)

    state._downloaded_bytes = probe_after_a_restart
    try:
        status = state.status()
    finally:
        release.set()
        state._thread.join(timeout = 5)

    assert status["bytes_total"] == 1000
    assert status["bytes_done"] == 0, "the probe read the restarted run's fields"


def test_mtmd_rejects_a_revision_that_is_not_an_immutable_commit(monkeypatch, tmp_path):
    """The other two backends validate the SHA; a bad one here would be pinned
    into --revision and then silently dropped by the revision record."""

    class _Meta:
        commit_hash = "main"
        etag = "etag"
        size = 1

    monkeypatch.setattr(mtmd_mod, "_claim_stt_repository", lambda repo: pytest.fail("claimed"))
    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "get_hf_file_metadata", lambda *a, **k: _Meta())
    state = mtmd_mod._MtmdDownloadState()
    state._run("qwen3-asr-0.6b", None, hub_cache = tmp_path)

    assert state.status()["error"] is not None
