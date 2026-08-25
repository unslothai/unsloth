# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Sequential HTTP transport and exact STT progress invariants."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from core.inference import (
    stt_download_worker,
    stt_ggml_sidecar,
    stt_mtmd_sidecar,
    stt_sidecar,
)
from hub.utils import download_registry


def _repo_dir(root: Path, repo: str) -> Path:
    return root / f"models--{repo.replace('/', '--')}"


def _case_insensitive(root: Path) -> bool:
    """macOS and Windows fold case, so two case variants are one directory there."""
    probe = root / "CaseProbe"
    probe.mkdir()
    try:
        return (root / "caseprobe").is_dir()
    finally:
        probe.rmdir()


def test_http_cache_preparation_fails_closed_when_blob_directory_is_unreadable(
    monkeypatch, tmp_path
):
    repo = "Org/Repo"
    entry = _repo_dir(tmp_path, repo)
    blobs = entry / "blobs"
    blobs.mkdir(parents = True)
    (entry / ".transport").write_text("xet", encoding = "utf-8")
    (blobs / "sparse.incomplete").write_bytes(b"x" * 64)
    real_iterdir = Path.iterdir

    def unreadable(self):
        if self == blobs:
            raise PermissionError("unreadable")
        return real_iterdir(self)

    monkeypatch.setattr(Path, "iterdir", unreadable)

    with pytest.raises(stt_sidecar.SttDownloadCacheError):
        stt_sidecar._prepare_stt_cache_for_http(repo, tmp_path)

    assert (entry / ".transport").read_text(encoding = "utf-8") == "xet"
    assert (blobs / "sparse.incomplete").is_file()


def test_http_cache_preparation_requires_marker_readback(monkeypatch, tmp_path):
    real_read_text = Path.read_text

    def unreadable_marker(self, *args, **kwargs):
        if self.name == download_registry.TRANSPORT_MARKER_NAME:
            raise PermissionError("unreadable")
        return real_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", unreadable_marker)

    with pytest.raises(stt_sidecar.SttDownloadCacheError):
        stt_sidecar._prepare_stt_cache_for_http("Org/Repo", tmp_path)


@pytest.mark.parametrize(
    "canonical_first",
    (False, True),
    ids = ("alias-first", "canonical-first"),
)
def test_http_cache_preparation_requires_every_case_variant_marker(
    monkeypatch, tmp_path, canonical_first
):
    if _case_insensitive(tmp_path):
        pytest.skip("case-folding filesystem: the alias and the canonical dir are one entry")
    repo = "Org/Repo"
    alias = _repo_dir(tmp_path, "org/repo")
    canonical = _repo_dir(tmp_path, repo)
    alias.mkdir(parents = True)
    canonical_blobs = canonical / "blobs"
    canonical_blobs.mkdir(parents = True)
    (alias / ".transport").write_text("http", encoding = "utf-8")
    (canonical / ".transport").write_text("xet", encoding = "utf-8")
    partial = canonical_blobs / "sparse.incomplete"
    partial.write_bytes(b"x" * 64)
    ordered_entries = (canonical, alias) if canonical_first else (alias, canonical)
    real_iterdir = Path.iterdir

    def ordered_unreadable(self):
        if self == tmp_path:
            return iter(ordered_entries)
        if self == canonical_blobs:
            raise PermissionError("unreadable")
        return real_iterdir(self)

    monkeypatch.setattr(Path, "iterdir", ordered_unreadable)

    with pytest.raises(stt_sidecar.SttDownloadCacheError):
        stt_sidecar._prepare_stt_cache_for_http(repo, tmp_path)

    spawned = []

    class Registry:
        @staticmethod
        def release_repository_owner(_repo, _owner):
            return True

    monkeypatch.setattr(
        stt_ggml_sidecar,
        "_claim_stt_repository",
        lambda _repo: (Registry(), object()),
    )
    monkeypatch.setattr("huggingface_hub.get_hf_file_metadata", lambda *_a, **_k: None)
    monkeypatch.setattr("huggingface_hub.hf_hub_url", lambda *_a, **_k: "https://example")
    monkeypatch.setattr(
        stt_download_worker,
        "spawn_download",
        lambda *_a, **_k: spawned.append(True),
    )

    state = stt_ggml_sidecar._GgmlDownloadState()
    state._run("tiny", None, hub_cache = tmp_path)

    assert spawned == []
    assert (canonical / ".transport").read_text(encoding = "utf-8") == "xet"
    assert partial.read_bytes() == b"x" * 64


def test_http_retry_keeps_its_owned_partial_and_cross_transport_retry_purges_it(tmp_path):
    repo = "Org/Repo"
    entry = _repo_dir(tmp_path, repo)
    blobs = entry / "blobs"
    blobs.mkdir(parents = True)

    stt_sidecar._prepare_stt_cache_for_http(repo, tmp_path)
    partial = blobs / "etag.incomplete"
    partial.write_bytes(b"x" * 64)
    stt_sidecar._prepare_stt_cache_for_http(repo, tmp_path)
    assert partial.stat().st_size == 64

    (entry / ".transport").write_text("xet", encoding = "utf-8")
    stt_sidecar._prepare_stt_cache_for_http(repo, tmp_path)
    assert not partial.exists()


def test_spawned_stt_worker_is_http_only_and_uses_the_captured_cache(monkeypatch, tmp_path):
    observed = {}

    class Process:
        pid = 12345

    def popen(argv, **kwargs):
        observed["argv"] = argv
        observed["env"] = kwargs["env"]
        return Process()

    monkeypatch.setattr(stt_download_worker.subprocess, "Popen", popen)
    monkeypatch.setattr("utils.process_lifetime.adopt_pid", lambda _pid: None)
    monkeypatch.setattr("utils.process_lifetime.child_popen_kwargs", lambda: {})

    stt_download_worker.spawn_download(
        ["--repo-id", "Org/Repo", "--filename", "model.bin"],
        hub_cache = tmp_path,
    )

    assert observed["env"]["HF_HUB_CACHE"] == str(tmp_path)
    assert observed["env"]["HF_HUB_DISABLE_XET"] == "1"
    assert observed["env"]["HF_HUB_ENABLE_HF_TRANSFER"] == "0"
    assert observed["argv"][-4:] == ["--repo-id", "Org/Repo", "--filename", "model.bin"]


def test_captured_cache_worker_stays_online_during_temporary_offline_guard(monkeypatch, tmp_path):
    fake_package = tmp_path / "modules" / "huggingface_hub"
    fake_package.mkdir(parents = True)
    observed_path = tmp_path / "worker-environment.json"
    fake_package.joinpath("__init__.py").write_text(
        """import json
import os

def hf_hub_download(**kwargs):
    with open(os.environ["STT_WORKER_ENV_PATH"], "w", encoding="utf-8") as handle:
        json.dump({
            "hub_cache": os.environ.get("HF_HUB_CACHE"),
            "hub_offline": os.environ.get("HF_HUB_OFFLINE"),
            "transformers_offline": os.environ.get("TRANSFORMERS_OFFLINE"),
        }, handle)
""",
        encoding = "utf-8",
    )
    monkeypatch.setenv("PYTHONPATH", str(tmp_path / "modules"))
    monkeypatch.setenv("STT_WORKER_ENV_PATH", str(observed_path))
    monkeypatch.delenv("HF_HUB_OFFLINE", raising = False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising = False)

    from utils.utils import force_hf_offline

    captured_cache = tmp_path / "captured-hub"
    with force_hf_offline():
        process = stt_download_worker.spawn_download(
            ["--repo-id", "Org/Repo", "--filename", "model.bin"],
            hub_cache = captured_cache,
        )
        stderr = stt_download_worker.reap_download(process)

    assert process.returncode == 0, stderr.decode("utf-8", "replace")
    assert json.loads(observed_path.read_text(encoding = "utf-8")) == {
        "hub_cache": str(captured_cache),
        "hub_offline": None,
        "transformers_offline": None,
    }


def test_worker_downloads_only_the_exact_repeated_filenames(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "huggingface_hub.hf_hub_download",
        lambda **kwargs: calls.append(kwargs),
    )

    assert (
        stt_download_worker.main(
            [
                "--repo-id",
                "Org/Repo",
                "--revision",
                "a" * 40,
                "--filename",
                "literal[1].json",
                "--filename",
                "weights/*.safetensors",
            ]
        )
        == 0
    )
    assert [call["filename"] for call in calls] == ["literal[1].json", "weights/*.safetensors"]


def test_selected_file_progress_covers_partial_final_and_windows_snapshot(tmp_path):
    repo = "Org/Repo"
    revision = "b" * 40
    entry = _repo_dir(tmp_path, repo)
    blobs = entry / "blobs"
    blobs.mkdir(parents = True)
    partial = blobs / "etag.incomplete"
    partial.write_bytes(b"x" * 40)
    kwargs = {
        "hub_cache": tmp_path,
        "repo": repo,
        "filename": "nested/model.bin",
        "size": 100,
        "blob_key": "etag",
        "revision": revision,
    }
    assert stt_sidecar._downloaded_file_bytes(**kwargs) == 40

    partial.rename(blobs / "etag")
    assert stt_sidecar._downloaded_file_bytes(**kwargs) == 40
    (blobs / "etag").unlink()
    snapshot = entry / "snapshots" / revision / "nested"
    snapshot.mkdir(parents = True)
    (snapshot / "model.bin").write_bytes(b"x" * 120)
    assert stt_sidecar._downloaded_file_bytes(**kwargs) == 100


def test_mtmd_progress_counts_only_its_two_exact_files(tmp_path):
    model_id = "qwen3-asr-0.6b"
    spec = stt_mtmd_sidecar.MTMD_STT_MODELS[model_id]
    entry = _repo_dir(tmp_path, spec.repo)
    blobs = entry / "blobs"
    blobs.mkdir(parents = True)
    (blobs / "main").write_bytes(b"x" * 80)
    (blobs / "mmproj.incomplete").write_bytes(b"x" * 15)
    (blobs / "unrelated").write_bytes(b"x" * 1000)
    state = stt_mtmd_sidecar._MtmdDownloadState()
    state._model_id = model_id
    state._hub_cache = tmp_path
    state._revision = "c" * 40
    state._selected_files = (
        stt_sidecar._SelectedHubFile(spec.model_file, 80, "main"),
        stt_sidecar._SelectedHubFile(spec.mmproj_file, 20, "mmproj"),
    )
    state._total_bytes = 100

    assert state._downloaded_bytes() == 95


def test_mtmd_run_holds_one_owner_and_one_cache_through_final_lookup(monkeypatch, tmp_path):
    model_id = "qwen3-asr-0.6b"
    spec = stt_mtmd_sidecar.MTMD_STT_MODELS[model_id]
    revision = "d" * 40
    events = []

    class Registry:
        def release_repository_owner(self, repo, owner):
            events.append(("release", repo, owner))
            return True

    owner = object()
    monkeypatch.setattr(
        stt_mtmd_sidecar,
        "_claim_stt_repository",
        lambda repo: (events.append(("claim", repo)) or (Registry(), owner)),
    )
    monkeypatch.setattr(
        stt_mtmd_sidecar,
        "_prepare_stt_cache_for_http",
        lambda repo, root: events.append(("prepare", repo, root)),
    )

    def metadata(url, token = None):
        name = spec.model_file if spec.model_file in url else spec.mmproj_file
        return SimpleNamespace(size = 10, etag = name, commit_hash = revision)

    monkeypatch.setattr("huggingface_hub.get_hf_file_metadata", metadata)
    monkeypatch.setattr(
        "huggingface_hub.hf_hub_url",
        lambda repo, filename, revision = None: f"https://example/{repo}/{filename}?rev={revision}",
    )

    class Process:
        returncode = 0

        def poll(self):
            return 0

        def terminate(self):
            events.append(("terminate",))

    def spawn(
        args,
        hf_token = None,
        *,
        hub_cache = None,
    ):
        events.append(("spawn", args, hub_cache))
        return Process()

    monkeypatch.setattr("core.inference.stt_download_worker.spawn_download", spawn)
    monkeypatch.setattr("core.inference.stt_download_worker.reap_download", lambda _process: b"")
    monkeypatch.setattr(
        stt_mtmd_sidecar,
        "_cached_model_paths",
        lambda model, *, hub_cache, revision: (
            events.append(("lookup", model, hub_cache, revision)) or ("model", "mmproj")
        ),
    )
    monkeypatch.setattr(
        stt_mtmd_sidecar,
        "_write_revision_record",
        lambda repo, rev: events.append(("record", repo, rev)),
    )

    state = stt_mtmd_sidecar._MtmdDownloadState()
    state._hub_cache = tmp_path
    state._run(model_id, None)

    spawn_event = next(event for event in events if event[0] == "spawn")
    assert spawn_event[2] == tmp_path
    assert spawn_event[1] == [
        "--repo-id",
        spec.repo,
        "--revision",
        revision,
        "--filename",
        spec.model_file,
        "--filename",
        spec.mmproj_file,
    ]
    assert [event[0] for event in events] == [
        "claim",
        "prepare",
        "spawn",
        "lookup",
        "record",
        "release",
    ]


def test_ggml_metadata_revision_pins_progress_worker_and_cached_lookup(monkeypatch, tmp_path):
    model_id = "tiny"
    repo = stt_ggml_sidecar.GGML_STT_REPOS[model_id]
    filename = stt_ggml_sidecar.GGML_STT_MODELS[model_id]
    revision_a = "a" * 40
    revision_b = "b" * 40
    entry = _repo_dir(tmp_path, repo)
    blobs = entry / "blobs"
    blobs.mkdir(parents = True)
    (blobs / "etag-a.incomplete").write_bytes(b"a" * 25)
    (blobs / "etag-b").write_bytes(b"b" * 100)
    observed = {}
    cached_model_path = stt_ggml_sidecar._cached_model_path

    class Registry:
        @staticmethod
        def release_repository_owner(_repo, _owner):
            return True

    class Process:
        returncode = 0

        @staticmethod
        def poll():
            return 0

    state = stt_ggml_sidecar._GgmlDownloadState()
    state._model_id = model_id
    state._hub_cache = tmp_path
    monkeypatch.setattr(
        stt_ggml_sidecar,
        "_claim_stt_repository",
        lambda _repo: (Registry(), object()),
    )
    monkeypatch.setattr(stt_ggml_sidecar, "_prepare_stt_cache_for_http", lambda *_args: None)
    monkeypatch.setattr(
        "huggingface_hub.get_hf_file_metadata",
        lambda *_args, **_kwargs: SimpleNamespace(
            size = 100,
            etag = "etag-a",
            commit_hash = revision_a,
        ),
    )
    monkeypatch.setattr(
        "huggingface_hub.hf_hub_url",
        lambda *_args, **_kwargs: f"https://example/{repo}/{filename}",
    )

    def spawn(
        args,
        hf_token = None,
        *,
        hub_cache = None,
    ):
        observed["args"] = args
        observed["hub_cache"] = hub_cache
        observed["progress"] = state._downloaded_bytes()
        observed["mutable_main"] = revision_b
        return Process()

    def lookup(
        model,
        *,
        hub_cache = None,
        revision = None,
    ):
        observed["lookup"] = (model, hub_cache, revision)
        return str(entry / "snapshots" / revision / filename)

    monkeypatch.setattr("core.inference.stt_download_worker.spawn_download", spawn)
    monkeypatch.setattr("core.inference.stt_download_worker.reap_download", lambda _process: b"")
    monkeypatch.setattr(stt_ggml_sidecar, "_cached_model_path", lookup)
    monkeypatch.setattr(
        stt_ggml_sidecar,
        "_write_revision_record",
        lambda recorded_repo, revision: observed.update(record = (recorded_repo, revision)),
    )

    state._run(model_id, None)

    assert observed["args"] == [
        "--repo-id",
        repo,
        "--revision",
        revision_a,
        "--filename",
        filename,
    ]
    assert observed["hub_cache"] == tmp_path
    assert observed["progress"] == 25
    assert observed["mutable_main"] == revision_b
    assert observed["lookup"] == (model_id, tmp_path, revision_a)
    assert observed["record"] == (repo, revision_a)

    monkeypatch.setattr(
        stt_ggml_sidecar,
        "_read_revision_record",
        lambda recorded_repo: observed["record"][1] if recorded_repo == repo else None,
    )
    monkeypatch.setattr(
        "huggingface_hub.hf_hub_download",
        lambda **kwargs: observed.update(later_lookup = kwargs) or "/cached/model.bin",
    )
    assert cached_model_path(model_id, hub_cache = tmp_path) == "/cached/model.bin"
    assert observed["later_lookup"]["revision"] == revision_a


def test_ggml_metadata_failure_spawns_nothing_and_leaks_no_owner(monkeypatch, tmp_path):
    events = []

    def metadata_failure(*_args, **_kwargs):
        raise RuntimeError("metadata unavailable")

    monkeypatch.setattr("huggingface_hub.get_hf_file_metadata", metadata_failure)
    monkeypatch.setattr("huggingface_hub.hf_hub_url", lambda *_args, **_kwargs: "https://example")
    monkeypatch.setattr(
        stt_ggml_sidecar,
        "_claim_stt_repository",
        lambda _repo: events.append("claim"),
    )
    monkeypatch.setattr(
        stt_download_worker,
        "spawn_download",
        lambda *_args, **_kwargs: events.append("spawn"),
    )

    state = stt_ggml_sidecar._GgmlDownloadState()
    state._run("tiny", None, hub_cache = tmp_path)

    assert events == []
    assert state.status()["error"] == "Download failed for 'tiny'."


def test_ggml_stale_record_falls_back_to_active_cache_main(monkeypatch, tmp_path):
    model_id = "tiny"
    repo = stt_ggml_sidecar.GGML_STT_REPOS[model_id]
    filename = stt_ggml_sidecar.GGML_STT_MODELS[model_id]
    stale_revision = "a" * 40
    active_revision = "b" * 40
    entry = _repo_dir(tmp_path, repo)
    (entry / "refs").mkdir(parents = True)
    (entry / "refs" / "main").write_text(active_revision, encoding = "utf-8")
    cached_file = entry / "snapshots" / active_revision / filename
    cached_file.parent.mkdir(parents = True)
    cached_file.write_bytes(b"model")
    records = []

    monkeypatch.setattr(stt_ggml_sidecar, "_read_revision_record", lambda _repo: stale_revision)
    monkeypatch.setattr(
        stt_ggml_sidecar,
        "_write_revision_record",
        lambda recorded_repo, revision: records.append((recorded_repo, revision)),
    )

    assert stt_ggml_sidecar._cached_model_path(model_id, hub_cache = tmp_path) == str(cached_file)
    assert records == [(repo, active_revision)]


def test_mtmd_stale_record_falls_back_to_one_active_cache_revision(monkeypatch, tmp_path):
    model_id = "qwen3-asr-0.6b"
    spec = stt_mtmd_sidecar.MTMD_STT_MODELS[model_id]
    stale_revision = "c" * 40
    active_revision = "d" * 40
    entry = _repo_dir(tmp_path, spec.repo)
    (entry / "refs").mkdir(parents = True)
    (entry / "refs" / "main").write_text(active_revision, encoding = "utf-8")
    snapshot = entry / "snapshots" / active_revision
    snapshot.mkdir(parents = True)
    model_file = snapshot / spec.model_file
    mmproj_file = snapshot / spec.mmproj_file
    model_file.write_bytes(b"model")
    mmproj_file.write_bytes(b"mmproj")
    records = []

    monkeypatch.setattr(stt_mtmd_sidecar, "_read_revision_record", lambda _repo: stale_revision)
    monkeypatch.setattr(
        stt_mtmd_sidecar,
        "_write_revision_record",
        lambda recorded_repo, revision: records.append((recorded_repo, revision)),
    )

    assert stt_mtmd_sidecar._cached_model_paths(model_id, hub_cache = tmp_path) == (
        str(model_file),
        str(mmproj_file),
    )
    assert records == [(spec.repo, active_revision)]


def test_start_captures_the_configured_hub_cache_once(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(
        stt_sidecar,
        "_capture_stt_hub_cache",
        lambda: (calls.append("capture") or tmp_path),
    )
    state = stt_sidecar._SnapshotDownloadState()
    observed = []
    monkeypatch.setattr(
        state,
        "_run",
        lambda repo, token, revision: observed.append((repo, state._hub_cache)),
    )

    state.start("tiny")
    state._thread.join(timeout = 5)

    assert calls == ["capture"]
    assert observed == [(stt_sidecar.STT_MODELS["tiny"], tmp_path)]


def test_live_studio_cache_wins_over_the_stale_startup_environment(monkeypatch, tmp_path):
    old_cache = tmp_path / "old" / "hub"
    new_cache = tmp_path / "new" / "hub"
    monkeypatch.setenv("HF_HUB_CACHE", str(old_cache))
    monkeypatch.setattr(
        "utils.hf_cache_settings.get_hf_cache_paths",
        lambda: SimpleNamespace(hub_cache = new_cache, source = "studio"),
    )

    assert stt_sidecar._capture_stt_hub_cache() == new_cache
    assert stt_sidecar._active_hf_hub_cache() == new_cache


@pytest.mark.parametrize(
    "state_factory",
    (
        stt_sidecar._SnapshotDownloadState,
        stt_ggml_sidecar._GgmlDownloadState,
        stt_mtmd_sidecar._MtmdDownloadState,
    ),
)
def test_each_stt_download_path_terminates_its_worker_on_cancel(state_factory):
    class Thread:
        @staticmethod
        def is_alive():
            return True

    class Process:
        terminated = False

        @staticmethod
        def poll():
            return None

        def terminate(self):
            self.terminated = True

    state = state_factory()
    process = Process()
    state._thread = Thread()
    state._process = process

    assert state.cancel() is True
    assert state._cancelled is True
    assert process.terminated is True
