# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import io
import logging

from hub.services import download_lifecycle
from hub.utils import download_registry
from hub.utils import state_dir


def _set_xet_reason(monkeypatch, reason):
    monkeypatch.setattr(
        download_lifecycle.download_registry,
        "download_transport_unavailable_reason",
        lambda _transport: reason,
    )


def _make_proc(rc, stderr = b""):
    class _Proc:
        pid = 4242

        def __init__(self):
            self._rc = rc
            self._waited = False
            self.killed = False
            self.stderr = io.BytesIO(stderr)

        def poll(self):
            return self._rc if self._waited else None

        def wait(self, timeout = None):
            self._waited = True
            return self._rc

        def kill(self):
            self.killed = True

    return _Proc()


class _ImmediateThread:
    def __init__(self, *, target, **_kwargs):
        self._target = target

    def start(self):
        self._target()


def test_resolve_effective_use_xet_keeps_http_when_not_requested(monkeypatch):
    _set_xet_reason(monkeypatch, "should not be consulted")
    assert download_lifecycle.resolve_effective_use_xet(False) is False


def test_resolve_effective_use_xet_keeps_xet_when_available(monkeypatch):
    _set_xet_reason(monkeypatch, None)
    assert download_lifecycle.resolve_effective_use_xet(True) is True


def test_resolve_effective_use_xet_downgrades_when_xet_unavailable(monkeypatch):
    _set_xet_reason(monkeypatch, "Xet transport is unavailable because hf_xet is not installed.")
    assert download_lifecycle.resolve_effective_use_xet(True) is False


def test_download_watcher_retries_xet_failure_over_http_for_model_and_dataset(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(state_dir, "cache_root", lambda: tmp_path / "state")
    monkeypatch.setattr(download_lifecycle.threading, "Thread", _ImmediateThread)
    real_register_worker = download_lifecycle.register_worker

    cases = [
        (
            "model",
            "Org/Model",
            "Q4_K_M",
            ["--repo-id", "Org/Model", "--variant", "Q4_K_M"],
            frozenset({"mainhash"}),
            frozenset({"mainhash", "mmprojhash"}),
            12,
            18,
        ),
        (
            "dataset",
            "Org/Data",
            None,
            ["--repo-id", "Org/Data", "--dataset"],
            frozenset(),
            frozenset(),
            0,
            0,
        ),
    ]

    for (
        repo_type,
        repo_id,
        variant,
        expected_args,
        blob_hashes,
        progress_blob_hashes,
        baseline_bytes,
        retry_baseline_bytes,
    ) in cases:
        registry = download_registry.DownloadRegistry()
        key = (
            download_registry.normalize_job_key(f"{repo_id}::{variant}")
            if variant is not None
            else download_registry.normalize_repo_key(repo_id)
        )
        assert registry.claim(
            key,
            download_registry.TRANSPORT_XET,
            repo_type = repo_type,
            repo_id = repo_id,
            variant = variant,
            blob_hashes = blob_hashes,
            progress_blob_hashes = progress_blob_hashes,
            completed_baseline_bytes = baseline_bytes,
        )
        original_generation = registry.current_generation(key)
        proc = _make_proc(1, b"xet failed")
        spawned = []
        retry_registers = []
        baseline_calls = []

        def fake_completed_blob_bytes(*args):
            baseline_calls.append(args)
            return retry_baseline_bytes

        def fake_spawn_worker(
            args,
            hf_token,
            *,
            use_xet,
            protected_blob_hashes = None,
        ):
            assert registry.get_job(key).state == "running"
            spawned.append((args, use_xet, protected_blob_hashes))
            return _make_proc(0, b"http retry")

        def fake_register_worker(*_args, **kwargs):
            retry_registers.append(kwargs)
            return True

        monkeypatch.setattr(download_registry, "completed_blob_bytes", fake_completed_blob_bytes)
        monkeypatch.setattr(download_lifecycle, "spawn_worker", fake_spawn_worker)
        monkeypatch.setattr(download_lifecycle, "register_worker", fake_register_worker)

        assert real_register_worker(
            registry,
            key,
            proc,
            hf_token = None,
            label = f"{repo_id}{f' [{variant}]' if variant else ''}",
            log_prefix = "Download",
            logger = logging.getLogger("test"),
            repo_type = repo_type,
            repo_id = repo_id,
            transport = download_registry.TRANSPORT_XET,
            watch_name = f"{repo_type}-watch",
        )

        assert spawned == [(expected_args, False, None)]
        assert (
            retry_registers and retry_registers[0]["transport"] == download_registry.TRANSPORT_HTTP
        )
        metadata = registry.get_job_metadata(key)
        assert metadata is not None
        assert metadata.transport == download_registry.TRANSPORT_HTTP
        assert metadata.blob_hashes == blob_hashes
        assert metadata.progress_blob_hashes == progress_blob_hashes
        assert metadata.completed_baseline_bytes == retry_baseline_bytes
        assert registry.current_generation(key) == original_generation
        assert baseline_calls == (
            [("model", "Org/Model", progress_blob_hashes)] if progress_blob_hashes else []
        )
        assert registry.get_job(key).state == "running"


def test_download_watcher_keeps_http_failure_terminal(monkeypatch, tmp_path):
    monkeypatch.setattr(state_dir, "cache_root", lambda: tmp_path / "state")
    monkeypatch.setattr(download_lifecycle.threading, "Thread", _ImmediateThread)
    real_register_worker = download_lifecycle.register_worker
    registry = download_registry.DownloadRegistry()
    key = download_registry.normalize_repo_key("Org/Data")
    assert registry.claim(
        key,
        download_registry.TRANSPORT_HTTP,
        repo_type = "dataset",
        repo_id = "Org/Data",
    )
    proc = _make_proc(1, b"http failed")

    def fake_register_worker(*_args, **_kwargs):
        raise AssertionError("HTTP failure should stay terminal")

    monkeypatch.setattr(download_lifecycle, "register_worker", fake_register_worker)

    assert real_register_worker(
        registry,
        key,
        proc,
        hf_token = None,
        label = "Org/Data",
        log_prefix = "Download",
        logger = logging.getLogger("test"),
        repo_type = "dataset",
        repo_id = "Org/Data",
        transport = download_registry.TRANSPORT_HTTP,
        watch_name = "dataset-watch",
    )

    assert registry.get_job(key).state == "error"


def test_http_retry_skip_without_metadata_sets_error(monkeypatch, tmp_path):
    monkeypatch.setattr(state_dir, "cache_root", lambda: tmp_path / "state")
    registry = download_registry.DownloadRegistry()
    key = download_registry.normalize_job_key("Org/Model::Q4_K_M")
    registry.set_job(key, "running")

    def fake_spawn_worker(*_args, **_kwargs):
        raise AssertionError("metadata-free retry should not spawn a worker")

    def fake_register_worker(*_args, **_kwargs):
        raise AssertionError("metadata-free retry should not register a worker")

    monkeypatch.setattr(download_lifecycle, "spawn_worker", fake_spawn_worker)
    monkeypatch.setattr(download_lifecycle, "register_worker", fake_register_worker)

    assert not download_lifecycle._try_http_retry(
        registry,
        key,
        hf_token = None,
        label = "Org/Model [Q4_K_M]",
        log_prefix = "Download",
        logger = logging.getLogger("test"),
        repo_type = "model",
        repo_id = "Org/Model",
        watch_name = "model-watch",
    )

    assert registry.get_job(key).state == "error"


def test_http_retry_skip_for_non_xet_transport_sets_error(monkeypatch, tmp_path):
    monkeypatch.setattr(state_dir, "cache_root", lambda: tmp_path / "state")
    registry = download_registry.DownloadRegistry()
    key = download_registry.normalize_repo_key("Org/Data")
    assert registry.claim(
        key,
        download_registry.TRANSPORT_HTTP,
        repo_type = "dataset",
        repo_id = "Org/Data",
    )

    def fake_spawn_worker(*_args, **_kwargs):
        raise AssertionError("non-XET retry should not spawn a worker")

    def fake_register_worker(*_args, **_kwargs):
        raise AssertionError("non-XET retry should not register a worker")

    monkeypatch.setattr(download_lifecycle, "spawn_worker", fake_spawn_worker)
    monkeypatch.setattr(download_lifecycle, "register_worker", fake_register_worker)

    assert not download_lifecycle._try_http_retry(
        registry,
        key,
        hf_token = None,
        label = "Org/Data",
        log_prefix = "Download",
        logger = logging.getLogger("test"),
        repo_type = "dataset",
        repo_id = "Org/Data",
        watch_name = "dataset-watch",
    )

    assert registry.get_job(key).state == "error"


def test_download_watcher_restores_xet_transport_when_http_retry_spawn_fails(monkeypatch, tmp_path):
    monkeypatch.setattr(state_dir, "cache_root", lambda: tmp_path / "state")
    monkeypatch.setattr(download_lifecycle.threading, "Thread", _ImmediateThread)
    real_register_worker = download_lifecycle.register_worker
    registry = download_registry.DownloadRegistry()
    key = download_registry.normalize_job_key("Org/Model::Q4_K_M")
    assert registry.claim(
        key,
        download_registry.TRANSPORT_XET,
        repo_type = "model",
        repo_id = "Org/Model",
        variant = "Q4_K_M",
        blob_hashes = frozenset({"mainhash"}),
        progress_blob_hashes = frozenset({"mainhash", "mmprojhash"}),
        completed_baseline_bytes = 12,
    )
    proc = _make_proc(1, b"xet failed")

    def fake_spawn_worker(*_args, **_kwargs):
        raise RuntimeError("HTTP retry spawn failed")

    def fake_register_worker(*_args, **_kwargs):
        raise AssertionError("HTTP retry should not register after spawn failure")

    monkeypatch.setattr(download_lifecycle, "spawn_worker", fake_spawn_worker)
    monkeypatch.setattr(download_lifecycle, "register_worker", fake_register_worker)

    assert real_register_worker(
        registry,
        key,
        proc,
        hf_token = None,
        label = "Org/Model [Q4_K_M]",
        log_prefix = "Download",
        logger = logging.getLogger("test"),
        repo_type = "model",
        repo_id = "Org/Model",
        transport = download_registry.TRANSPORT_XET,
        watch_name = "model-watch",
    )

    assert registry.get_job(key).state == "error"
    metadata = registry.get_job_metadata(key)
    assert metadata is not None
    assert metadata.transport == download_registry.TRANSPORT_XET


def test_download_watcher_preserves_pending_cancel_when_http_retry_claim_fails(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(state_dir, "cache_root", lambda: tmp_path / "state")
    monkeypatch.setattr(download_lifecycle.threading, "Thread", _ImmediateThread)
    real_register_worker = download_lifecycle.register_worker
    registry = download_registry.DownloadRegistry()
    key = download_registry.normalize_job_key("Org/Model::Q4_K_M")
    assert registry.claim(
        key,
        download_registry.TRANSPORT_XET,
        repo_type = "model",
        repo_id = "Org/Model",
        variant = "Q4_K_M",
        blob_hashes = frozenset({"mainhash"}),
        progress_blob_hashes = frozenset({"mainhash"}),
    )
    generation = registry.current_generation(key)
    proc = _make_proc(1, b"xet failed")
    markers = []
    original_claim = registry.claim

    def fake_spawn_worker(*_args, **_kwargs):
        raise AssertionError("failed retry claim should not spawn a worker")

    def fake_claim(*args, **kwargs):
        if kwargs.get("replace_active"):
            assert registry.mark_pending_cancel(key, generation)
            return False, "cancelling"
        return original_claim(*args, **kwargs)

    def fake_register_worker(*_args, **_kwargs):
        raise AssertionError("failed retry claim should not register a worker")

    def fake_persist_cancel_marker(*args, **_kwargs):
        markers.append(args)

    monkeypatch.setattr(registry, "claim", fake_claim)
    monkeypatch.setattr(download_lifecycle, "spawn_worker", fake_spawn_worker)
    monkeypatch.setattr(download_lifecycle, "register_worker", fake_register_worker)
    monkeypatch.setattr(download_registry, "persist_cancel_marker", fake_persist_cancel_marker)

    assert real_register_worker(
        registry,
        key,
        proc,
        hf_token = None,
        label = "Org/Model [Q4_K_M]",
        log_prefix = "Download",
        logger = logging.getLogger("test"),
        repo_type = "model",
        repo_id = "Org/Model",
        transport = download_registry.TRANSPORT_XET,
        watch_name = "model-watch",
    )

    assert registry.get_job(key).state == "cancelled"
    assert markers == [("model", "Org/Model", "Q4_K_M", download_registry.TRANSPORT_XET)]


def test_download_watcher_preserves_pending_cancel_when_http_retry_spawn_fails(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(state_dir, "cache_root", lambda: tmp_path / "state")
    monkeypatch.setattr(download_lifecycle.threading, "Thread", _ImmediateThread)
    real_register_worker = download_lifecycle.register_worker
    registry = download_registry.DownloadRegistry()
    key = download_registry.normalize_job_key("Org/Model::Q4_K_M")
    assert registry.claim(
        key,
        download_registry.TRANSPORT_XET,
        repo_type = "model",
        repo_id = "Org/Model",
        variant = "Q4_K_M",
        blob_hashes = frozenset({"mainhash"}),
        progress_blob_hashes = frozenset({"mainhash"}),
    )
    generation = registry.current_generation(key)
    proc = _make_proc(1, b"xet failed")
    markers = []

    def fake_spawn_worker(*_args, **_kwargs):
        assert registry.mark_pending_cancel(key, generation)
        raise RuntimeError("HTTP retry spawn failed")

    def fake_register_worker(*_args, **_kwargs):
        raise AssertionError("failed retry spawn should not register a worker")

    def fake_persist_cancel_marker(*args, **_kwargs):
        markers.append(args)

    monkeypatch.setattr(download_lifecycle, "spawn_worker", fake_spawn_worker)
    monkeypatch.setattr(download_lifecycle, "register_worker", fake_register_worker)
    monkeypatch.setattr(download_registry, "persist_cancel_marker", fake_persist_cancel_marker)

    assert real_register_worker(
        registry,
        key,
        proc,
        hf_token = None,
        label = "Org/Model [Q4_K_M]",
        log_prefix = "Download",
        logger = logging.getLogger("test"),
        repo_type = "model",
        repo_id = "Org/Model",
        transport = download_registry.TRANSPORT_XET,
        watch_name = "model-watch",
    )

    assert registry.get_job(key).state == "cancelled"
    assert markers == [("model", "Org/Model", "Q4_K_M", download_registry.TRANSPORT_XET)]


def test_download_watcher_persists_cancel_without_retry(monkeypatch, tmp_path):
    monkeypatch.setattr(state_dir, "cache_root", lambda: tmp_path / "state")
    monkeypatch.setattr(download_lifecycle.threading, "Thread", _ImmediateThread)
    real_register_worker = download_lifecycle.register_worker
    registry = download_registry.DownloadRegistry()
    key = download_registry.normalize_repo_key("Org/Data")
    assert registry.claim(
        key,
        download_registry.TRANSPORT_XET,
        repo_type = "dataset",
        repo_id = "Org/Data",
    )
    proc = _make_proc(130, b"cancelled")
    markers = []

    def fake_persist_cancel_marker(*args, **_kwargs):
        markers.append(args)

    def fake_register_worker(*_args, **_kwargs):
        raise AssertionError("cancelled workers should not retry")

    monkeypatch.setattr(download_registry, "persist_cancel_marker", fake_persist_cancel_marker)
    monkeypatch.setattr(download_lifecycle, "register_worker", fake_register_worker)

    assert real_register_worker(
        registry,
        key,
        proc,
        hf_token = None,
        label = "Org/Data",
        log_prefix = "Download",
        logger = logging.getLogger("test"),
        repo_type = "dataset",
        repo_id = "Org/Data",
        transport = download_registry.TRANSPORT_XET,
        watch_name = "dataset-watch",
    )

    assert registry.get_job(key).state == "cancelled"
    assert markers == [("dataset", "Org/Data", None, download_registry.TRANSPORT_XET)]
