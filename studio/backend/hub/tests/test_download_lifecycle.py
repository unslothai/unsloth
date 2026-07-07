# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import io
import json
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
            assert registry.get_job_metadata(key).transport == download_registry.TRANSPORT_HTTP
            if repo_type == "model" and variant:
                sibling_claimed, sibling_state = registry.claim(
                    "Org/Model::Q5_K_M",
                    download_registry.TRANSPORT_XET,
                    repo_type = "model",
                    repo_id = "Org/Model",
                    variant = "Q5_K_M",
                    blob_hashes = frozenset({"q5-main"}),
                    progress_blob_hashes = frozenset({"q5-main", "mmprojhash"}),
                )
                assert sibling_claimed is False
                assert sibling_state == "running"
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


def test_download_watcher_defers_http_retry_while_sibling_xet_variant_is_active(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(state_dir, "cache_root", lambda: tmp_path / "state")
    monkeypatch.setattr(download_lifecycle.threading, "Thread", _ImmediateThread)
    real_register_worker = download_lifecycle.register_worker
    registry = download_registry.DownloadRegistry()
    key_a = download_registry.normalize_job_key("Org/Model::Q4_K_M")
    key_b = download_registry.normalize_job_key("Org/Model::Q5_K_M")
    assert registry.claim(
        key_a,
        download_registry.TRANSPORT_XET,
        repo_type = "model",
        repo_id = "Org/Model",
        variant = "Q4_K_M",
        blob_hashes = frozenset({"mainhash-a"}),
        progress_blob_hashes = frozenset({"mainhash-a"}),
    )
    assert registry.claim(
        key_b,
        download_registry.TRANSPORT_XET,
        repo_type = "model",
        repo_id = "Org/Model",
        variant = "Q5_K_M",
        blob_hashes = frozenset({"mainhash-b"}),
        progress_blob_hashes = frozenset({"mainhash-b"}),
    )
    proc = _make_proc(1, b"xet failed")
    sleep_calls = []
    spawned = []

    def fake_sleep(seconds):
        sleep_calls.append(seconds)
        assert registry.get_job(key_a).state == "running"
        assert registry.get_job_metadata(key_a).transport == download_registry.TRANSPORT_XET
        assert registry.begin_delete("Org/Model", "Q4_K_M") is False
        registry.set_job(key_b, "complete")

    def fake_spawn_worker(
        args,
        hf_token,
        *,
        use_xet,
        protected_blob_hashes = None,
    ):
        assert use_xet is False
        assert registry.get_job_metadata(key_a).transport == download_registry.TRANSPORT_HTTP
        spawned.append((args, protected_blob_hashes))
        return _make_proc(0, b"http retry")

    monkeypatch.setattr(download_lifecycle.time, "sleep", fake_sleep)
    monkeypatch.setattr(download_lifecycle, "spawn_worker", fake_spawn_worker)

    assert real_register_worker(
        registry,
        key_a,
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

    assert sleep_calls == [0.05]
    assert spawned == [(["--repo-id", "Org/Model", "--variant", "Q4_K_M"], None)]
    metadata = registry.get_job_metadata(key_a)
    assert metadata is not None
    assert metadata.transport == download_registry.TRANSPORT_HTTP
    assert registry.get_job(key_a).state == "complete"
    assert registry.get_job(key_b).state == "complete"


def test_download_watcher_does_not_deadlock_when_sibling_xet_variants_both_fail(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(state_dir, "cache_root", lambda: tmp_path / "state")
    monkeypatch.setattr(download_lifecycle.threading, "Thread", _ImmediateThread)
    real_register_worker = download_lifecycle.register_worker
    registry = download_registry.DownloadRegistry()
    key_a = download_registry.normalize_job_key("Org/Model::Q4_K_M")
    key_b = download_registry.normalize_job_key("Org/Model::Q5_K_M")
    for key, variant, blob_hash in (
        (key_a, "Q4_K_M", "mainhash-a"),
        (key_b, "Q5_K_M", "mainhash-b"),
    ):
        assert registry.claim(
            key,
            download_registry.TRANSPORT_XET,
            repo_type = "model",
            repo_id = "Org/Model",
            variant = variant,
            blob_hashes = frozenset({blob_hash}),
            progress_blob_hashes = frozenset({blob_hash}),
        )

    spawned = []
    triggered_b = False

    def fake_spawn_worker(
        args,
        hf_token,
        *,
        use_xet,
        protected_blob_hashes = None,
    ):
        assert use_xet is False
        spawned.append(args)
        return _make_proc(0, b"http retry")

    def fake_sleep(_seconds):
        nonlocal triggered_b
        assert registry.get_job(key_a).state == "running"
        assert registry.get_job_metadata(key_a).transport == download_registry.TRANSPORT_XET
        assert not triggered_b
        triggered_b = True
        assert real_register_worker(
            registry,
            key_b,
            _make_proc(1, b"xet failed b"),
            hf_token = None,
            label = "Org/Model [Q5_K_M]",
            log_prefix = "Download",
            logger = logging.getLogger("test"),
            repo_type = "model",
            repo_id = "Org/Model",
            transport = download_registry.TRANSPORT_XET,
            watch_name = "model-watch-b",
        )

    monkeypatch.setattr(download_lifecycle.time, "sleep", fake_sleep)
    monkeypatch.setattr(download_lifecycle, "spawn_worker", fake_spawn_worker)

    assert real_register_worker(
        registry,
        key_a,
        _make_proc(1, b"xet failed a"),
        hf_token = None,
        label = "Org/Model [Q4_K_M]",
        log_prefix = "Download",
        logger = logging.getLogger("test"),
        repo_type = "model",
        repo_id = "Org/Model",
        transport = download_registry.TRANSPORT_XET,
        watch_name = "model-watch-a",
    )

    assert triggered_b
    assert spawned == [
        ["--repo-id", "Org/Model", "--variant", "Q5_K_M"],
        ["--repo-id", "Org/Model", "--variant", "Q4_K_M"],
    ]
    assert registry.get_job(key_a).state == "complete"
    assert registry.get_job(key_b).state == "complete"


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


def test_download_watcher_preserves_xet_marker_when_http_retry_is_cancelled_before_register(
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
        assert registry.get_job_metadata(key).transport == download_registry.TRANSPORT_HTTP
        assert registry.mark_pending_cancel(key, generation)
        return _make_proc(0, b"http retry")

    def fake_persist_cancel_marker(*args, **_kwargs):
        markers.append(args)

    monkeypatch.setattr(download_lifecycle, "spawn_worker", fake_spawn_worker)
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
    assert registry.get_job_metadata(key) is None


def test_download_watcher_honors_cancel_before_http_retry_spawn(monkeypatch, tmp_path):
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
    proc._waited = True
    markers = []

    def fake_drain_stderr_excerpt(_stream):
        assert (
            download_lifecycle.cancel_worker(
                registry,
                key,
                generation = generation,
                label = "Org/Model [Q4_K_M]",
                logger = logging.getLogger("test"),
            )
            == "cancelling"
        )
        return b"xet failed"

    def fake_spawn_worker(*_args, **_kwargs):
        raise AssertionError("cancelled retry should not spawn a worker")

    def fake_register_worker(*_args, **_kwargs):
        raise AssertionError("cancelled retry should not register a worker")

    def fake_persist_cancel_marker(*args, **_kwargs):
        markers.append(args)

    monkeypatch.setattr(download_lifecycle, "drain_stderr_excerpt", fake_drain_stderr_excerpt)
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


def test_download_watcher_preserves_xet_marker_when_http_retry_is_cancelled_after_register(
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
    markers = []

    class _CancellingProc:
        pid = 5252

        def __init__(self):
            self.stderr = io.BytesIO(b"cancelled")

        def wait(self, timeout = None):
            proc = registry.get_process(key)
            assert proc is self
            assert registry.request_cancel(key, self, generation)
            return -9

        def poll(self):
            return None

        def kill(self):
            raise AssertionError("registered retry worker should exit by cancellation")

    def fake_spawn_worker(*_args, **_kwargs):
        assert registry.get_job_metadata(key).transport == download_registry.TRANSPORT_HTTP
        return _CancellingProc()

    def fake_persist_cancel_marker(*args, **_kwargs):
        markers.append(args)

    monkeypatch.setattr(download_lifecycle, "spawn_worker", fake_spawn_worker)
    monkeypatch.setattr(download_registry, "persist_cancel_marker", fake_persist_cancel_marker)

    assert real_register_worker(
        registry,
        key,
        _make_proc(1, b"xet failed"),
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
    assert registry.get_job_metadata(key) is not None
    assert registry.get_job_metadata(key).transport == download_registry.TRANSPORT_HTTP


def test_http_retry_shutdown_and_breadcrumb_preserve_xet_marker(monkeypatch, tmp_path):
    monkeypatch.setattr(state_dir, "cache_root", lambda: tmp_path / "state")
    registry = download_registry.DownloadRegistry()
    key = download_registry.normalize_job_key("Org/Model::Q4_K_M")
    assert registry.claim(
        key,
        download_registry.TRANSPORT_HTTP,
        repo_type = "model",
        repo_id = "Org/Model",
        variant = "Q4_K_M",
        blob_hashes = frozenset({"mainhash"}),
        progress_blob_hashes = frozenset({"mainhash"}),
        cancel_marker_transport = download_registry.TRANSPORT_XET,
    )
    metadata = registry.get_job_metadata(key)
    assert metadata is not None
    assert metadata.transport == download_registry.TRANSPORT_HTTP
    assert metadata.cancel_marker_transport == download_registry.TRANSPORT_XET
    markers = []

    class _KillableProc:
        pid = 6262

        def __init__(self):
            self._rc = None
            self.killed = False

        def poll(self):
            return self._rc

        def kill(self):
            self.killed = True
            self._rc = -9

        def wait(self, timeout = None):
            return self._rc

    proc = _KillableProc()
    assert registry.register_process(key, proc)
    worker_files = list(state_dir.workers_dir().glob("*.json"))
    assert len(worker_files) == 1
    payload = json.loads(worker_files[0].read_text(encoding = "utf-8"))
    assert payload["transport"] == download_registry.TRANSPORT_HTTP
    assert payload["cancel_marker_transport"] == download_registry.TRANSPORT_XET

    def fake_persist_cancel_marker(*args, **_kwargs):
        markers.append(args)

    monkeypatch.setattr(download_registry, "persist_cancel_marker", fake_persist_cancel_marker)

    registry.terminate_all("download")

    assert proc.killed is True
    assert markers == [("model", "Org/Model", "Q4_K_M", download_registry.TRANSPORT_XET)]


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


def test_active_download_refs_include_waiting_http_retry(monkeypatch, tmp_path):
    monkeypatch.setattr(state_dir, "cache_root", lambda: tmp_path / "state")
    monkeypatch.setattr(download_lifecycle.threading, "Thread", _ImmediateThread)
    real_register_worker = download_lifecycle.register_worker
    registry = download_registry.DownloadRegistry()
    key_a = download_registry.normalize_job_key("Org/Model::Q4_K_M")
    key_b = download_registry.normalize_job_key("Org/Model::Q5_K_M")
    assert registry.claim(
        key_a,
        download_registry.TRANSPORT_XET,
        repo_type = "model",
        repo_id = "Org/Model",
        variant = "Q4_K_M",
        blob_hashes = frozenset({"mainhash-a"}),
        progress_blob_hashes = frozenset({"mainhash-a"}),
    )
    assert registry.claim(
        key_b,
        download_registry.TRANSPORT_XET,
        repo_type = "model",
        repo_id = "Org/Model",
        variant = "Q5_K_M",
        blob_hashes = frozenset({"mainhash-b"}),
        progress_blob_hashes = frozenset({"mainhash-b"}),
    )
    proc = _make_proc(1, b"xet failed")
    waiting_variants = []

    def fake_sleep(_seconds):
        # The Q4_K_M retry is released from the repo guard while it waits for the
        # slot; deletion stays blocked, so the listing must still surface it.
        assert registry.begin_delete("Org/Model", "Q4_K_M") is False
        refs = download_lifecycle.active_download_refs(registry, "Org/Model", with_variant = True)
        waiting_variants.append(sorted(ref.variant for ref in refs))
        registry.set_job(key_b, "complete")

    def fake_spawn_worker(
        args,
        hf_token,
        *,
        use_xet,
        protected_blob_hashes = None,
    ):
        assert use_xet is False
        return _make_proc(0, b"http retry")

    monkeypatch.setattr(download_lifecycle.time, "sleep", fake_sleep)
    monkeypatch.setattr(download_lifecycle, "spawn_worker", fake_spawn_worker)

    assert real_register_worker(
        registry,
        key_a,
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

    # While the HTTP retry waited for the slot, the listing must include both the
    # still-active sibling and the released retry so the frontend can adopt or
    # cancel that backend-running retry after a reload.
    assert waiting_variants == [["Q4_K_M", "Q5_K_M"]]
    assert registry.get_job(key_a).state == "complete"
