# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import io
import logging

import pytest

from hub.services import download_lifecycle
from hub.utils import download_registry, state_dir


class _Proc:
    pid = 4242

    def __init__(
        self,
        rc,
        stderr = b"",
    ):
        self.rc = rc
        self.stderr = io.BytesIO(stderr)
        self.waited = False

    def poll(self):
        return self.rc if self.waited else None

    def wait(self, timeout = None):
        self.waited = True
        return self.rc

    def kill(self):
        pass


class _ImmediateThread:
    def __init__(self, *, target, **_kwargs):
        self.target = target

    def start(self):
        self.target()


def test_resolve_effective_use_xet(monkeypatch):
    for requested, unavailable_reason, expected in (
        (False, "unused", False),
        (True, None, True),
        (True, "hf_xet is not installed", False),
    ):
        monkeypatch.setattr(
            download_lifecycle.download_registry,
            "download_transport_unavailable_reason",
            lambda _transport, reason = unavailable_reason: reason,
        )
        assert download_lifecycle.resolve_effective_use_xet(requested) is expected


def test_xet_failure_retries_over_http_for_model_and_dataset(monkeypatch, tmp_path):
    monkeypatch.setattr(state_dir, "cache_root", lambda: tmp_path / "state")
    monkeypatch.setattr(download_lifecycle.threading, "Thread", _ImmediateThread)
    # _ImmediateThread mutates the stdlib threading module, which the shared Zoo watchdog
    # also imports, so the watchdog would run INLINE and block in Event.wait() before
    # finalize_worker_exit could ever set its stop flag. Stub the seam instead.
    monkeypatch.setattr(download_lifecycle, "_start_stall_watchdog", lambda *a, **k: None)
    register_worker = download_lifecycle.register_worker

    for repo_type, repo_id, variant, expected_args in (
        ("model", "Org/Model", "Q4_K_M", ["--repo-id", "Org/Model", "--variant", "Q4_K_M"]),
        ("dataset", "Org/Data", None, ["--repo-id", "Org/Data", "--dataset"]),
    ):
        registry = download_registry.DownloadRegistry()
        key = download_registry.normalize_job_key(f"{repo_id}::{variant}" if variant else repo_id)
        assert registry.claim(
            key,
            download_registry.TRANSPORT_XET,
            repo_type = repo_type,
            repo_id = repo_id,
            variant = variant,
            blob_hashes = frozenset({"blob"}),
        )[0]
        generation = registry.current_generation(key)
        spawned = []

        def fake_spawn(
            args,
            _token,
            *,
            use_xet,
            protected_blob_hashes = None,
        ):
            spawned.append((args, use_xet, protected_blob_hashes))
            return _Proc(0)

        def fake_retry_register(*_args, **kwargs):
            assert kwargs["transport"] == download_registry.TRANSPORT_HTTP
            return True

        monkeypatch.setattr(download_lifecycle, "spawn_worker", fake_spawn)
        monkeypatch.setattr(download_lifecycle, "register_worker", fake_retry_register)
        assert register_worker(
            registry,
            key,
            _Proc(1, b"xet failed"),
            hf_token = None,
            label = repo_id,
            log_prefix = "Download",
            logger = logging.getLogger("test"),
            repo_type = repo_type,
            repo_id = repo_id,
            transport = download_registry.TRANSPORT_XET,
            watch_name = f"{repo_type}-watch",
        )

        metadata = registry.get_job_metadata(key)
        assert spawned == [(expected_args, False, None)]
        assert metadata.transport == download_registry.TRANSPORT_HTTP
        assert metadata.blob_hashes == frozenset({"blob"})
        assert registry.current_generation(key) == generation


def test_http_failure_remains_terminal(monkeypatch, tmp_path):
    monkeypatch.setattr(state_dir, "cache_root", lambda: tmp_path / "state")
    monkeypatch.setattr(download_lifecycle.threading, "Thread", _ImmediateThread)
    # _ImmediateThread mutates the stdlib threading module, which the shared Zoo watchdog
    # also imports, so the watchdog would run INLINE and block in Event.wait() before
    # finalize_worker_exit could ever set its stop flag. Stub the seam instead.
    monkeypatch.setattr(download_lifecycle, "_start_stall_watchdog", lambda *a, **k: None)
    register_worker = download_lifecycle.register_worker
    registry = download_registry.DownloadRegistry()
    key = download_registry.normalize_repo_key("Org/Data")
    assert registry.claim(
        key,
        download_registry.TRANSPORT_HTTP,
        repo_type = "dataset",
        repo_id = "Org/Data",
    )[0]
    monkeypatch.setattr(
        download_lifecycle,
        "register_worker",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("HTTP failures must not retry")
        ),
    )
    assert register_worker(
        registry,
        key,
        _Proc(1, b"http failed"),
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


def _run_completed_xet_worker(monkeypatch, tmp_path, *, bytes_before, bytes_after):
    """Drive one successful Xet worker to completion and report whether success was recorded."""
    monkeypatch.setattr(state_dir, "cache_root", lambda: tmp_path / "state")
    monkeypatch.setattr(download_lifecycle.threading, "Thread", _ImmediateThread)
    monkeypatch.setattr(download_lifecycle, "_start_stall_watchdog", lambda *a, **k: None)

    sizes = iter([bytes_before, bytes_after])
    monkeypatch.setattr(
        download_lifecycle, "_repo_bytes_on_disk", lambda *a, **k: next(sizes, bytes_after)
    )
    recorded = []
    monkeypatch.setattr(
        download_lifecycle, "_record_xet_success", lambda _logger: recorded.append(True)
    )

    registry = download_registry.DownloadRegistry()
    key = download_registry.normalize_job_key("Org/Model")
    assert registry.claim(
        key,
        download_registry.TRANSPORT_XET,
        repo_type = "model",
        repo_id = "Org/Model",
        variant = None,
        blob_hashes = frozenset({"blob"}),
    )[0]

    download_lifecycle.register_worker(
        registry,
        key,
        _Proc(0),
        hf_token = None,
        label = "Org/Model",
        log_prefix = "Download",
        logger = logging.getLogger("test"),
        repo_type = "model",
        repo_id = "Org/Model",
        transport = download_registry.TRANSPORT_XET,
        watch_name = "model-watch",
    )
    return recorded


def test_a_cached_xet_job_does_not_clear_the_failure_streak(monkeypatch, tmp_path):
    """A fully cached repo exits 0 without touching the network.

    Recording that as a Xet success wipes a correctly earned demotion, putting a machine that
    genuinely stalls back on Xet. Reachable from the UI's re-download action on an up-to-date model.
    """
    recorded = _run_completed_xet_worker(
        monkeypatch, tmp_path, bytes_before = 5_000, bytes_after = 5_000
    )
    assert recorded == []


def test_a_real_xet_transfer_does_clear_the_failure_streak(monkeypatch, tmp_path):
    """The streak must still reset on a job that actually moved bytes, or "two in a row" is wrong."""
    recorded = _run_completed_xet_worker(monkeypatch, tmp_path, bytes_before = 0, bytes_after = 5_000)
    assert recorded == [True]


def test_a_sibling_variants_bytes_do_not_count_as_this_jobs_progress(monkeypatch, tmp_path):
    """Two same-transport GGUF variants of one repo may run concurrently and share one blobs/ dir.

    A repo-wide measure credited a cached no-op worker with its sibling's bytes, which clears a
    legitimate stall streak and flips an already demoted verdict back to Xet.
    """
    seen = []

    def _fake_completed(
        repo_type,
        repo_id,
        blob_hashes,
        *,
        root = None,
    ):
        seen.append(frozenset(blob_hashes))
        return 1_000  # this variant's own blobs never grow: it was already cached

    monkeypatch.setattr(download_registry, "completed_blob_bytes", _fake_completed)
    monkeypatch.setattr(
        download_lifecycle,
        "_repo_bytes_on_disk",
        lambda *a, **k: pytest.fail("a variant job must not use the repo-wide measure"),
    )
    monkeypatch.setattr(state_dir, "cache_root", lambda: tmp_path / "state")
    monkeypatch.setattr(download_lifecycle.threading, "Thread", _ImmediateThread)
    monkeypatch.setattr(download_lifecycle, "_start_stall_watchdog", lambda *a, **k: None)

    recorded = []
    monkeypatch.setattr(
        download_lifecycle, "_record_xet_success", lambda _logger: recorded.append(True)
    )

    registry = download_registry.DownloadRegistry()
    key = download_registry.normalize_job_key("Org/Model::Q4_K_M")
    assert registry.claim(
        key,
        download_registry.TRANSPORT_XET,
        repo_type = "model",
        repo_id = "Org/Model",
        variant = "Q4_K_M",
        blob_hashes = frozenset({"mine"}),
    )[0]

    download_lifecycle.register_worker(
        registry,
        key,
        _Proc(0),
        hf_token = None,
        label = "Org/Model",
        log_prefix = "Download",
        logger = logging.getLogger("test"),
        repo_type = "model",
        repo_id = "Org/Model",
        transport = download_registry.TRANSPORT_XET,
        watch_name = "model-watch",
    )

    assert recorded == [], "a cached variant was credited with a sibling's bytes"
    assert seen and all(h == frozenset({"mine"}) for h in seen), seen


def _trip_xet_worker(monkeypatch, tmp_path, message):
    """Run a Xet worker whose watchdog trips with *message*; return the health failures recorded."""
    monkeypatch.setattr(state_dir, "cache_root", lambda: tmp_path / "state")
    monkeypatch.setattr(download_lifecycle.threading, "Thread", _ImmediateThread)

    def _start(registry, key, proc, *, on_stall, **kwargs):
        on_stall(message)          # the watchdog tripped
        return None

    monkeypatch.setattr(download_lifecycle, "_start_stall_watchdog", _start)
    monkeypatch.setattr(download_lifecycle, "spawn_worker", lambda *a, **k: _Proc(0))
    monkeypatch.setattr(download_lifecycle, "register_worker", lambda *a, **k: True)

    recorded = []
    monkeypatch.setattr(
        download_lifecycle, "_record_xet_failure", lambda m, _l: recorded.append(m)
    )

    registry = download_registry.DownloadRegistry()
    key = download_registry.normalize_job_key("Org/Model")
    assert registry.claim(
        key,
        download_registry.TRANSPORT_XET,
        repo_type = "model",
        repo_id = "Org/Model",
        variant = None,
        blob_hashes = frozenset({"blob"}),
    )[0]

    _register = download_lifecycle.__dict__["register_worker"]
    _real = getattr(download_lifecycle, "_REAL_REGISTER", None)
    (_real or _register)(
        registry,
        key,
        _Proc(1, b"killed"),
        hf_token = None,
        label = "Org/Model",
        log_prefix = "Download",
        logger = logging.getLogger("test"),
        repo_type = "model",
        repo_id = "Org/Model",
        transport = download_registry.TRANSPORT_XET,
        watch_name = "model-watch",
    )
    return recorded


def test_a_data_phase_stall_is_recorded_against_the_machine(monkeypatch, tmp_path):
    """A frozen partial with bytes already flowing is genuinely Xet misbehaving."""
    monkeypatch.setattr(download_lifecycle, "_REAL_REGISTER", download_lifecycle.register_worker,
                        raising = False)

    assert _trip_xet_worker(
        monkeypatch, tmp_path,
        "Download appears stalled (xet transport) -- no progress for 30s",
    ), "a real data-phase stall must still be recorded"


def test_a_pre_byte_trip_does_not_poison_the_machines_health_record(monkeypatch, tmp_path):
    """Two recorded failures pin this machine to HTTP for 24h, so only a real stall may count.

    A connect-phase trip means no byte ever arrived, which is as likely to be slow metadata, a long
    queue of HEADs, or a cache lock as a broken Xet. The HTTP retry still happens either way.
    """
    monkeypatch.setattr(download_lifecycle, "_REAL_REGISTER", download_lifecycle.register_worker,
                        raising = False)
    for message in (
        "Download did not start (xet transport) -- no data after 600s",
        "Download did not resume (xet transport) -- no data for 600s",
    ):
        assert _trip_xet_worker(monkeypatch, tmp_path, message) == [], message
