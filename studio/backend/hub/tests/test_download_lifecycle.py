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


def test_completion_invalidates_inventory_before_publishing_state(monkeypatch):
    events = []

    class _Registry:
        def cancel_requested(self, _key):
            return False

        def drop_process(self, _key, _proc):
            return True

        def get_job_metadata(self, _key):
            return None

        def set_job(self, _key, state):
            events.append(state)

    monkeypatch.setattr(
        download_lifecycle.hf_cache_scan,
        "invalidate_hf_cache_scans",
        lambda: events.append("invalidate"),
    )

    assert (
        download_lifecycle.finalize_worker_exit(
            _Registry(),
            "org/data",
            _Proc(0),
            hf_token = None,
            label = "org/data",
            log_prefix = "Download",
            logger = logging.getLogger("test"),
            repo_type = "dataset",
            repo_id = "org/data",
        )
        == "complete"
    )
    assert events == ["invalidate", "complete"]


def test_xet_failure_retries_over_http_for_model_and_dataset(monkeypatch, tmp_path):
    monkeypatch.setattr(state_dir, "cache_root", lambda: tmp_path / "state")
    monkeypatch.setattr(download_lifecycle.threading, "Thread", _ImmediateThread)
    # _ImmediateThread mutates the stdlib threading module the shared Zoo watchdog also imports, so it
    # would run INLINE and block in Event.wait() before finalize_worker_exit could stop it. Stub the
    # seam instead.
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
            allow_ambient_token = True,
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


def test_a_stalled_xet_worker_respawns_over_xet_keeping_its_claim(monkeypatch, tmp_path):
    """End-to-end through the real reclaim: the recovery worker spawns with use_xet=True, the job
    keeps the XET transport (so the UI and the .transport marker stay truthful), and the blob
    metadata survives the re-claim."""
    monkeypatch.setattr(state_dir, "cache_root", lambda: tmp_path / "state")
    monkeypatch.setattr(download_lifecycle.threading, "Thread", _ImmediateThread)

    def _start(registry, key, proc, *, on_stall, **kwargs):
        on_stall("Download appears stalled (xet transport) -- no progress for 30s")
        return None

    monkeypatch.setattr(download_lifecycle, "_start_stall_watchdog", _start)
    monkeypatch.setattr(
        download_lifecycle, "_record_xet_failure", lambda *a: pytest.fail("charged")
    )
    register_worker = download_lifecycle.register_worker

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
    generation = registry.current_generation(key)
    spawned = []

    def fake_spawn(
        args,
        _token,
        *,
        use_xet,
        protected_blob_hashes = None,
        allow_ambient_token = True,
    ):
        spawned.append((args, use_xet))
        return _Proc(0)

    def fake_retry_register(*_args, **kwargs):
        assert kwargs["transport"] == download_registry.TRANSPORT_XET
        assert kwargs["xet_attempt"] == 2
        return True

    monkeypatch.setattr(download_lifecycle, "spawn_worker", fake_spawn)
    monkeypatch.setattr(download_lifecycle, "register_worker", fake_retry_register)
    assert register_worker(
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

    assert spawned == [(["--repo-id", "Org/Model"], True)]
    metadata = registry.get_job_metadata(key)
    assert metadata.transport == download_registry.TRANSPORT_XET
    assert metadata.blob_hashes == frozenset({"blob"})
    assert registry.current_generation(key) == generation


def test_an_unspawnable_xet_retry_falls_through_to_http(monkeypatch, tmp_path):
    """The extra Xet worker is a bonus rung: if it cannot be spawned at all, the download must
    still get the HTTP fallback it would have had without the retry, not end in "error"."""
    monkeypatch.setattr(state_dir, "cache_root", lambda: tmp_path / "state")
    monkeypatch.setattr(download_lifecycle.threading, "Thread", _ImmediateThread)

    spawned = []

    def flaky_spawn(
        args,
        _token,
        *,
        use_xet,
        protected_blob_hashes = None,
        **_kw,
    ):
        spawned.append(use_xet)
        if use_xet:
            raise OSError("cannot fork")
        return _Proc(0)

    monkeypatch.setattr(download_lifecycle, "spawn_worker", flaky_spawn)
    monkeypatch.setattr(download_lifecycle, "register_worker", lambda *a, **k: True)

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

    assert download_lifecycle._try_transport_retry(
        registry,
        key,
        hf_token = None,
        label = "Org/Model",
        log_prefix = "Download",
        logger = logging.getLogger("test"),
        repo_type = "model",
        repo_id = "Org/Model",
        watch_name = "model-watch",
        retry_transport = download_registry.TRANSPORT_XET,
        xet_attempt = 2,
    )
    assert spawned == [True, False], "the failed XET respawn must be followed by an HTTP one"


def test_a_verdict_carried_onto_the_http_rung_is_still_charged(monkeypatch, tmp_path):
    """That fallthrough is the one path that hands a held stall to an HTTP worker. HTTP completing
    says nothing about Xet, so clearing the verdict there would lose the only evidence a repeatedly
    stalling machine ever produces."""
    monkeypatch.setattr(state_dir, "cache_root", lambda: tmp_path / "state")
    monkeypatch.setattr(download_lifecycle.threading, "Thread", _ImmediateThread)
    verdict = "Download appears stalled (xet transport) -- no progress for 30s"

    def _start(registry, key, proc, *, on_stall, **_kwargs):
        on_stall(verdict)
        return None

    spawned = []

    def flaky_spawn(
        args,
        _token,
        *,
        use_xet,
        protected_blob_hashes = None,
        **_kwargs,
    ):
        spawned.append(use_xet)
        if use_xet:
            raise OSError("cannot fork")
        return _Proc(0)

    monkeypatch.setattr(download_lifecycle, "_start_stall_watchdog", _start)
    monkeypatch.setattr(download_lifecycle, "spawn_worker", flaky_spawn)

    recorded = []
    monkeypatch.setattr(download_lifecycle, "_record_xet_failure", lambda m, _l: recorded.append(m))

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
    # Before the verdict: a double whose signature has fallen behind spawn_worker raises TypeError,
    # which reads as a spawn failure and still records the verdict, staying green.
    assert spawned == [True, False], "the HTTP rung never ran, so the verdict proves nothing"
    assert recorded == [verdict], "a real Xet stall was dropped when HTTP finished the download"


def test_http_failure_remains_terminal(monkeypatch, tmp_path):
    monkeypatch.setattr(state_dir, "cache_root", lambda: tmp_path / "state")
    monkeypatch.setattr(download_lifecycle.threading, "Thread", _ImmediateThread)
    # _ImmediateThread mutates the stdlib threading module the shared Zoo watchdog also imports, so stub
    # the seam instead.
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
    """A fully cached repo exits 0 without touching the network (the UI's re-download on an
    up-to-date model), and recording that as a Xet success wipes a correctly earned demotion."""
    recorded = _run_completed_xet_worker(
        monkeypatch, tmp_path, bytes_before = 5_000, bytes_after = 5_000
    )
    assert recorded == []


def test_a_real_xet_transfer_does_clear_the_failure_streak(monkeypatch, tmp_path):
    """The streak must still reset on a job that actually moved bytes, or "two in a row" is wrong."""
    recorded = _run_completed_xet_worker(monkeypatch, tmp_path, bytes_before = 0, bytes_after = 5_000)
    assert recorded == [True]


def test_a_sibling_variants_bytes_do_not_count_as_this_jobs_progress(monkeypatch, tmp_path):
    """Two same-transport GGUF variants of one repo may run concurrently over one blobs/ dir, and a
    repo-wide measure credited a cached no-op worker with its sibling's bytes, clearing a legitimate
    stall streak and flipping an already demoted verdict back to Xet."""
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


def _trip_xet_worker(
    monkeypatch,
    tmp_path,
    message,
    rc = 1,
    xet_attempt = 2,
    retries = None,
):
    """Run a Xet worker whose watchdog trips with *message*; return the health failures recorded.

    *xet_attempt* defaults to the LAST attempt of the default budget, since that is where the Xet
    phase ends and a held-back verdict is finally reported. Pass 1 to exercise the deferral.
    *retries* collects ``(retry_transport, xet_attempt, pending_xet_failure)`` for each respawn.
    """
    monkeypatch.setattr(state_dir, "cache_root", lambda: tmp_path / "state")
    monkeypatch.setattr(download_lifecycle.threading, "Thread", _ImmediateThread)

    def _start(registry, key, proc, *, on_stall, **kwargs):
        on_stall(message)
        return None

    monkeypatch.setattr(download_lifecycle, "_start_stall_watchdog", _start)
    monkeypatch.setattr(download_lifecycle, "spawn_worker", lambda *a, **k: _Proc(0))

    def _fake_register(*_args, **kwargs):
        if retries is not None:
            retries.append(
                (
                    kwargs.get("transport"),
                    kwargs.get("xet_attempt"),
                    kwargs.get("pending_xet_failure"),
                )
            )
        return True

    monkeypatch.setattr(download_lifecycle, "register_worker", _fake_register)

    recorded = []
    monkeypatch.setattr(download_lifecycle, "_record_xet_failure", lambda m, _l: recorded.append(m))

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
        _Proc(rc, b"killed" if rc else b""),
        hf_token = None,
        label = "Org/Model",
        log_prefix = "Download",
        logger = logging.getLogger("test"),
        repo_type = "model",
        repo_id = "Org/Model",
        transport = download_registry.TRANSPORT_XET,
        watch_name = "model-watch",
        xet_attempt = xet_attempt,
    )
    return recorded


def test_a_data_phase_stall_is_recorded_against_the_machine(monkeypatch, tmp_path):
    """A frozen partial with bytes already flowing is genuinely Xet misbehaving -- charged once the
    Xet phase is out of attempts."""
    monkeypatch.setattr(
        download_lifecycle, "_REAL_REGISTER", download_lifecycle.register_worker, raising = False
    )

    assert _trip_xet_worker(
        monkeypatch,
        tmp_path,
        "Download appears stalled (xet transport) -- no progress for 30s",
    ), "a real data-phase stall must still be recorded"


def test_a_first_stall_buys_another_xet_worker_and_records_nothing(monkeypatch, tmp_path):
    """A wedged Xet transfer usually clears on a fresh process, so the first data-phase stall
    respawns over XET. Nothing is recorded yet: if the retry succeeds the stall was noise, and
    charging both attempts would let ONE download hit the two-failure demotion threshold."""
    monkeypatch.setattr(
        download_lifecycle, "_REAL_REGISTER", download_lifecycle.register_worker, raising = False
    )
    retries = []
    verdict = "Download appears stalled (xet transport) -- no progress for 30s"
    assert (
        _trip_xet_worker(monkeypatch, tmp_path, verdict, xet_attempt = 1, retries = retries) == []
    ), "the first stall must not be charged to the machine"
    assert retries == [(download_registry.TRANSPORT_XET, 2, verdict)]


def test_the_last_xet_stall_falls_back_to_http_and_charges_once(monkeypatch, tmp_path):
    """Out of Xet attempts: the transport changes, and the single accumulated verdict is reported."""
    monkeypatch.setattr(
        download_lifecycle, "_REAL_REGISTER", download_lifecycle.register_worker, raising = False
    )
    retries = []
    recorded = _trip_xet_worker(
        monkeypatch,
        tmp_path,
        "Download appears stalled (xet transport) -- no progress for 30s",
        xet_attempt = 2,
        retries = retries,
    )
    assert len(recorded) == 1
    # The HTTP rung carries no pending verdict: it was just recorded.
    assert retries == [(download_registry.TRANSPORT_HTTP, 2, None)]


def test_a_pre_byte_trip_never_buys_another_xet_worker(monkeypatch, tmp_path):
    """Retrying "did not start" would buy a second full 600s connect window before HTTP ever
    starts, and that trip is as likely slow metadata as a broken Xet."""
    monkeypatch.setattr(
        download_lifecycle, "_REAL_REGISTER", download_lifecycle.register_worker, raising = False
    )
    retries = []
    _trip_xet_worker(
        monkeypatch,
        tmp_path,
        "Download did not start (xet transport) -- no data after 600s",
        xet_attempt = 1,
        retries = retries,
    )
    assert retries == [(download_registry.TRANSPORT_HTTP, 1, None)]


def test_the_attempts_knob_of_one_restores_the_straight_to_http_ladder(monkeypatch, tmp_path):
    monkeypatch.setenv("UNSLOTH_XET_ATTEMPTS", "1")
    monkeypatch.setattr(
        download_lifecycle, "_REAL_REGISTER", download_lifecycle.register_worker, raising = False
    )
    retries = []
    recorded = _trip_xet_worker(
        monkeypatch,
        tmp_path,
        "Download appears stalled (xet transport) -- no progress for 30s",
        xet_attempt = 1,
        retries = retries,
    )
    assert len(recorded) == 1
    assert retries == [(download_registry.TRANSPORT_HTTP, 1, None)]


def test_a_pre_byte_trip_does_not_poison_the_machines_health_record(monkeypatch, tmp_path):
    """Two recorded failures pin this machine to HTTP for 24h, and "did not start" means not one
    byte arrived, as likely slow metadata, a queue of HEADs or a cache lock as a broken Xet. The
    HTTP retry still happens either way."""
    monkeypatch.setattr(
        download_lifecycle, "_REAL_REGISTER", download_lifecycle.register_worker, raising = False
    )
    assert (
        _trip_xet_worker(
            monkeypatch,
            tmp_path,
            "Download did not start (xet transport) -- no data after 600s",
        )
        == []
    )


def test_a_post_byte_hang_between_files_is_recorded(monkeypatch, tmp_path):
    """ "did not resume" fires only after bytes HAVE flowed, so it is real Xet evidence, and it is
    the shape this worker hangs in most often since snapshot_download owns no partial between
    files. An earlier allow-list keyed on "no progress" silently dropped it."""
    monkeypatch.setattr(
        download_lifecycle, "_REAL_REGISTER", download_lifecycle.register_worker, raising = False
    )
    assert _trip_xet_worker(
        monkeypatch,
        tmp_path,
        "Download did not resume (xet transport) -- no data for 600s",
    ), "a post-byte Xet hang was not recorded against the machine"


def test_the_xet_baseline_is_sampled_before_the_worker_spawns(monkeypatch, tmp_path):
    """A fast child can finalize its blobs while we are still registering the process, so a later
    baseline shows no growth for a real transfer: the streak is never cleared and two stalls either
    side of it read as consecutive, demoting Auto for 24h."""
    monkeypatch.setattr(state_dir, "cache_root", lambda: tmp_path / "state")
    monkeypatch.setattr(download_lifecycle.threading, "Thread", _ImmediateThread)
    monkeypatch.setattr(download_lifecycle, "_start_stall_watchdog", lambda *a, **k: None)

    order = []
    sizes = iter([0, 5_000])

    monkeypatch.setattr(
        download_lifecycle,
        "_job_bytes_on_disk",
        lambda *a, **k: (order.append("sample"), next(sizes, 5_000))[1],
    )
    recorded = []
    monkeypatch.setattr(download_lifecycle, "_record_xet_success", lambda _l: recorded.append(True))

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

    def _spawn():
        order.append("spawn")
        return _Proc(0)

    download_lifecycle.launch_worker(
        registry,
        key,
        spawn = _spawn,
        hf_token = None,
        label = "Org/Model",
        log_prefix = "Download",
        logger = logging.getLogger("test"),
        repo_type = "model",
        repo_id = "Org/Model",
        transport = download_registry.TRANSPORT_XET,
        watch_name = "model-watch",
    )

    assert order[0] == "sample" and order[1] == "spawn", order
    assert recorded == [True], "a real transfer did not clear the failure streak"


def test_a_stall_verdict_racing_a_completed_worker_is_not_recorded(monkeypatch, tmp_path):
    """The watchdog appends its verdict BEFORE the kill lands, so a worker that completed in that
    instant would be charged a failure it did not earn, and on the completed path that also skips
    the success-clearing: two streak steps the wrong way from one race."""
    monkeypatch.setattr(
        download_lifecycle, "_REAL_REGISTER", download_lifecycle.register_worker, raising = False
    )
    assert (
        _trip_xet_worker(
            monkeypatch,
            tmp_path,
            "Download appears stalled (xet transport) -- no progress for 30s",
            rc = 0,
        )
        == []
    ), "a completed worker was charged a stall it raced"
