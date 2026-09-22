# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import logging
import os
import time

from hub.services import download_lifecycle
from hub.utils import download_heartbeat, download_registry
from hub.workers import hf_download

VERDICT = "Download appears stalled (xet transport) -- no progress for 30s"


class _Proc:
    pid = 4242

    def __init__(
        self,
        heartbeat,
        rc = None,
    ):
        self.args = ["python", "--heartbeat", str(heartbeat)]
        self.killed = False
        self.rc = rc

    def poll(self):
        return self.rc

    def kill(self):
        self.killed = True


class _Stop:
    def __init__(self):
        self.stopped = False

    def set(self):
        self.stopped = True


def _watchdog(monkeypatch, proc, on_stall):
    import utils.hf_xet_fallback as shim

    started = []

    def fake_start(*, on_stall, **_kwargs):
        stop = _Stop()
        started.append((on_stall, stop))
        return stop

    monkeypatch.setattr(shim, "start_watchdog", fake_start)
    registry = download_registry.DownloadRegistry()
    key = download_registry.normalize_job_key("Org/Model")
    handle = download_lifecycle._start_stall_watchdog(
        registry,
        key,
        proc,
        repo_type = "model",
        repo_id = "Org/Model",
        label = "Org/Model",
        log_prefix = "Download",
        logger = logging.getLogger("test"),
        on_stall = on_stall,
    )
    return handle, started


def test_a_verdict_is_overruled_while_the_worker_still_receives(monkeypatch, tmp_path):
    beat = tmp_path / "beat"
    beat.write_text("100")
    proc = _Proc(beat)
    stalled = []
    _handle, started = _watchdog(monkeypatch, proc, stalled.append)
    beat.write_text("250")
    started[0][0](VERDICT)
    assert not proc.killed
    assert stalled == []
    assert len(started) == 2
    started[1][0](VERDICT)
    assert proc.killed
    assert stalled == [VERDICT]
    assert len(started) == 2


def test_a_heartbeat_older_than_the_stall_window_does_not_save_the_worker(monkeypatch, tmp_path):
    beat = tmp_path / "beat"
    beat.write_text("100")
    proc = _Proc(beat)
    stalled = []
    _handle, started = _watchdog(monkeypatch, proc, stalled.append)
    beat.write_text("250")
    stale = time.time() - 600
    os.utime(beat, (stale, stale))
    started[0][0](VERDICT)
    assert proc.killed
    assert stalled == [VERDICT]
    assert len(started) == 1


def test_a_failed_re_arm_lets_the_verdict_stand(monkeypatch, tmp_path):
    import utils.hf_xet_fallback as shim

    beat = tmp_path / "beat"
    beat.write_text("100")
    proc = _Proc(beat)
    stalled = []
    calls = []

    def fake_start(*, on_stall, **_kwargs):
        calls.append(on_stall)
        if len(calls) > 1:
            raise RuntimeError("no watchdog")
        return _Stop()

    monkeypatch.setattr(shim, "start_watchdog", fake_start)
    registry = download_registry.DownloadRegistry()
    key = download_registry.normalize_job_key("Org/Model")
    download_lifecycle._start_stall_watchdog(
        registry,
        key,
        proc,
        repo_type = "model",
        repo_id = "Org/Model",
        label = "Org/Model",
        log_prefix = "Download",
        logger = logging.getLogger("test"),
        on_stall = stalled.append,
    )
    beat.write_text("250")
    calls[0](VERDICT)
    assert proc.killed
    assert stalled == [VERDICT]


def test_a_verdict_with_no_heartbeat_kills_as_before(monkeypatch, tmp_path):
    proc = _Proc(tmp_path / "missing")
    stalled = []
    _handle, started = _watchdog(monkeypatch, proc, stalled.append)
    started[0][0](VERDICT)
    assert proc.killed
    assert stalled == [VERDICT]
    assert len(started) == 1


def test_stopping_the_handle_stops_the_re_armed_watchdog(monkeypatch, tmp_path):
    beat = tmp_path / "beat"
    beat.write_text("1")
    proc = _Proc(beat)
    handle, started = _watchdog(monkeypatch, proc, lambda _m: None)
    beat.write_text("2")
    started[0][0](VERDICT)
    handle.set()
    assert started[1][1].stopped
    beat.write_text("3")
    started[1][0](VERDICT)
    assert not proc.killed
    assert started[2][1].stopped


def test_the_worker_progress_class_reports_transfer_bytes(tmp_path):
    beat = tmp_path / "beat"
    cls = hf_download._progress_class(str(beat))
    bar = cls(total = 10, disable = True)
    bar.update(4)
    bar.update(-3)
    bar.close()
    assert download_heartbeat.read(str(beat)) == 4
    assert hf_download._progress_class(None) is None


def test_the_stall_window_follows_the_environment(monkeypatch):
    monkeypatch.delenv("UNSLOTH_XET_STALL_TIMEOUT", raising = False)
    assert download_lifecycle._stall_window() == 30.0
    monkeypatch.setenv("UNSLOTH_XET_STALL_TIMEOUT", "12")
    assert download_lifecycle._stall_window() == 12.0
    monkeypatch.setenv("UNSLOTH_XET_STALL_TIMEOUT", "junk")
    assert download_lifecycle._stall_window() == 30.0


def test_heartbeat_writer_throttles_and_reader_rejects_garbage(tmp_path):
    beat = tmp_path / "beat"
    writer = download_heartbeat.HeartbeatWriter(str(beat), interval = 3600.0)
    writer.add(5)
    writer.add(7)
    assert download_heartbeat.read(str(beat)) == 5
    beat.write_text("not a number")
    assert download_heartbeat.read(str(beat)) is None
    assert download_heartbeat.read(None) is None


def test_worker_files_are_removed_once_the_worker_exits(tmp_path):
    beat = tmp_path / "beat"
    beat.write_text("1")
    (tmp_path / "beat.tmp").write_text("")
    proc = _Proc(beat)
    download_lifecycle._cleanup_worker_files(proc)
    assert beat.exists()
    proc.rc = 0
    download_lifecycle._cleanup_worker_files(proc)
    assert not beat.exists()
    assert not (tmp_path / "beat.tmp").exists()


def test_only_xet_workers_get_a_heartbeat(monkeypatch, tmp_path):
    spawned = []
    real_popen = download_lifecycle.subprocess.Popen

    def fake_popen(args, **kwargs):
        if "hub.workers.hf_download" not in args:
            return real_popen(args, **kwargs)
        spawned.append(args)
        return _Proc(tmp_path / "unused", rc = 0)

    monkeypatch.setattr(download_lifecycle.subprocess, "Popen", fake_popen)
    monkeypatch.setattr("huggingface_hub.utils.get_token_to_send", lambda token: None)
    monkeypatch.setattr("tempfile.tempdir", str(tmp_path))
    download_lifecycle.spawn_worker(
        ["--repo-id", "Org/Model"], None, use_xet = True, allow_ambient_token = False
    )
    download_lifecycle.spawn_worker(
        ["--repo-id", "Org/Model"], None, use_xet = False, allow_ambient_token = False
    )
    assert "--heartbeat" in spawned[0]
    assert spawned[0][spawned[0].index("--heartbeat") + 1].startswith(str(tmp_path))
    assert "--heartbeat" not in spawned[1]
