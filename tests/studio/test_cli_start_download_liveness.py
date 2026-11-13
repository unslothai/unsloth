# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""`unsloth start` must not kill an auto-started server whose model is still downloading."""

from __future__ import annotations

from urllib.parse import urlencode

import pytest
import typer

import unsloth_cli.commands.start as start_cli


BASE = "http://127.0.0.1:8888"
MODEL = "unsloth/Qwen3-Coder-480B-A35B-Instruct-GGUF"
ADAPTER = "owner/coder-lora"
ADAPTER_BASE = "owner/coder-base"
ADAPTER_BYTES = 20 * 1024**2
KEY_LINE = f"{start_cli._START_API_KEY_PREFIX}sk-unsloth-test\n"
EXPECTED_BYTES = 500 * 1024**3
STEP_S = 120.0
# A loop that keeps resetting its deadline never returns; ~7 days of fake clock at STEP_S.
MAX_ITERATIONS = 5000


def health_poll_line(n):
    """What `LoggingMiddleware` logs for the `/api/health` GET this loop just made."""
    return (
        f"2026-09-08 03:{n // 60:02d}:{n % 60:02d} [info     ] request_completed"
        "              method=GET path=/api/health process_time_ms=0.4 status_code=200\n"
    )


def failed_health_poll_record(n):
    """A health probe that raises: the JSON record, then the traceback `with_readable_traceback`
    echoes on the lines after it, each carrying `loggers.config._TRACEBACK_ECHO_PREFIX`."""
    return (
        f'{{"event": "request_failed", "path": "/api/health", "status_code": 500, "n": {n}}}\n'
        "| Traceback (most recent call last):\n"
        '|   File "/opt/unsloth/studio/backend/main.py", line 1796, in health\n'
        "|     return await _hardware_snapshot()\n"
        "| RuntimeError: hardware probe timed out\n"
    )


def load_watchdog_heartbeat(n):
    """`start_watchdog`'s on_heartbeat -> `{"type": "status"}` -> the orchestrator's
    `Subprocess status` INFO line. Timer driven, so it keeps coming while a load is wedged."""
    return (
        f"2026-09-08 03:{n // 60:02d}:{n % 60:02d} [info     ] Subprocess status: still loading\n"
    )


class FakeClock:
    """monotonic() only moves when the readiness loop sleeps, so runs are deterministic."""

    def __init__(self, step):
        self.now = 1000.0
        self.start = 1000.0
        self.step = step

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.now += self.step

    @property
    def elapsed(self):
        return self.now - self.start


class FakePopen:
    def poll(self):
        return None


class Harness:
    def __init__(
        self,
        monkeypatch,
        *,
        downloaded_bytes = 0,
        chunk_bytes = 0,
        chatter = None,
        fail_every = 0,
        fail_first = 0,
        unmeasured_every = 0,
        unmeasured_grows = False,
        vanish_every = 0,
        rebound = False,
        ready_at = None,
        tail = KEY_LINE,
        model = MODEL,
    ):
        self.clock = FakeClock(STEP_S)
        self.log_path = None
        self.chatter = chatter
        self.fail_every = fail_every
        self.fail_first = fail_first
        self.failures = 0
        self.unmeasured_every = unmeasured_every
        self.unmeasured_grows = unmeasured_grows
        self.unmeasured = 0
        self.vanish_every = vanish_every
        self.vanished = 0
        self.rebound = rebound
        self.downloaded_bytes = downloaded_bytes
        self.chunk_bytes = chunk_bytes
        self.ready_at = ready_at
        self.tail = tail
        self.model = model
        self.base_polls = 0
        self.server = FakePopen()
        self.iterations = 0
        self.polls = 0
        self.shutdowns = []
        monkeypatch.setattr(start_cli, "time", self.clock)
        monkeypatch.setattr(start_cli, "_http_json", self.http_json)
        monkeypatch.setattr(start_cli, "_log_tail", self.log_tail)
        monkeypatch.setattr(start_cli, "_studio_healthy", self.studio_healthy)
        monkeypatch.setattr(start_cli, "_shutdown_server", self.shutdowns.append)
        monkeypatch.setattr(start_cli, "_auto_served_server", None)
        monkeypatch.setattr(start_cli.atexit, "register", lambda *a, **k: None)
        monkeypatch.setattr(start_cli.subprocess, "Popen", lambda *a, **k: self.server)

    def http_json(
        self,
        method,
        url,
        token,
        payload = None,
        timeout = 30,
        error = None,
    ):
        if url.endswith("/active-downloads"):
            if self.model != ADAPTER:
                return {"downloads": []}
            return {"downloads": [{"repo_id": ADAPTER_BASE, "owner": "load", "state": "running"}]}
        if "download-progress" in url and urlencode({"repo_id": ADAPTER}) in url:
            return {
                "downloaded_bytes": ADAPTER_BYTES,
                "completed_bytes": ADAPTER_BYTES,
                "expected_bytes": ADAPTER_BYTES,
                "progress": 1.0,
                "cache_measured": True,
            }
        if urlencode({"repo_id": ADAPTER_BASE}) in url:
            self.base_polls += 1
        if "gguf-variants" in url:
            return {
                "default_variant": "Q4_K_M",
                "variants": [{"quant": "Q4_K_M", "download_size_bytes": EXPECTED_BYTES}],
            }
        if "download-progress" in url:
            self.polls += 1
            if self.polls <= self.fail_first or (
                self.fail_every and self.polls % self.fail_every == 0
            ):
                self.failures += 1
                raise TimeoutError("the server took too long to answer")
            if self.rebound:
                if self.polls % 2 == 0:
                    return {
                        "downloaded_bytes": 12 * 1024**3,
                        "expected_bytes": EXPECTED_BYTES,
                        "progress": 0,
                        "cache_measured": True,
                    }
                self.unmeasured += 1
                return {
                    "downloaded_bytes": 5 * 1024**3,
                    "expected_bytes": EXPECTED_BYTES,
                    "progress": 0,
                    "cache_measured": False,
                }
            if self.vanish_every and self.polls % self.vanish_every == 0:
                # A cleanly vanished mount is a MEASURED zero: the count drops, then returns.
                self.vanished += 1
                return {
                    "downloaded_bytes": 0,
                    "expected_bytes": EXPECTED_BYTES,
                    "progress": 0,
                    "cache_measured": True,
                }
            if self.unmeasured_every and self.polls % self.unmeasured_every == 0:
                # Unfinished scan: 200, cache_measured false; a growing count is a lower bound.
                self.unmeasured += 1
                if self.unmeasured_grows:
                    self.downloaded_bytes += self.chunk_bytes
                return {
                    "downloaded_bytes": self.downloaded_bytes if self.unmeasured_grows else 0,
                    "expected_bytes": EXPECTED_BYTES,
                    "progress": 0,
                    "cache_measured": False,
                }
            self.downloaded_bytes += self.chunk_bytes
            return {
                "downloaded_bytes": self.downloaded_bytes,
                "expected_bytes": EXPECTED_BYTES,
                "progress": self.downloaded_bytes / EXPECTED_BYTES,
                "cache_measured": True,
            }
        raise AssertionError(f"unexpected request: {method} {url}")

    def log_tail(
        self,
        path,
        lines = 20,
    ):
        self.log_path = path
        return self.tail

    def studio_healthy(
        self,
        base,
        timeout = 3.0,
    ):
        self.iterations += 1
        assert self.iterations <= MAX_ITERATIONS, (
            f"readiness loop still running after {self.iterations} passes "
            f"({self.clock.elapsed:.0f}s of fake clock); its deadline never expires"
        )
        if self.chatter and self.log_path is not None:
            with open(self.log_path, "ab") as handle:
                handle.write(self.chatter(self.iterations).encode())
        if self.ready_at is not None and self.iterations >= self.ready_at:
            self.tail = f"{KEY_LINE}Model loaded: {self.model}\n"
            return True
        return False

    def start(self):
        return start_cli._start_studio_server(BASE, self.model, start_cli.LoadOptions())[1]


def test_a_live_download_survives_past_the_idle_cap(monkeypatch):
    harness = Harness(
        monkeypatch,
        chunk_bytes = 1024**3,
        ready_at = 40,
    )

    server = harness.start()

    assert server is harness.server
    assert harness.shutdowns == []
    assert harness.polls >= 40
    assert harness.clock.elapsed > start_cli._SERVER_START_TIMEOUT_S


def test_a_stalled_download_still_times_out(monkeypatch, capsys):
    harness = Harness(
        monkeypatch,
        downloaded_bytes = 12 * 1024**3,
        chunk_bytes = 0,
    )

    with pytest.raises(typer.Exit):
        harness.start()

    assert harness.shutdowns == [harness.server]
    message = capsys.readouterr().err
    assert f"made no progress for {start_cli._SERVER_START_TIMEOUT_S}s" in message
    assert harness.clock.elapsed < 2 * start_cli._SERVER_START_TIMEOUT_S


def test_a_server_that_never_downloads_still_times_out(monkeypatch, capsys):
    harness = Harness(monkeypatch, tail = "starting\n")

    with pytest.raises(typer.Exit):
        harness.start()

    assert harness.polls == 0
    assert harness.shutdowns == [harness.server]
    assert f"made no progress for {start_cli._SERVER_START_TIMEOUT_S}s" in capsys.readouterr().err
    assert harness.clock.elapsed < 2 * start_cli._SERVER_START_TIMEOUT_S


@pytest.mark.parametrize(
    "chatter",
    [
        pytest.param(health_poll_line, id = "the loop's own health poll"),
        pytest.param(failed_health_poll_record, id = "traceback echo lines"),
        pytest.param(load_watchdog_heartbeat, id = "load watchdog heartbeat"),
    ],
)
def test_a_wedged_server_that_keeps_writing_still_times_out(monkeypatch, capsys, chatter):
    # The log grows on a timer while nothing loads; only fresh download bytes move the deadline.
    harness = Harness(
        monkeypatch,
        downloaded_bytes = EXPECTED_BYTES,
        chunk_bytes = 0,
        chatter = chatter,
    )

    with pytest.raises(typer.Exit):
        harness.start()

    assert harness.shutdowns == [harness.server]
    assert f"made no progress for {start_cli._SERVER_START_TIMEOUT_S}s" in capsys.readouterr().err
    assert harness.clock.elapsed < 2 * start_cli._SERVER_START_TIMEOUT_S


def test_a_transient_progress_error_does_not_blind_the_loop(monkeypatch):
    harness = Harness(
        monkeypatch,
        chunk_bytes = 1024**3,
        fail_every = 3,
        ready_at = 40,
    )

    server = harness.start()

    assert server is harness.server
    assert harness.shutdowns == []
    assert harness.failures >= 10
    assert harness.clock.elapsed > start_cli._SERVER_START_TIMEOUT_S


def test_an_unmeasured_reading_is_not_progress(monkeypatch, capsys):
    # An unfinished scan answers 200 with zero bytes; believing it makes the next reading
    # look like fresh growth and renews the deadline forever.
    harness = Harness(
        monkeypatch,
        downloaded_bytes = 12 * 1024**3,
        chunk_bytes = 0,
        unmeasured_every = 2,
    )

    with pytest.raises(typer.Exit):
        harness.start()

    assert harness.unmeasured > 0
    assert harness.shutdowns == [harness.server]
    assert f"made no progress for {start_cli._SERVER_START_TIMEOUT_S}s" in capsys.readouterr().err
    assert harness.clock.elapsed < 2 * start_cli._SERVER_START_TIMEOUT_S


def test_polling_keeps_probing_after_a_long_burst_of_errors(monkeypatch):
    # A run of failures may back the reader off but must never retire it.
    harness = Harness(
        monkeypatch,
        chunk_bytes = 1024**3,
        fail_first = 6,
        ready_at = 40,
    )

    server = harness.start()

    assert server is harness.server
    assert harness.shutdowns == []
    assert harness.failures >= 6
    assert harness.downloaded_bytes > 0
    assert harness.clock.elapsed > start_cli._SERVER_START_TIMEOUT_S


def test_a_growing_unmeasured_reading_still_counts(monkeypatch):
    # `cache_measured: false` with a growing count is a lower bound, not an unknown.
    harness = Harness(
        monkeypatch,
        chunk_bytes = 1024**3,
        unmeasured_every = 1,
        unmeasured_grows = True,
        ready_at = 40,
    )

    server = harness.start()

    assert server is harness.server
    assert harness.shutdowns == []
    assert harness.unmeasured >= 40
    assert harness.clock.elapsed > start_cli._SERVER_START_TIMEOUT_S


def test_a_partial_scan_rebound_is_not_progress(monkeypatch, capsys):
    # A partial scan must not lower the baseline, or a flapping mount reads as growth.
    harness = Harness(monkeypatch, rebound = True)

    with pytest.raises(typer.Exit):
        harness.start()

    assert harness.unmeasured > 0
    assert harness.shutdowns == [harness.server]
    assert f"made no progress for {start_cli._SERVER_START_TIMEOUT_S}s" in capsys.readouterr().err
    assert harness.clock.elapsed < 2 * start_cli._SERVER_START_TIMEOUT_S


def test_a_vanished_cache_mount_is_not_progress(monkeypatch, capsys):
    # A vanished mount is a MEASURED zero; following it down makes each remount read as growth.
    harness = Harness(
        monkeypatch,
        downloaded_bytes = 12 * 1024**3,
        chunk_bytes = 0,
        vanish_every = 2,
    )

    with pytest.raises(typer.Exit):
        harness.start()

    assert harness.vanished > 0
    assert harness.shutdowns == [harness.server]
    assert f"made no progress for {start_cli._SERVER_START_TIMEOUT_S}s" in capsys.readouterr().err
    assert harness.clock.elapsed < 2 * start_cli._SERVER_START_TIMEOUT_S


def test_a_lora_base_model_download_counts_as_progress(monkeypatch):
    harness = Harness(monkeypatch, model = ADAPTER, chunk_bytes = 1024**3, ready_at = 40)

    server = harness.start()

    assert server is harness.server
    assert harness.shutdowns == []
    assert harness.base_polls >= 40
    assert harness.clock.elapsed > start_cli._SERVER_START_TIMEOUT_S


def test_a_stalled_lora_base_model_still_times_out(monkeypatch, capsys):
    harness = Harness(
        monkeypatch,
        model = ADAPTER,
        downloaded_bytes = 12 * 1024**3,
        chunk_bytes = 0,
    )

    with pytest.raises(typer.Exit):
        harness.start()

    assert harness.base_polls > 0
    assert harness.shutdowns == [harness.server]
    assert f"made no progress for {start_cli._SERVER_START_TIMEOUT_S}s" in capsys.readouterr().err
    assert harness.clock.elapsed < 2 * start_cli._SERVER_START_TIMEOUT_S
