# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""`unsloth start` must not kill an auto-started server whose model is still downloading."""

from __future__ import annotations

import pytest
import typer

import unsloth_cli.commands.start as start_cli


BASE = "http://127.0.0.1:8888"
MODEL = "unsloth/Qwen3-Coder-480B-A35B-Instruct-GGUF"
KEY_LINE = f"{start_cli._START_API_KEY_PREFIX}sk-unsloth-test\n"
EXPECTED_BYTES = 500 * 1024**3
STEP_S = 120.0
# A loop that keeps resetting its deadline never returns, so stop it and say so rather
# than hanging the suite. At STEP_S this is ~7 days of fake wall clock.
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
        healthy = False,
        startup_key = None,
        startup_key_at = None,
        marker_at = None,
        ready_tail = None,
        flap_every = 0,
        step = STEP_S,
    ):
        self.clock = FakeClock(step)
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
        self.healthy = healthy
        self.startup_key = startup_key
        self.startup_key_at = startup_key_at
        self.marker_at = marker_at
        self.flap_every = flap_every
        self.ready_tail = ready_tail or f"{KEY_LINE}Model loaded: {MODEL}\n"
        self.mints = 0
        self.server = FakePopen()
        self.iterations = 0
        self.polls = 0
        self.shutdowns = []
        monkeypatch.setattr(start_cli, "time", self.clock)
        monkeypatch.setattr(start_cli, "_http_json", self.http_json)
        monkeypatch.setattr(start_cli, "_log_tail", self.log_tail)
        monkeypatch.setattr(start_cli, "_studio_healthy", self.studio_healthy)
        monkeypatch.setattr(start_cli, "_startup_api_key", self.startup_api_key)
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
                # Two roots, one intermittently unreadable and nothing downloading: the
                # partial scan sees less than the complete one, over and over.
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
                # A cache mount that disappears cleanly is a MEASURED absence -- no scan
                # error -- so the count drops to zero and returns on the next remount.
                self.vanished += 1
                return {
                    "downloaded_bytes": 0,
                    "expected_bytes": EXPECTED_BYTES,
                    "progress": 0,
                    "cache_measured": True,
                }
            if self.unmeasured_every and self.polls % self.unmeasured_every == 0:
                # A scan the server could not finish: 200, cache_measured false. Zero bytes
                # is the unreadable-root case; a growing count is the readable-root one,
                # which `snapshot_progress.py` documents as a real lower bound.
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
        # The real file the child writes to, so `_ServerLogProgress` reads it for real.
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
        if self.flap_every and self.iterations % self.flap_every == 0:
            return False
        if self.marker_at is not None and self.iterations >= self.marker_at:
            self.tail = KEY_LINE
        if self.ready_at is not None and self.iterations >= self.ready_at:
            self.tail = self.ready_tail
            return True
        return self.healthy

    def startup_api_key(self, base):
        self.mints += 1
        # Auth that only settles some way past the health gate.
        if self.startup_key_at is not None and self.iterations < self.startup_key_at:
            return None
        return self.startup_key

    def start(self):
        return start_cli._start_studio_server(BASE, MODEL, start_cli.LoadOptions())


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
    # The transfer outlives the cap in wall clock: the case that used to be killed.
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
    # Every one of these keeps the log growing on a timer while nothing loads, so log
    # growth cannot stand in for progress: only fresh download bytes may move the deadline.
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
    # One slow reading used to disable the reader for good, which would now leave a live
    # download with no signal at all and kill it at the cap.
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
    # `snapshot_progress_response` answers 200 with zero bytes when it cannot finish the
    # cache scan. Believing it would drop the count to zero, so the next real reading of
    # the same cached bytes would read as fresh growth and renew the deadline forever.
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
    # Bytes are the only thing separating a live transfer from a wedged server, so a run
    # of failures may back the reader off but must never retire it: the download it would
    # abandon is the one this whole path exists to keep alive. Six in a row is past the
    # point where the reader used to disable itself for the rest of the startup.
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
    assert harness.downloaded_bytes > 0  # it recovered and saw the transfer again
    assert harness.clock.elapsed > start_cli._SERVER_START_TIMEOUT_S


def test_a_growing_unmeasured_reading_still_counts(monkeypatch):
    # One cache root unreadable while the download lands in another: the backend answers
    # `cache_measured: false` with the readable root's real, growing byte count. That is a
    # lower bound, not an unknown, so rejecting it would kill a live transfer at the cap.
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
    assert harness.unmeasured >= 40  # every reading came back unmeasured
    assert harness.clock.elapsed > start_cli._SERVER_START_TIMEOUT_S


def test_a_partial_scan_rebound_is_not_progress(monkeypatch, capsys):
    # Nothing is downloading; one cache root simply comes and goes. Letting the partial
    # scan lower the baseline would turn the next complete scan of the very same bytes
    # into growth, and a wedged server would be renewed for as long as the mount flaps.
    harness = Harness(monkeypatch, rebound = True)

    with pytest.raises(typer.Exit):
        harness.start()

    assert harness.unmeasured > 0
    assert harness.shutdowns == [harness.server]
    assert f"made no progress for {start_cli._SERVER_START_TIMEOUT_S}s" in capsys.readouterr().err
    assert harness.clock.elapsed < 2 * start_cli._SERVER_START_TIMEOUT_S


def test_a_vanished_cache_mount_is_not_progress(monkeypatch, capsys):
    # Nothing is downloading; a cache mount just comes and goes. Its absence is reported
    # as a MEASURED zero, so a baseline that followed readings down would read every
    # remount as fresh growth and keep a wedged server waiting for as long as it flaps.
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


def test_an_older_child_is_polled_with_a_minted_key(monkeypatch):
    harness = Harness(
        monkeypatch,
        tail = "starting\n",
        healthy = True,
        startup_key = "sk-unsloth-minted",
        chunk_bytes = 1024**3,
        ready_at = 40,
        ready_tail = f"API Key: sk-unsloth-old\nModel loaded: {MODEL}\n",
    )

    server = harness.start()

    assert server is harness.server
    assert harness.shutdowns == []
    assert harness.mints == 1
    assert harness.polls >= 39
    assert harness.clock.elapsed > start_cli._SERVER_START_TIMEOUT_S


def test_an_older_child_with_no_mintable_key_still_times_out(monkeypatch, capsys):
    harness = Harness(monkeypatch, tail = "starting\n", healthy = True)

    with pytest.raises(typer.Exit):
        harness.start()

    assert harness.mints > 0
    assert harness.polls == 0
    assert harness.shutdowns == [harness.server]
    assert f"made no progress for {start_cli._SERVER_START_TIMEOUT_S}s" in capsys.readouterr().err


def test_a_minted_key_does_not_retire_the_early_key_marker(monkeypatch):
    harness = Harness(
        monkeypatch,
        tail = "starting\n",
        healthy = True,
        startup_key = "sk-unsloth-minted",
        chunk_bytes = 1024**3,
        marker_at = 3,
        ready_at = 40,
    )

    server = harness.start()

    assert server is harness.server
    assert harness.mints == 1
    assert harness.iterations >= 40


def test_a_ready_banner_is_not_mistaken_for_a_child_that_needs_a_key(monkeypatch):
    harness = Harness(
        monkeypatch,
        tail = f"API Key: sk-unsloth-old\nModel loaded: {MODEL}\n",
        healthy = True,
        startup_key = "sk-unsloth-minted",
    )

    server = harness.start()

    assert server is harness.server
    assert harness.mints == 0
    assert harness.polls == 0


def test_a_marker_one_poll_behind_health_does_not_mint(monkeypatch):
    # The child prints the marker just after its own health gate opens, so the loop can
    # see a healthy server one pass before the line lands. That is a normal launch, not
    # an older child, and it must not cost an extra key.
    harness = Harness(
        monkeypatch,
        tail = "starting\n",
        healthy = True,
        startup_key = "sk-unsloth-minted",
        chunk_bytes = 1024**3,
        marker_at = 1,
        ready_at = 40,
    )

    server = harness.start()

    assert server is harness.server
    assert harness.mints == 0
    assert harness.polls >= 38


def test_minting_is_retried_at_a_slow_cadence_not_every_poll(monkeypatch, capsys):
    # A server that never mints must not be asked once per sleep for 15 minutes, and
    # a handful of early failures must not retire the attempt for the whole window.
    harness = Harness(monkeypatch, tail = "starting\n", healthy = True, step = 2.0)

    with pytest.raises(typer.Exit):
        harness.start()

    polls = harness.iterations
    assert polls > 400, f"the loop only ran {polls} passes, so nothing was throttled"
    # Spelled out rather than read off the module so an implementation without the
    # cadence fails the rate assertion instead of erroring on a missing attribute.
    ceiling = harness.clock.elapsed / 30.0 + 2
    assert 3 < harness.mints <= ceiling, f"{harness.mints} mint attempts over {polls} passes"
    assert start_cli._KEY_MINT_RETRY_S == 30.0


def test_auth_that_settles_after_the_health_gate_still_starts_progress(monkeypatch):
    # `/api/health` can open before `/api/auth` is usable. Minting fails until it
    # settles; the download behind it must still be tracked instead of timing out.
    harness = Harness(
        monkeypatch,
        tail = "starting\n",
        healthy = True,
        startup_key = "sk-unsloth-minted",
        startup_key_at = 100,
        chunk_bytes = 1024**3,
        ready_at = 500,
        step = 2.0,
    )

    server = harness.start()

    assert server is harness.server
    assert harness.shutdowns == []
    assert harness.polls > 0
    assert harness.clock.elapsed > start_cli._SERVER_START_TIMEOUT_S


def test_a_health_probe_that_times_out_under_load_does_not_retire_minting(monkeypatch):
    # Health is polled with a 3s timeout while the disk is saturated by the download,
    # so it can miss a pass. A missed pass must not mean the loop never mints.
    harness = Harness(
        monkeypatch,
        tail = "starting\n",
        healthy = True,
        startup_key = "sk-unsloth-minted",
        chunk_bytes = 1024**3,
        flap_every = 2,
        ready_at = 41,
    )

    server = harness.start()

    assert server is harness.server
    assert harness.shutdowns == []
    assert harness.mints == 1
    assert harness.polls > 0
