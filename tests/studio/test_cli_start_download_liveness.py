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
        log_chunk = 0,
        ready_at = None,
        tail = KEY_LINE,
    ):
        self.clock = FakeClock(STEP_S)
        self.log_path = None
        self.log_chunk = log_chunk
        self.downloaded_bytes = downloaded_bytes
        self.chunk_bytes = chunk_bytes
        self.ready_at = ready_at
        self.tail = tail
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
        if "gguf-variants" in url:
            return {
                "default_variant": "Q4_K_M",
                "variants": [{"quant": "Q4_K_M", "download_size_bytes": EXPECTED_BYTES}],
            }
        if "download-progress" in url:
            self.polls += 1
            self.downloaded_bytes += self.chunk_bytes
            return {
                "downloaded_bytes": self.downloaded_bytes,
                "expected_bytes": EXPECTED_BYTES,
                "progress": self.downloaded_bytes / EXPECTED_BYTES,
            }
        raise AssertionError(f"unexpected request: {method} {url}")

    def log_tail(
        self,
        path,
        lines = 20,
    ):
        # The real file the child writes to, so `_log_size` runs its own stat().
        self.log_path = path
        return self.tail

    def studio_healthy(
        self,
        base,
        timeout = 3.0,
    ):
        self.iterations += 1
        if self.log_chunk and self.log_path is not None:
            with open(self.log_path, "ab") as handle:
                handle.write(b"." * self.log_chunk)
        if self.ready_at is not None and self.iterations >= self.ready_at:
            self.tail = f"{KEY_LINE}Model loaded: {MODEL}\n"
            return True
        return False

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


def test_a_load_that_keeps_logging_survives_past_the_idle_cap(monkeypatch):
    harness = Harness(
        monkeypatch,
        downloaded_bytes = EXPECTED_BYTES,
        chunk_bytes = 0,
        log_chunk = 4096,
        ready_at = 40,
    )

    server = harness.start()

    assert server is harness.server
    assert harness.shutdowns == []
    assert harness.clock.elapsed > start_cli._SERVER_START_TIMEOUT_S
