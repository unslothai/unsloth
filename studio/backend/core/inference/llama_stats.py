# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Translate llama-server's Prometheus /metrics into a periodic, vLLM-style
engine-stats log line (generation/prompt throughput, requests in flight).

llama-server already computes these (it needs `--metrics`); this lifts them
into Unsloth's structured log so the terminal shows serving health, not just
per-request access lines. Emitted only while there is activity.
"""

import os
import re
import threading
import time
import urllib.request

# Prometheus body lines: "llamacpp:<name>[{labels}] <value>" (skip "#" HELP/TYPE).
_METRIC_RE = re.compile(r"^llamacpp:(\w+)(?:\{[^}]*\})?\s+([0-9.eE+-]+)", re.MULTILINE)
_OFF = {"0", "false", "no", "off"}


class LlamaServerStatsLogger:
    """Daemon poller that logs vLLM-style engine stats from llama-server.

    Keeps retrying through transient scrape failures; the backend stops it via
    stop() on unload/reload, so a brief /metrics stall does not silence stats.
    """

    def __init__(
        self,
        base_url,
        logger,
        interval_s = 10.0,
    ):
        self._url = f"{base_url.rstrip('/')}/metrics"
        self._log = logger
        self._interval = max(1.0, float(interval_s))
        self._stop = threading.Event()
        self._thread = None

    def start(self):
        if self._thread is None:
            self._thread = threading.Thread(target = self._run, name = "llama-stats", daemon = True)
            self._thread.start()

    def stop(self):
        self._stop.set()

    def _scrape(self):
        try:
            with urllib.request.urlopen(self._url, timeout = 3) as r:
                if r.status != 200:
                    return None
                body = r.read().decode("utf-8", "replace")
        except Exception:
            return None
        out = {}
        for k, v in _METRIC_RE.findall(body):
            try:  # a malformed value must not kill the daemon thread
                out[k] = float(v)
            except ValueError:
                continue
        return out

    def _run(self):
        misses = 0
        prev = None  # (monotonic_t, tokens_predicted_total, prompt_tokens_total)
        while not self._stop.wait(self._interval):
            m = self._scrape()
            if not m:
                misses += 1
                if misses == 3:  # transient stall (load/GC); keep polling.
                    self._log.debug("engine_stats: /metrics scrape failing, still retrying")
                continue  # real shutdown is driven by stop() from _kill_process
            misses = 0
            # Generation tokens come from tokens_predicted_total (counter) and
            # predicted_tokens_seconds (gauge); n_decode_total counts
            # llama_decode() calls, not tokens, so it must not feed tok/s.
            now = time.monotonic()
            predicted = m.get("tokens_predicted_total", 0.0)
            prompt = m.get("prompt_tokens_total", 0.0)
            gen_delta = prompt_delta = 0.0
            if prev is not None and now > prev[0]:
                dt = now - prev[0]
                gen_delta = max(0.0, (predicted - prev[1]) / dt)
                prompt_delta = max(0.0, (prompt - prev[2]) / dt)
            prev = (now, predicted, prompt)
            # Prefer llama.cpp's own throughput gauges; fall back to the counter
            # delta for binaries that expose only the counters.
            gen_tps = m.get("predicted_tokens_seconds") or gen_delta
            prompt_tps = m.get("prompt_tokens_seconds") or prompt_delta
            running, waiting = (
                int(m.get("requests_processing", 0)),
                int(m.get("requests_deferred", 0)),
            )
            # Gate on real activity this tick so a stale gauge never logs at idle.
            if running or waiting or gen_delta or prompt_delta:
                self._log.info(
                    "engine_stats",
                    gen_tok_s = round(float(gen_tps), 1),
                    prompt_tok_s = round(float(prompt_tps), 1),
                    running = running,
                    waiting = waiting,
                )


def maybe_start_stats_logger(base_url, logger):
    """Start a stats logger unless UNSLOTH_STUDIO_ENGINE_STATS disables it."""
    if (os.environ.get("UNSLOTH_STUDIO_ENGINE_STATS", "1") or "").strip().lower() in _OFF:
        return None
    try:
        interval = float(os.environ.get("UNSLOTH_STUDIO_ENGINE_STATS_INTERVAL_S", "10"))
    except ValueError:
        interval = 10.0
    sl = LlamaServerStatsLogger(base_url, logger, interval)
    sl.start()
    return sl
