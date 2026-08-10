# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Translate llama-server's Prometheus /metrics into a periodic, vLLM-style
engine-stats log line (generation/prompt throughput, requests in flight).

llama-server already computes these (it needs `--metrics`); this lifts them
into Unsloth's structured log so the terminal shows serving health, not just
per-request access lines. Emitted only while there is activity.

Activity is logged as a burst, not as a tick every `interval_s`. A generation
running for a minute produced six near-identical lines ("about 250 tok/s") plus a
trailing one at `running=0`, which is repetition rather than information. Each
burst now logs its first tick, a heartbeat while it is still going, and one
closing line carrying the peak and mean of everything that was collapsed.
"""

import os
import re
import threading
import time
import urllib.request

# Prometheus body lines: "llamacpp:<name>[{labels}] <value>" (skip "#" HELP/TYPE).
_METRIC_RE = re.compile(r"^llamacpp:(\w+)(?:\{[^}]*\})?\s+([0-9.eE+-]+)", re.MULTILINE)
_OFF = {"0", "false", "no", "off"}


def _verbose_logging_requested() -> bool:
    """`unsloth studio --verbose` zeroes both access-log windows; reuse that signal
    so one flag restores every engine_stats tick along with every access line."""
    def _zero(name: str) -> bool:
        raw = (os.environ.get(name) or "").strip()
        try:
            return raw != "" and int(raw) <= 0
        except ValueError:
            return False
    return _zero("UNSLOTH_STUDIO_ACCESS_LOG_DEDUP_MS") and _zero(
        "UNSLOTH_STUDIO_ACCESS_LOG_POLL_DEDUP_MS"
    )


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
        heartbeat_s = 30.0,
    ):
        self._url = f"{base_url.rstrip('/')}/metrics"
        self._log = logger
        self._interval = max(1.0, float(interval_s))
        # While a burst is running, log at most this often. 0 restores a line per tick.
        self._heartbeat = max(0.0, float(heartbeat_s))
        self._verbose = _verbose_logging_requested()
        self._stop = threading.Event()
        self._thread = None
        # Burst accumulator: None when idle.
        self._burst = None

    def _begin_burst(self, now):
        self._burst = {
            "ticks": 0, "last_log": now,
            "peak_gen": 0.0, "sum_gen": 0.0,
            "peak_prompt": 0.0, "sum_prompt": 0.0,
            "peak_running": 0, "peak_waiting": 0,
            "collapsed": 0,
        }

    def _accumulate(self, gen_tps, prompt_tps, running, waiting):
        b = self._burst
        b["ticks"] += 1
        b["sum_gen"] += gen_tps
        b["sum_prompt"] += prompt_tps
        b["peak_gen"] = max(b["peak_gen"], gen_tps)
        b["peak_prompt"] = max(b["peak_prompt"], prompt_tps)
        b["peak_running"] = max(b["peak_running"], running)
        b["peak_waiting"] = max(b["peak_waiting"], waiting)

    def _close_burst(self):
        """Emit one line summarising everything the burst collapsed, then go idle."""
        b, self._burst = self._burst, None
        if b is None or b["ticks"] == 0:
            return
        # A single-tick burst already logged everything it had; nothing to summarise.
        if b["collapsed"] == 0:
            return
        self._log.info(
            "engine_stats",
            gen_tok_s = round(b["sum_gen"] / b["ticks"], 1),
            prompt_tok_s = round(b["sum_prompt"] / b["ticks"], 1),
            running = 0,
            waiting = 0,
            burst_ticks = b["ticks"],
            burst_collapsed = b["collapsed"],
            peak_gen_tok_s = round(b["peak_gen"], 1),
            peak_prompt_tok_s = round(b["peak_prompt"], 1),
            peak_running = b["peak_running"],
            peak_waiting = b["peak_waiting"],
        )

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
            active = bool(running or waiting or gen_delta or prompt_delta)
            gen_tps = round(float(gen_tps), 1)
            prompt_tps = round(float(prompt_tps), 1)

            if not active:
                # First idle tick after activity closes the burst with its summary.
                self._close_burst()
                continue

            if self._verbose or self._heartbeat <= 0:
                self._log.info(
                    "engine_stats",
                    gen_tok_s = gen_tps, prompt_tok_s = prompt_tps,
                    running = running, waiting = waiting,
                )
                continue

            starting = self._burst is None
            if starting:
                self._begin_burst(now)
            self._accumulate(gen_tps, prompt_tps, running, waiting)

            # Log the first tick of the burst, then at most one per heartbeat. The
            # ticks in between are the repetition this exists to remove; they stay
            # visible at debug, and their peak and mean reach the closing line.
            if starting or (now - self._burst["last_log"]) >= self._heartbeat:
                self._burst["last_log"] = now
                self._log.info(
                    "engine_stats",
                    gen_tok_s = gen_tps, prompt_tok_s = prompt_tps,
                    running = running, waiting = waiting,
                )
            else:
                self._burst["collapsed"] += 1
                self._log.debug(
                    "engine_stats (collapsed)",
                    gen_tok_s = gen_tps, prompt_tok_s = prompt_tps,
                    running = running, waiting = waiting,
                )


def maybe_start_stats_logger(base_url, logger):
    """Start a stats logger unless UNSLOTH_STUDIO_ENGINE_STATS disables it."""
    if (os.environ.get("UNSLOTH_STUDIO_ENGINE_STATS", "1") or "").strip().lower() in _OFF:
        return None
    try:
        interval = float(os.environ.get("UNSLOTH_STUDIO_ENGINE_STATS_INTERVAL_S", "10"))
    except ValueError:
        interval = 10.0
    # How often a still-running burst may log. 0 keeps the old line-per-tick behaviour.
    try:
        heartbeat = float(os.environ.get("UNSLOTH_STUDIO_ENGINE_STATS_LOG_EVERY_S", "30"))
    except ValueError:
        heartbeat = 30.0
    sl = LlamaServerStatsLogger(base_url, logger, interval, heartbeat)
    sl.start()
    return sl
