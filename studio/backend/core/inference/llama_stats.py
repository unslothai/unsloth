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
        stall_timeout_s = 180.0,
        on_stall = None,
    ):
        self._url = f"{base_url.rstrip('/')}/metrics"
        self._log = logger
        self._interval = max(1.0, float(interval_s))
        self._stop = threading.Event()
        self._thread = None
        # Stall watchdog: a held slot whose cumulative counters never advance.
        self._stall_timeout = max(0.0, float(stall_timeout_s))
        self._on_stall = on_stall
        self._last_counters = None
        self._stall_since = None
        self._stall_reported = False
        self._unmeasurable_reported = False

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

    def _stalled_for(self, now, running, predicted, prompt):
        """Seconds the engine has held work without advancing either counter.

        Progress is measured on the cumulative counters, never on the rate
        gauges: llama-server can keep reporting a stale-but-nonzero
        predicted_tokens_seconds while decode is wedged, so a rate-based check
        would miss exactly the case this exists to catch. Prefill counts as
        progress -- a large prompt legitimately generates no tokens for a long
        time -- so both counters must be static to call it a stall.
        """
        counters = (predicted, prompt)
        if not running or counters != self._last_counters:
            self._last_counters = counters
            self._stall_since = now if running else None
            self._stall_reported = False  # progress re-arms the watchdog
            return 0.0
        if self._stall_since is None:
            self._stall_since = now
            return 0.0
        return now - self._stall_since

    def _report_stall(self, running, waiting, stalled_for, predicted, prompt):
        self._stall_reported = True
        self._log.warning(
            "engine_stall_detected",
            running = running,
            waiting = waiting,
            stalled_s = round(stalled_for, 1),
            tokens_predicted_total = predicted,
            prompt_tokens_total = prompt,
        )
        if self._on_stall is None:
            return
        try:  # a failing reap hook must not kill the daemon thread
            self._on_stall(
                running = running,
                waiting = waiting,
                stalled_s = stalled_for,
            )
        except Exception as exc:
            self._log.warning("engine_stall_reap_failed", error = repr(exc))

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
            # A slot held with both counters frozen is a wedge, not slow decode:
            # without this the engine stays "generating" forever and the only
            # symptom is an endless run of identical info lines.
            #
            # Only act when progress is actually measurable. A build that does
            # not export these counter names reads 0.0 through .get() forever,
            # which is indistinguishable from a wedge -- reaping on that would
            # kill healthy generations, so say so once and never reap instead.
            measurable = (
                "tokens_predicted_total" in m and "prompt_tokens_total" in m
            )
            if not measurable:
                if not self._unmeasurable_reported:
                    self._unmeasurable_reported = True
                    self._log.warning(
                        "engine_progress_unmeasurable",
                        missing = sorted(
                            {"tokens_predicted_total", "prompt_tokens_total"} - set(m)
                        ),
                        detail = "stall watchdog disabled; llama-server /metrics lacks the token counters",
                    )
            else:
                stalled_for = self._stalled_for(now, running, predicted, prompt)
                if (
                    self._stall_timeout
                    and stalled_for >= self._stall_timeout
                    and not self._stall_reported
                ):
                    self._report_stall(running, waiting, stalled_for, predicted, prompt)
            # Gate on real activity this tick so a stale gauge never logs at idle.
            if running or waiting or gen_delta or prompt_delta:
                self._log.info(
                    "engine_stats",
                    gen_tok_s = round(float(gen_tps), 1),
                    prompt_tok_s = round(float(prompt_tps), 1),
                    running = running,
                    waiting = waiting,
                )


def maybe_start_stats_logger(base_url, logger, on_stall = None):
    """Start a stats logger unless UNSLOTH_STUDIO_ENGINE_STATS disables it."""
    if (os.environ.get("UNSLOTH_STUDIO_ENGINE_STATS", "1") or "").strip().lower() in _OFF:
        return None
    try:
        interval = float(os.environ.get("UNSLOTH_STUDIO_ENGINE_STATS_INTERVAL_S", "10"))
    except ValueError:
        interval = 10.0
    # Generously above any legitimate prefill; 0 disables the watchdog and keeps
    # the poller as a pure logger.
    try:
        stall_timeout = float(os.environ.get("UNSLOTH_STUDIO_ENGINE_STALL_TIMEOUT_S", "180"))
    except ValueError:
        stall_timeout = 180.0
    sl = LlamaServerStatsLogger(
        base_url,
        logger,
        interval,
        stall_timeout_s = stall_timeout,
        on_stall = on_stall,
    )
    sl.start()
    return sl
