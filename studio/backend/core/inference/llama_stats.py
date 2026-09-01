# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Translate llama-server's Prometheus /metrics into a periodic, vLLM-style
engine-stats log line (generation/prompt throughput, requests in flight).

llama-server already computes these (it needs `--metrics`); this lifts them
into Unsloth's structured log so the terminal shows serving health, not just
per-request access lines. Emitted only while there is activity.
"""

import math
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
        stall_timeout_s = 600.0,
    ):
        self._url = f"{base_url.rstrip('/')}/metrics"
        self._log = logger
        self._interval = max(1.0, float(interval_s))
        self._stop = threading.Event()
        self._thread = None
        # Stall reporting: a held slot that is not calling llama_decode() at all.
        self._stall_timeout = max(0.0, float(stall_timeout_s))
        self._last_decode = None
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

    def _stalled_for(self, now, running, decode_calls):
        """Seconds the engine has held a slot without calling llama_decode().

        Progress is n_decode_total, NOT the token counters. llama-server updates
        tokens_predicted_total once per generation, from callback_on_reset when
        the slot is released, and flushes prompt_tokens_total only on a decode
        that produced output. Both therefore sit still for the whole of a healthy
        long prefill and a healthy long decode, so treating them as a liveness
        signal would flag every slow generation. n_decode_total increments on
        every llama_decode() call, which is the thing that actually stops when
        the engine is wedged.

        Known blind spot: the counter's own help text excludes speculative and
        multimodal decoding, so a long image or audio encode can look static.
        That is one reason this only ever reports and never cancels anything.
        """
        if not running or decode_calls != self._last_decode:
            self._last_decode = decode_calls
            self._stall_since = now if running else None
            self._stall_reported = False  # progress re-arms the report
            return 0.0
        if self._stall_since is None:
            self._stall_since = now
            return 0.0
        return now - self._stall_since

    def _report_stall(self, running, waiting, stalled_for, decode_calls):
        """Log only. Cancelling a generation on this evidence is not safe.

        A held slot with no decode calls is the signature of a wedge, but the
        scrape cannot prove the engine is not about to resume, and it cannot see
        which of several in-flight generations owns the slot. Studio reaps a run
        that stops making progress from its own token stream instead, where the
        signal is per token and unambiguous. This exists so a wedge is visible in
        the log at all: the incident that prompted it ran 22 minutes without a
        single line above info.
        """
        self._stall_reported = True
        self._log.warning(
            "engine_no_decode_progress",
            running = running,
            waiting = waiting,
            stalled_s = round(stalled_for, 1),
            n_decode_total = decode_calls,
            detail = "llama-server holds a slot but has not called llama_decode(); "
            "a wedged engine looks like this, and so does a long multimodal encode",
        )

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
            # A held slot not calling llama_decode() is a wedge, and its only symptom
            # is an endless run of identical info lines. A build without n_decode_total
            # reads None and never "changes", accumulating the same way, so the message
            # is chosen at report time.
            decode_calls = m.get("n_decode_total")
            stalled_for = self._stalled_for(now, running, decode_calls)
            if self._stall_timeout and stalled_for >= self._stall_timeout:
                if decode_calls is None:
                    if not self._unmeasurable_reported:
                        self._unmeasurable_reported = True
                        self._log.warning(
                            "engine_progress_unmeasurable",
                            missing = "n_decode_total",
                            held_s = round(stalled_for, 1),
                            detail = "cannot tell a wedged engine from a working one; "
                            "llama-server /metrics lacks the decode counter",
                        )
                elif not self._stall_reported:
                    self._report_stall(running, waiting, stalled_for, decode_calls)
            # Gate on real activity this tick so a stale gauge never logs at idle.
            if running or waiting or gen_delta or prompt_delta:
                self._log.info(
                    "engine_stats",
                    gen_tok_s = round(float(gen_tps), 1),
                    prompt_tok_s = round(float(prompt_tps), 1),
                    running = running,
                    waiting = waiting,
                )


# A week already means "never" for a poll interval or a stall timeout. Bounded by
# threading.TIMEOUT_MAX as well, because the ceiling is platform specific and much lower
# than it looks: Linux accepts ~9.2e9 seconds, Windows about 49.7 days, since the timeout
# becomes a DWORD of milliseconds there. Picking a constant by hand got this wrong once
# already, so let the platform state its own limit.
_MAX_ENV_SECONDS = min(7.0 * 24.0 * 60.0 * 60.0, threading.TIMEOUT_MAX)


def _env_float(name, default, logger):
    """Seconds from the environment, rejecting anything that would silently do nothing.

    float() accepts non-finite text, and both spellings disable the reporting they were
    set to configure: max() drops nan so the stall line never arms, and an elapsed time
    can never reach inf.
    """
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    if not math.isfinite(value):
        logger.warning(
            "engine_stats_env_ignored",
            variable = name,
            value = raw,
            reason = "not a finite number",
        )
        return default
    if value > _MAX_ENV_SECONDS:
        # Event.wait() builds an absolute deadline, and one far enough out raises
        # "timestamp out of range for platform time_t" once the wait is entered, killing
        # the poll thread. Measured: a century still waits, 1e10 seconds does not.
        logger.warning(
            "engine_stats_env_clamped",
            variable = name,
            value = raw,
            applied_s = _MAX_ENV_SECONDS,
            reason = "longer than a timed wait can represent",
        )
        return _MAX_ENV_SECONDS
    return value


def maybe_start_stats_logger(base_url, logger):
    """Start a stats logger unless UNSLOTH_STUDIO_ENGINE_STATS disables it."""
    if (os.environ.get("UNSLOTH_STUDIO_ENGINE_STATS", "1") or "").strip().lower() in _OFF:
        return None
    interval = _env_float("UNSLOTH_STUDIO_ENGINE_STATS_INTERVAL_S", 10.0, logger)
    # Generously above any legitimate pause between decode calls; 0 silences the
    # stall line and keeps the poller as a pure stats logger.
    stall_timeout = _env_float("UNSLOTH_STUDIO_ENGINE_STALL_TIMEOUT_S", 600.0, logger)
    sl = LlamaServerStatsLogger(
        base_url,
        logger,
        interval,
        stall_timeout_s = stall_timeout,
    )
    sl.start()
    return sl
