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
import json
import urllib.request

# Prometheus body lines: "llamacpp:<name>[{labels}] <value>" (skip "#" HELP/TYPE).
_METRIC_RE = re.compile(r"^llamacpp:(\w+)(?:\{[^}]*\})?\s+([0-9.eE+-]+)", re.MULTILINE)
_OFF = {"0", "false", "no", "off"}


def fetch_llama_slots(base_url, timeout_s = 3.0, headers = None):
    """One ``GET /slots`` read as a list, or None if it could not be read.

    ``headers`` carries the backend's ``Authorization`` when the load was launched with
    ``--api-key`` (``UNSLOTH_DIRECT_STREAM=1``). llama.cpp exempts only ``/health`` and
    ``/v1/health`` from the key check, so an unauthenticated read of ``/slots`` answers
    401 and the ``except`` below turns that into None -- "cannot tell" -- which silently
    switches the whole exact-residency probe off in a supported mode.

    Added against the advice in the original design, which said to reuse the /metrics
    scraper and NOT add a slots poller. That advice was written before the residue was
    understood: /metrics reports requests_processing and token counters but nothing about
    cells still held by IDLE slots, and llama.cpp keeps a slot's prompt cache after its
    request finishes. Measured 2026-09-01, one idle slot held 16383 of a 16384 cache
    while the scheduler believed it was nearly empty. This is the only endpoint that can
    say so.

    None means "cannot tell" -- endpoint disabled, older build, socket error -- and must
    never be read as "the cache is empty".
    """
    url = f"{str(base_url).rstrip('/')}/slots"
    try:
        with urllib.request.urlopen(
            urllib.request.Request(url, headers = dict(headers or {})), timeout = timeout_s
        ) as r:
            if r.status != 200:
                return None
            payload = json.loads(r.read().decode("utf-8", "replace"))
    except Exception:
        return None
    return payload if isinstance(payload, list) else None


def erase_llama_slot(
    base_url,
    slot_id,
    timeout_s = 3.0,
    headers = None,
) -> int:
    """Drop one idle slot's cached prompt. Returns tokens erased, 0 on any failure.

    Cheaper than preempting: the cache belongs to a request that has already finished, so
    this costs a future prefix-cache hit rather than a running conversation's progress.
    """
    url = f"{str(base_url).rstrip('/')}/slots/{int(slot_id)}?action=erase"
    try:
        # Authorized for the same reason the read above is: a 401 here returns 0 tokens
        # erased, so a paused slot's cells are never released and the waiter it was freed
        # for waits out its deadline.
        request = urllib.request.Request(
            url, method = "POST", data = b"", headers = dict(headers or {})
        )
        with urllib.request.urlopen(request, timeout = timeout_s) as r:
            if r.status != 200:
                return 0
            payload = json.loads(r.read().decode("utf-8", "replace"))
    except Exception:
        return 0
    try:
        return max(0, int(payload.get("n_erased") or 0))
    except (AttributeError, TypeError, ValueError):
        return 0


def scrape_llama_metrics(base_url, timeout_s = 3.0):
    """One /metrics read as a {name: float} dict, or None if it could not be read.

    Split out of the daemon's own scrape so a caller needing a single sample (the
    preemption reclaim barrier) reuses this parser instead of adding a second one, or a
    ``GET /slots`` poller. None covers every reason the read did not happen: no
    ``--metrics``, a server still starting, a socket error. Callers must treat that as
    "cannot tell", never as "nothing is running".
    """
    url = f"{str(base_url).rstrip('/')}/metrics"
    try:
        with urllib.request.urlopen(url, timeout = timeout_s) as r:
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
        self._base_url = base_url.rstrip("/")
        self._url = f"{self._base_url}/metrics"
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
        return scrape_llama_metrics(self._base_url)

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
            self._stall_reported = False
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
                if misses == 3:  # transient stall (load/GC); keep polling
                    self._log.debug("engine_stats: /metrics scrape failing, still retrying")
                continue  # real shutdown is driven by stop() from _kill_process
            misses = 0
            # Generation tokens come from tokens_predicted_total (counter) and predicted_tokens_seconds (gauge);
            # n_decode_total counts llama_decode() calls, not tokens, so it must not feed tok/s.
            now = time.monotonic()
            predicted = m.get("tokens_predicted_total", 0.0)
            prompt = m.get("prompt_tokens_total", 0.0)
            gen_delta = prompt_delta = 0.0
            if prev is not None and now > prev[0]:
                dt = now - prev[0]
                gen_delta = max(0.0, (predicted - prev[1]) / dt)
                prompt_delta = max(0.0, (prompt - prev[2]) / dt)
            prev = (now, predicted, prompt)
            # Prefer llama.cpp's own throughput gauges; fall back to the counter delta for binaries that expose only the
            # counters.
            gen_tps = m.get("predicted_tokens_seconds") or gen_delta
            prompt_tps = m.get("prompt_tokens_seconds") or prompt_delta
            running, waiting = (
                int(m.get("requests_processing", 0)),
                int(m.get("requests_deferred", 0)),
            )
            # a build without n_decode_total reads None and never "changes", accumulating the same way
            # A held slot not calling llama_decode() is a wedge, and its only symptom is an endless run of identical
            # info lines. A build without n_decode_total reads None and never "changes", accumulating the same way, so
            # the message is chosen at report time.
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


# bounded by threading.TIMEOUT_MAX as well
# A week already means "never" for a poll interval or a stall timeout. Bounded by threading.TIMEOUT_MAX as well, because
# the ceiling is platform specific and much lower than it looks: Linux accepts ~9.2e9 seconds, Windows about 49.7 days,
# since the timeout becomes a DWORD of milliseconds there. Picking a constant by hand got this wrong once already, so
# let the platform state its own limit.
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
        # Event.wait() builds an absolute deadline
        # Event.wait() builds an absolute deadline, and one far enough out raises "timestamp out of range for platform
        # time_t" once the wait is entered, killing the poll thread. Measured: a century still waits, 1e10 seconds does
        # not.
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
    # Generously above any legitimate pause between decode calls; 0 silences the stall line and keeps the poller as a
    # pure stats logger.
    stall_timeout = _env_float("UNSLOTH_STUDIO_ENGINE_STALL_TIMEOUT_S", 600.0, logger)
    sl = LlamaServerStatsLogger(
        base_url,
        logger,
        interval,
        stall_timeout_s = stall_timeout,
    )
    sl.start()
    return sl
