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
        headers = None,
    ):
        self._url = f"{base_url.rstrip('/')}/metrics"
        self._headers = dict(headers or {})
        # No HTTPSHandler: it builds an SSL context this loopback scrape never needs.
        self._opener = urllib.request.OpenerDirector()
        for handler in (
            urllib.request.ProxyHandler({}),
            urllib.request.HTTPHandler(),
            urllib.request.HTTPDefaultErrorHandler(),
            urllib.request.HTTPErrorProcessor(),
        ):
            self._opener.add_handler(handler)
        self._log = logger
        self._interval = max(1.0, float(interval_s))
        self._stop = threading.Event()
        self._thread = None
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
            # No proxy: the bearer must only ever go to the loopback server.
            with self._opener.open(
                urllib.request.Request(self._url, headers = self._headers), timeout = 3
            ) as r:
                if r.status != 200:
                    return None
                body = r.read().decode("utf-8", "replace")
        except Exception:
            return None
        out = {}
        for k, v in _METRIC_RE.findall(body):
            try:  # a malformed value must not kill the daemon thread
                value = float(v)
            except ValueError:
                continue
            # float() turns huge digit strings into inf without raising.
            if math.isfinite(value):
                out[k] = value
        return out

    @staticmethod
    def _prompt_rate(base, tokens, seconds):
        """Baseline held until both move: /metrics rounds to six digits, so one can cross a boundary
        first."""
        if base is None or tokens < base[0] or seconds < base[1]:
            return 0.0, (tokens, seconds)
        d_tokens, d_seconds = tokens - base[0], seconds - base[1]
        if d_tokens <= 0.0 or d_seconds <= 0.0:
            return 0.0, base
        return d_tokens / d_seconds, (tokens, seconds)

    def _stalled_for(self, now, running, decode_calls):
        """Progress is n_decode_total, not token counters, which stall during healthy prefill and
        decode too."""
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
        """Log only: a scrape cannot prove the engine will not resume, so cancelling it is unsafe."""
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
        prev = None
        gen_base = prompt_base = None
        while not self._stop.wait(self._interval):
            m = self._scrape()
            if not m:
                misses += 1
                if misses == 3:
                    self._log.debug("engine_stats: /metrics scrape failing, still retrying")
                continue
            misses = 0
            now = time.monotonic()
            predicted = m.get("tokens_predicted_total", 0.0)
            prompt = m.get("prompt_tokens_total", 0.0)
            predicted_s = m.get("tokens_predicted_seconds_total", 0.0)
            prompt_s = m.get("prompt_seconds_total", 0.0)
            # A build without n_decode_total reads None; wedge message chosen at report time.
            decode_calls = m.get("n_decode_total")
            running, waiting = (
                int(m.get("requests_processing", 0)),
                int(m.get("requests_deferred", 0)),
            )
            prompt_delta, prompt_base = self._prompt_rate(prompt_base, prompt, prompt_s)
            gen_moved = gen_base is not None and (predicted, predicted_s) != gen_base
            gen_base = (predicted, predicted_s)
            decode_rate = None
            if prev is not None and now > prev[0] and None not in (decode_calls, prev[1]):
                decode_rate = max(0.0, (decode_calls - prev[1]) / (now - prev[0]))
            prev = (now, decode_calls)
            # The gauge resets on every /metrics read, so a zero is a real reading.
            gen_tps = m.get("predicted_tokens_seconds")
            prompt_measured = "prompt_tokens_seconds" in m or (
                "prompt_tokens_total" in m and "prompt_seconds_total" in m
            )
            prompt_tps = m.get("prompt_tokens_seconds") or prompt_delta
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
            if running or waiting or gen_tps or gen_moved or prompt_tps:
                # Absent, not 0.0: an unmeasured metric must not read as zero.
                fields = {}
                if gen_tps is not None:
                    fields["gen_tok_s"] = round(float(gen_tps), 1)
                if prompt_measured:
                    fields["prompt_tok_s"] = round(float(prompt_tps), 1)
                fields["running"], fields["waiting"] = running, waiting
                if decode_rate is not None:
                    fields["decode_calls_s"] = round(float(decode_rate), 1)
                self._log.info("engine_stats", **fields)


# TIMEOUT_MAX is platform specific (Windows ~49.7 days), so let the platform cap it.
_MAX_ENV_SECONDS = min(7.0 * 24.0 * 60.0 * 60.0, threading.TIMEOUT_MAX)


def _env_float(name, default, logger):
    """Rejects non-finite values, which would silently disable the stall and elapsed reporting."""
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
        # A far deadline makes Event.wait raise 'timestamp out of range', killing the thread.
        logger.warning(
            "engine_stats_env_clamped",
            variable = name,
            value = raw,
            applied_s = _MAX_ENV_SECONDS,
            reason = "longer than a timed wait can represent",
        )
        return _MAX_ENV_SECONDS
    return value


def maybe_start_stats_logger(
    base_url,
    logger,
    headers = None,
):
    """Start a stats logger unless UNSLOTH_STUDIO_ENGINE_STATS disables it."""
    if (os.environ.get("UNSLOTH_STUDIO_ENGINE_STATS", "1") or "").strip().lower() in _OFF:
        return None
    interval = _env_float("UNSLOTH_STUDIO_ENGINE_STATS_INTERVAL_S", 10.0, logger)
    # 0 disables the stall line.
    stall_timeout = _env_float("UNSLOTH_STUDIO_ENGINE_STALL_TIMEOUT_S", 600.0, logger)
    sl = LlamaServerStatsLogger(
        base_url,
        logger,
        interval,
        stall_timeout_s = stall_timeout,
        headers = headers,
    )
    sl.start()
    return sl
