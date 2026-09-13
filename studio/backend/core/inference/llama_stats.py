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


class SysmemFallbackWatch:
    """Windows CUDA only: warn once when a resident placement is silently paging to system RAM.

    Since driver 536.40 an oversized Windows CUDA allocation does not fail; WDDM demand-pages it,
    ``cudaMalloc`` returns success, and nothing in CUDA, NVML or nvidia-smi reports it (NVIDIA KB
    5490, https://nvidia.custhelp.com/app/answers/detail/a_id/5490, and
    https://forums.developer.nvidia.com/t/cudamalloc-with-sysmem-fallback/347791). There is no
    per-process query and no per-process way to turn it off, so this only ever reports.

    The fingerprint, from https://github.com/ollama/ollama/issues/16725: nvidia-smi free memory
    pinned at 0 to 2 percent of total, no out-of-memory anywhere, and generation an order of
    magnitude below what a resident placement should give. ``floor_gen_tok_s`` is the caller's
    expectation for the third part, derived from the host-to-device link rate and the weight bytes
    a token reads, so it is the ceiling the paging path could reach.

    Sampling is cheapest first: throughput rides the /metrics scrape already running, and the VRAM
    probe only runs once generation is already slow.
    """

    def __init__(
        self,
        vram_probe,
        floor_gen_tok_s,
        logger,
        *,
        free_fraction_max = 0.02,
        consecutive = 3,
    ):
        self._probe = vram_probe
        self._floor = float(floor_gen_tok_s)
        self._log = logger
        self._free_fraction_max = float(free_fraction_max)
        # One slow tick is a cold prompt or a scrape boundary; the fingerprint is sustained.
        self._consecutive = max(1, int(consecutive))
        self._streak = 0
        self._reported = False

    def observe(self, *, running, gen_tok_s):
        """Feed one /metrics tick. True on the tick the warning was emitted, else False."""
        if self._reported:
            return False
        if not running or not gen_tok_s or float(gen_tok_s) <= 0.0:
            self._streak = 0
            return False
        if float(gen_tok_s) > self._floor:
            self._streak = 0
            return False
        try:
            sample = self._probe()
        except Exception:
            sample = None
        # Unreadable is not evidence: an absent probe must not manufacture the fingerprint.
        if not sample:
            self._streak = 0
            return False
        pegged = []
        for _idx, free_mib, total_mib in sample:
            if int(total_mib) <= 0:
                continue
            pegged.append((int(free_mib) / float(total_mib)) <= self._free_fraction_max)
        if not pegged or not any(pegged):
            self._streak = 0
            return False
        self._streak += 1
        if self._streak < self._consecutive:
            return False
        self._reported = True
        self._log.warning(
            "cuda_sysmem_fallback_suspected",
            gen_tok_s = round(float(gen_tok_s), 2),
            expected_floor_tok_s = round(self._floor, 2),
            free_mib = [int(f) for _i, f, _t in sample],
            total_mib = [int(t) for _i, _f, t in sample],
            detail = "VRAM is full, nothing reported an out-of-memory error, and generation is "
            "an order of magnitude slow: on Windows the NVIDIA driver lets an oversized CUDA "
            "allocation succeed and pages it to system RAM instead of failing. The model loaded, "
            "it is just not resident. Set 'CUDA - Sysmem Fallback Policy' to 'Prefer No Sysmem "
            "Fallback' in NVIDIA Control Panel, 3D Settings (driver 546.01 or newer, needs an "
            "application restart), or load a smaller model or context.",
        )
        return True


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
        sysmem_watch = None,
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
        # Windows CUDA only and None everywhere else, so every other platform runs the loop it did.
        self._sysmem_watch = sysmem_watch

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
                value = float(v)
            except ValueError:
                continue
            # float() overflows a long digit string to inf without raising; _env_float
            # refuses the same text. No printed double reaches 1.8e308, so nothing is lost.
            if math.isfinite(value):
                out[k] = value
        return out

    @staticmethod
    def _prompt_rate(base, tokens, seconds):
        """Prompt tokens per second over the engine's OWN measure of the time they took.

        Prompt only: add_prompt(n, n, t_us) pairs those totals, while metrics_on_prediction()
        passes n_gen and n_gen - 1. The baseline is held until both move, since /metrics
        renders six significant digits and one can cross a boundary a scrape before the other.
        """
        if base is None or tokens < base[0] or seconds < base[1]:
            return 0.0, (tokens, seconds)  # first reading, or counters that went backwards
        d_tokens, d_seconds = tokens - base[0], seconds - base[1]
        if d_tokens <= 0.0 or d_seconds <= 0.0:
            return 0.0, base
        return d_tokens / d_seconds, (tokens, seconds)

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
        prev = None  # (monotonic_t, n_decode_total)
        gen_base = prompt_base = None  # (tokens, seconds) at the last tick that carried both
        while not self._stop.wait(self._interval):
            m = self._scrape()
            if not m:
                misses += 1
                if misses == 3:  # transient stall (load/GC); keep polling
                    self._log.debug("engine_stats: /metrics scrape failing, still retrying")
                continue  # real shutdown is driven by stop() from _kill_process
            misses = 0
            now = time.monotonic()
            predicted = m.get("tokens_predicted_total", 0.0)
            prompt = m.get("prompt_tokens_total", 0.0)
            predicted_s = m.get("tokens_predicted_seconds_total", 0.0)
            prompt_s = m.get("prompt_seconds_total", 0.0)
            # A build without n_decode_total reads None and never "changes", so the wedge
            # message is chosen at report time.
            decode_calls = m.get("n_decode_total")
            running, waiting = (
                int(m.get("requests_processing", 0)),
                int(m.get("requests_deferred", 0)),
            )
            prompt_delta, prompt_base = self._prompt_rate(prompt_base, prompt, prompt_s)
            gen_moved = gen_base is not None and (predicted, predicted_s) != gen_base
            gen_base = (predicted, predicted_s)
            # Calls, not tokens, and never fed into tok/s: the only counter that moves
            # within a tick, so the only sign of progress while a generation runs.
            decode_rate = None
            if prev is not None and now > prev[0] and None not in (decode_calls, prev[1]):
                decode_rate = max(0.0, (decode_calls - prev[1]) / (now - prev[0]))
            prev = (now, decode_calls)
            # The bucket empties on every /metrics read, ours or another client's, so a zero
            # gauge is a reading. The counters read a free first token against a millisecond.
            gen_tps = m.get("predicted_tokens_seconds")
            # /metrics renders one table, so a scrape with no prompt metric measured none.
            prompt_measured = "prompt_tokens_seconds" in m or (
                "prompt_tokens_total" in m and "prompt_seconds_total" in m
            )
            prompt_tps = m.get("prompt_tokens_seconds") or prompt_delta
            if self._sysmem_watch is not None:
                try:
                    self._sysmem_watch.observe(running = running, gen_tok_s = gen_tps)
                except Exception:  # a probe must never kill the stats thread
                    pass
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
            # Gate on real activity this tick, so an idle engine stays quiet.
            if running or waiting or gen_tps or gen_moved or prompt_tps:
                # Absent, not 0.0: a build with no gauge, no n_decode_total or no prompt
                # metric was never measured, and 0.0 states it was.
                fields = {}
                if gen_tps is not None:
                    fields["gen_tok_s"] = round(float(gen_tps), 1)
                if prompt_measured:
                    fields["prompt_tok_s"] = round(float(prompt_tps), 1)
                fields["running"], fields["waiting"] = running, waiting
                if decode_rate is not None:
                    fields["decode_calls_s"] = round(float(decode_rate), 1)
                self._log.info("engine_stats", **fields)


# A week already means "never" for a poll interval or a stall timeout. Bounded by threading.TIMEOUT_MAX as well,
# because the ceiling is platform specific and much lower than it looks: Linux accepts ~9.2e9 seconds, Windows about
# 49.7 days, since the timeout becomes a DWORD of milliseconds there. Picking a constant by hand got this wrong once
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
        # Event.wait() builds an absolute deadline, and one far enough out raises "timestamp out of range for platform
        # time_t" once the wait is entered, killing the poll thread. Measured: a century still waits, 1e10 seconds
        # does not.
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
    sysmem_watch = None,
):
    """Start a stats logger unless UNSLOTH_STUDIO_ENGINE_STATS disables it.

    ``sysmem_watch`` is the Windows CUDA sysmem-fallback detector, None everywhere else."""
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
        sysmem_watch = sysmem_watch,
    )
    sl.start()
    return sl
