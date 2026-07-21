# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for the llama-server /metrics -> engine_stats translator: generation
throughput comes from generated-token metrics (not llama_decode() calls), and
the unexposed kv_cache_usage_ratio is never fabricated into the log line."""

from core.inference.llama_stats import LlamaServerStatsLogger


class _Capture:
    def __init__(self):
        self.events = []

    def info(self, event, **kw):
        self.events.append((event, dict(kw)))

    def debug(self, *a, **k):
        pass


def _drive(snaps):
    """Run _run() synchronously over `snaps`, then stop deterministically."""
    cap = _Capture()
    lg = LlamaServerStatsLogger("http://127.0.0.1:0", cap)
    lg._interval = 0.001  # bypass the 1s floor for a fast, synchronous run
    state = {"i": 0}

    def fake_scrape():
        i = state["i"]
        state["i"] += 1
        if i >= len(snaps):
            lg.stop()
            return None
        return snaps[i]

    lg._scrape = fake_scrape
    lg._run()
    return [kw for ev, kw in cap.events if ev == "engine_stats"]


def test_gen_tok_s_uses_token_metrics_not_decode_calls():
    # tokens_predicted_total jumps 95 while n_decode_total only moves 9; the
    # gauge reports 95 tok/s. Decode-call rate (9) must not be reported.
    snaps = [
        {
            "tokens_predicted_total": 0.0,
            "prompt_tokens_total": 0.0,
            "n_decode_total": 0.0,
            "predicted_tokens_seconds": 95.0,
            "prompt_tokens_seconds": 30.0,
            "requests_processing": 1.0,
        },
        {
            "tokens_predicted_total": 95.0,
            "prompt_tokens_total": 30.0,
            "n_decode_total": 9.0,
            "predicted_tokens_seconds": 95.0,
            "prompt_tokens_seconds": 30.0,
            "requests_processing": 1.0,
        },
    ]
    stats = _drive(snaps)
    assert stats, "expected engine_stats while a request is processing"
    assert all(s["gen_tok_s"] == 95.0 for s in stats)
    assert all(s["prompt_tok_s"] == 30.0 for s in stats)


def test_kv_cache_pct_not_emitted_when_metric_absent():
    # llama.cpp does not expose kv_cache_usage_ratio, so it must not appear.
    snaps = [
        {
            "tokens_predicted_total": 0.0,
            "prompt_tokens_total": 0.0,
            "predicted_tokens_seconds": 10.0,
            "requests_processing": 1.0,
        },
        {
            "tokens_predicted_total": 10.0,
            "prompt_tokens_total": 5.0,
            "predicted_tokens_seconds": 10.0,
            "requests_processing": 1.0,
        },
    ]
    stats = _drive(snaps)
    assert stats
    assert all("kv_cache_pct" not in s for s in stats)


def test_scrape_parses_labelled_and_bare_metrics(monkeypatch):
    # Prometheus samples may carry labels; both labelled and bare lines parse.
    import core.inference.llama_stats as ls

    body = (
        'llamacpp:tokens_predicted_total{model="m"} 20\n'
        'llamacpp:prompt_tokens_total{model="m"} 5\n'
        "llamacpp:requests_processing 1\n"
        "# HELP llamacpp:ignored ignored\n"
    )

    class _Resp:
        status = 200

        def read(self):
            return body.encode()

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(ls.urllib.request, "urlopen", lambda *a, **k: _Resp())
    m = ls.LlamaServerStatsLogger("http://127.0.0.1:0", _Capture())._scrape()
    assert m["tokens_predicted_total"] == 20.0
    assert m["prompt_tokens_total"] == 5.0
    assert m["requests_processing"] == 1.0


def test_counter_delta_fallback_without_gauges():
    # Older binaries expose only the counters; throughput falls back to deltas.
    snaps = [
        {
            "tokens_predicted_total": 100.0,
            "prompt_tokens_total": 0.0,
            "requests_processing": 1.0,
        },
        {
            "tokens_predicted_total": 100.0,
            "prompt_tokens_total": 0.0,
            "requests_processing": 1.0,
        },
    ]
    stats = _drive(snaps)
    # running=1 keeps it emitting; gen_tok_s falls back to the (here zero) delta.
    assert stats and all(s["gen_tok_s"] >= 0.0 for s in stats)
