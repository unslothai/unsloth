# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""NF4 dequant cache bookkeeping on CPU tensors with a faked memory reading: budget from the safe point,
admission, hits, staleness, eviction under the margin, the out-of-memory fallback and the kill switch."""

import pytest
import torch
from unsloth.kernels import dequant_cache as dc

GiB = 2**30


@pytest.fixture(autouse = True)
def _fresh(monkeypatch):
    monkeypatch.setattr(dc, "_MODE", "auto")
    monkeypatch.setattr(dc, "_DISABLED", False)
    monkeypatch.setattr(dc, "_auto_allowed", lambda: True)
    dc.clear_dequant_cache()
    dc._TRANSIENT.clear()
    dc._SEEN.clear()
    dc._CALLS.clear()
    dc.STATS.clear()
    yield
    dc.clear_dequant_cache()
    dc._TRANSIENT.clear()
    dc._SEEN.clear()


def _memory(
    monkeypatch,
    usable,
    total = 16 * GiB,
    allocated = 3 * GiB,
    peak = 4 * GiB,
):
    stats = {"allocated_bytes.all.current": allocated, "allocated_bytes.all.peak": peak}
    monkeypatch.setattr(dc, "_usable", lambda index: (usable, total, stats))


def _weight(n = 64):
    W = torch.zeros(n, 1, dtype = torch.uint8)  # packed NF4 bytes: 2 values each
    absmax = torch.ones(4)
    return W, absmax


def _decoder(calls, n = 64):
    def decode(fresh):
        calls.append(fresh)
        return torch.full((8, 2 * n // 8), float(len(calls)), dtype = torch.bfloat16)

    return decode


def test_parse_mode():
    assert (
        dc.parse_mode("0") is None
        and dc.parse_mode("off") is None
        and dc.parse_mode("junk") is None
    )
    assert (
        dc.parse_mode(None) == "auto"
        and dc.parse_mode("auto") == "auto"
        and dc.parse_mode("1") == "auto"
    )
    assert dc.parse_mode("2") == 2 * GiB and dc.parse_mode("-1") is None


def test_budget_from_safe_point(monkeypatch):
    _memory(monkeypatch, usable = 10 * GiB)
    dc._SEEN.add(None)
    dc._on_optimizer_step(None, (), {})
    # usable - 2 x transient (peak - allocated) - margin (max(1 GiB, 5% of 16 GiB))
    assert dc._BUDGET[None] == 10 * GiB - 2 * GiB - GiB


def test_nothing_cached_before_first_step():
    W, absmax = _weight()
    calls = []
    for _ in range(3):
        dc._lookup(W, absmax, _decoder(calls), torch.bfloat16)
    assert calls == [False, False, False] and dc.cached_count() == 0


def test_admit_hit_transposed_and_stale(monkeypatch):
    _memory(monkeypatch, usable = 10 * GiB)
    dc._BUDGET[None] = GiB
    W, absmax = _weight()
    calls = []
    first = dc._lookup(W, absmax, _decoder(calls), torch.bfloat16)
    assert calls == [True] and dc.cached_count() == 1
    again = dc._lookup(W, absmax, _decoder(calls), torch.bfloat16)
    assert calls == [True] and again is first
    # The backward passes W.t(): the cached [N, K] comes back transposed, like fast_dequantize.
    back = dc._lookup(W.t(), absmax, _decoder(calls), torch.bfloat16)
    assert calls == [True] and torch.equal(back, first.t())
    W.add_(0)  # an in-place write bumps the version counter: decode again
    dc._lookup(W, absmax, _decoder(calls), torch.bfloat16)
    assert calls == [True, True] and dc.STATS["stale"] == 1


def test_budget_caps_admission(monkeypatch):
    _memory(monkeypatch, usable = 10 * GiB)
    dc._BUDGET[None] = 128 * 2 + 1  # room for one 128-element bf16 weight
    calls = []
    for _ in range(2):
        W, absmax = _weight()
        dc._lookup(W, absmax, _decoder(calls), torch.bfloat16)
    assert calls == [True, False] and dc.cached_count() == 1 and dc.STATS["declined"] == 1


def test_guard_evicts_under_margin(monkeypatch):
    _memory(monkeypatch, usable = 10 * GiB)
    dc._BUDGET[None] = GiB
    weights = [_weight() for _ in range(3)]
    for W, absmax in weights:
        dc._lookup(W, absmax, _decoder([]), torch.bfloat16)
    assert dc.cached_count() == 3
    _memory(monkeypatch, usable = 0)
    for _ in range(dc._CHECK_EVERY):
        dc._lookup(*weights[0], _decoder([]), torch.bfloat16)
    assert dc.cached_count() == 0 and dc.STATS["evictions"] == 3 and dc._BUDGET[None] == 0


def test_oom_while_admitting_disables(monkeypatch):
    _memory(monkeypatch, usable = 10 * GiB)
    dc._BUDGET[None] = GiB
    W, absmax = _weight()
    calls = []

    def decode(fresh):
        calls.append(fresh)
        if fresh:
            raise torch.cuda.OutOfMemoryError("fake")
        return torch.zeros(8, 16, dtype = torch.bfloat16)

    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    out = dc._lookup(W, absmax, decode, torch.bfloat16)
    assert calls == [True, False] and out.shape == (8, 16)
    assert dc._DISABLED and dc.cached_count() == 0 and dc.STATS["oom_clears"] == 1


def test_kill_switch(monkeypatch):
    monkeypatch.setattr(dc, "_MODE", None)
    W, absmax = _weight()
    assert not dc._eligible(W, torch.bfloat16)
    assert not dc.traced_route(W, None)
    dc._SEEN.add(None)
    _memory(monkeypatch, usable = 10 * GiB)
    dc._on_optimizer_step(None, (), {})
    assert dc._BUDGET == {}


def test_dead_weight_dropped_on_gc(monkeypatch):
    import gc

    _memory(monkeypatch, usable = 10 * GiB)
    dc._BUDGET[None] = GiB
    W, absmax = _weight()
    dc._lookup(W, absmax, _decoder([]), torch.bfloat16)
    assert dc.cached_count() == 1
    del W
    gc.collect()  # the gc callback frees the dense copy of a weight whose storage died
    assert dc.cached_count() == 0 and dc.STATS["stale"] == 1
