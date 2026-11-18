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
    monkeypatch.setattr(dc, "_ACTIVE", 1, raising = False)  # inside Trainer.train()
    monkeypatch.setattr(dc, "_foreign_bytes", lambda index: 0, raising = False)
    for name in ("_TRANSIENT", "_SEEN", "_CALLS", "_ROWS_SEEN", "_ROWS_NEW", "_FOREIGN"):
        getattr(dc, name, {}).clear()
    dc.clear_dequant_cache()
    dc.STATS.clear()
    yield
    dc.clear_dequant_cache()
    for name in ("_TRANSIENT", "_SEEN", "_ROWS_SEEN", "_ROWS_NEW", "_FOREIGN"):
        getattr(dc, name, {}).clear()


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


def _admit_at_safe_point(monkeypatch, rows):
    # One step with `rows`-row matmuls, then the safe point that measures it and sets the budget.
    _memory(monkeypatch, usable = 10 * GiB)
    W, absmax = _weight()
    dc._lookup(W, absmax, _decoder([]), torch.bfloat16, rows)
    dc._on_optimizer_step(None, (), {})
    calls = []
    dc._lookup(W, absmax, _decoder(calls), torch.bfloat16, rows)
    assert calls == [True] and dc.cached_count() == 1
    return W, absmax


def test_longer_batch_than_measured_drops_cache(monkeypatch):
    # The budget was measured on 128-row steps: a 4096-row batch (a long sequence late in training, a larger
    # evaluation batch) has a transient nobody measured, so it runs with nothing cached, like the base code.
    W, absmax = _admit_at_safe_point(monkeypatch, rows = 128)
    calls = []
    dc._lookup(W, absmax, _decoder(calls), torch.bfloat16, 128)
    assert calls == [] and dc.cached_count() == 1  # same size: served from the cache
    dc._lookup(W, absmax, _decoder(calls), torch.bfloat16, 4096)
    assert calls == [False] and dc.cached_count() == 0 and dc._BUDGET[None] == 0
    assert dc.STATS["rows_drops"] == 1
    dc._lookup(W, absmax, _decoder(calls), torch.bfloat16, 128)
    assert (
        calls == [False, False] and dc.cached_count() == 0
    )  # nothing re-admitted before the safe point
    # The next safe point has measured the 4096-row step: caching resumes, also for 4096 rows.
    dc._on_optimizer_step(None, (), {})
    dc._lookup(W, absmax, _decoder(calls), torch.bfloat16, 4096)
    assert calls == [False, False, True] and dc.cached_count() == 1


def test_nothing_cached_outside_trainer_train(monkeypatch):
    W, absmax = _admit_at_safe_point(monkeypatch, rows = 64)
    monkeypatch.setattr(dc, "_ACTIVE", 0)  # a custom loop, or a step after train() returned
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    dc._on_optimizer_step(None, (), {})
    assert dc.cached_count() == 0 and dc._BUDGET == {}


def test_per_process_memory_fraction_caps_usable(monkeypatch):
    # Studio's training VRAM budget: the allocator stops at fraction * total even with the card free.
    stats = {"allocated_bytes.all.current": 3 * GiB, "reserved_bytes.all.current": 4 * GiB}
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda index = None: (60 * GiB, 80 * GiB))
    monkeypatch.setattr(torch.cuda, "memory_stats", lambda index = None: stats)
    monkeypatch.setattr(dc, "_fraction", lambda index: 1.0)
    assert dc._usable(0)[0] == 61 * GiB
    monkeypatch.setattr(dc, "_fraction", lambda index: 0.1)  # 8 GiB allowed, 3 GiB allocated
    assert dc._usable(0)[0] == 5 * GiB


def test_other_process_on_the_gpu_turns_auto_off(monkeypatch):
    W, absmax = _admit_at_safe_point(monkeypatch, rows = 64)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(
        dc, "_foreign_bytes", lambda index: 3 * GiB
    )  # e.g. a chat model loaded meanwhile
    dc._on_optimizer_step(None, (), {})
    assert dc.cached_count() == 0 and dc._BUDGET[None] == 0 and dc.STATS["foreign_off"] == 1


def test_trainer_scope_hooks(monkeypatch):
    transformers_trainer = pytest.importorskip("transformers.trainer")
    from types import SimpleNamespace

    assert dc.install_trainer_scope()
    scope = [
        c
        for c in transformers_trainer.DEFAULT_CALLBACKS
        if c.__name__ == "UnslothDequantCacheScope"
    ]
    assert len(scope) == 1
    Trainer = transformers_trainer.Trainer
    assert getattr(Trainer.train, "_unsloth_dq_scope", False)
    assert getattr(Trainer.evaluate, "_unsloth_dq_scope", False)
    callback = scope[0]()
    monkeypatch.setattr(dc, "_ACTIVE", 0)
    callback.on_train_begin(None, None, SimpleNamespace())
    assert dc._ACTIVE == 1
    _admit_at_safe_point(monkeypatch, rows = 64)
    callback.on_step_end(None, None, SimpleNamespace(should_evaluate = True))  # evaluation comes next
    assert dc.cached_count() == 0 and dc._BUDGET == {}
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    callback.on_train_end(None, None, SimpleNamespace())
    assert dc._ACTIVE == 0


def test_auto_holds_weights_for_one_step_only(monkeypatch):
    # Between optimizer steps (callbacks, evaluation, saving, user code) auto mode holds nothing.
    W, absmax = _admit_at_safe_point(monkeypatch, rows = 64)
    dc._on_optimizer_step(None, (), {})
    assert dc.cached_count() == 0 and dc._BUDGET[None] > 0 and dc.STATS["step_drops"] == 1
    calls = []
    dc._lookup(W, absmax, _decoder(calls), torch.bfloat16, 64)  # the next step admits again
    assert calls == [True] and dc.cached_count() == 1
    # An explicit GiB cap keeps its weights across steps.
    monkeypatch.setattr(dc, "_MODE", float(GiB))
    dc._on_optimizer_step(None, (), {})
    assert dc.cached_count() == 1


def test_admission_keeps_the_transient_reserve(monkeypatch):
    # Memory taken after the safe point (a callback's tensor, a second model) is not in the budget: an
    # admission must still leave 2 x transient + margin usable.
    _memory(monkeypatch, usable = 10 * GiB)  # transient 1 GiB (peak 4 - allocated 3)
    dc._SEEN.add(None)
    dc._on_optimizer_step(None, (), {})
    assert dc._BUDGET[None] == 7 * GiB
    _memory(monkeypatch, usable = 3 * GiB - 1)  # 7 GiB allocated by someone else since then
    calls = []
    W, absmax = _weight()
    dc._lookup(W, absmax, _decoder(calls), torch.bfloat16)
    assert calls == [False] and dc.cached_count() == 0 and dc.STATS["declined"] == 1
    _memory(monkeypatch, usable = 3 * GiB + 4096)
    dc._lookup(W, absmax, _decoder(calls), torch.bfloat16)
    assert calls == [False, True] and dc.cached_count() == 1
