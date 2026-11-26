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
    for name in (
        "_TRANSIENT",
        "_SEEN",
        "_CALLS",
        "_ROWS_SEEN",
        "_ROWS_NEW",
        "_FOREIGN",
        "_CHECKED",
        "_SHRINK_PEAK",
        "_SHRINK_UNSAFE",
    ):
        getattr(dc, name, set()).clear()
    for name in ("_SHAPE_SEEN", "_SHAPE_NEW"):
        if hasattr(dc, name):
            getattr(dc, name)[:] = [0, 0]
    dc.clear_dequant_cache()
    dc.STATS.clear()
    dc._PRE_RAN[0] = 0
    dc._PADDED_SEEN[0] = dc._PADDED_NEW[0] = 0
    for name in ("_SHRINK_PEAK", "_SHRINK_UNSAFE"):
        getattr(dc, name, set()).clear()
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
    # The guard reads the driver first: on a GPU host the real free memory would hide the faked reading.
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda index = None: (usable, total))
    monkeypatch.setattr(dc, "_fraction", lambda index: 1.0, raising = False)


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


@pytest.mark.parametrize("storage", [torch.bfloat16, torch.float32])
def test_budget_counts_float_quant_storage_bytes(monkeypatch, storage):
    # FSDP2 QLoRA packs NF4 into bf16 / float32 storage: 4 or 8 values per element, not 2.
    _memory(monkeypatch, usable = 10 * GiB)
    W = torch.zeros(64, 1, dtype = storage)
    dense = 64 * W.element_size() * 2
    dc._BUDGET[None] = dense * 2 - 1  # one bf16 byte short of the real dense size
    calls = []

    def decode(fresh):
        calls.append(fresh)
        return torch.zeros(8, dense // 8, dtype = torch.bfloat16)

    dc._lookup(W, torch.ones(4), decode, torch.bfloat16)
    assert calls == [False] and dc.cached_count() == 0 and dc.STATS["declined"] == 1
    dc.clear_dequant_cache()
    dc._BUDGET[None] = dense * 2
    dc._lookup(W, torch.ones(4), decode, torch.bfloat16)
    assert calls == [False, True] and dc.cached_count() == 1


def test_whole_model_compile_gets_no_budget(monkeypatch):
    # torch.compile(model) traces the pre-forward shape guard away: without it a longer batch could allocate its
    # attention mask before any LoRA op drops the cache, so the safe point must not give a budget.
    _memory(monkeypatch, usable = 10 * GiB)
    model = torch.nn.Linear(1, 1)
    monkeypatch.setattr(
        dc, "_HOOKS", [model.register_forward_pre_hook(dc._model_pre_hook, with_kwargs = True)]
    )
    dc._SEEN.add(None)
    ids = torch.zeros(2, 8, dtype = torch.long)
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    dc._model_pre_hook(model, (), {"input_ids": ids})  # traced, never run eagerly
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: False)
    dc._on_optimizer_step(None, (), {})
    assert dc._BUDGET.get(None, 0) == 0
    dc._model_pre_hook(model, (), {"input_ids": ids})  # eager forward: the guard runs again
    dc._on_optimizer_step(None, (), {})
    assert dc._BUDGET[None] > 0
    dc._HOOKS[0].remove()


def test_first_padded_batch_of_a_measured_shape_drops_the_cache(monkeypatch):
    # Same [batch, sequence] as the measured steps, but padded: the forward builds a dense quadratic mask before
    # any LoRA op, which the unpadded steps never measured.
    _memory(monkeypatch, usable = 10 * GiB)
    ids = torch.zeros(2, 8, dtype = torch.long)
    full = torch.ones(2, 8, dtype = torch.long)
    dc._model_pre_hook(None, (), {"input_ids": ids, "attention_mask": full})
    dc._SEEN.add(None)
    dc._on_optimizer_step(None, (), {})
    W, absmax = _weight()
    dc._lookup(W, absmax, _decoder([]), torch.bfloat16)
    assert dc.cached_count() == 1
    dc._model_pre_hook(None, (), {"input_ids": ids, "attention_mask": full})
    assert dc.cached_count() == 1  # same shape, unpadded: kept
    padded = full.clone()
    padded[0, :3] = 0
    dc._model_pre_hook(None, (), {"input_ids": ids, "attention_mask": padded})
    assert dc.cached_count() == 0 and dc._BUDGET.get(None, 0) == 0
    dc._on_optimizer_step(None, (), {})  # the padded step is measured now
    dc._lookup(W, absmax, _decoder([]), torch.bfloat16)
    dc._model_pre_hook(None, (), {"input_ids": ids, "attention_mask": padded})
    assert dc.cached_count() == 1


def test_padded_mask_bound_is_joint_not_per_dimension(monkeypatch):
    # Measured padded [16, 1024] and [1, 4096]: a padded [4, 4096] fits both maxima but its dense mask is 4x larger.
    _memory(monkeypatch, usable = 10 * GiB)

    def step(b, s):
        mask = torch.ones(b, s, dtype = torch.long)
        mask[0, 0] = 0
        dc._model_pre_hook(
            None, (), {"input_ids": torch.zeros(b, s, dtype = torch.long), "attention_mask": mask}
        )

    step(16, 1024)
    step(1, 4096)
    dc._SEEN.add(None)
    dc._on_optimizer_step(None, (), {})
    W, absmax = _weight()
    dc._lookup(W, absmax, _decoder([]), torch.bfloat16)
    step(16, 1024)
    assert dc.cached_count() == 1  # measured
    step(4, 4096)
    assert dc.cached_count() == 0 and dc._BUDGET.get(None, 0) == 0


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
    handler = transformers_trainer.CallbackHandler([], None, None, None, None)
    assert getattr(handler.add_callback, "_unsloth_dq_scope", False)
    handler.add_callback(scope[0])
    callback = handler.callbacks[-1]
    assert callback.handler() is handler
    monkeypatch.setattr(dc, "_ACTIVE", 0)
    callback.on_train_begin(None, None, SimpleNamespace())
    assert dc._ACTIVE == 1  # only known callbacks
    # A user callback may allocate anything between optimizer steps: auto mode stays off for this run.
    UserCallback = type(
        "UserCallback", (transformers_trainer.TrainerCallback,), {"__module__": "__main__"}
    )
    handler.add_callback(UserCallback)
    callback.on_train_begin(None, None, SimpleNamespace())
    assert dc._ACTIVE == 0 and dc.STATS["user_callbacks_off"] == 1
    monkeypatch.setattr(dc, "_MODE", float(GiB))  # an explicit GiB cap opts in anyway
    callback.on_train_begin(None, None, SimpleNamespace())
    assert dc._ACTIVE == 1
    monkeypatch.setattr(dc, "_MODE", "auto")
    _admit_at_safe_point(monkeypatch, rows = 64)
    callback.on_step_end(None, None, SimpleNamespace(should_evaluate = True))  # evaluation comes next
    assert dc.cached_count() == 0 and dc._BUDGET == {}
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    callback.on_train_end(None, None, SimpleNamespace())
    assert dc._ACTIVE == 0


def test_admission_keeps_the_transient_reserve(monkeypatch):
    # Memory taken after the safe point (a callback's tensor, a second model) is not in the budget: an
    # admission must still leave 2 x transient + margin usable.
    _memory(monkeypatch, usable = 10 * GiB)  # transient 1 GiB (peak 4 - allocated 3)
    dc._SEEN.add(None)
    dc._on_optimizer_step(None, (), {})
    assert dc._BUDGET[None] == 7 * GiB
    _memory(monkeypatch, usable = 3 * GiB + 4096)  # 7 GiB allocated by someone else since then
    calls = []
    W, absmax = _weight()
    dc._lookup(W, absmax, _decoder(calls), torch.bfloat16)  # 256 bytes fit above the 3 GiB reserve
    assert calls == [True] and dc.cached_count() == 1 and dc._BUDGET[None] == 4096
    W2, absmax2 = _weight(4096)  # 16 KiB do not
    dc._lookup(W2, absmax2, _decoder(calls, 4096), torch.bfloat16)
    assert calls == [True, False] and dc.cached_count() == 1 and dc.STATS["declined"] == 1


def test_model_call_larger_than_measured_drops_cache_before_the_forward(monkeypatch):
    # The pre-hook sees the input before the model allocates for it (a [1, 1, s, s] attention mask grows with
    # the square of the sequence), so the cache is gone before that allocation, not at the first LoRA matmul.
    dc._model_pre_hook(None, (), {"input_ids": torch.zeros(4, 16, dtype = torch.long)})
    W, absmax = _admit_at_safe_point(monkeypatch, rows = 64)  # measured: 64 tokens, sequences of 16
    dc._model_pre_hook(None, (), {"input_ids": torch.zeros(4, 16, dtype = torch.long)})
    assert dc.cached_count() == 1
    dc._model_pre_hook(
        None, (), {"input_ids": torch.zeros(2, 32, dtype = torch.long)}
    )  # same tokens, longer
    assert dc.cached_count() == 0 and dc._BUDGET[None] == 0 and dc.STATS["shape_drops"] == 1
    dc._model_pre_hook(None, (), {"input_ids": torch.zeros(2, 32, dtype = torch.long)})
    dc._on_optimizer_step(None, (), {})
    dc._lookup(W, absmax, _decoder([]), torch.bfloat16, 64)
    dc._model_pre_hook(None, (), {"inputs_embeds": torch.zeros(2, 32, 8)})  # measured now: kept
    assert dc.cached_count() == 1


def test_replaced_quant_state_is_not_served_from_the_cache(monkeypatch):
    # Same packed storage and version, but a new quant state (absmax): the cached dense weight is stale.
    _memory(monkeypatch, usable = 10 * GiB)
    dc._BUDGET[None] = GiB
    W, absmax = _weight()
    calls = []
    first = dc._lookup(W, absmax, _decoder(calls), torch.bfloat16)
    assert calls == [True] and dc._lookup(W, absmax, _decoder(calls), torch.bfloat16) is first
    new_absmax = torch.full((4,), 2.0)
    out = dc._lookup(W, new_absmax, _decoder(calls), torch.bfloat16)
    assert calls == [True, True] and out is not first and dc.STATS["stale"] == 1


def _fake_allocator(monkeypatch):
    fake = {"allocated": 3 * GiB, "peak": 4 * GiB}  # step 1: 3 GiB resident, 1 GiB transient

    def usable(index):
        stats = {
            "allocated_bytes.all.current": fake["allocated"],
            "allocated_bytes.all.peak": fake["peak"],
        }
        return 10 * GiB, 16 * GiB, stats

    monkeypatch.setattr(dc, "_usable", usable)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    dc._SEEN.add(None)
    dc._on_optimizer_step(None, (), {})
    assert dc._TRANSIENT[None] == GiB and dc._BUDGET[None] == 7 * GiB
    W, absmax = _weight()
    dc._lookup(W, absmax, _decoder([]), torch.bfloat16)  # admitted
    fake.update(allocated = 5 * GiB, peak = 6 * GiB)  # steps with a 2 GiB cache, same 1 GiB transient
    return fake


def test_freed_cache_is_not_counted_as_activation_memory(monkeypatch):
    # The allocator's peak still holds cache bytes freed since: after a plain clear (evaluation) the last measured
    # transient is kept and the cache refills; it used to count the freed cache as activations for good.
    fake = _fake_allocator(monkeypatch)
    fake["allocated"] = 3 * GiB  # evaluation follows: the cache is dropped (2 GiB freed)
    dc.clear_dequant_cache()
    dc._on_optimizer_step(None, (), {})  # peak 6 GiB has not risen since the clear
    assert dc._TRANSIENT[None] == GiB and dc._BUDGET[None] == 7 * GiB
    assert dc.STATS["stale_peak_kept"] == 1
    fake["peak"] = 9 * GiB  # a real new peak after the clear is measured as usual
    dc._on_optimizer_step(None, (), {})
    assert dc._TRANSIENT[None] == 6 * GiB and None not in dc._SHRINK_PEAK


def test_stale_peak_after_a_size_drop_stays_conservative(monkeypatch):
    # After a drop before a larger call, the larger step's transient is unknown while the peak has not risen:
    # keeping the last estimate could refill the cache under a step that needs that memory.
    fake = _fake_allocator(monkeypatch)
    fake["allocated"] = 3 * GiB
    dc._drop_device(None, "shape_drops")
    dc._on_optimizer_step(None, (), {})
    # bound from the stale peak: 6 - 3 = 3 GiB transient -> budget 10 - 6 - 1
    assert (
        dc._BUDGET[None] == 3 * GiB
        and dc._TRANSIENT[None] == GiB
        and dc.STATS["stale_peak_bound"] == 1
    )


def test_never_resets_or_patches_torch_peak_stats(monkeypatch):
    # Accelerate, Studio, TRL and notebooks read these: the cache must leave them alone.
    import torch.cuda.memory as M

    originals = (
        torch.cuda.memory_stats,
        torch.cuda.reset_peak_memory_stats,
        M.memory_stats,
        M.reset_peak_memory_stats,
    )

    def refuse(*args, **kwargs):
        raise AssertionError("reset_peak_memory_stats called")

    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", refuse)
    monkeypatch.setattr(M, "reset_peak_memory_stats", refuse)
    monkeypatch.setattr(torch._C, "_cuda_resetPeakMemoryStats", refuse, raising = False)
    fake = _fake_allocator(monkeypatch)
    fake["allocated"] = 3 * GiB
    dc.clear_dequant_cache(release = True)
    dc._drop_device(None, "shape_drops")
    dc._on_optimizer_step(None, (), {})
    assert torch.cuda.memory_stats is originals[0] and M.memory_stats is originals[2]
    assert not hasattr(dc, "_install_peak_view") and not hasattr(dc, "_restart_peak")


def test_evaluation_shapes_do_not_widen_training_shapes():
    # An eval-mode call (bigger evaluation batch) drops the cache but is not "measured" for training steps.
    class Module:
        training = False

    dc._model_pre_hook(Module(), (), {"input_ids": torch.zeros(8, 512, dtype = torch.long)})
    assert dc._SHAPE_NEW == [0, 0]
    Module.training = True
    dc._model_pre_hook(Module(), (), {"input_ids": torch.zeros(2, 64, dtype = torch.long)})
    assert dc._SHAPE_NEW == [128, 64]


def test_host_bookkeeping_is_skipped_while_tracing(monkeypatch):
    # Under torch.compile neither the row count nor the model pre-hook may touch module state or add guards.
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    assert dc.rows_of(torch.zeros(4, 8)) == 0
    dc._model_pre_hook(None, (), {"input_ids": torch.zeros(2, 64, dtype = torch.long)})
    assert dc._SHAPE_NEW == [0, 0] and dc.STATS["shape_drops"] == 0


def test_save_forwards_only_its_parameters_with_an_mtp_head(monkeypatch):
    # unsloth_save_model forwards dict(locals()) to _push_with_mtp_head, which calls
    # unsloth_save_model(**save_kwargs) again: a local the cache release adds would be a TypeError there.
    import inspect
    import unsloth.save as save

    captured = {}
    monkeypatch.setattr(save, "_refuse_mtp_variant", lambda model, variant: None)
    monkeypatch.setattr(save, "_attached_mtp_head", lambda model: object())
    monkeypatch.setattr(
        save, "_push_with_mtp_head", lambda kwargs: captured.update(kwargs) or ("done", None)
    )
    released = []
    monkeypatch.setattr(dc, "clear_dequant_cache", lambda *a, **k: released.append(1))
    model = torch.nn.Linear(2, 2)
    assert save.unsloth_save_model(model, None, "repo", push_to_hub = True) == ("done", None)
    assert released, "the save path no longer releases the dequant cache"
    parameters = set(inspect.signature(inspect.unwrap(save.unsloth_save_model)).parameters)
    assert set(captured) == parameters, set(captured) ^ parameters


class _Training:
    training = True


def test_larger_step_while_the_cache_is_empty_stays_conservative(monkeypatch):
    # Evaluation empties the cache, then a larger training step runs uncached below the old peak: keeping the
    # small measured transient would refill the cache with too little room for that step's activations.
    fake = _fake_allocator(monkeypatch)
    dc._SHAPE_SEEN[:] = [128, 64]
    fake["allocated"] = 3 * GiB
    dc.clear_dequant_cache()  # evaluation
    dc._model_pre_hook(_Training(), (), {"input_ids": torch.zeros(2, 128, dtype = torch.long)})
    fake["peak"] = 6 * GiB  # the larger step peaked at 5 GiB, under the stale 6 GiB
    dc._on_optimizer_step(None, (), {})
    # stale bound: 6 - 3 = 3 GiB transient -> budget 10 - 6 - 1, not the 7 GiB the 1 GiB transient gives
    assert dc._BUDGET[None] == 3 * GiB and dc.STATS["stale_peak_bound"] == 1
    fake["peak"] = 9 * GiB  # a new peak is measured as usual
    dc._on_optimizer_step(None, (), {})
    assert dc._TRANSIENT[None] == 6 * GiB and None not in dc._SHRINK_UNSAFE


def test_more_rows_while_the_cache_is_empty_stays_conservative(monkeypatch):
    fake = _fake_allocator(monkeypatch)
    dc._ROWS_SEEN[None] = 64
    fake["allocated"] = 3 * GiB
    dc.clear_dequant_cache()
    W, absmax = _weight()
    dc._lookup(W, absmax, _decoder([]), torch.bfloat16, rows = 256)  # uncached, larger than measured
    dc._on_optimizer_step(None, (), {})
    assert dc._BUDGET[None] == 3 * GiB and dc.STATS["stale_peak_bound"] == 1


def test_next_training_run_does_not_reuse_the_last_transient(monkeypatch):
    # Another train() (another model, other data) after this one ends is not sized by what this run measured.
    fake = _fake_allocator(monkeypatch)
    fake["allocated"] = 3 * GiB
    dc._end_train()
    monkeypatch.setattr(dc, "_ACTIVE", 1)
    dc._on_optimizer_step(None, (), {})
    assert dc._BUDGET[None] == 3 * GiB and dc.STATS["stale_peak_bound"] == 1


def test_evaluation_rows_do_not_mark_growth(monkeypatch):
    # Evaluation batches run uncached and never size a training step: the refill after evaluation is kept.
    fake = _fake_allocator(monkeypatch)
    fake["allocated"] = 3 * GiB
    dc.clear_dequant_cache()  # Trainer.evaluate
    monkeypatch.setattr(dc, "_EVALUATING", [1])
    W, absmax = _weight()
    dc._lookup(W, absmax, _decoder([]), torch.bfloat16, rows = 4096)
    dc._EVALUATING[0] = 0
    dc._on_optimizer_step(None, (), {})
    assert dc._BUDGET[None] == 7 * GiB and dc.STATS["stale_peak_kept"] == 1
    assert dc._ROWS_SEEN.get(None, 0) < 4096


def test_waits_for_a_safe_point_with_the_checkpoint_skip_plan_in_place(monkeypatch):
    _memory(monkeypatch, usable = 10 * GiB)
    dc._SEEN.add(None)
    pending = [True]
    monkeypatch.setattr(dc, "_checkpoint_skip_pending", lambda: pending[0])
    dc._on_optimizer_step(None, (), {})
    assert dc._BUDGET[None] == 0 and dc.STATS["gc_skip_wait"] == 1
    pending[0] = False
    dc._on_optimizer_step(None, (), {})
    assert dc._BUDGET[None] == 10 * GiB - 2 * GiB - GiB
    pending[0] = True
    for _ in range(dc._GC_SKIP_WAIT_MAX + 2):
        dc._on_optimizer_step(None, (), {})
    assert dc._BUDGET[None] == 10 * GiB - 2 * GiB - GiB
    assert dc.STATS["gc_skip_wait"] == dc._GC_SKIP_WAIT_MAX


def test_checkpoint_skip_pending_reads_the_planner_module(monkeypatch):
    import sys, types

    fake = types.ModuleType("unsloth.models._utils")
    fake.checkpoint_skip_pending = lambda: True
    monkeypatch.setitem(sys.modules, "unsloth.models._utils", fake)
    assert dc._checkpoint_skip_pending() is True
    fake.checkpoint_skip_pending = lambda: False
    assert dc._checkpoint_skip_pending() is False
    monkeypatch.delitem(sys.modules, "unsloth.models._utils")
    assert dc._checkpoint_skip_pending() is False
