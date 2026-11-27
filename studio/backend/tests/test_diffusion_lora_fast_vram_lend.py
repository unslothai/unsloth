# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The nf4 dequant cache gives its idle VRAM back: per-step rebalance against what is free now, and an
explicit lend / return from the Studio parent (a dictation model loading on the card mid-run). CPU only:
the CUDA memory probes are faked and the cache holds CPU tensors."""

import queue
import sys
import threading
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from core.training import diffusion_lora_fast as fl  # noqa: E402

GIB = 1 << 30
MIB = 1 << 20


def _cache_with(
    n_entries,
    entry_bytes,
    *,
    cap,
    transient = 0,
    margin = 0,
):
    cache = fl.Nf4DequantCache()
    cache.plan(cap)
    cache.transient = transient
    cache.margin = margin
    for i in range(n_entries):
        cache.entries[i] = torch.empty(entry_bytes, dtype = torch.uint8)
        cache.used += entry_bytes
    cache.full = cache.used >= cache.budget
    return cache


def test_rebalance_evicts_newest_entries_when_another_process_takes_memory():
    cache = _cache_with(4, 100 * MIB, cap = 400 * MIB, transient = 50 * MIB, margin = 10 * MIB)
    # Room left: 30 free + 40 pool slack - 50 activations - 10 margin = 10 MiB: nothing to do.
    assert cache.rebalance(free = 30 * MIB, reserved = 1000 * MIB, allocated = 960 * MIB) == 0
    assert len(cache.entries) == 4
    # Another process took the 30 MiB and the pool slack is gone: room = -60 MiB, one 100 MiB entry covers it.
    evicted = cache.rebalance(free = 0, reserved = 960 * MIB, allocated = 960 * MIB)
    assert evicted == 100 * MIB
    assert sorted(cache.entries) == [0, 1, 2]  # the newest went first
    assert cache.used == 300 * MIB and cache.budget == cache.used and cache.full
    assert cache.shrinks == 1


def test_rebalance_grows_back_toward_the_plan_only_past_the_hysteresis():
    cache = _cache_with(2, 100 * MIB, cap = 4 * GIB, margin = 0)
    cache.budget = cache.used
    cache.full = True
    # 512 MiB of room: under the 1 GiB hysteresis, the shrunk budget stays.
    cache.rebalance(free = 512 * MIB, reserved = 0, allocated = 0)
    assert cache.budget == 200 * MIB and cache.full
    # The other process left: 3 GiB of room grows the budget (to 85% of it), never past the plan.
    cache.rebalance(free = 3 * GIB, reserved = 0, allocated = 0)
    assert cache.budget == 200 * MIB + int(3 * GIB * 0.85) and not cache.full
    cache.rebalance(free = 40 * GIB, reserved = 0, allocated = 0)
    assert cache.budget == 4 * GIB


def test_rebalance_counts_entries_still_to_come_and_other_reservations():
    cache = _cache_with(1, 100 * MIB, cap = 1 * GIB, margin = 0)
    cache.full = False  # 900 MiB of budget still to fill
    # 500 MiB free cannot hold the 900 MiB still promised: the cache stops growing at what it has.
    assert cache.rebalance(free = 500 * MIB, reserved = 0, allocated = 0) == 0
    assert cache.full and cache.budget == cache.used
    # A second planner (budgeted checkpointing) keeping 300 MiB of activations reserves them here.
    cache = _cache_with(3, 100 * MIB, cap = 300 * MIB, margin = 0)
    cache.reserve(300 * MIB)
    assert cache.rebalance(free = 100 * MIB, reserved = 0, allocated = 0) == 200 * MIB
    assert cache.used == 100 * MIB


def test_lend_frees_everything_and_caches_nothing_until_return():
    cache = _cache_with(3, 10 * MIB, cap = 1 * GIB)
    handle = fl.cache_vram_handler(cache)
    ack = handle("lend")
    assert ack["kind"] == "lend" and ack["freed_gib"] == round(30 * MIB / GIB, 3) and ack["lent"]
    assert not cache.entries and cache.used == 0 and cache.budget == 0
    # Lent: a forward caches nothing.
    assert cache.get(object(), torch.bfloat16) is None
    # A preview's suspend/resume inside the lend does not end it.
    cache.suspend()
    cache.resume()
    assert cache.lent
    ack = handle("return")
    assert not ack["lent"] and not cache.lent
    # Back: the next step's rebalance sizes the budget from what is free now (the new resident counted).
    cache.rebalance(free = 2 * GIB, reserved = 0, allocated = 0)
    assert 0 < cache.budget <= cache.cap


def test_vram_request_without_a_cache_is_acknowledged():
    fl.set_vram_request_handler(None)
    assert fl.handle_vram_request("lend") == {"kind": "lend", "cached_gib": 0.0, "freed_gib": 0.0}


def test_post_step_hook_rebalances_and_returns_memory_to_the_driver(monkeypatch):
    gib = GIB
    probe = {"free": 16 * gib}
    monkeypatch.delenv("UNSLOTH_DIFFUSION_DEQUANT_CACHE_GB", raising = False)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda: (probe["free"], 24 * gib))
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda: 6 * gib)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda: 4 * gib)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 5 * gib)
    emptied = []
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: emptied.append(1))
    p = torch.nn.Parameter(torch.zeros(4))
    opt = torch.optim.SGD([p], lr = 0.1)
    cache = fl.Nf4DequantCache()
    events = []
    handles = fl.attach_training_hooks(
        opt, cache, 1.0, None, lambda kind, st: events.append((kind, st))
    )
    p.grad = torch.ones(4)
    opt.step()  # step 1: plan
    assert cache.budget > 0 and cache.transient == 1 * gib
    for i in range(4):
        cache.entries[i] = torch.empty(1, dtype = torch.uint8)
        cache.sizes[i] = 1 * gib  # stands in for a 1 GiB weight
        cache.used += 1 * gib
    cache.full = True
    opt.step()  # room unchanged: nothing evicted
    assert not emptied and not events
    probe["free"] = 1 * gib  # another process took 15 GiB of the card
    opt.step()
    assert emptied and events and events[-1][0] == "dequant_cache_shrunk"
    assert events[-1][1]["evicted_gib"] > 0 and cache.used < 4 * gib
    for h in handles:
        h.remove()


def test_stop_queue_serves_vram_requests_without_stopping():
    from core.training.diffusion_lora_trainer import _drain_stop_queue

    fl.set_vram_request_handler(None)
    stop_q, event_q = queue.Queue(), queue.Queue()
    stop_q.put({"vram": "lend", "seq": 7})
    assert _drain_stop_queue(stop_q, event_q) is False  # a lend is not a stop
    ack = event_q.get_nowait()
    assert ack["type"] == "vram_request_ack" and ack["seq"] == 7 and ack["kind"] == "lend"
    # Mixed with a real stop: the stop still wins and the lend is still acknowledged.
    stop_q.put({"vram": "return", "seq": 8})
    stop_q.put({"save": False})
    assert _drain_stop_queue(stop_q, event_q) == {"save": False}
    assert event_q.get_nowait()["seq"] == 8


class _AliveProc:
    def is_alive(self):
        return True


def _service():
    from core.training.diffusion_training_service import DiffusionTrainingService

    svc = DiffusionTrainingService(target = lambda **kw: None)
    svc._stop_queue = queue.Queue()
    svc._proc = _AliveProc()
    return svc


def test_lend_vram_waits_for_the_trainer_and_returns_after_the_load():
    from core.training.diffusion_lora_trainer import _drain_stop_queue

    svc = _service()
    event_q = queue.Queue()

    def trainer_step():
        # The child's stop check between steps, then its pump forwarding the ack.
        while svc._stop_queue.empty():
            pass
        _drain_stop_queue(svc._stop_queue, event_q)
        svc._apply_event(event_q.get_nowait())

    t = threading.Thread(target = trainer_step)
    t.start()
    with svc.lend_vram("test load", timeout = 10) as acked:
        assert acked
        t.join()
    msg = svc._stop_queue.get_nowait()
    assert msg["vram"] == "return"
    assert not svc._vram_acks  # no waiter left behind


def test_lend_vram_proceeds_after_the_timeout_and_is_a_noop_when_idle():
    svc = _service()
    with svc.lend_vram("silent trainer", timeout = 0.05) as acked:
        assert acked is False
    sent = [svc._stop_queue.get_nowait()["vram"], svc._stop_queue.get_nowait()["vram"]]
    assert sent == ["lend", "return"]
    svc._proc = None
    with svc.lend_vram("nothing running") as acked:
        assert acked is False
    assert svc._stop_queue.empty()


def test_stt_cuda_load_borrows_from_diffusion_training(monkeypatch):
    from core.inference import stt_sidecar
    from core.training import diffusion_training_service as dts

    svc = _service()
    monkeypatch.setattr(dts, "get_diffusion_training_service", lambda: svc)
    calls = []
    import contextlib

    @contextlib.contextmanager
    def fake_lend(reason, timeout = 30.0):
        calls.append(reason)
        yield True

    monkeypatch.setattr(svc, "lend_vram", fake_lend)
    with stt_sidecar._lend_diffusion_training_vram("cpu", "x"):
        pass
    assert not calls  # a CPU load takes nothing from the card
    with stt_sidecar._lend_diffusion_training_vram("cuda", "dictation model load"):
        pass
    assert calls == ["dictation model load"]
