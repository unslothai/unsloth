# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Unlisted-arch dispatch of the fused-dequant int8 GEMM (``diffusion_int8_gemm.unlisted_config``): an sm80+ arch with
no ``_ARCH_CONFIG`` tile runs ``_FALLBACK_CONFIG`` only when it passes the bit-exact probe and is not slower than
cuBLAS ``_int_mm``. Capability, probe and timer are faked, so these run on CPU."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from core.inference import diffusion_convrot_quant as cq
from core.inference import diffusion_int8_gemm as g8
from core.inference import diffusion_qwenimage21_fused as qf

_ENVS = (
    g8.INT8_GEMM_ENV,
    g8.INT8_GEMM_UNLISTED_ENV,
    g8.INT8_GEMM_CONVROT_ENV,
    cq.INT8_ROTQUANT_ENV,
)


def _reset():
    for d in (
        g8._DEVICE_CFG,
        g8._DEVICE_TILES,
        g8._UNLISTED_PROBE,
        g8._UNLISTED_OK,
        g8._UNLISTED_TIMING,
    ):
        d.clear()
    g8._UNLISTED_CARDS.clear()


@pytest.fixture(autouse = True)
def _clean(monkeypatch):
    for name in _ENVS:
        monkeypatch.delenv(name, raising = False)
    _reset()
    yield
    _reset()


class _Fake:
    def __init__(
        self,
        monkeypatch,
        cap,
        probe = True,
        timing = None,
    ):
        self.probes, self.timed = [], []
        self.timing = timing if timing is not None else {"fused": 10.0, "stock": 20.0}
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "get_device_capability", lambda index = None: cap)
        monkeypatch.setattr(g8, "is_rocm_torch", lambda t: False)
        monkeypatch.setattr(g8, "_triton_version_ok", lambda version = None: True)
        monkeypatch.setattr(g8, "_kernels", lambda: object())
        monkeypatch.setattr(g8, "_op", lambda: "op")
        monkeypatch.setattr(g8, "_probe", lambda index, cfg: self.probes.append(cfg) or probe)
        monkeypatch.setattr(g8, "_unlisted_latency", self._latency)

    def _latency(self, index, cfg):
        self.timed.append(cfg)
        if isinstance(self.timing, Exception):
            raise self.timing
        return self.timing


@pytest.mark.parametrize(
    "cap, on",
    [
        ((7, 5), False),
        ((8, 0), False),  # listed: own tile
        ((8, 6), True),  # RTX 30xx / A6000 / A10
        ((8, 7), True),
        ((8, 9), False),
        ((9, 0), True),  # H100
        ((10, 0), False),  # measured slower than cuBLAS
        ((10, 3), False),
        ((12, 0), False),
        ((12, 1), True),
    ],
)
def test_arch_table(cap, on):
    assert g8.unlisted_wanted(cap) is on
    if cap >= (8, 0) and cap not in g8._UNLISTED_SKIP:  # exactly one route per sm80+ arch
        assert (g8.arch_config(cap) is not None) is not on


def test_fallback_tile_fits_a_99kb_part():
    bm, bn, bk, _gm, _warps, stages = g8._FALLBACK_CONFIG
    assert (bm * bk + bk * bn) * stages <= 99 * 1024


@pytest.mark.parametrize("env", [g8.INT8_GEMM_UNLISTED_ENV, g8.INT8_GEMM_ENV])
def test_kill_switches(monkeypatch, env):
    monkeypatch.setenv(env, "0")
    fake = _Fake(monkeypatch, (8, 6))
    assert not g8.unlisted_wanted((8, 6))
    assert g8.device_config(0) is None
    assert fake.probes == [] and fake.timed == []


@pytest.mark.parametrize(
    "timing, ok",
    [
        ({"fused": 10.0, "stock": 20.0}, True),
        ({"fused": 20.0, "stock": 20.0}, True),  # a tie is not slower
        ({"fused": 30.0, "stock": 20.0}, False),
        (RuntimeError("capture failed"), False),
    ],
)
def test_latency_gate(monkeypatch, timing, ok):
    fake = _Fake(monkeypatch, (8, 6), timing = timing)
    assert g8.device_config(0) == (g8._FALLBACK_CONFIG if ok else None)
    assert fake.probes == [g8._FALLBACK_CONFIG] and fake.timed == [g8._FALLBACK_CONFIG]
    assert g8._DEVICE_TILES[0] == ()


@pytest.mark.parametrize("env, name", [(g8.INT8_GEMM_UNLISTED_ENV, "1"), (g8.INT8_GEMM_ENV, "1")])
def test_force_skips_the_latency_check_not_the_probe(monkeypatch, env, name):
    monkeypatch.setenv(env, name)
    fake = _Fake(monkeypatch, (8, 6), timing = {"fused": 30.0, "stock": 20.0})
    assert g8.device_config(0) == g8._FALLBACK_CONFIG
    assert fake.probes == [g8._FALLBACK_CONFIG] and fake.timed == []


def test_failed_probe_keeps_stock_and_never_times(monkeypatch):
    fake = _Fake(monkeypatch, (9, 0), probe = False)
    assert g8.device_config(0) is None
    assert fake.timed == []


def test_decision_is_cached_per_device_and_the_switch_acts_between_loads(monkeypatch):
    fake = _Fake(monkeypatch, (8, 6))
    assert g8.device_config(0) == g8._FALLBACK_CONFIG
    assert g8.device_config(0) == g8._FALLBACK_CONFIG
    assert len(fake.probes) == 1 and len(fake.timed) == 1
    monkeypatch.setenv(g8.INT8_GEMM_UNLISTED_ENV, "0")
    assert g8.device_config(0) is None and g8._DEVICE_CFG[0] is None
    monkeypatch.delenv(g8.INT8_GEMM_UNLISTED_ENV)
    assert g8.device_config(0) == g8._FALLBACK_CONFIG
    assert len(fake.probes) == 1 and len(fake.timed) == 1
    assert g8.device_config(1) == g8._FALLBACK_CONFIG  # another card: its own probe and check
    assert len(fake.probes) == 2 and len(fake.timed) == 2


@pytest.mark.parametrize("cap", [(8, 9), (10, 0), (7, 5)])
def test_listed_skipped_and_pre_sm80_archs_never_time(monkeypatch, cap):
    fake = _Fake(monkeypatch, cap)
    g8.device_config(0)
    assert fake.timed == []
    if cap != (8, 9):
        assert fake.probes == [] and g8._DEVICE_CFG[0] is None


def test_failed_launch_sticks_on_an_unlisted_card(monkeypatch):
    _Fake(monkeypatch, (8, 6))
    assert g8.device_config(0) == g8._FALLBACK_CONFIG
    g8._DEVICE_CFG[None] = g8._FALLBACK_CONFIG  # CPU tensors report device index None
    g8._UNLISTED_CARDS.add(None)

    def boom(*args):
        raise RuntimeError("out of resources")

    monkeypatch.setattr(g8, "_launch", boom)
    monkeypatch.setattr(g8, "reference", lambda *args: "stock")
    a, w = torch.zeros(32, 1024, dtype = torch.int8), torch.zeros(64, 1024, dtype = torch.int8)
    assert g8._run(a, w, torch.ones(32), torch.ones(64), None) == "stock"
    assert g8._UNLISTED_PROBE[None] is False
    assert g8.unlisted_config(None) is None


@pytest.mark.parametrize(
    "gemm, rot, swapped",
    [(True, True, True), (True, False, False), (False, True, False), (False, False, False)],
)
def test_rotated_linears_swap_only_with_both_kernels(monkeypatch, gemm, rot, swapped):
    fake = _Fake(monkeypatch, (8, 6), timing = {"fused": 10.0 if gemm else 30.0, "stock": 20.0})
    monkeypatch.setattr(g8, "_unlisted_rotated", lambda index: rot)
    monkeypatch.setattr(cq, "is_rocm_torch", lambda t: False)
    monkeypatch.setattr(qf, "is_rocm_torch", lambda t: False)
    assert g8.rotated_swapped(0) is swapped
    # Qwen-Image-2.1 stock fusions (fused QKV, rotq_h16 via arm_rotated) stand down only when the swap takes them
    assert qf.arch_ok(torch.device("cuda", 0)) is not swapped
    assert qf._swap_covers(0) is swapped
    assert len(fake.timed) <= 1


@pytest.mark.parametrize("timing", [{"fused": 10.0, "stock": 20.0}, {"fused": 30.0, "stock": 20.0}])
def test_gate_takes_no_int8_fused_lock(monkeypatch, timing):
    """``diffusion_int8_fused._finalize`` asks ``rotated_ff`` (so this gate) while holding its lock: a gate that
    took it (``arm_rotated``) hung every load whose fused GEMM stayed off."""
    import threading

    from core.inference import diffusion_int8_fused as f8

    _Fake(monkeypatch, (8, 6), timing = timing)
    monkeypatch.setattr(f8, "_act_quant_device_ok", lambda index: True)
    monkeypatch.setattr(f8, "_act_quant_op", lambda: "actq")
    seen = []
    monkeypatch.setattr(
        cq, "unlisted_device_ok", lambda index, stock = None: seen.append(stock) or True
    )
    out = []
    with f8._LOCK:
        t = threading.Thread(target = lambda: out.append(g8.rotated_swapped(0)), daemon = True)
        t.start()
        t.join(10)
    assert not t.is_alive() and out == [timing["fused"] <= timing["stock"]]
    assert seen == ["actq"]


@pytest.mark.parametrize("cap", [(8, 0), (8, 9), (12, 0)])
def test_listed_archs_keep_the_static_gate(monkeypatch, cap):
    _Fake(monkeypatch, cap)
    assert g8.rotated_swapped(0) is None


def _model():
    lins = torch.nn.ModuleList([torch.nn.Linear(256, 256, bias = False) for _ in range(3)])
    return lins, {lins[0]: None, lins[1]: 256, lins[2]: 256}  # one plain, two rotated


def _swap(monkeypatch, gemm, rot):
    lins, groups = _model()
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda index = None: (8, 6))
    monkeypatch.setattr(
        g8, "_eligible", lambda m: ("v2", groups[m], None, m.weight) if m in groups else None
    )
    monkeypatch.setattr(g8, "_split_cards", lambda t: ())
    monkeypatch.setattr(g8, "device_config", lambda index: g8._FALLBACK_CONFIG if gemm else None)
    monkeypatch.setattr(cq, "rotquant_device_config", lambda index: None)
    monkeypatch.setattr(g8, "_unlisted_rotated", lambda index: rot)
    monkeypatch.setattr(g8, "_op", lambda: "op")
    g8._UNLISTED_CARDS.add(0)
    count = g8._swap_on(lins, None, 0)
    recs = [m.__dict__.get(g8._REC) for m in lins]
    g8.uninstall(lins)
    return count, recs


def test_swap_unlisted_with_rotq_h16_takes_every_linear(monkeypatch):
    count, recs = _swap(monkeypatch, gemm = True, rot = True)
    assert count == 3
    assert [r[2] for r in recs] == [
        False,
        True,
        True,
    ]  # rotation fused into the act quant on the rotated ones
    assert all(r[4] for r in recs)  # fused GEMM on all


def test_swap_unlisted_without_rotq_h16_leaves_rotated_linears_to_main(monkeypatch):
    count, recs = _swap(monkeypatch, gemm = True, rot = False)
    assert count == 1
    assert recs[0] is not None and recs[0][4] and recs[1] is None and recs[2] is None


def test_swap_unlisted_slower_gemm_swaps_nothing(monkeypatch):
    count, recs = _swap(monkeypatch, gemm = False, rot = True)
    assert count == 0 and recs == [None, None, None]


@pytest.mark.parametrize("env, deferred", [(None, True), ("0", False)])
def test_deferred_install_gates_in_an_unlisted_arch(monkeypatch, env, deferred):
    from core.inference import diffusion_int8_fused as f8

    _Fake(monkeypatch, (8, 6))
    if env is not None:
        monkeypatch.setenv(g8.INT8_GEMM_UNLISTED_ENV, env)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(f8, "resident_cuda_device", lambda t: None)
    monkeypatch.setattr(g8, "candidates", lambda t, rotated_only = False: 0 if rotated_only else 3)
    queued = []
    monkeypatch.setattr(f8, "run_on_first_call", lambda t, name, fn: queued.append(name))
    assert g8.install(torch.nn.Linear(64, 64)) == (3 if deferred else 0)
    assert queued == (["int8_gemm"] if deferred else [])


def test_switch_joins_the_compile_cache_key(monkeypatch):
    from core.inference import diffusion_compile_cache as cc

    assert g8.INT8_GEMM_UNLISTED_ENV not in cc.graph_switches()
    monkeypatch.setenv(g8.INT8_GEMM_UNLISTED_ENV, "0")
    assert cc.graph_switches()[g8.INT8_GEMM_UNLISTED_ENV] == "off"
