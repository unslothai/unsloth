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
    g8.INT8_GEMM_RACE_ENV,
    cq.INT8_ROTQUANT_ENV,
)


def _reset():
    for d in (
        g8._DEVICE_CFG,
        g8._DEVICE_TILES,
        g8._UNLISTED_PROBE,
        g8._UNLISTED_OK,
        g8._UNLISTED_TIMING,
        g8._UNLISTED_TILE,
        g8._UNLISTED_RACE,
    ):
        d.clear()
    g8._UNLISTED_CARDS.clear()


@pytest.fixture(autouse = True)
def _clean(monkeypatch, tmp_path):
    for name in _ENVS:
        monkeypatch.delenv(name, raising = False)
    # pre-Triton-3.8 gate; 3.8+ is in test_diffusion_int8_gemm.py
    monkeypatch.setattr(g8, "_triton_at_least", lambda major, minor: False)
    monkeypatch.setattr(g8, "_race_cache_file", lambda: tmp_path / g8._RACE_CACHE_NAME)
    monkeypatch.setattr(g8, "_race_key", lambda index: f"card{index}")
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
        race = None,
    ):
        self.probes, self.timed, self.raced = [], [], []
        self.timing = timing if timing is not None else {"fused": 10.0, "stock": 20.0}
        # None: every race tile fails its probe
        self.race = race
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "get_device_capability", lambda index = None: cap)
        monkeypatch.setattr(g8, "is_rocm_torch", lambda t: False)
        monkeypatch.setattr(g8, "_triton_version_ok", lambda version = None: True)
        monkeypatch.setattr(g8, "_kernels", lambda: object())
        monkeypatch.setattr(g8, "_op", lambda: "op")
        monkeypatch.setattr(g8, "_probe", self._probe_fn(probe))
        monkeypatch.setattr(g8, "_unlisted_latency", self._latency)
        monkeypatch.setattr(g8, "_unlisted_race", self._race)

    def _probe_fn(self, probe):
        def fn(index, cfg):
            self.probes.append(cfg)
            if cfg == g8._FALLBACK_CONFIG:
                return probe
            return self.race is not None and cfg in self.race

        return fn

    def _race(self, index, cfgs):
        self.raced.append(cfgs)
        if isinstance(self.race, Exception):
            raise self.race
        return {cfg: (self.race or {}).get(cfg, 10.0) for cfg in cfgs}

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
    assert fake.probes[0] == g8._FALLBACK_CONFIG and fake.timed == [g8._FALLBACK_CONFIG]
    assert fake.probes[1:] == list(g8._UNLISTED_RACE_TILES)
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
    n = len(fake.probes)
    assert len(fake.timed) == 1
    monkeypatch.setenv(g8.INT8_GEMM_UNLISTED_ENV, "0")
    assert g8.device_config(0) is None and g8._DEVICE_CFG[0] is None
    monkeypatch.delenv(g8.INT8_GEMM_UNLISTED_ENV)
    assert g8.device_config(0) == g8._FALLBACK_CONFIG
    assert len(fake.probes) == n and len(fake.timed) == 1
    assert g8.device_config(1) == g8._FALLBACK_CONFIG  # another card: its own probe and check
    assert len(fake.probes) == 2 * n and len(fake.timed) == 2


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


_A100_TILE, _L4_TILE = g8._UNLISTED_RACE_TILES


def test_race_tiles_fit_a_99kb_part_and_are_the_listed_tiles():
    # sm86 / sm87 parts have 99 KB of shared memory per block; a tile over it would only drop out at the probe
    for bm, bn, bk, _gm, _warps, stages in g8._UNLISTED_RACE_TILES:
        assert (bm * bk + bk * bn) * (stages - 1) <= 99 * 1024
    assert set(g8._UNLISTED_RACE_TILES) <= set(g8._ARCH_CONFIG.values())


def test_race_keeps_the_fastest_probed_tile(monkeypatch):
    fake = _Fake(monkeypatch, (8, 6), race = {_A100_TILE: 9.0, _L4_TILE: 7.0})
    assert g8.device_config(0) == _L4_TILE
    assert fake.raced == [(g8._FALLBACK_CONFIG, _A100_TILE, _L4_TILE)]
    assert g8._UNLISTED_TIMING[0]["tile"] == _L4_TILE and g8._UNLISTED_TIMING[0]["fused"] == 10.0
    assert g8._UNLISTED_TIMING[0]["race"] == {
        g8._FALLBACK_CONFIG: 10.0,
        _A100_TILE: 9.0,
        _L4_TILE: 7.0,
    }
    assert g8.device_config(0) == _L4_TILE and len(fake.raced) == 1


def test_race_tie_keeps_the_fallback(monkeypatch):
    _Fake(monkeypatch, (8, 6), race = {_A100_TILE: 10.0, _L4_TILE: 10.0})
    assert g8.device_config(0) == g8._FALLBACK_CONFIG


def test_race_times_only_tiles_that_pass_their_probe(monkeypatch):
    fake = _Fake(monkeypatch, (8, 6), race = {_A100_TILE: 5.0})
    assert g8.device_config(0) == _A100_TILE
    assert fake.raced == [(g8._FALLBACK_CONFIG, _A100_TILE)]


def test_no_probed_race_tile_never_races(monkeypatch):
    fake = _Fake(monkeypatch, (8, 6))
    assert g8.device_config(0) == g8._FALLBACK_CONFIG
    assert fake.raced == []


def test_failed_race_keeps_the_fallback(monkeypatch):
    _Fake(monkeypatch, (8, 6), race = RuntimeError("capture failed"))
    monkeypatch.setattr(g8, "_probe", lambda index, cfg: True)
    assert g8.device_config(0) == g8._FALLBACK_CONFIG


def test_the_race_winner_still_needs_the_latency_check(monkeypatch):
    fake = _Fake(monkeypatch, (8, 6), timing = {"fused": 30.0, "stock": 20.0}, race = {_L4_TILE: 1.0})
    assert g8.device_config(0) is None
    assert fake.timed == [_L4_TILE]


def test_a_faster_tile_can_pass_the_check_the_fallback_fails(monkeypatch):
    fake = _Fake(monkeypatch, (8, 6), race = {_A100_TILE: 8.0})
    # cuBLAS 20 us: the fallback (25 us) would keep it, the race winner (15 us) beats it
    monkeypatch.setattr(
        g8,
        "_unlisted_latency",
        lambda index, cfg: fake.timed.append(cfg)
        or {"fused": 15.0 if cfg == _A100_TILE else 25.0, "stock": 20.0},
    )
    assert g8.device_config(0) == _A100_TILE
    assert fake.timed == [_A100_TILE]
    assert g8._UNLISTED_TIMING[0]["tile"] == _A100_TILE and g8._UNLISTED_TIMING[0]["fused"] == 15.0


@pytest.mark.parametrize("env", [g8.INT8_GEMM_UNLISTED_ENV, g8.INT8_GEMM_ENV])
def test_force_never_races(monkeypatch, env):
    monkeypatch.setenv(env, "1")
    fake = _Fake(monkeypatch, (8, 6), race = {_L4_TILE: 1.0})
    assert g8.device_config(0) == g8._FALLBACK_CONFIG
    assert fake.raced == []


@pytest.mark.parametrize("cap", [(8, 0), (8, 9), (12, 0)])
def test_listed_archs_never_race(monkeypatch, cap):
    fake = _Fake(monkeypatch, cap, race = {_A100_TILE: 1.0, _L4_TILE: 1.0})
    assert g8.device_config(0) == g8._ARCH_CONFIG[cap]
    assert fake.raced == [] and fake.timed == []


def test_race_winner_is_reused_by_the_next_process(monkeypatch):
    fake = _Fake(monkeypatch, (8, 6), race = {_A100_TILE: 9.0, _L4_TILE: 7.0})
    assert g8.device_config(0) == _L4_TILE and len(fake.raced) == 1
    _reset()  # a new process: in-memory verdicts gone, the file stays
    fake.probes.clear()
    assert g8.device_config(0) == _L4_TILE
    assert len(fake.raced) == 1  # no second race
    assert fake.probes == [g8._FALLBACK_CONFIG, _L4_TILE]  # only the kept tile is probed again


def test_a_kept_fallback_is_reused_without_probing_the_race_tiles(monkeypatch):
    fake = _Fake(monkeypatch, (8, 6), race = {_A100_TILE: 11.0, _L4_TILE: 12.0})
    assert g8.device_config(0) == g8._FALLBACK_CONFIG
    _reset()
    fake.probes.clear()
    assert g8.device_config(0) == g8._FALLBACK_CONFIG
    assert len(fake.raced) == 1 and fake.probes == [g8._FALLBACK_CONFIG]


def test_another_card_or_toolchain_races_again(monkeypatch):
    fake = _Fake(monkeypatch, (8, 6), race = {_A100_TILE: 9.0, _L4_TILE: 7.0})
    assert g8.device_config(0) == _L4_TILE
    _reset()
    monkeypatch.setattr(g8, "_race_key", lambda index: "other-card")
    assert g8.device_config(0) == _L4_TILE and len(fake.raced) == 2


def test_kept_tile_that_now_fails_its_probe_falls_back(monkeypatch):
    fake = _Fake(monkeypatch, (8, 6), race = {_L4_TILE: 7.0})
    assert g8.device_config(0) == _L4_TILE
    _reset()
    fake.race = {}  # the kept tile no longer passes its probe
    assert g8.device_config(0) == g8._FALLBACK_CONFIG


@pytest.mark.parametrize("raw", ["", "{not json", '{"card0": [1, 2, 3]}', '{"card0": "x"}'])
def test_unreadable_or_foreign_cache_entries_are_a_miss(monkeypatch, tmp_path, raw):
    (tmp_path / g8._RACE_CACHE_NAME).write_text(raw)
    fake = _Fake(monkeypatch, (8, 6), race = {_L4_TILE: 7.0})
    assert g8.device_config(0) == _L4_TILE and len(fake.raced) == 1


@pytest.mark.parametrize("raw", ["0", "off"])
def test_race_switch_off_keeps_the_fallback_and_never_reads_the_file(monkeypatch, tmp_path, raw):
    (tmp_path / g8._RACE_CACHE_NAME).write_text('{"card0": [256, 128, 128, 8, 8, 3]}')
    monkeypatch.setenv(g8.INT8_GEMM_RACE_ENV, raw)
    fake = _Fake(monkeypatch, (8, 6), race = {_L4_TILE: 7.0})
    assert g8.device_config(0) == g8._FALLBACK_CONFIG
    assert fake.raced == [] and fake.probes == [g8._FALLBACK_CONFIG]


def test_race_switch_acts_between_loads_after_a_winner_is_kept(monkeypatch):
    fake = _Fake(monkeypatch, (8, 6), race = {_L4_TILE: 7.0})
    assert g8.device_config(0) == _L4_TILE
    monkeypatch.setenv(g8.INT8_GEMM_RACE_ENV, "0")
    assert g8.device_config(0) == g8._FALLBACK_CONFIG
    assert fake.timed[-1] == g8._FALLBACK_CONFIG
    monkeypatch.delenv(g8.INT8_GEMM_RACE_ENV)
    assert g8.device_config(0) == _L4_TILE
    assert len(fake.raced) == 1  # the kept winner, not a second race


def test_cache_keeps_a_bounded_number_of_cards(monkeypatch, tmp_path):
    import json

    path = tmp_path / g8._RACE_CACHE_NAME
    path.write_text(json.dumps({f"old{i}": list(g8._FALLBACK_CONFIG) for i in range(40)}))
    _Fake(monkeypatch, (8, 6), race = {_L4_TILE: 7.0})
    assert g8.device_config(0) == _L4_TILE
    table = json.loads(path.read_text())
    assert len(table) == g8._RACE_CACHE_MAX and table["card0"] == list(_L4_TILE)


def test_race_holds_each_tile_under_load_at_render_rows(monkeypatch):
    seen = {}

    def fake_graph_latency(
        index,
        fns,
        shapes = g8._UNLISTED_LATENCY_SHAPES,
        hold_s = 0.0,
    ):
        seen.update(shapes = shapes, hold_s = hold_s, arms = tuple(fns))
        return {k: 1.0 for k in fns}

    monkeypatch.setattr(g8, "_graph_latency", fake_graph_latency)
    g8._unlisted_race(0, (g8._FALLBACK_CONFIG, _L4_TILE))
    assert seen["hold_s"] == g8._RACE_HOLD_S > 0
    assert all(m == 4096 for m, _n, _k in seen["shapes"]) and len(seen["shapes"]) == 3
    assert seen["arms"] == (g8._FALLBACK_CONFIG, _L4_TILE)


class _Ev:
    def __init__(self, clock):
        self.clock, self.t = clock, None

    def record(self):
        self.t = self.clock[0]

    def synchronize(self):
        pass

    def elapsed_time(self, other):
        return (other.t - self.t) / 1000.0  # ms


def test_held_timing_skips_the_settle_window(monkeypatch):
    # a part that runs 1 us per replay for the first 50 ms, then 2 us (clocked down): the held time is the slow one
    import time

    clock = [0.0]
    wall = [0.0]
    monkeypatch.setattr(time, "perf_counter", lambda: wall[0])

    class _G:
        def replay(self):
            dt = 1.0 if wall[0] < 0.05 else 2.0
            clock[0] += dt
            wall[0] += 0.001

    us = g8._held_us(_G(), 0.1, _Ev(clock), _Ev(clock))
    assert us == pytest.approx(2.0)
