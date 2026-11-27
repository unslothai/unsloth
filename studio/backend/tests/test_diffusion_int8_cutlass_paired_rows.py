# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""MiniMax-H3-Image's large-row CUTLASS races take the pair check; other image families keep their winners (CPU).

Runs the real plan/race/pair check with timers replaced by a card clock model (``_Card``, L4 and RTX 6000 Ada)."""

from __future__ import annotations

import contextlib
import dataclasses
import math
import types

import pytest

from core.inference import diffusion_int8_cutlass as cut
from core.inference import diffusion_int8_gemm as g8

torch = pytest.importorskip("torch")


@dataclasses.dataclass(frozen = True)
class _Power:
    boost_mhz: float
    settle_s: float
    steady_mhz: dict  # {"today": alone, "cutlass": alone, "paired": a (tile, today's path) pair}


# L4: #19807's measured clocks. Ada: R9's step clocks, with an alone CUTLASS clock where tiles win alone.
POWER = {
    "l4": _Power(2040.0, 0.05, {"today": 1290.0, "cutlass": 855.0, "paired": 1171.0}),
    "rtx6000_ada": _Power(2505.0, 0.05, {"today": 1240.0, "cutlass": 1180.0, "paired": 1035.0}),
    # clock-stable card (A100-like): sustained timings match the short ones, so the short timings decide
    "stable": _Power(1410.0, 0.05, {"today": 1410.0, "cutlass": 1410.0, "paired": 1410.0}),
}
# tile speed at equal clock relative to today's path, and rows per tile
_TILE_SPEED = {0: 0.90, 1: 0.93, 2: 0.97}
_TILE_ROWS = {0: 128, 1: 128, 2: 256}

# (label, N, K, Linears, rows per call), from each model's config: H3-Image at 2048x2048, others at 1024x1024.
INVENTORIES = {
    "MiniMaxH3ImageTransformer2DModel": (
        ("qkv", 7168, 5376, 150, 4100),
        ("out", 5376, 7168, 50, 4100),
        ("ff_in", 28672, 5376, 50, 4100),
        ("ff_out", 5376, 14336, 50, 4100),
        ("context", 5376, 5120, 1, 512),
    ),
    "Flux2Transformer2DModel": (
        ("double_qkv", 3072, 3072, 30, 4096),
        ("double_ff_in", 18432, 3072, 10, 4096),
        ("double_ff_out", 3072, 9216, 10, 4096),
        ("single_in", 27648, 3072, 20, 4096),
        ("single_out", 3072, 12288, 20, 4096),
        ("txt_qkv", 3072, 3072, 30, 512),
    ),
    "QwenImageTransformer2DModel": (
        ("attn_img", 3072, 3072, 240, 4096),
        ("mlp_in_img", 12288, 3072, 60, 4096),
        ("mlp_out_img", 3072, 12288, 60, 4096),
        ("attn_txt", 3072, 3072, 240, 256),
    ),
}


def _is_today(name) -> bool:
    return name in ("today", "cublas")


class _Card:
    """Short bursts run at the current clock; sustained runs relax toward steady; paired calls use the pair clock."""

    def __init__(self, power: _Power):
        self.p = power
        self.clock = power.boost_mhz
        self.shape = (0, 0, 0)
        self.pairing = False
        self.paired_timings = 0

    def begin(self, m, n, k):
        self.shape = (int(m), int(n), int(k))
        self.clock = self.p.boost_mhz

    def _steady(self, name):
        if self.shape[0] < cut._LARGE_M:
            return self.p.boost_mhz
        if _is_today(name):
            return self.p.steady_mhz["today"]
        return self.p.steady_mhz["paired" if self.pairing else "cutlass"]

    def _work(self, name):
        m, n, k = self.shape
        speed = 1.0 if _is_today(name) else _TILE_SPEED[int(name)]
        return m * n * k * 1e-6 * (speed + (1.0 if self.pairing else 0.0))

    def time_us(
        self,
        fns,
        rounds = 3,
        reps = 3,
        warm = 1,
    ):
        return {name: self._work(name) / self.clock for name in fns}

    def time_sustained_us(
        self,
        fns,
        settle_s = 0.25,
        measure_s = 0.25,
    ):
        self.paired_timings += int(self.pairing)
        out = {}
        for name in fns:
            steady, tau, f0 = self._steady(name), self.p.settle_s, self.clock
            a, b = math.exp(-settle_s / tau), math.exp(-(settle_s + measure_s) / tau)
            mean = steady + (f0 - steady) * tau / measure_s * (a - b)
            self.clock = steady + (f0 - steady) * b
            out[name] = self._work(name) / mean
        return out


_STATE = (
    g8._CUT_PLAN,
    g8._CUT_FAMILY,
    g8._CUT_RACES,
    g8._CUT_DONE,
    g8._CUT_CKEY,
    g8._CUT_ROWS,
    g8._CUT_ROWS_PENDING,
    g8._CUT_LIVE,
    g8._CUT_TRITON_ON,
    cut._PROBED,
)


@pytest.fixture(autouse = True)
def _clean(monkeypatch):
    for env in (
        cut.PAIRED_ENV,
        cut.SUSTAINED_ENV,
        cut.CUTLASS_ENV,
        cut.ROW_RACE_ENV,
        cut.LIVE_RACE_ENV,
        cut.HUGE_RACE_ENV,
        cut.FILL_TIE_ENV,
    ):
        monkeypatch.delenv(env, raising = False)
    saved = [dict(d) for d in _STATE]
    paired = dict(getattr(cut, "_PAIRED_ROWS", {}))
    yield
    for d, old in zip(_STATE, saved):
        d.clear()
        d.update(old)
    if hasattr(cut, "_PAIRED_ROWS"):
        cut._PAIRED_ROWS.clear()
        cut._PAIRED_ROWS.update(paired)


# wrapped afresh by each load (a wrapper of a wrapper would count twice)
_REAL_RACE, _REAL_PAIRED = cut.race, cut.paired_cost_us


def _run_load(
    monkeypatch,
    card: str,
    dit: str,
    fresh: bool = True,
) -> dict:
    """One int8 load of ``dit`` on the modelled card; returns route, race inputs and image-row pair count.
    ``fresh=False``: keep the process state of earlier loads and flip the switch through ``_swap_on``, as production."""
    if fresh:
        for d in _STATE:
            d.clear()
    sim = _Card(POWER[card])
    races: list = []
    real_race, real_paired = _REAL_RACE, _REAL_PAIRED

    paired_at: list = []

    def race(index, w, ws, bias, xs_dtype, today, m, *a, **k):
        races.append(
            (
                int(w.shape[0]),
                int(w.shape[1]),
                str(xs_dtype),
                str(ws.dtype),
                today is not None,
                int(m),
            )
        )
        before = sim.paired_timings
        out = real_race(index, w, ws, bias, xs_dtype, today, m, *a, **k)
        if sim.paired_timings != before:
            paired_at.append(int(m))
        return out

    def paired_cost_us(fns, base_name):
        sim.pairing = True
        try:
            return real_paired(fns, base_name)
        finally:
            sim.pairing = False

    def eligible(a, w, xs, ws, bias, out_dtype):
        sim.begin(a.shape[0], w.shape[0], w.shape[1])
        return True

    tiles = tuple(sorted(_TILE_SPEED))
    to = torch.Tensor.to

    def to_device(self, *a, **k):
        if a and isinstance(a[0], (torch.device, str)) and str(a[0]).startswith("cuda"):
            return self
        return to(self, *a, **k)

    def lazy_randint(
        low,
        high,
        size,
        generator = None,
        dtype = None,
        device = None,
        **kw,
    ):
        return torch.empty(size, dtype = dtype or torch.int64)

    monkeypatch.setattr(cut, "race", race)
    monkeypatch.setattr(cut, "paired_cost_us", paired_cost_us)
    monkeypatch.setattr(cut, "time_us", sim.time_us)
    monkeypatch.setattr(cut, "time_sustained_us", sim.time_sustained_us)
    monkeypatch.setattr(cut, "eligible", eligible)
    monkeypatch.setattr(cut, "has_kernel", lambda *a, **k: True)
    monkeypatch.setattr(cut, "probed_tiles", lambda index: tiles)
    monkeypatch.setattr(
        cut, "tiles", lambda fam: tuple((_TILE_ROWS[t], 128, 64, 0, 0, 0, 0, 0) for t in tiles)
    )
    monkeypatch.setattr(g8, "cutlass_wanted", lambda cap: True)
    monkeypatch.setattr(g8, "_cut_cache_key", lambda index: None)
    monkeypatch.setattr(g8, "_xs_dtype", lambda *a, **k: torch.float32)
    monkeypatch.setattr(g8, "_huge_race_fits", lambda *a, **k: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: (8, 9))
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a, **k: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(torch.cuda, "device", lambda *a, **k: contextlib.nullcontext())
    monkeypatch.setattr(torch, "randint", lazy_randint)
    monkeypatch.setattr(torch.Tensor, "to", to_device)

    # absent on a tree without the family switch
    switch = getattr(cut, "set_paired_rows", None)
    if not fresh:
        monkeypatch.setattr(g8, "_split_cards", lambda t: [0])
        assert g8._swap_on(type(dit, (torch.nn.Module,), {})(), None, 0) == 0
    elif switch is not None:
        switch(0, type(dit, (), {})())
    g8._CUT_TRITON_ON[0] = True
    inv = INVENTORIES[dit]
    weights, recs = {}, []
    for _label, n, k, _count, _rows in inv:
        if (n, k) not in weights:
            w = types.SimpleNamespace(
                qdata = torch.empty(n, k, dtype = torch.int8),
                scale = torch.empty(n, 1, dtype = torch.bfloat16),
            )
            weights[(n, k)] = w
            recs.append((types.SimpleNamespace(bias = None), ("v2", None, False, w, True)))
    g8._cutlass_plan(0, recs, True)
    monkeypatch.setattr(g8, "_launch_or_ref", lambda *a, **k: None)
    for _label, n, k, _count, rows in inv:
        if cut._LARGE_M <= rows < cut._HUGE_M:
            a = torch.empty(rows, k, dtype = torch.int8)
            xs = torch.empty(rows, dtype = torch.float32)
            g8._live_row_race(
                0,
                a,
                torch.empty(n, k, dtype = torch.int8),
                xs,
                torch.empty(n, dtype = torch.bfloat16),
                None,
            )
    route = {}
    for label, n, k, _count, rows in inv:
        entry = (g8._CUT_PLAN.get(0) or {}).get(
            g8._cut_key(n, k, torch.float32, torch.bfloat16, False)
        ) or {}
        tile = cut.slot(rows, entry) if entry else None
        route[label] = "triton" if tile is None else f"tile{tile}"
    image_pairs = sum(1 for m in paired_at if m < cut._HUGE_M)
    return {"route": route, "races": races, "image_pairs": image_pairs}


def _image_rows_on_cutlass(r: dict, dit: str) -> int:
    return sum(
        count
        for label, _n, _k, count, rows in INVENTORIES[dit]
        if cut._LARGE_M <= rows < cut._HUGE_M and r["route"][label] != "triton"
    )


H3I = "MiniMaxH3ImageTransformer2DModel"


def test_ada_h3_image_2048_keeps_triton_on_image_rows(monkeypatch):
    """Ada: H3-Image tiles win alone but lose paired, so all 300 Linears keep today's path (origin/main keeps CUTLASS)."""
    r = _run_load(monkeypatch, "rtx6000_ada", H3I)
    assert _image_rows_on_cutlass(r, H3I) == 0, r["route"]
    # 5 load races at 4096 rows + 4 live races at 4160 rows
    assert r["image_pairs"] == 9, r
    monkeypatch.setenv(cut.PAIRED_ENV, "0")
    off = _run_load(monkeypatch, "rtx6000_ada", H3I)
    assert _image_rows_on_cutlass(off, H3I) == 300, off["route"]
    assert off["image_pairs"] == 0
    assert off["races"] == r["races"]


def test_l4_h3_image_routing_unchanged(monkeypatch):
    """L4: tiles already lose alone, so no winner reaches the pair check."""
    r = _run_load(monkeypatch, "l4", H3I)
    assert r["image_pairs"] == 0, r
    monkeypatch.setenv(cut.PAIRED_ENV, "0")
    off = _run_load(monkeypatch, "l4", H3I)
    assert r["route"] == off["route"] and r["races"] == off["races"]
    assert _image_rows_on_cutlass(r, H3I) == 0


# origin/main's routing for the other image families
_MAIN_ROUTES = {
    ("rtx6000_ada", "Flux2Transformer2DModel"): {
        "double_qkv": "tile0",
        "double_ff_in": "tile0",
        "double_ff_out": "tile0",
        "single_in": "tile0",
        "single_out": "tile0",
        "txt_qkv": "tile0",
    },
    ("rtx6000_ada", "QwenImageTransformer2DModel"): {
        "attn_img": "tile0",
        "mlp_in_img": "tile0",
        "mlp_out_img": "tile0",
        "attn_txt": "tile0",
    },
    ("l4", "Flux2Transformer2DModel"): {
        "double_qkv": "triton",
        "double_ff_in": "triton",
        "double_ff_out": "triton",
        "single_in": "triton",
        "single_out": "triton",
        "txt_qkv": "tile0",
    },
    ("l4", "QwenImageTransformer2DModel"): {
        "attn_img": "triton",
        "mlp_in_img": "triton",
        "mlp_out_img": "triton",
        "attn_txt": "tile0",
    },
}


@pytest.mark.parametrize("card", ["rtx6000_ada", "l4"])
@pytest.mark.parametrize("dit", ["Flux2Transformer2DModel", "QwenImageTransformer2DModel"])
def test_other_image_families_unchanged(monkeypatch, card, dit):
    """FLUX.2 klein and Qwen-Image-2.1 never pair below video rows: routing matches origin/main and pair check off."""
    r = _run_load(monkeypatch, card, dit)
    assert r["image_pairs"] == 0, r
    assert r["route"] == _MAIN_ROUTES[(card, dit)], r["route"]
    monkeypatch.setenv(cut.PAIRED_ENV, "0")
    off = _run_load(monkeypatch, card, dit)
    assert off["route"] == r["route"] and off["races"] == r["races"]
    assert not getattr(cut, "paired_rows", lambda index: False)(0)


def test_paired_for_rows_by_family(monkeypatch):
    assert cut.PAIRED_ROW_FAMILIES == (H3I,)
    cut.set_paired_rows(0, type("Flux2Transformer2DModel", (), {})())
    assert not cut.paired_for(4160, 0) and cut.paired_for(cut._RACE_M_HUGE, 0)
    assert cut.set_paired_rows(0, type(H3I, (), {})())
    assert cut.paired_for(4160, 0) and cut.paired_for(1024, 0) and not cut.paired_for(512, 0)
    assert not cut.paired_for(4160)
    assert not cut.paired_for(4160, 1)
    monkeypatch.setenv(cut.SUSTAINED_ENV, "huge")
    assert not cut.paired_for(4160, 0) and cut.paired_for(cut._RACE_M_HUGE, 0)
    monkeypatch.setenv(cut.SUSTAINED_ENV, "large")
    monkeypatch.setenv(cut.PAIRED_ENV, "0")
    assert not cut.paired_for(4160, 0) and not cut.paired_for(cut._RACE_M_HUGE, 0)


def test_swap_sets_the_family_switch_first(monkeypatch):
    """A FLUX.2 load after H3-Image on the same card races unpaired again, even on an early return."""
    cut.set_paired_rows(0, type(H3I, (), {})())
    monkeypatch.setattr(g8, "_split_cards", lambda t: [0])
    flux = type("Flux2Transformer2DModel", (torch.nn.Module,), {})()
    assert g8._swap_on(flux, None, 0) == 0
    assert not cut.paired_rows(0)
    h3i = type(H3I, (torch.nn.Module,), {})()
    g8._swap_on(h3i, None, 0)
    assert cut.paired_rows(0)


def test_race_cache_key_splits_only_the_paired_family(monkeypatch):
    """Only the paired family gets its own cache key (H3 video shares its weight shapes but races unpaired)."""
    monkeypatch.setattr(
        torch.cuda, "get_device_name", lambda *a, **k: "NVIDIA RTX 6000 Ada Generation"
    )
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: (8, 9))
    monkeypatch.setattr(cut, "library_path", lambda: None)
    monkeypatch.setattr(g8, "_sources_digest", lambda lib: "digest")
    cut.set_paired_rows(0, type("QwenImageTransformer2DModel", (), {})())
    plain = g8._cut_cache_key(0)
    assert plain and "paired_rows" not in plain
    cut.set_paired_rows(0, type(H3I, (), {})())
    assert g8._cut_cache_key(0) == plain + "|paired_rows"
    assert g8._cut_cache_key(1) == g8._cut_cache_key(1).replace("|paired_rows", "")


def test_h3_image_after_h3_video_in_one_process(monkeypatch):
    """H3 video (same 5 weight shapes, unpaired image-row races) then H3-Image on one card, no state reset in between:
    the second load must race paired again, not reuse the video load's in-process plan or live races."""
    video = "MiniMaxH3VideoTransformer3DModel"
    monkeypatch.setitem(INVENTORIES, video, INVENTORIES[H3I])
    vid = _run_load(monkeypatch, "rtx6000_ada", video)
    assert _image_rows_on_cutlass(vid, video) == 300 and vid["image_pairs"] == 0
    img = _run_load(monkeypatch, "rtx6000_ada", H3I, fresh = False)
    assert _image_rows_on_cutlass(img, H3I) == 0, img["route"]
    assert img["image_pairs"] == 9, img
    # and back: a video load after H3-Image keeps its unpaired winners
    again = _run_load(monkeypatch, "rtx6000_ada", video, fresh = False)
    assert again["route"] == vid["route"] and again["image_pairs"] == 0


def test_h3_image_short_timing_winners_skip_the_pair_check(monkeypatch):
    """Below video rows the pair check needs the sustained timings to have decided: a clock-stable card keeps its
    short-timing winners unpaired, as before this switch."""
    r = _run_load(monkeypatch, "stable", H3I)
    assert r["image_pairs"] == 0, r
    assert _image_rows_on_cutlass(r, H3I) == 300, r["route"]
