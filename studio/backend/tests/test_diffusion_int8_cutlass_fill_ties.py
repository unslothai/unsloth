# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A CUTLASS race's near-tied winners go to the tile whose last row tile is fullest at the race's row count (CPU).

L4, 3072x9216 at the 4224-row bucket of a 1024^2 FLUX.3 joint call: the race kept a 256-row tile that won by 2.1%
(1985.3 vs 2027.5 us) and then ran at 2252 us per call in the pipeline against the 128-row tile's 1779. At 4224 rows a
256-row tile leaves its 17th row tile half full, a 128-row tile fills all 33."""

from __future__ import annotations

import contextlib
import random

import pytest

from core.inference import diffusion_int8_cutlass as cut

torch = pytest.importorskip("torch")

# tile -> rows per tile, the sm_8x table (TBM of ui8_tile_info)
_ROWS = {0: 128, 1: 256, 2: 128, 3: 128, 4: 64, 5: 128, 7: 64, 8: 256, 9: 128, 12: 128, 13: 256}
# the misfired live race (sustained timings, us)
_MISFIRE = {"today": 2100.5, 0: 2027.5, 1: 1985.3, 2: 2033.9}


@pytest.fixture(autouse = True)
def _env(monkeypatch):
    monkeypatch.delenv("UNSLOTH_DIFFUSION_INT8_CUTLASS_FILL_TIES", raising = False)


def _fastest_rule(timing, top, base_name):
    """The race's pick before tie-breaking: the fastest tile, if it beats today's path by the margin."""
    best = min(top, key = lambda t: timing[t])
    return best if timing[best] < timing[base_name] * (1.0 - cut._MARGIN) else None


def test_the_measured_misfire_keeps_the_full_128_row_tile():
    assert cut.pick(_MISFIRE, [0, 1, 2], "today", 4224, _ROWS) == (0, 1)


def test_the_kill_switch_keeps_the_fastest_winner(monkeypatch):
    monkeypatch.setenv(cut.FILL_TIE_ENV, "0")
    assert cut.pick(_MISFIRE, [0, 1, 2], "today", 4224, _ROWS) == (1, 1)


def test_row_counts_every_tile_height_divides_keep_the_fastest_winner():
    rng = random.Random(0)
    for _ in range(2000):
        top = rng.sample(sorted(_ROWS), 3)
        timing = {t: rng.uniform(900.0, 1100.0) for t in top}
        timing["today"] = rng.uniform(900.0, 1150.0)
        for m in (512, 1024, 4096, 16384):
            got, _fastest = cut.pick(timing, top, "today", m, _ROWS)
            assert got == _fastest_rule(timing, top, "today"), (m, timing)


def test_a_partial_tile_past_the_tie_band_still_wins():
    timing = {"today": 2100.0, 0: 2027.5, 1: 1950.0}  # 1 is 3.8% faster
    assert cut.pick(timing, [0, 1], "today", 4224, _ROWS) == (1, 1)


def test_a_full_tile_that_loses_to_today_is_never_taken():
    timing = {
        "today": 2050.0,
        0: 2030.0,
        1: 1985.3,
    }  # 0 is within the tie band but not 2% under today
    assert cut.pick(timing, [0, 1], "today", 4224, _ROWS) == (1, 1)


def test_no_winner_keeps_todays_path():
    timing = {"today": 2000.0, 0: 1990.0, 1: 1980.0}
    assert cut.pick(timing, [0, 1], "today", 4224, _ROWS) == (None, None)


def test_equal_fill_keeps_the_faster_tile():
    timing = {"cublas": 2400.0, 0: 2030.0, 2: 2000.0, 1: 1990.0}
    assert cut.pick(timing, [0, 2, 1], "cublas", 4224, _ROWS) == (2, 1)


def test_unknown_tile_rows_keep_the_fastest_winner():
    assert cut.pick(_MISFIRE, [0, 1, 2], "today", 4224, {0: 128, 2: 128}) == (1, 1)


def test_last_tile_fill():
    assert cut.last_tile_fill(4224, 128) == 1.0
    assert cut.last_tile_fill(4224, 256) == 0.5
    assert cut.last_tile_fill(4176, 256) == pytest.approx(80 / 256)
    assert cut.last_tile_fill(4096, 256) == 1.0


def _fake_race(monkeypatch, timing):
    """``cut.race`` on the CPU: the card, library and timings faked, the pick its own."""
    dims = tuple((_ROWS.get(t, 128), 128, 64, 64, 64, 64, 3, 1) for t in range(15))
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda index: (8, 9))
    monkeypatch.setattr(torch.cuda, "device", lambda dev: contextlib.nullcontext())
    monkeypatch.setattr(cut, "probed_tiles", lambda index: tuple(range(15)))
    monkeypatch.setattr(cut, "tiles", lambda fam: dims)
    monkeypatch.setattr(cut, "has_kernel", lambda *a, **k: True)
    monkeypatch.setattr(cut, "eligible", lambda *a, **k: True)
    monkeypatch.setattr(cut, "sustained_for", lambda m: False)
    monkeypatch.setattr(
        cut,
        "time_us",
        lambda fns, **k: {name: float(timing.get(name, 9999.0)) for name in fns},
    )
    w = torch.randint(-127, 128, (64, 64), dtype = torch.int8)
    ws = torch.rand(64).to(torch.bfloat16)
    return lambda m: cut.race(0, w, ws, None, torch.bfloat16, lambda *a: None, m)


def test_the_race_keeps_the_full_tile_at_a_row_bucket(monkeypatch):
    race = _fake_race(monkeypatch, _MISFIRE)
    r = race(4224)
    assert r["tile"] == 0 and r["fill_tie"] == 1
    assert r["us"] == {"today": 2100.5, "1": 1985.3, "0": 2027.5, "2": 2033.9}


def test_the_race_at_4096_rows_is_unchanged(monkeypatch):
    race = _fake_race(monkeypatch, _MISFIRE)
    r = race(4096)
    assert r["tile"] == 1 and "fill_tie" not in r
