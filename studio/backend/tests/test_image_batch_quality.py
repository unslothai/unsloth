# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A measured batching win serves only where an offline G2c verdict passes for the family, card and execution path;
a missing or failing verdict keeps today's serial rules."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

from core.inference import diffusion_batch_quality as quality
from core.inference import diffusion_batch_race as race

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_image_batch_infra import _engine, _fake_generate, _verdicts  # noqa: E402

CARD = "NVIDIA RTX PRO 6000 Blackwell Server Edition"
PASS = {"verdict": "PASS", "n": 24, "upper": 0.006}


@pytest.fixture(autouse = True)
def _local_file(monkeypatch, tmp_path):
    monkeypatch.setattr(quality, "_file", lambda: tmp_path / quality.FILE_NAME)
    monkeypatch.setattr(quality, "SHIPPED", {})


def test_a_speed_winning_race_with_no_g2c_pass_stays_serial(monkeypatch):
    _verdicts(monkeypatch, {2: {"promoted": True, "arm": "batched"}, 4: {"promoted": False}})
    for engine in (_engine(device_name = CARD), _engine(device_name = CARD, cuda_graphs = (object(),))):
        assert engine.request_batching()["mode"] == "serial"
        assert engine.coalesce_batch_sizes(1024, 1024, 4.0) == frozenset()


@pytest.mark.parametrize(
    "stored",
    [
        {"verdict": "FAIL", "n": 24, "upper": 0.0136},
        {"verdict": "PASS", "n": 8, "upper": 0.004},  # too few pairs
        {"verdict": "PASS", "n": 24, "upper": 0.02},  # bound over eps whatever the label says
    ],
)
def test_a_failing_or_underpowered_verdict_stays_serial(monkeypatch, stored):
    _verdicts(monkeypatch, {2: {"promoted": True, "arm": "batched"}})
    assert quality.record("flux.2-klein", CARD, "graph", stored)
    engine = _engine(device_name = CARD)
    assert engine.request_batching()["mode"] == "serial"
    assert engine.coalesce_batch_sizes(1024, 1024, 4.0) == frozenset()


def test_a_pass_on_the_raced_path_lifts_the_serial_rule(monkeypatch):
    _verdicts(monkeypatch, {2: {"promoted": True, "arm": "batched"}, 4: {"promoted": False}})
    assert quality.record("flux.2-klein", CARD, "graph", PASS)
    engine = _engine(device_name = CARD)
    assert engine.request_batching()["mode"] == "measured"
    assert engine.coalesce_batch_sizes(1024, 1024, 4.0) == frozenset({2})


def test_the_verdict_is_per_path_family_and_card(monkeypatch):
    _verdicts(monkeypatch, {2: {"promoted": True, "arm": "batched"}})
    quality.record(
        "flux.2-klein", CARD, "eager", PASS
    )  # the eager path passed, the graphed arm won
    assert _engine(device_name = CARD).coalesce_batch_sizes(1024, 1024, 4.0) == frozenset()
    quality.record("flux.2-klein", "NVIDIA A100-SXM4-40GB", "graph", PASS)  # another card
    assert _engine(device_name = CARD).coalesce_batch_sizes(1024, 1024, 4.0) == frozenset()
    quality.record("hidream", CARD, "graph", PASS)  # another family
    assert _engine(device_name = CARD).coalesce_batch_sizes(1024, 1024, 4.0) == frozenset()
    quality.record("flux.2-klein", CARD, "any", PASS)
    assert _engine(device_name = CARD).coalesce_batch_sizes(1024, 1024, 4.0) == frozenset({2})


def test_no_card_identity_is_no_verdict(monkeypatch):
    _verdicts(monkeypatch, {2: {"promoted": True, "arm": "batched"}})
    quality.record("flux.2-klein", CARD, "any", PASS)
    assert _engine().coalesce_batch_sizes(1024, 1024, 4.0) == frozenset()  # no device, no class


def test_streamed_loads_keep_todays_rule_and_a_loss_still_drops_its_size(monkeypatch):
    # The gate only holds back promotions; it never adds a size or blocks the streamed default.
    _verdicts(monkeypatch, {4: {"promoted": False}})
    engine = _engine(device_name = CARD, offload_policy = "sequential")
    assert engine.request_batching()["mode"] == "any"
    sizes = engine.coalesce_batch_sizes(1024, 1024)
    assert 4 not in sizes and {2, 3} <= sizes


def test_the_gate_names_why_a_speed_win_does_not_serve():
    state = types.SimpleNamespace(
        family = types.SimpleNamespace(name = "flux.3-klein"), device_name = CARD
    )
    won = {"promoted": True, "arm": "batched_eager", "speedup_lb": 1.14}
    held = quality.gate(won, state)
    assert (
        held["promoted"] is False
        and "no G2c verdict" in held["quality_gate"]
        and "(eager)" in held["quality_gate"]
    )
    assert won["promoted"] is True, "the stored speed verdict is not rewritten"
    quality.record("flux.3-klein", CARD, "eager", {"verdict": "FAIL", "n": 24, "upper": 0.02})
    assert "does not pass" in quality.gate(won, state)["quality_gate"]
    quality.record("flux.3-klein", CARD, "eager", PASS)
    assert quality.gate(won, state) is won
    with pytest.raises(ValueError):
        quality.record("flux.3-klein", CARD, "compiled", PASS)


def test_the_race_itself_still_records_the_speed_verdict(monkeypatch):
    # Persisting the speed verdict is unchanged; serving it is what the gate decides.
    stored = {}
    monkeypatch.setattr(
        race, "record", lambda state, w, h, g, k, v, steps = None: stored.setdefault(k, v) is v
    )
    generate, clock, _ = _fake_generate(lambda rows: 1.0 + 0.5 * (rows - 1))
    report = race.run_race(
        generate,
        types.SimpleNamespace(cuda_graphs = ()),
        idle = lambda: True,
        width = 512,
        height = 512,
        sizes = (2,),
        repeats = 4,
        clock = lambda: clock["t"],
    )
    assert report["sizes"]["2"]["promoted"] is True and stored[2]["promoted"] is True


# ------------------------------------------------------------------------------------------- batch-exact (row 0 vs solo)


def test_the_eager_arms_replay_check_renders_its_solo_eager_too(monkeypatch):
    """G2a holds the batch shape fixed; the replay check is row 0 of the batch against its own solo render, and on the
    batched-eager arm that solo must run eager as well (not a graph replay), or the check compares two paths."""
    from core.inference import diffusion_cuda_graph as cg

    monkeypatch.setattr(race, "record", lambda *a, **k: True)
    seen = []
    handle = types.SimpleNamespace(set_bypass = lambda on: None, drop_batched = lambda size: 0)
    generate, clock, _ = _fake_generate(lambda rows: 1.0 if rows == 1 else 0.6 * rows)

    def recording(**kw):
        rows = len(kw.get("prompts") or [None])
        seen.append((rows, cg.eager_forced()))
        if rows > 1 and not cg.eager_forced():
            clock["t"] += 1.0  # the graphed arm loses, so the neighbour check runs on the eager arm
        return generate(**kw)

    report = race.run_race(
        recording,
        types.SimpleNamespace(cuda_graphs = (handle,)),
        idle = lambda: True,
        width = 512,
        height = 512,
        sizes = (2,),
        repeats = 3,
        clock = lambda: clock["t"],
    )
    verdict = report["sizes"]["2"]
    assert verdict["arm"] == "batched_eager" and verdict["replay_path"] == "eager"
    assert (1, True) in seen, "the eager arm's solo replay ran under forced_eager"
    assert verdict["b_exact"] is True and verdict["replay_exact"] is True


def test_b_exact_allows_only_the_same_shape_noise_floor():
    import numpy as np

    base = np.zeros((4, 4, 3), dtype = np.uint8)
    off_by_2, off_by_5 = base.copy(), base.copy()
    off_by_2[0, 0, 0], off_by_5[0, 0, 0] = 2, 5
    assert race._b_exact(base, base.copy(), base.copy()) is True  # bit-identical
    assert (
        race._b_exact(base, off_by_2, base.copy()) is False
    )  # deterministic path: any difference fails
    assert race._b_exact(base, off_by_2, off_by_5) is True  # within the A/A floor of that shape
    assert race._b_exact(base, off_by_5, off_by_2) is False  # further than the floor
    assert race._b_exact(None, base, base) is False


def test_seeded_joins_need_the_batch_exact_check():
    won = {"promoted": True, "neighbour_exact": True, "replay_exact": True}
    assert race.seeded_join_ok(won) is True  # a verdict from before the check
    assert race.seeded_join_ok(dict(won, b_exact = False)) is False
    assert race.seeded_join_ok(dict(won, replay_exact = False, b_exact = True)) is True
    assert race.seeded_join_ok(dict(won, neighbour_exact = False, b_exact = True)) is False


# ------------------------------------------------------------------------------------------- review fixes


def test_a_held_back_win_never_removes_a_size_the_default_policy_admits(monkeypatch):
    """A streamed load (policy any) or a static compile (compiled_sizes) admits sizes before any race; a win the gate
    holds back is not a loss, so it must not take its size away (the gate is never stricter than no race at all)."""
    _verdicts(monkeypatch, {2: {"promoted": True, "arm": "batched"}})
    streamed = _engine(device_name = CARD, offload_policy = "sequential")
    assert streamed.request_batching()["mode"] == "any"
    assert streamed.coalesce_batch_sizes(1024, 1024) is None
    # a real speed loss still drops its size
    _verdicts(monkeypatch, {2: {"promoted": True, "arm": "batched"}, 4: {"promoted": False}})
    sizes = streamed.coalesce_batch_sizes(1024, 1024)
    assert 4 not in sizes and 2 in sizes


def test_a_held_back_win_keeps_a_compiled_size(monkeypatch):
    from core.inference import diffusion as diffusion_module

    monkeypatch.setattr(diffusion_module, "compiled_shapes_are_static", lambda pipe, mode: True)
    _verdicts(monkeypatch, {2: {"promoted": True, "arm": "batched"}, 4: {"promoted": False}})
    engine = _engine(
        device_name = CARD,
        offload_policy = "sequential",
        speed_optims = ("compiled",),
        compile_cache_ctx = types.SimpleNamespace(shapes = [(1024, 1024, 2), (1024, 1024, 4)]),
    )
    assert engine.request_batching()["mode"] == "compiled_sizes"
    assert engine.coalesce_batch_sizes(1024, 1024) == frozenset({2})


def test_the_race_persists_the_path_it_ran_and_the_gate_reads_it(monkeypatch):
    """Without graph handles the race has only the ``batched`` arm, and that arm runs eager: the verdict says so, and
    only an eager (or any) G2c verdict may let it serve."""
    stored = {}
    monkeypatch.setattr(
        race, "record", lambda state, w, h, g, k, v, steps = None: stored.setdefault(k, v) is v
    )
    generate, clock, _ = _fake_generate(lambda rows: 1.0 + 0.5 * (rows - 1))
    report = race.run_race(
        generate,
        types.SimpleNamespace(cuda_graphs = ()),
        idle = lambda: True,
        width = 512,
        height = 512,
        sizes = (2,),
        repeats = 4,
        clock = lambda: clock["t"],
    )
    verdict = report["sizes"]["2"]
    assert (
        verdict["arm"] == "batched" and verdict["path"] == "eager" and stored[2]["path"] == "eager"
    )
    state = types.SimpleNamespace(
        family = types.SimpleNamespace(name = "flux.3-klein"), device_name = CARD
    )
    quality.record("flux.3-klein", CARD, "graph", PASS)
    assert (
        quality.gate(verdict, state)["promoted"] is False
    ), "a graph-only PASS does not vouch for eager"
    quality.record("flux.3-klein", CARD, "eager", PASS)
    assert quality.gate(verdict, state)["promoted"] is True


def test_a_graphed_race_with_handles_records_the_graph_path_and_a_fallback_records_eager(
    monkeypatch,
):
    stored = {}
    monkeypatch.setattr(
        race, "record", lambda state, w, h, g, k, v, steps = None: stored.setdefault(k, v) is v
    )
    handle = types.SimpleNamespace(set_bypass = lambda on: None, drop_batched = lambda size: 0)
    generate, clock, _ = _fake_generate(lambda rows: 1.0 + 0.5 * (rows - 1))
    report = race.run_race(
        generate,
        types.SimpleNamespace(cuda_graphs = (handle,)),
        idle = lambda: True,
        width = 512,
        height = 512,
        sizes = (2,),
        repeats = 4,
        accuracy = False,
        clock = lambda: clock["t"],
    )
    verdict = report["sizes"]["2"]
    assert verdict["path"] == ("eager" if verdict["arm"] == "batched_eager" else "graph")
    # A graphed win whose batch-size graph is gone falls back to the eager arm: its path follows.
    monkeypatch.setattr(race, "batched_graph_sizes", lambda state: set())
    graphed = {
        "promoted": True,
        "arm": "batched",
        "path": "graph",
        "arms": {"batched_eager": {"promoted": True}},
    }
    fallback = race.effective_verdict(graphed, 2, types.SimpleNamespace(cuda_graphs = (handle,)))
    assert fallback["graph_missing"] is True and fallback["path"] == "eager"
    assert quality.path_of(fallback) == "eager"
