# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A measured batching win serves only where an offline G2c verdict passes for the family, card and execution path;
a missing or failing verdict keeps today's serial rules."""

from __future__ import annotations

import asyncio
import sys
import types
from pathlib import Path

import httpx
import pytest

from core.inference import diffusion_batch_quality as quality
from core.inference import diffusion_batch_race as race

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_image_batch_infra import _engine, _fake_generate, _verdicts  # noqa: E402
from test_image_concurrency import studio  # noqa: E402,F401  (fixture)

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


# ------------------------------------------------------------------------------------------- arm choice, race sizes


def test_a_qualifying_arm_wins_over_a_faster_bound_that_fails_p95(monkeypatch):
    """Serial 5 s per image at batch 2; the graphed call is 4 s but one timed repeat takes 11 s (higher bound, p95 over
    the slack), the eager call is a steady 8 s (bound 1.25, p95 within). The eager arm passes both and must win."""
    from core.inference import diffusion_cuda_graph as cg

    stored, dropped = {}, []
    monkeypatch.setattr(
        race, "record", lambda state, w, h, g, k, v, steps = None: stored.setdefault(k, v) is v
    )
    handle = types.SimpleNamespace(
        set_bypass = lambda on: None, drop_batched = lambda size: dropped.append(size) or 0
    )
    clock = {"t": 0.0}
    graphed = {"n": 0}

    def generate(
        prompt = None,
        prompts = None,
        seeds = None,
        seed = None,
        batch_size = 1,
        **kw,
    ):
        rows = list(prompts) if prompts else [prompt]
        row_seeds = list(seeds) if seeds else [seed]
        if len(rows) == 1:
            clock["t"] += 5.0
        elif cg.eager_forced():
            clock["t"] += 8.0
        else:
            graphed["n"] += 1  # the first graphed call is the untimed capture warm-up
            clock["t"] += 11.0 if graphed["n"] == 5 else 4.0
        return {"images": [types.SimpleNamespace(p = p, s = s) for p, s in zip(rows, row_seeds)]}

    report = race.run_race(
        generate,
        types.SimpleNamespace(cuda_graphs = (handle,)),
        idle = lambda: True,
        width = 512,
        height = 512,
        sizes = (2,),
        repeats = 5,
        accuracy = False,
        clock = lambda: clock["t"],
    )
    verdict = report["sizes"]["2"]
    graphed_arm = verdict["arms"]["batched"]
    assert (
        graphed_arm["promoted"] is False
        and graphed_arm["speedup_lb"] > verdict["arms"]["batched_eager"]["speedup_lb"]
    )
    assert (
        verdict["promoted"] is True
        and verdict["arm"] == "batched_eager"
        and verdict["path"] == "eager"
    )
    assert stored[2]["promoted"] is True


def test_no_qualifying_arm_records_the_best_bound_as_the_loss():
    lost = race._pick_arm(
        {
            "batched": {"promoted": False, "speedup_lb": 1.2},
            "batched_eager": {"promoted": False, "speedup_lb": 0.9},
        }
    )
    assert lost == "batched"
    both = race._pick_arm(
        {
            "batched": {"promoted": True, "speedup_lb": 1.2},
            "batched_eager": {"promoted": True, "speedup_lb": 1.4},
        }
    )
    assert both == "batched_eager"


@pytest.mark.parametrize("sizes", [(16,), (2, 9), (1,), (0,)])
def test_the_race_refuses_a_size_it_has_no_prompts_for_before_any_render(sizes):
    calls = []
    with pytest.raises(ValueError, match = "batch sizes"):
        race.run_race(
            lambda **kw: calls.append(kw) or {"images": []},
            types.SimpleNamespace(cuda_graphs = ()),
            idle = lambda: True,
            width = 512,
            height = 512,
            sizes = sizes,
        )
    assert calls == [], "nothing renders before the sizes are checked"
    assert race.MAX_RACE_SIZE == min(len(race.RACE_PROMPTS), len(race.NEIGHBOUR_PROMPTS) + 1)


def test_the_race_endpoint_answers_422_for_an_unsupported_size(monkeypatch):
    import asyncio

    from fastapi import HTTPException

    import routes.inference as inference

    monkeypatch.setattr(inference.account_access, "managed_account", lambda: False)
    with pytest.raises(HTTPException) as caught:
        asyncio.run(
            inference.images_batching_race(
                inference.ImageBatchRaceRequest(sizes = [16]), current_subject = "x"
            )
        )
    assert caught.value.status_code == 422
    assert "2" in caught.value.detail and str(race.MAX_RACE_SIZE) in caught.value.detail


# ------------------------------------------------------------------------------------------- adapters


@pytest.mark.parametrize(
    "pipe",
    [
        types.SimpleNamespace(_unsloth_loras = (("style", "/tmp/style.safetensors", 0.8),)),
        types.SimpleNamespace(
            _unsloth_loras_baked = True, _unsloth_baked_lora_specs = (("style", 0.8),)
        ),
        types.SimpleNamespace(_unsloth_load_lora_specs = (("style", 0.8),)),
    ],
)
def test_the_race_refuses_a_load_with_active_adapters_before_any_render(monkeypatch, pipe):
    """The race renders without ``loras``, which generate() reads as none: it would unload or zero the user's adapters
    (or, baked under streamed graphs, refuse with a reload error mid-race). Verdicts are not keyed by adapter set, so
    the race refuses an adapted load up front."""
    engine = _engine(pipe = pipe, offload_policy = "sequential")
    rendered = []
    monkeypatch.setattr(engine, "generate", lambda **kw: rendered.append(kw) or {"images": []})
    with pytest.raises(race.RaceRefused, match = "adapter"):
        engine.batch_race(width = 512, height = 512)
    assert rendered == []


def test_an_adapter_free_load_still_races(monkeypatch):
    engine = _engine(
        pipe = types.SimpleNamespace(_unsloth_loras = (("off", "/tmp/off.safetensors", 0.0),)),
        offload_policy = "sequential",
    )
    called = []
    monkeypatch.setattr(race, "run_race", lambda *a, **k: called.append(k) or {"sizes": {}})
    engine.batch_race(width = 512, height = 512)
    assert called, "a zero-weight adapter is not an active selection"


# ------------------------------------------------------------------------------------------- deferred speed profile


def test_the_engine_refuses_to_race_while_the_speed_profile_is_still_deferred(monkeypatch):
    """A fresh Auto load engages its speed profile on its third render: inside a race that would install graphs after
    the race snapshotted its handles (graph timings saved as path eager, no eager arm, batch graphs left behind)."""
    engine = _engine(offload_policy = "sequential", speed_deferred = True)
    rendered = []
    monkeypatch.setattr(engine, "generate", lambda **kw: rendered.append(kw) or {"images": []})
    with pytest.raises(race.RaceRefused, match = "deferred"):
        engine.batch_race(width = 512, height = 512)
    assert rendered == []


def test_a_race_whose_execution_configuration_changes_stops_and_persists_nothing_more(monkeypatch):
    stored = {}
    monkeypatch.setattr(
        race, "record", lambda state, w, h, g, k, v, steps = None: stored.setdefault(k, v) is v
    )
    state = types.SimpleNamespace(cuda_graphs = (), speed_optims = ())
    generate, clock, calls = _fake_generate(lambda rows: 1.0 + 0.5 * (rows - 1))

    def engaging(**kw):
        if len(calls) == 2:  # the third render engages a profile: graphs appear under the race
            state.cuda_graphs = (
                types.SimpleNamespace(set_bypass = lambda on: None, drop_batched = lambda k: 0),
            )
            state.speed_optims = ("compiled", "cuda_graph")
        return generate(**kw)

    with pytest.raises(race.RaceRefused, match = "changed"):
        race.run_race(
            engaging,
            state,
            idle = lambda: True,
            width = 512,
            height = 512,
            sizes = (4, 2),
            repeats = 3,
            clock = lambda: clock["t"],
        )
    assert stored == {}, "no verdict is persisted for a measurement that straddled the change"


# ------------------------------------------------------------------------------------------- route capability check


def test_a_measured_moss_layer_major_load_coalesces_through_the_route(studio, monkeypatch):
    """MOSS fuses rows only on a layer-major load with a measured win (diffusion_moss_image.coalesce_key(pipe)). The
    route's family check has no pipe of its own: it must not refuse MOSS before the engine, which has the pipe, is
    asked (#19957)."""
    from test_image_batch_infra import _post_while_rendering
    from test_image_batching_followups import _groups, _interleaved, _plan

    from core.inference.diffusion_families import MOSS_IMAGE_FAMILY_NAME, detect_family

    fake = types.ModuleType("core.inference.diffusion_moss_image")
    fake.coalesce_key = lambda pipe: ("moss-image layer-major",)
    monkeypatch.setitem(sys.modules, "core.inference.diffusion_moss_image", fake)
    app, backend, _saved = studio
    backend.status = lambda: {
        "loaded": True,
        "repo_id": "x/moss",
        "family": MOSS_IMAGE_FAMILY_NAME,
        "base_repo": None,
    }
    _plan(
        backend,
        detect_family("", MOSS_IMAGE_FAMILY_NAME),
        types.SimpleNamespace(transformer = object()),
        offload_policy = "group",
    )
    responses = _post_while_rendering(app, backend, _interleaved("a fox", "a cat"), waiting = 2)
    assert all(r.status_code == 200 for r in responses)
    assert _groups(backend) == [sorted(["a fox", "a cat"])]


def test_a_moss_load_without_a_measured_win_still_renders_alone_through_the_route(
    studio, monkeypatch
):
    from test_image_batch_infra import _alone, _batched_calls
    from test_image_batching_followups import _interleaved, _plan

    from core.inference.diffusion_families import MOSS_IMAGE_FAMILY_NAME, detect_family

    fake = types.ModuleType("core.inference.diffusion_moss_image")
    fake.coalesce_key = lambda pipe: None
    monkeypatch.setitem(sys.modules, "core.inference.diffusion_moss_image", fake)
    app, backend, _saved = studio
    backend.status = lambda: {
        "loaded": True,
        "repo_id": "x/moss",
        "family": MOSS_IMAGE_FAMILY_NAME,
        "base_repo": None,
    }
    _plan(
        backend,
        detect_family("", MOSS_IMAGE_FAMILY_NAME),
        types.SimpleNamespace(transformer = object()),
        offload_policy = "group",
    )

    async def run():
        transport = httpx.ASGITransport(app = app)
        async with httpx.AsyncClient(transport = transport, base_url = "http://studio") as client:
            backend.rendering = True
            out = [
                await asyncio.wait_for(
                    client.post(path, json = body, headers = {"x-test-account": account}), 5
                )
                for path, account, body in _interleaved("a fox", "a cat")
            ]
            backend.rendering = False
            return out

    responses = asyncio.run(run())
    assert all(r.status_code == 200 for r in responses)
    assert _batched_calls(backend) == [] and sorted(_alone(backend)) == ["a cat", "a fox"]


# ------------------------------------------------------------------------------------------- eager wins need an enforceable path


def _eager_wins(monkeypatch, handle):
    from core.inference import diffusion_cuda_graph as cg

    stored = {}
    monkeypatch.setattr(
        race, "record", lambda state, w, h, g, k, v, steps = None: stored.setdefault(k, v) is v
    )
    clock = {"t": 0.0}

    def generate(
        prompt = None,
        prompts = None,
        seeds = None,
        seed = None,
        batch_size = 1,
        **kw,
    ):
        rows = list(prompts) if prompts else [prompt]
        row_seeds = list(seeds) if seeds else [seed]
        clock["t"] += (
            5.0 if len(rows) == 1 else (6.0 if cg.eager_forced() else 11.0)
        )  # the graphed arm loses
        return {"images": [types.SimpleNamespace(p = p, s = s) for p, s in zip(rows, row_seeds)]}

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
    return report["sizes"]["2"], stored


def test_an_eager_win_on_handles_that_cannot_drop_batched_graphs_is_not_promoted(monkeypatch):
    """Block-graph handles record on demand and have no drop_batched: later coalesced calls would replay graphs, so
    the eager verdict would not describe what serves. Withhold it."""
    blocks = types.SimpleNamespace(set_bypass = lambda on: None)  # no drop_batched
    verdict, stored = _eager_wins(monkeypatch, blocks)
    eager_arm = verdict["arms"]["batched_eager"]
    assert eager_arm["speedup_lb"] >= race.PROMOTE_LB, "the eager arm was fast enough"
    assert (
        eager_arm["promoted"] is False and "cannot keep batched calls eager" in eager_arm["reason"]
    )
    assert verdict["promoted"] is False
    assert stored[2]["promoted"] is False
    persisted_win = {
        "promoted": True,
        "arm": "batched_eager",
        "path": "eager",
        "neighbour_exact": True,
    }
    applied = race.effective_verdict(persisted_win, 2, types.SimpleNamespace(cuda_graphs = (blocks,)))
    assert applied["promoted"] is False


def test_an_eager_win_on_handles_that_drop_their_batched_graphs_is_promoted(monkeypatch):
    graphs = types.SimpleNamespace(set_bypass = lambda on: None, drop_batched = lambda size: 1)
    verdict, stored = _eager_wins(monkeypatch, graphs)
    assert verdict["promoted"] is True and verdict["arm"] == "batched_eager"
    applied = race.effective_verdict(dict(verdict), 2, types.SimpleNamespace(cuda_graphs = (graphs,)))
    assert applied["promoted"] is True
