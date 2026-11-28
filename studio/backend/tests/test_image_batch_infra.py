# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Batching infrastructure: the per-family capability, the measured race verdicts that may lift today's serial rules,
the seeded-join gate, the trace counters and the idle direct dispatch."""

from __future__ import annotations

import asyncio
import sys
import types
from pathlib import Path

import httpx
import pytest

from core.inference import diffusion_batch_adapters as adapters
from core.inference import diffusion_batch_race as race
from core.inference import diffusion_batch_trace as trace
from core.inference import image_request_batching as batching

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_image_batching_followups import (  # noqa: E402
    _alone,
    _groups,
    _interleaved,
    _plan,
    _qwen21_pipe,
)
from test_image_concurrency import (  # noqa: E402,F401  (fixture)
    _batched_calls,
    _post_while_rendering,
    studio,
)


@pytest.fixture(autouse = True)
def _fresh_trace():
    trace.reset()
    yield
    trace.reset()


@pytest.fixture(autouse = True)
def _quality_passes(monkeypatch):
    # These tests are about the speed race; the G2c quality gate has its own (test_image_batch_quality).
    from core.inference import diffusion_batch_quality as quality
    monkeypatch.setattr(quality, "quality_ok", lambda state, path: True)
    monkeypatch.setattr(quality, "any_quality_ok", lambda state: True)


# ------------------------------------------------------------------------------------------- capability


@pytest.mark.parametrize(
    "name",
    ["flux.3-klein", "flux.3-klein-base", "anima", "minimax-h3-image", "moss-image"],
)
def test_families_that_loop_one_image_per_forward_default_to_serial(name):
    from core.inference.diffusion_flux3 import FLUX3_FAMILY_NAMES

    if name.startswith("flux.3"):
        name = FLUX3_FAMILY_NAMES[0 if name == "flux.3-klein" else 1]
    family = types.SimpleNamespace(name = name)
    assert adapters.batched_forward(family) is False
    assert adapters.adapter_for(family) is None
    # The family saying so lifts the refusal (angle B's batched forward).
    assert adapters.batched_forward(types.SimpleNamespace(name = name, batched_forward = True)) is True


def test_layered_and_qwen21_default_to_serial_and_others_batch():
    assert (
        adapters.batched_forward(types.SimpleNamespace(name = "qwen-image-layered", layer_count = 2))
        is False
    )
    assert (
        adapters.batched_forward(types.SimpleNamespace(name = "qwen-image"), _qwen21_pipe()) is False
    )
    assert adapters.batched_forward(types.SimpleNamespace(name = "qwen-image"), None) is True
    assert adapters.batched_forward(types.SimpleNamespace(name = "z-image")) is True
    # A family can also opt out.
    assert (
        adapters.batched_forward(types.SimpleNamespace(name = "z-image", batched_forward = False))
        is False
    )
    assert adapters.adapter_for(types.SimpleNamespace(name = "z-image")) is adapters.DEFAULT_ADAPTER


def test_the_key_refusal_follows_the_capability():
    from core.inference.diffusion import coalesce_prompt_key

    assert (
        coalesce_prompt_key(types.SimpleNamespace(name = "minimax-h3-image"), None, "a fox") is None
    )
    assert coalesce_prompt_key(types.SimpleNamespace(name = "moss-image"), None, "a fox") is None
    assert (
        coalesce_prompt_key(types.SimpleNamespace(name = "qwen-image"), _qwen21_pipe(), "a fox")
        is None
    )
    lifted = types.SimpleNamespace(name = "minimax-h3-image", batched_forward = True)
    assert coalesce_prompt_key(lifted, None, "a fox") == ()
    q21 = types.SimpleNamespace(name = "qwen-image", batched_forward = True)
    assert coalesce_prompt_key(q21, _qwen21_pipe(), "a fox") == ()


def test_a_family_adapter_sets_its_own_layout_key():
    from core.inference.diffusion import coalesce_prompt_key

    class Lengths(adapters.BatchAdapter):
        name = "lengths"
        tokenizes = False

        def prepare(self, family, pipe, prompt):
            return ("words", len(prompt.split()))

    family = types.SimpleNamespace(name = "qwen-image", batched_forward = True, batch_adapter = Lengths())
    assert coalesce_prompt_key(family, _qwen21_pipe(), "a red fox") == ("words", 3)
    assert coalesce_prompt_key(family, _qwen21_pipe(), "a fox") != coalesce_prompt_key(
        family, _qwen21_pipe(), "a red fox"
    )


def test_in_request_chunks_follow_the_capability():
    from core.inference.diffusion import _generation_chunks
    from core.inference.diffusion_flux3 import FLUX3_FAMILY_NAMES

    jobs = [("a", 1), ("b", 2), ("c", 3)]
    flux3 = types.SimpleNamespace(name = FLUX3_FAMILY_NAMES[0])
    assert _generation_chunks(flux3, jobs, 0) == [[j] for j in jobs]
    fused = types.SimpleNamespace(name = FLUX3_FAMILY_NAMES[0], batched_forward = True)
    assert _generation_chunks(fused, jobs, 0) == [jobs]
    # MiniMax-H3 / MOSS keep one call per request (their lockstep / per-sample loops share the upload): unchanged.
    for name in ("minimax-h3-image", "moss-image"):
        assert _generation_chunks(types.SimpleNamespace(name = name), jobs, 0) == [jobs]


def test_a_family_that_says_it_fuses_batches_through_the_route(studio):
    from core.inference.diffusion_flux3 import FLUX3_FAMILY_NAMES

    app, backend, _saved = studio
    name = FLUX3_FAMILY_NAMES[0]
    backend.status = lambda: {
        "loaded": True,
        "repo_id": "x/flux3",
        "family": name,
        "base_repo": None,
    }
    import core.inference.diffusion_families as families

    real = families.detect_family
    family = types.SimpleNamespace(name = name, batched_forward = True)
    families.detect_family = lambda *a, **k: family
    try:
        _plan(backend, family, None, offload_policy = "group")
        responses = _post_while_rendering(app, backend, _interleaved("a fox", "a cat"), waiting = 2)
    finally:
        families.detect_family = real
    assert all(r.status_code == 200 for r in responses)
    assert _groups(backend) == [sorted(["a fox", "a cat"])]


def test_a_flux3_load_without_the_capability_renders_alone_and_says_why(studio):
    from core.inference.diffusion_flux3 import FLUX3_FAMILY_NAMES

    app, backend, _saved = studio
    name = FLUX3_FAMILY_NAMES[0]
    backend.status = lambda: {
        "loaded": True,
        "repo_id": "x/flux3",
        "family": name,
        "base_repo": None,
    }
    _plan(backend, types.SimpleNamespace(name = name), None, offload_policy = "group")

    async def run():
        transport = httpx.ASGITransport(app = app)
        async with httpx.AsyncClient(transport = transport, base_url = "http://studio") as client:
            backend.rendering = True
            out = []
            for path, account, body in _interleaved("a fox", "a cat"):
                # Finishes while another render holds the slot: it never waited to join a batch.
                out.append(
                    await asyncio.wait_for(
                        client.post(path, json = body, headers = {"x-test-account": account}), timeout = 5
                    )
                )
            backend.rendering = False
            return out

    responses = asyncio.run(run())
    assert all(r.status_code == 200 for r in responses)
    assert _alone(backend) == ["a fox", "a cat"]
    assert _batched_calls(backend) == []
    # The route admits FLUX.3; this load's family (no batched_forward) is refused by the engine plan.
    from core.inference.diffusion_families import detect_family

    assert adapters.batched_forward(detect_family("", name))
    assert trace.snapshot()["fallbacks"].get("engine policy or prompt layout") == 2


# ------------------------------------------------------------------------------------------- policy


def _engine(**state):
    from core.inference.diffusion import DiffusionBackend

    engine = DiffusionBackend()
    base = dict(
        family = types.SimpleNamespace(name = "flux.2-klein"),
        pipe = None,
        speed_optims = (),
        speed_mode = "off",
        cuda_graphs = (),
        compile_cache_ctx = None,
        offload_policy = "none",
    )
    base.update(state)
    engine._state = types.SimpleNamespace(**base)
    return engine


def _verdicts(monkeypatch, table):
    monkeypatch.setattr(
        race,
        "lookup",
        lambda state, w, h, g, sizes, steps = None: {k: v for k, v in table.items() if k in sizes},
    )
    monkeypatch.setattr(
        race, "any_promoted", lambda state: any(v.get("promoted") for v in table.values())
    )


def test_unmeasured_loads_keep_todays_rules(monkeypatch):
    _verdicts(monkeypatch, {})
    resident = _engine()
    assert resident.request_batching()["mode"] == "serial"
    assert resident.coalesce_batch_sizes(1024, 1024) == frozenset()
    graphs = _engine(offload_policy = "sequential", cuda_graphs = (object(),))
    assert graphs.request_batching()["mode"] == "serial"
    streamed = _engine(offload_policy = "sequential")
    assert streamed.request_batching()["mode"] == "any"
    assert streamed.coalesce_batch_sizes(1024, 1024) is None


def test_a_measured_win_lifts_the_resident_and_graph_rules_to_the_winning_sizes(monkeypatch):
    _verdicts(monkeypatch, {2: {"promoted": True}, 4: {"promoted": False}})
    for engine in (_engine(), _engine(offload_policy = "sequential", cuda_graphs = (object(),))):
        policy = engine.request_batching()
        assert policy["mode"] == "measured" and set(policy) == {"mode", "reason"}
        assert engine.coalesce_batch_sizes(1024, 1024, 4.0) == frozenset({2})
        assert engine.coalesce_plan("a fox", False) == ((), False)


def test_a_measured_loss_drops_that_size_from_a_streamed_load(monkeypatch):
    _verdicts(monkeypatch, {4: {"promoted": False}})
    engine = _engine(offload_policy = "sequential")
    assert engine.request_batching()["mode"] == "any"
    sizes = engine.coalesce_batch_sizes(1024, 1024)
    assert 4 not in sizes and {2, 3, 5} <= sizes


def test_per_call_activation_scales_and_serial_families_stay_serial_whatever_was_measured(
    monkeypatch,
):
    import core.inference.diffusion as diffusion_module

    _verdicts(monkeypatch, {2: {"promoted": True}})
    monkeypatch.setattr(
        diffusion_module, "_per_call_activation_scale", lambda state: "transformer.x"
    )
    assert _engine().request_batching()["mode"] == "serial"
    monkeypatch.setattr(diffusion_module, "_per_call_activation_scale", lambda state: None)
    h3 = _engine(family = types.SimpleNamespace(name = "minimax-h3-image"))
    assert h3.request_batching()["mode"] == "serial"
    assert h3.coalesce_plan("a fox", False) is None


def test_seeded_requests_join_only_where_the_path_is_neighbour_and_replay_exact(monkeypatch):
    engine = _engine()
    _verdicts(monkeypatch, {2: {"promoted": True, "neighbour_exact": True, "replay_exact": False}})
    assert engine.coalesce_seeded_ok(1024, 1024, 4.0, 1) is False
    _verdicts(monkeypatch, {2: {"promoted": True, "neighbour_exact": False, "replay_exact": True}})
    assert engine.coalesce_seeded_ok(1024, 1024, 4.0, 1) is False
    _verdicts(monkeypatch, {2: {"promoted": True, "neighbour_exact": True, "replay_exact": True}})
    assert engine.coalesce_seeded_ok(1024, 1024, 4.0, 1) is True
    _verdicts(monkeypatch, {})
    assert engine.coalesce_seeded_ok(1024, 1024, 4.0, 1) is False


def test_an_adapted_load_never_consumes_the_adapter_free_verdicts(monkeypatch):
    _verdicts(
        monkeypatch,
        {2: {"promoted": True, "neighbour_exact": True, "replay_exact": True}},
    )
    applied = types.SimpleNamespace(_unsloth_loras = [("style", "default", 0.8)])
    at_load = types.SimpleNamespace(_unsloth_load_lora_specs = (("style", 0.8),))
    zeroed = types.SimpleNamespace(_unsloth_loras = [("style", "default", 0.0)])
    assert _engine().coalesce_batch_sizes(1024, 1024, 4.0) == frozenset({2})
    assert _engine(pipe = zeroed).coalesce_batch_sizes(1024, 1024, 4.0) == frozenset({2})
    for pipe in (applied, at_load):
        engine = _engine(pipe = pipe)
        # Today's rule for a resident load: serial, and no seeded join on the base model's exactness.
        assert engine.coalesce_batch_sizes(1024, 1024, 4.0) == frozenset()
        assert engine.coalesce_seeded_ok(1024, 1024, 4.0, 1) is False
        streamed = _engine(pipe = pipe, offload_policy = "sequential")
        assert streamed.coalesce_batch_sizes(1024, 1024, 4.0) is None
        assert streamed.coalesce_seeded_ok(1024, 1024, 4.0, 1) is False


def test_a_seeded_request_joins_when_the_engine_says_its_path_is_exact():
    rendering = {"on": True}
    calls = []

    async def run_batch(
        prompts,
        seeds,
        owners = None,
    ):
        calls.append(list(seeds))
        return {"images": list(prompts), "seeds": list(seeds), "seed": seeds[0]}

    async def run(seeded_ok):
        rendering["on"] = True
        tasks = [
            asyncio.ensure_future(
                batching.generate(
                    "k",
                    prompt,
                    1,
                    seed,
                    busy = lambda: rendering["on"],
                    run_batch = run_batch,
                    seeded_ok = seeded_ok,
                )
            )
            for prompt, seed in (("a", 11), ("b", 22))
        ]
        for _ in range(100):
            await asyncio.sleep(0.01)
        rendering["on"] = False
        return await asyncio.gather(*tasks)

    assert asyncio.run(run(None)) == [None, None]
    assert trace.snapshot()["fallbacks"]["explicit seed"] == 2
    results = asyncio.run(run(lambda: True))
    assert calls == [[11, 22]]
    assert [r["seeds"] for r in results] == [[11], [22]]
    assert trace.snapshot()["call_sizes"] == {"2": 1}


# ------------------------------------------------------------------------------------------- verdicts


def test_promotion_needs_the_lower_bound_and_enough_repeats():
    fast = [([1.0, 1.0], 1.5), ([1.02, 0.98], 1.49), ([0.99, 1.01], 1.52), ([1.0, 1.0], 1.5)]
    verdict = race.judge(fast)
    assert verdict["promoted"] is True and verdict["speedup_lb"] >= race.PROMOTE_LB
    marginal = [([1.0, 1.0], 1.92), ([1.0, 1.0], 1.9), ([1.0, 1.0], 1.93), ([1.0, 1.0], 1.91)]
    assert race.judge(marginal)["promoted"] is False  # about 1.04x
    assert race.judge(fast[:2])["promoted"] is False  # fewer than MIN_REPEATS
    noisy = [([1.0, 1.0], 1.2), ([1.0, 1.0], 2.3), ([1.0, 1.0], 1.3), ([1.0, 1.0], 2.2)]
    assert race.judge(noisy)["promoted"] is False  # median speedup above 1.05, lower bound below


def test_burst_latencies_price_the_wait_behind_earlier_requests():
    serial, batched = race.burst_latencies([1.0, 1.0, 1.0, 1.0], 3.0)
    assert serial == [1.0, 2.0, 3.0, 4.0] and batched == [3.0] * 4
    assert race.percentile(serial, 95) == 4.0 and race.percentile(serial, 50) == 2.0


def test_verdicts_round_trip_through_the_probe_cache_losers_included(monkeypatch, tmp_path):
    import core.inference.diffusion_probe_cache as probe_cache

    monkeypatch.setattr(probe_cache, "_cache_file", lambda: tmp_path / "diffusion_quant_probe.json")
    monkeypatch.setattr(probe_cache, "fingerprint", lambda card: {"card": card, "torch": "x"})
    race._STORE.clear()
    race._UNIDENTIFIED.clear()
    state = types.SimpleNamespace(
        device = "cuda",
        gpu_ordinal = 0,
        family = types.SimpleNamespace(name = "flux.2-klein"),
        pipe = None,
        repo_id = "black-forest-labs/FLUX.2-klein-4B",
        offload_policy = "none",
        cuda_graphs = (),
        speed_optims = (),
    )
    assert race.lookup(state, 1024, 1024, 1.0, (2, 4)) == {}
    assert race.record(state, 1024, 1024, 1.0, 2, {"promoted": True, "speedup_lb": 1.2})
    assert race.record(state, 1024, 1024, 1.0, 4, {"promoted": False, "speedup_lb": 0.9})
    race._STORE.clear()  # a new process reads the file
    found = race.lookup(state, 1024, 1024, 1.0, (2, 3, 4))
    assert found == {
        2: {"promoted": True, "speedup_lb": 1.2},
        4: {"promoted": False, "speedup_lb": 0.9},
    }
    assert race.any_promoted(state) is True
    assert race.lookup(state, 768, 1024, 1.0, (2, 4)) == {}
    assert (
        race.lookup(state, 1024, 1024, 1.0, (2, 4), steps = 28) == {}
    ), "another step count is another verdict"
    assert race.record(state, 1024, 1024, 1.0, 2, {"promoted": False}, steps = 28)
    assert race.lookup(state, 1024, 1024, 1.0, (2,), steps = 28) == {2: {"promoted": False}}
    assert race.lookup(state, 1024, 1024, 1.0, (2,))[2]["promoted"] is True
    other = types.SimpleNamespace(**{**vars(state), "transformer_quant": "int8"})
    assert race.lookup(other, 1024, 1024, 1.0, (2, 4)) == {}, "another precision is another load"
    race._STORE.clear()


def test_a_card_without_a_fingerprint_is_not_reprobed_on_every_status_poll(monkeypatch, tmp_path):
    import core.inference.diffusion_probe_cache as probe_cache

    probes = []
    monkeypatch.setattr(probe_cache, "_cache_file", lambda: tmp_path / "diffusion_quant_probe.json")
    monkeypatch.setattr(probe_cache, "fingerprint", lambda card: probes.append(card))
    race._STORE.clear()
    race._UNIDENTIFIED.clear()
    state = types.SimpleNamespace(
        device = "cuda", gpu_ordinal = 0, cuda_graphs = (), offload_policy = "none"
    )
    for _ in range(5):
        assert race.lookup(state, 1024, 1024, 1.0, (2,)) == {}
        assert race.any_promoted(state) is False
    assert probes == ["cuda:0"]
    race._STORE.clear()
    race._UNIDENTIFIED.clear()


class _Image:
    def __init__(self, value):
        self.value = value

    def convert(self, _mode):
        import numpy as np
        return np.full((2, 2, 3), self.value % 251, dtype = "uint8")


def _fake_generate(cost, *, leak = False):
    """Time per call from ``cost(rows)``; each image derives from its own (prompt, seed) unless ``leak``."""
    clock = {"t": 0.0}
    calls = []

    def generate(
        prompt = None,
        prompts = None,
        seeds = None,
        seed = None,
        batch_size = 1,
        **kwargs,
    ):
        rows = list(prompts) if prompts else [prompt]
        row_seeds = list(seeds) if seeds else [seed]
        calls.append(len(rows))
        clock["t"] += cost(len(rows))
        noise = sum(len(p) for p in rows) if leak and len(rows) > 1 else 0
        return {"images": [_Image(hash((p, s)) + noise) for p, s in zip(rows, row_seeds)]}

    return generate, clock, calls


def test_the_race_warms_its_batches_inside_the_off_request_scope(monkeypatch):
    """Qwen-Image-2.1 compiles a mixed-length batch only off-request: the race's warm-ups must open that scope,
    or it splits the prompts by length and races separate forwards instead of the shared one."""
    import contextlib

    from core.inference import diffusion_qwenimage21 as q21

    monkeypatch.setattr(race, "record", lambda *a, **k: True)
    scoped: list = []
    inside = {"on": False}

    @contextlib.contextmanager
    def warming():
        inside["on"] = True
        try:
            yield
        finally:
            inside["on"] = False

    monkeypatch.setattr(q21, "warming_mixed_lengths", warming)
    generate, clock, _calls = _fake_generate(lambda rows: 1.0 + 0.5 * (rows - 1))

    def watched(*args, **kwargs):
        scoped.append((len(kwargs.get("prompts") or [None]), inside["on"]))
        return generate(*args, **kwargs)

    race.run_race(
        watched,
        types.SimpleNamespace(cuda_graphs = ()),
        idle = lambda: True,
        width = 512,
        height = 512,
        sizes = (2,),
        repeats = 1,
        clock = lambda: clock["t"],
    )
    batched = [on for rows, on in scoped if rows > 1]
    assert (
        batched and batched[0] is True
    ), "the untimed batched warm-up runs in the off-request scope"
    assert not any(on for rows, on in scoped if rows == 1), "solo renders never open it"
    assert batched[1:] and not any(batched[1:]), "timed arms run as a request would"


def test_the_race_promotes_a_real_win_and_reports_an_exact_path(monkeypatch):
    stored = {}
    monkeypatch.setattr(
        race, "record", lambda state, w, h, g, k, v, steps = None: stored.setdefault(k, v) is v
    )
    generate, clock, calls = _fake_generate(lambda rows: 1.0 + 0.5 * (rows - 1))
    report = race.run_race(
        generate,
        types.SimpleNamespace(cuda_graphs = ()),
        idle = lambda: True,
        width = 512,
        height = 512,
        sizes = (2, 4),
        repeats = 4,
        clock = lambda: clock["t"],
    )
    two, four = report["sizes"]["2"], report["sizes"]["4"]
    assert list(report["sizes"]) == ["4", "2"], "the largest batch size records first"
    assert two["promoted"] and four["promoted"]
    assert two["neighbour_exact"] is True and two["replay_exact"] is True
    assert set(stored) == {2, 4}


def test_the_race_catches_a_neighbour_leak_and_persists_a_loss(monkeypatch):
    stored = {}
    monkeypatch.setattr(
        race, "record", lambda state, w, h, g, k, v, steps = None: stored.setdefault(k, v) is v
    )
    generate, clock, _ = _fake_generate(lambda rows: float(rows), leak = True)
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
    assert verdict["promoted"] is False and verdict["neighbour_exact"] is False
    assert verdict["batched_graphs_dropped"] == 0
    assert stored[2]["promoted"] is False, "losers are persisted too"
    assert race.seeded_join_ok(verdict) is False


def test_the_race_never_runs_on_a_busy_engine_and_stops_when_a_request_waits(monkeypatch):
    stored = {}
    monkeypatch.setattr(
        race, "record", lambda state, w, h, g, k, v, steps = None: stored.setdefault(k, v) is v
    )
    generate, clock, calls = _fake_generate(lambda rows: 1.0)
    with pytest.raises(race.RaceRefused):
        race.run_race(generate, types.SimpleNamespace(), idle = lambda: False, width = 512, height = 512)
    assert calls == []
    budget = {"left": 5}

    def idle():
        budget["left"] -= 1
        return budget["left"] > 0

    report = race.run_race(
        generate,
        types.SimpleNamespace(cuda_graphs = ()),
        idle = idle,
        width = 512,
        height = 512,
        sizes = (2,),
        clock = lambda: clock["t"],
    )
    assert report["aborted"] and stored == {}


def test_the_race_skips_a_size_a_static_compile_would_have_to_compile(monkeypatch):
    monkeypatch.setattr(race, "record", lambda *a: True)
    generate, clock, calls = _fake_generate(lambda rows: 1.0)
    report = race.run_race(
        generate,
        types.SimpleNamespace(cuda_graphs = ()),
        idle = lambda: True,
        width = 512,
        height = 512,
        sizes = (3,),
        allowed_sizes = lambda k: k in (2, 4),
        clock = lambda: clock["t"],
    )
    assert "skipped" in report["sizes"]["3"] and calls == []


def test_the_engine_refuses_a_race_while_a_request_waits():
    import threading

    engine = _engine()
    engine._queued_generate_cancels.add(threading.Event())
    with pytest.raises(race.RaceRefused):
        engine.batch_race(width = 512, height = 512)


def test_the_engine_refuses_to_race_a_family_that_loops_one_image_per_forward():
    engine = _engine(family = types.SimpleNamespace(name = "moss-image"))
    with pytest.raises(race.RaceRefused):
        engine.batch_race(width = 512, height = 512)


def test_the_engine_race_records_its_batched_graphs_under_batch_rows(monkeypatch):
    from core.inference import diffusion_cuda_graph as cg

    seen = []
    engine = _engine()

    def generate(
        prompt = None,
        prompts = None,
        seeds = None,
        seed = None,
        **kwargs,
    ):
        seen.append(cg.current_batch_rows())
        rows = list(prompts) if prompts else [prompt]
        return {"images": [_Image(hash((p, s))) for p, s in zip(rows, list(seeds or [seed]))]}

    engine.generate = generate
    monkeypatch.setattr(race, "record", lambda *a: True)
    engine.batch_race(width = 512, height = 512, sizes = (2,), repeats = 3)
    assert (2, True) in seen and None in seen


# ------------------------------------------------------------------------------------------- trace


def test_idle_dispatch_reads_no_key_policy_or_status(monkeypatch):
    """G1: a free slot dispatches at once: no status read, no prompt key, no policy, no timer."""
    from routes.inference import _coalesced_image_generate

    import core.inference.diffusion_engine_router as router

    monkeypatch.setattr(router, "active_engine_name", lambda: router.ENGINE_DIFFUSERS)

    def boom(*_a, **_k):
        raise AssertionError("the idle path touched it")

    backend = types.SimpleNamespace(
        generation_holder = lambda: None,
        status = boom,
        coalesce_plan = boom,
        coalesce_batch_sizes = boom,
        request_batching = boom,
    )
    real_sleep = asyncio.sleep

    async def no_sleep(*_a, **_k):
        raise AssertionError("the idle path waited")

    monkeypatch.setattr(asyncio, "sleep", no_sleep)
    try:
        result = asyncio.run(
            _coalesced_image_generate(
                backend, None, prompt = "a fox", n = 1, seed = None, generate_kwargs = {}
            )
        )
    finally:
        monkeypatch.setattr(asyncio, "sleep", real_sleep)
    assert result is None
    counts = trace.snapshot()["counts"]
    assert counts["admitted"] == 1 and counts["idle_direct"] == 1


def test_the_trace_counts_coalesced_calls_and_the_status_carries_it(studio):
    app, backend, _saved = studio
    _plan(backend, types.SimpleNamespace(name = "z-image"), None, offload_policy = "group")
    responses = _post_while_rendering(
        app, backend, _interleaved("a fox", "a cat", "an owl"), waiting = 3
    )
    assert all(r.status_code == 200 for r in responses)
    snap = trace.snapshot()
    assert snap["call_sizes"] == {"3": 1}
    assert snap["counts"]["led"] == 1 and snap["counts"]["joined"] == 2
    assert snap["counts"]["coalesced_requests"] == 3
    from models.inference import DiffusionStatusResponse

    status = DiffusionStatusResponse(batch_trace = snap)
    assert status.model_dump()["batch_trace"]["call_sizes"] == {"3": 1}


def _toy_pipe(looped: bool):
    import torch

    class Block(torch.nn.Module):
        def forward(self, x):
            return x + 1

    class DiT(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.transformer_blocks = torch.nn.ModuleList([Block(), Block(), Block()])

        def forward(self, hidden_states):
            for block in self.transformer_blocks:
                hidden_states = block(hidden_states)
            return hidden_states

    class Pipe:
        def __init__(self):
            self.transformer = DiT()

        def __call__(
            self,
            rows: int,
            steps: int = 2,
        ):
            x = torch.zeros(rows, 4)
            for _ in range(steps):
                if looped:
                    x = torch.cat(
                        [self.transformer(hidden_states = x[i : i + 1]) for i in range(rows)]
                    )
                else:
                    x = self.transformer(hidden_states = x)
            return x

    return Pipe()


@pytest.mark.parametrize("looped", [False, True])
def test_the_trace_tells_a_fused_call_from_a_pipeline_loop(monkeypatch, looped):
    """G3: a list call that loops inside the pipeline shows invocations and block runs growing with the rows."""
    monkeypatch.setenv(trace.TRACE_ENV, "1")
    pipe = _toy_pipe(looped)
    with trace.denoiser_calls(pipe, 4):
        pipe(4)
    den = trace.snapshot()["denoiser"]
    assert den["rows"] == 8  # 4 rows x 2 steps either way
    per_step = den["invocations"] / 2
    assert per_step == (4 if looped else 1)
    assert den["block_runs"] == (24 if looped else 6)
    assert not pipe.transformer._forward_pre_hooks, "the hooks leave with the call"


def test_without_the_trace_env_no_hook_is_installed(monkeypatch):
    monkeypatch.delenv(trace.TRACE_ENV, raising = False)
    pipe = _toy_pipe(False)
    seen = []
    real = pipe.transformer.register_forward_pre_hook
    pipe.transformer.register_forward_pre_hook = lambda *a, **k: seen.append(1) or real(*a, **k)
    with trace.denoiser_calls(pipe, 4):
        pipe(4)
    assert seen == []
    assert trace.snapshot()["pipe_rows"] == {"4": 1}


def test_a_losing_size_drops_its_batched_graphs(monkeypatch):
    monkeypatch.setattr(race, "record", lambda *a: True)
    dropped = []
    handle = types.SimpleNamespace(
        drop_batched = lambda size: dropped.append(size) or 1, set_bypass = lambda on: None
    )
    generate, clock, _ = _fake_generate(lambda rows: float(rows))
    report = race.run_race(
        generate,
        types.SimpleNamespace(cuda_graphs = (handle,)),
        idle = lambda: True,
        width = 512,
        height = 512,
        sizes = (2,),
        repeats = 3,
        accuracy = False,
        clock = lambda: clock["t"],
    )
    assert report["sizes"]["2"]["promoted"] is False
    assert dropped == [2] and report["sizes"]["2"]["batched_graphs_dropped"] == 1


def test_a_resident_load_with_a_measured_win_coalesces_at_the_winning_size_only(
    studio, monkeypatch
):
    """Three waiting requests and a promoted size of 2: one call of two, the third renders alone (2 + 1)."""
    from core.inference.diffusion import DiffusionBackend

    app, backend, _saved = studio
    _verdicts(monkeypatch, {2: {"promoted": True}, 4: {"promoted": False}})
    _plan(backend, types.SimpleNamespace(name = "z-image"), None, offload_policy = "none")
    backend.coalesce_batch_sizes = types.MethodType(DiffusionBackend.coalesce_batch_sizes, backend)
    responses = _post_while_rendering(
        app, backend, _interleaved("a fox", "a cat", "an owl"), waiting = 3
    )
    assert all(r.status_code == 200 for r in responses)
    assert [len(call["prompts"]) for call in _batched_calls(backend)] == [2]
    assert len(_alone(backend)) == 1
    assert trace.snapshot()["call_sizes"] == {"2": 1}


def test_members_a_batch_left_out_coalesce_again_behind_it():
    """Admitted size 2, four requests waiting: two calls of two, nobody renders alone (the left-out pair waits for the
    first call and then shares the next one)."""
    state = {"blocker": True, "running": 0}
    calls = []

    async def run_batch(
        prompts,
        seeds,
        owners = None,
    ):
        calls.append(list(prompts))
        state["running"] += 1
        try:
            await asyncio.sleep(0.2)
        finally:
            state["running"] -= 1
        return {"images": list(prompts), "seeds": list(seeds), "seed": seeds[0]}

    def busy():
        return state["blocker"] or state["running"] > 0

    async def run():
        tasks = []
        for prompt in ("a", "b", "c", "d"):
            tasks.append(
                asyncio.ensure_future(
                    batching.generate(
                        "k",
                        prompt,
                        1,
                        None,
                        busy = busy,
                        run_batch = run_batch,
                        fits = lambda total: total == 2,
                    )
                )
            )
            await asyncio.sleep(0.02)
        await asyncio.sleep(0.1)
        state["blocker"] = False
        return await asyncio.gather(*tasks)

    results = asyncio.run(run())
    assert sorted(map(sorted, calls)) == [["a", "b"], ["c", "d"]]
    assert all(r is not None for r in results)
    assert trace.snapshot()["counts"].get("rejoined", 0) >= 2


def test_fbcache_decides_skips_from_the_whole_batch():
    """Why FBCache loads never coalesce: a row that alone would skip its blocks is computed because its neighbour
    changed, so one request decides another's image. (Studio's length guard and counter may wrap the method; both
    pass equal-shape calls through.)"""
    torch = pytest.importorskip("torch")
    fbc = pytest.importorskip("diffusers.hooks.first_block_cache")

    def decide(residual, previous):
        state = types.SimpleNamespace(head_block_residual = previous, tail_block_residuals = None)
        hook = types.SimpleNamespace(
            threshold = 0.1, state_manager = types.SimpleNamespace(get_state = lambda: state)
        )
        return bool(fbc.FBCHeadBlockHook._should_compute_remaining_blocks(hook, residual))

    prev = torch.ones(2, 4, 8)
    moved = prev.clone()
    moved[1] *= 3.0
    assert decide(prev[:1].clone(), prev[:1]) is False  # row 0 alone: skip
    assert decide(moved, prev) is True  # the same row 0 beside a changed neighbour: computed


@pytest.mark.parametrize("cache", ["fbcache", "static", None])
def test_an_fbcache_load_serves_serially_and_a_static_schedule_still_batches(cache):
    engine = _engine(offload_policy = "sequential", transformer_cache = cache, cache_auto = False)
    policy = engine.request_batching()
    if cache == "fbcache":
        assert policy["mode"] == "serial" and "FBCache" in policy["reason"]
        assert engine.coalesce_batch_sizes(1024, 1024) == frozenset()
        with pytest.raises(race.RaceRefused):
            engine.batch_race(width = 512, height = 512, steps = 8)
    else:
        assert policy["mode"] == "any"
        assert engine.coalesce_batch_sizes(1024, 1024) is None


def test_an_auto_cache_batches_only_below_the_fbcache_step_count():
    from core.inference.diffusion_cache import FBCACHE_MIN_STEPS

    engine = _engine(offload_policy = "sequential", transformer_cache = None, cache_auto = True)
    assert engine.request_batching()["mode"] == "any"
    assert engine.coalesce_batch_sizes(1024, 1024, None, FBCACHE_MIN_STEPS - 1) is None
    assert engine.coalesce_batch_sizes(1024, 1024, None, FBCACHE_MIN_STEPS) == frozenset()
    assert engine.coalesce_batch_sizes(1024, 1024) == frozenset(), "unknown steps count as engaged"


def test_the_halved_retry_runs_after_the_failed_call_released_its_frames():
    """G5: an out-of-memory batch is split into halves only once the failed call's traceback (and every tensor its
    frames hold) is gone; on a GPU the halves would otherwise start with the dead call's activations still allocated."""
    import gc
    import weakref

    class Activations:
        pass

    held: list = []
    alive_at_half: list = []

    async def run_batch(
        prompts,
        seeds,
        owners = None,
    ):
        if len(seeds) == 4:
            big = Activations()  # what the failed call's frame pins (latents, activations)
            held.append(weakref.ref(big))
            raise RuntimeError("CUDA out of memory. Tried to allocate 8.00 GiB")
        gc.collect()
        alive_at_half.append(held[0]() is not None)
        return {"images": list(prompts), "seeds": list(seeds), "seed": seeds[0]}

    results = _queue_four(run_batch)
    assert all(r is not None for r in results)
    assert alive_at_half == [
        False,
        False,
    ], "the failed call's frame was still alive during the halved retry"


def _queue_four(run_batch):
    state = {"blocker": True}

    async def run():
        tasks = []
        for prompt in ("a", "b", "c", "d"):
            tasks.append(
                asyncio.ensure_future(
                    batching.generate(
                        "k",
                        prompt,
                        1,
                        None,
                        busy = lambda: state["blocker"],
                        run_batch = run_batch,
                        splittable = lambda exc: "out of memory" in str(exc),
                    )
                )
            )
            await asyncio.sleep(0.02)
        await asyncio.sleep(0.1)
        state["blocker"] = False
        return await asyncio.gather(*tasks)

    return asyncio.run(run())


def test_the_batched_eager_arm_really_runs_eager(monkeypatch):
    """The eager arm flips every graph handle's set_bypass around its calls (and only its calls)."""
    monkeypatch.setattr(race, "record", lambda *a, **k: True)
    seen = []
    state_flag = {"bypassed": False}
    handle = types.SimpleNamespace(
        set_bypass = lambda on: state_flag.update(bypassed = bool(on)), drop_batched = lambda size: 0
    )
    generate, clock, _ = _fake_generate(lambda rows: 1.0)

    def recording(**kw):
        seen.append((len(kw.get("prompts") or [None]), state_flag["bypassed"]))
        return generate(**kw)

    race.run_race(
        recording,
        types.SimpleNamespace(cuda_graphs = (handle,)),
        idle = lambda: True,
        width = 512,
        height = 512,
        sizes = (2,),
        repeats = 3,
        accuracy = False,
        clock = lambda: clock["t"],
    )
    assert (2, True) in seen and (2, False) in seen
    assert all(
        not bypassed for rows, bypassed in seen if rows == 1
    ), "serial renders keep their graphs"
    assert state_flag["bypassed"] is False


def test_a_verdict_is_keyed_by_the_card_topology(monkeypatch):
    """A verdict on one card says nothing about a two-card split of the same load (or a CFG replica)."""
    import core.inference.diffusion_multigpu as mg

    pipe = types.SimpleNamespace()
    state = types.SimpleNamespace(pipe = pipe, offload_policy = "none", cuda_graphs = ())
    one = race.topology(state)
    monkeypatch.setattr(mg, "_PLACEMENTS", {id(pipe): None})
    split = {"primary": "cuda:0", "secondary": "cuda:1", "moved": ["text_encoder"]}

    class Box(dict):
        def get(
            self,
            key,
            default = None,
        ):
            return split if key is pipe else default

    monkeypatch.setattr(mg, "_PLACEMENTS", Box())
    two = race.topology(state)
    assert one != two and "split:cuda:0>cuda:1:text_encoder" in two
    assert race.load_key(state).endswith(two)


def test_a_graphed_verdict_admits_a_size_only_while_its_graph_exists(monkeypatch):
    """A verdict won on the graphed arm says nothing once that batch-size graph is gone (restart, LoRA reset, OOM):
    the size is then admitted only if the eager arm passed on its own."""
    graphed_win = {
        "promoted": True,
        "arm": "batched",
        "layout": race.layout_token(()),
        "graph_canvas": "c",
        "arms": {"batched": {"promoted": True}, "batched_eager": {"promoted": False}},
    }
    both_win = {
        "promoted": True,
        "arm": "batched",
        "layout": race.layout_token(()),
        "arms": {"batched": {"promoted": True}, "batched_eager": {"promoted": True}},
    }
    handle = types.SimpleNamespace(
        cache = {"k2": object()}, _batched = {"k2": 2}, _batched_canvas = {"k2": {"c"}}
    )
    with_graph = types.SimpleNamespace(cuda_graphs = (handle,))
    # step-graph handles (GraphedForward) can drop their batched graphs, so an eager fallback is enforceable
    without = types.SimpleNamespace(
        cuda_graphs = (
            types.SimpleNamespace(cache = {"k1": 1}, _batched = {}, drop_batched = lambda size: 0),
        )
    )
    assert race.effective_verdict(graphed_win, 2, with_graph, layout = ())["promoted"] is True
    assert race.effective_verdict(graphed_win, 2, without)["promoted"] is False
    assert race.effective_verdict(both_win, 2, without)["promoted"] is True
    assert not race.seeded_join_ok(
        race.effective_verdict(dict(both_win, neighbour_exact = True), 2, without)
    )
    restarted = types.SimpleNamespace(cuda_graphs = (types.SimpleNamespace(cache = {}, _batched = {}),))
    assert race.effective_verdict(graphed_win, 2, restarted)["promoted"] is False
    no_graphs = types.SimpleNamespace(cuda_graphs = ())
    assert race.effective_verdict(graphed_win, 2, no_graphs)["promoted"] is True


def test_an_eager_win_is_withheld_once_a_later_race_records_a_graph_of_that_size():
    """An eager-arm win at one canvas stands only while no batch-size graph of its size exists: a later race at another
    canvas or step count can record one with the same tensor shapes, which GraphedForward would replay for it."""
    eager_win = {"promoted": True, "arm": "batched_eager", "neighbour_exact": True, "b_exact": True}
    graphed_elsewhere = {
        "promoted": True,
        "arm": "batched",
        "layout": race.layout_token(()),
        "graph_canvas": "other",
        "arms": {"batched": {"promoted": True}, "batched_eager": {"promoted": True}},
    }
    dropped = types.SimpleNamespace(cache = {"k1": 1}, _batched = {}, drop_batched = lambda size: 0)
    assert race.effective_verdict(eager_win, 2, types.SimpleNamespace(cuda_graphs = (dropped,)))[
        "promoted"
    ]
    recorded = types.SimpleNamespace(
        cache = {"k2": object()},
        _batched = {"k2": 2},
        _batched_canvas = {"k2": {"other"}},
        drop_batched = lambda size: 0,
    )
    later = types.SimpleNamespace(cuda_graphs = (recorded,))
    for verdict in (eager_win, graphed_elsewhere):
        got = race.effective_verdict(verdict, 2, later, layout = ("bucket", 2))
        assert got["promoted"] is False and not race.seeded_join_ok(got)
    # Another size's graph replays nothing for size 2; a graphed win at its own canvas still holds.
    assert race.effective_verdict(eager_win, 4, later)["promoted"] is True
    assert race.effective_verdict(graphed_elsewhere, 2, later, layout = ())["promoted"] is True


def test_a_seeded_request_joins_only_when_every_size_it_could_render_at_is_exact(monkeypatch):
    """A streamed load (policy any) still admits the unmeasured sizes: one exact size 2 says nothing about 3-8."""
    from core.inference.image_request_batching import batch_max

    exact = {"promoted": True, "neighbour_exact": True, "replay_exact": True}
    streamed = _engine(offload_policy = "sequential")
    _verdicts(monkeypatch, {2: exact})
    assert streamed.coalesce_seeded_ok(1024, 1024, 4.0, 1) is False
    _verdicts(monkeypatch, {k: exact for k in range(2, batch_max() + 1)})
    assert streamed.coalesce_seeded_ok(1024, 1024, 4.0, 1) is True
    # A measured (resident) load admits only the winners: one exact size is enough there.
    _verdicts(monkeypatch, {2: exact, 4: {"promoted": False}})
    assert _engine().coalesce_seeded_ok(1024, 1024, 4.0, 1) is True


def test_an_eager_win_drops_the_graphs_dispatch_would_otherwise_replay(monkeypatch):
    """Dispatch never bypasses graphs: when the eager arm wins, the size's race-recorded graphs go, so requests run the
    arm that was measured."""
    monkeypatch.setattr(race, "record", lambda *a, **k: True)
    dropped = []
    flag = {"bypassed": False}
    handle = types.SimpleNamespace(
        set_bypass = lambda on: flag.update(bypassed = bool(on)),
        drop_batched = lambda size: dropped.append(size) or 1,
    )
    generate, clock, _ = _fake_generate(
        lambda rows: 1.0 if rows == 1 else (0.3 if flag["bypassed"] else 0.9) * rows
    )
    report = race.run_race(
        generate,
        types.SimpleNamespace(cuda_graphs = (handle,)),
        idle = lambda: True,
        width = 512,
        height = 512,
        sizes = (2,),
        repeats = 3,
        accuracy = False,
        clock = lambda: clock["t"],
    )
    verdict = report["sizes"]["2"]
    assert verdict["promoted"] is True and verdict["arm"] == "batched_eager"
    assert dropped == [2]


def test_a_static_compile_limits_the_race_on_a_resident_or_graphed_load(monkeypatch):
    """Resident weights return serial (measurable) before the static-compile rule: the race must still skip sizes the
    compile cache lacks."""
    import core.inference.diffusion as diffusion_module

    monkeypatch.setattr(diffusion_module, "compiled_shapes_are_static", lambda pipe, mode: True)
    seen = {}
    monkeypatch.setattr(race, "run_race", lambda *a, **k: seen.update(k) or {})
    for extra in ({}, {"offload_policy": "sequential", "cuda_graphs": (object(),)}):
        engine = _engine(
            speed_optims = ("compiled",),
            compile_cache_ctx = types.SimpleNamespace(shapes = [(512, 512, 2)]),
            **extra,
        )
        seen.clear()
        engine.batch_race(width = 512, height = 512)
        allowed = seen["allowed_sizes"]
        assert allowed is not None and allowed(2) and not allowed(4)


def test_the_eager_arm_holds_a_forced_eager_scope_through_generate(monkeypatch):
    """The per-chunk set_bypass(False) in generate() cannot undo it: the arm runs under cuda_graph.forced_eager()."""
    from core.inference import diffusion_cuda_graph as cg

    monkeypatch.setattr(race, "record", lambda *a, **k: True)
    seen = []
    handle = types.SimpleNamespace(set_bypass = lambda on: None, drop_batched = lambda size: 0)
    generate, clock, _ = _fake_generate(lambda rows: 1.0)

    def recording(**kw):
        seen.append((len(kw.get("prompts") or [None]), cg.eager_forced()))
        return generate(**kw)

    race.run_race(
        recording,
        types.SimpleNamespace(cuda_graphs = (handle,)),
        idle = lambda: True,
        width = 512,
        height = 512,
        sizes = (2,),
        repeats = 3,
        accuracy = False,
        clock = lambda: clock["t"],
    )
    assert (2, True) in seen and (2, False) in seen and (1, True) not in seen
    assert cg.eager_forced() is False


def test_a_graphed_verdict_holds_only_for_the_prompt_layout_it_was_raced_at(monkeypatch):
    """A FLUX.1 prompt in another T5 bucket runs the same canvas through other shapes: no graph of the race's layout
    serves it, so only the eager arm's verdict applies there (and when the layout is unknown)."""
    graphed_win = {
        "promoted": True,
        "arm": "batched",
        "layout": race.layout_token(("t5", 256)),
        "graph_canvas": "c",
        "arms": {"batched": {"promoted": True}, "batched_eager": {"promoted": False}},
    }
    handle = types.SimpleNamespace(
        cache = {"k2": object()}, _batched = {"k2": 2}, _batched_canvas = {"k2": {"c"}}
    )
    state = types.SimpleNamespace(cuda_graphs = (handle,))
    assert race.effective_verdict(graphed_win, 2, state, layout = ("t5", 256))["promoted"] is True
    assert race.effective_verdict(graphed_win, 2, state, layout = ("t5", 512))["promoted"] is False
    assert race.effective_verdict(graphed_win, 2, state)["promoted"] is False
    # The engine threads the request's layout through to the verdict.
    _verdicts(monkeypatch, {2: graphed_win})
    engine = _engine(cuda_graphs = (handle,))
    assert engine.coalesce_batch_sizes(1024, 1024, 4.0, layout = ("t5", 256)) == frozenset({2})
    assert engine.coalesce_batch_sizes(1024, 1024, 4.0, layout = ("t5", 512)) == frozenset()


def test_the_race_records_the_layout_of_its_prompts(monkeypatch):
    monkeypatch.setattr(race, "record", lambda *a, **k: True)
    generate, clock, _ = _fake_generate(lambda rows: 1.0 if rows == 1 else 0.4 * rows)
    report = race.run_race(
        generate,
        types.SimpleNamespace(cuda_graphs = ()),
        idle = lambda: True,
        width = 512,
        height = 512,
        sizes = (2,),
        repeats = 3,
        accuracy = False,
        layout = ("t5", 256),
        clock = lambda: clock["t"],
    )
    assert report["sizes"]["2"]["layout"] == race.layout_token(("t5", 256))


def test_an_out_of_memory_size_is_a_persisted_loss_and_the_race_goes_on(monkeypatch):
    stored = {}
    monkeypatch.setattr(
        race,
        "record",
        lambda state, w, h, g, size, verdict, steps = None: stored.update({size: verdict}) or True,
    )
    dropped = []
    handle = types.SimpleNamespace(
        set_bypass = lambda on: None, drop_batched = lambda size: dropped.append(size) or 1
    )
    generate, clock, _ = _fake_generate(lambda rows: 1.0 if rows == 1 else 0.4 * rows)

    def oom_at_four(**kw):
        if len(kw.get("prompts") or [None]) == 4:
            raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
        return generate(**kw)

    report = race.run_race(
        oom_at_four,
        types.SimpleNamespace(cuda_graphs = (handle,)),
        idle = lambda: True,
        width = 512,
        height = 512,
        sizes = (2, 4),
        repeats = 3,
        accuracy = False,
        clock = lambda: clock["t"],
    )
    four = report["sizes"]["4"]
    assert four["promoted"] is False and "out of memory" in four["failed"]
    assert stored[4]["promoted"] is False and 4 in dropped
    assert "2" in report["sizes"] and stored[2]["promoted"] is True


def test_a_failure_that_is_not_capacity_still_raises(monkeypatch):
    monkeypatch.setattr(race, "record", lambda *a, **k: True)

    def broken(**kw):
        raise KeyError("bug")

    with pytest.raises(KeyError):
        race.run_race(
            broken,
            types.SimpleNamespace(cuda_graphs = ()),
            idle = lambda: True,
            width = 512,
            height = 512,
            sizes = (2,),
            repeats = 3,
            accuracy = False,
        )


def test_an_aborted_size_drops_the_batched_graphs_its_warm_up_recorded(monkeypatch):
    """A request arriving after the batched warm-up (calls [1, 4]) aborts the size with no verdict: an otherwise serial
    load can never use the graph that warm-up recorded, so it goes before the aborted report returns."""
    monkeypatch.setattr(race, "record", lambda *a, **k: True)
    dropped = []
    handle = types.SimpleNamespace(
        set_bypass = lambda on: None, drop_batched = lambda size: dropped.append(size) or 1
    )
    generate, clock, calls = _fake_generate(lambda rows: 1.0)
    report = race.run_race(
        generate,
        types.SimpleNamespace(cuda_graphs = (handle,)),
        idle = lambda: len(calls) < 2,
        width = 512,
        height = 512,
        sizes = (4,),
        repeats = 3,
        accuracy = False,
        clock = lambda: clock["t"],
    )
    assert calls == [1, 4] and report["aborted"] is True
    assert dropped == [4] and report["sizes"]["4"]["batched_graphs_dropped"] == 1


def test_block_graph_handles_never_count_a_graphed_winner_as_missing():
    """BlockGraphSet (offloaded / layer-split loads) records each layout on demand, user calls included, and keeps no
    per-batch-size labels: only handles that refuse to record on a user's request (GraphedForward) decide."""
    import inspect

    from core.inference.diffusion_block_graph import BlockGraphSet

    assert "_batched" not in inspect.getsource(BlockGraphSet)
    graphed_win = {
        "promoted": True,
        "arm": "batched",
        "layout": race.layout_token(()),
        "arms": {"batched": {"promoted": True}, "batched_eager": {"promoted": False}},
    }
    blocks = types.SimpleNamespace(cache = {(0, "k4"): object()})
    assert (
        race.effective_verdict(
            graphed_win, 4, types.SimpleNamespace(cuda_graphs = (blocks,)), layout = ()
        )["promoted"]
        is True
    )
    # Beside a step-graph handle, that handle still decides.
    step = types.SimpleNamespace(cache = {"k1": object()}, _batched = {})
    mixed = types.SimpleNamespace(cuda_graphs = (blocks, step))
    assert race.effective_verdict(graphed_win, 4, mixed, layout = ())["promoted"] is False


def test_a_live_graph_at_one_canvas_never_admits_a_graphed_verdict_raced_at_another():
    """After a restart, racing 512x512 records a live batch-2 graph; a persisted batch-2 verdict for 1024x1024 that won
    only on the graphed arm must not ride it (graphs key on tensor shapes, so the 1024 batch runs eager): it falls back
    to its own eager arm, which lost, and seeded joins stay off."""
    layout = ("t5", 256)
    small = race.canvas_token(512, 512, None, None, layout)
    large = race.canvas_token(1024, 1024, None, None, layout)
    assert small != large
    graphed_win_large = {
        "promoted": True,
        "arm": "batched",
        "layout": race.layout_token(layout),
        "graph_canvas": large,
        "neighbour_exact": True,
        "replay_exact": True,
        "arms": {"batched": {"promoted": True}, "batched_eager": {"promoted": False}},
    }
    live_small = types.SimpleNamespace(
        cache = {"k512": object()}, _batched = {"k512": 2}, _batched_canvas = {"k512": {small}}
    )
    state = types.SimpleNamespace(cuda_graphs = (live_small,))
    applied = race.effective_verdict(graphed_win_large, 2, state, layout = layout)
    assert applied["promoted"] is False and applied["graph_missing"] is True
    assert not race.seeded_join_ok(applied)
    # The same verdict holds once a live batch-2 graph of its own canvas exists.
    live_small._batched_canvas["k512"].add(large)
    assert race.effective_verdict(graphed_win_large, 2, state, layout = layout)["promoted"] is True
    # A verdict persisted before canvases were tracked proves no graph shape: only its eager arm counts.
    legacy = {k: v for k, v in graphed_win_large.items() if k != "graph_canvas"}
    assert race.effective_verdict(legacy, 2, state, layout = layout)["promoted"] is False


def test_the_race_labels_its_batched_calls_and_verdicts_with_its_canvas(monkeypatch):
    from core.inference import diffusion_cuda_graph as cg

    monkeypatch.setattr(race, "record", lambda *a, **k: True)
    generate, clock, _ = _fake_generate(lambda rows: 1.0 if rows == 1 else 0.4 * rows)
    seen = []

    def traced(**kwargs):
        seen.append(cg._BATCH_CANVAS.get())
        return generate(**kwargs)

    report = race.run_race(
        traced,
        types.SimpleNamespace(cuda_graphs = ()),
        idle = lambda: True,
        width = 1024,
        height = 768,
        guidance = 3.5,
        steps = 4,
        sizes = (2,),
        repeats = 3,
        accuracy = False,
        layout = ("t5", 256),
        clock = lambda: clock["t"],
    )
    want = race.canvas_token(1024, 768, 3.5, 4, ("t5", 256))
    assert report["sizes"]["2"]["graph_canvas"] == want
    assert want in seen and None in seen  # solo renders carry no canvas label
    assert cg._BATCH_CANVAS.get() is None


def test_hidream_requests_batch_across_users():
    # G2a on L4 (int8, streamed, compiled): a target row is bit-identical whatever its neighbour is; the batched-vs-solo
    # move equals a one-rounding nudge of a solo run, so it is batch-size numerics, not cross-row coupling.
    from core.inference import diffusion_families as families
    for name in ("hidream-i1-full", "hidream-i1-dev", "hidream-i1-fast"):
        assert adapters.batched_forward(families.detect_family("", name)) is True
