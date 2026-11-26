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
    assert trace.snapshot()["fallbacks"].get("family renders one image per forward") == 2


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
        race, "lookup", lambda state, w, h, g, sizes: {k: v for k, v in table.items() if k in sizes}
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


def test_the_race_promotes_a_real_win_and_reports_an_exact_path(monkeypatch):
    stored = {}
    monkeypatch.setattr(race, "record", lambda state, w, h, g, k, v: stored.setdefault(k, v) is v)
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
    assert two["promoted"] and four["promoted"]
    assert two["neighbour_exact"] is True and two["replay_exact"] is True
    assert set(stored) == {2, 4}


def test_the_race_catches_a_neighbour_leak_and_persists_a_loss(monkeypatch):
    stored = {}
    monkeypatch.setattr(race, "record", lambda state, w, h, g, k, v: stored.setdefault(k, v) is v)
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
    assert stored[2]["promoted"] is False, "losers are persisted too"
    assert race.seeded_join_ok(verdict) is False


def test_the_race_never_runs_on_a_busy_engine_and_stops_when_a_request_waits(monkeypatch):
    stored = {}
    monkeypatch.setattr(race, "record", lambda state, w, h, g, k, v: stored.setdefault(k, v) is v)
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
