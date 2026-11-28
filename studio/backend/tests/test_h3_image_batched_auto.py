# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""MiniMax-H3-Image's batched forward is an Auto decision, on CPU with the tiny random release (fp32):

- Auto runs a k-image call as one batched forward only where a race verdict for this load, shape and k serves
  (speed promoted, rows engaged, every row exact); a missing, losing, inexact or other-size verdict keeps today's
  render; ``UNSLOTH_H3_IMAGE_BATCHED`` 1 / 0 overrides every verdict;
- the race times one k-image call with the batched forward off against on, promotes by the shared rule and checks
  every row bit for bit against the off arm and the first / last rows against their solo renders; a size whose
  batched arm cannot run all rows as one call is skipped, not judged;
- the engine's race endpoint races this family (it refused it before) instead of the request-sharing race."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
for p in (_BACKEND, Path(__file__).resolve().parent):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

pytestmark = pytest.mark.timeout(900)


@pytest.fixture(autouse = True)
def _env(monkeypatch):
    monkeypatch.setenv("UNSLOTH_H3_IMAGE_SOURCE_RESIZE", "0")
    monkeypatch.delenv("UNSLOTH_H3_IMAGE_BATCHED", raising = False)
    monkeypatch.delenv("UNSLOTH_H3_IMAGE_BATCH_TOKENS", raising = False)
    monkeypatch.delenv("UNSLOTH_H3_IMAGE_LOCKSTEP", raising = False)


@pytest.fixture(scope = "module")
def h3i():
    pytest.importorskip("torch")
    pytest.importorskip("diffusers")
    pytest.importorskip("transformers")
    pytest.importorskip("accelerate")
    try:
        from diffusers import DiffusionPipeline  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"diffusers pipelines unavailable: {exc}")
    from core.inference import diffusion_minimax_h3_image

    return diffusion_minimax_h3_image


@pytest.fixture(scope = "module")
def h3r():
    from core.inference import diffusion_minimax_h3_image_batch_race
    return diffusion_minimax_h3_image_batch_race


@pytest.fixture(scope = "module")
def pipe(h3i, tmp_path_factory):
    import _minimax_h3_image_tiny as tiny

    root = tmp_path_factory.mktemp("h3img_auto")
    ref = tiny.make_tiny_release(root)
    te = tiny.make_text_encoder(ref["tokenizer"])
    p = h3i.load_minimax_h3_image_pipeline(str(root), device = "cpu", text_encoder = te)
    yield p
    p.release()


@pytest.fixture
def resident(h3i, pipe):
    saved = pipe._unsloth_h3_plan
    pipe._unsloth_h3_plan = h3i.H3ImagePlan(
        placement = h3i.PLACE_RESIDENT,
        dit_gb = 1.0,
        te_gb = 1.0,
        vae_gb = 1.0,
        vram_gb = 24.0,
        host_gb = None,
    )
    yield pipe
    pipe._unsloth_h3_plan = saved


@pytest.fixture
def store(h3r, pipe, monkeypatch):
    """An in-memory verdict store for a fake loaded state holding ``pipe`` (CPU has no card identity)."""
    table: dict = {}
    state = types.SimpleNamespace(pipe = pipe)
    monkeypatch.setattr(h3r, "state_of", lambda p: state if p is pipe else None)
    monkeypatch.setattr(
        h3r, "lookup", lambda st, w, h, s, k: table.get((int(w), int(h), int(s), int(k)))
    )

    def record(st, w, h, s, k, verdict):
        table[(int(w), int(h), int(s), int(k))] = dict(verdict)
        return True

    monkeypatch.setattr(h3r, "record", record)
    table["state"] = state
    return table


SIZE = 256
CANVAS = [256, 256]  # the canvas the pipe reports at 256x256 (min_generation_area 1024)


def _render(
    pipe,
    seeds,
    prompt = "a cat",
):
    import torch
    return pipe(
        prompt = [prompt] * len(seeds),
        height = SIZE,
        width = SIZE,
        generator = [torch.Generator().manual_seed(s) for s in seeds],
        min_generation_area = 1024,
        output_type = "latent",
    ).images


def _serving(k, **over):
    v = {
        "promoted": True,
        "engaged": True,
        "rows_exact": True,
        "solo_exact": True,
        "size": k,
        "canvas": CANVAS,
    }
    v.update(over)
    return v


def _steps(h3i):
    return int(h3i.H3_IMAGE_STEPS)


def _equal_solo(pipe, seeds, together):
    import torch
    alone = [_render(pipe, [s])[0] for s in seeds]
    return all(a.shape == b.shape and torch.equal(a, b) for a, b in zip(alone, together))


def test_auto_without_a_verdict_renders_one_image_after_another(h3i, resident, store):
    out = _render(resident, [1, 2])
    assert "batched" not in resident._unsloth_h3_last_stats
    assert _equal_solo(resident, [1, 2], out)


def test_auto_with_a_serving_verdict_batches_and_every_row_equals_its_solo(h3i, resident, store):
    store[(SIZE, SIZE, _steps(h3i), 3)] = _serving(3)
    out = _render(resident, [4, 5, 6])
    assert resident._unsloth_h3_last_stats["batched"] == [3]
    assert _equal_solo(resident, [4, 5, 6], out)


@pytest.mark.parametrize(
    "verdict",
    [
        _serving(4),  # another k: 3 images were never raced
        _serving(3, promoted = False),  # a measured loss
        _serving(3, rows_exact = False),
        _serving(3, solo_exact = False),
        _serving(3, engaged = False),
        _serving(3, canvas = [512, 512]),  # raced at another canvas
    ],
)
def test_auto_keeps_one_after_another_without_a_serving_verdict_for_this_call(
    h3i, resident, store, verdict
):
    store[(SIZE, SIZE, _steps(h3i), int(verdict["size"]))] = verdict
    _render(resident, [4, 5, 6])
    # no row shares a block call (a canvas mismatch is only seen after set-up: batch 1 each, in lockstep)
    assert max(resident._unsloth_h3_last_stats.get("batched") or [1]) == 1


def test_auto_never_serves_smaller_layout_groups_than_the_raced_call(h3i, resident, store):
    import torch

    store[(SIZE, SIZE, _steps(h3i), 4)] = _serving(4)
    prompts = ["a cat", "a red fox in deep fresh snow at dawn"] * 2
    resident(
        prompt = prompts,
        height = SIZE,
        width = SIZE,
        generator = [torch.Generator().manual_seed(s) for s in (1, 2, 3, 4)],
        min_generation_area = 1024,
        output_type = "latent",
    )
    # two layouts would split into two batches of 2, a size the verdict for 4 never measured
    assert max(resident._unsloth_h3_last_stats.get("batched") or [1]) == 1


def test_the_env_overrides_every_verdict(h3i, resident, store, monkeypatch):
    store[(SIZE, SIZE, _steps(h3i), 2)] = _serving(2)
    monkeypatch.setenv("UNSLOTH_H3_IMAGE_BATCHED", "0")
    _render(resident, [1, 2])
    assert "batched" not in resident._unsloth_h3_last_stats
    store.pop((SIZE, SIZE, _steps(h3i), 2))
    monkeypatch.setenv("UNSLOTH_H3_IMAGE_BATCHED", "1")
    out = _render(resident, [1, 2])
    assert resident._unsloth_h3_last_stats["batched"] == [2]
    assert _equal_solo(resident, [1, 2], out)


def test_one_image_never_asks_for_a_verdict(h3i, resident, store, h3r, monkeypatch):
    asked = []
    monkeypatch.setattr(h3r, "lookup", lambda *a: asked.append(a))
    _render(resident, [7])
    assert asked == []


def test_the_auto_cap_is_the_measured_size_or_one(h3i, monkeypatch):
    b = h3i.h3_batch
    assert b.batched_mode() == b.MODE_AUTO
    assert b.batch_cap(4, 100.0, 0.1, 1.0) == 1
    assert b.batch_cap(4, 100.0, 0.1, 1.0, measured = 2) == 1
    assert b.batch_cap(4, 100.0, 0.1, 1.0, measured = 4) == 4
    # the measurement stands in for the default token budget (2048x2048 is 4108 rows per image) ...
    assert b.batch_cap(4, 100.0, 0.1, 1.0, seq = 4108, measured = 4) == 4
    # ... but not for the room: a raced batch that does not fit now runs batch 1, never a shape nobody raced
    assert b.batch_cap(4, 2.5, 0.1, 1.0, measured = 4) == 1
    monkeypatch.setenv(b.BATCH_TOKENS_ENV, "9000")
    assert b.batch_cap(4, 100.0, 0.1, 1.0, seq = 4108, measured = 4) == 1
    monkeypatch.delenv(b.BATCH_TOKENS_ENV)
    # forced on keeps today's budget; a race arm measures the real batch
    monkeypatch.setenv(b.BATCHED_ENV, "1")
    assert b.batch_cap(4, 100.0, 0.1, 1.0, seq = 4108) == 1
    monkeypatch.delenv(b.BATCHED_ENV)
    with b.forced_arm(b.MODE_ON):
        assert b.batched_mode() == b.MODE_ON
        assert b.batch_cap(4, 100.0, 0.1, 1.0, seq = 4108) == 4
    with b.forced_arm(b.MODE_OFF):
        assert b.batch_cap(4, 100.0, 0.1, 1.0, measured = 4) == 1
    assert b.batched_mode() == b.MODE_AUTO


def _generate(pipe, calls):
    import torch
    def generate(*, prompt, seeds, batch_size, width, height, steps, guidance, live_preview):
        # generate()'s default guidance (0) is refused by this guidance-distilled family
        assert batch_size == len(seeds) and live_preview is False and guidance == 1.0
        calls.append(len(seeds))
        out = pipe(
            prompt = [prompt] * len(seeds),
            height = height,
            width = width,
            num_inference_steps = steps,
            generator = [torch.Generator().manual_seed(s) for s in seeds],
            min_generation_area = 1024,
        ).images
        return {"images": out}

    return generate


def _clock(
    h3i,
    on_s,
    off_s = 1.0,
):
    t = [0.0]

    def clock():
        t[0] += on_s if h3i.h3_batch.batched_mode() == h3i.h3_batch.MODE_ON else off_s
        return t[0]

    return clock


def test_the_race_promotes_a_faster_exact_batch_and_auto_then_serves_it(h3i, h3r, resident, store):
    calls: list = []
    state = store["state"]
    report = h3r.run_race(
        _generate(resident, calls),
        state,
        idle = lambda: True,
        width = SIZE,
        height = SIZE,
        steps = _steps(h3i),
        sizes = (2,),
        repeats = 3,
        clock = _clock(h3i, 0.5),
    )
    v = report["sizes"]["2"]
    assert v["promoted"] and v["engaged"] and v["rows_exact"] and v["solo_exact"] and v["serves"]
    assert v["speedup_lb"] >= 1.05 and v["canvas"] == CANVAS and v["persisted"]
    # warm-ups 2, 3 repeats x 2 arms, then the first and last rows solo
    assert calls == [2, 2, 2, 2, 2, 2, 2, 2, 1, 1]
    assert h3i.h3_batch.batched_mode() == h3i.h3_batch.MODE_AUTO
    out = _render(resident, [8, 9])
    assert resident._unsloth_h3_last_stats["batched"] == [2]
    assert _equal_solo(resident, [8, 9], out)


def test_the_race_persists_a_loss_and_auto_keeps_one_after_another(h3i, h3r, resident, store):
    report = h3r.run_race(
        _generate(resident, []),
        store["state"],
        idle = lambda: True,
        width = SIZE,
        height = SIZE,
        steps = _steps(h3i),
        sizes = (2,),
        repeats = 3,
        clock = _clock(h3i, 1.2),
    )
    v = report["sizes"]["2"]
    assert not v["promoted"] and not v["serves"] and v["rows_exact"]
    assert "below" in v["reason"]
    assert store[(SIZE, SIZE, _steps(h3i), 2)]["promoted"] is False
    _render(resident, [8, 9])
    assert "batched" not in resident._unsloth_h3_last_stats


def test_a_win_without_the_exactness_check_or_with_an_inexact_row_does_not_serve(
    h3i, h3r, resident, store, monkeypatch
):
    kw = dict(idle = lambda: True, width = SIZE, height = SIZE, steps = _steps(h3i), sizes = (2,), repeats = 3)
    v = h3r.run_race(
        _generate(resident, []), store["state"], accuracy = False, clock = _clock(h3i, 0.5), **kw
    )["sizes"]["2"]
    assert not v["promoted"] and not v["serves"] and "accuracy off" in v["reason"]
    monkeypatch.setattr(h3r, "_all_same", lambda a, b: False)
    v = h3r.run_race(_generate(resident, []), store["state"], clock = _clock(h3i, 0.5), **kw)[
        "sizes"
    ]["2"]
    assert not v["promoted"] and not v["serves"] and "bit-identical" in v["reason"]


def test_a_size_whose_batched_arm_cannot_share_one_call_is_skipped(
    h3i, h3r, resident, store, monkeypatch
):
    monkeypatch.setattr(type(resident), "_compiled_blocks", lambda self: True)
    report = h3r.run_race(
        _generate(resident, []),
        store["state"],
        idle = lambda: True,
        width = SIZE,
        height = SIZE,
        steps = _steps(h3i),
        sizes = (2,),
        repeats = 3,
        clock = _clock(h3i, 0.5),
    )
    v = report["sizes"]["2"]
    assert v["engaged"] is False and "skipped" in v and not v["promoted"]
    assert (SIZE, SIZE, _steps(h3i), 2) not in store


def test_the_race_stops_when_a_request_waits(h3i, h3r, resident, store):
    from core.inference.diffusion_batch_race import RaceRefused

    with pytest.raises(RaceRefused):
        h3r.run_race(
            _generate(resident, []),
            store["state"],
            idle = lambda: False,
            width = SIZE,
            height = SIZE,
            steps = 6,
        )
    seen = iter([True, True, False])
    report = h3r.run_race(
        _generate(resident, []),
        store["state"],
        idle = lambda: next(seen, False),
        width = SIZE,
        height = SIZE,
        steps = _steps(h3i),
        sizes = (2,),
    )
    assert report["aborted"] and "aborted" in report["sizes"]["2"]
    assert h3i.h3_batch.batched_mode() == h3i.h3_batch.MODE_AUTO


def test_the_engine_races_this_family_instead_of_refusing_it(h3r, monkeypatch):
    from core.inference.diffusion import DiffusionBackend

    engine = DiffusionBackend()
    engine._state = types.SimpleNamespace(
        family = types.SimpleNamespace(name = "minimax-h3-image"),
        pipe = None,
        speed_optims = (),
        speed_mode = "off",
        cuda_graphs = (),
        compile_cache_ctx = None,
        offload_policy = "none",
    )
    seen = {}

    def fake(generate, state, **kw):
        seen.update(kw, state = state)
        return {"kind": "h3_image_batched_forward", "sizes": {}}

    monkeypatch.setattr(h3r, "run_race", fake)
    report = engine.batch_race(width = 1024, height = 1024, steps = None, sizes = (2, 4))
    assert report["kind"] == "h3_image_batched_forward"
    assert seen["state"] is engine._state and seen["width"] == 1024 and seen["sizes"] == (2, 4)
    assert seen["steps"] == 6 and seen["guidance"] is None
    # the backend registers itself, so a pipe call finds the loaded state that holds it
    engine._state.pipe = object()
    assert h3r.state_of(engine._state.pipe) is engine._state


def test_a_race_arm_follows_the_render_onto_the_render_thread(h3i):
    """Studio runs CUDA renders on a persistent render thread that copies the caller's context
    (diffusion_render_thread.run); a thread-local arm never reached the pipe there."""
    import contextvars
    import threading
    from concurrent.futures import ThreadPoolExecutor

    b = h3i.h3_batch
    with ThreadPoolExecutor(max_workers = 1) as pool:
        with b.forced_arm(b.MODE_ON):
            ctx = contextvars.copy_context()
            seen = pool.submit(ctx.run, b.batched_mode).result()
            # a request rendering beside the race (no copied context) keeps Auto
            other = []
            t = threading.Thread(target = lambda: other.append(b.batched_mode()))
            t.start()
            t.join()
        after = pool.submit(contextvars.copy_context().run, b.batched_mode).result()
    assert seen == b.MODE_ON and other == [b.MODE_AUTO] and after == b.MODE_AUTO
