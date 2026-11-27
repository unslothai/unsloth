# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""MiniMax-H3-Image batched forward, on CPU with the tiny random release in the native layout (fp32):

- G3: images of one packed layout run each block ONCE per step over all their rows (a forward-hook counter on the
  blocks), on a resident DiT and on the stream tier (one upload per step);
- G2a: a target row at the first, middle and last position is bit-identical when only a neighbour's prompt (same
  token count, or another count: its own batch), seed, reference image or activation magnitude changes;
- G1: one image renders through the unchanged single-image path; batched rows equal their solo renders (the patch
  projection and the output head run per row, at the batch-1 GEMM shape);
- per-row references (``row_images``) and mixed text lengths group by layout;
- the row-mod block forward at batch > 1 runs its row ops over the flattened rows, element for element."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
for p in (_BACKEND, Path(__file__).resolve().parent):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

pytestmark = pytest.mark.timeout(600)


@pytest.fixture(autouse = True)
def _env(monkeypatch):
    monkeypatch.setenv("UNSLOTH_H3_IMAGE_SOURCE_RESIZE", "0")
    monkeypatch.setenv("UNSLOTH_H3_IMAGE_BATCHED", "1")
    # no token budget: every same-layout row in one call
    monkeypatch.setenv("UNSLOTH_H3_IMAGE_BATCH_TOKENS", "1000000")
    monkeypatch.delenv("UNSLOTH_H3_IMAGE_LOCKSTEP", raising = False)


@pytest.fixture(scope = "module")
def h3i():
    pytest.importorskip("torch")
    pytest.importorskip("diffusers")
    pytest.importorskip("transformers")
    pytest.importorskip("accelerate")
    try:
        from diffusers import DiffusionPipeline  # noqa: F401
    except Exception as exc:  # noqa: BLE001 - a broken diffusers / hub pair
        pytest.skip(f"diffusers pipelines unavailable: {exc}")
    from core.inference import diffusion_minimax_h3_image

    return diffusion_minimax_h3_image


@pytest.fixture(scope = "module")
def pipe(h3i, tmp_path_factory):
    import _minimax_h3_image_tiny as tiny

    root = tmp_path_factory.mktemp("h3img_batched")
    ref = tiny.make_tiny_release(root)
    te = tiny.make_text_encoder(ref["tokenizer"])
    p = h3i.load_minimax_h3_image_pipeline(str(root), device = "cpu", text_encoder = te)
    yield p
    p.release()


@pytest.fixture
def resident(h3i, pipe):
    """The pipe as a resident DiT, with a block-call counter (G3) that records each call's batch."""
    saved = pipe._unsloth_h3_plan
    pipe._unsloth_h3_plan = h3i.H3ImagePlan(
        placement = h3i.PLACE_RESIDENT,
        dit_gb = 1.0,
        te_gb = 1.0,
        vae_gb = 1.0,
        vram_gb = 24.0,
        host_gb = None,
    )
    calls = []
    handles = [
        block.register_forward_pre_hook(lambda m, a, k = k: calls.append((k, int(a[0].shape[0]))))
        for k, block in enumerate(pipe.transformer.transformer_blocks)
    ]
    yield calls
    for h in handles:
        h.remove()
    pipe._unsloth_h3_plan = saved


def _render(pipe, prompts, seeds, **kw):
    import torch
    if isinstance(prompts, str):
        prompts = [prompts] * len(seeds)
    return pipe(
        prompt = prompts,
        height = 256,
        width = 256,
        generator = [torch.Generator().manual_seed(s) for s in seeds],
        min_generation_area = 1024,
        output_type = "latent",
        **kw,
    ).images


def _tokens(pipe, prompt):
    return len(pipe.tokenizer(prompt, add_special_tokens = False)["input_ids"])


def test_one_block_call_per_step_for_every_row_of_one_layout(h3i, pipe, resident):
    import torch

    steps, blocks = h3i.H3_IMAGE_STEPS, len(pipe.transformer.transformer_blocks)
    seeds = [11, 12, 13, 14]
    alone = [_render(pipe, "a cat", [s])[0] for s in seeds]
    resident.clear()
    together = _render(pipe, "a cat", seeds)
    assert len(resident) == steps * blocks
    assert {b for _k, b in resident} == {4}
    assert pipe._unsloth_h3_last_stats["batched"] == [4]
    assert len(together) == 4 and not torch.equal(together[0], together[1])
    for a, b in zip(alone, together):
        assert a.shape == b.shape and torch.equal(a, b)


def test_unset_or_off_keeps_one_call_per_image(h3i, pipe, resident, monkeypatch):
    import torch

    steps, blocks = h3i.H3_IMAGE_STEPS, len(pipe.transformer.transformer_blocks)
    monkeypatch.setenv(h3i.h3_batch.BATCHED_ENV, "0")
    assert pipe._together_refusal(2) == f"{h3i.h3_batch.BATCHED_ENV} is not 1"
    monkeypatch.delenv(h3i.h3_batch.BATCHED_ENV)
    assert pipe._together_refusal(2) == f"{h3i.h3_batch.BATCHED_ENV} is not 1"  # opt-in
    alone = [_render(pipe, "a cat", [s])[0] for s in (1, 2)]
    resident.clear()
    again = _render(pipe, "a cat", [1, 2])
    assert len(resident) == 2 * steps * blocks
    assert "batched" not in pipe._unsloth_h3_last_stats
    assert all(torch.equal(a, b) for a, b in zip(alone, again))


def test_one_image_renders_through_the_single_image_path(h3i, pipe, resident, monkeypatch):
    import torch

    assert pipe._together_refusal(1) == "one image"
    seen = []
    real = h3i.MiniMaxH3ImagePipeline._render_one
    monkeypatch.setattr(
        h3i.MiniMaxH3ImagePipeline,
        "_render_one",
        lambda self, *a, **k: seen.append(1) or real(self, *a, **k),
    )
    one = _render(pipe, "a cat", [5])
    assert seen == [1] and "batched" not in pipe._unsloth_h3_last_stats
    monkeypatch.setenv(h3i.h3_batch.BATCHED_ENV, "0")
    assert torch.equal(one[0], _render(pipe, "a cat", [5])[0])


@pytest.mark.parametrize("position", [0, 1, 2])
def test_a_neighbour_never_moves_the_target_row(h3i, pipe, resident, position):
    """G2a at a fixed batch of 3: the target's latents stay bit-identical whatever its neighbours' prompts (same token
    count) and seeds are; a neighbour of another token count runs as its own batch and leaves it equal too."""
    import torch

    assert _tokens(pipe, "a cat") == _tokens(pipe, "the cat") != _tokens(pipe, "a cat a")
    target_prompt, target_seed = "a cat", 7

    def batch(neighbours):
        prompts, seeds = [], []
        it = iter(neighbours)
        for k in range(3):
            p, s = (target_prompt, target_seed) if k == position else next(it)
            prompts.append(p)
            seeds.append(s)
        out = _render(pipe, prompts, seeds)
        return out[position], pipe._unsloth_h3_last_stats["batched"]

    base, shape = batch([("a cat", 1), ("a cat", 2)])
    assert shape == [3]
    rerun, _ = batch([("a cat", 1), ("a cat", 2)])
    assert torch.equal(base, rerun)  # the same-shape floor is 0 on CPU
    other_prompt, shape = batch([("the cat", 1), ("the cat", 2)])
    assert shape == [3] and torch.equal(base, other_prompt)
    other_seed, shape = batch([("a cat", 91), ("the cat", 92)])
    assert shape == [3] and torch.equal(base, other_seed)
    longer, shape = batch([("a cat a", 1), ("a cat", 2)])
    assert sorted(shape) == [1, 2]
    solo = _render(pipe, target_prompt, [target_seed])[0]
    torch.testing.assert_close(base, solo, rtol = 1e-5, atol = 1e-5)
    torch.testing.assert_close(longer, solo, rtol = 1e-5, atol = 1e-5)


def test_a_neighbour_of_huge_activations_never_moves_the_target_row(h3i, pipe):
    """G2a, activation magnitude: one batched forward, the neighbour's model input scaled by 1e4."""
    import torch

    transformer = pipe.transformer
    torch.manual_seed(0)
    jobs = []
    for seed in (1, 2, 3):
        job = pipe._render_setup(
            "a cat", [], [], 256, 256, torch.Generator().manual_seed(seed), None, 1024, 6
        )
        job["scheduler"] = pipe.scheduler
        pipe._denoise_setup(job, transformer)
        jobs.append(job)
    calls = [pipe._step_inputs(job, 0) for job in jobs]
    with torch.no_grad():
        want = h3i.h3_batch.split_rows(transformer(**h3i.h3_batch.merge_step_calls(calls))[0], 3)
        loud = [dict(c) for c in calls]
        loud[0]["hidden_states"] = calls[0]["hidden_states"] * 1e4
        loud[2]["hidden_states"] = calls[2]["hidden_states"] * 1e4
        got = h3i.h3_batch.split_rows(transformer(**h3i.h3_batch.merge_step_calls(loud))[0], 3)
    assert torch.equal(want[1], got[1])
    assert not torch.equal(want[0], got[0])


def _pictures():
    from PIL import Image

    import numpy as np

    rng = np.random.default_rng(0)
    return [Image.fromarray(rng.integers(0, 255, (256, 256, 3), dtype = np.uint8)) for _ in range(3)]


def test_per_row_references_batch_and_a_neighbours_reference_never_moves_the_target(
    h3i, pipe, resident
):
    import torch

    if pipe.processor is None:
        pytest.skip("no Qwen3-VL processor in this transformers")
    a, b, c = _pictures()
    steps, blocks = h3i.H3_IMAGE_STEPS, len(pipe.transformer.transformer_blocks)
    resident.clear()
    first = _render(pipe, ["a cat", "a cat", "a cat"], [1, 2, 3], row_images = [a, b, a])
    assert pipe._unsloth_h3_last_stats["batched"] == [3]
    assert len(resident) == steps * blocks
    swapped = _render(pipe, ["a cat", "a cat", "a cat"], [1, 2, 3], row_images = [a, c, a])
    assert torch.equal(first[0], swapped[0]) and torch.equal(first[2], swapped[2])
    assert not torch.equal(first[1], swapped[1])
    solo = _render(pipe, "a cat", [2], image = b)[0]
    torch.testing.assert_close(first[1], solo, rtol = 1e-5, atol = 1e-5)
    # a text-to-image row beside edits: its own batch
    mixed = _render(pipe, ["a cat", "a cat", "a cat"], [1, 2, 3], row_images = [a, None, a])
    assert sorted(pipe._unsloth_h3_last_stats["batched"]) == [1, 2]
    assert torch.equal(mixed[0], first[0])
    torch.testing.assert_close(mixed[1], _render(pipe, "a cat", [2])[0], rtol = 1e-5, atol = 1e-5)


def test_streamed_rows_share_one_upload_and_one_call_per_block(h3i, pipe, monkeypatch):
    import torch

    blocks = pipe.transformer.transformer_blocks
    onloads = []

    class Counting(h3i.ModuleListStreamer):
        def _onload(self, module, args):
            onloads.append(1)
            super()._onload(module, args)

    streamer = Counting(blocks, "cpu", resident = 1)
    streamer._unsloth_role = "transformer"
    streamer._unsloth_resident = 1
    rows = []
    counter = blocks[1].register_forward_pre_hook(lambda m, a: rows.append(int(a[0].shape[0])))
    saved = pipe._unsloth_h3_plan
    pipe._unsloth_h3_plan = h3i.H3ImagePlan(
        placement = h3i.PLACE_STREAM,
        dit_gb = 1.0,
        te_gb = 1.0,
        vae_gb = 1.0,
        vram_gb = 24.0,
        host_gb = None,
        dit_resident_blocks = 1,
        working_gb = h3i.stream_working_gb(h3i.LAYOUT_TODAY, 2048, 2048, 0, 24.0) + 50.0,
        slack_gb = 0.0,
        block_gb = 0.4,
        layout = h3i.LAYOUT_TODAY,
    )
    pipe._unsloth_h3_streamers.append(streamer)
    try:
        alone = [_render(pipe, "a cat", [s])[0] for s in (3, 4, 5)]
        onloads.clear()
        rows.clear()
        together = _render(pipe, "a cat", [3, 4, 5])
    finally:
        counter.remove()
        streamer.remove()
        pipe._unsloth_h3_streamers.remove(streamer)
        pipe._unsloth_h3_plan = saved
    steps = h3i.H3_IMAGE_STEPS
    assert len(onloads) == steps and rows == [3] * steps
    for a, b in zip(alone, together):
        torch.testing.assert_close(b, a, rtol = 1e-5, atol = 1e-5)


def test_the_batch_cap_is_what_the_room_leaves(h3i, pipe, resident, monkeypatch):
    import torch

    steps, blocks = h3i.H3_IMAGE_STEPS, len(pipe.transformer.transformer_blocks)
    monkeypatch.setattr(h3i, "resident_room_gb", lambda device: 0.5)
    monkeypatch.setattr(h3i.h3_batch, "batched_row_gb", lambda *a, **k: 1.0)
    out = _render(pipe, "a cat", [1, 2, 3])
    assert len(out) == 3
    # room for lockstep state only: one call per image
    assert pipe._unsloth_h3_last_stats["batched"] == [1, 1, 1]
    assert len(resident) == 3 * steps * blocks
    monkeypatch.setattr(h3i.h3_batch, "batched_row_gb", lambda *a, **k: 0.2)
    resident.clear()
    _render(pipe, "a cat", [1, 2, 3])
    assert pipe._unsloth_h3_last_stats["batched"] == [3]
    assert len(resident) == steps * blocks
    monkeypatch.setattr(h3i, "resident_room_gb", lambda device: 0.0)
    _render(pipe, "a cat", [1, 2, 3])
    assert "batched" not in pipe._unsloth_h3_last_stats
    bc = h3i.h3_batch.batch_cap
    assert bc(4, None, 0.1, 1.0) == 1
    assert bc(1, 100.0, 0.1, 1.0) == 1
    assert bc(4, 2.5, 0.1, 1.0) == 3
    assert bc(4, 100.0, 0.1, 1.0) == 4
    # rows per call stop at the token budget: 4 x 1036 (1024x1024) batch, 4108 (2048x2048) does not
    monkeypatch.delenv(h3i.h3_batch.BATCH_TOKENS_ENV)
    assert bc(4, 100.0, 0.1, 1.0, seq = 1036) == 4
    assert bc(4, 100.0, 0.1, 1.0, seq = 4108) == 1
    assert bc(4, 100.0, 0.1, 1.0, seq = 2316) == 1
    monkeypatch.setenv(h3i.h3_batch.BATCH_TOKENS_ENV, "9000")
    assert bc(4, 100.0, 0.1, 1.0, seq = 4108) == 2
    monkeypatch.setenv(h3i.h3_batch.BATCHED_ENV, "0")
    assert bc(4, 100.0, 0.1, 1.0) == 1
    del torch


def test_layout_key_for_a_coalescer(h3i, pipe):
    key = h3i.h3_batch.row_layout_key
    a = key(pipe, "a cat", width = 256, height = 256)
    assert a == key(pipe, "the cat", width = 256, height = 256)
    assert a != key(pipe, "a cat a", width = 256, height = 256)
    assert a != key(pipe, "a cat", width = 512, height = 256)
    assert a != key(pipe, "a cat", width = 256, height = 256, num_inference_steps = 4)


def test_row_mod_ops_over_a_batch_are_its_rows_flattened(h3i):
    import torch

    from core.inference import video_minimax_h3_rowmod as rowmod

    batch = h3i.h3_batch
    g = torch.Generator().manual_seed(0)
    x = torch.randn(3, 7, 16, generator = g)
    scale, shift, gate = (torch.randn(4, 16, generator = g) for _ in range(3))
    out = torch.randn(3, 7, 16, generator = g)
    idx = torch.randint(0, 4, (7,), generator = g)
    assert torch.equal(
        batch.batched_modulate(rowmod, x, scale, shift, idx),
        rowmod.reference_modulate(x, scale, shift, idx),
    )
    assert torch.equal(
        batch.batched_gated_residual(rowmod, x, gate, out, idx),
        rowmod.reference_gated_residual(x, gate, out, idx),
    )

    class Recorder:
        def __init__(self):
            self.shapes = []

        def modulate(self, x, scale, shift, idx):
            self.shapes.append((tuple(x.shape), int(idx.shape[0])))
            return rowmod.reference_modulate(x, scale, shift, idx)

    rec = Recorder()
    batch.batched_modulate(rec, x, scale, shift, idx)
    batch.batched_modulate(rec, x[:1], scale, shift, idx)
    assert rec.shapes == [((1, 21, 16), 21), ((1, 7, 16), 7)]


def test_the_batched_row_mod_block_forward_equals_the_stock_block(h3i, pipe):
    import types

    import torch

    from core.inference import video_minimax_h3_rowmod as rowmod

    block = pipe.transformer.transformer_blocks[0]
    assert rowmod.matches_stock_block(block)
    hidden = int(pipe.transformer.config.hidden_size)
    g = torch.Generator().manual_seed(1)
    dtype = block.norm1.weight.dtype if block.norm1.weight is not None else torch.float32
    x = torch.randn(2, 9, hidden, generator = g).to(dtype)
    temb = torch.randn(2, block.adaln_proj.linear.in_features, generator = g)
    idx = torch.randint(0, 6, (9,), generator = g)
    rope = pipe.transformer.rope(torch.randn(9, 3, generator = g).double())
    with torch.no_grad():
        want = block(x, temb, idx, rope)
        block.forward = types.MethodType(rowmod._row_mod_block_forward, block)
        block.__dict__[rowmod._FORWARD_MARK] = True
        try:
            assert h3i.h3_batch.install_batched_row_mod(pipe.transformer) == 1
            assert h3i.h3_batch.install_batched_row_mod(pipe.transformer) == 0
            got = block(x, temb, idx, rope)
            got1 = block(x[1:], temb, idx, rope)
        finally:
            rowmod.uninstall(pipe.transformer)
    assert "forward" not in block.__dict__
    assert torch.equal(want, got)
    assert torch.equal(want[1:], got1)
