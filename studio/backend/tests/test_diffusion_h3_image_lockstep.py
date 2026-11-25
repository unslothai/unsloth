# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""MiniMax-H3-Image stream tier, several images in one request: the images denoise together, each block running
every image as its own batch-1 call before the next block, so a streamed block is uploaded once per step for all of
them. On CPU with the tiny random release and a counting pageable streamer: the latents equal separate renders bit
for bit, the uploads halve, a plan without room (or the kill switch, or an out-of-memory) renders them one after
another, and a cancel stops both images and leaves the block hooks as they were."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
for p in (_BACKEND, Path(__file__).resolve().parent):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

pytestmark = pytest.mark.timeout(300)


@pytest.fixture(autouse = True)
def _picture_one_at_its_own_size(monkeypatch):
    monkeypatch.setenv("UNSLOTH_H3_IMAGE_SOURCE_RESIZE", "0")


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

    root = tmp_path_factory.mktemp("h3img_lockstep")
    ref = tiny.make_tiny_release(root)
    te = tiny.make_text_encoder(ref["tokenizer"])
    p = h3i.load_minimax_h3_image_pipeline(str(root), device = "cpu", text_encoder = te)
    yield p
    p.release()


def _counting_streamer(h3i):
    class Counting(h3i.ModuleListStreamer):
        """Counts uploads and knows which streamed blocks are on the device now."""

        def __init__(self, *a, **k):
            self.onloads = 0
            self.on: set = set()
            super().__init__(*a, **k)

        def _onload(self, module, args):
            self.onloads += 1
            self.on.add(id(module))
            super()._onload(module, args)

        def _offload(self, module, args, output):
            self.on.discard(id(module))
            super()._offload(module, args, output)

    return Counting


@pytest.fixture
def streamed(h3i, pipe):
    """The pipe as the stream tier: block 0 resident, block 1 streamed, a measured plan priced at 2048x2048."""
    blocks = pipe.transformer.transformer_blocks
    streamer = _counting_streamer(h3i)(blocks, "cpu", resident = 1)
    streamer._unsloth_role = "transformer"
    streamer._unsloth_resident = 1
    calls = []

    def on_device(module, _args):
        # registered after the streamer's onload hook: every call of a streamed block runs on its upload
        calls.append(id(module) in streamer.on)

    checks = [blocks[1].register_forward_pre_hook(on_device)]
    saved_plan = pipe._unsloth_h3_plan
    pipe._unsloth_h3_plan = h3i.H3ImagePlan(
        placement = h3i.PLACE_STREAM,
        dit_gb = 1.0,
        te_gb = 1.0,
        vae_gb = 1.0,
        vram_gb = 24.0,
        host_gb = None,
        dit_resident_blocks = 1,
        working_gb = h3i.stream_working_gb(h3i.LAYOUT_TODAY, 2048, 2048, 0, 24.0),
        slack_gb = 0.0,
        block_gb = 0.4,
        layout = h3i.LAYOUT_TODAY,
    )
    pipe._unsloth_h3_streamers.append(streamer)
    yield streamer, calls
    for handle in checks:
        handle.remove()
    streamer.remove()
    pipe._unsloth_h3_streamers.remove(streamer)
    pipe._unsloth_h3_plan = saved_plan


def _render(
    pipe,
    seeds,
    prompt = "a cat",
    **kw,
):
    import torch
    return pipe(
        prompt = prompt,
        height = 256,
        width = 256,
        num_images_per_prompt = 1 if isinstance(prompt, list) else len(seeds),
        generator = [torch.Generator().manual_seed(s) for s in seeds],
        min_generation_area = 1024,
        output_type = "latent",
        **kw,
    ).images


def _hooks_untouched(blocks):
    return all(
        not hasattr(hook, "_unsloth_gated")
        for block in blocks
        for table in (block._forward_pre_hooks, block._forward_hooks)
        for hook in table.values()
    )


def test_two_images_together_equal_separate_renders_on_half_the_uploads(
    h3i, pipe, streamed, monkeypatch
):
    import torch

    streamer, calls = streamed
    steps = h3i.H3_IMAGE_STEPS
    monkeypatch.setenv(h3i.LOCKSTEP_ENV, "0")
    alone = _render(pipe, [11, 12])
    assert streamer.onloads == 2 * steps
    assert "lockstep" not in pipe._unsloth_h3_last_stats
    monkeypatch.delenv(h3i.LOCKSTEP_ENV)
    streamer.onloads = 0
    calls.clear()
    together = _render(pipe, [11, 12])
    assert len(together) == len(alone) == 2
    for a, b in zip(alone, together):
        assert a.dtype == b.dtype and torch.equal(a, b)
    assert not torch.equal(together[0], together[1])
    # one upload per step serves both images, and both ran on it
    assert streamer.onloads == steps
    assert calls == [True] * (2 * steps)
    assert pipe._unsloth_h3_last_stats["lockstep"] == 2
    assert not streamer.on and _hooks_untouched(pipe.transformer.transformer_blocks)
    assert h3i.LOCKSTEP_GATE_ATTR not in pipe.transformer.__dict__


def test_per_image_prompts_and_one_shared_generator_keep_their_own_noise(
    h3i, pipe, streamed, monkeypatch
):
    import torch

    streamer, _calls = streamed

    def run(prompt, n):
        return pipe(
            prompt = prompt,
            height = 256,
            width = 256,
            num_images_per_prompt = n,
            generator = torch.Generator().manual_seed(5),
            min_generation_area = 1024,
            output_type = "latent",
        ).images

    for prompt, n in ((["a cat", "the cat a"], 1), ("a cat", 3)):
        monkeypatch.setenv(h3i.LOCKSTEP_ENV, "0")
        alone = run(prompt, n)
        monkeypatch.delenv(h3i.LOCKSTEP_ENV)
        streamer.onloads = 0
        together = run(prompt, n)
        assert len(together) == len(alone)
        assert all(torch.equal(a, b) for a, b in zip(alone, together))
        assert streamer.onloads == h3i.H3_IMAGE_STEPS
        assert pipe._unsloth_h3_last_stats["lockstep"] == len(alone)


def test_one_image_and_resident_placements_render_as_before(h3i, pipe, streamed):
    streamer, _calls = streamed
    _render(pipe, [3])
    assert streamer.onloads == h3i.H3_IMAGE_STEPS
    assert "lockstep" not in pipe._unsloth_h3_last_stats
    assert pipe._lockstep_refusal(1) == "one image"
    plan = pipe._unsloth_h3_plan
    plan.placement = h3i.PLACE_RESIDENT
    try:
        assert pipe._lockstep_refusal(2) == "the DiT is not streamed"
    finally:
        plan.placement = h3i.PLACE_STREAM


def test_a_plan_without_room_for_the_second_image_renders_them_one_after_another(
    h3i, pipe, streamed, monkeypatch
):
    import torch

    streamer, _calls = streamed
    monkeypatch.setenv(h3i.LOCKSTEP_ENV, "0")
    alone = _render(pipe, [21, 22])
    monkeypatch.delenv(h3i.LOCKSTEP_ENV)
    room = h3i.lockstep_room_gb(pipe._unsloth_h3_plan, 256, 256)
    # a second image's state larger than the room the plan leaves
    monkeypatch.setattr(h3i, "lockstep_state_gb", lambda *a, **k: room + 0.01)
    streamer.onloads = 0
    again = _render(pipe, [21, 22])
    assert streamer.onloads == 2 * h3i.H3_IMAGE_STEPS
    assert "lockstep" not in pipe._unsloth_h3_last_stats
    assert all(torch.equal(a, b) for a, b in zip(alone, again))


def test_the_room_is_what_the_plan_priced_past_one_image(h3i):
    def plan(**kw):
        base = dict(
            placement = h3i.PLACE_STREAM,
            dit_gb = 21.9,
            te_gb = 27.2,
            vae_gb = 2.34,
            vram_gb = 22.0,
            host_gb = None,
            dit_resident_blocks = 31,
            working_gb = h3i.stream_working_gb(h3i.LAYOUT_TODAY, 2048, 2048, 0, 22.0),
            slack_gb = 0.2,
            block_gb = 0.4,
            layout = h3i.LAYOUT_TODAY,
        )
        base.update(kw)
        return h3i.H3ImagePlan(**base)

    # at the 4 MP floor only the slack is left; 1024x1024 also leaves the floor's per-MP share
    assert h3i.lockstep_room_gb(plan(), 2048, 2048) == pytest.approx(0.2)
    credit = h3i.STREAM_WORKING_GB_PER_MP * (4096 * 4096 // 4 - 1024 * 1024) / 1e6
    assert h3i.lockstep_room_gb(plan(), 1024, 1024) == pytest.approx(0.2 + credit)
    assert h3i.lockstep_room_gb(plan(working_gb = None), 1024, 1024) is None
    assert h3i.lockstep_room_gb(plan(placement = h3i.PLACE_RESIDENT), 1024, 1024) is None
    assert h3i.lockstep_room_gb(None, 1024, 1024) is None
    # a 2048x2048 image's state (4096 image rows + 300 text rows at hidden 5376) is far under 0.2 GB
    each = h3i.lockstep_state_gb(4396, 300, 4096, 96, 5376, 96)
    assert 0.05 < each < 0.2


def test_room_for_two_of_three_pairs_two_and_renders_the_third_alone(
    h3i, pipe, streamed, monkeypatch
):
    import torch

    streamer, _calls = streamed
    monkeypatch.setenv(h3i.LOCKSTEP_ENV, "0")
    alone = _render(pipe, [1, 2, 3])
    monkeypatch.delenv(h3i.LOCKSTEP_ENV)
    seen = []
    real = h3i.MiniMaxH3ImagePipeline._lockstep_fits
    monkeypatch.setattr(
        h3i.MiniMaxH3ImagePipeline,
        "_lockstep_fits",
        lambda self, jobs, room: seen.append(len(jobs))
        or (len(jobs) <= 2 and real(self, jobs, room)),
    )
    streamer.onloads = 0
    together = _render(pipe, [1, 2, 3])
    assert seen == [2, 3]
    assert streamer.onloads == 2 * h3i.H3_IMAGE_STEPS
    assert all(torch.equal(a, b) for a, b in zip(alone, together))


def test_out_of_memory_together_falls_back_to_separate_renders(h3i, pipe, streamed, monkeypatch):
    import torch

    streamer, _calls = streamed
    monkeypatch.setenv(h3i.LOCKSTEP_ENV, "0")
    alone = _render(pipe, [31, 32])
    monkeypatch.delenv(h3i.LOCKSTEP_ENV)
    real = h3i.lockstep_blocks

    def oom(transformer, states):
        real(transformer, states[:1])
        raise torch.OutOfMemoryError("CUDA out of memory (test)")

    monkeypatch.setattr(h3i, "lockstep_blocks", oom)
    streamer.onloads = 0
    again = _render(pipe, [31, 32])
    assert all(torch.equal(a, b) for a, b in zip(alone, again))
    # the failed first step uploaded once, then each image streamed every step itself
    assert streamer.onloads == 1 + 2 * h3i.H3_IMAGE_STEPS
    assert _hooks_untouched(pipe.transformer.transformer_blocks)
    assert h3i.LOCKSTEP_GATE_ATTR not in pipe.transformer.__dict__


def test_out_of_memory_in_group_setup_falls_back_to_separate_renders(
    h3i, pipe, streamed, monkeypatch
):
    import torch

    streamer, _calls = streamed
    monkeypatch.setenv(h3i.LOCKSTEP_ENV, "0")
    alone = _render(pipe, [41, 42])
    monkeypatch.delenv(h3i.LOCKSTEP_ENV)
    real = h3i.MiniMaxH3ImagePipeline._render_setup
    seen = []

    def setup(self, *args, **kwargs):
        seen.append(1)
        # the second image's set-up inside the group runs out of memory; separate renders set up again
        if len(seen) == 2:
            raise torch.OutOfMemoryError("CUDA out of memory (test setup)")
        return real(self, *args, **kwargs)

    monkeypatch.setattr(h3i.MiniMaxH3ImagePipeline, "_render_setup", setup)
    streamer.onloads = 0
    again = _render(pipe, [41, 42])
    assert all(torch.equal(a, b) for a, b in zip(alone, again))
    assert streamer.onloads == 2 * h3i.H3_IMAGE_STEPS
    assert "lockstep" not in pipe._unsloth_h3_last_stats


def test_group_setup_records_no_autograd_graph(h3i, pipe, streamed, monkeypatch):
    import torch

    real = h3i.MiniMaxH3ImagePipeline._render_setup
    grad_on = []

    def setup(self, *args, **kwargs):
        grad_on.append(torch.is_grad_enabled())
        return real(self, *args, **kwargs)

    monkeypatch.setattr(h3i.MiniMaxH3ImagePipeline, "_render_setup", setup)
    with torch.enable_grad():
        _render(pipe, [51, 52])
    assert pipe._unsloth_h3_last_stats["lockstep"] == 2
    assert grad_on and not any(grad_on)


def test_the_room_is_priced_by_reference_images_not_their_patch_rows(h3i, pipe, monkeypatch):
    seen = []
    monkeypatch.setattr(
        h3i, "lockstep_room_gb", lambda plan, w, h, references = 0: seen.append(references) or 1.0
    )
    job = {"stats": {"canvas": (256, 256)}, "n_ref": 4096, "ref_images": 1}
    assert pipe._lockstep_room(job) == 1.0
    assert seen == [1]


def test_cancel_stops_both_images_and_the_callback_runs_once_per_step(h3i, pipe, streamed):
    streamer, _calls = streamed
    steps_seen = []

    def on_step(p, i, t, kwargs):
        steps_seen.append((i, tuple(kwargs["latents"].shape)))
        return kwargs

    import os

    os.environ[h3i.LOCKSTEP_ENV] = "0"
    try:
        _render(pipe, [41], callback_on_step_end = on_step)
    finally:
        del os.environ[h3i.LOCKSTEP_ENV]
    shape = steps_seen[0][1]
    steps_seen.clear()
    _render(pipe, [41, 42], callback_on_step_end = on_step)
    # once per step (for both images), with the first image's latents in the shape a single render passes
    assert [i for i, _ in steps_seen] == list(range(h3i.H3_IMAGE_STEPS))
    assert all(s == shape for _, s in steps_seen)

    def cancel(p, i, t, kwargs):
        if i == 1:
            p._interrupt = True
        return kwargs

    streamer.onloads = 0
    out = _render(pipe, [41, 42], callback_on_step_end = cancel)
    assert out == []
    assert streamer.onloads == 2  # steps 0 and 1 ran, for both images at once
    assert not streamer.on and _hooks_untouched(pipe.transformer.transformer_blocks)
    assert h3i.LOCKSTEP_GATE_ATTR not in pipe.transformer.__dict__


def test_the_gate_runs_only_the_streamer_hooks_once_per_block(h3i):
    import torch

    class Block(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = torch.nn.Linear(4, 4)

        def forward(self, x, temb, adaln, rope):
            return self.lin(x)

    class Owner(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.transformer_blocks = torch.nn.ModuleList([Block(), Block()])

    owner = Owner()
    log = []
    handles, others = [], []
    for k, block in enumerate(owner.transformer_blocks):
        handles.append(block.register_forward_pre_hook(lambda m, a, k = k: log.append(f"on{k}")))
        others.append(block.register_forward_pre_hook(lambda m, a, k = k: log.append(f"run{k}")))
        handles.append(block.register_forward_hook(lambda m, a, o, k = k: log.append(f"off{k}")))
    states = [[torch.randn(1, 4), None, None, None] for _ in range(3)]
    want = []
    for state in states:
        x = state[0]
        for block in owner.transformer_blocks:
            x = block.lin(x)
        want.append(x)
    gate = h3i.LockstepGate(owner.transformer_blocks, handles)
    assert gate.gated == 4
    owner.__dict__[h3i.LOCKSTEP_GATE_ATTR] = gate
    h3i.lockstep_blocks(owner, states)
    assert log == ["on0", "run0", "run0", "run0", "off0", "on1", "run1", "run1", "run1", "off1"]
    assert all(torch.equal(s[0], w) for s, w in zip(states, want))
    assert gate.first and gate.last
    gate.remove()
    owner.__dict__.pop(h3i.LOCKSTEP_GATE_ATTR)
    assert _hooks_untouched(owner.transformer_blocks)
    log.clear()
    owner.transformer_blocks[0](torch.randn(1, 4), None, None, None)
    assert log == ["on0", "run0", "off0"]
    # no gate: every hook runs on every call
    log.clear()
    h3i.lockstep_blocks(owner, states[:2])
    assert log.count("on0") == 2 and log.count("off1") == 2


def _cuda_stack():
    import torch

    class Block(torch.nn.Module):
        def __init__(self, width):
            super().__init__()
            self.a = torch.nn.Linear(width, 4 * width)
            self.b = torch.nn.Linear(4 * width, width)

        def forward(self, x, temb, adaln, rope):
            return x + self.b(torch.nn.functional.gelu(self.a(x * temb)))

    class Stack(torch.nn.Module):
        def __init__(
            self,
            width = 256,
            depth = 8,
        ):
            super().__init__()
            torch.manual_seed(0)
            self.transformer_blocks = torch.nn.ModuleList([Block(width) for _ in range(depth)])

        def forward(
            self,
            states,
            together = False,
        ):
            from core.inference import diffusion_minimax_h3_image as h3i

            states = [list(s) for s in states]
            if together:
                h3i.lockstep_blocks(self, states)
            else:
                for s in states[:1]:
                    for block in self.transformer_blocks:
                        s[0] = block(s[0], s[1], s[2], s[3])
            return [s[0] for s in states]

    return Stack


@pytest.mark.skipif(not __import__("torch").cuda.is_available(), reason = "needs CUDA")
def test_the_pinned_prefetch_streamer_serves_every_image_from_one_upload(h3i):
    import torch

    import core.inference.diffusion_offload_prefetch as op

    Stack = _cuda_stack()
    ref = Stack().cuda().eval().to(torch.bfloat16)
    stack = Stack().eval().to(torch.bfloat16)
    streamer = op.stream_module_list(stack, stack.transformer_blocks, "cuda", 5, spread = True)
    assert streamer is not None and streamer.streamed == 3
    images = [
        [
            torch.randn(1, 64, 256, device = "cuda", dtype = torch.bfloat16),
            torch.rand(1, 1, 256, device = "cuda", dtype = torch.bfloat16),
            None,
            None,
        ]
        for _ in range(2)
    ]
    with torch.no_grad():
        want = [ref([s])[0] for s in images]
        for _ in range(2):  # the first forward records the order
            alone = [stack([s])[0] for s in images]
        before = dict(streamer.stats())
        alone = [stack([s])[0] for s in images]
        mid = dict(streamer.stats())
        gate = h3i.LockstepGate(stack.transformer_blocks, streamer.handles)
        stack.__dict__[h3i.LOCKSTEP_GATE_ATTR] = gate
        try:
            together = stack(images, together = True)
            together = stack(images, together = True)
        finally:
            stack.__dict__.pop(h3i.LOCKSTEP_GATE_ATTR)
            gate.remove()
        after = dict(streamer.stats())
    torch.cuda.synchronize()
    assert all(torch.equal(a, w) for a, w in zip(alone, want))
    assert all(torch.equal(t, w) for t, w in zip(together, want))
    # separate: each image's forward copies the 3 streamed blocks; together: one forward copies them for both
    assert mid["copies"] - before["copies"] == 2 * 3
    assert after["copies"] - mid["copies"] == 2 * 3  # two lockstep forwards
    assert after["forwards"] - mid["forwards"] == 2
    assert after.get("missed", 0) == mid.get("missed", 0)
    assert after["prefetched"] > mid["prefetched"]
    assert streamer.prefetcher.inflight_bytes == 0 and not streamer.prefetcher.ready
    for i in streamer.streamed_indices:
        assert all(p.device.type == "cpu" for p in stack.transformer_blocks[i].parameters())
    streamer.remove()
