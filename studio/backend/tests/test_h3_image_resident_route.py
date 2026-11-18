# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""MiniMax-H3-Image on CPU: every placement the planner returns for the int8 DiT gets an int8 GEMM lever (a stream
plan that keeps all 50 blocks used to get none), the stream tier is priced from the measured working set, and a
render needing more working set than planned gives resident DiT blocks back to the stream (also after an OOM)."""

from __future__ import annotations

import copy
import itertools
import sys
import types
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
for p in (_BACKEND, Path(__file__).resolve().parent):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

INT8 = dict(dit_gb = 21.9, te_gb = 27.2, inflight = 2)
LINKS = [(None, None), (3.0, 270.0), (11.5, 270.0), (24.0, 272.0)]


@pytest.fixture(scope = "module")
def h3i():
    pytest.importorskip("torch")
    pytest.importorskip("diffusers")
    pytest.importorskip("transformers")
    from core.inference import diffusion_minimax_h3_image

    return diffusion_minimax_h3_image


@pytest.fixture(scope = "module")
def speed():
    from core.inference import diffusion_minimax_h3_image_speed
    return diffusion_minimax_h3_image_speed


@pytest.fixture(autouse = True)
def _env(monkeypatch):
    for name in (
        "UNSLOTH_H3_IMAGE_PARK",
        "UNSLOTH_H3_IMAGE_H2D_GIBS",
        "UNSLOTH_H3_IMAGE_PLACEMENT",
        "UNSLOTH_H3_IMAGE_MEASURED_WORKING_SET",
        "UNSLOTH_H3_IMAGE_INT8_GEMM_STREAMED",
    ):
        monkeypatch.delenv(name, raising = False)
    monkeypatch.setenv("UNSLOTH_H3_IMAGE_SOURCE_RESIZE", "0")


def _plans(h3i, te_gb = 27.2):
    """Every unforced plan over budgets x links x park settings x canvases x reference counts."""
    budgets = [round(6.0 + 0.1 * i, 1) for i in range(360)]  # 6.0 .. 41.9 GB
    canvases = [(2048, 2048), (4096, 4096), (2048, 1024)]
    for vram, (h2d, tops), park, (w, h), refs in itertools.product(
        budgets, LINKS, (None, True, False), canvases, (0, 1, 3)
    ):
        try:
            plan = h3i.plan_placement(
                vram_gb = vram,
                host_gb = 160.0,
                dit_gb = 21.9,
                te_gb = te_gb,
                inflight = 2,
                h2d_gib_s = h2d,
                block_tops = tops,
                park = park,
                width = w,
                height = h,
                references = refs,
            )
        except RuntimeError:
            continue  # refused at load with what it would need: nothing runs
        yield (vram, h2d, park, w, h, refs), plan


@pytest.mark.parametrize("te_gb", [27.2, 66.7])
@pytest.mark.parametrize("measured", ["1", "0"])
def test_every_planned_placement_has_an_int8_gemm(h3i, speed, monkeypatch, te_gb, measured):
    monkeypatch.setenv(h3i.MEASURED_WORKING_SET_ENV, measured)
    seen = set()
    for key, plan in _plans(h3i, te_gb):
        assert plan.placement != h3i.PLACE_SEQUENTIAL, (
            key,
            plan.reasons,
        )  # unforced: never for the int8 DiT
        resident = plan.dit_resident_blocks
        all_resident = plan.placement == h3i.PLACE_STREAM and resident == 50
        seen.add((plan.placement, all_resident))
        for pinned in (True, False):
            keeps = plan.placement == h3i.PLACE_STREAM and (pinned or all_resident)
            route = speed.int8_gemm_route(plan.placement, keeps_parameters = keeps)
            if plan.placement == h3i.PLACE_STREAM and not keeps:
                assert (
                    route is None
                )  # pageable streaming swaps parameters: the stock path, as before
                continue
            assert route in ("resident", "streamed"), (key, plan.placement, resident, pinned)
    # the case this guards: a stream plan that keeps every block
    assert (h3i.PLACE_STREAM, True) in seen and (h3i.PLACE_STREAM, False) in seen
    assert (h3i.PLACE_DIT_RESIDENT, False) in seen or te_gb > 30


def test_the_gemm_route_table(speed):
    route = speed.int8_gemm_route
    assert route("resident") == route("dit_resident") == "resident"
    assert route("stream", keeps_parameters = True) == "streamed"
    assert route("stream", keeps_parameters = False) is None
    assert route("stream", keeps_parameters = True, streamed_gemm = False) is None
    assert route("resident", fp16_contract = True) is None
    assert route("sequential") is None


def test_a_streamer_with_nothing_to_stream_keeps_parameters(h3i, speed):
    import torch

    blocks = torch.nn.ModuleList(torch.nn.Linear(4, 4) for _ in range(3))
    every = h3i.ModuleListStreamer(blocks, "cpu", 3)
    every._unsloth_role = "transformer"
    assert every.streamed == 0 and not every.handles
    pipe = types.SimpleNamespace(_unsloth_h3_streamers = [every])
    assert speed._dit_streamer_keeps_parameters(pipe)
    some = h3i.ModuleListStreamer(blocks, "cpu", 1)
    some._unsloth_role = "transformer"
    assert not speed._dit_streamer_keeps_parameters(
        types.SimpleNamespace(_unsloth_h3_streamers = [some])
    )
    some.remove()
    assert not speed._dit_streamer_keeps_parameters(types.SimpleNamespace(_unsloth_h3_streamers = []))


def test_the_measured_table_keeps_more_blocks_and_the_switch_restores_the_old_plan(
    h3i, monkeypatch
):
    card = dict(vram_gb = 25.06, host_gb = 84.44, h2d_gib_s = 3.0, block_tops = 220 * 0.67, **INT8)
    head = h3i.plan_placement(**card)
    monkeypatch.setenv(h3i.MEASURED_WORKING_SET_ENV, "0")
    old = h3i.plan_placement(**card)
    assert old.working_gb is None and old.dit_resident_blocks == 44  # main's plan for the x4 3090
    assert head.dit_resident_blocks == 50 and head.working_gb is not None
    # a smaller card keeps fewer blocks under either pricing, never more than fit
    for vram in (7.45, 11.75, 16.04, 24.64):
        monkeypatch.setenv(h3i.MEASURED_WORKING_SET_ENV, "1")
        new = h3i.plan_placement(**dict(card, vram_gb = vram))
        monkeypatch.setenv(h3i.MEASURED_WORKING_SET_ENV, "0")
        was = h3i.plan_placement(**dict(card, vram_gb = vram))
        assert new.dit_resident_blocks >= was.dit_resident_blocks, vram
        assert new.slack_gb >= 0 and new.dit_resident_blocks * new.block_gb + new.working_gb <= vram


def test_a_card_under_the_measured_floor_is_still_refused(h3i):
    with pytest.raises(RuntimeError, match = r"needs about [0-9.]+ GB of free GPU memory"):
        h3i.plan_placement(vram_gb = 4.0, host_gb = 80.0, **INT8)


def test_a_larger_canvas_or_more_references_gives_blocks_back(h3i, monkeypatch):
    plan = h3i.plan_placement(vram_gb = 24.64, host_gb = 160.0, h2d_gib_s = 3.0, block_tops = 270.0, **INT8)
    resident = plan.dit_resident_blocks
    assert h3i.render_resident_blocks(plan, 2048, 2048, 0) == resident
    big = h3i.render_resident_blocks(plan, 4096, 4096, 0)
    edit = h3i.render_resident_blocks(plan, 2048, 2048, 3)
    assert big < resident and edit < resident
    short = (
        h3i.stream_working_gb(plan.layout, 4096, 4096, 0, plan.vram_gb)
        - plan.working_gb
        - plan.slack_gb
    )
    assert (resident - big) * plan.block_gb >= short > (resident - big - 1) * plan.block_gb
    monkeypatch.setenv(h3i.MEASURED_WORKING_SET_ENV, "0")
    old = h3i.plan_placement(vram_gb = 24.64, host_gb = 160.0, h2d_gib_s = 3.0, block_tops = 270.0, **INT8)
    assert (
        h3i.render_resident_blocks(old, 4096, 4096, 0) is None
    )  # the old pricing has no per-render fit


def _tiny(h3i, tmp_path, monkeypatch, resident):
    import _minimax_h3_image_tiny as tiny

    root = tmp_path / "rel"
    ref = tiny.make_tiny_release(root)
    te = tiny.make_text_encoder(ref["tokenizer"])
    real = h3i.plan_placement

    def plan(**kw):
        p = real(**kw)
        p.placement, p.dit_resident_blocks, p.te_resident_layers = h3i.PLACE_STREAM, resident, 0
        p.parked = p.vae_parked = False
        p.layout, p.block_gb, p.slack_gb, p.working_gb = h3i.LAYOUT_TODAY, 0.4, 0.0, 2.65
        return p

    monkeypatch.setattr(h3i, "plan_placement", plan)
    return root, te


def _image(pipe):
    import torch
    out = pipe(prompt = "a red cube", height = 64, width = 64, generator = torch.Generator().manual_seed(5))
    return torch.tensor(list(out.images[0].getdata()))


def _dit_streamer(pipe):
    return next(s for s in pipe._unsloth_h3_streamers if s._unsloth_role == "transformer")


def test_a_render_needing_more_room_streams_more_blocks_with_the_same_image(
    h3i, tmp_path, monkeypatch
):
    import torch

    root, te = _tiny(h3i, tmp_path, monkeypatch, resident = 2)
    pipe = h3i.load_minimax_h3_image_pipeline(
        str(root), device = "cpu", text_encoder = copy.deepcopy(te)
    )
    try:
        assert _dit_streamer(pipe).streamed == 0
        want = _image(pipe)
        assert pipe._unsloth_h3_last_stats["fit"] == {"resident": 2}
        # the next render's working set is 0.5 GB past the plan's: one 0.4 GB block is not enough, two are
        monkeypatch.setattr(h3i, "stream_working_gb", lambda *a, **k: 3.15)
        got = _image(pipe)
        assert pipe._unsloth_h3_last_stats["fit"] == {"resident": 0, "demoted": True}
        assert _dit_streamer(pipe).streamed == 2 and pipe._unsloth_h3_plan.dit_resident_blocks == 0
        assert torch.equal(want, got)
        # the same render again fits what it already gave back
        assert torch.equal(want, _image(pipe))
        assert pipe._unsloth_h3_last_stats["fit"] == {"resident": 0}
    finally:
        pipe.release()


def test_an_out_of_memory_render_streams_more_blocks_and_runs_again(h3i, tmp_path, monkeypatch):
    import torch

    root, te = _tiny(h3i, tmp_path, monkeypatch, resident = 2)
    pipe = h3i.load_minimax_h3_image_pipeline(
        str(root), device = "cpu", text_encoder = copy.deepcopy(te)
    )
    try:
        want = _image(pipe)
        block = pipe.transformer.transformer_blocks[1]
        calls = {"n": 0}

        def boom(*_a, **_k):
            calls["n"] += 1
            if calls["n"] == 1:
                raise torch.OutOfMemoryError("CUDA out of memory. Tried to allocate 114 MiB")

        handle = block.register_forward_pre_hook(boom)
        got = _image(pipe)
        handle.remove()
        assert calls["n"] > 1 and pipe._unsloth_h3_plan.dit_resident_blocks == 0
        assert any("out of memory" in r for r in pipe._unsloth_h3_plan.reasons)
        assert torch.equal(want, got)
        # nothing left to give back: the error reaches the caller
        handle = block.register_forward_pre_hook(
            lambda *_a: (_ for _ in ()).throw(torch.OutOfMemoryError("again"))
        )
        with pytest.raises(torch.OutOfMemoryError):
            _image(pipe)
        handle.remove()
    finally:
        pipe.release()


def test_the_switch_off_keeps_the_old_render_path(h3i, tmp_path, monkeypatch):
    import torch

    root, te = _tiny(h3i, tmp_path, monkeypatch, resident = 2)
    monkeypatch.setenv(h3i.MEASURED_WORKING_SET_ENV, "0")
    pipe = h3i.load_minimax_h3_image_pipeline(
        str(root), device = "cpu", text_encoder = copy.deepcopy(te)
    )
    try:
        monkeypatch.setattr(h3i, "stream_working_gb", lambda *a, **k: 99.0)
        _image(pipe)
        assert pipe._unsloth_h3_last_stats["fit"] is None and _dit_streamer(pipe).streamed == 0
        block = pipe.transformer.transformer_blocks[1]
        handle = block.register_forward_pre_hook(
            lambda *_a: (_ for _ in ()).throw(torch.OutOfMemoryError("oom"))
        )
        with pytest.raises(torch.OutOfMemoryError):
            _image(pipe)
        handle.remove()
        assert pipe._unsloth_h3_plan.dit_resident_blocks == 2
    finally:
        pipe.release()


def test_demotion_never_leaves_an_installed_gemm_on_the_pageable_streamer(h3i, speed, monkeypatch):
    import torch

    blocks = torch.nn.ModuleList(torch.nn.Linear(4, 4) for _ in range(3))
    every = h3i.ModuleListStreamer(blocks, "cpu", 3)
    every._unsloth_role, every._unsloth_resident = "transformer", 3
    plan = h3i.H3ImagePlan(
        placement = h3i.PLACE_STREAM,
        dit_gb = 1.0,
        te_gb = 1.0,
        vae_gb = 1.0,
        vram_gb = 8.0,
        host_gb = None,
        dit_resident_blocks = 3,
        working_gb = 2.0,
        slack_gb = 0.0,
        block_gb = 0.4,
        layout = h3i.LAYOUT_TODAY,
    )
    pipe = types.SimpleNamespace(
        _unsloth_h3_plan = plan,
        _unsloth_h3_streamers = [every],
        transformer = types.SimpleNamespace(transformer_blocks = blocks),
        _device = lambda: torch.device("cpu"),
    )
    setattr(pipe, speed.LEVERS_ATTR, ("h3_int8_gemm_streamed",))
    assert not h3i.demote_dit_blocks(
        pipe, 1, "test"
    )  # CPU: no pinned streamer, and the GEMM needs one
    assert plan.dit_resident_blocks == 3 and pipe._unsloth_h3_streamers == [every]
    setattr(pipe, speed.LEVERS_ATTR, ())
    assert h3i.demote_dit_blocks(pipe, 1, "test")
    new = pipe._unsloth_h3_streamers[0]
    assert new is not every and new.streamed == 2 and plan.dit_resident_blocks == 1
    assert plan.working_gb == pytest.approx(2.8)
    assert not h3i.demote_dit_blocks(pipe, 1, "test")  # already there


def _measured_need_gb(h3i, plan):
    """Device GB the measured pricing books for ``plan`` (weights it keeps + the working set with its reserve)."""
    block = plan.block_gb
    layer = (plan.te_gb - h3i.TE_TOP_GB) / 50
    phases = [2 * block, 2 * layer + h3i.TE_VISION_GB, h3i.DIT_REFINER_GB]
    step_top = h3i.DIT_TOP_GB - h3i.DIT_REFINER_GB
    fixed = {
        h3i.LAYOUT_TODAY: plan.vae_gb + h3i.DIT_TOP_GB + h3i.TE_TOP_GB + block + max(block, layer),
        h3i.LAYOUT_PARKED: step_top + plan.vae_gb + max(phases),
        h3i.LAYOUT_PARKED_VAE: step_top + max(phases + [plan.vae_gb]),
    }[plan.layout]
    return fixed + plan.working_gb + plan.dit_resident_blocks * block


@pytest.mark.parametrize(
    "card",
    [
        dict(vram_gb = 7.73, host_gb = 80.53),  # RTX 6000 Ada capped to 8 GB
        dict(vram_gb = 25.06, host_gb = 84.44, h2d_gib_s = 3.0, block_tops = 220 * 0.67),  # 3090 on x4
        dict(vram_gb = 24.91, host_gb = 84.1, h2d_gib_s = 24.0, block_tops = 368.0),  # Ada 24 GB x16
        dict(vram_gb = 22.0, host_gb = 60.0, h2d_gib_s = 12.0, block_tops = 134.0),  # L4
        dict(vram_gb = 24.64, host_gb = 160.0, h2d_gib_s = 24.0, block_tops = 272.0),  # A100 capped, x16
        dict(vram_gb = 16.04, host_gb = 160.0, h2d_gib_s = 11.5, block_tops = 272.0),
        dict(vram_gb = 11.75, host_gb = 160.0, h2d_gib_s = 11.5, block_tops = 272.0),
    ],
)
def test_measured_plans_fit_their_budget_and_keep_at_least_as_many_blocks(h3i, monkeypatch, card):
    new = h3i.plan_placement(**card, **INT8)
    assert new.placement == h3i.PLACE_STREAM and new.working_gb is not None
    assert _measured_need_gb(h3i, new) <= card["vram_gb"] + 1e-6
    assert new.slack_gb == pytest.approx(card["vram_gb"] - _measured_need_gb(h3i, new), abs = 1e-6)
    monkeypatch.setenv(h3i.MEASURED_WORKING_SET_ENV, "0")
    old = h3i.plan_placement(**card, **INT8)
    if card.get("h2d_gib_s"):  # priced: the measured room never buys fewer resident blocks
        assert new.dit_resident_blocks >= old.dit_resident_blocks, (new.reasons, old.reasons)


@pytest.fixture
def pin_failure(monkeypatch):
    """The pinned streamer gets past its refusals on CPU and fails after moving the newly streamed blocks to the
    host (its copy stream is a CUDA one): the failure path a pinning error takes on a card."""
    from core.inference import diffusion_offload_prefetch as op

    monkeypatch.setattr(op, "_list_stream_refusal", lambda device: None)
    seen = []
    real = op._module_device

    def spy(module):
        seen.append(module)
        return real(module)

    monkeypatch.setattr(op, "_module_device", spy)
    return seen


def _fake_gemm(pipe, speed):
    setattr(
        pipe, speed.LEVERS_ATTR, tuple(speed.speed_levers(pipe) or ()) + ("h3_int8_gemm_streamed",)
    )


def test_a_pin_failure_under_the_gemm_keeps_every_block_resident(
    h3i, speed, tmp_path, monkeypatch, pin_failure
):
    import torch

    root, te = _tiny(h3i, tmp_path, monkeypatch, resident = 2)
    pipe = h3i.load_minimax_h3_image_pipeline(
        str(root), device = "cpu", text_encoder = copy.deepcopy(te)
    )
    try:
        want = _image(pipe)
        old = _dit_streamer(pipe)
        assert isinstance(old, h3i.ModuleListStreamer) and old.streamed == 0
        _fake_gemm(pipe, speed)
        pin_failure.clear()  # the load's encoder streamer took the same path
        assert not h3i.demote_dit_blocks(pipe, 0, "pin failure")
        blocks = list(pipe.transformer.transformer_blocks)
        assert pin_failure == blocks  # both newly streamed blocks were put back where they were
        assert _dit_streamer(pipe) is old and pipe._unsloth_h3_plan.dit_resident_blocks == 2
        assert all(p.device == pipe._device() for b in blocks for p in b.parameters())
        assert torch.equal(want, _image(pipe))
    finally:
        pipe.release()


def test_a_pin_failure_after_detaching_a_prefetch_streamer_rebuilds_the_old_split(
    h3i, speed, tmp_path, monkeypatch, pin_failure
):
    import torch

    from core.inference import diffusion_offload_prefetch as op

    root, te = _tiny(h3i, tmp_path, monkeypatch, resident = 1)
    pipe = h3i.load_minimax_h3_image_pipeline(
        str(root), device = "cpu", text_encoder = copy.deepcopy(te)
    )
    try:
        want = _image(pipe)
        # stand in for the pinned streamer a card builds: detach() hands back no groups on CPU
        old = _dit_streamer(pipe)
        old.remove()
        fake = op.PrefetchedListStreamer()
        fake._unsloth_role, fake._unsloth_resident = "transformer", 1
        streamers = pipe._unsloth_h3_streamers
        streamers[streamers.index(old)] = fake
        _fake_gemm(pipe, speed)
        assert not h3i.demote_dit_blocks(pipe, 0, "pin failure")
        new = _dit_streamer(pipe)
        # neither the new split nor the old one could be pinned: the old split, pageable, never no streamer
        assert (
            isinstance(new, h3i.ModuleListStreamer)
            and new._unsloth_resident == 1
            and new.streamed == 1
        )
        assert pipe._unsloth_h3_plan.dit_resident_blocks == 1
        assert torch.equal(want, _image(pipe))
    finally:
        pipe.release()


def test_the_pageable_last_resort_pads_the_int8_rows_again(h3i, speed, monkeypatch):
    """Both pinned rebuilds fail after an existing pinned streamer was detached: the pageable fallback swaps
    parameters, so the int8 GEMM takes torchao's stock path, which needs the row pads the GEMM install removed."""
    import torch

    pytest.importorskip("torchao")
    from torchao.quantization import quantize_

    from core.inference import diffusion_offload_prefetch as op
    from core.inference.diffusion_transformer_quant import _make_quant_config

    torch.manual_seed(0)

    def block():
        b = torch.nn.Module()
        b.attn = torch.nn.Module()
        b.attn.to_q = torch.nn.Linear(256, 64, bias = False)
        b.ff = torch.nn.Module()
        b.ff.net = torch.nn.Sequential(
            torch.nn.Linear(256, 64, bias = False)
        )  # not a quantized suffix
        return b.to(torch.bfloat16)

    blocks = torch.nn.ModuleList([block(), block()])
    quantize_(
        blocks, _make_quant_config("int8"), filter_fn = lambda m, fqn: fqn.endswith("attn.to_q")
    )
    transformer = types.SimpleNamespace(transformer_blocks = blocks)
    if not h3i.restore_row_pads(transformer):
        pytest.skip("this torchao gives no per-row int8 Linears")
    # what the int8 GEMM install leaves: the pads it serves itself are unwrapped
    for b in blocks:
        b.attn.to_q = b.attn.to_q.inner
    x = torch.randn(1, 37, 256, dtype = torch.bfloat16)
    with torch.no_grad():
        want = blocks[1].attn.to_q(x)

    monkeypatch.setattr(
        op, "_list_stream_refusal", lambda device: None
    )  # gets to the pin, fails there on CPU
    old = op.PrefetchedListStreamer()
    old._unsloth_role, old._unsloth_resident = "transformer", 1
    plan = h3i.H3ImagePlan(
        placement = h3i.PLACE_STREAM,
        dit_gb = 1.0,
        te_gb = 1.0,
        vae_gb = 1.0,
        vram_gb = 8.0,
        host_gb = None,
        dit_resident_blocks = 1,
        working_gb = 2.0,
        slack_gb = 0.0,
        block_gb = 0.4,
        layout = h3i.LAYOUT_TODAY,
    )
    pipe = types.SimpleNamespace(
        _unsloth_h3_plan = plan,
        _unsloth_h3_streamers = [old],
        transformer = transformer,
        _device = lambda: torch.device("cpu"),
    )
    setattr(pipe, speed.LEVERS_ATTR, ("h3_int8_gemm_streamed",))
    assert not h3i.demote_dit_blocks(pipe, 0, "pin failure")
    new = pipe._unsloth_h3_streamers[0]
    assert (
        isinstance(new, h3i.ModuleListStreamer) and new._unsloth_resident == 1 and new.streamed == 1
    )
    assert plan.dit_resident_blocks == 1
    for b in blocks:
        assert type(b.attn.to_q).__name__ == "H3ImageRowPad"
        assert type(b.attn.to_q.inner).__name__ != "H3ImageRowPad"  # wrapped once
        assert type(b.ff.net[0]).__name__ == "Linear"  # only the quantized suffixes
    seen = []
    blocks[1].attn.to_q.inner.register_forward_pre_hook(lambda m, a: seen.append(a[0].shape[0]))
    with torch.no_grad():
        got = blocks[1].attn.to_q(x)
    assert seen == [64] and torch.equal(got, want)
    assert h3i.restore_row_pads(transformer) == 0  # idempotent
