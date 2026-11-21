# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""MiniMax-H3-Image stream tier on CPU: parked layout planning, pricing and bit-identical parked renders."""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
for p in (_BACKEND, Path(__file__).resolve().parent):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

STREAM = dict(dit_gb = 21.9, te_gb = 27.2, inflight = 2)
# RTX 6000 Ada capped to 8 GB (the 8 GB OOM report: 7.73 GB free at load, 80.5 GB host)
CARD_8GB = dict(vram_gb = 7.73, host_gb = 80.53)
# RTX 3090 on PCIe 3.0 x4 (19 of 50 blocks streamed, 2.83 s/step); int8 GEMM ~220 TOPS
RTX3090_X4 = dict(vram_gb = 25.06, host_gb = 84.44, h2d_gib_s = 3.0, block_tops = 220 * 0.67)


@pytest.fixture(scope = "module")
def h3i():
    pytest.importorskip("torch")
    pytest.importorskip("diffusers")
    pytest.importorskip("transformers")
    from core.inference import diffusion_minimax_h3_image

    return diffusion_minimax_h3_image


@pytest.fixture(autouse = True)
def _auto_layout(monkeypatch):
    for name in (
        "UNSLOTH_H3_IMAGE_PARK",
        "UNSLOTH_H3_IMAGE_H2D_GIBS",
        "UNSLOTH_H3_IMAGE_PLACEMENT",
    ):
        monkeypatch.delenv(name, raising = False)
    monkeypatch.setenv("UNSLOTH_H3_IMAGE_SOURCE_RESIZE", "0")
    # old pricing; measured pricing is in test_h3_image_resident_route.py
    monkeypatch.setenv("UNSLOTH_H3_IMAGE_MEASURED_WORKING_SET", "0")


def _device_need_gb(
    h3i,
    plan,
    width = 2048,
    height = 2048,
):
    act = h3i.activation_gb(width, height) + 1.5
    block = (plan.dit_gb - h3i.DIT_TOP_GB) / 50
    layer = (plan.te_gb - h3i.TE_TOP_GB) / 50
    resident = plan.dit_resident_blocks * block
    if plan.parked:
        phase = max(2 * block, 2 * layer + h3i.TE_VISION_GB, h3i.DIT_REFINER_GB)
        if plan.vae_parked:
            return act + resident + h3i.DIT_TOP_GB - h3i.DIT_REFINER_GB + max(phase, plan.vae_gb)
        return act + resident + h3i.DIT_TOP_GB - h3i.DIT_REFINER_GB + plan.vae_gb + phase
    return act + resident + h3i.DIT_TOP_GB + h3i.TE_TOP_GB + plan.vae_gb + 2 * block


def test_an_8gb_card_gets_a_layout_that_fits_it(h3i):
    plan = h3i.plan_placement(**CARD_8GB, **STREAM)
    assert plan.placement == h3i.PLACE_STREAM
    assert plan.parked and plan.vae_parked, plan.reasons
    assert _device_need_gb(h3i, plan) <= CARD_8GB["vram_gb"]
    assert plan.host_need_gb is not None and plan.host_need_gb <= CARD_8GB["host_gb"]


def test_a_card_too_small_for_any_layout_is_refused_at_load(h3i):
    with pytest.raises(RuntimeError, match = r"needs about [0-9.]+ GB of free GPU memory"):
        h3i.plan_placement(vram_gb = 6.0, host_gb = 80.0, **STREAM)


def test_forcing_todays_layout_on_8gb_is_refused_not_run_out_of_memory(h3i, monkeypatch):
    with pytest.raises(RuntimeError, match = "free GPU memory"):
        h3i.plan_placement(**CARD_8GB, **STREAM, park = False)


def test_a_slow_link_keeps_more_blocks_on_the_card(h3i):
    unpriced = h3i.plan_placement(
        vram_gb = RTX3090_X4["vram_gb"], host_gb = RTX3090_X4["host_gb"], **STREAM
    )
    slow = h3i.plan_placement(**RTX3090_X4, **STREAM)
    assert not unpriced.parked and unpriced.dit_resident_blocks == 31  # the reported 3090 plan
    assert slow.parked and slow.h2d_gib_s == 3.0
    assert slow.dit_resident_blocks >= unpriced.dit_resident_blocks + 7, slow.reasons
    assert _device_need_gb(h3i, slow) <= RTX3090_X4["vram_gb"]
    assert "priced faster" in slow.reasons[-1]


@pytest.mark.parametrize("int8_tops", [150.0, 220.0, 284.0])
def test_the_slow_link_decision_holds_across_compute_readings(h3i, int8_tops):
    card = dict(RTX3090_X4, block_tops = int8_tops * h3i.INT8_BLOCK_EFFICIENCY)
    assert h3i.plan_placement(**card, **STREAM).parked


@pytest.mark.parametrize(
    "card",
    [
        dict(
            vram_gb = 24.91, host_gb = 84.1, h2d_gib_s = 24.0, block_tops = 368.0
        ),  # RTX 6000 Ada x16 at 24 GB
        dict(vram_gb = 22.0, host_gb = 60.0, h2d_gib_s = 12.0, block_tops = 134.0),  # L4
        dict(vram_gb = 24.64, host_gb = 160.0, h2d_gib_s = 24.0, block_tops = 272.0),  # A100 on a x16 link
    ],
)
def test_a_fast_link_keeps_todays_layout(h3i, card):
    priced = h3i.plan_placement(**card, **STREAM)
    unpriced = h3i.plan_placement(vram_gb = card["vram_gb"], host_gb = card["host_gb"], **STREAM)
    assert not priced.parked, priced.reasons
    assert priced.dit_resident_blocks == unpriced.dit_resident_blocks


def test_the_render_pricing_matches_the_reported_3090_step(h3i):
    per_step = h3i.stream_render_seconds(
        resident = 31,
        num_blocks = 50,
        block_gb = 0.398,
        steps = 1,
        h2d_gib_s = 3.0,
        block_tops = 220 * 0.67,
        width = 2048,
        height = 2048,
    )
    assert 2.6 < per_step < 3.1


def test_the_render_pricing_matches_the_a100_measurements(h3i):
    # A100 (11.5 GiB/s, int8 GEMM 406 TOPS): 30 resident + 20 streamed 0.953 s/step, all 50 streamed 1.593 s/step
    priced = dict(num_blocks = 50, block_gb = 0.398, steps = 1, h2d_gib_s = 11.5, width = 2048, height = 2048)
    tops = 406 * h3i.INT8_BLOCK_EFFICIENCY
    assert h3i.stream_render_seconds(resident = 30, block_tops = tops, **priced) == pytest.approx(
        0.953, rel = 0.08
    )
    assert h3i.stream_render_seconds(resident = 0, block_tops = tops, **priced) == pytest.approx(
        1.593, rel = 0.08
    )
    assert h3i.stream_render_seconds(resident = 50, block_tops = tops, **priced) == pytest.approx(
        0.578, rel = 0.08
    )


def test_env_switches(h3i, monkeypatch):
    monkeypatch.setenv(h3i.PARK_ENV, "1")
    assert h3i.park_setting() is True
    monkeypatch.setenv(h3i.PARK_ENV, "0")
    assert h3i.park_setting() is False
    monkeypatch.setenv(h3i.PARK_ENV, "auto")
    assert h3i.park_setting() is None
    monkeypatch.setenv(h3i.H2D_ENV, "2.5")
    assert h3i.link_measurements("cpu")[0] == 2.5
    big = dict(vram_gb = 40.0, host_gb = 160.0, **STREAM)
    assert h3i.plan_placement(**big, forced = "stream", park = True).parked


def test_host_parking_brings_weights_in_only_around_the_forward(h3i):
    import torch

    lin = torch.nn.Linear(4, 4)
    seen = []
    parked = h3i.park_on_host(lin, "meta", per_forward = True)
    lin.register_forward_pre_hook(lambda m, a: seen.append(m.weight.device.type))
    assert lin.weight.device.type == "cpu" and parked.bytes == 4 * 4 * 4 + 4 * 4
    out = lin(torch.empty(2, 4, device = "meta"))
    assert out.device.type == "meta" and seen == ["meta"]
    assert lin.weight.device.type == "cpu" and parked.depth == 0
    with pytest.raises(RuntimeError):
        lin(torch.empty(2, 3, device = "meta"))
    assert lin.weight.device.type == "cpu" and parked.depth == 0  # a raising forward still leaves
    assert h3i.park_on_host(lin, "meta") is parked
    assert h3i.unpark(lin) and len(lin._forward_pre_hooks) == 1
    assert not h3i.unpark(lin)


def test_host_lookup_is_bit_identical(h3i):
    import torch

    emb = torch.nn.Embedding(50, 8)
    ids = torch.tensor([[1, 5, 7, 49]])
    want = emb(ids).clone()
    h3i.lookup_on_host(emb, "cpu")
    assert torch.equal(emb(ids), want)


def _stream_plan(h3i, parked, vae_parked):
    real = h3i.plan_placement

    def plan(**kw):
        p = real(**kw)
        p.placement, p.dit_resident_blocks, p.te_resident_layers = h3i.PLACE_STREAM, 1, 0
        p.parked, p.vae_parked = parked, vae_parked
        return p

    return plan


def _render(h3i, root, te, parked, vae_parked, monkeypatch):
    import torch

    monkeypatch.setattr(h3i, "plan_placement", _stream_plan(h3i, parked, vae_parked))
    pipe = h3i.load_minimax_h3_image_pipeline(
        str(root), device = "cpu", text_encoder = copy.deepcopy(te)
    )
    try:
        assert pipe._unsloth_h3_plan.parked is parked
        marks = {
            "refiner": pipe.transformer.token_refiner,
            "vae": pipe.vae,
            "vision": pipe.text_encoder.model.visual,
            "embed": pipe.text_encoder.model.language_model.embed_tokens,
        }
        want = {k: parked for k in marks}
        want["vae"] = vae_parked
        assert {k: h3i.PARKED_ATTR in m.__dict__ for k, m in marks.items()} == want
        image = pipe(
            prompt = "a red cube", height = 64, width = 64, generator = torch.Generator().manual_seed(5)
        ).images[0]
        if vae_parked:
            assert pipe.vae.__dict__[h3i.PARKED_ATTR].depth == 0
            assert pipe.vae.post_quant_conv.weight.device.type == "cpu"
        if parked:
            assert pipe.transformer.token_refiner.__dict__[h3i.PARKED_ATTR].depth == 0
        return torch.tensor(list(image.getdata()))
    finally:
        pipe.release()


def test_the_parked_layout_renders_bit_identically(h3i, tmp_path, monkeypatch):
    import torch

    import _minimax_h3_image_tiny as tiny

    root = tmp_path / "rel"
    ref = tiny.make_tiny_release(root)
    te = tiny.make_text_encoder(ref["tokenizer"])
    today = _render(h3i, root, te, False, False, monkeypatch)
    assert torch.equal(today, _render(h3i, root, te, True, False, monkeypatch))
    assert torch.equal(today, _render(h3i, root, te, True, True, monkeypatch))


def test_a_parked_vae_keeps_its_fp32_fallback_across_decodes(h3i, tmp_path, monkeypatch):
    """After an fp16 overflow recasts the VAE to fp32, the next parked decode must upload fp32, not the snapshot."""
    import torch

    import _minimax_h3_image_tiny as tiny
    from core.inference import diffusion_vae_canvas as canvas

    root = tmp_path / "rel"
    ref = tiny.make_tiny_release(root)
    te = tiny.make_text_encoder(ref["tokenizer"])
    monkeypatch.setattr(h3i, "plan_placement", _stream_plan(h3i, True, True))
    pipe = h3i.load_minimax_h3_image_pipeline(str(root), device = "cpu", text_encoder = te)
    try:
        vae = pipe.vae
        assert canvas._block_linears(vae)
        assert h3i.unpark(vae)
        canvas.install(vae, "cpu", speed = False)
        canvas.set_decoder_dtype(vae, torch.float16)
        vae._unsloth_canvas_half = True
        h3i.park_on_host(vae, "cpu", pin = False)
        seen = []

        def fake_stock(z, *a, **k):
            seen.append(canvas._block_linears(vae)[0].weight.dtype)
            return torch.zeros(1, 3, 64, 64)

        vae.__dict__["_unsloth_canvas_stock_decode"] = fake_stock
        vae.use_tiling = False
        verdicts = iter(["nonfinite"])
        monkeypatch.setattr(canvas, "_probe", lambda out: next(verdicts, None))
        latents = torch.zeros(1, int(vae.config.latent_channels), 4, 4)
        pipe._decode(latents, 64, 64, "pil")
        assert seen == [torch.float16, torch.float32]  # overflowed in fp16, decoded again in fp32
        assert vae._unsloth_canvas_half is False
        assert canvas._block_linears(vae)[0].weight.device.type == "cpu"
        pipe._decode(latents, 64, 64, "pil")
        assert seen[2:] == [torch.float32]
        assert canvas._block_linears(vae)[0].weight.dtype == torch.float32
    finally:
        pipe.release()


def test_a_parked_module_keeps_tensors_its_phase_adds(h3i):
    import torch

    lin = torch.nn.Linear(4, 4)
    parked = h3i.park_on_host(lin, "cpu", pin = False)
    parked.enter()
    lin.register_buffer("extra", torch.ones(3))
    lin.weight = torch.nn.Parameter(torch.full((4, 4), 2.0), requires_grad = False)
    parked.leave()
    parked.enter()
    assert torch.equal(lin.extra, torch.ones(3)) and torch.equal(
        lin.weight, torch.full((4, 4), 2.0)
    )
    parked.leave()
    assert parked.bytes == 4 * 4 * 4 + 4 * 4 + 3 * 4


def test_a_parked_upload_that_runs_out_of_memory_rolls_back(h3i, monkeypatch):
    import torch

    lin = torch.nn.Linear(4, 4)
    parked = h3i.park_on_host(lin, "meta", pin = False)
    real_to = torch.Tensor.to
    calls = []

    def to(self, *args, **kwargs):
        if args and str(args[0]) == "meta":
            calls.append(1)
            if len(calls) == 2:
                raise torch.cuda.OutOfMemoryError("injected: the second tensor does not fit")
        return real_to(self, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", to)
    with pytest.raises(torch.cuda.OutOfMemoryError):
        parked.enter()
    # nothing is left half on the card, and the failed upload does not count as an entry
    assert parked.depth == 0
    assert lin.weight.device.type == "cpu" and lin.bias.device.type == "cpu"
    parked.enter()  # the retry uploads every tensor again
    assert parked.depth == 1 and len(calls) == 4
    assert lin.weight.device.type == "meta" and lin.bias.device.type == "meta"
    parked.leave()
    assert parked.depth == 0 and lin.weight.device.type == "cpu" and lin.bias.device.type == "cpu"
    assert parked.bytes == 4 * 4 * 4 + 4 * 4
