# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""TP-2 for Krea 2 on two real CUDA cards: the split matches one card, and the decode lend moves the first card's
shards to the second and back without changing a single output bit."""

from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
diffusers = pytest.importorskip("diffusers")
pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        not torch.cuda.is_available() or torch.cuda.device_count() < 2,
        reason = "needs two CUDA cards",
    ),
]
if not hasattr(diffusers, "Krea2Transformer2DModel"):
    pytest.skip("diffusers without Krea 2", allow_module_level = True)

from core.inference import diffusion_small_host as small_host  # noqa: E402
from core.inference import diffusion_tp2 as tp2  # noqa: E402

TEXT, GRID, LAYERS, DIM, BLOCKS = 12, 8, 3, 64, 4


def _model():
    torch.manual_seed(0)
    m = diffusers.Krea2Transformer2DModel(
        in_channels = 16,
        num_layers = BLOCKS,
        attention_head_dim = 64,
        num_attention_heads = 8,
        num_key_value_heads = 2,
        intermediate_size = 1024,
        timestep_embed_dim = 32,
        text_hidden_dim = DIM,
        num_text_layers = LAYERS,
        text_num_attention_heads = 2,
        text_num_key_value_heads = 2,
        text_intermediate_size = 64,
        num_layerwise_text_blocks = 1,
        num_refiner_text_blocks = 1,
        axes_dims_rope = (16, 24, 24),
    ).eval()
    with torch.no_grad():
        for p in m.parameters():
            p.normal_(0, 0.05)
    return m


def _inputs(device):
    g = torch.Generator().manual_seed(1)
    mask = torch.zeros(1, TEXT, dtype = torch.bool)
    mask[:, :7] = True
    hh, ww = torch.meshgrid(torch.arange(GRID), torch.arange(GRID), indexing = "ij")
    img = torch.stack([torch.zeros_like(hh), hh, ww], dim = -1).reshape(-1, 3)
    pos = torch.cat([torch.zeros(TEXT, 3, dtype = img.dtype), img])
    kw = {
        "hidden_states": torch.randn(1, GRID * GRID, 16, generator = g),
        "encoder_hidden_states": torch.randn(1, TEXT, LAYERS, DIM, generator = g),
        "timestep": torch.full((1,), 0.5),
        "position_ids": pos,
        "encoder_attention_mask": mask,
    }
    out = {k: v.to(device) for k, v in kw.items()}
    out["return_dict"] = False
    return out


def _run(m):
    with torch.no_grad():
        return m(**_inputs("cuda:0"))[0]


@pytest.mark.parametrize("int8", [False, True])
def test_krea2_tp2_on_two_cards_and_the_decode_lend(monkeypatch, int8):
    monkeypatch.delenv(tp2.DECODE_LEND_ENV, raising = False)
    ref = _model()
    if int8:
        monkeypatch.setattr(small_host, "_INT8_MIN_ELEMENTS", 1)
        small_host.quantize_int8_weight_(
            ref, compute_dtype = torch.float32, work_device = "cpu", keep_device = "cpu"
        )
    tp = copy.deepcopy(ref)
    ref = ref.to("cuda:0")
    devs = [torch.device("cuda", 0), torch.device("cuda", 1)]
    rec = tp2.install(tp, devs)
    assert rec["blocks"] == BLOCKS
    want = _run(ref)
    before = _run(tp)
    torch.testing.assert_close(before, want, rtol = 2e-4, atol = 2e-4)

    stack = tp.transformer_blocks[0]
    first = stack.shards()[0]
    torch.cuda.synchronize()
    used0 = torch.cuda.memory_allocated(0)
    restore = tp2.lend_for_decode(SimpleNamespace(transformer = tp))
    assert restore is not None
    assert all(p.device == devs[1] for blk in first for p in blk.parameters())
    # the first card gave back about one shard
    shard_bytes = sum(p.numel() * p.element_size() for blk in first for p in blk.parameters())
    assert used0 - torch.cuda.memory_allocated(0) >= 0.9 * shard_bytes
    assert tp2.lend_for_decode(SimpleNamespace(transformer = tp)) is None  # already lent
    restore()
    assert all(p.device == devs[0] for blk in first for p in blk.parameters())
    assert torch.equal(_run(tp), before)  # bit-identical after the round trip

    # a lend whose restore never ran is returned by the next denoiser call
    assert tp2.lend_for_decode(SimpleNamespace(transformer = tp)) is not None
    assert torch.equal(_run(tp), before)
    assert all(p.device == devs[0] for blk in first for p in blk.parameters())
    assert tp2.status(tp)["decode_lends"] == 2

    monkeypatch.setenv(tp2.DECODE_LEND_ENV, "0")
    assert tp2.lend_for_decode(SimpleNamespace(transformer = tp)) is None


def test_a_lend_that_fails_twice_is_returned_before_the_next_denoiser_call(monkeypatch):
    """Fault injection on real cards: the lend OOMs after one tensor reached cuda:1, then the rollback OOMs. The
    lend stays recorded, and the next call brings every tensor home and matches the pre-lend output bit for bit."""
    monkeypatch.delenv(tp2.DECODE_LEND_ENV, raising = False)
    tp = _model()
    devs = [torch.device("cuda", 0), torch.device("cuda", 1)]
    tp2.install(tp, devs)
    before = _run(tp)
    stack = tp.transformer_blocks[0]
    first = stack.shards()[0]
    real = tp2._move_card
    calls = {"away": 0, "home": 0}

    def flaky(entries, device):
        if torch.device(device) == devs[1]:
            calls["away"] += 1
            real(entries[:1], device)
            raise torch.OutOfMemoryError("injected lend OOM")
        calls["home"] += 1
        if calls["home"] == 1:
            raise torch.OutOfMemoryError("injected rollback OOM")
        real(entries, device)

    monkeypatch.setattr(tp2, "_move_card", flaky)
    assert tp2.lend_for_decode(SimpleNamespace(transformer = tp)) is None
    assert tp2._LENT_ATTR in stack.__dict__
    assert {p.device for blk in first for p in blk.parameters()} == set(devs)  # mixed, but recorded
    assert torch.equal(_run(tp), before)
    assert all(p.device == devs[0] for blk in first for p in blk.parameters())
    assert tp2._LENT_ATTR not in stack.__dict__
