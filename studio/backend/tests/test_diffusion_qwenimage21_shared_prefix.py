# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""One prompt rendered n times keeps one copy of the Qwen-Image-2.1 prefix K/V, bit-identical to n copies."""

from __future__ import annotations

import pytest

from core.inference import diffusion_qwenimage21 as q

torch = pytest.importorskip("torch")
qmod = pytest.importorskip("diffusers.models.transformers.transformer_qwenimage21")


@pytest.fixture(autouse = True)
def _stock_after(monkeypatch):
    monkeypatch.delenv(q.FAST_STEP_ENV, raising = False)
    monkeypatch.delenv(q.SHARED_PREFIX_KV_ENV, raising = False)
    q.uninstall()
    yield
    q.uninstall()


def _model():
    torch.manual_seed(0)
    m = qmod.QwenImage21Transformer2DModel(
        num_layers = 2,
        attention_head_dim = 16,
        num_attention_heads = 2,
        context_in_dim = 24,
        in_channels = 8,
        out_channels = 8,
        axes_dims_rope = (4, 6, 6),
    ).eval()
    with torch.no_grad():
        for p in m.parameters():
            p.normal_(0, 0.2)
    return m


def _inputs(
    batch,
    cond = (4, 4),
    text = 9,
    same = True,
    target = (4, 6),
):
    """``same``: every row is a copy of row 0 (num_images_per_prompt), else the text differs per row."""
    g = torch.Generator().manual_seed(3)
    th, tw = target
    ch, cw = cond
    vlm = torch.cat(
        [
            torch.zeros(text // 2, dtype = torch.bool),
            torch.ones(ch * cw // 4, dtype = torch.bool),
            torch.zeros(text - text // 2, dtype = torch.bool),
        ]
    )
    rows = 1 if same else batch
    text_rows = torch.randn(rows, vlm.numel(), 24, generator = g).expand(batch, -1, -1).contiguous()
    cond_rows = torch.randn(1, ch * cw, 8, generator = g).expand(batch, -1, -1).contiguous()
    return {
        "encoder_hidden_states": text_rows,
        "img_shapes": [[(1, ch, cw), (1, th, tw)]] * batch,
        "img_mask": torch.cat([vlm, torch.ones(th * tw // 4, dtype = torch.bool)])[None].repeat(
            batch, 1
        ),
        "cond": cond_rows,
        "latents": torch.randn(batch, th * tw, 8, generator = g),
    }


def _render(
    m,
    inp,
    steps = 4,
):
    outs = []
    kv = qmod.QwenImage21KVCache(len(m.transformer_blocks))
    lat = inp["latents"]
    with torch.inference_mode():
        for i in range(steps):
            hs = torch.cat([inp["cond"], lat], 1)
            out = m(
                hidden_states = hs,
                timestep = torch.full((hs.shape[0],), 1.0 - i / steps),
                encoder_hidden_states = inp["encoder_hidden_states"],
                encoder_hidden_states_mask = None,
                img_shapes = inp["img_shapes"],
                img_mask = inp["img_mask"],
                kv_cache = kv,
                kv_cache_mode = "extract" if i == 0 else "cached",
                return_dict = False,
            )[0]
            outs.append(out.clone())
            lat = lat - 0.1 * out[:, -lat.shape[1] :]
    return outs, kv


def _shared(kv, m):
    entries = [kv.get_layer(i) for i in range(len(m.transformer_blocks))]
    return [e.k.stride(0) == 0 and e.v.stride(0) == 0 for e in entries]


@pytest.mark.parametrize("batch", [2, 4])
def test_repeated_prompt_keeps_one_prefix_copy_and_matches_stock(batch):
    m = _model()
    ref, ref_kv = _render(m, _inputs(batch))
    assert not any(_shared(ref_kv, m))
    assert q.install()
    before = dict(q.SHARED_PREFIX_STATS)
    out, kv = _render(m, _inputs(batch))
    assert all(_shared(kv, m))
    assert q.SHARED_PREFIX_STATS["shared"] == before["shared"] + len(m.transformer_blocks)
    assert q.SHARED_PREFIX_STATS["bytes"] > before["bytes"]
    for a, b in zip(ref, out):
        assert torch.equal(a, b)
    for i in range(len(m.transformer_blocks)):
        assert torch.equal(kv.get_layer(i).k, ref_kv.get_layer(i).k)
        assert torch.equal(kv.get_layer(i).v, ref_kv.get_layer(i).v)


def test_distinct_rows_keep_every_copy():
    m = _model()
    ref, _ = _render(m, _inputs(2, same = False))
    assert q.install()
    out, kv = _render(m, _inputs(2, same = False))
    assert not any(_shared(kv, m))
    for a, b in zip(ref, out):
        assert torch.equal(a, b)


def test_single_image_and_kill_switch_keep_stock_entries(monkeypatch):
    m = _model()
    assert q.install()
    _, kv = _render(m, _inputs(1))
    assert not any(_shared(kv, m))
    monkeypatch.setenv(q.SHARED_PREFIX_KV_ENV, "0")
    _, kv = _render(m, _inputs(2))
    assert not any(_shared(kv, m))


class _Layer:
    def __init__(self, k, v):
        self.k, self.v = k, v


def test_rows_are_compared_bitwise():
    k = torch.ones(2, 3, 2, 4, dtype = torch.bfloat16)
    k[1, 0, 0, 0] = -0.0
    k[0, 0, 0, 0] = 0.0
    layers = [_Layer(k, k.clone())]
    assert q.share_prefix_rows(layers) == 0
    assert layers[0].k.stride(0) != 0


def test_equal_rows_free_all_but_one():
    row = torch.randn(1, 5, 2, 4).to(torch.bfloat16)
    k = row.repeat(3, 1, 1, 1)
    layers = [_Layer(k, k.clone()), _Layer(k.clone(), k.clone())]
    freed = q.share_prefix_rows(layers)
    assert freed == 4 * 2 * row.numel() * row.element_size()
    for layer in layers:
        assert layer.k.stride(0) == 0 and torch.equal(layer.k, k)
        assert torch.equal(torch.cat([layer.k, k], 1), torch.cat([k, k], 1))


def test_a_layer_with_distinct_rows_does_not_stop_the_others():
    row = torch.randn(1, 5, 2, 4).to(torch.bfloat16)
    same = row.repeat(2, 1, 1, 1)
    other = torch.randn(2, 5, 2, 4).to(torch.bfloat16)
    layers = [_Layer(other, other.clone()), _Layer(same, same.clone())]
    assert q.share_prefix_rows(layers) == 2 * row.numel() * row.element_size()
    assert layers[0].k.stride(0) != 0 and layers[1].k.stride(0) == 0
