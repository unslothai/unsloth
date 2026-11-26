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
    # Only a prefix too large to CUDA-graph is shared; the tiny model's K/V qualifies once the cap is 0.
    monkeypatch.setattr(q, "GRAPH_KV_MAX_BYTES", 0)
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


def _graph_count(m, monkeypatch, share):
    from torch._dynamo.utils import counters

    monkeypatch.setenv(q.SHARED_PREFIX_KV_ENV, "1" if share else "0")
    torch._dynamo.reset()
    counters.clear()
    outs, strides = [], []
    for same in (False, True, False):
        out, kv = _render(m, _inputs(2, same = same), steps = 2)
        outs.append(out)
        strides.append([kv.get_layer(i).k.stride(0) for i in range(len(m.transformer_blocks))])
    graphs = counters["stats"]["unique_graphs"]
    torch._dynamo.reset()
    return graphs, outs, strides


@pytest.mark.parametrize(
    "compile_kwargs",
    [{}, {"fullgraph": True}, {"fullgraph": True, "dynamic": True}],
    ids = ["default", "fullgraph", "dynamic"],
)
def test_shared_prefix_adds_no_dynamo_graphs(monkeypatch, compile_kwargs):
    """Same shapes, distinct -> identical -> distinct prompts: the compiled blocks never see a new K/V layout."""
    m = _model()
    assert q.install()
    m.compile_repeated_blocks(backend = "eager", **compile_kwargs)
    base_graphs, base_outs, base_strides = _graph_count(m, monkeypatch, False)
    densified = q.SHARED_PREFIX_STATS.get("densified", 0)
    graphs, outs, strides = _graph_count(m, monkeypatch, True)
    assert all(s != 0 for row in base_strides for s in row)
    assert all(s == 0 for s in strides[1]), "the identical-prompt render must share"
    assert graphs == base_graphs
    assert (
        q.SHARED_PREFIX_STATS.get("densified", 0) > densified
    ), "compiled blocks must get the dense copy"
    for a, b in zip(base_outs, outs):
        for x, y in zip(a, b):
            assert torch.equal(x, y)


def _graph_key_of(m, inp, kv):
    from core.inference.diffusion_cuda_graph import graph_key
    with torch.inference_mode():
        plan = q.graph_plan(
            m,
            (),
            dict(
                hidden_states = torch.cat([inp["cond"], inp["latents"]], 1),
                encoder_hidden_states = inp["encoder_hidden_states"],
                timestep = torch.ones(inp["latents"].shape[0]),
                img_mask = inp["img_mask"],
                img_shapes = inp["img_shapes"],
                kv_cache_mode = "cached",
                kv_cache = kv,
                return_dict = False,
            ),
        )
    return None if plan is None else graph_key(((), plan[1]))


def test_graph_key_is_the_same_for_identical_and_distinct_prompts(monkeypatch):
    monkeypatch.setattr(q, "GRAPH_KV_MAX_BYTES", 1 << 30)
    m = _model()
    assert q.install()
    keys = []
    for same in (False, True):
        inp = _inputs(2, same = same)
        _, kv = _render(m, inp, steps = 2)
        keys.append(_graph_key_of(m, inp, kv))
    assert keys[0] is not None and keys[0] == keys[1]
    # A prefix too large to graph is the only one shared, so a shared entry never reaches a graph key.
    monkeypatch.setattr(q, "GRAPH_KV_MAX_BYTES", 0)
    inp = _inputs(2)
    _, kv = _render(m, inp, steps = 2)
    assert all(_shared(kv, m))
    assert _graph_key_of(m, inp, kv) is None


class _OnCard1(torch.Tensor):
    @property
    def device(self):
        return torch.device("cuda", 1)


def test_capture_is_checked_on_the_operand_device(monkeypatch):
    """cuda:0 current while cuda:1 captures: the K/V on cuda:1 must not be compared (a host sync)."""
    import contextlib

    state = {"current": 0}

    @contextlib.contextmanager
    def device(index):
        old, state["current"] = state["current"], int(index)
        try:
            yield
        finally:
            state["current"] = old

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: state["current"])
    monkeypatch.setattr(torch.cuda, "device", device)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: state["current"] == 1)
    row = torch.randn(1, 5, 2, 4).to(torch.bfloat16)
    k = torch.Tensor._make_subclass(_OnCard1, row.repeat(2, 1, 1, 1))
    v = torch.Tensor._make_subclass(_OnCard1, row.repeat(2, 1, 1, 1))
    layer = _Layer(k, v)
    assert q.share_prefix_rows([layer]) == 0
    assert layer.k is k and layer.v is v


def test_oom_on_the_second_clone_propagates_and_keeps_the_entry_whole(monkeypatch):
    row = torch.arange(12, dtype = torch.bfloat16).view(1, 3, 2, 2)
    k = row.repeat(2, 1, 1, 1)
    v = k.clone()
    layer = _Layer(k, v)
    clone = torch.Tensor.clone
    calls = []

    def fail_second(t, *args, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            raise torch.cuda.OutOfMemoryError("simulated OOM on the V clone")
        return clone(t, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "clone", fail_second)
    with pytest.raises(torch.cuda.OutOfMemoryError):
        q.share_prefix_rows([layer])
    assert len(calls) == 2
    assert layer.k is k and layer.v is v
