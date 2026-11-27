# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Qwen-Image-2.1 mixed prompt lengths: per-row RoPE, neighbour invariance, grouping. CPU fp32, tiny model."""

from __future__ import annotations

import math

import pytest

from core.inference import diffusion_qwenimage21 as q
from core.inference import diffusion_qwenimage21_rope as rope

torch = pytest.importorskip("torch")
qmod = pytest.importorskip("diffusers.models.transformers.transformer_qwenimage21")


@pytest.fixture(autouse = True)
def _stock_after(monkeypatch):
    for env in (
        q.FAST_STEP_ENV,
        "UNSLOTH_DIFFUSION_Q21_ROW_LAYOUT",
        "UNSLOTH_DIFFUSION_Q21_MIXED_BATCH",
        rope.REAL_ROPE_ENV,
    ):
        monkeypatch.delenv(env, raising = False)
    q.uninstall()
    rope.uninstall()
    yield
    q.uninstall()
    rope.uninstall()


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


def _texts(*lengths, seed = 3):
    g = torch.Generator().manual_seed(seed)
    return [torch.randn(n, 24, generator = g) for n in lengths]


def _latents(
    batch,
    seed = 11,
    tokens = 24,
):
    return torch.randn(batch, tokens, 8, generator = torch.Generator().manual_seed(seed))


def _inputs(
    texts,
    latents,
    cond = None,
    longest = None,
):
    """Rows laid out as QwenImage21Pipeline does: right-padded, pad masked, condition slots ahead of text."""
    longest = longest or max(t.shape[0] for t in texts)
    batch = len(texts)
    lead = 3 if cond is not None else 0
    slots = 0 if cond is None else cond[0] * cond[1] // 4
    width = lead + slots + longest
    mask = torch.zeros(batch, width, dtype = torch.long)
    ehs = torch.zeros(batch, width, 24)
    vlm = torch.zeros(batch, width, dtype = torch.bool)
    g = torch.Generator().manual_seed(5)
    prefix = torch.randn(lead + slots, 24, generator = g)
    for row, t in enumerate(texts):
        n = lead + slots + t.shape[0]
        mask[row, :n] = 1
        ehs[row, : lead + slots] = prefix
        ehs[row, lead + slots : n] = t
        vlm[row, lead : lead + slots] = True
    shapes = [] if cond is None else [(1, *cond)]
    shapes.append((1, 4, 6))
    cond_latents = None
    if cond is not None:
        cond_latents = torch.randn(1, cond[0] * cond[1], 8, generator = g).expand(batch, -1, -1)
    return {
        "encoder_hidden_states": ehs,
        "encoder_hidden_states_mask": None if bool(mask.all()) else mask,
        "img_shapes": [shapes] * batch,
        "img_mask": torch.cat([vlm, torch.ones(batch, 6, dtype = torch.bool)], 1),
        "cond": cond_latents,
        "latents": latents,
    }


def _render(
    m,
    inp,
    steps = 3,
    cache = True,
    calls = None,
):
    kv = qmod.QwenImage21KVCache(len(m.transformer_blocks)) if cache else None
    lat = inp["latents"]
    with torch.inference_mode():
        for i in range(steps):
            hs = lat if inp["cond"] is None else torch.cat([inp["cond"], lat], 1)
            out = m(
                hidden_states = hs,
                timestep = torch.full((hs.shape[0],), 1.0 - i / steps),
                encoder_hidden_states = inp["encoder_hidden_states"],
                encoder_hidden_states_mask = inp["encoder_hidden_states_mask"],
                img_shapes = inp["img_shapes"],
                img_mask = inp["img_mask"],
                kv_cache = kv,
                kv_cache_mode = None if not cache else ("extract" if i == 0 else "cached"),
                return_dict = False,
            )[0]
            if calls is not None:
                calls.append(hs.shape[0])
            lat = lat - 0.1 * out[:, -lat.shape[1] :]
    return lat


def _psnr(a, b):
    mse = float(((a - b) ** 2).mean())
    return float("inf") if mse == 0 else 10 * math.log10(float(b.abs().max()) ** 2 / mse)


@pytest.mark.parametrize("cache", [True, False])
@pytest.mark.parametrize("cond", [None, (4, 4)])
def test_stock_lays_a_short_row_out_like_the_long_one_and_per_row_layout_fixes_it(cache, cond):
    """Stock gives the short row row 0's layout; per-row RoPE matches each lone render (fp32 rounding)."""
    m = _model()
    short, short2, long = _texts(5, 5, 13)
    lat = _latents(2)

    def arms():
        alone = _render(m, _inputs([short], lat[:1], cond), cache = cache)[0]
        equal = _render(m, _inputs([short, short2], lat, cond), cache = cache)[0]
        mixed = _render(m, _inputs([long, short], lat.flip(0), cond), cache = cache)[1]
        return alone, equal, mixed

    alone, equal, mixed = arms()
    assert torch.equal(equal, alone)
    assert _psnr(mixed, alone) < 70
    assert q.install() and q.row_layout_supported(m)
    alone, equal, mixed = arms()
    assert torch.equal(equal, alone)
    assert _psnr(mixed, alone) > 110
    assert torch.allclose(mixed, alone, rtol = 0, atol = 1e-5)


@pytest.mark.parametrize("cache", [True, False])
@pytest.mark.parametrize("cond", [None, (4, 4)])
@pytest.mark.parametrize("position", [0, 1, 2])
def test_a_row_is_bit_identical_whatever_its_neighbours_hold(cache, cond, position):
    """G2a at a fixed batch shape: change the neighbours' prompts (incl. token length), noise and magnitude."""
    assert q.install()
    m = _model()
    (target,) = _texts(6, seed = 21)
    longest = 13
    base = _texts(13, 9, seed = 22)
    other = [t * 40 for t in _texts(11, 13, seed = 23)]

    def run(neighbours, lat_seed):
        texts = list(neighbours)
        texts.insert(position, target)
        lat = _latents(3, seed = lat_seed)
        lat[position] = _latents(1, seed = 99)[0]
        return _render(m, _inputs(texts, lat, cond, longest = longest), cache = cache)[position]

    first = run(base, 1)
    assert torch.equal(first, run(base, 1))
    assert torch.equal(first, run(other, 2))


def test_per_row_tables_are_each_rows_solo_table():
    assert q.install()
    m = _model()
    short, long = _texts(5, 13)
    inp = _inputs([long, short], _latents(2), (4, 4))
    lay = q._layout_for(m, qmod, inp["img_mask"], inp["img_shapes"], torch.device("cpu"))
    rows = q._rows_for(
        m,
        lay,
        inp["img_mask"],
        inp["encoder_hidden_states_mask"],
        inp["img_shapes"][0],
        torch.device("cpu"),
    )
    assert rows is not None and rows.rotary_emb.shape == (2, lay.total, lay.rotary_emb.shape[1])
    assert torch.equal(rows.rotary_emb[0], lay.rotary_emb)
    solo = _inputs([short], _latents(1), (4, 4))
    solo_lay = q._layout_for(m, qmod, solo["img_mask"], solo["img_shapes"], torch.device("cpu"))
    valid = torch.ones(lay.total, dtype = torch.bool)
    valid[lay.text_positions] = inp["encoder_hidden_states_mask"][1].bool()[
        q._vlm_text(lay, inp["img_mask"].shape[1] - 6)
    ]
    assert torch.equal(rows.rotary_emb[1][valid], solo_lay.rotary_emb)
    assert torch.equal(rows.tails, rows.rotary_emb[:, lay.prefix_len :])
    uniform = _inputs([short, short], _latents(2), (4, 4))
    uniform["encoder_hidden_states_mask"] = torch.ones(
        2, uniform["img_mask"].shape[1] - 6, dtype = torch.long
    )
    assert (
        q._rows_for(
            m,
            lay,
            inp["img_mask"],
            uniform["encoder_hidden_states_mask"],
            inp["img_shapes"][0],
            torch.device("cpu"),
        )
        is None
    )


def test_rows_rope_is_the_stock_multiply_row_by_row():
    g = torch.Generator().manual_seed(0)
    x = torch.randn(3, 10, 2, 8, generator = g)
    freqs = torch.polar(torch.ones(3, 10, 4), torch.randn(3, 10, 4, generator = g))
    got = rope.rows_rope(x, freqs)
    for row in range(3):
        want = qmod.apply_rotary_emb_qwen(x[row : row + 1], freqs[row], use_real = False)
        assert torch.equal(got[row : row + 1], want)


def test_one_forward_per_step_for_every_row():
    """G3 at the forward: three prompt lengths, one denoiser call per step over all rows."""
    assert q.install()
    m = _model()
    calls: list = []
    _render(m, _inputs(_texts(4, 9, 13), _latents(3)), steps = 3, calls = calls)
    assert calls == [3, 3, 3]


def test_kill_switch_keeps_the_row_0_layout(monkeypatch):
    assert q.install()
    m = _model()
    short, long = _texts(5, 13)
    lat = _latents(2)
    fixed = _render(m, _inputs([long, short], lat))[1]
    monkeypatch.setenv(q.ROW_LAYOUT_ENV, "0")
    assert not q.row_layout_supported(m)
    q.uninstall()
    stock = _render(m, _inputs([long, short], lat))[1]
    assert q.install()
    assert torch.equal(_render(m, _inputs([long, short], lat))[1], stock)
    assert not torch.equal(fixed, stock)


def test_uniform_batches_stay_bit_identical_to_the_stock_forward():
    """Row layout only engages when rows pad differently: N=1, one prompt n times and equal lengths are unchanged."""
    m = _model()
    cases = [
        _inputs(_texts(7), _latents(1)),
        _inputs(_texts(7, 7), _latents(2)),
        _inputs(_texts(5), _latents(1), (4, 4)),
    ]
    want = [_render(m, inp) for inp in cases]
    assert q.install() and rope.rows_ready()
    for inp, ref in zip(cases, want):
        assert torch.equal(_render(m, inp), ref)


def test_graph_plan_replays_the_per_row_decode_bit_identically():
    """A decode step reuses the prefill's per-row tails; an unseen mask runs eager."""
    assert q.install()
    m = _model()
    inp = _inputs(_texts(5, 13), _latents(2))
    kv = qmod.QwenImage21KVCache(len(m.transformer_blocks))
    kwargs = {
        "hidden_states": inp["latents"],
        "timestep": torch.full((2,), 0.5),
        "encoder_hidden_states": inp["encoder_hidden_states"],
        "encoder_hidden_states_mask": inp["encoder_hidden_states_mask"],
        "img_shapes": inp["img_shapes"],
        "img_mask": inp["img_mask"],
        "kv_cache": kv,
        "return_dict": False,
    }
    with torch.inference_mode():
        m(**kwargs, kv_cache_mode = "extract")
        cached = {**kwargs, "kv_cache_mode": "cached"}
        plan = q.graph_plan(m, (), cached)
        assert plan is not None and plan[1]["rotary_emb"].dim() == 3
        want = m(**cached)[0]
        call, planned = plan
        assert torch.equal(call(**planned)[0], want)
        unseen = {**cached, "encoder_hidden_states_mask": inp["encoder_hidden_states_mask"].clone()}
        assert q.graph_plan(m, (), unseen) is None


def test_rows_wrapper_survives_a_real_rope_kill_switch(monkeypatch):
    stock = qmod.apply_rotary_emb_qwen
    assert q.install() and rope.rows_ready()
    wrapper = qmod.apply_rotary_emb_qwen
    assert wrapper is not stock and not rope.is_installed()
    monkeypatch.setenv(rope.REAL_ROPE_ENV, "0")
    assert rope.install() is False
    assert qmod.apply_rotary_emb_qwen is wrapper and rope.rows_ready() and not rope._ARMED[0]
    q.uninstall()
    assert qmod.apply_rotary_emb_qwen is stock


class QwenImage21Transformer2DModel:
    """Named like diffusers' class, without the fast step: stock layout, so rows split by prompt length."""


class _Pipe:
    prompt_template_t2i = "<sys> {} <end>"

    def __init__(self, transformer = None):
        self.transformer = (
            transformer if transformer is not None else QwenImage21Transformer2DModel()
        )

    @staticmethod
    def tokenizer(text):
        return {"input_ids": text.split()}


def test_layout_groups_merge_equal_lengths_across_the_whole_list():
    pipe = _Pipe()
    jobs = [
        ("red fox", 1),
        ("a long prompt here", 2),
        ("grey owl", 3),
        ("blue jay", 4),
        ("one more long one", 5),
    ]
    groups = q.layout_groups(pipe, jobs)
    assert groups == [[0, 2, 3], [1, 4]]
    order = [i for g in groups for i in g]
    assert sorted(order) == list(range(len(jobs)))
    rendered = [jobs[i] for i in order]
    assert q.text_length_runs(pipe, rendered) == [rendered[:3], rendered[3:]]
    restored = [None] * len(jobs)
    for k, i in enumerate(order):
        restored[i] = rendered[k]
    assert restored == jobs
    # The consecutive splitter keeps its old answer for the old caller.
    assert q.text_length_runs(pipe, jobs) == [[jobs[0]], [jobs[1]], [jobs[2], jobs[3]], [jobs[4]]]
    assert (
        q.layout_key(pipe, "red fox")
        == q.layout_key(pipe, "blue jay")
        != q.layout_key(pipe, "a long one")
    )


def test_unknown_lengths_group_with_nothing_and_one_prompt_is_one_group():
    pipe = _Pipe()
    pipe.tokenizer = None
    assert q.layout_groups(pipe, [("x y", 1), ("x z", 2)]) == [[0], [1]]
    assert q.layout_key(pipe, "x y") is None
    assert q.layout_groups(_Pipe(), [("same", 1), ("same", 2)]) == [[0, 1]]
    assert q.layout_groups(_Pipe(), []) == []
    assert q.layout_groups(object(), [("a", 1), ("b c d", 2)]) == [[0, 1]]


def test_mixed_lengths_share_one_call_when_streamed_or_forced(monkeypatch):
    """Resident (measured 0.75x on A100 int8) keeps the split by length; streamed (2.12x) or forced batches them."""
    from core.inference import diffusion_memory

    assert q.install()
    pipe = _Pipe(_model())
    jobs = [("red fox", 1), ("a long prompt here", 2), ("grey owl", 3)]
    assert q.row_layout_supported(pipe) and not q.mixed_lengths_batch(pipe)
    assert q.layout_groups(pipe, jobs) == [[0, 2], [1]]
    assert q.text_length_runs(pipe, jobs) == [[jobs[0]], [jobs[1]], [jobs[2]]]
    monkeypatch.setattr(diffusion_memory, "denoiser_residency", lambda p: (0, 32))
    assert q.mixed_lengths_batch(pipe)
    assert q.layout_groups(pipe, jobs) == [[0, 1, 2]]
    assert q.text_length_runs(pipe, jobs) == [jobs]
    assert q.layout_key(pipe, "red fox") == q.layout_key(pipe, "a long prompt here") == ()
    monkeypatch.setenv(q.MIXED_BATCH_ENV, "0")
    assert not q.mixed_lengths_batch(pipe)
    monkeypatch.setattr(diffusion_memory, "denoiser_residency", lambda p: (32, 32))
    monkeypatch.setenv(q.MIXED_BATCH_ENV, "1")
    assert q.text_length_runs(pipe, jobs) == [jobs]
    monkeypatch.setenv(q.ROW_LAYOUT_ENV, "0")
    assert not q.mixed_lengths_batch(pipe)


def test_compiled_blocks_batch_mixed_lengths_only_on_a_warm_graph(monkeypatch):
    """A user request never compiles the mixed signature; after an off-request warm, that size reuses the graph."""
    from torch._dynamo.utils import counters

    from core.inference import diffusion_memory

    assert q.install()
    torch._dynamo.reset()
    m = _model()
    for block in m.transformer_blocks:
        block.compile(backend = "eager")
    pipe = _Pipe(m)
    monkeypatch.setattr(diffusion_memory, "denoiser_residency", lambda p: (0, 32))
    jobs = [("red fox", 1), ("a long prompt here", 2), ("grey owl", 3)]
    try:
        _render(m, _inputs(_texts(13, 13, 13), _latents(3)))
        # streamed, but no mixed graph yet: split by length, and the coalescer keys by length
        assert not q.mixed_lengths_batch(pipe, 3) and not q.mixed_graph_warm(pipe, 3)
        assert q.text_length_runs(pipe, jobs) == [[jobs[0]], [jobs[1]], [jobs[2]]]
        assert q.layout_key(pipe, "red fox") != q.layout_key(pipe, "a long prompt here")
        with q.warming_mixed_lengths():
            assert q.mixed_lengths_batch(pipe, 3)
            _render(m, _inputs(_texts(4, 9, 13), _latents(3)))
        assert q.mixed_graph_warm(pipe, 3) and not q.mixed_graph_warm(pipe, 2)
        assert q.mixed_lengths_batch(pipe, 3) and q.text_length_runs(pipe, jobs) == [jobs]
        assert not q.mixed_lengths_batch(pipe, 2)
        graphs = counters["stats"]["unique_graphs"]
        calls: list = []
        _render(m, _inputs(_texts(13, 6, 2, seed = 9), _latents(3, seed = 4)), calls = calls)
        assert calls == [3, 3, 3]
        assert counters["stats"]["unique_graphs"] == graphs
    finally:
        torch._dynamo.reset()


def test_eager_blocks_need_no_warm(monkeypatch):
    from core.inference import diffusion_memory

    assert q.install()
    pipe = _Pipe(_model())
    monkeypatch.setattr(diffusion_memory, "denoiser_residency", lambda p: (0, 32))
    assert (
        q.mixed_graph_warm(pipe, 4)
        and q.mixed_lengths_batch(pipe, 4)
        and q.mixed_lengths_batch(pipe)
    )


def test_a_compile_below_the_offload_hooks_still_needs_a_warm(monkeypatch):
    """Streamed loads clear ``_compiled_call_impl``; Studio's compile mark still counts as compiled."""
    from core.inference import diffusion_memory

    assert q.install()
    m = _model()
    pipe = _Pipe(m)
    monkeypatch.setattr(diffusion_memory, "denoiser_residency", lambda p: (0, 32))
    assert q.mixed_lengths_batch(pipe, 3)
    m.__dict__["_unsloth_regional_compile_kwargs"] = {}
    assert not q.mixed_lengths_batch(pipe, 3) and not q.mixed_graph_warm(pipe, 3)
    with q.warming_mixed_lengths():
        _render(m, _inputs(_texts(4, 9, 13), _latents(3)))
    assert q.mixed_graph_warm(pipe, 3) and q.mixed_lengths_batch(pipe, 3)
