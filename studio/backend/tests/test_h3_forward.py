# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""CPU tests for the mirrored MiniMax-H3 denoiser forward (text stream once per render, packed sequence in place)
on a tiny random diffusers H3 transformer."""

from __future__ import annotations

import inspect

import pytest

torch = pytest.importorskip("torch")
h3_mod = pytest.importorskip("diffusers.models.transformers.transformer_minimax_h3")

from core.inference import video_minimax_h3_forward as h3f  # noqa: E402

needs_mirror = pytest.mark.skipif(
    h3f._forward_sha(h3_mod.MiniMaxH3Transformer3DModel) not in h3f._KNOWN_FORWARD_SHA,
    reason = "installed diffusers' H3 forward is not the mirrored source",
)

T_ROWS, V_ROWS, A_ROWS = 5, 12, 4


def _model():
    torch.manual_seed(0)
    m = h3_mod.MiniMaxH3Transformer3DModel(
        num_attention_heads = 2,
        attention_head_dim = 16,
        hidden_size = 32,
        num_layers = 2,
        num_refiner_layers = 2,
        ffn_dim = 64,
        in_channels = 4,
        audio_in_channels = 6,
        patch_size = (1, 1, 1),
        text_dim = 24,
        freq_dim = 16,
        time_embed_hidden_dim = 32,
        time_embed_dim = 16,
        rope_freq_dim = 2,
    ).eval()
    with torch.no_grad():
        for p in m.parameters():
            p.copy_(torch.randn_like(p) * 0.1)
    return m


def _layout():
    s = T_ROWS + V_ROWS + A_ROWS
    perm = torch.randperm(s, generator = torch.Generator().manual_seed(1))
    text_indices = perm[:T_ROWS].sort().values
    video_indices = perm[T_ROWS : T_ROWS + V_ROWS].sort().values
    audio_indices = perm[T_ROWS + V_ROWS :].sort().values
    token_tags = torch.zeros(s, dtype = torch.long)
    token_tags[video_indices] = 1
    token_tags[audio_indices] = 2
    position_ids = torch.randint(0, 6, (s, 3), generator = torch.Generator().manual_seed(2))
    timestep_indices = torch.zeros(s, dtype = torch.long)
    timestep_indices[video_indices[:3]] = 1
    return dict(
        token_tags = token_tags,
        position_ids = position_ids,
        video_indices = video_indices,
        audio_indices = audio_indices,
        text_indices = text_indices,
        timestep_indices = timestep_indices,
    )


def _step_inputs(i, enc):
    g = torch.Generator().manual_seed(100 + i)
    return dict(
        hidden_states = torch.randn(1, V_ROWS, 4, generator = g),
        audio_hidden_states = torch.randn(1, A_ROWS, 6, generator = g),
        encoder_hidden_states = enc,
        timestep = torch.tensor([1000.0 - 100 * i, 990.0]),
        return_dict = False,
    )


def _run(
    model,
    enc,
    steps = 4,
    layout = None,
):
    layout = layout or _layout()
    outs = []
    with torch.no_grad():
        for i in range(steps):
            outs.append(model(**_step_inputs(i, enc), **layout))
    return outs


def _enc(seed = 3):
    return torch.randn(1, T_ROWS, 24, generator = torch.Generator().manual_seed(seed))


class _Pipe:
    def __init__(self, transformer):
        self.transformer = transformer


def _assert_same(a, b):
    assert len(a) == len(b)
    for (va, aa), (vb, ab) in zip(a, b):
        assert torch.equal(va, vb) and torch.equal(aa, ab)


@needs_mirror
def test_bit_identical_and_text_stream_runs_once_per_render(monkeypatch):
    monkeypatch.delenv(h3f.REFINER_CACHE_ENV, raising = False)
    monkeypatch.delenv(h3f.INPLACE_PACK_ENV, raising = False)
    stock = _run(_model(), _enc())
    m = _model()
    calls = {"refiner": 0, "embedder": 0}
    m.token_refiner.register_forward_hook(
        lambda *a: calls.__setitem__("refiner", calls["refiner"] + 1)
    )
    m.context_embedder.register_forward_hook(
        lambda *a: calls.__setitem__("embedder", calls["embedder"] + 1)
    )
    assert h3f.install_h3_forward(m) is True
    assert h3f.install_h3_forward(m) is True  # idempotent
    with h3f.h3_render_scope(_Pipe(m)):
        got = _run(m, _enc())
    _assert_same(stock, got)
    assert calls == {"refiner": 1, "embedder": 1}
    stats = m.__dict__[h3f.STATS_ATTR]
    assert stats["text_misses"] == 1 and stats["text_hits"] == 3 and stats["inplace_packs"] == 4
    # the scope is gone after the render: nothing holds the conditioning
    assert h3f._SCOPE_ATTR not in m.__dict__


@needs_mirror
def test_cache_only_inside_a_render_and_keyed_on_the_conditioning(monkeypatch):
    monkeypatch.delenv(h3f.REFINER_CACHE_ENV, raising = False)
    m = _model()
    h3f.install_h3_forward(m)
    stats = m.__dict__[h3f.STATS_ATTR]
    _run(m, _enc(), steps = 2)  # no render scope: recomputed each call
    assert stats["text_misses"] == 0 and stats["text_hits"] == 0
    stock = _model()
    layout = _layout()
    with h3f.h3_render_scope(_Pipe(m)):
        a = _run(m, _enc(3), steps = 2, layout = layout)
        b = _run(
            m, _enc(4), steps = 2, layout = layout
        )  # another caption tensor: a miss, not the old rows
    assert stats["text_misses"] == 2 and stats["text_hits"] == 2
    _assert_same(a, _run(stock, _enc(3), steps = 2, layout = layout))
    _assert_same(b, _run(stock, _enc(4), steps = 2, layout = layout))
    # an in-place edit of the same tensor (normal tensor: version counter) misses too
    enc = _enc(5)
    with h3f.h3_render_scope(_Pipe(m)):
        _run(m, enc, steps = 1, layout = layout)
        enc.mul_(2.0)
        got = _run(m, enc, steps = 1, layout = layout)
    _assert_same(got, _run(stock, enc, steps = 1, layout = layout))
    assert stats["text_misses"] == 4


@needs_mirror
def test_inference_mode_render(monkeypatch):
    monkeypatch.delenv(h3f.REFINER_CACHE_ENV, raising = False)
    m, stock = _model(), _model()
    h3f.install_h3_forward(m)
    layout = _layout()
    with torch.inference_mode():
        enc = _enc()
        ref = [stock(**_step_inputs(i, enc), **layout) for i in range(3)]
        with h3f.h3_render_scope(_Pipe(m)):
            got = [m(**_step_inputs(i, enc), **layout) for i in range(3)]
    _assert_same(ref, got)
    assert m.__dict__[h3f.STATS_ATTR]["text_hits"] == 2


@needs_mirror
@pytest.mark.parametrize("env", [h3f.REFINER_CACHE_ENV, h3f.INPLACE_PACK_ENV])
def test_kill_switches(monkeypatch, env):
    monkeypatch.setenv(env, "0")
    m = _model()
    assert h3f.install_h3_forward(m) is True
    with h3f.h3_render_scope(_Pipe(m)):
        got = _run(m, _enc())
    _assert_same(_run(_model(), _enc()), got)
    stats = m.__dict__[h3f.STATS_ATTR]
    if env == h3f.REFINER_CACHE_ENV:
        assert stats["text_misses"] == 0 and stats["text_hits"] == 0 and stats["inplace_packs"] == 4
    else:
        assert stats["text_hits"] == 3 and stats["inplace_packs"] == 0
    monkeypatch.setenv(h3f.REFINER_CACHE_ENV, "0")
    monkeypatch.setenv(h3f.INPLACE_PACK_ENV, "0")
    fresh = _model()
    assert h3f.install_h3_forward(fresh) is False
    assert "forward" not in fresh.__dict__


@needs_mirror
def test_no_allocating_full_sequence_copies(monkeypatch):
    from torch.utils._python_dispatch import TorchDispatchMode

    class Count(TorchDispatchMode):
        def __init__(self):
            super().__init__()
            self.ops = {}

        def __torch_dispatch__(
            self,
            func,
            types,
            args = (),
            kwargs = None,
        ):
            name = str(func.overloadpacket)
            self.ops[name] = self.ops.get(name, 0) + 1
            return func(*args, **(kwargs or {}))

    def count(model, scope):
        with Count() as c:
            if scope:
                with h3f.h3_render_scope(_Pipe(model)):
                    _run(model, _enc(), steps = 2)
            else:
                _run(model, _enc(), steps = 2)
        return c.ops

    monkeypatch.delenv(h3f.INPLACE_PACK_ENV, raising = False)
    stock = count(_model(), False)
    m = _model()
    h3f.install_h3_forward(m)
    patched = count(m, True)
    assert stock.get("aten.index_copy", 0) == 6 and stock.get("aten.index_copy_", 0) == 0
    assert patched.get("aten.index_copy", 0) == 0 and patched.get("aten.index_copy_", 0) == 6
    # the refiner's attention + MLP ran for one of the two steps only
    assert patched.get("aten.mm", 0) + patched.get("aten.addmm", 0) < stock.get(
        "aten.mm", 0
    ) + stock.get("aten.addmm", 0)


@needs_mirror
def test_training_and_grad_keep_stock(monkeypatch):
    m = _model()
    h3f.install_h3_forward(m)
    layout = _layout()
    with h3f.h3_render_scope(_Pipe(m)):
        v, a = m(**_step_inputs(0, _enc()), **layout)
    assert v.requires_grad
    stats = m.__dict__[h3f.STATS_ATTR]
    assert stats["stock"] == 1 and stats["text_misses"] == 0


@needs_mirror
def test_signature_and_hooks_wrap_the_mirrored_forward():
    from diffusers.hooks import HookRegistry, ModelHook

    m = _model()
    h3f.install_h3_forward(m)
    stock_params = list(inspect.signature(h3_mod.MiniMaxH3Transformer3DModel.forward).parameters)
    # the modular denoise step filters its layout kwargs by this signature
    assert list(inspect.signature(m.forward).parameters) == stock_params[1:]

    seen = []

    class Pre(ModelHook):
        def pre_forward(self, module, *args, **kwargs):
            seen.append(1)
            return args, kwargs

    HookRegistry.check_if_exists_or_initialize(m).register_hook(Pre(), "probe")
    with h3f.h3_render_scope(_Pipe(m)):
        _run(m, _enc(), steps = 2)
    assert seen == [1, 1] and m.__dict__[h3f.STATS_ATTR]["text_hits"] == 1

    hooked = _model()
    HookRegistry.check_if_exists_or_initialize(hooked).register_hook(Pre(), "probe")
    assert h3f.install_h3_forward(hooked) is False  # never under an existing hook


def test_unknown_source_and_other_classes_keep_stock(monkeypatch):
    class Other(torch.nn.Module):
        def forward(self, x):
            return x

    assert h3f.install_h3_forward(Other()) is False
    assert h3f.install_h3_forward(None) is False
    m = _model()
    monkeypatch.setattr(h3f, "_KNOWN_FORWARD_SHA", frozenset({"0" * 16}))
    assert h3f.install_h3_forward(m) is False
    assert "forward" not in m.__dict__


@needs_mirror
def test_regional_compile_adds_no_graphs(monkeypatch):
    from torch._dynamo.testing import CompileCounter

    def graphs(install):
        torch._dynamo.reset()
        m = _model()
        if install:
            h3f.install_h3_forward(m)
        cnt = CompileCounter()
        m.compile_repeated_blocks(backend = cnt, fullgraph = True, dynamic = False)
        with h3f.h3_render_scope(_Pipe(m)):
            _run(m, _enc(), steps = 4)
        return cnt.frame_count

    stock, patched = graphs(False), graphs(True)
    assert patched <= stock


def _two_renders(model, fn):
    outs = []
    layout = _layout()
    for seed in (3, 4):
        enc = _enc(seed)
        with torch.no_grad(), h3f.h3_render_scope(_Pipe(model)):
            for i in range(3):
                outs.append(fn(**_step_inputs(i, enc), **layout))
    return outs


@needs_mirror
def test_regional_compile_still_uses_the_text_cache(monkeypatch):
    monkeypatch.delenv(h3f.REFINER_CACHE_ENV, raising = False)
    monkeypatch.delenv(h3f.INPLACE_PACK_ENV, raising = False)
    stock = _two_renders(_model(), _model())
    torch._dynamo.reset()
    m = _model()
    h3f.install_h3_forward(m)
    m.compile_repeated_blocks(backend = "eager", fullgraph = True, dynamic = False)
    _assert_same(stock, _two_renders(m, m))
    stats = m.__dict__[h3f.STATS_ATTR]
    assert stats["text_misses"] == 2 and stats["text_hits"] == 4 and stats["inplace_packs"] == 6


@needs_mirror
@pytest.mark.parametrize("fullgraph", [True, False])
def test_whole_forward_compile_one_graph_across_renders(monkeypatch, fullgraph):
    from torch._dynamo.testing import CompileCounter
    from torch._dynamo.utils import counters

    monkeypatch.delenv(h3f.REFINER_CACHE_ENV, raising = False)
    monkeypatch.delenv(h3f.INPLACE_PACK_ENV, raising = False)
    stock = _two_renders(_model(), _model())
    torch._dynamo.reset()
    counters.clear()
    m = _model()
    assert h3f.install_h3_forward(m) is True
    cnt = CompileCounter()
    compiled = torch.compile(m, backend = cnt, fullgraph = fullgraph, dynamic = False)
    _assert_same(stock, _two_renders(m, compiled))
    assert cnt.frame_count == 1
    assert sum(counters["graph_break"].values()) == 0
    torch._dynamo.reset()


@needs_mirror
def test_lora_scale_is_part_of_the_key(monkeypatch):
    monkeypatch.delenv(h3f.REFINER_CACHE_ENV, raising = False)
    m = _model()
    h3f.install_h3_forward(m)
    layout = _layout()
    enc = _enc()
    with torch.no_grad(), h3f.h3_render_scope(_Pipe(m)):
        m(**_step_inputs(0, enc), **layout)
        m(**_step_inputs(1, enc), **layout, attention_kwargs = {"scale": 0.5})
        m(**_step_inputs(2, enc), **layout, attention_kwargs = {"scale": 0.5})
    stats = m.__dict__[h3f.STATS_ATTR]
    assert stats["text_misses"] == 2 and stats["text_hits"] == 1
