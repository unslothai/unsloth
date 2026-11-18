# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""FLUX.3 base conditioning blocks once per render (diffusion_cond_blocks). The parity tests run a tiny random
``Flux3Transformer2DModel`` on CPU (skipped without a FLUX.3 diffusers build); the rest use fakes."""

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from core.inference import diffusion_cond_blocks as cb  # noqa: E402


def _flux3():
    diffusers = pytest.importorskip("diffusers")
    cls = getattr(diffusers, "Flux3Transformer2DModel", None)
    if cls is None:
        pytest.skip("diffusers has no Flux3Transformer2DModel")
    if cb._forward_sha(cls) not in cb._FLUX3_FORWARD_SHA and cls not in cb._PATCHED:
        pytest.skip("diffusers Flux3 forward differs from the one the cache mirrors")
    return cls


def _tiny(cls, flow_map = False):
    torch.manual_seed(0)
    model = cls(
        in_channels = 8,
        cond_in_channels = 8,
        num_layers = 1,
        num_single_layers = 2,
        attention_head_dim = 16,
        num_attention_heads = 2,
        joint_attention_dim = 24,
        timestep_channels = 16,
        axes_dims_rope = (4, 4, 4, 4),
        flow_map = flow_map,
    )
    return model.eval()


def _inputs(
    refs: bool,
    txt_len: int = 5,
    seed: int = 1,
):
    g = torch.Generator().manual_seed(seed)
    img = 16
    kw = {
        "hidden_states": torch.randn(1, img, 8, generator = g),
        "encoder_hidden_states": torch.randn(1, txt_len, 24, generator = g),
        "timestep": torch.tensor([0.7]),
        "img_ids": torch.randint(0, 8, (img, 4), generator = g).float(),
        "txt_ids": torch.randint(0, 8, (txt_len, 4), generator = g).float(),
        "return_dict": False,
    }
    if refs:
        kw["image_cond_hidden_states"] = torch.randn(1, 12, 8, generator = g)
        kw["image_cond_ids"] = torch.randint(0, 8, (12, 4), generator = g).float()
    return kw


def _steps(
    model,
    branches,
    steps = 3,
):
    outs = []
    for i in range(steps):
        for kw in branches:
            kw = dict(kw, timestep = torch.tensor([0.9 - 0.2 * i]))
            outs.append(model(**kw)[0])
    return outs


@pytest.mark.parametrize("refs", [False, True])
def test_cached_steps_match_stock_bitwise(refs, monkeypatch):
    cls = _flux3()
    monkeypatch.delenv(cb.COND_BLOCKS_ENV, raising = False)
    monkeypatch.setenv(cb.COMPACT_MOD_ENV, "0")
    model = _tiny(cls)
    # text-only renders keep the stock forward by default; the flag exercises the cached text path anyway
    model.__dict__["_unsloth_cond_text_only"] = not refs
    cond = _inputs(refs, seed = 1)
    uncond = dict(cond, encoder_hidden_states = torch.randn(1, 7, 24), txt_ids = torch.zeros(7, 4))
    with torch.no_grad():
        stock = _steps(model, [cond, uncond])
        assert cb.prepare(model)
        assert cb.begin_render(model)
        try:
            cached = _steps(model, [cond, uncond])
        finally:
            cb.end_render(model)
    for a, b in zip(stock, cached):
        assert torch.equal(a, b)
    stats = cb.status(model)
    # one miss per branch, the other steps reuse it
    assert (stats["misses"], stats["hits"]) == (2, 4)
    assert not model.__dict__["_unsloth_cond_cache"]


@pytest.mark.parametrize("refs", [False, True])
@pytest.mark.parametrize("armed", [False, True])
@pytest.mark.parametrize("fullgraph", [False, True])
def test_whole_transformer_compile_traces_once(refs, armed, fullgraph, monkeypatch):
    """Studio compiles blocks, but a caller may compile the whole denoiser: no counter may become a guard."""
    from torch._dynamo.testing import CompileCounter

    cls = _flux3()
    monkeypatch.delenv(cb.COND_BLOCKS_ENV, raising = False)
    monkeypatch.delenv(cb.COMPACT_MOD_ENV, raising = False)
    model = _tiny(cls)
    model.__dict__["_unsloth_cond_text_only"] = armed
    kw = _inputs(refs)
    with torch.no_grad():
        want = _tiny(cls)(**kw)[0]
        assert cb.prepare(model)
        if armed:
            assert cb.begin_render(model)
        model(
            **kw
        )  # an eager render first, as before Studio's deferred compile: the counters are already set
        torch._dynamo.reset()
        cc = CompileCounter()
        fn = torch.compile(model, backend = cc, fullgraph = fullgraph)
        try:
            outs = [fn(**kw)[0] for _ in range(12)]
        finally:
            if armed:
                cb.end_render(model)
            torch._dynamo.reset()
    assert cc.frame_count == 1
    for out in outs:
        torch.testing.assert_close(out, want)


def test_new_conditioning_tensor_recomputes(monkeypatch):
    cls = _flux3()
    monkeypatch.delenv(cb.COND_BLOCKS_ENV, raising = False)
    model = _tiny(cls)
    kw = _inputs(True)
    with torch.no_grad():
        assert cb.prepare(model) and cb.begin_render(model)
        first = model(**kw)[0]
        # an equal-shaped but different prompt in a new tensor must not hit the old entry
        other = dict(kw, encoder_hidden_states = kw["encoder_hidden_states"] + 1)
        got = model(**other)[0]
        cb.end_render(model)
        want = model(**other)[0]
    assert torch.equal(got, want) and not torch.equal(first, got)
    assert cb.status(model)["misses"] == 2


def test_in_place_edit_of_cached_source_is_not_a_hit(monkeypatch):
    """Same object and data_ptr still hits within a render; a fresh render after end_render recomputes."""
    cls = _flux3()
    monkeypatch.delenv(cb.COND_BLOCKS_ENV, raising = False)
    model = _tiny(cls)
    kw = _inputs(True)
    with torch.no_grad():
        cb.prepare(model)
        cb.begin_render(model)
        model(**kw)
        cb.end_render(model)
        kw["encoder_hidden_states"].mul_(2)
        want = model(**kw)[0]
        cb.begin_render(model)
        got = model(**kw)[0]
        cb.end_render(model)
    assert torch.equal(got, want)


def test_text_only_render_keeps_stock(monkeypatch):
    cls = _flux3()
    monkeypatch.delenv(cb.COND_BLOCKS_ENV, raising = False)
    monkeypatch.setenv(cb.COMPACT_MOD_ENV, "0")
    model = _tiny(cls)
    kw = _inputs(False)
    with torch.no_grad():
        want = model(**kw)[0]
        assert cb.prepare(model) and cb.begin_render(model)
        got = model(**kw)[0]
        cb.end_render(model)
    assert torch.equal(got, want)
    stats = cb.status(model)
    assert stats["hits"] == stats["misses"] == 0


def test_kill_switch_and_grad_and_lora_scale_keep_stock(monkeypatch):
    cls = _flux3()
    model = _tiny(cls)
    kw = _inputs(True)
    cb.prepare(model)
    monkeypatch.setenv(cb.COND_BLOCKS_ENV, "0")
    assert not cb.begin_render(model)
    with torch.no_grad():
        model(**kw)
    monkeypatch.delenv(cb.COND_BLOCKS_ENV)
    assert cb.begin_render(model)
    model(**kw)  # grad enabled: training path, never cached
    with torch.no_grad():
        model(**dict(kw, attention_kwargs = {"scale": 0.5}))
    stats = cb.status(model)
    assert stats["hits"] == stats["misses"] == 0
    assert stats["skipped"] == 1 and "LoRA" in stats["reason"]
    cb.end_render(model)


def test_flow_map_is_never_armed(monkeypatch):
    cls = _flux3()
    monkeypatch.delenv(cb.COND_BLOCKS_ENV, raising = False)
    model = _tiny(cls, flow_map = True)
    assert not cb.prepare(model)
    assert not cb.begin_render(model)
    assert "flow map" in cb.status(model)["reason"]


def test_offloaded_denoiser_declines_with_the_kill_switch(monkeypatch):
    cls = _flux3()
    monkeypatch.delenv(cb.COND_BLOCKS_ENV, raising = False)
    monkeypatch.setenv(cb.OFFLOAD_COND_ENV, "0")
    model = _tiny(cls)
    cb.prepare(model)
    model.single_transformer_blocks[0]._hf_hook = object()
    try:
        assert not cb.begin_render(model)
        assert "accelerate hook" in cb.status(model)["reason"]
    finally:
        del model.single_transformer_blocks[0]._hf_hook


def _group_offloaded(cls):
    from diffusers.hooks import apply_group_offloading

    model = _tiny(cls)
    cpu = torch.device("cpu")
    apply_group_offloading(
        model,
        onload_device = cpu,
        offload_device = cpu,
        offload_type = "block_level",
        num_blocks_per_group = 1,
    )
    return model


def _count_calls(blocks):
    calls = [0]
    for block in blocks:
        block.register_forward_pre_hook(lambda *_: calls.__setitem__(0, calls[0] + 1))
    return calls


def test_offloaded_conditioning_runs_once_inside_the_forward(monkeypatch):
    """Group offloaded: computed once per branch inside the forward, bitwise equal to the uncached forward."""
    cls = _flux3()
    monkeypatch.delenv(cb.COND_BLOCKS_ENV, raising = False)
    monkeypatch.delenv(cb.OFFLOAD_COND_ENV, raising = False)
    monkeypatch.setattr(
        cb, "_offload_device_ok", lambda t: True
    )  # the CUDA gate, opened for this CPU proxy
    model = _group_offloaded(cls)
    cond_calls = _count_calls(list(model.txt_mode_blocks) + list(model.mode_blocks["image_cond"]))
    image_calls = _count_calls(model.mode_blocks["image"])
    cond = _inputs(True, seed = 1)
    uncond = dict(cond, encoder_hidden_states = torch.randn(1, 7, 24), txt_ids = torch.zeros(7, 4))
    with torch.no_grad():
        assert cb.prepare(model)
        want = _steps(model, [cond, uncond])
        assert cond_calls[0] == 3 * 2 * 2
        cond_calls[0] = 0
        assert cb.begin_render(model)
        try:
            got = _steps(model, [cond, uncond])
        finally:
            cb.end_render(model)
    assert all(torch.equal(a, b) for a, b in zip(want, got))
    # 2 blocks x 2 branches once per render; image blocks every step
    assert cond_calls[0] == 2 * 2 and image_calls[0] == 2 * 3 * 2
    stats = cb.status(model)
    assert (stats["misses"], stats["hits"], stats["in_forward"]) == (2, 4, 2)
    assert not model.__dict__["_unsloth_cond_cache"]


def test_offloaded_cache_is_cuda_only(monkeypatch):
    cls = _flux3()
    monkeypatch.delenv(cb.COND_BLOCKS_ENV, raising = False)
    monkeypatch.delenv(cb.OFFLOAD_COND_ENV, raising = False)
    model = _group_offloaded(cls)
    kw = _inputs(True)
    with torch.no_grad():
        cb.prepare(model)
        want = model(**kw)[0]
        assert cb.begin_render(model)
        got = model(**kw)[0]
        cb.end_render(model)
    assert torch.equal(got, want)
    stats = cb.status(model)
    assert stats["hits"] == stats["misses"] == 0 and "CUDA only" in stats["reason"]


def test_offloaded_cache_declines_a_fixed_diffusers_prefetch_chain(monkeypatch):
    cls = _flux3()
    monkeypatch.delenv(cb.COND_BLOCKS_ENV, raising = False)
    model = _group_offloaded(cls)
    cb.prepare(model)
    assert cb.begin_render(model)
    cb.end_render(model)
    hooks = model.single_transformer_blocks[0]._diffusers_hook.hooks
    hook = next(iter(hooks.values()))
    hook.next_group = object()
    try:
        assert not cb.begin_render(model)
        assert "prefetch chain" in cb.status(model)["reason"]
    finally:
        hook.next_group = None


def test_offloaded_miss_is_claimed_only_by_its_own_call(monkeypatch):
    """The miss travels as ``Pending``: a call with other sources (a merged CFG batch) computes stock, caches nothing."""
    cls = _flux3()
    monkeypatch.delenv(cb.COND_BLOCKS_ENV, raising = False)
    monkeypatch.setattr(cb, "_offload_device_ok", lambda t: True)
    model = _group_offloaded(cls)
    kw = _inputs(True)
    with torch.no_grad():
        cb.prepare(model)
        want = model(**kw)[0]
        assert cb.begin_render(model)
        _, hooked = cb._pre_hook(model, (), dict(kw))
        pending = hooked[cb.COND_KWARG]
        assert isinstance(pending, cb.Pending)
        other = dict(kw, encoder_hidden_states = kw["encoder_hidden_states"].clone())
        got = model(**other, unsloth_cond = pending)[0]
        cb.end_render(model)
    assert torch.equal(got, want)
    stats = cb.status(model)
    assert stats["misses"] == 0 and stats.get("in_forward", 0) == 0


def test_a_pending_miss_is_never_graphed_or_a_cfg_template():
    from core.inference import diffusion_cfg_batch as cfg
    from core.inference import diffusion_cuda_graph as cg

    pending = cb.Pending(("k", 1), (None,) * 4)
    assert cg._uncapturable(cg.graph_key({cb.COND_KWARG: pending}))
    router = SimpleNamespace(template = ("stale",), b0 = None)
    cfg.CfgBatchRouter._record_template(router, (), {cb.COND_KWARG: pending}, None)
    assert router.template is None


def test_prefetch_order_follows_skipped_blocks():
    from core.inference import diffusion_offload_prefetch as op

    blocks = [torch.nn.Linear(1, 1) for _ in range(4)]
    groups = [SimpleNamespace(modules = [b]) for b in blocks] + [
        SimpleNamespace(modules = [torch.nn.Linear(1, 1)])
    ]
    ids = [id(g) for g in groups]
    pf = SimpleNamespace(
        active = True, seen = [], pos = 0, order = list(ids), on_order = True, by_id = dict(zip(ids, groups))
    )
    module = SimpleNamespace(**{op.PREFETCHER_ATTR: pf})
    op.match_skipped_order(module, blocks[:2], skipping = True)
    assert pf.order == ids[2:] and pf.on_order
    op.match_skipped_order(module, blocks[:2], skipping = False)
    assert pf.order == ids
    pf.seen = [ids[0]]
    op.match_skipped_order(module, blocks[:2], skipping = True)
    assert pf.order == ids


def _mod_rows(model):
    rows = []
    heads = [m.linear for m in model.early_stream_modulations.values()]
    heads += [m.linear for m in model.single_stream_modulations.values()] + [model.norm_out.linear]
    for head in heads:
        head.register_forward_pre_hook(lambda m, a: rows.append(int(a[0].shape[1])))
    return rows


@pytest.mark.parametrize(
    "flow_map,refs", [(False, False), (False, True), (True, False), (True, True)]
)
def test_compact_modulation_matches_stock_in_float64(flow_map, refs, monkeypatch):
    """One modulation row per sample equals the stock per-token rows up to float64 rounding."""
    cls = _flux3()
    monkeypatch.delenv(cb.COMPACT_MOD_ENV, raising = False)
    model = _tiny(cls, flow_map = flow_map).double()
    kw = _inputs(refs)
    kw = {
        k: v.double() if torch.is_tensor(v) and k.endswith("states") else v for k, v in kw.items()
    }
    if flow_map:
        kw["target_timestep"] = torch.tensor([0.3])
    with torch.no_grad():
        want = model(**kw)[0]
        cb.prepare(model)
        rows = _mod_rows(model)
        got = model(**kw)[0]
    assert rows and set(rows) == {1}
    assert torch.allclose(got, want, rtol = 1e-12, atol = 1e-13)
    assert cb.status(model)["compact"] == 1


def test_compact_cached_render_matches_compact_stock_bitwise(monkeypatch):
    cls = _flux3()
    monkeypatch.delenv(cb.COND_BLOCKS_ENV, raising = False)
    monkeypatch.delenv(cb.COMPACT_MOD_ENV, raising = False)
    model = _tiny(cls)
    cond = _inputs(True, seed = 1)
    uncond = dict(cond, encoder_hidden_states = torch.randn(1, 7, 24), txt_ids = torch.zeros(7, 4))
    with torch.no_grad():
        assert cb.prepare(model)
        want = _steps(model, [cond, uncond])
        assert cb.begin_render(model)
        try:
            got = _steps(model, [cond, uncond])
        finally:
            cb.end_render(model)
    assert all(torch.equal(a, b) for a, b in zip(want, got))
    assert model.__dict__["_unsloth_cond_cache"] == {}


def test_compact_kill_switch_grad_and_lora_scale_keep_stock(monkeypatch):
    cls = _flux3()
    model = _tiny(cls, flow_map = True)
    kw = dict(_inputs(True), target_timestep = torch.tensor([0.3]))
    monkeypatch.setenv(cb.COMPACT_MOD_ENV, "0")
    cb.prepare(model)
    assert not model.__dict__.get(cb._COMPACT_ATTR)
    with torch.no_grad():
        model(**kw)
    monkeypatch.delenv(cb.COMPACT_MOD_ENV)
    cb.prepare(model)
    assert model.__dict__.get(cb._COMPACT_ATTR)
    model(**kw)
    with torch.no_grad():
        model(**dict(kw, attention_kwargs = {"scale": 0.5}))
    assert cb.status(model).get("compact", 0) == 0


def test_pre_hook_stays_last(monkeypatch):
    """A hook added after the cache must run first, so conditioning blocks run inside its scope."""
    cls = _flux3()
    monkeypatch.delenv(cb.COND_BLOCKS_ENV, raising = False)
    model = _tiny(cls)
    cb.prepare(model)
    assert cb.begin_render(model)
    cb.end_render(model)
    order = []
    model.register_forward_pre_hook(lambda m, a: order.append("other"))
    handle = model.__dict__["_unsloth_cond_hook"]
    real = model._forward_pre_hooks[handle.id]

    def spy(m, a, k):
        order.append("cond")
        return real(m, a, k)

    model._forward_pre_hooks[handle.id] = spy
    assert cb.begin_render(model)
    assert next(reversed(model._forward_pre_hooks)) == handle.id
    with torch.no_grad():
        model(**_inputs(False))
    cb.end_render(model)
    assert order == ["other", "cond"]


def test_changed_diffusers_forward_is_refused():
    class Flux3Transformer2DModel(torch.nn.Module):
        config = SimpleNamespace(flow_map = False)

        def forward(self, hidden_states):
            return hidden_states

    model = Flux3Transformer2DModel()
    assert not cb.prepare(model)
    assert "diffusers forward changed" in cb.status(model)["reason"]
    assert Flux3Transformer2DModel.forward is not cb._flux3_cached_forward
    assert not cb.begin_render(model)


def test_other_transformers_are_untouched(monkeypatch):
    model = torch.nn.Linear(2, 2)
    assert not cb.prepare(model)
    assert not cb.begin_render(model)
    cb.end_render(model)
    assert not model._forward_pre_hooks
    assert not cb.begin_render(SimpleNamespace())
    cb.end_render(None)


def test_switch_values(monkeypatch):
    for value, want in (("", True), ("1", True), ("0", False), ("off", False), ("False", False)):
        monkeypatch.setenv(cb.COND_BLOCKS_ENV, value)
        assert cb.enabled() is want


def _span_inputs(
    batch: int,
    refs: int,
    dtype,
    seed: int = 3,
):
    g = torch.Generator().manual_seed(seed)
    img, txt = 16, 5
    kw = {
        "hidden_states": torch.randn(batch, img, 8, generator = g).to(dtype),
        "encoder_hidden_states": torch.randn(batch, txt, 24, generator = g).to(dtype),
        "timestep": torch.full((batch,), 0.7),
        "img_ids": torch.randint(0, 8, (img, 4), generator = g).float(),
        "txt_ids": torch.randint(0, 8, (txt, 4), generator = g).float(),
        "return_dict": False,
    }
    if refs:
        n = 7 * refs  # odd span length, one span of 7 tokens per reference image
        kw["image_cond_hidden_states"] = torch.randn(batch, n, 8, generator = g).to(dtype)
        kw["image_cond_ids"] = torch.randint(0, 8, (n, 4), generator = g).float()
    return kw


def _joint_mod_shapes(model):
    seen = []

    def spy(m, args, kwargs):
        span = kwargs.get(cb.SPAN_KWARG)
        if span is not None:
            assert tuple(span.shape[1:2]) == (1,) and kwargs[cb.SPAN_LEN_KWARG].numel() == 0
            # a block graph copies every input into a static buffer of the same strides: none may overlap itself
            from core.inference.diffusion_block_graph import _static_like
            for t in (kwargs["temb_mod"], span, kwargs[cb.SPAN_LEN_KWARG]):
                _static_like(t).copy_(t)
        seen.append((tuple(kwargs["temb_mod"].shape), span is not None))

    handles = [
        b.register_forward_pre_hook(spy, with_kwargs = True) for b in model.single_transformer_blocks
    ]
    return seen, handles


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("refs", [0, 1, 8])
def test_span_modulation_matches_per_token_rows_bitwise(dtype, batch, refs, monkeypatch):
    """One modulation row per span (image, references) is bitwise the per-token rows, and no per-token copy reaches
    the joint blocks."""
    cls = _flux3()
    monkeypatch.delenv(cb.COMPACT_MOD_ENV, raising = False)
    monkeypatch.delenv(cb.SPAN_MOD_ENV, raising = False)
    model = _tiny(cls).to(dtype)
    kw = _span_inputs(batch, refs, dtype)
    with torch.no_grad():
        assert cb.prepare(model)
        assert model.__dict__[cb._SPAN_ATTR] is True
        model.__dict__[cb._SPAN_ATTR] = False
        want = model(**kw)[0]
        model.__dict__[cb._SPAN_ATTR] = True
        seen, handles = _joint_mod_shapes(model)
        try:
            got = model(**kw)[0]
        finally:
            for h in handles:
                h.remove()
    assert torch.equal(
        got.view(torch.int16 if dtype == torch.bfloat16 else torch.int32),
        want.view(torch.int16 if dtype == torch.bfloat16 else torch.int32),
    )
    if refs:
        assert all(shape[1] == 1 and span for shape, span in seen), seen
    else:
        assert not any(span for _, span in seen)


def test_span_modulation_kill_switch(monkeypatch):
    cls = _flux3()
    monkeypatch.delenv(cb.COMPACT_MOD_ENV, raising = False)
    monkeypatch.setenv(cb.SPAN_MOD_ENV, "0")
    model = _tiny(cls)
    cb.prepare(model)
    assert model.__dict__[cb._SPAN_ATTR] is False
    seen, handles = _joint_mod_shapes(model)
    with torch.no_grad():
        model(**_span_inputs(1, 1, torch.float32))
    for h in handles:
        h.remove()
    assert seen and all(shape[1] > 1 and not span for shape, span in seen)


def test_span_block_without_the_kwarg_is_stock():
    cls = _flux3()
    model = _tiny(cls)
    block = model.single_transformer_blocks[0]
    assert cb.patch_single_class(type(block)) is None
    g = torch.Generator().manual_seed(5)
    h, e = torch.randn(1, 9, 32, generator = g), torch.randn(1, 4, 32, generator = g)
    mod, cmod = torch.randn(1, 9, 96, generator = g), torch.randn(1, 1, 96, generator = g)
    rope = (torch.ones(13, 16), torch.zeros(13, 16))
    stock = cb._PATCHED_SINGLE[type(block)]
    with torch.no_grad():
        want = stock(block, h, e, mod, cmod, rope)
        got = block(h, e, mod, cmod, rope)
    assert all(torch.equal(a, b) for a, b in zip(want, got))


@pytest.mark.parametrize("refs", [1, 3])
def test_span_blocks_compile_without_breaks_and_match_compiled_per_token_rows(refs, monkeypatch):
    """Studio compiles each block: the span forward traces whole (no graph break), once across two reference counts
    under dynamic shapes, and matches the per-token rows compiled the same way."""
    from torch._dynamo.testing import CompileCounterWithBackend

    cls = _flux3()
    monkeypatch.delenv(cb.COMPACT_MOD_ENV, raising = False)
    monkeypatch.delenv(cb.SPAN_MOD_ENV, raising = False)
    model = _tiny(cls)
    assert cb.prepare(model)
    blocks = list(model.single_transformer_blocks)
    eager = [b.forward for b in blocks]
    outs = {}
    counters = {}
    for span in (False, True):
        torch._dynamo.reset()
        cc = CompileCounterWithBackend("inductor")
        for b in blocks:
            b.forward = torch.compile(
                type(b).forward.__get__(b), backend = cc, fullgraph = True, dynamic = True
            )
        model.__dict__[cb._SPAN_ATTR] = span
        try:
            with torch.no_grad():
                outs[span] = [
                    model(**_span_inputs(1, n, torch.float32))[0] for n in (refs, refs + 1)
                ]
        finally:
            for b, f in zip(blocks, eager):
                b.forward = f
        counters[span] = cc.frame_count
    torch._dynamo.reset()
    assert counters[True] == len(blocks)
    assert all(torch.equal(a, b) for a, b in zip(outs[True], outs[False]))
