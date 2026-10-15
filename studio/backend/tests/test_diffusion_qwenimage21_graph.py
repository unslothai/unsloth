# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Graph plan checked on CPU; CUDA renders must match the stock diffusers forward bit for bit."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from core.inference import diffusion_capture_safe as cs
from core.inference import diffusion_cuda_graph as cg
from core.inference import diffusion_qwenimage21 as q

torch = pytest.importorskip("torch")
qmod = pytest.importorskip("diffusers.models.transformers.transformer_qwenimage21")

# pytest puts the rootdir on sys.path, not this directory
_TESTS_DIR = str(Path(__file__).resolve().parent)
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

from test_diffusion_qwenimage21 import CASES, _inputs, _model  # noqa: E402

_CUDA = torch.cuda.is_available()


@pytest.fixture(autouse = True)
def _stock_after(monkeypatch):
    for env in (q.FAST_STEP_ENV, cs.CAPTURE_SAFE_ENV, cg.CUDA_GRAPH_DISABLE_ENV):
        monkeypatch.delenv(env, raising = False)
    monkeypatch.setenv(cg.SPEED_CHECK_ENV, "0")  # tiny forwards on a shared card: timing is noise
    q.uninstall()
    yield
    q.uninstall()


def _call_kwargs(inp, hs, timestep, kv, mode):
    return dict(
        hidden_states = hs,
        timestep = timestep,
        encoder_hidden_states = inp["encoder_hidden_states"],
        encoder_hidden_states_mask = inp["encoder_hidden_states_mask"],
        img_shapes = inp["img_shapes"],
        img_mask = inp["img_mask"],
        attention_kwargs = None,
        kv_cache = kv,
        kv_cache_mode = mode,
        return_dict = False,
    )


def _render(
    m,
    inp,
    steps = 5,
    call = None,
):
    """The pipeline's loop: step 0 prefills the cache, later steps decode from it."""
    call = call or m
    outs = []
    kv = qmod.QwenImage21KVCache(len(m.transformer_blocks))
    lat = inp["latents"]
    with torch.inference_mode():
        for i in range(steps):
            hs = lat if inp["cond"] is None else torch.cat([inp["cond"], lat], 1)
            timestep = torch.full((hs.shape[0],), 1.0 - i / steps, device = hs.device)
            out = call(**_call_kwargs(inp, hs, timestep, kv, "extract" if i == 0 else "cached"))[0]
            outs.append(out.clone())
            lat = lat - 0.1 * out[:, -lat.shape[1] :]
    return outs


def _prefilled(m, inp):
    kv = qmod.QwenImage21KVCache(len(m.transformer_blocks))
    hs = inp["latents"] if inp["cond"] is None else torch.cat([inp["cond"], inp["latents"]], 1)
    t = torch.full((hs.shape[0],), 0.9, device = hs.device)
    with torch.inference_mode():
        m(**_call_kwargs(inp, hs, t, kv, "extract"))
    return kv, hs


@pytest.mark.parametrize("case", CASES)
def test_planned_decode_step_is_bit_identical_to_the_eager_fast_step(case):
    m = _model()
    inp = _inputs(**case)
    assert q.install()
    kv, hs = _prefilled(m, inp)
    t = torch.full((hs.shape[0],), 0.5)
    with torch.inference_mode():
        kwargs = _call_kwargs(inp, hs, t, kv, "cached")
        want = m(**kwargs)[0]
        plan = q.graph_plan(m, (), kwargs)
        assert plan is not None
        call, planned = plan
        got = call(**planned)[0]
    assert torch.equal(want, got)
    key = cg.graph_key(((), planned))
    assert not cg._uncapturable(key)
    assert not cg._has_float(key)
    assert len(planned["kv"]) == len(m.transformer_blocks)


def test_what_the_plan_leaves_eager(monkeypatch):
    m = _model()
    inp = _inputs(batch = 2, pad = 2)
    assert q.install()
    kv, hs = _prefilled(m, inp)
    t = torch.full((2,), 0.5)
    with torch.inference_mode():
        cached = _call_kwargs(inp, hs, t, kv, "cached")
        assert q.graph_plan(m, (), cached) is not None
        # the prefill fills the cache from Python, so it stays eager
        assert q.graph_plan(m, (), {**cached, "kv_cache_mode": "extract"}) is None
        assert q.graph_plan(m, (), {**cached, "kv_cache": None, "kv_cache_mode": None}) is None
        assert q.graph_plan(m, (), {**cached, "return_dict": True}) is None
        assert q.graph_plan(m, (), {**cached, "attention_kwargs": {"scale": 0.5}}) is None
        assert (
            q.graph_plan(m, (hs,), {k: v for k, v in cached.items() if k != "hidden_states"})
            is None
        )
        assert q.graph_plan(m, (), {**cached, "unknown": 1}) is None
        empty = qmod.QwenImage21KVCache(len(m.transformer_blocks))
        assert q.graph_plan(m, (), {**cached, "kv_cache": empty}) is None
        assert q.graph_plan(m, (), {**cached, "img_mask": "not a tensor"}) is None
    with torch.enable_grad():
        assert q.graph_plan(m, (), cached) is None
    monkeypatch.setattr(q, "GRAPH_KV_MAX_BYTES", 16)
    with torch.inference_mode():
        assert q.graph_plan(m, (), cached) is None
    monkeypatch.setattr(q, "GRAPH_KV_MAX_BYTES", 1 << 30)
    monkeypatch.setenv(q.FAST_STEP_ENV, "0")
    with torch.inference_mode():
        assert q.graph_plan(m, (), cached) is None


def test_resolve_follows_the_installed_fast_step(monkeypatch):
    cls = qmod.QwenImage21Transformer2DModel
    forward, why = cs.resolve(cls)
    assert forward is None and "is not installed" in why
    assert q.install()
    forward, why = cs.resolve(cls)
    assert why is None
    assert forward is vars(cls)["forward"]
    assert forward.__unsloth_graph_plan__ is q.graph_plan
    monkeypatch.setenv(cs.CAPTURE_SAFE_ENV, "0")
    forward, why = cs.resolve(cls)
    assert forward is None and cs.CAPTURE_SAFE_ENV in why
    monkeypatch.delenv(cs.CAPTURE_SAFE_ENV)
    q.uninstall()
    assert cs.resolve(cls)[0] is None


def test_graph_eligible_accepts_qwen_image_21_once_the_fast_step_is_installed(monkeypatch):
    import types

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    def eligible():
        return cg.graph_eligible(
            types.SimpleNamespace(device = "cuda", backend = "cuda"),
            family = types.SimpleNamespace(),
            pipe = types.SimpleNamespace(transformer = _model()),
            offload_active = False,
            cache_active = False,
            speed_mode = "default",
        )

    ok, why = eligible()
    assert ok is False and why.startswith(
        "QwenImage21Transformer2DModel forward is not capture-safe"
    )
    assert q.install()
    assert eligible() == (True, "eligible")


def test_a_failed_capture_falls_back_to_the_callers_own_call():
    """A planned call whose capture fails (here: host tensors) must run the stock call, not the planned kwargs."""
    m = _model()
    inp = _inputs(batch = 2, pad = 3)
    ref = _render(m, inp)
    assert q.install()
    handle = cg.GraphedForward(m).enable()
    try:
        got = _render(m, _inputs(batch = 2, pad = 3))
    finally:
        cg.uninstall_all([handle])
    for a, b in zip(ref, got):
        assert torch.equal(a, b)
    assert handle.poisoned and handle.stats["refused_host_tensor"] == 1
    assert cg._POOL_BOX[0] is None


def _cuda_inputs(**case):
    return {k: (v.to("cuda") if torch.is_tensor(v) else v) for k, v in _inputs(**case).items()}


def _graphed(m, **kw):
    handle = cg.GraphedForward(m, **kw).enable()
    assert handle.capture_safe and handle.plan is q.graph_plan
    return handle


@pytest.mark.skipif(not _CUDA, reason = "needs CUDA")
@pytest.mark.parametrize("case", CASES)
def test_graphed_render_is_bit_identical_to_the_stock_forward(case):
    m = _model().to("cuda")
    inp = _cuda_inputs(**case)
    ref = _render(m, inp)
    assert q.install()
    handle = _graphed(m)
    try:
        got = _render(m, inp)
        again = _render(m, _cuda_inputs(**case))
    finally:
        cg.uninstall_all([handle])
    for a, b, c in zip(ref, got, again):
        assert torch.equal(a, b)
        assert torch.equal(a, c)
    assert not handle.poisoned, handle.capture_error
    # two prefills eager, one capture, every decode step replayed (the capturing one too)
    assert handle.stats["planned_eager"] == 2
    assert handle.stats["captures"] == 1
    assert handle.stats["replays"] == 2 * (5 - 1)
    assert not handle.poisoned, handle.capture_error


@pytest.mark.skipif(not _CUDA, reason = "needs CUDA")
def test_evicting_the_only_graph_does_not_reuse_its_dead_pool():
    m = _model().to("cuda")
    lengths = (9, 11, 13)
    refs = [_render(m, _cuda_inputs(text = n, seed = n)) for n in lengths]
    assert q.install()
    handle = _graphed(m, max_graphs = 1)
    try:
        for n, ref in zip(lengths, refs):
            for a, b in zip(ref, _render(m, _cuda_inputs(text = n, seed = n))):
                assert torch.equal(a, b)
    finally:
        cg.uninstall_all([handle])
    assert not handle.poisoned, handle.capture_error
    assert handle.stats["captures"] == 3 and handle.stats["evictions"] == 2


@pytest.mark.skipif(not _CUDA, reason = "needs CUDA")
def test_a_new_prompt_length_recaptures_and_the_oldest_graph_is_evicted():
    m = _model().to("cuda")
    lengths = (9, 11, 9, 13)
    refs = [_render(m, _cuda_inputs(text = n, seed = n)) for n in lengths]
    assert q.install()
    handle = _graphed(m, max_graphs = 2)
    try:
        for n, ref in zip(lengths, refs):
            got = _render(m, _cuda_inputs(text = n, seed = n))
            for a, b in zip(ref, got):
                assert torch.equal(a, b)
    finally:
        cg.uninstall_all([handle])
    # 9 and 11 capture, 9 replays, 13 evicts 11 (LRU); a new length warms up on its first decode step
    assert handle.stats["captures"] == 3
    assert handle.stats["shape_warmups"] == 2
    assert handle.stats["evictions"] == 1
    assert handle.stats["cap_skips"] == 0
    assert handle.stats["replays"] == 4 * 4 - 2
    assert not handle.poisoned


@pytest.mark.skipif(not _CUDA, reason = "needs CUDA")
def test_a_new_prompt_length_warms_on_tensors_laid_out_like_the_capture(monkeypatch):
    """Compiled blocks guard on strides and storage offsets. The warm-up step of a new length must run on fresh copies
    (as the capture's statics are), else it compiles a variant for the caller's views and the capture that follows
    compiles inside the recording, which fails it (seen on the real model under block offload)."""
    m = _model().to("cuda")
    assert q.install()
    handle = _graphed(m)
    stock = q._graph_step
    seen: list = []

    def spy(self, **kw):
        if handle.stats["shape_warmups"] and not seen:
            leaves: list = []
            cg._flatten(kw, leaves)
            seen.append(leaves)
        return stock(self, **kw)

    monkeypatch.setattr(q, "_graph_step", spy)
    try:
        for n in (9, 11):
            _render(m, _cuda_inputs(text = n, seed = n))
    finally:
        cg.uninstall_all([handle])
    assert handle.stats["shape_warmups"] == 1 and handle.stats["captures"] == 2
    assert seen and all(t.storage_offset() == 0 for t in seen[0])


@pytest.mark.skipif(not _CUDA, reason = "needs CUDA")
def test_alternating_caches_of_one_shape_replay_their_own_prefix():
    """True CFG: cond and uncond share a graph key but not a KV cache; the sticky copy must follow identity."""
    m = _model().to("cuda")
    cond, uncond = _cuda_inputs(seed = 1), _cuda_inputs(seed = 2)
    assert q.install()

    def two_stream_render():
        out = []
        kvs = [qmod.QwenImage21KVCache(len(m.transformer_blocks)) for _ in range(2)]
        lat = cond["latents"]
        with torch.inference_mode():
            for i in range(4):
                t = torch.full((1,), 1.0 - i / 4, device = "cuda")
                mode = "extract" if i == 0 else "cached"
                a = m(**_call_kwargs(cond, lat, t, kvs[0], mode))[0][:, -lat.shape[1] :]
                b = m(**_call_kwargs(uncond, lat, t, kvs[1], mode))[0][:, -lat.shape[1] :]
                out += [a.clone(), b.clone()]
                lat = lat - 0.1 * (a + b)
        return out

    ref = two_stream_render()
    handle = _graphed(m)
    try:
        got = two_stream_render()
    finally:
        cg.uninstall_all([handle])
    for a, b in zip(ref, got):
        assert torch.equal(a, b)
    assert handle.stats["captures"] == 1 and handle.stats["replays"] == 6


@pytest.mark.skipif(not _CUDA, reason = "needs CUDA")
def test_graph_statics_match_the_eager_inputs_strides_and_inference_mode():
    """A static that is not an inference tensor where eager passes one selects another compiled variant."""
    m = _model().to("cuda")
    inp = _cuda_inputs()
    assert q.install()
    handle = _graphed(m)
    try:
        _render(m, inp)
        (entry,) = handle.cache.values()
        kv, hs = _prefilled(m, inp)
        with torch.inference_mode():
            # decode latents come from the previous step, made in inference mode
            hs = hs.clone()
            t = torch.full((1,), 0.5, device = "cuda")
            _, planned = q.graph_plan(m, (), _call_kwargs(inp, hs, t, kv, "cached"))
        live: list = []
        cg._flatten(((), planned), live)
        assert len(live) == len(entry.static)
        for src, dst in zip(live, entry.static):
            assert dst.stride() == src.stride()
            assert dst.is_inference() == src.is_inference()
    finally:
        cg.uninstall_all([handle])


@pytest.mark.skipif(not _CUDA, reason = "needs CUDA")
def test_planned_decode_steps_do_not_sync_the_host():
    m = _model().to("cuda")
    inp = _cuda_inputs(batch = 2, pad = 2)
    assert q.install()
    handle = _graphed(m)
    kv = qmod.QwenImage21KVCache(len(m.transformer_blocks))
    try:
        with torch.inference_mode():
            for i in range(5):
                t = torch.full((2,), 1.0 - i / 5, device = "cuda")
                # step 1 warm-up and capture synchronize by design; later replays must not
                if i >= 2:
                    torch.cuda.set_sync_debug_mode("error")
                try:
                    m(**_call_kwargs(inp, inp["latents"], t, kv, "extract" if i == 0 else "cached"))
                finally:
                    torch.cuda.set_sync_debug_mode("default")
    finally:
        cg.uninstall_all([handle])
    assert handle.stats["replays"] == 4


def _block_streamed(m):
    import types

    from core.inference import diffusion_memory as dm

    pipe = types.SimpleNamespace(transformer = m, components = {"transformer": m})
    assert dm._apply_group_offload(pipe, "cuda", None)
    return pipe


@pytest.mark.skipif(not _CUDA, reason = "needs CUDA")
@pytest.mark.parametrize("case", [CASES[1], CASES[3]])
def test_block_streamed_decode_steps_replay_bit_identically(case):
    """Under block offload the planned step enters through the offload hooks (which onload the top-level weights)
    and the copy-stream onloads are recorded with it: renders match the eager streamed and the resident forward."""
    pytest.importorskip("diffusers.hooks")
    from core.inference import diffusion_offload_prefetch as op

    resident = _render(_model().to("cuda"), _cuda_inputs(**case))
    assert q.install()
    m = _model()
    pipe = _block_streamed(m)
    eager = _render(m, _cuda_inputs(**case))
    handles, reason = cg.arm_after_placement(pipe)
    assert len(handles) == 1 and handles[0].placement.mode == "group", reason
    handle = handles[0]
    assert handle.plan is q.graph_plan and handle.placed is q.placed_step
    try:
        got = _render(m, _cuda_inputs(**case))
        again = _render(m, _cuda_inputs(**case))
        kv = qmod.QwenImage21KVCache(len(m.transformer_blocks))
        inp = _cuda_inputs(**case)
        hs = inp["latents"] if inp["cond"] is None else torch.cat([inp["cond"], inp["latents"]], 1)
        with torch.inference_mode():
            for i in range(3):
                t = torch.full((hs.shape[0],), 1.0 - i / 5, device = "cuda")
                if i:
                    torch.cuda.set_sync_debug_mode("error")
                try:
                    m(**_call_kwargs(inp, hs, t, kv, "extract" if i == 0 else "cached"))
                finally:
                    torch.cuda.set_sync_debug_mode("default")
    finally:
        cg.uninstall_all(handles)
    assert not handle.poisoned, handle.capture_error
    for r, e, a, b in zip(resident, eager, got, again):
        assert torch.equal(e, a) and torch.equal(e, b)
        assert torch.equal(r, a)
    assert handle.stats["captures"] == 1
    assert handle.stats["replays"] == 2 * (5 - 1) + 2
    assert next(m.transformer_blocks[-1].parameters()).device.type == "cpu"
    assert op.module_prefetcher(m).stats["missed"] == 0


def test_offload_placement_names_a_forward_that_cannot_record_under_the_hooks(monkeypatch):
    m = _model()

    def planned(*_a, **_k):
        return None

    planned.__unsloth_graph_plan__ = q.graph_plan
    monkeypatch.setattr(cs, "resolve", lambda cls: (planned, None))
    assert "no entry through the offload hooks" in cg._forward_refusal(m, "group")
    planned.__unsloth_graph_placed__ = q.placed_step
    assert cg._forward_refusal(m, "group") is None

    def rewrite(*_a, **_k):
        return None

    monkeypatch.setattr(cs, "resolve", lambda cls: (rewrite, None))
    assert "capture-safe forward" in cg._forward_refusal(m, "group")
    assert cg._forward_refusal(m, "model") is None
    monkeypatch.setattr(cs, "resolve", lambda cls: (None, "X forward is not capture-safe (why)"))
    assert cg._forward_refusal(m, "model") == "X forward is not capture-safe (why)"
