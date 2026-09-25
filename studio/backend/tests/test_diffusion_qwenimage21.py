# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Unit tests for ``diffusion_qwenimage21.py`` (the Qwen-Image-2.1 step without per-step host syncs).

CPU by default: a tiny random ``QwenImage21Transformer2DModel`` runs the pipeline's call pattern (one
``extract`` step, then ``cached`` steps) through the stock forward and the fast one, and every step
must match bit for bit. Skipped when the installed diffusers has no Qwen-Image 2.1.
"""

from __future__ import annotations

import importlib
import types

import pytest

from core.inference import diffusion_qwenimage21 as q

torch = pytest.importorskip("torch")
qmod = pytest.importorskip("diffusers.models.transformers.transformer_qwenimage21")


@pytest.fixture(autouse = True)
def _stock_after(monkeypatch):
    monkeypatch.delenv(q.FAST_STEP_ENV, raising = False)
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
    batch = 1,
    text = 9,
    cond = None,
    target = (4, 6),
    pad = 0,
    seed = 7,
):
    """Pipeline-shaped inputs: ``cond`` puts a condition image's slots in the middle of the text."""
    g = torch.Generator().manual_seed(seed)
    th, tw = target
    shapes = []
    latents = []
    if cond is None:
        vlm = torch.zeros(text, dtype = torch.bool)
    else:
        ch, cw = cond
        vlm = torch.cat(
            [
                torch.zeros(text // 2, dtype = torch.bool),
                torch.ones(ch * cw // 4, dtype = torch.bool),
                torch.zeros(text - text // 2, dtype = torch.bool),
            ]
        )
        shapes.append((1, ch, cw))
        latents.append(torch.randn(batch, ch * cw, 8, generator = g))
    shapes.append((1, th, tw))
    img_mask = torch.cat([vlm, torch.ones(th * tw // 4, dtype = torch.bool)])[None].repeat(batch, 1)
    mask = None
    if pad:
        mask = torch.ones(batch, vlm.numel(), dtype = torch.long)
        mask[-1, -pad:] = 0
    return {
        "encoder_hidden_states": torch.randn(batch, vlm.numel(), 24, generator = g),
        "encoder_hidden_states_mask": mask,
        "img_shapes": [shapes] * batch,
        "img_mask": img_mask,
        "cond": torch.cat(latents, 1) if latents else None,
        "latents": torch.randn(batch, th * tw, 8, generator = g),
    }


def _render(
    m,
    inp,
    steps = 4,
    cache = True,
):
    """The QwenImage21Pipeline denoise loop's transformer calls; returns every step's output."""
    outs = []
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
            outs.append(out.clone())
            lat = lat - 0.1 * out[:, -lat.shape[1] :]
    return outs


CASES = [
    {},
    {"batch": 2, "pad": 3},
    {"cond": (4, 4)},
    {"batch": 2, "cond": (4, 8), "target": (6, 4), "pad": 2},
]


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("cache", [True, False])
def test_fast_step_is_bit_identical_to_the_stock_forward(case, cache):
    m = _model()
    inp = _inputs(**case)
    ref = _render(m, inp, cache = cache)
    assert q.install()
    assert getattr(type(m).forward, "__unsloth_q21_fast_step__", False)
    first = _render(m, inp, cache = cache)
    # A second render with fresh mask objects finds the layout by content.
    again = _render(m, _inputs(**case), cache = cache)
    for a, b, c in zip(ref, first, again):
        assert torch.equal(a, b)
        assert torch.equal(a, c)


def test_layout_is_built_once_per_render_and_found_by_content():
    m = _model()
    assert q.install()
    calls = {"n": 0}
    real = q._build_layout

    def counting(*a, **k):
        calls["n"] += 1
        return real(*a, **k)

    q._build_layout = counting
    try:
        inp = _inputs()
        _render(m, inp, steps = 5)
        assert calls["n"] == 1
        _render(m, _inputs(), steps = 5)
        assert calls["n"] == 1
        _render(m, _inputs(text = 11), steps = 3)
        assert calls["n"] == 2
    finally:
        q._build_layout = real


def test_a_different_layout_of_the_same_length_is_not_reused():
    """Same shapes, image slots moved: the content key must tell them apart."""
    m = _model()
    a = _inputs(text = 8, cond = (4, 4))
    b = _inputs(text = 8, cond = (4, 4))
    vlm = b["img_mask"][0, :12].clone()
    b["img_mask"][:, :12] = vlm.roll(3)
    ref_b = _render(m, b)
    assert q.install()
    _render(m, a)
    got_b = _render(m, b)
    for x, y in zip(ref_b, got_b):
        assert torch.equal(x, y)


def test_grad_and_kill_switch_take_the_stock_forward(monkeypatch):
    m = _model()
    assert q.install()
    inp = _inputs()
    seen = []
    real = q._layout_for
    monkeypatch.setattr(q, "_layout_for", lambda *a, **k: seen.append(1) or real(*a, **k))
    kv = qmod.QwenImage21KVCache(len(m.transformer_blocks))
    with torch.enable_grad():
        m(
            hidden_states = inp["latents"],
            timestep = torch.ones(1),
            encoder_hidden_states = inp["encoder_hidden_states"],
            img_shapes = inp["img_shapes"],
            img_mask = inp["img_mask"],
            kv_cache = kv,
            kv_cache_mode = "extract",
            return_dict = False,
        )
    assert seen == []
    monkeypatch.setenv(q.FAST_STEP_ENV, "0")
    _render(m, inp, steps = 2)
    assert seen == []
    monkeypatch.delenv(q.FAST_STEP_ENV)
    _render(m, inp, steps = 2)
    assert seen


def test_stock_errors_are_kept():
    m = _model()
    assert q.install()
    inp = _inputs()
    with pytest.raises(ValueError, match = "kv_cache_mode"):
        with torch.inference_mode():
            m(
                hidden_states = inp["latents"],
                timestep = torch.ones(1),
                encoder_hidden_states = inp["encoder_hidden_states"],
                img_shapes = inp["img_shapes"],
                img_mask = inp["img_mask"],
                kv_cache_mode = "cached",
            )


def test_install_is_idempotent_and_uninstall_restores_the_stock_forward():
    cls = qmod.QwenImage21Transformer2DModel
    stock = vars(cls)["forward"]
    assert q.install() and q.install()
    patched = vars(cls)["forward"]
    assert patched is not stock
    assert patched.__unsloth_stock_forward__ is stock
    # The graph layer and hooks read the signature; it must be the stock one.
    import inspect

    assert inspect.signature(patched) == inspect.signature(stock)
    q.uninstall()
    assert vars(cls)["forward"] is stock
    q.uninstall()
    assert vars(cls)["forward"] is stock


def test_installed_diffusers_matches_the_fingerprints():
    """Fails when the pinned diffusers changes one of these functions: re-check the fast forward
    against the new stock one, then add the new digest."""
    assert q.why_unsupported(qmod) is None


def test_a_drifted_function_keeps_the_stock_forward(monkeypatch):
    fake = types.SimpleNamespace(**vars(qmod))

    def _qwenimage21_prefix_segments(image_ids, prefix_len):
        return []

    fake._qwenimage21_prefix_segments = _qwenimage21_prefix_segments
    assert "prefix_segments differs" in q.why_unsupported(fake)
    real_import = importlib.import_module
    monkeypatch.setattr(
        importlib,
        "import_module",
        lambda name, *a: fake if name == q._MODULE else real_import(name, *a),
    )
    cls = qmod.QwenImage21Transformer2DModel
    stock = vars(cls)["forward"]
    assert q.install() is False
    assert vars(cls)["forward"] is stock


def test_kill_switch_blocks_install(monkeypatch):
    monkeypatch.setenv(q.FAST_STEP_ENV, "off")
    assert q.install() is False
    assert not getattr(
        vars(qmod.QwenImage21Transformer2DModel)["forward"], "__unsloth_q21_fast_step__", False
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs CUDA for the sync check")
def test_cached_steps_do_not_sync_the_host():
    m = _model().to("cuda")
    inp = {
        k: (v.to("cuda") if torch.is_tensor(v) else v) for k, v in _inputs(batch = 2, pad = 2).items()
    }
    assert q.install()
    kv = qmod.QwenImage21KVCache(len(m.transformer_blocks))
    with torch.inference_mode():
        for i in range(4):
            if i >= 2:
                torch.cuda.set_sync_debug_mode("error")
            try:
                m(
                    hidden_states = inp["latents"],
                    timestep = torch.full((2,), 0.5, device = "cuda"),
                    encoder_hidden_states = inp["encoder_hidden_states"],
                    encoder_hidden_states_mask = inp["encoder_hidden_states_mask"],
                    img_shapes = inp["img_shapes"],
                    img_mask = inp["img_mask"],
                    kv_cache = kv,
                    kv_cache_mode = "extract" if i == 0 else "cached",
                    return_dict = False,
                )
            finally:
                torch.cuda.set_sync_debug_mode("default")


def test_digest_ignores_docstrings_comments_and_blank_lines_only():
    import linecache

    src = "def f(x):\n    '''doc'''\n    # note\n    y = x + 1  # why\n\n    return y\n"
    other = "def f(x):\n    y = x + 1\n    return y\n"
    changed = "def f(x):\n    y = x + 2\n    return y\n"
    digests = []
    for i, text in enumerate((src, other, changed)):
        name = f"<q21-digest-{i}>"
        linecache.cache[name] = (len(text), None, text.splitlines(True), name)
        scope: dict = {}
        exec(compile(text, name, "exec"), scope)
        digests.append(q._digest(scope["f"]))
    assert digests[0] == digests[1] != digests[2]


def test_install_for_pipe_only_touches_qwen_image_21_and_never_raises(monkeypatch):
    calls = []
    monkeypatch.setattr(q, "install", lambda logger = None: calls.append(1) or True)
    assert q.install_for_pipe(types.SimpleNamespace(transformer = object())) is False
    assert q.install_for_pipe(types.SimpleNamespace()) is False
    assert calls == []
    pipe = types.SimpleNamespace(
        transformer = qmod.QwenImage21Transformer2DModel.__new__(qmod.QwenImage21Transformer2DModel)
    )
    assert q.install_for_pipe(pipe) is True and calls == [1]
    monkeypatch.setattr(
        q, "install", lambda logger = None: (_ for _ in ()).throw(RuntimeError("boom"))
    )
    assert q.install_for_pipe(pipe) is False
