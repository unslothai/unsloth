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


def test_offloaded_denoiser_declines(monkeypatch):
    cls = _flux3()
    monkeypatch.delenv(cb.COND_BLOCKS_ENV, raising = False)
    model = _tiny(cls)
    cb.prepare(model)
    model.single_transformer_blocks[0]._hf_hook = object()
    try:
        assert not cb.begin_render(model)
        assert "accelerate hook" in cb.status(model)["reason"]
    finally:
        del model.single_transformer_blocks[0]._hf_hook


def test_pre_hook_stays_last(monkeypatch):
    """A hook registered after the cache (dynamic text dims installs at compile time) must still run first, so the
    conditioning blocks run inside its scope."""
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
