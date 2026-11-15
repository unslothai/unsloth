# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""CFG batching policy: every denoiser class Studio loads is classified once (batched by the router, batched by its
own pipeline, or not batchable), an offloaded load batches with no timing trial whenever memory allows, and the
newly batched classes match two plain calls in their pipeline's call shape. CPU only."""

from __future__ import annotations

import types

import pytest
import torch

from core.inference import diffusion_cfg_batch as cb
from core.inference import diffusion_families as df
from core.inference import video_families as vf


def _family_classes() -> set[str]:
    out: set[str] = set()
    for fam in (*df._FAMILIES, *vf._FAMILIES):
        for attr in ("transformer_class", "transformer2_class"):
            name = getattr(fam, attr, None)
            if name:
                out.add(str(name))
    return out


def test_every_family_denoiser_has_exactly_one_cfg_policy():
    tables = (set(cb.STATELESS_CLASSES), set(cb.PIPELINE_BATCHED), set(cb.NOT_BATCHABLE))
    for i, a in enumerate(tables):
        for b in tables[i + 1 :]:
            assert not (a & b), sorted(a & b)
    classified = set().union(*tables)
    missing = sorted(_family_classes() - classified)
    assert not missing, f"no CFG batching policy for {missing}"


def test_the_compiled_resident_and_measured_tables_only_name_batched_classes():
    assert cb.COMPILED_RESIDENT_CLASSES <= cb.STATELESS_CLASSES
    for cls, card, kind in cb.MEASURED_SLOWER:
        assert cls in cb.STATELESS_CLASSES and kind in ("eager", "compiled") and card


def test_the_default_mode_and_its_switches(monkeypatch):
    monkeypatch.delenv(cb.CFG_BATCH_ENV, raising = False)
    assert cb.cfg_batch_mode() == "auto"
    for raw, mode in (("timed", "timed"), ("force", "force"), ("0", "off"), ("1", "auto")):
        monkeypatch.setenv(cb.CFG_BATCH_ENV, raw)
        assert cb.cfg_batch_mode() == mode


def _offloaded(module):
    sub = torch.nn.Linear(1, 1)
    module.add_module("offloaded_group", sub)
    sub._diffusers_hook = types.SimpleNamespace(hooks = {"group_offloading": object()})
    return module


class _Pipe:
    def __init__(self, transformer):
        self.transformer = transformer


class _Calls(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.linspace(0.5, 1.5, 8))
        self.calls = []


class Krea2Transformer2DModel(_Calls):
    """A masked-text call (Krea2 / Lumina2 shape): 4-D latents, per-sample text with ``encoder_attention_mask``."""

    def forward(
        self,
        hidden_states,
        timestep,
        encoder_hidden_states,
        encoder_attention_mask,
        attention_kwargs = None,
        return_dict = True,
    ):
        self.calls.append(int(hidden_states.shape[0]))
        m = encoder_attention_mask.float().unsqueeze(-1)
        text = (encoder_hidden_states * m).sum(1) / m.sum(1)
        out = (
            hidden_states * self.w[:4].view(1, 4, 1, 1)
            + text.mean(-1).view(-1, 1, 1, 1)
            + timestep.view(-1, 1, 1, 1)
        )
        return (out,)


class Flux3Transformer2DModel(_Calls):
    """FLUX.3 modular call: ``img_ids`` / ``txt_ids`` without a batch dim, no text mask (padding is attended)."""

    def forward(
        self,
        hidden_states,
        encoder_hidden_states,
        timestep,
        img_ids,
        txt_ids,
        image_cond_hidden_states = None,
        image_cond_ids = None,
        target_timestep = None,
        attention_kwargs = None,
        return_dict = True,
    ):
        self.calls.append(int(hidden_states.shape[0]))
        assert img_ids.dim() == 2 and txt_ids.shape[0] == encoder_hidden_states.shape[1]
        text = encoder_hidden_states.mean(1, keepdim = True)
        return (hidden_states * self.w + text + timestep.view(-1, 1, 1) + txt_ids.float().mean(),)


class HunyuanImageTransformer2DModel(_Calls):
    """HunyuanImagePipeline guider call: two text encoders, each with a mask."""

    def forward(
        self,
        hidden_states,
        timestep,
        encoder_hidden_states,
        encoder_attention_mask,
        timestep_r = None,
        encoder_hidden_states_2 = None,
        encoder_attention_mask_2 = None,
        guidance = None,
        attention_kwargs = None,
        return_dict = True,
    ):
        self.calls.append(int(hidden_states.shape[0]))
        m = encoder_attention_mask.float().unsqueeze(-1)
        m2 = encoder_attention_mask_2.float().unsqueeze(-1)
        t1 = (encoder_hidden_states * m).sum(1) / m.sum(1)
        t2 = (encoder_hidden_states_2 * m2).sum(1) / m2.sum(1)
        return (
            hidden_states * 0.9
            + (t1.mean(-1) + t2.mean(-1)).view(-1, 1, 1, 1)
            + timestep.view(-1, 1, 1, 1),
        )


for _cls in (Krea2Transformer2DModel, Flux3Transformer2DModel, HunyuanImageTransformer2DModel):
    _cls.__module__ = "diffusers.models.fake"


def _loop(
    model,
    kind,
    steps = 6,
    neg_len = None,
):
    g = torch.Generator().manual_seed(0)
    preds = []
    if kind == "lumina2":
        lat = torch.randn(1, 4, 2, 2, generator = g)
        pos, neg = torch.randn(1, 6, 8, generator = g), torch.randn(1, 6, 8, generator = g)
        pm = torch.tensor([[1, 1, 1, 1, 1, 0]])
        nm = torch.tensor([[1, 0, 0, 0, 0, 0]])
        kw = lambda c: dict(
            encoder_hidden_states = pos if c else neg, encoder_attention_mask = pm if c else nm
        )  # noqa: E731
    elif kind == "flux3":
        lat = torch.randn(1, 4, 8, generator = g)
        T, Tn = 8, neg_len or 8
        pos, neg = torch.randn(1, T, 8, generator = g), torch.randn(1, Tn, 8, generator = g)
        img_ids = torch.zeros(4, 4)
        ids = {T: torch.arange(T * 4.0).view(T, 4), Tn: torch.arange(Tn * 4.0).view(Tn, 4)}
        kw = lambda c: dict(
            encoder_hidden_states = pos if c else neg, img_ids = img_ids, txt_ids = ids[T if c else Tn]
        )  # noqa: E731
    else:
        lat = torch.randn(1, 4, 2, 2, generator = g)
        pos, neg = torch.randn(1, 6, 8, generator = g), torch.randn(1, 6, 8, generator = g)
        pos2, neg2 = torch.randn(1, 3, 4, generator = g), torch.randn(1, 3, 4, generator = g)
        pm, nm = torch.tensor([[1, 1, 1, 1, 0, 0]]), torch.tensor([[1, 1, 0, 0, 0, 0]])
        pm2, nm2 = torch.tensor([[1, 1, 1]]), torch.tensor([[1, 0, 0]])
        kw = lambda c: dict(  # noqa: E731
            encoder_hidden_states = pos if c else neg,
            encoder_attention_mask = pm if c else nm,
            encoder_hidden_states_2 = pos2 if c else neg2,
            encoder_attention_mask_2 = pm2 if c else nm2,
        )
    for i in range(steps):
        t = (torch.tensor(1000.0 - 100 * i) / 1000).expand(lat.shape[0])
        c = model(hidden_states = lat, timestep = t, return_dict = False, **kw(True))[0]
        u = model(hidden_states = lat, timestep = t, return_dict = False, **kw(False))[0]
        pred = u + 3.5 * (c - u)
        preds.append(pred)
        lat = lat - 0.1 * pred
    return lat, preds


@pytest.mark.parametrize(
    "cls, kind",
    [
        (Krea2Transformer2DModel, "lumina2"),
        (Flux3Transformer2DModel, "flux3"),
        (HunyuanImageTransformer2DModel, "hunyuanimage"),
    ],
)
def test_a_newly_batched_class_matches_two_calls_from_step_two(monkeypatch, cls, kind):
    monkeypatch.delenv(cb.CFG_BATCH_ENV, raising = False)
    ref, _ = _loop(cls(), kind)
    model = _offloaded(cls())
    pipe = _Pipe(model)
    assert cb.install_cfg_batch(pipe, offload_policy = "group") is None
    got, _ = _loop(model, kind)
    torch.testing.assert_close(got, ref)
    router = cb.router_for(pipe)
    assert model.calls == [1, 1] + [2] * 5, router.describe()
    assert router.stats["hits"] == 5 and router.stats["misses"] == 0
    assert router.describe()["plain_pair_s"] is None  # no timing trial ran


def test_flux3_branches_in_different_text_windows_stay_two_calls(monkeypatch):
    """No transformer mask: padding the shorter window would change its attention, so that shape stays plain."""
    monkeypatch.delenv(cb.CFG_BATCH_ENV, raising = False)
    ref, _ = _loop(Flux3Transformer2DModel(), "flux3", neg_len = 4)
    model = _offloaded(Flux3Transformer2DModel())
    pipe = _Pipe(model)
    assert cb.install_cfg_batch(pipe, offload_policy = "group") is None
    got, _ = _loop(model, "flux3", neg_len = 4)
    torch.testing.assert_close(got, ref, rtol = 0, atol = 0)
    assert set(model.calls) == {1}
    assert "takes no mask" in cb.router_for(pipe).describe()["shape_off"]


def test_lumina2_unequal_text_lengths_pad_with_its_attention_mask():
    a = (
        (),
        {
            "hidden_states": torch.zeros(1, 2),
            "encoder_hidden_states": torch.ones(1, 3, 2),
            "encoder_attention_mask": torch.ones(1, 3, dtype = torch.long),
        },
    )
    b = (
        (),
        {
            "hidden_states": torch.zeros(1, 2),
            "encoder_hidden_states": torch.ones(1, 5, 2),
            "encoder_attention_mask": torch.ones(1, 5, dtype = torch.long),
        },
    )
    _args, kw = cb.merge_branches(a, b, 1)
    assert kw["encoder_hidden_states"].shape == (2, 5, 2)
    assert kw["encoder_attention_mask"].tolist() == [[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]]
    assert "encoder_hidden_states_mask" not in kw


@pytest.mark.parametrize(
    "name",
    [
        "QwenImage21Transformer2DModel",
        "Ideogram4Transformer2DModel",
        "Lumina2Transformer2DModel",
        "ZImageTransformer2DModel",
        "HiDreamImageTransformer2DModel",
    ],
)
def test_classified_but_not_router_batched_classes_decline_with_their_reason(monkeypatch, name):
    monkeypatch.delenv(cb.CFG_BATCH_ENV, raising = False)
    cls = type(name, (Krea2Transformer2DModel,), {})
    cls.__module__ = "diffusers.models.fake"
    reason = cb.cfg_batch_decline(_Pipe(_offloaded(cls())), offload_policy = "group")
    assert reason == {**cb.PIPELINE_BATCHED, **cb.NOT_BATCHABLE}[name]


def test_a_measured_slower_card_keeps_two_calls_unless_forced(monkeypatch):
    monkeypatch.delenv(cb.CFG_BATCH_ENV, raising = False)
    monkeypatch.setitem(
        cb.MEASURED_SLOWER, ("Krea2Transformer2DModel", "Tesla T4", "eager"), "1.04x slower"
    )
    monkeypatch.setattr(cb, "_cuda_latent", lambda latent: True)
    monkeypatch.setattr(cb, "_card_name", lambda latent: "Tesla T4")
    monkeypatch.setattr(cb.CfgBatchRouter, "_memory_ok", lambda self, latent, shape: True)
    model = _offloaded(Krea2Transformer2DModel())
    pipe = _Pipe(model)
    assert cb.install_cfg_batch(pipe, offload_policy = "group") is None
    _loop(model, "lumina2")
    assert set(model.calls) == {1}
    assert (
        "measured slower batched on this card (1.04x slower)"
        in cb.router_for(pipe).describe()["shape_off"]
    )
    monkeypatch.setattr(cb, "_card_name", lambda latent: "NVIDIA L4")
    other = _offloaded(Krea2Transformer2DModel())
    opipe = _Pipe(other)
    assert cb.install_cfg_batch(opipe, offload_policy = "group") is None
    _loop(other, "lumina2")
    assert other.calls[-1] == 2
    monkeypatch.setattr(cb, "_card_name", lambda latent: "Tesla T4")
    monkeypatch.setenv(cb.CFG_BATCH_ENV, "force")
    forced = _offloaded(Krea2Transformer2DModel())
    fpipe = _Pipe(forced)
    assert cb.install_cfg_batch(fpipe, offload_policy = "group") is None
    _loop(forced, "lumina2")
    assert forced.calls[-1] == 2


def test_the_memory_gate_still_keeps_a_render_plain_without_a_trial(monkeypatch):
    monkeypatch.delenv(cb.CFG_BATCH_ENV, raising = False)
    monkeypatch.setattr(cb.CfgBatchRouter, "_memory_ok", lambda self, latent, shape: False)
    ref, _ = _loop(Krea2Transformer2DModel(), "lumina2")
    model = _offloaded(Krea2Transformer2DModel())
    pipe = _Pipe(model)
    assert cb.install_cfg_batch(pipe, offload_policy = "group") is None
    got, _ = _loop(model, "lumina2")
    torch.testing.assert_close(got, ref, rtol = 0, atol = 0)
    assert set(model.calls) == {1} and "no room" in cb.router_for(pipe).describe()["shape_off"]


# Real diffusers classes (tiny random configs): batched vs two calls in fp32 on CPU, forced (CPU resident is plain).
_REAL = {
    "Lumina2Transformer2DModel": dict(
        sample_size = 8,
        patch_size = 2,
        in_channels = 4,
        hidden_size = 32,
        num_layers = 1,
        num_refiner_layers = 1,
        num_attention_heads = 2,
        num_kv_heads = 1,
        multiple_of = 16,
        axes_dim_rope = (4, 6, 6),
        axes_lens = (300, 64, 64),
        cap_feat_dim = 16,
    ),
    "HunyuanImageTransformer2DModel": dict(
        in_channels = 4,
        out_channels = 4,
        num_attention_heads = 2,
        attention_head_dim = 16,
        num_layers = 1,
        num_single_layers = 1,
        num_refiner_layers = 1,
        text_embed_dim = 16,
        text_embed_2_dim = 8,
        rope_axes_dim = (8, 8),
    ),
}


@pytest.mark.parametrize("name", sorted(_REAL))
def test_real_diffusers_class_batched_matches_two_calls(monkeypatch, name):
    diffusers = pytest.importorskip("diffusers")
    cls = getattr(diffusers, name, None)
    if cls is None:
        pytest.skip(f"diffusers {diffusers.__version__} has no {name}")
    monkeypatch.setenv(cb.CFG_BATCH_ENV, "force")
    torch.manual_seed(0)
    model = cls(**_REAL[name]).eval()
    g = torch.Generator().manual_seed(1)
    r = lambda *s: torch.randn(*s, generator = g)  # noqa: E731
    pm = torch.zeros(1, 12, dtype = torch.long)
    pm[:, :9] = 1
    nm = torch.zeros(1, 12, dtype = torch.long)
    nm[:, :3] = 1
    pos, neg = r(1, 12, 16), r(1, 12, 16)
    extra = {}
    if name == "HunyuanImageTransformer2DModel":
        pos2, neg2 = r(1, 6, 8), r(1, 6, 8)
        m2p, m2n = torch.ones(1, 6, dtype = torch.long), torch.tensor([[1, 0, 0, 0, 0, 0]])
        extra = {
            True: dict(encoder_hidden_states_2 = pos2, encoder_attention_mask_2 = m2p),
            False: dict(encoder_hidden_states_2 = neg2, encoder_attention_mask_2 = m2n),
        }
    lat0 = r(1, 4, 8, 8)

    def loop(m):
        lat, outs = lat0.clone(), []
        for i in range(4):
            t = torch.full((1,), 1.0 - i / 4)
            x = lat.clone()
            c = m(
                hidden_states = x,
                timestep = t,
                encoder_hidden_states = pos,
                encoder_attention_mask = pm,
                return_dict = False,
                **extra.get(True, {}),
            )[0]
            u = m(
                hidden_states = x,
                timestep = t,
                encoder_hidden_states = neg,
                encoder_attention_mask = nm,
                return_dict = False,
                **extra.get(False, {}),
            )[0]
            outs += [c, u]
            lat = lat - 0.1 * (u + 3.0 * (c - u))
        return outs

    with torch.no_grad():
        ref = loop(model)
        pipe = _Pipe(model)
        assert cb.install_cfg_batch(pipe, offload_policy = "none") is None
        got = loop(model)
        status = cb.cfg_batch_status(pipe)
        cb.uninstall_cfg_batch(pipe)
    assert status["batched_steps"] == 3 and status["misses"] == 0, status
    for a, b in zip(got, ref):
        torch.testing.assert_close(a, b, rtol = 1e-5, atol = 1e-5)


def test_a_streamed_offload_hook_removal_does_not_end_batching(monkeypatch):
    """diffusers' use_stream group offload removes its lazy-prefetch hook after the first forward and writes the bound
    class forward over ``module.forward``: the router takes its slot back and the render still batches."""
    monkeypatch.delenv(cb.CFG_BATCH_ENV, raising = False)
    model = _offloaded(HunyuanImageTransformer2DModel())
    pipe = _Pipe(model)
    assert cb.install_cfg_batch(pipe, offload_policy = "group") is None
    router = cb.router_for(pipe)
    real = router.orig_forward
    state = {"n": 0}

    def lazy_hook(*a, **k):  # the root hook: runs once, then restores the class forward
        state["n"] += 1
        out = real(*a, **k)
        if state["n"] == 1:
            model.__dict__["forward"] = types.MethodType(type(model).forward, model)
        return out

    router.orig_forward = lazy_hook
    ref, _ = _loop(HunyuanImageTransformer2DModel(), "hunyuanimage")
    got, _ = _loop(model, "hunyuanimage")
    torch.testing.assert_close(got, ref)
    assert model.__dict__["forward"] is router and router.stats["retaken"] == 1
    assert model.calls == [1, 1] + [2] * 5


def test_a_slot_taken_by_another_layer_is_not_retaken(monkeypatch):
    monkeypatch.delenv(cb.CFG_BATCH_ENV, raising = False)
    model = _offloaded(HunyuanImageTransformer2DModel())
    pipe = _Pipe(model)
    assert cb.install_cfg_batch(pipe, offload_policy = "group") is None
    router = cb.router_for(pipe)
    graphed = lambda *a, **k: router.orig_forward(*a, **k)  # noqa: E731 - a CUDA graph layer took the slot
    model.__dict__["forward"] = graphed
    _loop(model, "hunyuanimage")
    cb.end_cfg_batch_render(pipe)
    assert model.__dict__["forward"] is graphed and "retaken" not in router.stats
