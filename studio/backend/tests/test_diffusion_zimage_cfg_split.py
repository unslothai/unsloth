# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for ``diffusion_zimage_cfg_split.py``: a guided Z-Image call after guidance-off ones compiles nothing new."""

from __future__ import annotations

import types

import pytest

from core.inference import diffusion_zimage_cfg_split as zs

torch = pytest.importorskip("torch")
zmod = pytest.importorskip("diffusers.models.transformers.transformer_z_image")

_CONFIG = dict(
    all_patch_size = (2,),
    all_f_patch_size = (1,),
    in_channels = 16,
    dim = 32,
    n_layers = 2,
    n_refiner_layers = 1,
    n_heads = 2,
    n_kv_heads = 2,
    norm_eps = 1e-5,
    qk_norm = True,
    cap_feat_dim = 16,
    rope_theta = 256.0,
    t_scale = 1000.0,
    axes_dims = [8, 4, 4],
    axes_lens = [512, 64, 64],
)


@pytest.fixture(autouse = True)
def _clean(monkeypatch):
    monkeypatch.delenv(zs.ZIMAGE_CFG_SPLIT_ENV, raising = False)
    yield
    zs.uninstall_zimage_cfg_split()


def _model():
    torch.manual_seed(0)
    return zmod.ZImageTransformer2DModel(**_CONFIG).eval()


def _latent(seed):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(16, 1, 8, 8, generator = g)


def _caption(n, seed):
    g = torch.Generator().manual_seed(1000 + seed)
    return torch.randn(n, 16, generator = g)


def _guided_args(latent, t, cond_len, neg_len, seed):
    # What ZImagePipeline passes on a guided step: the latent repeated, the prompt then the negative embeddings.
    return (
        [latent, latent.clone()],
        t.repeat(2),
        [_caption(cond_len, seed), _caption(neg_len, seed + 1)],
    )


def _compile_blocks(model):
    """Studio's regional compile: every repeated block's forward through one dynamic dynamo code object."""
    for block in (*model.noise_refiner, *model.context_refiner, *model.layers):
        block.forward = torch.compile(block.forward, backend = "eager", dynamic = True)


def _graphs():
    from torch._dynamo.utils import counters
    return counters["stats"]["unique_graphs"]


def _run(model, *args):
    with torch.no_grad():
        return model(*args, return_dict = False)[0]


def test_guided_call_after_guidance_off_compiles_nothing_new():
    torch._dynamo.reset()
    model = _model()
    pipe = types.SimpleNamespace(transformer = model)
    assert zs.install_zimage_cfg_split(pipe, offload_policy = "none") is None
    _compile_blocks(model)
    t = torch.tensor([0.4])
    with zs.render(pipe, 1):
        # Guidance-off renders at two prompt lengths (two 32-token caption buckets).
        _run(model, [_latent(0)], t, [_caption(9, 0)])
        _run(model, [_latent(1)], t, [_caption(45, 1)])
        warm = _graphs()
        assert warm > 0
        # CFG turned on: captions in different buckets (a mask when batched), then in the same bucket (no mask).
        _run(model, *_guided_args(_latent(2), t, 45, 9, 2))
        _run(model, *_guided_args(_latent(3), t, 7, 9, 3))
        _run(model, *_guided_args(_latent(4), t, 70, 9, 4))
    assert _graphs() == warm
    assert zs.zimage_cfg_split_status(pipe) == {"armed": True, "split": 3, "batched": 0}


def test_batched_guided_call_recompiles_without_the_split(monkeypatch):
    """The defect the split removes: the doubled batch (and its attention mask) are new graphs."""
    torch._dynamo.reset()
    monkeypatch.setenv(zs.ZIMAGE_CFG_SPLIT_ENV, "0")
    model = _model()
    pipe = types.SimpleNamespace(transformer = model)
    assert zs.install_zimage_cfg_split(pipe, offload_policy = "none") is not None
    _compile_blocks(model)
    t = torch.tensor([0.4])
    with zs.render(pipe, 1):
        _run(model, [_latent(0)], t, [_caption(9, 0)])
        _run(model, [_latent(1)], t, [_caption(45, 1)])
        warm = _graphs()
        _run(model, *_guided_args(_latent(2), t, 45, 9, 2))
    assert _graphs() > warm


def test_split_output_is_each_branch_alone_and_matches_the_batched_call():
    model = _model()
    pipe = types.SimpleNamespace(transformer = model)
    t = torch.tensor([0.4])
    x, tt, cap = _guided_args(_latent(5), t, 45, 9, 5)
    batched = _run(model, x, tt, cap)
    alone = _run(model, x[:1], tt[:1], cap[:1]) + _run(model, x[1:], tt[1:], cap[1:])
    zs.install_zimage_cfg_split(pipe, offload_policy = "none")
    with zs.render(pipe, 1):
        split = _run(model, x, tt, cap)
        # return_dict=True (img2img / inpaint) joins the output dataclass the same way
        with torch.no_grad():
            split_dc = model(x, tt, cap).sample
    assert len(split) == 2
    for s, a, b, d in zip(split, alone, batched, split_dc):
        assert torch.equal(s, a)
        assert torch.equal(d, a)
        torch.testing.assert_close(s, b, atol = 1e-5, rtol = 1e-4)


def test_guidance_off_and_unarmed_calls_are_untouched():
    model = _model()
    pipe = types.SimpleNamespace(transformer = model)
    zs.install_zimage_cfg_split(pipe, offload_policy = "none")
    t = torch.tensor([0.4, 0.4])
    x, cap = [_latent(6), _latent(7)], [_caption(9, 6), _caption(12, 7)]
    with zs.render(pipe, 2):
        # Two images per call with guidance off: one batched call, as before.
        _run(model, x, t, cap)
    # Outside a render, nothing splits.
    _run(model, *_guided_args(_latent(8), torch.tensor([0.4]), 45, 9, 8))
    assert zs.zimage_cfg_split_status(pipe)["split"] == 0


@pytest.mark.parametrize("policy", ["group", "streaming", "sequential"])
def test_streamed_denoiser_is_not_armed(policy):
    pipe = types.SimpleNamespace(transformer = _model())
    assert "streamed" in zs.install_zimage_cfg_split(pipe, offload_policy = policy)
    assert zs.zimage_cfg_split_status(pipe)["armed"] is False


def test_step_cache_and_step_skip_keep_the_batched_call():
    model = _model()
    pipe = types.SimpleNamespace(transformer = model)
    zs.install_zimage_cfg_split(pipe, offload_policy = "none")
    args = _guided_args(_latent(9), torch.tensor([0.4]), 45, 9, 9)
    model._unsloth_step_cache = object()
    with zs.render(pipe, 1):
        _run(model, *args)
    del model._unsloth_step_cache

    class _Skip:
        _unsloth_outer_forward = True

        def __call__(self, *a, **k):
            return type(model).forward(model, *a, **k)

    model.__dict__["forward"] = _Skip()
    with zs.render(pipe, 1):
        _run(model, *args)
    assert zs.zimage_cfg_split_status(pipe) == {"armed": True, "split": 0, "batched": 2}


def test_not_zimage_and_uninstall_restores_the_class():
    assert (
        zs.install_zimage_cfg_split(
            types.SimpleNamespace(transformer = torch.nn.Linear(2, 2)), offload_policy = "none"
        )
        == "not a Z-Image denoiser"
    )
    cls = zmod.ZImageTransformer2DModel
    before = cls.__dict__.get("__call__")
    pipe = types.SimpleNamespace(transformer = _model())
    zs.install_zimage_cfg_split(pipe, offload_policy = "none")
    assert getattr(cls.__dict__.get("__call__"), "__unsloth_zimage_cfg_split__", False)
    zs.uninstall_zimage_cfg_split(pipe)
    assert cls.__dict__.get("__call__") is before
    assert zs.zimage_cfg_split_status(pipe) == {"armed": False}
