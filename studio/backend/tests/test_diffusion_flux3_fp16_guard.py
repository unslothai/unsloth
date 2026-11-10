# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""FLUX.3 fp16 residual rescale: kill switch, and finite fp32-equal output where plain fp16 overflows. CPU only."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
nn = torch.nn
F = torch.nn.functional

from core.inference import diffusion_fp16_guard as guard  # noqa: E402
from core.inference.diffusion import _resolve_diffusion_compute_dtype  # noqa: E402
from core.inference.diffusion_families import detect_family  # noqa: E402

_REAL_RECIPE_SUPPORTED = guard._recipe_supported

DISTILLED = "black-forest-labs/flux-3-image-klein"
BASE = "black-forest-labs/flux-3-image-klein-base"


@pytest.fixture(autouse = True)
def _clean_env(monkeypatch):
    monkeypatch.delenv(guard.FP16_GUARD_ENV, raising = False)
    monkeypatch.setattr(guard, "_recipe_supported", lambda fam, recipe: True)


def test_flux3_stays_fp16_behind_residual_guard():
    for repo in (DISTILLED, BASE):
        fam = detect_family(repo)
        assert fam.fp16_incompatible is True and fam.fp16_guard == "rescale_residual"
        assert guard.fp16_promotes_to_fp32(fam) is False
        assert _resolve_diffusion_compute_dtype(fam, torch.float16) is torch.float16
        assert _resolve_diffusion_compute_dtype(fam, torch.bfloat16) is torch.bfloat16


def test_global_kill_switch_promotes_flux3(monkeypatch):
    monkeypatch.setenv(guard.FP16_GUARD_ENV, "0")
    assert (
        _resolve_diffusion_compute_dtype(detect_family(DISTILLED), torch.float16) is torch.float32
    )


def test_unmatched_diffusers_source_promotes_flux3(monkeypatch):
    monkeypatch.setattr(guard, "_recipe_supported", lambda fam, recipe: False)
    assert (
        _resolve_diffusion_compute_dtype(detect_family(DISTILLED), torch.float16) is torch.float32
    )


@pytest.mark.parametrize("value, guarded", [("0", False), ("off", False), ("1", True), ("", True)])
def test_flux3_guard_kill_switch(value, guarded):
    backend = Path(__file__).resolve().parents[1]
    env = {**os.environ, "UNSLOTH_STUDIO_FLUX3_FP16_GUARD": value, "PYTHONPATH": str(backend)}
    code = (
        "from core.inference.diffusion_families import detect_family as d;"
        f"print(d({DISTILLED!r}).fp16_guard, d({BASE!r}).fp16_guard)"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], cwd = backend, env = env, capture_output = True, text = True
    )
    assert out.returncode == 0, out.stderr[-2000:]
    want = "rescale_residual" if guarded else "None"
    assert out.stdout.split()[-2:] == [want, want]


# --- A FLUX.3-shaped model: the diffusers forward bodies, tiny widths, weights raised until float16 overflows. ---


class _Attn(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.heads = heads
        self.to_q = nn.Linear(dim, dim, bias = False)
        self.to_k = nn.Linear(dim, dim, bias = False)
        self.to_v = nn.Linear(dim, dim, bias = False)
        self.norm_q = nn.RMSNorm(dim // heads, eps = 1e-6)
        self.norm_k = nn.RMSNorm(dim // heads, eps = 1e-6)
        self.to_out = nn.ModuleList([nn.Linear(dim, dim, bias = False), nn.Dropout(0.0)])

    def forward(self, hidden_states):
        q, k, v = (
            p(hidden_states).unflatten(-1, (self.heads, -1))
            for p in (self.to_q, self.to_k, self.to_v)
        )
        q = self.norm_q(q.float()).to(v.dtype).transpose(1, 2)
        k = self.norm_k(k.float()).to(v.dtype).transpose(1, 2)
        o = F.scaled_dot_product_attention(q, k, v.transpose(1, 2)).transpose(1, 2).flatten(2)
        return self.to_out[1](self.to_out[0](o))


class _SwiGLU(nn.Module):
    def forward(self, x):
        half = x.shape[-1] // 2
        return F.silu(x[..., :half]) * x[..., half:]


class _FF(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear_in = nn.Linear(dim, 6 * dim, bias = False)
        self.act_fn = _SwiGLU()
        self.linear_out = nn.Linear(3 * dim, dim, bias = False)

    def forward(self, x):
        return self.linear_out(self.act_fn(self.linear_in(x)))


class _Mode(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine = False, eps = 1e-6)
        self.attn = _Attn(dim, heads)
        self.ff = _FF(dim)

    def forward(self, hidden_states, temb_mod):
        shift, scale, gate = torch.chunk(temb_mod, 3, dim = -1)
        n = (1 + scale) * self.norm(hidden_states) + shift
        return hidden_states + gate * (self.attn(hidden_states = n) + self.ff(n))


class _Single(_Mode):
    def forward(self, hidden_states, encoder_hidden_states, temb_mod):
        shift, scale, gate = torch.chunk(temb_mod, 3, dim = -1)
        n = (1 + scale) * self.norm(hidden_states) + shift
        c = (1 + scale) * self.norm(encoder_hidden_states) + shift
        combined = torch.cat([c, n], dim = 1)
        out = self.attn(hidden_states = combined) + self.ff(combined)
        ctx, img = torch.split(out, [encoder_hidden_states.shape[1], hidden_states.shape[1]], dim = 1)
        return encoder_hidden_states + gate * ctx, hidden_states + gate * img


class _OutNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine = False, eps = 1e-6)

    def forward(self, x):
        return self.norm(x)


class _Flux3Like(nn.Module):
    def __init__(
        self,
        dim = 64,
        heads = 4,
        ctx_dim = 96,
    ):
        super().__init__()
        self.context_embedder = nn.Linear(ctx_dim, dim, bias = False)
        self.x_embedders = nn.ModuleDict({"image": nn.Linear(16, dim, bias = False)})
        self.txt_mode_blocks = nn.ModuleList([_Mode(dim, heads) for _ in range(2)])
        self.mode_blocks = nn.ModuleDict(
            {"image": nn.ModuleList([_Mode(dim, heads) for _ in range(2)])}
        )
        self.single_transformer_blocks = nn.ModuleList([_Single(dim, heads) for _ in range(3)])
        self.norm_out = _OutNorm(dim)
        self.proj_out = nn.Linear(dim, 16, bias = False)
        self.mod = nn.Parameter(torch.rand(3 * dim) * 0.5)

    def forward(self, hidden_states, encoder_hidden_states):
        mod = self.mod.to(hidden_states.dtype)
        enc = self.context_embedder(encoder_hidden_states)
        img = self.x_embedders["image"](hidden_states)
        for b in self.txt_mode_blocks:
            enc = b(enc, mod)
        for b in self.mode_blocks["image"]:
            img = b(img, mod)
        for b in self.single_transformer_blocks:
            enc, img = b(img, enc, mod)
        return self.proj_out(self.norm_out(img))


def _overflowing_model():
    torch.manual_seed(0)
    model = _Flux3Like()
    with torch.no_grad():
        model.context_embedder.weight.mul_(3000.0)
        for b in model.single_transformer_blocks:
            b.ff.linear_out.weight.mul_(2000.0)
    return model


def _inputs():
    torch.manual_seed(1)
    return torch.randn(1, 12, 16), torch.randn(1, 6, 96) * 30


def _fp16_ok():
    try:
        nn.Linear(4, 4).half()(torch.ones(1, 4, dtype = torch.float16))
        return True
    except Exception:  # noqa: BLE001
        return False


needs_fp16 = pytest.mark.skipif(not _fp16_ok(), reason = "this torch has no CPU float16 matmul")


@needs_fp16
def test_residual_guard_keeps_flux3_shape_finite_and_accurate():
    model = _overflowing_model()
    x, ctx = _inputs()
    with torch.no_grad():
        ref = model.float()(x, ctx)
        assert model.context_embedder(ctx).abs().max() > 65504
        half = model.half()
        stock = half(x.half(), ctx.half())
        assert not torch.isfinite(stock).all()
        assert guard.install_fp16_guard(half, "rescale_residual", torch.float16) == 7
        fixed = half(x.half(), ctx.half())
    assert torch.isfinite(fixed).all()
    err = (fixed.float() - ref).abs().max() / ref.abs().max()
    assert err < 2e-2, float(err)


@needs_fp16
def test_residual_guard_is_idempotent_and_removable():
    model = _overflowing_model().half()
    eps = [m.eps for m in model.modules() if isinstance(m, (nn.LayerNorm, nn.RMSNorm))]
    assert guard.install_fp16_guard(model, "rescale_residual", torch.float16) == 7
    assert guard.install_fp16_guard(model, "rescale_residual", torch.float16) == 7
    assert model.norm_out.norm.eps == 1e-6 / 1024**2
    assert guard.remove_fp16_guard(model) == 7
    assert [m.eps for m in model.modules() if isinstance(m, (nn.LayerNorm, nn.RMSNorm))] == eps
    assert not any(m._forward_pre_hooks or m._forward_hooks for m in model.modules())


@needs_fp16
def test_residual_guard_follows_a_wrapped_entry_projection():
    """A LoRA-style wrapper swapped into the entry slot after install scales the stream once."""

    class Wrap(nn.Module):
        def __init__(self, base):
            super().__init__()
            self.base_layer = base
            self.delta = nn.Linear(base.in_features, base.out_features, bias = False)

        def forward(self, x):
            return self.base_layer(x) + self.delta(x)

    x, ctx = _inputs()
    model2 = _overflowing_model()
    with torch.no_grad():
        model2.context_embedder = Wrap(model2.context_embedder)
        model2.context_embedder.delta.weight.normal_(0, 50.0)
        ref = model2(x, ctx)
        half2 = _overflowing_model().half()
        assert guard.install_fp16_guard(half2, "rescale_residual", torch.float16) == 7
        half2.context_embedder = Wrap(half2.context_embedder).half()
        half2.context_embedder.delta.weight.copy_(model2.context_embedder.delta.weight.half())
        out = half2(x.half(), ctx.half())
    assert torch.isfinite(out).all()
    err = (out.float() - ref).abs().max() / ref.abs().max()
    assert err < 2e-2, float(err)
    assert not half2.context_embedder.base_layer._forward_pre_hooks


def test_residual_guard_needs_bias_free_projections():
    model = _overflowing_model()
    model.single_transformer_blocks[0].ff.linear_out = nn.Linear(192, 64, bias = True)
    assert guard.install_fp16_guard(model.half(), "rescale_residual", torch.float16) == 6
    model = _overflowing_model()
    model.context_embedder = nn.Linear(96, 64, bias = True)
    assert guard.install_fp16_guard(model.half(), "rescale_residual", torch.float16) == 0


@pytest.mark.parametrize("dtype", ["bfloat16", "float32"])
def test_residual_guard_never_touches_non_fp16(dtype):
    model = _overflowing_model().to(getattr(torch, dtype))
    assert guard.install_fp16_guard(model, "rescale_residual", getattr(torch, dtype)) == 0
    assert not any(m._forward_pre_hooks or m._forward_hooks for m in model.modules())


@needs_fp16
def test_residual_guard_on_real_flux3_class():
    """The installed diffusers FLUX.3 transformer matches the recipe's source tokens and every block takes it."""
    mod = pytest.importorskip("diffusers.models.transformers.transformer_flux3")
    guard._SUPPORTED.clear()
    assert _REAL_RECIPE_SUPPORTED(detect_family(DISTILLED), "rescale_residual") is True
    torch.manual_seed(0)
    tr = mod.Flux3Transformer2DModel(
        num_layers = 1,
        num_single_layers = 2,
        attention_head_dim = 16,
        num_attention_heads = 2,
        joint_attention_dim = 48,
        axes_dims_rope = (4, 4, 4, 4),
        flow_map = True,
    )
    x = torch.randn(1, 16, 128)
    ctx = torch.randn(1, 8, 48)
    img_ids = torch.zeros(16, 4)
    txt_ids = torch.zeros(8, 4)
    t = torch.full((1,), 0.7)
    with torch.no_grad():
        ref = tr(x, ctx, t, img_ids, txt_ids, target_timestep = t * 0.5, return_dict = False)[0]
        half = tr.half()
        assert guard.install_fp16_guard(half, "rescale_residual", torch.float16) == 5
        out = half(
            x.half(),
            ctx.half(),
            t.half(),
            img_ids,
            txt_ids,
            target_timestep = (t * 0.5).half(),
            return_dict = False,
        )[0]
    assert torch.isfinite(out).all()
    err = (out.float() - ref).abs().max() / ref.abs().max()
    assert err < 3e-2, float(err)
