# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""fp16 guard for fp16-only cards: capability resolution, kill switch, bf16 untouched, and the post-norm rescale on a
tiny block shaped like Z-Image's (CPU, real torch)."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
nn = torch.nn
F = torch.nn.functional

from core.inference import diffusion_fp16_guard as guard  # noqa: E402
from core.inference.diffusion import _resolve_diffusion_compute_dtype  # noqa: E402
from core.inference.diffusion_families import _FAMILIES, detect_family  # noqa: E402
from core.inference.video_families import _FAMILIES as _VIDEO_FAMILIES  # noqa: E402


@pytest.fixture(autouse = True)
def _clean_env(monkeypatch):
    monkeypatch.delenv(guard.FP16_GUARD_ENV, raising = False)
    # The structural probe reads the installed diffusers; pin it so the resolution tests do not depend on it.
    monkeypatch.setattr(guard, "_recipe_supported", lambda fam, recipe: True)


def _video(name):
    return next(f for f in _VIDEO_FAMILIES if f.name == name)


def test_every_declared_recipe_is_known():
    for fam in (*_FAMILIES, *_VIDEO_FAMILIES):
        recipe = getattr(fam, "fp16_guard", None)
        assert recipe is None or recipe in guard.RECIPES, (fam.name, recipe)
        # a guard only means something on a family that would otherwise be promoted
        assert recipe is None or fam.fp16_incompatible, fam.name


def test_zimage_stays_fp16_with_guard():
    z = detect_family("Tongyi-MAI/Z-Image-Turbo")
    assert z.fp16_incompatible is True and z.fp16_guard == "rescale_post_norm"
    assert guard.fp16_promotes_to_fp32(z) is False
    assert _resolve_diffusion_compute_dtype(z, torch.float16) is torch.float16


def test_unguarded_incompatible_family_still_promotes():
    krea = detect_family("krea/Krea-2-Turbo")
    assert krea.fp16_incompatible is True and krea.fp16_guard is None
    assert _resolve_diffusion_compute_dtype(krea, torch.float16) is torch.float32


def test_video_families_declare_only_native():
    # The video loader resolves the dtype but installs no guard hooks, so a patching recipe there would render black.
    for fam in _VIDEO_FAMILIES:
        assert fam.fp16_guard in (None, "native"), fam.name


def test_video_capability_flags():
    for name in ("wan2.2-ti2v-5b", "hunyuanvideo-1.5", "hunyuanvideo-1.5-720p"):
        assert guard.fp16_promotes_to_fp32(_video(name)) is False, name
    # unmeasured families, and A14B (fp16 drifts from fp32 as far as bf16 does), keep the blanket promotion
    for name in ("ltx-2", "minimax-h3", "wan2.2-t2v-a14b"):
        assert guard.fp16_promotes_to_fp32(_video(name)) is True, name


def test_kill_switch_restores_promotion(monkeypatch):
    z = detect_family("Tongyi-MAI/Z-Image-Turbo")
    monkeypatch.setenv(guard.FP16_GUARD_ENV, "0")
    assert guard.fp16_promotes_to_fp32(z) is True
    assert _resolve_diffusion_compute_dtype(z, torch.float16) is torch.float32
    assert guard.fp16_promotes_to_fp32(_video("wan2.2-ti2v-5b")) is True


def test_unsupported_diffusers_block_keeps_promotion(monkeypatch):
    monkeypatch.setattr(guard, "_recipe_supported", lambda fam, recipe: False)
    z = detect_family("Tongyi-MAI/Z-Image-Turbo")
    assert _resolve_diffusion_compute_dtype(z, torch.float16) is torch.float32
    # "native" needs no patch, so no structural probe either
    monkeypatch.undo()
    monkeypatch.setattr(guard, "_SUPPORTED", {})
    assert guard._recipe_supported(_video("wan2.2-ti2v-5b"), "native") is True


def test_bf16_and_fp32_never_change():
    for fam in (*_FAMILIES, None):
        for dt in (torch.bfloat16, torch.float32):
            assert _resolve_diffusion_compute_dtype(fam, dt) is dt


def test_non_fp16_resolution_never_probes_diffusers(monkeypatch):
    def probe(fam, recipe):
        raise AssertionError("bf16 / fp32 must not import diffusers")

    monkeypatch.setattr(guard, "_recipe_supported", probe)
    z = detect_family("Tongyi-MAI/Z-Image-Turbo")
    assert _resolve_diffusion_compute_dtype(z, torch.bfloat16) is torch.bfloat16
    assert _resolve_diffusion_compute_dtype(z, torch.float32) is torch.float32


def test_probe_closes_dynamo_import_window_before_diffusers(monkeypatch):
    import importlib

    from utils import torch_warmup

    monkeypatch.undo()
    monkeypatch.setattr(guard, "_SUPPORTED", {})
    order = []
    monkeypatch.setattr(
        torch_warmup, "close_dynamo_import_window", lambda log: order.append("close")
    )
    real = importlib.import_module

    def tracking(name, *a, **k):
        if name == "diffusers":
            order.append("diffusers")
        return real(name, *a, **k)

    monkeypatch.setattr(importlib, "import_module", tracking)
    guard._recipe_supported(detect_family("Tongyi-MAI/Z-Image-Turbo"), "rescale_post_norm")
    assert order[:2] == ["close", "diffusers"], order


class _RMS(nn.Module):
    def __init__(
        self,
        dim,
        eps = 1e-5,
    ):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        var = x.float().pow(2).mean(-1, keepdim = True)
        return (x.float() * torch.rsqrt(var + self.eps)).to(x.dtype) * self.weight.to(x.dtype)


class _Attn(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.heads = heads
        self.to_q = nn.Linear(dim, dim, bias = False)
        self.to_k = nn.Linear(dim, dim, bias = False)
        self.to_v = nn.Linear(dim, dim, bias = False)
        self.to_out = nn.ModuleList([nn.Linear(dim, dim, bias = False), nn.Dropout(0.0)])
        self.norm_q = _RMS(dim // heads)
        self.norm_k = _RMS(dim // heads)

    def forward(
        self,
        hidden_states,
        attention_mask = None,
    ):
        b, s, d = hidden_states.shape
        q, k, v = (
            p(hidden_states).view(b, s, self.heads, -1) for p in (self.to_q, self.to_k, self.to_v)
        )
        q, k = self.norm_q(q), self.norm_k(k)
        w = torch.softmax(
            (q.float().transpose(1, 2) @ k.float().permute(0, 2, 3, 1)) / (d // self.heads) ** 0.5,
            -1,
        )
        o = (w.to(v.dtype) @ v.transpose(1, 2)).transpose(1, 2).reshape(b, s, d)
        return self.to_out[1](self.to_out[0](o))


class _FFN(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden, bias = False)
        self.w2 = nn.Linear(hidden, dim, bias = False)
        self.w3 = nn.Linear(dim, hidden, bias = False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class _Block(nn.Module):
    def __init__(
        self,
        dim = 64,
        heads = 4,
        hidden = 128,
    ):
        super().__init__()
        self.attention = _Attn(dim, heads)
        self.feed_forward = _FFN(dim, hidden)
        self.attention_norm1, self.attention_norm2 = _RMS(dim), _RMS(dim)
        self.ffn_norm1, self.ffn_norm2 = _RMS(dim), _RMS(dim)

    def forward(self, x):
        x = x + 0.5 * self.attention_norm2(self.attention(self.attention_norm1(x)))
        return x + 0.5 * self.ffn_norm2(self.feed_forward(self.ffn_norm1(x)))


def _overflowing_model():
    torch.manual_seed(0)
    model = nn.Sequential(_Block(), _Block())
    with torch.no_grad():
        for blk in model:
            # Z-Image-like: the FFN gate product and output, and the attention output, exceed float16's range.
            blk.feed_forward.w1.weight.mul_(40.0)
            blk.feed_forward.w3.weight.mul_(40.0)
            blk.feed_forward.w2.weight.mul_(400.0)
            blk.attention.to_v.weight.mul_(300.0)
            blk.attention.to_out[0].weight.mul_(300.0)
    return model


def _fp16_ok():
    try:
        nn.Linear(4, 4).half()(torch.ones(1, 4, dtype = torch.float16))
        return True
    except Exception:  # noqa: BLE001
        return False


needs_fp16 = pytest.mark.skipif(not _fp16_ok(), reason = "this torch has no CPU float16 matmul")


@needs_fp16
def test_guard_keeps_overflowing_block_finite_and_accurate():
    model = _overflowing_model()
    x = torch.randn(2, 16, 64)
    with torch.no_grad():
        ref = model.float()(x.float())
        assert ref.abs().max() < 1e3  # the residual stream itself is in range
        branch = model[0].feed_forward(model[0].ffn_norm1(x.float()))
        assert branch.abs().max() > 65504  # ...the pre-norm branch is not

        half = model.half()
        stock = half(x.half())
        assert not torch.isfinite(stock).all()  # stock fp16 overflows (black image)

        assert guard.install_fp16_guard(half, "rescale_post_norm", torch.float16) == 2
        fixed = half(x.half())
    assert torch.isfinite(fixed).all()
    err = (fixed.float() - ref).abs().max() / ref.abs().max()
    assert err < 2e-2, float(err)


@needs_fp16
def test_guard_is_idempotent_and_removable():
    model = _overflowing_model().half()
    eps_before = [m.eps for m in model.modules() if isinstance(m, _RMS)]
    assert guard.install_fp16_guard(model, "rescale_post_norm", torch.float16) == 2
    assert guard.install_fp16_guard(model, "rescale_post_norm", torch.float16) == 2
    assert model[0].attention_norm2.eps == 1e-5 / 16**2
    assert model[0].ffn_norm2.eps == 1e-5 / 128**2
    assert guard.remove_fp16_guard(model) == 2
    assert [m.eps for m in model.modules() if isinstance(m, _RMS)] == eps_before
    assert "forward" not in model[0].feed_forward.__dict__
    assert not model[0].attention._forward_pre_hooks and not model[0].attention._forward_hooks


@pytest.mark.parametrize("dtype", ["bfloat16", "float32"])
def test_guard_never_touches_non_fp16(dtype):
    dt = getattr(torch, dtype)
    torch.manual_seed(0)
    model = nn.Sequential(_Block(), _Block()).to(dt)
    x = torch.randn(2, 16, 64, dtype = dt)
    with torch.no_grad():
        before = model(x)
        assert guard.install_fp16_guard(model, "rescale_post_norm", dt) == 0
        after = model(x)
    assert torch.equal(before, after)
    assert all(m.eps == 1e-5 for m in model.modules() if isinstance(m, _RMS))
    assert not model[0].attention._forward_pre_hooks


@needs_fp16
def test_guard_inert_under_kill_switch_and_native(monkeypatch):
    model = _overflowing_model().half()
    assert guard.install_fp16_guard(model, "native", torch.float16) == 0
    assert guard.install_fp16_guard(model, None, torch.float16) == 0
    monkeypatch.setenv(guard.FP16_GUARD_ENV, "0")
    assert guard.install_fp16_guard(model, "rescale_post_norm", torch.float16) == 0
    assert not model[0].attention._forward_pre_hooks


def test_block_with_bias_is_not_guarded():
    torch.manual_seed(0)
    blk = _Block()
    blk.feed_forward.w2 = nn.Linear(128, 64, bias = True)
    assert guard.install_fp16_guard(nn.Sequential(blk), "rescale_post_norm", torch.float16) == 0


@needs_fp16
def test_guard_matches_real_zimage_block():
    """The recipe still recognises the installed diffusers ZImageTransformerBlock and is a no-op on in-range input."""
    zmod = pytest.importorskip("diffusers.models.transformers.transformer_z_image")
    z = detect_family("Tongyi-MAI/Z-Image-Turbo")
    guard._SUPPORTED.clear()
    torch.manual_seed(0)
    blk = zmod.ZImageTransformerBlock(0, 64, 4, 4, 1e-5, True, modulation = False).half()
    x = torch.randn(1, 8, 64, dtype = torch.float16)
    freqs = torch.polar(torch.ones(1, 8, 8), torch.randn(1, 8, 8))
    with torch.no_grad():
        before = blk(x, None, freqs).float()
        assert guard.install_fp16_guard(nn.Sequential(blk), z.fp16_guard, torch.float16) == 1
        after = blk(x, None, freqs).float()
    assert torch.isfinite(after).all()
    assert (after - before).abs().max() < 2e-2 * before.abs().max()


def test_real_diffusers_zimage_source_matches_recipe(monkeypatch):
    pytest.importorskip("diffusers.models.transformers.transformer_z_image")
    monkeypatch.undo()
    monkeypatch.setattr(guard, "_SUPPORTED", {})
    assert (
        guard._recipe_supported(detect_family("Tongyi-MAI/Z-Image-Turbo"), "rescale_post_norm")
        is True
    )


class _LoRALinear(nn.Module):
    """base + B @ A; peft's own injection imports torchao symbols some versions lack."""

    def __init__(self, base):
        super().__init__()
        self.base = base
        self.lora_A = nn.Linear(base.in_features, 4, bias = False)
        self.lora_B = nn.Linear(4, base.out_features, bias = False)

    def forward(self, x):
        return self.base(x) + self.lora_B(self.lora_A(x))


@needs_fp16
def test_guard_holds_through_a_lora_injected_after_install():
    """LoRA wrappers replace the Linear children after the guard is installed; the rescale lives on the parents, so the
    adapted branch is scaled as a whole."""
    targets = ["w1", "w2", "w3", "to_q", "to_k", "to_v", "to_out.0"]

    def adapted(model):
        torch.manual_seed(1)
        for name, module in list(model.named_modules()):
            if isinstance(module, nn.Linear) and any(name.endswith(t) for t in targets):
                parent, _, child = name.rpartition(".")
                setattr(model.get_submodule(parent), child, _LoRALinear(module))
        return model

    x = torch.randn(2, 16, 64)
    with torch.no_grad():
        ref_model = adapted(_overflowing_model()).float()
        ref = ref_model(x.float())
        half = _overflowing_model().half()
        assert guard.install_fp16_guard(half, "rescale_post_norm", torch.float16) == 2
        half = adapted(half.float()).half()
        assert isinstance(
            half.get_submodule(next(n for n, m in half.named_modules() if n.endswith("w3"))),
            _LoRALinear,
        )
        fixed = half(x.half())
    assert torch.isfinite(fixed).all()
    err = (fixed.float() - ref).abs().max() / ref.abs().max()
    assert err < 2e-2, float(err)
