# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Numerical + lifecycle tests for the per-arch eager fusions (``diffusion_arch_patches``).

Each per-arch patch only fuses ``a + b*c`` -> ``torch.addcmul`` (1-ULP, more accurate), so
the patched method/forward must match the stock diffusers one within fp tolerance. We also
check install/uninstall reversibility + idempotency, the kill-switch, and the body-drift
guard (a diffusers whose block body changed is left unpatched). Runs on CPU.
"""

from __future__ import annotations

import types

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("diffusers")

from core.inference import diffusion_arch_patches as ap  # noqa: E402


@pytest.fixture(autouse=True)
def _clean():
    ap.uninstall_arch_patches()
    yield
    ap.uninstall_arch_patches()


# ── qwen-image _modulate (modulation addcmul) ───────────────────────────────────


def test_qwen_modulate_matches_stock_global_and_indexed():
    from diffusers.models.transformers.transformer_qwenimage import QwenImageTransformerBlock as Q

    B, L, D = 2, 16, 64
    x = torch.randn(B, L, D)
    mod = torch.randn(B, 3 * D)
    # _modulate uses no real `self` state, so call it unbound with self=None.
    ref_x, ref_g = Q._modulate(None, x, mod)
    got_x, got_g = ap._qwen_modulate(None, x, mod)
    torch.testing.assert_close(got_x, ref_x, atol=1e-5, rtol=1e-4)
    assert torch.equal(got_g, ref_g)

    # per-token `index` branch (mod batch is 2*B).
    idx = torch.randint(0, 2, (B, L))
    mod2 = torch.randn(2 * B, 3 * D)
    ref2_x, ref2_g = Q._modulate(None, x, mod2, idx)
    got2_x, got2_g = ap._qwen_modulate(None, x, mod2, idx)
    torch.testing.assert_close(got2_x, ref2_x, atol=1e-5, rtol=1e-4)
    assert torch.equal(got2_g, ref2_g)


# ── z-image block forward (gated-residual addcmul) ──────────────────────────────


class _AttnStub(torch.nn.Module):
    """Deterministic stand-in for ZImageAttention so the block forward runs without RoPE /
    freqs_cis (the patch only changes the residual adds, which is what we validate)."""

    def __init__(self, dim):
        super().__init__()
        self.proj = torch.nn.Linear(dim, dim, bias=False)

    def forward(self, h, **kwargs):
        return self.proj(h)


def _zimage_block(dim=64, heads=4):
    from diffusers.models.transformers.transformer_z_image import ZImageTransformerBlock

    blk = ZImageTransformerBlock(
        layer_id=0, dim=dim, n_heads=heads, n_kv_heads=heads,
        norm_eps=1e-5, qk_norm=True, modulation=True,
    ).eval()
    blk.attention = _AttnStub(dim).eval()
    return blk


def _adaln_dim(dim):
    from diffusers.models.transformers.transformer_z_image import ADALN_EMBED_DIM
    return min(dim, ADALN_EMBED_DIM)


def test_zimage_forward_matches_stock_global_modulation():
    from diffusers.models.transformers.transformer_z_image import ZImageTransformerBlock

    torch.manual_seed(0)
    blk = _zimage_block()
    B, L, D = 2, 16, 64
    x = torch.randn(B, L, D)
    adaln = torch.randn(B, _adaln_dim(D))

    with torch.inference_mode():
        ref = ZImageTransformerBlock.forward(blk, x, None, None, adaln_input=adaln).clone()
        got = ap._zimage_forward(blk, x, None, None, adaln_input=adaln)
    torch.testing.assert_close(got, ref, atol=1e-5, rtol=1e-4)


def test_zimage_forward_matches_stock_per_token_modulation():
    from diffusers.models.transformers.transformer_z_image import ZImageTransformerBlock

    torch.manual_seed(1)
    blk = _zimage_block()
    B, L, D = 2, 16, 64
    x = torch.randn(B, L, D)
    ad = _adaln_dim(D)
    adaln_noisy = torch.randn(B, ad)
    adaln_clean = torch.randn(B, ad)
    noise_mask = torch.randint(0, 2, (B, L))

    with torch.inference_mode():
        ref = ZImageTransformerBlock.forward(
            blk, x, None, None, noise_mask=noise_mask,
            adaln_noisy=adaln_noisy, adaln_clean=adaln_clean,
        ).clone()
        got = ap._zimage_forward(
            blk, x, None, None, noise_mask=noise_mask,
            adaln_noisy=adaln_noisy, adaln_clean=adaln_clean,
        )
    torch.testing.assert_close(got, ref, atol=1e-5, rtol=1e-4)


# ── flux.1 / flux.2 block forwards (modulation + gated-residual addcmul) ─────────


class _Tuple2AttnStub(torch.nn.Module):
    """Double-stream attention stub -> (img_out, ctx_out)."""

    def __init__(self, dim):
        super().__init__()
        self.pi = torch.nn.Linear(dim, dim, bias=False)
        self.pc = torch.nn.Linear(dim, dim, bias=False)

    def forward(self, hidden_states, encoder_hidden_states=None, **kwargs):
        return self.pi(hidden_states), self.pc(encoder_hidden_states)


class _SingleAttnStub(torch.nn.Module):
    """Single-stream attention stub -> tensor."""

    def __init__(self, dim):
        super().__init__()
        self.p = torch.nn.Linear(dim, dim, bias=False)

    def forward(self, hidden_states, **kwargs):
        return self.p(hidden_states)


def _close_any(got, ref):
    if isinstance(ref, tuple):
        assert len(got) == len(ref)
        for g, r in zip(got, ref):
            torch.testing.assert_close(g, r, atol=1e-5, rtol=1e-4)
    else:
        torch.testing.assert_close(got, ref, atol=1e-5, rtol=1e-4)


D, H = 64, 4
B, L, LC = 2, 16, 8


def test_flux_double_forward_matches_stock():
    from diffusers.models.transformers.transformer_flux import FluxTransformerBlock

    torch.manual_seed(0)
    blk = FluxTransformerBlock(dim=D, num_attention_heads=H, attention_head_dim=D // H).eval()
    blk.attn = _Tuple2AttnStub(D).eval()
    hs, ehs, temb = torch.randn(B, L, D), torch.randn(B, LC, D), torch.randn(B, D)
    with torch.inference_mode():
        ref = FluxTransformerBlock.forward(blk, hs, ehs, temb)
        got = ap._flux_double_forward(blk, hs, ehs, temb)
    _close_any(got, ref)


def test_flux_single_forward_matches_stock():
    from diffusers.models.transformers.transformer_flux import FluxSingleTransformerBlock

    torch.manual_seed(1)
    blk = FluxSingleTransformerBlock(dim=D, num_attention_heads=H, attention_head_dim=D // H).eval()
    blk.attn = _SingleAttnStub(D).eval()
    hs, ehs, temb = torch.randn(B, L, D), torch.randn(B, LC, D), torch.randn(B, D)
    with torch.inference_mode():
        ref = FluxSingleTransformerBlock.forward(blk, hs, ehs, temb)
        got = ap._flux_single_forward(blk, hs, ehs, temb)
    _close_any(got, ref)


def test_flux2_double_forward_matches_stock():
    from diffusers.models.transformers.transformer_flux2 import Flux2TransformerBlock

    torch.manual_seed(2)
    blk = Flux2TransformerBlock(dim=D, num_attention_heads=H, attention_head_dim=D // H).eval()
    blk.attn = _Tuple2AttnStub(D).eval()
    hs, ehs = torch.randn(B, L, D), torch.randn(B, LC, D)
    tmi, tmt = torch.randn(B, 6 * D), torch.randn(B, 6 * D)
    with torch.inference_mode():
        ref = Flux2TransformerBlock.forward(blk, hs, ehs, tmi, tmt)
        got = ap._flux2_double_forward(blk, hs, ehs, tmi, tmt)
    _close_any(got, ref)


def test_flux2_single_forward_matches_stock():
    from diffusers.models.transformers.transformer_flux2 import Flux2SingleTransformerBlock

    torch.manual_seed(3)
    blk = Flux2SingleTransformerBlock(dim=D, num_attention_heads=H, attention_head_dim=D // H).eval()
    blk.attn = _SingleAttnStub(D).eval()
    hs, ehs, tm = torch.randn(B, L, D), torch.randn(B, LC, D), torch.randn(B, 3 * D)
    with torch.inference_mode():
        ref = Flux2SingleTransformerBlock.forward(blk, hs, ehs, tm)
        got = ap._flux2_single_forward(blk, hs, ehs, tm)
    _close_any(got, ref)


# ── lifecycle ───────────────────────────────────────────────────────────────────


def test_install_idempotent_and_reversible():
    from diffusers.models.transformers.transformer_qwenimage import QwenImageTransformerBlock as Q
    from diffusers.models.transformers.transformer_z_image import ZImageTransformerBlock as Z

    q_orig, z_orig = Q._modulate, Z.forward
    n1 = ap.install_arch_patches()
    n2 = ap.install_arch_patches()  # idempotent
    assert n1 == 6 and n2 == n1  # qwen + z-image + flux.1 x2 + flux.2 x2
    assert Q._modulate is not q_orig and Z.forward is not z_orig
    assert ap.is_installed()

    ap.uninstall_arch_patches()
    assert not ap.is_installed()
    assert Q._modulate is q_orig and Z.forward is z_orig  # exact restore
    ap.uninstall_arch_patches()  # idempotent


def test_kill_switch(monkeypatch):
    from diffusers.models.transformers.transformer_qwenimage import QwenImageTransformerBlock as Q

    monkeypatch.setenv("UNSLOTH_DIFFUSION_ARCH_PATCHES", "0")
    orig = Q._modulate
    assert ap.install_arch_patches() == 0
    assert not ap.is_installed()
    assert Q._modulate is orig


def test_body_drift_guard_skips_changed_block(monkeypatch):
    # If a resolver's body-check fails (diffusers changed the lines we rewrite), that patch
    # is skipped. Force the qwen resolver to see a drifted body.
    monkeypatch.setattr(ap, "_body_has", lambda fn, *needles: False)
    assert ap.install_arch_patches() == 0
    assert not ap.is_installed()
