# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Unit tests for ``diffusion_qwenimage21_rope.py`` (Qwen-Image-2.1 RoPE in real arithmetic when compiled).

The bit-identity tests compile the attention's QK-norm + RoPE with the stock complex form and with the
real form and compare the outputs exactly; they need CUDA and a torch whose inductor lowers ``addcmul``
to ``fma`` (2.11+). The install / guard tests run anywhere diffusers has Qwen-Image 2.1.
"""

from __future__ import annotations

import importlib
import types

import pytest

from core.inference import diffusion_qwenimage21_rope as rope

torch = pytest.importorskip("torch")
qmod = pytest.importorskip("diffusers.models.transformers.transformer_qwenimage21")


def _cuda_dtype():
    """bf16 where the card has it natively (sm80+), else fp16, as the Studio loaders pick."""
    return torch.bfloat16 if torch.cuda.get_device_capability() >= (8, 0) else torch.float16

_needs_exact = pytest.mark.skipif(
    not (torch.cuda.is_available() and rope.inductor_addcmul_is_fma()),
    reason = "needs CUDA and inductor's fma addcmul lowering (torch 2.11+)",
)


@pytest.fixture(autouse = True)
def _stock_after(monkeypatch):
    monkeypatch.delenv(rope.REAL_ROPE_ENV, raising = False)
    rope.uninstall()
    yield
    rope.uninstall()
    torch._dynamo.reset()


def _attention(device, heads = 4, dim_head = 32):
    torch.manual_seed(0)
    attn = qmod.QwenImage21Attention(dim = heads * dim_head, heads = heads, dim_head = dim_head)
    attn = attn.to(device, _cuda_dtype() if device == "cuda" else torch.bfloat16).eval()
    with torch.no_grad():
        for p in attn.parameters():
            p.normal_(0, 0.05)
        attn.norm_q.weight.normal_(1.0, 0.1)
        attn.norm_k.weight.normal_(1.0, 0.1)
    return attn


def _qk(attn, seq, device):
    g = torch.Generator(device = "cpu").manual_seed(3)
    x = torch.randn(2, seq, attn.inner_dim, generator = g).to(device, attn.to_q.weight.dtype)
    ang = torch.randn(seq, attn.inner_dim // attn.heads // 2, generator = g).to(device) * 40
    freqs = torch.polar(torch.ones_like(ang), ang)

    def prep(hidden_states, rotary_emb):
        q, k, _, _ = qmod._qwenimage21_prepare_qkv(attn, hidden_states, rotary_emb, None, None, None)
        return q, k

    return prep, x, freqs


def _compiled(prep, x, freqs, dynamic):
    torch._dynamo.reset()
    with torch.inference_mode():
        return torch.compile(prep, dynamic = dynamic)(x, freqs)


@_needs_exact
@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize("seq", [77, 1024])
def test_compiled_real_rope_is_bit_identical_to_the_complex_one(dynamic, seq):
    import torch._inductor.config as icfg

    prev = icfg.emulate_precision_casts
    icfg.emulate_precision_casts = True  # what Studio's compiled tiers set
    try:
        # Qwen-Image-2.1's own head layout: inductor's schedule for the norm in front of the RoPE depends
        # on it, and on a toy layout it can round the STOCK norm differently from eager in a few elements.
        attn = _attention("cuda", heads = 32, dim_head = 128)
        prep, x, freqs = _qk(attn, seq, "cuda")
        stock = _compiled(prep, x, freqs, dynamic)
        assert rope.install()
        real = _compiled(prep, x, freqs, dynamic)
    finally:
        icfg.emulate_precision_casts = prev
    assert torch.equal(stock[0], real[0])
    assert torch.equal(stock[1], real[1])


@_needs_exact
def test_the_real_form_runs_inside_the_compiled_block(monkeypatch):
    calls = []
    real = rope._real_rope
    monkeypatch.setattr(rope, "_real_rope", lambda x, f, fusion: calls.append(x.shape) or real(x, f, fusion))
    assert rope.install()
    attn = _attention("cuda")
    prep, x, freqs = _qk(attn, 64, "cuda")
    _compiled(prep, x, freqs, False)
    assert len(calls) == 2  # traced once for q, once for k


def _fake_fusion(monkeypatch):
    monkeypatch.setattr(rope, "inductor_addcmul_is_fma", lambda: True)
    monkeypatch.setattr(rope, "probe_fusion", lambda device: ("x", "x"))
    monkeypatch.setattr(rope, "_FUSION", {})
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)


def test_eager_calls_keep_the_complex_form(monkeypatch):
    _fake_fusion(monkeypatch)
    assert rope.install()
    monkeypatch.setattr(rope, "_real_rope", lambda *a: pytest.fail("real form used outside a compile"))
    attn = _attention("cpu")
    prep, x, freqs = _qk(attn, 16, "cpu")
    with torch.inference_mode():
        got = prep(x, freqs)
    rope.uninstall()
    with torch.inference_mode():
        want = prep(x, freqs)
    assert torch.equal(got[0], want[0]) and torch.equal(got[1], want[1])


def test_install_is_idempotent_reuses_one_wrapper_and_uninstall_restores(monkeypatch):
    _fake_fusion(monkeypatch)
    stock = qmod.apply_rotary_emb_qwen
    assert rope.install() and rope.install()
    wrapper = qmod.apply_rotary_emb_qwen
    assert wrapper is not stock and wrapper.__wrapped__ is stock
    rope.uninstall()
    assert qmod.apply_rotary_emb_qwen is stock
    assert rope.install()
    # The same object: dynamo guards on the global, so a new one would recompile every block.
    assert qmod.apply_rotary_emb_qwen is wrapper


def test_installed_diffusers_matches_the_fingerprints():
    """Fails when the pinned diffusers changes the RoPE or its caller: re-check the real form
    against the new stock one, then add the new digest."""
    assert rope.why_unsupported(qmod) is None


def test_a_drifted_rope_is_left_alone(monkeypatch):
    _fake_fusion(monkeypatch)
    fake = types.SimpleNamespace(**vars(qmod))

    def apply_rotary_emb_qwen(x, freqs_cis, use_real = True, use_real_unbind_dim = -1):
        return x

    fake.apply_rotary_emb_qwen = apply_rotary_emb_qwen
    assert "apply_rotary_emb_qwen differs" in rope.why_unsupported(fake)
    real_import = importlib.import_module
    monkeypatch.setattr(
        importlib, "import_module", lambda name, *a: fake if name == rope._MODULE else real_import(name, *a)
    )
    assert rope.install() is False
    assert fake.apply_rotary_emb_qwen is apply_rotary_emb_qwen


def test_kill_switch_and_non_fma_inductor_keep_the_complex_form(monkeypatch):
    stock = qmod.apply_rotary_emb_qwen
    monkeypatch.setattr(rope, "inductor_addcmul_is_fma", lambda: False)
    assert rope.install() is False
    assert qmod.apply_rotary_emb_qwen is stock
    _fake_fusion(monkeypatch)
    monkeypatch.setenv(rope.REAL_ROPE_ENV, "0")
    assert rope.install() is False
    assert qmod.apply_rotary_emb_qwen is stock
    monkeypatch.delenv(rope.REAL_ROPE_ENV)
    # A card whose complex multiply is neither fused form.
    monkeypatch.setattr(rope, "probe_fusion", lambda device: None)
    assert rope.install() is False
    assert qmod.apply_rotary_emb_qwen is stock


def test_addcmul_probe_reads_the_lowering():
    rope.inductor_addcmul_is_fma.cache_clear()
    try:
        from torch._inductor import lowering

        import inspect

        src = inspect.getsource(lowering.addcmul)
        want = "ops.fma(t1_val, t2_val, self_val)" in src and not torch.version.hip
    except Exception:  # noqa: BLE001 - torch without that lowering
        want = False
    assert rope.inductor_addcmul_is_fma() is want


def _fused(a, c, bd):
    """float32 fma(a, c, bd) for float32 numpy inputs in [1, 2): exact in float64, one rounding."""
    import numpy as np

    return (a.astype(np.float64) * c.astype(np.float64) + bd.astype(np.float64)).astype(np.float32)


@pytest.mark.parametrize("even", ["x", "s"])
@pytest.mark.parametrize("odd", ["x", "s"])
def test_classify_fusion_names_each_half(even, odd):
    import numpy as np

    rng = np.random.default_rng(0)
    a, b, c, d = (((rng.random(4096) + 1) * rng.choice([-1, 1], 4096)).astype(np.float32) for _ in range(4))
    real = _fused(a, c, -(b * d)) if even == "x" else _fused(-b, d, a * c)
    imag = _fused(b, c, a * d) if odd == "x" else _fused(a, d, b * c)
    assert rope.classify_fusion(a, b, c, d, real, imag) == (even, odd)
    # Unfused (two roundings) is neither form.
    assert rope.classify_fusion(a, b, c, d, a * c - b * d, imag) is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs CUDA")
def test_this_card_multiplies_complex_numbers_in_a_known_fused_form():
    assert rope.probe_fusion(torch.device("cuda")) in {("x", "x"), ("x", "s"), ("s", "x"), ("s", "s")}


@_needs_exact
@pytest.mark.parametrize("fusion", [("x", "x"), ("x", "s"), ("s", "x"), ("s", "s")])
def test_each_fusion_form_compiles_to_that_fma(fusion):
    """The compiled real form reproduces the fused form it was asked for, bit for bit, on inputs
    where the forms disagree: the check that inductor did not re-fuse the products."""
    import numpy as np

    g = torch.Generator(device = "cpu").manual_seed(1)
    x = (torch.rand(1, 64, 4, 32, generator = g) + 1) * (torch.randint(0, 2, (1, 64, 4, 32), generator = g) * 2 - 1)
    f = (torch.rand(64, 16, 2, generator = g) + 1) * (torch.randint(0, 2, (64, 16, 2), generator = g) * 2 - 1)
    freqs = torch.view_as_complex(f.contiguous())
    fn = torch.compile(lambda x, fr: rope._real_rope(x, fr, fusion))
    out = fn(x.cuda(), freqs.cuda()).cpu().numpy()
    a, b = x[..., 0::2].numpy(), x[..., 1::2].numpy()
    c = f[..., 0].unsqueeze(1).expand(64, 4, 16).unsqueeze(0).numpy()
    d = f[..., 1].unsqueeze(1).expand(64, 4, 16).unsqueeze(0).numpy()
    assert rope.classify_fusion(a, b, c, d, out[..., 0::2], out[..., 1::2]) == fusion
