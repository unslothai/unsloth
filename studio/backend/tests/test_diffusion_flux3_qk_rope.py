# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""CPU checks for diffusion_flux3_qk_rope gates; kernel bit-exactness is proven on the card by ``verdict``."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
tf3 = pytest.importorskip("diffusers.models.transformers.transformer_flux3")

from core.inference import diffusion_flux3_qk_rope as qr  # noqa: E402


@pytest.fixture(autouse = True)
def _clean(monkeypatch):
    monkeypatch.delenv(qr.QK_NORM_ROPE_ENV, raising = False)
    monkeypatch.delenv(qr.PATCH_ENV, raising = False)
    qr.reset()
    qr.STATS.update(fused = 0, stock = 0)
    yield
    qr.uninstall()
    qr.reset()


def _attn(dtype = torch.float32):
    torch.manual_seed(1)
    attn = tf3.Flux3Attention(query_dim = 64, heads = 4, dim_head = 16).to(dtype).eval()
    with torch.no_grad():
        attn.norm_q.weight.uniform_(0.5, 3.0)
        attn.norm_k.weight.uniform_(0.5, 3.0)
    return attn


def _rope(n, d = 16):
    g = torch.Generator().manual_seed(2)
    ang = torch.rand(n, d // 2, generator = g) * 60
    return torch.cos(ang).repeat_interleave(2, dim = -1), torch.sin(ang).repeat_interleave(2, dim = -1)


def _tiny():
    torch.manual_seed(0)
    return tf3.Flux3Transformer2DModel(
        in_channels = 8,
        cond_in_channels = 8,
        num_layers = 1,
        num_single_layers = 2,
        attention_head_dim = 16,
        num_attention_heads = 2,
        joint_attention_dim = 24,
        timestep_channels = 16,
        axes_dims_rope = (4, 4, 4, 4),
    ).eval()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("wdtype", [torch.float32, torch.float16])
def test_reference_is_the_stock_chain_bitwise(dtype, wdtype):
    g = torch.Generator().manual_seed(5)
    x = (torch.randn(2, 33, 3, 16, generator = g) * 10).to(dtype)
    norm = torch.nn.RMSNorm(16, eps = 1e-6).to(wdtype)
    with torch.no_grad():
        norm.weight.uniform_(0.5, 3.0)
        cos, sin = _rope(33)
        want = tf3.apply_rotary_emb(norm(x.float()), (cos, sin), sequence_dim = 1).to(dtype)
        got = qr.reference_qk(x, norm.weight, 1e-6, cos, sin, dtype)
    assert torch.equal(got, want)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_processor_on_cpu_is_the_stock_processor_bitwise(dtype):
    attn = _attn(dtype)
    stock = attn.processor
    x = torch.randn(2, 21, 64, generator = torch.Generator().manual_seed(3)).to(dtype)
    rope = _rope(21)
    with torch.no_grad():
        want = stock(attn, x, None, rope)
        got = qr.processor_class()()(attn, x, None, rope)
    assert torch.equal(got, want)
    assert qr.STATS == {"fused": 0, "stock": 1}


def _fake_card(
    monkeypatch,
    verdict = None,
    launch = None,
):
    """Pretend the CPU tensors are on a proven card: the fused branch runs ``launch`` (default: the reference)."""
    calls = []
    monkeypatch.setattr(qr, "eligible", lambda *a, **k: True)
    monkeypatch.setattr(
        qr,
        "verdict",
        lambda *a, **k: {"exact": True, "faster": True} if verdict is None else verdict,
    )

    def _launch(
        x,
        w,
        eps,
        cos,
        sin,
        out_dtype = None,
    ):
        calls.append(x.shape)
        if launch is not None:
            return launch(x, w, eps, cos, sin, out_dtype)
        return qr.reference_qk(x, w, eps, cos, sin, out_dtype or x.dtype)

    monkeypatch.setattr(qr, "launch", _launch)
    return calls


def _qk(
    attn,
    seq = 9,
    dtype = torch.float16,
):
    g = torch.Generator().manual_seed(7)
    q = (torch.randn(1, seq, 4, 16, generator = g) * 5).to(dtype)
    k = (torch.randn(1, seq, 4, 16, generator = g) * 5).to(dtype)
    return q, k


def _stock_qk(attn, q, k, rope, dtype):
    return (
        tf3.apply_rotary_emb(attn.norm_q(q.float()), rope, sequence_dim = 1).to(dtype),
        tf3.apply_rotary_emb(attn.norm_k(k.float()), rope, sequence_dim = 1).to(dtype),
    )


def test_fused_branch_engages_only_on_a_proven_card(monkeypatch):
    attn = _attn(torch.float16)
    q, k = _qk(attn)
    rope = _rope(9)
    calls = _fake_card(monkeypatch)
    with torch.no_grad():
        got = qr.qk_norm_rope(attn, q, k, torch.float16, rope, tf3.apply_rotary_emb)
        want = _stock_qk(attn, q, k, rope, torch.float16)
    assert len(calls) == 2 and qr.STATS["fused"] == 1
    assert all(torch.equal(a, b) for a, b in zip(got, want))


@pytest.mark.parametrize(
    "verdict", [{"exact": False, "faster": True}, {"exact": True, "faster": False}, None]
)
def test_unproven_or_slower_or_unprobed_card_takes_the_stock_chain(monkeypatch, verdict):
    attn = _attn(torch.float16)
    q, k = _qk(attn)
    calls = _fake_card(monkeypatch)
    monkeypatch.setattr(qr, "verdict", lambda *a, **k: verdict)
    with torch.no_grad():
        qr.qk_norm_rope(attn, q, k, torch.float16, _rope(9), tf3.apply_rotary_emb)
    assert calls == [] and qr.STATS == {"fused": 0, "stock": 1}


@pytest.mark.parametrize("env", ["QK_NORM_ROPE_ENV", "PATCH_ENV"])
@pytest.mark.parametrize("value", ["0", "off", "false", "no"])
def test_kill_switch(monkeypatch, value, env):
    attn = _attn(torch.float16)
    q, k = _qk(attn)
    calls = _fake_card(monkeypatch)
    monkeypatch.setenv(getattr(qr, env), value)
    with torch.no_grad():
        qr.qk_norm_rope(attn, q, k, torch.float16, _rope(9), tf3.apply_rotary_emb)
    assert calls == [] and qr.STATS["stock"] == 1
    assert not qr.wanted("cuda")


def test_compiling_frames_and_missing_rope_take_the_stock_lines(monkeypatch):
    attn = _attn(torch.float16)
    q, k = _qk(attn)
    calls = _fake_card(monkeypatch)
    with torch.no_grad():
        qr.qk_norm_rope(attn, q, k, torch.float16, None, tf3.apply_rotary_emb)
        monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
        qr.qk_norm_rope(attn, q, k, torch.float16, _rope(9), tf3.apply_rotary_emb)
    assert calls == []
    assert qr.STATS["stock"] == 1  # a compiling frame does not count


def test_patched_or_hooked_norms_take_the_stock_chain(monkeypatch):
    attn = _attn(torch.float16)
    q, k = _qk(attn)
    calls = _fake_card(monkeypatch)
    attn.norm_q.forward = (
        lambda x: x
    )  # instance override: the stock chain is no longer torch's RMSNorm
    with torch.no_grad():
        qr.qk_norm_rope(attn, q, k, torch.float16, _rope(9), tf3.apply_rotary_emb)
    del attn.norm_q.forward
    h = attn.norm_k.register_forward_hook(lambda *a: None)
    with torch.no_grad():
        qr.qk_norm_rope(attn, q, k, torch.float16, _rope(9), tf3.apply_rotary_emb)
    h.remove()
    assert qr.STATS == {"fused": 0, "stock": 2}
    assert len(calls) <= 2


def test_a_failed_launch_falls_back_and_stays_stock(monkeypatch):
    attn = _attn(torch.float16)
    q, k = _qk(attn)
    rope = _rope(9)

    def boom(*a, **k):
        raise RuntimeError("ptxas died")

    calls = _fake_card(monkeypatch, launch = boom)
    with torch.no_grad():
        got = qr.qk_norm_rope(attn, q, k, torch.float16, rope, tf3.apply_rotary_emb)
        qr.qk_norm_rope(attn, q, k, torch.float16, rope, tf3.apply_rotary_emb)
        want = _stock_qk(attn, q, k, rope, torch.float16)
    assert len(calls) == 1
    assert all(torch.equal(a, b) for a, b in zip(got, want))
    assert qr.STATS == {"fused": 0, "stock": 2}


def test_first_real_call_is_checked_once_and_a_mismatch_disables_the_key(monkeypatch):
    attn = _attn(torch.float16)
    q, k = _qk(attn)
    rope = _rope(9)
    found = {"exact": True, "faster": True}
    off_by_one = lambda x, w, eps, c, s, dt: qr.reference_qk(x, w, eps, c, s, dt or x.dtype) + 1
    calls = _fake_card(monkeypatch, launch = off_by_one)
    monkeypatch.setattr(qr, "verdict", lambda *a, **k: found)
    with torch.no_grad():
        got = qr.qk_norm_rope(attn, q, k, torch.float16, rope, tf3.apply_rotary_emb)
        want = _stock_qk(attn, q, k, rope, torch.float16)
        qr.qk_norm_rope(attn, q, k, torch.float16, rope, tf3.apply_rotary_emb)
    assert all(torch.equal(a, b) for a, b in zip(got, want))
    assert found["real_checked"] and not found["real_exact"] and not found["exact"]
    assert len(calls) == 1 and qr.STATS == {"fused": 0, "stock": 2}


def test_eligibility_rules():
    w = torch.ones(16)
    cos, sin = _rope(9)
    x = torch.zeros(1, 9, 4, 16, dtype = torch.float16)
    assert not qr.eligible(x, w, cos, sin, torch.float16)
    assert not qr.eligible(x.float(), w, cos, sin, torch.float32)


def test_install_swaps_stock_processors_and_uninstall_restores(monkeypatch):
    model = _tiny()
    pipe = SimpleNamespace(transformer = model)
    assert qr.install_for_pipe(pipe, "cpu") == 0
    monkeypatch.setattr(qr, "wanted", lambda device = "cuda": True)
    mods = [m for m in model.modules() if isinstance(m, tf3.Flux3Attention)]
    for m in mods:
        m.processor._attention_backend = "_native_cudnn"
    n = qr.install_for_pipe(pipe, "cuda")
    assert n == len(mods) > 0
    cls = qr.processor_class()
    for m in mods:
        assert type(m.processor) is cls and qr.is_qk_processor(m.processor)
        assert m.processor._attention_backend == "_native_cudnn"
    assert qr.install_for_pipe(pipe, "cuda") == 0  # idempotent
    qr.uninstall()
    for m in mods:
        assert type(m.processor) is tf3.Flux3AttnProcessor
        assert m.processor._attention_backend == "_native_cudnn"


def test_install_declines_a_changed_diffusers_processor(monkeypatch):
    from core.inference import diffusion_flux3_attn as fa

    monkeypatch.setattr(qr, "wanted", lambda device = "cuda": True)
    monkeypatch.setattr(fa, "_PROCESSOR_DIGESTS", frozenset({"0000000000000000"}))
    assert qr.install_for_pipe(SimpleNamespace(transformer = _tiny()), "cuda") == 0
    assert qr.install_for_pipe(SimpleNamespace(transformer = torch.nn.Linear(2, 2)), "cuda") == 0


def test_whole_transformer_forward_is_unchanged_on_cpu(monkeypatch):
    model = _tiny()
    torch.manual_seed(4)
    hs = torch.randn(1, 6, 8)
    ehs = torch.randn(1, 5, 24)
    img_ids = torch.zeros(6, 4)
    img_ids[:, 1] = torch.arange(6)
    txt_ids = torch.zeros(5, 4)
    txt_ids[:, 3] = torch.arange(5)
    kw = dict(
        hidden_states = hs,
        encoder_hidden_states = ehs,
        timestep = torch.tensor([0.5]),
        img_ids = img_ids,
        txt_ids = txt_ids,
    )
    try:
        with torch.no_grad():
            want = model(**kw).sample
    except TypeError as exc:
        pytest.skip(f"forward signature: {exc}")
    monkeypatch.setattr(qr, "wanted", lambda device = "cuda": True)
    assert qr.install_for_pipe(SimpleNamespace(transformer = model), "cuda") > 0
    with torch.no_grad():
        got = model(**kw).sample
    assert torch.equal(got, want)


def test_strided_and_convrot_paths_accept_the_swapped_processor():
    from core.inference import diffusion_flux_convrot_fused as cf

    proc = qr.processor_class()()
    assert qr.is_qk_processor(proc)
    assert not qr.is_qk_processor(tf3.Flux3AttnProcessor())
    assert cf.is_qk_processor is qr.is_qk_processor


def test_registered_in_the_fused_dit_group():
    from core.inference.diffusion_autoopt import patches as ap

    spec = {s.name: s for s in ap.specs(ap.STAGE_LOAD)}["flux3_qk_rope"]
    assert spec.group == ap.GROUP_FUSED_DIT
    assert spec.env == qr.QK_NORM_ROPE_ENV
    assert spec.backends == ("image",)
