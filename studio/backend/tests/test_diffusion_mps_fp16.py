# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""MPS fp16-headroom compute for Z-Image: engages only on MPS + bf16 + the env opt-in, the headroom identities are exact
(fp32 compute reproduces the stock forward bit for bit), fp16 compute stays finite where plain fp16 overflows, and
CUDA / CPU targets are never touched. Real diffusers Z-Image modules on CPU."""

from __future__ import annotations

import types

import pytest

torch = pytest.importorskip("torch")
zmod = pytest.importorskip("diffusers.models.transformers.transformer_z_image")
nn = torch.nn

from core.inference import diffusion_mps_fp16 as mf  # noqa: E402

MPS_BF16 = types.SimpleNamespace(device = "mps", dtype = torch.bfloat16)
CFG = dict(
    all_patch_size = (2,),
    all_f_patch_size = (1,),
    in_channels = 16,
    dim = 128,
    n_layers = 2,
    n_refiner_layers = 1,
    n_heads = 4,
    n_kv_heads = 4,
    norm_eps = 1e-5,
    qk_norm = True,
    cap_feat_dim = 32,
    rope_theta = 256.0,
    t_scale = 1000.0,
    axes_dims = [8, 12, 12],
    axes_lens = [256, 64, 64],
)


def _fp16_ok():
    try:
        nn.Linear(4, 4).half()(torch.ones(1, 4, dtype = torch.float16))
        return True
    except Exception:  # noqa: BLE001
        return False


needs_fp16 = pytest.mark.skipif(not _fp16_ok(), reason = "this torch has no CPU float16 matmul")


@pytest.fixture(autouse = True)
def _opt_in(monkeypatch):
    monkeypatch.setenv(mf.MPS_FP16_ENV, "1")


def _model(dtype = torch.bfloat16, seed = 0):
    torch.manual_seed(seed)
    model = zmod.ZImageTransformer2DModel(**CFG).eval()
    with torch.no_grad():
        for p in model.parameters():
            if p.dim() == 2:
                p.normal_(0, 0.05)
    return model.to(dtype)


def _inputs(dtype):
    g = torch.Generator().manual_seed(1)
    x = torch.randn(16, 1, 16, 16, generator = g).to(dtype)
    cap = torch.randn(24, 32, generator = g).to(dtype)
    return [x], torch.tensor([0.5]).to(dtype), [cap]


def _run(model, dtype):
    x, t, cap = _inputs(dtype)
    with torch.no_grad():
        out = model(x, t, cap, return_dict = False)[0]
    return (out[0] if isinstance(out, (list, tuple)) else out).float()


def _snapshot(model):
    procs = [type(m.processor) for m in model.modules() if hasattr(m, "processor")]
    ffn = [("forward" in m.__dict__) for m in model.modules() if isinstance(m, zmod.FeedForward)]
    sd = {k: (v.dtype, v.clone()) for k, v in model.state_dict().items()}
    return procs, ffn, sd


def _same(a, b):
    assert a[0] == b[0] and a[1] == b[1]
    assert a[2].keys() == b[2].keys()
    for k in a[2]:
        assert a[2][k][0] == b[2][k][0], k
        assert torch.equal(a[2][k][1], b[2][k][1]), k




@pytest.mark.parametrize(
    "target",
    [
        types.SimpleNamespace(device = "cuda", dtype = torch.bfloat16),
        types.SimpleNamespace(device = "cpu", dtype = torch.bfloat16),
        types.SimpleNamespace(device = "xpu", dtype = torch.bfloat16),
        types.SimpleNamespace(device = "mps", dtype = torch.float32),
        types.SimpleNamespace(device = "mps", dtype = torch.float16),
    ],
)
def test_never_engages_off_mps_bf16(target):
    model = _model()
    before = _snapshot(model)
    assert mf.unsupported_reason(model, target) is not None
    assert mf.install_mps_fp16(model, target) == 0
    _same(before, _snapshot(model))
    assert not mf.is_installed(model)


@pytest.mark.parametrize("value", [None, "", "0", "off", "false"])
def test_needs_the_env_opt_in(monkeypatch, value):
    if value is None:
        monkeypatch.delenv(mf.MPS_FP16_ENV, raising = False)
    else:
        monkeypatch.setenv(mf.MPS_FP16_ENV, value)
    model = _model()
    before = _snapshot(model)
    assert mf.install_mps_fp16(model, MPS_BF16) == 0
    _same(before, _snapshot(model))


def test_other_transformers_are_never_patched():
    root = nn.Sequential(nn.Linear(8, 8).to(torch.bfloat16))
    assert mf.unsupported_reason(root, MPS_BF16) == "transformer is not Z-Image"
    assert mf.install_mps_fp16(root, MPS_BF16) == 0


def test_engages_on_mps_bf16_and_keeps_the_stream_bf16():
    model = _model()
    n_blocks = CFG["n_layers"] + 2 * CFG["n_refiner_layers"]
    assert mf.unsupported_reason(model, MPS_BF16) is None
    assert mf.install_mps_fp16(model, MPS_BF16) == n_blocks
    assert mf.install_mps_fp16(model, MPS_BF16) == n_blocks  # idempotent
    assert mf.is_installed(model)
    assert model.dtype is torch.bfloat16
    fp16 = ("attention.to_q", "attention.to_k", "attention.to_v", "attention.to_out.0", "attention.norm_q",
            "attention.norm_k", "feed_forward.w1", "feed_forward.w2", "feed_forward.w3")
    for name, p in model.named_parameters():
        want = torch.float16 if any(f"{s}." in name for s in fp16) else torch.bfloat16
        assert p.dtype is want, name
    blk = model.layers[0]
    assert type(blk.attention.processor) is mf.processor_cls()
    assert "forward" in blk.feed_forward.__dict__


def test_remove_restores_the_stock_model():
    model = _model()
    before = _snapshot(model)
    assert mf.install_mps_fp16(model, MPS_BF16) > 0
    assert mf.remove_mps_fp16(model) == CFG["n_layers"] + 2 * CFG["n_refiner_layers"]
    after = _snapshot(model)
    assert before[0] == after[0] and before[1] == after[1] and not mf.is_installed(model)
    for k, (dtype, t) in before[2].items():
        assert after[2][k][0] == dtype, k
        # bf16 -> fp16 -> bf16 is exact except below fp16's normal range (|w| < 2**-14), off by < 2**-24 there.
        assert (after[2][k][1].float() - t.float()).abs().max() <= 2**-24, k


def test_attention_backend_survives_the_processor_swap():
    model = _model()
    for m in model.modules():
        if hasattr(m, "processor"):
            m.processor._attention_backend = "native"
    mf.install_mps_fp16(model, MPS_BF16)
    procs = [m.processor for m in model.modules() if hasattr(m, "processor")]
    assert procs and all(p._attention_backend == "native" for p in procs)


def test_quantized_storage_keeps_the_whole_model_bf16():
    """GGUF / torchao / bnb leaves would get fp16 activations against bf16 dequant: refuse the model, all or nothing."""
    model = _model()

    class _Gguf(nn.Linear):
        compute_dtype = torch.bfloat16

    old = model.layers[1].feed_forward.w2
    new = _Gguf(old.in_features, old.out_features, bias = False).to(torch.bfloat16)
    model.layers[1].feed_forward.w2 = new
    before = _snapshot(model)
    assert "quantized" in mf.unsupported_reason(model, MPS_BF16)
    assert mf.install_mps_fp16(model, MPS_BF16) == 0
    _same(before, _snapshot(model))


def test_diffusers_source_matches_the_reimplemented_attention():
    mf._SOURCE_OK.clear()
    assert mf._source_ok("processor", zmod.ZSingleStreamAttnProcessor.__call__, mf._PROCESSOR_TOKENS)
    assert mf._source_ok("ffn", zmod.FeedForward, mf._FFN_TOKENS)


def test_changed_diffusers_attention_keeps_bf16(monkeypatch):
    monkeypatch.setattr(mf, "_SOURCE_OK", {"processor": False})
    model = _model()
    assert mf.install_mps_fp16(model, MPS_BF16) == 0




def test_headroom_identities_are_exact_in_fp32(monkeypatch):
    """Power-of-two headroom is exact: fp32 compute reproduces the stock model bit for bit."""
    monkeypatch.setattr(mf, "COMPUTE_DTYPE", "float32")
    model = _model(torch.float32)
    ref = _run(model, torch.float32)
    assert mf.install_mps_fp16(model, MPS_BF16) > 0
    assert torch.equal(_run(model, torch.float32), ref)


def test_ffn_headroom_identity_exact_with_large_values(monkeypatch):
    monkeypatch.setattr(mf, "COMPUTE_DTYPE", "float32")
    torch.manual_seed(0)
    ffn = zmod.FeedForward(64, 160)
    with torch.no_grad():
        ffn.w1.weight.mul_(40.0)
        ffn.w3.weight.mul_(40.0)
        ffn.w2.weight.mul_(400.0)
        x = torch.randn(3, 9, 64)
        ref = ffn(x)
        assert ref.abs().max() > 65504
        patched = mf._ffn_forward(ffn, x)
    assert torch.equal(patched, ref)


def _overflowing(model):
    with torch.no_grad():
        for blk in (*model.layers, *model.noise_refiner, *model.context_refiner):
            blk.feed_forward.w1.weight.mul_(30.0)
            blk.feed_forward.w3.weight.mul_(30.0)
            blk.feed_forward.w2.weight.mul_(300.0)
            blk.attention.to_v.weight.mul_(200.0)
            blk.attention.to_out[0].weight.mul_(200.0)
    return model


@needs_fp16
def test_fp16_compute_finite_where_plain_fp16_overflows():
    ref = _run(_overflowing(_model(torch.float32)), torch.float32)
    # The branch outputs exceed fp16's range (plain fp16 is not finite) ...
    blk = _overflowing(_model(torch.float32)).layers[0]
    with torch.no_grad():
        branch = blk.feed_forward(torch.randn(1, 64, CFG["dim"]))
    assert branch.abs().max() > 65504
    plain = _run(_overflowing(_model(torch.float32)).half(), torch.float16)
    assert not torch.isfinite(plain).all()
    # ... and the headroom keeps them in range, as close to fp32 as the bf16 model is.
    bf16 = _run(_overflowing(_model(torch.float32)).to(torch.bfloat16), torch.bfloat16)
    model = _overflowing(_model(torch.float32)).to(torch.bfloat16)
    assert mf.install_mps_fp16(model, MPS_BF16) > 0
    got = _run(model, torch.bfloat16)
    assert torch.isfinite(got).all()
    rel = float((got - ref).norm() / ref.norm())
    rel_bf16 = float((bf16 - ref).norm() / ref.norm())
    assert rel < max(2 * rel_bf16, 1e-2), (rel, rel_bf16)


@needs_fp16
def test_attention_processor_matches_stock_in_fp32_and_fp16(monkeypatch):
    torch.manual_seed(0)
    blk = zmod.ZImageTransformerBlock(0, 64, 4, 4, 1e-5, True, modulation = False).float()
    with torch.no_grad():
        blk.attention.to_v.weight.mul_(300.0)
        blk.attention.to_out[0].weight.mul_(600.0)
    attn = blk.attention
    x = torch.randn(2, 12, 64)
    freqs = torch.polar(torch.ones(2, 12, 8), torch.randn(2, 12, 8))
    with torch.no_grad():
        ref = attn(x, freqs_cis = freqs)
        assert ref.abs().max() > 65504
        monkeypatch.setattr(mf, "COMPUTE_DTYPE", "float32")
        proc = mf.processor_cls()()
        assert torch.equal(proc(attn, x, freqs_cis = freqs), ref)
        monkeypatch.setattr(mf, "COMPUTE_DTYPE", "float16")
        attn.half()
        got = proc(attn, x.to(torch.bfloat16), freqs_cis = freqs).float()
    assert got.dtype is torch.float32 and torch.isfinite(got).all()
    assert float((got - ref).norm() / ref.norm()) < 1e-2


def test_ffn_output_saturates_instead_of_inf(monkeypatch):
    """A value past the headroom saturates at 65504 x headroom in the bf16 stream instead of injecting inf."""
    ffn = zmod.FeedForward(8, 16).to(torch.float16)
    with torch.no_grad():
        ffn.w1.weight.fill_(1.0)
        ffn.w3.weight.fill_(1.0)
        ffn.w2.weight.fill_(1e4)
        out = mf._ffn_forward(ffn, torch.full((1, 8), 30.0, dtype = torch.bfloat16)) if _fp16_ok() else None
    if out is None:
        pytest.skip("this torch has no CPU float16 matmul")
    assert out.dtype is torch.bfloat16 and torch.isfinite(out).all()
    # 65504 rounds to 65536 in bf16.
    assert float(out.abs().max()) == float(torch.tensor(65504.0).to(torch.bfloat16)) * mf.FFN_HEADROOM




class NativeWeightOnlyLinear(nn.Module):

    def __init__(self, lin):
        super().__init__()
        w = lin.weight.detach().float()
        scale = w.abs().amax(dim = 1).clamp(min = 1e-12) / 127.0
        self.register_buffer("weight_q", (w / scale[:, None]).round().clamp(-127, 127).to(torch.int8))
        self.register_buffer("weight_scale", scale)
        self.scheme, self.act_int8, self.bias, self.seen = "int8", False, None, []

    def forward(self, x):
        self.seen.append(x.dtype)
        w = self.weight_q.float() * self.weight_scale[:, None]
        return nn.functional.linear(x.float(), w).to(x.dtype)


@needs_fp16
def test_int8_weight_only_layers_keep_storage_and_get_fp16_activations():
    model = _model()
    q = []
    for blk in (*model.layers, *model.noise_refiner, *model.context_refiner):
        for parent, names in ((blk.attention, ("to_q", "to_k", "to_v")), (blk.feed_forward, ("w1", "w2", "w3"))):
            for n in names:
                layer = NativeWeightOnlyLinear(getattr(parent, n))
                setattr(parent, n, layer)
                q.append(layer)
    assert mf.install_mps_fp16(model, MPS_BF16) > 0
    assert all(layer.weight_q.dtype is torch.int8 and layer.weight_scale.dtype is torch.float32 for layer in q)
    out = _run(model, torch.bfloat16)
    assert torch.isfinite(out).all()
    assert q[0].seen and all(d is torch.float16 for layer in q for d in layer.seen)


class _LoRALinear(nn.Module):
    def __init__(self, base):
        super().__init__()
        self.base = base
        self.lora_A = nn.Linear(base.in_features, 4, bias = False).to(base.weight.dtype)
        self.lora_B = nn.Linear(4, base.out_features, bias = False).to(base.weight.dtype)

    def forward(self, x):
        return self.base(x) + self.lora_B(self.lora_A(x))


@needs_fp16
def test_lora_injected_after_install_rides_the_headroom():
    """Projections replaced after install (LoRA) are still called at run time and scaled as a whole."""

    def adapt(model):
        torch.manual_seed(1)
        for name, m in list(model.named_modules()):
            if isinstance(m, nn.Linear) and name.rsplit(".", 1)[-1] in ("w1", "w2", "w3", "to_q", "to_k", "to_v"):
                parent, _, child = name.rpartition(".")
                setattr(model.get_submodule(parent), child, _LoRALinear(m))
        return model

    ref = _run(adapt(_overflowing(_model(torch.float32))), torch.float32)
    model = _overflowing(_model(torch.float32)).to(torch.bfloat16)
    assert mf.install_mps_fp16(model, MPS_BF16) > 0
    adapt(model)
    got = _run(model, torch.bfloat16)
    assert torch.isfinite(got).all()
    assert float((got - ref).norm() / ref.norm()) < 2e-2
