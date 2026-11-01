# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""FP8 LoRA on a frozen 16-bit base (float8_training with LoRA, kernels/fp8_lora.py), on CPU: torch's CPU
_scaled_mm runs the same fp8 GEMMs, so numerics, caching, the LoRA kernels' backward routing, refusals and
torch.compile tracing are all checked without a GPU."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from unsloth import precision as P
from unsloth.kernels import fp8_lora as FL
from unsloth.kernels import fast_lora as FLK
from unsloth.models import float8_training as f8t

pytestmark = pytest.mark.skipif(
    not hasattr(torch, "float8_e4m3fn") or not hasattr(torch, "_scaled_mm"),
    reason = "needs torch float8 dtypes",
)

FP8_FWD_TOL = 0.06  # tensorwise e4m3 x e4m3: ~0.037 relative on Gaussian data
FP8_DGRAD_TOL = 0.09  # e5m2 dy x e4m3 W: ~0.059


def _rel(a, b):
    a, b = a.float(), b.float()
    return ((a - b).norm() / b.norm()).item()


def _cos(a, b):
    return F.cosine_similarity(a.flatten().float(), b.flatten().float(), dim = 0).item()


@pytest.fixture(autouse = True)
def _stats():
    FL.reset_fp8_lora_stats()
    yield


def _weight(N = 384, K = 256, dtype = torch.bfloat16, seed = 0):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(N, K, generator = g) * K**-0.5).to(dtype)


# ---- op numerics -----------------------------------------------------------------------------------------
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_fwd_and_dgrad_within_fp8_error(dtype):
    W = _weight(dtype = dtype)
    x = torch.randn(64, 256).to(dtype)
    dy = torch.randn(64, 384).to(dtype)
    state = FL.Fp8LoRABaseState(W)
    y = state.matmul(x, W)
    dx = state.dgrad(dy, W)
    ref_y = x.float() @ W.float().t()
    ref_dx = dy.float() @ W.float()
    assert y.dtype == dtype and dx.dtype == dtype
    e_fwd, e_dgrad = _rel(y, ref_y), _rel(dx, ref_dx)
    # Inside the fp8 band, and clearly not the 16-bit GEMM (which is ~0.002): fp8 really ran.
    assert 0.01 < e_fwd < FP8_FWD_TOL, e_fwd
    assert 0.01 < e_dgrad < FP8_DGRAD_TOL, e_dgrad
    # The transposed view (what LoRA_MLP's down-projection backward hands matmul_lora) is the dgrad.
    assert torch.equal(state.matmul(dy, W.t()), dx)
    assert FL.FP8_LORA_STATS["fwd"] == 1 and FL.FP8_LORA_STATS["dgrad"] == 2


def test_scaled_mm_counted_and_fp16_never_written_by_the_fp8_gemm(monkeypatch):
    seen = []
    real = torch._scaled_mm

    def spy(*args, **kwargs):
        seen.append(kwargs.get("out_dtype"))
        return real(*args, **kwargs)

    W = _weight(dtype = torch.float16)
    state = FL.Fp8LoRABaseState(W)
    monkeypatch.setattr(torch, "_scaled_mm", spy)
    with P.count_lowp_gemms() as counts:
        y = state.matmul(torch.randn(32, 256).half(), W)
        state.dgrad(torch.randn(32, 384).half(), W)
    assert counts["_scaled_mm"] == 2
    # survey8 risk: fp16 output of an fp8 GEMM overflows; the GEMM writes bf16 and the result is cast.
    assert seen == [torch.bfloat16, torch.bfloat16]
    assert y.dtype == torch.float16 and torch.isfinite(y).all()


def test_large_fp16_values_stay_finite():
    # Outputs near the fp16 limit: exact as the 16-bit GEMM would be, no fp8-specific overflow.
    W = torch.full((32, 64), 0.5, dtype = torch.float16)
    x = torch.full((16, 64), 1800.0, dtype = torch.float16)  # each output 57600 < 65504
    y = FL.Fp8LoRABaseState(W).matmul(x, W)
    assert torch.isfinite(y).all()
    assert abs(y.float().mean().item() - 57600.0) / 57600.0 < 0.07


def test_nan_propagates_for_the_grad_scaler():
    W = _weight()
    state = FL.Fp8LoRABaseState(W)
    dy = torch.randn(32, 384).bfloat16()
    dy[3, 5] = float("nan")
    assert torch.isnan(state.dgrad(dy, W)).any()


def test_decode_rows_use_the_16bit_weight():
    W = _weight()
    state = FL.Fp8LoRABaseState(W)
    x = torch.randn(1, 1, 256).bfloat16()
    torch.testing.assert_close(state.matmul(x, W), x @ W.t())
    assert FL.FP8_LORA_STATS["fwd"] == 0 and FL.FP8_LORA_STATS["fwd_16bit"] == 1


# ---- cache -----------------------------------------------------------------------------------------------
def test_quantized_once_and_requantized_after_an_in_place_change():
    W = _weight()
    state = FL.Fp8LoRABaseState(W)
    x = torch.randn(32, 256).bfloat16()
    for _ in range(5):
        state.refresh(W)
        state.matmul(x, W)
    assert state.quantizations == 1 and FL.FP8_LORA_STATS["quantize"] == 1
    w8 = state.w8
    # Both orientations come from one quantization: w8t is w8 transposed, byte for byte.
    assert torch.equal(state.w8t.view(torch.uint8), w8.t().contiguous().view(torch.uint8))
    with torch.no_grad():
        W.mul_(2.0)  # a LoRA merge / unmerge writes the base weight in place
    state.refresh(W)
    assert state.quantizations == 2
    assert _rel(state.matmul(x, W), x.float() @ W.float().t()) < FP8_FWD_TOL


# ---- the LoRA kernels: forward fp8, input gradient fp8, LoRA grads in 16 bit -------------------------------
class _Proj(torch.nn.Module):
    """A PEFT lora.Linear stand-in: base_layer + lora_A / lora_B ModuleDicts."""

    def __init__(self, K, N, r = 8, seed = 0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.base_layer = torch.nn.Linear(K, N, bias = False, dtype = torch.bfloat16)
        with torch.no_grad():
            self.base_layer.weight.copy_(torch.randn(N, K, generator = g) * K**-0.5)
        self.base_layer.weight.requires_grad_(False)
        self.lora_A = torch.nn.ModuleDict(
            {"default": torch.nn.Linear(K, r, bias = False, dtype = torch.bfloat16)}
        )
        self.lora_B = torch.nn.ModuleDict(
            {"default": torch.nn.Linear(r, N, bias = False, dtype = torch.bfloat16)}
        )
        with torch.no_grad():
            self.lora_A["default"].weight.copy_(torch.randn(r, K, generator = g) * 0.05)
            self.lora_B["default"].weight.copy_(torch.randn(N, r, generator = g) * 0.05)
        self.scaling = {"default": 2.0}
        self.disable_adapters = False
        self.merged = False
        self.active_adapters = ["default"]


def _silu_fwd(e, g):
    return F.silu(e.float()).to(e.dtype) * g


def _silu_bwd(DW, e, g):
    # swiglu_DWf_DW_dfg_kernel's contract: returns (h, df, de) = (f * g, DW * f, DW * g * f'(e)).
    ef = e.float()
    s = torch.sigmoid(ef)
    f = ef * s
    h = (f * g.float()).to(DW.dtype)
    df = (DW.float() * f).to(DW.dtype)
    de = (DW.float() * g.float() * (s * (1 + ef * (1 - s)))).to(DW.dtype)
    return h, df, de


def _lora_params(projs):
    return [p.lora_A["default"].weight for p in projs] + [p.lora_B["default"].weight for p in projs]


def _run_block(kind, fp8):
    torch.manual_seed(0)
    H, I = 256, 512
    if kind == "mlp":
        projs = [_Proj(H, I, seed = 1), _Proj(H, I, seed = 2), _Proj(I, H, seed = 3)]
    elif kind == "qkv":
        projs = [_Proj(H, H, seed = 1), _Proj(H, 128, seed = 2), _Proj(H, 128, seed = 3)]
    else:
        projs = [_Proj(H, H, seed = 4)]
    if fp8:
        for p in projs:
            FL.attach_fp8_lora_base(p.base_layer)
    X = (torch.randn(2, 32, H) * 0.5).bfloat16().requires_grad_(True)
    if kind == "mlp":
        args = []
        for p in projs:
            args += list(FLK.get_lora_parameters(p))
        out = FLK.LoRA_MLP.apply(X, *args, _silu_fwd, _silu_bwd, False)
        outs = (out,)
    elif kind == "qkv":
        args = []
        for p in projs:
            args += list(FLK.get_lora_parameters(p))
        outs = FLK.LoRA_QKV.apply(X, *args, False)
    else:
        outs = (FLK.LoRA_W.apply(X, *FLK.get_lora_parameters(projs[0])),)
    g = torch.Generator().manual_seed(9)
    loss = sum((o.float() * torch.randn(o.shape, generator = g)).sum() for o in outs)
    loss.backward()
    return [o.detach() for o in outs], X.grad, [p.grad for p in _lora_params(projs)], projs


@pytest.mark.parametrize("kind", ["mlp", "qkv", "o"])
def test_lora_kernels_run_fp8_and_keep_lora_grads_16bit(kind):
    outs_ref, dX_ref, grads_ref, _ = _run_block(kind, fp8 = False)
    FL.reset_fp8_lora_stats()
    outs, dX, grads, projs = _run_block(kind, fp8 = True)
    n = len(projs)
    # Every base GEMM ran in fp8: one forward and one input gradient per projection.
    assert FL.FP8_LORA_STATS["fwd"] == n, FL.FP8_LORA_STATS
    assert FL.FP8_LORA_STATS["dgrad"] == n, FL.FP8_LORA_STATS
    assert FL.FP8_LORA_STATS["fwd_16bit"] == 0 and FL.FP8_LORA_STATS["dgrad_16bit"] == 0
    for o, r in zip(outs, outs_ref):
        assert _rel(o, r) < FP8_FWD_TOL * 1.5
    assert _rel(dX, dX_ref) < FP8_DGRAD_TOL * 1.5
    for g, r, p in zip(grads, grads_ref, _lora_params(projs)):
        assert g is not None and g.dtype == p.dtype == torch.bfloat16
        assert _cos(g, r) > 0.99, _cos(g, r)
    for p in projs:
        assert p.base_layer.weight.grad is None


def test_peft_path_base_forward_and_backward():
    # PEFT lora.Linear calls base_layer(x): the patched forward runs the same two fp8 GEMMs.
    base = torch.nn.Linear(256, 384, bias = True, dtype = torch.bfloat16)
    base.requires_grad_(False)
    ref = torch.nn.Linear(256, 384, bias = True, dtype = torch.bfloat16)
    ref.load_state_dict(base.state_dict())
    ref.requires_grad_(False)
    FL.attach_fp8_lora_base(base)
    x = torch.randn(2, 16, 256).bfloat16()
    xa, xb = x.clone().requires_grad_(True), x.clone().requires_grad_(True)
    ya, yb = base(xa), ref(xb)
    dy = torch.randn_like(ya)
    ya.backward(dy)
    yb.backward(dy)
    assert _rel(ya, yb) < FP8_FWD_TOL and _rel(xa.grad, xb.grad) < FP8_DGRAD_TOL
    assert FL.FP8_LORA_STATS["fwd"] == 1 and FL.FP8_LORA_STATS["dgrad"] == 1
    # state_dict is untouched: the 16-bit weight is the source of truth for save / merge.
    assert set(base.state_dict()) == {"weight", "bias"}
    assert base.weight.dtype == torch.bfloat16


def _tiny_qwen3_peft():
    from transformers import Qwen3Config, Qwen3ForCausalLM
    from peft import LoraConfig, get_peft_model

    torch.manual_seed(0)
    config = Qwen3Config(
        vocab_size = 256,
        hidden_size = 128,
        intermediate_size = 256,
        num_hidden_layers = 2,
        num_attention_heads = 4,
        num_key_value_heads = 2,
        head_dim = 32,
    )
    model = Qwen3ForCausalLM(config).to(torch.bfloat16)
    lora = LoraConfig(
        r = 8,
        lora_alpha = 16,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        init_lora_weights = False,
    )
    return get_peft_model(model, lora)


def _loss_and_grads(model):
    torch.manual_seed(1)
    ids = torch.randint(0, 256, (2, 32))
    out = model(input_ids = ids, labels = ids)
    out.loss.backward()
    grads = {n: p.grad.clone() for n, p in model.named_parameters() if p.requires_grad}
    model.zero_grad(set_to_none = True)
    return out.loss.item(), grads


def test_enable_fp8_lora_on_a_peft_model():
    model = _tiny_qwen3_peft()
    loss_ref, grads_ref = _loss_and_grads(model)
    f8t.mark_fp8_lora(model.base_model.model, f8t.FP8_LORA_RECIPE)
    converted = f8t.enable_fp8_lora(model, allow_cpu = True)
    assert len(converted) == 2 * 7  # every LoRA'd projection; lm_head / embeddings untouched
    assert P.engaged(model)[f8t.FP8_LORA_RECIPE]["modules"] == 14
    loss, grads = _loss_and_grads(model)
    # Layer 0's q / k / v read the frozen embedding output, which needs no input gradient: 14 - 3 dgrads.
    assert FL.FP8_LORA_STATS["fwd"] == 14 and FL.FP8_LORA_STATS["dgrad"] == 11
    assert abs(loss - loss_ref) / loss_ref < 0.01
    assert grads.keys() == grads_ref.keys()
    for name, g in grads.items():
        assert g.dtype == grads_ref[name].dtype
        assert _cos(g, grads_ref[name]) > 0.98, (name, _cos(g, grads_ref[name]))
    # Quantized once per layer however many steps run.
    _loss_and_grads(model)
    assert FL.FP8_LORA_STATS["quantize"] == 14


def test_no_request_no_conversion_and_strict_zero_engagement(monkeypatch):
    model = _tiny_qwen3_peft()
    assert f8t.enable_fp8_lora(model) == []  # loaded without float8_training: untouched
    for module in model.modules():
        assert "_unsloth_fp8_lora" not in module.__dict__
    # Requested, but nothing eligible (CPU weights without allow_cpu): strict mode refuses the silent fallback.
    monkeypatch.setenv("UNSLOTH_STRICT_ENGAGE", "1")
    with pytest.raises(P.EngagementError):
        f8t.enable_fp8_lora(model, recipe = f8t.FP8_LORA_RECIPE)


def test_eligibility_reasons():
    assert FL.eligible_base(torch.nn.Linear(100, 64).bfloat16().requires_grad_(False)) is not None
    assert "trainable" in FL.eligible_base(torch.nn.Linear(64, 64).bfloat16())
    assert "dtype" in FL.eligible_base(torch.nn.Linear(64, 64).requires_grad_(False))
    assert FL.eligible_base(torch.nn.Linear(64, 64).bfloat16().requires_grad_(False)) is None
    sub = torch.nn.Linear(64, 64).bfloat16().requires_grad_(False)

    class _Sub(torch.Tensor):
        pass

    sub.weight = torch.nn.Parameter(sub.weight.detach().as_subclass(_Sub), requires_grad = False)
    assert "_Sub" in FL.eligible_base(sub)


# ---- torch.compile ---------------------------------------------------------------------------------------
def test_compile_aot_eager_fullgraph():
    torch._dynamo.reset()
    base = torch.nn.Linear(256, 384, bias = False, dtype = torch.bfloat16).requires_grad_(False)
    FL.attach_fp8_lora_base(base)
    proj = _Proj(256, 384)
    FL.attach_fp8_lora_base(proj.base_layer)

    def step(x):
        y = base(x)  # PEFT path
        W, W_quant, A, B, S = FLK.get_lora_parameters(proj)
        z = FLK.LoRA_W.apply(y[..., :256].contiguous(), W, W_quant, A, B, S)  # fast path
        return z.float().square().mean()

    x = torch.randn(2, 16, 256).bfloat16().requires_grad_(True)
    eager = step(x)
    eager.backward()
    g_eager = x.grad.clone()
    x.grad = None
    FL.reset_fp8_lora_stats()
    compiled = torch.compile(step, backend = "aot_eager", fullgraph = True)  # any graph break raises
    out = compiled(x)
    out.backward()
    torch.testing.assert_close(out, eager)
    torch.testing.assert_close(x.grad, g_eager)
    # The opaque ops ran inside the compiled graph: forward and backward GEMMs counted.
    assert FL.FP8_LORA_STATS["fwd"] == 2 and FL.FP8_LORA_STATS["dgrad"] == 2, FL.FP8_LORA_STATS
    explain = torch._dynamo.explain(step)(x)
    assert explain.graph_break_count == 0


# ---- refusals before load ---------------------------------------------------------------------------------
def test_resolve_refusals_and_names():
    R = f8t.FP8_LORA_RECIPE
    with pytest.raises(P.PrecisionRefused, match = "fp8 tensor cores"):
        f8t.resolve_fp8_lora("fp8_lora", gpu = "sm75")  # T4
    with pytest.raises(P.PrecisionRefused, match = "fp8 tensor cores"):
        f8t.resolve_fp8_lora("fp8_lora", gpu = "sm80")  # A100
    with pytest.raises(P.PrecisionRefused, match = "ROCm"):
        f8t.resolve_fp8_lora("fp8_lora", gpu = "rocm")
    with pytest.raises(P.PrecisionRefused, match = "no CUDA device"):
        f8t.resolve_fp8_lora("fp8_lora", gpu = None)
    with pytest.raises(P.PrecisionRefused, match = "load_in_4bit = False"):
        f8t.resolve_fp8_lora("fp8_lora", load_in_4bit = True, gpu = "sm89")
    with pytest.raises(P.PrecisionRefused, match = "qat_scheme"):
        f8t.resolve_fp8_lora("fp8_lora", qat_scheme = "fp8-fp8", gpu = "sm89")
    with pytest.raises(P.PrecisionRefused, match = "fp32"):
        f8t.resolve_fp8_lora("fp8_lora", dtype = torch.float32, gpu = "sm89")
    with pytest.raises(ValueError, match = "full finetuning recipe"):
        f8t.resolve_fp8_lora("rowwise", gpu = "sm89")
    assert f8t.resolve_fp8_lora("fp8_lora", gpu = "sm89", dtype = torch.bfloat16) == R
    assert f8t.resolve_fp8_lora("fp8_lora", gpu = "sm120", dtype = torch.float16) == R
    assert f8t.resolve_fp8_lora(False) is None and f8t.resolve_fp8_lora(None) is None
    gate = P.get_recipe(R).gate("sm89").status
    if gate != "named":
        # By name (True) only where it was measured faster.
        with pytest.raises(P.PrecisionRefused, match = "not been measured faster"):
            f8t.resolve_fp8_lora(True, gpu = "sm89")


def test_loader_refuses_before_any_load():
    from unsloth.models import loader
    for cls in (loader.FastLanguageModel, loader.FastModel):
        with pytest.raises(P.PrecisionRefused, match = "load_in_4bit = False"):
            cls.from_pretrained(
                model_name = "unsloth/does-not-exist", load_in_4bit = True, float8_training = "fp8_lora"
            )
        # No CUDA here: refused with the hardware reason, not a missing-repo error.
        with pytest.raises(P.PrecisionRefused, match = "no CUDA device|tensor cores"):
            cls.from_pretrained(
                model_name = "unsloth/does-not-exist",
                load_in_4bit = False,
                float8_training = "fp8_lora",
            )


def test_registry_row():
    r = P.get_recipe(f8t.FP8_LORA_RECIPE)
    assert r.kind == "frozen_base" and r.modes == ("lora",)
    assert r.fwd.a.dtype == "e4m3" and r.fwd.b.granularity == "tensor"
    assert r.dgrad.a.dtype == "e5m2" and not r.wgrad.low_precision
    assert "fp32" not in r.input_dtypes
    assert r.gate("sm75").status == "refused" and r.gate("sm80").status == "refused"


# ---- rowwise (per-token) activation / gradient scales: UNSLOTH_FP8_LORA_SCALING=row ------------------------
@pytest.fixture
def rowwise(monkeypatch):
    monkeypatch.setattr(FL, "_SCALING", "row")
    monkeypatch.setattr(FL, "_DGRAD_DTYPE", FL.E4M3)
    yield


def test_rowwise_numerics_and_outlier_tokens(rowwise):
    W = _weight()
    x = torch.randn(64, 256).bfloat16()
    x[0] *= 1e5  # one massive-activation token: tensorwise pushes the rest into e4m3 subnormals
    dy = torch.randn(64, 384).bfloat16()
    state = FL.Fp8LoRABaseState(W)
    assert state.w_inv_fwd.shape == (1, 384) and state.w_inv_dgrad.shape == (1, 256)
    y = state.matmul(x, W)
    ref = x.float() @ W.float().t()
    # Every other token keeps its own scale: the outlier row does not flush them toward zero.
    rest = _rel(y[1:], ref[1:])
    assert rest < FP8_FWD_TOL, rest
    FL._SCALING = "tensor"
    tensorwise = _rel(FL.Fp8LoRABaseState(W).matmul(x, W)[1:], ref[1:])
    FL._SCALING = "row"
    assert tensorwise > 2 * rest, (tensorwise, rest)
    dx = state.dgrad(dy, W)
    assert _rel(dx, dy.float() @ W.float()) < FP8_FWD_TOL


def test_rowwise_lora_kernels_and_peft_path(rowwise):
    outs_ref, dX_ref, grads_ref, _ = _run_block("mlp", fp8 = False)
    FL.reset_fp8_lora_stats()
    outs, dX, grads, projs = _run_block("mlp", fp8 = True)
    assert FL.FP8_LORA_STATS["fwd"] == 3 and FL.FP8_LORA_STATS["dgrad"] == 3
    assert _rel(outs[0], outs_ref[0]) < FP8_FWD_TOL * 1.5 and _rel(dX, dX_ref) < FP8_FWD_TOL * 1.5
    for g, r in zip(grads, grads_ref):
        assert _cos(g, r) > 0.99
    base = torch.nn.Linear(256, 384, bias = False, dtype = torch.bfloat16).requires_grad_(False)
    FL.attach_fp8_lora_base(base)
    x = torch.randn(2, 8, 256).bfloat16().requires_grad_(True)
    base(x).float().sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_loaded_adapter_is_converted_when_marked(monkeypatch):
    # An adapter repo loads as a PeftModel and skips get_peft_model: mark_fp8_lora converts it directly.
    model = _tiny_qwen3_peft()
    calls = []
    real = f8t.enable_fp8_lora
    monkeypatch.setattr(f8t, "enable_fp8_lora", lambda m, r = None, **k: calls.append(r) or real(m, r, allow_cpu = True))
    f8t.mark_fp8_lora(model, f8t.FP8_LORA_RECIPE)
    assert calls == [f8t.FP8_LORA_RECIPE]
    assert P.engaged(model)[f8t.FP8_LORA_RECIPE]["modules"] == 14
