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


def _weight(
    N = 384,
    K = 256,
    dtype = torch.bfloat16,
    seed = 0,
):
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
    # survey8 risk: fp16 output of an fp8 GEMM overflows; the GEMM writes float32 (bf16 rounding can still
    # overflow the cast back) and the result is cast.
    assert seen == [torch.float32, torch.float32]
    assert y.dtype == torch.float16 and torch.isfinite(y).all()


def test_large_fp16_values_stay_finite():
    # Outputs near the fp16 limit: exact as the 16-bit GEMM would be, no fp8-specific overflow.
    W = torch.full((32, 64), 0.5, dtype = torch.float16)
    x = torch.full((16, 64), 1800.0, dtype = torch.float16)  # each output 57600 < 65504
    y = FL.Fp8LoRABaseState(W).matmul(x, W)
    assert torch.isfinite(y).all()
    assert abs(y.float().mean().item() - 57600.0) / 57600.0 < 0.07


def test_fp16_results_at_the_fp16_limit_stay_finite():
    # Failing before: the GEMM wrote bf16, which rounds 65504 to 65536, an inf after the cast back to fp16.
    W = torch.full((32, 64), 0.5, dtype = torch.float16)
    x = torch.full((16, 64), 2047.0, dtype = torch.float16)  # each output 65504, the fp16 maximum
    state = FL.Fp8LoRABaseState(W)
    assert torch.isfinite(state.matmul(x, W)).all()
    dy = torch.full((16, 32), 4094.0, dtype = torch.float16)  # each dgrad output 65504
    assert torch.isfinite(state.dgrad(dy, W)).all()


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
def test_quantized_once_and_requantized_after_an_in_place_change(monkeypatch):
    monkeypatch.setattr(FL, "_CACHE_T", True)
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

    def __init__(
        self,
        K,
        N,
        r = 8,
        seed = 0,
    ):
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
        target_modules = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
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
    # sm120 runs per-token scales (faster there); sm89 tensorwise (rowwise _scaled_mm is slow on L4).
    assert f8t.resolve_fp8_lora("fp8_lora", gpu = "sm120", dtype = torch.float16) == "fp8_lora_rowwise"
    assert FL.scaling_for_class("sm120") == "row" and FL.scaling_for_class("sm89") == "tensor"
    assert f8t.resolve_fp8_lora(False) is None and f8t.resolve_fp8_lora(None) is None
    gate = P.get_recipe(R).gate("sm89").status
    if gate != "named":  # explicit until the gates in the registry evidence are met
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
    monkeypatch.setattr(
        f8t, "enable_fp8_lora", lambda m, r = None, **k: calls.append(r) or real(m, r, allow_cpu = True)
    )
    f8t.mark_fp8_lora(model, f8t.FP8_LORA_RECIPE)
    assert calls == [f8t.FP8_LORA_RECIPE]
    assert P.engaged(model)[f8t.FP8_LORA_RECIPE]["modules"] == 14


def test_default_no_transposed_cache_matches_cached(monkeypatch):
    W = _weight()
    dy = torch.randn(32, 384).bfloat16()
    lean = FL.Fp8LoRABaseState(W)
    assert (
        lean.w8t is None
    )  # default: one fp8 copy (+1 byte / weight), transposed inside the dgrad op
    monkeypatch.setattr(FL, "_CACHE_T", True)
    cached = FL.Fp8LoRABaseState(W)
    assert cached.w8t is not None
    torch.testing.assert_close(lean.dgrad(dy, W), cached.dgrad(dy, W), rtol = 0, atol = 0)


def test_compiled_graph_has_no_fp8_transpose_outside_the_ops():
    # sm89 Inductor writes corrupt bytes into 2D-tiled fp8 transposes (scripts/fp8_lora/l4_wgrad_bisect2.py):
    # every fp8 tensor op must stay inside the opaque fwd / dgrad ops, never in the traced graph.
    torch._dynamo.reset()
    base = torch.nn.Linear(256, 384, bias = False, dtype = torch.bfloat16).requires_grad_(False)
    FL.attach_fp8_lora_base(base)
    proj = _Proj(256, 384)
    FL.attach_fp8_lora_base(proj.base_layer)
    graphs = []

    def backend(gm, example_inputs):
        from torch._functorch.aot_autograd import aot_module_simplified
        def capture(g, _):
            graphs.append(g)
            return g

        return aot_module_simplified(gm, example_inputs, fw_compiler = capture, bw_compiler = capture)

    def step(x):
        y = base(x)
        W, W_quant, A, B, S = FLK.get_lora_parameters(proj)
        return FLK.LoRA_W.apply(y[..., :256].contiguous(), W, W_quant, A, B, S).float().sum()

    x = torch.randn(2, 16, 256).bfloat16().requires_grad_(True)
    torch.compile(step, backend = backend, fullgraph = True)(x).backward()
    assert len(graphs) >= 2  # forward and backward
    fp8 = (torch.float8_e4m3fn, torch.float8_e5m2)
    targets = []
    for g in graphs:
        for node in g.graph.nodes:
            val = node.meta.get("val")
            if node.op == "call_function" and isinstance(val, torch.Tensor) and val.dtype in fp8:
                targets.append(str(node.target))
    assert targets == [], targets
    names = {str(n.target) for g in graphs for n in g.graph.nodes if n.op == "call_function"}
    assert any("fp8_lora_fwd" in t for t in names) and any(
        "fp8_lora_dgrad" in t for t in names
    ), names


# ---- host master: the GPU keeps only the fp8 copy (memory at or below bf16 LoRA) ----------------------------
@pytest.fixture
def offload_on_cpu(monkeypatch):
    # The compute device is the CPU here, so offload to a separate host copy: every code path of the GPU case
    # runs, only the device copy itself is trivial (the GPU job checks that part).
    monkeypatch.setattr(FL, "_OFFLOAD_CPU_WEIGHTS", True)
    monkeypatch.setattr(FL, "_OFFLOAD_MASTER", True)
    monkeypatch.delenv("WORLD_SIZE", raising = False)


def _frozen_linear(
    K = 256,
    N = 384,
    bias = False,
    seed = 0,
):
    torch.manual_seed(seed)
    lin = torch.nn.Linear(K, N, bias = bias, dtype = torch.bfloat16)
    return lin.requires_grad_(False)


def test_host_master_leaves_only_fp8_bytes_on_the_compute_device(offload_on_cpu):
    lin, ref = _frozen_linear(), _frozen_linear()
    W0, param = lin.weight.detach().clone(), lin.weight
    state = FL.attach_fp8_lora_base(lin)
    ref_state = FL.attach_fp8_lora_base(ref, offload = False)
    assert state.offloaded and not ref_state.offloaded
    assert lin.weight is param  # optimizer / PEFT / named_parameters keep the same Parameter
    assert torch.equal(lin.weight, W0)  # the host master is the exact 16-bit weight
    numel = 256 * 384
    # Per converted layer: fp8 copy (1 byte / weight) + one fp32 scale; no 16-bit bytes on the device.
    assert state.resident_bytes(lin) == {
        "fp8": numel + 4,
        "master_device": 0,
        "master_host": 2 * numel,
    }
    # Before: the 16-bit master stayed beside the fp8 copy (3 bytes / weight on the device).
    assert ref_state.resident_bytes(ref) == {
        "fp8": numel + 4,
        "master_device": 2 * numel,
        "master_host": 0,
    }
    assert FL.fp8_lora_memory(torch.nn.Sequential(lin, ref))["total"] == {
        "fp8": 2 * (numel + 4),
        "master_device": 2 * numel,
        "master_host": 2 * numel,
    }


def test_host_master_gives_the_same_fp8_results(offload_on_cpu):
    lin, ref = _frozen_linear(bias = True), _frozen_linear(bias = True)
    FL.attach_fp8_lora_base(lin)
    FL.attach_fp8_lora_base(ref, offload = False)
    x = torch.randn(2, 16, 256).bfloat16()
    xa, xb = x.clone().requires_grad_(True), x.clone().requires_grad_(True)
    ya, yb = lin(xa), ref(xb)
    dy = torch.randn_like(ya)
    ya.backward(dy)
    yb.backward(dy)
    assert torch.equal(ya, yb) and torch.equal(xa.grad, xb.grad)
    assert FL.FP8_LORA_STATS["fwd"] == 2 and FL.FP8_LORA_STATS["dgrad"] == 2
    assert FL.FP8_LORA_STATS["quantize"] == 2  # quantized once each, never again from the host copy


def test_host_master_decode_rows_use_the_dequantized_fp8_weight(offload_on_cpu):
    lin = _frozen_linear()
    W0 = lin.weight.detach().clone()
    state = FL.attach_fp8_lora_base(lin)
    x = torch.randn(1, 1, 256).bfloat16()
    dense = (state.w8.float() * state.w_inv).bfloat16()
    torch.testing.assert_close(state.matmul(x, lin.weight), x @ dense.t())
    torch.testing.assert_close(lin(x), x @ dense.t())
    # Its orientation follows W: the transposed view gives the input-gradient form.
    dy = torch.randn(1, 1, 384).bfloat16()
    torch.testing.assert_close(state.matmul(dy, lin.weight.t()), dy @ dense)
    assert _rel(lin(x), x @ W0.t()) < FP8_FWD_TOL
    assert FL.FP8_LORA_STATS["fwd"] == 0


def test_host_master_dequantize_is_exact(offload_on_cpu):
    lin = _frozen_linear()
    W0 = lin.weight.detach().clone()
    state = FL.attach_fp8_lora_base(lin)
    # fast_dequantize (unsloth's merged save) gets the exact weight, never the fp8 reconstruction.
    from unsloth.kernels.utils import fast_dequantize

    assert torch.equal(fast_dequantize(lin.weight, state), W0)
    assert torch.equal(fast_dequantize(lin.weight.t(), state), W0.t())


def test_module_moves_keep_or_restore_the_host_master(offload_on_cpu):
    lin = _frozen_linear(bias = True)
    state = FL.attach_fp8_lora_base(lin)
    w8 = state.w8
    # The Trainer's model.to(device) onto the device the fp8 copy lives on: master stays in host RAM.
    lin.to(state.device)
    lin.to(torch.bfloat16)
    assert state.offloaded and state.w8 is w8
    # A real conversion brings the exact master back first, then converts it.
    W0 = lin.weight.detach().clone()
    lin.half()
    assert not state.offloaded and lin.weight.dtype == torch.float16
    assert torch.equal(lin.weight, W0.half())
    x = torch.randn(32, 256).half()
    y = lin(x)  # requantized from the converted weight
    assert (
        y.dtype == torch.float16
        and _rel(y, x.float() @ W0.float().t() + lin.bias.float()) < FP8_FWD_TOL
    )


def test_no_host_master_for_16bit_dgrad_or_distributed(offload_on_cpu, monkeypatch):
    monkeypatch.setattr(FL, "_DGRAD_DTYPE", None)
    assert not FL.offload_supported()
    monkeypatch.setattr(FL, "_DGRAD_DTYPE", "auto")
    assert FL.offload_supported()
    monkeypatch.setenv("WORLD_SIZE", "2")
    assert not FL.offload_supported()
    monkeypatch.delenv("WORLD_SIZE")
    monkeypatch.setattr(FL, "_OFFLOAD_MASTER", False)
    assert not FL.offload_supported()


def test_save_and_merge_exact_with_host_master(offload_on_cpu):
    import copy

    model = _tiny_qwen3_peft()
    ref = copy.deepcopy(model)
    frozen = {n: p.detach().clone() for n, p in model.named_parameters() if not p.requires_grad}
    f8t.mark_fp8_lora(model.base_model.model, f8t.FP8_LORA_RECIPE)
    converted = f8t.enable_fp8_lora(model, allow_cpu = True)
    assert len(converted) == 14
    assert P.engaged(model)[f8t.FP8_LORA_RECIPE]["detail"]["master_offloaded"] == 14
    memory = FL.fp8_lora_memory(model)
    assert len(memory["layers"]) == 14 and memory["total"]["master_device"] == 0
    # Train a few steps: LoRA moves, the frozen base must not.
    opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr = 0.1)
    for _ in range(3):
        _loss_and_grads_keep(model)
        opt.step()
        opt.zero_grad(set_to_none = True)
    sd = model.state_dict()
    for name, value in frozen.items():
        assert torch.equal(sd[name], value), name
    # Serialized base weights round trip bit for bit.
    from safetensors.torch import load, save

    base = {n: sd[n].contiguous() for n in frozen if "base_layer" in n}
    assert len(base) == 14
    loaded = load(save(base))
    assert all(torch.equal(loaded[n], frozen[n]) for n in base)
    # Same LoRA weights into the untouched copy, then merge both ways: bitwise equal to the 16-bit merge.
    ref.load_state_dict({n: p for n, p in sd.items() if "lora_" in n}, strict = False)
    from unsloth.save import _merge_lora

    mods, ref_mods = dict(model.named_modules()), dict(ref.named_modules())
    for name in converted:
        W, _ = _merge_lora(mods[name], name)
        W_ref, _ = _merge_lora(ref_mods[name], name)
        assert W.dtype == W_ref.dtype and torch.equal(W, W_ref), name
    merged = model.merge_and_unload().state_dict()
    merged_ref = ref.merge_and_unload().state_dict()
    assert merged.keys() == merged_ref.keys()
    for name in merged_ref:
        assert torch.equal(merged[name], merged_ref[name]), name
    # PEFT merge put the masters back on the compute device before adding the delta.
    assert all(
        not m.__dict__["_unsloth_fp8_lora"].offloaded
        for m in model.modules()
        if "_unsloth_fp8_lora" in m.__dict__
    )


def _loss_and_grads_keep(model):
    torch.manual_seed(1)
    ids = torch.randint(0, 256, (2, 32))
    model(input_ids = ids, labels = ids).loss.backward()


def test_square_tile_transpose_kernel_byte_exact_under_the_triton_interpreter():
    triton = pytest.importorskip("triton")
    import os
    import subprocess
    import sys
    import textwrap

    path = os.path.abspath(FL.__file__)
    code = textwrap.dedent(
        f"""
        import importlib.util, torch
        spec = importlib.util.spec_from_file_location("fp8_lora_standalone", {path!r})
        FL = importlib.util.module_from_spec(spec); spec.loader.exec_module(FL)
        assert FL.triton is not None
        for R, C in ((64, 64), (128, 192), (100, 37), (1, 300), (65, 129), (256, 64)):
            x = torch.randint(0, 255, (R, C), dtype = torch.uint8).view(torch.float8_e4m3fn)
            y = FL.fp8_transpose(x, use_triton = True)
            assert y.shape == (C, R) and y.is_contiguous() and y.dtype == x.dtype
            assert torch.equal(y.view(torch.uint8), x.t().contiguous().view(torch.uint8)), (R, C)
        print("TRANSPOSE_OK")
        """
    )
    env = dict(os.environ, TRITON_INTERPRET = "1", CUDA_VISIBLE_DEVICES = "")
    p = subprocess.run(
        [sys.executable, "-c", code], capture_output = True, text = True, env = env, timeout = 600
    )
    assert "TRANSPOSE_OK" in p.stdout, p.stdout[-2000:] + p.stderr[-4000:]
    # One BLOCK constexpr for both tile sides: non-square tiles miscompile fp8 transposed stores on sm89.
    import inspect

    src = inspect.getsource(FL.fp8_transpose)
    assert "_TRANSPOSE_BLOCK), triton.cdiv(C, _TRANSPOSE_BLOCK)" in src


def test_merge_of_a_deepcopy_merges_the_copy_only(offload_on_cpu):
    import copy

    model = _tiny_qwen3_peft()
    plain = copy.deepcopy(model)
    f8t.mark_fp8_lora(model.base_model.model, f8t.FP8_LORA_RECIPE)
    f8t.enable_fp8_lora(model, allow_cpu = True)
    twin = copy.deepcopy(model)
    name = "base_model.model.model.layers.0.mlp.down_proj"
    layer, twin_layer = dict(model.named_modules())[name], dict(twin.named_modules())[name]
    before = layer.base_layer.weight.detach().clone()
    twin_layer.merge()
    # The copy merged into its own base; the original layer is untouched and still unmerged.
    assert twin_layer.merged and not layer.merged
    assert torch.equal(layer.base_layer.weight, before)
    assert layer.base_layer.__dict__["_unsloth_fp8_lora"].offloaded
    plain_layer = dict(plain.named_modules())[name]
    plain_layer.merge()
    assert torch.equal(twin_layer.base_layer.weight, plain_layer.base_layer.weight)


def test_compiled_fp8_lora_ops_are_counted():
    # Under torch.compile the fp8 forward and dgrad are opaque custom ops; the counter names them.
    torch._dynamo.reset()
    W = _weight()
    state = FL.Fp8LoRABaseState(W)

    def run(x, dy):
        return state.matmul(x, W), state.dgrad(dy, W)

    compiled = torch.compile(run, backend = "aot_eager", fullgraph = True)
    x, dy = torch.randn(32, 256).bfloat16(), torch.randn(32, 384).bfloat16()
    compiled(x, dy)
    with P.count_lowp_gemms() as counts:
        compiled(x, dy)
    assert counts["unsloth::fp8_lora_fwd"] == 1 and counts["unsloth::fp8_lora_dgrad"] == 1, dict(
        counts
    )


def test_merge_requantizes_the_fp8_copy(offload_on_cpu):
    # Failing before: PEFT merges in place (same storage and version), so the fp8 cache kept the unmerged weight
    # while PEFT stopped adding the adapter: the merged model dropped the adapter.
    import copy

    model = _tiny_qwen3_peft()
    ref = copy.deepcopy(model)
    f8t.mark_fp8_lora(model.base_model.model, f8t.FP8_LORA_RECIPE)
    f8t.enable_fp8_lora(model, allow_cpu = True)
    ids = torch.randint(0, 256, (2, 32))
    with torch.no_grad():
        before = model(input_ids = ids).logits.float()
        model.merge_adapter()
        ref.merge_adapter()
        merged = model(input_ids = ids).logits.float()
        want = ref(input_ids = ids).logits.float()
    # fp8 of the merged weight (this test's random adapter is large): about 0.1; the adapter dropped: above 1.
    assert _rel(merged, want) < 0.3 and _rel(merged, before) < 0.3


def test_dora_layers_keep_the_master_on_the_device(offload_on_cpu):
    from transformers import Qwen3Config, Qwen3ForCausalLM
    from peft import LoraConfig, get_peft_model

    torch.manual_seed(0)
    config = Qwen3Config(
        vocab_size = 256,
        hidden_size = 128,
        intermediate_size = 256,
        num_hidden_layers = 1,
        num_attention_heads = 4,
        num_key_value_heads = 2,
        head_dim = 32,
    )
    model = get_peft_model(
        Qwen3ForCausalLM(config).to(torch.bfloat16),
        LoraConfig(r = 8, target_modules = ["q_proj", "v_proj"], use_dora = True),
    )
    f8t.mark_fp8_lora(model.base_model.model, f8t.FP8_LORA_RECIPE)
    assert len(f8t.enable_fp8_lora(model, allow_cpu = True)) == 2
    states = [
        m.base_layer.__dict__["_unsloth_fp8_lora"] for m in model.modules() if hasattr(m, "lora_A")
    ]
    assert states and not any(s.offloaded for s in states)


def test_an_added_adapter_lands_on_the_compute_device(offload_on_cpu, monkeypatch):
    # Failing before: PEFT placed a new adapter on base_layer.weight's device, the host for an offloaded master.
    from peft import LoraConfig

    model = _tiny_qwen3_peft()
    f8t.mark_fp8_lora(model.base_model.model, f8t.FP8_LORA_RECIPE)
    f8t.enable_fp8_lora(model, allow_cpu = True)
    layers = [m for m in model.modules() if hasattr(m, "lora_A")]
    compute = torch.device("meta")  # stands in for cuda; the master stays on the CPU host
    for m in layers:
        m.base_layer.__dict__["_unsloth_fp8_lora"].device = compute
    moved = []
    real_to = torch.nn.Module.to

    def spy(self, *args, **kwargs):
        if args and args[0] == compute:
            moved.append(self)
            return self
        return real_to(self, *args, **kwargs)

    monkeypatch.setattr(torch.nn.Module, "to", spy)
    # Requantizing onto the meta stand-in cannot run; placement is what this test checks.
    monkeypatch.setattr(FL.Fp8LoRABaseState, "refresh", lambda self, weight: self)
    model.add_adapter("second", LoraConfig(r = 4, target_modules = ["q_proj"]))
    added = [m for m in layers if "second" in m.lora_A]
    assert added and all(any(x is m.lora_A["second"] for x in moved) for m in added)
    monkeypatch.setattr(torch.nn.Module, "to", real_to)
    for m in layers:
        m.base_layer.__dict__["_unsloth_fp8_lora"].device = torch.device("cpu")
    model.add_adapter("third", LoraConfig(r = 4, target_modules = ["v_proj"], use_dora = True))
    dora = [m for m in layers if "third" in m.lora_A]
    assert dora and not any(m.base_layer.__dict__["_unsloth_fp8_lora"].offloaded for m in dora)


def test_low_cpu_mem_adapters_stay_on_meta_for_the_loader(offload_on_cpu):
    # Failing before: the placement wrapper moved PEFT's meta parameters (low_cpu_mem_usage) and raised.
    from peft import LoraConfig

    model = _tiny_qwen3_peft()
    f8t.mark_fp8_lora(model.base_model.model, f8t.FP8_LORA_RECIPE)
    f8t.enable_fp8_lora(model, allow_cpu = True)
    model.add_adapter("second", LoraConfig(r = 4, target_modules = ["q_proj"]), low_cpu_mem_usage = True)
    added = [m for m in model.modules() if hasattr(m, "lora_A") and "second" in m.lora_A]
    assert added and all(m.lora_A["second"].weight.device.type == "meta" for m in added)


def test_a_dora_adapter_restores_the_master_before_placement(offload_on_cpu, monkeypatch):
    # Failing before: PEFT placed the DoRA adapter (and its magnitude) by the host master, then the master moved.
    from peft import LoraConfig
    from peft.tuners.lora import LoraLayer

    model = _tiny_qwen3_peft()
    f8t.mark_fp8_lora(model.base_model.model, f8t.FP8_LORA_RECIPE)
    f8t.enable_fp8_lora(model, allow_cpu = True)
    seen = []
    real = LoraLayer._move_adapter_to_device_of_base_layer

    def spy(self, adapter_name, *args, **kwargs):
        if adapter_name == "dora":
            seen.append(self.base_layer.__dict__["_unsloth_fp8_lora"].offloaded)
        return real(self, adapter_name, *args, **kwargs)

    monkeypatch.setattr(LoraLayer, "_move_adapter_to_device_of_base_layer", spy)
    model.add_adapter("dora", LoraConfig(r = 4, target_modules = ["v_proj"], use_dora = True))
    assert seen and not any(seen)


def test_load_state_dict_requantizes_for_compiled_calls():
    # Failing before: refresh() is skipped while compiled, so new base weights kept the old fp8 copy.
    torch._dynamo.reset()
    lin = _frozen_linear()
    FL.attach_fp8_lora_base(lin, offload = False)
    compiled = torch.compile(lin, backend = "aot_eager", fullgraph = True)
    x = torch.randn(32, 256).bfloat16()
    assert compiled(x).abs().sum() > 0
    lin.load_state_dict({"weight": torch.zeros_like(lin.weight)})
    assert compiled(x).abs().sum() == 0


def test_a_dtype_move_requantizes_for_compiled_calls():
    # Failing before: .to() moved the master but kept the fp8 copy's dtype / device until an eager forward.
    lin = _frozen_linear()
    state = FL.attach_fp8_lora_base(lin, offload = False)
    lin.half()
    assert state.dtype == torch.float16 and state._key == state._weight_key(lin.weight)


def test_a_pissa_adapter_requantizes_the_rewritten_base(offload_on_cpu):
    # Failing before: PiSSA rewrites the base to its residual during add_adapter; compiled calls kept the old fp8.
    from peft import LoraConfig

    model = _tiny_qwen3_peft()
    f8t.mark_fp8_lora(model.base_model.model, f8t.FP8_LORA_RECIPE)
    f8t.enable_fp8_lora(model, allow_cpu = True)
    model.add_adapter(
        "pissa",
        LoraConfig(
            r = 8, lora_alpha = 16, target_modules = ["q_proj"], init_lora_weights = "pissa_niter_1"
        ),
    )
    layers = [m for m in model.modules() if hasattr(m, "lora_A") and "pissa" in m.lora_A]
    assert layers
    for m in layers:
        state = m.base_layer.__dict__["_unsloth_fp8_lora"]
        assert state._key == state._weight_key(m.base_layer.weight)


def test_mamba_raw_weight_projections_keep_the_master(offload_on_cpu):
    # Failing before: an offloaded x_proj / dt_proj / out_proj master met the device adapter delta in Unsloth's
    # fused Mamba path (W + delta), a device mismatch on the GPU.
    from peft import LoraConfig, get_peft_model

    class MambaMixer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.in_proj = torch.nn.Linear(64, 64, bias = False, dtype = torch.bfloat16)
            self.out_proj = torch.nn.Linear(64, 64, bias = False, dtype = torch.bfloat16)

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.mixer = MambaMixer()

    model = get_peft_model(Model(), LoraConfig(r = 4, target_modules = ["in_proj", "out_proj"]))
    f8t.mark_fp8_lora(model.base_model.model, f8t.FP8_LORA_RECIPE)
    assert len(f8t.enable_fp8_lora(model, allow_cpu = True)) == 2
    mixer = model.base_model.model.mixer
    assert not mixer.out_proj.base_layer.__dict__["_unsloth_fp8_lora"].offloaded
    assert mixer.in_proj.base_layer.__dict__["_unsloth_fp8_lora"].offloaded


def test_legacy_scaled_mm_tuple_return(monkeypatch):
    # torch 2.4 returns (output, amax) from torch._scaled_mm; the fp8 path must take the output.
    real = torch._scaled_mm
    monkeypatch.setattr(torch, "_scaled_mm", lambda *a, **k: (real(*a, **k), torch.zeros(())))
    W = _weight()
    state = FL.Fp8LoRABaseState(W)
    y = state.matmul(torch.randn(32, 256).bfloat16(), W)
    dx = state.dgrad(torch.randn(32, 384).bfloat16(), W)
    assert torch.is_tensor(y) and torch.is_tensor(dx)


@pytest.mark.parametrize("arch", ["falcon_h1", "wavlm"])
def test_parents_reading_a_child_weight_keep_its_master(offload_on_cpu, arch):
    # Failing before: only class names with "mamba" were kept; Falcon-H1's mixer and WavLM attention read
    # out_proj / q_proj .weight themselves, so an offloaded master met the device activations.

    if arch == "falcon_h1":
        mod = pytest.importorskip("transformers.models.falcon_h1.modeling_falcon_h1")
        parent_cls, leaf = mod.FalconH1Mixer, "out_proj"
    else:
        mod = pytest.importorskip("transformers.models.wavlm.modeling_wavlm")
        parent_cls, leaf = mod.WavLMAttention, "q_proj"
    parent = parent_cls.__new__(parent_cls)
    torch.nn.Module.__init__(parent)
    setattr(parent, leaf, torch.nn.Linear(4, 4))
    model = torch.nn.Module()
    model.block = parent
    assert f8t._mamba_raw_weight_leaf(model, f"block.{leaf}")


def test_refresh_under_inference_mode_builds_trainable_fp8_tensors():
    # Failing before: a merge / unmerge inside inference_mode rebuilt the cache as inference tensors, which the
    # next training step could not save for backward.
    lin = _frozen_linear()
    state = FL.attach_fp8_lora_base(lin, offload = False)
    with torch.inference_mode():
        state._key = None
        state.refresh(lin.weight)
    assert not state.w8.is_inference()
    x = torch.randn(32, 256).bfloat16().requires_grad_(True)
    lin(x).float().sum().backward()
    assert x.grad is not None


def test_grpo_out_of_place_merge_leaves_the_fp8_copy_alone(offload_on_cpu):
    # Failing before: the merge wrapper requantized the temporarily merged base into the live fp8 state, and the
    # sync helper restored only param.data, so compiled forwards kept the merged base under an unmerged adapter.
    from unsloth.models.rl_replacements import _unsloth_vllm_peft_merge_out_of_place

    model = _tiny_qwen3_peft()
    f8t.mark_fp8_lora(model.base_model.model, f8t.FP8_LORA_RECIPE)
    f8t.enable_fp8_lora(model, allow_cpu = True)
    layer = next(m for m in model.modules() if hasattr(m, "lora_A") and hasattr(m, "base_layer"))
    state = layer.base_layer.__dict__["_unsloth_fp8_lora"]
    w8, key, offloaded = state.w8, state._key, state.offloaded
    merged = _unsloth_vllm_peft_merge_out_of_place(layer.base_layer, [layer])
    assert merged
    assert state.w8 is w8 and state._key == key and state.offloaded == offloaded


def test_a_base_weight_shared_by_two_lora_layers_stays_16bit(offload_on_cpu):
    # Failing before: both readers got their own fp8 copy; merging through one left the other's copy stale (PEFT
    # merges in place: same storage and version), so the two layers disagreed after merge_adapter.
    from peft import LoraConfig, get_peft_model

    class Tied(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.a = torch.nn.Linear(64, 64, bias = False, dtype = torch.bfloat16)
            self.b = torch.nn.Linear(64, 64, bias = False, dtype = torch.bfloat16)
            self.b.weight = self.a.weight

    model = get_peft_model(
        Tied(), LoraConfig(r = 4, target_modules = ["a", "b"], init_lora_weights = False)
    )
    f8t.mark_fp8_lora(model.base_model.model, f8t.FP8_LORA_RECIPE)
    assert f8t.enable_fp8_lora(model, allow_cpu = True) == []
    inner = model.base_model.model
    model.merge_adapter()
    x = torch.randn(32, 64).bfloat16()
    with torch.no_grad():
        torch.testing.assert_close(inner.a(x), inner.b(x))


def test_block_swapped_layers_are_not_converted(offload_on_cpu):
    # Failing before: layers managed by block swap (offload_layers) were converted, and an evicted layer's empty
    # placeholder weight crashed quantization ("cannot compute the inf norm on an empty tensor").
    import types

    model = _tiny_qwen3_peft()
    layers = model.base_model.model.model.layers
    layers._unsloth_block_swap = types.SimpleNamespace(indices = [1])
    for m in layers[1].modules():
        if isinstance(m, torch.nn.Linear) and not m.weight.requires_grad:
            m.weight = torch.nn.Parameter(torch.empty(0, dtype = m.weight.dtype), requires_grad = False)
    f8t.mark_fp8_lora(model.base_model.model, f8t.FP8_LORA_RECIPE)
    converted = f8t.enable_fp8_lora(model, allow_cpu = True)
    assert converted and all(".layers.0." in name for name in converted)


def test_sharded_or_preloaded_weights_are_not_eligible():
    # Failing before: native FSDP2 sharded loading leaves a rank-local row chunk (tagged for the later gather) in a
    # plain Parameter, so its fp8 copy was quantized from one shard and the forward produced the wrong shape.
    linear = torch.nn.Linear(64, 64, bias = False).bfloat16().requires_grad_(False)
    linear.weight = torch.nn.Parameter(linear.weight.data[:32].clone(), requires_grad = False)
    assert "shard" in FL.eligible_base(linear)
    full = torch.nn.Linear(64, 64, bias = False).bfloat16().requires_grad_(False)
    full.weight.__dict__["_unsloth_fsdp2_preload"] = object()
    assert "shard" in FL.eligible_base(full)


def test_get_peft_model_converts_after_block_swap():
    # Failing before: FastLanguageModel.get_peft_model converted inside patch_peft_model, before install_block_swap
    # picked its layers, so streamed layers kept fp8 state over evicted placeholder weights.
    import inspect
    from unsloth.models.llama import FastLlamaModel

    get_src = inspect.getsource(FastLlamaModel.get_peft_model)
    assert "_defer_fp8_lora = True" in get_src
    swap = get_src.index("install_block_swap(")
    assert get_src.index("enable_fp8_lora(model)", swap) > swap
    patch_src = inspect.getsource(FastLlamaModel.patch_peft_model)
    assert "if not _defer_fp8_lora:" in patch_src


def test_column_major_weights_are_not_eligible():
    # Failing before: a [128, 64] weight stored as a transposed [64, 128] passed, and matmul_lora's
    # transpose check sent its forward to the input-gradient GEMM (shape error, or a wrong product when square).
    linear = torch.nn.Linear(64, 128, bias = False).bfloat16().requires_grad_(False)
    linear.weight = torch.nn.Parameter(torch.randn(64, 128).bfloat16().t(), requires_grad = False)
    assert "contiguous" in FL.eligible_base(linear)


@pytest.mark.parametrize("variant", ["bdlora", "velora"])
def test_a_variant_adapter_restores_the_master_before_placement(
    offload_on_cpu, monkeypatch, variant
):
    # Failing before: only DoRA brought the host master back, so BD-LoRA's block-diagonal A and VeLoRA's
    # BufferDict, both built by their variant init on base_layer.weight's device, stayed on the host.
    import peft
    from peft import LoraConfig
    from peft.tuners.lora import LoraLayer

    if variant == "bdlora":
        if not hasattr(peft, "BdLoraConfig"):
            pytest.skip("this PEFT has no BD-LoRA")
        config = LoraConfig(
            r = 4,
            target_modules = ["q_proj"],
            use_bdlora = peft.BdLoraConfig(nblocks = 2, target_modules_bd_a = ["q_proj"]),
        )
    else:
        if not hasattr(peft, "VeloraConfig"):
            pytest.skip("this PEFT has no VeLoRA")
        config = LoraConfig(
            r = 4,
            target_modules = ["q_proj"],
            velora_config = peft.VeloraConfig(num_groups = 2, init_type = "random"),
        )
    model = _tiny_qwen3_peft()
    f8t.mark_fp8_lora(model.base_model.model, f8t.FP8_LORA_RECIPE)
    f8t.enable_fp8_lora(model, allow_cpu = True)
    seen = []
    real = LoraLayer._move_adapter_to_device_of_base_layer

    def spy(self, adapter_name, *args, **kwargs):
        if adapter_name == variant:
            seen.append(self.base_layer.__dict__["_unsloth_fp8_lora"].offloaded)
        return real(self, adapter_name, *args, **kwargs)

    monkeypatch.setattr(LoraLayer, "_move_adapter_to_device_of_base_layer", spy)
    try:
        model.add_adapter(variant, config)
    except (TypeError, ValueError) as exc:
        pytest.skip(f"this PEFT cannot build {variant} here: {exc}")
    assert seen and not any(seen)
