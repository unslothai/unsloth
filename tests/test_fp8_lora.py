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
from real_accelerator import has_real_cuda

pytestmark = pytest.mark.skipif(
    not hasattr(torch, "float8_e4m3fn") or not hasattr(torch, "_scaled_mm"),
    reason = "needs torch float8 dtypes",
)


def _cpu_scaled_mm_works(rowwise = False):
    if not hasattr(torch, "float8_e4m3fn") or not hasattr(torch, "_scaled_mm"):
        return False
    try:
        a = torch.zeros(16, 16).to(torch.float8_e4m3fn)
        scale_a = torch.ones((16, 1) if rowwise else (), dtype = torch.float32)
        scale_b = torch.ones((1, 16) if rowwise else (), dtype = torch.float32)
        torch._scaled_mm(a, a.t(), scale_a = scale_a, scale_b = scale_b, out_dtype = torch.bfloat16)
        return True
    except (NotImplementedError, RuntimeError):
        return False


needs_cpu_scaled_mm = pytest.mark.skipif(
    not _cpu_scaled_mm_works(), reason = "this torch build has no CPU torch._scaled_mm kernel"
)

FP8_FWD_TOL = 0.06
FP8_DGRAD_TOL = 0.09


@pytest.mark.gpu
@pytest.mark.skipif(
    not has_real_cuda()
    or torch.version.hip is not None
    or torch.cuda.get_device_capability() < (8, 9),
    reason = "requires NVIDIA compute capability 8.9+ for FP8 CUDA cast kernels",
)
@pytest.mark.skipif(FL.triton is None, reason = "requires Triton for the CUDA cast kernels")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("fp8_dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
@pytest.mark.parametrize("rowwise", [False, True])
def test_cuda_casts_match_native_rounding(dtype, fp8_dtype, rowwise):
    limit = torch.finfo(fp8_dtype).max
    for seed in range(16):
        generator = torch.Generator(device = "cuda").manual_seed(seed)
        x = torch.randn(32, 256, device = "cuda", dtype = dtype, generator = generator)
        if seed == 0:
            x[0].fill_(-0.0)
            x[1].fill_(torch.finfo(dtype).tiny)
            x[2, 0] = torch.finfo(dtype).max
        amax = x.float().abs().amax(dim = 1, keepdim = True) if rowwise else x.float().abs().amax()
        amax = amax.clamp_min(1e-12)
        expected = (x.float() * (limit / amax)).clamp(-limit, limit).to(fp8_dtype)
        quantize = FL.quantize_rowwise if rowwise else FL.quantize_tensorwise
        actual, inverse = quantize(x, fp8_dtype)
        assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))
        assert torch.equal(inverse, amax / limit)


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


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@needs_cpu_scaled_mm
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
    assert 0.01 < e_fwd < FP8_FWD_TOL, e_fwd
    assert 0.01 < e_dgrad < FP8_DGRAD_TOL, e_dgrad
    assert torch.equal(state.matmul(dy, W.t()), dx)
    assert FL.FP8_LORA_STATS["fwd"] == 1 and FL.FP8_LORA_STATS["dgrad"] == 2


@needs_cpu_scaled_mm
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
    assert seen == [torch.float32, torch.float32]
    assert y.dtype == torch.float16 and torch.isfinite(y).all()


@needs_cpu_scaled_mm
def test_large_fp16_values_stay_finite():
    W = torch.full((32, 64), 0.5, dtype = torch.float16)
    x = torch.full((16, 64), 1800.0, dtype = torch.float16)  # each output 57600 < 65504
    y = FL.Fp8LoRABaseState(W).matmul(x, W)
    assert torch.isfinite(y).all()
    assert abs(y.float().mean().item() - 57600.0) / 57600.0 < 0.07


@needs_cpu_scaled_mm
def test_fp16_results_at_the_fp16_limit_stay_finite():
    W = torch.full((32, 64), 0.5, dtype = torch.float16)
    x = torch.full((16, 64), 2047.0, dtype = torch.float16)  # each output 65504, the fp16 maximum
    state = FL.Fp8LoRABaseState(W)
    assert torch.isfinite(state.matmul(x, W)).all()
    dy = torch.full((16, 32), 4094.0, dtype = torch.float16)  # each dgrad output 65504
    assert torch.isfinite(state.dgrad(dy, W)).all()


@needs_cpu_scaled_mm
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


@needs_cpu_scaled_mm
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
    assert torch.equal(state.w8t.view(torch.uint8), w8.t().contiguous().view(torch.uint8))
    with torch.no_grad():
        W.mul_(2.0)
    state.refresh(W)
    assert state.quantizations == 2
    assert _rel(state.matmul(x, W), x.float() @ W.float().t()) < FP8_FWD_TOL


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
@needs_cpu_scaled_mm
def test_lora_kernels_run_fp8_and_keep_lora_grads_16bit(kind):
    outs_ref, dX_ref, grads_ref, _ = _run_block(kind, fp8 = False)
    FL.reset_fp8_lora_stats()
    outs, dX, grads, projs = _run_block(kind, fp8 = True)
    n = len(projs)
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


@needs_cpu_scaled_mm
def test_peft_path_base_forward_and_backward():
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


@needs_cpu_scaled_mm
def test_enable_fp8_lora_on_a_peft_model():
    model = _tiny_qwen3_peft()
    loss_ref, grads_ref = _loss_and_grads(model)
    f8t.mark_fp8_lora(model.base_model.model, f8t.FP8_LORA_RECIPE)
    converted = f8t.enable_fp8_lora(model, allow_cpu = True)
    assert len(converted) == 2 * 7
    assert P.engaged(model)[f8t.FP8_LORA_RECIPE]["modules"] == 14
    loss, grads = _loss_and_grads(model)
    assert FL.FP8_LORA_STATS["fwd"] == 14 and FL.FP8_LORA_STATS["dgrad"] == 11
    assert abs(loss - loss_ref) / loss_ref < 0.01
    assert grads.keys() == grads_ref.keys()
    for name, g in grads.items():
        assert g.dtype == grads_ref[name].dtype
        assert _cos(g, grads_ref[name]) > 0.98, (name, _cos(g, grads_ref[name]))
    _loss_and_grads(model)
    assert FL.FP8_LORA_STATS["quantize"] == 14


@pytest.mark.parametrize("flag", [None, False, True])
@pytest.mark.parametrize("private_namespace", [False, True])
def test_fp8_loss_ignores_model_output_flags_only_on_its_head(monkeypatch, flag, private_namespace):
    import copy
    import types
    from unsloth_zoo.fused_losses import cross_entropy_loss as ce
    from unsloth_zoo.fused_losses import forward_adapter

    monkeypatch.setattr(ce, "_CHUNKED_CE", True)
    monkeypatch.setattr(ce, "_FUSED_CE_COMPILE_DISABLED", False)
    monkeypatch.setattr(ce, "TARGET_GB", 0)
    monkeypatch.setattr(ce, "N_CHUNKS", 0)
    monkeypatch.setattr(ce._chunked_logprob, "_FUSED_LINEAR", True)
    routes = []

    def observe_loss(**kwargs):
        extras = {
            k: kwargs[k] for k in ("output_attentions", "output_hidden_states") if k in kwargs
        }
        routes.append(ce._use_chunked_ce(extras, None))
        return kwargs["hidden_states"].sum() * 0

    monkeypatch.setattr(forward_adapter, "unsloth_fused_ce_loss", observe_loss)
    model = _tiny_qwen3_peft()
    owner = model.get_base_model()
    ordinary = _tiny_qwen3_peft().get_base_model()
    hidden = torch.zeros(2, 3, 128)
    labels = torch.zeros(2, 3, dtype = torch.long)

    def dispatch(self, hidden, labels):
        return unsloth_fused_lm_head_loss(
            hidden,
            self.get_output_embeddings(),
            labels,
            output_attentions = flag,
            output_hidden_states = flag,
        )

    original = forward_adapter.unsloth_fused_lm_head_loss
    namespace = {"__builtins__": __builtins__} if private_namespace else dispatch.__globals__
    monkeypatch.setitem(namespace, "unsloth_fused_lm_head_loss", original)
    function = types.FunctionType(dispatch.__code__, namespace, closure = dispatch.__closure__)
    attribute = "_old_forward" if private_namespace else "forward"
    setattr(owner, attribute, types.MethodType(function, owner))
    setattr(ordinary, attribute, types.MethodType(function, ordinary))

    def run(target):
        getattr(target, attribute)(hidden, labels)
        return routes[-1]

    assert not run(owner)
    f8t.mark_fp8_lora(owner, f8t.FP8_LORA_RECIPE)
    assert f8t.enable_fp8_lora(model, allow_cpu = True)
    assert run(owner)
    assert namespace["unsloth_fused_lm_head_loss"] is original
    assert forward_adapter.unsloth_fused_lm_head_loss is original
    assert getattr(ordinary, attribute).__func__ is function
    assert not run(ordinary)
    clone = copy.deepcopy(model)
    assert run(clone.get_base_model())
    f8t.restore_fp8_lora_for_inference(clone)
    assert not run(clone.get_base_model())
    assert run(owner)
    f8t.restore_fp8_lora_for_inference(model)
    assert not run(owner)


@pytest.mark.parametrize("decorated", [False, True])
@pytest.mark.parametrize("private_namespace", [False, True])
def test_fp8_loss_route_traces_fullgraph_without_changing_loss_or_input_gradient(
    monkeypatch, decorated, private_namespace
):
    import types
    from transformers.loss.loss_utils import ForCausalLMLoss
    from unsloth_zoo.fused_losses import forward_adapter

    def cpu_reference(hidden_states, lm_head_weight, lm_head_bias, labels, n_items, **kwargs):
        assert "output_attentions" not in kwargs
        assert "output_hidden_states" not in kwargs
        return ForCausalLMLoss(
            logits = F.linear(hidden_states, lm_head_weight, lm_head_bias),
            labels = labels,
            vocab_size = lm_head_weight.shape[0],
            num_items_in_batch = n_items,
        )

    monkeypatch.setattr(forward_adapter, "unsloth_fused_ce_loss", cpu_reference)
    model = _tiny_qwen3_peft()
    owner = model.get_base_model()
    head = model.get_output_embeddings()
    labels = torch.tensor([[1, 2, 3, -100], [-100, 3, 2, 1]])
    x = torch.randn(2, 4, 128, dtype = torch.bfloat16, requires_grad = True)

    def routed(self, hidden):
        return unsloth_fused_lm_head_loss(
            hidden,
            self.get_output_embeddings(),
            labels,
            num_items_in_batch = 5,
            output_attentions = None,
            output_hidden_states = None,
        )

    namespace = {"__builtins__": __builtins__} if private_namespace else routed.__globals__
    monkeypatch.setitem(
        namespace, "unsloth_fused_lm_head_loss", forward_adapter.unsloth_fused_lm_head_loss
    )
    function = types.FunctionType(routed.__code__, namespace, closure = routed.__closure__)
    if decorated:
        from transformers.utils.generic import can_return_tuple
        function = can_return_tuple(function)
    owner.forward = types.MethodType(function, owner)
    f8t.mark_fp8_lora(owner, f8t.FP8_LORA_RECIPE)
    f8t.enable_fp8_lora(model, allow_cpu = True)
    expected = ForCausalLMLoss(
        logits = head(x),
        labels = labels,
        vocab_size = 256,
        num_items_in_batch = 5,
    )
    (expected_grad,) = torch.autograd.grad(expected, x)
    torch._dynamo.reset()
    compiled = torch.compile(owner, backend = "aot_eager", fullgraph = True)
    actual = compiled(x)
    (actual_grad,) = torch.autograd.grad(actual, x)
    assert torch.equal(actual, expected)
    assert torch.equal(actual_grad, expected_grad)
    if decorated and private_namespace:
        from torch._dynamo.utils import counters

        before = counters["stats"]["unique_graphs"]
        compiled(x)
        stable = counters["stats"]["unique_graphs"] - before
        f8t.restore_fp8_lora_for_inference(model)
        f8t.enable_fp8_lora(model, allow_cpu = True)
        before = counters["stats"]["unique_graphs"]
        compiled(x)
        rebound = counters["stats"]["unique_graphs"] - before
        print("LOSS_REBIND_GRAPHS", stable, rebound)
        assert stable == rebound == 0


@pytest.mark.parametrize("return_dict", [False, True])
def test_fp8_loss_route_preserves_installed_transformers_decorators(monkeypatch, return_dict):
    from transformers.loss.loss_utils import ForCausalLMLoss
    from unsloth_zoo.fused_losses import forward_adapter, forward_install

    model = _tiny_qwen3_peft()
    owner = model.get_base_model()
    cls = type(owner)
    original = cls.forward
    monkeypatch.setattr(cls, "forward", original)
    for name in ("_PATCHED", "_FAILED", "_UNMATCHED"):
        monkeypatch.setattr(forward_install, name, dict(getattr(forward_install, name)))
    monkeypatch.setenv("UNSLOTH_FUSED_FORWARD", "1")
    monkeypatch.setenv("UNSLOTH_RETURN_LOGITS", "0")
    assert forward_install.install_for_class(cls), forward_install.audit()
    installed = cls.forward
    calls = []

    def cpu_reference(hidden_states, lm_head_weight, lm_head_bias, labels, n_items, **kwargs):
        calls.append(kwargs)
        assert "output_attentions" not in kwargs
        assert "output_hidden_states" not in kwargs
        return ForCausalLMLoss(
            logits = F.linear(hidden_states, lm_head_weight, lm_head_bias),
            labels = labels,
            vocab_size = lm_head_weight.shape[0],
            num_items_in_batch = n_items,
        )

    monkeypatch.setattr(forward_adapter, "unsloth_fused_ce_loss", cpu_reference)
    monkeypatch.setattr(FL, "_MIN_ROWS", 1024)
    owner.loss_function = forward_install._STOCK_CAUSAL_LM_LOSS
    f8t.mark_fp8_lora(owner, f8t.FP8_LORA_RECIPE)
    f8t.enable_fp8_lora(model, allow_cpu = True)
    assert owner.__dict__.get("_unsloth_fp8_lora_loss_forwards")
    ids = torch.tensor([[1, 3, 2, 5]])
    result = owner(
        input_ids = ids,
        labels = ids,
        use_cache = False,
        return_dict = return_dict,
        output_attentions = False,
        output_hidden_states = False,
    )
    assert calls
    assert isinstance(result, tuple) is (not return_dict)
    loss = result.loss if return_dict else result[0]
    assert torch.isfinite(loss)
    assert cls.forward is installed
    assert FL.FP8_LORA_STATS["fwd"] == 0
    assert FL.FP8_LORA_STATS["fwd_16bit"] > 0
    f8t.restore_fp8_lora_for_inference(model)
    assert owner.forward.__func__ is installed


def test_no_request_no_conversion_and_strict_zero_engagement(monkeypatch):
    model = _tiny_qwen3_peft()
    assert f8t.enable_fp8_lora(model) == []
    for module in model.modules():
        assert "_unsloth_fp8_lora" not in module.__dict__
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


@needs_cpu_scaled_mm
def test_compile_aot_eager_fullgraph():
    torch._dynamo.reset()
    base = torch.nn.Linear(256, 384, bias = False, dtype = torch.bfloat16).requires_grad_(False)
    FL.attach_fp8_lora_base(base)
    proj = _Proj(256, 384)
    FL.attach_fp8_lora_base(proj.base_layer)

    def step(x):
        y = base(x)
        W, W_quant, A, B, S = FLK.get_lora_parameters(proj)
        z = FLK.LoRA_W.apply(y[..., :256].contiguous(), W, W_quant, A, B, S)
        return z.float().square().mean()

    x = torch.randn(2, 16, 256).bfloat16().requires_grad_(True)
    eager = step(x)
    eager.backward()
    g_eager = x.grad.clone()
    x.grad = None
    FL.reset_fp8_lora_stats()
    compiled = torch.compile(step, backend = "aot_eager", fullgraph = True)
    out = compiled(x)
    out.backward()
    torch.testing.assert_close(out, eager)
    torch.testing.assert_close(x.grad, g_eager)
    assert FL.FP8_LORA_STATS["fwd"] == 2 and FL.FP8_LORA_STATS["dgrad"] == 2, FL.FP8_LORA_STATS
    explain = torch._dynamo.explain(step)(x)
    assert explain.graph_break_count == 0


def test_resolve_refusals_and_names():
    R = f8t.FP8_LORA_RECIPE
    with pytest.raises(P.PrecisionRefused, match = "fp8 tensor cores"):
        f8t.resolve_fp8_lora("fp8_lora", gpu = "sm75")
    with pytest.raises(P.PrecisionRefused, match = "fp8 tensor cores"):
        f8t.resolve_fp8_lora("fp8_lora", gpu = "sm80")
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
    assert f8t.resolve_fp8_lora("fp8_lora", gpu = "sm120", dtype = torch.float16) == "fp8_lora_rowwise"
    assert FL.scaling_for_class("sm120") == "row" and FL.scaling_for_class("sm89") == "tensor"
    assert f8t.resolve_fp8_lora(False) is None and f8t.resolve_fp8_lora(None) is None
    gate = P.get_recipe(R).gate("sm89").status
    if gate != "named":
        with pytest.raises(P.PrecisionRefused, match = "not been measured faster"):
            f8t.resolve_fp8_lora(True, gpu = "sm89")


@pytest.mark.parametrize("gpu", [None, "sm75", "rocm"])
def test_loader_refuses_before_any_load(monkeypatch, gpu):
    from unsloth.models import loader
    monkeypatch.setattr(P, "current_gpu_class", lambda: gpu)
    for cls in (loader.FastLanguageModel, loader.FastModel):
        with pytest.raises(P.PrecisionRefused, match = "load_in_4bit = False"):
            cls.from_pretrained(
                model_name = "unsloth/does-not-exist", load_in_4bit = True, float8_training = "fp8_lora"
            )
        with pytest.raises(P.PrecisionRefused):
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


@pytest.fixture
def rowwise(monkeypatch):
    if not _cpu_scaled_mm_works(rowwise = True):
        pytest.skip("this torch build has no CPU rowwise torch._scaled_mm kernel")
    monkeypatch.setattr(FL, "_SCALING", "row")
    monkeypatch.setattr(FL, "_DGRAD_DTYPE", FL.E4M3)
    yield


@needs_cpu_scaled_mm
def test_rowwise_numerics_and_outlier_tokens(rowwise):
    W = _weight()
    x = torch.randn(64, 256).bfloat16()
    x[0] *= 1e5
    dy = torch.randn(64, 384).bfloat16()
    state = FL.Fp8LoRABaseState(W)
    assert state.w_inv_fwd.shape == (1, 384) and state.w_inv_dgrad.shape == (1, 256)
    y = state.matmul(x, W)
    ref = x.float() @ W.float().t()
    rest = _rel(y[1:], ref[1:])
    assert rest < FP8_FWD_TOL, rest
    FL._SCALING = "tensor"
    tensorwise = _rel(FL.Fp8LoRABaseState(W).matmul(x, W)[1:], ref[1:])
    FL._SCALING = "row"
    assert tensorwise > 2 * rest, (tensorwise, rest)
    dx = state.dgrad(dy, W)
    assert _rel(dx, dy.float() @ W.float()) < FP8_FWD_TOL


@needs_cpu_scaled_mm
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
    model = _tiny_qwen3_peft()
    calls = []
    real = f8t.enable_fp8_lora
    monkeypatch.setattr(
        f8t, "enable_fp8_lora", lambda m, r = None, **k: calls.append(r) or real(m, r, allow_cpu = True)
    )
    f8t.mark_fp8_lora(model, f8t.FP8_LORA_RECIPE)
    assert calls == [f8t.FP8_LORA_RECIPE]
    assert P.engaged(model)[f8t.FP8_LORA_RECIPE]["modules"] == 14


@needs_cpu_scaled_mm
def test_default_no_transposed_cache_matches_cached(monkeypatch):
    W = _weight()
    dy = torch.randn(32, 384).bfloat16()
    lean = FL.Fp8LoRABaseState(W)
    assert lean.w8t is None
    monkeypatch.setattr(FL, "_CACHE_T", True)
    cached = FL.Fp8LoRABaseState(W)
    assert cached.w8t is not None
    torch.testing.assert_close(lean.dgrad(dy, W), cached.dgrad(dy, W), rtol = 0, atol = 0)


@needs_cpu_scaled_mm
def test_compiled_graph_has_no_fp8_transpose_outside_the_ops():
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
    assert len(graphs) >= 2
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


@pytest.fixture
def offload_on_cpu(monkeypatch):
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
    assert lin.weight is param
    assert torch.equal(lin.weight, W0)
    numel = 256 * 384
    assert state.resident_bytes(lin) == {
        "fp8": numel + 8,
        "master_device": 0,
        "master_host": 2 * numel,
    }
    assert ref_state.resident_bytes(ref) == {
        "fp8": numel + 8,
        "master_device": 2 * numel,
        "master_host": 0,
    }
    assert FL.fp8_lora_memory(torch.nn.Sequential(lin, ref))["total"] == {
        "fp8": 2 * (numel + 8),
        "master_device": 2 * numel,
        "master_host": 2 * numel,
    }


@needs_cpu_scaled_mm
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
    assert FL.FP8_LORA_STATS["quantize"] == 2


def test_host_master_decode_rows_use_the_dequantized_fp8_weight(offload_on_cpu):
    lin = _frozen_linear()
    W0 = lin.weight.detach().clone()
    state = FL.attach_fp8_lora_base(lin)
    x = torch.randn(1, 1, 256).bfloat16()
    dense = (state.w8.float() * state.w_inv).bfloat16()
    torch.testing.assert_close(state.matmul(x, lin.weight), x @ dense.t())
    torch.testing.assert_close(lin(x), x @ dense.t())
    dy = torch.randn(1, 1, 384).bfloat16()
    torch.testing.assert_close(state.matmul(dy, lin.weight.t()), dy @ dense)
    assert _rel(lin(x), x @ W0.t()) < FP8_FWD_TOL
    assert FL.FP8_LORA_STATS["fwd"] == 0


@needs_cpu_scaled_mm
@pytest.mark.parametrize("backend", ["llama", "vision"])
def test_generation_restores_exact_base_then_training_reenables_fp8(
    offload_on_cpu, monkeypatch, backend
):
    from unsloth.models import llama, vision

    owner = llama if backend == "llama" else vision
    api = owner.FastLlamaModel if backend == "llama" else owner.FastBaseModel
    monkeypatch.setenv("UNSLOTH_RETURN_HIDDEN_STATES", "0")
    monkeypatch.setattr(
        owner, "enable_fp8_lora", lambda model: f8t.enable_fp8_lora(model, allow_cpu = True)
    )
    model = _tiny_qwen3_peft().eval()
    ids = torch.randint(0, 256, (1, 32))
    with torch.no_grad():
        expected = model(ids).logits
    f8t.mark_fp8_lora(model.base_model.model, f8t.FP8_LORA_RECIPE)
    f8t.enable_fp8_lora(model, allow_cpu = True)
    assert FL.fp8_lora_memory(model)["layers"]
    api.for_inference(model)
    with torch.no_grad():
        actual = model(ids).logits
    assert torch.equal(actual, expected)
    assert not FL.fp8_lora_memory(model)["layers"]
    api.for_training(model, use_gradient_checkpointing = False)
    assert len(FL.fp8_lora_memory(model)["layers"]) == 14
    FL.reset_fp8_lora_stats()
    _, grads = _loss_and_grads(model)
    assert FL.FP8_LORA_STATS["fwd"] == 14
    assert any(torch.count_nonzero(grad) for grad in grads.values())


def test_enabling_an_offloaded_base_again_keeps_engagement(offload_on_cpu, monkeypatch):
    model = _tiny_qwen3_peft()
    f8t.mark_fp8_lora(model.base_model.model, f8t.FP8_LORA_RECIPE)
    expected = f8t.enable_fp8_lora(model, allow_cpu = True)
    states = [
        module.__dict__["_unsloth_fp8_lora"]
        for module in model.modules()
        if "_unsloth_fp8_lora" in module.__dict__
    ]
    for state in states:
        assert state.offloaded
        state.device = torch.device("cuda:0")
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (8, 9))
    monkeypatch.setattr(torch.version, "hip", None)
    monkeypatch.setenv("UNSLOTH_STRICT_ENGAGE", "1")
    assert f8t.enable_fp8_lora(model) == expected
    assert P.engaged(model)[f8t.FP8_LORA_RECIPE]["modules"] == len(expected)
    assert all(state.offloaded and state.quantizations == 1 for state in states)


def test_host_master_dequantize_is_exact(offload_on_cpu):
    lin = _frozen_linear()
    W0 = lin.weight.detach().clone()
    state = FL.attach_fp8_lora_base(lin)
    from unsloth.kernels.utils import fast_dequantize

    assert torch.equal(fast_dequantize(lin.weight, state), W0)
    assert torch.equal(fast_dequantize(lin.weight.t(), state), W0.t())


def test_repeated_inference_does_not_rescan_restored_layers(offload_on_cpu, monkeypatch):
    model = _tiny_qwen3_peft()
    f8t.mark_fp8_lora(model.base_model.model, f8t.FP8_LORA_RECIPE)
    f8t.enable_fp8_lora(model, allow_cpu = True)
    f8t.restore_fp8_lora_for_inference(model)
    calls = []
    original = model.modules

    def modules():
        calls.append(True)
        return original()

    monkeypatch.setattr(model, "modules", modules)
    f8t.restore_fp8_lora_for_inference(model)
    assert calls == []


@needs_cpu_scaled_mm
def test_module_moves_keep_or_restore_the_host_master(offload_on_cpu):
    lin = _frozen_linear(bias = True)
    state = FL.attach_fp8_lora_base(lin)
    w8 = state.w8
    lin.to(state.device)
    lin.to(torch.bfloat16)
    assert state.offloaded and state.w8 is w8
    W0 = lin.weight.detach().clone()
    lin.half()
    assert not state.offloaded and lin.weight.dtype == torch.float16
    assert torch.equal(lin.weight, W0.half())
    x = torch.randn(32, 256).half()
    y = lin(x)
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


@needs_cpu_scaled_mm
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
    opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr = 0.1)
    for _ in range(3):
        _loss_and_grads_keep(model)
        opt.step()
        opt.zero_grad(set_to_none = True)
    sd = model.state_dict()
    for name, value in frozen.items():
        assert torch.equal(sd[name], value), name
    from safetensors.torch import load, save

    base = {n: sd[n].contiguous() for n in frozen if "base_layer" in n}
    assert len(base) == 14
    loaded = load(save(base))
    assert all(torch.equal(loaded[n], frozen[n]) for n in base)
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
    assert all(
        not m.__dict__["_unsloth_fp8_lora"].offloaded
        for m in model.modules()
        if "_unsloth_fp8_lora" in m.__dict__
    )


@pytest.mark.parametrize("method", ["merge_and_unload", "unload"])
def test_unloaded_model_matches_its_reloaded_dense_weights(offload_on_cpu, method):
    import copy

    model = _tiny_qwen3_peft()
    f8t.mark_fp8_lora(model.base_model.model, f8t.FP8_LORA_RECIPE)
    assert len(f8t.enable_fp8_lora(model, allow_cpu = True)) == 14
    clone = copy.deepcopy(model)
    merged = getattr(clone, method)().eval()
    reloaded = type(merged)(copy.deepcopy(merged.config)).to(torch.bfloat16).eval()
    reloaded.load_state_dict(merged.state_dict())
    ids = torch.randint(0, 256, (2, 32))
    FL.reset_fp8_lora_stats()
    with torch.no_grad():
        actual = merged(input_ids = ids).logits
        expected = reloaded(input_ids = ids).logits
    assert torch.equal(actual, expected), (actual - expected).abs().max().item()
    assert FL.FP8_LORA_STATS["fwd"] == 0
    assert not FL.fp8_lora_memory(merged)["layers"]
    assert not merged.__dict__.get("_unsloth_fp8_lora_loss_forwards")
    assert len(FL.fp8_lora_memory(model)["layers"]) == 14


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
    assert twin_layer.merged and not layer.merged
    assert torch.equal(layer.base_layer.weight, before)
    assert layer.base_layer.__dict__["_unsloth_fp8_lora"].offloaded
    plain_layer = dict(plain.named_modules())[name]
    plain_layer.merge()
    assert torch.equal(twin_layer.base_layer.weight, plain_layer.base_layer.weight)


@needs_cpu_scaled_mm
def test_compiled_fp8_lora_ops_are_counted():
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


@needs_cpu_scaled_mm
def test_compiled_fused_fp8_lora_ops_are_counted(monkeypatch):
    torch._dynamo.reset()
    monkeypatch.setenv("UNSLOTH_STRICT_ENGAGE", "1")
    W = _weight()
    state = FL.Fp8LoRABaseState(W)
    A = torch.randn(16, 256).bfloat16()
    Bt = torch.randn(16, 384).bfloat16()

    def run(x, dy):
        return state.lora_first(x, W, x @ A.t(), Bt, 1.0), state.dgrad_accum(
            dy, W, torch.zeros_like(x)
        )

    compiled = torch.compile(run, backend = "aot_eager", fullgraph = True)
    x, dy = torch.randn(32, 256).bfloat16(), torch.randn(32, 384).bfloat16()
    compiled(x, dy)
    with P.count_lowp_gemms() as counts:
        compiled(x, dy)
    P.require("fp8_lora", sum(counts.values()))
    assert counts["unsloth::fp8_lora_first"] == counts["unsloth::fp8_lora_accum"] == 1
    assert counts["_scaled_mm"] == 0


@needs_cpu_scaled_mm
@pytest.mark.parametrize("compiled_first", [False, True])
def test_enable_fp8_refreshes_only_an_existing_block_compiler(monkeypatch, compiled_first):
    from unsloth.models import _block_compile

    model = _tiny_qwen3_peft()
    model._unsloth_block_compile_installed = compiled_first
    f8t.mark_fp8_lora(model.base_model.model, f8t.FP8_LORA_RECIPE)
    calls = []

    def refresh(model):
        assert FL.fp8_lora_memory(model)["layers"]
        calls.append(model)

    monkeypatch.setattr(_block_compile, "patch_block_compile", refresh)
    assert f8t.enable_fp8_lora(model, allow_cpu = True)
    assert calls == ([model] if compiled_first else [])


@needs_cpu_scaled_mm
def test_merge_requantizes_the_fp8_copy(offload_on_cpu):
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
    from peft import LoraConfig

    model = _tiny_qwen3_peft()
    f8t.mark_fp8_lora(model.base_model.model, f8t.FP8_LORA_RECIPE)
    f8t.enable_fp8_lora(model, allow_cpu = True)
    model.add_adapter("second", LoraConfig(r = 4, target_modules = ["q_proj"]), low_cpu_mem_usage = True)
    added = [m for m in model.modules() if hasattr(m, "lora_A") and "second" in m.lora_A]
    assert added and all(m.lora_A["second"].weight.device.type == "meta" for m in added)


def test_a_dora_adapter_restores_the_master_before_placement(offload_on_cpu, monkeypatch):
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


@needs_cpu_scaled_mm
def test_load_state_dict_requantizes_for_compiled_calls():
    torch._dynamo.reset()
    lin = _frozen_linear()
    FL.attach_fp8_lora_base(lin, offload = False)
    compiled = torch.compile(lin, backend = "aot_eager", fullgraph = True)
    x = torch.randn(32, 256).bfloat16()
    assert compiled(x).abs().sum() > 0
    lin.load_state_dict({"weight": torch.zeros_like(lin.weight)})
    assert compiled(x).abs().sum() == 0


def test_a_dtype_move_requantizes_for_compiled_calls():
    lin = _frozen_linear()
    state = FL.attach_fp8_lora_base(lin, offload = False)
    lin.half()
    assert state.dtype == torch.float16 and state._key == state._weight_key(lin.weight)


def test_a_pissa_adapter_requantizes_the_rewritten_base(offload_on_cpu):
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


@needs_cpu_scaled_mm
def test_legacy_scaled_mm_tuple_return(monkeypatch):
    real = torch._scaled_mm
    monkeypatch.setattr(torch, "_scaled_mm", lambda *a, **k: (real(*a, **k), torch.zeros(())))
    W = _weight()
    state = FL.Fp8LoRABaseState(W)
    y = state.matmul(torch.randn(32, 256).bfloat16(), W)
    dx = state.dgrad(torch.randn(32, 384).bfloat16(), W)
    assert torch.is_tensor(y) and torch.is_tensor(dx)


@pytest.mark.parametrize("arch", ["falcon_h1", "wavlm"])
def test_parents_reading_a_child_weight_keep_its_master(offload_on_cpu, arch):
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


@needs_cpu_scaled_mm
def test_refresh_under_inference_mode_builds_trainable_fp8_tensors():
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
    linear = torch.nn.Linear(64, 64, bias = False).bfloat16().requires_grad_(False)
    linear.weight = torch.nn.Parameter(linear.weight.data[:32].clone(), requires_grad = False)
    assert "shard" in FL.eligible_base(linear)
    full = torch.nn.Linear(64, 64, bias = False).bfloat16().requires_grad_(False)
    full.weight.__dict__["_unsloth_fsdp2_preload"] = object()
    assert "shard" in FL.eligible_base(full)


def test_get_peft_model_converts_after_block_swap():
    import inspect
    from unsloth.models.llama import FastLlamaModel

    get_src = inspect.getsource(FastLlamaModel.get_peft_model)
    assert "_defer_fp8_lora = True" in get_src
    swap = get_src.index("install_block_swap(")
    assert get_src.index("enable_fp8_lora(model)", swap) > swap
    patch_src = inspect.getsource(FastLlamaModel.patch_peft_model)
    assert "if not _defer_fp8_lora:" in patch_src


def test_column_major_weights_are_not_eligible():
    linear = torch.nn.Linear(64, 128, bias = False).bfloat16().requires_grad_(False)
    linear.weight = torch.nn.Parameter(torch.randn(64, 128).bfloat16().t(), requires_grad = False)
    assert "contiguous" in FL.eligible_base(linear)


@pytest.mark.parametrize("variant", ["bdlora", "velora"])
def test_a_variant_adapter_restores_the_master_before_placement(
    offload_on_cpu, monkeypatch, variant
):
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


@needs_cpu_scaled_mm
@pytest.mark.parametrize("error_type", [RuntimeError, ValueError])
@pytest.mark.parametrize("outputs", ["bf16", "bf16 and fp16"])
def test_rowwise_fp16_without_float32_out_stays_finite(rowwise, monkeypatch, error_type, outputs):
    real = torch._scaled_mm

    def refuse_fp32_rowwise(
        a,
        b,
        *args,
        scale_a = None,
        out_dtype = None,
        **kwargs,
    ):
        if out_dtype == torch.float32 and scale_a is not None and scale_a.numel() > 1:
            raise error_type(
                f"Only {outputs} high precision output types are supported for row-wise scaling."
            )
        return real(a, b, *args, scale_a = scale_a, out_dtype = out_dtype, **kwargs)

    monkeypatch.setattr(torch, "_scaled_mm", refuse_fp32_rowwise)
    monkeypatch.setattr(FL, "_ROWWISE_BF16_ONLY", set())
    W = torch.full((32, 64), 0.5, dtype = torch.float16)
    state = FL.Fp8LoRABaseState(W)
    assert state.w_inv_fwd.dim() == 2
    x = torch.full((16, 64), 2047.0, dtype = torch.float16)
    assert torch.isfinite(state.matmul(x, W)).all()
    assert torch.isfinite(state.matmul(x, W)).all()
    dy = torch.full((16, 32), 4094.0, dtype = torch.float16)
    assert torch.isfinite(state.dgrad(dy, W)).all()
    Wb = W.bfloat16()
    y = FL.Fp8LoRABaseState(Wb).matmul(torch.randn(16, 64).bfloat16(), Wb)
    assert y.dtype == torch.bfloat16


@pytest.mark.parametrize(
    "error",
    [
        torch.OutOfMemoryError("allocation failed"),
        RuntimeError("CUDA error: illegal memory access"),
    ],
)
def test_rowwise_failure_does_not_change_the_scaling_recipe(monkeypatch, error):
    monkeypatch.setattr(FL, "_ROWWISE_BF16_ONLY", set())
    calls = []

    def fail_once(*args):
        calls.append(args)
        if len(calls) == 1:
            raise error
        return torch.zeros(16, 16)

    monkeypatch.setattr(FL, "_scaled_mm", fail_once)
    x = torch.zeros(16, 16)
    with pytest.raises(type(error), match = str(error)):
        FL._rowwise_scaled_mm(x, x.t(), torch.ones(16, 1), torch.ones(1, 16), torch.float16, True)
    assert len(calls) == 1
    assert FL._ROWWISE_BF16_ONLY == set()


def test_vllm_server_merge_of_a_host_master_lands_on_the_adapter_device():
    from types import SimpleNamespace

    from unsloth.models.rl_replacements import _unsloth_vllm_merged_weight

    lora_A = torch.nn.Linear(64, 4, bias = False, device = "meta")
    lora_B = torch.nn.Linear(4, 32, bias = False, device = "meta")
    module = SimpleNamespace(
        active_adapters = ["default"],
        lora_A = {"default": lora_A},
        lora_B = {"default": lora_B},
        scaling = {"default": 2.0},
    )
    weight = torch.nn.Parameter(torch.randn(32, 64).bfloat16(), requires_grad = False)
    merged = _unsloth_vllm_merged_weight(module, weight)
    assert merged.device.type == "meta" and merged.dtype == torch.bfloat16
    lora_A.to_empty(device = "cpu")
    lora_B.to_empty(device = "cpu")
    module.lora_A["default"], module.lora_B["default"] = lora_A, lora_B
    assert _unsloth_vllm_merged_weight(module, weight).device.type == "cpu"


@needs_cpu_scaled_mm
def test_a_merge_under_inference_mode_leaves_a_trainable_master(offload_on_cpu, monkeypatch):
    real_to = torch.Tensor.to

    def allocating_to(self, *args, **kwargs):
        if torch.is_inference_mode_enabled():
            kwargs["copy"] = True
        return real_to(self, *args, **kwargs)

    model = _tiny_qwen3_peft()
    f8t.mark_fp8_lora(model.base_model.model, f8t.FP8_LORA_RECIPE)
    f8t.enable_fp8_lora(model, allow_cpu = True)
    monkeypatch.setattr(torch.Tensor, "to", allocating_to)
    with torch.inference_mode():
        model.merge_adapter()
        model.unmerge_adapter()
    monkeypatch.setattr(torch.Tensor, "to", real_to)
    ids = torch.randint(0, 256, (2, 16))
    model(input_ids = ids, labels = ids).loss.backward()
    assert any(p.grad is not None for n, p in model.named_parameters() if "lora_" in n)


def test_text_diffusion_refuses_fp8_lora_instead_of_dropping_it():
    import inspect

    from unsloth.models.loader import FastModel

    src = inspect.getsource(FastModel.from_pretrained)
    body = src[src.index("def _dispatch_diffusion():") :]
    body = body[: body.index("FastDiffusionModel.from_pretrained(")]
    assert "if fp8_lora is not None:" in body and "NotImplementedError" in body


def test_vllm_server_sync_of_a_host_master_without_an_active_delta_lands_on_the_device():
    from types import SimpleNamespace

    from unsloth.models.rl_replacements import _unsloth_vllm_merged_weight

    base = torch.nn.Linear(64, 32, bias = False).bfloat16().requires_grad_(False)
    base.__dict__["_unsloth_fp8_lora"] = SimpleNamespace(device = torch.device("meta"))
    module = SimpleNamespace(
        base_layer = base, active_adapters = ["other"], lora_A = {}, lora_B = {}, scaling = {}
    )
    assert _unsloth_vllm_merged_weight(module, base.weight).device.type == "meta"
    del base.__dict__["_unsloth_fp8_lora"]
    assert _unsloth_vllm_merged_weight(module, base.weight).device.type == "cpu"


def test_fast_language_model_forwards_fp8_lora_to_fastmodel():
    import inspect

    from unsloth.models.loader import FastLanguageModel

    src = inspect.getsource(FastLanguageModel.from_pretrained)
    assert '**({"float8_training": fp8_request} if fp8_lora is not None else {})' in src


@needs_cpu_scaled_mm
def test_fp8_lora_accumulates_small_adapter_before_base_rounding(monkeypatch):
    monkeypatch.setattr(FL, "_SCALING", "tensor")
    monkeypatch.setattr(FL, "_MIN_ROWS", 16)
    x = torch.ones(16, 16, dtype = torch.bfloat16)
    w = torch.zeros(16, 16, dtype = torch.bfloat16)
    w[:, 0], w[:, 1] = 1, 1 / 512
    a = torch.zeros(1, 16, dtype = torch.bfloat16)
    a[0, 0] = 1 / 256
    b = torch.ones(16, 1, dtype = torch.bfloat16)
    state = FL.Fp8LoRABaseState(w)
    delta = torch.empty_like(x).addmm_(x @ a.t(), b.t(), beta = 0)
    reference = delta.addmm_(x, w.t())
    actual = FLK.matmul_lora(x, w, state, a, b, 1)
    old_order = FLK.matmul_lora(x, w, state, a, b, 1, fused = False)
    assert torch.equal(reference, torch.full_like(x, 1 + 1 / 128))
    assert torch.equal(actual, reference)
    assert not torch.equal(old_order, reference)


@pytest.mark.parametrize("projection", ["gate_proj", "up_proj", "down_proj"])
def test_biased_native_mlp_keeps_its_forward(projection):
    from peft import LoraConfig, get_peft_model
    from transformers import LlamaConfig, LlamaForCausalLM

    model = get_peft_model(
        LlamaForCausalLM(
            LlamaConfig(
                vocab_size = 64,
                hidden_size = 32,
                intermediate_size = 64,
                num_hidden_layers = 1,
                num_attention_heads = 4,
                num_key_value_heads = 2,
                mlp_bias = True,
            )
        ),
        LoraConfig(r = 4, target_modules = ["gate_proj", "up_proj", "down_proj"]),
    )
    mlp = model.get_base_model().model.layers[0].mlp
    for name in ("gate_proj", "up_proj", "down_proj"):
        base = getattr(mlp, name).base_layer
        if name != projection:
            base.bias = None
        else:
            base.bias.data.fill_(0.25)
        base.__dict__["_unsloth_fp8_lora"] = FL.Fp8LoRABaseState(base.weight)
    original = mlp.forward
    f8t._prepare_fp8_lora_operators(model, allow_cpu = True)
    assert mlp.forward == original
    x = torch.randn(2, 16, 32)
    torch.testing.assert_close(mlp(x), original(x), rtol = 0, atol = 0)


@pytest.mark.parametrize(
    "weight_dtype, input_dtype",
    [
        (torch.bfloat16, torch.float32),
        (torch.float16, torch.bfloat16),
        (torch.bfloat16, torch.float16),
    ],
)
def test_mixed_native_norm_uses_its_original_equation(monkeypatch, weight_dtype, input_dtype):
    from unsloth.kernels import rms_layernorm

    def unsupported(*args, **kwargs):
        pytest.fail("mixed-dtype normalization entered the fused kernel")

    monkeypatch.setattr(rms_layernorm, "fast_rms_layernorm", unsupported)
    model = _tiny_qwen3_peft()
    norm = model.get_base_model().model.norm.to(weight_dtype)
    original = norm.forward
    f8t._prepare_fp8_lora_operators(model, allow_cpu = True)
    x = torch.randn(2, 16, norm.weight.numel()).to(input_dtype).requires_grad_(True)
    expected = original(x)
    for call in (norm, torch.compile(norm, backend = "aot_eager", fullgraph = True)):
        actual = call(x)
        assert actual.dtype == expected.dtype
        torch.testing.assert_close(actual, expected, rtol = 0, atol = 0)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_fp32_native_norm_keeps_its_promoted_output(dtype):
    model = _tiny_qwen3_peft()
    norm = model.get_base_model().model.norm.float()
    norm.weight.data.fill_(1.001)
    original = norm.forward
    f8t._prepare_fp8_lora_operators(model, allow_cpu = True)
    assert norm.forward == original
    x = torch.randn(2, 16, norm.weight.numel()).to(dtype)
    normalized = x.float() * torch.rsqrt(
        x.float().square().mean(-1, keepdim = True) + norm.variance_epsilon
    )
    expected = norm.weight * normalized.to(dtype)
    actual = norm(x)
    assert actual.dtype == expected.dtype == torch.float32
    torch.testing.assert_close(actual, expected, rtol = 0, atol = 0)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_norm_upcast_after_binding_keeps_the_native_equation(monkeypatch, dtype):
    from unsloth.kernels import rms_layernorm

    def unsupported(*args, **kwargs):
        pytest.fail("FP32 normalization entered the fused kernel")

    monkeypatch.setattr(rms_layernorm, "fast_rms_layernorm", unsupported)
    model = _tiny_qwen3_peft()
    norm = model.get_base_model().model.norm
    original = norm.forward
    f8t._prepare_fp8_lora_operators(model, allow_cpu = True)
    norm.float()
    x = torch.randn(2, 16, norm.weight.numel()).to(dtype)
    actual = norm(x)
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, original(x), rtol = 0, atol = 0)


@pytest.mark.parametrize(
    "mode", ["full", "gelu", "partial", "custom", "custom_class", "private_norm", "generated"]
)
@pytest.mark.parametrize("layout", ["3d", "2d", "strided_3d", "strided_2d"])
def test_native_operator_binding_preserves_keywords_and_restores_instances(
    monkeypatch, mode, layout
):
    import types
    from peft import LoraConfig, get_peft_model
    from transformers import Qwen3Config, Qwen3ForCausalLM
    from transformers.models.qwen3.modeling_qwen3 import Qwen3MLP, Qwen3RMSNorm
    from unsloth.kernels import rms_layernorm

    model = get_peft_model(
        Qwen3ForCausalLM(
            Qwen3Config(
                vocab_size = 64,
                hidden_size = 32,
                intermediate_size = 64,
                num_hidden_layers = 2,
                num_attention_heads = 4,
                num_key_value_heads = 2,
            )
        ).to(torch.bfloat16),
        LoraConfig(r = 4, target_modules = ["gate_proj", "up_proj", "down_proj"]),
    )
    owner = model.get_base_model()
    mlp = owner.model.layers[0].mlp
    norm = owner.model.norm
    original_mlp, original_norm = Qwen3MLP.forward, Qwen3RMSNorm.forward
    for layer in owner.model.layers:
        for part in ("gate_proj", "up_proj", "down_proj"):
            base = getattr(layer.mlp, part).base_layer
            base.__dict__["_unsloth_fp8_lora"] = FL.Fp8LoRABaseState(base.weight)
    if mode == "gelu":
        mlp.act_fn = torch.nn.GELU()
    elif mode == "partial":
        mlp.down_proj.base_layer.__dict__.pop("_unsloth_fp8_lora")
    elif mode in ("custom", "custom_class"):

        def residual(self, x):
            return original_mlp(self, x) + x

        if mode == "custom":
            mlp.forward = types.MethodType(residual, mlp)
        else:
            residual.__module__ = Qwen3MLP.__module__
            monkeypatch.setattr(Qwen3MLP, "forward", residual)
    elif mode == "private_norm":
        namespace = {k: v for k, v in original_norm.__globals__.items() if k != "__name__"}
        private = types.FunctionType(original_norm.__code__, namespace)
        assert private.__module__ is None
        norm.forward = types.MethodType(private, norm)
    elif mode == "generated":
        from unsloth_zoo.temporary_patches.utils import torch_compile_with_fallback
        for cls, original, argument in (
            (Qwen3MLP, original_mlp, "x"),
            (Qwen3RMSNorm, original_norm, "hidden_states"),
        ):
            helper = cls.__name__ + "_forward"
            namespace = {
                "__name__": "unsloth_compiled_module_test",
                helper: torch_compile_with_fallback(backend = "aot_eager")(original),
            }
            exec(
                f"def forward(self, {argument}):\n    return {helper}(self, {argument}={argument})\n",
                namespace,
            )
            monkeypatch.setattr(cls, "__module__", namespace["__name__"])
            monkeypatch.setattr(cls, "forward", namespace["forward"])
    installed_class_forward = Qwen3MLP.forward
    installed_norm_forward = Qwen3RMSNorm.forward
    saved_override = mlp.__dict__.get("forward")
    saved_norm_override = norm.__dict__.get("forward")
    x = torch.randn(2, 3, 64)[..., ::2] if layout.startswith("strided") else torch.randn(2, 3, 32)
    if layout.endswith("2d"):
        x = x.reshape(6, 32)
    x = x.to(torch.bfloat16).requires_grad_(True)
    expected_mlp, expected_norm = mlp(x = x), norm(hidden_states = x)
    calls = {"norm": 0, "mlp": 0}

    def norm_spy(
        layernorm,
        X,
        gemma = False,
        save_output = False,
        reuse_grad = False,
    ):
        if not torch.compiler.is_compiling():
            calls["norm"] += 1
        return original_norm(layernorm, X)

    def mlp_spy(
        self,
        X,
        inplace = True,
    ):
        assert X.ndim == 3 and X.is_contiguous()
        if not torch.compiler.is_compiling():
            calls["mlp"] += 1
        return original_mlp(self, X)

    monkeypatch.setattr(rms_layernorm, "fast_rms_layernorm", norm_spy)
    monkeypatch.setattr(FLK, "apply_lora_mlp_swiglu", mlp_spy)
    monkeypatch.setenv("UNSLOTH_FP8_LORA_FUSED_OPS", "0")
    f8t._prepare_fp8_lora_operators(model, allow_cpu = True)
    assert "_unsloth_fp8_lora_kernel_forwards" not in owner.__dict__
    monkeypatch.setenv("UNSLOTH_FP8_LORA_FUSED_OPS", "1")
    f8t._prepare_fp8_lora_operators(model, allow_cpu = True)
    installed = owner.__dict__["_unsloth_fp8_lora_kernel_forwards"]
    assert installed
    f8t._prepare_fp8_lora_operators(model, allow_cpu = True)
    assert owner.__dict__["_unsloth_fp8_lora_kernel_forwards"] is installed
    assert torch.equal(mlp(x = x), expected_mlp)
    assert torch.equal(norm(hidden_states = x), expected_norm)
    assert calls == {"norm": 1, "mlp": int(mode in ("full", "private_norm", "generated"))}
    assert (
        Qwen3MLP.forward is installed_class_forward
        and Qwen3RMSNorm.forward is installed_norm_forward
    )
    if mode == "private_norm" and layout == "strided_2d":
        reference_grad = torch.autograd.grad(expected_mlp.sum() + expected_norm.sum(), x)[0]
        compiled_mlp = torch.compile(mlp, backend = "aot_eager", fullgraph = True)
        compiled_norm = torch.compile(norm, backend = "aot_eager", fullgraph = True)
        actual_mlp, actual_norm = compiled_mlp(x = x), compiled_norm(hidden_states = x)
        torch.testing.assert_close(actual_mlp, expected_mlp)
        torch.testing.assert_close(actual_norm, expected_norm)
        actual_grad = torch.autograd.grad(actual_mlp.sum() + actual_norm.sum(), x)[0]
        torch.testing.assert_close(actual_grad, reference_grad)
        assert calls == {"norm": 1, "mlp": 1}
        from torch._dynamo.utils import counters

        before = counters["stats"]["unique_graphs"]
        compiled_mlp(x = x)
        compiled_norm(hidden_states = x)
        stable = counters["stats"]["unique_graphs"] - before
        f8t.restore_fp8_lora_for_inference(model)
        f8t._prepare_fp8_lora_operators(model, allow_cpu = True)
        before = counters["stats"]["unique_graphs"]
        compiled_mlp(x = x)
        compiled_norm(hidden_states = x)
        rebound = counters["stats"]["unique_graphs"] - before
        print("OPERATOR_REBIND_GRAPHS", stable, rebound)
        assert stable == rebound == 0

        def call(module, value):
            return module(value)

        shared = torch.compile(call, backend = "aot_eager", fullgraph = True)
        shared(norm, x)
        shared(mlp, x)
        before = counters["stats"]["unique_graphs"]
        other_norm = owner.model.layers[0].input_layernorm
        other_mlp = owner.model.layers[1].mlp
        torch.testing.assert_close(shared(other_norm, x), original_norm(other_norm, x))
        torch.testing.assert_close(shared(other_mlp, x), original_mlp(other_mlp, x))
        shared_growth = counters["stats"]["unique_graphs"] - before
        print("OPERATOR_SHARED_GRAPHS", shared_growth)
        assert shared_growth == 0
    f8t.restore_fp8_lora_for_inference(model)
    assert mlp.__dict__.get("forward") is saved_override
    assert norm.__dict__.get("forward") is saved_norm_override
    assert torch.equal(mlp(x = x), expected_mlp)
    assert torch.equal(norm(hidden_states = x), expected_norm)
    assert calls == {"norm": 1, "mlp": int(mode in ("full", "private_norm", "generated"))}
