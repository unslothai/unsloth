# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""qat_scheme = "nvfp4" / "mxfp8" shorthands and saving torchao NVFP4 / MX QAT linears, CPU only."""

from __future__ import annotations

import importlib.util
import warnings

import pytest
import torch
from real_accelerator import has_real_cuda

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("torchao") is None, reason = "require torchao"
)

import unsloth.models._utils as qat_utils
import unsloth.save as save_mod
from unsloth.models._utils import TorchAOConfig, _prepare_model_for_qat


def _tiny_qwen3():
    from transformers import Qwen3Config, Qwen3ForCausalLM

    torch.manual_seed(0)
    config = Qwen3Config(
        vocab_size = 256,
        hidden_size = 128,
        intermediate_size = 256,
        num_hidden_layers = 1,
        num_attention_heads = 4,
        num_key_value_heads = 1,
        head_dim = 32,
        tie_word_embeddings = False,
    )
    return Qwen3ForCausalLM(config).to(torch.bfloat16)


def _mx_linear():
    from torchao.prototype.qat import MXFakeQuantizedLinear
    return MXFakeQuantizedLinear


def _nvfp4_linear():
    from torchao.prototype.qat import NVFP4FakeQuantizedLinear
    return NVFP4FakeQuantizedLinear


def test_nvfp4_scheme_refuses_without_fp4_gemm():
    model = _tiny_qwen3()
    with pytest.raises(RuntimeError, match = "FP4 tensor cores"):
        _prepare_model_for_qat(model, "nvfp4")
    assert not any(isinstance(m, _nvfp4_linear()) for m in model.modules())


def test_nvfp4_scheme_layers_where_fp4_gemm_runs(monkeypatch):
    from torchao.prototype.mx_formats import NVFP4DynamicActivationNVFP4WeightConfig

    monkeypatch.setattr(qat_utils, "_qat_forward_runs", lambda *a: True)
    model = _tiny_qwen3()
    model.model.layers[0].odd = torch.nn.Linear(128, 40, bias = False)
    with warnings.catch_warnings(record = True):
        warnings.simplefilter("always")
        model = _prepare_model_for_qat(model, "nvfp4")
    ((base_config, _),) = model._torchao_config.base_config_and_filter_fns
    assert isinstance(base_config, NVFP4DynamicActivationNVFP4WeightConfig)
    assert (
        base_config.use_triton_kernel is False
    )  # the Triton quantizer needs the optional MSLK package
    layer = model.model.layers[0]
    for proj in (layer.self_attn.q_proj, layer.self_attn.k_proj, layer.mlp.down_proj):
        assert isinstance(proj, _nvfp4_linear())
    assert type(model.lm_head) is torch.nn.Linear  # 4-bit head costs too much accuracy
    assert type(layer.odd) is torch.nn.Linear


@pytest.mark.parametrize("lora", [False, True])
def test_nvfp4_needs_in_features_divisible_by_32(monkeypatch, lora):
    # _scaled_mm takes the packed FP4 K / 2 and needs it divisible by 16: K = 16, 48, 80 cannot run.
    monkeypatch.setattr(qat_utils, "_qat_forward_runs", lambda *a: True)
    if lora:
        from peft import LoraConfig, get_peft_model

        monkeypatch.setenv("UNSLOTH_QAT_FAKE_QUANTIZE_LORA", "1")
        model = get_peft_model(
            _tiny_qwen3(), LoraConfig(r = 16, lora_alpha = 16, target_modules = ["q_proj"])
        )
        model = _prepare_model_for_qat(model, "nvfp4")
        q_proj = model.base_model.model.model.layers[0].self_attn.q_proj
        assert isinstance(q_proj.base_layer, _nvfp4_linear())
        assert isinstance(q_proj.lora_A["default"], _nvfp4_linear())  # K = 128
        assert type(q_proj.lora_B["default"]) is torch.nn.Linear  # K = r = 16
        return
    widths = (16, 32, 48, 64, 80, 96)
    model = torch.nn.Sequential(*(torch.nn.Linear(k, 128, bias = False) for k in widths)).to(
        torch.bfloat16
    )
    with warnings.catch_warnings(record = True) as caught:
        warnings.simplefilter("always")
        model = _prepare_model_for_qat(model, "nvfp4")
    got = {k: isinstance(m, _nvfp4_linear()) for k, m in zip(widths, model)}
    assert got == {k: k % 32 == 0 for k in widths}
    assert any("in_features not divisible by 32" in str(w.message) for w in caught)


@pytest.mark.skipif(
    not has_real_cuda()
    or not torch.cuda.is_available()
    or torch.cuda.get_device_capability() < (10, 0),
    reason = "needs an FP4 tensor core GPU",
)
def test_nvfp4_admitted_widths_run_on_gpu():
    widths = (16, 32, 48, 64, 80, 96)
    model = torch.nn.ModuleList(torch.nn.Linear(k, 128, bias = False) for k in widths).to(
        "cuda", torch.bfloat16
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = _prepare_model_for_qat(model, "nvfp4")
    for k, layer in zip(widths, model):
        assert isinstance(layer, _nvfp4_linear()) == (k % 32 == 0), k
        x = torch.randn(16, k, device = "cuda", dtype = torch.bfloat16, requires_grad = True)
        layer(x).float().sum().backward()
        assert x.grad is not None


def test_nvfp4_plain_module_without_hf_heads(monkeypatch):
    monkeypatch.setattr(qat_utils, "_qat_forward_runs", lambda *a: True)
    model = torch.nn.Sequential(torch.nn.Linear(64, 128, bias = False)).to(torch.bfloat16)
    model = _prepare_model_for_qat(model, "nvfp4")
    assert isinstance(model[0], _nvfp4_linear())


def test_unknown_scheme_still_rejected():
    with pytest.raises(ValueError, match = "Unexpected QAT scheme"):
        _prepare_model_for_qat(_tiny_qwen3(), "mxfp6")


def test_mxfp8_scheme_refuses_without_mx_gemm():
    model = _tiny_qwen3()
    with pytest.raises(RuntimeError, match = "MX tensor cores"):
        _prepare_model_for_qat(model, "mxfp8")
    assert not any(isinstance(m, _mx_linear()) for m in model.modules())


def test_mxfp8_scheme_layers_where_mx_gemm_runs(monkeypatch):
    from torchao.prototype.mx_formats import MXDynamicActivationMXWeightConfig

    monkeypatch.setattr(qat_utils, "_qat_forward_runs", lambda *a: True)
    model = _tiny_qwen3()
    model.model.layers[0].odd = torch.nn.Linear(128, 48, bias = False)
    with warnings.catch_warnings(record = True) as caught:
        warnings.simplefilter("always")
        model = _prepare_model_for_qat(model, "mxfp8")
    ((base_config, _),) = model._torchao_config.base_config_and_filter_fns
    assert isinstance(base_config, MXDynamicActivationMXWeightConfig)
    assert base_config.activation_dtype == base_config.weight_dtype == torch.float8_e4m3fn
    layer = model.model.layers[0]
    for proj in (layer.self_attn.q_proj, layer.self_attn.k_proj, layer.mlp.down_proj):
        assert isinstance(proj, _mx_linear())
    assert type(model.lm_head) is torch.nn.Linear
    assert type(layer.odd) is torch.nn.Linear  # 48 is not a multiple of the 32-element block
    assert any("32-element scale blocks" in str(w.message) for w in caught)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_mxfp8_refuses_non_bfloat16_layers(monkeypatch, dtype):
    _probe_ok(monkeypatch)
    model = _tiny_qwen3().to(dtype)
    with pytest.raises(RuntimeError, match = "bfloat16 layers only"):
        _prepare_model_for_qat(model, "mxfp8")
    assert not any(isinstance(m, _mx_linear()) for m in model.modules())


@pytest.mark.parametrize("weight_dtype", [torch.bfloat16, torch.float32])
def test_mx_qat_linear_takes_float32_activations_under_autocast(weight_dtype):
    # Autocast feeds float32 activations into bfloat16 (or float32) linears.
    # torchao's CPU fake-quant backward cannot mix a float32 weight with bfloat16 activations.
    device = "cpu" if weight_dtype == torch.bfloat16 else "cuda"
    if device == "cuda" and not has_real_cuda():
        pytest.skip("float32 weights under autocast need the CUDA MX path")
    model = torch.nn.Sequential(torch.nn.Linear(64, 128, bias = False)).to(device, weight_dtype)
    model = _prepare_model_for_qat(model, _mx_torchao_config())
    assert isinstance(model[0], _mx_linear())
    x = torch.randn(2, 3, 64, device = device, requires_grad = True)
    with torch.autocast(device, dtype = torch.bfloat16):
        out = model(x)
    assert out.dtype == torch.bfloat16 and out.shape == (2, 3, 128)
    out.float().sum().backward()
    assert x.grad is not None and x.grad.dtype == torch.float32
    assert model[0].weight.grad is not None


def _mx_torchao_config():
    from torchao.prototype.mx_formats import MXDynamicActivationMXWeightConfig
    from torchao.quantization.quantize_.common.kernel_preference import KernelPreference
    return TorchAOConfig(
        qat_scheme = "custom-mx",
        base_config_and_filter_fns = [
            (
                MXDynamicActivationMXWeightConfig(kernel_preference = KernelPreference.EMULATED),
                lambda m, fqn: isinstance(m, torch.nn.Linear) and "lm_head" not in fqn,
            )
        ],
    )


def test_strip_restores_plain_linears():
    model = _prepare_model_for_qat(_tiny_qwen3(), _mx_torchao_config())
    q_proj = model.model.layers[0].self_attn.q_proj
    weight = q_proj.weight
    assert isinstance(q_proj, _mx_linear())
    qat_utils._strip_torchao_fake_quantization(model)
    assert not any(isinstance(m, _mx_linear()) for m in model.modules())
    q_proj = model.model.layers[0].self_attn.q_proj
    assert type(q_proj) is torch.nn.Linear
    assert torch.equal(q_proj.weight, weight)


def test_peft_save_merges_plain_linears(monkeypatch, tmp_path):
    from peft import LoraConfig, get_peft_model

    seen = {}
    monkeypatch.setattr(
        save_mod, "_unsloth_save_torchao_with_given_config", lambda **kw: seen.update(kw)
    )
    model = get_peft_model(_tiny_qwen3(), LoraConfig(r = 16, target_modules = ["q_proj", "down_proj"]))
    model = _prepare_model_for_qat(model, _mx_torchao_config())
    assert any(isinstance(m, _mx_linear()) for m in model.modules())
    save_mod._unsloth_save_torchao_with_attached_config(
        model = model, save_directory = str(tmp_path), tokenizer = object()
    )
    assert not any(isinstance(m, _mx_linear()) for m in model.modules())
    assert seen["torchao_config"] is not None


def _probe_ok(
    monkeypatch,
    bias_grad = True,
    calls = None,
    bad_widths = (),
):
    def forward_runs(
        base_config,
        device,
        dtype = torch.bfloat16,
        out_features = 128,
    ):
        if calls is not None and out_features == 128:
            calls.append((str(device), dtype))
        return out_features not in bad_widths

    monkeypatch.setattr(qat_utils, "_qat_forward_runs", forward_runs)
    monkeypatch.setattr(qat_utils, "_qat_bias_grad_supported", lambda *a: bias_grad, raising = False)


def test_nvfp4_refuses_float16_layers(monkeypatch):
    # torchao's non-Triton NVFP4 quantizer asserts bfloat16 / float32 at the first forward.
    _probe_ok(monkeypatch)
    model = _tiny_qwen3().to(torch.float16)
    with pytest.raises(RuntimeError, match = "bfloat16 or float32"):
        _prepare_model_for_qat(model, "nvfp4")
    assert not any(isinstance(m, _nvfp4_linear()) for m in model.modules())


def test_nvfp4_skips_peft_modules_to_save_head(monkeypatch):
    from peft import LoraConfig, get_peft_model

    _probe_ok(monkeypatch)
    model = get_peft_model(
        _tiny_qwen3(),
        LoraConfig(r = 16, target_modules = ["q_proj"], modules_to_save = ["lm_head"]),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = _prepare_model_for_qat(model, "nvfp4")
    head = model.base_model.model.lm_head
    assert not any(isinstance(m, _nvfp4_linear()) for m in head.modules())
    assert isinstance(model.base_model.model.model.layers[0].mlp.down_proj, _nvfp4_linear())


def test_nvfp4_probes_every_layer_placement(monkeypatch):
    calls = []
    _probe_ok(monkeypatch, calls = calls)
    model = _tiny_qwen3()
    model.model.layers[0].mlp.down_proj.to(torch.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _prepare_model_for_qat(model, "nvfp4")
    assert sorted(calls, key = str) == sorted(
        [("cpu", torch.bfloat16), ("cpu", torch.float32)], key = str
    )


def test_nvfp4_keeps_failing_narrow_widths_in_full_precision(monkeypatch):
    # tiny Qwen3 k_proj / v_proj have out_features 32 (one KV head of 32).
    _probe_ok(monkeypatch, bad_widths = (32,))
    model = _tiny_qwen3()
    attn = model.model.layers[0].self_attn
    with warnings.catch_warnings(record = True) as caught:
        warnings.simplefilter("always")
        model = _prepare_model_for_qat(model, "nvfp4")
    assert not isinstance(attn.k_proj, _nvfp4_linear())
    assert not isinstance(attn.v_proj, _nvfp4_linear())
    assert isinstance(attn.q_proj, _nvfp4_linear())
    assert any("out_features [32]" in str(w.message) for w in caught)


def test_nvfp4_converts_narrow_widths_that_run(monkeypatch):
    _probe_ok(monkeypatch)
    model = _tiny_qwen3()
    model = _prepare_model_for_qat(model, "nvfp4")
    assert isinstance(model.model.layers[0].self_attn.k_proj, _nvfp4_linear())


def test_nvfp4_probes_opted_in_adapter_placements(monkeypatch):
    calls = []
    _probe_ok(monkeypatch, calls = calls)
    monkeypatch.setenv("UNSLOTH_QAT_FAKE_QUANTIZE_LORA", "1")
    model = _tiny_qwen3()
    layer = model.model.layers[0].mlp
    layer.lora_A = torch.nn.Linear(256, 16, bias = False).to(torch.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _prepare_model_for_qat(model, "nvfp4")
    assert ("cpu", torch.float32) in calls


@pytest.mark.parametrize("scheme", ["nvfp4", "mxfp8"])
@pytest.mark.parametrize("bias_grad", [False, True])
def test_nvfp4_biased_layers_need_bias_grad(monkeypatch, bias_grad, scheme):
    _probe_ok(monkeypatch, bias_grad = bias_grad)
    linear_cls = _nvfp4_linear() if scheme == "nvfp4" else _mx_linear()
    model = _tiny_qwen3()
    layer = model.model.layers[0]
    layer.self_attn.q_proj = torch.nn.Linear(128, 128, bias = True).to(torch.bfloat16)
    with warnings.catch_warnings(record = True) as caught:
        warnings.simplefilter("always")
        model = _prepare_model_for_qat(model, scheme)
    assert isinstance(layer.self_attn.q_proj, linear_cls) is bias_grad
    assert isinstance(layer.mlp.down_proj, linear_cls)
    assert any("no bias" in str(w.message) for w in caught) is not bias_grad


def test_nvfp4_accepts_rank4_inputs(monkeypatch):
    linear_cls = _nvfp4_linear()

    def fake_2d_forward(self, x):
        assert x.dim() == 2, x.shape  # torchao's addmm dispatch takes 2D operands only
        return x @ self.weight.t()

    _probe_ok(monkeypatch)
    model = torch.nn.Sequential(torch.nn.Linear(64, 128, bias = False)).to(torch.bfloat16)
    model = _prepare_model_for_qat(model, "nvfp4")
    monkeypatch.setattr(linear_cls, "forward", fake_2d_forward)
    for shape in [(6, 64), (2, 3, 64), (2, 3, 5, 64)]:
        out = model(torch.randn(*shape, dtype = torch.bfloat16))
        assert out.shape == (*shape[:-1], 128)


def test_nvfp4_qat_forward_compiles_fullgraph(monkeypatch):
    # fullgraph = True: the patched forward must not be a compiler boundary.
    linear_cls = _nvfp4_linear()
    monkeypatch.setattr(linear_cls, "forward", lambda self, x: x @ self.weight.t())
    _probe_ok(monkeypatch)

    class Block(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = torch.nn.Linear(64, 128, bias = False)

        def forward(self, x):
            return self.proj(x * 2) + 1

    model = _prepare_model_for_qat(Block().to(torch.bfloat16), "nvfp4")
    assert isinstance(model.proj, linear_cls)
    torch._dynamo.reset()
    explain = torch._dynamo.explain(model)(torch.randn(4, 64, dtype = torch.bfloat16))
    assert explain.graph_break_count == 0 and explain.graph_count == 1
    x = torch.randn(2, 3, 64, dtype = torch.bfloat16)
    torch.testing.assert_close(torch.compile(model, fullgraph = True, backend = "eager")(x), model(x))


_needs_fp4_gpu = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() < (10, 0),
    reason = "needs an FP4 tensor core GPU",
)


@pytest.fixture
def exact_inductor(monkeypatch):
    # Inductor otherwise keeps silu(gate) * up in fp32 before the bf16 round, shifting fp4 codes (~2% grads).
    import torch._inductor.config as inductor_config

    monkeypatch.setattr(inductor_config, "emulate_precision_casts", True)
    torch._dynamo.reset()
    yield
    torch._dynamo.reset()


def _nvfp4_mlp(weight_dtype = torch.bfloat16):
    import torch.nn.functional as F

    class MLP(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_proj = torch.nn.Linear(256, 512, bias = False)
            self.up_proj = torch.nn.Linear(256, 512, bias = False)
            self.down_proj = torch.nn.Linear(512, 256, bias = False)

        def forward(self, x):
            return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

    torch.manual_seed(0)
    model = MLP().to("cuda", weight_dtype)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = _prepare_model_for_qat(model, "nvfp4")
    assert isinstance(model.gate_proj, _nvfp4_linear())
    return model


def _torchao_reference(model):
    """torchao's own eager NVFP4 QAT forward (its autograd Function), not the custom op."""
    import types
    from torchao.prototype.qat import NVFP4FakeQuantizedLinear

    ref = __import__("copy").deepcopy(model)
    for module in ref.modules():
        if isinstance(module, NVFP4FakeQuantizedLinear):
            module.forward = types.MethodType(NVFP4FakeQuantizedLinear.forward, module)
    return ref


def _grads(model, forward, x):
    model.zero_grad(set_to_none = True)
    x = x.detach().requires_grad_(True)
    out = forward(x)
    out.float().square().mean().backward()
    grads = {n: p.grad.float().clone() for n, p in model.named_parameters()}
    grads["x"] = x.grad.float().clone()
    return out.detach().float(), grads


def _assert_same(
    got,
    want,
    tol = 0.0,
):
    (out, grads), (ref_out, ref_grads) = got, want
    assert float((out - ref_out).norm() / ref_out.norm()) <= tol
    for name, ref in ref_grads.items():
        rel = float((grads[name] - ref).norm() / ref.norm())
        assert rel <= tol, (name, rel)


@_needs_fp4_gpu
def test_nvfp4_qat_custom_op_matches_torchao_eager():
    model = _nvfp4_mlp()
    x = torch.randn(2, 64, 256, device = "cuda", dtype = torch.bfloat16)
    ref = _torchao_reference(model)
    _assert_same(_grads(model, model, x), _grads(ref, ref, x))
    op = qat_utils._nvfp4_qat_linear_op()
    w = model.gate_proj.weight
    torch.library.opcheck(
        op,
        (x.view(-1, 256).detach().requires_grad_(True), w, None, True, True, True, False, None),
        # test_aot_dispatch_dynamic sums every output, which float8 scales do not support.
        test_utils = ("test_schema", "test_autograd_registration", "test_faketensor"),
    )


@_needs_fp4_gpu
@pytest.mark.parametrize("fullgraph", [False, True])
def test_nvfp4_qat_weight_grads_match_eager_under_compile(exact_inductor, fullgraph):
    # Inductor used to shrink the gate / up weight grads (~100% relative error against eager).
    model = _nvfp4_mlp()
    x = torch.randn(2, 64, 256, device = "cuda", dtype = torch.bfloat16)
    eager = _grads(model, model, x)
    compiled = torch.compile(model, fullgraph = fullgraph)
    _assert_same(_grads(model, compiled, x), eager)
    _assert_same(_grads(model, compiled, x), eager)  # warm call


@_needs_fp4_gpu
def test_nvfp4_qat_weight_grads_close_to_eager_with_default_inductor():
    model = _nvfp4_mlp()
    x = torch.randn(2, 64, 256, device = "cuda", dtype = torch.bfloat16)
    eager = _grads(model, model, x)
    torch._dynamo.reset()
    _assert_same(_grads(model, torch.compile(model, fullgraph = True), x), eager, 0.05)


@_needs_fp4_gpu
def test_nvfp4_qat_dynamic_batch_fullgraph_one_graph(exact_inductor):
    from torch._dynamo.testing import CompileCounterWithBackend

    model = _nvfp4_mlp()
    counter = CompileCounterWithBackend("inductor")
    compiled = torch.compile(model, fullgraph = True, dynamic = True, backend = counter)
    for batch in (2, 3, 4, 7):
        x = torch.randn(batch, 32, 256, device = "cuda", dtype = torch.bfloat16)
        torch._dynamo.mark_dynamic(x, 0)
        _assert_same(_grads(model, compiled, x), _grads(model, model, x))
    assert counter.frame_count == 1, counter.frame_count


@_needs_fp4_gpu
def test_nvfp4_qat_one_token_decode_fullgraph(exact_inductor):
    model = _nvfp4_mlp()
    model.requires_grad_(False)
    x = torch.randn(1, 1, 256, device = "cuda", dtype = torch.bfloat16)
    with torch.no_grad():
        want = model(x)
        got = torch.compile(model, fullgraph = True)(x)
    torch.testing.assert_close(got, want, rtol = 0, atol = 0)


@_needs_fp4_gpu
@pytest.mark.parametrize("weight_dtype", [torch.bfloat16, torch.float32])
def test_nvfp4_qat_autocast_grads_match_torchao(exact_inductor, weight_dtype):
    model = _nvfp4_mlp(weight_dtype)
    ref = _torchao_reference(model)
    x = torch.randn(2, 16, 256, device = "cuda", dtype = torch.bfloat16)

    def under_autocast(module):
        def run(t):
            with torch.autocast("cuda", dtype = torch.bfloat16):
                return module(t)

        return run

    want = _grads(ref, under_autocast(ref), x)
    _assert_same(_grads(model, under_autocast(model), x), want)
    compiled = torch.compile(under_autocast(model), fullgraph = True)
    _assert_same(_grads(model, compiled, x), want)


def _torchao_nvfp4_contract_forward(self, x):
    # What torchao's NVFP4 QAT forward accepts: nvfp4_quantize asserts these, then the output follows x.
    assert x.dim() == 2, x.shape
    assert x.dtype in (torch.bfloat16, torch.float32), f"{x.dtype} not supported"
    assert x.is_contiguous(), "Only support contiguous data for now"
    return (x.float() @ self.weight.float().t()).to(x.dtype)


def test_nvfp4_accepts_strided_inputs(monkeypatch):
    _probe_ok(monkeypatch)
    model = torch.nn.Sequential(torch.nn.Linear(64, 128, bias = False)).to(torch.bfloat16)
    ref = model[0].weight.detach().clone()
    model = _prepare_model_for_qat(model, "nvfp4")
    monkeypatch.setattr(_nvfp4_linear(), "forward", _torchao_nvfp4_contract_forward)
    base = torch.randn(4, 6, 128, dtype = torch.bfloat16)
    for x in [base[0, :, ::2], base[:, :, :64], base.transpose(0, 1)[..., 64:].unsqueeze(0)]:
        assert not x.is_contiguous()
        out = model(x)
        assert out.shape == (*x.shape[:-1], 128) and out.dtype == torch.bfloat16
        expected = (x.float() @ ref.float().t()).to(torch.bfloat16)
        torch.testing.assert_close(out, expected)


@pytest.mark.parametrize("weight_dtype", [torch.bfloat16, torch.float32])
def test_nvfp4_takes_float16_autocast_activations(monkeypatch, weight_dtype):
    # Linear(63, 64) stays plain (63 is not a multiple of 16) and emits float16 under autocast.
    _probe_ok(monkeypatch)
    torch.manual_seed(0)
    model = torch.nn.Sequential(
        torch.nn.Linear(63, 64, bias = False), torch.nn.Linear(64, 128, bias = False)
    ).to(weight_dtype)
    model = _prepare_model_for_qat(model, "nvfp4")
    assert not isinstance(model[0], _nvfp4_linear()) and isinstance(model[1], _nvfp4_linear())
    monkeypatch.setattr(_nvfp4_linear(), "forward", _torchao_nvfp4_contract_forward)
    x = torch.randn(2, 3, 63, dtype = weight_dtype, requires_grad = True)
    with torch.autocast("cpu", dtype = torch.float16):
        out = model(x)
    assert out.dtype == torch.float16 and out.shape == (2, 3, 128)
    out.float().sum().backward()
    assert x.grad is not None and model[1].weight.grad is not None


def test_nvfp4_float16_activations_under_bfloat16_autocast_return_bfloat16(monkeypatch):
    _probe_ok(monkeypatch)
    model = torch.nn.Sequential(torch.nn.Linear(64, 128, bias = False)).to(torch.bfloat16)
    model = _prepare_model_for_qat(model, "nvfp4")
    monkeypatch.setattr(_nvfp4_linear(), "forward", _torchao_nvfp4_contract_forward)
    x = torch.randn(2, 3, 64, dtype = torch.float16)
    with torch.autocast("cpu", dtype = torch.bfloat16):
        out = model(x)
    assert out.dtype == torch.bfloat16 and out.shape == (2, 3, 128)
