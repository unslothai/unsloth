# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Torchao-free weight-only int8 / fp8; ROCm simulated by patching ``torch_is_rocm`` / ``is_stubbed``."""

from __future__ import annotations

import sys
import types

import pytest

torch = pytest.importorskip("torch")

import core.inference.diffusion_native_quant as nq
import core.inference.diffusion_transformer_quant as tq


def _target(device = "cuda", dtype = None):
    return types.SimpleNamespace(device = device, dtype = torch.bfloat16 if dtype is None else dtype)


@pytest.fixture
def rocm(monkeypatch):
    monkeypatch.setattr(tq, "torch_is_rocm", lambda: True)
    monkeypatch.setattr(tq, "is_stubbed", lambda name: False)


@pytest.fixture
def win_stub(monkeypatch):
    monkeypatch.setattr(tq, "torch_is_rocm", lambda: False)
    monkeypatch.setattr(tq, "is_stubbed", lambda name: name == "torchao")


@pytest.fixture
def nvidia(monkeypatch):
    monkeypatch.setattr(tq, "torch_is_rocm", lambda: False)
    monkeypatch.setattr(tq, "is_stubbed", lambda name: False)


class _Block(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.to_q = torch.nn.Linear(dim, dim)
        self.ff = torch.nn.Linear(dim, 2 * dim, bias = False)
        self.out = torch.nn.Linear(2 * dim, dim)
        self.small = torch.nn.Linear(dim, 8)

    def forward(self, x):
        return self.out(torch.nn.functional.gelu(self.ff(self.to_q(x)))) + self.small(x).sum(
            -1, keepdim = True
        )


class _Toy(torch.nn.Module):
    def __init__(
        self,
        dim = 128,
        depth = 2,
    ):
        super().__init__()
        self.blocks = torch.nn.ModuleList(_Block(dim) for _ in range(depth))

    def forward(self, x):
        for b in self.blocks:
            x = x + b(x)
        return x


def _toy(dtype = torch.bfloat16):
    torch.manual_seed(0)
    return _Toy().to(dtype)


@pytest.mark.parametrize("scheme", ["int8", "fp8", "INT8", " fp8 "])
def test_rocm_and_stub_hosts_take_explicit_int8_and_fp8(rocm, scheme):
    assert tq.native_quant_host(_target())
    assert tq.native_quant_scheme(_target(), scheme) == scheme.strip().lower()


def test_windows_stub_host_is_native(win_stub):
    assert tq.native_quant_host(_target())
    assert tq.native_quant_scheme(_target(), "int8") == "int8"


@pytest.mark.parametrize("scheme", [None, "auto", "none", "off", "nvfp4", "mxfp8"])
def test_auto_and_other_schemes_never_go_native(rocm, scheme):
    assert tq.native_quant_scheme(_target(), scheme) is None


@pytest.mark.parametrize(
    "target",
    [
        _target(device = "cpu"),
        _target(device = "mps"),
        _target(device = "xpu"),
        _target(dtype = torch.float16),
        _target(dtype = torch.float32),
    ],
)
def test_non_cuda_or_non_bf16_is_never_native_even_on_rocm(rocm, target):
    assert not tq.native_quant_host(target)
    assert tq.native_quant_scheme(target, "int8") is None


@pytest.mark.parametrize("device", ["cuda", "cpu", "mps", "xpu"])
def test_nvidia_and_other_hosts_are_never_native(nvidia, device):
    assert not tq.native_quant_host(_target(device = device))
    for scheme in ("int8", "fp8", "nvfp4", "mxfp8", "auto"):
        assert tq.native_quant_scheme(_target(device = device), scheme) is None


def test_family_deny_list_still_applies(rocm, monkeypatch):
    monkeypatch.setitem(tq._FAMILY_SCHEME_DENY, "toy-family", frozenset({tq.TQ_INT8}))
    assert tq.native_quant_scheme(_target(), "int8", family = "toy-family") is None
    assert tq.native_quant_scheme(_target(), "fp8", family = "toy-family") == "fp8"
    # The shipped deny list only fences nvfp4 / mxfp8 off qwen-image, so int8 / fp8 run there.
    assert tq.native_quant_scheme(_target(), "int8", family = "qwen-image") == "int8"
    assert tq.native_quant_scheme(_target(), "fp8", family = "qwen-image-2.1") == "fp8"


def test_the_shared_selector_is_untouched_on_rocm(rocm):
    # Hosted prequant planners read this selector; AMD must never be handed a torchao checkpoint.
    for scheme in ("int8", "fp8", "auto"):
        assert tq.select_transformer_quant_scheme(_target(), scheme, unproven_ok = True) is None
    assert not tq.dense_transformer_supported(_target())


@pytest.mark.parametrize("scheme,bound", [("int8", 0.01), ("fp8", 0.04)])
def test_native_layer_reconstructs_the_weight(scheme, bound):
    torch.manual_seed(0)
    lin = torch.nn.Linear(256, 512).to(torch.bfloat16)
    err = nq.native_weight_error(lin, scheme)
    assert err is not None and err < bound
    layer = nq.native_linear_class()(lin, scheme)
    assert layer.weight_q.dtype == (torch.int8 if scheme == "int8" else torch.uint8)
    assert layer.weight_scale.dtype == torch.int32 and layer.weight_scale.shape == (512,)
    assert layer.weight_q.element_size() == 1
    assert layer.bias is lin.bias
    x = torch.randn(4, 256, dtype = torch.bfloat16)
    ref = lin(x).float()
    got = layer(x).float()
    assert layer(x).dtype == torch.bfloat16
    assert ((got - ref).norm() / ref.norm()).item() < bound * 2


def test_native_layer_rejects_an_unknown_scheme():
    with pytest.raises(ValueError):
        nq.native_linear_class()(torch.nn.Linear(8, 8), "nvfp4")


def test_zero_rows_do_not_divide_by_zero():
    lin = torch.nn.Linear(16, 4, bias = False).to(torch.bfloat16)
    with torch.no_grad():
        lin.weight.zero_()
    for scheme in ("int8", "fp8"):
        layer = nq.native_linear_class()(lin, scheme)
        assert torch.isfinite(layer.dequantized_weight(torch.float32)).all()
        assert layer(torch.ones(2, 16, dtype = torch.bfloat16)).abs().max().item() == 0


def test_native_buffers_move_with_module_to_and_state_dict():
    # Offload hooks move modules with Module.to(); torchao tensors reject that, plain buffers do not.
    lin = torch.nn.Linear(64, 64).to(torch.bfloat16)
    layer = nq.native_linear_class()(lin, "int8")
    keys = set(layer.state_dict())
    assert {"weight_q", "weight_scale", "bias"} <= keys and "weight" not in keys
    layer.to("meta")
    assert layer.weight_q.device.type == "meta" and layer.weight_q.dtype == torch.int8
    assert layer.weight_scale.dtype == torch.int32


@pytest.mark.parametrize("scheme", ["int8", "fp8"])
def test_a_module_wide_dtype_cast_leaves_the_stored_weights_alone(scheme):
    """A module ``.to(dtype)`` must not cast the integer-view payload or scales."""
    torch.manual_seed(0)
    lin = torch.nn.Linear(128, 64).to(torch.bfloat16)
    layer = nq.native_linear_class()(lin, scheme)
    x = torch.randn(3, 128, dtype = torch.bfloat16)
    before = layer(x).clone()
    q_before = layer.weight_q.clone()
    layer.to(torch.float16)
    assert layer.weight_q.dtype == q_before.dtype and torch.equal(layer.weight_q, q_before)
    assert layer.weight_scale.dtype == torch.int32
    assert torch.allclose(layer(x.to(torch.float16)).float(), before.float(), rtol = 1e-2, atol = 1e-2)


def _block_torchao(monkeypatch):
    # A None entry makes any `import torchao.quantization` raise, proving the branch never reaches it.
    monkeypatch.setitem(sys.modules, "torchao", None)
    monkeypatch.setitem(sys.modules, "torchao.quantization", None)


@pytest.mark.parametrize("scheme", ["int8", "fp8"])
def test_quantize_transformer_goes_native_on_rocm(rocm, monkeypatch, scheme):
    _block_torchao(monkeypatch)
    model = _toy()
    x = torch.randn(3, 128, dtype = torch.bfloat16)
    with torch.no_grad():
        ref = model(x).float()
    pipe = types.SimpleNamespace(transformer = model)
    logs = []
    logger = types.SimpleNamespace(
        info = lambda *a: logs.append(a), warning = lambda *a: pytest.fail(f"warned: {a}")
    )
    got = tq.quantize_transformer(pipe, _target(), mode = scheme, min_features = 64, logger = logger)
    assert got == scheme
    assert model._unsloth_runtime_quant == scheme
    swapped = {n for n, m in model.named_modules() if nq.is_native_linear(m)}
    assert swapped == {f"blocks.{i}.{n}" for i in range(2) for n in ("to_q", "ff", "out")}
    assert isinstance(model.blocks[0].small, torch.nn.Linear)
    assert tq.transformer_is_quantised(model) and nq.is_native_quantised(model)
    assert tq.dense_quant_blocker(pipe) is not None  # a second pass would compound the loss
    assert any("torchao-free" in str(a[0]) for a in logs)
    with torch.no_grad():
        out = model(x).float()
    assert ((out - ref).norm() / ref.norm()).item() < 0.05


def test_quantize_transformer_leaves_non_bf16_linears_dense(rocm, monkeypatch):
    _block_torchao(monkeypatch)
    model = _toy()
    model.blocks[1].ff.float()  # a deliberately fp32 layer, as Wan and Hunyuan keep
    tq.quantize_transformer(
        types.SimpleNamespace(transformer = model), _target(), mode = "int8", min_features = 64
    )
    assert isinstance(model.blocks[1].ff, torch.nn.Linear)
    assert nq.is_native_linear(model.blocks[0].ff)


def test_quantize_transformer_skips_lora_side_paths(rocm, monkeypatch):
    _block_torchao(monkeypatch)
    model = _toy()
    model.blocks[0].lora_A = torch.nn.Linear(128, 128).to(torch.bfloat16)
    tq.quantize_transformer(
        types.SimpleNamespace(transformer = model), _target(), mode = "fp8", min_features = 64
    )
    assert isinstance(model.blocks[0].lora_A, torch.nn.Linear)


def test_quantize_transformer_auto_on_rocm_stays_dense(rocm, monkeypatch):
    _block_torchao(monkeypatch)
    model = _toy()
    got = tq.quantize_transformer(
        types.SimpleNamespace(transformer = model), _target(), mode = "auto", min_features = 64
    )
    assert got is None and not tq.transformer_is_quantised(model)


def test_quantize_transformer_failure_is_reported_as_dirty(rocm, monkeypatch):
    # A partial conversion must read as quantised, not dense.
    model = _toy()
    real = nq.native_linear_class()
    calls = []

    def _boom(linear, scheme):
        calls.append(1)
        if len(calls) == 3:
            raise RuntimeError("out of memory")
        return real(linear, scheme)

    monkeypatch.setattr(nq, "native_linear_class", lambda: _boom)
    warned = []
    logger = types.SimpleNamespace(info = lambda *a: None, warning = lambda *a: warned.append(a))
    got = tq.quantize_transformer(
        types.SimpleNamespace(transformer = model),
        _target(),
        mode = "int8",
        min_features = 64,
        logger = logger,
    )
    assert got is None and warned
    assert tq.transformer_is_quantised(model)
    assert not getattr(model, "_unsloth_runtime_quant", None)


def test_the_pass_does_not_hold_every_dense_layer_alive(rocm, monkeypatch):
    import weakref

    model = _toy()
    dense_ff = weakref.ref(model.blocks[0].ff)
    tq.quantize_transformer(
        types.SimpleNamespace(transformer = model), _target(), mode = "int8", min_features = 64
    )
    assert dense_ff() is None


def test_quantize_transformer_on_nvidia_never_reaches_the_native_branch(nvidia, monkeypatch):
    monkeypatch.setattr(
        tq, "_quantize_native", lambda *a, **k: pytest.fail("native branch on an NVIDIA host")
    )
    seen = []
    monkeypatch.setattr(
        tq, "select_transformer_quant_scheme", lambda *a, **k: seen.append(a) or None
    )
    model = _toy()
    for scheme in ("int8", "fp8", "auto"):
        assert (
            tq.quantize_transformer(
                types.SimpleNamespace(transformer = model), _target(), mode = scheme
            )
            is None
        )
    assert len(seen) == 3


@pytest.mark.parametrize("scheme", ["int8", "fp8"])
def test_reading_weight_gives_the_dense_weight_without_storing_it(scheme):
    """PEFT DoRA reads ``base_layer.weight``: dense, rebuilt per read, not in the state dict."""
    torch.manual_seed(0)
    lin = torch.nn.Linear(128, 64).to(torch.bfloat16)
    layer = nq.native_linear_class()(lin, scheme)
    weight = layer.weight
    assert weight.dtype == torch.bfloat16 and weight.shape == (64, 128)
    assert torch.equal(weight, layer.dequantized_weight(torch.bfloat16))
    assert "weight" not in set(layer.state_dict())
    assert all(p is not weight for p in layer.parameters())


def test_a_dora_adapter_runs_on_a_native_base_layer(rocm, monkeypatch):
    peft = pytest.importorskip("peft")
    _block_torchao(monkeypatch)
    torch.manual_seed(0)
    model = _toy()
    cfg = peft.LoraConfig(r = 4, target_modules = ["to_q"], use_dora = True, init_lora_weights = False)
    wrapped = peft.inject_adapter_in_model(cfg, model)
    x = torch.randn(2, 128, dtype = torch.bfloat16)
    with torch.no_grad():
        ref = wrapped(x).float()
    tq.quantize_transformer(
        types.SimpleNamespace(transformer = wrapped), _target(), mode = "int8", min_features = 64
    )
    assert nq.is_native_linear(wrapped.blocks[0].to_q.base_layer)
    with torch.no_grad():
        out = wrapped(x).float()
    assert torch.isfinite(out).all()
    assert float((out - ref).norm() / ref.norm()) < 0.05


def test_peft_wrapped_base_layer_runs_native(rocm, monkeypatch):
    peft = pytest.importorskip("peft")
    _block_torchao(monkeypatch)
    model = _toy()
    cfg = peft.LoraConfig(r = 4, target_modules = ["to_q"], lora_alpha = 4)
    wrapped = peft.inject_adapter_in_model(cfg, model)
    x = torch.randn(2, 128, dtype = torch.bfloat16)
    tq.quantize_transformer(
        types.SimpleNamespace(transformer = wrapped), _target(), mode = "int8", min_features = 64
    )
    to_q = wrapped.blocks[0].to_q
    assert nq.is_native_linear(to_q.base_layer)
    assert isinstance(to_q.lora_A["default"], torch.nn.Linear)
    with torch.no_grad():
        assert torch.isfinite(wrapped(x)).all()


def _image_family(name):
    from core.inference.diffusion_families import _FAMILIES
    return next(f for f in _FAMILIES if f.name == name)


def _video_family(name):
    from core.inference.video_families import _FAMILIES
    return next(f for f in _FAMILIES if f.name == name)


def _image_gate(
    monkeypatch,
    fam,
    pinned,
    *,
    model_kind = "pipeline",
    memory_mode = None,
):
    import core.inference.diffusion as d
    monkeypatch.setattr(d, "effective_te_quant", lambda *a, **k: None)
    d.DiffusionBackend._assert_precision_for_target(
        None,
        fam,
        _target(),
        model_kind = model_kind,
        pinned = pinned,
        te_mode = None,
        memory_mode = memory_mode,
        cpu_offload = False,
    )


def _video_gate(
    monkeypatch,
    fam,
    pinned,
    *,
    model_kind = "pipeline",
    memory_mode = None,
):
    import core.inference.video as v
    monkeypatch.setattr(v, "effective_te_quant", lambda *a, **k: None)
    v._assert_video_precision_for_target(
        fam, _target(), model_kind = model_kind, transformer_quant = pinned, memory_mode = memory_mode
    )


@pytest.mark.parametrize("pinned", ["int8", "fp8"])
@pytest.mark.parametrize("memory_mode", [None, "balanced", "low_vram"])
def test_image_gate_admits_native_int8_fp8_on_rocm(rocm, monkeypatch, pinned, memory_mode):
    _image_gate(monkeypatch, _image_family("qwen-image-2.1"), pinned, memory_mode = memory_mode)


@pytest.mark.parametrize("pinned", ["nvfp4", "mxfp8"])
def test_image_gate_still_refuses_torchao_only_schemes_on_rocm(rocm, monkeypatch, pinned):
    with pytest.raises(RuntimeError, match = "transformer_quant"):
        _image_gate(monkeypatch, _image_family("qwen-image-2.1"), pinned)


@pytest.mark.parametrize("kind", ["gguf", "single_file"])
def test_image_gate_still_refuses_native_on_non_pipeline_loads(rocm, monkeypatch, kind):
    with pytest.raises(RuntimeError, match = "transformer_quant"):
        _image_gate(monkeypatch, _image_family("qwen-image-2.1"), "int8", model_kind = kind)


def test_image_gate_refuses_a_unet_family_on_rocm(rocm, monkeypatch):
    with pytest.raises(RuntimeError, match = "transformer_quant"):
        _image_gate(monkeypatch, _image_family("sdxl"), "int8")


def test_image_gate_refuses_a_denied_family_on_rocm(rocm, monkeypatch):
    monkeypatch.setitem(tq._FAMILY_SCHEME_DENY, "qwen-image-2.1", frozenset({tq.TQ_INT8}))
    with pytest.raises(RuntimeError, match = "transformer_quant"):
        _image_gate(monkeypatch, _image_family("qwen-image-2.1"), "int8")
    _image_gate(monkeypatch, _image_family("qwen-image-2.1"), "fp8")


def test_image_gate_on_nvidia_never_consults_the_native_scheme(nvidia, monkeypatch):
    import core.inference.diffusion as d

    monkeypatch.setattr(d, "native_quant_scheme", lambda *a, **k: pytest.fail("native on NVIDIA"))
    monkeypatch.setattr(d, "dense_transformer_supported", lambda target: False)
    with pytest.raises(RuntimeError, match = "transformer_quant"):
        _image_gate(monkeypatch, _image_family("qwen-image-2.1"), "int8")


@pytest.mark.parametrize("pinned", ["int8", "fp8"])
@pytest.mark.parametrize("memory_mode", [None, "balanced"])
def test_video_gate_admits_native_int8_fp8_on_rocm(rocm, monkeypatch, pinned, memory_mode):
    _video_gate(monkeypatch, _video_family("wan2.2-ti2v-5b"), pinned, memory_mode = memory_mode)


def test_video_gate_keeps_the_modular_workflow_off_the_native_path(rocm, monkeypatch):
    with pytest.raises(RuntimeError, match = "transformer_quant"):
        _video_gate(monkeypatch, _video_family("minimax-h3"), "int8")


@pytest.mark.parametrize("pinned", ["nvfp4", "mxfp8"])
def test_video_gate_still_refuses_torchao_only_schemes_on_rocm(rocm, monkeypatch, pinned):
    with pytest.raises(RuntimeError, match = "transformer_quant"):
        _video_gate(monkeypatch, _video_family("wan2.2-ti2v-5b"), pinned)


def test_video_gate_refuses_native_on_gguf_loads(rocm, monkeypatch):
    with pytest.raises(RuntimeError, match = "transformer_quant"):
        _video_gate(monkeypatch, _video_family("wan2.2-ti2v-5b"), "int8", model_kind = "gguf")


def _native_block(features = 64):
    lin = torch.nn.Linear(features, features, dtype = torch.bfloat16)
    return nq.native_linear_class()(lin, "int8")


def test_stream_group_offload_puts_native_weight_buffers_back_on_the_host():
    go = pytest.importorskip("diffusers.hooks.group_offloading")
    from core.inference.diffusion_memory import install_group_offload_buffer_restore

    install_group_offload_buffer_restore()
    layer = _native_block()
    host = {b: b.data for b in layer.buffers()}
    host.update({p: p.data for p in layer.parameters()})
    group = object.__new__(go.ModuleGroup)
    group.modules, group.parameters, group.buffers = [layer], [], []
    group.stream, group.record_stream = object(), True
    group.cpu_param_dict = dict(host)
    for tensor in list(layer.buffers()) + list(layer.parameters()):
        tensor.data = tensor.data.clone()  # stands in for the onloaded device copy
    group._offload_to_memory()
    assert all(b.data_ptr() == host[b].data_ptr() for b in layer.buffers())
    assert all(p.data_ptr() == host[p].data_ptr() for p in layer.parameters())


def test_buffer_restore_install_is_idempotent():
    go = pytest.importorskip("diffusers.hooks.group_offloading")
    from core.inference.diffusion_memory import install_group_offload_buffer_restore

    install_group_offload_buffer_restore()
    patched = go.ModuleGroup._offload_to_memory
    assert install_group_offload_buffer_restore() is False
    assert go.ModuleGroup._offload_to_memory is patched
    assert getattr(patched, "__wrapped__", None) is not None


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason = "stream group offload needs a CUDA device"
)
def test_native_int8_under_real_stream_group_offload_leaves_nothing_resident():
    pytest.importorskip("diffusers.hooks.group_offloading")
    from diffusers.hooks import apply_group_offloading

    from core.inference.diffusion_memory import install_group_offload_buffer_restore

    install_group_offload_buffer_restore()

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = torch.nn.ModuleList([_native_block(), _native_block()])

        def forward(self, x):
            for block in self.blocks:
                x = block(x)
            return x

    model = Model()
    apply_group_offloading(
        model,
        onload_device = torch.device("cuda"),
        offload_device = torch.device("cpu"),
        offload_type = "block_level",
        num_blocks_per_group = 1,
        use_stream = True,
        non_blocking = True,
        record_stream = True,
    )
    with torch.inference_mode():
        out = model(torch.randn(32, 64, device = "cuda", dtype = torch.bfloat16))
    torch.cuda.synchronize()
    assert out.device.type == "cuda"
    assert {b.device.type for b in model.buffers()} == {"cpu"}
