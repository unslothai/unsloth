# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""MiniMax-H3 attention levers on a tiny real H3 transformer quantized like the hosted one: bit-identity vs stock,
kill switches, refusals."""

from __future__ import annotations

import copy

import pytest
import torch

h3 = pytest.importorskip("diffusers.models.transformers.transformer_minimax_h3")
pytest.importorskip("torchao")

from core.inference import video_minimax_h3_attn as A  # noqa: E402
from core.inference.diffusion_convrot import is_rotated_linear, rotate_linears_  # noqa: E402

GROUP = 16


@pytest.fixture(autouse = True)
def _restore_process_flags():
    # set_attention_backend also sets diffusers' process-wide ACTIVE backend, and quantize_ sets inductor globals.
    import torch._inductor.config as inductor_config
    from diffusers.models.attention_dispatch import _AttentionBackendRegistry

    active_backend = _AttentionBackendRegistry._active_backend

    precision = torch.get_float32_matmul_precision()
    knobs = (
        "coordinate_descent_tuning",
        "coordinate_descent_check_all_directions",
        "force_fuse_int_mm_with_mul",
        "fx_graph_cache",
    )
    saved = {k: getattr(inductor_config, k) for k in knobs if hasattr(inductor_config, k)}
    yield
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    _AttentionBackendRegistry.set_active_backend(active_backend)
    torch.set_float32_matmul_precision(precision)
    for k, v in saved.items():
        setattr(inductor_config, k, v)


def _tiny_model() -> torch.nn.Module:
    torch.manual_seed(0)
    model = h3.MiniMaxH3Transformer3DModel(
        num_attention_heads = 2,
        attention_head_dim = 32,
        hidden_size = 64,
        num_layers = 2,
        num_refiner_layers = 1,
        ffn_dim = 128,
        in_channels = 4,
        audio_in_channels = 4,
        patch_size = (1, 2, 2),
        text_dim = 32,
        freq_dim = 32,
        time_embed_hidden_dim = 64,
        time_embed_dim = 32,
        rope_freq_dim = 4,
    )
    return model.eval()


def _quantize(
    model: torch.nn.Module,
    rotate: bool = True,
    version: int = 1,
) -> torch.nn.Module:
    from torchao.quantization import Int8DynamicActivationInt8WeightConfig, quantize_

    if rotate:
        fqns = [
            name
            for name, m in model.named_modules()
            if isinstance(m, torch.nn.Linear)
            and name.startswith("transformer_blocks.")
            and (".attn." in name or ".ff." in name)
        ]
        rotate_linears_(model, fqns, GROUP)

    def only_blocks(m, fqn):
        return isinstance(m, torch.nn.Linear) and fqn.startswith(
            ("transformer_blocks.", "token_refiner.")
        )

    try:
        config = Int8DynamicActivationInt8WeightConfig(version = version)
    except (TypeError, ValueError):
        if version != 1 or not hasattr(
            Int8DynamicActivationInt8WeightConfig, "__dataclass_fields__"
        ):
            pytest.skip(f"this torchao cannot build int8 weights version {version}")
        try:
            config = Int8DynamicActivationInt8WeightConfig()
            if getattr(config, "version", 1) != 1:
                pytest.skip("this torchao cannot build int8 weights version 1")
        except Exception:
            pytest.skip("this torchao cannot build int8 weights version 1")
    quantize_(model, config, filter_fn = only_blocks)
    return model


def _inputs(seed: int = 1) -> dict:
    g = torch.Generator().manual_seed(seed)
    n_text, n_video, n_audio = 5, 8, 3
    seq = n_text + n_video + n_audio
    text_idx = torch.arange(0, n_text)
    video_idx = torch.arange(n_text, n_text + n_video)
    audio_idx = torch.arange(n_text + n_video, seq)
    tags = torch.zeros(seq, dtype = torch.long)
    tags[text_idx] = 1
    tags[audio_idx] = 2
    pos = torch.randint(0, 6, (seq, 3), generator = g)
    return dict(
        hidden_states = torch.randn(1, n_video, 16, generator = g),
        audio_hidden_states = torch.randn(1, n_audio, 4, generator = g),
        encoder_hidden_states = torch.randn(1, n_text, 32, generator = g),
        timestep = torch.tensor([0.7, 0.2]),
        timestep_indices = torch.randint(0, 2, (seq,), generator = g),
        token_tags = tags,
        position_ids = pos,
        video_indices = video_idx,
        audio_indices = audio_idx,
        text_indices = text_idx,
        return_dict = False,
    )


def _run(model, **kw):
    with torch.no_grad():
        return model(**_inputs(), **kw)


@pytest.fixture(params = [1, 2], ids = ["v1_laqt", "v2_int8tensor"])
def quantized(request):
    return _quantize(_tiny_model(), version = request.param)


def _equal(a, b) -> bool:
    return all(torch.equal(x, y) for x, y in zip(a, b))


def test_fused_qkv_matches_the_three_projections_bit_for_bit(quantized):
    stock = copy.deepcopy(quantized)
    n = A.fuse_h3_qkv_(quantized)
    assert n == 3
    assert A.fused_qkv_count(quantized) == 3
    assert _equal(_run(stock), _run(quantized))


def test_fused_module_keeps_the_rotation_and_a_per_row_int8_weight(quantized):
    kind = type(quantized.transformer_blocks[0].attn.to_q.weight)
    A.fuse_h3_qkv_(quantized)
    attn = quantized.transformer_blocks[0].attn
    assert attn.fused_projections is True
    for name in ("to_q", "to_k", "to_v"):
        assert not hasattr(attn, name)
    assert is_rotated_linear(attn.to_qkv) and attn.to_qkv.convrot_groupsize == GROUP
    assert type(attn.to_qkv.weight) is kind and A._per_token_int8(attn.to_qkv.weight)
    assert tuple(attn.to_qkv.weight.shape) == (3 * 64, 64)
    refiner_attn = quantized.token_refiner.refiner_blocks[0].attn
    assert refiner_attn.fused_projections is True and not is_rotated_linear(refiner_attn.to_qkv)


def test_fused_qkv_kill_switch_leaves_the_model_alone(quantized, monkeypatch):
    monkeypatch.setenv(A.FUSED_QKV_ENV, "0")
    assert A.fuse_h3_qkv_(quantized) == 0
    assert A.fused_qkv_count(quantized) == 0
    assert hasattr(quantized.transformer_blocks[0].attn, "to_q")


def test_dense_projections_are_not_fused():
    model = _tiny_model()
    assert A.fuse_h3_qkv_(model) == 0
    assert hasattr(model.transformer_blocks[0].attn, "to_q")


def test_projections_with_a_hook_or_bias_are_not_fused(quantized):
    attn0 = quantized.transformer_blocks[0].attn
    attn1 = quantized.transformer_blocks[1].attn
    attn0.to_k.register_forward_pre_hook(lambda m, a: None)
    attn1.to_v.bias = torch.nn.Parameter(torch.zeros(64))
    A.fuse_h3_qkv_(quantized)
    assert not attn0.fused_projections and hasattr(attn0, "to_k")
    assert not attn1.fused_projections and hasattr(attn1, "to_v")
    assert quantized.token_refiner.refiner_blocks[0].attn.fused_projections


def test_padded_projections_are_not_fused(quantized):
    from core.inference.diffusion_quant_pad import PadToMinM
    from core.inference.diffusion_transformer_quant import apply_small_m_padding

    assert apply_small_m_padding(quantized, "int8", "minimax-h3")
    refiner = quantized.token_refiner.refiner_blocks[0].attn
    assert isinstance(refiner.to_q, PadToMinM)
    assert A.fuse_h3_qkv_(quantized) == 2
    assert not refiner.fused_projections and isinstance(refiner.to_q, PadToMinM)
    assert quantized.transformer_blocks[0].attn.fused_projections


def test_mixed_rotation_is_not_fused(quantized):
    attn = quantized.transformer_blocks[0].attn
    attn.to_k.convrot_groupsize = 64
    A.fuse_h3_qkv_(quantized)
    assert not attn.fused_projections


@pytest.mark.parametrize("fuse", [False, True])
def test_strided_processor_matches_stock_bit_for_bit(quantized, fuse):
    quantized.set_attention_backend("_native_math")
    if fuse:
        A.fuse_h3_qkv_(quantized)
    stock = copy.deepcopy(quantized)
    n = A.install_strided_attention(quantized)
    assert n == 3 and A.strided_attention_count(quantized) == 3
    assert (
        A._backend_value(quantized.transformer_blocks[0].attn.processor._attention_backend)
        == "_native_math"
    )
    assert _equal(_run(stock), _run(quantized))


def test_strided_processor_defers_to_stock_for_other_backends(quantized):
    A.install_strided_attention(quantized)
    proc = quantized.transformer_blocks[0].attn.processor
    proc._attention_backend = None
    stock = copy.deepcopy(quantized)
    for m in stock.modules():
        if isinstance(m, h3.MiniMaxH3Attention):
            m.set_processor(h3.MiniMaxH3AttnProcessor())
    assert _equal(_run(stock), _run(quantized))


def test_strided_kill_switch(quantized, monkeypatch):
    monkeypatch.setenv(A.STRIDED_ATTN_ENV, "0")
    assert A.install_strided_attention(quantized) == 0
    assert A.strided_attention_count(quantized) == 0


def test_install_is_idempotent(quantized):
    assert A.install_strided_attention(quantized) == 3
    assert A.install_strided_attention(quantized) == 0
    assert A.strided_attention_count(quantized) == 3


@pytest.mark.parametrize(
    "selected,cap,expected",
    [
        ("_native_cudnn", (8, 0), "_native_flash"),
        ("_native_cudnn", (8, 9), "_native_flash"),
        ("_native_cudnn", (12, 0), "_native_cudnn"),
        ("_native_cudnn", (9, 0), "_native_cudnn"),
        ("_native_cudnn", (10, 0), "_native_cudnn"),
        ("_native_cudnn", (8, 6), "_native_cudnn"),
        ("flash", (8, 0), "flash"),
        ("sage", (8, 9), "sage"),
        (None, (8, 0), None),
    ],
)
def test_arch_pick_moves_only_the_automatic_cudnn_pick_on_measured_archs(selected, cap, expected):
    assert A.h3_attention_backend(selected, cap) == expected


@pytest.mark.parametrize("requested", ["cudnn", "flash", "native"])
def test_arch_pick_keeps_an_explicit_request(requested):
    assert A.h3_attention_backend("_native_cudnn", (8, 0), requested = requested) == "_native_cudnn"
    assert A.h3_attention_backend("_native_cudnn", (8, 0), requested = "auto") == "_native_flash"


def test_h3_load_wires_the_request_and_the_speed_gate():
    import ast
    import inspect
    import textwrap

    from core.inference.video import VideoBackend

    tree = ast.parse(textwrap.dedent(inspect.getsource(VideoBackend._load_h3_modular_pipeline)))
    calls = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "h3_attention_backend"
    ]
    assert calls and all(any(k.arg == "requested" for k in c.keywords) for c in calls)
    gates = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.If)
        and any(
            getattr(c.func, "id", "") == "install_strided_attention"
            for c in ast.walk(n)
            if isinstance(c, ast.Call)
        )
    ]
    assert any("SPEED_OFF" in ast.unparse(g.test) for g in gates)


def test_arch_pick_kill_switch(monkeypatch):
    monkeypatch.setenv(A.ATTN_ARCH_ENV, "0")
    assert A.h3_attention_backend("_native_cudnn", (8, 0)) == "_native_cudnn"


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "cuDNN / flash SDPA need CUDA")
@pytest.mark.parametrize("backend", ["_native_cudnn", "_native_flash"])
def test_strided_processor_on_cuda_matches_the_stock_backend(backend):
    model = _tiny_model().to("cuda", torch.bfloat16)
    model.set_attention_backend(backend)
    stock = copy.deepcopy(model)
    assert A.install_strided_attention(model) == 3
    kw = {k: (v.to("cuda") if torch.is_tensor(v) else v) for k, v in _inputs().items()}
    for k in ("hidden_states", "audio_hidden_states", "encoder_hidden_states"):
        kw[k] = kw[k].to(torch.bfloat16)
    with torch.no_grad():
        assert _equal(stock(**kw), model(**kw))


from core.inference import video_minimax_h3_qknorm as Q  # noqa: E402


def _qk_inputs(
    device = "cpu",
    dtype = torch.bfloat16,
    seq = 37,
    heads = 3,
    dim = 128,
    rot = 96,
    strided = False,
):
    g = torch.Generator(device = device).manual_seed(3)
    if strided:
        qkv = torch.randn(1, seq, 3 * heads * dim, device = device, dtype = dtype, generator = g)
        x = qkv[..., : heads * dim].unflatten(-1, (heads, dim))
    else:
        x = torch.randn(1, seq, heads, dim, device = device, dtype = dtype, generator = g)
    w = (1 + 0.2 * torch.randn(dim, device = device, generator = g)).to(dtype)
    ang = torch.rand(seq, rot // 2, device = device, generator = g) * 30
    ang = torch.cat((ang, ang), dim = -1)
    return x, w, ang.cos(), ang.sin()


def test_qk_norm_rope_reference_is_the_stock_module_chain():
    x, w, cos, sin = _qk_inputs()
    norm = torch.nn.RMSNorm(128, eps = 1e-5).to(torch.bfloat16)
    norm.weight.data.copy_(w)
    stock = h3._apply_rotary_emb(norm(x), cos, sin)
    assert torch.equal(Q.reference_qk_norm_rope(x, w, cos, sin, 1e-5), stock)


def test_qk_norm_rope_op_off_cuda_is_the_stock_math():
    x, w, cos, sin = _qk_inputs()
    assert torch.equal(
        Q.qk_norm_rope(x, w, cos, sin, 1e-5), Q.reference_qk_norm_rope(x, w, cos, sin, 1e-5)
    )


def test_strided_processor_with_fused_qk_rope_matches_stock(quantized):
    quantized.set_attention_backend("_native_math")
    stock = copy.deepcopy(quantized)
    A.install_strided_attention(quantized)
    assert quantized.transformer_blocks[0].attn.processor._unsloth_qk_rope is True
    assert _equal(_run(stock), _run(quantized))


def test_fused_qk_rope_kill_switch(quantized, monkeypatch):
    monkeypatch.setenv(Q.QK_ROPE_ENV, "0")
    A.install_strided_attention(quantized)
    assert quantized.transformer_blocks[0].attn.processor._unsloth_qk_rope is False


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "the Triton kernel runs on CUDA only")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("strided", [False, True])
@pytest.mark.parametrize("seq", [1, 777])
def test_qk_kernel_matches_the_compiled_stock_math(dtype, strided, seq):
    import torch._inductor.config as ic

    x, w, cos, sin = _qk_inputs("cuda", dtype, seq = seq, heads = 56, strided = strided)
    ours = Q._launch(x, w, cos, sin, 1e-5)
    eager = Q.reference_qk_norm_rope(x, w, cos, sin, 1e-5)
    prev = ic.emulate_precision_casts
    ic.emulate_precision_casts = True
    try:
        compiled = torch.compile(Q.reference_qk_norm_rope, dynamic = False)(x, w, cos, sin, 1e-5)
    finally:
        ic.emulate_precision_casts = prev
    assert ours.is_contiguous() and ours.shape == x.shape
    # Only the sum-of-squares order is free: a few rows may land one rounding step away.
    budget = ours.numel() // (100000 if dtype == torch.bfloat16 else 20000)
    for ref in (eager, compiled):
        diff = (ours.float() - ref.float()).abs()
        assert int((ours != ref).sum()) <= max(2, budget)
        assert float(diff.max()) <= float((ref.float().abs().max() * 2**-7))
    # negative control: a one-ulp-scale perturbation must be caught
    bad = (ours.float() * (1 + 2**-7)).to(dtype)
    assert int((bad != eager).sum()) > ours.numel() // 2


def _graph_breaks(model) -> int:
    import torch._dynamo

    torch._dynamo.reset()
    with torch.no_grad():
        explained = torch._dynamo.explain(model)(**_inputs())
    torch._dynamo.reset()
    return explained.graph_break_count


@pytest.mark.parametrize("qk_rope", ["1", "0"])
def test_strided_processor_adds_no_graph_break(monkeypatch, qk_rope):
    # A break here splits every compiled block in two (2x slower steps).
    monkeypatch.setenv(Q.QK_ROPE_ENV, qk_rope)
    model = _tiny_model()
    model.set_attention_backend("_native_math")
    stock = _graph_breaks(model)
    assert A.install_strided_attention(model) > 0
    assert model.transformer_blocks[0].attn.processor._unsloth_qk_rope is (qk_rope == "1")
    assert _graph_breaks(model) == stock
