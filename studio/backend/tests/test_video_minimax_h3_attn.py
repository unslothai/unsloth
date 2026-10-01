# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""MiniMax-H3 attention fast path: fused int8 QKV, strided SDPA layout, per-arch kernel pick.

A tiny real ``MiniMaxH3Transformer3DModel`` (diffusers) is rotated and int8-quantized the way the hosted
pre-quantized denoiser is (ConvRot online rotation + torchao per-row dynamic int8), then every lever is checked for
bit-identity against the stock module on CPU, for its kill switch, and for refusing layouts it does not cover. The
cuDNN / flash arms of the strided processor run only where CUDA is present.
"""

from __future__ import annotations

import copy

import pytest
import torch

h3 = pytest.importorskip("diffusers.models.transformers.transformer_minimax_h3")
pytest.importorskip("torchao")

from core.inference import video_minimax_h3_attn as A  # noqa: E402
from core.inference.diffusion_convrot import is_rotated_linear, rotate_linears_  # noqa: E402

GROUP = 16


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


def _quantize(model: torch.nn.Module, rotate: bool = True, version: int = 1) -> torch.nn.Module:
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
        return isinstance(m, torch.nn.Linear) and fqn.startswith(("transformer_blocks.", "token_refiner."))

    # version 1 = the v1 LinearActivationQuantizedTensor the hosted .pt pickles (resident path on torchao 0.17);
    # version 2 = Int8Tensor (torchao >= 0.18, and what the streamed path rebuilds v1 into).
    try:
        config = Int8DynamicActivationInt8WeightConfig(version = version)
    except TypeError:
        if version != 1:
            pytest.skip("this torchao has no Int8Tensor config")
        config = Int8DynamicActivationInt8WeightConfig()
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
    # two blocks + one refiner block
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
    # The refiner's projections were never rotated: fused, and left unrotated.
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
    # the backend the dispatcher set survives the swap
    assert A._backend_value(quantized.transformer_blocks[0].attn.processor._attention_backend) == "_native_math"
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
