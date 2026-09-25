# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for the MiniMax-H3 video VAE speed layer; CPU tests hermetic, CUDA tests skip without a GPU."""

from __future__ import annotations

import copy
import types

import pytest
import torch

from core.inference import video_minimax_h3_vae as H

CUDA = torch.cuda.is_available() and not getattr(torch.version, "hip", None)
needs_cuda = pytest.mark.skipif(
    not CUDA or H._kernels() is None or not H._triton_version_ok(),
    reason = "needs an NVIDIA GPU with a Triton the kernels are verified on",
)


@pytest.fixture(autouse = True)
def _clean_env(monkeypatch):
    for name in (
        H.H3_VAE_FAST_ENV,
        H.H3_VAE_INT8_ENV,
        H.H3_VAE_TILE_BATCH_ENV,
        H.FP16_ACCUM_DISABLE_ENV,
    ):
        monkeypatch.delenv(name, raising = False)


def test_off_tier_engages_nothing():
    assert H.plan_h3_vae_levers("off", workflow = "fl2va") == ()
    assert H.plan_h3_vae_levers("OFF", workflow = "fl2va", consumer_gpu = True) == ()


@pytest.mark.parametrize("tier", ["default", "max", None, ""])
def test_default_and_max_get_the_near_lossless_set(tier):
    plan = H.plan_h3_vae_levers(tier, workflow = "fl2va", consumer_gpu = False)
    assert plan == (
        H.LEVER_FUSED_ENCODER,
        H.LEVER_FP16_ENCODER,
        H.LEVER_FUSED_DECODER,
        H.LEVER_TILE_BATCH,
    )


def test_eager_stays_rounding_level():
    plan = H.plan_h3_vae_levers("eager", workflow = "fl2va", consumer_gpu = True)
    assert plan == (H.LEVER_FUSED_ENCODER, H.LEVER_FUSED_DECODER, H.LEVER_TILE_BATCH)


def test_fp16_accumulation_only_under_max_on_consumer_gpus(monkeypatch):
    assert H.LEVER_FP16_ACCUM in H.plan_h3_vae_levers("max", workflow = "fl2va", consumer_gpu = True)
    assert H.LEVER_FP16_ACCUM not in H.plan_h3_vae_levers(
        "max", workflow = "fl2va", consumer_gpu = False
    )
    for tier in ("eager", "default"):
        assert H.LEVER_FP16_ACCUM not in H.plan_h3_vae_levers(
            tier, workflow = "fl2va", consumer_gpu = True
        )
    monkeypatch.setenv(H.FP16_ACCUM_DISABLE_ENV, "1")
    assert H.LEVER_FP16_ACCUM not in H.plan_h3_vae_levers(
        "max", workflow = "fl2va", consumer_gpu = True
    )


def test_int8_decoder_is_opt_in_only(monkeypatch):
    for tier in ("eager", "default", "max"):
        assert H.LEVER_INT8_DECODER not in H.plan_h3_vae_levers(
            tier, workflow = "fl2va", consumer_gpu = True
        )
    monkeypatch.setenv(H.H3_VAE_INT8_ENV, "1")
    assert H.LEVER_INT8_DECODER in H.plan_h3_vae_levers("default", workflow = "fl2va")


def test_t2va_skips_the_encoder_levers():
    plan = H.plan_h3_vae_levers("default", workflow = "t2va")
    assert H.LEVER_FUSED_ENCODER not in plan and H.LEVER_FP16_ENCODER not in plan
    assert H.LEVER_FUSED_DECODER in plan


def test_env_switches(monkeypatch):
    monkeypatch.setenv(H.H3_VAE_FAST_ENV, "0")
    assert H.plan_h3_vae_levers("max", workflow = "fl2va", consumer_gpu = True) == ()
    monkeypatch.delenv(H.H3_VAE_FAST_ENV)
    monkeypatch.setenv(H.H3_VAE_INT8_ENV, "1")
    assert H.plan_h3_vae_levers("off", workflow = "fl2va") == ()


def test_apply_is_a_no_op_off_cuda(monkeypatch):
    monkeypatch.setattr(H, "cuda_fast_path_available", lambda vae = None: False)
    vae = types.SimpleNamespace(decode = object(), encoder = object())
    stock_decode = vae.decode
    assert H.apply_h3_vae_speedups(vae, speed_mode = "max", workflow = "fl2va", consumer_gpu = True) == ()
    assert vae.decode is stock_decode
    assert H.apply_h3_vae_speedups(None, speed_mode = "max") == ()


@pytest.mark.skipif(CUDA, reason = "checks the real gate on a host without an NVIDIA GPU")
@pytest.mark.parametrize("tier", ["eager", "default", "max"])
def test_without_cuda_the_real_gate_leaves_the_stock_vae_untouched(tier):
    vae = _tiny_vae()
    encoder_forward, decoder_forward, decode = (
        vae.encoder.forward,
        vae.decoder.forward,
        vae.decode,
    )
    weights = {k: v.clone() for k, v in vae.state_dict().items()}
    assert H.plan_h3_vae_levers(tier, workflow = "fl2va")
    assert H.apply_h3_vae_speedups(vae, speed_mode = tier, workflow = "fl2va") == ()
    assert vae.encoder.forward == encoder_forward and vae.decoder.forward == decoder_forward
    assert vae.decode == decode
    for k, v in vae.state_dict().items():
        assert v.dtype == weights[k].dtype and torch.equal(v, weights[k]), k


def test_decode_scope_pins_and_restores_fp16_accumulation():
    matmul = torch.backends.cuda.matmul
    if not hasattr(matmul, "allow_fp16_accumulation"):
        pytest.skip("torch without allow_fp16_accumulation")
    seen = []

    class _VAE:
        def decode(self, z):
            seen.append(matmul.allow_fp16_accumulation)
            return z

    prev = matmul.allow_fp16_accumulation
    try:
        for flag_before, planned in ((True, False), (False, True), (False, False)):
            vae = _VAE()
            matmul.allow_fp16_accumulation = flag_before
            assert H._install_decode_scope(vae, fp16_accum = planned)
            assert vae.decode(3) == 3
            assert (
                seen[-1] is planned
            ), "the decode must run under the planned flag, not the process-wide one"
            assert (
                matmul.allow_fp16_accumulation is flag_before
            ), "the process-wide flag must be restored"
    finally:
        matmul.allow_fp16_accumulation = prev


def _tiny_vae():
    diffusers = pytest.importorskip("diffusers")
    cls = getattr(diffusers, "AutoencoderKLMiniMaxH3", None)
    if cls is None:
        pytest.skip("diffusers without AutoencoderKLMiniMaxH3")
    torch.manual_seed(0)
    vae = cls(
        block_out_channels = (16, 32),
        layers_per_block = 2,
        spatial_downsample_factors = (2, 2),
        temporal_downsample_factors = (2, 1),
        norm_num_groups = 8,
        latent_channels = 8,
        decoder_num_layers = 2,
        decoder_num_attention_heads = 2,
        decoder_attention_head_dim = 16,
        decoder_num_register_tokens = 2,
        decoder_ffn_mult = 2,
        decoder_rope_dim_ratio = 0.75,
        clip_length = 9,
        token_drop = 1,
    ).eval()
    with torch.no_grad():
        for name, p in vae.named_parameters():
            if name.endswith("scale1") or name.endswith("scale2"):
                p.normal_(0, 0.5)
            elif "norm" in name and name.endswith("weight"):
                p.normal_(1.0, 0.2)
            elif name.endswith("bias"):
                p.normal_(0, 0.1)
    vae.tile_sample_min_height = vae.tile_sample_min_width = 16
    vae.tile_sample_min_overlap_height = vae.tile_sample_min_overlap_width = 4
    return vae


DEVICES = ["cpu", pytest.param("cuda", marks = needs_cuda)]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("frames", [1, 9, 18])
def test_fused_encoder_matches_the_stock_encoder(frames, device, monkeypatch):
    monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", False)
    vae = _tiny_vae().to(device)
    fast = copy.deepcopy(vae)
    assert H._install_encoder(fast, fp16 = False)
    x = torch.randn(1, 3, frames, 24, 40, device = device)
    with torch.no_grad():
        ref = vae.encoder(x)
        got = fast.encoder(x)
    assert not getattr(
        fast.encoder, "_unsloth_fast_failed", False
    ), "the fused path fell back to stock"
    assert got.shape == ref.shape and got.dtype == ref.dtype
    torch.testing.assert_close(got, ref, rtol = 1e-4, atol = 1e-4)


@pytest.mark.parametrize("device", DEVICES)
def test_fused_fp16_encoder_returns_the_input_dtype_and_stays_close(device):
    vae = _tiny_vae().to(device)
    ref = copy.deepcopy(vae)
    assert H._install_encoder(vae, fp16 = True)
    x = torch.randn(1, 3, 9, 24, 40, device = device)
    with torch.no_grad():
        got, want = vae.encoder(x), ref.encoder(x)
    assert not getattr(vae.encoder, "_unsloth_fast_failed", False)
    assert got.dtype is torch.float32
    assert ((got - want).norm() / want.norm()) < 1e-2


def test_fused_encoder_fp16_casts_weights_and_returns_the_input_dtype():
    vae = _tiny_vae()
    assert H._install_encoder(vae, fp16 = True)
    convs = [m for m in vae.encoder.modules() if isinstance(m, torch.nn.Conv3d)]
    assert convs and all(m.weight.dtype is torch.float16 for m in convs)
    assert all(
        m.weight.is_contiguous(memory_format = torch.channels_last_3d)
        for m in convs
        if m is not vae.encoder.conv_in
    )
    assert vae.quant_conv.weight.dtype is torch.float32


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("frames", [1, 9])
def test_the_condition_encode_recipe_runs_through_the_fp16_encoder(frames, device):
    # keyframe path: vae.encode feeds the encoder output to a float32 quant_conv, so it must be float32
    from core.inference.video_minimax_h3 import trim_h3_video_vae

    vae = _tiny_vae().to(device)
    vae.register_to_config(latents_mean = [0.0] * 8, latents_std = [1.0] * 8)
    trim_h3_video_vae(vae, workflow = "fl2va")
    ref = copy.deepcopy(vae)
    assert H._install_encoder(vae, fp16 = True)
    pixels = torch.randint(0, 256, (1, 3, frames, 32, 48), dtype = torch.uint8, device = device)
    with torch.no_grad():
        try:
            from diffusers.modular_pipelines.minimax_h3.encoders import encode_vae_condition
        except ImportError:
            x = pixels.float() / 127.5 - 1.0
            got, want = (m.encode(x).latent_dist.mean for m in (vae, ref))
        else:
            args = (pixels, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            got, want = encode_vae_condition(vae, *args), encode_vae_condition(ref, *args)
    assert not getattr(vae.encoder, "_unsloth_fast_failed", False)
    assert vae.quant_conv.weight.dtype is torch.float32
    assert got.dtype is torch.float32 and got.shape == want.shape
    assert ((got - want).norm() / want.norm()) < 1e-2


@needs_cuda
def test_the_decoder_residual_stream_stays_float32_under_the_decode_autocast(monkeypatch):
    # the stock blocks keep the residual in float32 under float16 autocast, so must we
    from core.inference.video_minimax_h3 import trim_h3_video_vae

    vae = _tiny_vae().cuda()
    trim_h3_video_vae(vae, workflow = "fl2va")
    ref = copy.deepcopy(vae)
    assert H._install_decoder(vae)
    seen = []
    stack = H._fast_block_stack

    def spy(decoder, hidden_states, *args):
        seen.append(hidden_states.dtype)
        out = stack(decoder, hidden_states, *args)
        seen.append(out.dtype)
        return out

    monkeypatch.setattr(H, "_fast_block_stack", spy)
    z = torch.randn(1, 8, 3, 4, 5, device = "cuda")
    with torch.no_grad(), torch.autocast("cuda", dtype = torch.float16):
        got, want = vae.decoder(z), ref.decoder(z)
    assert not getattr(vae.decoder, "_unsloth_fast_failed", False)
    assert seen == [torch.float32, torch.float32]
    assert got.dtype == want.dtype
    torch.testing.assert_close(got.float(), want.float(), rtol = 2e-3, atol = 2e-3)


def test_norm_silu_pad_reference_matches_diffusers_ops():
    from torch.nn import functional as F

    vae = _tiny_vae()
    resnet = vae.encoder.down_blocks[0].resnets[0]
    x = torch.randn(1, 16, 5, 12, 10)
    ref = F.pad(F.silu(resnet.norm1(x)), (1, 1, 1, 1, 0, 0), mode = "reflect")
    ref = F.pad(ref, (0, 0, 0, 0, 2, 0))
    got = H.norm_silu_pad_reference(x, resnet.norm1, (1, 1, 1, 1), 2)
    assert got.is_contiguous(memory_format = torch.channels_last_3d)
    torch.testing.assert_close(got, ref, rtol = 1e-5, atol = 1e-5)


def test_single_frame_last_tap_equals_convolving_the_zero_frames():
    from torch.nn import functional as F

    vae = _tiny_vae()
    conv = vae.encoder.down_blocks[0].resnets[0].conv1
    x = torch.randn(1, 16, 1, 8, 8)
    padded = F.pad(F.pad(x, (1, 1, 1, 1, 0, 0), mode = "reflect"), (0, 0, 0, 0, 2, 0))
    ref = F.conv3d(padded, conv.weight, conv.bias)
    got, bias = H._causal_conv(conv, x)
    torch.testing.assert_close(got + bias.view(1, -1, 1, 1, 1), ref, rtol = 1e-5, atol = 1e-5)


def test_pending_bias_folds_through_a_1x1_conv():
    from torch.nn import functional as F

    conv = torch.nn.Conv3d(16, 32, 1)
    x = torch.randn(1, 16, 3, 5, 7)
    pending = torch.randn(16)
    ref = F.conv3d(x + pending.view(1, -1, 1, 1, 1), conv.weight, conv.bias)
    got, bias = H._causal_conv(conv, x, pending)
    torch.testing.assert_close(got + bias.view(1, -1, 1, 1, 1), ref, rtol = 1e-5, atol = 1e-5)


def test_pending_bias_is_added_before_the_norm_and_the_pad():
    from torch.nn import functional as F

    norm = torch.nn.GroupNorm(4, 16, eps = 1e-6)
    x = torch.randn(1, 16, 4, 6, 6)
    pending = torch.randn(16)
    y = x + pending.view(1, -1, 1, 1, 1)
    frames = y.permute(0, 2, 1, 3, 4).reshape(4, 16, 6, 6)
    ref = (
        F.silu(F.group_norm(frames, 4, norm.weight, norm.bias, 1e-6))
        .reshape(1, 4, 16, 6, 6)
        .permute(0, 2, 1, 3, 4)
    )
    ref = F.pad(F.pad(ref, (1, 1, 1, 1, 0, 0), mode = "reflect"), (0, 0, 0, 0, 2, 0))
    torch.testing.assert_close(
        H.norm_silu_pad(x, norm, (1, 1, 1, 1), 2, in_bias = pending), ref, rtol = 1e-5, atol = 1e-5
    )
    # a zero frame stays zero: the bias belongs to real frames only
    got = H.norm_silu_pad(x, None, (0, 1, 0, 1), 2, in_bias = pending)
    assert got[:, :, :2].abs().max() == 0


@pytest.mark.parametrize("device", DEVICES)
def test_fused_decoder_matches_the_stock_decoder(device):
    vae = _tiny_vae().to(device)
    fast = copy.deepcopy(vae)
    assert H._install_decoder(fast)
    z = torch.randn(1, 8, 3, 4, 5, device = device)
    with torch.no_grad():
        ref = vae.decoder(z)
        got = fast.decoder(z)
    assert not getattr(
        fast.decoder, "_unsloth_fast_failed", False
    ), "the fused path fell back to stock"
    assert got.shape == ref.shape
    torch.testing.assert_close(got, ref, rtol = 1e-4, atol = 1e-4)


def test_tile_batching_matches_the_stock_tile_loop(monkeypatch):
    vae = _tiny_vae()
    fast = copy.deepcopy(vae)
    assert H._install_tile_batch(fast)
    z = torch.randn(1, 8, 3, 6, 9)  # 24x36 px: 2x3 tiles of 16 px
    with torch.no_grad():
        ref = vae._decode_clip(z)
        for batch in ("1", "2", "4", "8"):
            monkeypatch.setenv(H.H3_VAE_TILE_BATCH_ENV, batch)
            torch.testing.assert_close(fast._decode_clip(z), ref, rtol = 1e-5, atol = 1e-5)


def test_int8_decoder_replaces_block_linears_and_stays_close():
    vae = _tiny_vae()
    fast = copy.deepcopy(vae)
    freed = H._install_int8_decoder(fast, keep_blocks = ())
    assert freed > 0
    lin = fast.decoder.transformer_blocks[0].attn.to_q
    assert lin.weight is None and lin._unsloth_wq.dtype is torch.int8
    assert fast.decoder.proj_out.weight is not None
    z = torch.randn(1, 8, 3, 4, 5)
    with torch.no_grad():
        ref = vae.decoder(z)
        got = fast.decoder(z)
    rel = (got - ref).norm() / ref.norm()
    assert rel < 0.05


def test_apply_skips_when_already_applied(monkeypatch):
    monkeypatch.setattr(H, "cuda_fast_path_available", lambda vae = None: True)
    vae = _tiny_vae()
    engaged = H.apply_h3_vae_speedups(
        vae, speed_mode = "default", workflow = "fl2va", consumer_gpu = True
    )
    assert set(engaged) == {
        H.LEVER_FUSED_ENCODER,
        H.LEVER_FP16_ENCODER,
        H.LEVER_FUSED_DECODER,
        H.LEVER_TILE_BATCH,
    }
    encoder_forward = vae.encoder.forward
    assert H.apply_h3_vae_speedups(vae, speed_mode = "max", workflow = "fl2va") == engaged
    assert vae.encoder.forward is encoder_forward


@needs_cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("channels_last", [False, True])
@pytest.mark.parametrize(
    "frames,pad,front,with_norm",
    [(9, (1, 1, 1, 1), 2, True), (1, (1, 1, 1, 1), 0, True), (7, (0, 1, 0, 1), 2, False)],
)
def test_norm_silu_pad_kernel(dtype, channels_last, frames, pad, front, with_norm):
    torch.manual_seed(0)
    norm = torch.nn.GroupNorm(8, 64, eps = 1e-6).cuda()
    with torch.no_grad():
        norm.weight.normal_()
        norm.bias.normal_()
    x = (torch.randn(1, 64, frames, 20, 36, device = "cuda") * 3 + 1).to(dtype)
    if channels_last:
        x = x.contiguous(memory_format = torch.channels_last_3d)
    n = norm if with_norm else None
    ref = H.norm_silu_pad_reference(x, n, pad, front)
    got = H.norm_silu_pad(x, n, pad, front)
    assert got.shape == ref.shape and got.stride() == ref.stride()
    tol = 1e-5 if dtype is torch.float32 else 1e-2
    torch.testing.assert_close(got.float(), ref.float(), rtol = tol, atol = tol)


@needs_cuda
def test_add_rmsnorm_kernel():
    torch.manual_seed(0)
    norm = torch.nn.RMSNorm(256, eps = 1e-5).cuda()
    with torch.no_grad():
        norm.weight.normal_(1, 0.1)
    h = torch.randn(300, 256, device = "cuda")
    o = torch.randn(300, 256, device = "cuda").half()
    s = torch.randn(256, device = "cuda")
    ref_h = h + o * s
    ref_n = norm(ref_h).half()
    got_n = H._add_rmsnorm(h, o, s, norm, torch.float16)
    assert torch.equal(h, ref_h)
    torch.testing.assert_close(got_n, ref_n, rtol = 2e-3, atol = 2e-3)


@needs_cuda
def test_qk_norm_rope_kernel_matches_the_reference_processor():
    torch.manual_seed(0)
    heads, dim = 4, 64
    x = torch.randn(2, 100, heads * dim, device = "cuda").half()
    angles = torch.rand(2, 100, 1, 24, device = "cuda") * 6.28
    angles = angles.tile(2)
    cos, sin = angles.cos().half(), angles.sin().half()
    ref = H.qk_norm_rope_reference(x, cos, sin, heads, 1e-5)
    got = x.clone()
    H._qk_norm_rope_(got, cos.reshape(-1, 48), sin.reshape(-1, 48), heads, 1e-5)
    assert (got == ref).float().mean() > 0.99
    torch.testing.assert_close(got.float(), ref.float(), rtol = 2e-3, atol = 2e-3)


@needs_cuda
def test_swiglu_kernel():
    x = torch.randn(64, 2 * 3000, device = "cuda").half()
    hidden, gate = x.chunk(2, dim = -1)
    got, ref = H._swiglu(x), hidden * torch.nn.functional.silu(gate)
    assert (got == ref).float().mean() > 0.999
    torch.testing.assert_close(got, ref, rtol = 2e-3, atol = 2e-3)


@needs_cuda
def test_int8_linear_kernel_path_matches_the_torch_path():
    torch.manual_seed(0)
    lin = torch.nn.Linear(256, 512).cuda().half()
    ref_lin = copy.deepcopy(lin)
    holder = torch.nn.Module()
    holder.transformer_blocks = torch.nn.ModuleList([torch.nn.ModuleDict({"l": lin})])
    vae = types.SimpleNamespace(decoder = holder)
    assert H._install_int8_decoder(vae, keep_blocks = ()) > 0
    x = torch.randn(3, 40, 256, device = "cuda").half()
    got = lin(x)
    ref = ref_lin(x)
    assert got.shape == ref.shape and got.dtype == ref.dtype
    assert ((got.float() - ref.float()).norm() / ref.float().norm()) < 0.02


@needs_cuda
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_int8_quant_rows_kernel_matches_the_torch_codes(dtype):
    g = torch.Generator(device = "cuda").manual_seed(0)
    rows, n = 4 * 1797, 2048
    x = torch.randn(rows, n, device = "cuda", generator = g)
    x = (x * (torch.rand(n, device = "cuda", generator = g) * 4 + 0.1)).to(dtype)
    # exact halves: absmax 127 gives s = 1, where half-to-even and half-away-from-zero disagree
    x[0, :6] = torch.tensor([127.0, 2.5, -3.5, 0.5, -0.5, 126.5], dtype = dtype)
    x[0, 6:] = 0
    xf = x.float()
    ref_s = xf.abs().amax(dim = 1).clamp(min = 1e-12) / 127.0
    ref_q = (xf / ref_s[:, None]).round().clamp(-127, 127).to(torch.int8)
    q = torch.empty(rows, n, dtype = torch.int8, device = "cuda")
    s = torch.empty(rows, dtype = torch.float32, device = "cuda")
    H._kernels().quant_rows[(rows,)](x, q, s, n, BLOCK_N = n, num_warps = 8)
    assert torch.equal(s, ref_s)
    assert int((q != ref_q).sum()) == 0
    assert q[0, :6].tolist() == [127, 2, -4, 0, 0, 126]


@needs_cuda
@pytest.mark.parametrize("res_layout", ["channels_last", "contiguous"])
def test_add_residual_kernel(res_layout):
    torch.manual_seed(0)
    out = (
        torch.randn(1, 64, 3, 10, 12, device = "cuda")
        .half()
        .contiguous(memory_format = torch.channels_last_3d)
    )
    res = torch.randn(1, 64, 3, 10, 12, device = "cuda").half()
    if res_layout == "channels_last":
        res = res.contiguous(memory_format = torch.channels_last_3d)
    ob, rb = torch.randn(64, device = "cuda").half(), torch.randn(64, device = "cuda")
    ref = (
        out.float() + ob.float().view(1, -1, 1, 1, 1) + res.float() + rb.view(1, -1, 1, 1, 1)
    ).half()
    got = H.add_residual(out.clone(memory_format = torch.channels_last_3d), ob, res, rb)
    torch.testing.assert_close(got.float(), ref.float(), rtol = 2e-3, atol = 2e-3)


@needs_cuda
def test_norm_silu_pad_kernel_with_pending_bias():
    torch.manual_seed(0)
    norm = torch.nn.GroupNorm(32, 256, eps = 1e-6).cuda()
    x = (
        torch.randn(1, 256, 5, 17, 23, device = "cuda")
        .half()
        .contiguous(memory_format = torch.channels_last_3d)
    )
    pending = torch.randn(256, device = "cuda")
    ref = H.norm_silu_pad_reference(x, norm, (1, 1, 1, 1), 2, in_bias = pending)
    got = H.norm_silu_pad(x, norm, (1, 1, 1, 1), 2, in_bias = pending)
    torch.testing.assert_close(got.float(), ref.float(), rtol = 1e-2, atol = 1e-2)


def test_int8_decoder_keeps_the_early_blocks_float_by_default(monkeypatch):
    monkeypatch.setattr(H, "H3_VAE_INT8_FLOAT_BLOCKS", 1)
    vae = _tiny_vae()
    blocks = vae.decoder.transformer_blocks
    H._install_int8_decoder(vae)
    assert blocks[0].attn.to_q.weight is not None
    assert blocks[1].attn.to_q.weight is None


def test_a_failing_fused_path_falls_back_to_stock_once_and_logs():
    calls = []

    def fast(self, x):
        calls.append("fast")
        raise RuntimeError("triton: no kernel image")

    def stock(self, x):
        calls.append("stock")
        return x + 1

    warnings = []
    holder = types.SimpleNamespace(
        _unsloth_logger = types.SimpleNamespace(warning = lambda *a: warnings.append(a))
    )
    forward = H._guarded(fast, stock, "decoder")
    assert forward(holder, 1) == 2
    assert forward(holder, 2) == 3
    assert calls == [
        "fast",
        "stock",
        "stock",
    ], "after one failure the fused path must not be retried"
    assert len(warnings) == 1


def test_an_input_the_stock_path_rejects_too_raises_without_disabling_the_fast_path():
    def fast(self, x):
        raise RuntimeError("Padding size should be less than the corresponding input dimension")

    def stock(self, x):
        raise RuntimeError("Padding size should be less than the corresponding input dimension")

    holder = types.SimpleNamespace()
    with pytest.raises(RuntimeError, match = "Padding size"):
        H._guarded(fast, stock, "encoder")(holder, 1)
    assert not getattr(holder, "_unsloth_fast_failed", False)


def test_a_non_reflect_padding_mode_keeps_the_stock_encoder():
    vae = _tiny_vae()
    for m in vae.encoder.modules():
        if hasattr(m, "spatial_padding_mode"):
            m.spatial_padding_mode = "replicate"
    forward = vae.encoder.forward
    assert not H._install_encoder(vae, fp16 = True)
    assert vae.encoder.forward == forward
    assert all(p.dtype is torch.float32 for p in vae.encoder.parameters())


def test_a_non_power_of_two_head_keeps_the_stock_decoder():
    vae = _tiny_vae()
    for block in vae.decoder.transformer_blocks:
        block.attn.dim_head = 12
    assert not H._install_decoder(vae)


def test_an_oom_in_the_fused_path_is_not_masked():
    def fast(self, x):
        raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")

    forward = H._guarded(fast, lambda self, x: x, "decoder")
    holder = types.SimpleNamespace()
    with pytest.raises(RuntimeError, match = "out of memory"):
        forward(holder, 1)
    assert not getattr(holder, "_unsloth_fast_failed", False)


def test_the_fp16_encoder_falls_back_to_the_stock_encoder(monkeypatch):
    vae = _tiny_vae()
    ref_vae = copy.deepcopy(vae)
    assert H._install_encoder(vae, fp16 = True)
    x = torch.randn(1, 3, 9, 24, 40)
    with torch.no_grad():
        fused = vae.encoder(x)

    def boom(*a, **k):
        raise RuntimeError("no triton")

    # the installed guard resolves the fused body at call time, so breaking it exercises the real fallback
    monkeypatch.setattr(H, "_fast_encoder_body", boom)
    with torch.no_grad():
        stock = vae.encoder(x)
        want = ref_vae.encoder(x)
    assert vae.encoder._unsloth_fast_failed
    assert stock.dtype is torch.float32
    # the stock forward over the float16 weights: float16 arithmetic, so close to both, equal to neither
    assert ((stock - want).norm() / want.norm()) < 1e-2
    assert ((stock - fused).norm() / fused.norm()) < 1e-2


def test_tile_batching_retries_one_tile_at_a_time_after_an_oom(monkeypatch):
    vae = _tiny_vae()
    ref = copy.deepcopy(vae)
    assert H._install_tile_batch(vae)
    monkeypatch.setenv(H.H3_VAE_TILE_BATCH_ENV, "4")
    stock_decoder = vae.decoder.forward
    seen = []

    def decoder(z):
        seen.append(z.shape[0])
        if z.shape[0] > 1:
            raise RuntimeError("CUDA out of memory")
        return stock_decoder(z)

    monkeypatch.setattr(vae.decoder, "forward", decoder)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    z = torch.randn(1, 8, 3, 6, 9)
    with torch.no_grad():
        torch.testing.assert_close(vae._decode_clip(z), ref._decode_clip(z), rtol = 1e-5, atol = 1e-5)
    assert seen[0] == 4 and set(seen[1:]) == {1}


def test_settle_drops_a_lever_that_fell_back_from_the_status():
    import dataclasses

    @dataclasses.dataclass(frozen = True)
    class _State:
        speed_optims: tuple

    state = _State(
        ("cudnn_benchmark", "h3_vae_fused_encoder", "h3_vae_fp16_encoder", "h3_vae_fused_decoder")
    )
    pipe = types.SimpleNamespace(
        vae = types.SimpleNamespace(
            encoder = types.SimpleNamespace(_unsloth_fast_failed = False),
            decoder = types.SimpleNamespace(_unsloth_fast_failed = True),
        )
    )
    H.settle_h3_vae_fallback(state, pipe)
    assert state.speed_optims == ("cudnn_benchmark", "h3_vae_fused_encoder", "h3_vae_fp16_encoder")
    H.settle_h3_vae_fallback(state, types.SimpleNamespace())  # no vae: a no-op, never raises


def test_int8_small_row_path_matches_the_int_mm_path():
    holder = torch.nn.Module()
    lin = torch.nn.Linear(64, 32)
    holder.transformer_blocks = torch.nn.ModuleList([torch.nn.ModuleDict({"l": lin})])
    assert H._install_int8_decoder(types.SimpleNamespace(decoder = holder), keep_blocks = ()) > 0
    x = torch.randn(64, 64)
    whole = lin(x)  # 64 rows: torch._int_mm
    sliced = torch.cat(
        [lin(x[i : i + 8]) for i in range(0, 64, 8)]
    )  # 8 rows: the dequantised float path
    assert ((whole - sliced).norm() / whole.norm()) < 0.02


def test_apply_holds_fp16_accumulation_off_outside_the_plan(monkeypatch):
    matmul = torch.backends.cuda.matmul
    if not hasattr(matmul, "allow_fp16_accumulation"):
        pytest.skip("torch without allow_fp16_accumulation")
    monkeypatch.setattr(H, "cuda_fast_path_available", lambda vae = None: True)
    seen = []
    vae = types.SimpleNamespace(decode = lambda z: seen.append(matmul.allow_fp16_accumulation) or z)
    prev = matmul.allow_fp16_accumulation
    try:
        matmul.allow_fp16_accumulation = True
        engaged = H.apply_h3_vae_speedups(
            vae, speed_mode = "default", workflow = "t2va", consumer_gpu = True
        )
        assert H.LEVER_FP16_ACCUM not in engaged
        vae.decode(0)
        assert seen == [False] and matmul.allow_fp16_accumulation is True
    finally:
        matmul.allow_fp16_accumulation = prev


def test_the_fast_path_is_for_the_h3_vae_class_only(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(H, "_kernels", lambda: object())
    monkeypatch.setattr(H, "_triton_version_ok", lambda: True)
    monkeypatch.setattr(H, "_triton_jit_toolchain_ok", lambda: True)

    class AutoencoderKLWan:
        pass

    class AutoencoderKLMiniMaxH3:
        pass

    if getattr(torch.version, "hip", None):
        pytest.skip("ROCm build")
    assert not H.cuda_fast_path_available(AutoencoderKLWan())
    assert H.cuda_fast_path_available(AutoencoderKLMiniMaxH3())


@pytest.mark.parametrize(
    "platform, probe, expected",
    [
        ("win32", lambda: False, False),  # Triton present but its JIT cannot find the CRT headers
        ("win32", lambda: True, True),
        (
            "win32",
            lambda: 1 / 0,
            True,
        ),  # the probe itself failing is not evidence; per-call fallback covers it
        ("linux", lambda: False, True),  # never asked off Windows
    ],
)
def test_the_fast_path_asks_the_windows_triton_toolchain(monkeypatch, platform, probe, expected):
    import core._msvc_env as msvc

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.version, "hip", None, raising = False)
    monkeypatch.setattr(H, "_kernels", lambda: object())
    monkeypatch.setattr(H, "_triton_version_ok", lambda: True)
    monkeypatch.setattr(H.sys, "platform", platform)
    monkeypatch.setattr(msvc, "crt_headers_reachable", probe)
    H._triton_jit_toolchain_ok.cache_clear()
    try:

        class AutoencoderKLMiniMaxH3:
            pass

        assert H.cuda_fast_path_available(AutoencoderKLMiniMaxH3()) is expected
    finally:
        H._triton_jit_toolchain_ok.cache_clear()


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("stock_fallback", [False, True])
def test_the_fp16_encoder_survives_diffusers_casting_the_pixels(device, stock_fallback):
    # diffusers 0.40.0 encode() casts pixels to the encoder dtype and feeds a float32 quant_conv
    get_parameter_dtype = pytest.importorskip("diffusers.models.modeling_utils").get_parameter_dtype

    vae = _tiny_vae().to(device)
    ref_vae = copy.deepcopy(vae)
    assert H._install_encoder(vae, fp16 = True)
    assert get_parameter_dtype(vae.encoder) is torch.float16
    vae.encoder._unsloth_fast_failed = stock_fallback
    x = torch.randn(1, 3, 9, 16, 16, device = device)
    with torch.no_grad():
        got = vae.encode(x.to(get_parameter_dtype(vae.encoder)), return_dict = False)[0].mean
        ref = ref_vae.encode(x, return_dict = False)[0].mean
    assert got.dtype is torch.float32 and vae.quant_conv.weight.dtype is torch.float32
    assert (got - ref).norm() / ref.norm() < 2e-2


def test_the_video_backend_wires_the_layer_into_the_h3_load_only():
    import ast
    import pathlib

    source = (pathlib.Path(H.__file__).parent / "video.py").read_text(encoding = "utf-8")
    tree = ast.parse(source)

    def calls(node, name):
        return [
            c
            for c in ast.walk(node)
            if isinstance(c, ast.Call)
            and (getattr(c.func, "id", None) == name or getattr(c.func, "attr", None) == name)
        ]

    methods = {
        n.name: n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    load = methods["_load_h3_modular_pipeline"]
    (apply_call,) = calls(load, "apply_h3_vae_speedups")
    assert len(calls(tree, "apply_h3_vae_speedups")) == 1, "only the H3 load engages the layer"
    assert {k.arg for k in apply_call.keywords} >= {"speed_mode", "workflow", "logger"}
    assert calls(load, "apply_speed_optims")[0].lineno < apply_call.lineno
    (settle,) = calls(methods["generate"], "settle_h3_vae_fallback")
    (compile_settle,) = calls(methods["generate"], "settle_compile_fallback")
    assert compile_settle.lineno < settle.lineno


@pytest.mark.skipif(H._kernels() is None, reason = "needs Triton")
def test_kernel_annotations_resolve_against_the_module_globals():
    # Triton <= 3.2 resolves kernel annotations in module globals; a closure-local ``tl`` NameErrors there
    k = H._kernels()
    names = [n for n in vars(k) if not n.startswith("_")]
    assert names
    for name in names:
        fn = getattr(getattr(k, name), "fn", None)
        assert fn is not None, name
        for arg, ann in fn.__annotations__.items():
            if isinstance(ann, str):
                eval(ann, fn.__globals__)  # noqa: S307 - a Triton annotation such as "tl.constexpr"


@pytest.mark.parametrize(
    "version, ok",
    [
        ("3.2.0", False),  # compiles, but wrong GroupNorm statistics for a channels-last input
        ("3.1.0", False),
        ("3.3.1", True),
        ("3.6.0", True),
        ("3.8.0.post28", True),  # triton-windows
        ("3.2.0+git35c6c7c6", False),
        ("not a version", False),
    ],
)
def test_the_fast_path_needs_a_verified_triton(monkeypatch, version, ok):
    assert H._triton_version_ok(version) is ok
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.version, "hip", None, raising = False)
    monkeypatch.setattr(H, "_kernels", lambda: object())
    monkeypatch.setattr(H, "_triton_jit_toolchain_ok", lambda: True)
    monkeypatch.setattr(H, "_triton_version_ok", lambda: ok)

    class AutoencoderKLMiniMaxH3:
        pass

    assert H.cuda_fast_path_available(AutoencoderKLMiniMaxH3()) is ok


@pytest.mark.parametrize(
    "before, planned, mid",
    [
        (True, False, False),  # the bf16 model that turned it on unloads mid-decode
        (False, False, True),  # a bf16 model loads and turns it on mid-decode
        (True, True, False),
        (False, True, True),
    ],
)
def test_decode_scope_keeps_a_flag_write_that_lands_mid_decode(before, planned, mid):
    from core.inference import diffusion_speed as S

    matmul = torch.backends.cuda.matmul
    if not hasattr(matmul, "allow_fp16_accumulation"):
        pytest.skip("torch without allow_fp16_accumulation")
    seen = {}

    class _VAE:
        def decode(self, z):
            seen["snapshot"] = S.snapshot_backend_flags()["matmul_fp16_accum"]
            S.restore_backend_flags({"matmul_fp16_accum": mid})
            seen["after_write"] = matmul.allow_fp16_accumulation
            return z

    prev = matmul.allow_fp16_accumulation
    try:
        matmul.allow_fp16_accumulation = before
        vae = _VAE()
        assert H._install_decode_scope(vae, fp16_accum = planned)
        vae.decode(0)
        assert seen["snapshot"] is before, "a snapshot taken mid-decode must see the process value"
        assert seen["after_write"] is planned, "the decode keeps its own value until it ends"
        assert (
            matmul.allow_fp16_accumulation is mid
        ), "the write made during the decode must survive it"
        assert not S._fp16_accum_scopes
    finally:
        matmul.allow_fp16_accumulation = prev


def test_fp16_accumulation_scopes_that_close_out_of_order():
    from core.inference import diffusion_speed as S

    matmul = torch.backends.cuda.matmul
    if not hasattr(matmul, "allow_fp16_accumulation"):
        pytest.skip("torch without allow_fp16_accumulation")
    prev = matmul.allow_fp16_accumulation
    try:
        matmul.allow_fp16_accumulation = True
        a, b = S.fp16_accumulation_scope(False), S.fp16_accumulation_scope(True)
        a.__enter__()
        assert matmul.allow_fp16_accumulation is False
        b.__enter__()
        assert matmul.allow_fp16_accumulation is True
        a.__exit__(None, None, None)
        assert matmul.allow_fp16_accumulation is True, "the scope still open keeps its value"
        b.__exit__(None, None, None)
        assert matmul.allow_fp16_accumulation is True and not S._fp16_accum_scopes
    finally:
        matmul.allow_fp16_accumulation = prev


def test_int8_install_leaves_the_decoder_float_when_quantization_fails(monkeypatch):
    vae = _tiny_vae()
    before = {k: v.clone() for k, v in vae.decoder.state_dict().items()}
    real, calls = H._quantize_int8_weight, []

    def flaky(w, hadamard):
        calls.append(1)
        if len(calls) == 3:
            raise RuntimeError("out of host memory")
        return real(w, hadamard)

    monkeypatch.setattr(H, "_quantize_int8_weight", flaky)
    with pytest.raises(RuntimeError):
        H._install_int8_decoder(vae, keep_blocks = ())
    assert len(calls) == 3
    int8_cls = H._int8_linear_class()
    for module in vae.decoder.modules():
        assert type(module) is not int8_cls
        assert not hasattr(module, "_unsloth_wq") and not hasattr(module, "_unsloth_rot")
    after = vae.decoder.state_dict()
    assert after.keys() == before.keys()
    for k, v in before.items():
        assert torch.equal(after[k], v), k


def test_apply_reports_int8_off_and_keeps_float_weights_when_it_fails(monkeypatch):
    monkeypatch.setattr(H, "cuda_fast_path_available", lambda vae = None: True)
    monkeypatch.setenv(H.H3_VAE_INT8_ENV, "1")

    def boom(w, hadamard):
        raise RuntimeError("out of host memory")

    monkeypatch.setattr(H, "_quantize_int8_weight", boom)
    vae = _tiny_vae()
    engaged = H.apply_h3_vae_speedups(
        vae, speed_mode = "default", workflow = "t2va", consumer_gpu = False
    )
    assert H.LEVER_INT8_DECODER not in engaged
    assert all(
        m.weight is not None for m in vae.decoder.modules() if isinstance(m, torch.nn.Linear)
    )


def test_int8_decoder_rotated_path_stays_close(monkeypatch):
    monkeypatch.setattr(H, "H3_VAE_INT8_ROT_GROUP", 16)
    vae = _tiny_vae()
    fast = copy.deepcopy(vae)
    H._install_int8_decoder(fast, keep_blocks = ())
    lin = fast.decoder.transformer_blocks[0].attn.to_q
    assert lin.weight is None and lin._unsloth_rot.dtype is torch.float16
    z = torch.randn(1, 8, 3, 4, 5)
    with torch.no_grad():
        ref = vae.decoder(z)
        got = fast.decoder(z)
    assert (got - ref).norm() / ref.norm() < 0.05
