# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""AutoencoderKL strips: the KL strip gate, the in-strip upsample with its carried conv bias (exact vs the stock decoder in
fp32 on CPU), and the fused path's Upsample2D."""

import pytest

torch = pytest.importorskip("torch")
diffusers = pytest.importorskip("diffusers")

import torch.nn.functional as TF  # noqa: E402

from core.inference import diffusion_vae_fused as F  # noqa: E402
from core.inference import diffusion_vae_strips as S  # noqa: E402

# reversed per decoder level: identity + upsample, halving after an upsample, general shortcut after an upsample, and
# an upsample in the last block (its bias then goes into conv_norm_out)
CONFIGS = [
    dict(block_out_channels = (32, 64, 128, 128), up_last = False),
    dict(block_out_channels = (16, 64, 64, 128), up_last = False),
    dict(block_out_channels = (32, 32, 64), up_last = True),
]


def _kl(
    block_out_channels,
    up_last = False,
    seed = 0,
):
    torch.manual_seed(seed)
    n = len(block_out_channels)
    vae = diffusers.AutoencoderKL(
        down_block_types = ("DownEncoderBlock2D",) * n,
        up_block_types = ("UpDecoderBlock2D",) * n,
        block_out_channels = block_out_channels,
        latent_channels = 4,
        norm_num_groups = 16,
        layers_per_block = 1,
    ).eval()
    if up_last:  # an upsampler in the last block too (a stage shape the bias carry must end in conv_norm_out)
        from diffusers.models.upsampling import Upsample2D
        c = vae.decoder.up_blocks[-1].resnets[-1].conv2.out_channels
        vae.decoder.up_blocks[-1].upsamplers = torch.nn.ModuleList(
            [Upsample2D(c, use_conv = True, out_channels = c)]
        )
    for p in vae.parameters():
        p.data.normal_(0, 0.2)
    return vae


def _ref_stats(
    x,
    norm,
    in_bias = None,
):
    x = x.double()
    if in_bias is not None:
        x = x + in_bias.double().view(1, -1, 1, 1)
    v = x.reshape(1, norm.num_groups, -1)
    rstd = 1.0 / torch.sqrt(v.var(-1, unbiased = False) + norm.eps)
    return v.mean(-1).float().reshape(-1), rstd.float().reshape(-1)


@pytest.fixture
def torch_stats(monkeypatch):
    monkeypatch.setattr(S, "_stats", _ref_stats)
    monkeypatch.setattr(S._LOCAL, "portable", True, raising = False)


@pytest.mark.parametrize("cfg", CONFIGS)
@pytest.mark.parametrize("strip_elems", [1 << 9, 1 << 24])
@pytest.mark.parametrize("carry", ["1", "0"])
def test_kl_strip_decode_matches_stock_fp32(torch_stats, monkeypatch, cfg, strip_elems, carry):
    monkeypatch.setattr(S, "STRIP_ELEMS", strip_elems)
    monkeypatch.setenv(S.STRIP_UPSAMPLE_ENV, carry)
    vae = _kl(**cfg)
    assert S.decoder_ok(vae.decoder, kernels = False)
    z = torch.randn(1, 4, 7, 9, generator = torch.Generator().manual_seed(1))
    with torch.no_grad():
        ref = vae.decoder(z)
        out = S._decode_one(vae.decoder, z.clone())
    assert out.shape == ref.shape
    assert (out - ref).abs().max().item() < 5e-5


def test_upsample_carries_its_bias_and_writes_the_conv_in_place(torch_stats, monkeypatch):
    monkeypatch.setattr(S, "STRIP_ELEMS", 1 << 9)
    vae = _kl(**CONFIGS[0])
    up = vae.decoder.up_blocks[0].upsamplers[0]
    x = torch.randn(1, up.channels, 5, 6).contiguous(memory_format = torch.channels_last)
    with torch.no_grad():
        ref = up(x)
        out, bias = S._upsample(up, x, defer_bias = True)
        plain, none = S._upsample(up, x, defer_bias = False)
    assert bias is up.conv.bias and none is None
    assert (out + bias.view(1, -1, 1, 1) - ref).abs().max().item() < 1e-5
    assert (plain - ref).abs().max().item() < 1e-5


def test_shortcut_bias_is_the_1x1_of_the_carried_bias():
    sc = torch.nn.Conv2d(8, 4, 1)
    b = torch.randn(8)
    x = torch.randn(1, 8, 3, 3)
    with torch.no_grad():
        ref = sc(x + b.view(1, -1, 1, 1))
        got = TF.conv2d(x, sc.weight, None) + S._shortcut_bias(sc, b).view(1, -1, 1, 1)
    assert (got - ref).abs().max().item() < 1e-5


def _runtime(monkeypatch):
    monkeypatch.setattr(F, "runtime_ok", lambda: True)
    monkeypatch.setattr(S, "_residual_kernel", lambda: object())


def test_kl_strips_install_by_structure_and_switch(monkeypatch):
    _runtime(monkeypatch)
    monkeypatch.delenv(S.STRIPS_ENV, raising = False)
    monkeypatch.delenv(S.KL_STRIPS_ENV, raising = False)
    vae = _kl(**CONFIGS[0])
    assert S.kl_strip_candidate(vae)
    assert S.install(vae) is True  # the card policy decides per decode
    assert not S.engaged(
        vae
    )  # CPU weights: no card that takes them, so the planner prices the fused decode
    monkeypatch.setenv(S.KL_STRIPS_ENV, "1")  # every decode: priced as strips wherever it runs
    assert S.engaged(vae)
    monkeypatch.delenv(S.KL_STRIPS_ENV)
    F.uninstall(vae)
    assert not S.engaged(vae)
    monkeypatch.setenv(S.KL_STRIPS_ENV, "0")
    assert S.install(_kl(**CONFIGS[0])) is False
    monkeypatch.setenv(S.KL_STRIPS_ENV, "1")
    monkeypatch.setenv(S.STRIPS_ENV, "0")  # the strips kill switch covers the KL family too
    assert S.install(_kl(**CONFIGS[0])) is False


def test_kl_strips_skip_a_vae_whose_decode_differs(monkeypatch):
    _runtime(monkeypatch)
    monkeypatch.setenv(S.KL_STRIPS_ENV, "1")

    class Odd(diffusers.AutoencoderKL):
        def _decode(
            self,
            z,
            return_dict = True,
        ):
            return super()._decode(z * 2, return_dict)

    vae = Odd(block_out_channels = (32, 64), norm_num_groups = 16)
    assert not S.kl_strip_candidate(vae)
    assert S.install(vae) is False


def test_kl_policy_by_card(monkeypatch):
    monkeypatch.delenv(S.KL_STRIPS_ENV, raising = False)
    monkeypatch.delenv(S.STRIPS_ENV, raising = False)
    monkeypatch.delenv(S.STRIP_UPSAMPLE_ENV, raising = False)
    S._card_cap.cache_clear()
    caps = {0: (8, 9), 1: (8, 0), 2: (7, 5), 3: (12, 0), 4: (9, 0), 5: (10, 0)}
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda i: caps[i])
    try:
        on = lambda: [S.kl_always_on(torch.device("cuda", i)) for i in range(6)]  # noqa: E731
        assert on() == [
            True,
            True,
            True,
            True,
            False,
            False,
        ]  # H100 / B200 unmeasured: only when the fused would not fit
        monkeypatch.setenv(
            S.STRIP_UPSAMPLE_ENV, "0"
        )  # the older upsample: A100 and RTX PRO 6000 measured slower
        assert on() == [True, False, True, False, False, False]
        monkeypatch.setenv(S.KL_STRIPS_ENV, "1")
        assert all(on())
        monkeypatch.setenv(S.KL_STRIPS_ENV, "0")
        assert not any(on())
    finally:
        S._card_cap.cache_clear()


def test_fused_path_patches_upsample2d_with_a_kill_switch(monkeypatch):
    monkeypatch.setattr(F, "runtime_ok", lambda: True)
    monkeypatch.delenv(F.FUSED_UPSAMPLE_ENV, raising = False)
    vae = _kl(**CONFIGS[0])
    ups = [m for m in vae.decoder.modules() if type(m).__name__ == "Upsample2D"]
    assert ups
    F.install_group_norm_vae(vae)
    assert all(getattr(m.__dict__.get("forward"), "_unsloth_vae_fused", False) for m in ups)
    x = torch.randn(1, ups[0].channels, 4, 5)
    with torch.no_grad():  # off CUDA: the stock forward, unchanged
        assert torch.equal(ups[0](x), type(ups[0]).forward(ups[0], x))
    F.uninstall(vae)
    assert all("forward" not in m.__dict__ for m in ups)
    monkeypatch.setenv(F.FUSED_UPSAMPLE_ENV, "0")
    vae = _kl(**CONFIGS[0])
    F.install_group_norm_vae(vae)
    assert not any(
        "forward" in m.__dict__ for m in vae.decoder.modules() if type(m).__name__ == "Upsample2D"
    )


def _cuda_triton() -> bool:
    return torch.cuda.is_available() and not getattr(torch.version, "hip", None) and F.runtime_ok()


@pytest.mark.skipif(not _cuda_triton(), reason = "needs NVIDIA CUDA + Triton")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_kl_strip_and_upsample_paths_on_cuda(monkeypatch, dtype):
    import copy

    # fp32 means fp32: sm80+ runs fp32 convs / matmuls in TF32 by default, and strips of other shapes pick other algorithms
    monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", False)
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)

    monkeypatch.setattr(S, "STRIP_ELEMS", 1 << 10)
    src = _kl(**CONFIGS[1]).to("cuda", dtype).to(memory_format = torch.channels_last)
    stock, fused, strips = copy.deepcopy(src), copy.deepcopy(src), copy.deepcopy(src)
    monkeypatch.setenv(S.KL_STRIPS_ENV, "0")
    F.install(fused)
    monkeypatch.setenv(S.KL_STRIPS_ENV, "1")
    F.install(strips)
    assert S.engaged(strips) and not S.engaged(fused)
    z = torch.randn(1, 4, 9, 13, device = "cuda", dtype = dtype)
    with torch.no_grad():
        r = stock.decode(z).sample.float()
        a = fused.decode(z).sample.float()
        b = strips.decode(z).sample.float()
        up = stock.decoder.up_blocks[0].upsamplers[0]
        x = torch.randn(1, up.channels, 6, 7, device = "cuda", dtype = dtype).contiguous(
            memory_format = torch.channels_last
        )
        y, _ = F.up_nearest2x(x.unsqueeze(2), False)
    assert torch.equal(y, TF.interpolate(x, scale_factor = 2.0, mode = "nearest"))
    assert strips.decoder._unsloth_vae_strips_calls == 1
    tol = 1e-4 if dtype == torch.float32 else 0.08
    assert (a - r).abs().max().item() < tol and (b - r).abs().max().item() < tol


def test_planner_prices_the_kl_strip_decode_when_it_runs(monkeypatch):
    from core.inference import diffusion_vae_tiling as vt

    vae = _kl(**CONFIGS[0]).to(torch.bfloat16)
    z = torch.zeros(1, 4, 256, 256)
    monkeypatch.setattr(S, "strip_elems", lambda device: 1 << 24)
    vae._unsloth_vae_fused_installed = 5
    assert vt.strips_decode_mib(vae, z, 256, 256) is None  # strips not engaged: the fused figure
    monkeypatch.setattr(S, "engaged", lambda v: v is vae)
    assert vt.strips_decode_mib(vae, z, 256, 256) == pytest.approx(0.048 * 256 * 256 + 128)
    monkeypatch.setattr(
        S, "largest_pass_elems", lambda dec, h, w: 2 * S._REF_PASS_ELEMS_PER_LATENT * h * w
    )
    assert vt.strips_decode_mib(vae, z, 256, 256) == pytest.approx(0.096 * 256 * 256 + 128)


def test_kl_strips_decline_off_their_cards_without_the_memory_reading(monkeypatch):
    """H100 / B200 (outside the measured cards): the KL forward goes straight to the fused decode, never pays the
    free-memory reading, and the planner keeps pricing the fused decode."""
    _runtime(monkeypatch)
    monkeypatch.delenv(S.KL_STRIPS_ENV, raising = False)
    vae = _kl(**CONFIGS[0])
    assert S.install(vae)
    calls = []
    monkeypatch.setattr(S, "fused_would_not_fit", lambda v, z: calls.append(1) or True)
    monkeypatch.setattr(S, "kl_always_on", lambda device: False)
    monkeypatch.setattr(S, "_call_ok", lambda d, z: True)
    monkeypatch.setattr(S, "worth_strips", lambda d, z: True)
    ran = []
    monkeypatch.setattr(S, "_decode_one", lambda d, z: ran.append(1))
    z = torch.randn(1, 4, 8, 8)
    with torch.no_grad():
        out = vae.decoder(z)
    assert out.shape[1] == 3 and not ran and not calls
    assert not S.engaged(vae)


@pytest.mark.skipif(not _cuda_triton(), reason = "needs NVIDIA CUDA + Triton")
def test_strip_upsample_kernel_launch_matches_torch():
    # The strip decoder swallows kernel errors and falls back, so call the launch directly.
    x = torch.randn(1, 64, 12, 20, device = "cuda", dtype = torch.bfloat16).contiguous(
        memory_format = torch.channels_last
    )
    ref = torch.empty(1, 10, 40, 64, device = "cuda", dtype = torch.bfloat16)
    out = torch.empty_like(ref)
    S._up_rows(x, 3, 5, ref, kernel = False)
    S._up_rows(x, 3, 5, out, kernel = True)
    assert torch.equal(ref, out)
