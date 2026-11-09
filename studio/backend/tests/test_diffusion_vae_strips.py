# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""diffusion_vae_strips: strip / halo / interleave bookkeeping against the stock decoder (CPU, the Triton launches
swapped for torch references), structural gating, kill switch, and the fused-vs-strip decode on CUDA."""

import pytest

torch = pytest.importorskip("torch")
diffusers = pytest.importorskip("diffusers")
if not hasattr(diffusers, "AutoencoderKLFlux2"):
    pytest.skip("diffusers without AutoencoderKLFlux2", allow_module_level = True)

import torch.nn.functional as TF  # noqa: E402

from core.inference import diffusion_vae_fused as F  # noqa: E402
from core.inference import diffusion_vae_strips as S  # noqa: E402

CONFIGS = [
    dict(block_out_channels = (32, 64, 128, 128)),  # equal, 128 -> 64 and 64 -> 32 halving in place
    dict(block_out_channels = (16, 64, 64, 128)),  # 64 -> 16: shortcut-buffer transition
    dict(block_out_channels = (32, 32, 64, 64), layers_per_block = 1),
]


def _vae(cfg, seed = 0):
    torch.manual_seed(seed)
    vae = diffusers.AutoencoderKLFlux2(latent_channels = 8, norm_num_groups = 16, **cfg).eval()
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


def _ref_norm_rows(
    x,
    a,
    b,
    norm,
    stats,
    in_bias = None,
):
    _, c, h, w = x.shape
    lo, hi = max(0, a - 1), min(h, b + 1)
    xs = x[:, :, lo:hi].float()
    if in_bias is not None:
        xs = xs + in_bias.float().view(1, -1, 1, 1)
    per = c // norm.num_groups
    mean = stats[0].repeat_interleave(per).view(1, -1, 1, 1)
    rstd = stats[1].repeat_interleave(per).view(1, -1, 1, 1)
    y = (xs - mean) * rstd * norm.weight.float().view(1, -1, 1, 1) + norm.bias.float().view(
        1, -1, 1, 1
    )
    buf, inner = S._padded_rows(c, hi - lo, int(a == 0), int(b == h), w, x)
    inner.copy_(TF.silu(y).to(x.dtype).permute(0, 2, 3, 1))
    return buf.permute(0, 3, 1, 2)


def _ref_add_rows(
    dst,
    y,
    bias,
    scale,
    res = None,
):
    v = (dst if res is None else res).float() + y.float()
    if bias is not None:
        v = v + bias.float().view(1, -1, 1, 1)
    dst.copy_((v / scale).to(dst.dtype))


@pytest.fixture
def torch_kernels(monkeypatch):
    monkeypatch.setattr(S, "_stats", _ref_stats)
    monkeypatch.setattr(S, "_norm_rows", _ref_norm_rows)
    monkeypatch.setattr(S, "_add_rows", _ref_add_rows)


@pytest.mark.parametrize("cfg", CONFIGS)
@pytest.mark.parametrize(
    "size, strip_elems", [((9, 13), 1 << 9), ((5, 17), 1 << 13), ((1, 3), 1 << 24)]
)
def test_strip_decode_matches_stock_fp32(torch_kernels, monkeypatch, cfg, size, strip_elems):
    # 1 << 9 forces one-row strips at every level: halos, the held row of the halving block, edge padding
    monkeypatch.setattr(S, "STRIP_ELEMS", strip_elems)
    vae = _vae(cfg)
    assert S.decoder_ok(vae.decoder)
    z = torch.randn(1, 8, *size, generator = torch.Generator().manual_seed(1))
    with torch.no_grad():
        ref = vae.decoder(z)
        out = S._decode_one(vae.decoder, z.clone())
    assert out.shape == ref.shape
    assert (out - ref).abs().max().item() < 2e-5


def test_halving_block_writes_inside_its_input(torch_kernels, monkeypatch):
    monkeypatch.setattr(S, "STRIP_ELEMS", 1 << 10)
    vae = _vae(CONFIGS[0])
    block = vae.decoder.up_blocks[2].resnets[0]
    assert block.conv1.in_channels == 2 * block.conv1.out_channels
    x = torch.randn(1, 128, 7, 11).contiguous(memory_format = torch.channels_last)
    with torch.no_grad():
        ref = block(x.clone(), None)
        out, hid = S._resnet(block, x, None)
    assert hid.shape == (1, 64, 7, 11)
    assert out.untyped_storage().data_ptr() == x.untyped_storage().data_ptr()
    assert (out - ref).abs().max().item() < 2e-5


def test_decoder_ok_rejects_unknown_shapes():
    vae = _vae(CONFIGS[0])
    assert S.decoder_ok(vae.decoder)
    vae.decoder.up_blocks[1].resnets[0].conv1.padding = (2, 2)
    assert not S.decoder_ok(vae.decoder)
    vae = _vae(CONFIGS[0])
    vae.decoder.up_blocks[0].upsamplers[0].interpolate = False
    assert not S.decoder_ok(vae.decoder)
    vae = _vae(CONFIGS[0])
    vae.decoder.up_blocks[0].resnets[
        0
    ]._hf_hook = object()  # accelerate offload: weights may not be on the card
    assert not S.decoder_ok(vae.decoder)


def test_install_gated_off_cpu_and_by_env(monkeypatch):
    vae = _vae(CONFIGS[0])
    monkeypatch.setattr(F, "runtime_ok", lambda: False)
    assert S.install(vae) is False
    monkeypatch.setattr(F, "runtime_ok", lambda: True)
    monkeypatch.setenv(S.STRIPS_ENV, "0")
    assert S.install(vae) is False
    assert "forward" not in vae.decoder.__dict__
    other = diffusers.AutoencoderKL(block_out_channels = (32, 64), norm_num_groups = 16)
    monkeypatch.delenv(S.STRIPS_ENV)
    assert S.install(other) is False


def _cuda_triton() -> bool:
    return torch.cuda.is_available() and not getattr(torch.version, "hip", None) and F.runtime_ok()


@pytest.mark.skipif(not _cuda_triton(), reason = "needs NVIDIA CUDA + Triton")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_strip_decode_matches_fused_cuda(monkeypatch, dtype):
    import copy

    monkeypatch.setattr(S, "STRIP_ELEMS", 1 << 10)
    src = _vae(CONFIGS[0]).to("cuda", dtype)
    fused, strips = copy.deepcopy(src), copy.deepcopy(src)
    monkeypatch.setenv(S.STRIPS_ENV, "0")
    F.install(fused)
    monkeypatch.delenv(S.STRIPS_ENV)
    monkeypatch.setenv(S.STRIPS_ENV, "1")  # every decode, whatever the card
    F.install(strips)
    assert S.engaged(strips) and not S.engaged(fused)
    z = torch.randn(1, 8, 9, 13, device = "cuda", dtype = dtype)
    with torch.no_grad():
        a = fused.decode(z).sample.float()
        b = strips.decode(z).sample.float()
    assert strips.decoder._unsloth_vae_strips_calls == 1
    tol = 1e-4 if dtype == torch.float32 else 0.08
    assert (a - b).abs().max().item() < tol
    F.uninstall(strips)
    assert "forward" not in strips.decoder.__dict__ and not S.engaged(strips)


def test_planner_prices_the_strip_decode_only_when_it_runs(monkeypatch):
    from core.inference import diffusion_vae_tiling as vt

    vae = _vae(CONFIGS[0]).to(torch.bfloat16)
    z = torch.zeros(1, 8, 256, 256)
    monkeypatch.setattr(S, "strip_elems", lambda device: 1 << 24)
    assert vt.strips_decode_mib(vae, z, 256, 256) is None  # not installed
    vae._unsloth_vae_fused_installed = 5
    monkeypatch.setattr(S, "engaged", lambda v: v is vae)
    big = vt.strips_decode_mib(vae, z, 256, 256)
    assert big == pytest.approx(0.048 * 256 * 256 + 128)
    # largest pass in one strip: the fused decode runs, so the fused figure prices it
    assert vt.strips_decode_mib(vae, z, 16, 16) is None
    assert vt._strips_tile_area(vae, z, big) == pytest.approx(256 * 256, rel = 1e-3)
    assert vt._strips_tile_area(vae, z, 100.0) is None
    vae._unsloth_vae_fused_failed = True
    assert vt.strips_decode_mib(vae, z, 256, 256) is None
    del vae._unsloth_vae_fused_failed
    vae.to(torch.float32)  # fp32 keeps the fused pricing
    assert vt.strips_decode_mib(vae, z, 256, 256) is None


def test_auto_policy_by_card_and_free_vram(monkeypatch):
    monkeypatch.delenv(S.STRIPS_ENV, raising = False)
    S._card_always.cache_clear()
    caps = {0: (8, 9), 1: (8, 0), 2: (7, 5), 3: (9, 0)}
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda i: caps[i])
    on = [S.always_on(torch.device("cuda", i)) for i in range(4)]
    S._card_always.cache_clear()
    assert on == [True, False, True, False]
    monkeypatch.setenv(S.STRIPS_ENV, "1")
    assert S.always_on(torch.device("cuda", 1))
    S._card_always.cache_clear()

    from core.inference import diffusion_vae_tiling as vt

    vae = _vae(CONFIGS[0])
    z = torch.zeros(1, 8, 128, 128)
    monkeypatch.setattr(vt, "_free_mib", lambda v, zz: (100.0, 1.0))
    assert S.fused_would_not_fit(vae, z)  # 0.09 * 128 * 128 MiB > 75 MiB
    monkeypatch.setattr(vt, "_free_mib", lambda v, zz: (1e6, 1.0))
    assert not S.fused_would_not_fit(vae, z)
    monkeypatch.setattr(vt, "_free_mib", lambda v, zz: None)  # off CUDA
    assert not S.fused_would_not_fit(vae, z)
