# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Windows ROCm per-tap Conv3d for video VAE decoders: equals F.conv3d, engages only over the size gate, installs only
on Windows ROCm, and the env switch turns it off per call."""

from __future__ import annotations

import ast
import logging
import sys
import types
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

torch = pytest.importorskip("torch")
import torch.nn.functional as F  # noqa: E402

from core.inference import video_vae_rocm_pertap as P  # noqa: E402

# (cin, cout, kernel, stride, padding, dilation, groups, T, H, W)
GEOMETRIES = {
    "ltx_k3_tpad0": (16, 24, (3, 3, 3), (1, 1, 1), (0, 1, 1), (1, 1, 1), 1, 7, 10, 12),
    "ltx_k1_shortcut": (16, 8, (1, 1, 1), (1, 1, 1), (0, 0, 0), (1, 1, 1), 1, 5, 9, 11),
    "k3_pad_all": (8, 12, (3, 3, 3), (1, 1, 1), (1, 1, 1), (1, 1, 1), 1, 6, 8, 8),
    "stride_t2_s2": (8, 8, (3, 3, 3), (2, 2, 2), (1, 1, 1), (1, 1, 1), 1, 9, 12, 10),
    "dilation_t2": (8, 8, (3, 3, 3), (1, 1, 1), (0, 1, 1), (2, 1, 1), 1, 9, 8, 8),
    "dilation_hw2": (8, 8, (3, 3, 3), (1, 1, 1), (0, 2, 2), (1, 2, 2), 1, 5, 9, 9),
    "groups2": (8, 12, (3, 3, 3), (1, 1, 1), (0, 1, 1), (1, 1, 1), 2, 6, 8, 8),
    "k2_t_only": (8, 8, (2, 1, 1), (1, 1, 1), (0, 0, 0), (1, 1, 1), 1, 6, 5, 5),
}


def _conv(
    geom: tuple,
    dtype,
    bias: bool = True,
    padding_mode: str = "zeros",
):
    cin, cout, k, s, p, d, g, *_ = geom
    torch.manual_seed(0)
    return torch.nn.Conv3d(cin, cout, k, s, p, d, g, bias = bias, padding_mode = padding_mode).to(dtype)


def _input(
    geom: tuple,
    dtype,
    batch: int = 1,
):
    cin, *_, T, H, W = geom
    g = torch.Generator().manual_seed(1)
    return torch.randn(batch, cin, T, H, W, generator = g).to(dtype)


@pytest.fixture
def engage_on_cpu(monkeypatch):
    """Every call on CPU takes the per-tap path (the real gate is 2 GiB on a ROCm device)."""
    monkeypatch.setattr(P, "ENGAGE_BYTES", 0)
    monkeypatch.setattr(P, "ENGAGE_DEVICE_TYPES", ("cpu",))
    monkeypatch.delenv(P.PERTAP_ENV, raising = False)


def _patch(conv):
    conv.forward = types.MethodType(P._forward, conv)
    return conv


@pytest.mark.parametrize("name", sorted(GEOMETRIES))
@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("batch", [1, 2])
def test_pertap_matches_conv3d_fp32(name, bias, batch, monkeypatch):
    geom = GEOMETRIES[name]
    conv = _conv(geom, torch.float32, bias)
    x = _input(geom, torch.float32, batch)
    ref = F.conv3d(x, conv.weight, conv.bias, conv.stride, conv.padding, conv.dilation, conv.groups)
    for chunk in (P.CHUNK_BYTES, 1):  # one chunk, then one output frame per chunk
        monkeypatch.setattr(P, "CHUNK_BYTES", chunk)
        with torch.no_grad():
            y = P.pertap_conv3d(
                x, conv.weight, conv.bias, conv.stride, conv.padding, conv.dilation, conv.groups
            )
        assert y.shape == ref.shape and y.dtype == ref.dtype and y.is_contiguous()
        torch.testing.assert_close(y, ref, rtol = 1e-5, atol = 1e-5)


@pytest.mark.parametrize("name", sorted(GEOMETRIES))
def test_pertap_bf16_no_worse_than_stock(name):
    geom = GEOMETRIES[name]
    conv = _conv(geom, torch.bfloat16)
    x = _input(geom, torch.bfloat16)
    w64, b64 = conv.weight.double(), conv.bias.double()
    exact = F.conv3d(x.double(), w64, b64, conv.stride, conv.padding, conv.dilation, conv.groups)
    with torch.no_grad():
        stock = conv(x)
        y = P.pertap_conv3d(
            x, conv.weight, conv.bias, conv.stride, conv.padding, conv.dilation, conv.groups
        )
    assert y.dtype == torch.bfloat16 and y.shape == stock.shape
    err_y = (y.double() - exact).abs().max().item()
    err_stock = (stock.double() - exact).abs().max().item()
    scale = exact.abs().max().item()
    assert err_y <= max(1.5 * err_stock, 1e-2 * scale)
    torch.testing.assert_close(y.float(), stock.float(), rtol = 2e-2, atol = 2e-2 * scale)


@pytest.mark.parametrize("mode", ["reflect", "replicate", "circular"])
def test_patched_forward_non_zero_padding_mode(mode, engage_on_cpu):
    geom = GEOMETRIES["k3_pad_all"]
    conv = _conv(geom, torch.float32, padding_mode = mode)
    x = _input(geom, torch.float32)
    with torch.no_grad():
        ref = conv(x)
        y = _patch(conv)(x)
    assert conv._unsloth_rocm_pertap_calls == 1
    torch.testing.assert_close(y, ref, rtol = 1e-5, atol = 1e-5)


@pytest.mark.parametrize("causal", [True, False])
def test_ltx2_causal_conv_padding(causal, engage_on_cpu):
    """The LTX-2 decoder pads time by repeating edge frames outside the Conv3d (temporal padding 0)."""
    mod = pytest.importorskip("diffusers.models.autoencoders.autoencoder_kl_ltx2")
    torch.manual_seed(0)
    layer = mod.LTX2VideoCausalConv3d(8, 12, 3).eval()
    x = torch.randn(1, 8, 5, 9, 7)
    with torch.no_grad():
        ref = layer(x, causal = causal)
        _patch(layer.conv)
        y = layer(x, causal = causal)
    assert layer.conv._unsloth_rocm_pertap_calls == 1
    torch.testing.assert_close(y, ref, rtol = 1e-5, atol = 1e-5)


def test_size_gate_input_and_output(monkeypatch):
    monkeypatch.setattr(P, "ENGAGE_DEVICE_TYPES", ("cpu",))
    monkeypatch.delenv(P.PERTAP_ENV, raising = False)
    narrow = torch.nn.Conv3d(4, 4, (3, 3, 3), padding = (0, 1, 1))
    wide_out = torch.nn.Conv3d(4, 64, (1, 1, 1))
    x = torch.randn(
        1, 4, 6, 8, 8
    )  # 1536 elements x 4 bytes in; narrow out 4*4*64*4, wide out 64*6*64*4
    in_bytes = x.numel() * 4
    with torch.no_grad():
        monkeypatch.setattr(P, "ENGAGE_BYTES", in_bytes)  # not exceeded: stock
        assert not P._pertap_eligible(narrow, x)
        monkeypatch.setattr(P, "ENGAGE_BYTES", in_bytes - 1)
        assert P._pertap_eligible(narrow, x)
        # output-only crossing: the input is under, the 16x wider output is over
        monkeypatch.setattr(P, "ENGAGE_BYTES", in_bytes * 4)
        assert not P._pertap_eligible(narrow, x)
        assert P._pertap_eligible(wide_out, x)
        # half-precision bytes count, not elements
        monkeypatch.setattr(P, "ENGAGE_BYTES", in_bytes // 2)
        assert not P._pertap_eligible(narrow.to(torch.float16), x.half())


def test_default_gate_is_2gib_and_device_cuda():
    assert P.ENGAGE_BYTES == 2**31
    assert P.ENGAGE_DEVICE_TYPES == ("cuda",)
    conv = torch.nn.Conv3d(4, 4, 3)
    x = torch.randn(1, 4, 4, 4, 4)
    assert not P._pertap_eligible(conv, x)  # CPU tensor never engages by default


def test_gate_counts_calls_and_falls_back(monkeypatch, engage_on_cpu):
    geom = GEOMETRIES["ltx_k3_tpad0"]
    conv = _patch(_conv(geom, torch.float32))
    x = _input(geom, torch.float32)
    with torch.no_grad():
        conv(x)
        monkeypatch.setattr(P, "ENGAGE_BYTES", 2**40)
        conv(x)  # under the gate: stock, not counted
    assert conv._unsloth_rocm_pertap_calls == 1
    # grad-tracking calls keep the stock op (autograd stays the stock graph)
    monkeypatch.setattr(P, "ENGAGE_BYTES", 0)
    y = conv(x)
    assert y.requires_grad and conv._unsloth_rocm_pertap_calls == 1


def test_kill_switch_per_call(monkeypatch, engage_on_cpu):
    geom = GEOMETRIES["ltx_k3_tpad0"]
    conv = _patch(_conv(geom, torch.float32))
    x = _input(geom, torch.float32)
    for value in ("0", "off", "false", "no"):
        monkeypatch.setenv(P.PERTAP_ENV, value)
        with torch.no_grad():
            conv(x)
        assert getattr(conv, "_unsloth_rocm_pertap_calls", 0) == 0
    monkeypatch.setenv(P.PERTAP_ENV, "1")
    with torch.no_grad():
        conv(x)
    assert conv._unsloth_rocm_pertap_calls == 1


# ------------------------------------------------------------------ install / platform gate
class AutoencoderKLLTX2Video(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = torch.nn.Sequential(
            torch.nn.Conv3d(4, 4, 3), torch.nn.ReLU(), torch.nn.Conv3d(4, 4, 1)
        )
        self.encoder = torch.nn.Sequential(torch.nn.Conv3d(4, 4, 3))


class AutoencoderKLWan(AutoencoderKLLTX2Video):
    pass


def _set_platform(monkeypatch, plat: str, hip, version: str):
    monkeypatch.setattr(P.sys, "platform", plat)
    monkeypatch.setattr(torch.version, "hip", hip, raising = False)
    monkeypatch.setattr(torch, "__version__", version, raising = False)
    monkeypatch.delenv(P.PERTAP_ENV, raising = False)


@pytest.mark.parametrize(
    "plat,hip,version,expected",
    [
        ("win32", "7.14.60850", "2.11.0+rocm7.14.0", 2),  # Windows ROCm wheel
        ("win32", None, "2.11.0+rocm7.14.0", 2),  # AMD SDK wheel: tag only in __version__
        ("linux", "7.13.0", "2.11.0+rocm7.13.0", 0),  # Linux ROCm: no cliff measured
        ("win32", None, "2.11.0+cu128", 0),  # Windows CUDA
        ("win32", None, "2.11.0+cpu", 0),  # Windows CPU
        ("darwin", None, "2.11.0", 0),  # macOS / MPS
        ("linux", None, "2.11.0+cu130", 0),  # Linux CUDA
    ],
)
def test_platform_gate(plat, hip, version, expected, monkeypatch):
    _set_platform(monkeypatch, plat, hip, version)
    vae = AutoencoderKLLTX2Video()
    assert P.install(vae) == expected
    patched = [m for m in vae.decoder.modules() if "forward" in m.__dict__]
    assert len(patched) == expected
    assert all("forward" not in m.__dict__ for m in vae.encoder.modules())


def test_install_on_pipe_target_device(monkeypatch):
    _set_platform(monkeypatch, "win32", "7.14", "2.11.0+rocm7.14.0")
    for device, expected in (("cuda", 2), ("cpu", 0), ("mps", 0), ("xpu", 0)):
        pipe = types.SimpleNamespace(vae = AutoencoderKLLTX2Video())
        assert P.install_on_pipe(pipe, types.SimpleNamespace(device = device)) == expected


def test_install_only_ltx2_and_idempotent(monkeypatch, caplog):
    _set_platform(monkeypatch, "win32", "7.14", "2.11.0+rocm7.14.0")
    assert P.install(AutoencoderKLWan()) == 0
    assert P.install(None) == 0
    vae = AutoencoderKLLTX2Video()
    first = vae.decoder[0]
    first.forward = lambda x: x  # something else already owns this conv's forward
    lg = logging.getLogger("pertap_test")
    with caplog.at_level(logging.INFO, logger = "pertap_test"):
        assert P.install(vae, lg) == 1
    assert "video.vae_rocm_pertap" in caplog.text
    assert P.install(vae) == 1
    assert not getattr(first, P._MARK, False)


def test_kill_switch_blocks_install(monkeypatch):
    _set_platform(monkeypatch, "win32", "7.14", "2.11.0+rocm7.14.0")
    monkeypatch.setenv(P.PERTAP_ENV, "0")
    vae = AutoencoderKLLTX2Video()
    assert P.install(vae) == 0
    assert all("forward" not in m.__dict__ for m in vae.modules())


def test_compiled_region_traces_stock_conv(engage_on_cpu):
    """Under torch.compile the patched forward traces the stock Conv3d: one graph, no break, no recompile."""
    geom = GEOMETRIES["ltx_k3_tpad0"]
    conv = _patch(_conv(geom, torch.float32))
    x = _input(geom, torch.float32)
    torch._dynamo.reset()
    compiled = torch.compile(conv, backend = "eager", fullgraph = True)
    with torch.no_grad(), torch._dynamo.config.patch(error_on_recompile = True):
        ref = F.conv3d(x, conv.weight, conv.bias, conv.stride, conv.padding)
        torch.testing.assert_close(compiled(x), ref)
        torch.testing.assert_close(compiled(x), ref)
    assert getattr(conv, "_unsloth_rocm_pertap_calls", 0) == 0
    torch._dynamo.reset()


def test_ltx2_decoder_end_to_end(monkeypatch, engage_on_cpu):
    """A tiny AutoencoderKLLTX2Video with Studio's LTX-2.3 decoder layout decodes the same through the per-tap path."""
    diffusers = pytest.importorskip("diffusers")
    cls = getattr(diffusers, "AutoencoderKLLTX2Video", None)
    if cls is None:
        pytest.skip("diffusers without AutoencoderKLLTX2Video")
    from core.inference.video_ltx2 import _VIDEO_VAE_CONFIG

    cfg = dict(_VIDEO_VAE_CONFIG)
    cfg.update(
        block_out_channels = (8, 16, 32, 32),
        decoder_block_out_channels = (8, 16, 16, 32),
        layers_per_block = (1, 1, 1, 1, 1),
        decoder_layers_per_block = (1, 1, 1, 1, 1),
        latent_channels = 8,
    )
    torch.manual_seed(0)
    try:
        vae = cls(**{k: v for k, v in cfg.items() if not k.startswith("_")}).eval()
    except TypeError as exc:
        pytest.skip(f"diffusers LTX-2 VAE config drift: {exc}")
    z = torch.randn(1, 8, 2, 2, 2)
    with torch.no_grad():
        ref = vae.decode(z, return_dict = False)[0]
        _set_platform(monkeypatch, "win32", "7.14", "2.11.0+rocm7.14.0")
        n = P.install(vae)
        y = vae.decode(z, return_dict = False)[0]
    convs = sum(isinstance(m, torch.nn.Conv3d) for m in vae.decoder.modules())
    assert n == convs > 0
    assert P.pertap_calls(vae) >= convs
    torch.testing.assert_close(y, ref, rtol = 1e-4, atol = 1e-4)


def test_video_backend_installs_after_frame_pad_fix():
    src = (_BACKEND / "core" / "inference" / "video.py").read_text(encoding = "utf-8")
    ast.parse(src)
    pad = src.index("install_frame_pad_fix(pipe, target, logger = logger)")
    pertap = src.index("install_vae_rocm_pertap(pipe, target, logger = logger)")
    assert 0 < pertap - pad < 600
    assert '"vae_rocm_pertap_conv"' in src
