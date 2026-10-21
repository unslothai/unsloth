# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""``_video_vae_half_decode`` on a tiny CPU stand-in for AutoencoderKLWan; capability probe monkeypatched."""

from __future__ import annotations

import logging
import types

import pytest

torch = pytest.importorskip("torch")

from core.inference import diffusion_speed as ds_mod  # noqa: E402
from core.inference.diffusion_speed import SPEED_DEFAULT, SPEED_OFF, apply_speed_optims  # noqa: E402


class _FakeWanVAE(torch.nn.Module):
    """Registration order matches AutoencoderKLWan, so ``dtype`` reads the (fp32) encoder like the real one."""

    def __init__(self, blow_up = False) -> None:
        super().__init__()
        self.encoder = torch.nn.Conv3d(3, 4, 3)
        self.quant_conv = torch.nn.Conv3d(4, 4, 1)
        self.post_quant_conv = torch.nn.Conv3d(4, 4, 1)
        self.decoder = torch.nn.Sequential(torch.nn.Conv3d(4, 4, 3), torch.nn.Conv2d(4, 4, 3))
        self.blow_up = blow_up
        self.seen: list = []

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    def decode(
        self,
        z,
        return_dict = True,
    ):
        weight = self.decoder[0].weight
        self.seen.append((z.dtype, weight.dtype))
        assert z.dtype == weight.dtype, "latents must reach the decoder in its own dtype"
        out = z * 2
        if self.blow_up and weight.dtype == torch.float16:
            out = out * float("nan")
        return (out,) if not return_dict else types.SimpleNamespace(sample = out)


def _pipe(**kw):
    return types.SimpleNamespace(vae = _FakeWanVAE(**kw))


def _target(device = "cuda", backend = "cuda"):
    return types.SimpleNamespace(device = device, backend = backend, dtype = torch.float16, ordinal = None)


def _wan():
    return types.SimpleNamespace(vae_force_fp32 = True, supports_torch_compile = False)


def _snapshot(vae):
    return {k: (v.dtype, v.stride(), v.detach().clone()) for k, v in vae.state_dict().items()}


def _unchanged(vae, snap):
    now = vae.state_dict()
    return (
        all(
            now[k].dtype == d and now[k].stride() == s and torch.equal(now[k], t)
            for k, (d, s, t) in snap.items()
        )
        and "decode" not in vae.__dict__
    )


@pytest.fixture
def sm75(monkeypatch):
    monkeypatch.setattr(ds_mod, "_fp16_compile_capable", lambda target: True)


def test_engages_on_nvidia_for_a_pinned_family(sm75):
    pipe = _pipe()
    assert ds_mod._video_vae_half_decode(pipe, _target(), _wan(), None) is True
    vae = pipe.vae
    for part in (vae.post_quant_conv, vae.decoder):
        assert all(p.dtype == torch.float16 for p in part.parameters())
    assert vae.decoder[0].weight.is_contiguous(memory_format = torch.channels_last_3d)
    assert not vae.decoder[0].weight.is_contiguous()
    assert vae.decoder[1].weight.is_contiguous(memory_format = torch.channels_last)
    assert vae.post_quant_conv.weight.is_contiguous(memory_format = torch.channels_last_3d)
    # Encoder stays fp32 so the pipeline still reads an fp32 vae.dtype.
    assert all(p.dtype == torch.float32 and p.is_contiguous() for p in vae.encoder.parameters())
    assert vae.dtype == torch.float32
    out = vae.decode(torch.randn(1, 4, 2, 5, 5), return_dict = False)[0]
    assert out.dtype == torch.float16 and vae.seen[-1] == (torch.float16, torch.float16)
    assert vae.decode(torch.randn(1, 4, 2, 5, 5)).sample.dtype == torch.float16


def test_second_expert_call_is_idempotent(sm75):
    pipe = _pipe()
    assert ds_mod._video_vae_half_decode(pipe, _target(), _wan(), None) is True
    wrapped = pipe.vae.decode
    assert ds_mod._video_vae_half_decode(pipe, _target(), _wan(), None) is True
    assert pipe.vae.decode is wrapped


@pytest.mark.parametrize(
    "family, target",
    [
        (
            types.SimpleNamespace(vae_force_fp32 = False),
            _target(),
        ),  # LTX-2 / HV1.5 / H3 / image families
        (types.SimpleNamespace(), _target()),
        (_wan(), _target(backend = "rocm")),
        (_wan(), _target(device = "mps", backend = "mps")),
        (_wan(), _target(device = "cpu", backend = "cpu")),
        (_wan(), _target(device = "xpu", backend = "xpu")),
        (_wan(), types.SimpleNamespace(device = "cuda")),  # no backend field: fail closed to fp32
    ],
)
def test_everything_else_keeps_the_fp32_pin(sm75, family, target):
    pipe = _pipe()
    snap = _snapshot(pipe.vae)
    assert ds_mod._video_vae_half_decode(pipe, target, family, None) is False
    assert _unchanged(pipe.vae, snap)


def test_below_sm75_keeps_the_fp32_pin(monkeypatch):
    monkeypatch.setattr(ds_mod, "_fp16_compile_capable", lambda target: False)
    pipe = _pipe()
    snap = _snapshot(pipe.vae)
    assert ds_mod._video_vae_half_decode(pipe, _target(), _wan(), None) is False
    assert _unchanged(pipe.vae, snap)


def test_non_finite_decode_reruns_and_stays_fp32(sm75, caplog):
    pipe = _pipe(blow_up = True)
    logger = logging.getLogger("test_vae_fp16")
    assert ds_mod._video_vae_half_decode(pipe, _target(), _wan(), logger) is True
    z = torch.randn(1, 4, 2, 5, 5)
    with caplog.at_level(logging.WARNING, logger = "test_vae_fp16"):
        out = pipe.vae.decode(z, return_dict = False)[0]
    assert "not finite" in caplog.text
    assert out.dtype == torch.float32 and torch.isfinite(out).all()
    assert torch.equal(out, z * 2)
    assert all(p.dtype == torch.float32 for p in pipe.vae.decoder.parameters())
    pipe.vae.decode(z)
    assert pipe.vae.seen[-1] == (torch.float32, torch.float32)


def test_speed_layer_wires_it_and_off_does_not(monkeypatch):
    calls = []
    monkeypatch.setattr(
        ds_mod,
        "_video_vae_half_decode",
        lambda pipe, target, family, logger: calls.append(family) or True,
    )
    for name in ("_vae_channels_last", "_enable_cudnn_benchmark", "_enable_fp16_accumulation"):
        monkeypatch.setattr(ds_mod, name, lambda *a, **k: False)
    pipe = types.SimpleNamespace(vae = None, transformer = types.SimpleNamespace())
    fam = _wan()
    target = types.SimpleNamespace(
        device = "cpu", dtype = torch.float32, supports_default_torch_compile = False
    )
    assert (
        apply_speed_optims(pipe, target, is_gguf = False, family = fam, speed_mode = SPEED_OFF)[
            "vae_fp16_decode"
        ]
        is False
    )
    assert calls == []
    applied = apply_speed_optims(pipe, target, is_gguf = False, family = fam, speed_mode = SPEED_DEFAULT)
    assert applied["vae_fp16_decode"] is True and calls == [fam]
