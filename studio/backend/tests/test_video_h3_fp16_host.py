# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""MiniMax-H3 (video) on a GPU without native bf16 (T4): the hosted INT8 / FP8 parts never engage there, so every
precision lands on the released weights promoted to float32 (~288 GB). The precision gate refuses a host that cannot
hold them before the eviction and the 144 GB pull, instead of downloading and then exhausting host RAM."""

import pytest
import torch

import core.inference.diffusion_memory as dm
import core.inference.video as v
from core.inference.diffusion_device import DiffusionDeviceTarget
from core.inference.video_families import _FAMILIES as VIDEO_FAMILIES

H3 = next(f for f in VIDEO_FAMILIES if f.name == "minimax-h3")
WAN = next(f for f in VIDEO_FAMILIES if f.name == "wan2.2-ti2v-5b")
GB = 1e9 / 1048576  # MiB per decimal GB


def _target(dtype):
    return DiffusionDeviceTarget(
        device = "cuda",
        dtype = dtype,
        backend = "cuda",
        vendor = "nvidia",
        supports_model_cpu_offload = True,
        supports_default_torch_compile = True,
        supports_pinned_transfer = True,
    )


@pytest.fixture
def host(monkeypatch):
    def _set(dtype, host_gb):
        monkeypatch.setattr(v, "resolve_diffusion_device_target", lambda *a, **k: _target(dtype))
        monkeypatch.setattr(dm, "_host_ram_capacity_mib", lambda: int(host_gb * GB))
        monkeypatch.setattr(v, "effective_te_quant", lambda *a, **k: None)

    return _set


@pytest.mark.parametrize("quant", [None, "auto", "none", "int8", "fp8"])
def test_t4_class_host_is_refused_before_the_pull(host, quant):
    host(torch.float16, 51)  # Colab T4 high-RAM
    with pytest.raises(RuntimeError) as err:
        v.assert_video_precision_available(H3, model_kind = "pipeline", transformer_quant = quant)
    msg = str(err.value)
    assert "no native bfloat16" in msg and "GGUF" in msg
    assert "about 288 GB" in msg and "float32" in msg and "has 51 GB" in msg


def test_shortfall_reads_the_family_sizes():
    msg = v.h3_fp16_host_shortfall(H3, _target(torch.float16), model_kind = "pipeline")
    # Needs the host reading; with this machine's RAM it is either None (>= 288 GB) or the refusal.
    assert msg is None or "denoiser 132.6, text encoder 133.6, VAEs 22.2" in msg


@pytest.mark.parametrize(
    "dtype, host_gb, fam, kind",
    [
        (torch.bfloat16, 51, H3, "pipeline"),  # bf16 cards keep the hosted parts: unchanged
        (
            torch.float16,
            400,
            H3,
            "pipeline",
        ),  # a host that holds the float32 weights is not refused
        (torch.float16, 51, H3, "gguf"),  # the GGUF build runs on the native engine
        (torch.float16, 51, WAN, "pipeline"),  # other video families are untouched
    ],
)
def test_other_hosts_and_builds_are_not_refused(host, dtype, host_gb, fam, kind):
    host(dtype, host_gb)
    assert v.h3_fp16_host_shortfall(fam, _target(dtype), model_kind = kind) is None
    if kind == "pipeline" and fam is H3:
        v.assert_video_precision_available(fam, model_kind = kind)  # unset precision: no refusal


def test_unreadable_host_ram_refuses_nothing(host, monkeypatch):
    host(torch.float16, 51)
    monkeypatch.setattr(dm, "_host_ram_capacity_mib", lambda: None)
    assert v.h3_fp16_host_shortfall(H3, _target(torch.float16), model_kind = "pipeline") is None


@pytest.mark.parametrize("host_gb, refused", [(200, False), (256, False), (180, True)])
def test_explicit_int8_conditioner_is_priced_at_its_hosted_size(host, host_gb, refused):
    # An explicit hosted int8 conditioner engages without bf16 (ConvRot dequantizes, then F.linear) and stays int8:
    # 132.6 + 27.2 + 22.2 = 182 GB, not the 288 GB dense float32 sum, so a 200-256 GB T4 / V100 host still loads.
    host(torch.float16, host_gb)
    if not refused:
        v.assert_video_precision_available(H3, model_kind = "pipeline", text_encoder_quant = "int8")
        return
    with pytest.raises(RuntimeError) as err:
        v.assert_video_precision_available(H3, model_kind = "pipeline", text_encoder_quant = "int8")
    assert "about 182 GB" in str(err.value) and "text encoder 27.2" in str(err.value)
