# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""LTX-2.5 on an fp16-only GPU widens to float32 in host RAM; small hosts are refused before the load."""

import pytest
import torch

import core.inference.diffusion_memory as dm
import core.inference.diffusion_te_prequant as tp
import core.inference.video as v
from core.inference.diffusion_device import DiffusionDeviceTarget
from core.inference.video_families import _FAMILIES as VIDEO_FAMILIES

LTX25 = next(f for f in VIDEO_FAMILIES if f.name == "ltx-2.5")
LTX2 = next(f for f in VIDEO_FAMILIES if f.name == "ltx-2")
WAN = next(f for f in VIDEO_FAMILIES if f.name == "wan2.2-ti2v-5b")
GIB = 1024
# float32 GiB: pipeline, dense encoder, hosted int8 encoder (dense tensors widened)
_GIB = 1e9 / 1024**3
PIPE_GIB = (42.0 + 9.2) * 2 * _GIB
DENSE_TE_GIB = 23.9 * 2 * _GIB
INT8_TE_GIB = (13.1 + (2 * 13.1 - 23.9)) * _GIB


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
    monkeypatch.delenv("UNSLOTH_DIFFUSION_SMALL_HOST", raising = False)
    monkeypatch.delenv("UNSLOTH_DIFFUSION_HOST_RAM_CHECK", raising = False)
    monkeypatch.delenv("UNSLOTH_DIFFUSION_HOST_RAM_LIMIT_MIB", raising = False)

    # an automatic pick takes the hosted int8 encoder only when the Hub confirms it; never ask the Hub here
    monkeypatch.setattr(tp, "te_prequant_sources_confirmed", lambda *a, **k: False)

    def _set(dtype, host_gib):
        monkeypatch.setattr(v, "resolve_diffusion_device_target", lambda *a, **k: _target(dtype))
        monkeypatch.setattr(dm, "_host_ram_capacity_mib", lambda: int(host_gib * GIB))
        monkeypatch.setattr(dm, "_available_system_memory_mib", lambda: int(host_gib * GIB * 0.9))

    return _set


@pytest.mark.parametrize("host_gib, shown", [(31.4, "31.4"), (12.7, "12.7")])
@pytest.mark.parametrize("quant", [None, "auto", "none", "int8", "fp8"])
def test_t4_class_host_is_refused_before_the_load(host, host_gib, shown, quant):
    host(torch.float16, host_gib)
    with pytest.raises(RuntimeError) as err:
        v.assert_video_precision_available(LTX25, model_kind = "pipeline", transformer_quant = quant)
    msg = str(err.value)
    assert msg.startswith("Not enough system RAM to load this model: LTX-2.5")
    assert "float32" in msg and "GGUF" in msg and "UNSLOTH_DIFFUSION_HOST_RAM_CHECK=0" in msg
    assert f"about {PIPE_GIB + DENSE_TE_GIB:.1f} GB of host memory" in msg
    assert f"{shown} GB is available" in msg


@pytest.mark.parametrize(
    "dtype, host_gib, fam, kind, te",
    [
        (torch.bfloat16, 31, LTX25, "pipeline", None),
        (torch.bfloat16, 12, LTX25, "pipeline", None),
        (torch.float16, 128, LTX25, "pipeline", "int8"),
        (torch.float16, 160, LTX25, "pipeline", "none"),
        (torch.float16, 31, LTX25, "gguf", None),
        (torch.float16, 31, LTX2, "pipeline", None),
        (torch.float16, 31, WAN, "pipeline", None),
    ],
)
def test_other_hosts_and_builds_are_not_refused(host, dtype, host_gib, fam, kind, te):
    host(dtype, host_gib)
    assert (
        v.ltx25_fp16_host_shortfall(fam, _target(dtype), model_kind = kind, text_encoder_quant = te)
        is None
    )
    if fam is LTX25 and kind == "pipeline":
        v.assert_video_precision_available(fam, model_kind = kind, text_encoder_quant = te)


@pytest.mark.parametrize("te", [None, "auto", "none"])
def test_96_gib_host_with_the_dense_encoder_is_refused(host, te):
    """The worker reads the bf16 encoder at float32 (~44.5 GiB) into the host RAM the DiT also widens into."""
    host(torch.float16, 96)
    with pytest.raises(RuntimeError) as err:
        v.assert_video_precision_available(LTX25, model_kind = "pipeline", text_encoder_quant = te)
    msg = str(err.value)
    assert f"about {PIPE_GIB + DENSE_TE_GIB:.1f} GB of host memory" in msg
    assert f"text encoder {DENSE_TE_GIB:.1f}" in msg and "96.0 GB is available" in msg


def test_int8_encoder_is_priced_at_its_hosted_size(host):
    need = PIPE_GIB + INT8_TE_GIB
    host(torch.float16, 96)
    msg = v.ltx25_fp16_host_shortfall(
        LTX25, _target(torch.float16), model_kind = "pipeline", text_encoder_quant = "int8"
    )
    assert msg is not None and f"about {need:.1f} GB" in msg
    host(torch.float16, need + 0.01)
    assert (
        v.ltx25_fp16_host_shortfall(
            LTX25, _target(torch.float16), model_kind = "pipeline", text_encoder_quant = "int8"
        )
        is None
    )
    # a custom pack has no hosted encoder: int8 is quantized from the dense float32 read
    assert (
        v.ltx25_fp16_host_shortfall(
            LTX25,
            _target(torch.float16),
            model_kind = "pipeline",
            text_encoder_quant = "int8",
            base = "someone/custom-ltx-2.5-pack",
        )
        is not None
    )


def test_auto_takes_the_int8_price_only_when_the_hosted_file_is_confirmed(host, monkeypatch):
    host(torch.float16, 128)
    shortfall = lambda: v.ltx25_fp16_host_shortfall(  # noqa: E731
        LTX25, _target(torch.float16), model_kind = "pipeline", text_encoder_quant = "auto"
    )
    assert shortfall() is not None
    monkeypatch.setattr(tp, "te_prequant_sources_confirmed", lambda *a, **k: True)
    assert shortfall() is None


@pytest.mark.parametrize(
    "env", ["UNSLOTH_DIFFUSION_SMALL_HOST", "UNSLOTH_DIFFUSION_HOST_RAM_CHECK"]
)
def test_kill_switches_restore_the_old_load(host, monkeypatch, env):
    host(torch.float16, 31)
    monkeypatch.setenv(env, "0")
    assert v.ltx25_fp16_host_shortfall(LTX25, _target(torch.float16), model_kind = "pipeline") is None
    v.assert_video_precision_available(LTX25, model_kind = "pipeline")


def test_unreadable_host_ram_refuses_nothing(host, monkeypatch):
    host(torch.float16, 31)
    monkeypatch.setattr(dm, "_host_ram_capacity_mib", lambda: None)
    assert v.ltx25_fp16_host_shortfall(LTX25, _target(torch.float16), model_kind = "pipeline") is None


def test_test_cap_caps_a_large_host(host, monkeypatch):
    host(torch.float16, 512)
    monkeypatch.setenv("UNSLOTH_DIFFUSION_HOST_RAM_LIMIT_MIB", str(30 * GIB))
    msg = v.ltx25_fp16_host_shortfall(LTX25, _target(torch.float16), model_kind = "pipeline")
    assert msg is not None and "30.0 GB is available" in msg


def test_boundary_is_the_widened_size_with_no_reserve(host):
    need_gib = PIPE_GIB + DENSE_TE_GIB
    host(torch.float16, need_gib + 0.01)
    assert v.ltx25_fp16_host_shortfall(LTX25, _target(torch.float16), model_kind = "pipeline") is None
    host(torch.float16, need_gib - 0.5)
    assert (
        v.ltx25_fp16_host_shortfall(LTX25, _target(torch.float16), model_kind = "pipeline")
        is not None
    )
