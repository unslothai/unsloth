# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""LTX-2.5 (video) on a GPU without native bf16 (T4): the load promotes float16 to float32 and no low-RAM route
serves it, so the 42 GB DiT and 9.2 GB of VAEs / connectors widen to float32 in host RAM (~95 GiB). A Kaggle T4x2
host (31 GB) was OOM-killed mid-load, right after the Gemma 4 worker started. The precision gate refuses such a host
before the eviction, the pull and the worker, with the normal memory-refusal message."""

import pytest
import torch

import core.inference.diffusion_memory as dm
import core.inference.video as v
from core.inference.diffusion_device import DiffusionDeviceTarget
from core.inference.video_families import _FAMILIES as VIDEO_FAMILIES

LTX25 = next(f for f in VIDEO_FAMILIES if f.name == "ltx-2.5")
LTX2 = next(f for f in VIDEO_FAMILIES if f.name == "ltx-2")
WAN = next(f for f in VIDEO_FAMILIES if f.name == "wan2.2-ti2v-5b")
GIB = 1024  # MiB per GiB


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

    def _set(dtype, host_gib):
        monkeypatch.setattr(v, "resolve_diffusion_device_target", lambda *a, **k: _target(dtype))
        monkeypatch.setattr(dm, "_host_ram_capacity_mib", lambda: int(host_gib * GIB))
        monkeypatch.setattr(dm, "_available_system_memory_mib", lambda: int(host_gib * GIB * 0.9))

    return _set


@pytest.mark.parametrize(
    "host_gib, shown", [(31.4, "31.4"), (12.7, "12.7")]
)  # Kaggle T4x2, Colab T4
@pytest.mark.parametrize("quant", [None, "auto", "none", "int8", "fp8"])
def test_t4_class_host_is_refused_before_the_load(host, host_gib, shown, quant):
    host(torch.float16, host_gib)
    with pytest.raises(RuntimeError) as err:
        v.assert_video_precision_available(LTX25, model_kind = "pipeline", transformer_quant = quant)
    msg = str(err.value)
    assert msg.startswith("Not enough system RAM to load this model: LTX-2.5")
    assert "float32" in msg and "GGUF" in msg and "UNSLOTH_DIFFUSION_HOST_RAM_CHECK=0" in msg
    # (42.0 + 9.2) decimal GB x 2 bytes, one decimal like the other host-RAM refusals
    assert "about 95.4 GB of host memory" in msg and f"{shown} GB is available" in msg


@pytest.mark.parametrize(
    "dtype, host_gib, fam, kind",
    [
        (torch.bfloat16, 31, LTX25, "pipeline"),  # bf16 cards keep their routes: unchanged
        (torch.bfloat16, 12, LTX25, "pipeline"),
        (
            torch.float16,
            128,
            LTX25,
            "pipeline",
        ),  # a host that holds the float32 weights is not refused
        (torch.float16, 31, LTX25, "gguf"),  # the GGUF DiT stays quantized
        (torch.float16, 31, LTX2, "pipeline"),  # other video families are untouched
        (torch.float16, 31, WAN, "pipeline"),
    ],
)
def test_other_hosts_and_builds_are_not_refused(host, dtype, host_gib, fam, kind):
    host(dtype, host_gib)
    assert v.ltx25_fp16_host_shortfall(fam, _target(dtype), model_kind = kind) is None
    if fam is LTX25 and kind == "pipeline":
        v.assert_video_precision_available(fam, model_kind = kind)  # unset precision: no refusal


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
    # the small-host test hook caps the reading, so a big host can rehearse the refusal
    host(torch.float16, 512)
    monkeypatch.setenv("UNSLOTH_DIFFUSION_HOST_RAM_LIMIT_MIB", str(30 * GIB))
    msg = v.ltx25_fp16_host_shortfall(LTX25, _target(torch.float16), model_kind = "pipeline")
    assert msg is not None and "30.0 GB is available" in msg


def test_boundary_is_the_widened_size_with_no_reserve(host):
    need_gib = (42.0 + 9.2) * 2 * 1e9 / 1048576 / GIB
    host(torch.float16, need_gib + 0.01)
    assert v.ltx25_fp16_host_shortfall(LTX25, _target(torch.float16), model_kind = "pipeline") is None
    host(torch.float16, need_gib - 0.5)
    assert (
        v.ltx25_fp16_host_shortfall(LTX25, _target(torch.float16), model_kind = "pipeline")
        is not None
    )
