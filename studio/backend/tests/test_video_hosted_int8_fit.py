# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""An explicit int8 LTX-2.5 load on ROCm RDNA that reads the hosted INT8 ConvRot DiT as native W8A8 Linears:

* the source check that commits the hosted file in place of the dense shards does not need torchao (the Windows ROCm
  torchao stub cannot deserialize a torchao checkpoint, but this read never uses torchao);
* the memory plan, and so the unified-memory guard, prices that file instead of the dense bf16 DiT.

Dense loads, AUTO, CUDA, pickle families and the kill switch keep the previous answers."""

from __future__ import annotations

import inspect
import sys
import types
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("diffusers")

BACKEND = Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from core.inference import diffusion_native_hosted as NH  # noqa: E402
from core.inference import diffusion_prequant as PQ  # noqa: E402
from core.inference import diffusion_transformer_quant as tq  # noqa: E402
from core.inference import video as V  # noqa: E402
from core.inference.diffusion_device import DiffusionDeviceTarget  # noqa: E402
from core.inference.diffusion_memory import (  # noqa: E402
    DeviceMemory,
    plan_diffusion_memory,
    unified_memory_shortfall_message,
)
from core.inference.video_families import detect_video_family  # noqa: E402

LTX = ("Lightricks/LTX-2.5-Diffusers", "ltx-2.5")
WAN = ("Wan-AI/Wan2.2-T2V-A14B-Diffusers", "wan2.2-t2v-a14b")
CONVROT = "LTX-2.5-INT8-ConvRot.safetensors"
GPU_BF16 = types.SimpleNamespace(device = "cuda", dtype = torch.bfloat16, ordinal = 0)
ENVS = (
    V.VIDEO_HOSTED_INT8_FIT_ENV,
    tq.ROCM_VIDEO_HOSTED_INT8_ENV,
    NH.NATIVE_HOSTED_ENV,
    "UNSLOTH_NATIVE_INT8_ACT",
    "UNSLOTH_NATIVE_INT8_ROT",
)


def _fam(pair = LTX):
    fam = detect_video_family(pair[0], override = pair[1])
    assert fam is not None
    return fam


@pytest.fixture(autouse = True)
def _clean_env(monkeypatch):
    for env in ENVS:
        monkeypatch.delenv(env, raising = False)


def _host(monkeypatch, *, rocm: bool):
    monkeypatch.setattr(tq, "torch_is_rocm", lambda: rocm)
    monkeypatch.setattr(tq, "is_stubbed", lambda name: False)
    monkeypatch.setattr(tq, "rocm_rdna_w8a8_ready", lambda ordinal = None: True)
    monkeypatch.setattr(tq, "_rdna_compile_tier_proven", lambda target: True)


# ---------------------------------------------------------------------------------------------------------------------
# Source check: the native hosted read needs no torchao.


@pytest.mark.parametrize(
    "pair, scheme, expect",
    [
        (LTX, "int8", True),
        (LTX, "fp8", False),
        (LTX, "auto", False),
        (LTX, None, False),
        (WAN, "int8", False),
    ],
)
def test_native_hosted_read_needs_no_torchao_only_for_convrot_safetensors(pair, scheme, expect):
    # Wan A14B's hosted int8 file is a torchao pickle: reading it needs a real torchao.
    assert V._video_native_hosted_reads_without_torchao(_fam(pair), scheme) is expect


@pytest.mark.parametrize(
    "env, value", [(V.VIDEO_HOSTED_INT8_FIT_ENV, "0"), (NH.NATIVE_HOSTED_ENV, "0")]
)
def test_native_hosted_torchao_free_kill_switches(monkeypatch, env, value):
    monkeypatch.setenv(env, value)
    assert V._video_native_hosted_reads_without_torchao(_fam(), "int8") is False


class _Sibling:
    def __init__(self, name, size):
        self.rfilename = name
        self.size = size


class _Api:
    token = None

    def __init__(self, names):
        self._names = names

    def model_info(
        self,
        repo,
        files_metadata = True,
    ):
        return types.SimpleNamespace(siblings = [_Sibling(n, 25_110_000_000) for n in self._names])


def _verified(
    monkeypatch,
    *,
    native_hosted,
    torchao,
    hub_names = (CONVROT,),
    local = False,
):
    """``_denoiser_prequant_verified`` for LTX-2.5 int8 with torchao readable or not (the Windows ROCm stub)."""
    seen = {}
    monkeypatch.setattr(
        V.VideoBackend, "_denoiser_prequant_covered", staticmethod(lambda *a, **k: True)
    )
    monkeypatch.setattr(
        PQ, "restricted_prequant_load_supported", lambda scheme = None, filename = None: torchao
    )
    real_hub = V.VideoBackend._denoiser_prequant_hub_files

    def _hub(
        fam,
        quant,
        base,
        api,
        h3_task = None,
        readable = None,
    ):
        seen["readable"] = readable
        return real_hub(fam, quant, base, _Api(hub_names), h3_task, readable = readable)

    monkeypatch.setattr(V.VideoBackend, "_denoiser_prequant_hub_files", staticmethod(_hub))

    def _cached(
        fam,
        quant,
        base,
        h3_task = None,
        online = None,
        hf_token = None,
        readable = None,
    ):
        seen["readable"] = readable
        return "unsloth/LTX-2.5-FP8"

    monkeypatch.setattr(V.VideoBackend, "_denoiser_prequant_cached_repo", staticmethod(_cached))
    monkeypatch.setattr(PQ, "hosted_prequant_refusal", lambda *a, **k: None)
    backend = V.VideoBackend.__new__(V.VideoBackend)
    ok = backend._denoiser_prequant_verified(
        _fam(),
        "int8",
        LTX[0],
        None,
        None,
        local_files_only = local,
        native_hosted = native_hosted,
    )
    return ok, seen.get("readable")


def test_stubbed_torchao_still_commits_the_native_hosted_convrot_file(monkeypatch):
    ok, readable = _verified(monkeypatch, native_hosted = True, torchao = False)
    assert ok is True
    # Only the safetensors file the native read opens counts; a torchao pickle twin never does.
    assert readable is not None and readable(CONVROT) and not readable("LTX-2.5-INT8.pt")


def test_stubbed_torchao_offline_uses_the_same_safetensors_filter(monkeypatch):
    ok, readable = _verified(monkeypatch, native_hosted = True, torchao = False, local = True)
    assert ok is True and readable is not None and not readable("transformer_int8.pt")


def test_stubbed_torchao_with_only_a_pickle_hosted_keeps_the_dense_shards(monkeypatch):
    ok, _ = _verified(
        monkeypatch, native_hosted = True, torchao = False, hub_names = ("LTX-2.5-INT8.pt",)
    )
    assert ok is False


def test_stubbed_torchao_seed_check_is_unchanged(monkeypatch):
    # A torchao seed (not the native read) still needs torchao.
    ok, _ = _verified(monkeypatch, native_hosted = False, torchao = False)
    assert ok is False


def test_stubbed_torchao_kill_switch_restores_the_refusal(monkeypatch):
    monkeypatch.setenv(V.VIDEO_HOSTED_INT8_FIT_ENV, "0")
    ok, _ = _verified(monkeypatch, native_hosted = True, torchao = False)
    assert ok is False


def test_real_torchao_keeps_the_existing_filter(monkeypatch):
    ok, readable = _verified(monkeypatch, native_hosted = True, torchao = True)
    assert ok is True and readable is None


# ---------------------------------------------------------------------------------------------------------------------
# Pricing: the planned hosted read is priced at the hosted int8 file.


def _dit_gb(**overrides):
    kwargs = dict(
        target = GPU_BF16,
        transformer_quant = "int8",
        kind = "pipeline",
        native_hosted_planned = "int8",
        source_dir = None,
    )
    kwargs.update(overrides)
    return V._video_hosted_read_dit_gb(_fam(kwargs.pop("pair", LTX)), **kwargs)


def test_rocm_explicit_int8_hosted_read_is_priced_at_the_file(monkeypatch):
    _host(monkeypatch, rocm = True)
    assert _dit_gb() == pytest.approx(25.11)


@pytest.mark.parametrize(
    "overrides",
    [
        {"native_hosted_planned": None},  # dense shards staged: dense price
        {"transformer_quant": "auto"},
        {"transformer_quant": None},
        {"transformer_quant": "fp8"},
        {"kind": "gguf"},
        {"pair": WAN},
    ],
)
def test_everything_else_keeps_the_dense_price(monkeypatch, overrides):
    _host(monkeypatch, rocm = True)
    assert _dit_gb(**overrides) is None


def test_cuda_keeps_the_dense_price(monkeypatch):
    _host(monkeypatch, rocm = False)
    assert _dit_gb() is None


@pytest.mark.parametrize(
    "env", [V.VIDEO_HOSTED_INT8_FIT_ENV, tq.ROCM_VIDEO_HOSTED_INT8_ENV, "UNSLOTH_NATIVE_INT8_ACT"]
)
def test_pricing_kill_switches(monkeypatch, env):
    _host(monkeypatch, rocm = True)
    monkeypatch.setenv(env, "0")
    assert _dit_gb() is None


def test_a_stored_precision_checkpoint_keeps_the_dense_price(monkeypatch):
    _host(monkeypatch, rocm = True)
    monkeypatch.setattr(V, "stored_denoiser_precision", lambda source: "fp8")
    assert _dit_gb(source_dir = "C:/models/ltx") is None


def test_load_pipeline_prices_the_hosted_read_and_reguards_a_miss():
    src = inspect.getsource(V.VideoBackend.load_pipeline)
    assert "denoiser_gb = hosted_read_dit_gb" in src
    assert "_video_hosted_read_dit_gb(" in src
    # A hosted miss (no read, or a read discarded as the wrong build) re-plans dense and re-runs the guard first.
    assert src.count("_reprice_dense_after_hosted_miss(") == 3
    begin = inspect.getsource(V.VideoBackend)
    assert "native_hosted = True," in begin


# ---------------------------------------------------------------------------------------------------------------------
# The guard arithmetic on the failing Windows gfx1151 host (77 GB free reported, 61 GB usable).

MIB_PER_GB = 1000.0**3 / (1024.0 * 1024.0)
GIB = 1024


def _ltx25_plan(memory, dit_gb):
    from core.inference.video_ltx25 import LTX25_WORKER_RESIDENT_GB

    fam = _fam()
    dit, te, vae = fam.bf16_components_gb
    companions = te + vae + LTX25_WORKER_RESIDENT_GB
    return plan_diffusion_memory(
        target = DiffusionDeviceTarget(
            device = "cuda",
            dtype = torch.bfloat16,
            backend = "cuda",
            vendor = "amd",
            supports_model_cpu_offload = True,
            supports_default_torch_compile = True,
            supports_pinned_transfer = True,
        ),
        device_memory = memory,
        model_dense_mib = int((dit_gb + companions) * MIB_PER_GB),
        runtime_headroom_mib = 4 * GIB,
        companion_dense_mib = int(companions * MIB_PER_GB),
        text_encoder_dense_mib = int(te * MIB_PER_GB),
    )


def test_windows_gfx1151_dense_is_refused_and_the_hosted_int8_dit_fits():
    # 80 GB total / 77 GB free reproduces the refusal's "only about 61 GB is usable".
    memory = DeviceMemory("cuda", "cuda", "unified_memory", free_mib = 77 * GIB, total_mib = 80 * GIB)
    dense = _ltx25_plan(memory, _fam().bf16_components_gb[0])
    hosted = _ltx25_plan(memory, 25.11)
    assert round(dense.estimates["safe_device_budget_mib"] / GIB) == 61
    assert "needs about 73 GB" in unified_memory_shortfall_message(dense, family = "ltx-2.5")
    # The guard itself is unchanged: the dense DiT is still refused on this device.
    assert unified_memory_shortfall_message(dense, family = "ltx-2.5") is not None
    assert unified_memory_shortfall_message(hosted, family = "ltx-2.5") is None
