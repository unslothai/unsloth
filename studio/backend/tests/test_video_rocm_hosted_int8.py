# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""An explicit int8 LTX-2.5 pick on ROCm RDNA reads the hosted INT8 ConvRot DiT as native W8A8 Linears instead of
quantising the dense shards weight-only at load. AUTO, CUDA and the kill switches keep the previous routes."""

from __future__ import annotations

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
from core.inference import diffusion_transformer_quant as tq  # noqa: E402
from core.inference import video as V  # noqa: E402
from core.inference.diffusion_memory import OFFLOAD_GROUP, OFFLOAD_NONE  # noqa: E402
from core.inference.video_families import detect_video_family  # noqa: E402

LTX = ("Lightricks/LTX-2.5-Diffusers", "ltx-2.5")
WAN = ("Wan-AI/Wan2.2-T2V-A14B-Diffusers", "wan2.2-t2v-a14b")
TI2V = ("Wan-AI/Wan2.2-TI2V-5B-Diffusers", "wan2.2-ti2v-5b")
GPU_BF16 = types.SimpleNamespace(device = "cuda", dtype = torch.bfloat16, ordinal = 0)
RESIDENT = types.SimpleNamespace(offload_policy = OFFLOAD_NONE)
STREAMED = types.SimpleNamespace(offload_policy = OFFLOAD_GROUP, stream_transformer = True)
ENVS = (
    tq.ROCM_VIDEO_HOSTED_INT8_ENV,
    NH.NATIVE_HOSTED_ENV,
    "UNSLOTH_NATIVE_INT8_ACT",
    "UNSLOTH_NATIVE_INT8_ROT",
)


def _fam(pair = LTX):
    fam = detect_video_family(pair[0], override = pair[1])
    assert fam is not None
    return fam


def _host(
    monkeypatch,
    *,
    rocm: bool,
    ready: bool = True,
    proven: bool = True,
):
    monkeypatch.setattr(tq, "torch_is_rocm", lambda: rocm)
    monkeypatch.setattr(tq, "is_stubbed", lambda name: False)
    monkeypatch.setattr(tq, "rocm_rdna_w8a8_ready", lambda ordinal = None: ready)
    monkeypatch.setattr(tq, "_rdna_compile_tier_proven", lambda target: proven)
    for env in ENVS:
        monkeypatch.delenv(env, raising = False)


@pytest.fixture
def rocm(monkeypatch):
    _host(monkeypatch, rocm = True)


@pytest.fixture
def nvidia(monkeypatch):
    _host(monkeypatch, rocm = False)


def test_ltx25_carries_the_rdna_explicit_w8a8_evidence():
    assert "16.17 vs 18.89 s/clip" in str(tq._FAMILY_RDNA_EXPLICIT_W8A8.get("ltx-2.5"))
    # AUTO is untouched: no AUTO W8A8 row, so gfx1151 AUTO keeps bf16.
    assert "ltx-2.5" not in tq._FAMILY_RDNA_AUTO_W8A8


@pytest.mark.parametrize(
    "requested, expect",
    [("int8", "int8"), ("auto", None), (None, None), ("fp8", None), ("off", None)],
)
def test_rocm_gate_takes_explicit_int8_only(rocm, requested, expect):
    assert V._video_rocm_hosted_scheme(_fam(), requested, GPU_BF16) == expect


@pytest.mark.parametrize(
    "env, value",
    [
        (tq.ROCM_VIDEO_HOSTED_INT8_ENV, "0"),
        (tq.ROCM_VIDEO_HOSTED_INT8_ENV, "off"),
        ("UNSLOTH_NATIVE_INT8_ACT", "0"),
        (NH.NATIVE_HOSTED_ENV, "0"),
        ("UNSLOTH_NATIVE_INT8_ROT", "0"),
    ],
)
def test_rocm_kill_switches(rocm, monkeypatch, env, value):
    monkeypatch.setenv(env, value)
    assert V._video_rocm_hosted_scheme(_fam(), "int8", GPU_BF16) is None


def test_rocm_kill_switch_restores_weight_only(rocm, monkeypatch):
    assert V._video_int8_act(GPU_BF16, _fam(), "int8") is True
    monkeypatch.setenv(tq.ROCM_VIDEO_HOSTED_INT8_ENV, "0")
    assert V._video_int8_act(GPU_BF16, _fam(), "int8") is False


@pytest.mark.parametrize("ready, proven", [(False, True), (True, False)])
def test_rocm_without_the_measured_w8a8_tier_keeps_weight_only(monkeypatch, ready, proven):
    _host(monkeypatch, rocm = True, ready = ready, proven = proven)
    assert V._video_rocm_hosted_scheme(_fam(), "int8", GPU_BF16) is None
    assert V._video_int8_act(GPU_BF16, _fam(), "int8") is False


@pytest.mark.parametrize("pair", [WAN, TI2V])
def test_rocm_other_video_families_are_unchanged(rocm, pair):
    # Wan A14B's hosted file is a torchao pickle; TI2V-5B has no hosted native file and no W8A8 evidence.
    assert V._video_rocm_hosted_scheme(_fam(pair), "int8", GPU_BF16) is None
    assert V._video_int8_act(GPU_BF16, _fam(pair), "int8") is False


@pytest.mark.parametrize("requested", ["int8", "auto", None, "fp8"])
@pytest.mark.parametrize("act", [None, "0", "1"])
def test_cuda_is_unchanged(nvidia, monkeypatch, requested, act):
    if act is not None:
        monkeypatch.setenv("UNSLOTH_NATIVE_INT8_ACT", act)
    assert V._video_rocm_hosted_scheme(_fam(), requested, GPU_BF16) is None
    assert V._video_int8_act(GPU_BF16, _fam(), requested) is tq.native_int8_act(GPU_BF16)


def test_rocm_auto_and_weight_only_defaults_are_unchanged(rocm):
    assert tq.native_int8_act(GPU_BF16) is False
    assert V._video_int8_act(GPU_BF16, _fam(), "auto") is False
    assert V._video_int8_act(GPU_BF16, _fam(), None) is False


def test_resident_spec_is_the_hosted_w8a8_build(rocm):
    spec = V._video_rocm_hosted_spec(
        _fam(),
        target = GPU_BF16,
        transformer_quant = "int8",
        plan = RESIDENT,
        kind = "pipeline",
        source_dir = None,
    )
    assert spec is not None
    assert spec["scheme"] == "int8" and spec["act_int8"] is True and spec["rot_group"] == 256
    # An offloaded plan keeps the offload spec, which now also asks for W8A8 on this pick.
    assert (
        V._video_rocm_hosted_spec(
            _fam(),
            target = GPU_BF16,
            transformer_quant = "int8",
            plan = STREAMED,
            kind = "pipeline",
            source_dir = None,
        )
        is None
    )
    off = V._video_native_cache_spec(
        _fam(),
        target = GPU_BF16,
        transformer_quant = "int8",
        plan = STREAMED,
        kind = "pipeline",
        source_dir = None,
    )
    assert off is not None and off["act_int8"] is True and spec == off
    for quant in ("auto", None):
        assert (
            V._video_rocm_hosted_spec(
                _fam(),
                target = GPU_BF16,
                transformer_quant = quant,
                plan = RESIDENT,
                kind = "pipeline",
                source_dir = None,
            )
            is None
        )


def test_hosted_metadata_check_accepts_the_rocm_spec_and_refuses_weight_only(rocm):
    from core.inference.diffusion_convrot import (
        CONVROT_KIND,
        ROTATION_FQNS_KEY,
        ROTATION_GROUP_KEY,
        ROTATION_KEY,
    )

    spec = V._video_rocm_hosted_spec(
        _fam(),
        target = GPU_BF16,
        transformer_quant = "int8",
        plan = RESIDENT,
        kind = "pipeline",
        source_dir = None,
    )
    meta = {
        "scheme": "int8",
        "quant_backend": "torchao",
        "family": "ltx-2.5",
        "component": "transformer",
        "torch_dtype": "bfloat16",
        ROTATION_KEY: CONVROT_KIND,
        ROTATION_GROUP_KEY: 256,
        ROTATION_FQNS_KEY: [],
        "min_features": spec["min_features"],
        "exclude_name_tokens": [t for t in spec["exclude_name_tokens"] if t != "lora_"],
    }
    assert NH.metadata_refusal(meta, spec, family = "ltx-2.5", component = "transformer") is None
    weight_only = dict(spec, act_int8 = False, rot_group = 0)
    assert NH.metadata_refusal(meta, weight_only, family = "ltx-2.5", component = "transformer")


def _plan(monkeypatch, requested):
    monkeypatch.setattr(V, "resolve_diffusion_device_target", lambda **k: GPU_BF16)
    backend = V.VideoBackend.__new__(V.VideoBackend)
    backend._state = None
    out: list = []
    got = backend._video_planned_auto_denoiser_scheme(
        _fam(),
        base = LTX[0],
        kind = "pipeline",
        transformer_quant = requested,
        speed_mode = None,
        native_hosted = out,
    )
    return got, out


def test_rocm_planner_stages_the_hosted_file_for_explicit_int8(rocm, monkeypatch):
    # ROCm seeds no torchao checkpoint (dense_transformer_supported is False there).
    assert not tq.dense_transformer_supported(GPU_BF16)
    got, out = _plan(monkeypatch, "int8")
    assert got == V.DENOISER_SEED_DECLINED and out == ["int8"]


@pytest.mark.parametrize("requested", ["auto", None])
def test_rocm_planner_keeps_auto_dense(rocm, monkeypatch, requested):
    assert _plan(monkeypatch, requested) == (None, [])


def test_rocm_planner_kill_switch(rocm, monkeypatch):
    monkeypatch.setenv(tq.ROCM_VIDEO_HOSTED_INT8_ENV, "0")
    assert _plan(monkeypatch, "int8") == (None, [])
