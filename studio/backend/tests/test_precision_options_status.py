# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Per-family denied precisions and probe failures reach the status and the download plan, so the Precision select
can disable a scheme with its reason before a load is refused (Wan2.2 TI2V 5B NVFP4, an FP8 the card cannot run)."""

from __future__ import annotations

import re
from types import SimpleNamespace

import pytest

from core.inference import diffusion_transformer_quant as tq
from models.inference import (
    DiffusionDownloadPlanResponse,
    DiffusionStatusResponse,
    VideoStatusResponse,
)

WAN_5B = "wan2.2-ti2v-5b"
SCHEMES = ("int8", "fp8", "nvfp4", "mxfp8")


@pytest.fixture(autouse = True)
def _no_probe(monkeypatch):
    """The helper is read by every status poll: it must never run a probe or read a stale card table."""

    def boom(*_a, **_k):
        raise AssertionError("precision_options must not spawn the smoke probe")

    monkeypatch.setattr(tq, "_run_smoke_probe", boom)
    monkeypatch.setattr(tq, "_child_probe_table", boom)
    monkeypatch.setattr(tq, "_persisted_probe_table", lambda card: None)
    monkeypatch.setattr(tq, "_SMOKE_CACHE", {})
    monkeypatch.setattr(tq, "_PRECISION_PERSISTED_TABLES", {})


def test_wan_nvfp4_carries_the_measured_gate_failure(monkeypatch):
    monkeypatch.setenv("UNSLOTH_NVFP4_DIFFUSION", "1")
    options = tq.precision_options(WAN_5B, "Wan-AI/Wan2.2-TI2V-5B-Diffusers", "cuda", ordinal = 0)
    assert set(options) == set(SCHEMES)
    assert options["nvfp4"] and "accuracy gate" in options["nvfp4"]
    assert "LPIPS 0.29" in options["nvfp4"]
    assert options["int8"] is None and options["fp8"] is None


def test_a_measured_failure_outlives_the_nvfp4_switch(monkeypatch):
    monkeypatch.setenv("UNSLOTH_NVFP4_DIFFUSION", "0")
    options = tq.precision_options(WAN_5B, None, None)
    assert options["nvfp4"] and "LPIPS" in options["nvfp4"]


def test_switched_off_nvfp4_without_a_measured_failure_is_left_to_the_caller(monkeypatch):
    monkeypatch.setenv("UNSLOTH_NVFP4_DIFFUSION", "0")
    assert tq.precision_options("flux.1", None, None) == {s: None for s in SCHEMES}


def test_a_family_with_no_deny_rules_nothing_out(monkeypatch):
    monkeypatch.setenv("UNSLOTH_NVFP4_DIFFUSION", "1")
    assert tq.precision_options("no-such-family", None, "cuda", ordinal = 0) == {
        s: None for s in SCHEMES
    }


def _target(
    ordinal,
    *,
    native = False,
    fp16_native = False,
    offload_native = False,
    monkeypatch = None,
):
    """An NVIDIA-shaped bf16 target whose native-path answers the test chooses."""
    monkeypatch.setattr(tq, "native_quant_host", lambda target: native)
    monkeypatch.setattr(tq, "fp16_card_native_host", lambda target, scheme: fp16_native)
    monkeypatch.setattr(tq, "native_offload_host", lambda target: offload_native)
    return SimpleNamespace(device = "cuda", dtype = "bfloat16", ordinal = ordinal)


def test_a_recorded_probe_failure_names_the_card(monkeypatch):
    monkeypatch.setenv("UNSLOTH_NVFP4_DIFFUSION", "1")
    tq._SMOKE_CACHE[("fp8", "cuda:1")] = False
    tq._SMOKE_CACHE[("int8", "cuda:1")] = True
    target = _target(1, monkeypatch = monkeypatch)
    options = tq.precision_options("no-such-family", None, "cuda", target = target, offload = False)
    assert options["fp8"] and "kernel probe" in options["fp8"]
    assert options["int8"] is None
    # Another card's verdict does not leak.
    other = SimpleNamespace(device = "cuda", dtype = "bfloat16", ordinal = 0)
    assert tq.precision_options("no-such-family", None, "cuda", target = other)["fp8"] is None


def test_the_persisted_table_counts_when_this_process_has_not_probed(monkeypatch):
    monkeypatch.setattr(tq, "_persisted_probe_table", lambda card: {"fp8": False, "int8": True})
    options = tq.precision_options(None, None, "cuda:0", target = _target(0, monkeypatch = monkeypatch))
    assert options["fp8"] and options["int8"] is None


def test_a_probe_verdict_never_applies_to_a_native_path(monkeypatch):
    """The torchao probe does not judge the native ROCm / MPS, fp16 Turing or native INT8 offload loads, which the
    precision check lets through, so their schemes stay pickable."""
    tq._SMOKE_CACHE[("fp8", "cuda:0")] = False
    tq._SMOKE_CACHE[("int8", "cuda:0")] = False
    rocm = _target(0, native = True, monkeypatch = monkeypatch)
    assert tq.precision_options(None, None, "cuda", target = rocm, offload = False)["fp8"] is None
    assert tq.precision_options(None, None, "cuda", target = rocm, offload = False)["int8"] is None
    # The modular workflow seeds torchao checkpoints, so a ROCm host still runs the probed kernels there.
    assert tq.precision_options(None, None, "cuda", target = rocm, offload = False, modular = True)["fp8"]
    turing = _target(0, fp16_native = True, monkeypatch = monkeypatch)
    assert tq.precision_options(None, None, "cuda", target = turing, offload = False)["int8"] is None
    nvidia = _target(0, offload_native = True, monkeypatch = monkeypatch)
    assert tq.precision_options(None, None, "cuda", target = nvidia, offload = True)["int8"] is None
    # Not known whether the load offloads: counted as possible, so no refusal.
    assert tq.precision_options(None, None, "cuda", target = nvidia, offload = None)["int8"] is None
    # Resident, the native offload path is out, so the torchao verdict stands; fp8 has no native offload path.
    assert tq.precision_options(None, None, "cuda", target = nvidia, offload = False)["int8"]
    assert tq.precision_options(None, None, "cuda", target = nvidia, offload = True)["fp8"]
    # A GGUF pick reaches dense quant only through torchao.
    assert tq.precision_options(None, None, "cuda", target = rocm, model_kind = "gguf")["fp8"]


def test_without_a_target_only_the_family_table_answers(monkeypatch):
    tq._SMOKE_CACHE[("fp8", "cuda:0")] = False
    assert tq.precision_options(None, None, "cuda", ordinal = 0)["fp8"] is None


def test_precision_target_builds_from_recorded_state():
    torch = pytest.importorskip("torch")
    target = tq.precision_target("cuda", "bfloat16", 1)
    assert target is not None and target.ordinal == 1 and target.dtype is torch.bfloat16
    assert tq.precision_target("cuda", "not-a-dtype", 0) is None
    assert tq.precision_target(None, "bfloat16", 0) is None


def test_an_unknown_card_index_never_initialises_cuda(monkeypatch):
    # "cuda" with no ordinal and no live CUDA context reads no probe verdict rather than calling current_device().
    key = tq._precision_card_key("cuda", None)
    assert key is None or key.startswith("cuda:")
    assert tq._precision_card_key(None, None) is None
    assert tq._precision_card_key("cuda", 2) == "cuda:2"


def test_lookup_errors_never_raise(monkeypatch):
    def broken(*_a, **_k):
        raise RuntimeError("table unreadable")

    monkeypatch.setattr(tq, "family_denies_scheme", broken)
    assert tq.precision_options(WAN_5B, None, "cuda:0") == {s: None for s in SCHEMES}


def test_plan_options_judge_a_pipeline_on_its_own_repo(monkeypatch):
    seen = {}

    def spy(
        family,
        base,
        device,
        *,
        ordinal = None,
        target = None,
        **_kw,
    ):
        seen.update(family = family, base = base, device = device, ordinal = ordinal)
        return {s: None for s in SCHEMES}

    monkeypatch.setattr(tq, "precision_options", spy)
    tq.plan_precision_options(
        WAN_5B, model_kind = "pipeline", repo_id = "a/b", base_repo = None, gpu_ordinal = 1
    )
    assert seen == {"family": WAN_5B, "base": "a/b", "device": "cuda", "ordinal": 1}
    assert (
        tq.plan_precision_options(
            None, model_kind = "pipeline", repo_id = "a/b", base_repo = None, gpu_ordinal = 0
        )
        is None
    )


def test_the_response_models_carry_the_field():
    payload = {"nvfp4": "Measured and failed", "fp8": None}
    assert (
        DiffusionStatusResponse(loaded = True, precision_options = payload).precision_options == payload
    )
    assert VideoStatusResponse(loaded = True, precision_options = payload).precision_options == payload
    assert DiffusionDownloadPlanResponse(precision_options = payload).precision_options == payload
    assert DiffusionStatusResponse(loaded = False).precision_options is None
    assert (
        DiffusionStatusResponse(loaded = True, precision_options_card = 1).precision_options_card == 1
    )
    assert VideoStatusResponse(loaded = True, precision_options_card = 0).precision_options_card == 0


def test_status_and_plan_routes_fill_it():
    """Wiring: both status() dicts and both download-plan routes set the field."""
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    image = (root / "core/inference/diffusion.py").read_text(encoding = "utf-8")
    video = (root / "core/inference/video.py").read_text(encoding = "utf-8")
    flat_image = re.sub(r"\s+", " ", image)
    flat_video = re.sub(r"\s+", " ", video)
    # The status passes the resident target, so a probe verdict only lands where torchao runs.
    for flat in (flat_image, flat_video):
        assert re.search(
            r'"precision_options": precision_options\( ?(state\.family\.name|fam\.name),', flat
        )
        assert "target = precision_target( state.device, state.dtype," in flat
        assert "model_kind = state.kind," in flat
        # The verdicts describe the next load, whose Memory pick may offload: the resident's policy is not assumed.
        assert "model_kind = state.kind, # These verdicts describe the NEXT load" in flat
        assert "offload = None," in flat
        assert (
            "offload = state.offload_policy" not in flat
            and "offload = (state.offload_policy" not in flat
        )
        assert '"precision_options_card": None,' in flat
        assert (
            '"precision_options_card": physical_gpu_id_for_ordinal( state.placed_ordinal if state.placed_ordinal is not None else state.gpu_ordinal )'
            in flat
        )
    for route in ("routes/inference.py", "routes/video.py"):
        text = (root / route).read_text(encoding = "utf-8")
        assert '"precision_options": plan_precision_options(' in text
        assert "target = plan_target," in text


def test_the_status_card_is_the_physical_id_the_ui_picks(monkeypatch):
    """Under CUDA_VISIBLE_DEVICES=1,0 torch ordinal 0 is physical GPU 1: the status must name 1, or picking physical 1
    hides its own probe failure and picking physical 0 inherits it."""
    from core.inference import diffusion_device
    from utils.hardware import hardware

    monkeypatch.setattr(hardware, "get_parent_visible_gpu_ids", lambda: [1, 0])
    assert diffusion_device.physical_gpu_id_for_ordinal(0) == 1
    assert diffusion_device.physical_gpu_id_for_ordinal(1) == 0
    assert diffusion_device.physical_gpu_id_for_ordinal(2) is None
    assert diffusion_device.physical_gpu_id_for_ordinal(None) is None
    monkeypatch.setattr(hardware, "get_parent_visible_gpu_ids", lambda: [4, 5])
    assert diffusion_device.physical_gpu_id_for_ordinal(1) == 5

    def broken():
        raise RuntimeError("mask unreadable")

    monkeypatch.setattr(hardware, "get_parent_visible_gpu_ids", broken)
    assert diffusion_device.physical_gpu_id_for_ordinal(0) is None


def test_h3_fp8_stays_selectable_where_its_stored_fallback_runs(monkeypatch):
    """MiniMax-H3 runs its hosted fp8 codes stored where torchao's fp8 GEMM fails, so a cached fp8 probe failure must
    not disable fp8 for it; other families keep the verdict."""
    from core.inference import video_minimax_h3_native_quant as h3

    tq._SMOKE_CACHE[("fp8", "cuda:0")] = False
    target = _target(0, monkeypatch = monkeypatch)
    monkeypatch.setattr(h3, "h3_hosted_fp8_storage_host", lambda target: True)
    assert (
        tq.precision_options("minimax-h3", None, "cuda", target = target, offload = False)["fp8"]
        is None
    )
    assert tq.precision_options("no-such-family", None, "cuda", target = target, offload = False)["fp8"]
    # Without the stored-fp8 host (no float8 storage, a non-bf16 card), the torchao verdict stands for H3 too.
    monkeypatch.setattr(h3, "h3_hosted_fp8_storage_host", lambda target: False)
    assert tq.precision_options("minimax-h3", None, "cuda", target = target, offload = False)["fp8"]


def test_the_h3_load_check_and_the_status_share_one_host_rule(monkeypatch):
    from core.inference import video_minimax_h3_native_quant as h3
    monkeypatch.setattr(h3, "h3_hosted_fp8_storage_host", lambda target: False)
    assert h3.h3_hosted_fp8_native(SimpleNamespace(device = "cuda", dtype = None)) is False
