# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Explicit fp8 on a conventional video pipeline whose plan moves the DiT: the hosted torchao fp8 denoiser is seeded
and streamed through group offload (``_video_torchao_seed_placement``), as the image path does, instead of the load
being refused. CPU only: the placement rule and both planner call sites, with the plan and the fp8 probe spoofed."""

import types

import pytest
import torch

import core.inference.diffusion_memory as dm
import core.inference.video as vid
from core.inference.video_families import detect_video_family

LTX25 = "Lightricks/LTX-2.5-Diffusers"
CUDA_BF16 = types.SimpleNamespace(device = "cuda", dtype = torch.bfloat16, ordinal = 0)


def _plan(policy, **kw):
    return types.SimpleNamespace(
        offload_policy = policy,
        stream_transformer = kw.pop("stream_transformer", policy != "none"),
        stream_text_encoders = kw.pop("stream_text_encoders", True),
        reasons = (),
        **kw,
    )


@pytest.fixture
def fam():
    return detect_video_family(LTX25)


@pytest.fixture
def fp8_gemm(monkeypatch):
    """A card whose fp8 probe passes (sm_89+) unless a test flips ``state["ok"]``."""
    state = {"ok": True}
    monkeypatch.setattr(
        vid,
        "select_transformer_quant_scheme",
        lambda target, scheme, family = None, **kw: scheme if state["ok"] else None,
    )
    monkeypatch.setattr(
        vid, "plan_keeps_transformer_resident", lambda p: p.offload_policy == "none"
    )
    return state


@pytest.mark.parametrize(
    "policy, placed_policy, want",
    [
        ("group", "group", "group"),  # the L4 whole-card plan: denoiser and text encoders streamed
        ("streaming", "streaming", "streaming"),
        (
            "model",
            "model",
            "streaming",
        ),  # whole-module hooks cannot move a Float8Tensor: stream instead
        (
            "model",
            "streaming",
            "streaming",
        ),  # the quantised parts do not fit whole (torchao_offload_plan)
        ("sequential", None, None),  # never measured with torchao
        ("group", None, None),  # torchao / diffusers too old, or no up-front pin budget
    ],
)
def test_placement_follows_torchao_offload_plan(
    monkeypatch, fam, fp8_gemm, policy, placed_policy, want
):
    plan = _plan(policy)
    seen = []

    def offload_plan(p, scheme):
        seen.append(scheme)
        return (
            None
            if placed_policy is None
            else (p if placed_policy == policy else _plan(placed_policy))
        )

    monkeypatch.setattr(dm, "torchao_offload_plan", offload_plan)
    monkeypatch.setattr(dm, "torchao_scheme_streams", lambda scheme, **kw: True)
    monkeypatch.setattr(dm, "_torchao_stream_pinnable", lambda p, reclaimable_host_mib = 0: True)
    monkeypatch.setattr(dm, "torchao_streaming_plan", lambda p: _plan("streaming"))
    got = vid._video_torchao_seed_placement(fam, "fp8", CUDA_BF16, plan)
    assert seen == ["fp8"]
    assert (None if got is None else got.offload_policy) == want


def test_whole_module_plan_without_a_pin_budget_is_not_streamed(monkeypatch, fam, fp8_gemm):
    monkeypatch.setattr(dm, "torchao_offload_plan", lambda p, s: p)
    monkeypatch.setattr(dm, "torchao_scheme_streams", lambda scheme, **kw: True)
    monkeypatch.setattr(dm, "_torchao_stream_pinnable", lambda p, reclaimable_host_mib = 0: False)
    assert vid._video_torchao_seed_placement(fam, "fp8", CUDA_BF16, _plan("model")) is None


@pytest.mark.parametrize(
    "case",
    ["int8", "resident", "no_plan", "kill_switch", "no_fp8_gemm", "fp16", "cpu", "modular"],
)
def test_placement_declines(monkeypatch, fam, fp8_gemm, case):
    """int8 keeps its torchao-free twin; a resident plan needs nothing; no fp8 GEMM (A100), a non-bf16 or non-CUDA
    target, the kill switch and the modular workflow (MiniMax-H3 places its own denoiser) keep today's decline."""
    calls = []
    monkeypatch.setattr(dm, "torchao_offload_plan", lambda p, s: calls.append(s) or p)
    scheme, target, plan = "fp8", CUDA_BF16, _plan("group")
    if case == "int8":
        scheme = "int8"
    elif case == "resident":
        plan = _plan("none")
    elif case == "no_plan":
        plan = None
    elif case == "kill_switch":
        monkeypatch.setenv(vid.VIDEO_FP8_STREAM_ENV, "0")
    elif case == "no_fp8_gemm":
        fp8_gemm["ok"] = False
    elif case == "fp16":
        target = types.SimpleNamespace(device = "cuda", dtype = torch.float16, ordinal = 0)
    elif case == "cpu":
        target = types.SimpleNamespace(device = "cpu", dtype = torch.bfloat16, ordinal = None)
    elif case == "modular":
        fam = types.SimpleNamespace(name = "minimax-h3", modular_workflow = "t2v")
    assert vid._video_torchao_seed_placement(fam, scheme, target, plan) is None
    assert calls == []


def _planner(monkeypatch, fam, *, resident, placement):
    backend = vid.VideoBackend()
    backend._state = types.SimpleNamespace(pipe = None)
    monkeypatch.setattr(vid, "_video_auto_denoiser_scheme", lambda fam, **kw: "fp8")
    monkeypatch.setattr(vid, "_pipeline_device_mib", lambda pipe, ordinal = None: 0)
    monkeypatch.setattr(vid, "resolve_diffusion_device_target", lambda **_k: CUDA_BF16)
    judged = _plan("group")

    def stays(fam, **kw):
        if kw.get("plan_out") is not None:
            kw["plan_out"].append(judged)
        return resident

    monkeypatch.setattr(vid, "_video_seed_stays_resident", stays)
    seen = []
    monkeypatch.setattr(
        vid,
        "_video_torchao_seed_placement",
        lambda fam, scheme, target, plan: seen.append((scheme, plan)) or placement,
    )
    monkeypatch.setattr(vid, "_video_native_hosted_scheme", lambda fam, scheme, target: None)
    native: list = []
    got = backend._video_planned_auto_denoiser_scheme(
        fam,
        base = LTX25,
        kind = "pipeline",
        transformer_quant = "fp8",
        speed_mode = None,
        native_hosted = native,
    )
    return got, seen, judged


def test_planner_seeds_the_hosted_fp8_when_it_streams(monkeypatch, fam):
    got, seen, judged = _planner(monkeypatch, fam, resident = False, placement = _plan("group"))
    assert got == "fp8"
    assert seen == [("fp8", judged)]  # judged on the same artifact-sized plan, no second settle


def test_planner_still_declines_when_it_cannot_stream(monkeypatch, fam):
    got, seen, _ = _planner(monkeypatch, fam, resident = False, placement = None)
    assert got == vid.DENOISER_SEED_DECLINED
    assert seen


def test_planner_resident_seed_skips_the_streaming_check(monkeypatch, fam):
    got, seen, _ = _planner(monkeypatch, fam, resident = True, placement = None)
    assert got == "fp8"
    assert seen == []


def test_stays_resident_hands_back_the_plan_it_judged(monkeypatch, fam):
    planned = _plan("group")
    monkeypatch.setattr(vid, "_video_seed_plan", lambda fam, **kw: planned)
    monkeypatch.setattr(vid, "plan_keeps_transformer_resident", lambda p: False)
    out: list = []
    assert (
        vid._video_seed_stays_resident(
            fam,
            target = CUDA_BF16,
            scheme = "fp8",
            memory_mode = None,
            text_encoder_quant = None,
            base_repo = LTX25,
            plan_out = out,
        )
        is False
    )
    assert out == [planned]
    # no sizes: unchanged answer (a scheme counts as resident), nothing handed back
    monkeypatch.setattr(vid, "_video_seed_plan", lambda fam, **kw: None)
    out.clear()
    assert vid._video_seed_stays_resident(
        fam,
        target = CUDA_BF16,
        scheme = "fp8",
        memory_mode = None,
        text_encoder_quant = None,
        base_repo = LTX25,
        plan_out = out,
    )
    assert out == []
