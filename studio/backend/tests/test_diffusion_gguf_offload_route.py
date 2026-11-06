# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A GGUF pick whose memory plan offloads loads the cached hosted int8 / fp8 checkpoint under the same placement."""

from __future__ import annotations

import types
from dataclasses import replace

import pytest

from core.inference import diffusion_gguf_route as route
from core.inference import diffusion_memory as mem
from core.inference.diffusion_memory import (
    OFFLOAD_GROUP,
    OFFLOAD_NONE,
    OFFLOAD_SEQUENTIAL,
    OFFLOAD_STREAMING,
    DeviceMemory,
    plan_diffusion_memory,
)


@pytest.fixture(autouse = True)
def _roomy_host(monkeypatch):
    monkeypatch.setattr(mem, "_pin_budget_mib", lambda: 10**7)
    monkeypatch.setattr(mem, "_pinned_memory_capped", lambda: False)
    monkeypatch.delenv(mem.GROUP_OFFLOAD_PIN_ENV, raising = False)
    monkeypatch.setattr(mem, "_installed_diffusers_version", lambda: (0, 40))
    monkeypatch.delenv(route.GGUF_OFFLOAD_PREQUANT_ENV, raising = False)


def _plan(policy, *, stream_transformer = True):
    device = DeviceMemory("cuda", "cuda:0", "discrete_vram", 11_000, 12_288)
    plan = plan_diffusion_memory(
        target = types.SimpleNamespace(supports_model_cpu_offload = True),
        device_memory = device,
        model_dense_mib = 15_000,
        runtime_headroom_mib = 2_000,
        companion_dense_mib = 8_000,
        text_encoder_dense_mib = 7_800,
    )
    return replace(plan, offload_policy = policy, stream_transformer = stream_transformer)


def _ao(version = (0, 18)):
    return lambda plan, scheme: mem.torchao_offload_plan(plan, scheme, torchao_version = version)


INT8 = types.SimpleNamespace(scheme = "int8", prequant = True, transient_transformer_mib = 7_000)
DENSE = types.SimpleNamespace(scheme = "int8", prequant = False, transient_transformer_mib = 20_000)


def _decide(
    plan,
    candidate = INT8,
    *,
    auto = True,
    gguf = "qwen-image-2.1-Q4_K_M.gguf",
    cached = True,
    version = (0, 18),
):
    return route.gguf_offload_prequant_placement(
        plan,
        candidate,
        scheme = getattr(candidate, "scheme", None),
        auto = auto,
        gguf_filename = gguf,
        prequant_cached = cached,
        torchao_offload_plan = _ao(version),
    )


@pytest.mark.parametrize("policy", [OFFLOAD_STREAMING, OFFLOAD_GROUP])
def test_offloading_q4_pick_takes_the_cached_int8_checkpoint(policy):
    placement, reason = _decide(_plan(policy))
    assert reason is None
    assert placement is not None and placement.offload_policy == policy


def test_sequential_offload_keeps_the_gguf():
    assert _decide(_plan(OFFLOAD_SEQUENTIAL)) == (None, None)


def test_torchao_too_old_to_stream_int8_keeps_the_gguf():
    assert _decide(_plan(OFFLOAD_STREAMING), version = (0, 16)) == (None, None)


@pytest.mark.parametrize("value", ["0", "off", "false", "no"])
def test_kill_switch_restores_the_resident_only_rule(monkeypatch, value):
    monkeypatch.setenv(route.GGUF_OFFLOAD_PREQUANT_ENV, value)
    assert _decide(_plan(OFFLOAD_STREAMING)) == (None, None)


def test_never_builds_dense_under_offload():
    assert _decide(_plan(OFFLOAD_STREAMING), DENSE) == (None, None)


def test_auto_never_downloads_a_second_denoiser():
    placement, reason = _decide(_plan(OFFLOAD_STREAMING), cached = False)
    assert placement is None and "not cached" in reason


def test_explicit_scheme_may_fetch_its_checkpoint():
    placement, reason = _decide(_plan(OFFLOAD_STREAMING), cached = False, auto = False)
    assert reason is None and placement is not None


@pytest.mark.parametrize(
    "gguf, outranks",
    [
        ("qwen-image-2.1-Q2_K.gguf", False),
        ("qwen-image-2.1-Q3_K_M.gguf", False),
        ("qwen-image-2.1-Q4_K_M.gguf", False),
        ("qwen-image-2.1-Q5_K_S.gguf", False),
        ("qwen-image-2.1-Q5_K_M.gguf", False),
        ("model-UD-Q4_K_XL.gguf", False),
        ("model-IQ4_XS.gguf", False),
        ("model-MXFP4.gguf", False),
        ("qwen-image-2.1-Q6_K.gguf", True),
        ("qwen-image-2.1-Q6_K_XL.gguf", True),
        ("model-UD-Q6_K_XL.gguf", True),
        ("qwen-image-2.1-Q8_0.gguf", True),
        ("qwen-image-2.1-F16.gguf", True),
        ("qwen-image-2.1-BF16.gguf", True),
        ("model.gguf", True),
        ("model-MXFP8.gguf", True),
        ("model-IQ6_XS.gguf", True),
        ("model-PQ8_0.gguf", True),
        ("model-TQ1_0.gguf", False),
        ("model-Q5_0.gguf", False),
        (None, True),
    ],
)
def test_accuracy_gate_by_quant(gguf, outranks):
    assert route.gguf_outranks_scheme(gguf, "int8") is outranks


@pytest.mark.parametrize(
    "gguf, outranks",
    [
        ("x-Q4_K_M.gguf", False),
        ("x-Q5_K_M.gguf", True),
        ("x-Q6_K.gguf", True),
        ("x-MXFP8.gguf", True),
    ],
)
def test_fp8_only_replaces_4_bit_and_narrower(gguf, outranks):
    assert route.gguf_outranks_scheme(gguf, "fp8") is outranks


def test_auto_keeps_a_q8_pick_explicit_int8_still_swaps():
    placement, reason = _decide(_plan(OFFLOAD_STREAMING), gguf = "qwen-image-2.1-Q8_0.gguf")
    assert placement is None and "Q8_0" in reason and "at least as accurate" in reason
    placement, reason = _decide(
        _plan(OFFLOAD_STREAMING), gguf = "qwen-image-2.1-Q8_0.gguf", auto = False
    )
    assert placement is not None and reason is None


def test_swap_reason_names_what_ran_and_how_to_keep_the_gguf():
    text = route.gguf_offload_swap_reason(
        "int8",
        "prequant:unsloth/Qwen-Image-2.1-FP8/Qwen-Image-2.1-INT8.safetensors",
        types.SimpleNamespace(offload_policy = OFFLOAD_STREAMING),
        "qwen-image-2.1-Q4_K_M.gguf",
    )
    assert (
        "Q4_K_M GGUF pick was replaced by unsloth/Qwen-Image-2.1-FP8/Qwen-Image-2.1-INT8.safetensors"
        in text
    )
    assert "'streaming'" in text and "Precision to Off" in text


def test_resident_plans_are_not_this_rules_business():
    placement, _ = _decide(_plan(OFFLOAD_NONE))
    assert placement.offload_policy == OFFLOAD_NONE
