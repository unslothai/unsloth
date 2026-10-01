# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""INT8 under streamed offload on torchao 0.17 (what a fresh install resolves: torch < 2.12 -> torchao 0.17).

The planner used to require torchao 0.18 for int8 under any tier that streams the denoiser, so every offloading
install got the fp8 transformer. 0.17 already ships the pinnable ``Int8Tensor``; only the v1 int8 class it still
defaults to is unpinnable, and that is rebuilt before streaming. Version-simulated, CPU only."""

from __future__ import annotations

import types

import pytest

from core.inference import diffusion_memory as mem
from core.inference.diffusion_memory import (
    OFFLOAD_GROUP,
    OFFLOAD_MODEL,
    OFFLOAD_STREAMING,
    MemoryPlan,
    torchao_offload_plan,
    torchao_scheme_streams,
    torchao_survives_plan,
)

ENV = "UNSLOTH_DIFFUSION_INT8_STREAM_TORCHAO17"
AO17, AO18, AO16 = (0, 17), (0, 18), (0, 16)


@pytest.fixture(autouse = True)
def _host(monkeypatch):
    # hermetic: the installed torchao / diffusers of the test host never decide these tests
    monkeypatch.setattr(mem, "_installed_diffusers_version", lambda: (0, 41))
    monkeypatch.setattr(mem, "_int8_tensor_pinnable", lambda: True, raising = False)
    monkeypatch.delenv(ENV, raising = False)


def _plan(policy, stream_transformer = True):
    # Qwen-Image-2.1 int8 at a 16 GB budget: denoiser 6922 MiB, companions 9603 MiB
    return MemoryPlan(
        requested_mode = "auto",
        offload_policy = policy,
        vae_tiling = False,
        vae_slicing = True,
        device_memory = None,
        estimates = {
            "safe_device_budget_mib": 13320,
            "model_dense_mib": 16525,
            "companion_dense_mib": 9603,
            "text_encoder_dense_mib": 8959,
            "runtime_headroom_mib": 8192,
            "base_overhead_mib": 2048,
        },
        reasons = (),
        stream_transformer = stream_transformer,
    )


def test_int8_streams_on_torchao_017():
    assert torchao_scheme_streams("int8", torchao_version = AO17) is True


@pytest.mark.parametrize("policy", [OFFLOAD_GROUP, OFFLOAD_STREAMING])
def test_int8_survives_a_streamed_plan_on_torchao_017(monkeypatch, policy):
    monkeypatch.setattr(mem, "_torchao_stream_pinnable", lambda *a, **k: True)
    plan = _plan(policy)
    assert torchao_survives_plan(plan, "int8", torchao_version = AO17) is True
    assert torchao_offload_plan(plan, "int8", torchao_version = AO17).offload_policy == policy


def test_int8_whole_module_too_big_streams_on_torchao_017(monkeypatch):
    monkeypatch.setattr(mem, "_torchao_stream_pinnable", lambda *a, **k: True)
    plan = _plan(OFFLOAD_MODEL)
    placed = torchao_offload_plan(plan, "int8", torchao_version = AO17)
    assert placed is not None and placed.offload_policy == OFFLOAD_STREAMING


def test_kill_switch_restores_the_018_floor(monkeypatch):
    monkeypatch.setenv(ENV, "0")
    assert torchao_scheme_streams("int8", torchao_version = AO17) is False
    assert torchao_scheme_streams("int8", torchao_version = AO18) is True


def test_torchao_without_a_pinnable_int8_tensor_keeps_the_018_floor(monkeypatch):
    monkeypatch.setattr(mem, "_int8_tensor_pinnable", lambda: False)
    assert torchao_scheme_streams("int8", torchao_version = AO17) is False


@pytest.mark.parametrize("version", [AO16, (0, 14)])
def test_older_torchao_still_declines_int8_streaming(version):
    assert torchao_scheme_streams("int8", torchao_version = version) is False


def test_fp8_and_other_schemes_unchanged():
    assert torchao_scheme_streams("fp8", torchao_version = AO17) is True
    assert torchao_scheme_streams("fp8", torchao_version = AO16) is False
    assert torchao_scheme_streams("nvfp4", torchao_version = AO18) is False


# ---- apply time: a v1 int8 module is rebuilt before the stream, else it would copy synchronously


class _V1Weight:
    pass


_V1Weight.__module__ = "torchao.quantization.linear_activation_quantized_tensor"
_V1Weight.__name__ = "LinearActivationQuantizedTensor"


class _Int8Weight:
    pass


_Int8Weight.__module__ = "torchao.quantization"
_Int8Weight.__name__ = "Int8Tensor"


class _Module:
    def __init__(self, cls):
        self.weights = [cls()]

    def parameters(self):
        return iter(self.weights)


STREAM_KWARGS = {"use_stream": True, "non_blocking": True, "record_stream": True, "low_cpu_mem_usage": False}


def _rebuild_to_int8(module):
    module.weights = [_Int8Weight()]
    return 1


@pytest.fixture
def _apply_env(monkeypatch):
    monkeypatch.setattr(mem, "install_group_offload_torchao_swap_retry", lambda: True)
    monkeypatch.setattr(mem, "_freeze_torchao_weights", lambda module: None)


def test_v1_int8_is_rebuilt_and_keeps_the_copy_stream(monkeypatch, _apply_env):
    import core.inference.prequant_legacy_int8 as legacy

    monkeypatch.setattr(legacy, "convert_legacy_int8_weights", _rebuild_to_int8)
    module = _Module(_V1Weight)
    out = mem._torchao_group_offload_kwargs(module, dict(STREAM_KWARGS), [0])
    assert type(module.weights[0]).__name__ == "Int8Tensor"
    assert out["use_stream"] is True and out.get("record_stream") is True


def test_v1_int8_kill_switch_keeps_v1_and_synchronous_copies(monkeypatch, _apply_env):
    import core.inference.prequant_legacy_int8 as legacy

    monkeypatch.setenv(ENV, "0")
    monkeypatch.setattr(legacy, "convert_legacy_int8_weights", _rebuild_to_int8)
    module = _Module(_V1Weight)
    out = mem._torchao_group_offload_kwargs(module, dict(STREAM_KWARGS), [0])
    assert type(module.weights[0]).__name__ == "LinearActivationQuantizedTensor"
    assert out["use_stream"] is False


def test_v1_int8_that_fails_validation_falls_back_to_synchronous(monkeypatch, _apply_env):
    import core.inference.prequant_legacy_int8 as legacy

    monkeypatch.setattr(legacy, "convert_legacy_int8_weights", lambda module: 0)
    module = _Module(_V1Weight)
    out = mem._torchao_group_offload_kwargs(module, dict(STREAM_KWARGS), [0])
    assert out["use_stream"] is False


def test_int8_tensor_module_is_untouched(monkeypatch, _apply_env):
    import core.inference.prequant_legacy_int8 as legacy

    calls = []
    monkeypatch.setattr(legacy, "convert_legacy_int8_weights", lambda m: calls.append(m) or 0)
    out = mem._torchao_group_offload_kwargs(_Module(_Int8Weight), dict(STREAM_KWARGS), [0])
    assert calls == [] and out["use_stream"] is True
