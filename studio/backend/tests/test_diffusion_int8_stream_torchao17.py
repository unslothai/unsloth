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


# ---- apply time: a v1 int8 module (0.17's default int8 class) keeps the copy stream through the pin-op shim


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


STREAM_KWARGS = {
    "use_stream": True,
    "non_blocking": True,
    "record_stream": True,
    "low_cpu_mem_usage": False,
}


@pytest.fixture
def _apply_env(monkeypatch):
    monkeypatch.setattr(mem, "install_group_offload_torchao_swap_retry", lambda: True)
    monkeypatch.setattr(mem, "_freeze_torchao_weights", lambda module: None)


def test_v1_int8_keeps_the_copy_stream_once_its_pin_ops_exist(monkeypatch, _apply_env):
    monkeypatch.setattr(mem, "install_torchao_v1_int8_pin_ops", lambda: True, raising = False)
    out = mem._torchao_group_offload_kwargs(_Module(_V1Weight), dict(STREAM_KWARGS), [0])
    assert (
        out["use_stream"] is True
        and out.get("record_stream") is True
        and out.get("non_blocking") is True
    )


def test_v1_int8_without_its_pin_ops_falls_back_to_synchronous(monkeypatch, _apply_env):
    monkeypatch.setattr(mem, "install_torchao_v1_int8_pin_ops", lambda: False, raising = False)
    out = mem._torchao_group_offload_kwargs(_Module(_V1Weight), dict(STREAM_KWARGS), [0])
    assert out["use_stream"] is False


def test_pin_op_shim_honours_the_kill_switch(monkeypatch):
    monkeypatch.setenv(ENV, "0")
    monkeypatch.setattr(mem, "_V1_INT8_PIN_OPS_INSTALLED", False, raising = False)
    assert mem.install_torchao_v1_int8_pin_ops() is False


def test_int8_tensor_module_keeps_the_stream(monkeypatch, _apply_env):
    out = mem._torchao_group_offload_kwargs(_Module(_Int8Weight), dict(STREAM_KWARGS), [0])
    assert out["use_stream"] is True


def _v1_int8_stack():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA (pinned host memory)")
    try:
        from torchao.quantization.linear_activation_quantized_tensor import (  # noqa: F401
            LinearActivationQuantizedTensor,
        )
    except Exception:  # noqa: BLE001
        pytest.skip("this torchao has no v1 int8 classes (0.18+)")
    return torch


def test_v1_int8_streams_bit_identically_with_the_shim():
    """Real torchao <= 0.17 + CUDA: the v1 weights pin, stream on the copy stream, and match resident exactly."""
    import copy

    torch = _v1_int8_stack()
    from diffusers.hooks import apply_group_offloading
    from torchao.quantization import Int8DynamicActivationInt8WeightConfig, quantize_

    assert mem.install_torchao_v1_int8_pin_ops() is True
    torch.manual_seed(0)
    blocks = torch.nn.Sequential(
        *[
            torch.nn.Sequential(
                torch.nn.Linear(256, 512), torch.nn.GELU(), torch.nn.Linear(512, 256)
            )
            for _ in range(3)
        ]
    ).to(torch.bfloat16)
    offloaded = copy.deepcopy(blocks)
    # set_inductor_config = False as Studio builds it: the bare config sets float32 matmul precision process-wide
    quantize_(offloaded, Int8DynamicActivationInt8WeightConfig(set_inductor_config = False))
    offloaded.requires_grad_(False)
    assert type(next(offloaded.parameters())).__name__ == "LinearActivationQuantizedTensor"
    resident = copy.deepcopy(offloaded).cuda()
    kwargs = mem._torchao_group_offload_kwargs(
        offloaded,
        {
            "onload_device": torch.device("cuda"),
            "offload_device": torch.device("cpu"),
            "offload_type": "block_level",
            "num_blocks_per_group": 1,
            **STREAM_KWARGS,
        },
        [0],
    )
    assert kwargs["use_stream"] is True
    apply_group_offloading(offloaded, **kwargs)
    x = torch.randn(2, 16, 256, dtype = torch.bfloat16, device = "cuda")
    with torch.no_grad():
        want = resident(x)
        for _ in range(3):
            assert torch.equal(offloaded(x), want)


def test_v1_int8_payload_returns_to_the_host_after_each_forward():
    """The streamed v1 payload (int8 data + scales) must leave the GPU on offload, else the whole denoiser accumulates
    there: diffusers restores torchao weights through ``tensor_data_names``, which the v1 classes do not declare."""
    torch = _v1_int8_stack()
    from diffusers.hooks import apply_group_offloading
    from torchao.quantization import Int8DynamicActivationInt8WeightConfig, quantize_

    assert mem.install_torchao_v1_int8_pin_ops() is True
    torch.manual_seed(0)
    blocks = torch.nn.Sequential(
        *[
            torch.nn.Sequential(
                torch.nn.Linear(1024, 1024), torch.nn.GELU(), torch.nn.Linear(1024, 1024)
            )
            for _ in range(4)
        ]
    ).to(torch.bfloat16)
    quantize_(blocks, Int8DynamicActivationInt8WeightConfig(set_inductor_config = False))
    blocks.requires_grad_(False)
    kwargs = mem._torchao_group_offload_kwargs(
        blocks,
        {
            "onload_device": torch.device("cuda"),
            "offload_device": torch.device("cpu"),
            "offload_type": "block_level",
            "num_blocks_per_group": 1,
            **STREAM_KWARGS,
        },
        [0],
    )
    assert kwargs["use_stream"] is True
    apply_group_offloading(blocks, **kwargs)
    x = torch.randn(2, 16, 1024, dtype = torch.bfloat16, device = "cuda")
    with torch.no_grad():
        for _ in range(2):
            blocks(x)
    torch.cuda.synchronize()
    for block in blocks:
        for linear in (block[0], block[2]):
            impl = linear.weight.original_weight_tensor.tensor_impl
            assert impl.int_data.device.type == "cpu"
            assert impl.scale.device.type == "cpu"
