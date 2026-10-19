# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Behaviour of the two small-card fixes through entry points that exist before and after them.

Imports only names present on main, so on main these fail by ASSERTION, not ImportError:

1. The pipeline seed call in diffusion.py, evaluated as written, drives ``load_prequantized_transformer`` (checkpoint
   I/O faked) under an offloading Qwen-Image-2.1 plan: the module must be materialised on the host, not the GPU.
2. The post-load refinement expression in diffusion.py, evaluated as written, applied to a balanced 16 GB plan and a
   pipe whose loaded fp8 text encoder is 8960 MiB: the encoder must not stay resident.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest
import torch

import core.inference.diffusion as diffusion
import core.inference.diffusion_memory as dm
import core.inference.diffusion_prequant as prequant
from core.inference.diffusion_memory import (
    DeviceMemory,
    OFFLOAD_GROUP,
    OFFLOAD_MODEL,
    plan_diffusion_memory,
    torchao_streaming_plan,
)

BACKEND = Path(__file__).resolve().parents[1]
DIFFUSION_SRC = (BACKEND / "core/inference/diffusion.py").read_text(encoding = "utf-8")
MIB = 1024 * 1024
SIZES_AUTO = dict(model_dense_mib = 19630, companion_dense_mib = 12182, text_encoder_dense_mib = 10847)
SIZES_EXPLICIT_INT8 = dict(
    model_dense_mib = 31566, companion_dense_mib = 18024, text_encoder_dense_mib = 16689
)


class _Target:
    supports_model_cpu_offload = True


def _plan(
    free,
    total,
    mode = None,
    sizes = SIZES_AUTO,
):
    return plan_diffusion_memory(
        target = _Target(),
        device_memory = DeviceMemory("cuda", "cuda", "discrete_vram", free, total),
        runtime_headroom_mib = 8192,
        requested_mode = mode,
        **sizes,
    )


def _call(name):
    calls = [
        n
        for n in ast.walk(ast.parse(DIFFUSION_SRC))
        if isinstance(n, ast.Call) and getattr(n.func, "id", None) == name
    ]
    assert len(calls) == 1, f"expected one {name}() call in diffusion.py"
    return calls[0]


# ---------------------------------------------------------------- 1. seed placement


class _FakeDiT(torch.nn.Module):
    placed: list = []

    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(4, 4)

    @classmethod
    def from_config(cls, config):
        return cls()

    def to(self, *args, **kwargs):  # record, never move (CPU-only test host)
        _FakeDiT.placed.append(str(args[0] if args else kwargs.get("device")))
        return self


@pytest.fixture
def fake_checkpoint(monkeypatch):
    _FakeDiT.placed = []
    state = {k: v.clone() for k, v in torch.nn.Linear(4, 4).state_dict().items()}
    state = {f"proj.{k}": v for k, v in state.items()}
    monkeypatch.setattr(prequant, "_resolve_checkpoint_path", lambda *a, **k: "fake.safetensors")
    monkeypatch.setattr(
        prequant, "_load_prequant_checkpoint", lambda *a, **k: {"state_dict": state, "metadata": {}}
    )
    monkeypatch.setattr(prequant, "_validate_checkpoint", lambda *a, **k: True)
    monkeypatch.setattr(prequant, "_verify_packed_fingerprint", lambda *a, **k: True)
    monkeypatch.setattr(prequant, "_pin_kernel_preference", lambda *a, **k: 0)
    monkeypatch.setattr(prequant, "_load_transformer_config", lambda *a, **k: {})


class _Source:
    kind = "repo"
    location = "unsloth/Qwen-Image-2.1-FP8"
    filename = "Qwen-Image-2.1-INT8.safetensors"


def _seed_through_call_site(
    plan,
    device = "cuda",
    scheme = "int8",
):
    """Evaluate every keyword of diffusion.py's seed call, then run the loader with the ones it accepts."""
    call = _call("denoiser_prequant_pipe_kwargs")
    scope = dict(vars(diffusion))
    scope.update(plan = plan, device = device, pipeline_seed_scheme = scheme)
    kwargs = {}
    for kw in call.keywords:
        if kw.arg in ("device", "placement_device"):
            kwargs[kw.arg] = eval(compile(ast.Expression(kw.value), "<seed-call>", "eval"), scope)
    accepted = inspect.signature(prequant.load_prequantized_transformer).parameters
    kwargs = {k: v for k, v in kwargs.items() if k in accepted}
    module = prequant.load_prequantized_transformer(
        _FakeDiT, "Qwen/Qwen-Image-2.1", _Source(), dtype = torch.bfloat16, scheme = scheme, **kwargs
    )
    assert module is not None, "fake checkpoint load failed"
    return _FakeDiT.placed[-1]


@pytest.mark.parametrize(
    "free,total", [(7788, 8188), (11888, 12288), (15976, 16376)], ids = ["8GB", "12GB", "16GB"]
)
def test_offloading_plan_materialises_the_seed_on_the_host(fake_checkpoint, free, total):
    plan = _plan(free, total)
    if plan.offload_policy == OFFLOAD_MODEL:
        plan = torchao_streaming_plan(plan)  # what torchao_offload_plan makes of it for int8
    assert not dm.plan_keeps_transformer_resident(plan)
    assert _seed_through_call_site(plan) == "cpu"


def test_resident_plan_keeps_the_seed_on_the_device(fake_checkpoint):
    plan = _plan(24176, 24576)  # transformer resident, encoders streamed
    assert dm.plan_keeps_transformer_resident(plan)
    assert _seed_through_call_site(plan) == "cuda"


# ---------------------------------------------------------------- 2. balanced refinement


def _meta(mib):
    m = torch.nn.Module()
    m.w = torch.nn.Parameter(
        torch.empty(mib * MIB, dtype = torch.uint8, device = "meta"), requires_grad = False
    )
    return m


class _Pipe:
    def __init__(self):
        self.transformer = _meta(7448)
        self.text_encoder = _meta(8960)  # fp8 Qwen3-VL as loaded
        self.vae = _meta(645)
        self.components = {
            "transformer": self.transformer,
            "text_encoder": self.text_encoder,
            "vae": self.vae,
        }


def _refine_as_loaded(pipe, plan):
    """The expression diffusion.py assigns to ``refined_plan`` after the load, evaluated as written."""
    for node in ast.walk(ast.parse(DIFFUSION_SRC)):
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and getattr(node.targets[0], "id", None) == "refined_plan"
        ):
            scope = dict(vars(diffusion))
            scope.update(pipe = pipe, plan = plan)
            return eval(compile(ast.Expression(node.value), "<refine>", "eval"), scope)
    raise AssertionError("refined_plan assignment not found in diffusion.py")


@pytest.fixture
def no_torchao(monkeypatch):
    monkeypatch.setattr(dm, "_pipe_denoisers_hold_torchao", lambda pipe: False)


@pytest.mark.parametrize("sizes", [SIZES_AUTO, SIZES_EXPLICIT_INT8], ids = ["auto_quant", "int8"])
def test_balanced_16gb_does_not_keep_the_9gib_encoder_resident(no_torchao, sizes):
    plan = _plan(15976, 16376, "balanced", sizes)
    assert (
        plan.offload_policy == OFFLOAD_GROUP and not plan.stream_text_encoders
    )  # the plan main ships
    final = _refine_as_loaded(_Pipe(), plan)
    encoder_resident = final.offload_policy == OFFLOAD_GROUP and not final.stream_text_encoders
    assert not encoder_resident


@pytest.mark.parametrize("free,total", [(11888, 12288), (7788, 8188)], ids = ["12GB", "8GB"])
def test_balanced_small_cards_leave_group_offload(no_torchao, free, total):
    final = _refine_as_loaded(_Pipe(), _plan(free, total, "balanced", SIZES_EXPLICIT_INT8))
    assert final.offload_policy != OFFLOAD_GROUP


def test_balanced_24gb_keeps_its_placement(no_torchao):
    plan = _plan(24176, 24576, "balanced", SIZES_EXPLICIT_INT8)
    final = _refine_as_loaded(_Pipe(), plan)
    assert final.offload_policy == OFFLOAD_GROUP and not final.stream_text_encoders
