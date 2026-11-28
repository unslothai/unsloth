# SPDX-License-Identifier: AGPL-3.0-only
# Unsloth - 2x faster, 60% less VRAM LLM finetuning
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
"""MultiLoraExperts picks its grouped mode only where Zoo's grouped GEMMs run as kernels. Zoo's Triton self-check passes
on every ROCm arch, but its policy sends LoRA GEMMs there only on Linux gfx1151 (or when forced): elsewhere, with
torch._grouped_mm failing its probe, the grouped mode looped per group over E * (U + 1) groups, slower than one adapter
at a time. CPU only: device capability, ROCm detection and the probes are stubbed."""

import ast
from pathlib import Path
import sys
import types

import pytest
import torch

moe_utils = pytest.importorskip("unsloth_zoo.temporary_patches.moe_utils")
moe_grouped_fp16 = pytest.importorskip("unsloth_zoo.temporary_patches.moe_grouped_fp16")


def _load():
    path = Path(__file__).resolve().parents[1] / "unsloth/kernels/multi_lora.py"
    tree = ast.parse(path.read_text(encoding = "utf-8"), filename = str(path))
    keep = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in (
            "_zoo_moe_utils",
            "_triton_routes_lora",
            "_grouped_supported",
        ):
            keep.append(node)
        elif isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id in ("_HALF", "_ZOO_GROUPED") for t in node.targets
        ):
            keep.append(node)
    tree.body = keep
    mod = types.ModuleType("_multi_lora_grouped_policy_source")
    mod.__dict__.update(torch = torch)
    exec(compile(tree, str(path), "exec"), mod.__dict__)
    return mod


ml = _load()


@pytest.fixture
def stubbed(monkeypatch):
    if moe_grouped_fp16.triton is None or moe_utils._GROUPED_MM_TRITON_OP is None:
        pytest.fail("needs Triton and torch.library.custom_op for Zoo's grouped GEMM policy")
    for name in (
        "UNSLOTH_MOE_GROUPED_TRITON",
        "UNSLOTH_DISABLE_MOE_TRITON",
        "UNSLOTH_MOE_GROUPED_TRITON_MAX_ROWS",
    ):
        monkeypatch.delenv(name, raising = False)
    monkeypatch.setattr(moe_utils, "_TRITON_GROUPED_MM_POLICY", {})
    monkeypatch.setattr(moe_utils, "_grouped_mm_supported_on", lambda device: False)
    monkeypatch.setattr(moe_grouped_fp16, "triton_grouped_available", lambda device: True)

    def setup(
        rocm,
        cap,
        platform = "linux",
    ):
        monkeypatch.setattr(moe_utils, "is_rocm_torch", lambda _torch: rocm)
        monkeypatch.setattr(moe_utils, "_triton_grouped_mm_capability", lambda index: cap)
        monkeypatch.setattr(sys, "platform", platform)
        moe_utils._TRITON_GROUPED_MM_POLICY.clear()

    return setup


DEV = torch.device("cuda", 0)


@pytest.mark.parametrize(
    "cap, platform",
    [((9, 4), "linux"), ((11, 0), "linux"), ((12, 0), "linux"), ((11, 5), "win32")],
)
def test_rocm_arch_without_triton_lora_route_runs_one_adapter_at_a_time(stubbed, cap, platform):
    stubbed(rocm = True, cap = cap, platform = platform)
    assert moe_utils._triton_grouped_mm_max_rows(0, "lora") == 0
    assert ml._grouped_supported(DEV, torch.bfloat16) is False


def test_rocm_gfx1151_linux_keeps_grouped_mode(stubbed):
    stubbed(rocm = True, cap = (11, 5))
    assert ml._grouped_supported(DEV, torch.bfloat16) is True


def test_forced_triton_keeps_grouped_mode_on_any_rocm_arch(stubbed, monkeypatch):
    stubbed(rocm = True, cap = (9, 4))
    monkeypatch.setenv("UNSLOTH_MOE_GROUPED_TRITON", "1")
    assert ml._grouped_supported(DEV, torch.float16) is True


def test_triton_turned_off_falls_back_to_the_grouped_mm_probe(stubbed, monkeypatch):
    stubbed(rocm = False, cap = (8, 0))
    assert ml._grouped_supported(DEV, torch.bfloat16) is True
    monkeypatch.setenv("UNSLOTH_MOE_GROUPED_TRITON", "0")
    assert ml._grouped_supported(DEV, torch.bfloat16) is False
    monkeypatch.setattr(moe_utils, "_grouped_mm_supported_on", lambda device: True)
    assert ml._grouped_supported(DEV, torch.bfloat16) is True


@pytest.mark.parametrize("cap", [(7, 5), (8, 0), (8, 6), (8, 9), (12, 0)])
def test_cuda_triton_archs_unchanged(stubbed, cap):
    stubbed(rocm = False, cap = cap)
    assert ml._grouped_supported(DEV, torch.bfloat16) is True
