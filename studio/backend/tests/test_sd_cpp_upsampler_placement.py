# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import types

import pytest

import utils.hardware as hardware
from core.inference import diffusion_engine_router
from core.inference import flux3_prompt_upsampler as pup
from core.inference.sd_cpp_backend import _SdState


def _native_state(**overrides):
    fields = dict(
        repo_id = "unsloth/FLUX.3-klein-GGUF",
        base_repo = "unsloth/FLUX.3-klein",
        family = types.SimpleNamespace(name = "flux3-klein"),
        device = "cuda",
        files = types.SimpleNamespace(),
    )
    fields.update(overrides)
    return _SdState(**fields)


def _activate(monkeypatch, state, visible):
    monkeypatch.setattr(
        diffusion_engine_router,
        "get_active_diffusion_engine",
        lambda: types.SimpleNamespace(_state = state),
    )
    monkeypatch.setattr(hardware, "get_parent_visible_gpu_ids", lambda: list(visible))


@pytest.mark.parametrize(
    "visible, physical, expected",
    [([0, 1], 1, 1), ([0, 1], 0, 0), ([3, 5], 5, 1), ([3, 5], 4, None)],
)
def test_upsampler_reads_the_native_sd_cpp_card(monkeypatch, visible, physical, expected):
    _activate(monkeypatch, _native_state(physical_gpu_id = physical), visible)
    assert pup._resident_cuda_ordinal() == expected


def test_unresolved_or_off_torch_native_placement_stays_unplaced(monkeypatch):
    _activate(monkeypatch, _native_state(physical_gpu_id = None), [0, 1])
    assert pup._resident_cuda_ordinal() is None
    _activate(monkeypatch, _native_state(physical_gpu_id = 1, off_torch_device = "Vulkan1"), [0, 1])
    assert pup._resident_cuda_ordinal() is None


def test_llama_server_is_masked_to_the_native_card(monkeypatch):
    torch = pytest.importorskip("torch")
    _activate(monkeypatch, _native_state(physical_gpu_id = 5), [3, 5])
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3,5")
    monkeypatch.delenv("HIP_VISIBLE_DEVICES", raising = False)
    monkeypatch.delenv("LLAMA_ARG_DEVICE", raising = False)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    from core.inference.llama_cpp import LlamaCppBackend

    monkeypatch.setattr(LlamaCppBackend, "_torch_is_rocm", staticmethod(lambda _torch: False))
    assert pup._child_device_mask(pup._resident_cuda_ordinal()) == {"CUDA_VISIBLE_DEVICES": "5"}
