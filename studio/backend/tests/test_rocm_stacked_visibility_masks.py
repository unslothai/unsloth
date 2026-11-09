# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Stacked ROCm masks re-index each other, so the probe declines rather than guess a physical id."""

from __future__ import annotations

import subprocess
import sys
import types

import pytest

from core.inference.llama_cpp import LlamaCppBackend
from utils import hardware
from utils.hardware import amd


def _payload(*gpus: tuple[int, int, int]):
    """amd-smi ``metric`` output for (amd-smi gpu id, used MiB, total MiB) triples."""
    return [
        {
            "gpu": idx,
            "mem_usage": {
                "used_vram": {"value": used, "unit": "MB"},
                "total_vram": {"value": total, "unit": "MB"},
            },
        }
        for idx, used, total in gpus
    ]


def _fake_amd_smi(metric, hip_by_gpu = None):
    """Stub amd-smi: ``metric`` returns the VRAM rows, ``list -e`` the id mapping."""

    def _run(*args, **kwargs):
        if args and args[0] == "list":
            if hip_by_gpu is None:
                return None
            return [{"gpu": gpu, "hip_id": hip} for gpu, hip in hip_by_gpu.items()]
        return metric

    return _run


def _hip_sees(
    monkeypatch,
    count,
    totals = None,
):
    """Declare HIP's own inventory, which is not this host's: how many devices it
    opens, and the total memory it reports per physical id."""
    monkeypatch.setattr(LlamaCppBackend, "_rocm_hip_device_count", staticmethod(lambda: count))
    monkeypatch.setattr(
        LlamaCppBackend,
        "_rocm_total_memory_mib_by_physical_id",
        staticmethod(lambda: dict(totals or {})),
    )


@pytest.fixture
def three_identical_cards(monkeypatch):
    """Three same-model cards, so the total-memory check cannot tell the wrong card from the right one."""
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(LlamaCppBackend, "_torch_is_rocm", staticmethod(lambda torch: True))
    monkeypatch.setattr(LlamaCppBackend, "_rocm_hip_is_reachable", staticmethod(lambda: True))
    monkeypatch.setattr(
        LlamaCppBackend, "_rocm_unified_memory_gpu_ids", staticmethod(lambda: set())
    )
    for _var in (
        "HIP_VISIBLE_DEVICES",
        "ROCR_VISIBLE_DEVICES",
        "CUDA_VISIBLE_DEVICES",
        "GPU_DEVICE_ORDINAL",
    ):
        monkeypatch.delenv(_var, raising = False)
    monkeypatch.setattr(
        amd,
        "_run_amd_smi",
        _fake_amd_smi(
            _payload((0, 4096, 24576), (1, 23552, 24576), (2, 23552, 24576)),
            {0: 0, 1: 1, 2: 2},
        ),
    )
    _hip_sees(monkeypatch, 3, {0: 24576, 1: 24576, 2: 24576})


class TestStackedMasksAreNotResolvable:
    """Both layers filtering means the resolved ids are not the ids HIP opened."""

    def test_a_rocr_mask_under_a_hip_mask_defers_to_torch(self, three_identical_cards, monkeypatch):
        """Under ROCR 1,2, HIP 0 is physical GPU 1, so reading the HIP mask alone reports the wrong card."""
        monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "1,2")
        monkeypatch.setenv("HIP_VISIBLE_DEVICES", "0")
        _hip_sees(monkeypatch, 1, {0: 24576})
        assert LlamaCppBackend._get_gpu_memory_amd_smi() == []

    def test_the_cuda_twin_stacks_the_same_way(self, three_identical_cards, monkeypatch):
        """CUDA_VISIBLE_DEVICES is a HIP-layer twin on AMD, so it indexes the post-ROCr list too."""
        monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "1,2")
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
        _hip_sees(monkeypatch, 1, {0: 24576})
        assert LlamaCppBackend._get_gpu_memory_amd_smi() == []

    def test_an_empty_rocr_mask_under_a_hip_mask_defers(self, three_identical_cards, monkeypatch):
        """``ROCR_VISIBLE_DEVICES=""`` hides every agent (case A1 in ROCR-Runtime's
        ``amd_filter_device.h``), so the HIP mask indexes an empty list. Set is set:
        the value is not what makes the two stack."""
        monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "")
        monkeypatch.setenv("HIP_VISIBLE_DEVICES", "0")
        _hip_sees(monkeypatch, 0, {})
        assert LlamaCppBackend._get_gpu_memory_amd_smi() == []

    def test_the_load_still_gets_torchs_figures(self, three_identical_cards, monkeypatch):
        """End to end: deferring is not dropping the GPU. The torch branch answers
        for the one device HIP opened, in HIP's own index space, which is what
        _get_gpu_memory returned before the amd-smi branch existed."""
        monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "1,2")
        monkeypatch.setenv("HIP_VISIBLE_DEVICES", "0")
        _hip_sees(monkeypatch, 1, {0: 24576})
        monkeypatch.setattr(
            LlamaCppBackend, "_is_vulkan_backend", staticmethod(lambda binary: False)
        )

        def _no_nvidia_smi(*args, **kwargs):
            raise FileNotFoundError("nvidia-smi")

        monkeypatch.setattr(subprocess, "run", _no_nvidia_smi)
        torch_mod = types.ModuleType("torch")
        torch_mod.cuda = types.SimpleNamespace(
            is_available = lambda: True,
            device_count = lambda: 1,
            mem_get_info = lambda *a: (0, 0),
        )
        monkeypatch.setitem(sys.modules, "torch", torch_mod)
        monkeypatch.setattr(
            hardware, "trusted_mem_get_info", lambda *a, **k: (1024**3, 24 * 1024**3)
        )
        assert LlamaCppBackend._get_gpu_memory() == [(0, 1024, 24576)]


class TestOneMaskIsStillAnswerable:
    """The guard is about two layers filtering at once, not about the variables
    existing. A single mask maps to physical ids exactly as before."""

    def test_a_lone_rocr_mask_still_answers(self, three_identical_cards, monkeypatch):
        """No HIP layer above it, so the ROCR ids ARE the physical ids
        ``_resolve_visible_physical_ids`` reads."""
        monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "1,2")
        _hip_sees(monkeypatch, 2, {1: 24576, 2: 24576})
        assert LlamaCppBackend._get_gpu_memory_amd_smi() == [(1, 1024, 24576), (2, 1024, 24576)]

    def test_a_lone_hip_mask_still_answers(self, three_identical_cards, monkeypatch):
        """Nothing filtered below it, so its ordinals are physical ids too."""
        monkeypatch.setenv("HIP_VISIBLE_DEVICES", "0")
        _hip_sees(monkeypatch, 1, {0: 24576})
        assert LlamaCppBackend._get_gpu_memory_amd_smi() == [(0, 20480, 24576)]

    def test_hip_and_cuda_together_are_one_layer(self, three_identical_cards, monkeypatch):
        """Both name the HIP layer and clr reads whichever of them is set FIRST,
        never both, so they do not compose and the HIP value is the whole mask."""
        monkeypatch.setenv("HIP_VISIBLE_DEVICES", "0")
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2")
        _hip_sees(monkeypatch, 1, {0: 24576})
        assert LlamaCppBackend._get_gpu_memory_amd_smi() == [(0, 20480, 24576)]

    def test_windows_has_no_rocr_layer_to_stack(self, three_identical_cards, monkeypatch):
        """Windows has no ROCr layer, so a stray ROCR variable must not disable the branch there."""
        monkeypatch.setattr(sys, "platform", "win32")
        monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "1,2")
        monkeypatch.setenv("HIP_VISIBLE_DEVICES", "0")
        _hip_sees(monkeypatch, 1, {0: 24576})
        assert LlamaCppBackend._get_gpu_memory_amd_smi() == [(0, 20480, 24576)]
