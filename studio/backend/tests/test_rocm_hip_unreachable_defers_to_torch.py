# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The amd-smi VRAM branch must not answer for a host where HIP cannot open a device.

amd-smi and HIP do not read the same thing. amd-smi reports the driver's inventory
over sysfs and libdrm's ``/dev/dri/renderD*``; HIP needs ``/dev/kfd`` and a matching
HSA runtime. The two come apart on real hosts: a container started with
``--device=/dev/dri`` but no ``--device=/dev/kfd`` (AMD's own container docs require
BOTH), and a torch wheel built against a ROCm the installed runtime does not match.
There amd-smi lists the card and ``hipGetDeviceCount`` returns 0, so
``torch.cuda.is_available()`` is False while ``_torch_is_rocm()`` stays True.

The llama-server child is a HIP process too, so it dies exactly where torch did.
Before this branch existed the torch fallback returned ``[]`` for such a host and the
load went to CPU, which is the right answer; the amd-smi branch returns first and
hands placement a device nothing can open.

Neither existing guard catches it, because both are themselves torch readers that
fail open: ``_rocm_unified_memory_gpu_ids`` returns ``set()`` and
``_rocm_arch_by_physical_id`` returns ``{}`` the moment ``is_available()`` is False.
So an APU on such a host is not even recognised as one, and amd-smi's dedicated
carve-out is accepted as the whole pool.

Gating costs nothing this path was not already paying: ``_rocm_unified_memory_gpu_ids``
calls ``is_available()`` and ``get_device_properties()`` further down the same
function. Neither creates a primary context (measured: no compute-apps entry, against
612 MiB for ``mem_get_info``), because ``is_available()`` is ``hipGetDeviceCount``
and not ``_lazy_init``.

torch, ROCm detection and amd-smi are all mocked; this repository has no AMD GPU.
"""

from __future__ import annotations

import subprocess
import sys
import types

import pytest

from core.inference.llama_cpp import LlamaCppBackend
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


@pytest.fixture
def rocm(monkeypatch):
    """A ROCm host with one card, no mask, and an amd-smi that answers.

    The three mask vars are cleared from the environment as well as patched: the
    probe asks whether a mask is SET before asking what it resolves to, so the
    shell's own CUDA_VISIBLE_DEVICES would otherwise read as a mask that resolves to
    nothing (#8662 for the same trap in the APU tests)."""
    monkeypatch.setattr(LlamaCppBackend, "_torch_is_rocm", staticmethod(lambda torch: True))
    for _var in ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"):
        monkeypatch.delenv(_var, raising = False)
    monkeypatch.setattr(amd, "_run_amd_smi", lambda *a, **k: _payload((0, 4096, 24576)))


def _fake_torch(monkeypatch, *, hip_up: bool):
    """A ROCm torch whose HIP either can or cannot open the card amd-smi just listed."""
    torch_mod = types.ModuleType("torch")
    torch_mod.version = types.SimpleNamespace(hip = "6.2.41134")
    torch_mod.cuda = types.SimpleNamespace(
        is_available = lambda: hip_up,
        device_count = lambda: 1 if hip_up else 0,
        get_device_properties = lambda _o: types.SimpleNamespace(
            name = "Radeon RX 7900 XTX", gcnArchName = "gfx1100"
        ),
    )
    monkeypatch.setitem(sys.modules, "torch", torch_mod)
    return torch_mod


class TestHipCannotOpenTheDevice:
    @pytest.fixture
    def hip_is_down(self, rocm, monkeypatch):
        return _fake_torch(monkeypatch, hip_up = False)

    def test_the_amd_smi_branch_declines(self, hip_is_down):
        assert LlamaCppBackend._get_gpu_memory_amd_smi() == []

    def test_and_the_whole_probe_reports_no_gpu(self, hip_is_down, monkeypatch):
        """End to end: the pre-branch answer for this host, which is CPU."""
        monkeypatch.setattr(
            LlamaCppBackend, "_is_vulkan_backend", staticmethod(lambda binary: False)
        )

        def _no_nvidia_smi(*args, **kwargs):
            raise FileNotFoundError("nvidia-smi")

        monkeypatch.setattr(subprocess, "run", _no_nvidia_smi)
        assert LlamaCppBackend._get_gpu_memory() == []

    def test_no_existing_guard_would_have_caught_it(self, hip_is_down):
        """Both are torch readers that fail open, which is why the gate is explicit."""
        assert LlamaCppBackend._rocm_unified_memory_gpu_ids() == set()
        assert LlamaCppBackend._rocm_arch_by_physical_id() == {}


def test_a_reachable_hip_still_answers_from_amd_smi(rocm, monkeypatch):
    """The gate must not cost the saving on a working host: no mem_get_info here."""
    _fake_torch(monkeypatch, hip_up = True)
    assert LlamaCppBackend._get_gpu_memory_amd_smi() == [(0, 20480, 24576)]
