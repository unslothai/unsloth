# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""amd-smi VRAM must match HIP's inventory or not answer: masks, device cgroups and pools can differ."""

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
    """Declare HIP's own inventory: how many devices it opens, and the total memory
    it reports per physical id. Empty totals mean torch could not describe the
    devices, which the probe fails open on."""
    monkeypatch.setattr(LlamaCppBackend, "_rocm_hip_device_count", staticmethod(lambda: count))
    monkeypatch.setattr(
        LlamaCppBackend,
        "_rocm_total_memory_mib_by_physical_id",
        staticmethod(lambda: dict(totals or {})),
    )


@pytest.fixture
def rocm(monkeypatch):
    """Clears every visibility variable, since a set CUDA mask would read as a mask resolving to nothing."""
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
    _hip_sees(monkeypatch, 2)


@pytest.fixture
def two_cards(rocm, monkeypatch):
    """A plain two-card host: amd-smi and HIP agree on both devices."""
    monkeypatch.setattr(
        amd,
        "_run_amd_smi",
        _fake_amd_smi(_payload((0, 4096, 24576), (1, 8192, 16384)), {0: 0, 1: 1}),
    )


class TestGpuDeviceOrdinalIsHonoured:
    """The resolver does not read GPU_DEVICE_ORDINAL, and HIP may ignore it, so the branch declines."""

    def test_a_masked_process_is_not_offered_the_hidden_card(self, two_cards, monkeypatch):
        """GPU_DEVICE_ORDINAL=1 is the mask this repository already models, in
        ``test_overlay_skips_under_gpu_device_ordinal``: physical GPU 1 as torch
        ordinal 0. The branch must not rank GPU 0 behind it."""
        monkeypatch.setenv("GPU_DEVICE_ORDINAL", "1")
        assert LlamaCppBackend._get_gpu_memory_amd_smi() == []

    def test_a_full_ordinal_list_defers_too(self, two_cards, monkeypatch):
        """Every card is still visible, but the variable also REORDERS them, so the
        ids this branch would hand the child are not the ones it named."""
        monkeypatch.setenv("GPU_DEVICE_ORDINAL", "1,0")
        assert LlamaCppBackend._get_gpu_memory_amd_smi() == []

    def test_an_unset_or_blank_variable_is_not_a_mask(self, two_cards, monkeypatch):
        assert LlamaCppBackend._get_gpu_memory_amd_smi() == [(0, 20480, 24576), (1, 8192, 16384)]
        monkeypatch.setenv("GPU_DEVICE_ORDINAL", "   ")
        assert LlamaCppBackend._gpu_device_ordinal_active() is False
        assert LlamaCppBackend._get_gpu_memory_amd_smi() != []


class TestTheInventoryMustBeTheOneHipOpens:
    """A count HIP does not agree with means something outside the visibility
    variables filtered the device set."""

    def test_a_device_cgroup_container_defers_to_torch(self, two_cards, monkeypatch):
        """One renderD node, no env var: amd-smi lists both host cards, HIP opens
        one. Answering would let the fit pin a GPU the container cannot access."""
        _hip_sees(monkeypatch, 1)
        assert LlamaCppBackend._get_gpu_memory_amd_smi() == []

    def test_the_container_still_gets_torchs_single_device(self, two_cards, monkeypatch):
        """End to end: the torch branch answers for the one device HIP opened."""
        _hip_sees(monkeypatch, 1)
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
            hardware, "trusted_mem_get_info", lambda *a, **k: (20 * 1024**3, 24 * 1024**3)
        )
        assert LlamaCppBackend._get_gpu_memory() == [(0, 20480, 24576)]

    def test_a_row_amd_smi_omitted_does_not_pass_for_a_one_card_host(self, rocm, monkeypatch):
        """amd-smi omitting a row leaves one row that passes the single-GPU shortcut, hiding the card."""
        monkeypatch.setattr(amd, "_run_amd_smi", _fake_amd_smi(_payload((0, 4096, 24576)), None))
        assert LlamaCppBackend._get_gpu_memory_amd_smi() == []

    def test_a_masked_subset_is_measured_against_the_mask(self, rocm, monkeypatch):
        """HIP_VISIBLE_DEVICES=1 hides a card from HIP, not from amd-smi, so one
        device each is agreement rather than a filtered inventory."""
        monkeypatch.setenv("HIP_VISIBLE_DEVICES", "1")
        _hip_sees(monkeypatch, 1)
        monkeypatch.setattr(
            amd,
            "_run_amd_smi",
            _fake_amd_smi(_payload((0, 4096, 24576), (1, 8192, 16384)), {0: 0, 1: 1}),
        )
        assert LlamaCppBackend._get_gpu_memory_amd_smi() == [(1, 8192, 16384)]


class TestAnApuTheClassifierMisses:
    """A Phoenix iGPU without is_integrated is classed discrete, so the APU deferral never fires."""

    @pytest.fixture
    def phoenix(self, rocm, monkeypatch):
        monkeypatch.setattr(amd, "_run_amd_smi", _fake_amd_smi(_payload((0, 128, 512)), None))
        _hip_sees(monkeypatch, 1, {0: 16384})

    def test_the_branch_declines_rather_than_reporting_the_carve_out(self, phoenix):
        assert LlamaCppBackend._get_gpu_memory_amd_smi() == []

    def test_the_probe_still_reports_the_pool_the_model_runs_in(self, phoenix, monkeypatch):
        """End to end: _get_gpu_memory keeps the torch branch's figures, which is
        what it returned before this branch existed."""
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
            hardware, "trusted_mem_get_info", lambda *a, **k: (12 * 1024**3, 16 * 1024**3)
        )
        assert LlamaCppBackend._get_gpu_memory() == [(0, 12288, 16384)]

    def test_a_partitioned_card_defers_the_other_way_round(self, rocm, monkeypatch):
        """The mismatch is symmetric: an MI300 partition reports a slice to HIP
        while amd-smi reports the whole card."""
        monkeypatch.setattr(amd, "_run_amd_smi", _fake_amd_smi(_payload((0, 4096, 196608)), None))
        _hip_sees(monkeypatch, 1, {0: 24576})
        assert LlamaCppBackend._get_gpu_memory_amd_smi() == []

    def test_two_readings_of_one_pool_are_accepted(self, rocm, monkeypatch):
        """amd-smi and HIP do not report a card's total to the byte (reserved pages,
        MB against MiB), so the comparison has the same 10% margin the System tab's
        overlay uses."""
        monkeypatch.setattr(amd, "_run_amd_smi", _fake_amd_smi(_payload((0, 4096, 24576)), None))
        _hip_sees(monkeypatch, 1, {0: 24400})
        assert LlamaCppBackend._get_gpu_memory_amd_smi() == [(0, 20480, 24576)]

    def test_a_device_torch_cannot_describe_fails_open(self, rocm, monkeypatch):
        """A device torch cannot describe is omitted, so missing evidence must not count against the
        branch."""
        monkeypatch.setattr(amd, "_run_amd_smi", _fake_amd_smi(_payload((0, 4096, 24576)), None))
        _hip_sees(monkeypatch, 1, {})
        assert LlamaCppBackend._get_gpu_memory_amd_smi() == [(0, 20480, 24576)]
