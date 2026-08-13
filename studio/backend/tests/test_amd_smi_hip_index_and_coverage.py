# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The amd-smi VRAM branch must answer in HIP's index space, for every visible device.

Two index spaces exist on an AMD host and they are not the same number. amd-smi's
``gpu`` id is "an enumeration index assigned in discovery order" over the KFD/sysfs
view; HIP's id is what ``HIP_VISIBLE_DEVICES`` names, what torch reports as
``cuda:N``, and what ``_get_gpu_memory``'s callers feed back to the llama-server
child. AMD ships the mapping between them as ``amd-smi list -e`` (``hip_id``, ROCm
6.4.0+, ``amdsmi_get_gpu_enumeration_info``), whose whole documented purpose is
"mapping physical-to-logical GPU IDs"; the library derives it from the KFD node id,
not from the discovery order, and the two disagree on real hardware (MI350X in
SPX/NPS1). Answering with the wrong one associates a card's VRAM with another
card's id, so the planner pins the wrong GPU.

amd-smi also enumerates every card regardless of ``ROCR_VISIBLE_DEVICES`` /
``HIP_VISIBLE_DEVICES`` (it reads KFD directly), so the branch has to apply the mask
itself, and a mask it cannot read is not the same thing as no mask at all.

Third: the branch either answers for every visible device or declines. A subset is
indistinguishable from a complete answer to the caller, which reads a non-empty list
as the final word and never asks torch.

torch, ROCm detection and amd-smi are all mocked; this repository has no AMD GPU.
"""

from __future__ import annotations

import subprocess
import sys
import types

import pytest

from core.inference.llama_cpp import LlamaCppBackend
from utils import hardware
from utils.hardware import amd


@pytest.fixture
def rocm(monkeypatch):
    """A ROCm host whose nvidia-smi probe finds nothing, with no APUs and no mask."""
    monkeypatch.setattr(LlamaCppBackend, "_torch_is_rocm", staticmethod(lambda torch: True))
    monkeypatch.setattr(
        LlamaCppBackend, "_rocm_unified_memory_gpu_ids", staticmethod(lambda: set())
    )
    for _var in ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"):
        monkeypatch.delenv(_var, raising = False)


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


def _enumeration(hip_by_gpu: dict):
    """``amd-smi list -e`` output: the amd-smi gpu id -> HIP id mapping."""
    return [{"gpu": gpu, "hip_id": hip} for gpu, hip in hip_by_gpu.items()]


def _fake_amd_smi(metric, enumeration = None):
    """Stub amd-smi: ``metric`` returns the VRAM rows, ``list`` the id mapping."""

    def _run(*args, **kwargs):
        if args and args[0] == "list":
            return enumeration
        return metric

    return _run


class TestTheAmdSmiIdIsTranslatedToHip:
    """Defect A: the branch keyed rows by amd-smi's gpu id and compared that number
    against HIP visibility tokens."""

    def test_a_reordered_host_reports_each_cards_memory_under_its_hip_id(self, rocm, monkeypatch):
        """amd-smi gpu 1 is HIP 0 here. Keyed by amd-smi's number, the 96 GiB card's
        memory lands on HIP id 1 and the planner pins the 16 GiB card for a model
        only the big one fits."""
        monkeypatch.setattr(
            amd,
            "_run_amd_smi",
            _fake_amd_smi(
                _payload((0, 1024, 16384), (1, 2048, 98304)),
                _enumeration({0: 1, 1: 0}),
            ),
        )
        assert LlamaCppBackend._get_gpu_memory_amd_smi() == [
            (0, 96256, 98304),
            (1, 15360, 16384),
        ]

    def test_the_mask_is_matched_in_hip_space(self, rocm, monkeypatch):
        """HIP_VISIBLE_DEVICES=0 names the 96 GiB card, which amd-smi calls gpu 1."""
        monkeypatch.setenv("HIP_VISIBLE_DEVICES", "0")
        monkeypatch.setattr(
            amd,
            "_run_amd_smi",
            _fake_amd_smi(
                _payload((0, 1024, 16384), (1, 2048, 98304)),
                _enumeration({0: 1, 1: 0}),
            ),
        )
        assert LlamaCppBackend._get_gpu_memory_amd_smi() == [(0, 96256, 98304)]

    def test_a_multi_gpu_host_declines_when_the_mapping_is_unavailable(self, rocm, monkeypatch):
        """amd-smi before ROCm 6.4.0 has no ``list -e``. Identity is an assumption
        there, not a fact, so hand the host to torch."""
        monkeypatch.setattr(
            amd,
            "_run_amd_smi",
            _fake_amd_smi(_payload((0, 1024, 16384), (1, 2048, 98304)), None),
        )
        assert LlamaCppBackend._get_gpu_memory_amd_smi() == []

    def test_a_partial_mapping_declines(self, rocm, monkeypatch):
        """``hip_id`` is "N/A" when the library cannot read the device's KFD node."""
        monkeypatch.setattr(
            amd,
            "_run_amd_smi",
            _fake_amd_smi(
                _payload((0, 1024, 16384), (1, 2048, 98304)),
                [{"gpu": 0, "hip_id": 0}, {"gpu": 1, "hip_id": "N/A"}],
            ),
        )
        assert LlamaCppBackend._get_gpu_memory_amd_smi() == []

    def test_a_single_gpu_host_needs_no_mapping(self, rocm, monkeypatch):
        """One card on both sides forces the mapping, so the common ROCm desktop
        keeps the context saving on any amd-smi version."""
        monkeypatch.setattr(amd, "_run_amd_smi", _fake_amd_smi(_payload((0, 4096, 24576)), None))
        assert LlamaCppBackend._get_gpu_memory_amd_smi() == [(0, 20480, 24576)]


class TestAnUnreadableMaskDeclines:
    """Defect A, second half: ``_resolve_visible_physical_ids`` returns None both for
    "no mask" and for a mask it cannot parse. ROCr accepts UUID tokens, so reading
    the second as the first offers a deliberately hidden card for ranking."""

    def test_a_uuid_rocr_mask_is_not_read_as_no_mask(self, rocm, monkeypatch):
        monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "GPU-a1b2c3d4e5f60718")
        monkeypatch.setattr(
            amd,
            "_run_amd_smi",
            _fake_amd_smi(
                _payload((0, 1024, 16384), (1, 2048, 98304)),
                _enumeration({0: 0, 1: 1}),
            ),
        )
        assert LlamaCppBackend._get_gpu_memory_amd_smi() == []

    def test_an_empty_mask_still_means_no_gpus(self, rocm, monkeypatch):
        monkeypatch.setenv("HIP_VISIBLE_DEVICES", "")
        monkeypatch.setattr(amd, "_run_amd_smi", _fake_amd_smi(_payload((0, 1024, 16384)), None))
        assert LlamaCppBackend._get_gpu_memory_amd_smi() == []


class TestAPartialAnswerDefersToTorch:
    """Defect B: a row amd-smi cannot size is dropped, and what is left is a
    non-empty list the caller takes as the whole host.

    The field shape: an APU beside a dGPU, where the shared pool reports total 0.
    ``_get_gpu_memory`` returns one device, so the "fewer than 2 usable GPUs" arm
    disables tensor parallelism on a host the torch branch reports as two.
    """

    @pytest.fixture
    def apu_plus_dgpu(self, rocm, monkeypatch):
        monkeypatch.setattr(
            amd,
            "_run_amd_smi",
            _fake_amd_smi(
                _payload((0, 512, 0), (1, 1024, 8192)),
                _enumeration({0: 0, 1: 1}),
            ),
        )

    def test_the_branch_declines_rather_than_answering_for_one_card(self, apu_plus_dgpu):
        assert LlamaCppBackend._get_gpu_memory_amd_smi() == []

    def test_the_probe_still_sees_both_devices(self, apu_plus_dgpu, monkeypatch):
        """End to end: the torch branch answers, and both GPUs survive, so the
        tensor-parallel decision is the one main makes."""
        monkeypatch.setattr(
            LlamaCppBackend, "_is_vulkan_backend", staticmethod(lambda binary: False)
        )

        def _no_nvidia_smi(*args, **kwargs):
            raise FileNotFoundError("nvidia-smi")

        monkeypatch.setattr(subprocess, "run", _no_nvidia_smi)
        torch_mod = types.ModuleType("torch")
        torch_mod.cuda = types.SimpleNamespace(
            is_available = lambda: True,
            device_count = lambda: 2,
            mem_get_info = lambda *a: (0, 0),
        )
        monkeypatch.setitem(sys.modules, "torch", torch_mod)
        monkeypatch.setattr(
            hardware,
            "trusted_mem_get_info",
            lambda ordinal, *a, **k: (
                (16 * 1024**3, 32 * 1024**3) if ordinal == 0 else (7 * 1024**3, 8 * 1024**3)
            ),
        )
        assert LlamaCppBackend._get_gpu_memory() == [(0, 16384, 32768), (1, 7168, 8192)]

    def test_a_visible_device_amd_smi_never_enumerated_declines(self, rocm, monkeypatch):
        """The mask names two cards and amd-smi answered for one; the other is not
        "absent", it is unmeasured."""
        monkeypatch.setenv("HIP_VISIBLE_DEVICES", "0,1")
        monkeypatch.setattr(amd, "_run_amd_smi", _fake_amd_smi(_payload((0, 1024, 16384)), None))
        assert LlamaCppBackend._get_gpu_memory_amd_smi() == []
