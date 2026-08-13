# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""``_get_gpu_memory`` must not open a HIP context to read VRAM on ROCm.

``torch.cuda.mem_get_info`` creates a primary context the process never releases:
measured at 612 MiB on a B200 with CUDA 13, and reported in the 692-712 MiB range
elsewhere. ``get_device_properties`` and ``get_device_capability`` do not, which is
why the System tab already reads totals through properties.

NVIDIA hosts were already safe because the probe asks nvidia-smi first. ROCm hosts
were not: they fell straight through to torch, so a Studio backend serving GGUF
models paid a permanent ~700 MiB for a number it only reads, on a card whose models
run in a llama-server CHILD that then cannot use it. amd-smi answers the same
question from a subprocess.

The torch fallback stays for hosts with no smi tool at all. Callers read ``[]`` as
"no GPU" -- ``_resolve_auto`` picks a backend from it, ``_gpu_available`` gates the
embedder, and the loader's fit drops to CPU -- so returning "unknown" there would
silently move inference off the GPU. Accepting the context is the lesser cost.

torch, ROCm detection and amd-smi are all mocked: this repository has no AMD GPU
and no ROCm CI, so none of this is a hardware validation.
"""

from __future__ import annotations

import json
import subprocess

import pytest

from core.inference.llama_cpp import LlamaCppBackend
from utils.hardware import amd


@pytest.fixture
def rocm(monkeypatch):
    """A ROCm host whose nvidia-smi probe finds nothing, with no APUs."""
    monkeypatch.setattr(LlamaCppBackend, "_torch_is_rocm", staticmethod(lambda torch: True))
    monkeypatch.setattr(
        LlamaCppBackend, "_rocm_unified_memory_gpu_ids", staticmethod(lambda: set())
    )
    monkeypatch.setattr(
        LlamaCppBackend, "_resolve_visible_physical_ids", staticmethod(lambda: None)
    )


def _payload(*gpus: tuple[int, int, int]):
    """amd-smi ``metric`` output for (id, used MiB, total MiB) triples."""
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


def test_amd_smi_supplies_free_and_total_without_torch(rocm, monkeypatch):
    monkeypatch.setattr(amd, "_run_amd_smi", lambda *a, **k: _payload((0, 4096, 24576)))
    assert LlamaCppBackend._get_gpu_memory_amd_smi() == [(0, 20480, 24576)]


def test_the_visibility_mask_is_honoured(rocm, monkeypatch):
    """amd-smi enumerates every card; a masked-out GPU must not be offered."""
    monkeypatch.setattr(LlamaCppBackend, "_resolve_visible_physical_ids", staticmethod(lambda: [1]))
    monkeypatch.setattr(
        amd, "_run_amd_smi", lambda *a, **k: _payload((0, 0, 24576), (1, 8192, 16384))
    )
    assert LlamaCppBackend._get_gpu_memory_amd_smi() == [(1, 8192, 16384)]


def test_a_unified_memory_apu_keeps_its_host_reserve(rocm, monkeypatch):
    """Same shared-pool treatment as the torch branch: total 0, reserve taken."""
    monkeypatch.setattr(LlamaCppBackend, "_rocm_unified_memory_gpu_ids", staticmethod(lambda: {0}))
    monkeypatch.setattr(LlamaCppBackend, "_available_system_memory_mib", staticmethod(lambda: 8192))
    monkeypatch.setattr(amd, "_run_amd_smi", lambda *a, **k: _payload((0, 0, 65536)))
    ((idx, free_mib, total_mib),) = LlamaCppBackend._get_gpu_memory_amd_smi()
    assert (idx, total_mib) == (0, 0)  # shared pool reports no VRAM total
    assert 0 < free_mib < 8192  # capped at system RAM, then reserved against


@pytest.mark.parametrize(
    "payload",
    [None, [], [{"gpu": 0, "power": {}}], [{"gpu": 0, "vram": {"used": 1, "total": 0}}]],
)
def test_an_unusable_amd_smi_falls_through(rocm, monkeypatch, payload):
    """Absent, empty, VRAM-less and zero-total answers all defer to the caller."""
    monkeypatch.setattr(amd, "_run_amd_smi", lambda *a, **k: payload)
    assert LlamaCppBackend._get_gpu_memory_amd_smi() == []


def test_the_branch_is_inert_off_rocm(monkeypatch):
    """An NVIDIA host must not spawn amd-smi; its numbers stay exactly as they were."""
    monkeypatch.setattr(LlamaCppBackend, "_torch_is_rocm", staticmethod(lambda torch: False))

    def _boom(*a, **k):
        raise AssertionError("amd-smi must not run off ROCm")

    monkeypatch.setattr(amd, "_run_amd_smi", _boom)
    assert LlamaCppBackend._get_gpu_memory_amd_smi() == []


def test_used_above_total_cannot_become_negative_free(rocm, monkeypatch):
    """A stale reading mid-reset clamps to 0 rather than offering negative VRAM."""
    monkeypatch.setattr(amd, "_run_amd_smi", lambda *a, **k: _payload((0, 9000, 8192)))
    assert LlamaCppBackend._get_gpu_memory_amd_smi() == [(0, 0, 8192)]


def test_free_is_derived_the_way_nvidia_smi_reports_it(rocm, monkeypatch):
    """Parity with the nvidia-smi branch: free == total - used, in MiB."""
    monkeypatch.setattr(amd, "_run_amd_smi", lambda *a, **k: _payload((0, 5000, 16384)))
    assert amd.get_gpu_vram_mib() == {0: (16384 - 5000, 16384)}


class TestTheArchGateAppliesToThisBranchToo:
    """#7624's gate lives in ``_get_gpu_memory``, and this branch returns before the
    torch one that carries it. An llama-server placement must lose a device the
    installed prebuilt has no kernels for here as well, or reading VRAM through
    amd-smi reintroduces the "device kernel image is invalid" crash.

    The #7624 shape: a gfx1101 dGPU beside a gfx1036 iGPU the build does not cover.
    """

    @pytest.fixture
    def mixed_host(self, rocm, tmp_path, monkeypatch):
        """Both cards visible to amd-smi; only the dGPU's arch is in the marker."""
        (tmp_path / "UNSLOTH_PREBUILT_INFO.json").write_text(
            json.dumps({"mapped_targets": ["gfx1100", "gfx1101"]}), encoding = "utf-8"
        )
        monkeypatch.setattr(
            LlamaCppBackend,
            "_rocm_arch_by_physical_id",
            staticmethod(lambda: {0: "gfx1101", 1: "gfx1036"}),
        )
        monkeypatch.setattr(
            amd, "_run_amd_smi", lambda *a, **k: _payload((0, 4096, 24576), (1, 1024, 16384))
        )
        return str(tmp_path / "build" / "bin" / "llama-server")

    def test_an_uncovered_device_is_dropped(self, mixed_host):
        assert LlamaCppBackend._get_gpu_memory_amd_smi(mixed_host, for_llama_server = True) == [
            (0, 20480, 24576)
        ]

    def test_and_kept_for_every_other_caller(self, mixed_host):
        # _resolve_auto's winner runs under PyTorch, which covers the card fine.
        assert LlamaCppBackend._get_gpu_memory_amd_smi(mixed_host) == [
            (0, 20480, 24576),
            (1, 15360, 16384),
        ]

    def test_the_gate_survives_the_full_probe(self, mixed_host, monkeypatch):
        """End to end through _get_gpu_memory, which is where the ordering trap is:
        a truthy amd-smi answer returns before the torch branch can gate it."""
        monkeypatch.setattr(
            LlamaCppBackend, "_is_vulkan_backend", staticmethod(lambda binary: False)
        )

        def _no_nvidia_smi(*args, **kwargs):
            raise FileNotFoundError("nvidia-smi")

        monkeypatch.setattr(subprocess, "run", _no_nvidia_smi)
        assert LlamaCppBackend._get_gpu_memory(mixed_host, for_llama_server = True) == [
            (0, 20480, 24576)
        ]

    def test_unknown_coverage_keeps_every_device(self, rocm, tmp_path, monkeypatch):
        """No install marker (source build, pre-#7624 install): fail open, the same
        way the torch branch does."""
        monkeypatch.setattr(
            LlamaCppBackend,
            "_rocm_arch_by_physical_id",
            staticmethod(lambda: {0: "gfx1101", 1: "gfx1036"}),
        )
        monkeypatch.setattr(
            amd, "_run_amd_smi", lambda *a, **k: _payload((0, 4096, 24576), (1, 1024, 16384))
        )
        binary = str(tmp_path / "build" / "bin" / "llama-server")
        assert len(LlamaCppBackend._get_gpu_memory_amd_smi(binary, for_llama_server = True)) == 2

    def test_a_device_with_no_reported_arch_is_kept(self, mixed_host, monkeypatch):
        """Torch could not describe it, so there is no evidence against it."""
        monkeypatch.setattr(
            LlamaCppBackend, "_rocm_arch_by_physical_id", staticmethod(lambda: {0: "gfx1101"})
        )
        assert len(LlamaCppBackend._get_gpu_memory_amd_smi(mixed_host, for_llama_server = True)) == 2
