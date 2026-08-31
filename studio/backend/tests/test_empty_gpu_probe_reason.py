# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""_explain_empty_gpu_probe: why _get_gpu_memory came back empty.

From a real report. A user's model ran on the CPU and their whole log said:

    GGUF size: 16.4 GB, ... GPUs free: [], selected: None, --fit: on
    Load mode: the whole load (34.3 GB) fits without paging

which records the verdict and none of the reason, so the question actually being
asked -- why did my model go to system RAM -- could not be answered from it. The
probe returns [] for at least six unrelated reasons and every one lands on the CPU;
these tests pin that each names itself, and that the explainer never creates the HIP
context the amd-smi branch exists to avoid.
"""

import sys
import types

import pytest

from core.inference.llama_cpp import LlamaCppBackend


def _fake_torch(
    *,
    available = True,
    count = 2,
    rocm = True,
):
    cuda = types.SimpleNamespace(
        is_available = lambda: available,
        device_count = lambda: count,
        mem_get_info = lambda i: (_ for _ in ()).throw(
            AssertionError("mem_get_info must not be called: it creates a HIP context")
        ),
    )
    return types.SimpleNamespace(
        cuda = cuda,
        version = types.SimpleNamespace(hip = "6.2" if rocm else None, cuda = None if rocm else "12.4"),
    )


@pytest.fixture(autouse = True)
def _not_vulkan(monkeypatch):
    monkeypatch.setattr(LlamaCppBackend, "_is_vulkan_backend", staticmethod(lambda b = None: False))
    monkeypatch.setattr(
        LlamaCppBackend, "_find_llama_server_binary", staticmethod(lambda: "/opt/llama-server")
    )


def _clear_masks(monkeypatch):
    for var in (
        "CUDA_VISIBLE_DEVICES",
        "HIP_VISIBLE_DEVICES",
        "ROCR_VISIBLE_DEVICES",
        "GPU_DEVICE_ORDINAL",
    ):
        monkeypatch.delenv(var, raising = False)


class TestItNamesTheCause:
    def test_a_vulkan_build_says_so(self, monkeypatch):
        monkeypatch.setattr(
            LlamaCppBackend, "_is_vulkan_backend", staticmethod(lambda b = None: True)
        )
        assert "Vulkan" in LlamaCppBackend._explain_empty_gpu_probe()

    def test_no_torch_at_all(self, monkeypatch):
        _clear_masks(monkeypatch)
        monkeypatch.setitem(sys.modules, "torch", None)
        assert "torch is not importable" in LlamaCppBackend._explain_empty_gpu_probe()

    def test_torch_without_a_usable_device(self, monkeypatch):
        _clear_masks(monkeypatch)
        monkeypatch.setitem(sys.modules, "torch", _fake_torch(available = False))
        assert "no usable CUDA or HIP device" in LlamaCppBackend._explain_empty_gpu_probe()

    def test_zero_devices_enumerated(self, monkeypatch):
        _clear_masks(monkeypatch)
        monkeypatch.setitem(sys.modules, "torch", _fake_torch(count = 0))
        assert "enumerated 0 devices" in LlamaCppBackend._explain_empty_gpu_probe()

    def test_the_arch_gate_dropping_every_card(self, monkeypatch):
        """The #7624 case: a build with no kernels for the cards actually present."""
        _clear_masks(monkeypatch)
        monkeypatch.setitem(sys.modules, "torch", _fake_torch())
        monkeypatch.setattr(
            LlamaCppBackend,
            "_installed_llama_gfx_archs",
            staticmethod(lambda b = None: frozenset({"gfx1100"})),
        )
        monkeypatch.setattr(
            LlamaCppBackend,
            "_rocm_arch_by_physical_id",
            staticmethod(lambda: {0: "gfx1030", 1: "gfx1030"}),
        )
        msg = LlamaCppBackend._explain_empty_gpu_probe()
        assert "arch gate dropped every device" in msg
        assert "gfx1100" in msg and "gfx1030" in msg, "name both sides or it is not actionable"

    def test_rocm_devices_present_but_both_probes_declined(self, monkeypatch):
        """The reported host: cards visible to torch, amd-smi and the fallback empty."""
        _clear_masks(monkeypatch)
        monkeypatch.setitem(sys.modules, "torch", _fake_torch())
        monkeypatch.setattr(
            LlamaCppBackend, "_installed_llama_gfx_archs", staticmethod(lambda b = None: None)
        )
        msg = LlamaCppBackend._explain_empty_gpu_probe()
        assert "both declined" in msg and "3" not in msg
        assert "2 ROCm device" in msg

    def test_a_covered_arch_does_not_blame_the_gate(self, monkeypatch):
        _clear_masks(monkeypatch)
        monkeypatch.setitem(sys.modules, "torch", _fake_torch())
        monkeypatch.setattr(
            LlamaCppBackend,
            "_installed_llama_gfx_archs",
            staticmethod(lambda b = None: frozenset({"gfx1030"})),
        )
        monkeypatch.setattr(
            LlamaCppBackend, "_rocm_arch_by_physical_id", staticmethod(lambda: {0: "gfx1030"})
        )
        assert "arch gate" not in LlamaCppBackend._explain_empty_gpu_probe()


class TestItReportsTheVisibilityMask:
    @pytest.mark.parametrize(
        "var", ["CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "GPU_DEVICE_ORDINAL"]
    )
    def test_an_empty_mask_is_called_out(self, monkeypatch, var):
        """A container or a stale mask hides every device, and looks identical to
        having no GPU unless the mask is printed."""
        _clear_masks(monkeypatch)
        monkeypatch.setenv(var, "")
        monkeypatch.setitem(sys.modules, "torch", _fake_torch(available = False))
        assert f"{var} is empty" in LlamaCppBackend._explain_empty_gpu_probe()

    def test_a_set_mask_is_quoted(self, monkeypatch):
        _clear_masks(monkeypatch)
        monkeypatch.setenv("HIP_VISIBLE_DEVICES", "3")
        monkeypatch.setitem(sys.modules, "torch", _fake_torch(available = False))
        assert "HIP_VISIBLE_DEVICES='3'" in LlamaCppBackend._explain_empty_gpu_probe()

    def test_no_mask_adds_nothing(self, monkeypatch):
        _clear_masks(monkeypatch)
        monkeypatch.setitem(sys.modules, "torch", _fake_torch(available = False))
        assert "VISIBLE_DEVICES" not in LlamaCppBackend._explain_empty_gpu_probe()


class TestItIsSafeToCallOnAnyPath:
    def test_it_never_touches_mem_get_info(self, monkeypatch):
        """Reading memory creates a HIP primary context the backend never gives
        back (~700 MiB), which is the whole reason the amd-smi branch exists. The
        fake raises if it is called, so reaching a verdict here proves it was not."""
        _clear_masks(monkeypatch)
        monkeypatch.setitem(sys.modules, "torch", _fake_torch())
        monkeypatch.setattr(
            LlamaCppBackend, "_installed_llama_gfx_archs", staticmethod(lambda b = None: None)
        )
        assert LlamaCppBackend._explain_empty_gpu_probe()

    def test_a_raising_helper_still_yields_a_reason(self, monkeypatch):
        def boom(binary = None):
            raise RuntimeError("marker unreadable")

        _clear_masks(monkeypatch)
        monkeypatch.setitem(sys.modules, "torch", _fake_torch())
        monkeypatch.setattr(LlamaCppBackend, "_installed_llama_gfx_archs", staticmethod(boom))
        msg = LlamaCppBackend._explain_empty_gpu_probe()
        assert "could not be determined" in msg and "RuntimeError" in msg

    def test_it_is_never_empty(self, monkeypatch):
        """An unexplained empty list is the defect; an empty explanation restores it."""
        _clear_masks(monkeypatch)
        for torch_mod in (None, _fake_torch(available = False), _fake_torch(count = 0)):
            monkeypatch.setitem(sys.modules, "torch", torch_mod)
            assert LlamaCppBackend._explain_empty_gpu_probe().strip()


class TestMacOSIsNotACpuVerdict:
    """This probe reads CUDA and HIP only, so it is empty on every Mac -- including an
    Apple Silicon one whose model is about to run entirely on the Metal GPU. There is a
    whole "No GPU is enumerated on Metal" branch in load_model for that state. Warning
    there would tell every Mac user the opposite of what happens."""

    @pytest.mark.parametrize("torch_state", ["absent", "no_cuda", "rocm_like"])
    def test_the_reason_names_metal_rather_than_blaming_torch(self, monkeypatch, torch_state):
        _clear_masks(monkeypatch)
        monkeypatch.setattr(sys, "platform", "darwin")
        monkeypatch.setitem(
            sys.modules,
            "torch",
            None if torch_state == "absent" else _fake_torch(available = torch_state != "no_cuda"),
        )
        msg = LlamaCppBackend._explain_empty_gpu_probe()
        assert "Metal" in msg
        assert "torch" not in msg, "on macOS the probe's blind spot is the answer, not torch"

    def test_the_load_site_does_not_warn_on_macos(self):
        """Source-level, because load_model cannot be driven from a unit test. The
        simulation in temp/simgpu mirrors this predicate across the platform matrix."""
        import inspect

        from core.inference import llama_cpp as mod

        src = inspect.getsource(mod.LlamaCppBackend.load_model)
        at = src.find("No GPU was available for this load")
        assert at != -1, "the empty-probe warning is gone"
        guard = src[max(0, at - 900) : at]
        assert 'sys.platform != "darwin"' in guard, "macOS must be excluded from this warning"
        assert "not _arch_gate_forced_cpu" in guard, "the arch gate warns better; do not double up"
