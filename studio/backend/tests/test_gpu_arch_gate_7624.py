# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""
Regression tests for #7624 / #7669: multi-GPU auto-selection on ROCm must not
pick a device the installed llama.cpp prebuilt has no kernels for.

Covers the pieces of the fix: _installed_llama_gfx_archs (mapped_targets from
the UNSLOTH_PREBUILT_INFO.json install marker, via llama_cpp_freshness),
_rocm_arch_by_physical_id, the opt-in per-device gate in _get_gpu_memory's torch
fallback, the "device kernel image is invalid" crash marker, and the retry set
that crash falls back to.

Also pins the two things the gate must NOT do: filter the probe for torch
callers (the RAG sentence-transformers pick runs under PyTorch, where an
unsupported-by-llama.cpp device is usually still fine), and displace the
unified-memory APU accounting that shares this loop.
"""

import json
import subprocess
import sys
import types

import pytest

# Imported eagerly so the fake torch below cannot be in place when the real
# module graph is first loaded.
from core.inference.llama_cpp import _IGPU_HOST_RESERVE_MIB, LlamaCppBackend
from core.training.worker import _rocm_classify_unified_memory  # noqa: F401


def _binary_with_marker(tmp_path, payload):
    """Lay out <root>/UNSLOTH_PREBUILT_INFO.json with a binary path below it,
    matching the managed install layout the marker walk-up covers."""
    (tmp_path / "UNSLOTH_PREBUILT_INFO.json").write_text(json.dumps(payload), encoding = "utf-8")
    return str(tmp_path / "build" / "bin" / "llama-server")


class TestInstalledLlamaGfxArchs:
    def test_reads_mapped_targets(self, tmp_path):
        binary = _binary_with_marker(
            tmp_path, {"mapped_targets": ["gfx1100", "GFX1101", "gfx1102:xnack-"]}
        )
        archs = LlamaCppBackend._installed_llama_gfx_archs(binary)
        assert archs == frozenset({"gfx1100", "gfx1101", "gfx1102"})

    def test_no_marker_is_unknown(self, tmp_path):
        # Source build / custom-linked dir: no marker anywhere above the binary.
        assert LlamaCppBackend._installed_llama_gfx_archs(str(tmp_path / "llama-server")) is None

    def test_no_binary_is_unknown(self, monkeypatch):
        monkeypatch.setattr(
            LlamaCppBackend, "_find_llama_server_binary", staticmethod(lambda: None)
        )
        assert LlamaCppBackend._installed_llama_gfx_archs() is None

    def test_pre_7669_install_is_unknown(self, tmp_path):
        # Older installs have no mapped_targets key: fail open.
        binary = _binary_with_marker(tmp_path, {"asset": "app-windows-x64-rocm-gfx110X.zip"})
        assert LlamaCppBackend._installed_llama_gfx_archs(binary) is None

    def test_empty_targets_is_unknown(self, tmp_path):
        # Non-ROCm bundles record []: fail open rather than drop every GPU.
        binary = _binary_with_marker(tmp_path, {"mapped_targets": []})
        assert LlamaCppBackend._installed_llama_gfx_archs(binary) is None

    def test_unreadable_marker_is_unknown(self, tmp_path, monkeypatch):
        # A raising marker read must answer "unknown", never propagate: it runs
        # inside the GPU probe, where an exception would drop every device.
        import utils.llama_cpp_freshness as freshness

        def _boom(_binary):
            raise OSError("marker read failed")

        monkeypatch.setattr(freshness, "read_install_marker", _boom)
        assert LlamaCppBackend._installed_llama_gfx_archs(str(tmp_path / "llama-server")) is None


class TestKernelImageInvalidMarker:
    def test_detects_rocm_arch_mismatch(self):
        tail = (
            "load_model: loading model 'x.gguf'\n"
            "E ROCm error: device kernel image is invalid\n"
            "E   current device: 0, in function ggml_cuda_kernel_launch"
        )
        assert LlamaCppBackend._kernel_image_invalid(tail)

    def test_ignores_other_crashes(self):
        assert not LlamaCppBackend._kernel_image_invalid("out of memory")
        assert not LlamaCppBackend._kernel_image_invalid("")
        assert not LlamaCppBackend._kernel_image_invalid(None)


class _FakeProps:
    """hipDeviceProp_t stand-in exposing the canonical arch attribute."""

    _ATTR = "gcnArchName"

    def __init__(self, arch):
        setattr(self, self._ATTR, arch)


def _fake_torch(
    archs,
    free_mib,
    *,
    props_cls = _FakeProps,
    hip = "7.1.0",
    version_str = "",
):
    torch = types.ModuleType("torch")
    torch.version = types.SimpleNamespace() if hip is None else types.SimpleNamespace(hip = hip)
    torch.__version__ = version_str
    torch.cuda = types.SimpleNamespace(
        is_available = lambda: True,
        device_count = lambda: len(archs),
        mem_get_info = lambda o: (free_mib[o] * 1024 * 1024, 32 * 1024**3),
        get_device_properties = lambda o: props_cls(archs[o]),
    )
    return torch


@pytest.fixture
def rocm_probe_env(tmp_path, monkeypatch):
    """Force _get_gpu_memory down the torch/ROCm fallback, hermetically.
    The fake binary lives under tmp_path so a test can plant an install
    marker there (or not, for the unknown-coverage case)."""

    def _no_nvidia_smi(*args, **kwargs):
        raise FileNotFoundError("nvidia-smi")

    monkeypatch.setattr(subprocess, "run", _no_nvidia_smi)
    fake_binary = str(tmp_path / "build" / "bin" / "llama-server")
    monkeypatch.setattr(
        LlamaCppBackend, "_find_llama_server_binary", staticmethod(lambda: fake_binary)
    )
    for var in ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"):
        monkeypatch.delenv(var, raising = False)


class TestGpuArchGate:
    def test_igpu_dropped_when_arch_unsupported(self, tmp_path, monkeypatch, rocm_probe_env):
        # dGPU (gfx1101) + iGPU (gfx1036) whose shared-RAM "free memory"
        # outranks the dGPU's free VRAM -- the #7624 / #7669 shape. Only the
        # dGPU may survive enumeration.
        _binary_with_marker(
            tmp_path, {"mapped_targets": ["gfx1100", "gfx1101", "gfx1102", "gfx1103"]}
        )
        monkeypatch.setitem(
            sys.modules, "torch", _fake_torch(["gfx1101", "gfx1036"], [12049, 12176])
        )
        assert LlamaCppBackend._get_gpu_free_memory(for_llama_server = True) == [(0, 12049)]

    def test_unknown_coverage_keeps_all_devices(self, tmp_path, monkeypatch, rocm_probe_env):
        # No install marker (source build / custom link): behavior unchanged.
        monkeypatch.setitem(
            sys.modules, "torch", _fake_torch(["gfx1101", "gfx1036"], [12049, 12176])
        )
        assert LlamaCppBackend._get_gpu_free_memory(for_llama_server = True) == [
            (0, 12049),
            (1, 12176),
        ]

    def test_unknown_device_arch_fails_open(self, tmp_path, monkeypatch, rocm_probe_env):
        # A device torch can't describe is kept, never silently dropped.
        _binary_with_marker(tmp_path, {"mapped_targets": ["gfx1101"]})
        monkeypatch.setitem(sys.modules, "torch", _fake_torch(["gfx1101", ""], [12049, 12176]))
        assert LlamaCppBackend._get_gpu_free_memory(for_llama_server = True) == [
            (0, 12049),
            (1, 12176),
        ]

    @pytest.mark.parametrize("attr", ["gcn_arch_name", "arch_name", "gfx_arch_name"])
    def test_alternate_arch_attribute_spellings(self, attr, tmp_path, monkeypatch, rocm_probe_env):
        # AMD SDK / Radeon wheels may omit the canonical gcnArchName. Reading
        # only that attribute would leave the arch map empty and fail the gate
        # open, which is exactly the crash this PR exists to prevent.
        props_cls = type("_AltProps", (_FakeProps,), {"_ATTR": attr})
        _binary_with_marker(tmp_path, {"mapped_targets": ["gfx1101"]})
        monkeypatch.setitem(
            sys.modules,
            "torch",
            _fake_torch(["gfx1101", "gfx1036"], [12049, 12176], props_cls = props_cls),
        )
        assert LlamaCppBackend._rocm_arch_by_physical_id() == {0: "gfx1101", 1: "gfx1036"}
        assert LlamaCppBackend._get_gpu_free_memory(for_llama_server = True) == [(0, 12049)]

    def test_amd_sdk_wheel_without_version_hip_is_gated(
        self, tmp_path, monkeypatch, rocm_probe_env
    ):
        # AMD SDK / Radeon ROCm wheels can leave torch.version.hip unset while
        # __version__ still identifies ROCm. A bare version.hip test would skip
        # the gate there; the shared _torch_is_rocm predicate does not.
        _binary_with_marker(tmp_path, {"mapped_targets": ["gfx1101"]})
        monkeypatch.setitem(
            sys.modules,
            "torch",
            _fake_torch(
                ["gfx1101", "gfx1036"], [12049, 12176], hip = None, version_str = "2.6.0+rocm6.4"
            ),
        )
        assert LlamaCppBackend._get_gpu_free_memory(for_llama_server = True) == [(0, 12049)]

    def test_non_rocm_wheel_is_never_gated(self, tmp_path, monkeypatch, rocm_probe_env):
        # A CUDA wheel must not consult a ROCm marker at all.
        _binary_with_marker(tmp_path, {"mapped_targets": ["gfx1101"]})
        monkeypatch.setitem(
            sys.modules,
            "torch",
            _fake_torch(
                ["gfx1101", "gfx1036"], [12049, 12176], hip = None, version_str = "2.6.0+cu124"
            ),
        )
        assert LlamaCppBackend._get_gpu_free_memory(for_llama_server = True) == [
            (0, 12049),
            (1, 12176),
        ]


class TestTorchCallersStayUnfiltered:
    """The gate answers "what can llama-server run on", not "what GPUs exist".

    _get_gpu_memory is also the torch-free GPU check behind the RAG backend
    pick, which resolves to sentence-transformers (PyTorch). A device the
    installed llama.cpp prebuilt has no kernels for is usually still a perfectly
    good torch device, so gating that probe would silently move embeddings to
    the CPU -- a working path broken by the fix.
    """

    def test_probe_is_unfiltered_by_default(self, tmp_path, monkeypatch, rocm_probe_env):
        _binary_with_marker(tmp_path, {"mapped_targets": ["gfx1101"]})
        monkeypatch.setitem(
            sys.modules, "torch", _fake_torch(["gfx1101", "gfx1036"], [12049, 12176])
        )
        assert LlamaCppBackend._get_gpu_free_memory() == [(0, 12049), (1, 12176)]
        assert LlamaCppBackend._get_gpu_memory() == [(0, 12049, 32768), (1, 12176, 32768)]

    def test_rag_auto_still_picks_sentence_transformers(
        self, tmp_path, monkeypatch, rocm_probe_env
    ):
        # Every visible device is unsupported by the prebuilt; the torch caller
        # must still see a GPU and choose the torch backend.
        from core.rag import embeddings

        _binary_with_marker(tmp_path, {"mapped_targets": ["gfx1101"]})
        monkeypatch.setitem(sys.modules, "torch", _fake_torch(["gfx1036"], [12176]))
        assert embeddings._resolve_auto() == "sentence-transformers"

    def test_embed_llama_server_probe_opts_in(self, tmp_path, monkeypatch, rocm_probe_env):
        # The GGUF embedding backend IS llama-server, so the same unsupported
        # device must not count as an available GPU there.
        import utils.hardware as uh
        from core.rag.embed_llama_server import LlamaServerBackend

        monkeypatch.setattr(uh, "is_apple_silicon", lambda: False)
        _binary_with_marker(tmp_path, {"mapped_targets": ["gfx1101"]})
        monkeypatch.setitem(sys.modules, "torch", _fake_torch(["gfx1036"], [12176]))
        assert LlamaServerBackend._gpu_available() is False
        # Same host, supported arch: still a GPU.
        monkeypatch.setitem(sys.modules, "torch", _fake_torch(["gfx1101"], [12176]))
        assert LlamaServerBackend._gpu_available() is True


class TestArchGateCoexistsWithUnifiedMemory:
    """Both behaviours share the torch fallback loop and neither may displace
    the other: the APU keeps its host-RAM reserve and its total-0 reporting,
    while the unsupported device is still dropped."""

    def test_apu_reserve_and_arch_gate_together(self, tmp_path, monkeypatch, rocm_probe_env):
        # gfx1151 Strix Halo (unified memory, supported by the bundle),
        # gfx1101 dGPU (supported), gfx1036 (NOT in the bundle).
        _binary_with_marker(tmp_path, {"mapped_targets": ["gfx1101", "gfx1151"]})
        monkeypatch.setitem(
            sys.modules,
            "torch",
            _fake_torch(["gfx1151", "gfx1101", "gfx1036"], [20000, 12049, 12176]),
        )
        monkeypatch.setattr(
            LlamaCppBackend, "_available_system_memory_mib", staticmethod(lambda: 30000)
        )
        assert LlamaCppBackend._get_gpu_memory(for_llama_server = True) == [
            # Shared pool: host reserve held back, total reported 0.
            (0, 20000 - _IGPU_HOST_RESERVE_MIB, 0),
            # Discrete supported card: untouched.
            (1, 12049, 32768),
            # gfx1036 dropped by the arch gate.
        ]

    def test_apu_reserve_survives_unknown_coverage(self, tmp_path, monkeypatch, rocm_probe_env):
        # No marker: the gate fails open, and the APU accounting still applies.
        monkeypatch.setitem(sys.modules, "torch", _fake_torch(["gfx1151"], [20000]))
        monkeypatch.setattr(
            LlamaCppBackend, "_available_system_memory_mib", staticmethod(lambda: 30000)
        )
        assert LlamaCppBackend._get_gpu_memory(for_llama_server = True) == [
            (0, 20000 - _IGPU_HOST_RESERVE_MIB, 0)
        ]


class TestArchCrashRetrySet:
    """The reactive recovery for markerless / custom-linked builds, where the
    proactive gate cannot know what the binary covers."""

    def test_prefers_never_selected_devices(self):
        assert LlamaCppBackend._arch_crash_retry_gpu_ids([0], [0, 1, 2]) == [1, 2]

    def test_narrows_to_discrete_when_every_gpu_was_selected(self, monkeypatch):
        # The planner took both cards, so there is no untouched device left.
        # Dropping the shared-pool APU still leaves a launchable dGPU (#7624).
        monkeypatch.setattr(
            LlamaCppBackend, "_rocm_unified_memory_gpu_ids", staticmethod(lambda: {0})
        )
        assert LlamaCppBackend._arch_crash_retry_gpu_ids([0, 1], [0, 1]) == [1]

    def test_no_retry_when_nothing_would_change(self, monkeypatch):
        # No unified device among the selection: the respawn would be identical.
        monkeypatch.setattr(
            LlamaCppBackend, "_rocm_unified_memory_gpu_ids", staticmethod(lambda: set())
        )
        assert LlamaCppBackend._arch_crash_retry_gpu_ids([0, 1], [0, 1]) == []

    def test_no_retry_when_narrowing_empties_the_selection(self, monkeypatch):
        monkeypatch.setattr(
            LlamaCppBackend, "_rocm_unified_memory_gpu_ids", staticmethod(lambda: {0, 1})
        )
        assert LlamaCppBackend._arch_crash_retry_gpu_ids([0, 1], [0, 1]) == []

    def test_single_gpu_host_has_no_retry(self, monkeypatch):
        def _boom():
            raise AssertionError("must not probe on a single-GPU host")

        monkeypatch.setattr(LlamaCppBackend, "_rocm_unified_memory_gpu_ids", staticmethod(_boom))
        assert LlamaCppBackend._arch_crash_retry_gpu_ids([0], [0]) == []

    def test_empty_selection_is_a_no_op(self):
        assert LlamaCppBackend._arch_crash_retry_gpu_ids([], [0, 1]) == []

    def test_classifier_failure_is_not_fatal(self, monkeypatch):
        def _boom():
            raise RuntimeError("hip enumeration failed")

        monkeypatch.setattr(LlamaCppBackend, "_rocm_unified_memory_gpu_ids", staticmethod(_boom))
        assert LlamaCppBackend._arch_crash_retry_gpu_ids([0, 1], [0, 1]) == []
