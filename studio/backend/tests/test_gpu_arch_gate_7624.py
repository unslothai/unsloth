# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Regression tests for #7624 / #7669: multi-GPU auto-selection on ROCm must not
pick a device the installed llama.cpp prebuilt has no kernels for.

Covers _installed_llama_gfx_archs (mapped_targets from the
UNSLOTH_PREBUILT_INFO.json marker, via llama_cpp_freshness),
_rocm_arch_by_physical_id, the opt-in per-device gate in _get_gpu_memory's torch
fallback, the "device kernel image is invalid" crash marker, and the retry set.

Also pins the two things the gate must NOT do: filter the probe for torch callers
(the RAG sentence-transformers pick runs under PyTorch, where a device llama.cpp
lacks kernels for is usually still fine), and displace the unified-memory APU
accounting that shares this loop.
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


# Every shape the install marker can arrive in, as (label, contents). It is written
# by the installer but read back after arbitrary upgrades, partial writes, disk
# damage and hand-editing, so the reader parses untrusted input. A callable receives
# the marker path and lays out something other than a plain JSON file.
_MARKER_CORPUS = [
    ("absent_file", None),
    ("empty_file", ""),
    ("invalid_json", "{not json"),
    ("truncated_json", '{"mapped_targets": ["gfx1030"'),
    ("marker_is_a_list", '["gfx1030"]'),
    ("marker_is_null", "null"),
    ("marker_is_a_string", '"hello"'),
    ("marker_is_empty_dict", "{}"),
    ("targets_null", '{"mapped_targets": null}'),
    ("targets_empty_string", '{"mapped_targets": ""}'),
    ("targets_bare_string", '{"mapped_targets": "gfx1030"}'),
    ("targets_empty_list", '{"mapped_targets": []}'),
    ("targets_list_of_null", '{"mapped_targets": [null]}'),
    ("targets_list_of_int", '{"mapped_targets": [123]}'),
    ("targets_blank_strings", '{"mapped_targets": ["", "  "]}'),
    ("targets_dict", '{"mapped_targets": {"a": "gfx1030"}}'),
    ("targets_huge_list", json.dumps({"mapped_targets": [f"gfx{i:04d}" for i in range(5000)]})),
    ("cuda_bundle", '{"asset": "app-linux-x64-cuda12.tar.gz"}'),
    ("vulkan_bundle", '{"asset": "app-linux-x64-vulkan.tar.gz", "mapped_targets": []}'),
    ("non_utf8_bytes", lambda p: p.write_bytes(b"\xff\xfe\x00garbage")),
    ("directory_in_place_of_file", lambda p: p.mkdir()),
    (
        "permission_denied",
        lambda p: (p.write_text(json.dumps({"mapped_targets": ["gfx1030"]})), p.chmod(0o000)),
    ),
    # Forwards compatibility: mapped_targets is remote data (it comes from the
    # release manifest, which versions independently of this code), so a future
    # publish can put a token in it that no device will ever report.
    ("future_generic_target", '{"mapped_targets": ["gfx11-generic"]}'),
    ("future_generic_target_dashed", '{"mapped_targets": ["gfx10-3-generic"]}'),
    ("future_family_label", '{"mapped_targets": ["gfx110X"]}'),
    ("future_mixed_concrete_and_generic", '{"mapped_targets": ["gfx1100", "gfx11-generic"]}'),
]


class TestInstalledLlamaGfxArchsCorpus:
    """The gate must degrade to "unknown" on every malformed marker. Raising is
    fatal because this runs inside the GPU probe; returning a non-None set no device
    can match is just as fatal and much quieter, since _get_gpu_memory then drops
    every GPU and llama-server silently runs on the CPU. The contract is None (fail
    open) or a set of concrete archs, never anything between."""

    @pytest.mark.parametrize("label,payload", _MARKER_CORPUS, ids = [c[0] for c in _MARKER_CORPUS])
    def test_never_raises_and_never_fails_closed(self, label, payload, tmp_path):
        import utils.llama_cpp_freshness as freshness

        root = tmp_path / label
        root.mkdir()
        marker = root / "UNSLOTH_PREBUILT_INFO.json"
        if callable(payload):
            payload(marker)
        elif payload is not None:
            marker.write_text(payload, encoding = "utf-8")
        # The walk-up memoizes per binary path; each case gets its own root.
        freshness._marker_cache.clear()

        archs = LlamaCppBackend._installed_llama_gfx_archs(
            str(root / "build" / "bin" / "llama-server")
        )

        if archs is None:
            return
        assert isinstance(archs, frozenset)
        # Not merely non-empty: every token must be one a device can report,
        # otherwise the set gates every GPU off the host.
        assert archs, f"{label} returned an empty set, which would drop every GPU"
        assert all(
            LlamaCppBackend._CONCRETE_GFX_ARCH.match(a) for a in archs
        ), f"{label} returned unmatchable tokens {sorted(archs)}"

    def test_empty_set_is_converted_to_none(self, tmp_path):
        # `return archs or None` is what does this. Pinned explicitly because a
        # frozenset() return would type-check fine and fail closed at runtime.
        binary = _binary_with_marker(tmp_path, {"mapped_targets": ["", "   ", ":"]})
        assert LlamaCppBackend._installed_llama_gfx_archs(binary) is None

    def test_non_list_targets_are_rejected_before_iteration(self, tmp_path, monkeypatch):
        # Pins the isinstance(targets, list) guard alone. Every non-list json.loads
        # can produce is ALSO caught downstream (a bare "gfx1030" iterates into single
        # characters, none a concrete arch), so the corpus above cannot tell the guard
        # from its absence. A tuple is not JSON-reachable, which is the point: it
        # isolates the guard from the token check.
        import utils.llama_cpp_freshness as freshness
        monkeypatch.setattr(
            freshness, "read_install_marker", lambda _b: {"mapped_targets": ("gfx1030",)}
        )
        assert LlamaCppBackend._installed_llama_gfx_archs(str(tmp_path / "llama-server")) is None

    def test_concrete_targets_still_gate(self, tmp_path):
        # The corpus is all about degrading safely; this pins that the normal
        # case is untouched, so "never fails closed" cannot be met by gutting
        # the feature.
        binary = _binary_with_marker(tmp_path, {"mapped_targets": ["gfx1030", "gfx90a"]})
        assert LlamaCppBackend._installed_llama_gfx_archs(binary) == frozenset(
            {"gfx1030", "gfx90a"}
        )


class TestForwardsCompatibleArchTokens:
    """A future manifest may record a target that is not a concrete per-device arch:
    ROCm 6.3+ ships generic code objects (gfx11-generic) covering a family, and this
    repo's manifest already carries umbrella labels (gfx110X, gfx120X) in the sibling
    gfx_target field. rocminfo/torch still report the CONCRETE arch, so exact-set
    membership against a generic token matches nothing and would drop every GPU. Fail
    open on any token we cannot interpret."""

    @pytest.mark.parametrize(
        "token",
        [
            "gfx11-generic",
            "gfx10-3-generic",
            "gfx9-4-generic",
            "gfx110X",
            "gfx120X",
            "gfx103X",
            "generic",
            "all",
            "native",
        ],
    )
    def test_non_concrete_token_disables_the_gate(self, token, tmp_path):
        binary = _binary_with_marker(tmp_path, {"mapped_targets": [token]})
        assert LlamaCppBackend._installed_llama_gfx_archs(binary) is None

    def test_one_bad_token_disables_the_whole_gate(self, tmp_path):
        # Deliberately all-or-nothing. Keeping just the concrete half of
        # ["gfx1100", "gfx11-generic"] would gate a gfx1101 device off a build
        # that the generic object actually covers.
        binary = _binary_with_marker(tmp_path, {"mapped_targets": ["gfx1100", "gfx11-generic"]})
        assert LlamaCppBackend._installed_llama_gfx_archs(binary) is None

    @pytest.mark.parametrize(
        "token", ["gfx803", "gfx900", "gfx906", "gfx908", "gfx90a", "gfx90c", "gfx942", "gfx950"]
    )
    def test_real_concrete_archs_are_accepted(self, token, tmp_path):
        # The published manifest's mapped_targets entries, plus the CDNA parts
        # whose trailing hex letter a digits-only pattern would wrongly reject.
        binary = _binary_with_marker(tmp_path, {"mapped_targets": [token]})
        assert LlamaCppBackend._installed_llama_gfx_archs(binary) == frozenset({token})

    def test_every_published_mapped_target_is_accepted(self, tmp_path):
        # Verbatim from llama-prebuilt-manifest.json of the unslothai/llama.cpp
        # release (b10360-mix-87da1a2): the union of every ROCm bundle's
        # mapped_targets. A pattern rejecting any of these turns the gate off for
        # real installs.
        published = [
            "gfx908", "gfx90a",
            "gfx1030", "gfx1031", "gfx1032", "gfx1034",
            "gfx1100", "gfx1101", "gfx1102", "gfx1103",
            "gfx1150", "gfx1151",
            "gfx1200", "gfx1201",
        ]  # fmt: skip
        binary = _binary_with_marker(tmp_path, {"mapped_targets": published})
        assert LlamaCppBackend._installed_llama_gfx_archs(binary) == frozenset(published)

    def test_generic_target_keeps_every_gpu(self, tmp_path, monkeypatch, rocm_probe_env):
        # End to end: without the guard this host loses BOTH cards and drops to
        # CPU, because neither gfx1100 nor gfx1101 equals "gfx11-generic".
        _binary_with_marker(tmp_path, {"mapped_targets": ["gfx11-generic"]})
        monkeypatch.setitem(
            sys.modules, "torch", _fake_torch(["gfx1100", "gfx1101"], [12000, 13000])
        )
        assert LlamaCppBackend._get_gpu_free_memory(for_llama_server = True) == [
            (0, 12000),
            (1, 13000),
        ]


class TestKernelImageInvalidMarker:
    def test_detects_rocm_arch_mismatch(self):
        tail = (
            "load_model: loading model 'x.gguf'\n"
            "E ROCm error: device kernel image is invalid\n"
            "E   current device: 0, in function ggml_cuda_kernel_launch"
        )
        assert LlamaCppBackend._kernel_image_invalid(tail)

    def test_detects_the_no_binary_for_gpu_spelling(self):
        # hipErrorNoBinaryForGpu: the same arch mismatch, and the code whose
        # documented cause IS "compiled for a different GPU architecture". A
        # build that raises this one must reach the retry too.
        tail = (
            "ggml-cuda.cu:76: ROCm error\n"
            "  no kernel image is available for execution on the device\n"
            "  current device: 1"
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
    _get_gpu_memory is also the torch-free check behind the RAG backend pick, which
    resolves to sentence-transformers (PyTorch), and a device the prebuilt lacks
    kernels for is usually still a good torch device -- so gating that probe would
    move embeddings to the CPU, a working path broken by the fix."""

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
        """#7624: the one selected GPU IS the whole host, so there is nothing to
        retry on and the unified-memory narrowing must be skipped outright. The spy
        counts rather than raises because the branch runs under
        ``except Exception: return []``, which swallows an AssertionError and returns
        the very [] the guard should produce -- passing either way. Counting is what
        observes the branch being skipped."""
        calls = []

        def _spy():
            calls.append(1)
            return set()

        monkeypatch.setattr(LlamaCppBackend, "_rocm_unified_memory_gpu_ids", staticmethod(_spy))
        assert LlamaCppBackend._arch_crash_retry_gpu_ids([0], [0]) == []
        assert calls == [], "single-GPU host must not reach the unified-memory probe"

    def test_empty_selection_is_a_no_op(self):
        assert LlamaCppBackend._arch_crash_retry_gpu_ids([], [0, 1]) == []

    def test_classifier_failure_is_not_fatal(self, monkeypatch):
        def _boom():
            raise RuntimeError("hip enumeration failed")

        monkeypatch.setattr(LlamaCppBackend, "_rocm_unified_memory_gpu_ids", staticmethod(_boom))
        assert LlamaCppBackend._arch_crash_retry_gpu_ids([0, 1], [0, 1]) == []
