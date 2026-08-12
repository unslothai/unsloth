# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Consumer-side pins for the #7624 / #7669 ROCm arch gate.

``test_gpu_arch_gate_7624.py`` pins the gate itself. This file pins the routing
decision around it: WHICH callers of ``_get_gpu_memory`` /
``_get_gpu_free_memory`` opt in, and what the ones that stay unfiltered are
guaranteed to keep doing.

Three claims are load-bearing and none of them is obvious from the call sites:

* the gate is INERT on a Vulkan build, which is what lets the Vulkan-ordinal
  preflight and the route-level ordinal checks stay unfiltered without an
  explicit ``for_llama_server = False``;
* placement opts in for AUTOMATIC selection only, so an explicit pin on a
  device the prebuilt has no kernels for still reaches that device (and its own
  crash message) instead of being quietly relocated or dropped;
* ``_wait_for_vram_settle`` stays unfiltered -- it measures driver reclaim, it
  does not place work.

Mock-based throughout: there is no AMD hardware or ROCm CI here.
"""

from __future__ import annotations

import os
import struct
import subprocess
import sys
import types
from pathlib import Path
from unittest.mock import patch

import pytest

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from core.inference.llama_cpp import GgufLoadIntent, LlamaCppBackend  # noqa: E402

_REAL_POPEN = subprocess.Popen


# ── Vulkan inertness ────────────────────────────────────────────────


@pytest.fixture
def vulkan_probe(monkeypatch):
    """A Vulkan build with one discrete device and one iGPU, and every ROCm
    arch helper booby-trapped: reaching one of them means the gate ran on a
    code path that has no gfx arches to speak of."""

    def _explode(*_args, **_kwargs):
        raise AssertionError("ROCm arch gate consulted on a Vulkan build")

    monkeypatch.setattr(LlamaCppBackend, "_is_vulkan_backend", staticmethod(lambda _b = None: True))
    monkeypatch.setattr(
        LlamaCppBackend, "_find_llama_server_binary", staticmethod(lambda: "/fake/llama-server")
    )
    monkeypatch.setattr(
        LlamaCppBackend,
        "_run_vulkan_probe",
        staticmethod(
            lambda _b = None: [
                {
                    "index": 0,
                    "free_mib": 12049,
                    "is_igpu": False,
                    "total_mib": 16384,
                    "name": "RX 7800 XT",
                },
                {
                    "index": 1,
                    "free_mib": 12176,
                    "is_igpu": True,
                    "total_mib": 65536,
                    "name": "Radeon Graphics",
                },
            ]
        ),
    )
    monkeypatch.setattr(LlamaCppBackend, "_installed_llama_gfx_archs", staticmethod(_explode))
    monkeypatch.setattr(LlamaCppBackend, "_rocm_arch_by_physical_id", staticmethod(_explode))
    monkeypatch.setattr(LlamaCppBackend, "_torch_is_rocm", staticmethod(_explode))


class TestGateIsInertOnVulkanBuilds:
    """The Vulkan branch returns before ``for_llama_server`` is ever read, so
    the preflight at load_model and the ordinal checks in routes/inference.py
    are unaffected by the flag they do not pass. Verified rather than assumed:
    the whole justification for leaving those three sites unfiltered rests on
    it."""

    def test_flag_changes_nothing_on_a_vulkan_build(self, vulkan_probe):
        gated = LlamaCppBackend._get_gpu_memory("/fake/llama-server", for_llama_server = True)
        plain = LlamaCppBackend._get_gpu_memory("/fake/llama-server")
        assert gated == plain
        # Real rows, not two empty lists agreeing with each other.
        assert [row[0] for row in plain] == [0, 1]

    def test_free_memory_wrapper_is_inert_too(self, vulkan_probe):
        assert LlamaCppBackend._get_gpu_free_memory(
            "/fake/llama-server", for_llama_server = True
        ) == LlamaCppBackend._get_gpu_free_memory("/fake/llama-server")

    def test_vulkan_ordinal_preflight_sees_every_ordinal(self, vulkan_probe):
        # The preflight's issubset check (#7239) must keep enumerating the iGPU
        # ordinal, or a legitimate explicit pin on it would 400.
        assert {g[0] for g in LlamaCppBackend._get_gpu_memory("/fake/llama-server")} == {0, 1}


# ── Placement: automatic only ───────────────────────────────────────


def _write_gguf(path: Path, architecture: str = "llama") -> Path:
    def string(value: str) -> bytes:
        data = value.encode()
        return struct.pack("<Q", len(data)) + data

    metadata = string("general.architecture") + struct.pack("<I", 8) + string(architecture)
    path.write_bytes(struct.pack("<IIQQ", 0x46554747, 3, 0, 1) + metadata)
    return path


def _backend(tmp_path: Path, memory, *, gated_out = frozenset()):
    """Placement harness whose GPU probe HONORS ``for_llama_server``.

    ``gated_out`` stands in for the devices whose gfx arch is absent from the
    installed prebuilt's mapped_targets, so the test observes the gate's real
    consequence for placement rather than only the argument it was called with.
    Every call is recorded in ``backend._probe_calls``.
    """
    backend = LlamaCppBackend()
    calls: list[bool] = []
    backend._probe_calls = calls

    def _probe(_binary = None, *, for_llama_server = False):
        calls.append(for_llama_server)
        rows = list(memory)
        if for_llama_server:
            rows = [row for row in rows if row[0] not in gated_out]
        return rows

    backend._get_gpu_memory = _probe
    backend._get_gpu_free_memory = lambda _binary = None, **kw: [
        (index, free) for index, free, _total in _probe(_binary, **kw)
    ]
    backend._read_gguf_metadata = lambda _path: None
    backend._can_estimate_kv = lambda: False
    backend._get_gguf_size_bytes = lambda _path: 1024
    backend._mmproj_vram_bytes = lambda _path: 0
    backend._resolve_launch_mmproj_path = lambda **kwargs: None
    backend._apu_ram_shortfall_message = lambda *args, **kwargs: None
    backend._amd_apu_wants_unified_memory = lambda *args, **kwargs: False
    backend._find_llama_server_binary = lambda include_denied = False: "/fake/llama-server"
    backend._is_vulkan_backend = lambda _binary = None: False
    backend._wait_for_health = lambda timeout: True
    backend._detect_audio_type_strict = lambda: None
    backend._apply_detected_audio = lambda _detected: True
    return backend, _write_gguf(tmp_path / "model.gguf")


def _launch(backend, gguf, **load_kwargs):
    captured: dict = {}

    def fake_popen(cmd, **kwargs):
        if not cmd or str(cmd[0]) != "/fake/llama-server":
            return _REAL_POPEN(cmd, **kwargs)
        captured["cmd"] = list(cmd)
        captured["env"] = kwargs.get("env") or dict(os.environ)
        return type(
            "Process",
            (),
            {
                "pid": 123,
                "stdout": (),
                "poll": lambda self: None,
                "terminate": lambda self: None,
                "wait": lambda self, timeout = None: 0,
                "kill": lambda self: None,
            },
        )()

    with patch.object(subprocess, "Popen", side_effect = fake_popen):
        assert backend.load_model(
            GgufLoadIntent(gguf_path = str(gguf), model_identifier = "test", **load_kwargs)
        )
    return captured


class TestPlacementOptsInForAutoOnly:
    """``for_llama_server = not gpu_ids`` at the placement probe."""

    def test_automatic_placement_opts_in(self, tmp_path):
        backend, gguf = _backend(tmp_path, [(0, 12049, 16384), (1, 40000, 65536)])
        _launch(backend, gguf)
        assert backend._probe_calls, "placement never probed the GPUs"
        assert any(backend._probe_calls), "automatic placement did not opt into the arch gate"

    def test_explicit_pin_does_not_opt_in(self, tmp_path):
        backend, gguf = _backend(tmp_path, [(0, 12049, 16384), (1, 40000, 65536)])
        _launch(backend, gguf, gpu_ids = [1])
        assert backend._probe_calls, "placement never probed the GPUs"
        assert not any(backend._probe_calls), "an explicit pin was silently arch-gated"

    def test_explicit_pin_on_an_uncovered_gpu_still_reaches_that_gpu(self, tmp_path):
        # GPU 1 is the device the installed prebuilt has no kernels for. The
        # user pinned it anyway. It must be the device the child is given, so
        # llama-server produces its own "device kernel image is invalid" and
        # the user learns which card is wrong -- not be relocated onto GPU 0
        # nor dropped to CPU behind their back.
        backend, gguf = _backend(
            tmp_path, [(0, 12049, 16384), (1, 40000, 65536)], gated_out = frozenset({1})
        )
        captured = _launch(backend, gguf, gpu_ids = [1])
        env = captured["env"]
        assert env.get("HIP_VISIBLE_DEVICES") == "1" or env.get("CUDA_VISIBLE_DEVICES") == "1", (
            f"explicit pin on the uncovered GPU did not reach the child: "
            f"HIP={env.get('HIP_VISIBLE_DEVICES')!r} CUDA={env.get('CUDA_VISIBLE_DEVICES')!r}"
        )

    def test_automatic_placement_avoids_the_uncovered_gpu(self, tmp_path):
        # The #7624 shape: the uncovered device reports the larger free pool and
        # would win the free-VRAM rank. Automatic placement must land on GPU 0.
        backend, gguf = _backend(
            tmp_path, [(0, 12049, 16384), (1, 40000, 65536)], gated_out = frozenset({1})
        )
        captured = _launch(backend, gguf)
        env = captured["env"]
        pinned = env.get("HIP_VISIBLE_DEVICES") or env.get("CUDA_VISIBLE_DEVICES")
        assert pinned is None or "1" not in pinned.split(","), (
            f"automatic placement selected the uncovered GPU: {pinned!r}"
        )


# ── Unfiltered by design ────────────────────────────────────────────


class TestWaitForVramSettleStaysUnfiltered:
    """The settle poll measures the driver reclaiming a dead child's
    allocations. It places nothing, so it must not pay for a marker read and an
    arch enumeration on every sample -- and narrowing its device list would
    change the ``len(curr) != len(prev)`` short-circuit it relies on."""

    def test_probe_is_never_gated(self, monkeypatch):
        import time as _time

        seen: list[dict] = []

        def _probe(binary = None, **kwargs):
            seen.append(dict(kwargs))
            return [(0, 12049)]

        monkeypatch.setattr(LlamaCppBackend, "_get_gpu_free_memory", staticmethod(_probe))
        LlamaCppBackend._wait_for_vram_settle(
            max_wait = 0.05, interval = 0.01, since_kill = _time.monotonic()
        )
        assert seen, "the settle poll never probed"
        assert all(not call.get("for_llama_server", False) for call in seen), (
            f"the settle poll opted into the arch gate: {seen}"
        )


class TestRagAutoStaysUnfiltered:
    """``core/rag/embeddings.py::_resolve_auto`` picks between
    sentence-transformers (PyTorch) and llama-server. The PyTorch winner runs
    on devices the llama.cpp prebuilt knows nothing about, so gating this probe
    would push bulk embedding indexing to the CPU for no reason."""

    def test_resolve_auto_never_gates(self, monkeypatch):
        from core.rag import embeddings

        seen: list[dict] = []

        def _probe(binary = None, **kwargs):
            seen.append(dict(kwargs))
            return [(0, 12049)]

        monkeypatch.setattr(LlamaCppBackend, "_get_gpu_free_memory", staticmethod(_probe))
        assert embeddings._resolve_auto() == "sentence-transformers"
        assert seen, "_resolve_auto never probed"
        assert all(not call.get("for_llama_server", False) for call in seen), (
            f"_resolve_auto opted into the arch gate: {seen}"
        )


class TestEmbedLlamaServerOptsIn:
    """The GGUF embedding backend IS a llama-server process, so it is the one
    RAG-side probe that must be gated."""

    def test_gpu_available_opts_in(self, monkeypatch):
        import utils.hardware as uh
        from core.rag.embed_llama_server import LlamaServerBackend

        monkeypatch.setattr(uh, "is_apple_silicon", lambda: False)
        seen: list[dict] = []

        def _probe(binary = None, **kwargs):
            seen.append(dict(kwargs))
            return [(0, 12049)]

        monkeypatch.setattr(LlamaCppBackend, "_get_gpu_free_memory", staticmethod(_probe))
        assert LlamaServerBackend._gpu_available() is True
        assert seen and all(call.get("for_llama_server") is True for call in seen), (
            f"the embedding llama-server probe was not gated: {seen}"
        )


# ── Crash recovery edge cases ───────────────────────────────────────


def _unified(monkeypatch, ids):
    monkeypatch.setattr(
        LlamaCppBackend, "_rocm_unified_memory_gpu_ids", staticmethod(lambda: set(ids))
    )


class TestArchCrashRetryEdgeCases:
    """``_arch_crash_retry_gpu_ids`` decides where a "device kernel image is
    invalid" crash respawns. Every answer it gives must be a strict improvement
    on the set that just crashed, or empty."""

    def test_none_inputs(self):
        assert LlamaCppBackend._arch_crash_retry_gpu_ids(None, None) == []
        assert LlamaCppBackend._arch_crash_retry_gpu_ids(None, [0, 1]) == []
        assert LlamaCppBackend._arch_crash_retry_gpu_ids([0], None) == []

    def test_empty_enumeration_with_a_multi_gpu_selection(self, monkeypatch):
        # Nothing enumerated (the probe failed after the crash) but two devices
        # were selected: narrowing is still the honest answer.
        _unified(monkeypatch, {1})
        assert LlamaCppBackend._arch_crash_retry_gpu_ids([0, 1], []) == [0]

    def test_duplicates_collapse(self, monkeypatch):
        _unified(monkeypatch, {0})
        assert LlamaCppBackend._arch_crash_retry_gpu_ids([0, 0, 1, 1], [0, 0, 1, 1]) == [1]
        assert LlamaCppBackend._arch_crash_retry_gpu_ids([0, 0], [0, 0, 1, 1, 2]) == [1, 2]

    def test_selected_ids_absent_from_the_enumeration(self, monkeypatch):
        # A stale selection naming a device the post-crash probe no longer sees.
        # The untouched enumerated device is still the right retry.
        _unified(monkeypatch, set())
        assert LlamaCppBackend._arch_crash_retry_gpu_ids([5], [0, 1]) == [0, 1]
        # ... and when the enumeration is a strict subset of the selection there
        # is no untouched device, so it falls through to the narrowing.
        _unified(monkeypatch, {7})
        assert LlamaCppBackend._arch_crash_retry_gpu_ids([7, 8], [7]) == [8]

    def test_single_gpu_host_never_probes_and_never_retries(self, monkeypatch):
        # A raising spy would be vacuous here: the narrowing branch swallows
        # every exception into []. Count the calls instead, so "took the short
        # circuit" and "took the long way and failed" stay distinguishable.
        calls: list[int] = []

        def _counting():
            calls.append(1)
            return set()

        monkeypatch.setattr(
            LlamaCppBackend, "_rocm_unified_memory_gpu_ids", staticmethod(_counting)
        )
        assert LlamaCppBackend._arch_crash_retry_gpu_ids([0], [0]) == []
        assert LlamaCppBackend._arch_crash_retry_gpu_ids([3], [3]) == []
        assert calls == [], "a single-GPU host enumerated devices it cannot use"

    def test_all_devices_unified(self, monkeypatch):
        _unified(monkeypatch, {0, 1, 2})
        # Every selected device is unified: narrowing empties the set, no retry.
        assert LlamaCppBackend._arch_crash_retry_gpu_ids([0, 1, 2], [0, 1, 2]) == []

    def test_no_device_unified(self, monkeypatch):
        _unified(monkeypatch, set())
        # Narrowing changes nothing, so the respawn would crash identically.
        assert LlamaCppBackend._arch_crash_retry_gpu_ids([0, 1, 2], [0, 1, 2]) == []

    def test_retry_set_never_contains_a_device_that_just_crashed_alone(self, monkeypatch):
        # The narrowing branch may only ever return a strict subset.
        _unified(monkeypatch, {0})
        selected = [0, 1, 2]
        out = LlamaCppBackend._arch_crash_retry_gpu_ids(selected, selected)
        assert out == [1, 2]
        assert set(out) < set(selected)

    @pytest.mark.parametrize(
        "selected,enumerated,unified",
        [
            ([0], [0], set()),
            ([0], [0, 1], set()),
            ([0, 1], [0, 1], {0}),
            ([0, 1], [0, 1], set()),
            ([0, 1], [0, 1], {0, 1}),
            ([0, 1, 2], [0, 1, 2, 3], {0}),
            ([5], [0, 1], {0}),
            ([0, 1], [], {1}),
        ],
    )
    def test_every_answer_differs_from_the_set_that_crashed(
        self, selected, enumerated, unified, monkeypatch
    ):
        """The one guarantee the single retry needs: the respawn is never the
        launch that just died. Either a different device set, or nothing."""
        _unified(monkeypatch, unified)
        out = LlamaCppBackend._arch_crash_retry_gpu_ids(selected, enumerated)
        assert not out or set(out) != set(selected), (
            f"retry would respawn the identical selection {sorted(set(selected))}"
        )

    def test_the_decision_is_stateless_and_so_must_not_be_looped(self, monkeypatch):
        """Pinned deliberately, as a hazard note rather than a bug.

        ``_arch_crash_retry_gpu_ids`` has no memory of what already failed, so
        the "prefer the never-selected devices" branch is symmetric: on a
        two-GPU host it maps [0] to [1] and [1] back to [0]. That is safe today
        only because the launch path applies it exactly once, in a straight-line
        ``if`` (see TestArchCrashRetryFiresAtMostOnce). Wrapping it in a retry
        loop without threading through the already-tried set would ping-pong
        forever, so make that consequence visible here.
        """
        _unified(monkeypatch, set())
        assert LlamaCppBackend._arch_crash_retry_gpu_ids([0], [0, 1]) == [1]
        assert LlamaCppBackend._arch_crash_retry_gpu_ids([1], [0, 1]) == [0]


class TestKernelImageInvalidDoesNotFalsePositive:
    """The marker gates a device-narrowing respawn, so a false positive would
    silently move an unrelated failure onto a different card."""

    @pytest.mark.parametrize(
        "output",
        [
            # Realistic llama.cpp / ggml output that mentions the words but is
            # not the arch mismatch.
            "ggml_cuda_compute_forward: RMS_NORM failed\nCUDA error: invalid argument",
            "load_model: error loading model: invalid model file magic",
            "llama_model_load: error loading model: check_tensor_dims: tensor "
            "'blk.0.attn_q.weight' has wrong shape",
            "ggml-cuda.cu: kernel launch failed",
            "error: invalid value for --n-gpu-layers",
            "ROCm error: out of memory",
            "hipErrorInvalidDeviceFunction: invalid device function",
            "warning: the kernel module amdgpu is out of date",
            "srv    load_model: the image is invalid",
            "device kernel image is valid",
            "common_init_from_params: failed to load model 'x.gguf'",
        ],
    )
    def test_unrelated_output_does_not_match(self, output):
        assert not LlamaCppBackend._kernel_image_invalid(output)

    @pytest.mark.parametrize(
        "output",
        [
            "ROCm error: device kernel image is invalid",
            "ggml-cuda.cu:76: ROCm error\n  device kernel image is invalid\n  current device: 1",
            "DEVICE KERNEL IMAGE IS INVALID".lower(),
        ],
    )
    def test_the_real_crash_matches(self, output):
        assert LlamaCppBackend._kernel_image_invalid(output)

    def test_matching_is_case_sensitive_by_design(self):
        # HIP emits the message lowercase. Documenting the actual contract so a
        # future "make it case-insensitive" change is a deliberate one.
        assert not LlamaCppBackend._kernel_image_invalid("Device Kernel Image Is Invalid")

    def test_non_string_input_is_not_a_match(self):
        assert not LlamaCppBackend._kernel_image_invalid(None)
        assert not LlamaCppBackend._kernel_image_invalid("")


class TestArchCrashRetryFiresAtMostOnce:
    """The recovery is one straight-line ``if`` in the launch path, not a loop.
    Pinned here so a future refactor into a retry loop has to notice."""

    def test_source_has_a_single_archfallback_spawn(self):
        source = Path(LlamaCppBackend.__module__.replace(".", "/"))
        path = Path(__file__).resolve().parent.parent / "core" / "inference" / "llama_cpp.py"
        assert path.exists(), source
        text = path.read_text(encoding = "utf-8")
        assert text.count('label = "-archfallback"') == 1
        assert text.count("_arch_crash_retry_gpu_ids(") == 2  # definition + the one call site

    def test_a_second_pass_over_the_same_state_yields_nothing(self, monkeypatch):
        # After the retry narrows [0, 1] to [1], re-running the decision on the
        # narrowed set is a single-GPU selection and answers [].
        _unified(monkeypatch, {0})
        first = LlamaCppBackend._arch_crash_retry_gpu_ids([0, 1], [0, 1])
        assert first == [1]
        assert LlamaCppBackend._arch_crash_retry_gpu_ids(first, [0, 1]) == [0]
        assert LlamaCppBackend._arch_crash_retry_gpu_ids([0], [0]) == []
