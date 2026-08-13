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


def _backend(
    tmp_path: Path,
    memory,
    *,
    gated_out = frozenset(),
):
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
        assert pinned is None or "1" not in pinned.split(
            ","
        ), f"automatic placement selected the uncovered GPU: {pinned!r}"


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
        assert all(
            not call.get("for_llama_server", False) for call in seen
        ), f"the settle poll opted into the arch gate: {seen}"


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
        assert all(
            not call.get("for_llama_server", False) for call in seen
        ), f"_resolve_auto opted into the arch gate: {seen}"


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
        assert seen and all(
            call.get("for_llama_server") is True for call in seen
        ), f"the embedding llama-server probe was not gated: {seen}"


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
        assert not out or set(out) != set(
            selected
        ), f"retry would respawn the identical selection {sorted(set(selected))}"

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
            # hipErrorNoBinaryForGpu: HIP's other code for the same mismatch,
            # and the one whose documented cause is code compiled for a
            # different GPU arch. Neither field log happened to show it, but a
            # different ROCm or ggml build raises it for the same iGPU pick.
            "ROCm error: no kernel image is available for execution on the device",
            "ggml-cuda.cu:76: ROCm error\n"
            "  no kernel image is available for execution on the device\n"
            "  current device: 1",
            # The same code raised during backend init rather than at a kernel
            # launch: ggml prints it through a different format string, so the
            # match has to be on the message and not on the "ROCm error:" prefix.
            "ggml_cuda_init: failed to initialize ROCm: "
            "no kernel image is available for execution on the device",
            # cudaErrorNoKernelImageForDevice. Same string, same defect, and the
            # retry is not ROCm-gated, so an NVIDIA host with a build that has
            # no kernels for one of its cards recovers the same way.
            "CUDA error: no kernel image is available for execution on the device",
        ],
    )
    def test_the_real_crash_matches(self, output):
        assert LlamaCppBackend._kernel_image_invalid(output)

    @pytest.mark.parametrize(
        "output",
        [
            "Device Kernel Image Is Invalid",
            "ROCm error: DEVICE KERNEL IMAGE IS INVALID",
            "No kernel image is available for execution on the device",
            "ROCM ERROR: NO KERNEL IMAGE IS AVAILABLE FOR EXECUTION ON THE DEVICE",
        ],
    )
    def test_matching_is_case_insensitive(self, output):
        # hipGetErrorString is lowercase, but the layers that reprint it are not
        # consistent about that, and a missed match costs the whole recovery.
        # Safe because the markers are specific enough that folding case cannot
        # pull an unrelated failure in: the corpus above runs unchanged.
        assert LlamaCppBackend._kernel_image_invalid(output)

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


class TestArchRetryDropsTensorSplit:
    """``--tensor-split`` weights are positional over the child's VISIBLE
    devices (llama.cpp parses them into ``params.tensor_split[i]`` by position,
    then copies the first ``n_devices()``). The arch-crash retry masks a device
    out, which re-indexes the survivors, so the crashed set's shares would land
    on the wrong cards and could overcommit one. The retry drops the flag."""

    def test_the_two_token_form_goes_with_its_value(self):
        cmd = ["llama-server", "-m", "x.gguf", "--tensor-split", "10,20,30", "-ngl", "-1"]
        assert LlamaCppBackend._without_tensor_split(cmd) == [
            "llama-server",
            "-m",
            "x.gguf",
            "-ngl",
            "-1",
        ]

    def test_the_short_and_equals_forms_go_too(self):
        assert LlamaCppBackend._without_tensor_split(["s", "-ts", "1,2", "-ngl", "-1"]) == [
            "s",
            "-ngl",
            "-1",
        ]
        # An "=" form carries its value in the token, so nothing may be skipped
        # after it -- dropping the next token would eat --ngl's flag.
        assert LlamaCppBackend._without_tensor_split(["s", "--tensor-split=1,2", "-ngl"]) == [
            "s",
            "-ngl",
        ]
        # llama.cpp normalises long-option underscores; _flag_name mirrors it.
        assert LlamaCppBackend._without_tensor_split(["s", "--tensor_split", "1,2"]) == ["s"]

    def test_a_command_without_a_split_reports_nothing_to_do(self):
        cmd = ["llama-server", "-m", "x.gguf", "--split-mode", "tensor", "-ngl", "-1"]
        assert LlamaCppBackend._without_tensor_split(cmd) is None
        # Known limitation, pinned rather than fixed: the scan is positional, so
        # a VALUE spelled exactly like the flag is removed as if it were one --
        # and the two-token form would then swallow the argument after it. No
        # Studio-built argv can reach this (the only free-text values are the
        # model path and the HF-derived --alias, and llama.cpp's own value
        # tokens are numbers or enum words), so the cost of teaching the scanner
        # every flag's arity is not worth paying. If a caller-supplied value can
        # ever be "-ts" or "--split-mode", this is where it breaks.
        assert LlamaCppBackend._without_tensor_split(["s", "--alias", "-ts"]) == ["s", "--alias"]

    def test_only_the_split_is_removed(self):
        cmd = [
            "llama-server",
            "-m",
            "x.gguf",
            "--split-mode",
            "tensor",
            "--tensor-split",
            "10,20,30",
            "--flash-attn",
            "on",
            "-ngl",
            "-1",
        ]
        out = LlamaCppBackend._without_tensor_split(cmd)
        assert "--tensor-split" not in out and "10,20,30" not in out
        assert out == [
            "llama-server",
            "-m",
            "x.gguf",
            "--split-mode",
            "tensor",
            "--flash-attn",
            "on",
            "-ngl",
            "-1",
        ]

    def test_the_retry_call_site_drops_it_before_respawning(self):
        # Source-level, like test_source_has_a_single_archfallback_spawn: the
        # respawn is one straight-line block with no test seam, so pin that the
        # drop happens between narrowing the device set and the respawn.
        path = Path(__file__).resolve().parent.parent / "core" / "inference" / "llama_cpp.py"
        text = path.read_text(encoding = "utf-8")
        # Two call sites: the arch-crash retry, and the manual-split launch the
        # gate narrows. Both mask devices out from under a positional ratio.
        assert text.count("self._without_tensor_split(") == 2
        block = text.split("_arch_crash_retry_gpu_ids(\n")[-1].split('label = "-archfallback"')[0]
        assert "_without_tensor_split(cmd)" in block
        assert "self._tensor_split = None" in block  # /status must not report a dropped split


class TestArchRetryRestoresTheMemoryPolicy:
    """_spawn_and_wait's --fit retry appends a page-lock to its OWN argv and
    writes the Model Memory record back. The arch-crash respawn starts from
    `cmd`, which never carried that lock, so without a restore the backend
    reports page-locking as active on an unlocked child and the duplicate-load
    comparator then declines the reload that would apply it."""

    @staticmethod
    def _source():
        path = Path(__file__).resolve().parent.parent / "core" / "inference" / "llama_cpp.py"
        return path.read_text(encoding = "utf-8")

    def test_the_launch_snapshots_what_cmd_means(self):
        text = self._source()
        # Snapshotted at the launch, not re-derived at the retry: re-probing
        # residency for the SURVIVING devices would mark an APU survivor
        # mlock-applicable against a lock-free argv, which turns every later
        # duplicate load into a reload on a path that is quiet today.
        assert "_mem_policy_for_cmd = (" in text
        _snap = [
            _line.strip().rstrip(",")
            for _line in text.split("_mem_policy_for_cmd = (")[1].split(")")[0].splitlines()
            if _line.strip()
        ]
        assert _snap == [
            "_mem_host_resident",
            "self._memory_state",
            "self._memory_policy_active",
            "self._memory_mlock_applicable",
        ]

    def test_the_retry_call_site_restores_it_before_respawning(self):
        text = self._source()
        block = text.split("_arch_crash_retry_gpu_ids(\n")[-1].split('label = "-archfallback"')[0]
        assert "= _mem_policy_for_cmd" in block
        _restored = [
            _line.strip().rstrip(",")
            for _line in block.split(") = _mem_policy_for_cmd")[0].split("(")[-1].splitlines()
            if _line.strip()
        ]
        # Exact names, in the snapshot's order -- a tuple unpack cannot report a
        # mismatch, so a renamed or reordered target restores the wrong field.
        # _mem_host_resident is in it, or the respawn's own --fit retry reads the
        # crashed launch's re-armed lock as already held and skips re-arming.
        assert _restored == [
            "_mem_host_resident",
            "self._memory_state",
            "self._memory_policy_active",
            "self._memory_mlock_applicable",
        ]


class TestEmbedLlamaServerPinsTheGatedGpus:
    """Knowing a supported GPU EXISTS is not the same as launching on it. The
    embed child enumerates every ROCm agent, and that HSA enumeration is what
    dies on an arch the prebuilt has no kernels for -- so on a mixed host
    (unsupported iGPU + supported dGPU) the gate passes and the server still
    crashes unless the surviving ids are carried into the launch env."""

    @staticmethod
    def _probes(
        monkeypatch,
        *,
        gated,
        everything,
        archs = frozenset({"gfx1030"}),
    ):
        """Stub a ROCm host, the gate marker, and both probes. Returns the
        per-call kwargs seen, so a test can COUNT calls -- raising inside a spy
        here would be swallowed by the caller's ``except Exception``."""
        seen: list[dict] = []

        def _probe(binary = None, *, for_llama_server = False):
            seen.append({"binary": binary, "for_llama_server": for_llama_server})
            rows = gated if for_llama_server else everything
            return [(idx, free, 0) for idx, free in rows]

        monkeypatch.setattr(LlamaCppBackend, "_host_torch_is_rocm", staticmethod(lambda: True))
        monkeypatch.setattr(
            LlamaCppBackend, "_installed_llama_gfx_archs", staticmethod(lambda _b = None: archs)
        )
        monkeypatch.setattr(LlamaCppBackend, "_get_gpu_memory", staticmethod(_probe))
        return seen

    def test_a_narrowing_gate_yields_the_surviving_ids(self, monkeypatch):
        self._probes(monkeypatch, gated = [(1, 24000)], everything = [(0, 60000), (1, 24000)])
        from core.rag.embed_llama_server import LlamaServerBackend
        assert LlamaServerBackend._arch_gated_gpu_ids("/fake/llama-server") == [1]

    def test_full_coverage_needs_no_mask(self, monkeypatch):
        self._probes(
            monkeypatch, gated = [(0, 60000), (1, 24000)], everything = [(0, 60000), (1, 24000)]
        )
        from core.rag.embed_llama_server import LlamaServerBackend
        assert LlamaServerBackend._arch_gated_gpu_ids("/fake/llama-server") == []

    def test_unknown_coverage_fails_open_without_probing(self, monkeypatch):
        # NVIDIA, CPU-only, Vulkan and macOS have no mapped_targets marker. The
        # marker check comes first, so neither probe may run at all.
        seen = self._probes(
            monkeypatch, gated = [(1, 24000)], everything = [(0, 1), (1, 24000)], archs = None
        )
        from core.rag.embed_llama_server import LlamaServerBackend

        assert LlamaServerBackend._arch_gated_gpu_ids("/fake/llama-server") == []
        assert seen == [], f"the GPU probe ran despite unknown arch coverage: {seen}"

    def test_a_non_rocm_host_never_probes(self, monkeypatch):
        seen = self._probes(monkeypatch, gated = [(1, 24000)], everything = [(0, 1), (1, 24000)])
        monkeypatch.setattr(LlamaCppBackend, "_host_torch_is_rocm", staticmethod(lambda: False))
        from core.rag.embed_llama_server import LlamaServerBackend

        assert LlamaServerBackend._arch_gated_gpu_ids("/fake/llama-server") == []
        assert seen == [], f"a non-ROCm host paid for the gate probes: {seen}"

    def test_a_probe_failure_never_blocks_the_spawn(self, monkeypatch):
        monkeypatch.setattr(LlamaCppBackend, "_host_torch_is_rocm", staticmethod(lambda: True))
        monkeypatch.setattr(
            LlamaCppBackend,
            "_installed_llama_gfx_archs",
            staticmethod(lambda _b = None: (_ for _ in ()).throw(RuntimeError("marker"))),
        )
        from core.rag.embed_llama_server import LlamaServerBackend

        assert LlamaServerBackend._arch_gated_gpu_ids("/fake/llama-server") == []

    @staticmethod
    def _spy_visibility(monkeypatch):
        calls: list[tuple] = []
        monkeypatch.setattr(
            LlamaCppBackend,
            "_emit_child_gpu_visibility",
            staticmethod(lambda env, pinned, **kw: calls.append((pinned, kw))),
        )
        return calls

    def test_the_launch_env_masks_the_child_to_them(self, monkeypatch):
        self._probes(monkeypatch, gated = [(1, 24000)], everything = [(0, 60000), (1, 24000)])
        calls = self._spy_visibility(monkeypatch)
        from core.rag.embed_llama_server import LlamaServerBackend

        env = LlamaServerBackend()._build_env("/fake/llama-server", use_gpu = True)
        # prefer_rocr: a HIP-only mask still lets HSA enumerate the unsupported
        # agent, which is the segfault the pin exists to avoid.
        assert calls == [("1", {"prefer_rocr": True})], calls
        assert env.get("CUDA_VISIBLE_DEVICES") != ""  # the CPU sentinel, not this path

    def test_an_ungated_host_leaves_the_env_alone(self, monkeypatch):
        self._probes(
            monkeypatch, gated = [(0, 60000), (1, 24000)], everything = [(0, 60000), (1, 24000)]
        )
        calls = self._spy_visibility(monkeypatch)
        from core.rag.embed_llama_server import LlamaServerBackend

        LlamaServerBackend()._build_env("/fake/llama-server", use_gpu = True)
        assert calls == [], f"an unnarrowed host was masked anyway: {calls}"

    def test_a_uuid_mask_leaves_the_inherited_mask_in_place(self, monkeypatch):
        """A ROCr mask may name UUIDs, not indices ("a list of device indices or
        UUIDs", e.g. ROCR_VISIBLE_DEVICES="0,GPU-DEADBEEFDEADBEEF"). The gate's
        ids are then torch ordinals with no known physical mapping, and pinning
        them would replace that mask with numbers ROCr resolves against the
        whole host -- exposing cards the parent hid, the dropped one included."""
        self._probes(monkeypatch, gated = [(1, 24000)], everything = [(0, 60000), (1, 24000)])
        calls = self._spy_visibility(monkeypatch)
        # A torch module has to be importable, not just _torch_is_rocm patched:
        # _active_gpu_visibility_mask reads the ROCr mask only inside its `import
        # torch` try, so on a host without torch it answers from
        # CUDA_VISIBLE_DEVICES instead and an unmappable ROCr mask reads back as
        # "no mask". Without this the test silently checks the wrong branch and
        # fails on any runner whose dependency set omits torch.
        monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace())
        monkeypatch.setattr(LlamaCppBackend, "_torch_is_rocm", staticmethod(lambda _t: True))
        monkeypatch.setattr(sys, "platform", "linux")
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising = False)
        monkeypatch.delenv("HIP_VISIBLE_DEVICES", raising = False)
        monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "GPU-DEADBEEFDEADBEEF")
        from core.rag.embed_llama_server import LlamaServerBackend

        env = LlamaServerBackend()._build_env("/fake/llama-server", use_gpu = True)
        assert calls == [], f"an unmappable mask was rewritten with ordinals: {calls}"
        assert env["ROCR_VISIBLE_DEVICES"] == "GPU-DEADBEEFDEADBEEF"

    def test_the_cpu_path_hides_devices_at_both_layers(self, monkeypatch):
        """CPU has to mean CPU on ROCm too. HIP consults CUDA_VISIBLE_DEVICES
        only when HIP_VISIBLE_DEVICES is unset, so a blank CUDA mask alone leaves
        an inherited HIP pin in charge and the child keeps a device (and the
        VRAM its context costs) on a load that chose the CPU."""
        self._probes(monkeypatch, gated = [(1, 24000)], everything = [(0, 60000), (1, 24000)])
        calls = self._spy_visibility(monkeypatch)
        monkeypatch.setenv("HIP_VISIBLE_DEVICES", "0")
        monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "0")
        from core.rag.embed_llama_server import LlamaServerBackend

        env = LlamaServerBackend()._build_env("/fake/llama-server", use_gpu = False)
        assert calls == []  # no pin: the gate has nothing to narrow on a CPU load
        assert env["CUDA_VISIBLE_DEVICES"] == ""
        assert env["HIP_VISIBLE_DEVICES"] == "-1"
        # ROCR hides agents BELOW HIP, so clearing it would expose more of them
        # to the enumeration that dies on an uncovered arch. Left as inherited.
        assert env["ROCR_VISIBLE_DEVICES"] == "0"


class TestTheGateNeverRewritesAnUnmappableMask:
    """ROCr and CUDA both accept UUID device tokens, and CUDA also MIG ids.
    ``_resolve_visible_physical_ids`` answers None for those, so the probe's
    ids fall back to torch ordinals -- fine for RANKING (the arch map falls
    back the same way), unusable as a PIN: the runtime would resolve those
    numbers against the whole host. The survivor pin fails open there, so the
    inherited mask (and the launch it describes) is exactly what it was."""

    @staticmethod
    def _rocm_host(monkeypatch, *, gated, everything):
        seen: list[bool] = []

        def _probe(binary = None, *, for_llama_server = False):
            seen.append(for_llama_server)
            rows = gated if for_llama_server else everything
            return [(idx, free, 0) for idx, free in rows]

        monkeypatch.setattr(LlamaCppBackend, "_host_torch_is_rocm", staticmethod(lambda: True))
        monkeypatch.setattr(
            LlamaCppBackend,
            "_installed_llama_gfx_archs",
            staticmethod(lambda _b = None: frozenset({"gfx1030"})),
        )
        monkeypatch.setattr(LlamaCppBackend, "_get_gpu_memory", staticmethod(_probe))
        for _var in ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"):
            monkeypatch.delenv(_var, raising = False)
        return seen

    def test_an_unmappable_mask_stops_the_pin_before_either_probe(self, monkeypatch):
        seen = self._rocm_host(monkeypatch, gated = [(1, 24000)], everything = [(0, 60000), (1, 24000)])
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-deadbeefdeadbeef")

        assert LlamaCppBackend._arch_gate_survivors("/fake/llama-server") == []
        assert seen == [], f"the host paid for the probes it cannot use: {seen}"

    def test_an_index_mask_still_narrows(self, monkeypatch):
        # The fail-open must key on "set but unparseable", not on "unset": an
        # ordinary numeric mask still maps back, so the gate keeps working.
        self._rocm_host(monkeypatch, gated = [(1, 24000)], everything = [(0, 60000), (1, 24000)])
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")

        assert LlamaCppBackend._arch_gate_survivors("/fake/llama-server") == [1]

    def test_no_mask_at_all_still_narrows(self, monkeypatch):
        self._rocm_host(monkeypatch, gated = [(1, 24000)], everything = [(0, 60000), (1, 24000)])

        assert LlamaCppBackend._arch_gate_survivors("/fake/llama-server") == [1]

    @pytest.mark.parametrize(
        "mask, unmappable",
        [
            (None, False),  # no mask: nothing inherited to lose
            ("", False),  # empty mask: resolves to "no devices", not unknown
            ("0,1", False),
            (" 2 ", False),
            ("GPU-DEADBEEFDEADBEEF", True),
            ("0,GPU-DEADBEEFDEADBEEF", True),  # the mixed form ROCm documents
            ("MIG-GPU-4a2b/1/0", True),
        ],
    )
    def test_which_masks_count_as_unmappable(self, monkeypatch, mask, unmappable):
        monkeypatch.delenv("HIP_VISIBLE_DEVICES", raising = False)
        monkeypatch.delenv("ROCR_VISIBLE_DEVICES", raising = False)
        if mask is None:
            monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising = False)
        else:
            monkeypatch.setenv("CUDA_VISIBLE_DEVICES", mask)

        assert LlamaCppBackend._visibility_mask_is_unmappable() is unmappable


class TestCpuSentinelDropsAnInheritedDevicePick:
    """llama.cpp reads LLAMA_ARG_DEVICE / LLAMA_ARG_MAIN_GPU as the env spelling
    of --device / --main-gpu (common/arg.cpp set_env), and neither the chat
    forced-CPU launch nor the embedding CPU launch passes those flags. Masking
    every device away while leaving an inherited pick in place is the one
    combination llama.cpp cannot serve: it rejects a device name that no longer
    enumerates, so the child exits instead of running on the CPU we chose. The
    file already treats an inherited LLAMA_ARG_SPLIT_MODE / LLAMA_ARG_FIT as
    live input, so this is the same rule, not a new one.
    """

    def test_the_cpu_sentinel_clears_the_pick(self):
        env = {"LLAMA_ARG_DEVICE": "HIP0", "LLAMA_ARG_MAIN_GPU": "1", "PATH": "/usr/bin"}
        LlamaCppBackend._emit_child_gpu_visibility(env, "-1")
        assert "LLAMA_ARG_DEVICE" not in env
        assert "LLAMA_ARG_MAIN_GPU" not in env
        assert env["CUDA_VISIBLE_DEVICES"] == "-1"
        assert env["PATH"] == "/usr/bin"  # nothing else touched

    @pytest.mark.parametrize("pinned", ["0", "1,2", ""])
    def test_a_real_pin_keeps_it(self, pinned):
        """Only the CPU sentinel clears the pick. With devices visible the
        inherited selection is still the user's, and it still resolves."""
        env = {"LLAMA_ARG_DEVICE": "HIP0"}
        LlamaCppBackend._emit_child_gpu_visibility(env, pinned)
        assert env["LLAMA_ARG_DEVICE"] == "HIP0"

    def test_the_embedding_cpu_launch_clears_it_too(self, tmp_path, monkeypatch):
        from core.rag.embed_llama_server import LlamaServerBackend

        monkeypatch.setenv("LLAMA_ARG_DEVICE", "HIP0")
        monkeypatch.setenv("LLAMA_ARG_MAIN_GPU", "0")
        backend = LlamaServerBackend.__new__(LlamaServerBackend)
        env = backend._build_env(str(tmp_path / "llama-server"), use_gpu = False)
        assert "LLAMA_ARG_DEVICE" not in env
        assert "LLAMA_ARG_MAIN_GPU" not in env
        assert env["CUDA_VISIBLE_DEVICES"] == ""
        assert env["HIP_VISIBLE_DEVICES"] == "-1"

    def test_the_embedding_gpu_launch_keeps_it(self, tmp_path, monkeypatch):
        from core.rag.embed_llama_server import LlamaServerBackend

        monkeypatch.setenv("LLAMA_ARG_DEVICE", "HIP0")
        monkeypatch.setattr(
            LlamaCppBackend, "_arch_gate_survivors", staticmethod(lambda _b = None: [])
        )
        backend = LlamaServerBackend.__new__(LlamaServerBackend)
        env = backend._build_env(str(tmp_path / "llama-server"), use_gpu = True)
        assert env["LLAMA_ARG_DEVICE"] == "HIP0"


class TestArchForcedCpuHoldsNoVram:
    """An arch-gated launch is a zero-VRAM launch that arrived through an
    AUTOMATIC request, which is exactly what routes/inference.py:8206 means by
    "recovery may turn an automatic GPU request into a zero-VRAM load". Without
    it in holds_no_vram the CPU-only server keeps the CHAT claim, blocks an
    image/video pipeline from coexisting, and can be unloaded mid-load by an
    owner it never competed with.
    """

    def _backend(self):
        backend = LlamaCppBackend()
        backend._gpu_memory_mode = "auto"
        backend._gpu_layers = -1
        backend._gpu_offload_active = False
        return backend

    def test_an_automatic_arch_gated_launch_holds_no_vram(self):
        backend = self._backend()
        assert backend.holds_no_vram is False, "precondition: auto mode alone is not zero-VRAM"
        backend._arch_gate_forced_cpu = True
        assert backend.holds_no_vram is True

    def test_the_manual_zero_offload_rule_is_unchanged(self):
        backend = LlamaCppBackend()
        backend._gpu_memory_mode = "manual"
        backend._gpu_layers = 0
        backend._gpu_offload_active = False
        assert backend.holds_no_vram is True
        backend._gpu_offload_active = True
        assert backend.holds_no_vram is False
        backend._gpu_offload_active = None
        assert backend.holds_no_vram is False

    def test_an_ordinary_gpu_load_still_holds_vram(self):
        backend = self._backend()
        backend._gpu_offload_active = True
        assert backend.holds_no_vram is False


# ── The gated split still dedupes an identical repeat load ─────────────────


class _StubProcess:
    """``is_loaded`` only asks "is not None"; nothing here is ever spawned."""

    def terminate(self):
        pass

    def wait(self, timeout = None):
        return 0

    def kill(self):
        pass

    def poll(self):
        return 0


_GATED_REQUEST = dict(
    model_identifier = "owner/repo",
    hf_variant = "Q4_K_M",
    n_ctx = 8192,
    gpu_memory_mode = "manual",
    gpu_layers = 99,
    tensor_split = (0.5, 0.5),
)


def _post_gate_backend(dropped, *, live_split = None):
    """Live state after a manual-split launch the arch gate normalized: the ratio
    left the argv (``_tensor_split`` is None), and the drop was recorded."""
    backend = LlamaCppBackend()
    backend._process = _StubProcess()
    backend._healthy = True
    backend._model_identifier = "owner/repo"
    backend._hf_variant = "Q4_K_M"
    backend._requested_n_ctx = 8192
    backend._requested_n_parallel = 1
    backend._requested_spec_mode = "auto"
    backend._gpu_memory_mode = "manual"
    backend._gpu_layers = 99
    backend._tensor_split = live_split
    backend._arch_gate_dropped_tensor_split = dropped
    return backend


class TestGatedSplitStillDeduplicates:
    """The gate drops a ratio sized for GPUs the installed build has no kernels
    for. The UI re-sends that same ratio on every Apply, and
    ``_runtime_matches_intent`` compares it against the live ``_tensor_split``,
    so without the recorded drop each identical request reads as new and
    respawns the same already-normalized server -- a multi-second teardown and
    reload of a multi-GB model, on every Apply, forever."""

    def test_identical_repeat_request_matches(self):
        backend = _post_gate_backend((0.5, 0.5))
        assert backend.adopt_load_intent_if_matched(GgufLoadIntent(**_GATED_REQUEST)) is True

    def test_a_different_ratio_still_reloads(self):
        """The excuse is for the ratio that was dropped, not for any ratio."""
        backend = _post_gate_backend((0.5, 0.5))
        changed = dict(_GATED_REQUEST, tensor_split = (0.9, 0.1))
        assert backend.adopt_load_intent_if_matched(GgufLoadIntent(**changed)) is False

    def test_a_launch_that_dropped_nothing_records_nothing(self):
        """No recorded drop means no excuse: a live server with no split must
        still reload for a request that asks for one."""
        backend = _post_gate_backend(None)
        assert backend.adopt_load_intent_if_matched(GgufLoadIntent(**_GATED_REQUEST)) is False

    def test_no_drop_never_excuses_a_live_split(self):
        """Both sides None must not read as "the gate dropped this": a server
        RUNNING a ratio has to reload for a request that asks for none."""
        backend = _post_gate_backend(None, live_split = [0.5, 0.5])
        no_split = dict(_GATED_REQUEST, tensor_split = None)
        assert backend.adopt_load_intent_if_matched(GgufLoadIntent(**no_split)) is False
