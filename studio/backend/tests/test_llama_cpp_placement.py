# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Focused integration tests for explicit GGUF GPU placement."""

from __future__ import annotations

import os
import struct
import subprocess
import sys
import threading
import types
from pathlib import Path
from unittest.mock import patch

import pytest

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)


def _stub_module(name: str, **attrs):
    if name in sys.modules:
        return
    try:
        __import__(name)
        return
    except Exception:
        module = types.ModuleType(name)
        for key, value in attrs.items():
            setattr(module, key, value)
        sys.modules[name] = module


_stub_module("loggers", get_logger = lambda name: __import__("logging").getLogger(name))
_stub_module("structlog", get_logger = lambda *a, **k: __import__("logging").getLogger("stub"))
_stub_module(
    "jwt",
    decode = lambda *a, **k: {},
    ExpiredSignatureError = type("ExpiredSignatureError", (Exception,), {}),
    InvalidTokenError = type("InvalidTokenError", (Exception,), {}),
)
if "httpx" not in sys.modules:
    try:
        import httpx  # noqa: F401
    except Exception:
        module = types.ModuleType("httpx")
        for name in (
            "ConnectError",
            "TimeoutException",
            "ReadTimeout",
            "ReadError",
            "RemoteProtocolError",
            "CloseError",
        ):
            setattr(module, name, type(name, (Exception,), {}))
        module.Timeout = type("Timeout", (), {"__init__": lambda self, *a, **k: None})
        module.Client = type(
            "Client",
            (),
            {
                "__init__": lambda self, **kwargs: None,
                "__enter__": lambda self: self,
                "__exit__": lambda self, *args: None,
            },
        )
        sys.modules["httpx"] = module

from core.inference.llama_cpp import GgufLoadIntent, LlamaCppBackend, _loader_path_var

_REAL_POPEN = subprocess.Popen


def _write_gguf(path: Path, architecture: str = "llama") -> Path:
    def string(value: str) -> bytes:
        data = value.encode()
        return struct.pack("<Q", len(data)) + data

    metadata = string("general.architecture") + struct.pack("<I", 8) + string(architecture)
    path.write_bytes(struct.pack("<IIQQ", 0x46554747, 3, 0, 1) + metadata)
    return path


def _backend(tmp_path: Path, *, vulkan: bool, memory):
    backend = LlamaCppBackend()
    gguf = _write_gguf(tmp_path / "model.gguf")
    backend._get_gpu_memory = lambda _binary = None, **_kw: list(memory)
    backend._get_gpu_free_memory = lambda _binary = None, **_kw: [
        (index, free) for index, free, _total in memory
    ]
    backend._read_gguf_metadata = lambda _path: None
    backend._can_estimate_kv = lambda: False
    backend._get_gguf_size_bytes = lambda _path: 1024
    backend._mmproj_vram_bytes = lambda _path: 0
    backend._resolve_launch_mmproj_path = lambda **kwargs: None
    backend._apu_ram_shortfall_message = lambda *args, **kwargs: None
    # Off by default: the host-RAM preflight is not what most of these cells are about,
    # and it now runs on every launch. The tests that ARE about it restore the real one.
    backend._launch_host_shortfall_message = lambda *args, **kwargs: None
    backend._amd_apu_wants_unified_memory = lambda *args, **kwargs: False
    backend._find_llama_server_binary = lambda include_denied = False: "/fake/llama-server"
    backend._is_vulkan_backend = lambda _binary = None: vulkan
    backend._wait_for_health = lambda timeout: True
    backend._detect_audio_type_strict = lambda: None
    backend._apply_detected_audio = lambda _detected: True
    return backend, gguf


def _launch(backend, gguf, **load_kwargs):
    captured = {}

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
            GgufLoadIntent(
                gguf_path = str(gguf),
                model_identifier = "test",
                **load_kwargs,
            )
        )
    return captured


def _launch_warns(backend, gguf, **load_kwargs):
    """A load whose weights outgrow fast memory: it LAUNCHES, and says so.

    These cases used to raise. The spill is mmap'd, so refusing cost users a load
    llama.cpp itself supports. The advisory is the whole remaining contract, so assert
    it rather than only that the launch survived.
    """
    captured = _launch(backend, gguf, **load_kwargs)
    assert "does not fit in GPU memory" in (backend.last_load_warning or "")
    return captured


def test_vulkan_selection_uses_ordinals_and_owns_device_flags(tmp_path):
    backend, gguf = _backend(
        tmp_path,
        vulkan = True,
        memory = [(0, 10_000, 16_000), (1, 8_000, 16_000)],
    )
    backend._select_gpus = lambda *args, **kwargs: ([1], False)

    result = _launch(
        backend,
        gguf,
        gpu_ids = [0, 1],
        extra_args = ["--device", "Vulkan0", "--main-gpu", "0", "--top-k", "5"],
    )

    cmd = result["cmd"]
    assert cmd[cmd.index("--device") + 1] == "Vulkan1"
    assert cmd.count("--device") == 1
    assert "--main-gpu" not in cmd
    assert cmd[cmd.index("--top-k") + 1] == "5"
    assert backend.requested_gpu_ids == [0, 1]
    assert backend.gpu_ids == [1]


@pytest.mark.parametrize(
    "gpu_ids,extra_args,expected_draft,user_device_survives",
    [
        (None, None, "Vulkan1", False),
        (None, ["--device", "Vulkan1", "-dev=Vulkan0"], "Vulkan0", True),
        ([1], ["--device", "Vulkan1", "-dev=Vulkan0"], "Vulkan1", False),
    ],
)
def test_vulkan_fit_and_mtp_drafter_follow_placement_owner(
    tmp_path, gpu_ids, extra_args, expected_draft, user_device_survives
):
    backend, gguf = _backend(
        tmp_path,
        vulkan = True,
        memory = [(0, 24_000, 0), (1, 8_000, 16_000)],
    )
    planned = []

    def fallback(_model_size, gpus, *args, **kwargs):
        planned.append(list(gpus))
        return None, True

    backend._select_gpus = fallback
    backend.probe_server_capabilities = lambda _binary = None: {
        "mtp_token": "draft-mtp",
        "spec_draft_n_max_flag": "--spec-draft-n-max",
    }
    backend._resolve_launch_mtp_path = lambda **_kwargs: "/fake/mtp.gguf"
    result = _launch(
        backend,
        gguf,
        mtp_draft_path = "/fake/mtp.gguf",
        speculative_type = "mtp",
        gpu_ids = gpu_ids,
        extra_args = extra_args,
    )

    assert planned
    assert all(gpus == [(1, 8_000)] for gpus in planned)
    cmd = result["cmd"]
    assert cmd[cmd.index("--device") + 1] == "Vulkan1"
    assert cmd[cmd.index("--spec-draft-device") + 1] == expected_draft
    assert ("-dev=Vulkan0" in cmd) is user_device_survives


@pytest.mark.parametrize("use_fit", [False, True])
def test_dspark_composed_argv_respects_placement_fit_decision(tmp_path, use_fit):
    backend, gguf = _backend(
        tmp_path,
        vulkan = False,
        memory = [(0, 24_000, 24_000)],
    )
    sidecar = tmp_path / "dspark-model-Q8_0.gguf"
    sidecar.write_bytes(b"draft")
    backend._select_gpus = lambda *args, **kwargs: (None, True) if use_fit else ([0], False)
    backend.probe_server_capabilities = lambda _binary = None: {
        "supports_dspark": True,
        "spec_draft_n_max_flag": "--spec-draft-n-max",
    }

    result = _launch(
        backend,
        gguf,
        dspark_draft_path = str(sidecar),
        speculative_type = "dspark",
    )

    cmd = result["cmd"]
    assert cmd.count("--fit") == 1
    assert cmd[cmd.index("--fit") + 1] == ("on" if use_fit else "off")
    # DSpark engages under either placement: --fit on only means llama.cpp skips
    # the sidecar's memory reserve, it does not refuse to load it.
    assert cmd[cmd.index("--model-draft") + 1] == str(sidecar)
    assert cmd[cmd.index("--spec-type") + 1] == "draft-dspark"
    assert backend.spec_fallback_reason is None


def test_dspark_keeps_a_user_fit_flag(tmp_path):
    """A caller's --fit is theirs to set: the sidecar loads under either value,
    so Unsloth has no reason to rewrite it."""
    backend, gguf = _backend(tmp_path, vulkan = False, memory = [(0, 24_000, 24_000)])
    sidecar = tmp_path / "dspark-model-Q8_0.gguf"
    sidecar.write_bytes(b"draft")
    backend._select_gpus = lambda *args, **kwargs: ([0], False)
    backend.probe_server_capabilities = lambda _binary = None: {
        "supports_dspark": True,
        "spec_draft_n_max_flag": "--spec-draft-n-max",
    }

    result = _launch(
        backend,
        gguf,
        dspark_draft_path = str(sidecar),
        speculative_type = "dspark",
        extra_args = ["--fit", "on", "--top-k", "5"],
        gpu_ids = [0],
    )

    cmd = result["cmd"]
    assert cmd[len(cmd) - 1 - cmd[::-1].index("--fit") + 1] == "on"
    assert cmd[cmd.index("--top-k") + 1] == "5"
    assert cmd[cmd.index("--spec-type") + 1] == "draft-dspark"


def test_pass_through_dspark_loads_under_an_auto_fit_placement(tmp_path):
    """Manual + Auto layers emits --fit on and a user-owned --spec-type returns
    from _build_speculative_flags early. Nothing rewrites the placement: llama.cpp
    only skips the sidecar's memory reserve under fitting, it still loads it."""
    backend, gguf = _backend(tmp_path, vulkan = False, memory = [(0, 24_000, 24_000)])
    sidecar = tmp_path / "dspark-model-Q8_0.gguf"
    sidecar.write_bytes(b"draft")

    result = _launch(
        backend,
        gguf,
        gpu_memory_mode = "manual",
        gpu_layers = -1,
        extra_args = ["--spec-type", "draft-dspark", "--model-draft", str(sidecar)],
    )

    cmd = result["cmd"]
    assert cmd.count("--fit") == 1
    assert cmd[cmd.index("--fit") + 1] == "on"
    assert cmd[cmd.index("--spec-type") + 1] == "draft-dspark"


def test_cuda_selection_uses_visibility_and_removes_environment_placement(tmp_path, monkeypatch):
    monkeypatch.setenv("LLAMA_ARG_DEVICE", "CUDA0")
    monkeypatch.setenv("LLAMA_ARG_MAIN_GPU", "0")
    backend, gguf = _backend(
        tmp_path,
        vulkan = False,
        memory = [(0, 10_000, 16_000), (1, 8_000, 16_000)],
    )
    backend._select_gpus = lambda *args, **kwargs: ([1], False)

    result = _launch(backend, gguf, gpu_ids = [1])

    assert result["env"]["CUDA_VISIBLE_DEVICES"] == "1"
    assert "LLAMA_ARG_DEVICE" not in result["env"]
    assert "LLAMA_ARG_MAIN_GPU" not in result["env"]


def test_backend_detection_accepts_versioned_vulkan_soname(tmp_path):
    binary = tmp_path / "llama-server"
    binary.write_bytes(b"x")
    lib_dir = tmp_path / "lib"
    lib_dir.mkdir()
    prefix = "" if sys.platform == "win32" else "lib"
    extension = "dll" if sys.platform == "win32" else "so"
    (lib_dir / f"{prefix}ggml-vulkan.{extension}.0").write_bytes(b"x")

    with patch("core.inference.llama_cpp._llama_lib_dir", return_value = lib_dir):
        assert LlamaCppBackend._is_vulkan_backend(str(binary)) is True
        assert LlamaCppBackend._backend_lacks_gpu_lib(str(binary)) is False


def test_cpu_only_detection_requires_a_proven_split_library_layout(tmp_path):
    binary = tmp_path / "llama-server"
    binary.write_bytes(b"x")
    lib_dir = tmp_path / "lib"
    lib_dir.mkdir()
    prefix = "" if sys.platform == "win32" else "lib"
    extension = "dll" if sys.platform == "win32" else "so"
    (lib_dir / f"{prefix}ggml-cpu.{extension}").write_bytes(b"x")

    with patch("core.inference.llama_cpp._llama_lib_dir", return_value = lib_dir):
        assert LlamaCppBackend._backend_lacks_gpu_lib(str(binary)) is True

    (lib_dir / f"{prefix}ggml-vulkan.{extension}").write_bytes(b"x")
    with patch("core.inference.llama_cpp._llama_lib_dir", return_value = lib_dir):
        assert LlamaCppBackend._backend_lacks_gpu_lib(str(binary)) is False


def test_diffusion_does_not_reinterpret_vulkan_ordinals(tmp_path):
    gguf = _write_gguf(tmp_path / "diffusion.gguf", "diffusion-gemma")
    backend = LlamaCppBackend()
    backend._find_llama_server_binary = lambda include_denied = False: "/fake/llama-server"
    backend._is_vulkan_backend = lambda _binary = None: True
    backend._get_gpu_memory = lambda _binary = None, **_kw: [(1, 8_000, 8_000)]
    backend._download_gguf = lambda **kwargs: str(gguf)
    backend._read_gguf_metadata = lambda _path: setattr(backend, "_is_diffusion", True)
    backend._start_diffusion_server = lambda **kwargs: pytest.fail(
        "Vulkan ordinal reached the CUDA diffusion runner"
    )

    with pytest.raises(ValueError, match = "no defined mapping"):
        backend.load_model(
            GgufLoadIntent(
                hf_repo = "renamed/model",
                hf_variant = "Q4_K_M",
                model_identifier = "renamed/model",
                speculative_type = "off",
                gpu_ids = [1],
            )
        )


# ── Auto drops a drafter the VRAM cannot hold ─────────────────────────


def _hybrid_mtp_backend(
    tmp_path: Path,
    *,
    partial_offload: bool,
    memory = None,
):
    backend, gguf = _backend(
        tmp_path,
        vulkan = False,
        memory = [(0, 12 * 1024, 12 * 1024)] if memory is None else memory,
    )

    def read_metadata(_path):
        backend._nextn_predict_layers = 1
        backend._n_layers = 65
        backend._n_kv_heads = 4
        backend._n_heads = 24
        backend._embedding_length = 5120
        backend._kv_key_length = 256
        backend._kv_value_length = 256
        backend._full_attention_interval = 4
        backend._ssm_inner_size = 6144
        backend._ssm_state_size = 128
        backend._ssm_group_count = 16
        backend._ssm_conv_kernel = 4

    backend._read_gguf_metadata = read_metadata
    placement = (None, True) if partial_offload else ([0], False)
    backend._select_gpus = lambda *args, **kwargs: placement
    backend._select_gpus_split_aware = lambda *args, **kwargs: placement
    backend.probe_server_capabilities = lambda _binary = None: {
        "mtp_token": "draft-mtp",
        "supports_ngram_mod": True,
        "spec_draft_n_max_flag": "--spec-draft-n-max",
    }
    return backend, gguf


def test_auto_disables_embedded_hybrid_mtp_under_partial_offload(tmp_path):
    backend, gguf = _hybrid_mtp_backend(tmp_path, partial_offload = True)

    result = _launch(
        backend,
        gguf,
        speculative_type = "auto",
        n_ctx = 4096,
        n_parallel = 4,
    )

    cmd = result["cmd"]
    assert cmd[cmd.index("--fit") + 1] == "on"
    assert "draft-mtp" not in cmd
    assert "ngram-mod" not in cmd
    assert cmd[cmd.index("--spec-type") + 1] == "none"
    assert backend.spec_fallback_reason == "mtp_partial_offload"


def test_forced_embedded_hybrid_mtp_survives_partial_offload(tmp_path):
    backend, gguf = _hybrid_mtp_backend(tmp_path, partial_offload = True)

    result = _launch(
        backend,
        gguf,
        speculative_type = "mtp",
        n_ctx = 4096,
        n_parallel = 4,
    )

    cmd = result["cmd"]
    assert cmd[cmd.index("--fit") + 1] == "on"
    assert cmd[cmd.index("--spec-type") + 1] == "draft-mtp"
    assert backend.spec_fallback_reason is None


def test_auto_keeps_embedded_hybrid_mtp_when_fully_offloaded(tmp_path):
    backend, gguf = _hybrid_mtp_backend(tmp_path, partial_offload = False)

    result = _launch(
        backend,
        gguf,
        speculative_type = "auto",
        n_ctx = 4096,
        n_parallel = 4,
    )

    cmd = result["cmd"]
    assert cmd[cmd.index("--fit") + 1] == "off"
    assert cmd[cmd.index("--spec-type") + 1] == "draft-mtp"
    assert backend.spec_fallback_reason is None


def test_auto_disables_embedded_hybrid_mtp_with_manual_partial_layers(tmp_path):
    backend, gguf = _hybrid_mtp_backend(tmp_path, partial_offload = False)

    result = _launch(
        backend,
        gguf,
        speculative_type = "auto",
        gpu_memory_mode = "manual",
        gpu_layers = 42,
        n_ctx = 4096,
        n_parallel = 4,
    )

    cmd = result["cmd"]
    assert cmd[cmd.index("--gpu-layers") + 1] == "42"
    assert cmd[cmd.index("--fit") + 1] == "off"
    assert cmd[cmd.index("--spec-type") + 1] == "none"
    assert backend.spec_fallback_reason == "mtp_partial_offload"


@pytest.mark.parametrize("gpu_layers", [0, 66])
def test_auto_keeps_embedded_hybrid_mtp_without_manual_partial_layers(tmp_path, gpu_layers):
    backend, gguf = _hybrid_mtp_backend(tmp_path, partial_offload = False)

    result = _launch(
        backend,
        gguf,
        speculative_type = "auto",
        gpu_memory_mode = "manual",
        gpu_layers = gpu_layers,
        n_ctx = 4096,
        n_parallel = 4,
    )

    cmd = result["cmd"]
    assert cmd[cmd.index("--gpu-layers") + 1] == str(gpu_layers)
    assert cmd[cmd.index("--spec-type") + 1] == "draft-mtp"
    assert backend.spec_fallback_reason is None


def test_auto_keeps_embedded_hybrid_mtp_without_a_gpu(tmp_path):
    # No GPU is probed, so nothing selects a placement and `--fit on` stays --
    # the same command a CPU-only box and a Metal Mac emit. There is nothing to
    # partially offload to there, and the rollback copies cost no VRAM, so the
    # CPU MTP policy stands.
    backend, gguf = _hybrid_mtp_backend(tmp_path, partial_offload = True, memory = [])

    result = _launch(
        backend,
        gguf,
        speculative_type = "auto",
        n_ctx = 4096,
        n_parallel = 4,
    )

    cmd = result["cmd"]
    assert cmd[cmd.index("--fit") + 1] == "on"
    assert "draft-mtp" in cmd[cmd.index("--spec-type") + 1]
    assert backend.spec_fallback_reason is None


def test_auto_keeps_embedded_hybrid_mtp_when_the_device_selection_is_cpu(tmp_path):
    # A GPU is probed, but the extras take the model off it. llama.cpp then runs
    # on the CPU whatever the fitter decides, so nothing is partially offloaded
    # and the rollback copies cost no VRAM.
    backend, gguf = _hybrid_mtp_backend(tmp_path, partial_offload = True)

    result = _launch(
        backend,
        gguf,
        speculative_type = "auto",
        extra_args = ["--device", "none"],
        n_ctx = 4096,
        n_parallel = 4,
    )

    cmd = result["cmd"]
    assert cmd[cmd.index("--fit") + 1] == "on"
    assert "draft-mtp" in cmd[cmd.index("--spec-type") + 1]
    assert backend.spec_fallback_reason is None


def test_a_hand_pinned_device_is_gpu_evidence_when_the_probe_found_none(tmp_path):
    # A failed probe is not evidence of no GPU: the extras can still point the
    # child at one and ask for a partial count, which is the placement this
    # fallback exists for. Same flag _device_selection_is_cpu reads for the CPU
    # answer, so the two sides agree.
    backend, gguf = _hybrid_mtp_backend(tmp_path, partial_offload = True, memory = [])

    result = _launch(
        backend,
        gguf,
        speculative_type = "auto",
        extra_args = ["--device", "Vulkan0", "--gpu-layers", "42"],
        n_ctx = 4096,
        n_parallel = 4,
    )

    cmd = result["cmd"]
    assert cmd[cmd.index("--device") + 1] == "Vulkan0"
    assert cmd[cmd.index("--spec-type") + 1] == "none"
    assert backend.spec_fallback_reason == "mtp_partial_offload"


def test_partial_offload_stand_down_records_the_draft_depth_it_decided_at(tmp_path):
    backend, gguf = _hybrid_mtp_backend(tmp_path, partial_offload = True)

    result = _launch(
        backend,
        gguf,
        speculative_type = "auto",
        spec_draft_n_max = 3,
        n_ctx = 4096,
        n_parallel = 4,
    )

    cmd = result["cmd"]
    assert cmd[cmd.index("--spec-type") + 1] == "none"
    assert backend.spec_fallback_reason == "mtp_partial_offload"
    # Nothing drafts, so the flag is not emitted -- but the depth priced the
    # rollback copies that made this placement partial, so it is recorded for the
    # reload comparison (test_llama_cpp_mtp_detection.py owns that half).
    assert "--spec-draft-n-max" not in cmd
    assert backend.spec_draft_n_max == 3


def test_manual_auto_layers_is_not_evidence_of_partial_offload(tmp_path):
    # Manual mode empties the probed GPU set to hand sizing to llama.cpp, so its
    # --fit on is the value this path starts at, not a finding. Reading it as
    # partial offload disabled MTP on a card with room for every layer.
    backend, gguf = _hybrid_mtp_backend(tmp_path, partial_offload = True)

    result = _launch(
        backend,
        gguf,
        speculative_type = "auto",
        gpu_memory_mode = "manual",
        gpu_layers = -1,
        n_ctx = 4096,
        n_parallel = 4,
    )

    cmd = result["cmd"]
    assert cmd[cmd.index("--fit") + 1] == "on"
    assert "draft-mtp" in cmd[cmd.index("--spec-type") + 1]
    assert backend.spec_fallback_reason is None


def test_manual_auto_layers_still_reads_a_pass_through_layer_count(tmp_path):
    # The evidence Manual mode does carry: a concrete count in the extras. That
    # still stands the drafter down, so declining to guess costs nothing where the
    # user actually said where the layers go.
    backend, gguf = _hybrid_mtp_backend(tmp_path, partial_offload = True)

    result = _launch(
        backend,
        gguf,
        speculative_type = "auto",
        gpu_memory_mode = "manual",
        gpu_layers = -1,
        extra_args = ["--gpu-layers", "42"],
        n_ctx = 4096,
        n_parallel = 4,
    )

    cmd = result["cmd"]
    assert cmd[cmd.index("--spec-type") + 1] == "none"
    assert backend.spec_fallback_reason == "mtp_partial_offload"


def test_auto_disables_embedded_hybrid_mtp_for_final_partial_layer_override(tmp_path):
    backend, gguf = _hybrid_mtp_backend(tmp_path, partial_offload = False)

    result = _launch(
        backend,
        gguf,
        speculative_type = "auto",
        extra_args = ["--gpu-layers", "42"],
        n_ctx = 4096,
        n_parallel = 4,
    )

    cmd = result["cmd"]
    assert cmd[-2:] == ["--gpu-layers", "42"]
    assert cmd[cmd.index("--spec-type") + 1] == "none"
    assert backend.spec_fallback_reason == "mtp_partial_offload"


def test_auto_reports_the_binary_not_the_placement_when_the_build_lacks_mtp(tmp_path):
    # Nothing to stand down: this build cannot run MTP at all, so the placement
    # story would send the user to force a mode it does not have, and hide the
    # update affordance the binary fallback carries.
    backend, gguf = _hybrid_mtp_backend(tmp_path, partial_offload = True)
    backend.probe_server_capabilities = lambda _binary = None: {
        "mtp_token": None,
        "mtp_probe_inconclusive": False,
        "supports_ngram_mod": False,
        "spec_draft_n_max_flag": "--spec-draft-n-max",
    }

    result = _launch(
        backend,
        gguf,
        speculative_type = "auto",
        n_ctx = 4096,
        n_parallel = 4,
    )

    cmd = result["cmd"]
    assert "--spec-type" not in cmd
    assert "--spec-default" in cmd
    assert backend.spec_fallback_reason == "binary_no_mtp"


def test_auto_classifies_placement_on_the_device_flags_the_child_gets(tmp_path):
    # An explicit gpu_ids pick owns placement, so the launch drops the stale
    # --device none from the extras further down. Classifying before that strip
    # would read CPU-only for a load that partially offloads.
    backend, gguf = _hybrid_mtp_backend(tmp_path, partial_offload = True)

    result = _launch(
        backend,
        gguf,
        speculative_type = "auto",
        gpu_ids = [0],
        extra_args = ["--device", "none"],
        n_ctx = 4096,
        n_parallel = 4,
    )

    cmd = result["cmd"]
    # The strip already ran: the child never sees the CPU device the classifier
    # would otherwise have believed.
    assert "--device" not in cmd
    assert cmd[cmd.index("--spec-type") + 1] == "none"
    assert backend.spec_fallback_reason == "mtp_partial_offload"


def _hybrid_reserve_backend(tmp_path: Path, *, caps = None):
    """A Hybrid Mamba target on one 24 GB card with the MTP-overhead math live.

    The drafter's own KV is stubbed away so the only moving term is the target's
    recurrent rollback state, which is what the reserve has to keep charging.
    """
    gb = 1024**3
    backend, gguf = _backend(tmp_path, vulkan = False, memory = [(0, 24_576, 24_576)])
    sidecar = tmp_path / "dflash-model-Q8_0.gguf"
    sidecar.write_bytes(b"draft")
    backend._get_gguf_size_bytes = lambda path: 0 if str(path) == str(sidecar) else 8 * gb
    backend._can_estimate_kv = lambda: True
    backend._compute_buffer_ctx_bytes = lambda *args, **kwargs: 0
    backend._estimate_compute_buffer_bytes = lambda **kwargs: 1
    backend._mtp_draft_kv_bytes = lambda *args, **kwargs: 0
    backend._select_gpus = lambda *args, **kwargs: ([0], False)
    backend._select_gpus_split_aware = lambda *args, **kwargs: ([0], False)

    def read_metadata(_path):
        backend._nextn_predict_layers = 1
        backend._n_layers = 65
        backend._n_kv_heads = 4
        backend._n_heads = 24
        backend._embedding_length = 5120
        backend._kv_key_length = 256
        backend._kv_value_length = 256
        backend._full_attention_interval = 4
        backend._ssm_inner_size = 6144
        backend._ssm_state_size = 128
        backend._ssm_group_count = 16
        backend._ssm_conv_kernel = 4

    backend._read_gguf_metadata = read_metadata
    backend.probe_server_capabilities = lambda _binary = None: (
        caps
        or {
            "mtp_token": "draft-mtp",
            "supports_dflash": True,
            "supports_ngram_mod": True,
            "spec_draft_n_max_flag": "--spec-draft-n-max",
            # Or the launch clamps the four slots to one and the per-slot state,
            # which is what these tests measure, shrinks with them.
            "supports_kv_unified": True,
        }
    )
    return backend, gguf, sidecar


def _recorded_mtp_reserve(backend, gguf, **load_kwargs):
    """The bytes the fit was asked to hold back for speculation."""
    charged, _fns = _recorded_mtp_reserve_and_callbacks(backend, gguf, **load_kwargs)
    return charged


def _recorded_mtp_reserve_and_callbacks(backend, gguf, **load_kwargs):
    """The reserve the fit saw, plus the callback objects it was handed."""
    charged = []
    callbacks = []
    _fit = backend._fit_context_to_vram

    def recording_fit(requested, *args, **kwargs):
        fn = kwargs.get("mtp_overhead_fn")
        callbacks.append(fn)
        charged.append(0 if fn is None else int(fn(requested) or 0))
        return _fit(requested, *args, **kwargs)

    backend._fit_context_to_vram = recording_fit
    _launch(backend, gguf, **load_kwargs)
    assert charged, "the fit never ran, so this proves nothing"
    return charged, callbacks


def test_a_cpu_pinned_drafter_still_pays_the_hybrid_target_rollback(tmp_path):
    # -ngld 0 moves the drafter's weights and KV to host memory, but the rollback
    # snapshots live in the TARGET context, so they stay on the GPU. Releasing the
    # whole reserve here undercounts them and the fit can pick a placement that
    # spills.
    backend, gguf, sidecar = _hybrid_reserve_backend(tmp_path)

    charged = _recorded_mtp_reserve(
        backend,
        gguf,
        dflash_draft_path = str(sidecar),
        speculative_type = "dflash",
        n_ctx = 8192,
        n_parallel = 4,
        extra_args = ["--spec-draft-ngl", "0"],
    )

    # After the launch: the GGUF dims land when the load reads the metadata.
    expected = backend._mamba_recurrent_state_bytes(n_parallel = 4) * 2
    assert expected > 0
    assert set(charged) == {expected}


def test_the_cpu_drafter_reserve_still_reprices_per_slot_candidate(tmp_path):
    # _slots_that_fit_on_gpu re-prices the reserve for each candidate slot count
    # through the callback's _np / _n_ubatch keywords. A replacement that takes
    # neither raises TypeError there, and the broad GPU-selection handler swallows
    # it into --fit on, throwing the whole placement plan away.
    backend, gguf, sidecar = _hybrid_reserve_backend(tmp_path)

    _charged, callbacks = _recorded_mtp_reserve_and_callbacks(
        backend,
        gguf,
        dflash_draft_path = str(sidecar),
        speculative_type = "dflash",
        n_ctx = 8192,
        n_parallel = 4,
        extra_args = ["--spec-draft-ngl", "0"],
    )

    fn = callbacks[0]
    assert fn is not None
    for slots in (1, 2, 4):
        assert fn(8192, _np = slots, _n_ubatch = 512) == (
            backend._mamba_recurrent_state_bytes(n_parallel = slots) * 2
        )
    # Per-slot state, not per-token: context does not move it.
    assert fn(2048, _np = 4, _n_ubatch = 512) == fn(131072, _np = 4, _n_ubatch = 512)


@pytest.mark.parametrize(
    ("spec_type", "pays_rollback"),
    [("draft-dflash", True), ("draft-eagle3", True), ("draft-simple", False)],
)
def test_a_pass_through_drafter_pays_the_rollback_its_type_calls_for(
    tmp_path, spec_type, pays_rollback
):
    # need_n_rs_seq lists every draft-model type but draft-simple, so the extras
    # path has to read the type rather than assume either answer.
    backend, gguf, sidecar = _hybrid_reserve_backend(tmp_path)

    charged = _recorded_mtp_reserve(
        backend,
        gguf,
        speculative_type = "auto",
        n_ctx = 8192,
        n_parallel = 4,
        extra_args = [
            "--spec-type",
            spec_type,
            "--model-draft",
            str(sidecar),
            "--spec-draft-n-max",
            "2",
        ],
    )

    rollback = backend._mamba_recurrent_state_bytes(n_parallel = 4) * 2
    assert rollback > 0
    assert set(charged) == {rollback if pays_rollback else 0}


@pytest.mark.parametrize("requested_depth", [None, 2])
def test_a_pass_through_spec_block_budgets_the_depth_the_build_defaults_to(
    tmp_path, requested_depth
):
    # Unsloth emits no --spec-draft-n-max when the extras own the spec block, so
    # the child runs at the build's own default. Budgeting Unsloth's 2 instead
    # under-reserves the rollback copies, which scale directly with it -- and a
    # request field carries no further than the platform default does, since
    # neither is emitted.
    backend, gguf, _sidecar = _hybrid_reserve_backend(
        tmp_path,
        caps = {
            "mtp_token": "draft-mtp",
            "supports_ngram_mod": True,
            "spec_draft_n_max_flag": "--spec-draft-n-max",
            "spec_draft_n_max_default": 16,
            "supports_kv_unified": True,
        },
    )
    charged = _recorded_mtp_reserve(
        backend,
        gguf,
        speculative_type = "auto",
        spec_draft_n_max = requested_depth,
        n_ctx = 8192,
        n_parallel = 4,
        extra_args = ["--spec-type", "draft-mtp"],
    )

    base = backend._mamba_recurrent_state_bytes(n_parallel = 4)
    assert base > 0
    assert set(charged) == {16 * base}


def test_a_legacy_build_inherits_its_own_draft_depth_variable(tmp_path, monkeypatch):
    # A legacy build spells the pair --draft-max / LLAMA_ARG_DRAFT_MAX. Reading only
    # the post-rename name budgets the build default while the child drafts at the
    # inherited one.
    backend, gguf, _sidecar = _hybrid_reserve_backend(
        tmp_path,
        caps = {
            "mtp_token": "draft-mtp",
            "supports_ngram_mod": True,
            "spec_draft_n_max_flag": "--draft-max",
            "spec_draft_n_max_default": 8,
            "supports_kv_unified": True,
        },
    )
    monkeypatch.setenv("LLAMA_ARG_DRAFT_MAX", "32")

    charged = _recorded_mtp_reserve(
        backend,
        gguf,
        speculative_type = "auto",
        n_ctx = 8192,
        n_parallel = 4,
        extra_args = ["--spec-type", "draft-mtp"],
    )

    base = backend._mamba_recurrent_state_bytes(n_parallel = 4)
    assert base > 0
    assert set(charged) == {32 * base}


def test_a_post_rename_build_ignores_the_legacy_depth_variable(tmp_path, monkeypatch):
    # LLAMA_ARG_DRAFT_MAX is the twin of the removed --draft-max, so a build that
    # advertises the modern flag never reads it. Pricing a stale value there would
    # budget a depth the child does not draft at.
    backend, gguf, _sidecar = _hybrid_reserve_backend(
        tmp_path,
        caps = {
            "mtp_token": "draft-mtp",
            "supports_ngram_mod": True,
            "spec_draft_n_max_flag": "--spec-draft-n-max",
            "spec_draft_n_max_default": 16,
            "supports_kv_unified": True,
        },
    )
    monkeypatch.delenv("LLAMA_ARG_SPEC_DRAFT_N_MAX", raising = False)
    monkeypatch.setenv("LLAMA_ARG_DRAFT_MAX", "32")

    charged = _recorded_mtp_reserve(
        backend,
        gguf,
        speculative_type = "auto",
        n_ctx = 8192,
        n_parallel = 4,
        extra_args = ["--spec-type", "draft-mtp"],
    )

    base = backend._mamba_recurrent_state_bytes(n_parallel = 4)
    assert base > 0
    assert set(charged) == {16 * base}


def test_an_unreadable_help_budgets_the_deepest_shipped_draft_depth(tmp_path):
    # The probe timed out, or the help line carries no default. The child is still
    # drafting at whatever the build defaults to, so Unsloth's own explicit-mode 2
    # would under-reserve the rollback copies by up to eight times.
    backend, gguf, _sidecar = _hybrid_reserve_backend(
        tmp_path,
        caps = {
            "mtp_token": "draft-mtp",
            "supports_ngram_mod": True,
            "spec_draft_n_max_flag": "--spec-draft-n-max",
            "supports_kv_unified": True,
        },
    )

    charged = _recorded_mtp_reserve(
        backend,
        gguf,
        speculative_type = "auto",
        n_ctx = 8192,
        n_parallel = 4,
        extra_args = ["--spec-type", "draft-mtp"],
    )

    base = backend._mamba_recurrent_state_bytes(n_parallel = 4)
    assert base > 0
    assert set(charged) == {LlamaCppBackend._UNKNOWN_SPEC_DRAFT_N_MAX * base}


def test_an_explicit_pin_the_probe_cannot_see_is_not_a_partial_verdict(tmp_path):
    # The probe answered nothing, but the pick still pins the child to those
    # devices, so the launch does offload to them: the probe-only view read this
    # as a CPU-only box, and the GPU-evidence guard is right to accept the pin.
    #
    # That is where the pin's authority stops. Every planner branch is gated on a
    # non-empty `gpus`, so with an empty probe none of them ran and the `--fit on`
    # below is the default use_fit starts at, not a finding that the model does
    # not fit -- the same reasoning _partially_offloads_layers already applies to
    # Manual mode. Standing MTP down here would cost the drafting win on a card
    # that may well hold every layer, so Auto keeps MTP until something actually
    # says the placement is partial.
    backend, gguf = _hybrid_mtp_backend(tmp_path, partial_offload = True, memory = [])

    result = _launch(
        backend,
        gguf,
        speculative_type = "auto",
        gpu_ids = [0],
        n_ctx = 4096,
        n_parallel = 4,
    )

    cmd = result["cmd"]
    assert cmd[cmd.index("--fit") + 1] == "on"
    assert cmd[cmd.index("--spec-type") + 1] == "draft-mtp"
    assert backend.spec_fallback_reason != "mtp_partial_offload"


def test_an_unseen_pin_with_a_concrete_layer_count_still_stands_down(tmp_path):
    # The other half: a fixed 42 of 65 blocks is partial placement on its own
    # evidence, so the empty probe costs the stand-down nothing here.
    backend, gguf = _hybrid_mtp_backend(tmp_path, partial_offload = True, memory = [])

    result = _launch(
        backend,
        gguf,
        speculative_type = "auto",
        gpu_ids = [0],
        n_ctx = 4096,
        n_parallel = 4,
        extra_args = ["--gpu-layers", "42"],
    )

    cmd = result["cmd"]
    assert cmd[cmd.index("--spec-type") + 1] == "none"
    assert backend.spec_fallback_reason == "mtp_partial_offload"


def _tight_vram_backend(tmp_path: Path, *, drafter_gb: float):
    """One 24 GB card, a 16 GB target and a drafter of the caller's size.

    The fit terms are stubbed to constants so the only variable is whether the
    drafter's reserve clears the pin budget.
    """
    gb = 1024**3
    backend, gguf = _backend(tmp_path, vulkan = False, memory = [(0, 24_576, 24_576)])
    sidecar = tmp_path / "dspark-model-Q8_0.gguf"
    sidecar.write_bytes(b"draft")
    backend._get_gguf_size_bytes = lambda path: (
        int(drafter_gb * gb) if str(path) == str(sidecar) else 16 * gb
    )
    backend._can_estimate_kv = lambda: True
    backend._estimate_kv_cache_bytes = lambda *args, **kwargs: 1 * gb
    backend._compute_buffer_ctx_bytes = lambda *args, **kwargs: 0
    # Positive, or the fit swaps in its 5 GB flat reserve and swamps the numbers.
    backend._estimate_compute_buffer_bytes = lambda **kwargs: 1
    backend._mtp_draft_kv_bytes = lambda *args, **kwargs: 0
    backend._estimate_mtp_overhead_bytes = lambda *args, **kwargs: int(drafter_gb * gb)
    backend._fit_context_to_vram = lambda requested, *args, **kwargs: requested
    backend._select_gpus = lambda *args, **kwargs: ([0], False)
    backend._select_gpus_split_aware = lambda *args, **kwargs: ([0], False)
    backend.probe_server_capabilities = lambda _binary = None: {
        "supports_dspark": True,
        "spec_draft_n_max_flag": "--spec-draft-n-max",
    }
    return backend, gguf, sidecar


def test_auto_drops_the_drafter_when_only_the_target_fits(tmp_path):
    """Model fits, drafter does not: Auto keeps the context and runs without it.

    The alternative today is a silently smaller context (or --fit offload, where
    decode collapses), paid for a speed option the user never asked for.
    """
    backend, gguf, sidecar = _tight_vram_backend(tmp_path, drafter_gb = 12.0)

    result = _launch(
        backend,
        gguf,
        dspark_draft_path = str(sidecar),
        speculative_type = "auto",
        n_ctx = 8192,
    )

    cmd = result["cmd"]
    assert "--model-draft" not in cmd
    assert "draft-dspark" not in cmd
    assert cmd[cmd.index("-c") + 1] == "8192"
    assert backend.spec_fallback_reason == "drafter_no_vram"
    # Names the drafter Auto had resolved, so the notice does not read "MTP", and
    # keeps the resolved path so a repeat Apply dedupes instead of relaunching.
    assert backend.spec_drafter_kind == "dspark"
    assert backend.mtp_draft_path == str(sidecar)


def test_auto_keeps_a_drafter_that_fits(tmp_path):
    """The drop is scoped to the shortfall: with room for both, nothing changes."""
    backend, gguf, sidecar = _tight_vram_backend(tmp_path, drafter_gb = 1.5)

    result = _launch(
        backend,
        gguf,
        dspark_draft_path = str(sidecar),
        speculative_type = "auto",
        n_ctx = 8192,
    )

    cmd = result["cmd"]
    assert cmd[cmd.index("--model-draft") + 1] == str(sidecar)
    assert cmd[cmd.index("--spec-type") + 1] == "draft-dspark"
    assert backend.spec_fallback_reason is None


def test_forcing_the_drafter_overrides_the_vram_drop(tmp_path):
    """Only Auto is second-guessed. An explicit choice launches the drafter and
    lets the existing context reduction pay for it."""
    backend, gguf, sidecar = _tight_vram_backend(tmp_path, drafter_gb = 12.0)

    result = _launch(
        backend,
        gguf,
        dspark_draft_path = str(sidecar),
        speculative_type = "dspark",
        n_ctx = 8192,
    )

    cmd = result["cmd"]
    assert cmd[cmd.index("--model-draft") + 1] == str(sidecar)
    assert cmd[cmd.index("--spec-type") + 1] == "draft-dspark"
    assert backend.spec_fallback_reason is None


def test_an_embedded_mtp_head_is_dropped_too(tmp_path):
    """No sidecar file to blank, so the drop has to reach the flags themselves.

    An embedded head still costs a draft KV and a verify graph, and the fit
    reserved neither; emitting --spec-type draft-mtp anyway would OOM the load.
    """
    backend, gguf, _sidecar = _tight_vram_backend(tmp_path, drafter_gb = 12.0)
    backend._read_gguf_metadata = lambda _path: setattr(backend, "_nextn_predict_layers", 1)
    backend.probe_server_capabilities = lambda _binary = None: {
        "mtp_token": "draft-mtp",
        "supports_ngram_mod": True,
        "spec_draft_n_max_flag": "--spec-draft-n-max",
    }

    result = _launch(backend, gguf, speculative_type = "auto", n_ctx = 8192)

    cmd = result["cmd"]
    assert "draft-mtp" not in cmd
    assert cmd[cmd.index("--spec-type") + 1] == "ngram-mod"
    assert cmd[cmd.index("-c") + 1] == "8192"
    assert backend.spec_fallback_reason == "drafter_no_vram"


def test_the_vram_drop_does_not_emit_ngram_mod_on_a_build_without_it(tmp_path):
    """`ngram-mod` is a value in llama.cpp's --spec-type enum, so a build that
    predates it aborts on the flag instead of ignoring it. The MLA and sub-3B
    fallbacks gate on the capability for exactly that reason; this one has to too,
    or the drop turns a slower load into a load that never starts."""
    backend, gguf, _sidecar = _tight_vram_backend(tmp_path, drafter_gb = 12.0)
    backend._read_gguf_metadata = lambda _path: setattr(backend, "_nextn_predict_layers", 1)
    backend.probe_server_capabilities = lambda _binary = None: {
        "mtp_token": "draft-mtp",
        "supports_ngram_mod": False,
        "spec_draft_n_max_flag": "--spec-draft-n-max",
    }

    result = _launch(backend, gguf, speculative_type = "auto", n_ctx = 8192)

    cmd = result["cmd"]
    assert "ngram-mod" not in cmd
    assert "--spec-type" not in cmd
    assert "draft-mtp" not in cmd
    assert cmd[cmd.index("-c") + 1] == "8192"
    assert backend.spec_fallback_reason == "drafter_no_vram"


def test_a_standalone_model_draft_in_extras_is_not_auto_dropped(tmp_path):
    """--model-draft alone sets no --spec-type, so neither extras probe fires, but
    llama-server loads whatever it names regardless of the spec type (load_model
    gates the draft model on has_dft(), i.e. "a draft path was given"). Dropping it
    releases the reserve for a drafter the child still loads, and it is an explicit
    user choice besides."""
    backend, gguf, sidecar = _tight_vram_backend(tmp_path, drafter_gb = 12.0)
    user_draft = tmp_path / "my-drafter.gguf"
    user_draft.write_bytes(b"draft")

    result = _launch(
        backend,
        gguf,
        dspark_draft_path = str(sidecar),
        speculative_type = "auto",
        n_ctx = 8192,
        extra_args = ["--model-draft", str(user_draft)],
    )

    cmd = result["cmd"]
    assert "ngram-mod" not in cmd
    assert cmd[cmd.index("--spec-type") + 1] == "draft-dspark"
    assert backend.spec_fallback_reason is None


def test_a_busy_second_gpu_does_not_condemn_a_drafter_the_first_one_holds(tmp_path):
    """A whole-pool figure is not the ceiling it looks like.

    A card with almost nothing free adds ~0 to the pooled budget, but a two-GPU
    layer split still charges its 1 GiB pipeline overhead, so pricing the drafter
    over the whole pool can reject one the single healthy GPU holds comfortably.
    The probe walks the same ranked subsets the placement loop does, so the 1-GPU
    placement it would actually pick is the one that decides.
    """
    backend, gguf, sidecar = _tight_vram_backend(tmp_path, drafter_gb = 5.0)
    # GPU1 is in use by something else: 800 MiB free of 24 GiB.
    backend._get_gpu_memory = lambda _binary = None: [
        (0, 24_576, 24_576),
        (1, 800, 24_576),
    ]
    backend._get_gpu_free_memory = lambda _binary = None: [(0, 24_576), (1, 800)]

    result = _launch(
        backend,
        gguf,
        dspark_draft_path = str(sidecar),
        speculative_type = "auto",
        n_ctx = 8192,
    )

    cmd = result["cmd"]
    assert cmd[cmd.index("--model-draft") + 1] == str(sidecar)
    assert cmd[cmd.index("--spec-type") + 1] == "draft-dspark"
    assert backend.spec_fallback_reason is None


def test_a_cpu_offloaded_sidecar_releases_the_byte_accurate_reserve(tmp_path):
    """-ngld 0 puts the drafter in host memory, and a separate sidecar displaces
    the embedded head that mtp_overhead_fn was sized from, so nothing speculative
    is GPU-resident. The flat fraction already stands down here; the byte-accurate
    callback did not, so the fit went on charging GPU bytes for a drafter that
    allocates none, cutting the context or taking --fit for them.
    """
    backend, gguf, sidecar = _tight_vram_backend(tmp_path, drafter_gb = 12.0)
    backend._nextn_predict_layers = 1
    backend.probe_server_capabilities = lambda _binary = None: {
        "supports_dspark": True,
        "supports_mtp": True,
        "mtp_token": "mtp",
        "spec_draft_n_max_flag": "--spec-draft-n-max",
    }
    charged = []
    _fit = backend._fit_context_to_vram

    def recording_fit(requested, *args, **kwargs):
        fn = kwargs.get("mtp_overhead_fn")
        charged.append(0 if fn is None else int(fn(requested) or 0))
        return _fit(requested, *args, **kwargs)

    backend._fit_context_to_vram = recording_fit

    _launch(
        backend,
        gguf,
        dspark_draft_path = str(sidecar),
        speculative_type = "auto",
        n_ctx = 8192,
        extra_args = ("--spec-draft-ngl", "0"),
    )

    assert charged, "the fit never ran, so this proves nothing"
    assert set(charged) == {0}


def test_an_mla_model_keeps_the_reason_that_actually_dropped_its_drafter(tmp_path):
    """An MLA embedded-MTP model has no drafter to save: Auto drops it by policy,
    because llama.cpp's MLA/DSA MTP path is slower than no speculation at all. If
    the VRAM branch claims it first, the notice tells the user to force MTP at a
    smaller context, i.e. to buy a known regression with their context length."""
    backend, gguf, _sidecar = _tight_vram_backend(tmp_path, drafter_gb = 12.0)
    # Embedded head, MLA geometry, no sidecar: exactly the GLM-5.2 shape.
    backend._nextn_predict_layers = 1
    backend._kv_lora_rank = 512
    backend.probe_server_capabilities = lambda _binary = None: {
        "supports_mtp": True,
        "mtp_token": "mtp",
        "supports_ngram_mod": True,
        "spec_draft_n_max_flag": "--spec-draft-n-max",
    }

    result = _launch(backend, gguf, speculative_type = "auto", n_ctx = 8192)

    cmd = result["cmd"]
    assert "draft-mtp" not in cmd
    assert cmd[cmd.index("--spec-type") + 1] == "ngram-mod"
    assert backend.spec_fallback_reason == "mla_mtp_disabled"


def test_tensor_parallel_keeps_its_own_sizing(tmp_path):
    """_plan_tensor_parallel reserves a per-device tensor buffer on geometry this
    layer-split probe does not model, so under tensor mode the probe stands down
    rather than decide the drafter's fate on numbers that are not that load's."""
    # Two cards that only hold the 16 GB target together, so the layer-split
    # probe would condemn the drafter if it were allowed to answer here.
    backend, gguf, sidecar = _tight_vram_backend(tmp_path, drafter_gb = 12.0)
    backend._get_gpu_memory = lambda _binary = None: [
        (0, 12_288, 12_288),
        (1, 12_288, 12_288),
    ]
    backend._get_gpu_free_memory = lambda _binary = None: [(0, 12_288), (1, 12_288)]
    backend._tensor_split_aborts = lambda *args, **kwargs: False

    result = _launch(
        backend,
        gguf,
        dspark_draft_path = str(sidecar),
        speculative_type = "auto",
        tensor_parallel = True,
        n_ctx = 8192,
    )

    assert backend.spec_fallback_reason != "drafter_no_vram"
    assert "--model-draft" in result["cmd"]


def test_a_tensor_request_that_aborted_before_is_probed_as_the_layer_load_it_is(tmp_path):
    """A recorded --split-mode tensor abort downgrades the load to a layer split
    before anything is planned, and the layer planner does reserve the Auto drafter
    (paying for it in context). Gating the probe on the REQUESTED tensor flag would
    hand that load the silent context cut the probe exists to prevent."""
    backend, gguf, sidecar = _tight_vram_backend(tmp_path, drafter_gb = 12.0)
    backend._tensor_split_aborts = lambda *args, **kwargs: True

    result = _launch(
        backend,
        gguf,
        dspark_draft_path = str(sidecar),
        speculative_type = "auto",
        tensor_parallel = True,
        n_ctx = 8192,
    )

    cmd = result["cmd"]
    assert "--split-mode" not in cmd
    assert "--model-draft" not in cmd
    assert cmd[cmd.index("-c") + 1] == "8192"
    assert backend.spec_fallback_reason == "drafter_no_vram"


def test_a_single_gpu_tensor_request_is_probed_as_the_layer_load_it_is(tmp_path):
    """Same shape, the commonest cause: tensor parallelism needs >= 2 usable GPUs,
    so a one-card request is downgraded to a layer split and must be probed."""
    backend, gguf, sidecar = _tight_vram_backend(tmp_path, drafter_gb = 12.0)
    backend._tensor_split_aborts = lambda *args, **kwargs: False

    result = _launch(
        backend,
        gguf,
        dspark_draft_path = str(sidecar),
        speculative_type = "auto",
        tensor_parallel = True,
        n_ctx = 8192,
    )

    cmd = result["cmd"]
    assert "--split-mode" not in cmd
    assert "--model-draft" not in cmd
    assert cmd[cmd.index("-c") + 1] == "8192"
    assert backend.spec_fallback_reason == "drafter_no_vram"


@pytest.mark.parametrize(
    "n_gpus, model_gb, aborts, load_kwargs",
    [
        # One row per strip site in load_model. Placement is the whole point, so
        # the rows are the sites, not the drop reasons -- two manual-mode drops
        # share a strip and would be one row's worth of coverage twice.
        (2, 1, True, {}),  # a recorded --split-mode tensor abort
        (1, 1, False, {}),  # fewer than 2 GPUs clear the compute-buffer reserve
        (2, 80, False, {}),  # pooled VRAM cannot hold the weights
        (2, 1, False, {"gpu_memory_mode": "manual"}),  # Auto layers: --fit owns memory
        # gpu_ids, not n_gpus: this guard counts the selection (or torch's visible
        # devices), so without a pin it passes only because torch is absent here.
        (2, 1, False, {"gpu_memory_mode": "manual", "gpu_layers": 20, "gpu_ids": [0]}),
    ],
)
def test_a_dropped_tensor_request_launches_as_a_layer_split(
    tmp_path, n_gpus, model_gb, aborts, load_kwargs
):
    """A downgrade has to land a working layer split, not merely lose a flag: the
    server comes up in layer mode and the user's unrelated extras still reach it.
    Extras are appended last, so a --split-mode tensor left among them would
    re-engage the mode the downgrade just dropped."""
    backend, gguf = _backend(
        tmp_path,
        vulkan = False,
        memory = [(i, 24_000, 24_000) for i in range(n_gpus)],
    )
    backend._tensor_split_aborts = lambda *args, **kwargs: aborts
    # _backend stubs the weights at 1 KB; only a real size trips the pooled-VRAM case.
    backend._get_gguf_size_bytes = lambda _path: model_gb * 1024**3

    cmd = _launch(
        backend,
        gguf,
        tensor_parallel = True,
        extra_args = ["--split-mode", "tensor", "--tensor-split", "3,1", "--top-k", "5"],
        **load_kwargs,
    )["cmd"]

    # The load is the layer split the downgrade chose ...
    assert backend.tensor_parallel is False
    # ... it still carries the user's unrelated extras ...
    assert "--top-k" in cmd
    # ... and not the split-mode group -- --tensor-split rides with the mode, so a
    # strip narrowed to --split-mode alone leaves the user's ratio behind.
    assert "--split-mode" not in cmd
    assert "--tensor-split" not in cmd


def test_the_probe_prices_the_drafter_at_a_context_the_weakest_card_can_hold(tmp_path):
    """The compute buffer is replicated on every device of a layer split, so a
    pooled budget can price a context the smallest card cannot hold; the placement
    loop catches that with _every_gpu_holds_reserve and caps to what it does hold.

    A probe comparing pooled footprints only condemns the drafter at that
    unattainable context, even though both fit at the context the real placement
    must use. The numbers (Auto context, native 8192): the target alone fits on the
    big card, the pair fits pooled at 8192, but the 1.5 GB card cannot hold the
    1 GiB pipeline overhead plus its own 8192-token buffer copy, so 5888 is the real
    ceiling -- and at 5888 the drafter fits.
    """
    mib = 1024**2
    backend, gguf, sidecar = _tight_vram_backend(tmp_path, drafter_gb = 1.0)
    backend._read_gguf_metadata = lambda _path: setattr(backend, "_context_length", 8192)
    backend._get_gpu_memory = lambda _binary = None: [
        (0, 19_588, 19_588),
        (1, 1_546, 1_546),
    ]
    backend._get_gpu_free_memory = lambda _binary = None: [(0, 19_588), (1, 1_546)]
    # Context-linear, so the per-device reserve (and the drafter) shrink with a cap.
    backend._compute_buffer_ctx_bytes = lambda n_ctx, *args, **kwargs: n_ctx * 83_886
    backend._estimate_mtp_overhead_bytes = lambda ctx, *args, **kwargs: ctx * 94_371
    # Sanity on the geometry the assertions below rest on (MiB).
    assert 1024 + 8192 * 83_886 / mib > 1_546 * 0.97
    assert 1024 + 5888 * 83_886 / mib <= 1_546 * 0.97

    result = _launch(
        backend,
        gguf,
        dspark_draft_path = str(sidecar),
        speculative_type = "auto",
        # 0 = Auto context (the branch that caps); the native 8192 above is the target.
        n_ctx = 0,
    )

    cmd = result["cmd"]
    assert cmd[cmd.index("--model-draft") + 1] == str(sidecar)
    assert cmd[cmd.index("--spec-type") + 1] == "draft-dspark"
    assert backend.spec_fallback_reason is None


def test_the_drop_actually_releases_the_reserve_the_fit_charges(tmp_path):
    """The drop has to reach the fit, not just the launch.

    Every _mtp_bytes site in the fit is unconditional and _fit_context_to_vram
    calls any non-None mtp_overhead_fn whatever mtp_engaged says, so clearing
    _mtp_will_engage alone still let the planner shrink the context for a drafter
    it no longer launches. Deliberately does NOT stub _fit_context_to_vram or the
    GPU selectors: the point is the context the real fit arrives at.
    """
    gb = 1024**3
    mib = 1024**2
    backend, gguf = _backend(tmp_path, vulkan = False, memory = [(0, 24_576, 24_576)])
    sidecar = tmp_path / "dspark-model-Q8_0.gguf"
    sidecar.write_bytes(b"draft")
    backend._get_gguf_size_bytes = lambda path: 6 * gb if str(path) == str(sidecar) else 16 * gb
    backend._read_gguf_metadata = lambda _path: setattr(backend, "_context_length", 8192)
    backend._can_estimate_kv = lambda: True
    # Context-linear, so an unreleased drafter reserve is paid for in context.
    backend._estimate_kv_cache_bytes = lambda ctx, *args, **kwargs: int(ctx * 0.5 * mib)
    backend._compute_buffer_ctx_bytes = lambda *args, **kwargs: 0
    backend._estimate_compute_buffer_bytes = lambda **kwargs: 1
    backend._mtp_draft_kv_bytes = lambda *args, **kwargs: 0
    backend._estimate_mtp_overhead_bytes = lambda *args, **kwargs: 6 * gb
    backend.probe_server_capabilities = lambda _binary = None: {
        "supports_dspark": True,
        "supports_ngram_mod": True,
        "spec_draft_n_max_flag": "--spec-draft-n-max",
    }

    result = _launch(
        backend,
        gguf,
        dspark_draft_path = str(sidecar),
        speculative_type = "auto",
        n_ctx = 0,
    )

    cmd = result["cmd"]
    # 16 GB + a 4 GB KV at 8192 clears the 23.3 GB pin budget; + 6 GB does not.
    assert "--model-draft" not in cmd
    assert backend.spec_fallback_reason == "drafter_no_vram"
    # The whole point: native context survives, rather than being cut to pay for
    # a drafter that is not launching.
    assert cmd[cmd.index("-c") + 1] == "8192"
    assert cmd[cmd.index("--fit") + 1] == "off"


def test_a_cpu_offloaded_sidecar_is_not_probed_because_a_head_also_exists(tmp_path):
    """The drafter that launches decides, not one that merely exists.

    llama.cpp loads the draft model on has_dft(), so a separate sidecar wins over
    an embedded head; pinned to CPU it takes no GPU reserve, and there is nothing
    for the shortfall probe to drop. Keying the exemption on "no embedded head"
    dropped a sidecar that was never on the GPU in the first place.
    """
    backend, gguf, sidecar = _tight_vram_backend(tmp_path, drafter_gb = 12.0)
    backend._read_gguf_metadata = lambda _path: setattr(backend, "_nextn_predict_layers", 1)

    result = _launch(
        backend,
        gguf,
        dspark_draft_path = str(sidecar),
        speculative_type = "auto",
        n_ctx = 8192,
        extra_args = ["--spec-draft-ngl", "0"],
    )

    assert backend.spec_fallback_reason != "drafter_no_vram"
    assert "--model-draft" in result["cmd"]


def test_a_cpu_offloaded_sidecar_reserves_no_gpu_despite_an_embedded_head(tmp_path):
    """The exemption has to reach the reserve, not just the probe.

    _mtp_reserves_gpu kept the flat fraction and draft-compute reserve alive for
    an embedded head the launch never uses, so the context still shrank for GPU
    memory nothing allocates. One definition now serves both.
    """
    backend, gguf, sidecar = _tight_vram_backend(tmp_path, drafter_gb = 12.0)

    def _meta(_path):
        backend._nextn_predict_layers = 1
        backend._context_length = 8192

    backend._read_gguf_metadata = _meta
    reserved = []
    backend._fit_context_to_vram = lambda requested, *a, **k: (
        reserved.append(k.get("mtp_engaged")) or requested
    )

    _launch(
        backend,
        gguf,
        dspark_draft_path = str(sidecar),
        speculative_type = "auto",
        n_ctx = 0,
        extra_args = ["--spec-draft-ngl", "0"],
    )

    assert reserved, "the fit never ran"
    assert not any(reserved), f"mtp_engaged should be False throughout, got {reserved}"


def _shrink_to_hold_both_backend(
    tmp_path, *, model_gb, native_ctx, kv_mib_per_tok, mtp_mib_per_tok
):
    """Two 24 GiB cards and a DSpark sidecar, with every VRAM term context-linear.

    ``kv_mib_per_tok`` prices the target's cache and ``mtp_mib_per_tok`` the whole
    drafter reserve, weights included: the stub replaces
    ``_estimate_mtp_overhead_bytes`` outright, so the sidecar's file size never
    reaches the arithmetic and only these rates decide what holds what.
    """
    gb = 1024**3
    backend, gguf = _backend(
        tmp_path, vulkan = False, memory = [(0, 24_576, 24_576), (1, 24_576, 24_576)]
    )
    sidecar = tmp_path / "dspark-model-Q8_0.gguf"
    sidecar.write_bytes(b"draft")
    backend._get_gguf_size_bytes = lambda path: (
        8 * gb if str(path) == str(sidecar) else int(model_gb * gb)
    )
    backend._read_gguf_metadata = lambda _path: setattr(backend, "_context_length", native_ctx)
    backend._can_estimate_kv = lambda: True
    backend._estimate_kv_cache_bytes = lambda ctx, *a, **k: int(ctx * kv_mib_per_tok * 1024**2)
    backend._compute_buffer_ctx_bytes = lambda *a, **k: 0
    backend._estimate_compute_buffer_bytes = lambda **k: 1
    backend._mtp_draft_kv_bytes = lambda *a, **k: 0
    backend._estimate_mtp_overhead_bytes = lambda ctx, *a, **k: int(ctx * mtp_mib_per_tok * 1024**2)
    backend.probe_server_capabilities = lambda _binary = None: {
        "supports_dspark": True,
        "supports_ngram_mod": True,
        "spec_draft_n_max_flag": "--spec-draft-n-max",
    }
    return backend, gguf, sidecar


def test_a_subset_that_can_shrink_to_hold_both_is_where_the_decision_lands(tmp_path):
    """The placement loop does not walk past a subset that fails with the drafter.

    It re-caps the context WITH the drafter charged and accepts that subset at
    whatever is left, so a smaller context holding both here IS the placement the
    load takes, and the drafter gets paid for in context. Believing a later,
    larger subset would rescue it keeps the drafter and shrinks the context, which
    is the trade this exists to refuse.

    The shrink has to land ABOVE the fit floor to be a measurement rather than the
    floor itself, which is what makes this scenario 32768-native rather than 8192.
    On one card, budget 23838 MiB: the target alone is 14336 + 320 = 14656 MiB and
    holds 32768 (+8192 MiB of cache) with room to spare, the drafter's 8192 MiB on
    top does not, and 14336 + 544 leaves 8958 MiB, which at 0.5 MiB/token both ways
    re-caps to 17664. That is the shrink the loop takes and the drafter is refused
    for. Two cards WOULD have held both at the full 32768 (32288 of 47677 MiB
    pooled), so the refusal is a choice and not an absence of alternatives; see
    test_widening_beats_shrinking_below_the_fit_floor for the case where the shrink
    is not available and widening is what happens instead.
    """
    backend, gguf, sidecar = _shrink_to_hold_both_backend(
        tmp_path,
        model_gb = 14,
        native_ctx = 32_768,
        kv_mib_per_tok = 0.25,
        mtp_mib_per_tok = 0.25,
    )

    result = _launch(
        backend,
        gguf,
        dspark_draft_path = str(sidecar),
        speculative_type = "auto",
        n_ctx = 0,
    )

    cmd = result["cmd"]
    assert "--model-draft" not in cmd
    assert backend.spec_fallback_reason == "drafter_no_vram"
    assert cmd[cmd.index("-c") + 1] == "32768"
    # One card: the point is that the loop stopped here rather than widening.
    assert result["env"].get("CUDA_VISIBLE_DEVICES") == "0"


def test_widening_beats_shrinking_below_the_fit_floor(tmp_path):
    """The other side of the same branch: no shrink is on offer, so the loop widens.

    Same machine, sized so one card can only hold both BELOW the fit floor. The
    re-cap with the drafter charged is bounded by that floor, so it cannot return
    the shrink, the caller re-prices its answer and rejects it, and the loop walks
    on to the two-card subset that holds the target AND the drafter at the full
    context. Keeping the drafter for free is the whole reason the floor exists, and
    the placement is a real fit rather than a floored one: 17952 MiB of weights and
    overhead plus 4096 of cache plus 6144 of drafter is 28192 of a 47677 MiB pool,
    about 14 GiB on each 24 GiB card.

    Pinning it because the shape that would flip it back is silent: the re-cap
    helper returns ``min_ctx`` rather than 0 when even ``min_ctx`` does not fit, so
    the only thing standing between this and a one-card placement that OOMs is the
    caller's own footprint re-check.
    """
    backend, gguf, sidecar = _shrink_to_hold_both_backend(
        tmp_path,
        model_gb = 16,
        native_ctx = 8192,
        kv_mib_per_tok = 0.5,
        mtp_mib_per_tok = 0.75,
    )

    result = _launch(
        backend,
        gguf,
        dspark_draft_path = str(sidecar),
        speculative_type = "auto",
        n_ctx = 0,
    )

    cmd = result["cmd"]
    assert "--model-draft" in cmd
    assert backend.spec_fallback_reason is None
    assert cmd[cmd.index("-c") + 1] == "8192"
    assert result["env"].get("CUDA_VISIBLE_DEVICES") == "0,1"


def _restore_host_guard(backend):
    """Put the real preflight back on a harness that stubs it off by default."""
    backend._launch_host_shortfall_message = LlamaCppBackend._launch_host_shortfall_message.__get__(
        backend
    )
    return backend


def _offload_backend(tmp_path, *, gguf_gb, free_mib, avail_mib, monkeypatch, **kwargs):
    backend, gguf = _backend(tmp_path, vulkan = False, memory = [(0, free_mib, 6141)])
    _restore_host_guard(backend)
    backend._get_gguf_size_bytes = lambda _path: int(gguf_gb * 1024**3)
    # no subset holds the model, so --fit on owns placement and spills to host ram
    backend._select_gpus = lambda *args, **kw: (None, True)
    for name, value in kwargs.items():
        setattr(backend, name, value)
    monkeypatch.setattr(
        LlamaCppBackend, "_available_system_memory_mib", staticmethod(lambda: avail_mib)
    )
    return backend, gguf


def test_weights_larger_than_vram_plus_ram_still_load_with_a_warning(tmp_path, monkeypatch):
    """The field case: a 13.3 GB GGUF on a 6 GB laptop card holding 4877 MiB free needs
    about 8.5 GB of host RAM, which a 10 GB host cannot hold. It loads anyway, paging
    the remainder from disk, and says so."""
    backend, gguf = _offload_backend(
        tmp_path, gguf_gb = 13.3, free_mib = 4877, avail_mib = 10_000, monkeypatch = monkeypatch
    )

    _launch_warns(backend, gguf)


def test_the_same_load_on_a_large_ram_host_still_launches(tmp_path, monkeypatch):
    """Deliberate CPU offload stays supported; only a shortfall refuses."""
    backend, gguf = _offload_backend(
        tmp_path, gguf_gb = 13.3, free_mib = 4877, avail_mib = 64_000, monkeypatch = monkeypatch
    )

    assert "--fit" in _launch(backend, gguf)["cmd"]


def test_free_vram_offsets_the_charge(tmp_path, monkeypatch):
    """Same model and same host RAM as the refusal above, but a card big enough to hold
    it. The VRAM credit is what separates the two, so the charge is the shortfall and
    not the model size."""
    backend, gguf = _offload_backend(
        tmp_path, gguf_gb = 13.3, free_mib = 20_000, avail_mib = 10_000, monkeypatch = monkeypatch
    )

    assert "--fit" in _launch(backend, gguf)["cmd"]


@pytest.mark.parametrize(
    "memory",
    [
        [(0, 12 * 1024, 0)],
        [(0, 12 * 1024, 0), (1, 12 * 1024, 0)],
    ],
    ids = ["one-shared-device", "two-shared-devices"],
)
def test_vulkan_igpu_shared_memory_is_not_counted_twice(tmp_path, monkeypatch, memory):
    """Shared Vulkan rows and host RAM describe one pool."""
    backend, gguf = _backend(
        tmp_path,
        vulkan = True,
        memory = memory,
    )
    _restore_host_guard(backend)
    backend._get_gguf_size_bytes = lambda _path: 20 * 1024**3
    backend._select_gpus = lambda *args, **kwargs: (None, True)
    monkeypatch.setattr(
        LlamaCppBackend, "_available_system_memory_mib", staticmethod(lambda: 14 * 1024)
    )
    monkeypatch.setattr(LlamaCppBackend, "_cgroup_available_memory_mib", staticmethod(lambda: None))

    _launch_warns(backend, gguf)


def test_vulkan_igpu_heap_can_hold_weights_missing_from_host_available(tmp_path, monkeypatch):
    """A firmware carve-out remains usable when host-available RAM is low."""
    backend, gguf = _backend(
        tmp_path,
        vulkan = True,
        memory = [(0, 107 * 1024, 0)],
    )
    _restore_host_guard(backend)
    backend._get_gguf_size_bytes = lambda _path: int(16.5 * 1024**3)
    monkeypatch.setattr(
        LlamaCppBackend, "_available_system_memory_mib", staticmethod(lambda: 13 * 1024)
    )
    monkeypatch.setattr(
        LlamaCppBackend, "_total_system_memory_mib", staticmethod(lambda: 32 * 1024)
    )
    monkeypatch.setattr(LlamaCppBackend, "_cgroup_available_memory_mib", staticmethod(lambda: None))

    assert _launch(backend, gguf)["cmd"]


@pytest.mark.parametrize(
    "gguf_mib,admitted",
    [(4096, True), (8192, True), (8256, False), (8704, False), (9216, False)],
    ids = ["4-gib", "8-gib", "8.06-gib", "8.5-gib", "9-gib"],
)
def test_vulkan_igpu_backing_bound_preserves_placement_and_host_headroom(
    tmp_path, monkeypatch, gguf_mib, admitted
):
    """The raw planner reading never lets host-backed credit lose system headroom."""
    backend, gguf = _backend(tmp_path, vulkan = True, memory = [(0, 15 * 1024, 0)])
    _restore_host_guard(backend)
    backend._get_gpu_memory = lambda _binary = None, **_kw: (
        LlamaCppBackend._get_gpu_free_memory_vulkan(_binary)
    )
    backend._get_gguf_size_bytes = lambda _path: gguf_mib * 1024**2
    monkeypatch.setattr(
        LlamaCppBackend,
        "_run_vulkan_probe",
        staticmethod(
            lambda _binary = None: [
                {
                    "index": 0,
                    "free_mib": 16 * 1024,
                    "is_igpu": True,
                    "total_mib": 16 * 1024,
                    "name": "Vulkan0",
                }
            ]
        ),
    )
    monkeypatch.setattr(
        LlamaCppBackend, "_available_system_memory_mib", staticmethod(lambda: 10 * 1024)
    )
    monkeypatch.setattr(
        LlamaCppBackend, "_total_system_memory_mib", staticmethod(lambda: 16 * 1024)
    )
    monkeypatch.setattr(LlamaCppBackend, "_cgroup_available_memory_mib", staticmethod(lambda: None))

    if not admitted:
        _launch_warns(backend, gguf)
        return

    cmd = _launch(backend, gguf)["cmd"]
    assert cmd[cmd.index("-ngl") + 1] == "-1"
    assert cmd[cmd.index("--fit") + 1] == "off"
    assert cmd[cmd.index("--device") + 1] == "Vulkan0"


@pytest.mark.parametrize(
    "placement",
    [
        {"gpu_memory_mode": "manual", "gpu_layers": 0},
        {"gpu_memory_mode": "manual", "gpu_layers": 8},
        {"extra_args": ["--device", "none"]},
        {"extra_args": ["-ngl", "0"]},
    ],
    ids = ["manual-zero-offload", "manual-partial-offload", "device-none", "extras-zero-offload"],
)
def test_vulkan_igpu_heap_is_not_credited_to_a_host_resident_launch(
    tmp_path, monkeypatch, placement
):
    """Only a full GPU offload may credit the shared heap."""
    backend, gguf = _backend(tmp_path, vulkan = True, memory = [(0, 107 * 1024, 0)])
    _restore_host_guard(backend)
    backend._get_gguf_size_bytes = lambda _path: int(16.5 * 1024**3)
    monkeypatch.setattr(
        LlamaCppBackend, "_available_system_memory_mib", staticmethod(lambda: 13 * 1024)
    )
    monkeypatch.setattr(
        LlamaCppBackend, "_total_system_memory_mib", staticmethod(lambda: 32 * 1024)
    )
    monkeypatch.setattr(LlamaCppBackend, "_cgroup_available_memory_mib", staticmethod(lambda: None))

    _launch_warns(backend, gguf, **placement)


def test_a_device_pin_decides_whether_the_shared_heap_is_reachable(tmp_path, monkeypatch):
    """Only a selected shared device contributes its heap."""

    def _mixed():
        backend, gguf = _backend(
            tmp_path, vulkan = True, memory = [(0, 6 * 1024, 8 * 1024), (1, 94641, 0)]
        )
        _restore_host_guard(backend)
        backend._get_gguf_size_bytes = lambda _path: 30 * 1024**3
        # gpu_layers=33 fully offloads this 32-layer model.
        backend._n_layers = 32
        return backend, gguf

    monkeypatch.setattr(
        LlamaCppBackend, "_available_system_memory_mib", staticmethod(lambda: 13 * 1024)
    )
    monkeypatch.setattr(
        LlamaCppBackend, "_total_system_memory_mib", staticmethod(lambda: 32 * 1024)
    )
    monkeypatch.setattr(LlamaCppBackend, "_cgroup_available_memory_mib", staticmethod(lambda: None))
    manual = {"gpu_memory_mode": "manual", "gpu_layers": 33}

    backend, gguf = _mixed()
    _launch_warns(backend, gguf, extra_args = ["--device", "Vulkan0"], **manual)

    backend, gguf = _mixed()
    assert _launch(backend, gguf, extra_args = ["--device", "Vulkan1"], **manual)["cmd"]


def test_an_unselected_card_does_not_shrink_what_the_shared_heap_must_hold(tmp_path, monkeypatch):
    """Only selected cards reduce the bytes assigned to the shared heap."""
    backend, gguf = _backend(
        tmp_path, vulkan = True, memory = [(0, 24 * 1024, 24 * 1024), (1, 10 * 1024, 0)]
    )
    _restore_host_guard(backend)
    backend._get_gguf_size_bytes = lambda _path: 30 * 1024**3
    backend._n_layers = 32
    monkeypatch.setattr(
        LlamaCppBackend, "_available_system_memory_mib", staticmethod(lambda: 4 * 1024)
    )
    monkeypatch.setattr(
        LlamaCppBackend, "_total_system_memory_mib", staticmethod(lambda: 32 * 1024)
    )
    monkeypatch.setattr(LlamaCppBackend, "_cgroup_available_memory_mib", staticmethod(lambda: None))

    _launch_warns(
        backend,
        gguf,
        gpu_memory_mode = "manual",
        gpu_layers = 33,
        extra_args = ["--device", "Vulkan1"],
    )


@pytest.mark.parametrize(
    "split",
    [{"tensor_split": [1.0, 0.0]}, {"extra_args": ["--tensor-split", "1,0"]}],
    ids = ["picker-share", "user-flag"],
)
def test_an_explicit_tensor_split_leaves_the_shared_heap_uncredited(tmp_path, monkeypatch, split):
    """An ambiguous tensor split must not credit a shared heap."""
    backend, gguf = _backend(tmp_path, vulkan = True, memory = [(0, 6 * 1024, 8 * 1024), (1, 94641, 0)])
    _restore_host_guard(backend)
    backend._get_gguf_size_bytes = lambda _path: 30 * 1024**3
    backend._n_layers = 32
    monkeypatch.setattr(
        LlamaCppBackend, "_available_system_memory_mib", staticmethod(lambda: 4 * 1024)
    )
    monkeypatch.setattr(
        LlamaCppBackend, "_total_system_memory_mib", staticmethod(lambda: 32 * 1024)
    )
    monkeypatch.setattr(LlamaCppBackend, "_cgroup_available_memory_mib", staticmethod(lambda: None))

    _launch_warns(
        backend,
        gguf,
        gpu_memory_mode = "manual",
        gpu_layers = 33,
        gpu_ids = [0, 1],
        **split,
    )


def _mixed_vulkan(tmp_path, monkeypatch, memory):
    """A 30 GiB GGUF on a host with 4 GiB of RAM left, full manual offload."""
    backend, gguf = _backend(tmp_path, vulkan = True, memory = memory)
    _restore_host_guard(backend)
    backend._get_gguf_size_bytes = lambda _path: 30 * 1024**3
    backend._n_layers = 32
    monkeypatch.setattr(
        LlamaCppBackend, "_available_system_memory_mib", staticmethod(lambda: 4 * 1024)
    )
    monkeypatch.setattr(
        LlamaCppBackend, "_total_system_memory_mib", staticmethod(lambda: 32 * 1024)
    )
    monkeypatch.setattr(LlamaCppBackend, "_cgroup_available_memory_mib", staticmethod(lambda: None))
    return backend, gguf


@pytest.mark.parametrize(
    "extras",
    [["--split-mode", "none", "--main-gpu", "0"], ["-sm", "none"]],
    ids = ["with-main-gpu", "bare"],
)
def test_split_mode_none_leaves_a_second_device_heap_uncredited(tmp_path, monkeypatch, extras):
    """Split mode none cannot select a shared heap among multiple devices."""
    backend, gguf = _mixed_vulkan(tmp_path, monkeypatch, [(0, 6 * 1024, 8 * 1024), (1, 94641, 0)])

    # Both devices pinned, so the split mode is the only thing left to decide.
    _launch_warns(
        backend,
        gguf,
        gpu_memory_mode = "manual",
        gpu_layers = 33,
        extra_args = ["--device", "Vulkan0,Vulkan1", *extras],
    )


def test_an_unpinned_launch_beside_a_discrete_card_leaves_the_heap_uncredited(
    tmp_path, monkeypatch
):
    """llama.cpp drops integrated GPUs when its own device list finds a discrete one."""
    backend, gguf = _mixed_vulkan(tmp_path, monkeypatch, [(0, 94641, 0), (1, 6 * 1024, 8 * 1024)])

    _launch_warns(backend, gguf, gpu_memory_mode = "manual", gpu_layers = 33)


def test_a_pin_still_reaches_the_heap_beside_a_discrete_card(tmp_path, monkeypatch):
    """Naming the shared device puts it back in llama.cpp's list."""
    backend, gguf = _mixed_vulkan(tmp_path, monkeypatch, [(0, 94641, 0), (1, 6 * 1024, 8 * 1024)])

    assert _launch(
        backend,
        gguf,
        gpu_memory_mode = "manual",
        gpu_layers = 33,
        extra_args = ["--device", "Vulkan0"],
    )["cmd"]


def test_split_mode_none_still_credits_a_lone_shared_device(tmp_path, monkeypatch):
    """A lone shared device remains reachable under split mode none."""
    backend, gguf = _mixed_vulkan(tmp_path, monkeypatch, [(0, 94641, 0)])

    assert _launch(
        backend,
        gguf,
        gpu_memory_mode = "manual",
        gpu_layers = 33,
        extra_args = ["--split-mode", "none"],
    )["cmd"]


def test_vulkan_igpu_heap_does_not_bypass_a_cgroup_limit(tmp_path, monkeypatch):
    """A shared Vulkan heap remains subject to the process cgroup limit."""
    backend, gguf = _backend(
        tmp_path,
        vulkan = True,
        memory = [(0, 64 * 1024, 0)],
    )
    _restore_host_guard(backend)
    backend._apu_ram_shortfall_message = LlamaCppBackend._apu_ram_shortfall_message
    backend._get_gguf_size_bytes = lambda _path: 20 * 1024**3
    monkeypatch.setattr(
        LlamaCppBackend, "_available_system_memory_mib", staticmethod(lambda: 64 * 1024)
    )
    monkeypatch.setattr(
        LlamaCppBackend, "_cgroup_available_memory_mib", staticmethod(lambda: 8 * 1024)
    )

    # The cgroup ceiling still governs what the message prices, it just no longer
    # stops the load.
    _launch(backend, gguf)
    assert "unified-memory APU" in (backend.last_load_warning or "")


def test_a_card_resident_model_is_not_refused_by_a_container_ceiling(tmp_path, monkeypatch):
    """A card-resident model is independent of the cgroup memory budget."""
    backend, gguf = _offload_backend(
        tmp_path,
        gguf_gb = 23.4,
        free_mib = 24 * 1024,
        avail_mib = 1024,
        monkeypatch = monkeypatch,
    )
    backend._apu_ram_shortfall_message = LlamaCppBackend._apu_ram_shortfall_message
    monkeypatch.setattr(LlamaCppBackend, "_cgroup_available_memory_mib", staticmethod(lambda: 1024))

    assert _launch(backend, gguf)["cmd"]


def test_unknown_available_ram_abstains(tmp_path, monkeypatch):
    backend, gguf = _offload_backend(
        tmp_path, gguf_gb = 13.3, free_mib = 4877, avail_mib = None, monkeypatch = monkeypatch
    )

    assert _launch(backend, gguf)["cmd"]


def test_an_unsized_model_abstains(tmp_path, monkeypatch):
    """A GGUF whose size cannot be read leaves nothing to price."""
    backend, gguf = _offload_backend(
        tmp_path, gguf_gb = 13.3, free_mib = 4877, avail_mib = 10_000, monkeypatch = monkeypatch
    )
    backend._get_gguf_size_bytes = lambda _path: (_ for _ in ()).throw(OSError("stat failed"))

    assert _launch(backend, gguf)["cmd"]


@pytest.mark.parametrize(
    "extra_args",
    [
        ["-ngl", "0"],
        ["--mlock"],
        ["--no-mmap"],
        ["--device", "none"],
        ["--no-kv-offload"],
    ],
    ids = ["zero-layers", "mlock", "no-mmap", "cpu-device", "cpu-kv"],
)
def test_placement_flags_never_turn_an_allowed_load_into_a_refusal(
    tmp_path, monkeypatch, extra_args
):
    """The floor prices weights against the whole free pool and models no placement.
    Each of these moves bytes onto the host or narrows the reachable VRAM, so a guard
    that read them could only refuse MORE. Leaving them out cannot invent a refusal,
    which is the property that keeps this check free of llama.cpp placement modelling."""
    backend, gguf = _offload_backend(
        tmp_path, gguf_gb = 13.3, free_mib = 4877, avail_mib = 64_000, monkeypatch = monkeypatch
    )

    assert _launch(backend, gguf, extra_args = extra_args)["cmd"]


def test_the_guard_reads_the_model_the_child_opens(tmp_path, monkeypatch):
    """Sizing comes from the argv path, not from the planner's earlier pick, so a
    fallback that rewrote -m is priced as launched."""
    seen = []
    backend, gguf = _offload_backend(
        tmp_path, gguf_gb = 13.3, free_mib = 4877, avail_mib = 10_000, monkeypatch = monkeypatch
    )
    real_size = backend._get_gguf_size_bytes

    def _record(path):
        seen.append(str(path))
        return real_size(path)

    backend._get_gguf_size_bytes = _record
    _launch_warns(backend, gguf)

    assert str(gguf) in seen


def test_the_env_escape_is_now_a_no_op_that_only_silences_the_warning(tmp_path, monkeypatch):
    """UNSLOTH_ALLOW_HOST_OFFLOAD was the opt-out from a refusal that no longer exists.

    Both arms must load. Unset, the load carries the advisory; set, it loads just the
    same and says nothing. Kept honoured so existing scripts and docs keep working."""
    warned, gguf = _offload_backend(
        tmp_path, gguf_gb = 13.3, free_mib = 4877, avail_mib = 10_000, monkeypatch = monkeypatch
    )
    monkeypatch.delenv("UNSLOTH_ALLOW_HOST_OFFLOAD", raising = False)
    assert "--fit" in _launch(warned, gguf)["cmd"]
    assert "does not fit in GPU memory" in (warned.last_load_warning or "")

    allowed_dir = tmp_path / "allowed"
    allowed_dir.mkdir()
    allowed, gguf2 = _offload_backend(
        allowed_dir, gguf_gb = 13.3, free_mib = 4877, avail_mib = 10_000, monkeypatch = monkeypatch
    )
    monkeypatch.setenv("UNSLOTH_ALLOW_HOST_OFFLOAD", "1")
    assert "--fit" in _launch(allowed, gguf2)["cmd"]
    assert allowed.last_load_warning is None


def test_a_wildly_oversized_model_still_loads(tmp_path, monkeypatch):
    """The regression this change exists for.

    A 67.6 GB quant on a 14.5 GiB card with 51 GB of RAM (the measured Colab T4
    high-RAM shape) used to come back as HTTP 400 from /api/inference/load, on a model
    llama.cpp itself will run straight off the mmap. No shortfall may block a load
    again, however large the gap: the cost is reported, not enforced."""
    backend, gguf = _offload_backend(
        tmp_path,
        gguf_gb = 67.6,
        free_mib = 14_848,
        avail_mib = 51_000,
        monkeypatch = monkeypatch,
    )
    monkeypatch.delenv("UNSLOTH_ALLOW_HOST_OFFLOAD", raising = False)

    assert _launch(backend, gguf)["cmd"], "the oversized load never spawned llama-server"
    assert "does not fit in GPU memory" in (backend.last_load_warning or "")


# An unmapped load is the one shape the advisory's premise does not cover. The spill
# is only survivable because the weights are mmap'd; upstream sets use_mmap for
# mmap/mmap+mlock/auto alone (llama-model-loader.cpp), so `none` and `mlock` read every
# byte into a buffer llama.cpp allocates and an oversized model fails outright instead
# of paging. The guard still never refuses: it overrides the mode and says so.
_UNMAPPED_ARGV = [
    ["--no-mmap"],
    ["--load-mode", "none"],
    ["--load-mode=none"],
    ["--no-direct-io"],
]
_UNMAPPED_IDS = ["no-mmap", "load-mode-none", "load-mode-none-equals", "no-direct-io"]


@pytest.mark.parametrize("extra_args", _UNMAPPED_ARGV, ids = _UNMAPPED_IDS)
def test_an_oversized_unmapped_load_is_remapped_instead_of_refused(
    tmp_path, monkeypatch, extra_args
):
    """It launches, the argv the child gets is pageable, and the warning names the
    override -- a silent one would leave the user's own setting quietly undone."""
    backend, gguf = _offload_backend(
        tmp_path, gguf_gb = 13.3, free_mib = 4877, avail_mib = 10_000, monkeypatch = monkeypatch
    )
    monkeypatch.delenv("UNSLOTH_ALLOW_HOST_OFFLOAD", raising = False)

    cmd = _launch(backend, gguf, extra_args = extra_args)["cmd"]

    assert cmd, "the unmapped oversized load never spawned llama-server"
    assert not _unmapped_tokens(cmd), f"the child still loads unmapped: {cmd}"
    assert "memory mapping instead" in (backend.last_load_warning or "")


@pytest.mark.parametrize("extra_args", _UNMAPPED_ARGV, ids = _UNMAPPED_IDS)
def test_an_unmapped_load_that_fits_is_left_exactly_as_asked(tmp_path, monkeypatch, extra_args):
    """The control. Same request on a host with room: no shortfall, so nothing is
    overridden and no warning is invented. Loading unmapped is a legitimate choice."""
    backend, gguf = _offload_backend(
        tmp_path, gguf_gb = 13.3, free_mib = 4877, avail_mib = 64_000, monkeypatch = monkeypatch
    )

    cmd = _launch(backend, gguf, extra_args = extra_args)["cmd"]

    assert _unmapped_tokens(cmd) == list(
        extra_args
    ), f"the fitting load lost the mode it asked for: {cmd}"
    assert backend.last_load_warning is None


def test_the_override_keeps_a_lock_rather_than_dropping_it(tmp_path, monkeypatch):
    """`mlock` is unmapped too, but it also says "keep this in RAM". The pageable
    equivalent is `mmap+mlock`, which upstream mmaps, so the request survives the
    override instead of being silently discarded."""
    backend, gguf = _offload_backend(
        tmp_path, gguf_gb = 13.3, free_mib = 4877, avail_mib = 10_000, monkeypatch = monkeypatch
    )
    monkeypatch.delenv("UNSLOTH_ALLOW_HOST_OFFLOAD", raising = False)

    cmd = _launch(backend, gguf, extra_args = ["--load-mode", "mlock"])["cmd"]

    assert cmd, "the unmapped oversized load never spawned llama-server"
    _modes = [cmd[i + 1] for i, tok in enumerate(cmd) if tok == "--load-mode" and i + 1 < len(cmd)]
    assert _modes == ["mmap+mlock"], f"the lock was not carried onto a mapping: {cmd}"
    assert "memory mapping instead" in (backend.last_load_warning or "")


def test_the_override_reaches_the_env_twin_llama_cpp_reads_first(tmp_path, monkeypatch):
    """llama.cpp resolves LLAMA_ARG_* before argv, so an inherited selector survives
    stripping the tokens. Unsloth emits no load-mode flag of its own here, so without
    the env half the child would still load unmapped with nothing in the argv to show
    it."""
    backend, gguf = _offload_backend(
        tmp_path, gguf_gb = 13.3, free_mib = 4877, avail_mib = 10_000, monkeypatch = monkeypatch
    )
    monkeypatch.delenv("UNSLOTH_ALLOW_HOST_OFFLOAD", raising = False)
    monkeypatch.setenv("LLAMA_ARG_NO_MMAP", "1")

    launched = _launch(backend, gguf)

    assert launched["cmd"], "the unmapped oversized load never spawned llama-server"
    assert "LLAMA_ARG_NO_MMAP" not in launched["env"]
    assert "memory mapping instead" in (backend.last_load_warning or "")


def _override_log(monkeypatch):
    """Collect the pageable-override log line. structlog, so caplog cannot see it."""
    import core.inference.llama_cpp as llama_cpp

    lines = []
    monkeypatch.setattr(
        llama_cpp.logger,
        "warning",
        lambda msg, *a, **kw: lines.append(msg % a if a else msg),
    )
    return lines


@pytest.mark.parametrize("extra_args", _UNMAPPED_ARGV, ids = _UNMAPPED_IDS)
def test_the_warning_opt_out_never_disables_the_pageable_override(
    tmp_path, monkeypatch, extra_args
):
    """UNSLOTH_ALLOW_HOST_OFFLOAD silences the warning and nothing else.

    The override is what lets an oversized unmapped load finish at all: "none" and
    "mlock" allocate the whole model in host RAM, so the same shortfall is an OOM kill
    rather than slow paging. Gating it on the message meant setting the deprecated
    escape turned a load that works into one that is killed, which is the opposite of
    what an opt-out from a REFUSAL was ever meant to do. The message goes; the rewrite
    stays, and the log still names it so the load is traceable."""
    backend, gguf = _offload_backend(
        tmp_path, gguf_gb = 13.3, free_mib = 4877, avail_mib = 10_000, monkeypatch = monkeypatch
    )
    monkeypatch.setenv("UNSLOTH_ALLOW_HOST_OFFLOAD", "1")
    logged = _override_log(monkeypatch)

    cmd = _launch(backend, gguf, extra_args = extra_args)["cmd"]

    assert cmd, "the unmapped oversized load never spawned llama-server"
    assert not _unmapped_tokens(cmd), f"the opt-out left the child loading unmapped: {cmd}"
    # The opt-out did its one job, and only that job.
    assert backend.last_load_warning is None
    assert [line for line in logged if "Overriding the unmapped load mode" in line], logged


@pytest.mark.parametrize("extra_args", _UNMAPPED_ARGV, ids = _UNMAPPED_IDS)
def test_the_opt_out_on_a_fitting_unmapped_load_changes_nothing(tmp_path, monkeypatch, extra_args):
    """The control for the case above. Silenced or not, a load with room to run is
    left exactly as asked and nothing is logged about an override."""
    backend, gguf = _offload_backend(
        tmp_path, gguf_gb = 13.3, free_mib = 4877, avail_mib = 64_000, monkeypatch = monkeypatch
    )
    monkeypatch.setenv("UNSLOTH_ALLOW_HOST_OFFLOAD", "1")
    logged = _override_log(monkeypatch)

    cmd = _launch(backend, gguf, extra_args = extra_args)["cmd"]

    assert _unmapped_tokens(cmd) == list(extra_args), f"the fitting load lost its mode: {cmd}"
    assert backend.last_load_warning is None
    assert not [line for line in logged if "Overriding the unmapped load mode" in line], logged


def _apu_backend(tmp_path, *, gguf_gb, avail_mib, monkeypatch):
    """A ROCm unified-memory APU: the weights load into system RAM, and the APU's
    reported GPU pool IS that RAM.

    The discrete host guard is left stubbed off, as it effectively is on this host:
    ``_shared_gpu_ids`` is populated for Vulkan alone (the Vulkan probe is what reports
    an iGPU), so on ROCm the APU's pool is credited against the weights as if it were
    dedicated VRAM, the spill prices out at zero and the guard abstains. The APU
    preflight is the only reading that sees the shortfall."""
    backend, gguf = _backend(tmp_path, vulkan = False, memory = [(0, 60_000, 60_000)])
    backend._get_gguf_size_bytes = lambda _path: int(gguf_gb * 1024**3)
    backend._amd_apu_wants_unified_memory = lambda *_a, **_kw: True
    backend._apu_ram_shortfall_message = LlamaCppBackend._apu_ram_shortfall_message
    # nothing pinned, so the preflight re-asks the gate; no marker here, so it abstains
    backend._arch_gate_survivors = lambda _binary = None: []
    backend._select_gpus = lambda *args, **kw: (None, True)
    monkeypatch.setattr(
        LlamaCppBackend, "_available_system_memory_mib", staticmethod(lambda: avail_mib)
    )
    monkeypatch.delenv("UNSLOTH_ALLOW_HOST_OFFLOAD", raising = False)
    return backend, gguf


@pytest.mark.parametrize("extra_args", _UNMAPPED_ARGV, ids = _UNMAPPED_IDS)
def test_an_oversized_unmapped_apu_load_is_paged_before_it_launches(
    tmp_path, monkeypatch, extra_args
):
    """The APU preflight only RECORDED its shortfall, and the discrete guard next door
    cannot re-find it (the APU's reported pool covers the weights), so an unmapped
    oversized APU load reached the child unchanged and was OOM-killed. It is the
    condition that drives the override, not whichever guard happened to word it."""
    backend, gguf = _apu_backend(
        tmp_path, gguf_gb = 64.6, avail_mib = 46 * 1024, monkeypatch = monkeypatch
    )

    cmd = _launch(backend, gguf, extra_args = extra_args)["cmd"]

    assert cmd, "the unmapped oversized APU load never spawned llama-server"
    assert not _unmapped_tokens(cmd), f"the APU child still loads unmapped: {cmd}"
    assert "unified-memory APU" in (backend.last_load_warning or "")
    assert "memory mapping instead" in (backend.last_load_warning or "")


@pytest.mark.parametrize("extra_args", _UNMAPPED_ARGV, ids = _UNMAPPED_IDS)
def test_an_unmapped_apu_load_that_fits_is_left_exactly_as_asked(tmp_path, monkeypatch, extra_args):
    """The control. Same APU, same request, room to run: no shortfall, so the mode the
    user chose survives untouched and no warning is invented."""
    backend, gguf = _apu_backend(
        tmp_path, gguf_gb = 64.6, avail_mib = 92 * 1024, monkeypatch = monkeypatch
    )

    cmd = _launch(backend, gguf, extra_args = extra_args)["cmd"]

    assert _unmapped_tokens(cmd) == list(extra_args), f"the fitting APU load lost it: {cmd}"
    assert backend.last_load_warning is None


def _unmapped_tokens(cmd):
    """The tokens in ``cmd`` that select a mode llama.cpp does not mmap."""
    out = []
    for i, token in enumerate(cmd):
        if token in ("--no-mmap", "-no-mmap", "--no-direct-io", "-ndio"):
            out.append(token)
        elif token in ("--load-mode", "-lm") and i + 1 < len(cmd):
            if cmd[i + 1].strip().lower() in ("none", "mlock"):
                out.extend([token, cmd[i + 1]])
        elif token.split("=", 1)[0] in ("--load-mode", "-lm") and "=" in token:
            if token.split("=", 1)[1].strip().lower() in ("none", "mlock"):
                out.append(token)
    return out


def _load_intent(gguf, **kwargs):
    return GgufLoadIntent(gguf_path = str(gguf), model_identifier = "test", **kwargs)


def _host_totals(
    monkeypatch,
    backend,
    *,
    vram_total_mib,
    ram_total_mib,
    vram_free_mib = None,
):
    """Pin what the preflight reads: the physical ceilings, and a free VRAM figure low
    enough to stand for a card the resident model has not given back yet."""
    free = vram_total_mib if vram_free_mib is None else vram_free_mib
    backend._get_gpu_memory = lambda _binary = None, **_kw: [(0, free, vram_total_mib)]
    monkeypatch.setattr(
        LlamaCppBackend, "_total_system_memory_mib", staticmethod(lambda: ram_total_mib)
    )


def test_the_route_precheck_refuses_before_the_gpu_handoff(tmp_path, monkeypatch):
    """`acquire_for(CHAT)` evicts a resident Images/Video pipeline and the reload
    confirmation cancels the running generations, both before the launch guard can read the
    finished argv. The route asks first, so a pick no reclaim can rescue, 100 GB against a
    24 GB card and 10 GB of RAM, tears nothing down."""
    backend, gguf = _offload_backend(
        tmp_path, gguf_gb = 100, free_mib = 20_000, avail_mib = 10_000, monkeypatch = monkeypatch
    )
    _host_totals(monkeypatch, backend, vram_total_mib = 24_000, ram_total_mib = 32_000)

    verdict = backend.host_offload_warning_for_intent(_load_intent(gguf))
    assert verdict is not None and "does not fit in GPU memory" in verdict


def test_the_route_precheck_credits_capacity_the_handoff_is_about_to_reclaim(tmp_path, monkeypatch):
    """The resident llama-server, Unsloth model and media pipeline hold VRAM, and through a
    host KV cache, CPU-offloaded weights and locked mappings they hold RAM too. The route and
    load_model reclaim all of it after this runs, so pricing against either free reading
    refused a switch the reclaimed machine handles outright and made switching on a busy
    machine impossible. Both physical totals are what bound the launch.

    30 GB against a 24 GB card leaves about 6.7 GB on the host, which 3 GB of MemAvailable
    cannot hold and the machine's own 64 GB holds easily."""
    backend, gguf = _offload_backend(
        tmp_path, gguf_gb = 30, free_mib = 900, avail_mib = 3_000, monkeypatch = monkeypatch
    )
    # 900 MiB free VRAM and 3 GB MemAvailable: the model being replaced still holds both
    _host_totals(
        monkeypatch, backend, vram_total_mib = 24_000, ram_total_mib = 64_000, vram_free_mib = 900
    )

    assert backend.host_offload_warning_for_intent(_load_intent(gguf)) is None


def test_the_route_precheck_only_refuses_what_the_launch_would(tmp_path, monkeypatch):
    """Abstains on an undownloaded repo, a device whose total the probe cannot read, an
    unreadable pool, unreadable total RAM and the escape. So it can never reject a load the
    launch would allow."""
    backend, gguf = _offload_backend(
        tmp_path, gguf_gb = 100, free_mib = 20_000, avail_mib = 10_000, monkeypatch = monkeypatch
    )
    _host_totals(monkeypatch, backend, vram_total_mib = 24_000, ram_total_mib = 32_000)

    assert backend.host_offload_warning_for_intent(_load_intent(gguf, hf_repo = "org/repo")) is None
    # an igpu or a MIG/vGPU line reports total 0, so the ceiling is unknown
    backend._get_gpu_memory = lambda _binary = None, **_kw: [(0, 20_000, 0)]
    assert backend.host_offload_warning_for_intent(_load_intent(gguf)) is None
    backend._get_gpu_memory = lambda _binary = None, **_kw: []
    assert backend.host_offload_warning_for_intent(_load_intent(gguf)) is None
    _host_totals(monkeypatch, backend, vram_total_mib = 24_000, ram_total_mib = None)
    assert backend.host_offload_warning_for_intent(_load_intent(gguf)) is None
    _host_totals(monkeypatch, backend, vram_total_mib = 24_000, ram_total_mib = 32_000)
    monkeypatch.setenv("UNSLOTH_ALLOW_HOST_OFFLOAD", "1")
    assert backend.host_offload_warning_for_intent(_load_intent(gguf)) is None


def test_an_arch_gated_cpu_launch_prices_the_whole_model(tmp_path, monkeypatch):
    """The arch gate empties the pool AND masks every card, so the child is knowingly
    on the CPU rather than unprobed. Abstaining there ran an oversized GGUF wholly from
    RAM with no preflight, which is the OOM this guard exists to stop."""
    backend, gguf = _backend(tmp_path, vulkan = False, memory = [])
    _restore_host_guard(backend)
    backend._get_gguf_size_bytes = lambda _path: int(13.3 * 1024**3)
    backend._select_gpus = lambda *args, **kw: (None, True)
    monkeypatch.setattr(
        LlamaCppBackend, "_available_system_memory_mib", staticmethod(lambda: 10_000)
    )

    assert (
        backend._launch_host_shortfall_message(
            ["llama-server", "-m", str(gguf)], [], child_has_no_gpu = True
        )
        is not None
    )


def test_a_masked_off_child_takes_no_vram_credit(tmp_path, monkeypatch):
    """Manual zero-offload masks the child off cards the planner still probed. Crediting
    that VRAM would offset a spill the child cannot place there."""
    backend, gguf = _backend(tmp_path, vulkan = False, memory = [(0, 20_000, 24_000)])
    _restore_host_guard(backend)
    backend._get_gguf_size_bytes = lambda _path: int(13.3 * 1024**3)
    monkeypatch.setattr(
        LlamaCppBackend, "_available_system_memory_mib", staticmethod(lambda: 10_000)
    )
    argv = ["llama-server", "-m", str(gguf)]

    assert backend._launch_host_shortfall_message(argv, [(0, 20_000)]) is None
    assert (
        backend._launch_host_shortfall_message(argv, [(0, 20_000)], child_has_no_gpu = True)
        is not None
    )


def test_an_unprobed_pool_still_abstains_when_nothing_was_masked(tmp_path, monkeypatch):
    """The abstention survives: only the launch saying it masked the child off every
    card prices the full model, not a pool that merely came back empty."""
    backend, gguf = _backend(tmp_path, vulkan = False, memory = [])
    _restore_host_guard(backend)
    backend._get_gguf_size_bytes = lambda _path: int(13.3 * 1024**3)
    monkeypatch.setattr(
        LlamaCppBackend, "_available_system_memory_mib", staticmethod(lambda: 10_000)
    )

    assert backend._launch_host_shortfall_message(["llama-server", "-m", str(gguf)], []) is None


def test_a_gpu_less_host_running_a_cpu_only_build_still_abstains(tmp_path, monkeypatch):
    """Unsloth installs a CPU-only prebuilt on a host with no GPU, so that host probes an
    empty pool AND reports a build with no GPU backend. Letting the build state alone
    charge the whole model refused a 7.5 GB GGUF with 9 GB of RAM, which loads on main,
    and blamed GPU memory on a machine that has no GPU."""
    backend, gguf = _backend(tmp_path, vulkan = False, memory = [])
    _restore_host_guard(backend)
    backend._get_gguf_size_bytes = lambda _path: int(7.5 * 1024**3)
    backend._select_gpus = lambda *args, **kw: (None, True)
    backend._binary_ships_no_gpu_backend = lambda _binary = None, _env = None: True
    monkeypatch.setattr(
        LlamaCppBackend, "_available_system_memory_mib", staticmethod(lambda: 9_216)
    )

    assert _launch(backend, gguf)["cmd"]


def test_a_gpu_less_host_still_abstains_on_a_zero_offload_request(tmp_path, monkeypatch):
    """gpu_layers=0 is a request, not a probe result, so it says nothing about whether a
    card exists. Charging the whole model on an empty pool repeats the CPU-only-build
    refusal on the same GPU-less host."""
    backend, gguf = _backend(tmp_path, vulkan = False, memory = [])
    _restore_host_guard(backend)
    backend._get_gguf_size_bytes = lambda _path: int(7.5 * 1024**3)
    backend._select_gpus = lambda *args, **kw: (None, True)
    monkeypatch.setattr(
        LlamaCppBackend, "_available_system_memory_mib", staticmethod(lambda: 9_216)
    )

    assert _launch(backend, gguf, gpu_memory_mode = "manual", gpu_layers = 0)["cmd"]


def test_a_cpu_only_build_takes_no_vram_credit(tmp_path, monkeypatch):
    """A split-library build shipping no cuda/hip/vulkan backend cannot offload, so the
    cards the hardware probe still enumerates are unreachable. Crediting their VRAM
    priced a spill the child never takes: it places the whole model in RAM."""
    backend, gguf = _backend(tmp_path, vulkan = False, memory = [(0, 16_384, 24_000)])
    _restore_host_guard(backend)
    backend._get_gguf_size_bytes = lambda _path: 20 * 1024**3
    monkeypatch.setattr(
        LlamaCppBackend, "_available_system_memory_mib", staticmethod(lambda: 8_192)
    )
    argv = ["llama-server", "-m", str(gguf)]

    # 20 GiB - 16 GiB free VRAM reads as a 4 GiB spill an 8 GiB host can hold.
    assert backend._launch_host_shortfall_message(argv, [(0, 16_384)]) is None
    assert (
        backend._launch_host_shortfall_message(argv, [(0, 16_384)], child_has_no_gpu = True)
        is not None
    )


def test_an_unknown_backend_layout_keeps_its_vram_credit(tmp_path):
    """Fails open on a static or unrecognised layout, so a custom GPU build is never
    mistaken for a CPU-only one and refused."""
    assert LlamaCppBackend._binary_ships_no_gpu_backend("/nonexistent/llama-server") is False


def test_the_launch_reports_a_cpu_only_build_to_the_guard(tmp_path, monkeypatch):
    """End to end: the call site must pass the CPU-only-build state, not just accept it.
    A 20 GiB model over 16 GiB of free VRAM reads as a 4 GiB spill an 8 GiB host holds,
    so only the build state separates the launch from the refusal."""
    gpu_build, gguf = _offload_backend(
        tmp_path, gguf_gb = 20, free_mib = 16_384, avail_mib = 8_192, monkeypatch = monkeypatch
    )
    gpu_build._binary_ships_no_gpu_backend = lambda _binary = None, _env = None: False
    assert _launch(gpu_build, gguf)["cmd"]

    cpu_dir = tmp_path / "cpu"
    cpu_dir.mkdir()
    cpu_build, gguf2 = _offload_backend(
        cpu_dir,
        gguf_gb = 20,
        free_mib = 16_384,
        avail_mib = 8_192,
        monkeypatch = monkeypatch,
    )
    cpu_build._binary_ships_no_gpu_backend = lambda _binary = None, _env = None: True
    _launch_warns(cpu_build, gguf2)


def test_an_empty_gpu_pool_abstains(tmp_path, monkeypatch):
    """_get_gpu_memory swallows a failed probe as [], so an empty pool cannot be told
    from a host with no GPU. Pricing the full model there would refuse a load that
    llama-server's own enumeration can still place on a card."""
    backend, gguf = _backend(tmp_path, vulkan = False, memory = [])
    _restore_host_guard(backend)
    backend._get_gguf_size_bytes = lambda _path: int(13.3 * 1024**3)
    monkeypatch.setattr(
        LlamaCppBackend, "_available_system_memory_mib", staticmethod(lambda: 10_000)
    )

    assert _launch(backend, gguf)["cmd"]


@pytest.mark.parametrize("accelerator", ["sycl", "opencl", "musa", "cann"])
def test_a_non_cuda_accelerator_build_keeps_its_vram_credit(tmp_path, accelerator):
    """_installed_ggml_backends reads only cuda, hip and vulkan, so a split-library build
    shipping any other supported ggml accelerator looked CPU-only. Pricing its weights
    against RAM refused loads the accelerator can hold."""
    binary = tmp_path / "llama-server"
    binary.write_bytes(b"x")
    lib_dir = tmp_path / "lib"
    lib_dir.mkdir()
    prefix = "" if sys.platform == "win32" else "lib"
    extension = "dll" if sys.platform == "win32" else "so"
    (lib_dir / f"{prefix}ggml-cpu.{extension}").write_bytes(b"x")
    (lib_dir / f"{prefix}ggml-{accelerator}.{extension}").write_bytes(b"x")

    with patch("core.inference.llama_cpp._llama_lib_dir", return_value = lib_dir):
        assert LlamaCppBackend._binary_ships_no_gpu_backend(str(binary)) is False
        # the narrower pre-existing helper is what misreads this layout
        assert LlamaCppBackend._backend_lacks_gpu_lib(str(binary)) is True


def test_a_genuinely_cpu_only_layout_is_still_recognised(tmp_path):
    binary = tmp_path / "llama-server"
    binary.write_bytes(b"x")
    lib_dir = tmp_path / "lib"
    lib_dir.mkdir()
    prefix = "" if sys.platform == "win32" else "lib"
    extension = "dll" if sys.platform == "win32" else "so"
    (lib_dir / f"{prefix}ggml-cpu.{extension}").write_bytes(b"x")
    (lib_dir / f"{prefix}ggml-base.{extension}").write_bytes(b"x")

    with patch("core.inference.llama_cpp._llama_lib_dir", return_value = lib_dir):
        assert LlamaCppBackend._binary_ships_no_gpu_backend(str(binary)) is True


def test_an_rpc_launch_abstains(tmp_path, monkeypatch):
    """--rpc places layers on remote devices this cannot size, so refusing on local
    capacity alone would block a viable distributed launch."""
    backend, gguf = _offload_backend(
        tmp_path, gguf_gb = 13.3, free_mib = 4877, avail_mib = 10_000, monkeypatch = monkeypatch
    )
    argv = ["llama-server", "-m", str(gguf)]

    assert backend._launch_host_shortfall_message(argv, [(0, 4877)]) is not None
    assert (
        backend._launch_host_shortfall_message([*argv, "--rpc", "10.0.0.2:50052"], [(0, 4877)])
        is None
    )
    assert backend._launch_host_shortfall_message([*argv, "--rpc", "  "], [(0, 4877)]) is not None


def test_an_rpc_env_launch_abstains(tmp_path, monkeypatch):
    """llama.cpp reads LLAMA_ARG_RPC as the environment twin of --rpc, so the guard has
    to see the child environment or it refuses the same distributed launch."""
    backend, gguf = _offload_backend(
        tmp_path, gguf_gb = 13.3, free_mib = 4877, avail_mib = 10_000, monkeypatch = monkeypatch
    )
    argv = ["llama-server", "-m", str(gguf)]

    assert backend._launch_host_shortfall_message(argv, [(0, 4877)], {}) is not None
    assert (
        backend._launch_host_shortfall_message(
            argv, [(0, 4877)], {"LLAMA_ARG_RPC": "10.0.0.2:50052"}
        )
        is None
    )


def test_an_external_backend_path_keeps_its_vram_credit(tmp_path):
    """GGML_BACKEND_PATH points the child at plugins outside the lib directory, so a
    cpu-only layout beside the binary is no longer proof the child cannot offload."""
    binary = tmp_path / "llama-server"
    binary.write_bytes(b"x")
    lib_dir = tmp_path / "lib"
    lib_dir.mkdir()
    prefix = "" if sys.platform == "win32" else "lib"
    extension = "dll" if sys.platform == "win32" else "so"
    (lib_dir / f"{prefix}ggml-cpu.{extension}").write_bytes(b"x")

    with patch("core.inference.llama_cpp._llama_lib_dir", return_value = lib_dir):
        assert LlamaCppBackend._binary_ships_no_gpu_backend(str(binary), {}) is True
        assert (
            LlamaCppBackend._binary_ships_no_gpu_backend(
                str(binary), {"GGML_BACKEND_PATH": "/opt/ggml-cuda"}
            )
            is False
        )


def test_a_paravirtual_metal_launch_prices_the_whole_model(tmp_path, monkeypatch):
    """A virtualised Apple GPU rewrites the command to --gpu-layers 0 --device none, and
    Metal hosts leave the pool empty, so the abstention swallowed a placement the launch
    already knew was CPU-only."""
    backend, gguf = _backend(tmp_path, vulkan = False, memory = [])
    _restore_host_guard(backend)
    backend._get_gguf_size_bytes = lambda _path: int(13.3 * 1024**3)
    monkeypatch.setattr(
        LlamaCppBackend, "_available_system_memory_mib", staticmethod(lambda: 10_000)
    )
    argv = ["llama-server", "-m", str(gguf), "--gpu-layers", "0", "--device", "none"]

    assert backend._launch_host_shortfall_message(argv, [], {}) is None
    assert backend._launch_host_shortfall_message(argv, [], {}, child_has_no_gpu = True) is not None


def test_the_launched_load_mode_is_recorded_in_the_memory_state(tmp_path, monkeypatch):
    """A --load-mode the launch emitted has to reach _memory_state.

    "none" reads the weights into an anonymous host buffer (llama-model-loader
    sets use_mmap only for mmap / mmap+mlock / auto), so it IS a reservation. Left
    out of the record, a later "Don't reserve system RAM" is judged satisfied by
    the running child and Apply keeps the reservation instead of relaunching.
    """
    import utils.model_memory_settings as mm
    from core.inference.llama_server_args import memory_state_satisfies_settings

    monkeypatch.setattr(mm, "get_model_memory_settings", lambda: (False, False))
    monkeypatch.setattr(mm, "get_keep_resident", lambda: False)
    monkeypatch.setattr(mm, "get_no_ram_reserve", lambda: False)
    monkeypatch.setattr(mm, "should_mlock", lambda: False)

    caps = dict(LlamaCppBackend.probe_server_capabilities.__func__(LlamaCppBackend, None))
    caps["supports_load_mode"] = True
    backend, gguf = _backend(tmp_path, vulkan = False, memory = [(0, 40000, 48000)])
    with patch.object(LlamaCppBackend, "probe_server_capabilities", lambda *a, **k: caps):
        captured = _launch(backend, gguf, load_mode = "none")

    assert captured["cmd"][captured["cmd"].index("--load-mode") + 1] == "none"
    # (mlock, reserves_ram)
    assert backend._memory_state == (False, True)

    # Both consumers now see the child contradicting the setting.
    monkeypatch.setattr(mm, "get_no_ram_reserve", lambda: True)
    assert (
        memory_state_satisfies_settings(
            backend._memory_state,
            backend._memory_policy_active,
            backend._memory_mlock_applicable,
        )
        is False
    )


def test_a_fit_derived_load_mode_is_recorded_too(tmp_path, monkeypatch):
    """Same record, for the mode the fit supplies rather than the user.

    The fit's "none" reserves exactly as much host RAM as a hand-picked one, so a
    launch that took it must not report itself as non-reserving to the settings
    route and the duplicate-load comparator.
    """
    import utils.model_memory_settings as mm
    from core.inference.llama_server_args import memory_state_satisfies_settings

    monkeypatch.setattr(mm, "get_model_memory_settings", lambda: (False, False))
    monkeypatch.setattr(mm, "get_keep_resident", lambda: False)
    monkeypatch.setattr(mm, "get_no_ram_reserve", lambda: False)
    monkeypatch.setattr(mm, "should_mlock", lambda: False)

    caps = dict(LlamaCppBackend.probe_server_capabilities.__func__(LlamaCppBackend, None))
    caps["supports_load_mode"] = True
    backend, gguf = _backend(tmp_path, vulkan = False, memory = [(0, 40000, 48000)])
    with (
        patch.object(LlamaCppBackend, "probe_server_capabilities", lambda *a, **k: caps),
        patch.object(LlamaCppBackend, "_fit_derived_load_mode", return_value = "none"),
    ):
        captured = _launch(backend, gguf)  # no per-model pick: the fit supplies it

    assert captured["cmd"][captured["cmd"].index("--load-mode") + 1] == "none"
    assert backend._fit_load_mode_flags == ["--load-mode", "none"]
    assert backend._memory_state == (False, True)

    monkeypatch.setattr(mm, "get_no_ram_reserve", lambda: True)
    assert (
        memory_state_satisfies_settings(
            backend._memory_state,
            backend._memory_policy_active,
            backend._memory_mlock_applicable,
        )
        is False
    )


# ── Tensor parallelism keeps the requested KV cache type ─────────────


def _tensor_backend(tmp_path):
    backend, gguf = _backend(
        tmp_path,
        vulkan = False,
        memory = [(0, 24_000, 24_000), (1, 24_000, 24_000)],
    )
    backend._tensor_split_aborts = lambda *args, **kwargs: False
    return backend, gguf


@pytest.mark.parametrize("kv_type", ["q8_0", "q4_0"])
def test_tensor_mode_emits_the_requested_quantized_kv(tmp_path, kv_type):
    """llama.cpp runs a quantized KV cache under --split-mode tensor (ggml-org/
    llama.cpp#23792), so the requested type reaches the child verbatim. Two types,
    so a q8_0-only carve-out cannot pass."""
    backend, gguf = _tensor_backend(tmp_path)

    cmd = _launch(backend, gguf, tensor_parallel = True, cache_type_kv = kv_type)["cmd"]

    assert cmd[cmd.index("--split-mode") + 1] == "tensor"
    assert cmd[cmd.index("--cache-type-k") + 1] == kv_type
    assert cmd[cmd.index("--cache-type-v") + 1] == kv_type
    # The recorded type /status reports and the reload matcher compares against.
    assert backend.cache_type_kv == kv_type


def test_an_unknown_kv_type_is_still_refused_in_tensor_mode(tmp_path):
    """_valid_cache_types drops a type llama.cpp's kv_cache_type_from_str does not
    know, emitting no flag rather than aborting the child. Tensor mode does not
    widen it."""
    backend, gguf = _tensor_backend(tmp_path)

    cmd = _launch(backend, gguf, tensor_parallel = True, cache_type_kv = "q3_K")["cmd"]

    assert cmd[cmd.index("--split-mode") + 1] == "tensor"
    assert "--cache-type-k" not in cmd
    assert "--cache-type-v" not in cmd
    assert backend.cache_type_kv is None


def test_tensor_mode_keeps_an_inherited_quantized_kv_env(tmp_path, monkeypatch):
    """The tensor-branch env scrub owns the split, not the cache type: an
    LLAMA_ARG_CACHE_TYPE_K/_V reaches the child untouched, while the tensor split
    Unsloth emits itself is still cleared. The inherited type also reaches tensor
    placement accounting -- priced as banded/f16 instead, an Inkling child's dense
    fallback OOMs an auto context the plan advertised as fitting."""
    monkeypatch.setenv("LLAMA_ARG_CACHE_TYPE_K", "q8_0")
    monkeypatch.setenv("LLAMA_ARG_CACHE_TYPE_V", "q8_0")
    monkeypatch.setenv("LLAMA_ARG_TENSOR_SPLIT", "9,1")
    backend, gguf = _tensor_backend(tmp_path)
    planned = {}
    real_plan = backend._plan_tensor_parallel

    with patch.object(
        backend,
        "_plan_tensor_parallel",
        side_effect = lambda *a, **kw: planned.update(kw) or real_plan(*a, **kw),
    ):
        captured = _launch(backend, gguf, tensor_parallel = True)
    env, cmd = captured["env"], captured["cmd"]

    assert env["LLAMA_ARG_CACHE_TYPE_K"] == "q8_0"
    assert env["LLAMA_ARG_CACHE_TYPE_V"] == "q8_0"
    assert "LLAMA_ARG_TENSOR_SPLIT" not in env
    assert planned["cache_type_kv"] == "q8_0"
    # Budget-only adoption: the env stays the source of truth for the child.
    assert "--cache-type-k" not in cmd
    assert "--cache-type-v" not in cmd


# ── UNSLOTH_ALLOW_HOST_OFFLOAD is warning-scoped on the APU path too ────────
# The variable's whole remaining contract is that it silences a message. The APU
# preflight priced RAM with a helper that never reads it, so setting the deprecated
# escape silenced the discrete guard and left the APU advisory in memory_warning.


@pytest.mark.parametrize("extra_args", _UNMAPPED_ARGV, ids = _UNMAPPED_IDS)
def test_the_opt_out_silences_the_apu_advisory_and_keeps_the_override(
    tmp_path, monkeypatch, extra_args
):
    """Both halves at once, because they pull in opposite directions: the message goes,
    and the pageable rewrite that makes the load survivable stays. The verdict is read
    before the opt-out, so only the recording is suppressed."""
    backend, gguf = _apu_backend(
        tmp_path, gguf_gb = 64.6, avail_mib = 46 * 1024, monkeypatch = monkeypatch
    )
    monkeypatch.setenv("UNSLOTH_ALLOW_HOST_OFFLOAD", "1")
    logged = _override_log(monkeypatch)

    cmd = _launch(backend, gguf, extra_args = extra_args)["cmd"]

    assert cmd, "the unmapped oversized APU load never spawned llama-server"
    assert not _unmapped_tokens(cmd), f"the opt-out left the APU child unmapped: {cmd}"
    assert backend.last_load_warning is None, (
        "UNSLOTH_ALLOW_HOST_OFFLOAD is documented as silencing the warning, but the "
        f"APU advisory came back: {backend.last_load_warning}"
    )
    # Silenced, not invisible: the log is what keeps an overridden load traceable.
    assert [line for line in logged if "Overriding the unmapped load mode" in line], logged


def test_the_opt_out_leaves_a_fitting_apu_load_alone(tmp_path, monkeypatch):
    """The control. Nothing to say and nothing to override, so the mode the user asked
    for survives and no warning is invented either way."""
    backend, gguf = _apu_backend(
        tmp_path, gguf_gb = 64.6, avail_mib = 92 * 1024, monkeypatch = monkeypatch
    )
    monkeypatch.setenv("UNSLOTH_ALLOW_HOST_OFFLOAD", "1")

    cmd = _launch(backend, gguf, extra_args = ["--no-mmap"])["cmd"]

    assert _unmapped_tokens(cmd) == ["--no-mmap"], cmd
    assert backend.last_load_warning is None


def _apu_and_discrete_shortfall_backend(tmp_path, monkeypatch, *, avail_mib):
    """A unified-memory APU whose reported VRAM pool does NOT cover the weights, so the
    APU preflight and the discrete host guard both find a shortfall on the same load.

    The overlap is the point: two guards, two messages, and _record_load_warning keeps
    the first."""
    backend, gguf = _offload_backend(
        tmp_path,
        gguf_gb = 13.3,
        free_mib = 4877,
        avail_mib = avail_mib,
        monkeypatch = monkeypatch,
        _amd_apu_wants_unified_memory = lambda *_a, **_kw: True,
        _apu_ram_shortfall_message = LlamaCppBackend._apu_ram_shortfall_message,
        # nothing pinned, so the preflight re-asks the gate; no marker, so it abstains
        _arch_gate_survivors = lambda _binary = None: [],
    )
    monkeypatch.delenv("UNSLOTH_ALLOW_HOST_OFFLOAD", raising = False)
    return backend, gguf


def _host_guard_spy(backend):
    """Record what the discrete host guard returned, so a test can pin that it really
    did fire rather than assuming the overlap it is about."""
    seen = []
    real = backend._launch_host_shortfall_message

    def _spy(*args, **kwargs):
        message = real(*args, **kwargs)
        seen.append(message)
        return message

    backend._launch_host_shortfall_message = _spy
    return seen


@pytest.mark.parametrize("extra_args", _UNMAPPED_ARGV, ids = _UNMAPPED_IDS)
def test_the_override_note_reaches_the_warning_the_route_returns(tmp_path, monkeypatch, extra_args):
    """When BOTH guards warn, the APU preflight recorded first and first notice wins,
    so appending the override note to the launch guard's message alone wrote it onto a
    string _record_load_warning then discards. The user was told nothing about Unsloth
    undoing the non-mmap mode they chose, on a load whose argv really did change."""
    backend, gguf = _apu_and_discrete_shortfall_backend(tmp_path, monkeypatch, avail_mib = 10_000)
    seen = _host_guard_spy(backend)

    cmd = _launch(backend, gguf, extra_args = extra_args)["cmd"]

    assert cmd, "the unmapped oversized load never spawned llama-server"
    assert not _unmapped_tokens(cmd), f"the child still loads unmapped: {cmd}"
    # The precondition this cell exists for: two shortfalls on one load.
    assert any(msg and "does not fit in GPU memory" in msg for msg in seen), seen
    warning = backend.last_load_warning or ""
    assert "unified-memory APU" in warning, warning
    assert (
        "memory mapping instead" in warning
    ), f"the override never reached the warning the route returns: {warning}"


def test_the_note_is_appended_once_when_only_the_launch_guard_warned(tmp_path, monkeypatch):
    """The control against a double append. With no APU notice recorded there is
    nothing to amend, so the note arrives exactly once, through the launch guard's own
    message."""
    backend, gguf = _offload_backend(
        tmp_path, gguf_gb = 13.3, free_mib = 4877, avail_mib = 10_000, monkeypatch = monkeypatch
    )
    monkeypatch.delenv("UNSLOTH_ALLOW_HOST_OFFLOAD", raising = False)

    _launch(backend, gguf, extra_args = ["--no-mmap"])

    warning = backend.last_load_warning or ""
    assert "does not fit in GPU memory" in warning
    assert warning.count("memory mapping instead") == 1, warning


# ── Repricing after the text-only fallback drops a CPU-pinned projector ────────
# The APU preflight charges model_size + the projector the launch pinned to CPU,
# because both land in the same system RAM. When llama-server then fails on the
# projector and the retry strips --mmproj, the child that serves the session never
# reads those bytes: the advisory the route hands back described a load that is not
# running, and could have been the only reason an unmapped launch was remapped.
_PROJECTOR_ABORT_OUT = (
    "srv    load_model: loading model 'model.gguf'\nclip.cpp:4391: Unknown projector type\n"
)


def _apu_pinned_projector_backend(tmp_path, monkeypatch, *, gguf_gb, mmproj_gb, avail_mib):
    """An APU whose weights fit in RAM on their own and only overflow it once the
    CPU-pinned vision projector is charged alongside them."""
    backend, gguf = _apu_backend(
        tmp_path, gguf_gb = gguf_gb, avail_mib = avail_mib, monkeypatch = monkeypatch
    )
    mmproj = _write_gguf(tmp_path / "mmproj.gguf", architecture = "clip")
    backend._resolve_launch_mmproj_path = lambda **_kw: str(mmproj)
    backend._mmproj_vram_bytes = lambda _path: int(mmproj_gb * 1024**3)
    return backend, gguf


def _launch_with_text_only_fallback(backend, gguf, **load_kwargs):
    """Every spawn that still carries --mmproj aborts on the projector; the text-only
    retry comes up healthy. Mirrors the real recovery: the session ends up serving a
    child that loaded the weights and nothing else."""
    captured = {"cmds": []}

    def fake_popen(cmd, **kwargs):
        if not cmd or str(cmd[0]) != "/fake/llama-server":
            return _REAL_POPEN(cmd, **kwargs)
        captured["cmds"].append(list(cmd))
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

    def fake_health(timeout = None):
        launched = captured["cmds"][-1] if captured["cmds"] else []
        if "--mmproj" in launched:
            backend._stdout_lines = _PROJECTOR_ABORT_OUT.splitlines()
            return False
        backend._stdout_lines = []
        return True

    backend._wait_for_health = fake_health

    with patch.object(subprocess, "Popen", side_effect = fake_popen):
        assert backend.load_model(
            GgufLoadIntent(
                gguf_path = str(gguf),
                model_identifier = "test",
                is_vision = True,
                **load_kwargs,
            )
        )
    return captured


@pytest.mark.parametrize("unmapped", [False, True], ids = ["pageable", "unmapped"])
def test_the_text_only_fallback_reprices_the_projector_it_dropped(tmp_path, monkeypatch, unmapped):
    """40 GB of weights fit the 44 GB this APU can spare; the 8 GB projector pinned
    beside them does not, so the preflight warns (and, unmapped, remaps the load).
    The projector then fails to start and the session serves text-only, holding only
    the weights -- which fit. The advisory has to describe THAT child.

    The override is not taken back. Its argv and its env reached the child long before
    this point, so the running server is memory-mapped whatever the message says; an
    override that turns out to have been unnecessary is reported, not un-reported.
    """
    backend, gguf = _apu_pinned_projector_backend(
        tmp_path, monkeypatch, gguf_gb = 40.0, mmproj_gb = 8.0, avail_mib = 46 * 1024
    )
    extra_args = ["--no-mmproj-offload"] + (["--no-mmap"] if unmapped else [])

    captured = _launch_with_text_only_fallback(backend, gguf, extra_args = extra_args)

    assert "--mmproj" in captured["cmds"][0], captured["cmds"][0]
    assert "--mmproj" not in captured["cmds"][-1], captured["cmds"][-1]
    warning = backend.last_load_warning or ""
    assert "unified-memory APU" not in warning, (
        "the response still warns about a shortfall the resident model does not have: " f"{warning}"
    )
    if unmapped:
        # The one thing that IS still true of the running child.
        assert "memory mapping instead" in warning, warning
    else:
        assert backend.last_load_warning is None, warning


def test_the_reprice_reads_the_pool_the_preflight_saw_not_the_one_the_model_is_in(
    tmp_path, monkeypatch
):
    """The reprice asks "would the weights alone have fit?", so it has to weigh them
    against the RAM that was free BEFORE they were loaded.

    By the time it runs, the text-only child is healthy and holding the weights, so
    system RAM has fallen by roughly their size. Re-reading it there charges
    ``model_size`` against ``avail - model_size`` and finds a shortfall for a model
    that demonstrably just started: the load would keep an advisory saying it does not
    fit while it is serving. Modelled here by dropping the pool between the preflight
    and the fallback, which a fixed ``avail_mib`` in the other tests cannot express.
    """
    backend, gguf = _apu_pinned_projector_backend(
        tmp_path, monkeypatch, gguf_gb = 40.0, mmproj_gb = 8.0, avail_mib = 46 * 1024
    )
    # 46 GB free at the preflight; the resident 40 GB of weights leave 6 GB by the time
    # the text-only retry is healthy. Only the first reading is the one being repriced.
    readings = iter([46 * 1024])
    monkeypatch.setattr(
        LlamaCppBackend,
        "_available_system_memory_mib",
        staticmethod(lambda: next(readings, 6 * 1024)),
    )

    _launch_with_text_only_fallback(backend, gguf, extra_args = ["--no-mmproj-offload"])

    assert backend.last_load_warning is None, (
        "the reprice charged the weights against a pool they are already occupying, so "
        f"a model that started fine still warns it does not fit: {backend.last_load_warning}"
    )


def test_a_projector_the_weights_alone_still_outgrow_keeps_its_warning(tmp_path, monkeypatch):
    """The control. Same fallback, but the weights on their own are already too big
    for this APU, so dropping the projector changes nothing the user needs to know and
    the advisory stays."""
    backend, gguf = _apu_pinned_projector_backend(
        tmp_path, monkeypatch, gguf_gb = 64.6, mmproj_gb = 8.0, avail_mib = 46 * 1024
    )

    _launch_with_text_only_fallback(backend, gguf, extra_args = ["--no-mmproj-offload"])

    assert "unified-memory APU" in (backend.last_load_warning or "")


# ── The resident advisory outlives an attempt that never touched the server ───
def _load_rejected(backend, intent, **load_kwargs):
    """Run a load that is expected to stand down before the Phase 1 teardown, and
    report nothing about it: what the caller asserts is the server left behind."""
    try:
        return backend.load_model(intent, **load_kwargs)
    except (RuntimeError, ValueError, FileNotFoundError):
        return False


def test_a_rejected_load_leaves_the_resident_advisory_alone(tmp_path, monkeypatch):
    """An oversized model is serving, and its advisory is the only thing telling the
    user why generation crawls. A load that is refused before the teardown leaves that
    child running, so retiring its notice reported memory_warning: null for a model
    that is still paging -- including on the very next already_loaded answer."""
    backend, gguf = _apu_backend(
        tmp_path, gguf_gb = 64.6, avail_mib = 46 * 1024, monkeypatch = monkeypatch
    )

    resident = _launch(backend, gguf)
    assert resident["cmd"]
    warned = backend.last_load_warning
    assert "unified-memory APU" in (warned or "")

    # 1. The in-app update refusal: the first check in load_model, above everything.
    backend._llama_update_in_progress = True
    assert not _load_rejected(
        backend, GgufLoadIntent(gguf_path = str(gguf), model_identifier = "other")
    )
    backend._llama_update_in_progress = False
    assert backend.last_load_warning == warned, (
        "the update refusal retired the advisory of a server it never touched: "
        f"{backend.last_load_warning}"
    )

    # 2. A cancel that lands before the teardown, on a different model.
    cancelled = threading.Event()
    cancelled.set()
    assert not _load_rejected(
        backend,
        GgufLoadIntent(gguf_path = str(gguf), model_identifier = "other"),
        load_cancel_event = cancelled,
    )
    assert (
        backend.last_load_warning == warned
    ), f"the cancelled load retired the resident advisory: {backend.last_load_warning}"

    # 3. ...and the already_loaded fast path still carries it, which is what the route
    # reads for memory_warning on a repeat /load.
    assert backend.load_model(GgufLoadIntent(gguf_path = str(gguf), model_identifier = "test"))
    assert (
        backend.last_load_warning == warned
    ), f"already_loaded answered with no memory_warning: {backend.last_load_warning}"


# ── Repricing after an auto-selected Vulkan backend is replayed on CPU ─────────
# The replay launches with --gpu-layers 0 --device none, so the child that serves the
# session is credited no VRAM at all and pages the whole model from system RAM. The
# preflight priced a GPU placement that is now dead: its spill figure understates what
# the running child holds, and a model that FIT in VRAM was never priced against host
# RAM at all, which is the case the user most needs told about.
def _vulkan_cpu_replay_backend(tmp_path, monkeypatch, *, gguf_gb, free_mib, avail_mib):
    """A host whose auto-selected Vulkan build hard-crashes at startup, with a discrete
    card (Vulkan reports total 0 only for an iGPU, so this pool is real VRAM) and a
    staged CPU runtime for the replay."""
    backend, gguf = _backend(tmp_path, vulkan = True, memory = [(0, free_mib, free_mib)])
    _restore_host_guard(backend)
    backend._get_gguf_size_bytes = lambda _path: int(gguf_gb * 1024**3)
    backend._select_gpus = lambda *_a, **_kw: (None, True)
    backend.probe_server_capabilities = lambda _binary: {"found": True}
    # The real _prepare_cpu_fallback_launch and _cpu_isolated_replay run: what is being
    # asserted is how the REPLAY argv prices, so it has to be the argv the code builds.
    backend._cpu_isolated_binary = lambda _binary: "/fake/llama-server"
    backend._llama_server_env_for_binary = lambda _binary: {_loader_path_var(): ""}
    backend._record_server_pid = lambda _pid: None
    backend._clear_server_pid = lambda: None
    monkeypatch.setattr(
        LlamaCppBackend, "_is_vulkan_backend", staticmethod(lambda _binary = None: True)
    )
    monkeypatch.setattr(
        LlamaCppBackend, "_vulkan_prebuilt_was_auto_selected", staticmethod(lambda _binary: True)
    )
    monkeypatch.setattr(
        LlamaCppBackend, "_available_system_memory_mib", staticmethod(lambda: avail_mib)
    )
    monkeypatch.delenv("UNSLOTH_ALLOW_HOST_OFFLOAD", raising = False)
    return backend, gguf


def _launch_with_vulkan_cpu_replay(
    backend,
    gguf,
    *,
    crash = True,
    **load_kwargs,
):
    """Every launch that can still reach a GPU dies of a signal, which is what a broken
    Vulkan backend does; only the device-less replay comes up healthy. Written against
    the argv rather than the attempt number so the intermediate rungs (the --flash-attn
    off retry) crash too instead of masking the fallback under test.

    ``crash=False`` is the control: the same host, the same model, one launch that
    works, so nothing is repriced.
    """
    captured = {"cmds": []}

    def _is_cpu_replay(cmd):
        return "--device" in cmd and cmd[cmd.index("--device") + 1] == "none"

    def fake_popen(cmd, **kwargs):
        if not cmd or str(cmd[0]) != "/fake/llama-server":
            return _REAL_POPEN(cmd, **kwargs)
        rc = -11 if (crash and not _is_cpu_replay(list(cmd))) else None
        captured["cmds"].append(list(cmd))
        captured["env"] = kwargs.get("env") or dict(os.environ)
        return type(
            "Process",
            (),
            {
                "pid": 123,
                "stdout": (),
                "returncode": rc,
                "poll": lambda self: rc,
                "terminate": lambda self: None,
                "wait": lambda self, timeout = None: rc,
                "kill": lambda self: None,
            },
        )()

    def fake_health(timeout = None):
        if crash and not _is_cpu_replay(captured["cmds"][-1] if captured["cmds"] else []):
            backend._stdout_lines = ["ggml_vulkan: Device memory allocation failed"]
            return False
        backend._stdout_lines = []
        return True

    backend._wait_for_health = fake_health

    with patch.object(subprocess, "Popen", side_effect = fake_popen):
        assert backend.load_model(
            GgufLoadIntent(
                gguf_path = str(gguf),
                model_identifier = "test",
                **load_kwargs,
            )
        )
    if crash:
        assert len(captured["cmds"]) > 1, captured["cmds"]
        assert backend._cpu_fallback_reason == "vulkan_startup_crash"
        assert _is_cpu_replay(captured["cmds"][-1]), captured["cmds"][-1]
    return captured


def test_a_model_that_fits_vram_but_not_ram_is_warned_once_it_lands_on_cpu(tmp_path, monkeypatch):
    """20 GB of weights on a 24 GB card: nothing spills, so the preflight has nothing
    to say. The Vulkan backend then crashes and the replay runs on no GPU at all, so
    the whole 20 GB has to come out of a 12 GB host and pages from disk for the rest of
    the session -- and memory_warning was null for exactly that load."""
    backend, gguf = _vulkan_cpu_replay_backend(
        tmp_path, monkeypatch, gguf_gb = 20.0, free_mib = 24 * 1024, avail_mib = 12 * 1024
    )

    _launch_with_vulkan_cpu_replay(backend, gguf)

    warning = backend.last_load_warning or ""
    assert "does not fit in GPU memory" in warning, (
        "the CPU-only child pages the whole model from disk and says nothing about it: "
        f"{warning!r}"
    )
    assert "About 20 GB" in warning, warning


def test_a_gpu_placement_warning_does_not_survive_onto_the_cpu_child(tmp_path, monkeypatch):
    """20 GB of weights on an 8 GB card spill 12 GB, which a 13 GB host cannot hold, so
    the preflight warns about 12 GB. That placement then dies. The child that serves the
    session holds all 20 GB in RAM, so the figure it inherited describes an offload it
    no longer performs -- and understates the one it does."""
    backend, gguf = _vulkan_cpu_replay_backend(
        tmp_path, monkeypatch, gguf_gb = 20.0, free_mib = 8 * 1024, avail_mib = 13 * 1024
    )

    _launch_with_vulkan_cpu_replay(backend, gguf)

    warning = backend.last_load_warning or ""
    assert (
        "About 20 GB" in warning
    ), f"the CPU-only child still reports the dead GPU placement's spill: {warning!r}"
    assert "About 12 GB" not in warning, warning


def test_a_vulkan_load_that_never_falls_back_keeps_its_own_advisory(tmp_path, monkeypatch):
    """The control. Same host and same model as above, but the GPU launch comes up:
    nothing is replayed, so the advisory is the GPU placement's own spill, untouched."""
    backend, gguf = _vulkan_cpu_replay_backend(
        tmp_path, monkeypatch, gguf_gb = 20.0, free_mib = 8 * 1024, avail_mib = 13 * 1024
    )

    captured = _launch_with_vulkan_cpu_replay(backend, gguf, crash = False)

    assert len(captured["cmds"]) == 1, captured["cmds"]
    assert backend._cpu_fallback_reason is None
    warning = backend.last_load_warning or ""
    assert "About 12 GB" in warning, warning
    assert "About 20 GB" not in warning, warning


def test_the_cpu_reprice_carries_the_pageable_override_note_it_did_not_revert(
    tmp_path, monkeypatch
):
    """An unmapped oversized load is remapped before the first spawn: force_pageable_load
    rewrites the argv and the shared child env in place. The replay is built from that
    argv and that env and strips placement alone, so the CPU child really is
    memory-mapped -- the note stays true and is carried onto the repriced text verbatim,
    not rebuilt and not taken back."""
    backend, gguf = _vulkan_cpu_replay_backend(
        tmp_path, monkeypatch, gguf_gb = 20.0, free_mib = 8 * 1024, avail_mib = 13 * 1024
    )

    captured = _launch_with_vulkan_cpu_replay(backend, gguf, extra_args = ["--no-mmap"])

    replay = captured["cmds"][-1]
    assert not _unmapped_tokens(replay), f"the CPU child loads unmapped: {replay}"
    warning = backend.last_load_warning or ""
    assert "About 20 GB" in warning, warning
    assert warning.count("memory mapping instead") == 1, warning


@pytest.mark.parametrize("extra_args", _UNMAPPED_ARGV, ids = _UNMAPPED_IDS)
def test_a_replay_that_loses_its_vram_is_repaged_before_it_spawns(
    tmp_path, monkeypatch, extra_args
):
    """20 GB of weights on a 24 GB card, loaded unmapped by request, on a 12 GB host.

    The preflight leaves the mode alone and is right to: nothing spills, so there is no
    shortfall and an unmapped load that fits is exactly what was asked for. Then Vulkan
    hard-crashes and the replay appends "--gpu-layers 0 --fit off --device none", which
    credits it no VRAM at all. The same 20 GB now has to come out of a 12 GB host, and
    unmapped that is one allocation of the whole file rather than a mapping that pages:
    an OOM kill, not a slow load. So the replay is priced on its own footprint and
    repaged before it spawns, exactly as the main launch path would have done.
    """
    backend, gguf = _vulkan_cpu_replay_backend(
        tmp_path, monkeypatch, gguf_gb = 20.0, free_mib = 24 * 1024, avail_mib = 12 * 1024
    )

    captured = _launch_with_vulkan_cpu_replay(backend, gguf, extra_args = extra_args)

    gpu_attempt, replay = captured["cmds"][0], captured["cmds"][-1]
    assert _unmapped_tokens(gpu_attempt) == list(
        extra_args
    ), f"the fitting GPU launch lost the mode it asked for: {gpu_attempt}"
    assert not _unmapped_tokens(
        replay
    ), f"the CPU replay holds the whole model in host RAM unmapped: {replay}"
    # The advisory has to describe the child that is running: the whole model against
    # host RAM, and the override that is why it can page at all.
    warning = backend.last_load_warning or ""
    assert "About 20 GB" in warning, warning
    assert warning.count("memory mapping instead") == 1, warning
    # And the record the reload comparator reads, or the next Apply judges this child
    # against a mode it no longer runs.
    assert backend._memory_state == (False, False), backend._memory_state


@pytest.mark.parametrize("extra_args", _UNMAPPED_ARGV, ids = _UNMAPPED_IDS)
def test_a_replay_the_host_can_actually_hold_keeps_the_mode_it_was_asked_for(
    tmp_path, monkeypatch, extra_args
):
    """The control. Same crash, same replay with no VRAM credited, but a 64 GB host
    holds all 20 GB outright. Nothing is oversized, so loading unmapped is the
    legitimate choice it always was and no override and no warning are invented."""
    backend, gguf = _vulkan_cpu_replay_backend(
        tmp_path, monkeypatch, gguf_gb = 20.0, free_mib = 24 * 1024, avail_mib = 64 * 1024
    )

    captured = _launch_with_vulkan_cpu_replay(backend, gguf, extra_args = extra_args)

    replay = captured["cmds"][-1]
    assert _unmapped_tokens(replay) == list(
        extra_args
    ), f"the CPU replay lost the mode the user asked for: {replay}"
    assert backend.last_load_warning is None, backend.last_load_warning


def test_the_opt_out_silences_the_replay_warning_without_licensing_the_oom(tmp_path, monkeypatch):
    """UNSLOTH_ALLOW_HOST_OFFLOAD is warning-scoped, the same contract the main launch
    path holds it to: it hides the message, it does not hand the child a load it cannot
    complete. So the replay is still repaged and the override is still logged."""
    backend, gguf = _vulkan_cpu_replay_backend(
        tmp_path, monkeypatch, gguf_gb = 20.0, free_mib = 24 * 1024, avail_mib = 12 * 1024
    )
    monkeypatch.setenv("UNSLOTH_ALLOW_HOST_OFFLOAD", "1")

    captured = _launch_with_vulkan_cpu_replay(backend, gguf, extra_args = ["--no-mmap"])

    assert not _unmapped_tokens(captured["cmds"][-1]), captured["cmds"][-1]
    assert backend.last_load_warning is None, backend.last_load_warning


def test_an_effective_lock_survives_the_replay_override_as_a_mapped_one(tmp_path, monkeypatch):
    """force_pageable_load's own rule, reached through this rung: "keep this in RAM" is
    honoured over a mapping the kernel can fall back on, so --load-mode mlock becomes
    mmap+mlock rather than losing the lock."""
    backend, gguf = _vulkan_cpu_replay_backend(
        tmp_path, monkeypatch, gguf_gb = 20.0, free_mib = 24 * 1024, avail_mib = 12 * 1024
    )

    captured = _launch_with_vulkan_cpu_replay(backend, gguf, extra_args = ["--load-mode", "mlock"])

    replay = captured["cmds"][-1]
    assert not _unmapped_tokens(replay), replay
    assert "mmap+mlock" in replay, replay


def test_a_shadowed_lock_is_not_resurrected_by_the_replay_override(tmp_path, monkeypatch):
    """The other half of that rule. "--mlock --no-mmap" already runs unlocked, so
    dropping only the selector would page-lock the whole oversized mapping into the RAM
    this override exists to keep pageable."""
    backend, gguf = _vulkan_cpu_replay_backend(
        tmp_path, monkeypatch, gguf_gb = 20.0, free_mib = 24 * 1024, avail_mib = 12 * 1024
    )

    captured = _launch_with_vulkan_cpu_replay(backend, gguf, extra_args = ["--mlock", "--no-mmap"])

    replay = captured["cmds"][-1]
    assert not _unmapped_tokens(replay), replay
    assert "--mlock" not in replay and "mmap+mlock" not in replay, replay


def test_the_cpu_reprice_reads_the_pool_the_preflight_saw_not_the_one_the_model_is_in(
    tmp_path, monkeypatch
):
    """The reprice asks "does the whole model fit in host RAM?", so it has to weigh it
    against the RAM that was free BEFORE the replay loaded it.

    By the time it runs, the CPU child is healthy and holding every byte, so a fresh
    reading is the pool minus the model being priced: it would charge 20 GB against the
    4 GB left over and warn that a model which demonstrably just started does not fit.
    """
    backend, gguf = _vulkan_cpu_replay_backend(
        tmp_path, monkeypatch, gguf_gb = 20.0, free_mib = 24 * 1024, avail_mib = 24 * 1024
    )
    # 24 GB free at the preflight, 4 GB once the 20 GB of weights are resident.
    readings = iter([24 * 1024])
    monkeypatch.setattr(
        LlamaCppBackend,
        "_available_system_memory_mib",
        staticmethod(lambda: next(readings, 4 * 1024)),
    )

    _launch_with_vulkan_cpu_replay(backend, gguf)

    assert backend.last_load_warning is None, (
        "the reprice charged the weights against a pool they already occupy, so a model "
        f"that started fine still warns it does not fit: {backend.last_load_warning!r}"
    )


@pytest.mark.parametrize("extra_args", _UNMAPPED_ARGV, ids = _UNMAPPED_IDS)
def test_an_unmapped_load_is_priced_against_the_cards_the_pin_left_it(
    tmp_path, monkeypatch, extra_args
):
    """VRAM on a card this launch pinned away is not VRAM the child can spend.

    Two 24 GB cards, and Unsloth pins the child to one of them. A 25 GB model is
    smaller than the pair and larger than the card it actually gets, so the spill
    is real and the 3 GB host cannot hold it. Summing both cards prices that spill
    at zero, leaves the unmapped request standing, and the child then allocates the
    offloaded weights in host RAM and is OOM-killed: the override exists to turn
    exactly that into a slow load.
    """
    backend, gguf = _backend(
        tmp_path, vulkan = False, memory = [(0, 20_000, 24_576), (1, 20_000, 24_576)]
    )
    _restore_host_guard(backend)
    backend._get_gguf_size_bytes = lambda _path: int(25 * 1024**3)
    backend._select_gpus = lambda *args, **kw: ([0], False)
    monkeypatch.setattr(
        LlamaCppBackend, "_available_system_memory_mib", staticmethod(lambda: 3_000)
    )
    monkeypatch.delenv("UNSLOTH_ALLOW_HOST_OFFLOAD", raising = False)

    result = _launch(backend, gguf, extra_args = extra_args)
    cmd = result["cmd"]

    assert cmd, "the pinned unmapped load never spawned llama-server"
    assert not _unmapped_tokens(
        cmd
    ), f"the child still loads unmapped, priced against a card the pin hid: {cmd}"
    assert "memory mapping instead" in (backend.last_load_warning or "")


def test_a_pinned_load_that_fits_the_card_it_got_keeps_its_mode(tmp_path, monkeypatch):
    """The control, and the one that stops the fix double-counting a pin.

    Same pin, same single reachable card, but the model fits it. Nothing is
    oversized, so the request survives untouched and no warning is invented. This
    is also what fails if the reachable set is ever derived from an INHERITED
    CUDA_VISIBLE_DEVICES: the probe has already applied that mask, so subtracting
    it again would price a card the child really does have as absent.
    """
    backend, gguf = _backend(
        tmp_path, vulkan = False, memory = [(0, 20_000, 24_576), (1, 20_000, 24_576)]
    )
    _restore_host_guard(backend)
    backend._get_gguf_size_bytes = lambda _path: int(8 * 1024**3)
    backend._select_gpus = lambda *args, **kw: ([0], False)
    monkeypatch.setattr(
        LlamaCppBackend, "_available_system_memory_mib", staticmethod(lambda: 3_000)
    )
    monkeypatch.delenv("UNSLOTH_ALLOW_HOST_OFFLOAD", raising = False)

    cmd = _launch(backend, gguf, extra_args = ["--no-mmap"])["cmd"]

    assert _unmapped_tokens(cmd) == ["--no-mmap"], f"the fitting load lost its mode: {cmd}"
    assert backend.last_load_warning is None


def test_a_restored_cpu_fallback_is_priced_against_host_ram(tmp_path, monkeypatch):
    """A persisted cpu_fallback load reconstructs the CPU-only replay directly, with
    no crash to price around, so nothing upstream ever warned about it.

    The Vulkan probe returns an empty pool on this shape, which leaves
    `_child_has_no_gpu` False and makes the preflight abstain, so there is no notice
    for the amend on that path to add to. The child then serves the whole model out of
    system RAM and `memory_warning` came back null for it, which is the session that
    most needs the advisory.
    """
    backend, gguf = _vulkan_cpu_replay_backend(
        tmp_path, monkeypatch, gguf_gb = 20.0, free_mib = 4_000, avail_mib = 12_000
    )
    # The probe comes back EMPTY, which is the shape that makes the preflight abstain:
    # _child_has_no_gpu credits _cpu_only_zero_offload only when a device was detected,
    # so with no rows it stays False and the guard cannot tell an unreadable pool from
    # a host with no GPU. A row reporting zero free is a different, already-warned case.
    backend._get_gpu_memory = lambda _binary = None, **_kw: []
    backend._get_gpu_free_memory = lambda _binary = None, **_kw: []

    _launch_with_vulkan_cpu_replay(backend, gguf, crash = False, cpu_fallback = True)

    warning = backend.last_load_warning or ""
    assert (
        "20 GB" in warning
    ), f"a restored CPU-only session reported nothing about paging the model: {warning!r}"


def test_a_restored_cpu_fallback_the_host_can_hold_says_nothing(tmp_path, monkeypatch):
    """The control. Same restored path, a host with room: no shortfall, so no advisory
    is invented for a session that is running comfortably."""
    backend, gguf = _vulkan_cpu_replay_backend(
        tmp_path, monkeypatch, gguf_gb = 20.0, free_mib = 4_000, avail_mib = 64_000
    )
    backend._get_gpu_memory = lambda _binary = None, **_kw: []
    backend._get_gpu_free_memory = lambda _binary = None, **_kw: []

    _launch_with_vulkan_cpu_replay(backend, gguf, crash = False, cpu_fallback = True)

    assert backend.last_load_warning is None
