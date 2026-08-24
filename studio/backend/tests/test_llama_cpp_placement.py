# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Focused integration tests for explicit GGUF GPU placement."""

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

from core.inference.llama_cpp import GgufLoadIntent, LlamaCppBackend

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
    so Studio has no reason to rewrite it."""
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
    backend._get_gguf_size_bytes = lambda path: (0 if str(path) == str(sidecar) else 8 * gb)
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
    backend.probe_server_capabilities = lambda _binary = None: caps or {
        "mtp_token": "draft-mtp",
        "supports_dflash": True,
        "supports_ngram_mod": True,
        "spec_draft_n_max_flag": "--spec-draft-n-max",
        # Or the launch clamps the four slots to one and the per-slot state,
        # which is what these tests measure, shrinks with them.
        "supports_kv_unified": True,
    }
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
    # Studio emits no --spec-draft-n-max when the extras own the spec block, so
    # the child runs at the build's own default. Budgeting Studio's 2 instead
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
    # drafting at whatever the build defaults to, so Studio's own explicit-mode 2
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


def test_a_subset_that_can_shrink_to_hold_both_is_where_the_decision_lands(tmp_path):
    """The placement loop does not walk past a subset that fails with the drafter.

    It re-caps the context WITH the drafter charged and accepts that subset at
    whatever is left, so a smaller context holding both here IS the placement the
    load takes, and the drafter gets paid for in context. Believing a later,
    larger subset would rescue it keeps the drafter and shrinks the context, which
    is the trade this exists to refuse.
    """
    gb = 1024**3
    backend, gguf = _backend(
        tmp_path, vulkan = False, memory = [(0, 24_576, 24_576), (1, 24_576, 24_576)]
    )
    sidecar = tmp_path / "dspark-model-Q8_0.gguf"
    sidecar.write_bytes(b"draft")
    backend._get_gguf_size_bytes = lambda path: 8 * gb if str(path) == str(sidecar) else 16 * gb
    backend._read_gguf_metadata = lambda _path: setattr(backend, "_context_length", 8192)
    backend._can_estimate_kv = lambda: True
    # Context-linear throughout, so GPU0 alone can shrink its way to holding both.
    backend._estimate_kv_cache_bytes = lambda ctx, *a, **k: int(ctx * 0.5 * 1024**2)
    backend._compute_buffer_ctx_bytes = lambda *a, **k: 0
    backend._estimate_compute_buffer_bytes = lambda **k: 1
    backend._mtp_draft_kv_bytes = lambda *a, **k: 0
    backend._estimate_mtp_overhead_bytes = lambda ctx, *a, **k: int(ctx * 0.75 * 1024**2)
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
    assert "--model-draft" not in cmd
    assert backend.spec_fallback_reason == "drafter_no_vram"
    assert cmd[cmd.index("-c") + 1] == "8192"


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


def test_weights_larger_than_vram_plus_ram_are_refused(tmp_path, monkeypatch):
    """The field case: a 13.3 GB GGUF on a 6 GB laptop card holding 4877 MiB free needs
    about 8.5 GB of host RAM, which a 10 GB host cannot hold. Unrefused, the mmap'd
    remainder thrashes until the OS kills Studio and the desktop session."""
    backend, gguf = _offload_backend(
        tmp_path, gguf_gb = 13.3, free_mib = 4877, avail_mib = 10_000, monkeypatch = monkeypatch
    )

    with pytest.raises(RuntimeError, match = "does not fit in GPU memory"):
        _launch(backend, gguf)


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

    with pytest.raises(RuntimeError, match = "does not fit in GPU memory"):
        _launch(backend, gguf)


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
    backend._get_gpu_memory = (
        lambda _binary = None, **_kw: LlamaCppBackend._get_gpu_free_memory_vulkan(_binary)
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
        with pytest.raises(RuntimeError, match = "does not fit in GPU memory"):
            _launch(backend, gguf)
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

    with pytest.raises(RuntimeError, match = "does not fit in GPU memory"):
        _launch(backend, gguf, **placement)


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
    with pytest.raises(RuntimeError, match = "does not fit in GPU memory"):
        _launch(backend, gguf, extra_args = ["--device", "Vulkan0"], **manual)

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

    with pytest.raises(RuntimeError, match = "does not fit in GPU memory"):
        _launch(
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

    with pytest.raises(RuntimeError, match = "does not fit in GPU memory"):
        _launch(
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
    with pytest.raises(RuntimeError, match = "does not fit in GPU memory"):
        _launch(
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

    with pytest.raises(RuntimeError, match = "does not fit in GPU memory"):
        _launch(backend, gguf, gpu_memory_mode = "manual", gpu_layers = 33)


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

    with pytest.raises(RuntimeError, match = "unified-memory APU"):
        _launch(backend, gguf)


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
    with pytest.raises(RuntimeError, match = "does not fit in GPU memory"):
        _launch(backend, gguf)

    assert str(gguf) in seen


def test_the_env_escape_loads_a_variant_the_guard_refuses(tmp_path, monkeypatch):
    """The picker still offers a variant `classifyGgufFit` calls "oom", and no load
    field carries a force, so an unconditional refusal leaves that selection with no way
    through. UNSLOTH_ALLOW_HOST_OFFLOAD=1 abstains, and the refusal names it."""
    refused, gguf = _offload_backend(
        tmp_path, gguf_gb = 13.3, free_mib = 4877, avail_mib = 10_000, monkeypatch = monkeypatch
    )
    with pytest.raises(RuntimeError, match = "UNSLOTH_ALLOW_HOST_OFFLOAD=1"):
        _launch(refused, gguf)

    allowed_dir = tmp_path / "allowed"
    allowed_dir.mkdir()
    allowed, gguf2 = _offload_backend(
        allowed_dir, gguf_gb = 13.3, free_mib = 4877, avail_mib = 10_000, monkeypatch = monkeypatch
    )
    monkeypatch.setenv("UNSLOTH_ALLOW_HOST_OFFLOAD", "1")
    assert "--fit" in _launch(allowed, gguf2)["cmd"]


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

    verdict = backend.host_offload_refusal_for_intent(_load_intent(gguf))
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

    assert backend.host_offload_refusal_for_intent(_load_intent(gguf)) is None


def test_the_route_precheck_only_refuses_what_the_launch_would(tmp_path, monkeypatch):
    """Abstains on an undownloaded repo, a device whose total the probe cannot read, an
    unreadable pool, unreadable total RAM and the escape. So it can never reject a load the
    launch would allow."""
    backend, gguf = _offload_backend(
        tmp_path, gguf_gb = 100, free_mib = 20_000, avail_mib = 10_000, monkeypatch = monkeypatch
    )
    _host_totals(monkeypatch, backend, vram_total_mib = 24_000, ram_total_mib = 32_000)

    assert backend.host_offload_refusal_for_intent(_load_intent(gguf, hf_repo = "org/repo")) is None
    # an igpu or a MIG/vGPU line reports total 0, so the ceiling is unknown
    backend._get_gpu_memory = lambda _binary = None, **_kw: [(0, 20_000, 0)]
    assert backend.host_offload_refusal_for_intent(_load_intent(gguf)) is None
    backend._get_gpu_memory = lambda _binary = None, **_kw: []
    assert backend.host_offload_refusal_for_intent(_load_intent(gguf)) is None
    _host_totals(monkeypatch, backend, vram_total_mib = 24_000, ram_total_mib = None)
    assert backend.host_offload_refusal_for_intent(_load_intent(gguf)) is None
    _host_totals(monkeypatch, backend, vram_total_mib = 24_000, ram_total_mib = 32_000)
    monkeypatch.setenv("UNSLOTH_ALLOW_HOST_OFFLOAD", "1")
    assert backend.host_offload_refusal_for_intent(_load_intent(gguf)) is None


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
    """Studio installs a CPU-only prebuilt on a host with no GPU, so that host probes an
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
    with pytest.raises(RuntimeError, match = "does not fit in GPU memory"):
        _launch(cpu_build, gguf2)


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
