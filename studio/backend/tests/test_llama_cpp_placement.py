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
    backend._get_gpu_memory = lambda _binary = None: list(memory)
    backend._get_gpu_free_memory = lambda _binary = None: [
        (index, free) for index, free, _total in memory
    ]
    backend._read_gguf_metadata = lambda _path: None
    backend._can_estimate_kv = lambda: False
    backend._get_gguf_size_bytes = lambda _path: 1024
    backend._mmproj_vram_bytes = lambda _path: 0
    backend._resolve_launch_mmproj_path = lambda **kwargs: None
    backend._apu_ram_shortfall_message = lambda *args, **kwargs: None
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
    backend._select_gpus = lambda *args, **kwargs: ((None, True) if use_fit else ([0], False))
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
    backend._get_gpu_memory = lambda _binary = None: [(1, 8_000, 8_000)]
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
    backend._get_gguf_size_bytes = lambda path: (6 * gb if str(path) == str(sidecar) else 16 * gb)
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
    backend._get_gguf_size_bytes = lambda path: (8 * gb if str(path) == str(sidecar) else 16 * gb)
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
