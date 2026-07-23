# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for GGUF memory placement mode and explicit GPU selection (#7164)."""

from __future__ import annotations

import os
import struct
import subprocess
import sys
import types as _types
from pathlib import Path
from unittest.mock import patch

_REAL_POPEN = subprocess.Popen

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)


def _install_stub_if_absent(name: str, build):
    """Install a stub for ``name`` only when the real module isn't importable, so a
    stub never shadows a real module and breaks unrelated tests in a combined pytest
    run. The stub stays a pure fallback for minimal environments."""
    if name in sys.modules:
        return
    try:
        __import__(name)
        return
    except Exception:
        sys.modules[name] = build()


def _build_loggers_stub():
    mod = _types.ModuleType("loggers")
    mod.get_logger = lambda name: __import__("logging").getLogger(name)
    return mod


def _build_structlog_stub():
    mod = _types.ModuleType("structlog")
    mod.get_logger = lambda *a, **k: __import__("logging").getLogger("stub")
    return mod


def _build_httpx_stub():
    mod = _types.ModuleType("httpx")
    for _exc_name in (
        "ConnectError",
        "TimeoutException",
        "ReadTimeout",
        "ReadError",
        "RemoteProtocolError",
        "CloseError",
    ):
        setattr(mod, _exc_name, type(_exc_name, (Exception,), {}))
    mod.Timeout = type("T", (), {"__init__": lambda s, *a, **k: None})
    mod.Client = type(
        "Client",
        (),
        {
            "__init__": lambda self, **kw: None,
            "__enter__": lambda self: self,
            "__exit__": lambda self, *a: None,
        },
    )
    return mod


def _build_jwt_stub():
    mod = _types.ModuleType("jwt")
    mod.decode = lambda *a, **k: {}
    mod.ExpiredSignatureError = type("ExpiredSignatureError", (Exception,), {})
    mod.InvalidTokenError = type("InvalidTokenError", (Exception,), {})
    return mod


_install_stub_if_absent("loggers", _build_loggers_stub)
_install_stub_if_absent("structlog", _build_structlog_stub)
_install_stub_if_absent("httpx", _build_httpx_stub)
_install_stub_if_absent("jwt", _build_jwt_stub)

import pytest

from core.inference.llama_cpp import LlamaCppBackend

_GGUF_MAGIC = 0x46554747
_VTYPE_STRING = 8


def _enc_string(s: str) -> bytes:
    b = s.encode("utf-8")
    return struct.pack("<Q", len(b)) + b


def _enc_kv_string(key: str, value: str) -> bytes:
    return _enc_string(key) + struct.pack("<I", _VTYPE_STRING) + _enc_string(value)


def _write_minimal_gguf(path: Path, *, arch: str = "llama") -> Path:
    body = _enc_kv_string("general.architecture", arch)
    header = struct.pack("<IIQQ", _GGUF_MAGIC, 3, 0, 1)
    path.write_bytes(header + body)
    return path


class _FakeProcess:
    """Minimal stand-in so is_loaded returns True."""

    def terminate(self):
        pass

    def wait(self, timeout = None):
        return 0

    def kill(self):
        pass

    def poll(self):
        return 0


def _loaded_backend(**overrides):
    backend = LlamaCppBackend()
    backend._process = _FakeProcess()
    backend._healthy = True
    backend._model_identifier = "owner/repo"
    backend._hf_variant = "Q4_K_M"
    backend._requested_n_ctx = 8192
    backend._cache_type_kv = None
    backend._speculative_type = None
    backend._requested_spec_mode = "auto"
    backend._chat_template_override = None
    backend._is_vision = False
    backend._extra_args = None
    backend._extra_args_source = None
    backend._gguf_path = None
    backend._gpu_ids = None
    backend._memory_mode = None
    for key, value in overrides.items():
        setattr(backend, key, value)
    if "_gpu_ids" in overrides and "_requested_gpu_ids" not in overrides:
        backend._requested_gpu_ids = list(overrides["_gpu_ids"] or []) or None
    return backend


# ── _memory_mode_flags ───────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "mode,expected",
    [
        (None, []),
        ("auto", []),
        ("AUTO", []),
        ("pinned", ["--mlock"]),
        ("PINNED", ["--mlock"]),
        ("resident", ["--no-mmap", "--mlock"]),
        ("RESIDENT", ["--no-mmap", "--mlock"]),
        ("", []),
    ],
)
def test_memory_mode_flags_maps_modes(mode, expected):
    assert LlamaCppBackend._memory_mode_flags(mode) == expected


@pytest.mark.parametrize(
    "mode,expected",
    [
        (None, []),
        ("auto", []),
        ("pinned", ["--load-mode", "mlock"]),
        ("resident", ["--load-mode", "none"]),
    ],
)
def test_memory_mode_flags_use_unified_load_mode(mode, expected):
    assert (
        LlamaCppBackend._memory_mode_flags(mode, supports_load_mode = True)
        == expected
    )


# ── _already_in_target_state ─────────────────────────────────────────────────


def _base_target_state_kwargs(backend):
    return {
        "model_identifier": "owner/repo",
        "hf_variant": "Q4_K_M",
        "n_ctx": 8192,
        "cache_type_kv": None,
        "speculative_type": None,
        "chat_template_override": None,
        "extra_args": None,
        "is_vision": False,
        "gpu_ids": backend.gpu_ids,
        "memory_mode": backend.memory_mode,
    }


def test_already_in_target_state_matches_same_memory_mode():
    backend = _loaded_backend(_memory_mode = "resident")
    kwargs = _base_target_state_kwargs(backend)
    assert backend._already_in_target_state(**kwargs) is True


def test_already_in_target_state_rejects_different_memory_mode():
    backend = _loaded_backend(_memory_mode = "resident")
    kwargs = _base_target_state_kwargs(backend)
    kwargs["memory_mode"] = "pinned"
    assert backend._already_in_target_state(**kwargs) is False


def test_already_in_target_state_keeps_device_extras_without_gpu_ids():
    # Without gpu_ids, --device is not stripped, so a genuine extras change still reloads.
    backend = _loaded_backend(_gpu_ids = None, _extra_args = ["--flash-attn", "on"])
    kwargs = _base_target_state_kwargs(backend)
    kwargs["gpu_ids"] = None
    kwargs["extra_args"] = ["--flash-attn", "on", "--device", "CUDA0"]
    assert backend._already_in_target_state(**kwargs) is False


def test_already_in_target_state_strips_device_extras_under_gpu_ids():
    # load_model stores device-stripped extras when gpu_ids owns placement, so a
    # duplicate /load carrying a user --device must strip the same way before the
    # dedupe compare, else it needlessly restarts an already-correct server (#7188).
    backend = _loaded_backend(_gpu_ids = [0, 1], _extra_args = ["--flash-attn", "on"])
    kwargs = _base_target_state_kwargs(backend)
    kwargs["gpu_ids"] = [0, 1]
    kwargs["extra_args"] = ["--flash-attn", "on", "--device", "CUDA0"]
    assert backend._already_in_target_state(**kwargs) is True


# ── GPU selection: reject-before behaviour (host-residency / Vulkan hardening) ─
# The pure gpu_ids filter/order/normalize picker tests from #7188 are dropped here:
# #6414 already covers the physical-ID picker on main. What remains is the
# resolvability preflight the route runs (assert_requested_gpu_ids_resolvable) to
# reject a bad selection BEFORE the active model is torn down (#7188).


def _fit_fallback_backend(
    tmp_path,
    gpu_memory,
    *,
    vulkan = False,
):
    """Backend stubbed like the --fit-fallback test but with a configurable probe."""
    gguf = tmp_path / "model.gguf"
    _write_minimal_gguf(gguf)

    backend = LlamaCppBackend()
    backend._get_gpu_memory = lambda _binary = None: list(gpu_memory)
    backend._read_gguf_metadata = lambda _p: None
    backend._can_estimate_kv = lambda: False
    backend._get_gguf_size_bytes = lambda _p: 1024
    backend._mmproj_vram_bytes = lambda _p: 0
    backend._resolve_launch_mmproj_path = lambda **k: None
    backend._apu_ram_shortfall_message = lambda *a, **k: None
    backend._amd_apu_wants_unified_memory = lambda *a, **k: False
    backend._find_llama_server_binary = lambda include_denied = False: "/fake/llama-server"
    backend._is_vulkan_backend = lambda _binary = None: vulkan
    backend._get_gpu_free_memory = lambda _binary = None: list(gpu_memory)
    backend._wait_for_health = lambda timeout: True
    backend._detect_audio_type_strict = lambda: None
    backend._apply_detected_audio = lambda _d: True
    return backend, gguf


def test_empty_probe_preserves_explicit_gpu_ids(tmp_path):
    """A CUDA/ROCm build with an empty probe must still honor a route-validated explicit
    gpu_ids, pinning via CUDA_VISIBLE_DEVICES + --fit on instead of raising (#7164)."""
    from utils.hardware import DeviceType

    backend, gguf = _fit_fallback_backend(tmp_path, gpu_memory = [])  # empty probe

    captured = {}

    def _make_fake_popen(cmd, **kwargs):
        if not cmd or str(cmd[0]) != "/fake/llama-server":
            return _REAL_POPEN(cmd, **kwargs)

        class _FakePopen:
            pid = 12345

            def __init__(self, cmd, **kwargs):
                captured["env"] = kwargs.get("env") or dict(os.environ)
                captured["cmd"] = list(cmd)

            def poll(self):
                return None

        return _FakePopen(cmd, **kwargs)

    with (
        patch.object(subprocess, "Popen", side_effect = _make_fake_popen),
        # CUDA host with telemetry down: the mask fallback only applies on CUDA/ROCm.
        patch("utils.hardware.get_device", return_value = DeviceType.CUDA),
        patch("utils.hardware.get_parent_visible_gpu_ids", return_value = [0, 1]),
    ):
        assert backend.load_model(
            gguf_path = str(gguf),
            model_identifier = "test",
            gpu_ids = [1],
        )

    assert captured["env"]["CUDA_VISIBLE_DEVICES"] == "1"
    cmd = captured["cmd"]
    assert "--fit" in cmd and cmd[cmd.index("--fit") + 1] == "on"


def test_empty_probe_still_rejects_vulkan_gpu_ids(tmp_path):
    """A Vulkan build cannot map physical ids to ggml ordinals without a probe, so an
    empty Vulkan probe must keep rejecting the selection (tracked as #7201) rather than
    silently spreading the load across every device. The route runs this preflight via
    assert_requested_gpu_ids_resolvable BEFORE any teardown (the merged impl moved the
    check out of load_model into the route)."""
    backend, _gguf = _fit_fallback_backend(tmp_path, gpu_memory = [], vulkan = True)

    with patch("utils.hardware.get_parent_visible_gpu_ids", return_value = []):
        with pytest.raises(ValueError, match = "do not match any visible GPUs"):
            backend.assert_requested_gpu_ids_resolvable([0])


def test_torchless_vulkan_populated_probe_uses_identity_ordinals(tmp_path):
    """A torch-less Vulkan host has an empty parent-visible mask, so gpu_ids ARE the
    Vulkan ordinals: with a populated probe the selection loads (pinned via --device
    Vulkan<i>) instead of raising (#7188)."""
    backend, gguf = _fit_fallback_backend(
        tmp_path, gpu_memory = [(0, 10000, 16000), (1, 8000, 16000)], vulkan = True
    )
    backend._select_gpus = lambda *a, **k: ([1], False)

    captured = {}

    def _make_fake_popen(cmd, **kwargs):
        if not cmd or str(cmd[0]) != "/fake/llama-server":
            return _REAL_POPEN(cmd, **kwargs)

        class _FakePopen:
            pid = 321

            def __init__(self, cmd, **kwargs):
                captured["cmd"] = list(cmd)

            def poll(self):
                return None

        return _FakePopen(cmd, **kwargs)

    with (
        patch.object(subprocess, "Popen", side_effect = _make_fake_popen),
        patch("utils.hardware.get_parent_visible_gpu_ids", return_value = []),
    ):
        assert backend.load_model(
            gguf_path = str(gguf),
            model_identifier = "test",
            gpu_ids = [0, 1],
        )
    cmd = captured["cmd"]
    assert "--device" in cmd and cmd[cmd.index("--device") + 1] == "Vulkan1"
    assert backend.gpu_ids == [1]
    assert backend.requested_gpu_ids == [0, 1]


def test_vulkan_fit_keeps_discrete_device_selected(tmp_path):
    """Crossing into --fit must not make an integrated GPU eligible again.

    A mixed iGPU/discrete host can fit a small context on the discrete card but
    require CPU offload at a larger context. The latter still needs an explicit
    Vulkan device pin even when the user did not use the GPU picker.
    """
    backend, gguf = _fit_fallback_backend(
        tmp_path,
        # total=0 is the Vulkan probe's integrated-GPU marker.
        gpu_memory = [(0, 30000, 0), (1, 14000, 16000)],
        vulkan = True,
    )
    backend._select_gpus = lambda *a, **k: (None, True)

    captured = {}

    def _make_fake_popen(cmd, **kwargs):
        if not cmd or str(cmd[0]) != "/fake/llama-server":
            return _REAL_POPEN(cmd, **kwargs)

        class _FakePopen:
            pid = 322

            def __init__(self, cmd, **kwargs):
                captured["cmd"] = list(cmd)

            def poll(self):
                return None

        return _FakePopen(cmd, **kwargs)

    with (
        patch.object(subprocess, "Popen", side_effect = _make_fake_popen),
        patch("utils.hardware.get_parent_visible_gpu_ids", return_value = []),
    ):
        assert backend.load_model(
            gguf_path = str(gguf),
            model_identifier = "test",
        )

    cmd = captured["cmd"]
    assert "--fit" in cmd and cmd[cmd.index("--fit") + 1] == "on"
    assert "--device" in cmd and cmd[cmd.index("--device") + 1] == "Vulkan1"


def test_vulkan_rejects_duplicate_gpu_ids(tmp_path):
    """The CUDA resolver's duplicate check is skipped for the Vulkan path, so the branch
    must reject duplicates itself rather than let set() silently collapse them (#7188)."""
    backend, _gguf = _fit_fallback_backend(
        tmp_path, gpu_memory = [(0, 10000, 16000), (1, 8000, 16000)], vulkan = True
    )
    with patch("utils.hardware.get_parent_visible_gpu_ids", return_value = []):
        with pytest.raises(ValueError, match = "unique and non-negative"):
            backend.assert_requested_gpu_ids_resolvable([0, 0])


def test_empty_probe_rejects_gpu_ids_without_gpu_backend(tmp_path):
    """A non-Vulkan build with an empty probe AND empty parent-visible mask has no GPU
    backend, so the backend must reject gpu_ids rather than pin a non-existent device
    and silently run on CPU (#7188)."""
    from utils.hardware import DeviceType

    backend, _gguf = _fit_fallback_backend(tmp_path, gpu_memory = [])  # empty probe

    with (
        # CUDA host (the mask governs placement) but no mask -> genuinely GPU-less.
        patch("utils.hardware.get_device", return_value = DeviceType.CUDA),
        patch("utils.hardware.get_parent_visible_gpu_ids", return_value = []),
    ):
        with pytest.raises(ValueError, match = "no GPU backend"):
            backend.assert_requested_gpu_ids_resolvable([0])


def test_empty_probe_rejects_gpu_ids_outside_parent_mask(tmp_path):
    """Empty non-Vulkan probe but a set parent-visible mask (real GPU, no telemetry):
    a request outside that mask is a genuine 'no such GPU', so it must still raise
    rather than pin an id the host can't offer (#7188)."""
    from utils.hardware import DeviceType

    backend, _gguf = _fit_fallback_backend(tmp_path, gpu_memory = [])  # empty probe

    with (
        # CUDA host with telemetry down: the mask [0, 1] is real, but 9 is outside it.
        patch("utils.hardware.get_device", return_value = DeviceType.CUDA),
        patch("utils.hardware.get_parent_visible_gpu_ids", return_value = [0, 1]),
    ):
        with pytest.raises(ValueError, match = "do not match any visible GPUs"):
            backend.assert_requested_gpu_ids_resolvable([9])


def test_vulkan_gpu_ids_strips_conflicting_user_device(tmp_path):
    """On Vulkan --device is the only pin. With explicit gpu_ids, a user --device in
    extras must be stripped so it can't override Unsloth's pin (#7188). Unsloth's --device
    survives; unrelated extras pass through."""
    backend, gguf = _fit_fallback_backend(
        tmp_path, gpu_memory = [(0, 10000, 16000), (1, 8000, 16000)], vulkan = True
    )
    backend._select_gpus = lambda *a, **k: ([0], False)

    captured = {}

    def _make_fake_popen(cmd, **kwargs):
        if not cmd or str(cmd[0]) != "/fake/llama-server":
            return _REAL_POPEN(cmd, **kwargs)

        class _FakePopen:
            pid = 999

            def __init__(self, cmd, **kwargs):
                captured["cmd"] = list(cmd)

            def poll(self):
                return None

        return _FakePopen(cmd, **kwargs)

    with (
        patch.object(subprocess, "Popen", side_effect = _make_fake_popen),
        patch("utils.hardware.get_parent_visible_gpu_ids", return_value = [0, 1]),
    ):
        assert backend.load_model(
            gguf_path = str(gguf),
            model_identifier = "test",
            gpu_ids = [0],
            extra_args = ["--device", "Vulkan1", "--top-k", "5"],
        )

    cmd = captured["cmd"]
    # Unsloth pinned Vulkan0 (gpu_indices=[0]); the user's Vulkan1 override is gone.
    assert "Vulkan1" not in cmd
    device_idxs = [i for i, tok in enumerate(cmd) if tok == "--device"]
    assert len(device_idxs) == 1
    assert cmd[device_idxs[0] + 1] == "Vulkan0"
    # Unrelated user extras still pass through.
    assert "--top-k" in cmd and cmd[cmd.index("--top-k") + 1] == "5"


def test_populated_probe_nonmatching_gpu_ids_still_raises(tmp_path):
    """When the probe DID enumerate GPUs but none match the request, the id is genuinely
    absent and the load must still raise (the empty-probe relaxation must not swallow
    a real 'no such GPU' error)."""
    backend, _gguf = _fit_fallback_backend(tmp_path, gpu_memory = [(0, 10000, 16000)])

    with patch("utils.hardware.get_parent_visible_gpu_ids", return_value = [0]):
        with pytest.raises(ValueError, match = "do not match any visible GPUs"):
            backend.assert_requested_gpu_ids_resolvable([9])


@pytest.mark.parametrize(
    "bad_ids, match",
    [
        ([99], "do not match any visible GPUs"),  # out-of-range on a Vulkan host
        ([0, 0], "unique and non-negative"),  # duplicate ids
    ],
)
def test_invalid_gpu_ids_rejected_before_teardown(tmp_path, bad_ids, match):
    """A bad gpu_ids (a typo like [99], or a duplicate) must be rejected by the route's
    resolvability preflight BEFORE the active model is torn down. The merged impl does
    the check in the route (assert_requested_gpu_ids_resolvable) rather than a Phase-0
    block inside load_model, so exercise that helper directly; a running server is never
    reached, hence never killed (#7188)."""
    backend, _gguf = _fit_fallback_backend(tmp_path, gpu_memory = [(0, 10000, 16000)], vulkan = True)
    killed = []
    backend._kill_process = lambda: killed.append(True)

    with patch("utils.hardware.get_parent_visible_gpu_ids", return_value = []):
        with pytest.raises(ValueError, match = match):
            backend.assert_requested_gpu_ids_resolvable(bad_ids)

    assert killed == [], f"active model was torn down before rejecting gpu_ids={bad_ids}"


def test_valid_gpu_ids_pass_resolvable_preflight(tmp_path):
    """The resolvability preflight must NOT over-reject a VALID selection: a good gpu_ids
    passes without raising so the route proceeds to load."""
    backend, _gguf = _fit_fallback_backend(tmp_path, gpu_memory = [(0, 10000, 16000)], vulkan = True)

    with patch("utils.hardware.get_parent_visible_gpu_ids", return_value = []):
        # Does not raise.
        backend.assert_requested_gpu_ids_resolvable([0])


def test_gpu_ids_preserved_on_fit_fallback(tmp_path):
    """When _select_gpus falls back to --fit on, still pin CUDA_VISIBLE_DEVICES."""
    gguf = tmp_path / "model.gguf"
    _write_minimal_gguf(gguf)

    backend = LlamaCppBackend()
    backend._get_gpu_memory = lambda _binary = None: [
        (0, 10000, 16000),
        (1, 8000, 16000),
        (2, 6000, 16000),
    ]
    backend._read_gguf_metadata = lambda _p: None
    backend._can_estimate_kv = lambda: False
    backend._get_gguf_size_bytes = lambda _p: 1024
    backend._mmproj_vram_bytes = lambda _p: 0
    backend._resolve_launch_mmproj_path = lambda **k: None
    backend._apu_ram_shortfall_message = lambda *a, **k: None
    backend._amd_apu_wants_unified_memory = lambda *a, **k: False
    backend._find_llama_server_binary = lambda include_denied = False: "/fake/llama-server"
    # Force a --fit on fallback.
    backend._select_gpus = lambda *a, **k: (None, True)
    backend._wait_for_health = lambda timeout: True
    backend._detect_audio_type_strict = lambda: None
    backend._apply_detected_audio = lambda _d: True

    captured_envs = []

    def _make_fake_popen(cmd, **kwargs):
        if not cmd or str(cmd[0]) != "/fake/llama-server":
            return _REAL_POPEN(cmd, **kwargs)

        class _FakePopen:
            pid = 12345

            def __init__(self, cmd, **kwargs):
                captured_envs.append(kwargs.get("env") or dict(os.environ))

            def poll(self):
                return None

        return _FakePopen(cmd, **kwargs)

    with patch.object(subprocess, "Popen", side_effect = _make_fake_popen):
        backend.load_model(
            gguf_path = str(gguf),
            model_identifier = "test",
            gpu_ids = [1, 2],
        )

    assert captured_envs, "llama-server was not spawned"
    assert captured_envs[-1]["CUDA_VISIBLE_DEVICES"] == "1,2"


@pytest.mark.parametrize("gpu_ids,scrubbed", [([1, 2], True), (None, False)])
def test_gpu_ids_scrubs_inherited_llama_arg_device(tmp_path, monkeypatch, gpu_ids, scrubbed):
    """An explicit gpu_ids pin owns device placement, so an inherited LLAMA_ARG_DEVICE (the
    env form of llama.cpp --device) is scrubbed from the child env -- otherwise it would
    steer offload off the pinned cards (or to 'none' -> CPU) while /load reports a GPU-pinned
    success. Without gpu_ids the operator's LLAMA_ARG_DEVICE inheritance is left intact,
    mirroring the memory/split/tensor env scrubs (backwards compatible) (#7188)."""
    monkeypatch.setenv("LLAMA_ARG_DEVICE", "CUDA3")

    gguf = tmp_path / "model.gguf"
    _write_minimal_gguf(gguf)

    backend = LlamaCppBackend()
    backend._get_gpu_memory = lambda _binary = None: [
        (0, 10000, 16000),
        (1, 8000, 16000),
        (2, 6000, 16000),
    ]
    backend._read_gguf_metadata = lambda _p: None
    backend._can_estimate_kv = lambda: False
    backend._get_gguf_size_bytes = lambda _p: 1024
    backend._mmproj_vram_bytes = lambda _p: 0
    backend._resolve_launch_mmproj_path = lambda **k: None
    backend._apu_ram_shortfall_message = lambda *a, **k: None
    backend._amd_apu_wants_unified_memory = lambda *a, **k: False
    backend._find_llama_server_binary = lambda include_denied = False: "/fake/llama-server"
    backend._select_gpus = lambda *a, **k: ((list(gpu_ids) if gpu_ids else [0]), False)
    backend._wait_for_health = lambda timeout: True
    backend._detect_audio_type_strict = lambda: None
    backend._apply_detected_audio = lambda _d: True

    captured_envs = []

    def _make_fake_popen(cmd, **kwargs):
        if not cmd or str(cmd[0]) != "/fake/llama-server":
            return _REAL_POPEN(cmd, **kwargs)

        class _FakePopen:
            pid = 12345

            def __init__(self, cmd, **kwargs):
                captured_envs.append(kwargs.get("env") or dict(os.environ))

            def poll(self):
                return None

        return _FakePopen(cmd, **kwargs)

    with patch.object(subprocess, "Popen", side_effect = _make_fake_popen):
        backend.load_model(
            gguf_path = str(gguf),
            model_identifier = "test",
            gpu_ids = gpu_ids,
        )

    assert captured_envs, "llama-server was not spawned"
    assert ("LLAMA_ARG_DEVICE" not in captured_envs[-1]) == scrubbed


def test_memory_mode_clears_inherited_mmap_env_vars(tmp_path):
    """An explicit memory_mode must clear stale LLAMA_ARG_* env vars."""
    gguf = tmp_path / "model.gguf"
    _write_minimal_gguf(gguf)

    backend = LlamaCppBackend()
    backend._get_gpu_memory = lambda _binary = None: [(0, 10000, 16000)]
    backend._read_gguf_metadata = lambda _p: None
    backend._can_estimate_kv = lambda: False
    backend._get_gguf_size_bytes = lambda _p: 1024
    backend._mmproj_vram_bytes = lambda _p: 0
    backend._resolve_launch_mmproj_path = lambda **k: None
    backend._apu_ram_shortfall_message = lambda *a, **k: None
    backend._amd_apu_wants_unified_memory = lambda *a, **k: False
    backend._find_llama_server_binary = lambda include_denied = False: "/fake/llama-server"
    backend._select_gpus = lambda *a, **k: ([0], False)
    backend._wait_for_health = lambda timeout: True
    backend._detect_audio_type_strict = lambda: None
    backend._apply_detected_audio = lambda _d: True

    base_env = dict(os.environ)
    base_env.update(
        {
            "LLAMA_ARG_LOAD_MODE": "dio",
            "LLAMA_ARG_MLOCK": "1",
            "LLAMA_ARG_MMAP": "1",
            "LLAMA_ARG_NO_MMAP": "1",
            "LLAMA_ARG_DIO": "1",
        }
    )
    backend._llama_server_env_for_binary = lambda _binary: dict(base_env)

    captured_envs = []

    def _make_fake_popen(cmd, **kwargs):
        if not cmd or str(cmd[0]) != "/fake/llama-server":
            return _REAL_POPEN(cmd, **kwargs)

        class _FakePopen:
            pid = 12345

            def __init__(self, cmd, **kwargs):
                captured_envs.append(kwargs.get("env") or dict(os.environ))

            def poll(self):
                return None

        return _FakePopen(cmd, **kwargs)

    with patch.object(subprocess, "Popen", side_effect = _make_fake_popen):
        backend.load_model(
            gguf_path = str(gguf),
            model_identifier = "test",
            memory_mode = "auto",
        )

    assert captured_envs, "llama-server was not spawned"
    env = captured_envs[-1]
    for var in (
        "LLAMA_ARG_LOAD_MODE",
        "LLAMA_ARG_MLOCK",
        "LLAMA_ARG_MMAP",
        "LLAMA_ARG_NO_MMAP",
        "LLAMA_ARG_DIO",
    ):
        assert var not in env


@pytest.mark.parametrize(
    "mode,user_flag,winning",
    [
        ("resident", "--mmap", "--no-mmap"),
        ("pinned", "--no-mmap", None),
        ("auto", "--mlock", None),
    ],
)
def test_memory_mode_strips_conflicting_extra_args(tmp_path, mode, user_flag, winning):
    """When a placement mode is applied, a conflicting --mmap/--no-mmap/--mlock left
    in extra_args must be stripped so llama.cpp's last-wins parsing can't run a
    placement that disagrees with the stored memory_mode (#7164). auto emits no
    memory flag, so the user's --mlock is dropped rather than pinning the child."""
    gguf = tmp_path / "model.gguf"
    _write_minimal_gguf(gguf)
    backend = _mem_env_backend(gguf)

    captured_cmds = []

    def _make_fake_popen(cmd, **kwargs):
        if not cmd or str(cmd[0]) != "/fake/llama-server":
            return _REAL_POPEN(cmd, **kwargs)

        class _FakePopen:
            pid = 12345

            def __init__(self, cmd, **kwargs):
                captured_cmds.append(list(cmd))

            def poll(self):
                return None

        return _FakePopen(cmd, **kwargs)

    with patch.object(subprocess, "Popen", side_effect = _make_fake_popen):
        backend.load_model(
            gguf_path = str(gguf),
            model_identifier = "test",
            memory_mode = mode,
            extra_args = [user_flag],
        )

    assert captured_cmds, "llama-server was not spawned"
    cmd = captured_cmds[-1]
    # The caller's conflicting flag is gone.
    assert user_flag not in cmd
    # Only Unsloth's own memory flags (if any) remain, and the last mmap/no-mmap
    # flag reflects Unsloth's mode, not the stripped user flag.
    mmap_flags = [a for a in cmd if a in ("--mmap", "--no-mmap")]
    if winning is None:
        assert "--mmap" not in cmd  # user --mmap/--no-mmap fully stripped
    else:
        assert mmap_flags[-1] == winning


def test_vulkan_gpu_ids_used_as_direct_ordinals_not_remapped(tmp_path):
    """gpu_ids are Vulkan device ordinals matched directly against the Vulkan probe,
    NEVER remapped through the CUDA/HIP parent mask. Vulkan enumerates independently of
    CUDA_VISIBLE_DEVICES (its order can even reverse the CUDA order), so a remap would
    pin the wrong device. A CUDA mask of [2, 3] must not shift the requested ordinal
    (#7188)."""
    gguf = tmp_path / "model.gguf"
    _write_minimal_gguf(gguf)

    backend = LlamaCppBackend()
    backend._is_vulkan_backend = lambda _binary = None: True
    # Vulkan reports two devices at ordinals 0 and 1.
    backend._get_gpu_memory = lambda _binary = None: [(0, 10000, 16000), (1, 9000, 16000)]
    backend._get_gpu_free_memory = lambda _binary = None: [(0, 10000), (1, 9000)]
    backend._read_gguf_metadata = lambda _p: None
    backend._can_estimate_kv = lambda: False
    backend._get_gguf_size_bytes = lambda _p: 1024
    backend._mmproj_vram_bytes = lambda _p: 0
    backend._resolve_launch_mmproj_path = lambda **k: None
    backend._apu_ram_shortfall_message = lambda *a, **k: None
    backend._amd_apu_wants_unified_memory = lambda *a, **k: False
    backend._find_llama_server_binary = lambda include_denied = False: "/fake/llama-server"
    # Select the single candidate that survives the gpu_ids filter (its Vulkan ordinal).
    backend._select_gpus = lambda requested_total, gpus, **k: ([gpus[0][0]], False)
    backend._wait_for_health = lambda timeout: True
    backend._detect_audio_type_strict = lambda: None
    backend._apply_detected_audio = lambda _d: True

    captured_cmds = []

    def _make_fake_popen(cmd, **kwargs):
        if not cmd or str(cmd[0]) != "/fake/llama-server":
            return _REAL_POPEN(cmd, **kwargs)

        class _FakePopen:
            pid = 12345

            def __init__(self, cmd, **kwargs):
                captured_cmds.append(list(cmd))

            def poll(self):
                return None

        return _FakePopen(cmd, **kwargs)

    with (
        patch.object(
            subprocess,
            "Popen",
            side_effect = _make_fake_popen,
        ),
        # A CUDA/HIP mask of [2, 3] must NOT remap the requested Vulkan ordinal.
        patch(
            "utils.hardware.get_parent_visible_gpu_ids",
            return_value = [2, 3],
        ),
    ):
        backend.load_model(
            gguf_path = str(gguf),
            model_identifier = "test",
            gpu_ids = [1],
        )

    assert captured_cmds, "llama-server was not spawned"
    cmd = captured_cmds[-1]
    assert "--device" in cmd
    # gpu_ids=[1] pins Vulkan ordinal 1 directly, not remapped to Vulkan0 via the mask.
    assert cmd[cmd.index("--device") + 1] == "Vulkan1"


def test_has_gpu_backend_accepts_parent_mask_when_probe_empty():
    """A real GPU with telemetry down (empty probe) but a numeric parent-visible mask
    must count as a backend, so /load does not 400 a selection assert_requested_gpu_ids_
    resolvable would accept via the mask fallback. No probe AND no mask is GPU-less (#7188)."""
    backend = LlamaCppBackend()
    backend._find_llama_server_binary = lambda include_denied = False: "/fake/llama-server"
    backend._get_gpu_memory = lambda _binary = None: []  # telemetry unavailable
    with patch("utils.hardware.get_parent_visible_gpu_ids", return_value = [0, 1]):
        assert backend.has_gpu_backend() is True
    with patch("utils.hardware.get_parent_visible_gpu_ids", return_value = []):
        assert backend.has_gpu_backend() is False


def test_partial_gpu_ids_match_is_rejected():
    """A partial match such as [0, 99] against a probe of [0] must be rejected, not
    silently narrowed to [0] (which would place the model on fewer GPUs than the caller
    asked for). Every requested id must be present, not just one (#7188)."""
    probe = [(0, 10000, 16000)]
    for is_vulkan in (True, False):
        with pytest.raises(ValueError, match = "do not match any visible GPUs"):
            LlamaCppBackend._assert_gpu_ids_resolvable([0, 99], probe, is_vulkan)
        # A full match still passes.
        LlamaCppBackend._assert_gpu_ids_resolvable([0], probe, is_vulkan)


def test_duplicate_or_negative_gpu_ids_rejected_on_all_backends():
    """Duplicate/negative ids are invalid on every backend, not just Vulkan: a non-Vulkan
    populated probe must not collapse [0, 0] into {0} and pin one GPU while recording the
    duplicate (a torch-less CUDA host defers here, skipping the CUDA resolver) (#7188)."""
    probe = [(0, 10000, 16000), (1, 8000, 16000)]
    for is_vulkan in (True, False):
        for bad in ([0, 0], [-1], [0, -1]):
            with pytest.raises(ValueError, match = "unique and non-negative"):
                LlamaCppBackend._assert_gpu_ids_resolvable(bad, probe, is_vulkan)
        # A valid unique selection still passes.
        LlamaCppBackend._assert_gpu_ids_resolvable([0, 1], probe, is_vulkan)


def test_cpu_only_build_rejects_gpu_ids():
    """A CPU-only llama.cpp build ignores CUDA_VISIBLE_DEVICES, so an explicit pin can't be
    honored: reject it before teardown even with a populated nvidia-smi probe on a CUDA host,
    instead of reporting a GPU-pinned load that silently runs on CPU (#7188)."""
    probe = [(0, 10000, 16000)]
    with pytest.raises(ValueError, match = "CPU-only build"):
        LlamaCppBackend._assert_gpu_ids_resolvable([0], probe, False, backend_lacks_gpu_lib = True)
    # With a GPU backend lib present the same pin is accepted.
    LlamaCppBackend._assert_gpu_ids_resolvable([0], probe, False, backend_lacks_gpu_lib = False)


def test_backend_lacks_gpu_lib_detection(tmp_path):
    """_backend_lacks_gpu_lib is True ONLY for a clear CPU-only split-lib layout (a
    ggml-cpu/base lib with no gpu sibling); a gpu lib, or an unrecognized/static layout
    with no ggml libs, returns False so a valid custom GPU build is never falsely rejected
    (#7188)."""
    ext = "dll" if sys.platform == "win32" else "so"
    pre = "" if sys.platform == "win32" else "lib"

    def _lib_dir_with(*names):
        d = tmp_path / ("libs_" + ("_".join(names) or "empty"))
        d.mkdir()
        for nm in names:
            (d / f"{pre}ggml-{nm}.{ext}").write_bytes(b"x")
        return d

    binary = str(tmp_path / "llama-server")
    (tmp_path / "llama-server").write_bytes(b"x")

    # CPU-only split layout -> True.
    with patch("core.inference.llama_cpp._llama_lib_dir", return_value = _lib_dir_with("cpu")):
        assert LlamaCppBackend._backend_lacks_gpu_lib(binary) is True
    # Any GPU ggml lib present -> False (pin can be honored).
    for gpu in ("cuda", "hip", "vulkan"):
        with patch(
            "core.inference.llama_cpp._llama_lib_dir", return_value = _lib_dir_with("cpu", gpu)
        ):
            assert LlamaCppBackend._backend_lacks_gpu_lib(binary) is False
    # No ggml libs at all (static / unrecognized) -> False (never falsely reject).
    with patch("core.inference.llama_cpp._llama_lib_dir", return_value = _lib_dir_with()):
        assert LlamaCppBackend._backend_lacks_gpu_lib(binary) is False

    # Versioned sonames (e.g. libggml-cuda.so.0) are matched too: a versioned GPU lib
    # next to an unversioned CPU lib is a real GPU build -> False (#7188).
    for gpu in ("cuda", "hip", "vulkan"):
        d = tmp_path / f"libsv_cpu_{gpu}"
        d.mkdir()
        (d / f"{pre}ggml-cpu.{ext}").write_bytes(b"x")
        (d / f"{pre}ggml-{gpu}.{ext}.0").write_bytes(b"x")
        with patch("core.inference.llama_cpp._llama_lib_dir", return_value = d):
            assert LlamaCppBackend._backend_lacks_gpu_lib(binary) is False
    # A versioned CPU-only lib is still recognized as CPU-only -> True.
    d = tmp_path / "libsv_cpu_only"
    d.mkdir()
    (d / f"{pre}ggml-cpu.{ext}.0").write_bytes(b"x")
    with patch("core.inference.llama_cpp._llama_lib_dir", return_value = d):
        assert LlamaCppBackend._backend_lacks_gpu_lib(binary) is True


def test_is_vulkan_backend_matches_versioned_soname(tmp_path):
    """_is_vulkan_backend matches versioned Vulkan sonames (libggml-vulkan.so.0) too, so a
    distro/split-lib Vulkan install without the dev-only unversioned symlink is still detected
    as Vulkan; otherwise the route treats Vulkan ordinals as CUDA ids and never emits the
    --device Vulkan<i> pin (#7188). Shares the matcher with _backend_lacks_gpu_lib."""
    ext = "dll" if sys.platform == "win32" else "so"
    pre = "" if sys.platform == "win32" else "lib"
    binary = str(tmp_path / "llama-server")
    (tmp_path / "llama-server").write_bytes(b"x")
    counter = {"n": 0}

    def _dir(*files):
        counter["n"] += 1
        d = tmp_path / f"vkdir_{counter['n']}"
        d.mkdir()
        for f in files:
            (d / f).write_bytes(b"x")
        return d

    # Versioned-only Vulkan lib (no unversioned symlink) -> detected as Vulkan.
    with patch(
        "core.inference.llama_cpp._llama_lib_dir", return_value = _dir(f"{pre}ggml-vulkan.{ext}.0")
    ):
        assert LlamaCppBackend._is_vulkan_backend(binary) is True
    # Unversioned Vulkan lib -> still detected (regression).
    with patch(
        "core.inference.llama_cpp._llama_lib_dir", return_value = _dir(f"{pre}ggml-vulkan.{ext}")
    ):
        assert LlamaCppBackend._is_vulkan_backend(binary) is True
    # A CUDA/HIP sibling (versioned or not) means a multi-backend build -> defer to that backend.
    for sib in (f"{pre}ggml-cuda.{ext}", f"{pre}ggml-hip.{ext}.0"):
        with patch(
            "core.inference.llama_cpp._llama_lib_dir",
            return_value = _dir(f"{pre}ggml-vulkan.{ext}.0", sib),
        ):
            assert LlamaCppBackend._is_vulkan_backend(binary) is False
    # No Vulkan lib at all -> not a Vulkan build.
    with patch(
        "core.inference.llama_cpp._llama_lib_dir", return_value = _dir(f"{pre}ggml-cpu.{ext}")
    ):
        assert LlamaCppBackend._is_vulkan_backend(binary) is False


def test_empty_non_cuda_probe_rejects_gpu_ids_even_with_stray_mask():
    """On a Metal/SYCL/CPU (non-CUDA) backend the launcher's CUDA_VISIBLE_DEVICES is
    ignored (SYCL keys off ONEAPI_DEVICE_SELECTOR, Metal off none), so an empty probe
    with a stray parent-visible mask must NOT accept gpu_ids -- the pin would be silently
    dropped onto the default device. A CUDA host with the same empty probe + mask is the
    real telemetry-down case and still falls through (#7188)."""
    from utils.hardware import DeviceType
    with patch("utils.hardware.get_parent_visible_gpu_ids", return_value = [0]):
        # A stray CUDA mask on a non-CUDA host must not be trusted to honor the pin.
        for dev in (DeviceType.MLX, DeviceType.XPU, DeviceType.CPU):
            with patch("utils.hardware.get_device", return_value = dev):
                with pytest.raises(ValueError, match = "does not support explicit GPU selection"):
                    LlamaCppBackend._assert_gpu_ids_resolvable([0], [], False)
        # CUDA host (ROCm reports CUDA too): empty probe + valid mask is telemetry-down,
        # so the selection is accepted for the loader to pin via CUDA_VISIBLE_DEVICES.
        with patch("utils.hardware.get_device", return_value = DeviceType.CUDA):
            LlamaCppBackend._assert_gpu_ids_resolvable([0], [], False)


def test_explicit_gpu_ids_strips_stored_device_extra_args(tmp_path):
    """With explicit gpu_ids a user --device is dropped from BOTH the command and the
    PERSISTED extras, so a later same-model reload that inherits these (after gpu_ids is
    cleared) can't resurrect the dropped --device and override auto placement (#7188)."""
    backend, gguf = _fit_fallback_backend(tmp_path, gpu_memory = [(0, 10000, 16000)], vulkan = True)
    backend._select_gpus = lambda requested_total, gpus, **k: ([gpus[0][0]], False)

    def _make_fake_popen(cmd, **kwargs):
        if not cmd or str(cmd[0]) != "/fake/llama-server":
            return _REAL_POPEN(cmd, **kwargs)

        class _FakePopen:
            pid = 12345

            def __init__(self, cmd, **kwargs):
                pass

            def poll(self):
                return None

        return _FakePopen(cmd, **kwargs)

    with (
        patch.object(subprocess, "Popen", side_effect = _make_fake_popen),
        patch("utils.hardware.get_parent_visible_gpu_ids", return_value = []),
    ):
        backend.load_model(
            gguf_path = str(gguf),
            model_identifier = "test",
            gpu_ids = [0],
            extra_args = ["--device", "Vulkan3", "--top-k", "5"],
        )

    stored = backend._extra_args or []
    assert "--device" not in stored and "Vulkan3" not in stored
    assert "--top-k" in stored  # unrelated extras are preserved


def test_memory_mode_auto_matches_none_in_target_state():
    """An explicit 'auto' request should not reload a load that omitted the field."""
    backend = _loaded_backend(_memory_mode = None)
    kwargs = _base_target_state_kwargs(backend)
    kwargs["memory_mode"] = "auto"
    assert backend._already_in_target_state(**kwargs) is True


def test_explicit_auto_reloads_when_child_inherited_mem_env():
    """When the live child was launched with no mode but inherited operator
    LLAMA_ARG_* placement flags, an explicit 'auto' request must reload so the
    scrub runs -- otherwise it dedups to already-loaded and stays mlocked (#7164)."""
    backend = _loaded_backend(_memory_mode = None, _launched_with_inherited_mem_env = True)
    kwargs = _base_target_state_kwargs(backend)
    kwargs["memory_mode"] = "auto"
    assert backend._already_in_target_state(**kwargs) is False


def test_omitted_mode_does_not_reload_child_with_inherited_mem_env():
    """A request that also omits the mode (memory_mode=None) does NOT reload the
    inherited-env child: omitting keeps the operator env, so there's nothing to
    scrub and no spurious reload (which would be rejected during training)."""
    backend = _loaded_backend(_memory_mode = None, _launched_with_inherited_mem_env = True)
    kwargs = _base_target_state_kwargs(backend)
    kwargs["memory_mode"] = None
    assert backend._already_in_target_state(**kwargs) is True


def test_explicit_auto_matches_scrubbed_child():
    """Once the child was launched clean (no inherited env), an explicit 'auto'
    re-Apply dedups to already-loaded -- no needless reload."""
    backend = _loaded_backend(_memory_mode = None, _launched_with_inherited_mem_env = False)
    kwargs = _base_target_state_kwargs(backend)
    kwargs["memory_mode"] = "auto"
    assert backend._already_in_target_state(**kwargs) is True


def test_memory_mode_pinned_does_not_match_none():
    backend = _loaded_backend(_memory_mode = None)
    kwargs = _base_target_state_kwargs(backend)
    kwargs["memory_mode"] = "pinned"
    assert backend._already_in_target_state(**kwargs) is False


def test_load_response_and_status_round_trip_placement_fields():
    """gpu_ids and gguf_memory_mode are accepted by the response schemas so
    status-hydrated requests can preserve explicit placement settings."""
    from models.inference import InferenceStatusResponse, LoadResponse

    load_resp = LoadResponse(
        status = "loaded",
        model = "m",
        display_name = "m",
        is_gguf = True,
        inference = {},
        gpu_ids = [0, 1],
        gguf_memory_mode = "resident",
    )
    assert load_resp.gpu_ids == [0, 1]
    assert load_resp.gguf_memory_mode == "resident"

    status_resp = InferenceStatusResponse(
        is_gguf = True,
        gpu_ids = [0, 1],
        gguf_memory_mode = "pinned",
    )
    assert status_resp.gpu_ids == [0, 1]
    assert status_resp.gguf_memory_mode == "pinned"


@pytest.mark.parametrize(
    "mode,expected_requested,expected_canonical",
    [
        ("auto", "auto", None),
        ("AUTO", "auto", None),
        ("pinned", "pinned", "pinned"),
        (None, None, None),
    ],
)
def test_requested_memory_mode_preserves_explicit_auto(
    tmp_path, mode, expected_requested, expected_canonical
):
    """The response echoes requested_memory_mode, not the canonical placement. An explicit
    "auto" must survive as "auto" (not collapse to null) so the UI can restore it and a
    later reload re-runs the inherited-env scrub instead of letting LLAMA_ARG_MLOCK creep
    back; canonical _memory_mode still maps "auto" -> None for placement (#7188)."""
    gguf = tmp_path / "model.gguf"
    _write_minimal_gguf(gguf)
    backend = _mem_env_backend(gguf)

    with patch.object(subprocess, "Popen"):
        assert backend.load_model(gguf_path = str(gguf), model_identifier = "t", memory_mode = mode)
    assert backend.requested_memory_mode == expected_requested
    assert backend.memory_mode == expected_canonical


# ── inherited LLAMA_ARG_* mmap/mlock env is scrubbed when a mode is set ───────


def _mem_env_backend(gguf):
    backend = LlamaCppBackend()
    backend._get_gpu_memory = lambda _binary = None: [(0, 10000, 16000)]
    backend._read_gguf_metadata = lambda _p: None
    backend._can_estimate_kv = lambda: False
    backend._get_gguf_size_bytes = lambda _p: 1024
    backend._mmproj_vram_bytes = lambda _p: 0
    backend._resolve_launch_mmproj_path = lambda **k: None
    backend._apu_ram_shortfall_message = lambda *a, **k: None
    backend._amd_apu_wants_unified_memory = lambda *a, **k: False
    backend._find_llama_server_binary = lambda include_denied = False: "/fake/llama-server"
    backend._select_gpus = lambda *a, **k: ([0], False)
    backend._wait_for_health = lambda timeout: True
    backend._detect_audio_type_strict = lambda: None
    backend._apply_detected_audio = lambda _d: True
    return backend


@pytest.mark.parametrize(
    "mode,scrubbed",
    [("auto", True), ("pinned", True), ("resident", True), (None, False)],
)
def test_memory_mode_scrubs_inherited_mmap_env(tmp_path, monkeypatch, mode, scrubbed):
    """An explicit memory_mode strips inherited LLAMA_ARG_MLOCK/NO_MMAP/MMAP so
    llama-server can't run a placement Unsloth didn't select (#7164). memory_mode=None
    leaves operator env untouched (backwards compatible); the reload-dedup handles a
    later explicit 'auto' (see the target-state tests below)."""
    monkeypatch.setenv("LLAMA_ARG_MLOCK", "1")
    monkeypatch.setenv("LLAMA_ARG_NO_MMAP", "1")
    monkeypatch.setenv("LLAMA_ARG_MMAP", "true")
    monkeypatch.setenv("LLAMA_ARG_LOAD_MODE", "dio")
    monkeypatch.setenv("LLAMA_ARG_DIO", "1")

    gguf = tmp_path / "model.gguf"
    _write_minimal_gguf(gguf)
    backend = _mem_env_backend(gguf)

    captured_envs = []

    def _make_fake_popen(cmd, **kwargs):
        if not cmd or str(cmd[0]) != "/fake/llama-server":
            return _REAL_POPEN(cmd, **kwargs)

        class _FakePopen:
            pid = 12345

            def __init__(self, cmd, **kwargs):
                captured_envs.append(kwargs.get("env") or {})

            def poll(self):
                return None

        return _FakePopen(cmd, **kwargs)

    with patch.object(subprocess, "Popen", side_effect = _make_fake_popen):
        backend.load_model(
            gguf_path = str(gguf),
            model_identifier = "test",
            memory_mode = mode,
        )

    assert captured_envs, "llama-server was not spawned"
    env = captured_envs[-1]
    for var in (
        "LLAMA_ARG_LOAD_MODE",
        "LLAMA_ARG_MLOCK",
        "LLAMA_ARG_NO_MMAP",
        "LLAMA_ARG_MMAP",
        "LLAMA_ARG_DIO",
    ):
        assert (var not in env) == scrubbed


# ── diffusion GGUFs clear host-residency memory-mode state ───────────────────
# The route rejects known DiffusionGemma Vulkan pins and explicit host-memory modes
# before teardown. load_model retains a post-download Vulkan guard for a remote
# uncached model whose architecture was not known to the route (#7239). A successful
# diffusion load also clears any host-residency memory_mode carried from a prior
# llama-server load because the diffusion runner has no --mlock/--no-mmap support.


def test_remote_diffusion_load_rejects_vulkan_ordinal_after_download(tmp_path):
    """A renamed remote DiffusionGemma may be unclassifiable until its downloaded
    GGUF header is read. Never pass its Vulkan ordinal to the CUDA diffusion runner."""
    gguf = tmp_path / "renamed.gguf"
    _write_minimal_gguf(gguf, arch = "diffusion-gemma")

    backend = LlamaCppBackend()
    backend._find_llama_server_binary = lambda include_denied = False: "/fake/llama-server"
    backend._is_vulkan_backend = lambda _binary = None: True
    backend._download_gguf = lambda **_kwargs: str(gguf)
    backend._read_gguf_metadata = lambda _path: setattr(backend, "_is_diffusion", True)
    backend._start_diffusion_server = lambda **_kwargs: pytest.fail(
        "Vulkan ordinal reached the CUDA diffusion runner"
    )

    with pytest.raises(ValueError, match = "no defined mapping"):
        backend.load_model(
            hf_repo = "renamed/model",
            hf_variant = "Q4_K_M",
            model_identifier = "renamed/model",
            speculative_type = "off",
            gpu_ids = [1],
        )


def test_confirmed_diffusion_allows_physical_gpu_id_on_vulkan_build(tmp_path):
    """The route validates known DiffusionGemma pins as CUDA physical IDs. A
    Vulkan llama.cpp build does not change the diffusion runner's index space."""
    gguf = tmp_path / "diffusion.gguf"
    _write_minimal_gguf(gguf, arch = "diffusion-gemma")

    backend = LlamaCppBackend()
    backend._find_llama_server_binary = lambda include_denied = False: "/fake/llama-server"
    backend._is_vulkan_backend = lambda _binary = None: True
    backend._read_gguf_metadata = lambda _path: setattr(backend, "_is_diffusion", True)
    captured = {}
    backend._start_diffusion_server = lambda **kwargs: captured.update(kwargs) or True

    assert backend.load_model(
        gguf_path = str(gguf),
        model_identifier = "diffusion/model",
        speculative_type = "off",
        gpu_ids = [1],
        gpu_ids_are_vulkan_ordinals = False,
    )
    assert captured["gpu_ids"] == [1]


def test_remote_diffusion_rejects_explicit_memory_mode_after_download(tmp_path):
    gguf = tmp_path / "renamed.gguf"
    _write_minimal_gguf(gguf, arch = "diffusion-gemma")

    backend = LlamaCppBackend()
    backend._find_llama_server_binary = lambda include_denied = False: "/fake/llama-server"
    backend._is_vulkan_backend = lambda _binary = None: False
    backend._download_gguf = lambda **_kwargs: str(gguf)
    backend._read_gguf_metadata = lambda _path: setattr(backend, "_is_diffusion", True)
    backend._start_diffusion_server = lambda **_kwargs: pytest.fail(
        "unsupported memory mode reached the diffusion runner"
    )

    with pytest.raises(ValueError, match = "host-memory modes are not supported"):
        backend.load_model(
            hf_repo = "renamed/model",
            hf_variant = "Q4_K_M",
            model_identifier = "renamed/model",
            speculative_type = "off",
            memory_mode = "resident",
        )


@pytest.mark.parametrize("mode", [None, "auto", "AUTO", ""])
def test_diffusion_load_clears_stale_memory_mode(tmp_path, mode):
    """auto/blank/None is the allowed no-op default for diffusion. A successful load
    must clear any _memory_mode left by a prior llama-server load so reload-dedup
    doesn't force a needless kill+restart of the diffusion server. (The single-device
    gpu_ids collapse is recorded inside _start_diffusion_server, exercised by #6414's
    picker tests; this stub replaces the runner, so only the memory-mode clear, which
    load_model performs itself, is asserted here.)"""
    gguf = tmp_path / "diffusion.gguf"
    _write_minimal_gguf(gguf, arch = "diffusion-gemma")

    backend = LlamaCppBackend()
    backend._read_gguf_metadata = lambda _p: setattr(backend, "_is_diffusion", True)
    backend._find_llama_server_binary = lambda include_denied = False: "/fake/llama-server"
    backend._is_vulkan_backend = lambda _binary = None: False
    backend._start_diffusion_server = lambda **kw: True

    # Simulate leftover placement state from a previous llama-server GGUF load.
    backend._memory_mode = "resident"
    backend._requested_memory_mode = "resident"
    backend._launched_with_inherited_mem_env = True

    assert (
        backend.load_model(
            gguf_path = str(gguf),
            model_identifier = "d",
            memory_mode = mode,
        )
        is True
    )
    assert backend._memory_mode is None
    assert backend._requested_memory_mode is None
    assert backend._launched_with_inherited_mem_env is False


def test_local_chat_gguf_in_diffusion_path_not_prekilled(tmp_path):
    """A local chat GGUF whose path contains "diffusion" is NOT a diffusion model: the
    header (read via _classify_diffusion_gguf) decides, not the path string, so explicit
    gpu_ids on such a local GGUF must load normally, not be rejected on the path (#7188)."""
    backend, gguf = _fit_fallback_backend(tmp_path, gpu_memory = [(0, 10000, 16000)])
    backend._select_gpus = lambda *a, **k: ([0], False)

    with patch.object(subprocess, "Popen"):
        assert (
            backend.load_model(
                gguf_path = str(gguf),
                model_identifier = "/models/diffusion/chat.gguf",
                gpu_ids = [0],
            )
            is True
        )
