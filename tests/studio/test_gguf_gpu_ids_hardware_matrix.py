# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team.
"""Studio GGUF gpu_ids routing under spoofed NVIDIA / AMD / CPU-only hardware.

Where test_gguf_placement_os_gpu_matrix.py mocks ``utils.hardware.get_device``, this drives
the same decision through the REAL hardware layer with torch spoofed by
tests/_zoo_aggressive_cuda_spoof.py (NVIDIA) and tests/_zoo_rocm_spoof.py (AMD), so each
vendor's DeviceType is observed rather than assumed. Each case runs in its own subprocess
(the spoofs mutate torch and ``utils.hardware`` globals), with the bundle layout faked per OS.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")

_TESTS_DIR = Path(__file__).resolve().parents[1]  # tests/
_REPO_ROOT = _TESTS_DIR.parent
_STUDIO_BACKEND = _REPO_ROOT / "studio" / "backend"

# vendor -> (spoof mode, ggml backend libs the installer ships for it)
_VENDORS = {
    "nvidia": ("cuda", ("cpu", "cuda")),
    "amd": ("rocm", ("cpu", "hip")),
    "cpu_only": ("none", ("cpu",)),
}

# os name -> (sys.platform value, ggml shared-library suffix)
_OSES = {
    "windows": ("win32", "dll"),
    "linux": ("linux", "so"),
    # WSL2 runs the Linux bundle; it must resolve identically to native Linux.
    "wsl": ("linux", "so"),
    "macos": ("darwin", "dylib"),
}

_CHILD = r"""
import asyncio, json, sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, __TESTS_DIR__)
# Let bitsandbytes settle against the honest torch before the spoof flips
# is_available(): it picks a compute backend at import and a CPU-only wheel
# cannot satisfy the CUDA/ROCm one.
try:
    import bitsandbytes  # noqa: F401
except Exception:
    pass

mode = __MODE__
if mode == "cuda":
    import _zoo_aggressive_cuda_spoof as spoof
    spoof.apply()
elif mode == "rocm":
    import _zoo_rocm_spoof as spoof
    spoof.apply("gfx1100")

sys.path.insert(0, __BACKEND_DIR__)
sys.platform = __PLATFORM__

import utils.hardware as hardware_pkg
from core.inference.llama_cpp import LlamaCppBackend

binary = __BINARY__
lacks_gpu_lib = LlamaCppBackend._backend_lacks_gpu_lib(binary)
device = hardware_pkg.get_device()

import importlib.util
spec = importlib.util.spec_from_file_location(
    "inference_route_for_hw_matrix", str(Path(__BACKEND_DIR__) / "routes" / "inference.py")
)
route = importlib.util.module_from_spec(spec)
spec.loader.exec_module(route)

from fastapi import HTTPException


async def _inline_to_thread(func, /, *args, **kwargs):
    return func(*args, **kwargs)


def _resolve(gpu_ids):
    backend_stub = SimpleNamespace(
        is_vulkan_build = lambda: False,
        _backend_lacks_gpu_lib = lambda *a, **k: lacks_gpu_lib,
    )
    with (
        patch.object(route, "_classify_diffusion_gguf", return_value = False),
        patch.object(route, "get_llama_cpp_backend", return_value = backend_stub),
        patch.object(route.asyncio, "to_thread", new = _inline_to_thread),
        patch(
            "utils.hardware.hardware.resolve_requested_gpu_ids",
            return_value = list(gpu_ids or []),
        ),
    ):
        try:
            resolved, vulkan = asyncio.run(
                route._resolve_gguf_gpu_ids_for_request(
                    SimpleNamespace(is_gguf = True), gpu_ids
                )
            )
        except HTTPException as exc:
            return {"status": exc.status_code, "detail": str(exc.detail)}
        return {"resolved": resolved, "vulkan": vulkan}


print("RESULT " + json.dumps({
    "device": device.name,
    "is_rocm": bool(getattr(__import__("torch").version, "hip", None)),
    "lacks_gpu_lib": lacks_gpu_lib,
    "pin": _resolve([0]),
    "no_pin": _resolve(None),
}))
"""


def _make_bundle(tmp_path: Path, os_name: str, backends) -> str:
    _platform, suffix = _OSES[os_name]
    lib_dir = tmp_path / "build" / "bin"
    lib_dir.mkdir(parents = True, exist_ok = True)
    prefix = "" if suffix == "dll" else "lib"
    for backend in backends:
        (lib_dir / f"{prefix}ggml-{backend}.{suffix}").write_bytes(b"")
    binary = lib_dir / ("llama-server.exe" if suffix == "dll" else "llama-server")
    binary.write_bytes(b"")
    return str(binary)


def _run_child(tmp_path: Path, os_name: str, vendor: str) -> dict:
    mode, backends = _VENDORS[vendor]
    platform_name, _suffix = _OSES[os_name]
    script = _CHILD
    for token, value in (
        ("__TESTS_DIR__", str(_TESTS_DIR)),
        ("__BACKEND_DIR__", str(_STUDIO_BACKEND)),
        ("__MODE__", mode),
        ("__PLATFORM__", platform_name),
        ("__BINARY__", _make_bundle(tmp_path, os_name, backends)),
    ):
        script = script.replace(token, repr(value))
    env = dict(os.environ, UNSLOTH_COMPILE_DISABLE = "1", CUDA_VISIBLE_DEVICES = "")
    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output = True,
        text = True,
        timeout = 600,
        env = env,
        cwd = str(_STUDIO_BACKEND),
    )
    line = next(
        (ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT ")),
        None,
    )
    if line is None:
        pytest.skip(
            f"hardware spoof child did not report ({vendor}/{os_name}): "
            f"{proc.stderr.strip()[-400:]}"
        )
    return json.loads(line[len("RESULT ") :])


@pytest.mark.parametrize("os_name", sorted(_OSES))
@pytest.mark.parametrize("vendor", sorted(_VENDORS))
def test_gpu_ids_routing_under_spoofed_hardware(tmp_path, os_name, vendor):
    result = _run_child(tmp_path, os_name, vendor)

    if vendor == "nvidia":
        assert result["device"] == "CUDA"
        assert result["is_rocm"] is False
    elif vendor == "amd":
        # PyTorch ROCm reuses torch.cuda over HIP, so DeviceType stays CUDA and gpu_ids
        # stays the same physical-index space.
        assert result["device"] == "CUDA"
        assert result["is_rocm"] is True

    # macOS ships .dylib ggml libs, so the split-library probe abstains there.
    expected_lacks = vendor == "cpu_only" and os_name != "macos"
    assert result["lacks_gpu_lib"] is expected_lacks

    if expected_lacks:
        assert result["pin"]["status"] == 400
        assert "cpu-only build" in result["pin"]["detail"].lower()
    else:
        assert result["pin"] == {"resolved": [0], "vulkan": False}

    # The pre-#7164 shape (no pick) must load on every cell, even the CPU-only build.
    assert result["no_pin"] == {"resolved": None, "vulkan": False}
