# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""OS x GPU matrix for the GGUF placement round-trip (#7210).

Spoofs [Windows, Linux, WSL, macOS] x [NVIDIA, AMD, CPU-only] plus the Vulkan bundle, with
no GPU required, over: the bundle layout ``_backend_lacks_gpu_lib`` reads, ``/api/system``'s
``gguf_gpu_ids_supported`` picker gate, and ``/load``'s ``_resolve_gguf_gpu_ids_for_request``,
which must agree with it. Also pins the pre-existing paths this must not disturb: no gpu_ids,
a single-GPU host, a GPU-less host, and a CPU-only build.
"""

from __future__ import annotations

import asyncio
import importlib.util
import logging
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from fastapi import HTTPException

_BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

import utils.hardware as hardware_pkg  # noqa: E402
from core.inference.llama_cpp import LlamaCppBackend  # noqa: E402


async def _inline_to_thread(func, /, *args, **kwargs):
    return func(*args, **kwargs)


def _load_module(name: str, relative_path: str):
    spec = importlib.util.spec_from_file_location(name, _BACKEND_ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# name -> (sys.platform value, shared-library suffix used by the llama.cpp bundle)
OS_PROFILES = {
    "windows": ("win32", "dll"),
    # WSL2 runs the Linux bundle, so every decision below must match native Linux exactly.
    "wsl": ("linux", "so"),
    "linux": ("linux", "so"),
    "macos": ("darwin", "dylib"),
}

# name -> (ggml backend libs present, DeviceType, gpu_ids expected to work)
GPU_PROFILES = {
    # NVIDIA: CUDA physical ids, the original gpu_ids namespace.
    "nvidia": (("cpu", "cuda"), "CUDA", True),
    # AMD/ROCm: torch-ROCm reuses torch.cuda, so DeviceType stays CUDA; the HIP lib proves the pin.
    "amd": (("cpu", "hip"), "CUDA", True),
    # CPU-only llama.cpp: a pin would be silently ignored, so it must 400.
    "cpu_only": (("cpu",), "CPU", False),
}


def _make_bundle(tmp_path: Path, os_name: str, backends) -> str:
    """Lay out a llama.cpp bundle the way the installer does for ``os_name``."""
    _platform, suffix = OS_PROFILES[os_name]
    lib_dir = tmp_path / os_name / "build" / "bin"
    lib_dir.mkdir(parents = True, exist_ok = True)
    prefix = "" if suffix == "dll" else "lib"
    for backend in backends:
        (lib_dir / f"ggml-{backend}.{suffix}").write_bytes(b"")
        (lib_dir / f"{prefix}ggml-{backend}.{suffix}").write_bytes(b"")
    binary = lib_dir / ("llama-server.exe" if suffix == "dll" else "llama-server")
    binary.write_bytes(b"")
    return str(binary)


def _gguf_config():
    return SimpleNamespace(is_gguf = True)


def _resolve(
    route,
    gpu_ids,
    *,
    binary,
    vulkan,
    device,
    resolved = None,
    diffusion = False,
):
    """Drive _resolve_gguf_gpu_ids_for_request with the host fully spoofed."""
    backend = SimpleNamespace(
        is_vulkan_build = lambda: vulkan,
        _backend_lacks_gpu_lib = lambda *a, **k: LlamaCppBackend._backend_lacks_gpu_lib(binary),
    )
    with (
        patch.object(route, "_classify_diffusion_gguf", return_value = diffusion),
        patch.object(route, "get_llama_cpp_backend", return_value = backend),
        patch.object(route.asyncio, "to_thread", new = _inline_to_thread),
        patch.object(
            hardware_pkg, "get_device", return_value = getattr(hardware_pkg.DeviceType, device)
        ),
        patch(
            "utils.hardware.hardware.resolve_requested_gpu_ids",
            return_value = list(resolved if resolved is not None else (gpu_ids or [])),
        ),
        patch.object(
            LlamaCppBackend, "_find_llama_server_binary", staticmethod(lambda *a, **k: binary)
        ),
        patch.object(
            LlamaCppBackend,
            "_get_gpu_memory",
            staticmethod(lambda *a, **k: [(0, 8000, 16000), (1, 8000, 16000)]),
        ),
    ):
        return asyncio.run(route._resolve_gguf_gpu_ids_for_request(_gguf_config(), gpu_ids))


def _system_gpu_ids_supported(
    main_mod,
    *,
    binary,
    vulkan,
    device,
    vulkan_devices = None,
):
    """Drive /api/system's picker gate with the host fully spoofed."""
    devices = [
        {"index": 0, "index_kind": "physical", "name": "GPU0", "memory_total_gb": 8.0},
        {"index": 1, "index_kind": "physical", "name": "GPU1", "memory_total_gb": 8.0},
    ]
    # Resolve the real bundle layout BEFORE patching the classifier, so the stub can't recurse.
    lacks_gpu_lib = LlamaCppBackend._backend_lacks_gpu_lib(binary)
    main_mod._system_gpu_cache = None
    with (
        patch.object(
            hardware_pkg,
            "get_backend_visible_gpu_info",
            return_value = {"available": True, "devices": devices},
        ),
        patch.object(hardware_pkg, "get_visible_gpu_utilization", return_value = {"devices": []}),
        patch.object(
            hardware_pkg, "get_device", return_value = getattr(hardware_pkg.DeviceType, device)
        ),
        patch.object(LlamaCppBackend, "_is_vulkan_backend", staticmethod(lambda *a: vulkan)),
        patch.object(
            LlamaCppBackend,
            "_get_vulkan_gpu_info",
            staticmethod(lambda *a: list(vulkan_devices or [])),
        ),
        patch.object(
            LlamaCppBackend,
            "_backend_lacks_gpu_lib",
            staticmethod(lambda *a, **k: lacks_gpu_lib),
        ),
    ):
        return main_mod._get_cached_system_gpu_info(logging.getLogger("matrix"))


@pytest.fixture(scope = "module")
def route():
    return _load_module("inference_route_for_os_gpu_matrix", "routes/inference.py")


@pytest.fixture(scope = "module")
def main_mod():
    return _load_module("main_for_os_gpu_matrix", "main.py")


@pytest.mark.parametrize("os_name", sorted(OS_PROFILES))
@pytest.mark.parametrize("gpu_name", sorted(GPU_PROFILES))
def test_gpu_ids_decision_matrix(tmp_path, route, main_mod, os_name, gpu_name):
    """One consistent verdict per (OS, GPU) cell across picker gate and /load."""
    backends, device, pin_supported = GPU_PROFILES[gpu_name]
    platform_name, _suffix = OS_PROFILES[os_name]
    binary = _make_bundle(tmp_path, os_name, backends)

    with patch.object(sys, "platform", platform_name):
        lacks_gpu_lib = LlamaCppBackend._backend_lacks_gpu_lib(binary)
        # macOS ships .dylib ggml libs, so the probe abstains instead of rejecting a Metal build.
        expected_lacks = (gpu_name == "cpu_only") and os_name != "macos"
        assert lacks_gpu_lib is expected_lacks

        info = _system_gpu_ids_supported(main_mod, binary = binary, vulkan = False, device = device)
        assert info["gguf_gpu_ids_supported"] is not expected_lacks
        # The device list is independent of the picker gate: hiding it never zeroes fit badges.
        assert len(info["devices"]) == 2

        if pin_supported or os_name == "macos":
            resolved, vulkan_ordinals = _resolve(
                route, [0, 1], binary = binary, vulkan = False, device = device
            )
            assert resolved == [0, 1]
            assert vulkan_ordinals is False
        else:
            with pytest.raises(HTTPException) as exc:
                _resolve(route, [0, 1], binary = binary, vulkan = False, device = device)
            assert exc.value.status_code == 400
            assert "cpu-only build" in exc.value.detail.lower()


@pytest.mark.parametrize("os_name", ["windows", "linux", "wsl"])
@pytest.mark.parametrize("gpu_name", sorted(GPU_PROFILES))
def test_vulkan_bundle_matrix(tmp_path, route, main_mod, os_name, gpu_name):
    """A Vulkan bundle pins by ggml ordinal on every host it ships for (not macOS: Metal)."""
    _backends, device, _pin = GPU_PROFILES[gpu_name]
    platform_name, _suffix = OS_PROFILES[os_name]
    binary = _make_bundle(tmp_path, os_name, ("cpu", "vulkan"))

    with patch.object(sys, "platform", platform_name):
        # A Vulkan bundle is never a CPU-only build, whatever torch reports.
        assert LlamaCppBackend._backend_lacks_gpu_lib(binary) is False

        resolved, vulkan_ordinals = _resolve(
            route, [0, 1], binary = binary, vulkan = True, device = device
        )
        assert resolved == [0, 1]
        assert vulkan_ordinals is True, "Vulkan builds must tag ids as ggml ordinals"

        probed = [
            {"index": 0, "index_kind": "vulkan", "name": "V0"},
            {"index": 1, "index_kind": "vulkan", "name": "V1"},
        ]
        info = _system_gpu_ids_supported(
            main_mod,
            binary = binary,
            vulkan = True,
            device = device,
            vulkan_devices = probed,
        )
        assert info["gguf_gpu_ids_supported"] is True
        assert [d["index_kind"] for d in info["gguf_gpu_devices"]] == ["vulkan", "vulkan"]
        # The picker list is additive: the general metrics keep describing PyTorch.
        assert [d["index_kind"] for d in info["devices"]] == ["physical", "physical"]


@pytest.mark.parametrize("os_name", ["windows", "linux", "wsl"])
def test_vulkan_probe_failure_keeps_devices_and_hides_picker(tmp_path, route, main_mod, os_name):
    """A missing ICD must not read as "no GPU": keep torch's list, drop the picker."""
    platform_name, _suffix = OS_PROFILES[os_name]
    binary = _make_bundle(tmp_path, os_name, ("cpu", "vulkan"))

    with patch.object(sys, "platform", platform_name):
        info = _system_gpu_ids_supported(
            main_mod, binary = binary, vulkan = True, device = "CUDA", vulkan_devices = []
        )
    assert info["gguf_gpu_ids_supported"] is False
    assert info["available"] is True
    assert len(info["devices"]) == 2
    assert info["gguf_gpu_devices"] == []


def test_vulkan_ordinals_never_replace_the_pytorch_gpu_metrics(tmp_path, main_mod):
    """A forced-Vulkan llama.cpp build on a CUDA/ROCm host must not rewrite the general
    device list: `devices` is summed for the VRAM monitor, Hub fit filtering and training
    model sizing, none of which run on ggml's Vulkan devices."""
    binary = _make_bundle(tmp_path, "linux", ("cpu", "vulkan"))
    # A shared-memory iGPU torch cannot see, four times the real accelerator budget.
    probed = [
        {"index": 0, "index_kind": "vulkan", "name": "iGPU", "memory_total_gb": 64.0},
    ]

    with patch.object(sys, "platform", "linux"):
        info = _system_gpu_ids_supported(
            main_mod, binary = binary, vulkan = True, device = "CUDA", vulkan_devices = probed
        )

    assert [d["name"] for d in info["devices"]] == ["GPU0", "GPU1"]
    assert sum(d["memory_total_gb"] for d in info["devices"]) == 16.0
    assert info["backend"] != "vulkan", "the general metrics still describe PyTorch"
    # The ordinals the picker needs are published on their own channel.
    assert info["gguf_gpu_devices"] == probed
    assert info["gguf_gpu_ids_supported"] is True


def test_system_gpu_payload_stays_additive_for_pre_feature_clients(tmp_path, main_mod):
    """The Vulkan build only adds a key: `available`, `backend` and `devices` read exactly
    as they do without one, so a pre-#7164 frontend is unaffected by the picker feature."""
    binary = _make_bundle(tmp_path, "linux", ("cpu", "vulkan"))
    probed = [{"index": 0, "index_kind": "vulkan", "name": "iGPU", "memory_total_gb": 64.0}]

    with patch.object(sys, "platform", "linux"):
        without = _system_gpu_ids_supported(main_mod, binary = binary, vulkan = False, device = "CUDA")
        with_vulkan = _system_gpu_ids_supported(
            main_mod, binary = binary, vulkan = True, device = "CUDA", vulkan_devices = probed
        )

    for key in ("available", "backend", "devices"):
        assert with_vulkan[key] == without[key], f"{key} must keep its pre-feature meaning"
    assert with_vulkan["gguf_gpu_devices"] == probed


# ── Pre-existing paths this feature must not disturb ────────────────


@pytest.mark.parametrize("os_name", sorted(OS_PROFILES))
@pytest.mark.parametrize("gpu_name", sorted(GPU_PROFILES))
@pytest.mark.parametrize("gpu_ids", [None, []])
def test_request_without_gpu_ids_never_rejects(tmp_path, route, os_name, gpu_name, gpu_ids):
    """A client that sends no pick (every pre-#7164 client) is untouched, even on CPU-only."""
    backends, device, _pin = GPU_PROFILES[gpu_name]
    platform_name, _suffix = OS_PROFILES[os_name]
    binary = _make_bundle(tmp_path, os_name, backends)

    with patch.object(sys, "platform", platform_name):
        assert _resolve(route, gpu_ids, binary = binary, vulkan = False, device = device) == (
            None,
            False,
        )
        assert _resolve(route, gpu_ids, binary = binary, vulkan = True, device = device) == (
            None,
            False,
        )


@pytest.mark.parametrize("os_name", sorted(OS_PROFILES))
def test_single_gpu_host_still_accepts_a_pin(tmp_path, route, os_name):
    """Studio hides the picker below 2 GPUs, but a stored [0] must still load."""
    platform_name, _suffix = OS_PROFILES[os_name]
    binary = _make_bundle(tmp_path, os_name, ("cpu", "cuda"))

    with patch.object(sys, "platform", platform_name):
        resolved, vulkan_ordinals = _resolve(
            route, [0], binary = binary, vulkan = False, device = "CUDA", resolved = [0]
        )
    assert resolved == [0]
    assert vulkan_ordinals is False


@pytest.mark.parametrize("os_name", sorted(OS_PROFILES))
def test_gpu_less_host_reports_no_picker_and_keeps_loading(tmp_path, route, main_mod, os_name):
    """No GPU at all: the picker is gone and an omitted pick still loads."""
    platform_name, _suffix = OS_PROFILES[os_name]
    binary = _make_bundle(tmp_path, os_name, ("cpu",))

    main_mod._system_gpu_cache = None
    with patch.object(sys, "platform", platform_name):
        lacks_gpu_lib = LlamaCppBackend._backend_lacks_gpu_lib(binary)
    with (
        patch.object(sys, "platform", platform_name),
        patch.object(
            hardware_pkg,
            "get_backend_visible_gpu_info",
            return_value = {"available": False, "devices": []},
        ),
        patch.object(hardware_pkg, "get_visible_gpu_utilization", return_value = {"devices": []}),
        patch.object(hardware_pkg, "get_device", return_value = hardware_pkg.DeviceType.CPU),
        patch.object(LlamaCppBackend, "_is_vulkan_backend", staticmethod(lambda *a: False)),
        patch.object(
            LlamaCppBackend,
            "_backend_lacks_gpu_lib",
            staticmethod(lambda *a, **k: lacks_gpu_lib),
        ),
    ):
        info = main_mod._get_cached_system_gpu_info(logging.getLogger("matrix"))
        assert info["available"] is False
        assert info["devices"] == []
        # macOS abstains from the split-library probe, so only the CPU bundle's hosts hide it.
        assert info["gguf_gpu_ids_supported"] is (os_name == "macos")

        assert _resolve(route, None, binary = binary, vulkan = False, device = "CPU") == (
            None,
            False,
        )


def test_load_and_status_schemas_accept_pre_feature_payloads():
    """Old clients omit the new fields; old servers omit them from responses."""
    from models.inference import InferenceStatusResponse, LoadRequest, LoadResponse

    request = LoadRequest(model_path = "unsloth/model-GGUF", gguf_variant = "Q4_K_M")
    assert request.gpu_ids is None
    assert request.gguf_memory_mode is None

    # An old response still validates, and the new fields read as unset, not as a placement.
    load = LoadResponse(
        status = "loaded",
        model = "unsloth/model-GGUF",
        display_name = "model",
        inference = {},
    )
    assert load.gguf_memory_mode is None
    assert load.requested_gpu_ids is None

    status = InferenceStatusResponse()
    assert status.gguf_memory_mode is None
    assert status.requested_gpu_ids is None
    assert status.gpu_ids is None


def test_diffusion_on_a_vulkan_bundle_resolves_physical_ids(tmp_path, route, main_mod):
    """A diffusion GGUF opts out of the Vulkan ordinal namespace at /load.

    DiffusionGemma never runs through llama.cpp, so its gpu_ids stay CUDA physical ids even on
    a Vulkan bundle, while /api/system still advertises ggml ordinals: the namespaces collide on
    the same integers, which is why model-config-page.tsx hides the picker here.
    """
    binary = _make_bundle(tmp_path, "linux", ("cpu", "vulkan"))
    probed = [
        {"index": 0, "index_kind": "vulkan", "name": "V0"},
        {"index": 1, "index_kind": "vulkan", "name": "V1"},
    ]

    with patch.object(sys, "platform", "linux"):
        info = _system_gpu_ids_supported(
            main_mod, binary = binary, vulkan = True, device = "CUDA", vulkan_devices = probed
        )
        # The picker is offered ggml Vulkan ordinals, with no model in the loop.
        assert info["gguf_gpu_ids_supported"] is True
        assert [d["index_kind"] for d in info["gguf_gpu_devices"]] == ["vulkan", "vulkan"]

        _ids, normal_vulkan = _resolve(
            route, [1], binary = binary, vulkan = True, device = "CUDA", diffusion = False
        )
        assert normal_vulkan is True

        ids, diffusion_vulkan = _resolve(
            route, [1], binary = binary, vulkan = True, device = "CUDA", diffusion = True
        )
        assert ids == [1]
        assert diffusion_vulkan is False, "diffusion ids must resolve as CUDA physical ids"

        # And the CUDA requirement still stands for every other host.
        for device in ("CPU", "XPU"):
            with pytest.raises(HTTPException) as exc:
                _resolve(route, [1], binary = binary, vulkan = True, device = device, diffusion = True)
            assert exc.value.status_code == 400
            assert "diffusiongemma" in exc.value.detail.lower()
