# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A CPU torch beside an AMD or Arc GPU installs the Vulkan sd.cpp build (#13666).

Vega, RDNA1 and Polaris get no ROCm wheel, so torch is CPU-only there while llama.cpp already
renders on the card through Vulkan. Every hardware answer is stubbed."""

from __future__ import annotations

import io
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from core.inference import diffusion_engine_router as r
from core.inference import sd_cpp_backend
from core.inference.diffusion_families import detect_family
from core.inference.sd_cpp_engine import ENGINE_SD_CPP

_STUDIO = Path(__file__).resolve().parents[2]
if str(_STUDIO) not in sys.path:
    sys.path.insert(0, str(_STUDIO))

_ENVS = (
    "UNSLOTH_DIFFUSION_ENGINE",
    "UNSLOTH_DIFFUSION_SD_CPP",
    "UNSLOTH_DIFFUSION_SD_CPP_DEVICE",
    "UNSLOTH_DIFFUSION_SD_CPP_VULKAN_FALLBACK",
    "ROCR_VISIBLE_DEVICES",
    "HIP_VISIBLE_DEVICES",
    "CUDA_VISIBLE_DEVICES",
    "ZE_AFFINITY_MASK",
    "GGML_VK_VISIBLE_DEVICES",
)

_VEGA = {"vendor": "amd", "index": 0, "name": None, "source": "sysfs-drm"}
_ARC_LINUX = {"vendor": "intel", "index": 0, "name": None, "xpu_class": True}
_UHD_LINUX = {"vendor": "intel", "index": 0, "name": None, "xpu_class": False}
_ARC_WINDOWS = {"vendor": "intel", "index": 0, "name": "Intel(R) Arc(TM) A770 Graphics"}
_UHD_WINDOWS = {"vendor": "intel", "index": 0, "name": "Intel(R) UHD Graphics 770"}
_RTX = {"vendor": "nvidia", "index": 0, "name": "NVIDIA GeForce RTX 3060"}


def _host(
    *devices,
    unanswered = (),
    unknown = False,
):
    return {"unknown": unknown, "unanswered": list(unanswered), "devices": list(devices)}


def _platform(monkeypatch, system, machine):
    monkeypatch.setattr(r, "sys", SimpleNamespace(platform = system, modules = sys.modules))
    monkeypatch.setattr(r, "platform", SimpleNamespace(machine = lambda: machine))


@pytest.fixture(autouse = True)
def _pinned_host(monkeypatch):
    for e in _ENVS:
        monkeypatch.delenv(e, raising = False)
    _platform(monkeypatch, "linux", "x86_64")
    monkeypatch.setattr(r, "_physical_inventory", lambda: _host(_VEGA))
    monkeypatch.setattr(r, "_vulkan_loader_loads", lambda: True)
    monkeypatch.setattr(r, "_linux_render_node_accessible", lambda vendors: True)
    import utils.hardware.amd as amd

    monkeypatch.setattr(amd, "the_vulkan_loader_has_no_usable_driver", lambda: False)
    monkeypatch.setattr(r, "_hardware_vulkan_driver", lambda vendors: True)
    monkeypatch.setattr(r, "_linux_nvidia_pci_present", lambda: False)


@pytest.mark.parametrize(
    "devices,expected",
    [
        ((_VEGA,), "vulkan"),
        ((_ARC_LINUX,), "vulkan"),
        # Unpinned sd.cpp could take the UHD iGPU first.
        ((_UHD_LINUX, _VEGA), "auto"),
        ((_ARC_LINUX, _VEGA), "vulkan"),
        ((_UHD_LINUX,), "auto"),
        ((), "auto"),
        ((_VEGA, _RTX), "auto"),
        ((_RTX,), "auto"),
    ],
)
def test_cpu_torch_vulkan_follows_the_cards(monkeypatch, devices, expected):
    monkeypatch.setattr(r, "_physical_inventory", lambda: _host(*devices))
    assert r.image_install_accelerator("cpu") == expected


@pytest.mark.parametrize("device,expected", [(_ARC_WINDOWS, "vulkan"), (_UHD_WINDOWS, "auto")])
def test_windows_intel_is_read_by_name(monkeypatch, device, expected):
    _platform(monkeypatch, "win32", "AMD64")
    monkeypatch.setattr(r, "_physical_inventory", lambda: _host(device))
    assert r.image_install_accelerator("cpu") == expected


@pytest.mark.parametrize(
    "inventory",
    [_host(_VEGA, unanswered = ["nvidia"]), _host(unknown = True)],
)
def test_an_unanswered_nvidia_probe_keeps_cpu(monkeypatch, inventory):
    monkeypatch.setattr(r, "_physical_inventory", lambda: inventory)
    assert r.image_install_accelerator("cpu") == "auto"


@pytest.mark.parametrize(
    "var,value",
    [
        ("ROCR_VISIBLE_DEVICES", "0"),
        ("HIP_VISIBLE_DEVICES", "0"),
        ("CUDA_VISIBLE_DEVICES", "0"),
        ("ZE_AFFINITY_MASK", "0"),
        ("GGML_VK_VISIBLE_DEVICES", ""),
    ],
)
def test_a_mask_vulkan_does_not_read_keeps_cpu(monkeypatch, var, value):
    monkeypatch.setenv(var, value)
    assert r.image_install_accelerator("cpu") == "auto"


@pytest.mark.parametrize("value", ["", "default"])
def test_an_unset_equivalent_ze_mask_is_not_a_mask(monkeypatch, value):
    monkeypatch.setenv("ZE_AFFINITY_MASK", value)
    assert r.image_install_accelerator("cpu") == "vulkan"


@pytest.mark.parametrize("value", ["abc", "-1", "0,x", ""])
def test_a_malformed_ggml_vulkan_mask_keeps_cpu(monkeypatch, value):
    # Selects nothing or throws.
    monkeypatch.setenv("GGML_VK_VISIBLE_DEVICES", value)
    assert r.image_install_accelerator("cpu") == "auto"


@pytest.mark.parametrize(
    "output",
    [
        "vk::Queue::submit: ErrorDeviceLost",
        "VK_ERROR_DEVICE_LOST",
        "vk::createInstance: ErrorIncompatibleDriver",
    ],
)
def test_vulkan_failures_are_recognised(output):
    assert sd_cpp_backend.output_shows_accelerator_failure(output)


def test_an_incompatible_vulkan_driver_is_decisive():
    assert sd_cpp_backend.output_shows_decisive_accelerator_failure("ErrorIncompatibleDriver")
    assert not sd_cpp_backend.output_shows_decisive_accelerator_failure("ErrorDeviceLost")
    assert not sd_cpp_backend.output_shows_accelerator_failure("VK_ERROR_OUT_OF_DEVICE_MEMORY")


def test_a_ggml_vulkan_mask_is_honoured_but_never_vouches_for_a_mixed_host(monkeypatch):
    # Its ordinals index the raw Vulkan list, so they cannot say which card a UHD + Radeon host gets.
    monkeypatch.setenv("GGML_VK_VISIBLE_DEVICES", "1")
    assert r.image_install_accelerator("cpu") == "vulkan"
    monkeypatch.setattr(r, "_physical_inventory", lambda: _host(_UHD_LINUX, _VEGA))
    assert r.image_install_accelerator("cpu") == "auto"


def test_an_nvidia_card_nvidia_smi_cannot_see_keeps_cpu(monkeypatch):
    monkeypatch.setattr(r, "_linux_nvidia_pci_present", lambda: True)
    assert r.image_install_accelerator("cpu") == "auto"


@pytest.mark.parametrize(
    "vendors,manifests,expected",
    [
        ({"amd"}, ["/usr/share/vulkan/icd.d/radeon_icd.x86_64.json"], True),
        ({"amd"}, ["/usr/share/vulkan/icd.d/lvp_icd.x86_64.json"], False),
        # VK_ICD_FILENAMES picked another vendor's driver: it lists no AMD card.
        ({"amd"}, ["/etc/vulkan/icd.d/intel_icd.json"], False),
        (
            {"intel"},
            ["/usr/share/vulkan/icd.d/lvp_icd.x86_64.json", "/etc/vulkan/icd.d/intel_icd.json"],
            True,
        ),
        ({"amd"}, [], False),
    ],
)
def test_only_a_driver_for_the_listed_cards_counts(monkeypatch, vendors, manifests, expected):
    import utils.hardware.amd as amd

    monkeypatch.undo()
    _platform(monkeypatch, "linux", "x86_64")
    monkeypatch.setattr(amd, "_vulkan_icd_manifest_paths", lambda: list(manifests))
    monkeypatch.setattr(amd, "_loadable_icd_manifests", lambda paths = None: list(manifests))
    assert r._hardware_vulkan_driver(vendors) is expected


def test_an_unmounted_render_node_keeps_cpu(monkeypatch):
    monkeypatch.setattr(r, "_linux_render_node_accessible", lambda vendors: False)
    assert r.image_install_accelerator("cpu") == "auto"
    _platform(monkeypatch, "win32", "AMD64")
    assert r.image_install_accelerator("cpu") == "vulkan"


@pytest.mark.parametrize(
    "system,machine", [("linux", "aarch64"), ("win32", "ARM64"), ("darwin", "arm64")]
)
def test_hosts_without_a_vulkan_bundle_keep_cpu(monkeypatch, system, machine):
    _platform(monkeypatch, system, machine)
    assert r.image_install_accelerator("cpu") == "auto"


def test_no_vulkan_driver_or_loader_keeps_cpu(monkeypatch):
    import utils.hardware.amd as amd

    monkeypatch.setattr(r, "_hardware_vulkan_driver", lambda vendors: False)
    assert r.image_install_accelerator("cpu") == "auto"
    monkeypatch.setattr(r, "_hardware_vulkan_driver", lambda vendors: True)
    monkeypatch.setattr(r, "_vulkan_loader_loads", lambda: False)
    assert r.image_install_accelerator("cpu") == "auto"


def test_the_fallback_knob_opts_out(monkeypatch):
    monkeypatch.setenv("UNSLOTH_DIFFUSION_SD_CPP_VULKAN_FALLBACK", "0")
    assert r.image_install_accelerator("cpu") == "auto"


@pytest.mark.parametrize(
    "backend,expected", [("cuda", "cuda"), ("rocm", "rocm"), ("xpu", "vulkan")]
)
def test_gpu_torch_backends_are_unchanged(backend, expected):
    assert r.image_install_accelerator(backend) == expected


def test_video_keeps_torchs_build():
    assert r._install_accelerator_for("cpu") == "auto"


def test_a_vulkan_build_that_cannot_run_falls_to_cpu(monkeypatch):
    monkeypatch.setattr(sd_cpp_backend, "_accelerator_runtime_failures", {})
    monkeypatch.setattr(sd_cpp_backend, "_stored_accelerator_runtime_failures", lambda: {})
    monkeypatch.setattr(sd_cpp_backend, "_persist_accelerator_runtime_failures", lambda _r: None)
    monkeypatch.setattr(sd_cpp_backend, "_accelerator_fingerprint", lambda *_a: {"bundle": "t"})
    assert sd_cpp_backend.fallback_accelerator_for("vulkan") == "cpu"
    assert sd_cpp_backend.preferred_accelerator("vulkan") == "vulkan"
    sd_cpp_backend.note_accelerator_runtime_failure("vulkan", proven = True)
    assert sd_cpp_backend.preferred_accelerator("vulkan") == "cpu"
    # A ROCm host whose Vulkan rung also failed reaches CPU instead of retrying Vulkan.
    assert sd_cpp_backend.preferred_accelerator("rocm") == "rocm"
    sd_cpp_backend.note_accelerator_runtime_failure("rocm", proven = True)
    assert sd_cpp_backend.preferred_accelerator("rocm") == "cpu"


def test_selection_installs_the_vulkan_build(monkeypatch):
    monkeypatch.setattr(
        r, "resolve_diffusion_device_target", lambda: SimpleNamespace(backend = "cpu", device = "cpu")
    )
    monkeypatch.setattr(r, "_off_torch_warned", set())
    monkeypatch.setattr(sd_cpp_backend, "_installed_accelerator_of", lambda _b: "vulkan")
    monkeypatch.setattr(
        r,
        "get_active_diffusion_engine",
        lambda: SimpleNamespace(
            status = lambda: {"loaded": False, "repo_id": None}, unload = lambda: None
        ),
    )
    monkeypatch.setattr(r, "_server_binary_runnable", lambda *_a, **_k: True)
    monkeypatch.setattr(r, "SdCppEngine", lambda **_: SimpleNamespace(version = lambda: "sd-cli v0"))
    asked: list[str] = []

    def _ensure(**kwargs):
        asked.append(kwargs["accelerator"])
        return None

    monkeypatch.setattr(r, "ensure_sd_server_binary", _ensure)
    monkeypatch.setattr(
        r, "ensure_sd_cpp_binary", lambda **kw: asked.append(kw["accelerator"]) or "/opt/sd/sd-cli"
    )
    saved = (r._active_engine_name, r._fallback_reason)
    try:
        r.select_and_activate_engine(detect_family("z-image"))
        assert r.active_engine_name() == ENGINE_SD_CPP
    finally:
        r._active_engine_name, r._fallback_reason = saved
    assert asked and set(asked) == {"vulkan"}


_PINNED_RELEASE_ASSETS = [
    "sd-master-951-f89d9b1-uafac7a7-bin-Linux-Ubuntu-22.04-x86_64-cuda12.zip",
    "sd-master-951-f89d9b1-uafac7a7-bin-Linux-Ubuntu-22.04-x86_64-vulkan.zip",
    "sd-master-951-f89d9b1-uafac7a7-bin-Linux-Ubuntu-22.04-x86_64.zip",
    "sd-master-951-f89d9b1-uafac7a7-bin-Linux-Ubuntu-24.04-aarch64.zip",
    "sd-master-951-f89d9b1-uafac7a7-bin-win-cpu-x64.zip",
    "sd-master-951-f89d9b1-uafac7a7-bin-win-cuda12-x64.zip",
    "sd-master-951-f89d9b1-uafac7a7-bin-win-vulkan-x64.zip",
]


@pytest.mark.parametrize(
    "system,machine,expected",
    [
        (
            "Linux",
            "x86_64",
            "sd-master-951-f89d9b1-uafac7a7-bin-Linux-Ubuntu-22.04-x86_64-vulkan.zip",
        ),
        ("Windows", "AMD64", "sd-master-951-f89d9b1-uafac7a7-bin-win-vulkan-x64.zip"),
        ("Linux", "aarch64", None),
    ],
)
def test_the_pinned_release_has_the_vulkan_bundle_it_routes_to(system, machine, expected):
    from install_sd_cpp_prebuilt import resolve_release_asset
    assert (
        resolve_release_asset(
            _PINNED_RELEASE_ASSETS, system = system, machine = machine, accelerator = "vulkan"
        )
        == expected
    )


def test_only_a_render_node_of_the_listed_vendor_counts(monkeypatch, tmp_path):
    import builtins
    import glob

    monkeypatch.undo()  # the real helper, not the fixture's stub
    vendors = {"renderD128": "0x1af4", "renderD129": "0x1002"}  # virtio, then AMD
    real_open = builtins.open

    def _open(path, *a, **k):
        name = str(path).split("/")[-3] if str(path).startswith("/sys/class/drm/") else None
        if name in vendors:
            return io.StringIO(vendors[name] + "\n")
        return real_open(path, *a, **k)

    accessible = set()
    monkeypatch.setattr(glob, "glob", lambda pattern: [f"/dev/dri/{n}" for n in vendors])
    monkeypatch.setattr(builtins, "open", _open)
    monkeypatch.setattr(r.os, "access", lambda node, mode: node in accessible)
    accessible.add("/dev/dri/renderD128")
    assert r._linux_render_node_accessible({"amd"}) is False
    accessible.add("/dev/dri/renderD129")
    assert r._linux_render_node_accessible({"amd"}) is True
    assert r._linux_render_node_accessible({"intel"}) is False


@pytest.mark.parametrize("var", ["VK_LOADER_DRIVERS_DISABLE", "VK_LOADER_DRIVERS_SELECT"])
def test_a_windows_driver_filter_the_walk_cannot_apply_keeps_cpu(monkeypatch, var):
    import utils.hardware.amd as amd

    monkeypatch.undo()
    _platform(monkeypatch, "win32", "AMD64")
    monkeypatch.setattr(amd, "_vulkan_icd_manifest_paths", lambda: [])
    assert r._hardware_vulkan_driver({"amd"}) is True
    monkeypatch.setenv(var, "*")
    assert r._hardware_vulkan_driver({"amd"}) is False
