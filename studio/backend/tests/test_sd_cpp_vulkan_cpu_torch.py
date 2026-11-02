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
    monkeypatch.setattr(r, "_accessible_render_vendors", lambda vendors: set(vendors))
    import utils.hardware.amd as amd

    monkeypatch.setattr(amd, "the_vulkan_loader_has_no_usable_driver", lambda: False)
    monkeypatch.setattr(r, "_hardware_vulkan_driver", lambda vendors: True)
    monkeypatch.setattr(r, "_nvidia_adapter_present", lambda: False)
    monkeypatch.setattr(r, "_reachable_intel_arc_render_node", lambda: True, raising = False)


@pytest.mark.parametrize(
    "devices,expected",
    [
        ((_VEGA,), "vulkan"),
        ((_ARC_LINUX,), "vulkan"),
        # Unpinned sd.cpp takes a discrete GPU before an iGPU: the card renders, not the UHD iGPU.
        ((_UHD_LINUX, _VEGA), "vulkan"),
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
    if var == "ZE_AFFINITY_MASK":
        monkeypatch.setattr(r, "_physical_inventory", lambda: _host(_ARC_LINUX))
    monkeypatch.setenv(var, value)
    assert r.image_install_accelerator("cpu") == "auto"


@pytest.mark.parametrize(
    "device,var",
    [
        (_ARC_LINUX, "CUDA_VISIBLE_DEVICES"),
        (_ARC_LINUX, "HIP_VISIBLE_DEVICES"),
        (_ARC_LINUX, "ROCR_VISIBLE_DEVICES"),
        (_VEGA, "ZE_AFFINITY_MASK"),
    ],
)
def test_a_mask_for_another_vendor_leaves_the_card_alone(monkeypatch, device, var):
    monkeypatch.setattr(r, "_physical_inventory", lambda: _host(device))
    monkeypatch.setenv(var, "0")
    assert r.image_install_accelerator("cpu") == "vulkan"


def test_an_unmounted_uhd_node_does_not_block_the_amd_card(monkeypatch):
    monkeypatch.setattr(r, "_physical_inventory", lambda: _host(_UHD_LINUX, _VEGA))
    monkeypatch.setattr(r, "_accessible_render_vendors", lambda vendors: {"amd"} & set(vendors))
    assert r.image_install_accelerator("cpu") == "vulkan"
    # Both nodes mounted: the dedicated card is still the one sd.cpp takes.
    monkeypatch.setattr(r, "_accessible_render_vendors", lambda vendors: set(vendors))
    assert r.image_install_accelerator("cpu") == "vulkan"


@pytest.mark.parametrize("arc_node_reachable, expected", [(True, "vulkan"), (False, "auto")])
def test_a_uhd_beside_an_arc_needs_the_arcs_own_node(monkeypatch, arc_node_reachable, expected):
    # The vendor-level node check passes on the iGPU's node alone: a container mounting only it sees the UHD.
    monkeypatch.setattr(r, "_physical_inventory", lambda: _host(_UHD_LINUX, _ARC_LINUX))
    monkeypatch.setattr(
        r, "_reachable_intel_arc_render_node", lambda: arc_node_reachable, raising = False
    )
    assert r.image_install_accelerator("cpu") == expected


@pytest.mark.parametrize(
    "arc_node_reachable, drivers, expected",
    [
        # The Arc node is not mounted and only the Intel driver loads: Intel Vulkan sees the UHD alone and the AMD
        # card has no driver, so the iGPU would render.
        (False, {"intel"}, "auto"),
        # The AMD card's own driver loads: it renders, whatever the Arc's node.
        (False, {"amd"}, "vulkan"),
        (False, {"amd", "intel"}, "vulkan"),
        # The Arc's node is reachable: the Intel driver drives the Arc, a discrete GPU sd.cpp takes first.
        (True, {"intel"}, "vulkan"),
    ],
)
def test_an_arc_beside_an_amd_card_and_a_uhd_still_needs_its_own_node(
    monkeypatch, arc_node_reachable, drivers, expected
):
    monkeypatch.setattr(r, "_physical_inventory", lambda: _host(_VEGA, _ARC_LINUX, _UHD_LINUX))
    monkeypatch.setattr(
        r, "_reachable_intel_arc_render_node", lambda: arc_node_reachable, raising = False
    )
    monkeypatch.setattr(r, "_hardware_vulkan_driver", lambda vendors: bool(set(vendors) & drivers))
    assert r.image_install_accelerator("cpu") == expected


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
        "vk::createInstance: ErrorIncompatibleDriver",
        "vk::createInstance: ErrorInitializationFailed",
    ],
)
def test_vulkan_failures_are_recognised(output):
    assert sd_cpp_backend.output_shows_accelerator_failure(output)


@pytest.mark.parametrize("output", ["vk::Queue::submit: ErrorDeviceLost", "VK_ERROR_DEVICE_LOST"])
def test_a_lost_vulkan_device_is_not_a_build_failure(output):
    # Recoverable by recreating the device (a TDR, a scheduler timeout): no evidence about the build.
    assert not sd_cpp_backend.output_shows_accelerator_failure(output)


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
    monkeypatch.setattr(r, "_nvidia_adapter_present", lambda: True)
    assert r.image_install_accelerator("cpu") == "auto"


@pytest.mark.parametrize(
    "vendors,manifests,expected",
    [
        ({"amd"}, ["/usr/share/vulkan/icd.d/radeon_icd.x86_64.json"], True),
        ({"amd"}, ["/usr/share/vulkan/icd.d/radv_icd.x86_64.json"], True),
        ({"amd"}, ["/etc/vulkan/icd.d/amd_icd64.json"], True),
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
    monkeypatch.setattr(r, "_accessible_render_vendors", lambda vendors: set())
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
    "sd-master-982-0518b34-uaba3a1c-bin-Linux-Ubuntu-22.04-x86_64-cuda12.zip",
    "sd-master-982-0518b34-uaba3a1c-bin-Linux-Ubuntu-22.04-x86_64-vulkan.zip",
    "sd-master-982-0518b34-uaba3a1c-bin-Linux-Ubuntu-22.04-x86_64.zip",
    "sd-master-982-0518b34-uaba3a1c-bin-Linux-Ubuntu-24.04-aarch64.zip",
    "sd-master-982-0518b34-uaba3a1c-bin-win-cpu-x64.zip",
    "sd-master-982-0518b34-uaba3a1c-bin-win-cuda12-x64.zip",
    "sd-master-982-0518b34-uaba3a1c-bin-win-vulkan-x64.zip",
]


@pytest.mark.parametrize(
    "system,machine,expected",
    [
        (
            "Linux",
            "x86_64",
            "sd-master-982-0518b34-uaba3a1c-bin-Linux-Ubuntu-22.04-x86_64-vulkan.zip",
        ),
        ("Windows", "AMD64", "sd-master-982-0518b34-uaba3a1c-bin-win-vulkan-x64.zip"),
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
    vendors = {"renderD128": "0x8086", "renderD129": "0x1002"}  # Intel, then AMD
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
    assert r._accessible_render_vendors({"amd"}) == set()
    accessible.add("/dev/dri/renderD129")
    assert r._accessible_render_vendors({"amd"}) == {"amd"}
    assert r._accessible_render_vendors({"amd", "intel"}) == {"amd", "intel"}
    # A virtio-gpu (Venus) node beside the card: unpinned sd.cpp could pick it.
    vendors["renderD130"] = "0x1af4"
    assert r._accessible_render_vendors({"amd"}) == {"amd"}  # not openable here
    accessible.add("/dev/dri/renderD130")
    assert r._accessible_render_vendors({"amd"}) == set()
    del vendors["renderD130"]
    # A node whose vendor cannot be read: unknown, so it blocks only while openable.
    nodes = [*vendors, "renderD999"]
    monkeypatch.setattr(glob, "glob", lambda pattern: [f"/dev/dri/{n}" for n in nodes])
    assert r._accessible_render_vendors({"amd"}) == {"amd"}
    accessible.add("/dev/dri/renderD999")
    assert r._accessible_render_vendors({"amd"}) == set()


@pytest.mark.parametrize("var", ["VK_LOADER_DRIVERS_DISABLE", "VK_LOADER_DRIVERS_SELECT"])
def test_a_windows_driver_filter_the_walk_cannot_apply_keeps_cpu(monkeypatch, var):
    import utils.hardware.amd as amd

    monkeypatch.undo()
    _platform(monkeypatch, "win32", "AMD64")
    monkeypatch.setattr(amd, "_vulkan_icd_manifest_paths", lambda: [])
    assert r._hardware_vulkan_driver({"amd"}) is True
    monkeypatch.setenv(var, "*")
    assert r._hardware_vulkan_driver({"amd"}) is False


def test_the_render_node_and_the_driver_must_share_a_vendor(monkeypatch):
    # AMD + Arc: only the AMD node is mounted and only an Intel driver loads.
    monkeypatch.setattr(r, "_physical_inventory", lambda: _host(_VEGA, _ARC_LINUX))
    monkeypatch.setattr(r, "_accessible_render_vendors", lambda vendors: {"amd"} & set(vendors))
    monkeypatch.setattr(r, "_hardware_vulkan_driver", lambda vendors: "intel" in vendors)
    assert r.image_install_accelerator("cpu") == "auto"
    monkeypatch.setattr(r, "_hardware_vulkan_driver", lambda vendors: "amd" in vendors)
    assert r.image_install_accelerator("cpu") == "vulkan"


def test_a_uhd_beside_an_amd_card_needs_the_cards_own_driver(monkeypatch):
    # Both nodes mounted, but only an Intel ICD loads: Vulkan then enumerates the iGPU alone and the AMD card is not a
    # Vulkan device at all, so the UHD would render.
    monkeypatch.setattr(r, "_physical_inventory", lambda: _host(_UHD_LINUX, _VEGA))
    monkeypatch.setattr(r, "_hardware_vulkan_driver", lambda vendors: "intel" in vendors)
    assert r.image_install_accelerator("cpu") == "auto"
    monkeypatch.setattr(r, "_hardware_vulkan_driver", lambda vendors: "amd" in vendors)
    assert r.image_install_accelerator("cpu") == "vulkan"


@pytest.mark.parametrize(
    "records,live,expected",
    [
        ({}, None, False),
        (None, ["AMD Radeon RX 7900 XTX"], True),  # registry unreadable
        (
            {1: {"name": "NVIDIA GeForce RTX 4090"}},
            ["AMD Radeon RX 7900 XTX", "NVIDIA GeForce RTX 4090"],
            True,
        ),
        # A stale record of a card that was removed.
        ({1: {"name": "NVIDIA GeForce RTX 3060"}}, ["AMD Radeon RX 7900 XTX"], False),
        ({1: {"name": "NVIDIA GeForce RTX 3060"}}, None, True),
    ],
)
def test_windows_nvidia_without_nvidia_smi_is_found_in_the_adapter_registry(
    monkeypatch, records, live, expected
):
    import utils.hardware.hardware as hw

    monkeypatch.undo()
    _platform(monkeypatch, "win32", "AMD64")
    seen = []

    def _records(vendor_id, distinguish_failure = False):
        seen.append(vendor_id)
        return records

    monkeypatch.setattr(hw, "_windows_amd_adapter_records_by_luid", _records)
    monkeypatch.setattr(hw, "_windows_live_adapter_names", lambda: live)
    assert r._nvidia_adapter_present() is expected
    assert seen == [0x10DE]


def _empty_failure_records(monkeypatch, stored = None):
    monkeypatch.setattr(sd_cpp_backend, "_accelerator_runtime_failures", {})
    monkeypatch.setattr(
        sd_cpp_backend, "_stored_accelerator_runtime_failures", lambda: dict(stored or {})
    )
    monkeypatch.setattr(sd_cpp_backend, "_persist_accelerator_runtime_failures", lambda _r: None)
    monkeypatch.setattr(sd_cpp_backend, "_accelerator_fingerprint", lambda *_a: {"bundle": "t"})


@pytest.mark.parametrize("cpu_torch", [True, False])
def test_two_lost_vulkan_devices_keep_the_vulkan_build(monkeypatch, cpu_torch):
    """An Arc / ROCm-on-Vulkan host (and a CPU-torch one) that lost the device twice stays on the GPU build."""
    _empty_failure_records(monkeypatch)
    monkeypatch.setattr(sd_cpp_backend, "_vulkan_cpu_rung_applies", lambda: cpu_torch)
    monkeypatch.setattr(sd_cpp_backend, "_installed_accelerator_of", lambda _b: "vulkan")
    for _ in range(2):
        sd_cpp_backend.note_accelerator_failure_from_output(
            "/x/sd-cli", "vk::Queue::submit: ErrorDeviceLost"
        )
    assert sd_cpp_backend.preferred_accelerator("vulkan") == "vulkan"


def test_ambiguous_vulkan_strikes_move_only_a_cpu_torch_host(monkeypatch):
    """Two ambiguous failures: the CPU-torch route falls to CPU sd.cpp, an XPU / ROCm host keeps Vulkan."""
    _empty_failure_records(monkeypatch)
    for _ in range(2):
        sd_cpp_backend.note_accelerator_runtime_failure("vulkan", proven = False)
    monkeypatch.setattr(sd_cpp_backend, "_vulkan_cpu_rung_applies", lambda: True)
    assert sd_cpp_backend.preferred_accelerator("vulkan") == "cpu"
    monkeypatch.setattr(sd_cpp_backend, "_vulkan_cpu_rung_applies", lambda: False)
    assert sd_cpp_backend.preferred_accelerator("vulkan") == "vulkan"
    # Decisive evidence still moves every host: a build that cannot run must not leave it with nothing.
    sd_cpp_backend.note_accelerator_runtime_failure("vulkan", proven = True)
    assert sd_cpp_backend.preferred_accelerator("vulkan") == "cpu"


def test_a_vulkan_record_from_before_device_lost_stopped_counting_is_ignored(monkeypatch):
    """Two ambiguous strikes persisted under the old rule may be two lost devices: they no longer divert."""
    monkeypatch.setattr(sd_cpp_backend, "_vulkan_cpu_rung_applies", lambda: True)
    old = {"vulkan": {"strikes": 2, "proven": False, "fingerprint": {"bundle": "t"}}}
    _empty_failure_records(monkeypatch, stored = old)
    assert sd_cpp_backend.preferred_accelerator("vulkan") == "vulkan"
    # A decisive one written then still holds.
    proven = {"vulkan": {"strikes": 1, "proven": True, "fingerprint": {"bundle": "t"}}}
    _empty_failure_records(monkeypatch, stored = proven)
    assert sd_cpp_backend.preferred_accelerator("vulkan") == "cpu"


def test_an_old_rule_record_keeps_only_the_proven_cards_known():
    """Card A proven, card B ambiguous only, no unscoped proof: B must not divert on A's failure."""
    old = {
        "strikes": 2,
        "proven": True,
        "fingerprint": {"bundle": "t"},
        "cards": ["Card A", "Card B"],
        "per_card": {
            "Card A": {"strikes": 1, "proven": True},
            "Card B": {"strikes": 1, "proven": False},
        },
    }
    kept = sd_cpp_backend._decisive_evidence_only(old)
    assert kept["cards"] == ["Card A"]
    fp = {"bundle": "t"}
    assert sd_cpp_backend._record_diverts(kept, fp, card = "Card A") is True
    assert sd_cpp_backend._record_diverts(kept, fp, card = "Card B") is False


@pytest.mark.parametrize("card,diverts", [("Card A", True), ("Card B", False), (None, True)])
def test_an_old_rule_proof_without_tallies_keeps_its_cards(card, diverts):
    """Proof naming cards but no per-card tallies diverted only those cards; migrating it must not widen it."""
    old = {"strikes": 1, "proven": True, "fingerprint": {"bundle": "t"}, "cards": ["Card A"]}
    fp = {"bundle": "t"}
    assert sd_cpp_backend._record_diverts(old, fp, card = card) is diverts
    kept = sd_cpp_backend._record_under_current_rule("vulkan", old)
    assert sd_cpp_backend._record_diverts(kept, fp, card = card) is diverts


def test_the_fallback_status_counts_the_same_evidence_as_the_runtime(monkeypatch):
    """Settings must not report a diversion the runtime decision no longer makes."""
    monkeypatch.setattr(sd_cpp_backend, "sd_cpp_vulkan_fallback_enabled", lambda: True)
    monkeypatch.setattr(sd_cpp_backend, "_vulkan_cpu_rung_applies", lambda: True)
    old = {"vulkan": {"strikes": 2, "proven": False, "fingerprint": {"bundle": "t"}}}
    _empty_failure_records(monkeypatch, stored = old)
    assert sd_cpp_backend.preferred_accelerator("vulkan") == "vulkan"
    assert sd_cpp_backend.accelerator_runtime_failure_state()["diverting"] is False
    # Current ambiguous strikes off the CPU-torch route: kept on Vulkan, and reported so.
    _empty_failure_records(monkeypatch)
    for _ in range(2):
        sd_cpp_backend.note_accelerator_runtime_failure("vulkan", proven = False)
    monkeypatch.setattr(sd_cpp_backend, "_vulkan_cpu_rung_applies", lambda: False)
    assert sd_cpp_backend.preferred_accelerator("vulkan") == "vulkan"
    assert sd_cpp_backend.accelerator_runtime_failure_state()["diverting"] is False
    monkeypatch.setattr(sd_cpp_backend, "_vulkan_cpu_rung_applies", lambda: True)
    assert sd_cpp_backend.accelerator_runtime_failure_state()["diverting"] is True


@pytest.mark.parametrize("value", ["0 1", "0, 1", "1 "])
def test_a_space_separated_ggml_vulkan_mask_is_a_mask(monkeypatch, value):
    # ggml reads commas as spaces.
    monkeypatch.setenv("GGML_VK_VISIBLE_DEVICES", value)
    assert r.image_install_accelerator("cpu") == "vulkan"


_LISTING = (
    "CPU\tAMD Ryzen 9\nVulkan0\tAMD Radeon RX 7800 XT (RADV NAVI32)\nROCm0\tAMD Radeon RX 7800 XT\n"
)


def test_device_listing_readers_share_one_probe(monkeypatch, tmp_path):
    binary = tmp_path / "sd-cli"
    binary.write_bytes(b"x")
    calls: list = []

    def _probe(_binary, *args):
        calls.append(args)
        return _LISTING

    monkeypatch.setattr(sd_cpp_backend, "_LAST_DEVICE_LISTING", {})
    monkeypatch.setattr(sd_cpp_backend, "_sd_cpp_probe_output", _probe)
    assert sd_cpp_backend.sd_cpp_device_name_for_ordinal(str(binary), 0) == "ROCm0"
    sd_cpp_backend.sd_cpp_device_named(str(binary), "AMD Radeon RX 7800 XT")
    assert sd_cpp_backend.sd_cpp_listed_device(str(binary))[0] == "vulkan"
    assert calls == [("--list-devices",)]
    # A different device selection is a different listing.
    monkeypatch.setenv("GGML_VK_VISIBLE_DEVICES", "0")
    sd_cpp_backend.sd_cpp_listed_device(str(binary))
    assert len(calls) == 2


def test_a_hardware_inventory_change_reprobes_the_listing(monkeypatch, tmp_path):
    # Same file, same env, but the eGPU was unplugged (or the driver reset): the old listing names a device that is gone.
    from utils.hardware import gpu_query, hardware

    binary = tmp_path / "sd-cli"
    binary.write_bytes(b"x")
    calls: list = []

    def _probe(_binary, *args):
        calls.append(args)
        return _LISTING

    monkeypatch.setattr(sd_cpp_backend, "_LAST_DEVICE_LISTING", {})
    monkeypatch.setattr(sd_cpp_backend, "_sd_cpp_probe_output", _probe)
    sd_cpp_backend.sd_cpp_listed_device(str(binary))
    sd_cpp_backend.sd_cpp_listed_device(str(binary))
    assert len(calls) == 1
    # An explicit re-detection (hotplug, driver reset).
    gpu_query.invalidate_static("test hotplug")
    sd_cpp_backend.sd_cpp_listed_device(str(binary))
    assert len(calls) == 2
    # A periodic inventory refresh that found a different device set.
    before = [{"vendor": "amd", "name": "AMD Radeon RX 7800 XT"}]
    monkeypatch.setattr(hardware, "_physical_gpu_inventory_cache", (0.0, {"devices": before}))
    monkeypatch.setattr(hardware, "_probe_physical_gpu_inventory", lambda: {"devices": []})
    hardware._run_physical_gpu_inventory_probe()
    sd_cpp_backend.sd_cpp_listed_device(str(binary))
    assert len(calls) == 3
    # A refresh that found the same devices keeps the listing.
    hardware._run_physical_gpu_inventory_probe()
    sd_cpp_backend.sd_cpp_listed_device(str(binary))
    assert len(calls) == 3


def test_an_unreadable_listing_is_not_cached(monkeypatch, tmp_path):
    binary = tmp_path / "sd-cli"
    binary.write_bytes(b"x")
    answers = [None, _LISTING]
    monkeypatch.setattr(sd_cpp_backend, "_LAST_DEVICE_LISTING", {})
    monkeypatch.setattr(sd_cpp_backend, "_sd_cpp_probe_output", lambda *_a: answers.pop(0))
    assert sd_cpp_backend.sd_cpp_device_name_for_ordinal(str(binary), 0) is None
    assert sd_cpp_backend.sd_cpp_device_name_for_ordinal(str(binary), 0) == "ROCm0"
