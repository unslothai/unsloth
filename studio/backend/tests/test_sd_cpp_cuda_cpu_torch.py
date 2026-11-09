# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""No GPU torch beside an NVIDIA card running llama.cpp's CUDA build gets the CUDA sd.cpp build (#16803)."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from core.inference import diffusion_engine_router as r
from core.inference import sd_cpp_backend
from core.inference import sd_cpp_engine
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
    "CUDA_VISIBLE_DEVICES",
)

_RTX = {"vendor": "nvidia", "index": 0, "name": "NVIDIA GeForce RTX 3060"}
_VEGA = {"vendor": "amd", "index": 0, "name": None, "source": "sysfs-drm"}


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
def _nvidia_host_with_cuda_llama(monkeypatch):
    for e in _ENVS:
        monkeypatch.delenv(e, raising = False)
    _platform(monkeypatch, "win32", "AMD64")
    monkeypatch.setattr(r, "_physical_inventory", lambda: _host(_RTX))
    # The Vulkan route is never taken beside NVIDIA; keep its own probes off the real host.
    monkeypatch.setattr(r, "_nvidia_adapter_present", lambda: True)
    _llama_marker(monkeypatch, "cuda13")


def _llama_marker(monkeypatch, runtime_line):
    import utils.llama_cpp_freshness as freshness
    monkeypatch.setattr(sd_cpp_engine, "_llama_server_binary", lambda: "/opt/llama/llama-server")
    monkeypatch.setattr(freshness, "read_install_marker", lambda _b: {"runtime_line": runtime_line})


def _empty_failure_records(monkeypatch, stored = None):
    monkeypatch.setattr(sd_cpp_backend, "_accelerator_runtime_failures", {})
    monkeypatch.setattr(
        sd_cpp_backend, "_stored_accelerator_runtime_failures", lambda: dict(stored or {})
    )
    monkeypatch.setattr(sd_cpp_backend, "_persist_accelerator_runtime_failures", lambda _r: None)
    monkeypatch.setattr(sd_cpp_backend, "_accelerator_fingerprint", lambda *_a: {"bundle": "t"})


@pytest.mark.parametrize("system,machine", [("win32", "AMD64"), ("linux", "x86_64")])
def test_a_cpu_torch_beside_cuda_llama_installs_the_cuda_build(monkeypatch, system, machine):
    _platform(monkeypatch, system, machine)
    assert r.image_install_accelerator("cpu") == "cuda"


def test_llama_cpp_without_cuda_keeps_cpu(monkeypatch):
    # A CPU or Vulkan llama.cpp proves nothing about the CUDA driver.
    _llama_marker(monkeypatch, "vulkan")
    assert r.image_install_accelerator("cpu") == "auto"


@pytest.mark.parametrize(
    "inventory",
    [
        _host(),
        _host(_VEGA),
        _host(unknown = True),
        _host(unanswered = ["nvidia"]),
    ],
)
def test_no_listed_nvidia_card_keeps_cpu(monkeypatch, inventory):
    monkeypatch.setattr(r, "_physical_inventory", lambda: inventory)
    assert r.image_install_accelerator("cpu") == "auto"


@pytest.mark.parametrize("value", ["", "-1", " -1 "])
def test_a_mask_hiding_every_card_keeps_cpu(monkeypatch, value):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", value)
    assert r.image_install_accelerator("cpu") == "auto"


def test_a_mask_naming_a_card_still_uses_it(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    assert r.image_install_accelerator("cpu") == "cuda"


@pytest.mark.parametrize(
    "system,machine", [("linux", "aarch64"), ("darwin", "arm64"), ("win32", "ARM64")]
)
def test_hosts_without_a_cuda_bundle_keep_cpu(monkeypatch, system, machine):
    _platform(monkeypatch, system, machine)
    assert r.image_install_accelerator("cpu") == "auto"


def test_the_fallback_knob_opts_out(monkeypatch):
    # Without the CPU rung a CUDA build that cannot run would leave the host with nothing.
    monkeypatch.setenv("UNSLOTH_DIFFUSION_SD_CPP_VULKAN_FALLBACK", "0")
    assert r.image_install_accelerator("cpu") == "auto"


@pytest.mark.parametrize(
    "backend,expected", [("cuda", "cuda"), ("rocm", "rocm"), ("xpu", "vulkan"), ("mps", "auto")]
)
def test_gpu_torch_backends_are_unchanged(backend, expected):
    assert r.image_install_accelerator(backend) == expected


def test_an_off_torch_card_still_wins(monkeypatch):
    monkeypatch.setenv("UNSLOTH_DIFFUSION_SD_CPP_DEVICE", "nvidia:0")
    assert r.image_install_accelerator("cpu") == "cuda"


def test_video_keeps_torchs_build():
    assert r._install_accelerator_for("cpu") == "auto"


def test_a_cuda_build_that_cannot_run_falls_to_cpu_only_on_a_cpu_torch(monkeypatch):
    _empty_failure_records(monkeypatch)
    monkeypatch.setattr(sd_cpp_backend, "_cuda_cpu_rung_applies", lambda: True)
    assert sd_cpp_backend.fallback_accelerator_for("cuda") == "cpu"
    assert sd_cpp_backend.preferred_accelerator("cuda") == "cuda"
    sd_cpp_backend.note_accelerator_runtime_failure("cuda", proven = True)
    assert sd_cpp_backend.preferred_accelerator("cuda") == "cpu"
    # A CUDA torch keeps diffusers on the card: the CPU rung is never its fallback.
    monkeypatch.setattr(sd_cpp_backend, "_cuda_cpu_rung_applies", lambda: False)
    assert sd_cpp_backend.fallback_accelerator_for("cuda") is None
    assert sd_cpp_backend.preferred_accelerator("cuda") == "cuda"


def test_a_cuda_torch_records_no_cuda_failure(monkeypatch):
    _empty_failure_records(monkeypatch)
    monkeypatch.setattr(sd_cpp_backend, "_cuda_cpu_rung_applies", lambda: False)
    sd_cpp_backend.note_accelerator_runtime_failure("cuda", proven = True)
    assert sd_cpp_backend._accelerator_runtime_failures == {}


def test_two_ambiguous_cuda_strikes_move_a_cpu_torch_host(monkeypatch):
    _empty_failure_records(monkeypatch)
    monkeypatch.setattr(sd_cpp_backend, "_cuda_cpu_rung_applies", lambda: True)
    sd_cpp_backend.note_accelerator_runtime_failure("cuda", proven = False)
    assert sd_cpp_backend.preferred_accelerator("cuda") == "cuda"
    sd_cpp_backend.note_accelerator_runtime_failure("cuda", proven = False)
    assert sd_cpp_backend.preferred_accelerator("cuda") == "cpu"


def test_a_cuda_build_the_loader_refuses_installs_the_cpu_one(monkeypatch):
    _empty_failure_records(monkeypatch)
    monkeypatch.setattr(sd_cpp_backend, "_cuda_cpu_rung_applies", lambda: True)
    inst = sd_cpp_backend._installer_module()
    calls: list[str] = []

    def _install(**kwargs):
        calls.append(kwargs["accelerator"])
        if kwargs["accelerator"] == "cuda":
            raise inst.UnrunnableBuildError(
                "STATUS_DLL_NOT_FOUND", accelerator = "cuda", asset = "x", tag = "t"
            )
        return "/opt/sd/sd-cli"

    monkeypatch.setattr(sd_cpp_engine, "host_cuda_runtime_major", lambda: "13")
    path, used = sd_cpp_backend._install_down_the_ladder(_install, "cuda")
    assert (path, used) == ("/opt/sd/sd-cli", "cpu")
    assert calls == ["cuda", "cpu"]
    assert sd_cpp_backend.preferred_accelerator("cuda") == "cpu"


def test_the_cuda_install_asks_for_llama_cpps_cuda_line(monkeypatch):
    monkeypatch.setattr(sd_cpp_engine, "host_cuda_runtime_major", lambda: "13")
    assert sd_cpp_backend._install_kwargs("cuda") == {"accelerator": "cuda", "cuda_major": "13"}


def test_host_cuda_runtime_major_prefers_llama_then_torch(monkeypatch):
    monkeypatch.setattr(sd_cpp_engine, "llama_cpp_cuda_major", lambda: "12")
    assert sd_cpp_engine.host_cuda_runtime_major() == "12"
    monkeypatch.setattr(sd_cpp_engine, "llama_cpp_cuda_major", lambda: None)
    fake_torch = SimpleNamespace(version = SimpleNamespace(cuda = "13.0"))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    assert sd_cpp_engine.host_cuda_runtime_major() == "13"


def test_llama_cpp_cuda_major_reads_the_install_marker(monkeypatch):
    markers = {"cuda13": "13", "cuda12": "12", "vulkan": None, "cpu": None}
    for line, expected in markers.items():
        _llama_marker(monkeypatch, line)
        assert sd_cpp_engine.llama_cpp_cuda_major() == expected
    monkeypatch.setattr(sd_cpp_engine, "_llama_server_binary", lambda: None)
    assert sd_cpp_engine.llama_cpp_cuda_major() is None


def test_selection_installs_the_cuda_build(monkeypatch):
    monkeypatch.setattr(
        r, "resolve_diffusion_device_target", lambda: SimpleNamespace(backend = "cpu", device = "cpu")
    )
    monkeypatch.setattr(r, "_off_torch_warned", set())
    monkeypatch.setattr(sd_cpp_backend, "_installed_accelerator_of", lambda _b: "cuda")
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
    assert asked and set(asked) == {"cuda"}


_PINNED_RELEASE_ASSETS = [
    "sd-master-982-0518b34-uba6ac30-bin-Linux-Ubuntu-22.04-x86_64-cuda12.zip",
    "sd-master-982-0518b34-uba6ac30-bin-Linux-Ubuntu-22.04-x86_64-cuda13.zip",
    "sd-master-982-0518b34-uba6ac30-bin-Linux-Ubuntu-22.04-x86_64-vulkan.zip",
    "sd-master-982-0518b34-uba6ac30-bin-Linux-Ubuntu-22.04-x86_64.zip",
    "sd-master-982-0518b34-uba6ac30-bin-Linux-Ubuntu-24.04-aarch64.zip",
    "sd-master-982-0518b34-uba6ac30-bin-win-cpu-x64.zip",
    "sd-master-982-0518b34-uba6ac30-bin-win-cuda12-x64.zip",
    "sd-master-982-0518b34-uba6ac30-bin-win-cuda13-x64.zip",
    "sd-master-982-0518b34-uba6ac30-bin-win-vulkan-x64.zip",
]


@pytest.mark.parametrize(
    "system,machine,major,expected",
    [
        ("Windows", "AMD64", "13", "sd-master-982-0518b34-uba6ac30-bin-win-cuda13-x64.zip"),
        ("Windows", "AMD64", "12", "sd-master-982-0518b34-uba6ac30-bin-win-cuda12-x64.zip"),
        (
            "Linux",
            "x86_64",
            "13",
            "sd-master-982-0518b34-uba6ac30-bin-Linux-Ubuntu-22.04-x86_64-cuda13.zip",
        ),
        ("Linux", "aarch64", "13", None),
    ],
)
def test_the_pinned_release_has_the_cuda_bundle_it_routes_to(system, machine, major, expected):
    inst = sd_cpp_backend._installer_module()
    assert (
        inst.resolve_release_asset(
            _PINNED_RELEASE_ASSETS,
            system = system,
            machine = machine,
            accelerator = "cuda",
            cuda_major = major,
        )
        == expected
    )


def test_a_cuda_build_that_cannot_start_steps_down_within_the_same_load(monkeypatch):
    """Without torch nothing but sd.cpp renders, so the first load must not end on diffusers."""
    _empty_failure_records(monkeypatch)
    monkeypatch.setattr(sd_cpp_backend, "_cuda_cpu_rung_applies", lambda: True)
    monkeypatch.setattr(
        r, "resolve_diffusion_device_target", lambda: SimpleNamespace(backend = "cpu", device = "cpu")
    )
    monkeypatch.setattr(r, "_off_torch_warned", set())
    monkeypatch.setattr(
        sd_cpp_backend, "_installed_accelerator_of", lambda b: b.split("/")[2] if b else None
    )
    monkeypatch.setattr(
        r,
        "get_active_diffusion_engine",
        lambda: SimpleNamespace(
            status = lambda: {"loaded": False, "repo_id": None}, unload = lambda: None
        ),
    )
    monkeypatch.setattr(r, "_server_binary_runnable", lambda b, *_a, **_k: "/cuda/" not in b)
    monkeypatch.setattr(
        r,
        "SdCppEngine",
        lambda binary = None, **_: SimpleNamespace(
            version = lambda: None if "/cuda/" in binary else "sd-cli v0"
        ),
    )
    asked: list[str] = []

    def _ensure(stem):
        def _inner(**kwargs):
            asked.append(kwargs["accelerator"])
            klass = "cpu" if kwargs["accelerator"] in ("auto", "cpu") else kwargs["accelerator"]
            return f"/sd/{klass}/{stem}"

        return _inner

    monkeypatch.setattr(r, "ensure_sd_server_binary", _ensure("sd-server"))
    monkeypatch.setattr(r, "ensure_sd_cpp_binary", _ensure("sd-cli"))
    saved = (r._active_engine_name, r._fallback_reason)
    try:
        r.select_and_activate_engine(detect_family("z-image"))
        assert r.active_engine_name() == ENGINE_SD_CPP
    finally:
        r._active_engine_name, r._fallback_reason = saved
    assert asked[:2] == ["cuda", "cuda"]
    assert asked[2:] and set(asked[2:]) == {"cpu"}
    # Proven on the spot: the next load starts at the CPU build.
    assert sd_cpp_backend.preferred_accelerator("cuda") == "cpu"
