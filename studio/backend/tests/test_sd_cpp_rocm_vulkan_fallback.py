# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The ROCm -> Vulkan fallback rung for the native stable-diffusion.cpp bundle (#9278, #8814)."""

from __future__ import annotations

import inspect
import threading
import types
from pathlib import Path

import pytest

from core.inference.video import VideoBackend, _detect_load_family
from core.inference import sd_cpp_backend as sd_backend

H3_REPO = "leejet/MiniMax-H3-GGUF"
H3_FILE = "minimax_h3_fl2va-Q4_K_M.gguf"

_BANNER = "stable-diffusion.cpp version unknown, commit unknown\n"
_H3_HELP = _BANNER + "  --ref-video   MiniMax-H3 Ref2VA reference video frame directory\n"

# The ROCm build on an unsupported card does not answer at all, hence the raw verdict.
_DEVICES_ROCM = "ROCm0\tAMD Radeon RX 7900 XTX\nCPU\tAMD Ryzen 9\n"
_DEVICES_VULKAN = "Vulkan0\tAMD Radeon RX 7900 XTX\nCPU\tAMD Ryzen 9\n"
_DEVICES_CPU_ONLY = "CPU\tAMD Ryzen 9\n"
# MISSING: no such build here. None: installed, cannot be asked. A string: its --list-devices answer.
MISSING = object()

PLATFORMS = ["linux", "wsl", "win32"]

ALL_PLATFORMS = ["linux", "wsl", "win32", "darwin"]

# On macOS the vendor axis collapses: no CUDA, no ROCm, the GPU backend is Metal.
_BACKEND_FOR = {
    "darwin": {"nvidia": "mps", "amd": "mps", "cpu_only": "cpu"},
    "other": {"nvidia": "cuda", "amd": "rocm", "cpu_only": "cpu"},
}
_DEVICE_FOR = {"mps": "mps", "cpu": "cpu", "cuda": "cuda", "rocm": "cuda"}


def _corner(platform: str, vendor: str) -> tuple[str, str]:
    backend = _BACKEND_FOR["darwin" if platform == "darwin" else "other"][vendor]
    return backend, _DEVICE_FOR[backend]


# accelerator_class folds the "auto" default onto the plain build it names "cpu".
_FIRST_ENSURE = {"cuda": "cuda", "rocm": "rocm", "mps": "cpu", "cpu": "cpu"}

_DEVICES_CUDA = "CUDA0\tNVIDIA GeForce RTX 4090\nCPU\tIntel Core i9\n"


def _devices_for(backend: str, *, rocm_runs: bool) -> dict:
    """A Vulkan build is offered in EVERY corner on purpose: it must be shown untaken everywhere but one."""
    if backend == "cuda":
        return {"cuda": _DEVICES_CUDA, "vulkan": _DEVICES_VULKAN, "cpu": _DEVICES_CPU_ONLY}
    if backend == "rocm":
        return dict(_ROCM_WORKS) if rocm_runs else dict(_ROCM_BROKEN)
    return {"cpu": _DEVICES_CPU_ONLY, "vulkan": _DEVICES_VULKAN}


def _noted_accelerators(store: dict) -> list:
    records = store.get("sd_cpp_accelerator_runtime_failures") or {}
    return sorted(
        k
        for k, v in records.items()
        if isinstance(v, dict) and (v.get("proven") or v.get("strikes", 0) >= 2)
    )


def _recorded_strikes(store: dict, klass: str = "rocm") -> int:
    record = (store.get("sd_cpp_accelerator_runtime_failures") or {}).get(klass) or {}
    return int(record.get("strikes", 0))


class _PlanInfo:
    def __init__(self, siblings) -> None:
        self.siblings = siblings


class _Engine:
    def __init__(self, binary) -> None:
        self.binary = binary

    def version(self):
        return "stub-version"


@pytest.fixture
def fake_settings(monkeypatch):
    from storage import studio_db

    store: dict = {}

    def _get(key, fallback = None):
        return store.get(key, fallback)

    def _upsert(settings, **_kwargs):
        store.update(settings)
        return dict(store)

    monkeypatch.setattr(studio_db, "get_app_setting", _get)
    monkeypatch.setattr(studio_db, "upsert_app_settings", _upsert)
    return store


@pytest.fixture(autouse = True)
def _clean_process_state(monkeypatch):
    """raising = False so these still report what they measured on a tree without the note."""
    from core.inference import sd_cpp_backend

    monkeypatch.setattr(sd_cpp_backend, "_accelerator_runtime_failures", {}, raising = False)
    monkeypatch.setattr(
        sd_cpp_backend,
        "_HOST_FINGERPRINT_MEMO",
        {"runtime": "6.4.0", "gpus": ["AMD Radeon RX 7900 XTX"]},
        raising = False,
    )
    monkeypatch.delenv("UNSLOTH_DIFFUSION_SD_CPP_VULKAN_FALLBACK", raising = False)


@pytest.fixture
def h3_amd_host(monkeypatch, tmp_path):
    """``devices``: accelerator class -> ``--list-devices`` text, MISSING for "no such build here", None for "cannot be asked"."""
    from core.inference import sd_cpp_backend, sd_cpp_engine
    from core.inference import video as video_mod

    def _setup(*, platform: str, backend: str, device: str, devices: dict):
        monkeypatch.setattr(
            sd_cpp_engine.sys, "platform", "linux" if platform == "wsl" else platform
        )
        monkeypatch.setattr(
            video_mod,
            "resolve_diffusion_device_target",
            lambda: types.SimpleNamespace(backend = backend, device = device, dtype = None),
        )
        monkeypatch.setattr(sd_cpp_backend, "_install_allowed", lambda: True)
        monkeypatch.setattr(sd_cpp_backend, "is_managed_binary", lambda _b: True)
        monkeypatch.setattr(sd_cpp_engine, "SdCppEngine", _Engine)

        ensured: list[str] = []
        by_binary = {f"/opt/sd/{accel}/sd-cli": accel for accel in devices}

        def _ensure(*, allow_install = True, accelerator = "cpu"):
            klass = "cpu" if accelerator in ("auto", "cpu", "", None) else accelerator
            ensured.append(klass)
            if devices.get(klass, MISSING) is MISSING:
                return None
            return f"/opt/sd/{klass}/sd-cli"

        monkeypatch.setattr(sd_cpp_backend, "ensure_sd_cpp_binary", _ensure)

        def _probe(binary, *args):
            if args == ("--list-devices",):
                answer = devices.get(by_binary.get(binary, ""), MISSING)
                return None if answer is MISSING else answer
            return _H3_HELP

        monkeypatch.setattr(sd_cpp_backend, "_sd_cpp_probe_output", _probe)
        monkeypatch.setattr(
            sd_cpp_backend, "_installed_accelerator_of", lambda b: by_binary.get(b or "")
        )

        class _Api:
            def __init__(self, **_kwargs):
                pass

            def model_info(self, repo, *_args, **_kwargs):
                return _PlanInfo([])

        monkeypatch.setattr("huggingface_hub.HfApi", _Api)

        downloads: list[str] = []

        def _download(_repo, wanted, *_args, **_kwargs):
            downloads.append(wanted)
            path = tmp_path / Path(wanted).name
            path.write_bytes(b"x")
            return str(path)

        monkeypatch.setattr("utils.hf_xet_fallback.hf_hub_download_with_xet_fallback", _download)

        def run():
            fam = _detect_load_family(H3_REPO, None, "minimax-h3")
            assert fam is not None
            obj = VideoBackend()
            obj._run_load_h3_native(
                fam = fam,
                token = None,
                cancel_event = threading.Event(),
                repo_id = H3_REPO,
                gguf_filename = H3_FILE,
            )
            return obj

        return types.SimpleNamespace(run = run, ensured = ensured, downloads = downloads)

    return _setup


_ROCM_WORKS = {"rocm": _DEVICES_ROCM, "vulkan": _DEVICES_VULKAN, "cpu": _DEVICES_CPU_ONLY}
# #8814 / #9278: the ROCm build is installed and simply cannot be asked anything.
_ROCM_BROKEN = {"rocm": None, "vulkan": _DEVICES_VULKAN, "cpu": _DEVICES_CPU_ONLY}
_ROCM_CPU_ONLY = {"rocm": _DEVICES_CPU_ONLY, "vulkan": _DEVICES_VULKAN, "cpu": _DEVICES_CPU_ONLY}
_VULKAN_ONLY = {"rocm": MISSING, "vulkan": _DEVICES_VULKAN, "cpu": _DEVICES_CPU_ONLY}
_CPU_ONLY = {"rocm": MISSING, "vulkan": MISSING, "cpu": _DEVICES_CPU_ONLY}


@pytest.mark.parametrize("platform", PLATFORMS)
def test_a_working_rocm_build_is_left_alone(h3_amd_host, fake_settings, platform):
    host = h3_amd_host(platform = platform, backend = "rocm", device = "cuda", devices = _ROCM_WORKS)
    backend_obj = host.run()
    assert host.ensured == ["rocm"]
    assert backend_obj._state.device == "cuda"
    assert fake_settings == {}


@pytest.mark.parametrize("platform", PLATFORMS)
@pytest.mark.parametrize(
    "state,devices,diverts",
    [
        ("rocm_unrunnable", _ROCM_BROKEN, False),
        ("rocm_cpu_only", _ROCM_CPU_ONLY, True),
        # The ensure answers None for a failed download too, and a record with no binary carries no bundle tag to retire: a strike, not proof.
        ("no_rocm_asset", _VULKAN_ONLY, False),
    ],
)
def test_a_rocm_build_that_cannot_run_falls_back_to_vulkan(
    h3_amd_host, fake_settings, platform, state, devices, diverts
):
    """On main every one of these commits ``native_device = "cpu"`` or refuses; both reporters' cards run Vulkan."""
    host = h3_amd_host(platform = platform, backend = "rocm", device = "cuda", devices = devices)
    backend_obj = host.run()
    assert host.ensured == ["rocm", "vulkan"], host.ensured
    assert backend_obj._state.device == "cuda"
    assert len(host.downloads) == 4
    assert _noted_accelerators(fake_settings) == (["rocm"] if diverts else [])
    assert _recorded_strikes(fake_settings) == 1


@pytest.mark.parametrize("platform", PLATFORMS)
def test_the_cpu_rung_is_still_reached_when_vulkan_cannot_run_either(
    h3_amd_host, fake_settings, platform
):
    host = h3_amd_host(platform = platform, backend = "rocm", device = "cuda", devices = _CPU_ONLY)
    backend_obj = host.run()
    assert host.ensured == ["rocm", "vulkan", "cpu"], host.ensured
    assert backend_obj._state.device == "cpu"
    assert fake_settings == {}


@pytest.mark.parametrize("platform", PLATFORMS)
def test_a_cuda_host_never_takes_the_vulkan_rung(h3_amd_host, fake_settings, platform):
    host = h3_amd_host(
        platform = platform,
        backend = "cuda",
        device = "cuda",
        devices = {"cuda": _DEVICES_CPU_ONLY, "vulkan": _DEVICES_VULKAN, "cpu": _DEVICES_CPU_ONLY},
    )
    backend_obj = host.run()
    assert host.ensured == ["cuda", "cpu"], host.ensured
    assert backend_obj._state.device == "cpu"
    assert fake_settings == {}


@pytest.mark.parametrize("platform", PLATFORMS)
def test_the_fallback_can_be_switched_off(h3_amd_host, fake_settings, monkeypatch, platform):
    monkeypatch.setenv("UNSLOTH_DIFFUSION_SD_CPP_VULKAN_FALLBACK", "0")
    host = h3_amd_host(platform = platform, backend = "rocm", device = "cuda", devices = _ROCM_BROKEN)
    backend_obj = host.run()
    assert "vulkan" not in host.ensured
    assert backend_obj._state.device == "cuda"
    assert fake_settings == {}


@pytest.mark.parametrize("platform", PLATFORMS)
def test_a_recorded_failure_skips_the_rocm_build_on_the_next_load(
    h3_amd_host, fake_settings, platform
):
    fake_settings["sd_cpp_accelerator_runtime_failures"] = ["rocm"]
    host = h3_amd_host(platform = platform, backend = "rocm", device = "cuda", devices = _ROCM_WORKS)
    backend_obj = host.run()
    assert host.ensured == ["vulkan"], host.ensured
    assert backend_obj._state.device == "cuda"


def test_clearing_the_note_restores_the_host_accelerator(fake_settings):
    from core.inference import sd_cpp_backend

    sd_cpp_backend.note_accelerator_runtime_failure("rocm")
    assert sd_cpp_backend.accelerator_runtime_failed("rocm") is True
    assert sd_cpp_backend.preferred_accelerator("rocm") == "vulkan"
    sd_cpp_backend.clear_accelerator_runtime_failures()
    assert sd_cpp_backend.accelerator_runtime_failed("rocm") is False
    assert sd_cpp_backend.preferred_accelerator("rocm") == "rocm"


@pytest.mark.parametrize(
    "accelerator,expected",
    [("rocm", "vulkan"), ("vulkan", None), ("cuda", None), ("cpu", None), ("auto", None)],
)
def test_only_rocm_has_a_rung_below_it(accelerator, expected):
    from core.inference.sd_cpp_backend import fallback_accelerator_for
    assert fallback_accelerator_for(accelerator) == expected


@pytest.mark.parametrize(
    "output,expected",
    [
        ("ROCm error: CUBLAS_STATUS_INVALID_VALUE at hipblasSetStream (ggml-cuda.cu:1679)", True),
        ("ggml_cuda_mul_mat_q: unspecified launch failure at mmq.cu:145", True),
        ("hipErrorNoBinaryForGpu: Unable to find code object for all current devices", True),
        # #8814's line is a WARNING from builds that then render fine, so it must not count.
        ("Warning: Attempting to use CK on an unsupported architecture!", False),
        ("sd-cli exited 1. Last output:\nerror: failed to open prompt file", False),
        ("", False),
        (None, False),
    ],
)
def test_only_a_gpu_backend_failure_moves_the_preference(output, expected):
    from core.inference.sd_cpp_backend import output_shows_accelerator_failure
    assert output_shows_accelerator_failure(output) is expected


def test_a_generation_that_dies_in_hipblas_records_the_failure(fake_settings, monkeypatch):
    from core.inference import sd_cpp_backend
    from core.inference import video as video_mod

    monkeypatch.setattr(sd_cpp_backend, "_installed_accelerator_of", lambda _b: "rocm")
    video_mod._note_sd_cpp_accelerator_failure(
        "/opt/sd/rocm/sd-cli",
        "sd-cli exited 1. Last output:\nROCm error: CUBLAS_STATUS_INVALID_VALUE at hipblasSetStream",
    )
    assert _noted_accelerators(fake_settings) == ["rocm"]
    assert sd_cpp_backend.preferred_accelerator("rocm") == "vulkan"


def test_an_ordinary_generation_failure_records_nothing(fake_settings, monkeypatch):
    from core.inference import sd_cpp_backend
    from core.inference import video as video_mod

    monkeypatch.setattr(sd_cpp_backend, "_installed_accelerator_of", lambda _b: "rocm")
    video_mod._note_sd_cpp_accelerator_failure(
        "/opt/sd/rocm/sd-cli", "sd-cli exited 1. Last output:\nggml_new_object: not enough space"
    )
    assert fake_settings == {}
    assert sd_cpp_backend.preferred_accelerator("rocm") == "rocm"


def test_the_engine_router_installs_the_preferred_accelerator(fake_settings, monkeypatch):
    from core.inference import diffusion_engine_router as router

    fake_settings["sd_cpp_accelerator_runtime_failures"] = ["rocm"]
    asked: list[str] = []

    monkeypatch.setattr(router, "_install_allowed", lambda: True)
    monkeypatch.setattr(
        router,
        "ensure_sd_server_binary",
        lambda **kwargs: asked.append(kwargs["accelerator"]) or None,
    )
    monkeypatch.setattr(
        router,
        "ensure_sd_cpp_binary",
        lambda **kwargs: asked.append(kwargs["accelerator"]) or None,
    )
    monkeypatch.setattr(
        router,
        "resolve_diffusion_device_target",
        lambda: types.SimpleNamespace(backend = "rocm", device = "cuda", dtype = None),
    )
    monkeypatch.setattr(router, "family_sd_cpp_supported", lambda _fam: True)
    monkeypatch.setattr(router, "_activate", lambda name, reason = None: name)
    monkeypatch.setenv("UNSLOTH_DIFFUSION_ENGINE", "sd_cpp")
    router.select_and_activate_engine(_detect_load_family(H3_REPO, None, "minimax-h3"))
    assert asked == ["vulkan", "vulkan"], asked


@pytest.mark.parametrize("platform", ALL_PLATFORMS)
@pytest.mark.parametrize("vendor", ["nvidia", "amd", "cpu_only"])
def test_every_healthy_corner_installs_exactly_what_it_did_before(
    h3_amd_host, fake_settings, platform, vendor
):
    backend, device = _corner(platform, vendor)
    host = h3_amd_host(
        platform = platform,
        backend = backend,
        device = device,
        devices = _devices_for(backend, rocm_runs = True),
    )
    backend_obj = host.run()
    assert host.ensured == [_FIRST_ENSURE[backend]], host.ensured
    assert backend_obj._state.device == device
    assert fake_settings == {}


@pytest.mark.parametrize("platform", ALL_PLATFORMS)
@pytest.mark.parametrize("vendor", ["nvidia", "amd", "cpu_only"])
def test_only_the_amd_corners_with_a_rocm_asset_take_the_new_rung(
    h3_amd_host, fake_settings, platform, vendor
):
    backend, device = _corner(platform, vendor)
    host = h3_amd_host(
        platform = platform,
        backend = backend,
        device = device,
        devices = _devices_for(backend, rocm_runs = False),
    )
    backend_obj = host.run()
    takes_the_rung = backend == "rocm"
    if takes_the_rung:
        assert host.ensured == ["rocm", "vulkan"], host.ensured
        assert _noted_accelerators(fake_settings) == []
        assert _recorded_strikes(fake_settings) == 1
    else:
        assert host.ensured == [_FIRST_ENSURE[backend]], host.ensured
        assert fake_settings == {}
    assert backend_obj._state.device == device


@pytest.mark.parametrize("platform", ALL_PLATFORMS)
def test_the_release_assets_each_platform_resolves_are_unchanged(platform):
    import importlib.util
    import sys as _sys
    from pathlib import Path as _Path

    root = _Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location(
        "_r3c_install_sd_cpp_prebuilt", root / "install_sd_cpp_prebuilt.py"
    )
    module = importlib.util.module_from_spec(spec)
    _sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    assets = [
        "sd-master-linux-x64.zip",
        "sd-master-linux-x64-rocm.zip",
        "sd-master-linux-x64-vulkan.zip",
        "sd-master-bin-win-avx2-x64.zip",
        "sd-master-bin-win-cuda12-x64.zip",
        "sd-master-bin-win-rocm-x64.zip",
        "sd-master-bin-win-vulkan-x64.zip",
        "sd-master-bin-macos-arm64.zip",
    ]
    system = {"linux": "Linux", "wsl": "Linux", "win32": "Windows", "darwin": "Darwin"}[platform]
    machine = "arm64" if platform == "darwin" else "x86_64"

    def resolved(accel):
        return module.resolve_release_asset(
            assets, system = system, machine = machine, accelerator = accel
        )

    if platform == "darwin":
        # One asset, whatever is asked for, so macOS has no rung to take.
        assert resolved("auto") == "sd-master-bin-macos-arm64.zip"
        assert resolved("rocm") == "sd-master-bin-macos-arm64.zip"
        assert resolved("vulkan") == "sd-master-bin-macos-arm64.zip"
        return
    assert resolved("rocm") is not None
    assert resolved("vulkan") is not None
    assert resolved("rocm") != resolved("vulkan")
    if platform == "win32":
        assert resolved("cuda") == "sd-master-bin-win-cuda12-x64.zip"
        assert resolved("auto") == "sd-master-bin-win-avx2-x64.zip"
    else:
        # Upstream publishes no Linux CUDA archive, which is why the CPU rung exists at all. Unchanged here.
        assert resolved("cuda") is None
        assert resolved("auto") == "sd-master-linux-x64.zip"


@pytest.mark.parametrize(
    "backend,noted,expected",
    [
        ("rocm", ["rocm"], "vulkan"),
        ("rocm", [], "rocm"),
        ("cuda", ["rocm"], "cuda"),
        ("xpu", ["rocm"], "vulkan"),  # Intel already installs the vulkan build; nothing moves.
        ("mps", ["rocm"], "auto"),
        ("cpu", ["rocm"], "auto"),
    ],
)
def test_the_image_path_resolves_the_preferred_accelerator(
    fake_settings, monkeypatch, backend, noted, expected
):
    """The single funnel for the image path's ensures: ``_accelerator_changed`` compares the tree against this answer, so a call site that disagreed would reinstall on every load."""
    from core.inference import sd_cpp_backend

    fake_settings["sd_cpp_accelerator_runtime_failures"] = list(noted)
    monkeypatch.setattr(
        sd_cpp_backend,
        "resolve_diffusion_device_target",
        lambda: types.SimpleNamespace(backend = backend, device = "cuda", dtype = None),
    )
    assert sd_cpp_backend.SdCppDiffusionBackend._resolved_accelerator() == expected


def test_a_single_ambiguous_failure_never_diverts_a_working_rocm_host(fake_settings, monkeypatch):
    """ "hip error" / "unspecified launch failure" also come out of a driver reset, so one must leave a working host alone."""
    from core.inference import sd_cpp_backend
    from core.inference import video as video_mod

    monkeypatch.setattr(sd_cpp_backend, "_installed_accelerator_of", lambda _b: "rocm")
    video_mod._note_sd_cpp_accelerator_failure(
        "/opt/sd/rocm/sd-cli", "sd-cli exited 1. Last output:\nHIP error: out of memory"
    )
    assert sd_cpp_backend.accelerator_runtime_failed("rocm") is False
    assert sd_cpp_backend.preferred_accelerator("rocm") == "rocm"
    assert _noted_accelerators(fake_settings) == []


def test_a_second_ambiguous_failure_does_divert(fake_settings, monkeypatch):
    from core.inference import sd_cpp_backend
    from core.inference import video as video_mod

    monkeypatch.setattr(sd_cpp_backend, "_installed_accelerator_of", lambda _b: "rocm")
    for _ in range(2):
        video_mod._note_sd_cpp_accelerator_failure(
            "/opt/sd/rocm/sd-cli",
            "sd-cli exited 1. Last output:\nggml_cuda_mul_mat_q: unspecified launch failure",
        )
    assert sd_cpp_backend.accelerator_runtime_failed("rocm") is True
    assert sd_cpp_backend.preferred_accelerator("rocm") == "vulkan"


@pytest.mark.parametrize(
    "output,decisive",
    [
        ("ROCm error: CUBLAS_STATUS_INVALID_VALUE at hipblasSetStream", True),
        ("hipErrorNoBinaryForGpu: Unable to find code object for all current devices", True),
        ("no kernel image is available for execution on the device", True),
        ("invalid device function", True),
        ("ggml_cuda_mul_mat_q: unspecified launch failure at mmq.cu:145", False),
        ("ROCm error: something went wrong", False),
        # The same defect class printed by a layer that does not name the build, but both also have mundane causes, so both are only counted.
        ("rocBLAS error: Could not initialize Tensile host: No devices found", False),
        (
            "Memory access fault by GPU node-1 (Agent handle: 0x55d) on address 0x7f18. "
            "Reason: Page not present or supervisor privilege.",
            False,
        ),
    ],
)
def test_the_marker_tiers_split_evidence_from_suspicion(output, decisive):
    from core.inference.sd_cpp_backend import (
        output_shows_accelerator_failure,
        output_shows_decisive_accelerator_failure,
    )
    assert output_shows_accelerator_failure(output) is True
    assert output_shows_decisive_accelerator_failure(output) is decisive


@pytest.mark.parametrize(
    "output",
    [
        "sd-cli exited 1. Last output:\nrocBLAS error: Could not initialize Tensile host: No devices found",
        "sd-cli exited 134. Last output:\nMemory access fault by GPU node-1 (Agent handle: 0x55d) "
        "on address 0x7f18. Reason: Page not present or supervisor privilege.",
    ],
)
def test_the_rocm_failures_that_name_no_build_are_counted_not_ignored(
    fake_settings, monkeypatch, output
):
    """Neither names the build, so neither matched any marker; both also occur on hosts whose ROCm is fine."""
    from core.inference import sd_cpp_backend
    from core.inference import video as video_mod

    monkeypatch.setattr(sd_cpp_backend, "_installed_accelerator_of", lambda _b: "rocm")
    video_mod._note_sd_cpp_accelerator_failure("/opt/sd/rocm/sd-cli", output)
    assert sd_cpp_backend.accelerator_runtime_failed("rocm") is False
    assert sd_cpp_backend.preferred_accelerator("rocm") == "rocm"

    video_mod._note_sd_cpp_accelerator_failure("/opt/sd/rocm/sd-cli", output)
    assert sd_cpp_backend.accelerator_runtime_failed("rocm") is True
    assert sd_cpp_backend.preferred_accelerator("rocm") == "vulkan"


@pytest.mark.parametrize(
    "output",
    [
        "ROCm error: out of memory",
        "HIP error: out of memory",
        "hipErrorOutOfMemory",
        "ggml_backend_cuda_buffer_type_alloc_buffer: failed to allocate 4096 MiB",
        "sd-cli exited 1. Last output:\nnot enough memory to allocate the compute buffer",
        "hipErrorNoBinaryForGpu reported while out of memory",
        # The wordings are the ones this repository's own OOM classifier in `utils.utils` already recognises.
        "sd-cli exited 1. Last output:\nROCm error: CUBLAS_STATUS_ALLOC_FAILED",
        "sd-cli exited 1. Last output:\nrocBLAS error: memory allocation failed",
        "sd-cli exited 1. Last output:\nhipMalloc: cannot allocate memory",
        "sd-cli exited 1. Last output:\nggml_backend_alloc_ctx_tensors: allocation failure",
        "sd-cli exited 1. Last output:\nROCm error: out of device memory",
    ],
)
def test_an_exhausted_card_is_never_read_as_an_unusable_build(output):
    """An OOM is a statement about the REQUEST, not about the build, and no fingerprint change can ever retire a note written from one."""
    from core.inference.sd_cpp_backend import (
        output_shows_accelerator_failure,
        output_shows_capacity_failure,
        output_shows_decisive_accelerator_failure,
    )

    assert output_shows_capacity_failure(output) is True
    assert output_shows_accelerator_failure(output) is False
    assert output_shows_decisive_accelerator_failure(output) is False


def test_repeated_out_of_memory_never_diverts_the_host(fake_settings, monkeypatch):
    from core.inference import sd_cpp_backend
    from core.inference import video as video_mod

    monkeypatch.setattr(sd_cpp_backend, "_installed_accelerator_of", lambda _b: "rocm")
    for _ in range(sd_cpp_backend._AMBIGUOUS_FAILURE_STRIKES + 2):
        video_mod._note_sd_cpp_accelerator_failure(
            "/opt/sd/rocm/sd-cli",
            "sd-cli exited 1. Last output:\nROCm error: out of memory",
        )

    assert sd_cpp_backend.accelerator_runtime_failed("rocm") is False
    assert sd_cpp_backend.preferred_accelerator("rocm") == "rocm"
    assert _noted_accelerators(fake_settings) == []


def test_one_decisive_failure_is_enough(fake_settings, monkeypatch):
    from core.inference import sd_cpp_backend
    from core.inference import video as video_mod

    monkeypatch.setattr(sd_cpp_backend, "_installed_accelerator_of", lambda _b: "rocm")
    video_mod._note_sd_cpp_accelerator_failure(
        "/opt/sd/rocm/sd-cli",
        "sd-cli exited 1. Last output:\nROCm error: CUBLAS_STATUS_INVALID_VALUE at hipblasSetStream",
    )
    assert sd_cpp_backend.accelerator_runtime_failed("rocm") is True
    assert _noted_accelerators(fake_settings) == ["rocm"]


@pytest.mark.parametrize(
    "changed,still_applies",
    [
        ({}, True),
        # torch.version.hip is the ROCm the torch WHEEL was built against, so a driver-only upgrade does not move it.
        ({"runtime": "7.0.0"}, False),
        ({"bundle": "master-b9999"}, False),
        ({"gpus": ["AMD Radeon RX 9070 XT"]}, False),
        # Unknown on either side is NOT a mismatch: these components are best-effort reads.
        ({"runtime": None}, True),
        ({"gpus": None}, True),
    ],
)
def test_a_record_is_retired_by_the_things_it_is_a_fact_about(changed, still_applies):
    from core.inference.sd_cpp_backend import _fingerprint_still_applies

    stored = {"bundle": "master-b1000", "runtime": "6.4.0", "gpus": ["AMD Radeon RX 7900 XTX"]}
    current = dict(stored)
    current.update(changed)
    assert _fingerprint_still_applies(stored, current) is still_applies


def test_a_driver_upgrade_makes_the_host_try_rocm_again(fake_settings, monkeypatch):
    from core.inference import sd_cpp_backend

    sd_cpp_backend.note_accelerator_runtime_failure("rocm")
    assert sd_cpp_backend.preferred_accelerator("rocm") == "vulkan"

    monkeypatch.setattr(
        sd_cpp_backend,
        "_HOST_FINGERPRINT_MEMO",
        {"runtime": "7.0.0", "gpus": ["AMD Radeon RX 7900 XTX"]},
        raising = False,
    )
    # The in-process mirror is keyed on the old fingerprint too, so both halves have to retire.
    assert sd_cpp_backend.accelerator_runtime_failed("rocm") is False
    assert sd_cpp_backend.preferred_accelerator("rocm") == "rocm"


def test_strikes_do_not_accumulate_across_a_fingerprint_change(fake_settings, monkeypatch):
    from core.inference import sd_cpp_backend

    sd_cpp_backend.note_accelerator_runtime_failure("rocm", proven = False)
    monkeypatch.setattr(
        sd_cpp_backend,
        "_HOST_FINGERPRINT_MEMO",
        {"runtime": "7.0.0", "gpus": ["AMD Radeon RX 7900 XTX"]},
        raising = False,
    )
    sd_cpp_backend.note_accelerator_runtime_failure("rocm", proven = False)
    assert sd_cpp_backend.accelerator_runtime_failed("rocm") is False


def test_the_settings_route_reports_and_clears_the_record(fake_settings, monkeypatch):
    """This preference lives in settings, not in the managed tree, so reinstalling does not clear it."""
    from core.inference import sd_cpp_backend
    from routes import settings as settings_routes

    sd_cpp_backend.note_accelerator_runtime_failure("rocm")
    state = settings_routes._diffusion_accelerator_fallback_response()
    assert state.diverting is True
    assert [r.accelerator for r in state.records] == ["rocm"]
    assert state.records[0].fallback == "vulkan"
    assert state.records[0].proven is True

    cleared = settings_routes.clear_diffusion_accelerator_fallback(current_subject = "owner")
    assert cleared.diverting is False
    assert cleared.records == []
    assert sd_cpp_backend.accelerator_runtime_failed("rocm") is False
    assert sd_cpp_backend.preferred_accelerator("rocm") == "rocm"
    assert _noted_accelerators(fake_settings) == []


def test_the_report_does_not_claim_a_diversion_the_switch_turned_off(fake_settings, monkeypatch):
    """With the switch off ``preferred_accelerator`` leaves ROCm selected however many strikes stand, so reporting the record's own verdict told that host its loads were redirected."""
    from core.inference import sd_cpp_backend
    from routes import settings as settings_routes

    sd_cpp_backend.note_accelerator_runtime_failure("rocm")
    monkeypatch.setenv("UNSLOTH_DIFFUSION_SD_CPP_VULKAN_FALLBACK", "0")

    assert sd_cpp_backend.preferred_accelerator("rocm") == "rocm"
    assert sd_cpp_backend.fallback_accelerator_for("rocm") is None

    state = settings_routes._diffusion_accelerator_fallback_response()
    assert state.enabled is False
    assert state.diverting is False, "reported a diversion the switch had turned off"
    assert state.records[0].diverting is False
    assert state.records[0].accelerator == "rocm"
    assert state.records[0].proven is True
    assert state.records[0].stale is False
    assert sd_cpp_backend.accelerator_runtime_failed("rocm") is True

    monkeypatch.delenv("UNSLOTH_DIFFUSION_SD_CPP_VULKAN_FALLBACK", raising = False)
    back = settings_routes._diffusion_accelerator_fallback_response()
    assert back.enabled is True
    assert back.diverting is True
    assert back.records[0].diverting is True


def test_a_stale_record_is_reported_as_stale_rather_than_hidden(fake_settings, monkeypatch):
    from core.inference import sd_cpp_backend
    from routes import settings as settings_routes

    sd_cpp_backend.note_accelerator_runtime_failure("rocm")
    monkeypatch.setattr(
        sd_cpp_backend,
        "_HOST_FINGERPRINT_MEMO",
        {"runtime": "7.0.0", "gpus": ["AMD Radeon RX 7900 XTX"]},
        raising = False,
    )
    state = settings_routes._diffusion_accelerator_fallback_response()
    assert state.records[0].stale is True
    assert state.records[0].diverting is False
    assert state.diverting is False


def test_the_early_list_shape_is_still_read(fake_settings):
    """An early build of this feature stored a bare list of names; read it as what it meant."""
    from core.inference import sd_cpp_backend

    fake_settings["sd_cpp_accelerator_runtime_failures"] = ["rocm"]
    assert sd_cpp_backend.accelerator_runtime_failed("rocm") is True
    assert sd_cpp_backend.preferred_accelerator("rocm") == "vulkan"


@pytest.mark.parametrize(
    "stored", [None, "", [], {}, "not json", '["rocm"]', {"rocm": "yes"}, {"": {}}, 17]
)
def test_an_unreadable_record_never_breaks_a_load(fake_settings, stored):
    from core.inference import sd_cpp_backend

    fake_settings["sd_cpp_accelerator_runtime_failures"] = stored
    assert sd_cpp_backend.accelerator_runtime_failed("rocm") in (True, False)
    assert sd_cpp_backend.preferred_accelerator("rocm") in ("rocm", "vulkan")


# Before the fix the CPU-only Vulkan binary inherited listed_accelerator=True from the unreadable ROCm probe.
_ROCM_UNKNOWN_VULKAN_CPU_ONLY = {
    "rocm": None,
    "vulkan": _DEVICES_CPU_ONLY,
    "cpu": _DEVICES_CPU_ONLY,
}


@pytest.mark.parametrize("platform", PLATFORMS)
def test_a_cpu_only_fallback_build_does_not_inherit_the_gpu_reading(
    h3_amd_host, fake_settings, platform
):
    host = h3_amd_host(
        platform = platform,
        backend = "rocm",
        device = "cuda",
        devices = _ROCM_UNKNOWN_VULKAN_CPU_ONLY,
    )
    backend_obj = host.run()
    assert host.ensured == ["rocm", "vulkan", "cpu"], host.ensured
    assert backend_obj._state.device == "cpu"
    assert _noted_accelerators(fake_settings) == []


@pytest.mark.parametrize("platform", PLATFORMS)
def test_an_unreadable_fallback_never_raises_the_reading(h3_amd_host, fake_settings, platform):
    host = h3_amd_host(
        platform = platform,
        backend = "rocm",
        device = "cuda",
        devices = {"rocm": _DEVICES_CPU_ONLY, "vulkan": None, "cpu": _DEVICES_CPU_ONLY},
    )
    backend_obj = host.run()
    assert host.ensured == ["rocm", "vulkan", "cpu"], host.ensured
    assert backend_obj._state.device == "cpu"
    assert _noted_accelerators(fake_settings) == []


_T, _F, _N, _X = "accel", "cpu_only", "unreadable", "missing"

_ANSWER = {_T: _DEVICES_VULKAN, _F: _DEVICES_CPU_ONLY, _N: None, _X: MISSING}

# (rocm, vulkan) -> (ensured, committed device, what main committed, why). "same": untouched.
_ROUTING = {
    (_T, _T): (["rocm"], "cuda", "cuda", "same"),
    (_T, _F): (["rocm"], "cuda", "cuda", "same"),
    (_T, _N): (["rocm"], "cuda", "cuda", "same"),
    (_T, _X): (["rocm"], "cuda", "cuda", "same"),
    (_F, _T): (["rocm", "vulkan"], "cuda", "cpu", "upgraded on positive evidence"),
    (_F, _F): (["rocm", "vulkan", "cpu"], "cpu", "cpu", "same"),
    (_F, _N): (["rocm", "vulkan", "cpu"], "cpu", "cpu", "same"),
    (_F, _X): (["rocm", "vulkan", "cpu"], "cpu", "cpu", "same"),
    (_N, _T): (["rocm", "vulkan"], "cuda", "cuda", "upgraded on positive evidence"),
    # The one corner whose committed DEVICE moves down, and main's commit there was already a load that fails minutes in.
    (_N, _F): (["rocm", "vulkan", "cpu"], "cpu", "cuda", "was a failed load"),
    (_N, _N): (["rocm", "vulkan"], "cuda", "cuda", "same"),
    (_N, _X): (["rocm", "vulkan"], "cuda", "cuda", "same"),
    (_X, _T): (["rocm", "vulkan"], "cuda", "cpu", "upgraded on positive evidence"),
    (_X, _F): (["rocm", "vulkan", "cpu"], "cpu", "cpu", "same"),
    (_X, _N): (["rocm", "vulkan", "cpu"], "cpu", "cpu", "same"),
    (_X, _X): (["rocm", "vulkan", "cpu"], "cpu", "cpu", "same"),
}


@pytest.mark.parametrize("corner", sorted(_ROUTING), ids = lambda c: f"rocm_{c[0]}-vulkan_{c[1]}")
def test_the_whole_load_routing_space_is_enumerated(h3_amd_host, fake_settings, corner):
    """No input that committed a device on main commits a DIFFERENT device now, except where main's commit was already a failed load."""
    rocm_answer, vulkan_answer = corner
    expected_ensured, expected_device, _main, _why = _ROUTING[corner]
    host = h3_amd_host(
        platform = "linux",
        backend = "rocm",
        device = "cuda",
        devices = {
            "rocm": _ANSWER[rocm_answer],
            "vulkan": _ANSWER[vulkan_answer],
            "cpu": _DEVICES_CPU_ONLY,
        },
    )
    backend_obj = host.run()
    assert host.ensured == expected_ensured, host.ensured
    assert backend_obj._state.device == expected_device
    upgraded = _why == "upgraded on positive evidence"
    # ...and only where the host's own build ANSWERED; an unreadable probe is not an answer.
    assert _noted_accelerators(fake_settings) == (
        ["rocm"] if (upgraded and rocm_answer not in (_N, _X)) else []
    )
    assert _recorded_strikes(fake_settings) == (1 if upgraded else 0)


def _main_would_commit(rocm_answer: str) -> str:
    """main never installs the fallback build, so its committed device is a function of the ROCm column alone; the `main` column of ``_ROUTING`` is DERIVED rather than asserted by hand."""
    if rocm_answer == _X:
        return "cpu"
    verdict = {_T: True, _F: False, _N: None}[rocm_answer]
    return "cuda" if (True if verdict is None else verdict) else "cpu"


def test_the_main_column_is_what_main_actually_commits():
    for (rocm_answer, _vulkan), (_ensured, _device, main_device, _why) in _ROUTING.items():
        assert main_device == _main_would_commit(rocm_answer), (rocm_answer, main_device)


def test_no_corner_loses_a_gpu_it_previously_kept():
    for corner, (_ensured, device, main_device, why) in _ROUTING.items():
        if main_device == "cuda" and device != "cuda":
            assert why == "was a failed load", corner


def test_every_upgrade_required_positive_fallback_evidence():
    for (_rocm, vulkan), (_ensured, _device, _main, why) in _ROUTING.items():
        if why == "upgraded on positive evidence":
            assert vulkan == _T, (_rocm, vulkan)


@pytest.fixture
def unpinned_fingerprint(monkeypatch):
    from core.inference import sd_cpp_backend

    monkeypatch.setattr(sd_cpp_backend, "_HOST_FINGERPRINT_MEMO", None, raising = False)
    monkeypatch.setattr(
        sd_cpp_backend, "_RUNTIME_FINGERPRINT_MEMO", {"runtime": "6.4.0"}, raising = False
    )
    return sd_cpp_backend


def _inventory(*names):
    return {
        "available": bool(names),
        "devices": [{"name": n} for n in names],
        "sources": ["test"],
        "unknown": False,
    }


_UNKNOWN_INVENTORY = {"available": False, "devices": [], "sources": [], "unknown": True}


def test_a_cold_inventory_read_is_not_frozen_for_the_process(unpinned_fingerprint, monkeypatch):
    """``get_physical_gpu_inventory(block = False)`` on a COLD cache returns the unknown sentinel, so memoising the whole host half carried "no cards" for the life of the process."""
    from utils.hardware import hardware

    answers = [_UNKNOWN_INVENTORY, _inventory("AMD Radeon RX 7900 XTX")]
    monkeypatch.setattr(
        hardware,
        "get_physical_gpu_inventory",
        lambda *, block = True: answers.pop(0) if answers else _inventory("AMD Radeon RX 7900 XTX"),
    )

    cold = unpinned_fingerprint._host_fingerprint()
    assert cold == {"runtime": "6.4.0", "gpus": None}
    warm = unpinned_fingerprint._host_fingerprint()
    assert warm == {"runtime": "6.4.0", "gpus": ["AMD Radeon RX 7900 XTX"]}


def test_a_card_added_while_studio_runs_retires_the_record(
    unpinned_fingerprint, fake_settings, monkeypatch
):
    from routes import settings as settings_routes
    from utils.hardware import hardware

    cards = [_inventory("AMD Radeon RX 7900 XTX")]
    monkeypatch.setattr(hardware, "get_physical_gpu_inventory", lambda *, block = True: cards[0])

    unpinned_fingerprint.note_accelerator_runtime_failure("rocm")
    assert unpinned_fingerprint.preferred_accelerator("rocm") == "vulkan"

    cards[0] = _inventory("AMD Radeon RX 7900 XTX", "AMD Radeon RX 9070 XT")
    assert unpinned_fingerprint.accelerator_runtime_failed("rocm") is False
    assert unpinned_fingerprint.preferred_accelerator("rocm") == "rocm"
    assert settings_routes._diffusion_accelerator_fallback_response().records[0].stale is True


def test_the_runtime_half_is_still_read_once(unpinned_fingerprint, monkeypatch):
    """The cards are re-read per call; the wheel label cannot move inside a process, so it is not."""
    import torch

    from utils.hardware import hardware

    cards = [_inventory("first")]
    monkeypatch.setattr(hardware, "get_physical_gpu_inventory", lambda *, block = True: cards[0])
    monkeypatch.setattr(unpinned_fingerprint, "_RUNTIME_FINGERPRINT_MEMO", None, raising = False)
    monkeypatch.setattr(torch.version, "hip", "6.4.0", raising = False)

    first = unpinned_fingerprint._host_fingerprint()
    assert first == {"runtime": "6.4.0", "gpus": ["first"]}
    monkeypatch.setattr(torch.version, "hip", "7.0.0", raising = False)
    cards[0] = _inventory("second")
    assert unpinned_fingerprint._host_fingerprint() == {"runtime": "6.4.0", "gpus": ["second"]}


# The Linux gfx1151 runner's device: no name, because the reading came from sysfs-drm rather than amd-smi.
_UNNAMED_GFX1151 = {
    "vendor": "amd",
    "index": 0,
    "name": None,
    "memory_total_gb": 64.0,
    "source": "sysfs-drm",
    "gfx_candidates": ["gfx11", "gfx1151"],
}


def _raw_inventory(*devices):
    return {
        "available": bool(devices),
        "devices": list(devices),
        "sources": ["test"],
        "unknown": False,
    }


def test_a_card_the_os_cannot_name_still_reaches_the_fingerprint(unpinned_fingerprint, monkeypatch):
    """The gfx1151 Linux runner answers from sysfs-drm with ``name = None``; keying on the name alone dropped its only device."""
    from utils.hardware import hardware

    monkeypatch.setattr(
        hardware,
        "get_physical_gpu_inventory",
        lambda *, block = True: _raw_inventory(dict(_UNNAMED_GFX1151)),
    )
    assert unpinned_fingerprint._host_fingerprint() == {
        "runtime": "6.4.0",
        "gpus": ["gfx11/gfx1151"],
    }


def test_a_named_card_carries_its_target_as_well(unpinned_fingerprint, monkeypatch):
    from utils.hardware import hardware

    named = dict(_UNNAMED_GFX1151, name = "AMD Radeon(TM) 8060S Graphics")
    monkeypatch.setattr(
        hardware, "get_physical_gpu_inventory", lambda *, block = True: _raw_inventory(named)
    )
    assert unpinned_fingerprint._host_fingerprint() == {
        "runtime": "6.4.0",
        "gpus": ["AMD Radeon(TM) 8060S Graphics@gfx11/gfx1151"],
    }


def test_two_cards_sharing_a_generic_name_are_not_one_card(unpinned_fingerprint, monkeypatch):
    """A gfx1103 APU and a gfx1151 Strix Halo both answer ``AMD Radeon(TM) Graphics``."""
    from utils.hardware import hardware

    first = dict(
        _UNNAMED_GFX1151,
        name = "AMD Radeon(TM) Graphics",
        gfx_candidates = ["gfx11", "gfx1103"],
    )
    second = dict(
        _UNNAMED_GFX1151,
        name = "AMD Radeon(TM) Graphics",
        gfx_candidates = ["gfx11", "gfx1151"],
    )
    cards = [_raw_inventory(first)]
    monkeypatch.setattr(hardware, "get_physical_gpu_inventory", lambda *, block = True: cards[0])

    unpinned_fingerprint.note_accelerator_runtime_failure("rocm")
    assert unpinned_fingerprint.preferred_accelerator("rocm") == "vulkan"
    cards[0] = _raw_inventory(second)
    assert unpinned_fingerprint.accelerator_runtime_failed("rocm") is False


def test_two_unnamed_cards_of_different_targets_are_not_one_card(unpinned_fingerprint, monkeypatch):
    from utils.hardware import hardware

    first = dict(_UNNAMED_GFX1151, gfx_candidates = ["gfx11", "gfx1100"])
    second = dict(_UNNAMED_GFX1151, gfx_candidates = ["gfx12", "gfx1201"])
    cards = [_raw_inventory(first)]
    monkeypatch.setattr(hardware, "get_physical_gpu_inventory", lambda *, block = True: cards[0])

    unpinned_fingerprint.note_accelerator_runtime_failure("rocm")
    assert unpinned_fingerprint.preferred_accelerator("rocm") == "vulkan"
    cards[0] = _raw_inventory(second)
    assert unpinned_fingerprint.accelerator_runtime_failed("rocm") is False


def test_a_device_reporting_nothing_but_a_vendor_still_contributes(
    unpinned_fingerprint, monkeypatch
):
    from utils.hardware import hardware
    monkeypatch.setattr(
        hardware,
        "get_physical_gpu_inventory",
        lambda *, block = True: _raw_inventory({"vendor": "amd", "index": 1}),
    )
    assert unpinned_fingerprint._host_fingerprint()["gpus"] == ["amd:1"]


def test_an_empty_device_list_is_still_no_cards(unpinned_fingerprint, monkeypatch):
    from utils.hardware import hardware
    monkeypatch.setattr(hardware, "get_physical_gpu_inventory", lambda *, block = True: {})
    assert unpinned_fingerprint._host_fingerprint()["gpus"] is None


# The gfx1151 Windows runner: the ROCm asset ships no hipBLAS, so every invocation exits 0xC0000135 printing nothing.
_WINDOWS_DLL_FAILURE = "sd-cli exited 3221225781. Last output:\n"


def test_a_build_that_never_started_is_a_decisive_build_failure():
    from core.inference.sd_cpp_backend import (
        output_shows_accelerator_failure,
        output_shows_decisive_accelerator_failure,
    )
    assert output_shows_accelerator_failure(_WINDOWS_DLL_FAILURE) is True
    assert output_shows_decisive_accelerator_failure(_WINDOWS_DLL_FAILURE) is True


@pytest.mark.parametrize(
    "status",
    [3221225781, 3221225785, 3221225794, -1073741515, -1073741511, -1073741502],
)
def test_every_image_load_status_is_recognised_signed_or_unsigned(status):
    from core.inference.sd_cpp_backend import output_shows_image_load_failure
    assert output_shows_image_load_failure(f"sd-cli exited {status}. Last output:\n") is True


@pytest.mark.parametrize("status", [1, 2, 134, 139, -9, -11, 3221225477])
def test_an_ordinary_non_zero_exit_is_not_a_build_failure(status):
    from core.inference.sd_cpp_backend import (
        output_shows_decisive_accelerator_failure,
        output_shows_image_load_failure,
    )

    text = f"sd-cli exited {status}. Last output:\nsomething went wrong\n"
    assert output_shows_image_load_failure(text) is False
    assert output_shows_decisive_accelerator_failure(text) is False


def test_an_out_of_memory_alongside_an_image_load_status_is_still_capacity():
    from core.inference.sd_cpp_backend import (
        output_shows_accelerator_failure,
        output_shows_decisive_accelerator_failure,
    )

    text = "sd-cli exited 3221225781. Last output:\nROCm error: out of memory\n"
    assert output_shows_accelerator_failure(text) is False
    assert output_shows_decisive_accelerator_failure(text) is False


def test_the_number_alone_is_not_enough():
    from core.inference.sd_cpp_backend import output_shows_image_load_failure
    assert output_shows_image_load_failure("seed 3221225781 produced a nice image") is False


def test_the_vulkan_fallback_pins_the_card_that_was_selected(monkeypatch):
    """``Vulkan0`` is not the physical index: the ordinal lookup is confined to the CUDA/ROCm namespace, so after the fallback sd.cpp took its own default device."""
    from core.inference import sd_cpp_backend

    listing = (
        "CPU\tAMD Ryzen 9 7950X\n"
        "Vulkan0\tAMD Radeon RX 7600 (RADV NAVI33)\n"
        "Vulkan1\tAMD Radeon RX 7900 XTX (RADV NAVI31)\n"
    )
    monkeypatch.setattr(
        sd_cpp_backend,
        "_sd_cpp_probe_output",
        lambda binary, *args: listing if args == ("--list-devices",) else None,
    )
    assert sd_cpp_backend.sd_cpp_device_name_for_ordinal("/opt/sd/vulkan/sd-cli", 1) is None
    assert (
        sd_cpp_backend.sd_cpp_device_named("/opt/sd/vulkan/sd-cli", "AMD Radeon RX 7900 XTX")
        == "Vulkan1"
    )
    assert (
        sd_cpp_backend.sd_cpp_device_named("/opt/sd/vulkan/sd-cli", "AMD Radeon RX 7600")
        == "Vulkan0"
    )
    from core.inference.sd_cpp_args import device_backend_flags

    assert device_backend_flags("Vulkan1") == [
        "--backend",
        "diffusion=Vulkan1,te=Vulkan1,vae=Vulkan1",
    ]


def test_two_identical_cards_are_not_pinned_on_a_guess(monkeypatch):
    from core.inference import sd_cpp_backend

    listing = (
        "Vulkan0\tAMD Radeon RX 7900 XTX (RADV NAVI31)\n"
        "Vulkan1\tAMD Radeon RX 7900 XTX (RADV NAVI31)\n"
    )
    monkeypatch.setattr(
        sd_cpp_backend,
        "_sd_cpp_probe_output",
        lambda binary, *args: listing if args == ("--list-devices",) else None,
    )
    assert (
        sd_cpp_backend.sd_cpp_device_named("/opt/sd/vulkan/sd-cli", "AMD Radeon RX 7900 XTX")
        is None
    )
    # Both namespaces walk one vendor's GPUs in the order the driver reports them.
    assert (
        sd_cpp_backend.sd_cpp_device_named(
            "/opt/sd/vulkan/sd-cli", "AMD Radeon RX 7900 XTX", position = 1
        )
        == "Vulkan1"
    )
    assert (
        sd_cpp_backend.sd_cpp_device_named(
            "/opt/sd/vulkan/sd-cli", "AMD Radeon RX 7900 XTX", position = 0
        )
        == "Vulkan0"
    )
    assert (
        sd_cpp_backend.sd_cpp_device_named(
            "/opt/sd/vulkan/sd-cli", "AMD Radeon RX 7900 XTX", position = 5
        )
        is None
    )
    assert sd_cpp_backend.sd_cpp_device_named("/opt/sd/vulkan/sd-cli", "NVIDIA RTX 4090") is None
    assert sd_cpp_backend.sd_cpp_device_named("/opt/sd/vulkan/sd-cli", None) is None


def test_the_h3_load_resolves_the_pin_by_name_when_the_index_says_nothing():
    import inspect
    from core.inference import video as video_mod

    source = inspect.getsource(video_mod)
    assert "sd_cpp_device_named(" in source
    ordinal_call = source.index("sd_cpp_device_name_for_ordinal(binary, native_ordinal)")
    named_call = source.index("sd_cpp_device_named(\n", ordinal_call)
    assert ordinal_call < named_call
    assert "_physical_card_name(native_ordinal)" in source
    assert "position = selected_position" in source


_VISIBILITY_VARS = (
    "ROCR_VISIBLE_DEVICES",
    "HIP_VISIBLE_DEVICES",
    "CUDA_VISIBLE_DEVICES",
    "GPU_DEVICE_ORDINAL",
)


def _no_visibility_mask(monkeypatch):
    for variable in _VISIBILITY_VARS:
        monkeypatch.delenv(variable, raising = False)


def test_the_position_among_identical_cards_is_what_is_carried(monkeypatch):
    from core.inference import video as video_mod

    _no_visibility_mask(monkeypatch)

    class _FakeCuda:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def device_count():
            return 3

        @staticmethod
        def get_device_name(index):
            return ["AMD Radeon RX 7600", "AMD Radeon RX 7900 XTX", "AMD Radeon RX 7900 XTX"][index]

    monkeypatch.setitem(__import__("sys").modules, "torch", type("torch", (), {"cuda": _FakeCuda}))
    assert video_mod._physical_card_name(0) == ("AMD Radeon RX 7600", 0)
    assert video_mod._physical_card_name(1) == ("AMD Radeon RX 7900 XTX", 0)
    assert video_mod._physical_card_name(2) == ("AMD Radeon RX 7900 XTX", 1)
    assert video_mod._physical_card_name(9) == (None, None)
    assert video_mod._physical_card_name(None) == (None, None)


def _pinned_torch(monkeypatch, names):
    class _FakeCuda:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def device_count():
            return len(names)

        @staticmethod
        def get_device_name(index):
            return names[index]

    monkeypatch.setitem(__import__("sys").modules, "torch", type("torch", (), {"cuda": _FakeCuda}))


def _pinned_inventory(
    monkeypatch,
    devices,
    hip_by_row = None,
):
    """Two index spaces: masks and torch speak HIP ids, the inventory's ``index`` is amd-smi's discovery row. Default here is the identity mapping."""
    from utils.hardware import amd, hardware

    monkeypatch.setattr(
        hardware,
        "get_physical_gpu_inventory",
        lambda *, block = True: {
            "available": True,
            "devices": list(devices),
            "sources": ["test"],
            "unknown": False,
        },
    )
    if hip_by_row is None:
        hip_by_row = {device["index"]: device["index"] for device in devices}
    monkeypatch.setattr(amd, "get_hip_id_by_gpu_index", lambda: hip_by_row)


def test_a_visibility_mask_is_translated_before_the_tie_is_broken(monkeypatch):
    """The Vulkan child gets no mask -- Vulkan does not read those variables -- so the count has to be taken over the PHYSICAL cards."""
    from core.inference import video as video_mod

    _no_visibility_mask(monkeypatch)
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "1")
    _pinned_torch(monkeypatch, ["AMD Radeon RX 7900 XTX"])
    _pinned_inventory(
        monkeypatch,
        [
            {"vendor": "amd", "index": 0, "name": "AMD Radeon RX 7900 XTX"},
            {"vendor": "amd", "index": 1, "name": "AMD Radeon RX 7900 XTX"},
        ],
    )

    assert video_mod._physical_card_name(0) == ("AMD Radeon RX 7900 XTX", 1)


def test_the_masks_compose_the_way_rocm_applies_them(monkeypatch):
    """ROCR filters the agents the runtime reports and HIP then indexes into WHAT IS LEFT."""
    from core.inference import video as video_mod

    _no_visibility_mask(monkeypatch)
    monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "1,2,3")
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "2")  # -> physical 3
    _pinned_torch(monkeypatch, ["AMD Radeon RX 7900 XTX"])
    _pinned_inventory(
        monkeypatch,
        [{"vendor": "amd", "index": index, "name": "AMD Radeon RX 7900 XTX"} for index in range(4)],
    )

    assert video_mod._physical_card_name(0) == ("AMD Radeon RX 7900 XTX", 3)


def test_a_hip_id_is_translated_into_the_inventorys_own_row(monkeypatch):
    """A mask and torch name HIP device ids; the inventory's ``index`` is amd-smi's discovery row, and they do not coincide on every host."""
    from core.inference import video as video_mod

    _no_visibility_mask(monkeypatch)
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "0")  # HIP id 0 ...
    _pinned_torch(monkeypatch, ["AMD Radeon RX 7900 XTX"])
    _pinned_inventory(
        monkeypatch,
        [
            {"vendor": "amd", "index": 0, "name": "AMD Radeon RX 7600"},
            {"vendor": "amd", "index": 1, "name": "AMD Radeon RX 7900 XTX"},
            {"vendor": "amd", "index": 2, "name": "AMD Radeon RX 7900 XTX"},
        ],
        # ... which is probe row 2 on this host, not row 0.
        hip_by_row = {0: 2, 1: 1, 2: 0},
    )

    assert video_mod._physical_card_name(0) == ("AMD Radeon RX 7900 XTX", 1)


def test_an_unreadable_hip_mapping_declines_the_tie_break(monkeypatch):
    """``get_hip_id_by_gpu_index`` answers None when any device lacks a usable id, and tells callers not to assume the identity mapping."""
    from core.inference import video as video_mod

    _no_visibility_mask(monkeypatch)
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "1")
    _pinned_torch(monkeypatch, ["AMD Radeon RX 7900 XTX"])
    _pinned_inventory(
        monkeypatch,
        [
            {"vendor": "amd", "index": 0, "name": "AMD Radeon RX 7900 XTX"},
            {"vendor": "amd", "index": 1, "name": "AMD Radeon RX 7900 XTX"},
        ],
        hip_by_row = None,
    )
    from utils.hardware import amd

    monkeypatch.setattr(amd, "get_hip_id_by_gpu_index", lambda: None)

    assert video_mod._physical_card_name(0) == ("AMD Radeon RX 7900 XTX", None)


def test_a_mask_this_cannot_read_declines_the_tie_break(monkeypatch):
    """A mask may name UUIDs, which say nothing about enumeration order."""
    from core.inference import video as video_mod

    _no_visibility_mask(monkeypatch)
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "GPU-a1b2c3d4e5f60718")
    _pinned_torch(monkeypatch, ["AMD Radeon RX 7900 XTX"])
    _pinned_inventory(
        monkeypatch,
        [
            {"vendor": "amd", "index": 0, "name": "AMD Radeon RX 7900 XTX"},
            {"vendor": "amd", "index": 1, "name": "AMD Radeon RX 7900 XTX"},
        ],
    )

    assert video_mod._physical_card_name(0) == ("AMD Radeon RX 7900 XTX", None)


def test_an_unnamed_physical_card_withholds_the_tie_break(monkeypatch):
    from core.inference import video as video_mod

    _no_visibility_mask(monkeypatch)
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "1")
    _pinned_torch(monkeypatch, ["AMD Radeon(TM) Graphics"])
    _pinned_inventory(
        monkeypatch,
        [
            {"vendor": "amd", "index": 0, "name": None, "gfx_candidates": ["gfx11", "gfx1151"]},
            {"vendor": "amd", "index": 1, "name": None, "gfx_candidates": ["gfx11", "gfx1151"]},
        ],
    )

    assert video_mod._physical_card_name(0) == ("AMD Radeon(TM) Graphics", None)


def test_the_tie_break_counts_the_same_group_the_matcher_will(monkeypatch):
    """The matcher can only index Vulkan devices by marketing name, so the tie-break has to count that same group rather than the gfx-bearing identity."""
    from core.inference import video as video_mod

    _no_visibility_mask(monkeypatch)
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "1")
    _pinned_torch(monkeypatch, ["AMD Radeon(TM) Graphics"])
    _pinned_inventory(
        monkeypatch,
        [
            {
                "vendor": "amd",
                "index": 0,
                "name": "AMD Radeon(TM) Graphics",
                "gfx_candidates": ["gfx11", "gfx1103"],
            },
            {
                "vendor": "amd",
                "index": 1,
                "name": "AMD Radeon(TM) Graphics",
                "gfx_candidates": ["gfx11", "gfx1151"],
            },
        ],
    )

    assert video_mod._physical_card_name(0) == ("AMD Radeon(TM) Graphics", 1)
    _pinned_inventory(
        monkeypatch,
        [
            {"vendor": "amd", "index": 0, "name": "AMD Radeon RX 7600"},
            {"vendor": "amd", "index": 1, "name": "AMD Radeon(TM) Graphics"},
        ],
    )
    assert video_mod._physical_card_name(0) == ("AMD Radeon(TM) Graphics", 0)


def test_a_strike_never_forgets_what_the_last_one_knew(monkeypatch):
    """An unknown component reads as compatible, so overwriting a known one with None makes the note permanent."""
    from core.inference import sd_cpp_backend

    previous = {"bundle": "b1", "runtime": "rocm6.2", "gpus": "gfx1100"}
    current = {"bundle": None, "runtime": "rocm6.2", "gpus": None}
    merged = sd_cpp_backend._fingerprint_with_known_fields_kept(previous, current)
    assert merged == {"bundle": "b1", "runtime": "rocm6.2", "gpus": "gfx1100"}

    changed = {"bundle": "b2", "runtime": "rocm6.2", "gpus": "gfx1201"}
    assert sd_cpp_backend._fingerprint_with_known_fields_kept(previous, changed) == changed
    assert sd_cpp_backend._fingerprint_with_known_fields_kept(None, current) == current


def test_a_second_strike_taken_blind_still_expires_when_the_cards_change(
    fake_settings, monkeypatch
):
    """If the second strike was taken while the GPU list was unreadable the record used to carry no cards, and no later change could retire it."""
    from core.inference import sd_cpp_backend
    from core.inference import video as video_mod

    monkeypatch.setattr(sd_cpp_backend, "_installed_accelerator_of", lambda _b: "rocm")
    monkeypatch.setattr(
        sd_cpp_backend,
        "_accelerator_fingerprint",
        lambda: {"bundle": "b1", "runtime": "rocm6.2", "gpus": "gfx1100"},
    )
    video_mod._note_sd_cpp_accelerator_failure(
        "/opt/sd/rocm/sd-cli", "sd-cli exited 1. Last output:\nROCm error: no kernel image"
    )
    # The second strike is taken while the cards cannot be read.
    monkeypatch.setattr(
        sd_cpp_backend,
        "_accelerator_fingerprint",
        lambda: {"bundle": None, "runtime": "rocm6.2", "gpus": None},
    )
    video_mod._note_sd_cpp_accelerator_failure(
        "/opt/sd/rocm/sd-cli", "sd-cli exited 1. Last output:\nROCm error: no kernel image"
    )
    assert sd_cpp_backend.accelerator_runtime_failed("rocm") is True

    # A new card. The record has to expire, which it cannot do against a fingerprint of Nones.
    monkeypatch.setattr(
        sd_cpp_backend,
        "_accelerator_fingerprint",
        lambda: {"bundle": "b1", "runtime": "rocm6.2", "gpus": "gfx1201"},
    )
    assert sd_cpp_backend.accelerator_runtime_failed("rocm") is False


def test_a_related_model_name_is_not_tie_broken_by_position(monkeypatch):
    """``AMD Radeon RX 7600`` is contained in ``AMD Radeon RX 7600 XT``, and the position counts only cards of the SELECTED name."""
    from core.inference import sd_cpp_backend

    listing = (
        "Vulkan0\tAMD Radeon RX 7600 XT (RADV NAVI33)\n"
        "Vulkan1\tAMD Radeon RX 7600 (RADV NAVI33)\n"
    )
    monkeypatch.setattr(
        sd_cpp_backend,
        "_sd_cpp_probe_output",
        lambda binary, *args: listing if args == ("--list-devices",) else None,
    )
    assert (
        sd_cpp_backend.sd_cpp_device_named(
            "/opt/sd/vulkan/sd-cli", "AMD Radeon RX 7600", position = 0
        )
        == "Vulkan1"
    )
    assert (
        sd_cpp_backend.sd_cpp_device_named(
            "/opt/sd/vulkan/sd-cli", "AMD Radeon RX 7600 XT", position = 0
        )
        == "Vulkan0"
    )

    mixed = (
        "Vulkan0\tAMD Radeon RX 7600 XT Special Edition\n"
        "Vulkan1\tAMD Radeon RX 7600 Special Edition\n"
    )
    monkeypatch.setattr(
        sd_cpp_backend,
        "_sd_cpp_probe_output",
        lambda binary, *args: mixed if args == ("--list-devices",) else None,
    )
    assert (
        sd_cpp_backend.sd_cpp_device_named(
            "/opt/sd/vulkan/sd-cli", "AMD Radeon RX 7600", position = 0
        )
        is None
    )
    assert (
        sd_cpp_backend.sd_cpp_device_named(
            "/opt/sd/vulkan/sd-cli", "AMD Radeon RX 7600", position = 1
        )
        is None
    )


def test_a_build_recorded_as_unrunnable_is_not_accepted_back_from_the_ensure(monkeypatch):
    """An ensure does not promise the accelerator asked for: offline or after a failed download it returns whatever usable build is in the tree."""
    from core.inference import sd_cpp_backend

    monkeypatch.setattr(
        sd_cpp_backend, "_installed_accelerator_of", lambda binary: "rocm", raising = False
    )
    monkeypatch.setattr(
        sd_cpp_backend,
        "accelerator_runtime_failed",
        lambda accelerator, card = None: accelerator == "rocm",
        raising = False,
    )
    assert sd_cpp_backend.usable_or_recorded_failure("/opt/sd/rocm/sd-cli", "vulkan") is None
    assert sd_cpp_backend.usable_or_recorded_failure("/opt/sd/rocm/sd-cli", "cpu") is None

    # Asked for ROCm and handed back ROCm: kept, because with the fallback switched off preferred_accelerator
    # asks for ROCm on purpose and that opt-out means run it anyway.
    assert (
        sd_cpp_backend.usable_or_recorded_failure("/opt/sd/rocm/sd-cli", "rocm")
        == "/opt/sd/rocm/sd-cli"
    )

    monkeypatch.setattr(
        sd_cpp_backend, "_installed_accelerator_of", lambda binary: "vulkan", raising = False
    )
    assert (
        sd_cpp_backend.usable_or_recorded_failure("/opt/sd/vulkan/sd-cli", "rocm")
        == "/opt/sd/vulkan/sd-cli"
    )
    monkeypatch.setattr(
        sd_cpp_backend, "_installed_accelerator_of", lambda binary: None, raising = False
    )
    assert (
        sd_cpp_backend.usable_or_recorded_failure("/usr/local/bin/sd", "rocm")
        == "/usr/local/bin/sd"
    )
    assert sd_cpp_backend.usable_or_recorded_failure(None, "rocm") is None


def test_the_image_router_checks_its_ensures_against_the_record_too():
    """The image router's own ensures are probed for runnability, which a ROCm build that only dies mid-render passes."""
    import inspect
    from core.inference import diffusion_engine_router as router

    body = inspect.getsource(router.select_and_activate_engine)
    ensures = body.count("ensure_sd_server_binary(") + body.count("ensure_sd_cpp_binary(")
    assert ensures == 2, ensures
    # Through the one gate both ensures share, so a refactor giving them a common path needs no rewrite here.
    assert body.count("_accept(") == ensures + 1, body[:400]
    assert "usable_or_recorded_failure(candidate, install_accelerator, selected_card)" in body


def test_every_ensure_in_the_h3_load_is_checked_against_the_record():
    import inspect
    from core.inference import video as video_mod

    source = inspect.getsource(video_mod)
    load = source[source.index("allow_install = _install_allowed()") :]
    ensures = load.count("ensure_h3_sd_cpp_binary(")
    guarded = load.count("usable_or_recorded_failure(")
    assert ensures >= 3, ensures
    assert guarded == ensures, (guarded, ensures)


def test_the_note_can_be_pinned_to_the_build_that_failed(monkeypatch):
    """The load installs the fallback bundle BEFORE it records the failure, so a live reading would describe the replacement build."""
    from core.inference import sd_cpp_backend

    stored: dict = {}
    monkeypatch.setattr(
        sd_cpp_backend, "_stored_accelerator_runtime_failures", lambda: stored, raising = False
    )
    monkeypatch.setattr(
        sd_cpp_backend,
        "_write_accelerator_runtime_failures",
        lambda records: stored.update(records),
        raising = False,
    )
    monkeypatch.setattr(
        sd_cpp_backend,
        "_accelerator_fingerprint",
        lambda: {"bundle": "after-the-install"},
        raising = False,
    )
    monkeypatch.setattr(sd_cpp_backend, "_accelerator_runtime_failures", {}, raising = False)

    sd_cpp_backend.note_accelerator_runtime_failure(
        "rocm", fingerprint = {"bundle": "the-build-that-failed"}
    )
    record = (stored or sd_cpp_backend._accelerator_runtime_failures).get("rocm")
    assert record, stored
    assert record["fingerprint"]["bundle"] == "the-build-that-failed", record

    monkeypatch.setattr(sd_cpp_backend, "_accelerator_runtime_failures", {}, raising = False)
    stored.clear()
    sd_cpp_backend.note_accelerator_runtime_failure("rocm")
    record = (stored or sd_cpp_backend._accelerator_runtime_failures).get("rocm")
    assert record["fingerprint"]["bundle"] == "after-the-install", record


def test_the_load_path_reads_the_fingerprint_before_it_installs_the_fallback():
    import inspect
    from core.inference import video as video_mod

    source = inspect.getsource(video_mod)
    # It is the FAILED build's own root that is fingerprinted, not the current default.
    read = source.index("failed_fingerprint = _accelerator_fingerprint(binary)")
    install = source.index("fallback_binary = usable_or_recorded_failure(")
    note = source.index("note_accelerator_runtime_failure(\n")
    assert read < install < note, (read, install, note)
    assert "fingerprint = failed_fingerprint" in source[note : note + 900]
    window = source[note : note + 700]
    # `proven` still needs the probe to have RUN and ANSWERED, and now also that a bare negative
    # answer is explained: "CPU only" reads the same whether the runtime is missing or the card busy.
    assert "accelerator_probe_ran" in window
    assert "accelerator_verdict is not None" in window
    assert "accelerator_probe_failure_is_decisive(accelerator)" in window


def test_a_singleton_match_does_not_answer_for_a_position_it_cannot_hold(monkeypatch):
    from core.inference import sd_cpp_backend

    listing = "Vulkan0\tAMD Radeon RX 7900 XTX (RADV NAVI31)\n"
    monkeypatch.setattr(
        sd_cpp_backend,
        "_sd_cpp_probe_output",
        lambda binary, *args: listing if args == ("--list-devices",) else None,
    )
    assert (
        sd_cpp_backend.sd_cpp_device_named(
            "/opt/sd/vulkan/sd-cli", "AMD Radeon RX 7900 XTX", position = 1
        )
        is None
    )
    assert (
        sd_cpp_backend.sd_cpp_device_named(
            "/opt/sd/vulkan/sd-cli", "AMD Radeon RX 7900 XTX", position = 0
        )
        == "Vulkan0"
    )
    assert (
        sd_cpp_backend.sd_cpp_device_named("/opt/sd/vulkan/sd-cli", "AMD Radeon RX 7900 XTX")
        == "Vulkan0"
    )


def test_the_fingerprint_reads_the_root_that_owns_the_binary(monkeypatch):
    """A build served out of the legacy tree beside the Unsloth home must not be recorded under the default root's tag, or nothing can ever retire the record."""
    from pathlib import Path

    from core.inference import sd_cpp_backend

    asked: "list" = []

    class _Installer:
        @staticmethod
        def read_install_record(root):
            asked.append(str(root))
            return {"tag": f"tag-for-{Path(root).name}"}

    monkeypatch.setattr(sd_cpp_backend, "_installer_module", lambda: _Installer, raising = False)
    monkeypatch.setattr(sd_cpp_backend, "_host_fingerprint", lambda: {}, raising = False)
    monkeypatch.setattr(
        sd_cpp_backend, "managed_install_root", lambda: Path("/roots/current"), raising = False
    )
    monkeypatch.setattr(
        sd_cpp_backend,
        "owning_managed_root",
        lambda binary: Path("/roots/legacy") if binary else None,
        raising = False,
    )

    monkeypatch.setattr(sd_cpp_backend, "find_sd_cpp_binary", lambda: None, raising = False)
    assert sd_cpp_backend._accelerator_fingerprint("/roots/legacy/bin/sd")["bundle"] == (
        "tag-for-legacy"
    )
    assert sd_cpp_backend._accelerator_fingerprint()["bundle"] == "tag-for-current"
    assert asked == ["/roots/legacy", "/roots/current"], asked

    # Named nothing while the finder serves the legacy tree (every CONSULTATION, since `accelerator_runtime_failed` passes no binary), it is that root too.
    asked.clear()
    monkeypatch.setattr(
        sd_cpp_backend, "find_sd_cpp_binary", lambda: "/roots/legacy/bin/sd", raising = False
    )
    assert sd_cpp_backend._accelerator_fingerprint()["bundle"] == "tag-for-legacy"
    assert asked == ["/roots/legacy"], asked


def test_the_decided_class_is_read_under_the_claim_that_validated_it():
    """Read after the claim is released, an install that replaced the tree in between is recorded as the class this load decided on."""
    import inspect
    from core.inference import video as video_mod

    source = inspect.getsource(video_mod)
    probe = source.index("accelerator_verdict = sd_cpp_accelerator_device_verdict(binary)")
    read = source.index("decided_accelerator = _installed_accelerator_of(binary)", probe)
    collapse = source.index("listed_accelerator = accelerator_verdict_keeps_gpu(", probe)
    assert probe < read < collapse, (probe, read, collapse)
    assert source.count("decided_accelerator = fallback_class") == 2, source.count(
        "decided_accelerator = fallback_class"
    )
    assert "_UNREAD_ACCELERATOR" in source


def test_a_second_unreadable_rocm_probe_does_divert(h3_amd_host, fake_settings):
    """One unreadable probe can be a timeout or a transient fault; repeating under the same fingerprint is a different claim."""
    from core.inference import sd_cpp_backend

    for _ in range(sd_cpp_backend._AMBIGUOUS_FAILURE_STRIKES - 1):
        host = h3_amd_host(platform = "linux", backend = "rocm", device = "cuda", devices = _ROCM_BROKEN)
        host.run()
        assert _noted_accelerators(fake_settings) == []
    host = h3_amd_host(platform = "linux", backend = "rocm", device = "cuda", devices = _ROCM_BROKEN)
    host.run()
    assert _noted_accelerators(fake_settings) == ["rocm"]


def test_an_image_generation_that_dies_in_hipblas_records_it_too(fake_settings, monkeypatch):
    """The recorder was reached only from the video path, so an image-only host never moved off ROCm."""
    from core.inference import sd_cpp_backend

    monkeypatch.setattr(sd_cpp_backend, "_installed_accelerator_of", lambda _b: "rocm")
    backend = sd_cpp_backend.SdCppDiffusionBackend.__new__(sd_cpp_backend.SdCppDiffusionBackend)
    backend._engine = types.SimpleNamespace(binary = "/opt/sd/rocm/sd-cli")
    cancel = threading.Event()
    source = inspect.getsource(sd_cpp_backend.SdCppDiffusionBackend.generate)
    assert "note_accelerator_failure_from_output(" in source, source[-2000:]

    sd_cpp_backend.note_accelerator_failure_from_output(
        getattr(getattr(backend, "_engine", None), "binary", None),
        "sd-cli exited 1. Last output:\nROCm error: CUBLAS_STATUS_INVALID_VALUE at hipblasSetStream",
        source = "diffusion",
    )
    assert _noted_accelerators(fake_settings) == ["rocm"]
    assert sd_cpp_backend.preferred_accelerator("rocm") == "vulkan"
    assert not cancel.is_set()


def test_a_cancelled_image_generation_records_nothing(fake_settings, monkeypatch):
    from core.inference import sd_cpp_backend

    source = inspect.getsource(sd_cpp_backend.SdCppDiffusionBackend.generate)
    handler = source.index("note_accelerator_failure_from_output(")
    window = source[max(0, handler - 1400) : handler]
    assert "cancel.is_set()" in window, window
    assert "DIFFUSION_CANCELLED_MSG not in str(exc)" in window, window


def test_the_availability_probe_reads_the_same_record_selection_does(fake_settings, monkeypatch):
    """Counting a binary selection would refuse predicted native while the load went to diffusers, whose planner was never run."""
    from core.inference import diffusion_engine_router as router
    from core.inference import sd_cpp_backend

    monkeypatch.setattr(
        router, "resolve_diffusion_device_target", lambda: types.SimpleNamespace(backend = "cuda")
    )
    monkeypatch.setattr(router, "_install_accelerator_for", lambda _backend: "rocm")
    monkeypatch.setattr(router, "ensure_sd_server_binary", lambda **_k: None)
    monkeypatch.setattr(router, "ensure_sd_cpp_binary", lambda **_k: "/opt/sd/rocm/sd-cli")
    monkeypatch.setattr(
        router, "SdCppEngine", lambda binary: types.SimpleNamespace(version = lambda: "1.0")
    )
    assert router.native_binary_installed() is True

    monkeypatch.setattr(sd_cpp_backend, "_installed_accelerator_of", lambda _b: "rocm")
    sd_cpp_backend.note_accelerator_runtime_failure("rocm")
    assert sd_cpp_backend.preferred_accelerator("rocm") == "vulkan"
    assert router.native_binary_installed() is False


def test_a_server_that_dies_mid_render_is_recorded_against_its_own_binary(
    fake_settings, monkeypatch
):
    """``_resolve_backend`` returns no engine in server mode, so reading ``self._engine`` there passes None and the recorder returns immediately."""
    import inspect

    from core.inference import sd_cpp_backend

    source = inspect.getsource(sd_cpp_backend.SdCppDiffusionBackend.generate)
    handler = source.index("note_accelerator_failure_from_output(")
    window = source[max(0, handler - 1400) : handler]
    assert 'getattr(state, "server", None), "binary"' in window, window

    monkeypatch.setattr(sd_cpp_backend, "_installed_accelerator_of", lambda _b: "rocm")
    sd_cpp_backend.note_accelerator_failure_from_output(
        "/opt/sd/rocm/sd-server",
        "sd-server exited 1. Last output:\nROCm error: CUBLAS_STATUS_INVALID_VALUE at hipblasSetStream",
        source = "diffusion",
    )
    assert _noted_accelerators(fake_settings) == ["rocm"]


def test_the_image_pin_follows_the_card_into_the_vulkan_namespace(monkeypatch):
    """The ordinal lookup answers None in the Vulkan namespace, so sd.cpp took its own default device while this load reserved another card."""
    from core.inference import sd_cpp_backend

    monkeypatch.setattr(
        sd_cpp_backend,
        "_sd_cpp_probe_output",
        lambda *_a: "Vulkan0\tAMD Radeon RX 7600\nVulkan1\tAMD Radeon RX 7900 XTX (RADV NAVI31)\n",
    )
    monkeypatch.setattr(
        sd_cpp_backend, "physical_card_name", lambda ordinal: ("AMD Radeon RX 7900 XTX", 0)
    )
    flags = sd_cpp_backend._offload_with_device_pin_impl(["--offload-to-cpu"], "/sd/sd-cli", 1)
    assert flags == ["--offload-to-cpu", "--backend", "diffusion=Vulkan1,te=Vulkan1,vae=Vulkan1"]

    monkeypatch.setattr(sd_cpp_backend, "physical_card_name", lambda ordinal: (None, None))
    assert sd_cpp_backend._offload_with_device_pin_impl(["--offload-to-cpu"], "/sd/sd-cli", 1) == [
        "--offload-to-cpu"
    ]


def test_the_image_pin_tells_two_identical_cards_apart(monkeypatch):
    """Getting this wrong is worse than not pinning: it writes a --backend for the OTHER card while the load accounts for the selected one."""
    from core.inference import sd_cpp_backend

    monkeypatch.setattr(
        sd_cpp_backend,
        "_sd_cpp_probe_output",
        lambda *_a: (
            "Vulkan0\tAMD Radeon RX 7900 XTX (RADV NAVI31)\n"
            "Vulkan1\tAMD Radeon RX 7900 XTX (RADV NAVI31)\n"
        ),
    )
    for position, expected in ((0, "Vulkan0"), (1, "Vulkan1")):
        monkeypatch.setattr(
            sd_cpp_backend,
            "physical_card_name",
            lambda _ordinal, _p = position: ("AMD Radeon RX 7900 XTX", _p),
        )
        flags = sd_cpp_backend._offload_with_device_pin_impl([], "/sd/sd-cli", position)
        assert flags == [
            "--backend",
            f"diffusion={expected},te={expected},vae={expected}",
        ], (position, flags)

    # A third card of that name physically, but only two in the Vulkan namespace: the pin is dropped.
    monkeypatch.setattr(
        sd_cpp_backend, "physical_card_name", lambda _ordinal: ("AMD Radeon RX 7900 XTX", 2)
    )
    assert sd_cpp_backend._offload_with_device_pin_impl([], "/sd/sd-cli", 2) == []


@pytest.mark.parametrize("platform", PLATFORMS)
def test_a_failed_fetch_does_not_divert_a_host_whose_rocm_build_works(
    h3_amd_host, fake_settings, platform
):
    """The ensure answers None for a failed download exactly as for a missing asset, and with no binary the fingerprint carries no bundle tag to retire the record."""
    from core.inference import sd_cpp_backend

    failed_fetch = h3_amd_host(
        platform = platform, backend = "rocm", device = "cuda", devices = _VULKAN_ONLY
    )
    assert failed_fetch.run()._state.device == "cuda"
    assert failed_fetch.ensured == ["rocm", "vulkan"], failed_fetch.ensured
    assert _recorded_strikes(fake_settings) == 1
    assert _noted_accelerators(fake_settings) == []
    assert sd_cpp_backend.preferred_accelerator("rocm") == "rocm"

    recovered = h3_amd_host(platform = platform, backend = "rocm", device = "cuda", devices = _ROCM_WORKS)
    assert recovered.run()._state.device == "cuda"
    assert recovered.ensured == ["rocm"], recovered.ensured

    again = h3_amd_host(platform = platform, backend = "rocm", device = "cuda", devices = _VULKAN_ONLY)
    again.run()
    assert _recorded_strikes(fake_settings) == 2
    assert _noted_accelerators(fake_settings) == ["rocm"]


@pytest.mark.parametrize("platform", PLATFORMS)
def test_a_substituted_cpu_build_is_not_evidence_about_rocm(
    h3_amd_host, fake_settings, monkeypatch, platform
):
    """The ensure deliberately keeps a usable build of the WRONG class, so the binary that answers is not always the one on trial."""
    from core.inference import sd_cpp_backend

    host = h3_amd_host(
        platform = platform,
        backend = "rocm",
        device = "cuda",
        devices = {
            "rocm": MISSING,
            "vulkan": _DEVICES_VULKAN,
            "cpu": _DEVICES_CPU_ONLY,
        },
    )
    ensure = sd_cpp_backend.ensure_sd_cpp_binary

    def _substituting_ensure(*, allow_install = True, accelerator = "cpu"):
        if accelerator == "rocm":
            return "/opt/sd/cpu/sd-cli"
        return ensure(allow_install = allow_install, accelerator = accelerator)

    monkeypatch.setattr(sd_cpp_backend, "ensure_sd_cpp_binary", _substituting_ensure)

    assert host.run()._state.device == "cuda"
    assert _noted_accelerators(fake_settings) == []
    assert _recorded_strikes(fake_settings) == 1
    assert sd_cpp_backend.preferred_accelerator("rocm") == "rocm"


def test_a_server_that_starts_and_dies_is_recorded_from_its_own_output(fake_settings, monkeypatch):
    """A ROCm build that cannot come up at all fails inside the load, where the error carries the child's own output."""
    import inspect

    from core.inference import sd_cpp_backend

    source = inspect.getsource(sd_cpp_backend.SdCppDiffusionBackend._run_load)
    start = source.index("sd-server failed to start")
    window = source[start : start + 900]
    assert "note_accelerator_failure_from_output(" in window, window
    assert "server_binary" in window, window

    monkeypatch.setattr(sd_cpp_backend, "_installed_accelerator_of", lambda _b: "rocm")
    sd_cpp_backend.note_accelerator_failure_from_output(
        "/opt/sd/rocm/sd-server",
        "sd-server exited 1. Last output:\nROCm error: no kernel image is available for "
        "execution on the device",
    )
    assert _noted_accelerators(fake_settings) == ["rocm"]


def test_a_build_that_cannot_launch_at_all_is_counted(fake_settings, monkeypatch):
    """0xC0000135 arrives as a large POSITIVE exit code, so the probe accepts it; a missing execute bit fails identically, so it is only counted."""
    from core.inference import sd_cpp_backend

    monkeypatch.setattr(sd_cpp_backend, "_installed_accelerator_of", lambda _b: "rocm")
    sd_cpp_backend.note_unlaunchable_accelerator_build("/opt/sd/rocm/sd-server")
    assert _recorded_strikes(fake_settings) == 1
    assert _noted_accelerators(fake_settings) == [], "one launch failure must not divert"
    sd_cpp_backend.note_unlaunchable_accelerator_build("/opt/sd/rocm/sd-server")
    assert _noted_accelerators(fake_settings) == ["rocm"], "a host that keeps failing is moved"

    monkeypatch.setattr(sd_cpp_backend, "_installed_accelerator_of", lambda _b: "cpu")
    sd_cpp_backend.note_unlaunchable_accelerator_build("/opt/sd/cpu/sd-server")
    monkeypatch.setattr(sd_cpp_backend, "_installed_accelerator_of", lambda _b: None)
    sd_cpp_backend.note_unlaunchable_accelerator_build("/somewhere/else/sd-server")
    assert _noted_accelerators(fake_settings) == ["rocm"]
    assert _recorded_strikes(fake_settings, "cpu") == 0


def test_both_unlaunchable_load_paths_record_before_they_raise():
    import inspect

    from core.inference import sd_cpp_backend

    source = inspect.getsource(sd_cpp_backend.SdCppDiffusionBackend._run_load)
    for raised in (
        'raise RuntimeError("sd-server binary is present but not runnable.")',
        'raise RuntimeError("sd-cli binary is present but not runnable.")',
    ):
        arm = source[: source.index(raised)]
        assert "note_unlaunchable_accelerator_build(" in arm[-400:], raised


def test_a_resident_server_does_not_cost_the_reload_its_native_engine(fake_settings, monkeypatch):
    """A resident server executes out of the tree, so both ensures decline the install; the load stops it and lands the deferred install itself."""
    from core.inference import diffusion_engine_router as router
    from core.inference import sd_cpp_backend

    monkeypatch.setattr(router, "_install_allowed", lambda: True)
    monkeypatch.setattr(router, "ensure_sd_server_binary", lambda **_k: "/opt/sd/rocm/sd-server")
    monkeypatch.setattr(router, "ensure_sd_cpp_binary", lambda **_k: "/opt/sd/rocm/sd-cli")
    monkeypatch.setattr(router, "_server_binary_runnable", lambda _b: True)
    monkeypatch.setattr(
        router, "SdCppEngine", lambda binary: types.SimpleNamespace(version = lambda: "1")
    )
    monkeypatch.setattr(
        router,
        "resolve_diffusion_device_target",
        lambda: types.SimpleNamespace(backend = "rocm", device = "cuda", dtype = None),
    )
    monkeypatch.setattr(router, "family_sd_cpp_supported", lambda _fam: True)
    monkeypatch.setattr(router, "_activate", lambda name, reason = None: name)
    monkeypatch.setattr(sd_cpp_backend, "_installed_accelerator_of", lambda _b: "rocm")
    monkeypatch.setenv("UNSLOTH_DIFFUSION_ENGINE", "sd_cpp")
    sd_cpp_backend.note_accelerator_runtime_failure("rocm", proven = True)
    assert sd_cpp_backend.preferred_accelerator("rocm") == "vulkan"

    family = _detect_load_family(H3_REPO, None, "minimax-h3")

    monkeypatch.setattr(router, "_managed_tree_in_use", lambda: True)
    assert (
        router.select_and_activate_engine(family) == "sd_cpp"
    ), "the reload that should have upgraded to Vulkan behind the teardown went to diffusers"

    monkeypatch.setattr(router, "_managed_tree_in_use", lambda: False)
    assert router.select_and_activate_engine(family) == "diffusers"

    # With installing switched off the deferred upgrade hands back the same ROCm path.
    monkeypatch.setattr(router, "_managed_tree_in_use", lambda: True)
    monkeypatch.setattr(router, "_install_allowed", lambda: False)
    assert (
        router.select_and_activate_engine(family) == "diffusers"
    ), "a deferred upgrade that can never install kept the condemned build"


def test_the_router_counts_a_binary_it_rejects_for_not_launching(fake_settings, monkeypatch):
    """Selection runs BEFORE the load, so the load's own recorders never see this build.

    ONE strike, though the server and the CLI both failed: they are two executables out of a single
    install, so the second record would say nothing new while carrying the bundle straight to the
    two-strike diversion bar, condemning ROCm on one install event rather than on two independent
    failures. The run still falls back to diffusers, which is the part that protects the request."""
    from core.inference import diffusion_engine_router as router
    from core.inference import sd_cpp_backend

    monkeypatch.setattr(router, "_install_allowed", lambda: True)
    monkeypatch.setattr(router, "ensure_sd_server_binary", lambda **_k: "/opt/sd/rocm/sd-server")
    monkeypatch.setattr(router, "ensure_sd_cpp_binary", lambda **_k: "/opt/sd/rocm/sd-cli")
    monkeypatch.setattr(router, "_server_binary_runnable", lambda _b: False)
    monkeypatch.setattr(
        router, "SdCppEngine", lambda binary: types.SimpleNamespace(version = lambda: None)
    )
    monkeypatch.setattr(
        router,
        "resolve_diffusion_device_target",
        lambda: types.SimpleNamespace(backend = "rocm", device = "cuda", dtype = None),
    )
    monkeypatch.setattr(router, "family_sd_cpp_supported", lambda _fam: True)
    monkeypatch.setattr(router, "_activate", lambda name, reason = None: name)
    monkeypatch.setattr(sd_cpp_backend, "_installed_accelerator_of", lambda _b: "rocm")
    monkeypatch.setenv("UNSLOTH_DIFFUSION_ENGINE", "sd_cpp")

    chosen = router.select_and_activate_engine(_detect_load_family(H3_REPO, None, "minimax-h3"))
    assert chosen == "diffusers", chosen
    assert _recorded_strikes(fake_settings) == 1
    # One strike is not a diversion: ROCm is still what the next load will try.
    assert sd_cpp_backend.accelerator_runtime_failed("rocm", None) is False
    assert _noted_accelerators(fake_settings) == []


def _backend_with_a_deferred_upgrade(
    monkeypatch,
    *,
    delivered,
    requested = "vulkan",
):
    from core.inference import sd_cpp_backend

    backend = sd_cpp_backend.SdCppDiffusionBackend.__new__(sd_cpp_backend.SdCppDiffusionBackend)
    monkeypatch.setattr(
        sd_cpp_backend.SdCppDiffusionBackend,
        "_upgrade_server_after_teardown",
        lambda _self, _binary: delivered,
        raising = False,
    )
    monkeypatch.setattr(
        sd_cpp_backend.SdCppDiffusionBackend,
        "_resolved_accelerator",
        lambda _self, _card = None: requested,
        raising = False,
    )
    # Built with __new__, so give it the per-load field the resolutions read.
    backend._loading_card = None
    monkeypatch.setattr(
        sd_cpp_backend,
        "_installed_accelerator_of",
        lambda binary: "rocm" if binary and "rocm" in binary else "vulkan",
    )
    sd_cpp_backend.note_accelerator_runtime_failure("rocm", proven = True)
    return backend


def test_a_deferred_upgrade_that_did_not_deliver_does_not_start_the_condemned_build(
    fake_settings, monkeypatch
):
    """``_upgrade_server_after_teardown`` is never fatal: offline or on a failed download it hands back the path it was given."""
    backend = _backend_with_a_deferred_upgrade(monkeypatch, delivered = "/opt/sd/rocm/sd-server")
    with pytest.raises(RuntimeError, match = "recorded as failing"):
        backend._upgraded_or_refused("/opt/sd/rocm/sd-server", mode = "server", engine = None)


def test_a_deferred_upgrade_that_delivered_is_started(fake_settings, monkeypatch):
    backend = _backend_with_a_deferred_upgrade(monkeypatch, delivered = "/opt/sd/vulkan/sd-server")
    assert (
        backend._upgraded_or_refused("/opt/sd/rocm/sd-server", mode = "server", engine = None)
        == "/opt/sd/vulkan/sd-server"
    )


def test_the_build_that_was_asked_for_is_still_run_after_the_teardown(fake_settings, monkeypatch):
    """With the fallback switched off the request is ROCm on purpose, and that opt-out means run it anyway."""
    backend = _backend_with_a_deferred_upgrade(
        monkeypatch, delivered = "/opt/sd/rocm/sd-server", requested = "rocm"
    )
    assert (
        backend._upgraded_or_refused("/opt/sd/rocm/sd-server", mode = "server", engine = None)
        == "/opt/sd/rocm/sd-server"
    )


def test_a_serverless_upgrade_is_judged_by_the_cli_this_load_will_run(fake_settings, monkeypatch):
    """A one-shot load resolves to sd-cli precisely BECAUSE the deferral suppressed the install."""
    backend = _backend_with_a_deferred_upgrade(monkeypatch, delivered = None)
    engine = types.SimpleNamespace(binary = "/opt/sd/rocm/sd-cli")
    with pytest.raises(RuntimeError, match = "recorded as failing"):
        backend._upgraded_or_refused(None, mode = "oneshot", engine = engine)

    engine = types.SimpleNamespace(binary = "/opt/sd/vulkan/sd-cli")
    assert backend._upgraded_or_refused(None, mode = "oneshot", engine = engine) is None


def test_the_load_path_takes_the_deferred_upgrade_through_the_record_check(fake_settings):
    import inspect

    from core.inference import sd_cpp_backend

    body = inspect.getsource(sd_cpp_backend.SdCppDiffusionBackend._run_load)
    assert "_upgraded_or_refused(" in body
    assert (
        "_upgrade_server_after_teardown(" not in body
    ), "the load path went around the record check again"


def _inventory_of(
    monkeypatch,
    devices,
    hip_by_row = None,
):
    import types as _types

    from utils.hardware import amd as amd_module
    from utils.hardware import hardware as hardware_module

    monkeypatch.setattr(
        hardware_module,
        "get_physical_gpu_inventory",
        lambda **_k: {"unknown": False, "devices": devices},
        raising = False,
    )
    monkeypatch.setattr(
        amd_module,
        "get_hip_id_by_gpu_index",
        lambda: hip_by_row if hip_by_row is not None else {d["index"]: d["index"] for d in devices},
        raising = False,
    )
    return _types.SimpleNamespace()


def test_one_card_that_cannot_run_the_build_does_not_divert_the_other(fake_settings, monkeypatch):
    """One ROCm bundle, two gfx targets: the build can carry code for one card on this host and none for the other."""
    from core.inference import sd_cpp_backend

    sd_cpp_backend.note_accelerator_runtime_failure("rocm", proven = True, card = "Card A@gfx1201")
    assert sd_cpp_backend.accelerator_runtime_failed("rocm", "Card A@gfx1201") is True
    assert sd_cpp_backend.accelerator_runtime_failed("rocm", "Card B@gfx1100") is False
    assert sd_cpp_backend.accelerator_runtime_failed("rocm") is True
    assert sd_cpp_backend.preferred_accelerator("rocm", "Card B@gfx1100") == "rocm"
    assert sd_cpp_backend.preferred_accelerator("rocm", "Card A@gfx1201") == "vulkan"


def test_a_second_card_failing_the_same_build_is_added_not_substituted(fake_settings, monkeypatch):
    from core.inference import sd_cpp_backend

    sd_cpp_backend.note_accelerator_runtime_failure("rocm", proven = True, card = "Card A@gfx1201")
    sd_cpp_backend.note_accelerator_runtime_failure("rocm", proven = True, card = "Card B@gfx1100")
    assert sd_cpp_backend.accelerator_runtime_failed("rocm", "Card A@gfx1201") is True
    assert sd_cpp_backend.accelerator_runtime_failed("rocm", "Card B@gfx1100") is True
    assert sd_cpp_backend.accelerator_runtime_failed("rocm", "Card C@gfx1030") is False


def test_a_record_written_before_the_cards_were_named_still_diverts(fake_settings, monkeypatch):
    from core.inference import sd_cpp_backend
    sd_cpp_backend.note_accelerator_runtime_failure("rocm", proven = True)
    assert sd_cpp_backend.accelerator_runtime_failed("rocm", "Card B@gfx1100") is True


def test_evidence_no_card_was_named_for_survives_a_later_card_tally(fake_settings, monkeypatch):
    """A failure recorded without a card applies to every card. A later failure naming card A must
    add to it, not hide it: before, A's fresh one-strike tally replaced it for A, and the new card
    list excluded every other card from it, a proven host-wide record included."""
    from core.inference import sd_cpp_backend

    sd_cpp_backend.note_accelerator_runtime_failure("rocm", proven = False)
    sd_cpp_backend.note_accelerator_runtime_failure("rocm", proven = False, card = "Card A@gfx1201")
    for _ in range(2):  # in process, then read back from the store
        assert sd_cpp_backend.accelerator_runtime_failed("rocm", "Card A@gfx1201") is True
        assert sd_cpp_backend.accelerator_runtime_failed("rocm", "Card B@gfx1100") is False
        sd_cpp_backend._accelerator_runtime_failures.clear()

    fake_settings.clear()
    sd_cpp_backend.note_accelerator_runtime_failure("rocm", proven = True)
    sd_cpp_backend.note_accelerator_runtime_failure("rocm", proven = False, card = "Card A@gfx1201")
    for _ in range(2):
        assert sd_cpp_backend.accelerator_runtime_failed("rocm", "Card A@gfx1201") is True
        assert sd_cpp_backend.accelerator_runtime_failed("rocm", "Card B@gfx1100") is True
        sd_cpp_backend._accelerator_runtime_failures.clear()


def test_the_condemned_build_is_still_refused_on_the_card_that_failed(fake_settings, monkeypatch):
    from core.inference import sd_cpp_backend

    monkeypatch.setattr(sd_cpp_backend, "_installed_accelerator_of", lambda _b: "rocm")
    sd_cpp_backend.note_accelerator_runtime_failure("rocm", proven = True, card = "Card A@gfx1201")
    assert (
        sd_cpp_backend.usable_or_recorded_failure("/opt/sd/rocm/sd-cli", "vulkan", "Card A@gfx1201")
        is None
    )
    assert (
        sd_cpp_backend.usable_or_recorded_failure("/opt/sd/rocm/sd-cli", "vulkan", "Card B@gfx1100")
        == "/opt/sd/rocm/sd-cli"
    )


def test_the_selected_card_is_read_through_the_visibility_mask(fake_settings, monkeypatch):
    from core.inference import sd_cpp_backend

    devices = [
        {"index": 0, "name": "Card A", "gfx_candidates": ["gfx1201"], "vendor": "amd"},
        {"index": 1, "name": "Card B", "gfx_candidates": ["gfx1100"], "vendor": "amd"},
    ]
    _inventory_of(monkeypatch, devices)
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "1")
    monkeypatch.delenv("ROCR_VISIBLE_DEVICES", raising = False)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising = False)

    identity = sd_cpp_backend.selected_card_identity(0)
    assert identity == sd_cpp_backend._card_identity(devices[1]), identity


def test_an_unreadable_enumeration_names_no_card(fake_settings, monkeypatch):
    from core.inference import sd_cpp_backend
    from utils.hardware import hardware as hardware_module

    monkeypatch.setattr(
        hardware_module,
        "get_physical_gpu_inventory",
        lambda **_k: {"unknown": True},
        raising = False,
    )
    assert sd_cpp_backend.selected_card_identity(0) is None
    assert sd_cpp_backend.selected_card_identity(None) is None


def test_the_route_tells_the_selection_which_card_it_picked(fake_settings):
    import inspect

    from core.inference import diffusion_engine_router as router

    body = inspect.getsource(router.select_and_activate_engine)
    assert "_selected_card(gpu_ordinal)" in body
    assert (
        "preferred_accelerator(\n            _install_accelerator_for(backend), selected_card\n        )"
        in body
    )

    import routes.inference as inference_routes

    route_source = inspect.getsource(inference_routes)
    call = route_source.split("                select_and_activate_engine,", 1)[1][:600]
    # The ordinal the route ALREADY resolved, never the id list: re-resolving re-ranks a multi-card
    # pick by free VRAM, so selection could answer for a card this load does not run on.
    assert "gpu_ordinal = gpu_ordinal" in call, call
    assert "gpu_ids = request.gpu_ids" not in call, call


def test_the_selection_never_re_ranks_a_multi_card_pick_for_itself(monkeypatch):
    """Ranking is what turns several ids into one ordinal, and free VRAM moves as the load stages."""
    from core.inference import diffusion_device
    from core.inference import diffusion_engine_router as router
    from core.inference import sd_cpp_backend

    monkeypatch.setattr(
        diffusion_device,
        "resolve_selected_cuda_ordinal",
        lambda *_a, **_k: pytest.fail("selection re-resolved the ordinal for itself"),
        raising = False,
    )
    monkeypatch.setattr(sd_cpp_backend, "selected_card_identity", lambda ordinal: f"Card {ordinal}")
    assert router._selected_card(1) == "Card 1"
    assert router._selected_card(None) is None


def test_the_backend_resolution_asks_about_the_card_this_load_selected(fake_settings, monkeypatch):
    from core.inference import sd_cpp_backend

    monkeypatch.setattr(
        sd_cpp_backend,
        "resolve_diffusion_device_target",
        lambda: types.SimpleNamespace(backend = "rocm", device = "cuda", dtype = None),
        raising = False,
    )
    sd_cpp_backend.note_accelerator_runtime_failure("rocm", proven = True, card = "Card A@gfx1201")

    backend = sd_cpp_backend.SdCppDiffusionBackend.__new__(sd_cpp_backend.SdCppDiffusionBackend)
    assert backend._resolved_accelerator("Card B@gfx1100") == "rocm"
    assert backend._resolved_accelerator("Card A@gfx1201") == "vulkan"
    assert backend._resolved_accelerator() == "vulkan"


def test_every_in_load_resolution_reads_the_same_card(fake_settings):
    """``_accelerator_changed`` compares the tree against this answer, so a resolution that skipped the card would reinstall over what the others chose."""
    import inspect

    from core.inference import sd_cpp_backend

    source = inspect.getsource(sd_cpp_backend)
    assert (
        source.count("self._resolved_accelerator()") == 0
    ), "a resolution inside the load still asks without the card"
    run_load = inspect.getsource(sd_cpp_backend.SdCppDiffusionBackend._run_load)
    assert "self._loading_card = selected_card_identity(gpu_ordinal)" in run_load
    assert run_load.index("_loading_card = selected_card_identity") < run_load.index(
        "self._resolve_backend()"
    )


def test_a_video_render_failure_names_the_card_it_was_rendering_on(fake_settings, monkeypatch):
    from core.inference import sd_cpp_backend, video as video_mod

    monkeypatch.setattr(sd_cpp_backend, "_installed_accelerator_of", lambda _b: "rocm")
    monkeypatch.setattr(
        video_mod,
        "_note_sd_cpp_accelerator_failure",
        video_mod._note_sd_cpp_accelerator_failure,
        raising = False,
    )
    video_mod._note_sd_cpp_accelerator_failure(
        "/opt/sd/rocm/sd-cli",
        "ROCm error: CUBLAS_STATUS_INVALID_VALUE at hipblasSetStream",
        card = "Card 1@gfx1201",
    )
    assert sd_cpp_backend.accelerator_runtime_failed("rocm", "Card 1@gfx1201") is True
    assert sd_cpp_backend.accelerator_runtime_failed("rocm", "Card 0@gfx1201") is False


_CARD_A = "AMD Radeon RX 7900 XTX@gfx1100"
_CARD_B = "AMD Radeon RX 9070 XT@gfx1201"


def test_one_cards_decisive_failure_is_not_proof_about_another(fake_settings):
    """A record naming several cards carried ONE verdict, so the first ambiguous error on an
    otherwise-working card inherited the other card's proof and diverted it on the spot."""
    from core.inference import sd_cpp_backend

    sd_cpp_backend.note_accelerator_runtime_failure("rocm", proven = True, card = _CARD_A)
    sd_cpp_backend.note_accelerator_runtime_failure("rocm", proven = False, card = _CARD_B)
    assert sd_cpp_backend.accelerator_runtime_failed("rocm", _CARD_A) is True
    assert sd_cpp_backend.accelerator_runtime_failed("rocm", _CARD_B) is False
    # B's own second strike is B's own evidence, and that does convict it.
    sd_cpp_backend.note_accelerator_runtime_failure("rocm", proven = False, card = _CARD_B)
    assert sd_cpp_backend.accelerator_runtime_failed("rocm", _CARD_B) is True


def test_one_ambiguous_failure_on_each_of_two_cards_convicts_neither(fake_settings):
    """Shared strike counts let one ambiguous failure per card satisfy the two-strike threshold for both."""
    from core.inference import sd_cpp_backend

    assert sd_cpp_backend._AMBIGUOUS_FAILURE_STRIKES == 2
    sd_cpp_backend.note_accelerator_runtime_failure("rocm", proven = False, card = _CARD_A)
    sd_cpp_backend.note_accelerator_runtime_failure("rocm", proven = False, card = _CARD_B)
    assert sd_cpp_backend.accelerator_runtime_failed("rocm", _CARD_A) is False
    assert sd_cpp_backend.accelerator_runtime_failed("rocm", _CARD_B) is False
    # A caller that cannot name its card is still answered with the host-wide tally, as before.
    assert sd_cpp_backend.accelerator_runtime_failed("rocm") is True


def test_the_per_card_tallies_survive_the_store(fake_settings):
    """Dropped on the way out, the next process reads the union back and diverts the working card."""
    from core.inference import sd_cpp_backend

    sd_cpp_backend.note_accelerator_runtime_failure("rocm", proven = True, card = _CARD_A)
    sd_cpp_backend.note_accelerator_runtime_failure("rocm", proven = False, card = _CARD_B)
    # The next process has only the store.
    sd_cpp_backend._accelerator_runtime_failures.clear()
    assert sd_cpp_backend.accelerator_runtime_failed("rocm", _CARD_A) is True
    assert sd_cpp_backend.accelerator_runtime_failed("rocm", _CARD_B) is False


def test_a_record_written_before_the_per_card_tallies_still_applies(fake_settings):
    """Back-compat: a record with cards but no per-card notes keeps the verdict it was saved with."""
    from core.inference import sd_cpp_backend

    record = sd_cpp_backend._normalise_failure_record(
        "rocm", {"strikes": 1, "proven": True, "fingerprint": {}, "cards": [_CARD_A]}
    )
    assert "per_card" not in record, record
    assert sd_cpp_backend._record_diverts(record, {}, _CARD_A) is True
    assert sd_cpp_backend._record_diverts(record, {}, _CARD_B) is False


def test_two_launch_failures_on_one_card_do_not_divert_another(fake_settings, monkeypatch):
    from core.inference import sd_cpp_backend

    monkeypatch.setattr(sd_cpp_backend, "_installed_accelerator_of", lambda _b: "rocm")
    for _ in range(sd_cpp_backend._AMBIGUOUS_FAILURE_STRIKES):
        sd_cpp_backend.note_unlaunchable_accelerator_build("/opt/sd/rocm/sd-server", card = _CARD_A)
    assert sd_cpp_backend.accelerator_runtime_failed("rocm", _CARD_A) is True
    assert sd_cpp_backend.accelerator_runtime_failed("rocm", _CARD_B) is False


def test_the_router_counts_its_launch_failures_against_the_card_it_selected(fake_settings):
    """Both router-side recorders: a cards-less note reads as host-wide and moves every card."""
    from core.inference import diffusion_engine_router as router

    body = inspect.getsource(router.select_and_activate_engine)
    calls = body.count("note_unlaunchable_accelerator_build(")
    assert calls == 1, body  # one bundle, one recorder
    assert body.count("card = selected_card") == calls, body


def test_the_h3_load_scopes_its_record_lookup_to_the_card_it_selected(fake_settings):
    """The render failures this path writes are card-scoped, so reading them back must be too."""
    from core.inference import video as video_mod

    source = inspect.getsource(video_mod)
    load = source[source.index("allow_install = _install_allowed()") :]
    assert "selected_card = selected_card_identity(gpu_ordinal)" in load
    assert "preferred_accelerator(_install_accelerator_for(target.backend), selected_card)" in load
    start = 0
    guarded = 0
    while True:
        found = load.find("usable_or_recorded_failure(", start)
        if found < 0:
            break
        window = load[found : found + 260]
        assert "selected_card" in window, window
        guarded += 1
        start = found + 1
    assert guarded == load.count("ensure_h3_sd_cpp_binary("), guarded
    note = load.index("note_accelerator_runtime_failure(")
    assert "card = selected_card" in load[note : note + 900], load[note : note + 900]


def test_a_cancelled_workers_card_does_not_leak_into_the_replacement_load(fake_settings):
    """``unload`` clears ``_loading`` before the cancelled worker exits, so a replacement load's
    thread can write this while the old worker is still inside ``_run_load``; shared, whichever
    wrote last decided the accelerator for both."""
    from core.inference import sd_cpp_backend

    backend = sd_cpp_backend.SdCppDiffusionBackend.__new__(sd_cpp_backend.SdCppDiffusionBackend)
    seen: dict = {}
    ready = {_CARD_A: threading.Event(), _CARD_B: threading.Event()}
    go = threading.Event()

    def worker(card):
        backend._loading_card = card
        ready[card].set()
        go.wait(10)
        seen[card] = backend._loading_card

    first = threading.Thread(target = worker, args = (_CARD_A,))
    first.start()
    assert ready[_CARD_A].wait(10)
    second = threading.Thread(target = worker, args = (_CARD_B,))
    second.start()
    assert ready[_CARD_B].wait(10)
    go.set()
    first.join(10)
    second.join(10)
    assert seen == {_CARD_A: _CARD_A, _CARD_B: _CARD_B}, seen
    # Off a load thread -- a generation re-resolving sd-cli -- the last COMMITTED load's card stands.
    # Neither worker here reached the _state commit, so there is none, and None is the honest answer:
    # a started load is not a loaded model, and naming card B while nothing is loaded is what sends a
    # one-shot generation to the wrong build. Committing publishes it, which the next test pins.
    assert backend._loading_card is None
    backend._committed_loading_card = _CARD_A
    assert backend._loading_card == _CARD_A


# ── the host runtime preflight, from the two failure shapes measured on real hardware ──────────
class TestRocmRuntimePreflight:
    """The ROCm prebuilt ships no HIP or BLAS runtime and takes all of it from the host, so a host
    without one can be identified BEFORE the 244 MB download rather than after a failed load.

    Both shapes this guards were measured, not imagined:
      Linux, no ROCm    ``libggml-hip.so`` fails to dlopen with "libhipblas.so.3: cannot open shared
                        object file", sd-cli catches it, loads the CPU backend, exits 0 and lists CPU
                        only. No error text and no non-zero exit, so no marker can ever fire.
      Windows, no DLLs  exit 0xC0000135, zero bytes, already decisive via the exit status.
    """

    @staticmethod
    def _sonames_that(fail: set) -> object:
        import ctypes

        real = ctypes.CDLL

        def _fake(name, *a, **k):
            if name in fail:
                raise OSError(f"{name}: cannot open shared object file: No such file or directory")
            if name in sd_backend._ROCM_RUNTIME_SONAMES:
                return object()
            return real(name, *a, **k)

        return _fake

    def test_a_host_missing_hipblas_is_diverted_without_downloading(self, monkeypatch):
        import ctypes

        monkeypatch.setattr(sd_backend.os, "name", "posix")
        monkeypatch.setattr(sd_backend.sys, "platform", "linux")
        monkeypatch.setattr(ctypes, "CDLL", self._sonames_that({"libhipblas.so.3"}))
        assert sd_backend.rocm_runtime_resolvable() is False
        # A negative probe is now EXPLAINED, so one occurrence suffices instead of two.
        assert sd_backend.accelerator_probe_failure_is_decisive("rocm") is True

    def test_a_host_with_rocm_keeps_rocm(self, monkeypatch):
        import ctypes

        monkeypatch.setattr(sd_backend.os, "name", "posix")
        monkeypatch.setattr(sd_backend.sys, "platform", "linux")
        monkeypatch.setattr(ctypes, "CDLL", self._sonames_that(set()))
        assert sd_backend.rocm_runtime_resolvable() is True
        # ROCm is present, so a negative probe stays ambiguous and the two-strike rule protects it.
        assert sd_backend.accelerator_probe_failure_is_decisive("rocm") is False

    def test_an_unanswerable_host_is_not_diverted(self, monkeypatch):
        """None must never divert. A host that cannot be asked behaves exactly as it did before,
        which is what keeps this from being a new way to lose a working GPU."""
        monkeypatch.setattr(sd_backend, "rocm_runtime_resolvable", lambda: None)
        assert sd_backend.accelerator_probe_failure_is_decisive("rocm") is False

    def test_windows_is_left_to_the_exit_status(self, monkeypatch):
        """On Windows the same condition is already decisive from the real binary's 0xC0000135, so
        the preflight declines to guess at DLL search order."""
        monkeypatch.setattr(sd_backend.os, "name", "nt")
        assert sd_backend.rocm_runtime_resolvable() is None

    def test_the_preflight_never_touches_a_non_rocm_accelerator(self, monkeypatch):
        """Only rocm has a rung below it. A CUDA or Vulkan host must not consult this at all."""
        called = []
        monkeypatch.setattr(
            sd_backend, "rocm_runtime_resolvable", lambda: called.append(1) or False
        )
        assert sd_backend.accelerator_probe_failure_is_decisive("cuda") is False
        assert sd_backend.accelerator_probe_failure_is_decisive("vulkan") is False
        assert called == []

    def test_the_off_switch_beats_the_preflight(self, monkeypatch):
        """UNSLOTH_DIFFUSION_SD_CPP_VULKAN_FALLBACK=0 means see the ROCm failure, including this one."""
        import ctypes

        monkeypatch.setenv("UNSLOTH_DIFFUSION_SD_CPP_VULKAN_FALLBACK", "0")
        monkeypatch.setattr(sd_backend.os, "name", "posix")
        monkeypatch.setattr(sd_backend.sys, "platform", "linux")
        monkeypatch.setattr(ctypes, "CDLL", self._sonames_that({"libhipblas.so.3"}))
        # The off switch removes the rung, so nothing about rocm can be decisive for a fallback.
        assert sd_backend.accelerator_probe_failure_is_decisive("rocm") is False


# ── the two index spaces, and the mask that does not exist on Windows ──────────────────────
class TestSelectedCardIndexSpaces:
    """``selected_card_identity`` turns a torch ordinal into a card. Getting the wrong card is worse
    than getting none: the failure is persisted against it, so the card that really fails keeps being
    retried while a healthy one is diverted to Vulkan for the rest of the install's life."""

    def test_an_unmasked_host_still_translates_the_hip_id(self, fake_settings, monkeypatch):
        """No mask means the ordinal IS the HIP id, not an inventory row. amd-smi's discovery order is
        a different index space, so the unmasked path needs the same mapping the masked one does."""
        from core.inference import sd_cpp_backend

        devices = [
            {"index": 0, "name": "Card A", "gfx_candidates": ["gfx1201"], "vendor": "amd"},
            {"index": 1, "name": "Card B", "gfx_candidates": ["gfx1100"], "vendor": "amd"},
        ]
        # HIP id 0 is probe row 1 on this host: the spaces disagree, which is the whole point.
        _inventory_of(monkeypatch, devices, hip_by_row = {0: 1, 1: 0})
        for variable in ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"):
            monkeypatch.delenv(variable, raising = False)

        # Torch ordinal 0 -> HIP id 0 -> probe row 1 -> Card B. Before the fix this read row 0, Card A.
        assert sd_cpp_backend.selected_card_identity(0) == sd_cpp_backend._card_identity(devices[1])

    def test_a_multi_gpu_host_with_no_mapping_names_no_card(self, fake_settings, monkeypatch):
        """``amd-smi list -e`` arrived in ROCm 6.4. Without it, which row a HIP id means is a guess,
        and a guess here pins the fallback to whichever card the guess happened to land on."""
        from core.inference import sd_cpp_backend

        devices = [
            {"index": 0, "name": "Card A", "gfx_candidates": ["gfx1201"], "vendor": "amd"},
            {"index": 1, "name": "Card B", "gfx_candidates": ["gfx1100"], "vendor": "amd"},
        ]
        _inventory_of(monkeypatch, devices, hip_by_row = None)
        from utils.hardware import amd as amd_module

        monkeypatch.setattr(amd_module, "get_hip_id_by_gpu_index", lambda: None, raising = False)
        for variable in ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"):
            monkeypatch.delenv(variable, raising = False)

        assert sd_cpp_backend.selected_card_identity(0) is None

    def test_a_single_gpu_host_with_no_mapping_still_names_its_card(
        self, fake_settings, monkeypatch
    ):
        """One card is the case this fallback exists for (gfx1151 Strix Halo, every APU). With a
        single inventory row the identity mapping is the only one there is, so declining would cost
        failure attribution on every pre-6.4 ROCm host and buy no safety."""
        from core.inference import sd_cpp_backend
        from utils.hardware import amd as amd_module

        devices = [{"index": 0, "name": "Card A", "gfx_candidates": ["gfx1151"], "vendor": "amd"}]
        _inventory_of(monkeypatch, devices)
        monkeypatch.setattr(amd_module, "get_hip_id_by_gpu_index", lambda: None, raising = False)
        for variable in ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"):
            monkeypatch.delenv(variable, raising = False)

        assert sd_cpp_backend.selected_card_identity(0) == sd_cpp_backend._card_identity(devices[0])

    def test_a_cold_inventory_still_names_the_selected_card_off_the_event_loop(self, monkeypatch):
        """On a host whose torch works nothing reads the inventory at startup, so the first load met
        the non-blocking "unknown" answer and recorded its failure against every card. A worker
        thread reads it blocking; the event loop still never does."""
        import asyncio

        from core.inference import sd_cpp_backend
        from utils.hardware import amd as amd_module
        from utils.hardware import hardware as hardware_module

        devices = [
            {"index": 0, "name": "Card A", "gfx_candidates": ["gfx1201"], "vendor": "amd"},
            {"index": 1, "name": "Card B", "gfx_candidates": ["gfx1100"], "vendor": "amd"},
        ]
        calls: list[bool] = []

        def _inventory(*, block = True):
            calls.append(block)
            if block:
                return {"unknown": False, "devices": devices}
            return {"unknown": True, "devices": []}

        monkeypatch.setattr(hardware_module, "get_physical_gpu_inventory", _inventory)
        monkeypatch.setattr(amd_module, "get_hip_id_by_gpu_index", lambda: {0: 0, 1: 1})
        for variable in ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"):
            monkeypatch.delenv(variable, raising = False)

        assert sd_cpp_backend.selected_card_identity(1) == sd_cpp_backend._card_identity(devices[1])
        assert calls == [False, True]

        calls.clear()

        async def _on_loop():
            return sd_cpp_backend.selected_card_identity(1)

        assert asyncio.run(_on_loop()) is None
        assert calls == [False]

    def test_another_vendors_row_at_the_same_index_is_not_the_amd_card(self, monkeypatch):
        """Inventory indices are vendor-local, and an Intel iGPU beside an AMD card is an ordinary
        desktop. The iGPU's row 0 answered for AMD row 0 when it was listed first."""
        from core.inference import sd_cpp_backend

        devices = [
            {"index": 0, "name": "Intel(R) UHD Graphics 770", "vendor": "intel"},
            {
                "index": 0,
                "name": "AMD Radeon RX 7900 XTX",
                "gfx_candidates": ["gfx1100"],
                "vendor": "amd",
            },
        ]
        _inventory_of(monkeypatch, devices, hip_by_row = {0: 0})
        for variable in ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"):
            monkeypatch.delenv(variable, raising = False)

        assert sd_cpp_backend.selected_card_identity(0) == sd_cpp_backend._card_identity(devices[1])
        assert sd_cpp_backend._physical_position_of(0) == ("AMD Radeon RX 7900 XTX", 0)

    def test_windows_ignores_a_stale_rocr_mask(self, monkeypatch):
        """Windows HIP has no ROCr layer, so ROCR_VISIBLE_DEVICES masks nothing there. Reading it
        would turn a leftover ``ROCR_VISIBLE_DEVICES=1`` into "ordinal 0 is card 1"."""
        from core.inference import sd_cpp_backend

        monkeypatch.setattr(sd_cpp_backend.sys, "platform", "win32")
        env = {"ROCR_VISIBLE_DEVICES": "1"}

        # No mask in force on Windows, so the ordinal passes through and nothing is masked.
        assert sd_cpp_backend._physical_index_of(0, env = env) == (0, False)

    def test_linux_still_reads_the_rocr_mask(self, monkeypatch):
        """The same variable on Linux is a real mask, and the fix must not disarm it there."""
        from core.inference import sd_cpp_backend

        monkeypatch.setattr(sd_cpp_backend.sys, "platform", "linux")
        env = {"ROCR_VISIBLE_DEVICES": "1"}

        assert sd_cpp_backend._physical_index_of(0, env = env) == (1, True)

    def test_windows_still_reads_the_hip_mask(self, monkeypatch):
        """HIP_VISIBLE_DEVICES IS honoured by Windows HIP; only the ROCr variable is absent there."""
        from core.inference import sd_cpp_backend

        monkeypatch.setattr(sd_cpp_backend.sys, "platform", "win32")
        env = {"HIP_VISIBLE_DEVICES": "2,3", "ROCR_VISIBLE_DEVICES": "1"}

        # Composing the stale ROCR mask in would have made this 3 rather than 2.
        assert sd_cpp_backend._physical_index_of(0, env = env) == (2, True)


def _recorded_proven(store: dict, klass: str = "rocm") -> bool:
    record = (store.get("sd_cpp_accelerator_runtime_failures") or {}).get(klass) or {}
    return bool(record.get("proven", False))


class TestACpuOnlyAnswerIsOnlyProofWhenItIsExplained:
    """The ROCm build answering ``--list-devices`` with CPU only is the Linux failure shape: exit 0, no
    GPU enumerated. A busy or masked card produces the same text, so on its own it is one strike, not
    proof. It becomes proof when the HIP/BLAS runtime the prebuilt needs cannot be loaded at all."""

    def _run(self, h3_amd_host, monkeypatch, *, resolvable):
        from core.inference import sd_cpp_backend

        monkeypatch.setattr(
            sd_cpp_backend, "rocm_runtime_resolvable", lambda: resolvable, raising = False
        )
        host = h3_amd_host(platform = "linux", backend = "rocm", device = "cuda", devices = _ROCM_CPU_ONLY)
        host.run()

    def test_a_loadable_runtime_makes_it_ambiguous(self, h3_amd_host, fake_settings, monkeypatch):
        """The runtime IS there, so CPU-only says nothing conclusive: one strike, two needed to divert."""
        self._run(h3_amd_host, monkeypatch, resolvable = True)
        assert _recorded_proven(fake_settings) is False
        assert _recorded_strikes(fake_settings) == 1

    def test_an_unloadable_runtime_makes_it_decisive(self, h3_amd_host, fake_settings, monkeypatch):
        """No hipblas / rocblas / amdhip64 on this host explains the CPU-only answer, so one is enough."""
        self._run(h3_amd_host, monkeypatch, resolvable = False)
        assert _recorded_proven(fake_settings) is True


class TestTheCommittedCardIsPublishedAtTheCommit:
    """``_committed_loading_card`` is what a caller OFF the load thread reads. It must name the load
    that took, not the one that started, or a one-shot generation resolves the other card's build."""

    def test_a_load_in_flight_does_not_publish_its_card(self):
        from core.inference import sd_cpp_backend

        backend_obj = sd_cpp_backend.SdCppDiffusionBackend.__new__(
            sd_cpp_backend.SdCppDiffusionBackend
        )
        backend_obj._loading_card_store = lambda: types.SimpleNamespace()
        backend_obj._committed_loading_card = "Card A@gfx1100"

        # A worker starting on card B writes only its own thread-local view.
        backend_obj._loading_card = "Card B@gfx1201"
        assert backend_obj._committed_loading_card == "Card A@gfx1100"

    def test_the_setter_is_thread_local_only(self):
        """Pinned on the source: the publish belongs beside the _state commit, past the supersession
        check, not in the setter, which a superseded worker can still reach late."""
        import inspect

        from core.inference import sd_cpp_backend

        setter = inspect.getsource(sd_cpp_backend.SdCppDiffusionBackend._loading_card.fset)
        code = "\n".join(line for line in setter.splitlines() if not line.strip().startswith("#"))
        assert "_committed_loading_card" not in code, code

        source = inspect.getsource(sd_cpp_backend)
        commit = source.index("self._state = state\n")
        assert "self._committed_loading_card = self._loading_card" in source[commit : commit + 500]


def test_an_ensure_that_hands_back_the_failed_build_is_not_a_working_fallback(
    h3_amd_host, fake_settings, monkeypatch
):
    """When the Vulkan rung cannot be installed, ensure deliberately keeps the usable build it already
    has, which is the ROCm one that just probed negative. If the second probe of that SAME executable
    answers yes (the first was transient: a busy card, a masked card), accepting it would record ROCm
    as failed and pin a preference to a rung that was never installed. The verdict has to be attributed
    to the class that actually produced it."""
    from core.inference import sd_cpp_backend

    # No vulkan build to install; the ensure falls back to handing out the rocm one it already has.
    devices = {"rocm": _DEVICES_ROCM, "cpu": _DEVICES_CPU_ONLY}
    host = h3_amd_host(platform = "linux", backend = "rocm", device = "cuda", devices = devices)

    real_ensure = sd_cpp_backend.ensure_sd_cpp_binary

    def _ensure(*, allow_install = True, accelerator = "cpu"):
        if accelerator == "vulkan":
            return "/opt/sd/rocm/sd-cli"  # the failed build, handed back as "the best we have"
        return real_ensure(allow_install = allow_install, accelerator = accelerator)

    monkeypatch.setattr(sd_cpp_backend, "ensure_sd_cpp_binary", _ensure)
    # First probe negative, every later one positive: the transient case this guards.
    calls = {"n": 0}

    def _verdict(binary, *a, **k):
        calls["n"] += 1
        return calls["n"] > 1

    # video.py imports this lazily from sd_cpp_backend inside the load, so the module attribute is
    # what it resolves at call time.
    monkeypatch.setattr(sd_cpp_backend, "sd_cpp_accelerator_device_verdict", _verdict)

    # The tree now holds a build that is not the accelerator this load decided on, which the existing
    # identity check refuses rather than running. That is the right end for a transient cause: the
    # message asks for a retry, and the retry probes positive and loads ROCm normally.
    with pytest.raises(RuntimeError, match = "different accelerator"):
        host.run()

    # The point of the test: NOTHING is persisted. Before the class check, this same run recorded
    # rocm with proven=True off one transient probe of the very binary that was handed back.
    assert _noted_accelerators(fake_settings) == [], fake_settings


class TestTheRouterRecordsTheBundleNotTheServer:
    """sd-server and sd-cli ship in the same bundle. Only the server failing to launch says nothing
    about the accelerator, and a strike for it on every otherwise successful one-shot load reaches the
    two-strike threshold and diverts a working ROCm host to Vulkan for good.

    These assert the ORDER of the three statements in the selection block rather than driving a load:
    the behaviour is "the record happens after the CLI verdict, not before", and the surrounding
    function activates a global engine, which a unit test should not be doing to reach one branch.
    """

    @staticmethod
    def _selection_source():
        from core.inference import diffusion_engine_router as router
        return inspect.getsource(router.select_and_activate_engine)

    def test_both_verdicts_are_held_until_the_other_has_answered(self):
        source = self._selection_source()
        held_server = source.index("unlaunchable_server = server_binary")
        cli_probe = source.index("SdCppEngine(binary = binary).version() is None")
        held_cli = source.index("unlaunchable_cli = binary")
        recorded = source.index("note_unlaunchable_accelerator_build(")
        assert held_server < cli_probe < held_cli < recorded, (
            held_server,
            cli_probe,
            held_cli,
            recorded,
        )

    def test_a_bundle_where_nothing_runs_is_still_recorded(self):
        source = self._selection_source()
        # The dead-bundle case must still reach the recorder, not be dropped along with the deferral.
        assert (
            "if binary is None and server_binary is None and (unlaunchable_cli or unlaunchable_server):"
            in source
        )


def test_the_download_plan_predicts_for_the_card_the_load_will_select():
    """Card-scoped records mean a card-less prediction reads a per-card failure as host-wide. Selection
    is given the ordinal and clears the working card, so the two disagree and the plan stages the
    diffusers files a native load never opens."""
    from core.inference import diffusion_engine_router as router

    assert "gpu_ordinal" in inspect.signature(router.predict_engine).parameters
    assert "gpu_ordinal" in inspect.signature(router.native_binary_installed).parameters
    # The predictor must scope the record lookup exactly the way selection does.
    source = inspect.getsource(router.native_binary_installed)
    assert "_selected_card(gpu_ordinal)" in source
    assert source.count("selected_card") >= 3  # accelerator + both usable_or_recorded_failure calls


def test_both_routes_predict_with_the_ordinal_they_already_resolved():
    """The records are per card, so a host-wide prediction reads one card's failure as every card's.
    Both routes resolve an ordinal for other reasons already, so the fix costs no second resolution --
    and must not add one: ranking reads free VRAM per candidate and opens a CUDA context on each,
    which the download plan defers until training is known idle."""
    import inspect

    from routes import inference as routes

    source = inspect.getsource(routes)
    # Every prediction is card-scoped.
    assert "predict_engine(fam, model_kind = kind)" not in source
    # And the download plan resolves exactly once, after the training state is known.
    plan = inspect.getsource(routes.diffusion_download_plan)
    assert plan.count("_selected_gpu_ordinal(") == 1, plan.count("_selected_gpu_ordinal(")
    training = plan.index("training = fam is not None")
    resolved = plan.index("_selected_gpu_ordinal(")
    predicted = plan.index("predict_engine(")
    assert training < resolved < predicted, (training, resolved, predicted)


def test_an_unmasked_ordinal_is_still_a_hip_id_not_an_inventory_row(monkeypatch):
    """Unmasked does not make the ordinal the inventory position. HIP orders by node id and amd-smi
    by discovery row, so counting same-name cards in torch order names the OTHER card of a matched
    pair and pins Vulkan to a GPU the request never selected."""
    from core.inference import video as video_mod

    _no_visibility_mask(monkeypatch)
    _pinned_torch(monkeypatch, ["AMD Radeon RX 7900 XTX", "AMD Radeon RX 7900 XTX"])
    _pinned_inventory(
        monkeypatch,
        [
            {"vendor": "amd", "index": 0, "name": "AMD Radeon RX 7900 XTX"},
            {"vendor": "amd", "index": 1, "name": "AMD Radeon RX 7900 XTX"},
        ],
        # The two spaces disagree: HIP id 0 is inventory row 1 and vice versa.
        hip_by_row = {0: 1, 1: 0},
    )

    assert video_mod._physical_card_name(0) == ("AMD Radeon RX 7900 XTX", 1)
    assert video_mod._physical_card_name(1) == ("AMD Radeon RX 7900 XTX", 0)


def test_an_unmasked_ordinal_still_counts_when_no_mapping_exists(monkeypatch):
    """amd-smi missing, or older than the ROCm 6.4 that added ``list -e``: the torch-order count is
    the only answer there is, and it stays the answer."""
    from core.inference import video as video_mod
    from utils.hardware import amd, hardware

    _no_visibility_mask(monkeypatch)
    _pinned_torch(monkeypatch, ["AMD Radeon RX 7900 XTX", "AMD Radeon RX 7900 XTX"])
    monkeypatch.setattr(hardware, "get_physical_gpu_inventory", lambda *, block = True: None)
    monkeypatch.setattr(amd, "get_hip_id_by_gpu_index", lambda: {})

    assert video_mod._physical_card_name(1) == ("AMD Radeon RX 7900 XTX", 1)


def test_the_loading_card_does_not_outlive_the_load_on_a_pooled_thread(monkeypatch):
    """The card is the LOAD's, not the worker's. Load threads are pooled, so a thread-local left set
    makes a later off-load resolution on the same worker -- one-shot generation re-resolving sd-cli
    -- answer with a finished load's card instead of the committed one, and pick the fallback for a
    card that never failed."""
    from core.inference import sd_cpp_backend

    backend = sd_cpp_backend.SdCppDiffusionBackend.__new__(sd_cpp_backend.SdCppDiffusionBackend)
    backend._committed_loading_card = "AMD Radeon RX 7900 XTX@0"

    # Off any load: the committed card is the answer.
    assert backend._loading_card == "AMD Radeon RX 7900 XTX@0"

    # During a load on this thread the load's own selection wins, including an explicit None.
    backend._loading_card = "AMD Radeon RX 7600@1"
    assert backend._loading_card == "AMD Radeon RX 7600@1"
    backend._loading_card = None
    assert backend._loading_card is None

    # And once the load is over the worker has no own answer again.
    backend._clear_loading_card()
    assert backend._loading_card == "AMD Radeon RX 7900 XTX@0"


def test_constructing_the_backend_does_not_claim_a_card_for_its_thread():
    """``__init__`` runs on whichever executor thread built the backend. An explicit None written
    there would be that thread's own answer for the life of the process, so a generate later
    scheduled onto the same worker would read None rather than the committed card."""
    import inspect

    from core.inference import sd_cpp_backend

    source = inspect.getsource(sd_cpp_backend.SdCppDiffusionBackend.__init__)
    assert "self._loading_card =" not in source

    # And the load clears it on the way out, on every path.
    run_load = inspect.getsource(sd_cpp_backend.SdCppDiffusionBackend._run_load)
    finally_block = run_load[run_load.rindex("finally:") :]
    assert "_clear_loading_card()" in finally_block


def test_the_loading_card_store_exists_before_any_load():
    """Created lazily, two overlapping first loads could each build a store, and the one that lost
    the assignment dropped the card it wrote. Built with the backend, every worker shares it."""
    from core.inference import sd_cpp_backend

    backend = sd_cpp_backend.SdCppDiffusionBackend()
    assert isinstance(backend.__dict__.get("_loading_cards"), threading.local)
    assert not hasattr(backend._loading_cards, "card")


@pytest.mark.parametrize(
    "returncode, runnable",
    [
        (0xC0000135, False),
        (0xC0000139, False),
        (0xC0000142, False),
        (127, False),
        (0, True),
        (1, True),
    ],
)
def test_the_server_probe_refuses_a_windows_loader_death(monkeypatch, returncode, runnable):
    """A ROCm sd-server with no HIP DLLs exits 0xC0000135 with no output. Read as runnable, the
    router never recorded it, and the load's start failure had nothing to classify, so every
    forced-native image load retried the same build instead of ever reaching Vulkan."""
    import subprocess

    from core.inference import sd_cpp_backend

    monkeypatch.setattr(
        subprocess, "run", lambda *_a, **_k: types.SimpleNamespace(returncode = returncode)
    )
    assert sd_cpp_backend._server_binary_runnable("/opt/sd/rocm/sd-server") is runnable


def test_a_damaged_cli_beside_a_healthy_server_is_not_a_strike(fake_settings, monkeypatch):
    """The mirror of the held sd-server verdict. sd-cli losing its execute bit while the server runs
    says nothing about the accelerator: the server is what this load uses and the load succeeds. A
    strike there is a strike on a working host, and two of them divert a healthy ROCm bundle to
    Vulkan for good."""
    from core.inference import diffusion_engine_router as router
    from core.inference import sd_cpp_backend

    monkeypatch.setattr(router, "_install_allowed", lambda: True)
    monkeypatch.setattr(router, "ensure_sd_server_binary", lambda **_k: "/opt/sd/rocm/sd-server")
    monkeypatch.setattr(router, "ensure_sd_cpp_binary", lambda **_k: "/opt/sd/rocm/sd-cli")
    monkeypatch.setattr(router, "_server_binary_runnable", lambda _b: True)  # server is fine
    monkeypatch.setattr(
        router, "SdCppEngine", lambda binary: types.SimpleNamespace(version = lambda: None)
    )  # cli is not
    monkeypatch.setattr(
        router,
        "resolve_diffusion_device_target",
        lambda: types.SimpleNamespace(backend = "rocm", device = "cuda", dtype = None),
    )
    monkeypatch.setattr(router, "family_sd_cpp_supported", lambda _fam: True)
    monkeypatch.setattr(router, "_activate", lambda name, reason = None: name)
    monkeypatch.setattr(sd_cpp_backend, "_installed_accelerator_of", lambda _b: "rocm")
    monkeypatch.setenv("UNSLOTH_DIFFUSION_ENGINE", "sd_cpp")

    chosen = router.select_and_activate_engine(_detect_load_family(H3_REPO, None, "minimax-h3"))
    assert chosen == "sd_cpp", chosen  # the healthy server carries the load
    assert _recorded_strikes(fake_settings) == 0  # and nothing was held against ROCm
