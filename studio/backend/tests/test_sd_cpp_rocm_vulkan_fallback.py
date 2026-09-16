# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The ROCm -> Vulkan rung for the native stable-diffusion.cpp bundle (#9278, #8814).

A ROCm sd.cpp prebuilt is published for one generic ROCm target rather than per gfx arch, so a card
whose hipBLAS kernels that build does not carry cannot run it: #9278 (RX 9070 XT, gfx1201, Windows)
dies with ``CUBLAS_STATUS_INVALID_VALUE at hipblasSetStream`` right after tensor loading, and #8814
(RX 7900 XTX, gfx1100, Linux) never gets further than ``stable-diffusion.cpp could not be installed
or started for MiniMax-H3``. Both cards run Vulkan, and the upstream release carries a
``-vulkan`` asset for Linux and for Windows alongside the ROCm one, which the installer already
resolves (``_LINUX_ACCEL_TOKEN`` / ``_WINDOWS_ACCEL_TOKEN``).

Parametrised over ``platform x accelerator state`` because both halves decide the outcome: the
rung must be taken on every platform that has a Vulkan asset, must NOT be taken on a host whose
problem is something else (a CUDA host with only the CPU prebuilt has always fallen to CPU and must
keep doing exactly that), and must leave the CPU fallback reachable when Vulkan cannot run either.

Everything here is faked: no GPU, no network, no sd-cli. The binaries are strings and the probes
answer canned ``--help`` / ``--list-devices`` text, which is how the rest of the sd.cpp suite
(``test_sd_cpp_h3_matrix.py``) drives this code path.
"""

from __future__ import annotations

import threading
import types
from pathlib import Path

import pytest

from core.inference.video import VideoBackend, _detect_load_family

H3_REPO = "leejet/MiniMax-H3-GGUF"
H3_FILE = "minimax_h3_fl2va-Q4_K_M.gguf"

_BANNER = "stable-diffusion.cpp version unknown, commit unknown\n"
_H3_HELP = _BANNER + "  --ref-video   MiniMax-H3 Ref2VA reference video frame directory\n"

# sd-cli --list-devices output per build. The ROCm build on an unsupported card is the interesting
# one: it does not answer at all, which is what an sd-cli that dies during backend init looks like
# from here, and is why the load has to read the raw verdict rather than the collapsed one.
_DEVICES_ROCM = "ROCm0\tAMD Radeon RX 7900 XTX\nCPU\tAMD Ryzen 9\n"
_DEVICES_VULKAN = "Vulkan0\tAMD Radeon RX 7900 XTX\nCPU\tAMD Ryzen 9\n"
_DEVICES_CPU_ONLY = "CPU\tAMD Ryzen 9\n"
# Three states per accelerator, and they are not the same thing: MISSING is "no such build can be
# installed on this host" (the ensure returns nothing), None is "the build is installed and cannot
# be asked anything", which is what an sd-cli that dies during backend init looks like from here,
# and a string is what it answers --list-devices with.
MISSING = object()

# sys.platform values that have both a ROCm and a Vulkan sd.cpp asset. WSL is a Linux platform
# string, listed separately because it is a distinct host shape in every other sd.cpp test.
PLATFORMS = ["linux", "wsl", "win32"]

# Every host shape Studio ships on, including the one with no ROCm story at all. macOS is listed
# because the point of the matrix below is the corners that must NOT change, and a platform whose
# release carries a single Metal asset (install_sd_cpp_prebuilt.resolve_release_asset returns the
# darwin zip whatever accelerator is asked for) is the strongest of those.
ALL_PLATFORMS = ["linux", "wsl", "win32", "darwin"]

# GPU vendor -> the device backend resolve_diffusion_device_target settles on, per platform. On
# macOS the vendor axis genuinely collapses: there is no CUDA and no ROCm there, the GPU backend is
# Metal, so both GPU cells are "mps" and that is the fact the matrix should state rather than skip.
_BACKEND_FOR = {
    "darwin": {"nvidia": "mps", "amd": "mps", "cpu_only": "cpu"},
    "other": {"nvidia": "cuda", "amd": "rocm", "cpu_only": "cpu"},
}
_DEVICE_FOR = {"mps": "mps", "cpu": "cpu", "cuda": "cuda", "rocm": "cuda"}


def _corner(platform: str, vendor: str) -> tuple[str, str]:
    """(backend, device) for one cell of the platform x vendor matrix."""
    backend = _BACKEND_FOR["darwin" if platform == "darwin" else "other"][vendor]
    return backend, _DEVICE_FOR[backend]


# The accelerator a corner asks the installer for first: diffusion_engine_router._INSTALL_ACCELERATOR
# with its "auto" default, and accelerator_class folding "auto" onto the plain build it names "cpu".
_FIRST_ENSURE = {"cuda": "cuda", "rocm": "rocm", "mps": "cpu", "cpu": "cpu"}

_DEVICES_CUDA = "CUDA0\tNVIDIA GeForce RTX 4090\nCPU\tIntel Core i9\n"


def _devices_for(backend: str, *, rocm_runs: bool) -> dict:
    """The --list-devices answer of each per-accelerator build present on a host of this shape. A
    Vulkan build is offered in EVERY corner on purpose: the fix must be shown to leave it untaken
    everywhere except the one corner it is for."""
    if backend == "cuda":
        return {"cuda": _DEVICES_CUDA, "vulkan": _DEVICES_VULKAN, "cpu": _DEVICES_CPU_ONLY}
    if backend == "rocm":
        return dict(_ROCM_WORKS) if rocm_runs else dict(_ROCM_BROKEN)
    # cpu and mps both install the plain build; macOS has no vulkan asset at all, but offering one
    # here only makes the "never taken" assertion stronger.
    return {"cpu": _DEVICES_CPU_ONLY, "vulkan": _DEVICES_VULKAN}


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
    """An in-memory app-settings store, so the persisted preference is exercised for real without
    touching a database. Returned so a test can read what was written."""
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
    """The in-process half of the note is module state; a leak would make these order dependent.

    ``raising = False`` so the behavioural tests below still report what they measured, rather than
    every one of them erroring in setup, on a tree that does not carry the note at all."""
    from core.inference import sd_cpp_backend

    monkeypatch.setattr(
        sd_cpp_backend, "_accelerator_runtime_failures", set(), raising = False
    )
    monkeypatch.delenv("UNSLOTH_DIFFUSION_SD_CPP_VULKAN_FALLBACK", raising = False)


@pytest.fixture
def h3_amd_host(monkeypatch, tmp_path):
    """Run `_run_load_h3_native` against a host whose per-accelerator builds are described by
    ``devices``: a mapping of accelerator class -> ``--list-devices`` text, or None for "no such
    build can be installed here". Records every accelerator the load asked to ensure."""
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
        # binary path -> the accelerator it was installed for, so the probes and the "which build is
        # committed" assertions agree without the test having to hard-code paths.
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


# (label, what each accelerator's build answers --list-devices with)
_ROCM_WORKS = {"rocm": _DEVICES_ROCM, "vulkan": _DEVICES_VULKAN, "cpu": _DEVICES_CPU_ONLY}
# #8814 / #9278: the ROCm build is installed and simply cannot be asked anything.
_ROCM_BROKEN = {"rocm": None, "vulkan": _DEVICES_VULKAN, "cpu": _DEVICES_CPU_ONLY}
# The same host where the ROCm build starts but enumerates no accelerator of its own.
_ROCM_CPU_ONLY = {"rocm": _DEVICES_CPU_ONLY, "vulkan": _DEVICES_VULKAN, "cpu": _DEVICES_CPU_ONLY}
# No ROCm asset for this host at all, which is what a release that ships only Vulkan looks like.
_VULKAN_ONLY = {"rocm": MISSING, "vulkan": _DEVICES_VULKAN, "cpu": _DEVICES_CPU_ONLY}
# Nothing but the CPU build runs here; the load must still commit, on the CPU, as it always has.
_CPU_ONLY = {"rocm": MISSING, "vulkan": MISSING, "cpu": _DEVICES_CPU_ONLY}


@pytest.mark.parametrize("platform", PLATFORMS)
def test_a_working_rocm_build_is_left_alone(h3_amd_host, fake_settings, platform):
    """The negative control, and the one that matters most: a host where ROCm works must not be
    moved to Vulkan, must not install it, and must not record anything."""
    host = h3_amd_host(
        platform = platform, backend = "rocm", device = "cuda", devices = _ROCM_WORKS
    )
    backend_obj = host.run()
    assert host.ensured == ["rocm"]
    assert backend_obj._state.device == "cuda"
    assert fake_settings == {}


@pytest.mark.parametrize("platform", PLATFORMS)
@pytest.mark.parametrize(
    "state,devices",
    [("rocm_unrunnable", _ROCM_BROKEN), ("rocm_cpu_only", _ROCM_CPU_ONLY), ("no_rocm_asset", _VULKAN_ONLY)],
)
def test_a_rocm_build_that_cannot_run_falls_back_to_vulkan(
    h3_amd_host, fake_settings, platform, state, devices
):
    """The fix. On main every one of these commits ``native_device = "cpu"`` (or refuses), because
    the only rung below ROCm was the CPU build. The Vulkan build runs on both reporters' cards."""
    host = h3_amd_host(platform = platform, backend = "rocm", device = "cuda", devices = devices)
    backend_obj = host.run()
    assert host.ensured == ["rocm", "vulkan"], host.ensured
    # Committed on the GPU, on the Vulkan build, with the bundle fetched exactly once.
    assert backend_obj._state.device == "cuda"
    assert len(host.downloads) == 4
    # And remembered, so the next load does not pay for the broken ROCm build again.
    assert fake_settings["sd_cpp_accelerator_runtime_failures"] == ["rocm"]


@pytest.mark.parametrize("platform", PLATFORMS)
def test_the_cpu_rung_is_still_reached_when_vulkan_cannot_run_either(
    h3_amd_host, fake_settings, platform
):
    """The rung is added, not substituted: a host with no working GPU build ends up exactly where
    it does on main, on the CPU build, and records no preference it could not act on."""
    host = h3_amd_host(platform = platform, backend = "rocm", device = "cuda", devices = _CPU_ONLY)
    backend_obj = host.run()
    assert host.ensured == ["rocm", "vulkan", "cpu"], host.ensured
    assert backend_obj._state.device == "cpu"
    assert fake_settings == {}


@pytest.mark.parametrize("platform", PLATFORMS)
def test_a_cuda_host_never_takes_the_vulkan_rung(h3_amd_host, fake_settings, platform):
    """A CUDA host with only the CPU prebuilt is the case the CPU fallback was written for, and it
    has nothing to do with per-arch ROCm builds. It must reach the CPU rung directly."""
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
    """An operator who would rather see the ROCm failure than be moved to Vulkan keeps that."""
    monkeypatch.setenv("UNSLOTH_DIFFUSION_SD_CPP_VULKAN_FALLBACK", "0")
    host = h3_amd_host(
        platform = platform, backend = "rocm", device = "cuda", devices = _ROCM_BROKEN
    )
    backend_obj = host.run()
    assert "vulkan" not in host.ensured
    assert backend_obj._state.device == "cuda"  # unchanged from main: the unreadable probe keeps it
    assert fake_settings == {}


@pytest.mark.parametrize("platform", PLATFORMS)
def test_a_recorded_failure_skips_the_rocm_build_on_the_next_load(
    h3_amd_host, fake_settings, platform
):
    """The persisted half. Once this host has been shown the ROCm build does not run, the NEXT
    load must not install and probe it again: it asks for Vulkan first."""
    fake_settings["sd_cpp_accelerator_runtime_failures"] = ["rocm"]
    host = h3_amd_host(
        platform = platform, backend = "rocm", device = "cuda", devices = _ROCM_WORKS
    )
    backend_obj = host.run()
    assert host.ensured == ["vulkan"], host.ensured
    assert backend_obj._state.device == "cuda"


def test_clearing_the_note_restores_the_host_accelerator(fake_settings):
    """A driver update or a new card is a new host, so the note has to be clearable."""
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
    """One-way and one-deep: nothing falls back FROM Vulkan, and CUDA either works or the host has
    no CUDA asset at all, which the CPU rung has always covered."""
    from core.inference.sd_cpp_backend import fallback_accelerator_for

    assert fallback_accelerator_for(accelerator) == expected


@pytest.mark.parametrize(
    "output,expected",
    [
        # #9278 verbatim, both shapes the reporter saw.
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
    """A render fails for all sorts of reasons. Only the ones that name the GPU build may persist a
    preference away from this host's own accelerator."""
    from core.inference.sd_cpp_backend import output_shows_accelerator_failure

    assert output_shows_accelerator_failure(output) is expected


def test_a_generation_that_dies_in_hipblas_records_the_failure(fake_settings, monkeypatch):
    """The other half of #9278: the ROCm build starts, loads the tensors and dies minutes into the
    render. Nothing about the install is wrong, so without this the next load picks it again."""
    from core.inference import sd_cpp_backend
    from core.inference import video as video_mod

    monkeypatch.setattr(sd_cpp_backend, "_installed_accelerator_of", lambda _b: "rocm")
    video_mod._note_sd_cpp_accelerator_failure(
        "/opt/sd/rocm/sd-cli",
        "sd-cli exited 1. Last output:\nROCm error: CUBLAS_STATUS_INVALID_VALUE at hipblasSetStream",
    )
    assert fake_settings["sd_cpp_accelerator_runtime_failures"] == ["rocm"]
    assert sd_cpp_backend.preferred_accelerator("rocm") == "vulkan"


def test_an_ordinary_generation_failure_records_nothing(fake_settings, monkeypatch):
    """The guard on the above. An out-of-memory or a bad argument must not move a working host."""
    from core.inference import sd_cpp_backend
    from core.inference import video as video_mod

    monkeypatch.setattr(sd_cpp_backend, "_installed_accelerator_of", lambda _b: "rocm")
    video_mod._note_sd_cpp_accelerator_failure(
        "/opt/sd/rocm/sd-cli", "sd-cli exited 1. Last output:\nggml_new_object: not enough space"
    )
    assert fake_settings == {}
    assert sd_cpp_backend.preferred_accelerator("rocm") == "rocm"


def test_the_engine_router_installs_the_preferred_accelerator(fake_settings, monkeypatch):
    """The image path ensures the same bundle, so a host that cannot run the ROCm build must not be
    handed it there either."""
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
    # prefer_native, so the ensure pair runs on a GPU host at all.
    monkeypatch.setenv("UNSLOTH_DIFFUSION_ENGINE", "sd_cpp")
    router.select_and_activate_engine(_detect_load_family(H3_REPO, None, "minimax-h3"))
    assert asked == ["vulkan", "vulkan"], asked


# ---------------------------------------------------------------------------
# The platform x vendor matrix.
#
# The fix touches the accelerator every Studio host installs sd.cpp for, so the thing that has to be
# demonstrated is not only that the broken corner is fixed but that the other eleven are untouched.
# Both halves are driven from the same fixture and the same table, so a corner cannot be quietly
# dropped from one of them.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("platform", ALL_PLATFORMS)
@pytest.mark.parametrize("vendor", ["nvidia", "amd", "cpu_only"])
def test_every_healthy_corner_installs_exactly_what_it_did_before(
    h3_amd_host, fake_settings, platform, vendor
):
    """A host whose own accelerator works is never moved, on any platform, for any vendor: one
    ensure, for its own accelerator, and nothing recorded. A Vulkan build is available in every
    corner, so "not taken" here is a real choice rather than an absence."""
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
    """The same matrix with the ROCm build unable to start. Exactly one column of one row shape
    changes: an AMD host on a platform that has a ROCm asset. macOS resolves its GPU to Metal and
    never asks for ROCm, so its cells are unchanged even in the AMD column, and a CUDA or CPU host
    is untouched because ROCm is not what it asked for."""
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
        assert fake_settings["sd_cpp_accelerator_runtime_failures"] == ["rocm"]
    else:
        assert host.ensured == [_FIRST_ENSURE[backend]], host.ensured
        assert fake_settings == {}
    # Every corner still commits on the device it committed on before: the GPU for a GPU host (the
    # AMD one now on the Vulkan build rather than on nothing), the CPU for a CPU host.
    assert backend_obj._state.device == device


@pytest.mark.parametrize("platform", ALL_PLATFORMS)
def test_the_release_assets_each_platform_resolves_are_unchanged(platform):
    """The other end of the matrix: the fallback is only worth taking where the installer can
    actually resolve a Vulkan asset, and it must not have changed what any accelerator resolves.

    macOS is the reason this is asserted rather than assumed: its release carries one Metal zip and
    ``resolve_release_asset`` ignores the accelerator there, so asking it for Vulkan hands back the
    same Metal build, which is why the fallback must never be reached on that platform at all.
    """
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
        # One asset, whatever is asked for. So there is no ROCm build to fail and no Vulkan build
        # to fall back to: the matrix above is right to expect macOS never to move.
        assert resolved("auto") == "sd-master-bin-macos-arm64.zip"
        assert resolved("rocm") == "sd-master-bin-macos-arm64.zip"
        assert resolved("vulkan") == "sd-master-bin-macos-arm64.zip"
        return
    # Everywhere else both rungs exist, which is what makes the fallback reachable.
    assert resolved("rocm") is not None
    assert resolved("vulkan") is not None
    assert resolved("rocm") != resolved("vulkan")
    if platform == "win32":
        assert resolved("cuda") == "sd-master-bin-win-cuda12-x64.zip"
        assert resolved("auto") == "sd-master-bin-win-avx2-x64.zip"
    else:
        # Upstream publishes no Linux CUDA archive, which is the pre-existing reason the CPU rung
        # exists at all. Unchanged here.
        assert resolved("cuda") is None
        assert resolved("auto") == "sd-master-linux-x64.zip"


@pytest.mark.parametrize(
    "backend,noted,expected",
    [
        # The corner the fix is for: the image path must not hand back the build this host has
        # already been shown cannot start.
        ("rocm", ["rocm"], "vulkan"),
        # And every other corner is the accelerator the host asked for, unchanged.
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
    """``SdCppDiffusionBackend._resolved_accelerator`` is the single funnel for the image path's three
    ensures (``_resolve_engine``, ``_resolve_backend``, the deferred upgrade and the mid-load re-resolve), so the preference
    has to be applied in it rather than at any one of them: ``_accelerator_changed`` compares the
    installed tree against this answer, and a call site that disagreed would reinstall over the
    tree the others just wrote, on every load."""
    from core.inference import sd_cpp_backend

    fake_settings["sd_cpp_accelerator_runtime_failures"] = list(noted)
    monkeypatch.setattr(
        sd_cpp_backend,
        "resolve_diffusion_device_target",
        lambda: types.SimpleNamespace(backend = backend, device = "cuda", dtype = None),
    )
    assert sd_cpp_backend.SdCppDiffusionBackend._resolved_accelerator() == expected
