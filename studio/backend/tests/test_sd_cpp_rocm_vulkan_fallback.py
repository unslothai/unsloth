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


def _noted_accelerators(store: dict) -> list:
    """The accelerators the persisted record currently DIVERTS, in order.

    The stored value is a map of accelerator -> {strikes, proven, fingerprint} rather than a bare
    list of names, because a name outlives the thing it is a fact about. Tests assert through this
    so they describe the behaviour ("rocm is diverted") rather than the storage layout.
    """
    records = store.get("sd_cpp_accelerator_runtime_failures") or {}
    return sorted(
        k
        for k, v in records.items()
        if isinstance(v, dict) and (v.get("proven") or v.get("strikes", 0) >= 2)
    )


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

    monkeypatch.setattr(sd_cpp_backend, "_accelerator_runtime_failures", {}, raising = False)
    # The host half of the fingerprint is memoised per process. Pin it to one known value so these
    # tests neither read this machine's real GPUs nor leak a memo into each other; the fingerprint
    # tests below override it deliberately.
    monkeypatch.setattr(
        sd_cpp_backend,
        "_HOST_FINGERPRINT_MEMO",
        {"runtime": "6.4.0", "gpus": ["AMD Radeon RX 7900 XTX"]},
        raising = False,
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
    host = h3_amd_host(platform = platform, backend = "rocm", device = "cuda", devices = _ROCM_WORKS)
    backend_obj = host.run()
    assert host.ensured == ["rocm"]
    assert backend_obj._state.device == "cuda"
    assert fake_settings == {}


@pytest.mark.parametrize("platform", PLATFORMS)
@pytest.mark.parametrize(
    "state,devices",
    [
        ("rocm_unrunnable", _ROCM_BROKEN),
        ("rocm_cpu_only", _ROCM_CPU_ONLY),
        ("no_rocm_asset", _VULKAN_ONLY),
    ],
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
    assert _noted_accelerators(fake_settings) == ["rocm"]


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
    host = h3_amd_host(platform = platform, backend = "rocm", device = "cuda", devices = _ROCM_BROKEN)
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
    host = h3_amd_host(platform = platform, backend = "rocm", device = "cuda", devices = _ROCM_WORKS)
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
    assert _noted_accelerators(fake_settings) == ["rocm"]
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
        assert _noted_accelerators(fake_settings) == ["rocm"]
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


# ---------------------------------------------------------------------------
# Release safety: the persistent preference has to be invalidated by the things it is a fact about,
# must not be set by one transient error, and must have a way back that is not "edit the database".
# ---------------------------------------------------------------------------


def test_a_single_ambiguous_failure_never_diverts_a_working_rocm_host(fake_settings, monkeypatch):
    """The core of the P2 finding. "hip error", "rocm error" and "unspecified launch failure" all
    come out of a wedged queue, a driver reset or a card another process is mistreating, none of
    which proves this BUILD cannot serve this card. One of them must leave a working host alone."""
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
    """The other side of it: a host producing these repeatedly under one fingerprint is not having
    bad luck, and #9278 with offload enabled reports exactly this string."""
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
        # The same defect class as the decisive list, printed by a layer that does not name the
        # build: rocBLAS finding no Tensile kernels for this gfx target, and the HSA runtime abort
        # the same situation ends in once it reaches a kernel launch. Both have mundane causes too
        # (/dev/kfd missing inside a container; flash attention or multi-GPU P2P on a host whose
        # ROCm is otherwise fine), so both are counted rather than acted on.
        ("rocBLAS error: Could not initialize Tensile host: No devices found", False),
        (
            "Memory access fault by GPU node-1 (Agent handle: 0x55d) on address 0x7f18. "
            "Reason: Page not present or supervisor privilege.",
            False,
        ),
    ],
)
def test_the_marker_tiers_split_evidence_from_suspicion(output, decisive):
    """Only a message that names the BUILD having no code for the card is acted on immediately.
    Every marker is still recognised as accelerator-related; the tier decides what it costs."""
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
    """The gap this closes. These are the two shapes a generic ROCm build most often fails in on a
    card it carries no kernels for, and neither of them names the build, so neither matched any
    marker: the host was left failing on ROCm with no rung taken and nothing counted.

    They are AMBIGUOUS, not decisive, because both also come out of hosts whose ROCm is fine (a
    container without /dev/kfd for the first, flash attention or multi-GPU P2P for the second), so
    one occurrence must still leave a working host exactly where it was.
    """
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
        # The exact shape that made this necessary: ROCm reports an exhausted card through the
        # same prefix "rocm error" the ambiguous tier matches on.
        "ROCm error: out of memory",
        "HIP error: out of memory",
        "hipErrorOutOfMemory",
        "ggml_backend_cuda_buffer_type_alloc_buffer: failed to allocate 4096 MiB",
        "sd-cli exited 1. Last output:\nnot enough memory to allocate the compute buffer",
        # An allocation failure whose tail also carries a build-shaped string must not divert the
        # host on one occurrence either.
        "hipErrorNoBinaryForGpu reported while out of memory",
    ],
)
def test_an_exhausted_card_is_never_read_as_an_unusable_build(output):
    """An OOM is a statement about the REQUEST, not about the build.

    "ROCm error: out of memory" contains "rocm error", so it counted as an ambiguous strike, and
    two of them under one fingerprint moved the host to Vulkan permanently. Nothing about the host
    changed, so the fingerprint can never retire that note, yet the same build renders the same
    model perfectly at a smaller size. `output_shows_accelerator_failure` already promised this in
    its own docstring: "an out-of-memory never persists a preference away from the host's own
    accelerator".
    """
    from core.inference.sd_cpp_backend import (
        output_shows_accelerator_failure,
        output_shows_capacity_failure,
        output_shows_decisive_accelerator_failure,
    )

    assert output_shows_capacity_failure(output) is True
    assert output_shows_accelerator_failure(output) is False
    assert output_shows_decisive_accelerator_failure(output) is False


def test_repeated_out_of_memory_never_diverts_the_host(fake_settings, monkeypatch):
    """Past the strike count, because one occurrence was never the thing that was wrong."""
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
    """A decisive message is the defect itself, so it does not wait for a second occurrence."""
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
        # Nothing moved: the record still describes this host.
        ({}, True),
        # torch.version.hip is the ROCm version the torch WHEEL was built against, so this moves
        # when the user reinstalls torch from a different ROCm index. A driver-only upgrade under
        # the same wheel does NOT move it, which is why the cards and the reset route carry the
        # rest of the retirement story.
        ({"runtime": "7.0.0"}, False),
        # A new sd.cpp bundle is a new build, and a new build can carry the missing kernels.
        ({"bundle": "master-b9999"}, False),
        # A different card is a different question entirely.
        ({"gpus": ["AMD Radeon RX 9070 XT"]}, False),
        # Unknown on either side is NOT a mismatch: these components are best-effort reads, and a
        # probe that intermittently answers "unknown" must not make the host flip build to build.
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
    """End to end on the same point: a diverted host that upgrades ROCm is not stuck on Vulkan."""
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
    """Ambiguous strikes counted against the old driver are not evidence about the new one."""
    from core.inference import sd_cpp_backend

    sd_cpp_backend.note_accelerator_runtime_failure("rocm", proven = False)
    monkeypatch.setattr(
        sd_cpp_backend,
        "_HOST_FINGERPRINT_MEMO",
        {"runtime": "7.0.0", "gpus": ["AMD Radeon RX 7900 XTX"]},
        raising = False,
    )
    sd_cpp_backend.note_accelerator_runtime_failure("rocm", proven = False)
    # One strike under each fingerprint, so neither reaches the threshold.
    assert sd_cpp_backend.accelerator_runtime_failed("rocm") is False


def test_the_settings_route_reports_and_clears_the_record(fake_settings, monkeypatch):
    """The way back. A preference that outlives the condition that set it needs a reset that is not
    "edit the application database", and reinstalling does not clear this one: it lives in
    settings, not in the managed tree."""
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
    # Both halves, or this process would keep diverting after the persisted record is gone.
    assert sd_cpp_backend.accelerator_runtime_failed("rocm") is False
    assert sd_cpp_backend.preferred_accelerator("rocm") == "rocm"
    assert _noted_accelerators(fake_settings) == []


def test_a_stale_record_is_reported_as_stale_rather_than_hidden(fake_settings, monkeypatch):
    """A user looking at the setting should be able to see that the note is already inert."""
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
    """An early build of this feature stored a bare list of names. Read it as what it meant rather
    than discarding it, so a dev-build user is not silently un-diverted."""
    from core.inference import sd_cpp_backend

    fake_settings["sd_cpp_accelerator_runtime_failures"] = ["rocm"]
    assert sd_cpp_backend.accelerator_runtime_failed("rocm") is True
    assert sd_cpp_backend.preferred_accelerator("rocm") == "vulkan"


@pytest.mark.parametrize(
    "stored", [None, "", [], {}, "not json", '["rocm"]', {"rocm": "yes"}, {"": {}}, 17]
)
def test_an_unreadable_record_never_breaks_a_load(fake_settings, stored):
    """The store is best effort in both directions: a value this cannot parse costs the preference
    and nothing else, and must never raise into a load."""
    from core.inference import sd_cpp_backend

    fake_settings["sd_cpp_accelerator_runtime_failures"] = stored
    assert sd_cpp_backend.accelerator_runtime_failed("rocm") in (True, False)
    assert sd_cpp_backend.preferred_accelerator("rocm") in ("rocm", "vulkan")


# ---------------------------------------------------------------------------
# Finding 2: the swapped-in fallback binary must carry its OWN reading.
# ---------------------------------------------------------------------------


# The ROCm build cannot be asked anything (verdict None, which keeps the GPU), and the Vulkan build
# that replaces it decisively enumerates the CPU only. Before the fix the CPU-only Vulkan binary
# inherited listed_accelerator=True from the unreadable ROCm probe.
_ROCM_UNKNOWN_VULKAN_CPU_ONLY = {
    "rocm": None,
    "vulkan": _DEVICES_CPU_ONLY,
    "cpu": _DEVICES_CPU_ONLY,
}


@pytest.mark.parametrize("platform", PLATFORMS)
def test_a_cpu_only_fallback_build_does_not_inherit_the_gpu_reading(
    h3_amd_host, fake_settings, platform
):
    """The swapped-in binary has to carry its own verdict.

    Carrying the binary while keeping the reading taken from the build it replaced left
    native_device on the GPU with a CPU-only binary attached. The claimed re-vet after the component
    fetch then rejected the mismatch, so the cost was a load that failed minutes in rather than CPU
    work billed as GPU work, but it is a load that did not need to fail."""
    host = h3_amd_host(
        platform = platform,
        backend = "rocm",
        device = "cuda",
        devices = _ROCM_UNKNOWN_VULKAN_CPU_ONLY,
    )
    backend_obj = host.run()
    # The CPU rung is reached, which is the whole point: it commits rather than failing later.
    assert host.ensured == ["rocm", "vulkan", "cpu"], host.ensured
    assert backend_obj._state.device == "cpu"
    # Nothing is recorded: the fallback was not shown to be better, so there is no preference worth
    # persisting and the next load tries this host's own accelerator again.
    assert _noted_accelerators(fake_settings) == []


@pytest.mark.parametrize("platform", PLATFORMS)
def test_an_unreadable_fallback_never_raises_the_reading(h3_amd_host, fake_settings, platform):
    """The guard on the fix. The swap may only LOWER the reading: a fallback probe that says
    nothing must leave a decisive CPU-only verdict alone rather than promoting it to "has a GPU"."""
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


# ---------------------------------------------------------------------------
# Release safety, checklist item 7: exactness of untouched paths, by ENUMERATION rather than by
# reading the diff. The load path's whole input space is (what the host's own build answers) x
# (what the fallback build answers), each of them four-valued: MISSING (no such build can be
# installed), None (installed, cannot be asked), False (answers, CPU only), True (answers, has an
# accelerator device).
#
# `main` in the table is what the unmodified code committed for that corner, derived from its two
# rules: listed = (verdict is None) or verdict, and the CPU rung is taken when not listed.
# ---------------------------------------------------------------------------

_T, _F, _N, _X = "accel", "cpu_only", "unreadable", "missing"

_ANSWER = {_T: _DEVICES_VULKAN, _F: _DEVICES_CPU_ONLY, _N: None, _X: MISSING}

# (rocm answer, vulkan answer) -> (ensured, committed device, what main committed, why the change
# is allowed). "same" means the corner is untouched.
_ROUTING = {
    # The host's own build works. The rung is never reached, on any fallback answer.
    (_T, _T): (["rocm"], "cuda", "cuda", "same"),
    (_T, _F): (["rocm"], "cuda", "cuda", "same"),
    (_T, _N): (["rocm"], "cuda", "cuda", "same"),
    (_T, _X): (["rocm"], "cuda", "cuda", "same"),
    # The host's own build answers CPU only. main went straight to the CPU rung.
    (_F, _T): (["rocm", "vulkan"], "cuda", "cpu", "upgraded on positive evidence"),
    (_F, _F): (["rocm", "vulkan", "cpu"], "cpu", "cpu", "same"),
    (_F, _N): (["rocm", "vulkan", "cpu"], "cpu", "cpu", "same"),
    (_F, _X): (["rocm", "vulkan", "cpu"], "cpu", "cpu", "same"),
    # The host's own build cannot be asked: #8814 and #9278. main kept the GPU on a build that
    # cannot start, which is the defect.
    (_N, _T): (["rocm", "vulkan"], "cuda", "cuda", "upgraded on positive evidence"),
    # The one corner whose committed DEVICE moves down. main committed the GPU -- it never installs
    # the fallback build at all here, so its answer depends only on the ROCm column -- but it
    # committed it on an sd-cli that could not be asked for its devices, which is an sd-cli that
    # dies during backend init. That load fails at generation time, minutes in. A load that was
    # already going to fail is the only thing a new branch is allowed to capture.
    (_N, _F): (["rocm", "vulkan", "cpu"], "cpu", "cuda", "was a failed load"),
    (_N, _N): (["rocm", "vulkan"], "cuda", "cuda", "same"),
    (_N, _X): (["rocm", "vulkan"], "cuda", "cuda", "same"),
    # No ROCm asset for this host at all.
    (_X, _T): (["rocm", "vulkan"], "cuda", "cpu", "upgraded on positive evidence"),
    (_X, _F): (["rocm", "vulkan", "cpu"], "cpu", "cpu", "same"),
    (_X, _N): (["rocm", "vulkan", "cpu"], "cpu", "cpu", "same"),
    (_X, _X): (["rocm", "vulkan", "cpu"], "cpu", "cpu", "same"),
}


@pytest.mark.parametrize("corner", sorted(_ROUTING), ids = lambda c: f"rocm_{c[0]}-vulkan_{c[1]}")
def test_the_whole_load_routing_space_is_enumerated(h3_amd_host, fake_settings, corner):
    """Every corner of the load path's input space, with its committed device pinned.

    The property this establishes is the one a release-safety claim needs: no input that committed
    a device on main commits a DIFFERENT device now, except where main's commit was already a
    failed load, and every upgrade is taken only on the fallback build positively enumerating an
    accelerator of its own.
    """
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
    # And the preference is written in exactly the corners that were upgraded, never in one where
    # the fallback was not shown to be better.
    assert _noted_accelerators(fake_settings) == (
        ["rocm"] if _why == "upgraded on positive evidence" else []
    )


def _main_would_commit(rocm_answer: str) -> str:
    """The device the UNMODIFIED code commits for a corner, from main's own two rules.

    main never installs the fallback build, so its answer is a function of the ROCm column alone:
    ``listed = sd_cpp_lists_accelerator_device(binary)``, which collapses "could not be asked" to
    True, and the CPU rung is taken exactly when ``listed`` is false. A MISSING build means no
    binary at all, which is not listed either.

    Written out so the `main` column of ``_ROUTING`` is DERIVED rather than asserted by hand. The
    two properties below read that column, so a wrong entry in it would have silently licensed a
    device downgrade; one was found there (the unreadable / CPU-only corner had been annotated with
    a failure mode that belongs to an intermediate state of this branch, not to main).
    """
    if rocm_answer == _X:
        return "cpu"
    verdict = {_T: True, _F: False, _N: None}[rocm_answer]
    return "cuda" if (True if verdict is None else verdict) else "cpu"


def test_the_main_column_is_what_main_actually_commits():
    """The column the two properties below rest on, checked against main's rules rather than
    trusted. Verified once against a real run of 2ab07c9c4 with the same harness, corner by
    corner; this keeps it true as the table is edited."""
    for (rocm_answer, _vulkan), (_ensured, _device, main_device, _why) in _ROUTING.items():
        assert main_device == _main_would_commit(rocm_answer), (rocm_answer, main_device)


def test_no_corner_loses_a_gpu_it_previously_kept():
    """The enumeration read as a property rather than as a table, so a future edit to _ROUTING
    cannot quietly encode a regression: a corner may only move off the GPU if what main did there
    was already a failed load."""
    for corner, (_ensured, device, main_device, why) in _ROUTING.items():
        if main_device == "cuda" and device != "cuda":
            assert why == "was a failed load", corner


def test_every_upgrade_required_positive_fallback_evidence():
    """The other half: a corner is only allowed to gain the GPU when the fallback build answered
    --list-devices with an accelerator of its own."""
    for (_rocm, vulkan), (_ensured, _device, _main, why) in _ROUTING.items():
        if why == "upgraded on positive evidence":
            assert vulkan == _T, (_rocm, vulkan)


# ---------------------------------------------------------------------------
# The cards half of the fingerprint, against the REAL _host_fingerprint rather than the pinned memo
# the tests above use. The whole point of that component is that a card change retires the record,
# and it can only do that if it is actually read.
# ---------------------------------------------------------------------------


@pytest.fixture
def unpinned_fingerprint(monkeypatch):
    """Undo the autouse memo pin, so _host_fingerprint runs for real, and give the runtime half a
    known value so only the cards vary."""
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
    """Measured on a real host before this was fixed: ``get_physical_gpu_inventory(block = False)``
    on a COLD cache returns the unknown sentinel and only schedules the probe, so the first call in
    a process sees no cards at all. Memoising the whole host half then carried that ``None`` for the
    life of the process -- past the refresh landing a second later, and past any card added or
    removed while Studio runs.

    That is the component the record's retirement is supposed to rest on, and the note is written by
    a LOAD, so the record a card change should retire is exactly the one most likely to have been
    written with no cards in it. The runtime half is still memoised; it is a string baked into the
    torch wheel and genuinely cannot move inside one process.
    """
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
    """An eGPU plugged in, or a second card added, inside the life of one process. Before the memo
    was narrowed this could not retire anything until Studio was restarted, and the settings route
    reported the note as live rather than stale."""
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
    """The cards are re-read per call; the wheel label is not. Both matter: re-reading the cards is
    what the test above needs, and not paying for the torch attribute read on every load is what
    keeps this off the load path's cost."""
    import torch

    from utils.hardware import hardware

    cards = [_inventory("first")]
    monkeypatch.setattr(hardware, "get_physical_gpu_inventory", lambda *, block = True: cards[0])
    monkeypatch.setattr(unpinned_fingerprint, "_RUNTIME_FINGERPRINT_MEMO", None, raising = False)
    monkeypatch.setattr(torch.version, "hip", "6.4.0", raising = False)

    first = unpinned_fingerprint._host_fingerprint()
    assert first == {"runtime": "6.4.0", "gpus": ["first"]}
    # A wheel label cannot actually change inside a process, so a later read of it is not taken.
    monkeypatch.setattr(torch.version, "hip", "7.0.0", raising = False)
    cards[0] = _inventory("second")
    assert unpinned_fingerprint._host_fingerprint() == {"runtime": "6.4.0", "gpus": ["second"]}
