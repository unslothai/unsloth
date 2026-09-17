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

import inspect
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


def _recorded_strikes(store: dict, klass: str = "rocm") -> int:
    """Strikes standing against *klass*, diverting or not.

    A record that does not divert is not the same as no record: an unreadable probe leaves a
    strike, and it is the accumulation of them under one fingerprint that eventually moves the
    host. Tests that assert "not diverted yet" say which of the two they mean through this.
    """
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
    "state,devices,diverts",
    [
        # The host's own build is installed and cannot be asked anything. The fallback
        # enumerating a device says nothing about WHY, so this is a strike, not a proof.
        ("rocm_unrunnable", _ROCM_BROKEN, False),
        # It answered, and what it answered was "no accelerator of my own". That, with the
        # fallback listing one, is the evidence the record is for.
        ("rocm_cpu_only", _ROCM_CPU_ONLY, True),
        # Nothing was obtained, which is NOT the same as "no such build exists": the ensure
        # returns None for a download that failed as readily as for an asset this host has no
        # build of, and nothing about the ROCm build was observed either way. Proven, one bad
        # fetch would divert a healthy host for good, and it would not even age out: with no
        # binary there is no owning root, so the fingerprint carries no bundle tag and a
        # release that ships the asset cannot retire the record. A strike, like the
        # unreadable probe above.
        ("no_rocm_asset", _VULKAN_ONLY, False),
    ],
)
def test_a_rocm_build_that_cannot_run_falls_back_to_vulkan(
    h3_amd_host, fake_settings, platform, state, devices, diverts
):
    """The fix. On main every one of these commits ``native_device = "cpu"`` (or refuses), because
    the only rung below ROCm was the CPU build. The Vulkan build runs on both reporters' cards."""
    host = h3_amd_host(platform = platform, backend = "rocm", device = "cuda", devices = devices)
    backend_obj = host.run()
    assert host.ensured == ["rocm", "vulkan"], host.ensured
    # Committed on the GPU, on the Vulkan build, with the bundle fetched exactly once.
    assert backend_obj._state.device == "cuda"
    assert len(host.downloads) == 4
    # And remembered, so the next load does not pay for the broken ROCm build again -- where
    # what was seen amounts to proof. Where it does not, the strike is still kept, so a host
    # that keeps failing this way is diverted by the second one.
    assert _noted_accelerators(fake_settings) == (["rocm"] if diverts else [])
    assert _recorded_strikes(fake_settings) == 1


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
        # rocm_runs=False here is the build that cannot be asked anything, so one load is a
        # strike rather than a diversion.
        assert _noted_accelerators(fake_settings) == []
        assert _recorded_strikes(fake_settings) == 1
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
        # "out of memory" is not the only way a card says it is full, and these two carry a
        # build-shaped string of their own. The wordings are the ones this repository's own
        # OOM classifier in `utils.utils` already recognises.
        "sd-cli exited 1. Last output:\nROCm error: CUBLAS_STATUS_ALLOC_FAILED",
        "sd-cli exited 1. Last output:\nrocBLAS error: memory allocation failed",
        "sd-cli exited 1. Last output:\nhipMalloc: cannot allocate memory",
        "sd-cli exited 1. Last output:\nggml_backend_alloc_ctx_tensors: allocation failure",
        "sd-cli exited 1. Last output:\nROCm error: out of device memory",
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


def test_the_report_does_not_claim_a_diversion_the_switch_turned_off(fake_settings, monkeypatch):
    """`enabled: false, diverting: true` is a contradiction, and the wrong half is the one the
    operator acts on.

    With UNSLOTH_DIFFUSION_SD_CPP_VULKAN_FALLBACK=0, `fallback_accelerator_for` returns None and
    `preferred_accelerator` leaves ROCm selected however many strikes the record holds. The
    settings route was reporting the record's own verdict, so the one host that explicitly asked
    not to be moved off its accelerator was told its loads were being redirected.
    """
    from core.inference import sd_cpp_backend
    from routes import settings as settings_routes

    sd_cpp_backend.note_accelerator_runtime_failure("rocm")
    monkeypatch.setenv("UNSLOTH_DIFFUSION_SD_CPP_VULKAN_FALLBACK", "0")

    # The premise, measured rather than assumed: nothing is actually being diverted.
    assert sd_cpp_backend.preferred_accelerator("rocm") == "rocm"
    assert sd_cpp_backend.fallback_accelerator_for("rocm") is None

    state = settings_routes._diffusion_accelerator_fallback_response()
    assert state.enabled is False
    assert state.diverting is False, "reported a diversion the switch had turned off"
    assert state.records[0].diverting is False
    # Nothing is hidden: the record itself is still fully visible next to `enabled: false`, so
    # an operator can see what WOULD happen if they turned the rung back on.
    assert state.records[0].accelerator == "rocm"
    assert state.records[0].proven is True
    assert state.records[0].stale is False
    # And the record's own verdict is untouched, because it is a fact about the host rather
    # than about the switch. `preferred_accelerator` is what consults the switch.
    assert sd_cpp_backend.accelerator_runtime_failed("rocm") is True

    # Turn it back on and the same record diverts again, so this gates on the switch and not on
    # something the test did to the record.
    monkeypatch.delenv("UNSLOTH_DIFFUSION_SD_CPP_VULKAN_FALLBACK", raising = False)
    back = settings_routes._diffusion_accelerator_fallback_response()
    assert back.enabled is True
    assert back.diverting is True
    assert back.records[0].diverting is True


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
    upgraded = _why == "upgraded on positive evidence"
    # ...and only where the host's own build ANSWERED. A probe that could not be read is not
    # evidence about the build: the fallback enumerating a device of its own says nothing about
    # why the first one was unreadable, and persisting that as proven skipped an otherwise
    # healthy, faster ROCm build on every later load. A build that was never obtained is the
    # same non-answer: the ensure returns None for a download that failed exactly as it does
    # for an asset that does not exist, so one bad fetch would divert the host for good.
    assert _noted_accelerators(fake_settings) == (
        ["rocm"] if (upgraded and rocm_answer not in (_N, _X)) else []
    )
    # The unreadable one still leaves a strike, which is what eventually diverts a host where
    # this keeps happening. (_X is not a probe at all: no build of that accelerator exists here,
    # and the bundle tag in the fingerprint retires that record when a release ships one.)
    assert _recorded_strikes(fake_settings) == (1 if upgraded else 0)


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


# ---------------------------------------------------------------------------
# What the runners measured, as tests. Both of these are shapes taken off a real
# Strix Halo (gfx1151) box rather than off an issue report, one per OS.
# ---------------------------------------------------------------------------


# The exact device the Linux gfx1151 runner's inventory answered with: no name, because the
# reading came from sysfs-drm rather than amd-smi.
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
    """Measured on the gfx1151 Linux CI runner: the inventory answers from sysfs-drm with
    ``name = None`` and the gfx target in ``gfx_candidates``. Keying the component on ``name``
    alone dropped the only device, so the cards half was ``None`` on a machine with a card in it,
    and a record written there could not be retired by a card change at all -- on precisely the
    AMD hosts this feature exists for."""
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
    """The Windows gfx1151 runner names its card, so this is the other half of the same
    measurement. The name is kept because it is the legible half, and the gfx target is kept
    beside it because it is the half the stored claim is actually about."""
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
    """AMD reports a great many distinct targets under one generic marketing name: a gfx1103
    laptop APU and a gfx1151 Strix Halo both answer `AMD Radeon(TM) Graphics`. Keying on the name
    alone left the fingerprint unchanged across that swap, so a `no code objects for this card`
    record written on the first host kept diverting the second to Vulkan even though its build
    has the code objects."""
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
    """The identity has to distinguish gfx targets, or swapping a 7900 XTX for a 9070 XT on a host
    that names neither would leave the record standing."""
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
    """A record written against no cards at all must stay "cannot tell", not "these cards"."""
    from utils.hardware import hardware

    monkeypatch.setattr(hardware, "get_physical_gpu_inventory", lambda *, block = True: {})
    assert unpinned_fingerprint._host_fingerprint()["gpus"] is None


# ── The failure that prints nothing ──────────────────────────────────────────


# Measured on the gfx1151 Windows 11 runner: the Windows ROCm asset ships stable-diffusion.dll and
# no hipBLAS, the box has only the driver's amdhip64_6.dll, and every invocation exits 0xC0000135
# in 0.02s having printed zero bytes. This is how sd_cpp_engine surfaces that.
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
    """Windows reports these as unsigned NTSTATUS; a signed reading of the same number is what a
    POSIX-minded caller would print. Both have to match or the recognition depends on who read
    the exit code."""
    from core.inference.sd_cpp_backend import output_shows_image_load_failure
    assert output_shows_image_load_failure(f"sd-cli exited {status}. Last output:\n") is True


@pytest.mark.parametrize("status", [1, 2, 134, 139, -9, -11, 3221225477])
def test_an_ordinary_non_zero_exit_is_not_a_build_failure(status):
    """A crash, a signal or a bad argument says nothing about the build carrying kernels for this
    card, and reading them as decisive would divert a working host on one bad render."""
    from core.inference.sd_cpp_backend import (
        output_shows_decisive_accelerator_failure,
        output_shows_image_load_failure,
    )

    text = f"sd-cli exited {status}. Last output:\nsomething went wrong\n"
    assert output_shows_image_load_failure(text) is False
    assert output_shows_decisive_accelerator_failure(text) is False


def test_an_out_of_memory_alongside_an_image_load_status_is_still_capacity():
    """Capacity is checked first everywhere else; it must stay first here too, or a message that
    happens to carry both numbers would divert a host whose card was merely full."""
    from core.inference.sd_cpp_backend import (
        output_shows_accelerator_failure,
        output_shows_decisive_accelerator_failure,
    )

    text = "sd-cli exited 3221225781. Last output:\nROCm error: out of memory\n"
    assert output_shows_accelerator_failure(text) is False
    assert output_shows_decisive_accelerator_failure(text) is False


def test_the_number_alone_is_not_enough():
    """The status is only evidence when it is an EXIT status. A prompt or a log line that happens
    to contain the digits must not divert anything."""
    from core.inference.sd_cpp_backend import output_shows_image_load_failure
    assert output_shows_image_load_failure("seed 3221225781 produced a nice image") is False


def test_the_vulkan_fallback_pins_the_card_that_was_selected(monkeypatch):
    """`Vulkan0` is not the physical index the user picked.

    The ordinal lookup is confined to the CUDA/ROCm namespace, so after the fallback it
    answers None, no `--backend` is written, and sd.cpp takes its own default device --
    normally the first card -- while the load record and the arbiter claim name the card that
    WAS selected. The card's own name is the one thing the two namespaces agree on.
    """
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
    # The physical index means nothing here, which is the defect.
    assert sd_cpp_backend.sd_cpp_device_name_for_ordinal("/opt/sd/vulkan/sd-cli", 1) is None
    # The name does.
    assert (
        sd_cpp_backend.sd_cpp_device_named("/opt/sd/vulkan/sd-cli", "AMD Radeon RX 7900 XTX")
        == "Vulkan1"
    )
    assert (
        sd_cpp_backend.sd_cpp_device_named("/opt/sd/vulkan/sd-cli", "AMD Radeon RX 7600")
        == "Vulkan0"
    )
    # And it becomes a real pin rather than sd.cpp's default device.
    from core.inference.sd_cpp_args import device_backend_flags

    assert device_backend_flags("Vulkan1") == [
        "--backend",
        "diffusion=Vulkan1,te=Vulkan1,vae=Vulkan1",
    ]


def test_two_identical_cards_are_not_pinned_on_a_guess(monkeypatch):
    """Two of the same card produce two identical descriptions. Picking either would be a
    guess dressed as a pin, which is what this exists to stop: the load runs on sd.cpp's own
    choice, as it does today, and says so."""
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
    # Unless the selection's place in the run of identical cards is known: the second card of
    # that name physically is the second Vulkan entry of that name, because both namespaces
    # walk one vendor's GPUs in the order the driver reports them.
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
    # And a position the device list cannot honour pins nothing rather than guessing.
    assert (
        sd_cpp_backend.sd_cpp_device_named(
            "/opt/sd/vulkan/sd-cli", "AMD Radeon RX 7900 XTX", position = 5
        )
        is None
    )
    # An unreadable probe and a card nothing answers to are the same answer.
    assert sd_cpp_backend.sd_cpp_device_named("/opt/sd/vulkan/sd-cli", "NVIDIA RTX 4090") is None
    assert sd_cpp_backend.sd_cpp_device_named("/opt/sd/vulkan/sd-cli", None) is None


def test_the_h3_load_resolves_the_pin_by_name_when_the_index_says_nothing():
    """The matcher is only worth anything if the load reaches for it."""
    import inspect
    from core.inference import video as video_mod

    source = inspect.getsource(video_mod)
    assert "sd_cpp_device_named(" in source
    ordinal_call = source.index("sd_cpp_device_name_for_ordinal(binary, native_ordinal)")
    named_call = source.index("sd_cpp_device_named(\n", ordinal_call)
    # Second, not instead: a build whose devices ARE in the physical namespace is unchanged.
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
    """An unmasked host: torch's list and the physical list are the same list."""
    for variable in _VISIBILITY_VARS:
        monkeypatch.delenv(variable, raising = False)


def test_the_position_among_identical_cards_is_what_is_carried(monkeypatch):
    """A name cannot separate two 7900 XTXs; "this is the second one" can."""
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
    # An index the host does not have, and no selection at all, are both "cannot tell".
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
    """The inventory, plus the amd-smi mapping from its probe rows to HIP device ids.

    Two index spaces: a visibility mask and torch both speak HIP ids, while the inventory's
    own `index` is amd-smi's discovery row. They coincide on most hosts, so the default here
    is the identity mapping, and a test that needs them to disagree passes its own.
    """
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
    """The count has to be taken over the PHYSICAL cards.

    `HIP_VISIBLE_DEVICES` filters and reorders what torch enumerates, and the Vulkan child
    gets no equivalent mask -- Vulkan does not read those variables -- so it walks every
    card. With two identical cards and only physical card 1 visible, torch's own list makes
    the selection "the first card of that name", and the pin then named physical card 0 while
    Studio reserved and accounted for card 1.
    """
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
    """ROCR filters the agents the runtime reports and HIP then indexes into WHAT IS LEFT,
    not into the physical order, so the two levels have to be composed in that order."""
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
    """Two index spaces wearing one number.

    A visibility mask names HIP device ids, derived from the KFD node id, and that is what
    torch reports as `cuda:N`. The inventory's `index` is amd-smi's own discovery row, which
    its docstring says is not a pin. They coincide on most hosts and not on all, so on a host
    where they disagree, counting positions with the HIP number selected the wrong row and
    pinned Vulkan to a card Studio had not reserved.
    """
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
    """`get_hip_id_by_gpu_index` answers None when any device lacks a usable id -- an older
    amd-smi rejects `list -e` outright. Assuming the identity mapping there is exactly what
    its docstring tells callers not to do, so the position is withheld and the pin falls back
    to the name, which is still unambiguous for a card that is alone of its kind."""
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
    """A mask may name UUIDs, which say nothing about enumeration order. The name still pins
    a card that is alone of its kind; the POSITION is withheld, so `sd_cpp_device_named`
    refuses to choose between identical cards rather than pinning the wrong one."""
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
    """The inventory answers from sysfs-drm with no marketing name on exactly the AMD hosts
    this feature is about, and the matcher has nothing but the name to index Vulkan devices
    by. So a row the OS did not name cannot establish the grouping the tie-break counts in,
    and the position is withheld rather than guessed: `sd_cpp_device_named` then declines to
    choose between identical cards instead of pinning the wrong one. A card alone of its name
    is still pinned by that name."""
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
    """AMD ships many gfx targets under one marketing name, and the matcher can only index
    Vulkan devices by that name. Counting on the gfx-bearing identity instead made a gfx1103
    APU and a gfx1151 card that both report `AMD Radeon(TM) Graphics` two groups, so
    selecting the second gave position 0 and the pin named the first Vulkan device."""
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
    # And a card of a DIFFERENT name before it does not count toward the group.
    _pinned_inventory(
        monkeypatch,
        [
            {"vendor": "amd", "index": 0, "name": "AMD Radeon RX 7600"},
            {"vendor": "amd", "index": 1, "name": "AMD Radeon(TM) Graphics"},
        ],
    )
    assert video_mod._physical_card_name(0) == ("AMD Radeon(TM) Graphics", 0)


def test_a_strike_never_forgets_what_the_last_one_knew(monkeypatch):
    """A component that cannot be read right now is recorded as None, and an unknown reads as
    compatible.

    So overwriting a known GPU or bundle value with None loses the only thing that could ever
    invalidate the record: once it is diverting the host, a later card or bundle change is
    compared against an unknown and skipped, and the note becomes permanent.
    """
    from core.inference import sd_cpp_backend

    previous = {"bundle": "b1", "runtime": "rocm6.2", "gpus": "gfx1100"}
    current = {"bundle": None, "runtime": "rocm6.2", "gpus": None}
    merged = sd_cpp_backend._fingerprint_with_known_fields_kept(previous, current)
    assert merged == {"bundle": "b1", "runtime": "rocm6.2", "gpus": "gfx1100"}

    # A component the new reading DOES have wins: that is a real change, and it is what
    # retires a stale note.
    changed = {"bundle": "b2", "runtime": "rocm6.2", "gpus": "gfx1201"}
    assert sd_cpp_backend._fingerprint_with_known_fields_kept(previous, changed) == changed
    # Nothing to merge from is not an error.
    assert sd_cpp_backend._fingerprint_with_known_fields_kept(None, current) == current


def test_a_second_strike_taken_blind_still_expires_when_the_cards_change(
    fake_settings, monkeypatch
):
    """The consequence, end to end. Two ambiguous strikes divert the host; if the second was
    taken while the GPU list was unreadable, the record used to carry no cards at all and no
    later change could retire it."""
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
    """Containment matches two different cards, and the position does not count them.

    `AMD Radeon RX 7600` is contained in `AMD Radeon RX 7600 XT`, so on a mixed host both
    devices land in the candidate list. The position handed in counts only the cards of the
    SELECTED name in the physical enumeration, so it never included the XT: applying it here
    pins whichever of the two Vulkan enumeration happened to put first, and the load then runs
    on, and accounts for, a different GPU.
    """
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
    # The driver tag is dropped and the rest compared for equality, so each card is found as
    # itself rather than as a substring of the other.
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

    # And where only containment can match, a candidate list that mixes models is an
    # ambiguity rather than a tie: nothing is pinned, whatever position says.
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
    """An ensure does not promise the accelerator that was asked for.

    With installing switched off, offline, or after a failed download it returns whatever
    usable build is already in the managed tree, which on a host that recorded a ROCm crash
    and cannot fetch the Vulkan rung is the ROCm build. That build still answers
    `--list-devices`, so every runnability probe passes and the load commits the very build
    the record exists to avoid -- and the failure the record describes is a crash minutes
    into the render, after a multi-tens-of-GB download.
    """
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
    # Asked for Vulkan, handed back the condemned ROCm build: refused.
    assert sd_cpp_backend.usable_or_recorded_failure("/opt/sd/rocm/sd-cli", "vulkan") is None
    assert sd_cpp_backend.usable_or_recorded_failure("/opt/sd/rocm/sd-cli", "cpu") is None

    # Asked for ROCm and handed back ROCm: kept. With
    # UNSLOTH_DIFFUSION_SD_CPP_VULKAN_FALLBACK=0 preferred_accelerator deliberately asks for
    # ROCm again despite the record, and that opt-out means run it anyway; refusing here
    # would send the load to the CPU rung instead, which is the opposite of the promise.
    assert (
        sd_cpp_backend.usable_or_recorded_failure("/opt/sd/rocm/sd-cli", "rocm")
        == "/opt/sd/rocm/sd-cli"
    )

    # A build with no record against it is handed straight back, and so is a user-supplied
    # one whose class is unrecorded: unknown is not a failure.
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
    """The video ladder was not the boundary.

    The image router runs its own two ensures and then probes them for runnability, which a
    ROCm build that only dies mid-render passes: without the same check the router selects
    native and the backend runs the condemned build.
    """
    import inspect
    from core.inference import diffusion_engine_router as router

    body = inspect.getsource(router.select_and_activate_engine)
    ensures = body.count("ensure_sd_server_binary(") + body.count("ensure_sd_cpp_binary(")
    assert ensures == 2, ensures
    # Through the one gate both ensures share, which is `usable_or_recorded_failure` plus the
    # deferred-upgrade exception below it. Counting the raw calls instead would have to be
    # rewritten by any refactor that gives the two ensures a common path, which is the shape
    # they now have.
    assert body.count("_accept(") == ensures + 1, body[:400]
    assert "usable_or_recorded_failure(candidate, install_accelerator, selected_card)" in body


def test_every_ensure_in_the_h3_load_is_checked_against_the_record():
    """One guarded ensure is not the boundary: the fallback ensure and the CPU ensure hand
    back the same wrong-accelerator build under the same conditions."""
    import inspect
    from core.inference import video as video_mod

    source = inspect.getsource(video_mod)
    load = source[source.index("allow_install = _install_allowed()") :]
    ensures = load.count("ensure_h3_sd_cpp_binary(")
    guarded = load.count("usable_or_recorded_failure(")
    assert ensures >= 3, ensures
    assert guarded == ensures, (guarded, ensures)


def test_the_note_can_be_pinned_to_the_build_that_failed(monkeypatch):
    """The bundle tag is read live out of the managed install record.

    The load path installs the fallback bundle BEFORE it records the failure, so a reading
    taken at the note describes the build that replaced the failed one. A newer release's
    ROCm asset would then match the stored tag and the record would suppress the retry that
    might have worked on it.
    """
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

    # A caller that has installed nothing still gets the live reading.
    monkeypatch.setattr(sd_cpp_backend, "_accelerator_runtime_failures", {}, raising = False)
    stored.clear()
    sd_cpp_backend.note_accelerator_runtime_failure("rocm")
    record = (stored or sd_cpp_backend._accelerator_runtime_failures).get("rocm")
    assert record["fingerprint"]["bundle"] == "after-the-install", record


def test_the_load_path_reads_the_fingerprint_before_it_installs_the_fallback():
    """The ordering is the whole of it, so it is pinned in the source rather than only in the
    behaviour of the helper."""
    import inspect
    from core.inference import video as video_mod

    source = inspect.getsource(video_mod)
    # And it is the FAILED build's own root that is fingerprinted, not the current default:
    # a binary the finder served out of the legacy tree beside the Unsloth home would
    # otherwise carry a tag belonging to an unrelated install.
    read = source.index("failed_fingerprint = _accelerator_fingerprint(binary)")
    install = source.index("fallback_binary = usable_or_recorded_failure(")
    note = source.index("note_accelerator_runtime_failure(\n")
    assert read < install < note, (read, install, note)
    assert "fingerprint = failed_fingerprint" in source[note : note + 400]
    # And the note is only PROVEN where the host's own build answered.
    assert (
        "proven = accelerator_probe_ran and accelerator_verdict is not None"
        in source[note : note + 400]
    )


def test_a_singleton_match_does_not_answer_for_a_position_it_cannot_hold(monkeypatch):
    """One device answering is not proof that it is the card that was selected.

    The position counts the cards of that name in the PHYSICAL enumeration, so a selection of
    the second of two identical cards against a build that enumerates one of them is
    unresolved: returning the singleton meant the graph could run on one card while the
    arbiter reserved and accounted for the other.
    """
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
    # Position 0 IS the one device, and no position at all is the unchanged single-match case.
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
    """`_installed_accelerator_of` reads the binary's own root and this read the default one.

    A ROCm build the finder served out of the legacy tree beside the Unsloth home would
    therefore be recorded under a tag that is None or belongs to an unrelated install, so
    replacing or upgrading the legacy bundle could never invalidate the record and the host
    stayed diverted to Vulkan until it was cleared by hand.
    """
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
    # Named nothing and nothing discovered, it is the current root.
    assert sd_cpp_backend._accelerator_fingerprint()["bundle"] == "tag-for-current"
    assert asked == ["/roots/legacy", "/roots/current"], asked

    # And named nothing while the finder serves the legacy tree -- which is every
    # CONSULTATION, since `accelerator_runtime_failed` passes no binary -- it is the legacy
    # root as well. Reading the current one there compared a record written against one
    # bundle with another bundle's tag: stale at once if that root has any tag of its own,
    # and never retired at all if it has none, so replacing the legacy build could not clear
    # the diversion.
    asked.clear()
    monkeypatch.setattr(
        sd_cpp_backend, "find_sd_cpp_binary", lambda: "/roots/legacy/bin/sd", raising = False
    )
    assert sd_cpp_backend._accelerator_fingerprint()["bundle"] == "tag-for-legacy"
    assert asked == ["/roots/legacy"], asked


def test_the_decided_class_is_read_under_the_claim_that_validated_it():
    """Read after the claim is released, an install that replaces the managed tree in between
    is recorded as the class this load decided on: the re-vet then compares ROCm with ROCm,
    both builds answer the device probe the same way, and the load commits the very build the
    fallback existed to avoid."""
    import inspect
    from core.inference import video as video_mod

    source = inspect.getsource(video_mod)
    probe = source.index("accelerator_verdict = sd_cpp_accelerator_device_verdict(binary)")
    read = source.index("decided_accelerator = _installed_accelerator_of(binary)", probe)
    # Inside the same `with` block as the probe, which ends at the line that collapses it.
    collapse = source.index("listed_accelerator = accelerator_verdict_keeps_gpu(", probe)
    assert probe < read < collapse, (probe, read, collapse)
    # And each rung that REPLACES the binary records its own class rather than inheriting one.
    assert source.count("decided_accelerator = fallback_class") == 2, source.count(
        "decided_accelerator = fallback_class"
    )
    assert "_UNREAD_ACCELERATOR" in source


def test_a_second_unreadable_rocm_probe_does_divert(h3_amd_host, fake_settings):
    """The strike is not a no-op: it accumulates, it just is not a proof on its own.

    One unreadable probe can be a timeout, a nonzero exit or a transient fault, and diverting
    on it permanently skipped an otherwise healthy, faster ROCm build. Repeating under the same
    fingerprint is a different claim, and that one does move the host.
    """
    from core.inference import sd_cpp_backend

    for _ in range(sd_cpp_backend._AMBIGUOUS_FAILURE_STRIKES - 1):
        host = h3_amd_host(platform = "linux", backend = "rocm", device = "cuda", devices = _ROCM_BROKEN)
        host.run()
        assert _noted_accelerators(fake_settings) == []
    host = h3_amd_host(platform = "linux", backend = "rocm", device = "cuda", devices = _ROCM_BROKEN)
    host.run()
    assert _noted_accelerators(fake_settings) == ["rocm"]


def test_an_image_generation_that_dies_in_hipblas_records_it_too(fake_settings, monkeypatch):
    """The recorder was reached only from the video path.

    The same sd-cli, run by the image path, produces the same hipBLAS failure, and on an
    image-only host nothing recorded it: `preferred_accelerator` never moved, so every retry
    reinstalled and ran the ROCm build again and the Vulkan rung below it was never reached.
    """
    from core.inference import sd_cpp_backend

    monkeypatch.setattr(sd_cpp_backend, "_installed_accelerator_of", lambda _b: "rocm")
    backend = sd_cpp_backend.SdCppDiffusionBackend.__new__(sd_cpp_backend.SdCppDiffusionBackend)
    backend._engine = types.SimpleNamespace(binary = "/opt/sd/rocm/sd-cli")
    cancel = threading.Event()
    source = inspect.getsource(sd_cpp_backend.SdCppDiffusionBackend.generate)
    # The handler is on the branch that runs BOTH image paths, so neither the server mode nor
    # the one-shot mode can be the one that is not covered.
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
    """A cancel unwinds through the same RuntimeError, and it says nothing about the build."""
    from core.inference import sd_cpp_backend

    source = inspect.getsource(sd_cpp_backend.SdCppDiffusionBackend.generate)
    handler = source.index("note_accelerator_failure_from_output(")
    window = source[max(0, handler - 1400) : handler]
    assert "cancel.is_set()" in window, window
    assert "DIFFUSION_CANCELLED_MSG not in str(exc)" in window, window


def test_the_availability_probe_reads_the_same_record_selection_does(fake_settings, monkeypatch):
    """The prediction and the selection are read together.

    The prediction decides which planner stages the download; selection decides what loads. A
    record that condemns this host's accelerator makes selection refuse a binary that is still
    on disk and still answers its runnability probe, so counting it here predicted native while
    the load went to diffusers -- and an offline load then had none of the diffusers assets,
    because the planner for that engine was never run.
    """
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
    # Nothing recorded: the binary on disk is available, exactly as before.
    assert router.native_binary_installed() is True

    monkeypatch.setattr(sd_cpp_backend, "_installed_accelerator_of", lambda _b: "rocm")
    sd_cpp_backend.note_accelerator_runtime_failure("rocm")
    assert sd_cpp_backend.preferred_accelerator("rocm") == "vulkan"
    # Now selection would refuse that binary as a substitute for the Vulkan build it asked
    # for, so the probe must not answer that native is available either.
    assert router.native_binary_installed() is False


def test_a_server_that_dies_mid_render_is_recorded_against_its_own_binary(
    fake_settings, monkeypatch
):
    """The resident sd-server is the preferred path, and it is not the engine.

    `_resolve_backend` returns no engine in server mode, so reading `self._engine` there
    passes None and the recorder returns immediately: a ROCm server that starts fine and then
    dies in hipBLAS recorded nothing, and every reload ran ROCm again rather than taking the
    Vulkan rung this whole change is for.
    """
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
    """The image path built its pin from the physical ordinal alone.

    `Vulkan0` is not the index the user picked, so the ordinal lookup answered None, no
    --backend was written, and sd.cpp took its own default device while this load reserved and
    accounted for the card that WAS selected. On a multi-GPU host that is an overcommit of one
    card and a reservation against another. The card's own name is what the two namespaces
    agree on, which is what the video path already matches by.
    """
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

    # An unresolvable card still pins nothing, which is the behaviour every build had before.
    monkeypatch.setattr(sd_cpp_backend, "physical_card_name", lambda ordinal: (None, None))
    assert sd_cpp_backend._offload_with_device_pin_impl(["--offload-to-cpu"], "/sd/sd-cli", 1) == [
        "--offload-to-cpu"
    ]


def test_the_image_pin_tells_two_identical_cards_apart(monkeypatch):
    """The case a name match alone cannot decide, which is the common multi-GPU box.

    Two of the same card produce two identical descriptions, so the pin has to carry the
    position within that run of identical names. Getting this wrong is worse than not pinning
    at all: it writes a --backend for the OTHER card while the load reserves and accounts for
    the selected one, which is the exact overcommit the pin exists to prevent. A position
    outside the matches has to pin nothing rather than fall back to the first.
    """
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

    # A third card of that name physically, but only two in the Vulkan namespace: the counts
    # disagree, so the position means nothing and the pin is dropped.
    monkeypatch.setattr(
        sd_cpp_backend, "physical_card_name", lambda _ordinal: ("AMD Radeon RX 7900 XTX", 2)
    )
    assert sd_cpp_backend._offload_with_device_pin_impl([], "/sd/sd-cli", 2) == []


@pytest.mark.parametrize("platform", PLATFORMS)
def test_a_failed_fetch_does_not_divert_a_host_whose_rocm_build_works(
    h3_amd_host, fake_settings, platform
):
    """The recoverable half of "no binary was obtained".

    The ensure answers None for a download that failed exactly as it does for an asset this
    host has no build of, so a transient fetch failure used to be recorded as PROOF that the
    ROCm build cannot run here -- permanently, since with no binary there is no owning root
    and the fingerprint carries no bundle tag for a later release to retire. This drives the
    load twice: once while the fetch produces nothing, then again on the same host with the
    asset available, and the second load must still try ROCm and commit it.
    """
    from core.inference import sd_cpp_backend

    failed_fetch = h3_amd_host(
        platform = platform, backend = "rocm", device = "cuda", devices = _VULKAN_ONLY
    )
    assert failed_fetch.run()._state.device == "cuda"
    assert failed_fetch.ensured == ["rocm", "vulkan"], failed_fetch.ensured
    # A strike, not a diversion.
    assert _recorded_strikes(fake_settings) == 1
    assert _noted_accelerators(fake_settings) == []
    assert sd_cpp_backend.preferred_accelerator("rocm") == "rocm"

    recovered = h3_amd_host(platform = platform, backend = "rocm", device = "cuda", devices = _ROCM_WORKS)
    assert recovered.run()._state.device == "cuda"
    assert recovered.ensured == ["rocm"], recovered.ensured

    # And a host where it keeps happening is still diverted, by the second strike: the
    # ambiguity is what makes one of them insufficient, not a reason to ignore them.
    again = h3_amd_host(platform = platform, backend = "rocm", device = "cuda", devices = _VULKAN_ONLY)
    again.run()
    assert _recorded_strikes(fake_settings) == 2
    assert _noted_accelerators(fake_settings) == ["rocm"]


@pytest.mark.parametrize("platform", PLATFORMS)
def test_a_substituted_cpu_build_is_not_evidence_about_rocm(
    h3_amd_host, fake_settings, monkeypatch, platform
):
    """The ensure keeps a usable build of the WRONG class when the install fails.

    That is deliberate -- "a usable binary of the wrong accelerator is still better than none"
    -- but it means the binary that answers `--list-devices` is not always the one whose class
    is on trial. A CPU build honestly enumerating no accelerator was recorded as proof that
    ROCm cannot run here, under a bundle tag the other assets of that release share, so no
    later release retired it and every future load skipped ROCm.
    """
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
            # The install failed and a CPU build was already on disk.
            return "/opt/sd/cpu/sd-cli"
        return ensure(allow_install = allow_install, accelerator = accelerator)

    monkeypatch.setattr(sd_cpp_backend, "ensure_sd_cpp_binary", _substituting_ensure)

    assert host.run()._state.device == "cuda"
    # A strike, because something did answer and it was not ROCm -- never a proven failure.
    assert _noted_accelerators(fake_settings) == []
    assert _recorded_strikes(fake_settings) == 1
    assert sd_cpp_backend.preferred_accelerator("rocm") == "rocm"


def test_a_server_that_starts_and_dies_is_recorded_from_its_own_output(fake_settings, monkeypatch):
    """The recorder was wired to the generate paths only.

    A ROCm build that cannot come up at all fails inside the load, where the error carries the
    child's own output, and nothing there was reading it: the load fell back to one-shot, the
    render failed later or the load failed outright, and every retry chose ROCm again because
    no note had been written.
    """
    import inspect

    from core.inference import sd_cpp_backend

    source = inspect.getsource(sd_cpp_backend.SdCppDiffusionBackend._run_load)
    start = source.index("sd-server failed to start")
    window = source[start : start + 900]
    assert "note_accelerator_failure_from_output(" in window, window
    assert "server_binary" in window, window

    # And the recorder really does act on the text that error carries.
    monkeypatch.setattr(sd_cpp_backend, "_installed_accelerator_of", lambda _b: "rocm")
    sd_cpp_backend.note_accelerator_failure_from_output(
        "/opt/sd/rocm/sd-server",
        "sd-server exited 1. Last output:\nROCm error: no kernel image is available for "
        "execution on the device",
    )
    assert _noted_accelerators(fake_settings) == ["rocm"]


def test_a_build_that_cannot_launch_at_all_is_counted(fake_settings, monkeypatch):
    """A build that dies in the dynamic loader prints nothing to classify.

    On Windows that is an unsigned status like 0xC0000135 -- a dependent DLL missing, which is
    what a ROCm build looks like on a host with no HIP runtime -- and it arrives as a large
    POSITIVE exit code, so the pre-download probe accepts it and the failure surfaces as a
    start that never comes up. Counted rather than acted on: a missing execute bit or an
    interrupted extraction fail identically and say nothing about the accelerator.
    """
    from core.inference import sd_cpp_backend

    monkeypatch.setattr(sd_cpp_backend, "_installed_accelerator_of", lambda _b: "rocm")
    sd_cpp_backend.note_unlaunchable_accelerator_build("/opt/sd/rocm/sd-server")
    assert _recorded_strikes(fake_settings) == 1
    assert _noted_accelerators(fake_settings) == [], "one launch failure must not divert"
    sd_cpp_backend.note_unlaunchable_accelerator_build("/opt/sd/rocm/sd-server")
    assert _noted_accelerators(fake_settings) == ["rocm"], "a host that keeps failing is moved"

    # A build with no rung below it is left alone, and so is a binary with no recorded class.
    monkeypatch.setattr(sd_cpp_backend, "_installed_accelerator_of", lambda _b: "cpu")
    sd_cpp_backend.note_unlaunchable_accelerator_build("/opt/sd/cpu/sd-server")
    monkeypatch.setattr(sd_cpp_backend, "_installed_accelerator_of", lambda _b: None)
    sd_cpp_backend.note_unlaunchable_accelerator_build("/somewhere/else/sd-server")
    assert _noted_accelerators(fake_settings) == ["rocm"]
    assert _recorded_strikes(fake_settings, "cpu") == 0


def test_both_unlaunchable_load_paths_record_before_they_raise():
    """The two places the load gives up on a present-but-unrunnable build."""
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
    """The first reload after a mid-render ROCm failure is the one the fallback is FOR.

    That server is still resident, so it is still executing out of the managed tree, and an
    accelerator upgrade replaces the binaries in that tree: both ensures decline the install
    and hand back the ROCm build. Refusing it here made `native_available` false and sent the
    reload to diffusers, which downloads a different set of assets entirely and only reaches
    Vulkan on some later load, after that switch happened to unload the server. The load path
    stops the server and then lands the deferred install itself.
    """
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
    # The record that makes the ROCm build a refused substitute, and the resident server that
    # makes it the only thing either ensure can return.
    sd_cpp_backend.note_accelerator_runtime_failure("rocm", proven = True)
    assert sd_cpp_backend.preferred_accelerator("rocm") == "vulkan"

    family = _detect_load_family(H3_REPO, None, "minimax-h3")

    monkeypatch.setattr(router, "_managed_tree_in_use", lambda: True)
    assert (
        router.select_and_activate_engine(family) == "sd_cpp"
    ), "the reload that should have upgraded to Vulkan behind the teardown went to diffusers"

    # And with the tree free, nothing changes: an ensure that still hands back the condemned
    # build has no teardown coming to fix it, so it is refused exactly as before.
    monkeypatch.setattr(router, "_managed_tree_in_use", lambda: False)
    assert router.select_and_activate_engine(family) == "diffusers"

    # Nor does a busy tree help when there is no install to wait for: with installing
    # switched off the deferred upgrade hands back the same ROCm path, and starting it is
    # starting the known-failing build.
    monkeypatch.setattr(router, "_managed_tree_in_use", lambda: True)
    monkeypatch.setattr(router, "_install_allowed", lambda: False)
    assert (
        router.select_and_activate_engine(family) == "diffusers"
    ), "a deferred upgrade that can never install kept the condemned build"


def test_the_router_counts_a_binary_it_rejects_for_not_launching(fake_settings, monkeypatch):
    """Selection runs BEFORE the load, so the load's own recorders never see this build.

    A freshly installed ROCm binary that cannot launch -- the server exiting 126/127 on a
    missing shared library, or `sd-cli --version` exiting nonzero -- was cleared here and
    diffusers was selected, with nothing persisted. Every later forced-native request then
    reinstalled the same build, rejected it the same way, and the Vulkan rung below it was
    never reached.
    """
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
    # Both probes rejected a build, so both are counted -- and a strike is all either is: a
    # missing execute bit fails identically and says nothing about the accelerator.
    assert _recorded_strikes(fake_settings) == 2
    assert _noted_accelerators(fake_settings) == ["rocm"]


def _backend_with_a_deferred_upgrade(
    monkeypatch,
    *,
    delivered,
    requested = "vulkan",
):
    """A backend whose deferred install has just run, returning ``delivered``."""
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
        lambda _self: requested,
        raising = False,
    )
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
    """The router kept this native selection on the strength of an install that had not run.

    It has run by now, and `_upgrade_server_after_teardown` is never fatal: offline, on a
    failed download, or with an archive that carries no server it hands back the path it was
    given. Committing that is committing the build the record condemns, which dies mid-render
    a full download later.
    """
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
    """With the Vulkan fallback switched off the request is ROCm again on purpose, and that
    opt-out means run it anyway -- the same boundary the selection draws."""
    backend = _backend_with_a_deferred_upgrade(
        monkeypatch, delivered = "/opt/sd/rocm/sd-server", requested = "rocm"
    )
    assert (
        backend._upgraded_or_refused("/opt/sd/rocm/sd-server", mode = "server", engine = None)
        == "/opt/sd/rocm/sd-server"
    )


def test_a_serverless_upgrade_is_judged_by_the_cli_this_load_will_run(fake_settings, monkeypatch):
    """A one-shot load resolves to sd-cli precisely BECAUSE the deferral suppressed the
    install, and its binary comes out of the same archive, so the server path says nothing."""
    backend = _backend_with_a_deferred_upgrade(monkeypatch, delivered = None)
    engine = types.SimpleNamespace(binary = "/opt/sd/rocm/sd-cli")
    with pytest.raises(RuntimeError, match = "recorded as failing"):
        backend._upgraded_or_refused(None, mode = "oneshot", engine = engine)

    engine = types.SimpleNamespace(binary = "/opt/sd/vulkan/sd-cli")
    assert backend._upgraded_or_refused(None, mode = "oneshot", engine = engine) is None


def test_the_load_path_takes_the_deferred_upgrade_through_the_record_check(fake_settings):
    """Placement: the helper above proves only what it does once reached."""
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
    """The host's own card enumeration, with the HIP mapping a mask is translated through."""
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
    """One ROCm bundle, two gfx targets: the build can have code for one card on this host and
    none for the other. Recorded against the accelerator alone, the first card's crash sent
    every later load to Vulkan, including one that explicitly selected the supported card."""
    from core.inference import sd_cpp_backend

    sd_cpp_backend.note_accelerator_runtime_failure("rocm", proven = True, card = "Card A@gfx1201")
    assert sd_cpp_backend.accelerator_runtime_failed("rocm", "Card A@gfx1201") is True
    assert sd_cpp_backend.accelerator_runtime_failed("rocm", "Card B@gfx1100") is False
    # And a selection that cannot say which card it is about to use is not narrowed.
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
    """An older record names no card, and reading that as "not this card" would hand the build
    back to the very host that recorded it."""
    from core.inference import sd_cpp_backend

    sd_cpp_backend.note_accelerator_runtime_failure("rocm", proven = True)
    assert sd_cpp_backend.accelerator_runtime_failed("rocm", "Card B@gfx1100") is True


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
    """A mask filters and reorders what torch enumerates, so ordinal 0 under `HIP_VISIBLE_DEVICES=1`
    is the second physical card -- and recording the first one's identity would condemn the wrong
    card."""
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
    """Which leaves every record applying, exactly as it did before."""
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
    """Placement: the narrowing above is worth nothing if the selection never hears the card."""
    import inspect

    from core.inference import diffusion_engine_router as router

    body = inspect.getsource(router.select_and_activate_engine)
    assert "_selected_card(gpu_ids)" in body
    assert (
        "preferred_accelerator(\n            _install_accelerator_for(backend), selected_card\n        )"
        in body
    )

    import routes.inference as inference_routes

    route_source = inspect.getsource(inference_routes)
    call = route_source.split("                select_and_activate_engine,", 1)[1][:600]
    assert "gpu_ids = request.gpu_ids" in call, call
