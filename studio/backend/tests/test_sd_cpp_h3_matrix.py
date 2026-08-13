# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The MiniMax-H3 native preflight across the host matrix.

``_run_load_h3_native`` decides three things before it commits a runtime: which binary to run,
which device to commit it on, and -- since #8507 -- whether to spend the four-file download at all.
Those interact: the accelerator probe only runs on a GPU target, the CPU fallback rewrites both the
binary and the device, and every refusal has to happen before the bundle is fetched.

Parametrised over ``platform x GPU vendor x binary state`` because the combinations are what broke:
a CPU-only prebuilt on a CUDA host has its own fallback path, and a refusal reached through that
fallback used to download the bundle first on the way to failing.
"""

from __future__ import annotations

import threading
import types
from pathlib import Path

import pytest

from core.inference.video import VideoBackend, _detect_load_family
from core.inference.video_families import VIDEO_CANCELLED_MSG


H3_REPO = "leejet/MiniMax-H3-GGUF"
H3_FILE = "minimax_h3_fl2va-Q4_K_M.gguf"

_BANNER = "stable-diffusion.cpp version unknown, commit unknown\n"
_H3_HELP = _BANNER + "  --ref-video   MiniMax-H3 Ref2VA reference video frame directory\n"
_PRE_H3_HELP = _BANNER + "  -M, --mode    run mode, one of [img_gen, vid_gen, upscale]\n"
# Debian/Ubuntu's find-and-replace `sd`, and Homebrew ships the same tool on macOS.
_UNRELATED_HELP = "sd 1.0.0\nFind & replace CLI\n\nUSAGE:\n    sd <find> <replace-with>\n"

# (label, DiffusionDeviceTarget.backend, .device)
HARDWARE = [
    ("nvidia", "cuda", "cuda"),
    ("amd", "rocm", "cuda"),
    ("apple", "mps", "mps"),
    ("cpu", "cpu", "cpu"),
]
# sys.platform values. WSL is a Linux platform string with a Windows kernel underneath, so it is
# the linux row -- listed separately because it is where PATH picks up a Windows-side install.
PLATFORMS = ["linux", "wsl", "darwin", "win32"]


class _PlanInfo:
    def __init__(self, siblings) -> None:
        self.siblings = siblings


class _Engine:
    def __init__(self, binary) -> None:
        self.binary = binary

    def version(self):
        return "stub-version"


@pytest.fixture
def h3_host(monkeypatch, tmp_path):
    """Drive `_run_load_h3_native` on a chosen host with a chosen binary, recording downloads."""
    from core.inference import video as video_mod
    from core.inference import sd_cpp_backend, sd_cpp_engine

    def _setup(
        *,
        platform: str,
        backend: str,
        device: str,
        help_text: str | None,
        managed: bool = False,
        lists_accelerator: bool = True,
    ):
        monkeypatch.setattr(
            sd_cpp_engine.sys, "platform", "linux" if platform == "wsl" else platform
        )
        monkeypatch.setattr(
            video_mod,
            "resolve_diffusion_device_target",
            lambda: types.SimpleNamespace(backend = backend, device = device, dtype = None),
        )
        monkeypatch.setattr(sd_cpp_backend, "_install_allowed", lambda: True)
        monkeypatch.setattr(sd_cpp_backend, "is_managed_binary", lambda _b: managed)
        monkeypatch.setattr(sd_cpp_engine, "SdCppEngine", _Engine)

        binary = None if help_text is None else "/opt/sd/sd-cli"
        monkeypatch.setattr(sd_cpp_backend, "ensure_sd_cpp_binary", lambda **_kwargs: binary)

        def _probe(_binary, *args):
            if args == ("--list-devices",):
                if lists_accelerator:
                    return "CUDA0\tNVIDIA H100 PCIe\nCPU\tIntel(R) Xeon(R)\n"
                return "CPU\tIntel(R) Xeon(R)\n"
            return help_text

        monkeypatch.setattr(sd_cpp_backend, "_sd_cpp_probe_output", _probe)

        asset_calls: list[str] = []

        class _Api:
            def __init__(self, **_kwargs):
                pass

            def model_info(self, repo, *_args, **_kwargs):
                asset_calls.append(repo)
                return _PlanInfo([])

        monkeypatch.setattr("huggingface_hub.HfApi", _Api)

        downloads: list[str] = []

        def _download(_repo, wanted, *_args, **_kwargs):
            downloads.append(wanted)
            path = tmp_path / Path(wanted).name
            path.write_bytes(b"x")
            return str(path)

        monkeypatch.setattr("utils.hf_xet_fallback.hf_hub_download_with_xet_fallback", _download)

        def run(cancel_event: threading.Event | None = None):
            fam = _detect_load_family(H3_REPO, None, "minimax-h3")
            assert fam is not None
            backend_obj = VideoBackend()
            backend_obj._run_load_h3_native(
                fam = fam,
                token = None,
                cancel_event = cancel_event or threading.Event(),
                repo_id = H3_REPO,
                gguf_filename = H3_FILE,
            )
            return backend_obj

        return types.SimpleNamespace(run = run, downloads = downloads, asset_calls = asset_calls)

    return _setup


@pytest.mark.parametrize("platform", PLATFORMS)
@pytest.mark.parametrize("hw_label,backend,device", HARDWARE)
@pytest.mark.parametrize(
    "state,help_text,expected",
    [
        ("unrelated_binary", _UNRELATED_HELP, "is not stable-diffusion.cpp"),
        ("pre_h3_binary", _PRE_H3_HELP, "does not advertise MiniMax-H3"),
        ("no_binary", None, "could not be installed or started"),
    ],
)
def test_h3_preflight_refuses_before_downloading(
    h3_host, platform, hw_label, backend, device, state, help_text, expected
):
    """Every refusal, on every host, costs zero downloads.

    The bundle is tens of GB. A refusal that arrives after it has been fetched is the failure mode
    the H3 gate was written to prevent, and it was reachable on all of these hosts: the gate ran
    after the download loop, and a None binary was not rejected until later still.
    """
    host = h3_host(platform = platform, backend = backend, device = device, help_text = help_text)
    with pytest.raises(RuntimeError, match = expected):
        host.run()
    assert host.downloads == []


@pytest.mark.parametrize("platform", PLATFORMS)
@pytest.mark.parametrize("hw_label,backend,device", HARDWARE)
def test_h3_preflight_admits_a_capable_build_and_downloads_once(
    h3_host, platform, hw_label, backend, device
):
    """The other direction: a genuine H3 build is not refused anywhere, and the load proceeds to
    fetch exactly the four files. Guards against an identity check that is too strict."""
    host = h3_host(platform = platform, backend = backend, device = device, help_text = _H3_HELP)
    backend_obj = host.run()
    assert len(host.downloads) == 4
    assert backend_obj._state is not None
    # A GPU target keeps its device when the build offers an accelerator; CPU/MPS are unchanged.
    assert backend_obj._state.device == device


@pytest.mark.parametrize("platform", PLATFORMS)
@pytest.mark.parametrize(
    "hw_label,backend,device", [h for h in HARDWARE if h[1] in ("cuda", "rocm")]
)
def test_h3_gpu_host_falls_back_to_the_cpu_build(h3_host, platform, hw_label, backend, device):
    """Upstream publishes no Linux CUDA archive, so a GPU host routinely ends up on the CPU
    prebuilt. It must still load, committed on the CPU rather than on a GPU it never ran on."""
    host = h3_host(
        platform = platform,
        backend = backend,
        device = device,
        help_text = _H3_HELP,
        lists_accelerator = False,
    )
    backend_obj = host.run()
    assert len(host.downloads) == 4
    assert backend_obj._state is not None
    assert backend_obj._state.device == "cpu"


@pytest.mark.parametrize("platform", PLATFORMS)
@pytest.mark.parametrize("hw_label,backend,device", HARDWARE)
def test_h3_cancellation_during_the_preflight_stops_before_the_asset_calls(
    h3_host, platform, hw_label, backend, device, monkeypatch
):
    """The ensure takes no cancel_event and can spend minutes installing the prebuilt, so a cancel
    arriving during it is already late. It must not then cost four more model_info round trips."""
    from core.inference import sd_cpp_backend

    cancelled = threading.Event()
    host = h3_host(platform = platform, backend = backend, device = device, help_text = _H3_HELP)
    original = sd_cpp_backend.ensure_h3_sd_cpp_binary

    def _cancel_midway(**kwargs):
        cancelled.set()  # the user hits cancel while the install is running
        return original(**kwargs)

    monkeypatch.setattr(sd_cpp_backend, "ensure_h3_sd_cpp_binary", _cancel_midway)

    with pytest.raises(RuntimeError, match = VIDEO_CANCELLED_MSG):
        host.run(cancel_event = cancelled)
    assert host.downloads == []
    # The point is not that it eventually raises -- the download loop always would. It is that a
    # cancelled load stops before paying for the four sequential size-estimate round trips.
    assert host.asset_calls == []


@pytest.mark.parametrize("platform", PLATFORMS)
@pytest.mark.parametrize("hw_label,backend,device", HARDWARE)
def test_h3_revets_a_user_supplied_binary_swapped_during_the_download(
    h3_host, platform, hw_label, backend, device, monkeypatch
):
    """Vetting before the download means the vet-to-commit window is now the whole download.

    A user-supplied binary is not ours to reinstall, so nothing else guards it: if the file at that
    path changes while the bundle is fetched, the re-vet is the only thing standing between the
    replacement and being recorded as the identity every later generation compares against."""
    from core.inference import sd_cpp_backend

    host = h3_host(platform = platform, backend = backend, device = device, help_text = _H3_HELP)

    # The preflight sees an H3 build; by the time the download is done the path holds a pre-H3 one.
    swapped = {"done": False}
    real_probe = sd_cpp_backend._sd_cpp_probe_output

    def _probe(binary, *args):
        if args == ("--help",) and swapped["done"]:
            return _PRE_H3_HELP
        return real_probe(binary, *args)

    def _download(*args, **kwargs):
        swapped["done"] = True
        return _real_download(*args, **kwargs)

    _real_download = None
    import utils.hf_xet_fallback as xet

    _real_download = xet.hf_hub_download_with_xet_fallback
    monkeypatch.setattr(sd_cpp_backend, "_sd_cpp_probe_output", _probe)
    monkeypatch.setattr(xet, "hf_hub_download_with_xet_fallback", _download)

    with pytest.raises(RuntimeError, match = "changed while this model was loading"):
        host.run()
    assert len(host.downloads) == 4  # the swap is caught after the fetch, not before it


@pytest.mark.parametrize("platform", PLATFORMS)
@pytest.mark.parametrize("hw_label,backend,device", HARDWARE)
def test_h3_revet_checks_identity_not_just_the_h3_marker(
    h3_host, platform, hw_label, backend, device, monkeypatch
):
    """The post-download re-vet asks both of the preflight's questions, not only capability.

    --ref-video is a plain option name that unrelated reference-video tools expose too, so a swap
    to one of those would clear a marker-only re-check and then be recorded by _sd_cli_identity as
    the vetted build every later generation compares against."""
    from core.inference import sd_cpp_backend

    host = h3_host(platform = platform, backend = backend, device = device, help_text = _H3_HELP)

    swapped = {"done": False}
    real_probe = sd_cpp_backend._sd_cpp_probe_output

    def _probe(binary, *args):
        # Not sd.cpp, but it does carry the H3 marker -- capability alone would wave it through.
        if args == ("--help",) and swapped["done"]:
            return "reference-video-cli 2.1\n  --ref-video PATH   reference clip\n"
        return real_probe(binary, *args)

    import utils.hf_xet_fallback as xet

    real_download = xet.hf_hub_download_with_xet_fallback

    def _download(*args, **kwargs):
        swapped["done"] = True
        return real_download(*args, **kwargs)

    monkeypatch.setattr(sd_cpp_backend, "_sd_cpp_probe_output", _probe)
    monkeypatch.setattr(xet, "hf_hub_download_with_xet_fallback", _download)

    with pytest.raises(RuntimeError, match = "changed while this model was loading"):
        host.run()
    assert len(host.downloads) == 4


@pytest.mark.parametrize("platform", PLATFORMS)
@pytest.mark.parametrize("hw_label,backend,device", HARDWARE)
def test_h3_cancellation_precedes_the_binary_install(h3_host, platform, hw_label, backend, device):
    """The preflight can download and extract the sd-cli prebuilt and takes no cancel_event, so a
    load cancelled before its worker started must not reach it."""
    from core.inference import sd_cpp_backend

    host = h3_host(platform = platform, backend = backend, device = device, help_text = _H3_HELP)
    ensures: list[str] = []
    original = sd_cpp_backend.ensure_h3_sd_cpp_binary

    def _spy(**kwargs):
        ensures.append("called")
        return original(**kwargs)

    sd_cpp_backend.ensure_h3_sd_cpp_binary = _spy
    try:
        cancelled = threading.Event()
        cancelled.set()
        with pytest.raises(RuntimeError):
            host.run(cancel_event = cancelled)
    finally:
        sd_cpp_backend.ensure_h3_sd_cpp_binary = original
    assert ensures == []
    assert host.downloads == []
