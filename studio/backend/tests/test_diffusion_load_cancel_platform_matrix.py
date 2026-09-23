# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Load cancellation across [Windows, Linux, WSL, macOS] x [NVIDIA, AMD, Intel XPU, Apple MPS, CPU].

Simulation notice. This host is Linux with an NVIDIA card. The OS axis is `sys.platform` /
`platform.system` / `platform.release` (the WSL marker), the way
`test_gpu_arch_gate_os_matrix_7624.py` and `test_llama_extra_args_platforms.py` do it, and the device
axis is the documented per-instance seam `DiffusionBackend._pick_device_and_dtype`, the way
`_force_cuda_target` in `test_diffusion_backend.py` does it. So these cells prove that the
cancellation LOGIC is device- and platform-independent -- which is the claim under test, since none
of it branches on either -- and they do NOT prove ROCm kernels, Metal placement, oneAPI or real VRAM
reclamation. Only hardware can do that.

What each cell asserts, from the three defects this file was added with:
  * an eject cancels a load that was already in flight (the epoch, not a counter, decides);
  * a load that arrives AFTER the eject started waits for teardown and then runs -- it is not
    refused with "Diffusion load was cancelled.", which is a cancellation that never happened and,
    through the route's 409, surfaced to the user as a failed model switch;
  * both fences drain: `_unload_waiters` and `_teardown_waiters` return to zero, so no later load or
    generation inherits a raised fence.
"""

import importlib.util
import sys
import threading
from pathlib import Path

import pytest

from core.inference.diffusion import DiffusionBackend

# The fake torch/diffusers runtime is reused rather than re-declared. Loaded by path because the
# test directory is not a package, which is the same idiom test_mlx_context_platform_matrix.py uses.
_SIBLING = "test_diffusion_backend"
if _SIBLING in sys.modules:
    _backend_tests = sys.modules[_SIBLING]
else:
    _spec = importlib.util.spec_from_file_location(
        _SIBLING, Path(__file__).with_name(f"{_SIBLING}.py")
    )
    _backend_tests = importlib.util.module_from_spec(_spec)
    sys.modules[_SIBLING] = _backend_tests
    _spec.loader.exec_module(_backend_tests)

fake_runtime = _backend_tests.fake_runtime

# (sys.platform, platform.system(), platform.release() marker)
OS_CELLS = {
    "windows": ("win32", "Windows", "10.0.22631"),
    "linux": ("linux", "Linux", "6.8.0-45-generic"),
    # Indistinguishable from Linux except for this marker, which is exactly why the row exists.
    "wsl": ("linux", "Linux", "5.15.153.1-microsoft-standard-WSL2"),
    "macos": ("darwin", "Darwin", "24.1.0"),
}

# (device string handed to the loader, dtype attribute name on the fake torch)
VENDOR_CELLS = {
    "nvidia": ("cuda:0", "bfloat16"),
    "amd": ("cuda:0", "float16"),  # ROCm reports through the CUDA device string
    "xpu": ("xpu:0", "bfloat16"),
    "mps": ("mps", "float16"),
    "cpu": ("cpu", "float32"),
}

# Cells a user can actually be in: no CUDA/ROCm/XPU on macOS, no Metal anywhere else.
CELLS = [
    (os_key, vendor)
    for os_key in OS_CELLS
    for vendor in VENDOR_CELLS
    if (vendor == "mps") == (os_key == "macos") or vendor == "cpu"
]


@pytest.fixture
def cell(request, monkeypatch, fake_runtime):  # noqa: F811
    """Place the process in one [OS, vendor] cell and hand back a backend pinned to it."""
    import platform as platform_mod
    import sys

    os_key, vendor = request.param
    platform_name, system_name, release = OS_CELLS[os_key]
    device, dtype_name = VENDOR_CELLS[vendor]

    monkeypatch.setattr(sys, "platform", platform_name)
    monkeypatch.setattr(platform_mod, "system", lambda: system_name)
    monkeypatch.setattr(platform_mod, "release", lambda: release)
    if os_key == "wsl":
        monkeypatch.setenv("WSL_DISTRO_NAME", "Ubuntu")
    else:
        monkeypatch.delenv("WSL_DISTRO_NAME", raising=False)

    backend = DiffusionBackend()
    dtype = getattr(sys.modules["torch"], dtype_name)
    monkeypatch.setattr(backend, "_pick_device_and_dtype", lambda ordinal=None: (device, dtype))
    backend._cell = (os_key, vendor, device)
    return backend


@pytest.mark.parametrize("cell", CELLS, indirect=True, ids=[f"{o}-{v}" for o, v in CELLS])
def test_an_eject_cancels_the_load_that_was_already_in_flight(cell, tmp_path):
    backend = cell
    (tmp_path / "model.gguf").write_bytes(b"weights")
    token = backend._load_token
    backend.unload()
    # The worker carries the epoch it was given; the eject bumped it, so the worker is cancelled.
    with pytest.raises(RuntimeError, match="cancelled"):
        backend.load_pipeline(
            str(tmp_path),
            gguf_filename="model.gguf",
            base_repo="base/repo",
            family_override="z-image",
            _load_token=token,
        )
    assert backend._unload_waiters == 0 and backend._teardown_waiters == 0


@pytest.mark.parametrize("cell", CELLS, indirect=True, ids=[f"{o}-{v}" for o, v in CELLS])
def test_a_load_arriving_during_an_eject_queues_and_then_runs(cell, tmp_path, monkeypatch):
    backend = cell
    (tmp_path / "model.gguf").write_bytes(b"weights")
    held, release, outcome = threading.Event(), threading.Event(), {}
    dispatched = threading.Event()
    monkeypatch.setattr(backend, "_run_load", lambda **kwargs: dispatched.set())

    def hold_pipeline_lock():
        # Stands in for the multi-minute constructor (or the active denoise) the eject must wait for.
        with backend._lock:
            held.set()
            release.wait(10)

    def eject():
        try:
            backend.unload()
            outcome["eject"] = "done"
        except BaseException as exc:  # noqa: BLE001
            outcome["eject"] = repr(exc)

    def replacement():
        try:
            backend.begin_load(
                str(tmp_path),
                gguf_filename="model.gguf",
                base_repo="base/repo",
                family_override="z-image",
            )
            outcome["load"] = "accepted"
        except BaseException as exc:  # noqa: BLE001
            outcome["load"] = repr(exc)

    holder = threading.Thread(target=hold_pipeline_lock, daemon=True)
    holder.start()
    assert held.wait(5)
    ejector = threading.Thread(target=eject, daemon=True)
    ejector.start()
    for _ in range(500):
        if backend._unload_waiters:
            break
        threading.Event().wait(0.01)
    assert backend._unload_waiters == 1, "the eject never raised its fence"

    waiter = threading.Thread(target=replacement, daemon=True)
    waiter.start()
    waiter.join(0.5)
    assert waiter.is_alive() and "load" not in outcome, f"{backend._cell}: the load should queue"

    release.set()
    holder.join(5)
    ejector.join(5)
    waiter.join(5)
    assert outcome.get("eject") == "done", outcome
    assert outcome.get("load") == "accepted", f"{backend._cell}: {outcome}"
    assert dispatched.wait(5)
    assert backend._unload_waiters == 0 and backend._teardown_waiters == 0
