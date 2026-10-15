# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Pre-load unsloth_zoo.device_type with torch.cuda.is_available mocked, so its @cache captures cuda."""

from __future__ import annotations

# Must run before torch is imported anywhere below, so it is not a fixture.
import importlib.util as _ilu  # noqa: E402
import pathlib as _pathlib  # noqa: E402

_iso = _pathlib.Path(__file__).resolve()
for _up in _iso.parents:
    _candidate = _up / "tests" / "_shared" / "compile_cache_isolation.py"
    if _candidate.is_file():
        _spec = _ilu.spec_from_file_location("_unsloth_compile_cache_isolation", _candidate)
        _mod = _ilu.module_from_spec(_spec)
        _spec.loader.exec_module(_mod)  # sets the env vars on import
        break

# --- one CPU share per xdist worker -----------------------------------------------
# Also before torch is imported. Each xdist worker otherwise starts a torch / OpenMP pool
# as wide as the machine, so `-n 4` on a 4 vCPU runner runs 16 spinning threads on 4 cores
# and CPU training tests slow down by an order of magnitude: test_decision_model's Clef
# training took >330 s there against ~50 s alone. An explicit thread setting is kept.
# Loaded by path: tests/_shared is put on sys.path only below.
import os as _os  # noqa: E402

_share_path = _iso.parent / "_shared" / "xdist_cpu_share.py"
if _share_path.is_file():
    _spec = _ilu.spec_from_file_location("xdist_cpu_share", _share_path)
    _mod = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    _mod.apply(_os.environ)

# --- shared test helpers on sys.path -----------------------------------------------
# tests/_shared holds no package marker and pytest only puts a *test file's* own
# directory on sys.path, so tests/python/, tests/studio/install/ and tests/security/
# cannot reach it by import. Adding it here (this conftest is collected for anything
# under tests/) is what lets all four levels share one module rather than each growing
# a private copy -- see tests/_shared/unsloth_pwsh_runner.py for the case that forced it.
import sys as _sys  # noqa: E402

_shared_dir = _iso.parent / "_shared"
if _shared_dir.is_dir() and str(_shared_dir) not in _sys.path:
    _sys.path.insert(0, str(_shared_dir))

import importlib.util
import os
import sys
import types

import pytest

# A test that hides nvidia-smi to fake a CPU host must not find the real GPUs through NVML.
os.environ.setdefault("UNSLOTH_NVIDIA_LIBRARY_PROBE", "0")


@pytest.fixture(autouse = True)
def _contain_installer_venv_root(tmp_path_factory, monkeypatch):
    """Imported in the body because tests/_shared is added to sys.path further down this file."""
    from installer_venv_root import contain_installer_venv_root
    contain_installer_venv_root(monkeypatch, tmp_path_factory)


def _has_real_accelerator() -> bool:
    """Calls the probe before any spoof; the name is pinned by a test that reads this file's AST."""
    from real_accelerator import has_real_accelerator
    return has_real_accelerator()


def _preload_device_type(package: str, prereqs: tuple[str, ...] = ()) -> bool:
    """Pre-load <package>.device_type under a mocked is_available()==True so its
    @cache captures "cuda"; prereqs are submodules to load first (e.g. 'utils').
    Returns False if anything is unimportable, so the caller falls back to a stub."""
    target = f"{package}.device_type"
    if target in sys.modules:
        return True
    pkg_spec = importlib.util.find_spec(package)
    if pkg_spec is None or not pkg_spec.submodule_search_locations:
        return False
    pkg_path = pkg_spec.submodule_search_locations[0]

    skeleton_already = package in sys.modules
    if not skeleton_already:
        skel = types.ModuleType(package)
        skel.__path__ = [pkg_path]
        skel.__spec__ = pkg_spec
        skel.__package__ = package
        sys.modules[package] = skel

    try:
        for prereq in prereqs:
            full = f"{package}.{prereq}"
            if full in sys.modules:
                continue
            prereq_path = os.path.join(pkg_path, f"{prereq}.py")
            prereq_spec = importlib.util.spec_from_file_location(full, prereq_path)
            prereq_mod = importlib.util.module_from_spec(prereq_spec)
            sys.modules[full] = prereq_mod
            prereq_spec.loader.exec_module(prereq_mod)

        device_type_path = os.path.join(pkg_path, "device_type.py")
        dt_spec = importlib.util.spec_from_file_location(target, device_type_path)
        dt_mod = importlib.util.module_from_spec(dt_spec)
        sys.modules[target] = dt_mod

        import torch

        _orig_is_avail = torch.cuda.is_available
        torch.cuda.is_available = lambda: True  # type: ignore[assignment]
        try:
            dt_spec.loader.exec_module(dt_mod)
        finally:
            torch.cuda.is_available = _orig_is_avail
    except Exception:
        sys.modules.pop(target, None)
        return False
    finally:
        if not skeleton_already:
            sys.modules.pop(package, None)

    return True


def _patch_torch_cuda_for_import() -> None:
    """Stub the torch.cuda.* probes fired at import time once DEVICE_TYPE is
    forced to "cuda"; returning plausible Ampere values lets the import finish
    (real-tensor tests still run on CPU)."""
    try:
        import torch.cuda.memory as _cuda_memory  # type: ignore

        # Zero free is an exhausted card, which callers treat as fatal.
        _cuda_memory.mem_get_info = lambda *a, **k: (60 * 1024**3, 80 * 1024**3)
    except Exception:
        pass
    try:
        import torch
        torch.cuda.get_device_capability = lambda *a, **k: (8, 0)
        torch.cuda.is_bf16_supported = lambda *a, **k: True
    except Exception:
        pass


def _install_device_type_stub(name: str) -> None:
    stub = types.ModuleType(name)
    stub.DEVICE_TYPE = "cuda"
    stub.DEVICE_TYPE_TORCH = "cuda"
    stub.DEVICE_COUNT = 1
    stub.ALLOW_PREQUANTIZED_MODELS = False
    stub.is_hip = lambda: False
    stub.get_device_type = lambda: "cuda"
    stub.get_device_count = lambda: 1
    stub.device_synchronize = lambda *a, **k: None
    stub.device_empty_cache = lambda *a, **k: None
    stub.device_is_bf16_supported = lambda *a, **k: False
    stub.arch_lacks_bf16 = lambda arch: (
        str(arch or "").split(":", 1)[0].strip().lower().startswith("gfx10")
    )
    # gfx101x (RDNA1) only; gfx103x (RDNA2) must not match.
    stub.arch_lacks_buffer_ops = lambda arch: (
        str(arch or "").split(":", 1)[0].strip().lower().startswith("gfx101")
    )
    stub.apply_gfx101x_triton_workaround = lambda *a, **k: False
    stub.gfx101x_triton_workaround_applied = lambda: False
    stub.hip_visible_archs = lambda: []
    sys.modules[name] = stub


def _preimport_bitsandbytes() -> None:
    """Import bitsandbytes before the CUDA spoof; inside it, a CPU-only torch raises AttributeError."""
    try:
        import bitsandbytes  # noqa: F401
    except Exception:
        pass


if not _has_real_accelerator():
    _preimport_bitsandbytes()
    if not _preload_device_type("unsloth_zoo", prereqs = ("utils",)):
        _install_device_type_stub("unsloth_zoo.device_type")
    if not _preload_device_type("unsloth"):
        _install_device_type_stub("unsloth.device_type")
    _patch_torch_cuda_for_import()


# Import unsloth to apply its import-time upstream-drift fixes; ImportError is swallowed.


def _apply_upstream_import_fixes_for_tests() -> None:
    try:
        import unsloth  # noqa: F401  # runs unsloth/import_fixes.py
    except Exception:
        pass


_apply_upstream_import_fixes_for_tests()
