# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Hardware profiles for CPU-only tests, and one context manager that makes torch look like that card.

Real profiles are the cards Studio ships for or an external tester benchmarks on; synthetic ones are future
architectures, each with an explicit ``parent`` (a real profile) whose resources it copies, so a test can ask "does
the next architecture keep every lever its parent has". Values are what a policy reads (capability, gfx name, name,
VRAM, SM count, opt-in shared memory, host RAM, host link), not marketing specs.

``mock_profile(profile)`` fakes ``torch.cuda`` (availability, capability, name, properties, mem_get_info, bf16
support), the ROCm / HIP identity (``torch.version.hip``, ``gcnArchName``), MPS availability, host RAM and Triton
availability for that profile. No GPU, no network, no kernel launch.
"""

from __future__ import annotations

import contextlib
import dataclasses
import sys
import types
from collections import namedtuple
from typing import Any, Iterator, Optional

GIB = 1024**3

# Host-to-device link classes (what a residency / streaming planner prices).
LINK_GBPS = {
    "pcie3x4": 3.5,
    "pcie3x16": 12.0,
    "pcie4x16": 24.0,
    "pcie5x16": 50.0,
    "unified": 200.0,
    "none": 0.0,
}

# SDPA kernels torch really runs per platform (what ``diffusion_attention.available_sdpa_kernels`` would probe).
_SDPA_SM80 = ("cudnn", "flash", "mem_efficient", "math")
_SDPA_SM75 = ("mem_efficient", "math")
_SDPA_ROCM = ("flash", "mem_efficient", "math")
_SDPA_MPS = ("mem_efficient", "math")


@dataclasses.dataclass(frozen = True)
class GpuProfile:
    key: str
    name: str  # torch.cuda.get_device_name
    vendor: str  # nvidia | amd | apple | cpu
    segment: str  # consumer | workstation | datacenter | apu | laptop | host
    capability: Optional[tuple] = None  # CUDA (major, minor); ROCm reports gfx11.5 as (11, 5)
    gfx: Optional[str] = None  # ROCm gcnArchName
    vram_gb: float = 0.0
    host_ram_gb: float = 64.0
    link: str = "pcie4x16"
    sm_count: int = 0
    smem_optin: int = 0  # shared_memory_per_block_optin, bytes
    triton: bool = True
    sdpa: tuple = ()
    parent: Optional[str] = None  # synthetic profiles only
    synthetic: bool = False

    @property
    def cuda(self) -> bool:
        return self.vendor in ("nvidia", "amd")

    @property
    def rocm(self) -> bool:
        return self.vendor == "amd"

    @property
    def mps(self) -> bool:
        return self.vendor == "apple"

    @property
    def link_gbps(self) -> float:
        return LINK_GBPS[self.link]

    @property
    def arch(self) -> str:
        if self.gfx:
            return self.gfx
        if self.capability:
            return f"sm{self.capability[0]}{self.capability[1]}"
        return self.vendor


def _nv(key, name, segment, cap, vram, host, link, sms, smem, **kw) -> GpuProfile:
    return GpuProfile(
        key = key,
        name = name,
        vendor = "nvidia",
        segment = segment,
        capability = cap,
        vram_gb = vram,
        host_ram_gb = host,
        link = link,
        sm_count = sms,
        smem_optin = smem,
        sdpa = _SDPA_SM75 if cap < (8, 0) else _SDPA_SM80,
        **kw,
    )


_SMEM_SM75 = 64 * 1024
_SMEM_SM80 = 163 * 1024
_SMEM_SM86 = 99 * 1024
_SMEM_SM90 = 227 * 1024

REAL_PROFILES: tuple = (
    _nv("t4", "Tesla T4", "datacenter", (7, 5), 15.0, 31.0, "pcie3x16", 40, _SMEM_SM75),
    _nv(
        "a100_40",
        "NVIDIA A100-SXM4-40GB",
        "datacenter",
        (8, 0),
        39.6,
        83.0,
        "pcie4x16",
        108,
        _SMEM_SM80,
    ),
    _nv(
        "a100_80",
        "NVIDIA A100-SXM4-80GB",
        "datacenter",
        (8, 0),
        79.2,
        167.0,
        "pcie4x16",
        108,
        _SMEM_SM80,
    ),
    # The external tester's 3090 sits on a PCIe 3.0 x4 slot (R5 #18183): keep that link, it is what regressed.
    _nv(
        "rtx3090",
        "NVIDIA GeForce RTX 3090",
        "consumer",
        (8, 6),
        24.0,
        64.0,
        "pcie3x4",
        82,
        _SMEM_SM86,
    ),
    _nv("a10g", "NVIDIA A10G", "datacenter", (8, 6), 22.0, 32.0, "pcie4x16", 80, _SMEM_SM86),
    _nv("l4", "NVIDIA L4", "datacenter", (8, 9), 22.5, 53.6, "pcie4x16", 58, _SMEM_SM86),
    _nv(
        "rtx6000_ada",
        "NVIDIA RTX 6000 Ada Generation",
        "workstation",
        (8, 9),
        48.0,
        128.0,
        "pcie4x16",
        142,
        _SMEM_SM86,
    ),
    _nv(
        "h100",
        "NVIDIA H100 80GB HBM3",
        "datacenter",
        (9, 0),
        79.6,
        200.0,
        "pcie5x16",
        132,
        _SMEM_SM90,
    ),
    _nv("b200", "NVIDIA B200", "datacenter", (10, 0), 178.4, 1024.0, "pcie5x16", 148, _SMEM_SM90),
    _nv(
        "rtx5090",
        "NVIDIA GeForce RTX 5090",
        "consumer",
        (12, 0),
        31.4,
        64.0,
        "pcie5x16",
        170,
        _SMEM_SM86,
    ),
    _nv(
        "rtx_pro6000",
        "NVIDIA RTX PRO 6000 Blackwell Workstation Edition",
        "workstation",
        (12, 0),
        95.0,
        256.0,
        "pcie5x16",
        188,
        _SMEM_SM86,
    ),
    GpuProfile(
        key = "gfx1151",
        name = "AMD Radeon 8060S Graphics",
        vendor = "amd",
        segment = "apu",
        capability = (11, 5),
        gfx = "gfx1151",
        vram_gb = 96.0,
        host_ram_gb = 128.0,
        link = "unified",
        sm_count = 40,
        smem_optin = 64 * 1024,
        sdpa = _SDPA_ROCM,
    ),
    GpuProfile(
        key = "mps",
        name = "Apple M4 Max",
        vendor = "apple",
        segment = "laptop",
        vram_gb = 48.0,
        host_ram_gb = 64.0,
        link = "unified",
        triton = False,
        sdpa = _SDPA_MPS,
    ),
    GpuProfile(
        key = "cpu",
        name = "cpu",
        vendor = "cpu",
        segment = "host",
        host_ram_gb = 64.0,
        link = "none",
        triton = False,
    ),
)


def _child(parent: GpuProfile, key: str, **changes) -> GpuProfile:
    """A future arch: the parent's resources, a new identity."""
    return dataclasses.replace(parent, key = key, parent = parent.key, synthetic = True, **changes)


_BY_KEY = {p.key: p for p in REAL_PROFILES}

SYNTHETIC_PROFILES: tuple = (
    _child(
        _BY_KEY["b200"], "sm103", name = "NVIDIA Synthetic sm103 (B200 resources)", capability = (10, 3)
    ),
    _child(
        _BY_KEY["b200"], "sm110", name = "NVIDIA Synthetic sm110 (B200 resources)", capability = (11, 0)
    ),
    _child(
        _BY_KEY["rtx5090"],
        "sm121",
        name = "NVIDIA GeForce Synthetic sm121 (RTX 5090 resources)",
        capability = (12, 1),
    ),
    _child(
        _BY_KEY["rtx_pro6000"],
        "sm130",
        name = "NVIDIA RTX Synthetic sm130 (RTX PRO 6000 resources)",
        capability = (13, 0),
    ),
    _child(
        _BY_KEY["gfx1151"],
        "gfx1250",
        name = "AMD Radeon Synthetic gfx1250 (gfx1151 resources)",
        capability = (12, 5),
        gfx = "gfx1250",
    ),
)

PROFILES: dict = {p.key: p for p in REAL_PROFILES + SYNTHETIC_PROFILES}


def profile(key: str) -> GpuProfile:
    return PROFILES[key]


# ------------------------------------------------------------------------------------------------------- mocking
_VirtualMemory = namedtuple(
    "svmem", "total available percent used free active inactive buffers cached shared slab"
)


class _Patch:
    """Attribute patcher that also works outside pytest (records and restores, missing attributes deleted)."""

    _MISSING = object()

    def __init__(self) -> None:
        self._undo: list = []

    def setattr(self, obj: Any, name: str, value: Any) -> None:
        old = (
            obj.__dict__.get(name, self._MISSING)
            if isinstance(obj, type)
            else getattr(obj, name, self._MISSING)
        )
        self._undo.append((obj, name, old))
        setattr(obj, name, value)

    def undo(self) -> None:
        for obj, name, old in reversed(self._undo):
            if old is self._MISSING:
                try:
                    delattr(obj, name)
                except AttributeError:
                    pass
            else:
                setattr(obj, name, old)
        self._undo.clear()


def device_properties(p: GpuProfile) -> Any:
    cap = p.capability or (0, 0)
    return types.SimpleNamespace(
        name = p.name,
        major = int(cap[0]),
        minor = int(cap[1]),
        total_memory = int(p.vram_gb * GIB),
        multi_processor_count = int(p.sm_count),
        shared_memory_per_block_optin = int(p.smem_optin),
        shared_memory_per_multiprocessor = int(p.smem_optin + 1024),
        gcnArchName = (p.gfx + ":sramecc-:xnack-") if p.gfx else "",
        is_integrated = int(p.link == "unified"),
        L2_cache_size = 50 * 1024 * 1024,
        warp_size = 32 if p.vendor == "nvidia" else 64,
        uuid = f"GPU-synthetic-{p.key}",
        pci_bus_id = 1,
        pci_device_id = 0,
        pci_domain_id = 0,
    )


def free_bytes(p: GpuProfile) -> int:
    """Free VRAM at load time: the whole card less what a CUDA context and the desktop hold."""
    reserve = 0.6 if p.segment in ("consumer", "workstation") else 0.4
    return int(max(p.vram_gb - reserve, 0.0) * GIB)


# Module helpers that answer "does Triton import / which version" from the installed wheel. Pinned per profile so
# the answer is the profile's, not the host's (CPU torch from pip ships no Triton).
PINNED_TRITON_VERSION = (3, 6)


def _triton_overrides(p: GpuProfile) -> list:
    has = bool(p.triton and p.cuda)

    def at_least(major: int, minor: int) -> bool:
        return has and PINNED_TRITON_VERSION >= (int(major), int(minor))

    def version_ok(version: Optional[str] = None) -> bool:
        if version is not None:
            import re
            m = re.match(r"(\d+)\.(\d+)", str(version))
            return bool(m) and (int(m.group(1)), int(m.group(2))) >= (3, 2)
        return has

    return [
        ("core.inference.diffusion_int8_attention", "triton_available", lambda: has),
        ("core.inference.diffusion_int8_gemm", "_triton_at_least", at_least),
        ("core.inference.diffusion_int8_gemm", "_triton_version_ok", version_ok),
        ("core.inference.diffusion_convrot_quant", "_triton_ok", lambda: has),
        (
            "core.inference.diffusion_grouped_int8_gemm",
            "_triton_version",
            lambda: PINNED_TRITON_VERSION if has else None,
        ),
    ]


@contextlib.contextmanager
def mock_profile(
    p: GpuProfile,
    *,
    device_count: int = 1,
    extra: Any = None,
) -> Iterator[GpuProfile]:
    """Make torch (and Studio's Triton probes) report ``p`` for the duration.

    ``extra`` is an optional callable ``extra(patch, p)`` for test-specific patches applied inside the same undo
    scope. Every patch is undone on exit, also on error.
    """
    import torch

    patch = _Patch()
    props = device_properties(p)
    cuda = p.cuda
    try:
        patch.setattr(torch.cuda, "is_available", lambda: cuda)
        patch.setattr(torch.cuda, "device_count", lambda: device_count if cuda else 0)
        patch.setattr(torch.cuda, "current_device", lambda: 0)
        patch.setattr(torch.cuda, "is_initialized", lambda: cuda)

        def no_device(*a: Any, **k: Any) -> None:
            # Anything that really touches the card (a tensor moved to it, an on-device probe) fails at once, as it
            # does on CPU torch: a CUDA build on a GPU-less host would otherwise spend seconds probing the driver.
            raise RuntimeError(f"profile stand-in {p.key}: no real device to run on")

        patch.setattr(torch.cuda, "_lazy_init", no_device)

        def capability(device: Any = None) -> tuple:
            if not cuda:
                raise RuntimeError("Found no NVIDIA driver on your system")
            return tuple(p.capability)

        def name(device: Any = None) -> str:
            if not cuda:
                raise RuntimeError("Found no NVIDIA driver on your system")
            return p.name

        def properties(device: Any = None) -> Any:
            if not cuda:
                raise RuntimeError("Found no NVIDIA driver on your system")
            return props

        def mem_get_info(device: Any = None) -> tuple:
            if not cuda:
                raise RuntimeError("Found no NVIDIA driver on your system")
            return free_bytes(p), int(p.vram_gb * GIB)

        def bf16(including_emulation: bool = True) -> bool:
            return bool(cuda and (p.rocm or (p.capability or (0, 0)) >= (8, 0)))

        patch.setattr(torch.cuda, "get_device_capability", capability)
        patch.setattr(torch.cuda, "get_device_name", name)
        patch.setattr(torch.cuda, "get_device_properties", properties)
        patch.setattr(torch.cuda, "mem_get_info", mem_get_info)
        patch.setattr(torch.cuda, "is_bf16_supported", bf16)
        patch.setattr(torch.version, "hip", "6.4.43482" if p.rocm else None)
        patch.setattr(torch.version, "cuda", "12.8" if p.vendor == "nvidia" else None)
        mps = getattr(torch.backends, "mps", None)
        if mps is not None:
            patch.setattr(mps, "is_available", lambda: p.mps)
            patch.setattr(mps, "is_built", lambda: p.mps)
        try:
            import psutil

            total = int(p.host_ram_gb * GIB)
            avail = int(total * 0.85)
            vm = _VirtualMemory(total, avail, 15.0, total - avail, avail, 0, 0, 0, 0, 0, 0)
            patch.setattr(psutil, "virtual_memory", lambda: vm)
        except Exception:  # noqa: BLE001 - psutil-less hosts: nothing reads it then
            pass
        for mod_name, attr, value in _triton_overrides(p):
            mod = sys.modules.get(mod_name)
            if mod is None:
                try:
                    mod = __import__(mod_name, fromlist = [attr])
                except Exception:  # noqa: BLE001 - a module that does not import here has no probe to pin
                    continue
            if hasattr(mod, attr):
                patch.setattr(mod, attr, value)
        try:
            # The prebuilt CUTLASS INT8 GEMM wheel counts as installed on NVIDIA (Studio installs it there); whether
            # it serves the arch stays the policy's call (``family``), the .so is never loaded.
            from core.inference import diffusion_int8_cutlass as cut
            wheel = types.SimpleNamespace(_name = "libunsloth_int8_cutlass.so (profile stand-in)")
            patch.setattr(
                cut,
                "library",
                lambda: wheel if (p.vendor == "nvidia" and cut.mode() != "off") else None,
            )
        except Exception:  # noqa: BLE001
            pass
        try:
            from core.inference import diffusion_attention
            patch.setattr(
                diffusion_attention,
                "_sdpa_capability",
                lambda target: tuple(p.sdpa)
                if str(getattr(target, "device", "")) != "cpu"
                else ("math",),
            )
        except Exception:  # noqa: BLE001
            pass
        try:
            # Studio's detected device (cached from the host at startup) and its ROCm flag follow the profile.
            from utils import hardware as uh
            from utils.hardware import hardware as hw

            device_type = (
                uh.DeviceType.CUDA if cuda else uh.DeviceType.CPU
            )  # MPS / CPU: torch decides
            patch.setattr(uh, "get_device", lambda: device_type)
            patch.setattr(hw, "get_device", lambda: device_type)
            patch.setattr(hw, "IS_ROCM", bool(p.rocm))
        except Exception:  # noqa: BLE001
            pass
        try:
            # The ROCm regional-compile probe compiles a block on the card: the profile's card passes it (it is a
            # runtime check on real hardware, like a race); its per-card memo starts empty.
            from core.inference import diffusion_rocm_compile as rc
            patch.setattr(
                rc, "_probe", lambda torch_mod, index: (True, "profile stand-in: probe passes")
            )
            patch.setattr(rc, "_result", {})
        except Exception:  # noqa: BLE001
            pass
        _clear_device_caches()
        if extra is not None:
            extra(patch, p)
        yield p
    finally:
        patch.undo()
        _clear_device_caches()


# lru caches keyed by device index: one profile's answer must not leak into the next.
_DEVICE_CACHES = (
    ("core.inference.diffusion_vae_strips", "_card_strip_elems"),
    ("core.inference.diffusion_vae_strips", "_card_cap"),
)


def _clear_device_caches() -> None:
    for mod_name, attr in _DEVICE_CACHES:
        fn = getattr(sys.modules.get(mod_name), attr, None)
        clear = getattr(fn, "cache_clear", None)
        if callable(clear):
            clear()


# Env prefixes a policy reads its switches from; cleared while a profile resolves so a developer's shell (or a CI
# default) never decides a golden cell.
POLICY_ENV_PREFIXES = (
    "UNSLOTH_DIFFUSION_",
    "UNSLOTH_H3_",
    "UNSLOTH_VIDEO_",
    "UNSLOTH_IMAGE_",
    "UNSLOTH_SD_CPP_",
)
POLICY_ENV_NAMES = ("TORCHDYNAMO_DISABLE", "UNSLOTH_COMPILE_DISABLE")


@contextlib.contextmanager
def clean_policy_env() -> Iterator[None]:
    import os

    saved = {
        k: v
        for k, v in os.environ.items()
        if k.startswith(POLICY_ENV_PREFIXES) or k in POLICY_ENV_NAMES
    }
    for k in saved:
        os.environ.pop(k, None)
    try:
        yield
    finally:
        for k in [
            k for k in os.environ if k.startswith(POLICY_ENV_PREFIXES) or k in POLICY_ENV_NAMES
        ]:
            os.environ.pop(k, None)
        os.environ.update(saved)
