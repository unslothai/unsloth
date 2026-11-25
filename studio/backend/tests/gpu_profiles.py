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
import functools
import dataclasses
import sys
import types
from collections import namedtuple
from typing import Any, Callable, Iterator, Optional

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

    def defer(self, fn: Any) -> None:
        """Run ``fn()`` at undo time (in reverse order with the restores)."""
        self._undo.append((fn, ("__call__",), None))

    def setitem(self, mapping: Any, key: Any, value: Any) -> None:
        self._undo.append((mapping, ("__item__", key), mapping.get(key, self._MISSING)))
        mapping[key] = value

    def undo(self) -> None:
        for obj, name, old in reversed(self._undo):
            if name == ("__call__",):
                try:
                    obj()
                except Exception:  # noqa: BLE001 - a cache clear never fails the restore
                    pass
                continue
            if isinstance(name, tuple) and name[:1] == ("__item__",):
                if old is self._MISSING:
                    obj.pop(name[1], None)
                else:
                    obj[name[1]] = old
                continue
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
        # Triton present = the INT8 GEMM kernels build (they compile lazily at the first launch)
        (
            "core.inference.diffusion_int8_gemm",
            "_kernels",
            lambda: types.SimpleNamespace() if has else None,
        ),
        ("core.inference.diffusion_convrot_quant", "_triton_ok", lambda: has),
        (
            "core.inference.diffusion_grouped_int8_gemm",
            "_triton_version",
            lambda: PINNED_TRITON_VERSION if has else None,
        ),
    ]


# The host environment every profile declares, whatever the host running the test has: Linux (macOS for MPS), torch
# with allow_fp16_accumulation (>= 2.7), torchao installed, Triton per profile, the prebuilt CUTLASS INT8 GEMM wheel
# on NVIDIA, a CUDA 13.0 NVRTC (targets sm75 .. sm121, not compute_130), no Intel XPU. test_the_matrix_is_host_
# independent flips each of these on the host and requires the identical matrix.
DECLARED_NVRTC_ARCHS = ((7, 5), (12, 1))


class _DeclaredCudaDriver:
    """The CUDA driver (libcuda) an NVIDIA profile declares, for code that only asks whether it loads (the model-offload
    graphs' stable device range needs its virtual-memory API). Any real driver call fails loudly: the guard never
    reserves memory on the host's card."""

    def __getattr__(self, name: str) -> Any:
        raise RuntimeError(f"the lever guard does not call the CUDA driver ({name})")


def _pin_cuda_driver(put: Callable, p: GpuProfile) -> None:
    """NVIDIA profiles have a CUDA driver with virtual memory management; ROCm, MPS and CPU have none, whatever the
    host has (a CPU CI runner has no libcuda, a GPU host does)."""

    def no_driver() -> Any:
        raise OSError("CUDA driver library not found")

    try:
        from core.inference import diffusion_stable_arena as arena
    except Exception:  # noqa: BLE001 - nothing reads the driver then
        return
    if hasattr(arena, "_driver"):
        driver = _DeclaredCudaDriver()
        put(arena, "_driver", (lambda: driver) if p.vendor == "nvidia" else no_driver)


def _pin_metal(put: Callable, p: GpuProfile) -> None:
    """Every real Metal / MPS reading Studio makes, pinned to the profile: the MPS profile declares macOS 14+ (bf16
    works) with its unified memory as Metal's working set; every other profile has no MPS, as on Linux. A Mac host's
    own device is never touched (on a paravirtual runner the first MPS op crashes the process)."""
    import torch

    def no_mps(*_a: Any, **_k: Any) -> Any:
        raise RuntimeError("MPS backend is not available")

    working_set = int(p.vram_gb * GIB) if p.mps else 0
    mps = getattr(torch, "mps", None)
    if mps is not None:
        for name, value in (
            ("recommended_max_memory", (lambda: working_set) if p.mps else no_mps),
            ("driver_allocated_memory", (lambda: 0) if p.mps else no_mps),
            ("current_allocated_memory", (lambda: 0) if p.mps else no_mps),
            ("synchronize", (lambda: None) if p.mps else no_mps),
            ("empty_cache", lambda: None),
            ("set_per_process_memory_fraction", lambda fraction: None),
        ):
            if hasattr(mps, name):
                put(mps, name, value)
    for mod_name, attr, value in (
        # Studio's on-device bf16 probe (a real MPS op): macOS 14+ on the MPS profile
        (
            "core.inference.diffusion_device",
            "_mps_supports_bfloat16",
            lambda torch_mod: bool(p.mps),
        ),
        # the VAE's Metal sync (a real MPS tensor round trip)
        ("core.inference.diffusion_device", "mps_synchronize", lambda: None),
        # MLX / torch.mps / Metal.framework reading of the working set
        (
            "utils.hardware.hardware",
            "_metal_recommended_working_set_bytes",
            lambda *, allow_import = True: working_set,
        ),
    ):
        try:
            mod = __import__(mod_name, fromlist = [attr])
        except Exception:  # noqa: BLE001 - a module that does not import here reads nothing
            continue
        if hasattr(mod, attr):
            put(mod, attr, value)


def declared_os(p: GpuProfile) -> str:
    return "darwin" if p.mps else "linux"


def _wrap(old: Any, new: Any) -> Any:
    """``new`` carrying ``old``'s metadata and ``__wrapped__`` (Dynamo's import reads it off torch functions)."""
    if callable(old):
        try:
            functools.update_wrapper(new, old)
        except Exception:  # noqa: BLE001 - a builtin without the attributes: keep new as is
            pass
    return new


class _MatmulWithFp16Accum:
    """``torch.backends.cuda.matmul`` for a torch older than ``allow_fp16_accumulation`` (2.7): the flag exists."""

    def __init__(self, inner: Any) -> None:
        object.__setattr__(self, "_inner", inner)
        object.__setattr__(self, "allow_fp16_accumulation", False)

    def __getattr__(self, name: str) -> Any:
        return getattr(object.__getattribute__(self, "_inner"), name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "allow_fp16_accumulation":
            object.__setattr__(self, name, value)
        else:
            setattr(object.__getattribute__(self, "_inner"), name, value)


def _ensure_importable(patch: Any, name: str) -> None:
    """``import <name>`` succeeds in the mock scope (a bare package stand-in where the host lacks it or it fails)."""
    try:
        __import__(name)
        return
    except Exception:  # noqa: BLE001 - absent, or broken on this torch
        pass
    stub = types.ModuleType(name)
    stub.__path__ = []  # a package: its submodules still fail to import, as on a broken install
    stub.__version__ = "0.0+profile-stand-in"
    patch.setitem(sys.modules, name, stub)


def _device_index(device: Any) -> int:
    if device is None:
        return 0
    if isinstance(device, int):
        return device
    if isinstance(device, str):
        return int(device.split(":", 1)[1]) if ":" in device else 0
    index = getattr(device, "index", None)  # torch.device
    return int(index) if isinstance(index, int) else 0


@contextlib.contextmanager
def mock_profile(
    p: GpuProfile,
    *,
    device_count: int = 1,
    ordinals: Optional[dict] = None,
    extra: Any = None,
) -> Iterator[GpuProfile]:
    """Make torch (and Studio's host probes) report ``p`` for the duration.

    ``ordinals`` maps a device index to its own profile for mixed-card hosts (``{1: PROFILES["h100"]}``); every
    other index is ``p``. ``extra`` is an optional callable ``extra(patch, p)`` for test-specific patches applied
    inside the same undo scope. Every patch is undone on exit, also on error.
    """
    import torch

    try:
        import torch._dynamo  # noqa: F401 - imported before any torch function is replaced (it inspects them)
    except Exception:  # noqa: BLE001
        pass
    patch = _Patch()
    cards = {0: p, **(ordinals or {})}
    count = max(device_count, max(cards) + 1)
    cuda = p.cuda

    def card(device: Any = None) -> GpuProfile:
        if not cuda:
            raise RuntimeError("Found no NVIDIA driver on your system")
        return cards.get(_device_index(device), p)

    def put(obj: Any, name: str, value: Any) -> None:
        patch.setattr(obj, name, _wrap(getattr(obj, name, None), value))

    try:
        put(torch.cuda, "is_available", lambda: cuda)
        put(torch.cuda, "device_count", lambda: count if cuda else 0)
        put(torch.cuda, "current_device", lambda: 0)
        put(torch.cuda, "is_initialized", lambda: cuda)
        # nothing in the guard records a CUDA graph (a CPU build raises on the real query)
        if hasattr(torch.cuda, "is_current_stream_capturing"):
            put(torch.cuda, "is_current_stream_capturing", lambda: False)

        def no_device(*a: Any, **k: Any) -> None:
            # Anything that really touches the card (a tensor moved to it, an on-device probe) fails at once, as it
            # does on CPU torch: a CUDA build on a GPU-less host would otherwise spend seconds probing the driver.
            raise RuntimeError(f"profile stand-in {p.key}: no real device to run on")

        put(torch.cuda, "_lazy_init", no_device)
        put(torch.cuda, "get_device_capability", lambda device = None: tuple(card(device).capability))
        put(torch.cuda, "get_device_name", lambda device = None: card(device).name)
        put(
            torch.cuda, "get_device_properties", lambda device = None: device_properties(card(device))
        )

        def mem_get_info(device: Any = None) -> tuple:
            q = card(device)
            return free_bytes(q), int(q.vram_gb * GIB)

        def bf16(including_emulation: bool = True) -> bool:
            return bool(cuda and (p.rocm or (p.capability or (0, 0)) >= (8, 0)))

        put(torch.cuda, "mem_get_info", mem_get_info)
        put(torch.cuda, "is_bf16_supported", bf16)
        patch.setattr(torch.version, "hip", "6.4.43482" if p.rocm else None)
        patch.setattr(torch.version, "cuda", "12.8" if p.vendor == "nvidia" else None)
        mps = getattr(torch.backends, "mps", None)
        if mps is not None:
            put(mps, "is_available", lambda: p.mps)
            put(mps, "is_built", lambda: p.mps)
        _pin_metal(put, p)
        _pin_cuda_driver(put, p)
        xpu = getattr(torch, "xpu", None)
        if (
            xpu is not None
        ):  # no profile is an Intel XPU: a host one must not turn CPU / MPS rows into xpu
            put(xpu, "is_available", lambda: False)
            put(xpu, "device_count", lambda: 0)
        # Declared OS (Windows hosts read sys.platform for the compile runtime, Triton and more).
        patch.setattr(sys, "platform", declared_os(p))
        # torch >= 2.7: allow_fp16_accumulation exists (Backend CI's floor, 2.4, lacks it).
        matmul = torch.backends.cuda.matmul
        if not hasattr(matmul, "allow_fp16_accumulation"):
            patch.setattr(torch.backends.cuda, "matmul", _MatmulWithFp16Accum(matmul))
        # torch >= 2.7: the e8m0 dtype MXFP8 needs exists (read by hasattr before any capability gate).
        if not hasattr(torch, "float8_e8m0fnu"):
            patch.setattr(torch, "float8_e8m0fnu", getattr(torch, "float8_e4m3fn", torch.uint8))
        # torchao installed (it fails to import on torch < 2.11 for torchao 0.18, and is absent on some hosts).
        _ensure_importable(patch, "torchao")
        for mod_name, attr, value in (
            ("core.inference.diffusion_precision", "torchao_quantize_importable", lambda: True),
            (
                "core.inference.diffusion_transformer_quant",
                "torchao_unavailable_reason",
                lambda: None,
            ),
            ("core.inference.prequant_safetensors", "safetensors_prequant_supported", lambda: True),
        ):
            try:
                mod = __import__(mod_name, fromlist = [attr])
                if hasattr(mod, attr):
                    put(mod, attr, value)
            except Exception:  # noqa: BLE001
                pass
        try:
            from core import _torchao_stub
            real_is_stubbed = _torchao_stub.is_stubbed
            put(
                _torchao_stub,
                "is_stubbed",
                lambda package: False if package == "torchao" else real_is_stubbed(package),
            )
        except Exception:  # noqa: BLE001
            pass
        try:
            # The real compile-runtime decision, read under the declared OS (sys.platform is pinned above): its
            # lru cache is dropped on entry and exit, so neither the host's answer nor this profile's leaks.
            from core.inference import diffusion_speed

            gate = diffusion_speed.torch_compile_runtime_available
            clear = getattr(gate, "cache_clear", None)
            if clear is not None:
                clear()
                patch.defer(clear)
        except Exception:  # noqa: BLE001
            pass
        try:
            import psutil

            total = int(p.host_ram_gb * GIB)
            avail = int(total * 0.85)
            vm = _VirtualMemory(total, avail, 15.0, total - avail, avail, 0, 0, 0, 0, 0, 0)
            patch.setattr(psutil, "virtual_memory", lambda: vm)
        except Exception:  # noqa: BLE001 - psutil-less hosts: nothing reads it then
            pass
        for mod_name, attr, value in _triton_overrides(p) + _toolchain_overrides(p):
            mod = sys.modules.get(mod_name)
            if mod is None:
                try:
                    mod = __import__(mod_name, fromlist = [attr])
                except Exception:  # noqa: BLE001 - a module that does not import here has no probe to pin
                    continue
            if hasattr(mod, attr):
                put(mod, attr, value)
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


def _toolchain_overrides(p: GpuProfile) -> list:
    """The declared CUDA toolchain (``DECLARED_NVRTC_ARCHS``), for the INT8 attention fallback's NVRTC gate."""
    lo, hi = DECLARED_NVRTC_ARCHS

    def nvrtc_arch_supported(capability: Any) -> bool:
        try:
            cap = (int(capability[0]), int(capability[1]))
        except Exception:  # noqa: BLE001
            return False
        return lo <= cap <= hi

    return [
        (
            "core.inference.diffusion_int8_attention_cuda",
            "nvrtc_arch_supported",
            nvrtc_arch_supported,
        )
    ]


# lru caches keyed by device index: one profile's answer must not leak into the next.
_DEVICE_CACHES = (
    ("core.inference.diffusion_vae_strips", "_card_strip_elems"),
    ("core.inference.diffusion_vae_strips", "_card_cap"),
)
# Per-device fact caches with a reset hook.
_DEVICE_RESETS = (("core.inference.diffusion_int8_attention", "reset_device_facts"),)


def _clear_device_caches() -> None:
    for mod_name, attr in _DEVICE_CACHES:
        fn = getattr(sys.modules.get(mod_name), attr, None)
        clear = getattr(fn, "cache_clear", None)
        if callable(clear):
            clear()
    for mod_name, attr in _DEVICE_RESETS:
        fn = getattr(sys.modules.get(mod_name), attr, None)
        if callable(fn):
            fn()


# Env prefixes a policy reads its switches from; cleared while a profile resolves so a developer's shell (or a CI
# default) never decides a golden cell.
# Every Studio lever, opt-out and kill switch is an UNSLOTH_* variable (plus torch's compile switches): all of them
# are cleared for a profile, so an operator's shell (UNSLOTH_DISABLE_CUDA_GRAPH=1, UNSLOTH_STUDIO_FLUX3_FP16=1, ...)
# never moves a cell. Only the guard's own controls and import plumbing pass.
POLICY_ENV_PREFIXES = ("UNSLOTH_",)
POLICY_ENV_NAMES = ("TORCHDYNAMO_DISABLE", "TORCH_COMPILE_DISABLE")
POLICY_ENV_KEEP = (
    "UNSLOTH_UPDATE_GPU_GOLDEN",
    "UNSLOTH_GPU_GOLDEN_REASON",
    "UNSLOTH_IS_PRESENT",
    "UNSLOTH_COMPILE_LOCATION",
)


def _policy_env(name: str) -> bool:
    return (
        name.startswith(POLICY_ENV_PREFIXES) or name in POLICY_ENV_NAMES
    ) and name not in POLICY_ENV_KEEP


# The matrix child starts from an allowlist, not the parent's environment: process plumbing only (paths, locale,
# temp dirs, the venv, the Windows system variables Python needs), so a setting read at import time (FLUX.3's fp16
# opt-in) cannot reach the golden either.
CHILD_ENV_ALLOW = (
    "PATH",
    "PATHEXT",
    "HOME",
    "USER",
    "LOGNAME",
    "USERNAME",
    "SHELL",
    "TERM",
    "LANG",
    "LANGUAGE",
    "TMPDIR",
    "TMP",
    "TEMP",
    "PYTHONPATH",
    "PYTHONHOME",
    "PYTHONIOENCODING",
    "PYTHONUTF8",
    "VIRTUAL_ENV",
    "CONDA_PREFIX",
    "LD_LIBRARY_PATH",
    "DYLD_LIBRARY_PATH",
    "DYLD_FALLBACK_LIBRARY_PATH",
    "SYSTEMROOT",
    "SYSTEMDRIVE",
    "WINDIR",
    "COMSPEC",
    "APPDATA",
    "LOCALAPPDATA",
    "USERPROFILE",
    "PROGRAMDATA",
    "PROGRAMFILES",
    "PROGRAMFILES(X86)",
    "NUMBER_OF_PROCESSORS",
    "PROCESSOR_ARCHITECTURE",
    "HF_HOME",
    "HF_HUB_CACHE",
    "HF_HUB_OFFLINE",
    "XDG_CACHE_HOME",
    "TRITON_CACHE_DIR",
    "TORCHINDUCTOR_CACHE_DIR",
    "TORCH_EXTENSIONS_DIR",
    "CUDA_CACHE_PATH",
    "PYTHONUNBUFFERED",
    # the host's device masks only limit what the real driver enumerates (every profile mocks its own devices); without
    # them a many-GPU host initialises every card
    "CUDA_VISIBLE_DEVICES",
    "HIP_VISIBLE_DEVICES",
    "ROCR_VISIBLE_DEVICES",
)
CHILD_ENV_ALLOW_PREFIXES = ("LC_",)


def child_env_base(environ: Any = None) -> dict:
    """The allowlisted part of ``environ`` (default ``os.environ``) a matrix child inherits."""
    import os

    environ = os.environ if environ is None else environ
    allow = {n.upper() for n in CHILD_ENV_ALLOW}
    return {
        k: v
        for k, v in environ.items()
        if k.upper() in allow or k.upper().startswith(CHILD_ENV_ALLOW_PREFIXES)
    }


@contextlib.contextmanager
def clean_policy_env() -> Iterator[None]:
    import os

    saved = {k: v for k, v in os.environ.items() if _policy_env(k)}
    for k in saved:
        os.environ.pop(k, None)
    try:
        yield
    finally:
        for k in [k for k in os.environ if _policy_env(k)]:
            os.environ.pop(k, None)
        os.environ.update(saved)
