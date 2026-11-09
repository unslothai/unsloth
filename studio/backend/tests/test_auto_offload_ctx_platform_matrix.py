# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Raising the Auto offload context may change only the emitted context, not device placement."""

from __future__ import annotations

import dataclasses
import importlib.util
import inspect
import sys
from pathlib import Path
from typing import Optional

import pytest

# Reuse both harnesses by path; the tests dir is not a package.
_TESTS_DIR = Path(__file__).resolve().parent


def _load(module_name: str, file_name: str):
    spec = importlib.util.spec_from_file_location(module_name, _TESTS_DIR / file_name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_placement = _load("_placement_harness_auto_offload", "test_llama_cpp_placement.py")
_platforms = _load("_platform_harness_auto_offload", "test_llama_extra_args_platforms.py")

_backend = _placement._backend
_launch = _placement._launch
_apply_platform = _platforms._apply_platform
PLATFORMS = _platforms.PLATFORMS

from core.inference.llama_cpp import (  # noqa: E402
    _AUTO_OFFLOAD_CTX,
    _FIT_MIN_CTX,
    _IGPU_HOST_RESERVE_MIB,
    LlamaCppBackend,
    _apply_igpu_host_reserve_mib,
)
from utils.hardware import hardware as _hw  # noqa: E402

GIB = 1024**3
MIB = 1024**2


# Line tracer over load_model tells which placement arm actually ran.
# Anchors are found by source search and asserted unique, so renames fail loudly.

_LOAD_MODEL = inspect.unwrap(LlamaCppBackend.load_model)
_SOURCE_LINES, _SOURCE_FIRST = inspect.getsourcelines(_LOAD_MODEL)


def _anchor_lines(needle: str) -> list[int]:
    return [_SOURCE_FIRST + offset for offset, line in enumerate(_SOURCE_LINES) if needle in line]


def _one_anchor(needle: str) -> int:
    hits = _anchor_lines(needle)
    assert len(hits) == 1, f"anchor is no longer unique in load_model: {needle!r} -> {hits}"
    return hits[0]


_NATIVE_CTX_ANCHORS = _anchor_lines("native_ctx_for_cap = self._context_length or effective_ctx")
assert len(_NATIVE_CTX_ANCHORS) == 2, _NATIVE_CTX_ANCHORS

ARM_ANCHORS = {
    "tensor-parallel": _one_anchor("self._plan_tensor_parallel("),
    "measured-kv": min(_NATIVE_CTX_ANCHORS),
    "file-size-only": _one_anchor("Falling back to file-size-only GPU selection"),
    "apple-metal": _one_anchor("_apple_fit_budget_mib = int("),
}

SITE_A = _one_anchor("effective_ctx = min(_AUTO_OFFLOAD_CTX, effective_ctx)")
_AWARDS = _anchor_lines("gpu_indices = sorted(idx for idx, _ in subset)")
assert len(_AWARDS) == 2 and max(_AWARDS) > SITE_A, (_AWARDS, SITE_A)
SITE_A_AWARD = max(_AWARDS)
SITE_B = _one_anchor("if use_fit and not explicit_ctx:")


def _traced(call):
    """Run ``call`` and return ``(result, executed_line_numbers)`` for load_model."""
    hits: set[int] = set()

    def _local(frame, event, _arg):
        if event == "line":
            hits.add(frame.f_lineno)
        return _local

    def _global(frame, _event, _arg):
        return _local if frame.f_code is _LOAD_MODEL.__code__ else None

    previous = sys.gettrace()
    sys.settrace(_global)
    try:
        return call(), hits
    finally:
        sys.settrace(previous)


# APU HIP free is unusable (free == total), so system RAM caps it first.
HOST_AVAILABLE_MIB = 24_000
APU_RAW_FREE_MIB = 32_768
APU_FREE_MIB = _apply_igpu_host_reserve_mib(min(APU_RAW_FREE_MIB, HOST_AVAILABLE_MIB), True)
IGPU_RAW_FREE_MIB = 12_000
IGPU_FREE_MIB = _apply_igpu_host_reserve_mib(IGPU_RAW_FREE_MIB, True)


@dataclasses.dataclass(frozen = True)
class Accelerator:
    """total_mib of 0 marks a shared pool (iGPU or APU); placement reads it as no known headroom."""

    label: str
    vulkan: bool
    memory: tuple
    apple_budget_bytes: int = 0
    is_rocm: bool = False


ACCELERATORS = [
    Accelerator(label, vulkan, tuple(memory)) for label, vulkan, memory in _platforms.ACCELERATORS
] + [
    Accelerator("amd-rocm", False, ((0, 12_000, 16_000),), is_rocm = True),
    Accelerator("amd-apu", False, ((0, APU_FREE_MIB, 0),), is_rocm = True),
    Accelerator("vulkan-igpu", True, ((0, IGPU_FREE_MIB, 0),)),
    Accelerator("apple-metal", False, (), apple_budget_bytes = 16 * GIB),
]

MATRIX = [
    pytest.param(platform, accelerator, id = f"{platform[0]}-{accelerator.label}")
    for platform in PLATFORMS
    for accelerator in ACCELERATORS
]

# fits: subset loop awards residency; overflows: the only way to reach Site A.
FITS = 0.35
OVERFLOWS = 1.30
NATIVE_CTX = 131_072
KV_MIB_PER_CTX = 0.5


@dataclasses.dataclass(frozen = True)
class Outcome:
    arm: str
    site: Optional[str]
    ctx: Optional[int]
    fit: Optional[str]
    gpu_indices: Optional[tuple]
    awarded: bool

    def placement(self) -> tuple:
        """Everything the constant is forbidden to move."""
        return (self.arm, self.site, self.fit, self.gpu_indices, self.awarded)


def _flag(cmd, *names) -> Optional[str]:
    for index, token in enumerate(cmd):
        if token in names and index + 1 < len(cmd):
            return cmd[index + 1]
    return None


def _selected_devices(cmd, env) -> Optional[tuple]:
    """Devices as the child sees them: a visibility mask, or --device VulkanN; mask -1 is the CPU pin."""
    device = _flag(cmd, "--device", "-dev")
    if device:
        return tuple(
            int(name.strip().lower().removeprefix("vulkan"))
            for name in device.split(",")
            if name.strip()
        )
    for name in ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"):
        mask = env.get(name)
        if mask:
            return tuple(int(part) for part in mask.split(",") if part.strip())
    return None


def _subdir(tmp_path, name):
    path = tmp_path / name
    path.mkdir(parents = True, exist_ok = True)
    return path


def cell_backend(
    tmp_path,
    monkeypatch,
    platform,
    accelerator: Accelerator,
    *,
    model_fraction: float = OVERFLOWS,
    estimate_kv: bool = True,
    native_ctx: int = NATIVE_CTX,
):
    """A backend wearing one cell's platform and accelerator."""
    _apply_platform(monkeypatch, platform)
    # No inherited mask, so a pin in the child env is one this launch wrote.
    for name in ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"):
        monkeypatch.delenv(name, raising = False)
    monkeypatch.setattr(_hw, "IS_ROCM", accelerator.is_rocm, raising = False)
    monkeypatch.setattr(
        LlamaCppBackend,
        "_apple_metal_memory_budget_bytes",
        staticmethod(lambda: accelerator.apple_budget_bytes),
    )
    backend, gguf = _backend(tmp_path, vulkan = accelerator.vulkan, memory = list(accelerator.memory))

    free_mib = sum(row[1] for row in accelerator.memory) or 16_384
    model_bytes = int(model_fraction * free_mib * MIB)
    backend._get_gguf_size_bytes = lambda _path: model_bytes
    backend._read_gguf_metadata = lambda _path: setattr(backend, "_context_length", native_ctx)
    backend._can_estimate_kv = lambda: estimate_kv
    backend._estimate_kv_cache_bytes = lambda ctx, *_a, **_kw: int(ctx * KV_MIB_PER_CTX * MIB)
    backend._compute_buffer_ctx_bytes = lambda *_a, **_kw: 0
    backend._estimate_compute_buffer_bytes = lambda **_kw: 1
    return backend, gguf


def run_cell(tmp_path, monkeypatch, platform, accelerator: Accelerator, **kwargs) -> Outcome:
    """Drive one cell through the real ``load_model`` and report what it did."""
    load_kwargs = kwargs.pop("load_kwargs", {})
    backend, gguf = cell_backend(tmp_path, monkeypatch, platform, accelerator, **kwargs)
    result, hits = _traced(lambda: _launch(backend, gguf, n_ctx = 0, **load_kwargs))
    arms = [name for name, line in ARM_ANCHORS.items() if line in hits]
    assert len(arms) <= 1, f"more than one placement arm ran: {arms}"
    site = "A" if SITE_A in hits else ("B" if SITE_B in hits else None)
    ctx = _flag(result["cmd"], "-c", "--ctx-size")
    return Outcome(
        arm = arms[0] if arms else "none",
        site = site,
        ctx = int(ctx) if ctx is not None else None,
        fit = _flag(result["cmd"], "--fit"),
        gpu_indices = _selected_devices(result["cmd"], result["env"]),
        awarded = SITE_A_AWARD in hits,
    )


@pytest.mark.parametrize("platform,accelerator", MATRIX)
def test_an_overflowing_model_reaches_the_expected_arm_and_pins_nothing(
    tmp_path, monkeypatch, platform, accelerator
):
    """An overflowing model reaches Site A and must pin no device, leaving placement to --fit."""
    outcome = run_cell(tmp_path, monkeypatch, platform, accelerator)

    if accelerator.memory:
        assert outcome.arm == "measured-kv"
        assert outcome.site == "A"
        # The change buys context, never residency.
        assert outcome.awarded is False
        assert outcome.fit == "on"
        assert outcome.ctx == _AUTO_OFFLOAD_CTX
    elif accelerator.apple_budget_bytes:
        assert outcome.arm == "apple-metal"
        assert outcome.site is None
    else:
        assert outcome.arm == "none"
        assert outcome.site is None


@pytest.mark.parametrize("platform,accelerator", MATRIX)
def test_a_model_that_fits_never_reaches_either_site(tmp_path, monkeypatch, platform, accelerator):
    """G1. The far more common shape: the subset loop awards, so the constant is
    never read. Pinned devices and ``--fit off`` are the evidence the loop won."""
    outcome = run_cell(tmp_path, monkeypatch, platform, accelerator, model_fraction = FITS)

    assert outcome.site is None
    if accelerator.memory:
        assert outcome.arm == "measured-kv"
        assert outcome.gpu_indices == (accelerator.memory[0][0],)
        assert outcome.fit == "off"
        assert outcome.ctx is not None and outcome.ctx > _AUTO_OFFLOAD_CTX


@pytest.mark.parametrize("accelerator", ACCELERATORS, ids = [a.label for a in ACCELERATORS])
def test_wsl_is_indistinguishable_from_native_linux(tmp_path, monkeypatch, accelerator):
    """G5. The only WSL detector on this path is a loader-path decision, so the
    context math must not be able to tell the two apart on any accelerator."""
    linux = next(p for p in PLATFORMS if p[0] == "linux")
    wsl2 = next(p for p in PLATFORMS if p[0] == "wsl2")

    for fraction in (FITS, OVERFLOWS):
        on_linux = run_cell(
            _subdir(tmp_path, f"linux-{fraction}"),
            monkeypatch,
            linux,
            accelerator,
            model_fraction = fraction,
        )
        on_wsl = run_cell(
            _subdir(tmp_path, f"wsl-{fraction}"),
            monkeypatch,
            wsl2,
            accelerator,
            model_fraction = fraction,
        )
        assert on_wsl == on_linux


@pytest.mark.parametrize("platform", PLATFORMS, ids = [p[0] for p in PLATFORMS])
def test_the_file_size_only_arm_relabels_the_context_without_moving_a_device(
    tmp_path, monkeypatch, platform
):
    """Site B. Reached only without KV metadata, and inert with respect to
    placement by construction: ``_select_gpus`` has already returned above it and
    nothing below re-runs it. Only the number the UI is told changes."""
    accelerator = next(a for a in ACCELERATORS if a.label == "nvidia-single")
    outcome = run_cell(tmp_path, monkeypatch, platform, accelerator, estimate_kv = False)

    assert outcome.arm == "file-size-only"
    assert outcome.site == "B"
    assert outcome.gpu_indices is None
    assert outcome.fit == "on"
    assert outcome.ctx == _AUTO_OFFLOAD_CTX


@pytest.mark.parametrize("platform", PLATFORMS, ids = [p[0] for p in PLATFORMS])
def test_metal_auto_still_floors_at_the_fit_minimum(tmp_path, monkeypatch, platform):
    """Metal Auto must floor at _FIT_MIN_CTX, like discrete GPUs; no separate Metal literal."""
    metal = next(a for a in ACCELERATORS if a.label == "apple-metal")
    on_metal = run_cell(_subdir(tmp_path, "metal"), monkeypatch, platform, metal)

    assert on_metal.arm == "apple-metal"
    assert on_metal.ctx == _FIT_MIN_CTX

    discrete = next(a for a in ACCELERATORS if a.label == "nvidia-single")
    on_discrete = run_cell(_subdir(tmp_path, "discrete"), monkeypatch, platform, discrete)
    assert on_discrete.ctx == _AUTO_OFFLOAD_CTX
    assert on_discrete.ctx == on_metal.ctx


@pytest.mark.parametrize("platform", PLATFORMS, ids = [p[0] for p in PLATFORMS])
@pytest.mark.parametrize("gpu_layers", [-1, 8], ids = ["auto-layers", "explicit-layers"])
def test_manual_memory_mode_bypasses_both_sites_on_a_gpu_box(
    tmp_path, monkeypatch, platform, gpu_layers
):
    """G7. Two sites clear ``gpus`` before the chain, so a manual load takes no arm
    at all even with cards enumerated, and neither site can be reached."""
    accelerator = next(a for a in ACCELERATORS if a.label == "nvidia-multi")
    outcome = run_cell(
        tmp_path,
        monkeypatch,
        platform,
        accelerator,
        load_kwargs = {"gpu_memory_mode": "manual", "gpu_layers": gpu_layers},
    )

    assert outcome.arm == "none"
    assert outcome.site is None
    assert outcome.gpu_indices is None


@pytest.mark.parametrize("platform", PLATFORMS, ids = [p[0] for p in PLATFORMS])
def test_the_rocm_arch_gate_drops_an_amd_host_onto_the_cpu_path(tmp_path, monkeypatch, platform):
    """G8. Every present device gated out (#7624) empties ``_gpu_mem``, so an AMD
    box with real cards takes the same no-arm path a CPU-only box takes and never
    reaches either site."""
    accelerator = next(a for a in ACCELERATORS if a.label == "amd-rocm")
    _apply_platform(monkeypatch, platform)
    monkeypatch.setattr(_hw, "IS_ROCM", True, raising = False)
    monkeypatch.setattr(
        LlamaCppBackend, "_apple_metal_memory_budget_bytes", staticmethod(lambda: 0)
    )
    for name in ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"):
        monkeypatch.delenv(name, raising = False)
    backend, gguf = _backend(tmp_path, vulkan = False, memory = list(accelerator.memory))
    present = list(accelerator.memory)

    backend._get_gpu_memory = lambda _binary = None, for_llama_server = False, **_kw: (
        [] if for_llama_server else list(present)
    )
    backend._host_torch_is_rocm = lambda: True
    backend._installed_llama_gfx_archs = lambda _binary: frozenset({"gfx1030"})
    backend._rocm_arch_by_physical_id = lambda: {row[0]: "gfx1033" for row in present}
    backend._get_gguf_size_bytes = lambda _path: int(OVERFLOWS * 12_000 * MIB)
    backend._read_gguf_metadata = lambda _path: setattr(backend, "_context_length", NATIVE_CTX)
    backend._can_estimate_kv = lambda: True
    backend._estimate_kv_cache_bytes = lambda ctx, *_a, **_kw: int(ctx * KV_MIB_PER_CTX * MIB)
    backend._compute_buffer_ctx_bytes = lambda *_a, **_kw: 0
    backend._estimate_compute_buffer_bytes = lambda **_kw: 1

    result, hits = _traced(lambda: _launch(backend, gguf, n_ctx = 0))

    assert not [name for name, line in ARM_ANCHORS.items() if line in hits]
    assert SITE_A not in hits and SITE_B not in hits
    assert _selected_devices(result["cmd"], result["env"]) == (-1,)


def test_the_shared_memory_cells_carry_a_zero_total_and_a_reduced_free():
    """G9. Documents the numbers the two shared-pool cells above were built from,
    so a change to the reserve or to the APU cap shows up here rather than as a
    silently different matrix. Both reach the sites with ``total_mib == 0``."""
    apu = next(a for a in ACCELERATORS if a.label == "amd-apu")
    igpu = next(a for a in ACCELERATORS if a.label == "vulkan-igpu")

    assert apu.memory[0][2] == 0 and igpu.memory[0][2] == 0
    assert APU_FREE_MIB == HOST_AVAILABLE_MIB - _IGPU_HOST_RESERVE_MIB == 22_976
    assert IGPU_FREE_MIB == IGPU_RAW_FREE_MIB - _IGPU_HOST_RESERVE_MIB == 10_976
    assert _apply_igpu_host_reserve_mib(12_000, False) == 12_000


def _rocm_torch(free_mib: int, total_mib: int, reserved_mib: int):
    """The two readings ``trusted_mem_get_info`` consults, and nothing else."""
    import types

    torch = types.ModuleType("torch")
    torch.cuda = types.SimpleNamespace(
        mem_get_info = lambda _device = None: (free_mib * MIB, total_mib * MIB),
        memory_reserved = lambda _device = None: reserved_mib * MIB,
    )
    return torch


@pytest.mark.parametrize(
    "os_key,expected_free_mib",
    [("linux", 16_000), ("win32", 10_384)],
    ids = ["linux-rocm", "windows-rocm"],
)
def test_windows_rocm_feeds_a_smaller_free_reading_into_the_planner(
    monkeypatch, os_key, expected_free_mib
):
    """Windows ROCm caps free VRAM at total minus reserved, as WDDM reports the process budget."""
    monkeypatch.setitem(sys.modules, "torch", _rocm_torch(16_000, 16_384, 6_000))
    monkeypatch.setattr(_hw.sys, "platform", os_key)
    monkeypatch.setattr(_hw, "IS_ROCM", True, raising = False)

    assert _hw.rocm_windows_free_is_untrusted() is (os_key == "win32")
    free_bytes, total_bytes = _hw.trusted_mem_get_info(0)
    assert free_bytes // MIB == expected_free_mib
    assert total_bytes // MIB == 16_384


def test_the_windows_rocm_cap_is_what_pushes_a_load_into_the_fallback(tmp_path, monkeypatch):
    """Windows' smaller free reading pushes identical AMD hardware into the offload fallback."""
    windows = next(p for p in PLATFORMS if p[0] == "windows")
    linux = next(p for p in PLATFORMS if p[0] == "linux")
    model_mib = 10 * 1024

    def _run(platform, free_mib, subdir):
        accelerator = Accelerator("amd-rocm", False, ((0, free_mib, 16_384),), is_rocm = True)
        backend, gguf = cell_backend(
            _subdir(tmp_path, subdir), monkeypatch, platform, accelerator, model_fraction = 1.0
        )
        backend._get_gguf_size_bytes = lambda _path: model_mib * MIB
        result, hits = _traced(lambda: _launch(backend, gguf, n_ctx = 0))
        return _selected_devices(result["cmd"], result["env"]), SITE_A in hits

    on_linux = _run(linux, 16_000, "linux")
    on_windows = _run(windows, 10_384, "windows")

    assert on_linux == ((0,), False)
    assert on_windows == (None, True)
