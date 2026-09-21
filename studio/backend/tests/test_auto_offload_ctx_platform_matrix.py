# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""[OS x GPU vendor] for the Auto host-offload context (``_AUTO_OFFLOAD_CTX``).

The failure mode guarded: raising the context Auto settles on when no discrete-GPU
subset can hold the model (4096 -> 8192) is a change to a value that is read inside
the *placement* decision, so it could move which devices a load pins, whether
``--fit`` owns placement, or which arm of ``load_model``'s placement chain runs at
all. It must not. The only thing allowed to differ is the emitted context, and only
on the two arms that reach one of the two sites.

The chain under test, in source order:

  1. tensor-parallel        -- ``_plan_tensor_parallel`` owns everything
  2. measured-KV            -- gpus + ``_can_estimate_kv()``; holds SITE A, the
                               subset loop's ``else:`` plus its residency re-check
  3. file-size-only         -- gpus, no KV metadata; holds SITE B, which only
                               relabels the context ``--fit`` was already given
  4. Apple unified memory   -- no gpus, a Metal budget; floors at _FIT_MIN_CTX
                               like every other arm, since its four hardcoded
                               4096s were replaced by the constant
  5. (no arm)               -- no gpus and no Metal budget: CPU, no context math

Each cell records the arm taken, the emitted context, ``--fit`` and the pinned
device list. The arm is read from a line tracer over ``load_model`` rather than
inferred from the output, because "the arm ran and changed nothing" and "the arm
never ran" produce the same argv on most cells.

Simulation notice: this suite runs on one host. Only Linux/NVIDIA is native.
Windows, WSL2 and macOS are ``sys.platform`` / ``platform.release`` monkeypatches
via the shared ``_apply_platform`` seam, Metal is a non-zero
``_apple_metal_memory_budget_bytes`` with an empty GPU probe, and every AMD cell is
a memory shape plus ``utils.hardware.IS_ROCM``. No ROCm runtime, no Metal device
and no Windows kernel is exercised. The authoritative signal for those remains the
per-OS CI matrix on real runners; this is the branch coverage one host can give.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import inspect
import sys
from pathlib import Path
from typing import Optional

import pytest

# Both harnesses already exist; by path, because the tests dir is not a package.
# test_llama_cpp_placement owns the module stubs, the fake GGUF, the fake GPU probe
# and the captured Popen; test_llama_extra_args_platforms owns the OS seam and the
# accelerator list this file extends rather than replaces.
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


# ── the arm tracer ───────────────────────────────────────────────────────────
#
# load_model is one function with the whole placement chain inside it, so the arm
# taken is not observable from the outside: an arm that ran and left placement
# alone emits the same argv as one that was skipped. A line tracer over that single
# code object is the cheapest thing that can tell them apart.
#
# Anchors are resolved by source search at import time, not by hardcoded line
# numbers, and every one of them is asserted unique-or-known-count below. A rename
# that moves an anchor fails loudly with the anchor named, which is the intended
# behaviour: the table this file prints would otherwise silently start describing
# the wrong branch.

_LOAD_MODEL = inspect.unwrap(LlamaCppBackend.load_model)
_SOURCE_LINES, _SOURCE_FIRST = inspect.getsourcelines(_LOAD_MODEL)


def _anchor_lines(needle: str) -> list[int]:
    return [_SOURCE_FIRST + offset for offset, line in enumerate(_SOURCE_LINES) if needle in line]


def _one_anchor(needle: str) -> int:
    hits = _anchor_lines(needle)
    assert len(hits) == 1, f"anchor is no longer unique in load_model: {needle!r} -> {hits}"
    return hits[0]


# Both the measured-KV arm and the Apple arm open by re-reading the native context,
# so one search yields both, ordered by position in the chain.
_NATIVE_CTX_ANCHORS = _anchor_lines("native_ctx_for_cap = self._context_length or effective_ctx")
assert len(_NATIVE_CTX_ANCHORS) == 2, _NATIVE_CTX_ANCHORS

# First executable statement of each arm's body, so a hit means the arm was TAKEN.
# An `elif` header line would only mean its condition was evaluated.
ARM_ANCHORS = {
    "tensor-parallel": _one_anchor("self._plan_tensor_parallel("),
    "measured-kv": min(_NATIVE_CTX_ANCHORS),
    "file-size-only": _one_anchor("Falling back to file-size-only GPU selection"),
    "apple-metal": _one_anchor("_apple_fit_budget_mib = int("),
}

# Site A: the measured-KV subset loop's `else:`, which lowers the context and then
# re-checks whether any subset can hold the model at that lower context.
SITE_A = _one_anchor("effective_ctx = min(_AUTO_OFFLOAD_CTX, effective_ctx)")
# The line the re-check awards residency on. Two subset loops assign it; the award
# is the one after Site A.
_AWARDS = _anchor_lines("gpu_indices = sorted(idx for idx, _ in subset)")
assert len(_AWARDS) == 2 and max(_AWARDS) > SITE_A, (_AWARDS, SITE_A)
SITE_A_AWARD = max(_AWARDS)
# Site B: the file-size-only arm's relabel. Placement was already decided by
# _select_gpus on the line above, so nothing downstream of this can move a device.
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
        # Restored unconditionally: leaking a trace function would slow, and under
        # a coverage run silently displace, every test that follows this one.
        sys.settrace(previous)


# ── the matrix ───────────────────────────────────────────────────────────────

# Both shared-memory cells report their pool through the same helper the real
# probes use, so the numbers in the table are the product's own arithmetic rather
# than invented ones. An APU's HIP free reading is unusable (Windows reports
# free == total, #7072), so system RAM caps it first; a Vulkan iGPU's reading is
# already host RAM, so only the reserve applies.
HOST_AVAILABLE_MIB = 24_000
APU_RAW_FREE_MIB = 32_768
APU_FREE_MIB = _apply_igpu_host_reserve_mib(min(APU_RAW_FREE_MIB, HOST_AVAILABLE_MIB), True)
IGPU_RAW_FREE_MIB = 12_000
IGPU_FREE_MIB = _apply_igpu_host_reserve_mib(IGPU_RAW_FREE_MIB, True)


@dataclasses.dataclass(frozen = True)
class Accelerator:
    """One column of the matrix.

    ``memory`` rows are ``(index, free_mib, total_mib)`` exactly as
    ``_get_gpu_memory`` answers. A ``total_mib`` of 0 is the shared-pool marker both
    the Vulkan iGPU and the ROCm APU paths emit, because that "total" would be
    system RAM; the placement math reads it as "no absolute headroom known".
    """

    label: str
    vulkan: bool
    memory: tuple
    apple_budget_bytes: int = 0
    is_rocm: bool = False


# The four the extra-args matrix already ships, kept byte-identical so both suites
# describe the same hardware, plus the four this change makes worth separating.
ACCELERATORS = [
    Accelerator(label, vulkan, tuple(memory)) for label, vulkan, memory in _platforms.ACCELERATORS
] + [
    # ROCm without Vulkan: the torch fallback probe, not amd-smi, and the only
    # vendor for which sys.platform changes the free VRAM figure (see G6 below).
    Accelerator("amd-rocm", False, ((0, 12_000, 16_000),), is_rocm = True),
    # Unified-memory APU: no dedicated VRAM at all.
    Accelerator("amd-apu", False, ((0, APU_FREE_MIB, 0),), is_rocm = True),
    Accelerator("vulkan-igpu", True, ((0, IGPU_FREE_MIB, 0),)),
    # Apple Silicon: nothing enumerates, and the Metal budget is the only signal.
    Accelerator("apple-metal", False, (), apple_budget_bytes = 16 * GIB),
]

MATRIX = [
    pytest.param(platform, accelerator, id = f"{platform[0]}-{accelerator.label}")
    for platform in PLATFORMS
    for accelerator in ACCELERATORS
]

# What every cell loads. "fits" is small enough for the cell's own pool, so the
# subset loop awards residency and no site is reached; "overflows" is large enough
# that nothing holds it, which is the only way to reach Site A.
FITS = 0.35
OVERFLOWS = 1.30
NATIVE_CTX = 131_072
# KV linear in the context at 0.5 MiB per token, so the fit has something to price
# and a lower context genuinely buys room. Any monotone function would do.
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
    """The devices placement actually chose, as the child sees them.

    ``backend.gpu_ids`` only carries an explicit user pick, so an automatic
    selection is invisible there. What placement emits instead is a visibility
    mask (CUDA / HIP / ROCR) or, on a Vulkan build, ``--device VulkanN``. A mask of
    -1 is the deliberate CPU pin. Note that a Vulkan build names its devices even
    when ``--fit`` owns placement, so this is the device set the child may use, not
    proof that residency was awarded; ``Outcome.awarded`` is that proof.
    """
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
    # No inherited visibility mask: a mask in the child env has to be one THIS
    # launch wrote, or "placement pinned nothing" reads as a pin on any developer
    # box that exports CUDA_VISIBLE_DEVICES.
    for name in ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"):
        monkeypatch.delenv(name, raising = False)
    monkeypatch.setattr(_hw, "IS_ROCM", accelerator.is_rocm, raising = False)
    monkeypatch.setattr(
        LlamaCppBackend,
        "_apple_metal_memory_budget_bytes",
        staticmethod(lambda: accelerator.apple_budget_bytes),
    )
    backend, gguf = _backend(tmp_path, vulkan = accelerator.vulkan, memory = list(accelerator.memory))

    # Sized against the cell's own reported free memory so "fits" and "overflows"
    # mean the same thing on a 12 GB card and on a 24 GB shared pool. With no GPU
    # the size decides nothing, so a fixed 16 GB stands in.
    free_mib = sum(row[1] for row in accelerator.memory) or 16_384
    model_bytes = int(model_fraction * free_mib * MIB)
    backend._get_gguf_size_bytes = lambda _path: model_bytes
    backend._read_gguf_metadata = lambda _path: setattr(backend, "_context_length", native_ctx)
    backend._can_estimate_kv = lambda: estimate_kv
    backend._estimate_kv_cache_bytes = lambda ctx, *_a, **_kw: int(ctx * KV_MIB_PER_CTX * MIB)
    # Held at zero so the context that comes out is the KV decision alone; the
    # compute buffer has its own suite (test_compute_buffer.py).
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


# ── G1 / G4 / G5: the arm and the placement on every cell ────────────────────


@pytest.mark.parametrize("platform,accelerator", MATRIX)
def test_an_overflowing_model_reaches_the_expected_arm_and_pins_nothing(
    tmp_path, monkeypatch, platform, accelerator
):
    """G1. A model no subset can hold: the placement half of the answer.

    This is the only shape that reaches Site A, so it is the shape where the
    constant could do damage. On every cell the answer is the same one the 4096
    fallback gave: no device is pinned and ``--fit`` owns placement.
    """
    outcome = run_cell(tmp_path, monkeypatch, platform, accelerator)

    if accelerator.memory:
        assert outcome.arm == "measured-kv"
        assert outcome.site == "A"
        # The whole claim of the change: it buys context, never residency. The
        # re-check under Site A is the only code that could award any, and it does
        # not fire on a single cell of this matrix.
        assert outcome.awarded is False
        assert outcome.fit == "on"
        assert outcome.ctx == _AUTO_OFFLOAD_CTX
    elif accelerator.apple_budget_bytes:
        # G3: Metal is a mutually exclusive arm and never sees the constant.
        assert outcome.arm == "apple-metal"
        assert outcome.site is None
    else:
        # G4: no device and no Metal budget falls off the end of the chain.
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
        # A fitted context, not the fallback: the two must not be confusable.
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


# ── G3: the Metal arm, measured against the discrete one ─────────────────────


@pytest.mark.parametrize("platform", PLATFORMS, ids = [p[0] for p in PLATFORMS])
def test_metal_auto_still_floors_at_the_fit_minimum(tmp_path, monkeypatch, platform):
    """G3. Pins the CURRENT Metal behaviour so it cannot drift silently.

    This used to measure an ASYMMETRY: the Apple arm held four hardcoded 4096s while
    a discrete GPU got ``_AUTO_OFFLOAD_CTX``, so the same offloading model published
    half the context on a Mac. Those four sites now read ``_FIT_MIN_CTX``, the
    asymmetry is gone by design, and what is worth pinning is the symmetry -- the two
    arms agreeing is the property that regresses if anyone re-introduces a separate
    Metal literal.

    The equality is asserted against the CONSTANTS on both sides rather than against
    8192, so the pair keeps agreeing through the next floor move instead of failing
    here. The old form asserted ``on_discrete.ctx == 2 * on_metal.ctx``, which was a
    correct reading of the ratio at the time and is exactly the shape that turns a
    deliberate change into a puzzling failure.

    Parametrised over every platform on purpose: the arm is gated on a non-zero
    Metal budget, not on ``sys.platform``, and that is worth having on record.
    """
    metal = next(a for a in ACCELERATORS if a.label == "apple-metal")
    on_metal = run_cell(_subdir(tmp_path, "metal"), monkeypatch, platform, metal)

    assert on_metal.arm == "apple-metal"
    assert on_metal.ctx == _FIT_MIN_CTX

    discrete = next(a for a in ACCELERATORS if a.label == "nvidia-single")
    on_discrete = run_cell(_subdir(tmp_path, "discrete"), monkeypatch, platform, discrete)
    assert on_discrete.ctx == _AUTO_OFFLOAD_CTX
    # Same model, same published context, whichever kind of memory it lands in.
    assert on_discrete.ctx == on_metal.ctx


# ── G7: manual memory mode ───────────────────────────────────────────────────


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


# ── G8: the ROCm arch gate ───────────────────────────────────────────────────


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

    # The gate's own shape: the llama-server probe returns nothing, the ungated one
    # still sees the cards, and no installed arch covers them.
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
    # -1 is the CPU mask the gate writes so the child cannot enumerate the cards
    # it has no kernels for. Placement chose CPU, not a device.
    assert _selected_devices(result["cmd"], result["env"]) == (-1,)


# ── G9: the two shared-memory pools that are not Metal ───────────────────────


def test_the_shared_memory_cells_carry_a_zero_total_and_a_reduced_free():
    """G9. Documents the numbers the two shared-pool cells above were built from,
    so a change to the reserve or to the APU cap shows up here rather than as a
    silently different matrix. Both reach the sites with ``total_mib == 0``."""
    apu = next(a for a in ACCELERATORS if a.label == "amd-apu")
    igpu = next(a for a in ACCELERATORS if a.label == "vulkan-igpu")

    assert apu.memory[0][2] == 0 and igpu.memory[0][2] == 0
    # The APU's HIP reading is capped by system RAM before the reserve is taken.
    assert APU_FREE_MIB == HOST_AVAILABLE_MIB - _IGPU_HOST_RESERVE_MIB == 22_976
    # The iGPU's reading is already host RAM, so only the reserve applies.
    assert IGPU_FREE_MIB == IGPU_RAW_FREE_MIB - _IGPU_HOST_RESERVE_MIB == 10_976
    # A discrete card is never touched by the reserve.
    assert _apply_igpu_host_reserve_mib(12_000, False) == 12_000


# ── G6: Windows ROCm reaches the fallback more often than Linux ROCm ─────────


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
    """G6, first half: identical hardware, two different numbers.

    ``rocm_windows_free_is_untrusted`` is True only for win32 + IS_ROCM, and
    ``trusted_mem_get_info`` then caps free at ``total - reserved``. WDDM
    virtualises video memory, so the driver's 16000 is the process's own budget
    rather than the card's residency; the cap is the only ceiling there is.
    """
    monkeypatch.setitem(sys.modules, "torch", _rocm_torch(16_000, 16_384, 6_000))
    monkeypatch.setattr(_hw.sys, "platform", os_key)
    monkeypatch.setattr(_hw, "IS_ROCM", True, raising = False)

    assert _hw.rocm_windows_free_is_untrusted() is (os_key == "win32")
    free_bytes, total_bytes = _hw.trusted_mem_get_info(0)
    assert free_bytes // MIB == expected_free_mib
    assert total_bytes // MIB == 16_384


def test_the_windows_rocm_cap_is_what_pushes_a_load_into_the_fallback(tmp_path, monkeypatch):
    """G6, second half: the smaller number changes the outcome.

    Same card, same model. Linux keeps the driver's 16000 MiB and the subset loop
    awards residency; Windows sees 10384 MiB, nothing holds the model, and the load
    lands on Site A. So Windows AMD users meet the new Auto offload context on
    hardware where Linux AMD users never see it. Not a regression the constant
    introduced -- the same asymmetry sent them to 4096 before -- but it is the cell
    where the new value is most often visible.

    The weights are 10 GiB rather than the 12 GiB this started at. At 12 GiB both
    sides offload now: 12288 + 320 MiB leaves 2900 of a 15508 MiB Linux budget,
    which held a 4096 context and does not hold an 8192 one, so the cell stopped
    contrasting anything the moment the fit floor moved. 10 GiB restores the
    contrast and leaves it a measurement rather than a floor -- Linux pins at 9728,
    1536 above the floor -- so the next floor move shows up as this test failing
    only once it would really have changed the placement.
    """
    windows = next(p for p in PLATFORMS if p[0] == "windows")
    linux = next(p for p in PLATFORMS if p[0] == "linux")
    # 10 GiB of weights: inside a 16000 MiB budget with room for a KV at the fit
    # floor, outside the 10384 MiB one before any context is priced at all.
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
