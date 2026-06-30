# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Unit tests for the diffusion memory planner (``diffusion_memory.py``).

Hermetic and CPU-only: no torch, diffusers, GPU, or network. The device target
and the device-memory snapshot are constructed directly, so the planner's policy
matrix and the applier's pipeline calls are exercised in isolation.
"""

from __future__ import annotations

import types

import pytest

from core.inference.diffusion_memory import (
    MEMORY_MODE_BALANCED,
    MEMORY_MODE_FAST,
    MEMORY_MODE_LOW_VRAM,
    OFFLOAD_GROUP,
    OFFLOAD_MODEL,
    OFFLOAD_NONE,
    OFFLOAD_SEQUENTIAL,
    DeviceMemory,
    MemoryPlan,
    apply_memory_plan,
    estimate_gguf_dense_mib,
    estimate_image_runtime_mib,
    infer_gguf_quant_label,
    normalize_memory_mode,
    plan_diffusion_memory,
    snapshot_device_memory,
)


def _target(
    *,
    device = "cuda",
    backend = "cuda",
    supports_offload = True,
):
    """A duck-typed stand-in for DiffusionDeviceTarget (only the fields the
    planner / snapshot read)."""
    return types.SimpleNamespace(
        device = device,
        backend = backend,
        supports_model_cpu_offload = supports_offload,
    )


def _discrete(free_mib, total_mib = None):
    return DeviceMemory("cuda", "cuda", "discrete_vram", free_mib, total_mib or free_mib)


# ── mode normalisation ────────────────────────────────────────────────────────


def test_normalize_memory_mode_accepts_and_rejects():
    assert normalize_memory_mode(None) is None
    assert normalize_memory_mode("  ") is None
    assert normalize_memory_mode("LOW-VRAM") == "low_vram"
    assert normalize_memory_mode("Balanced") == "balanced"
    with pytest.raises(ValueError):
        normalize_memory_mode("ultra")


# ── filename / size estimates ─────────────────────────────────────────────────


@pytest.mark.parametrize(
    "filename,expected",
    [
        ("z-image-turbo-Q4_K_M.gguf", "Q4_K_M"),
        ("flux1-dev-Q8_0.gguf", "Q8_0"),
        ("model-BF16.gguf", "BF16"),
        ("qwen-image-IQ4_XS.gguf", "IQ4_XS"),
        ("no-quant-here.gguf", None),
        (None, None),
    ],
)
def test_infer_gguf_quant_label(filename, expected):
    assert infer_gguf_quant_label(filename) == expected


def test_estimate_gguf_dense_mib_expansion():
    # 4-bit roughly quadruples once dequantised to bf16; F16 is already dense.
    assert estimate_gguf_dense_mib(1000, "Q4_K_M") == 4000
    assert estimate_gguf_dense_mib(1000, "Q8_0") == 2000
    assert estimate_gguf_dense_mib(1000, "BF16") == 1000
    assert estimate_gguf_dense_mib(None, "Q4_K_M") is None
    # Unknown quant falls back to the conservative 4-bit-ish factor.
    assert estimate_gguf_dense_mib(1000, None) == 4000


def test_estimate_image_runtime_scales_by_family():
    base = estimate_image_runtime_mib(family = "z-image")
    # Distilled / turbo families get a discount; editing pipelines need more.
    turbo = estimate_image_runtime_mib(family = "z-image-turbo")
    edit = estimate_image_runtime_mib(family = "flux-kontext-edit")
    assert turbo < base < edit
    assert turbo >= 1024  # never below the floor


# ── planner: device classes ───────────────────────────────────────────────────


def test_cpu_target_never_offloads_but_tiles():
    plan = plan_diffusion_memory(
        target = _target(device = "cpu", backend = "cpu", supports_offload = False),
        device_memory = DeviceMemory("cpu", "cpu", "system_memory", 8000, 16000),
        model_dense_mib = 4000,
        runtime_headroom_mib = 2000,
    )
    assert plan.offload_policy == OFFLOAD_NONE
    # CPU/MPS have no separate device pool, so VAE tiling is on to cap the spike.
    assert plan.vae_tiling and plan.vae_slicing


def test_mps_unified_never_auto_offloads():
    plan = plan_diffusion_memory(
        target = _target(device = "mps", backend = "mps", supports_offload = False),
        device_memory = DeviceMemory("mps", "mps", "unified_memory", 4000, 32000),
        model_dense_mib = 20000,
        runtime_headroom_mib = 4000,
    )
    assert plan.offload_policy == OFFLOAD_NONE
    assert any("unified" in r for r in plan.reasons)


def test_unified_cuda_skips_offload_even_if_offload_capable():
    # An integrated CUDA SoC reports unified memory; CPU offload would free nothing.
    plan = plan_diffusion_memory(
        target = _target(device = "cuda", backend = "cuda", supports_offload = True),
        device_memory = DeviceMemory("cuda", "cuda", "unified_memory", 2000, 16000),
        model_dense_mib = 12000,
        runtime_headroom_mib = 4000,
    )
    assert plan.offload_policy == OFFLOAD_NONE


# ── planner: auto budget tiers on a discrete GPU ──────────────────────────────


def test_auto_resident_when_roomy():
    # 80 GB card, ~16 GB model: fits with headroom -> stay resident (bit-identical).
    plan = plan_diffusion_memory(
        target = _target(),
        device_memory = _discrete(80000),
        model_dense_mib = 12000,
        runtime_headroom_mib = 4000,
    )
    assert plan.offload_policy == OFFLOAD_NONE
    assert plan.vae_tiling is False and plan.vae_slicing is False  # roomy -> no tiling


def test_auto_model_offload_on_tight_fit():
    # 24 GB free -> reserve max(2048, 2400)=2400 -> budget 21600, 0.85*budget=18360.
    # required = 16000+4000+1000 = 21000: over 0.85*budget but still under budget
    # -> whole-module offload.
    plan = plan_diffusion_memory(
        target = _target(),
        device_memory = _discrete(24000, 24000),
        model_dense_mib = 16000,
        runtime_headroom_mib = 4000,
        base_overhead_mib = 1000,
    )
    assert plan.offload_policy == OFFLOAD_MODEL
    assert plan.vae_tiling is True  # offloading -> device is tight -> tile


def test_auto_group_offload_when_transformer_overflows_but_companions_fit():
    # Big transformer pushes the resident total over budget, but the companions
    # (text encoder + VAE) still fit -> stream the transformer (fast, moderate cut).
    plan = plan_diffusion_memory(
        target = _target(),
        device_memory = _discrete(8000, 8000),
        model_dense_mib = 40000,
        companion_dense_mib = 1500,
        runtime_headroom_mib = 1000,
        base_overhead_mib = 1000,
    )
    assert plan.offload_policy == OFFLOAD_GROUP


def test_legacy_cpu_offload_forces_whole_module_over_auto_group():
    # Legacy cpu_offload=True (no explicit memory_mode) must keep its historical
    # whole-module meaning even when auto would otherwise pick the lighter group
    # offload on a tight card.
    plan = plan_diffusion_memory(
        target = _target(),
        device_memory = _discrete(8000, 8000),
        model_dense_mib = 40000,
        companion_dense_mib = 1500,
        runtime_headroom_mib = 1000,
        base_overhead_mib = 1000,
        explicit_offload = True,
    )
    assert plan.offload_policy == OFFLOAD_MODEL


def test_explicit_balanced_mode_still_honored_over_legacy_cpu_offload():
    # An explicit memory_mode wins over the legacy flag: balanced stays group offload.
    plan = plan_diffusion_memory(
        target = _target(),
        device_memory = _discrete(8000, 8000),
        model_dense_mib = 40000,
        companion_dense_mib = 1500,
        runtime_headroom_mib = 1000,
        base_overhead_mib = 1000,
        explicit_offload = True,
        requested_mode = "balanced",
    )
    assert plan.offload_policy == OFFLOAD_GROUP


def test_auto_model_offload_when_companions_exceed_budget():
    # The text encoder itself is too big to stay resident -> offload everything.
    plan = plan_diffusion_memory(
        target = _target(),
        device_memory = _discrete(8000, 8000),
        model_dense_mib = 40000,
        companion_dense_mib = 30000,
        runtime_headroom_mib = 4000,
    )
    assert plan.offload_policy == OFFLOAD_MODEL


def test_auto_model_offload_when_companion_size_unknown():
    # Without a companion estimate the planner can't prove group fits -> safest cut.
    plan = plan_diffusion_memory(
        target = _target(),
        device_memory = _discrete(8000, 8000),
        model_dense_mib = 40000,
        runtime_headroom_mib = 4000,
    )
    assert plan.offload_policy == OFFLOAD_MODEL


def test_auto_stays_resident_when_budget_unknown():
    plan = plan_diffusion_memory(
        target = _target(),
        device_memory = _discrete(None, None),
        model_dense_mib = 40000,
        runtime_headroom_mib = 4000,
    )
    assert plan.offload_policy == OFFLOAD_NONE
    assert any("unknown" in r for r in plan.reasons)


# ── planner: explicit modes + cpu_offload override ────────────────────────────


def test_explicit_modes_force_policy_regardless_of_budget():
    roomy = _discrete(80000)
    assert (
        plan_diffusion_memory(
            target = _target(),
            device_memory = roomy,
            model_dense_mib = 1000,
            runtime_headroom_mib = 1000,
            requested_mode = MEMORY_MODE_FAST,
        ).offload_policy
        == OFFLOAD_NONE
    )
    assert (
        plan_diffusion_memory(
            target = _target(),
            device_memory = roomy,
            model_dense_mib = 1000,
            runtime_headroom_mib = 1000,
            requested_mode = MEMORY_MODE_BALANCED,
        ).offload_policy
        == OFFLOAD_GROUP
    )
    assert (
        plan_diffusion_memory(
            target = _target(),
            device_memory = roomy,
            model_dense_mib = 1000,
            runtime_headroom_mib = 1000,
            requested_mode = MEMORY_MODE_LOW_VRAM,
        ).offload_policy
        == OFFLOAD_MODEL
    )


def test_fast_falls_back_to_model_offload_when_it_does_not_fit():
    plan = plan_diffusion_memory(
        target = _target(),
        device_memory = _discrete(8000, 8000),
        model_dense_mib = 40000,
        runtime_headroom_mib = 4000,
        requested_mode = MEMORY_MODE_FAST,
    )
    assert plan.offload_policy == OFFLOAD_MODEL


def test_explicit_cpu_offload_overrides_resident_auto_choice():
    # Roomy GPU -> auto would stay resident, but cpu_offload=True forces offload.
    plan = plan_diffusion_memory(
        target = _target(),
        device_memory = _discrete(80000),
        model_dense_mib = 4000,
        runtime_headroom_mib = 2000,
        explicit_offload = True,
    )
    assert plan.offload_policy == OFFLOAD_MODEL
    assert any("explicit cpu_offload" in r for r in plan.reasons)


def test_explicit_cpu_offload_ignored_on_cpu_target():
    plan = plan_diffusion_memory(
        target = _target(device = "cpu", backend = "cpu", supports_offload = False),
        device_memory = DeviceMemory("cpu", "cpu", "system_memory", 8000, 16000),
        model_dense_mib = 4000,
        runtime_headroom_mib = 2000,
        explicit_offload = True,
    )
    assert plan.offload_policy == OFFLOAD_NONE


# ── snapshot ──────────────────────────────────────────────────────────────────


def test_snapshot_cpu_target_uses_system_memory(monkeypatch):
    import core.inference.diffusion_memory as mem

    monkeypatch.setattr(mem, "_system_memory_mib", lambda: (16000, 9000))
    snap = snapshot_device_memory(_target(device = "cpu", backend = "cpu"))
    assert snap.memory_kind == "system_memory"
    assert snap.free_mib == 9000 and snap.total_mib == 16000


def test_snapshot_cuda_reads_mem_get_info(monkeypatch):
    import sys

    fake_torch = types.ModuleType("torch")
    fake_torch.cuda = types.SimpleNamespace(
        mem_get_info = lambda: (10 * 1024 * 1024 * 1024, 24 * 1024 * 1024 * 1024),
        get_device_properties = lambda i: types.SimpleNamespace(integrated = False),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    snap = snapshot_device_memory(_target())
    assert snap.memory_kind == "discrete_vram"
    assert snap.free_mib == 10 * 1024 and snap.total_mib == 24 * 1024


def test_snapshot_never_raises_on_probe_failure(monkeypatch):
    import sys

    fake_torch = types.ModuleType("torch")

    def _boom():
        raise RuntimeError("no cuda")

    fake_torch.cuda = types.SimpleNamespace(mem_get_info = _boom)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    snap = snapshot_device_memory(_target())
    assert snap.free_mib is None and snap.total_mib is None


# ── applier ───────────────────────────────────────────────────────────────────


class _RecordingPipe:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def to(self, device):
        self.calls.append(f"to:{device}")
        return self

    def enable_model_cpu_offload(self):
        self.calls.append("model_offload")

    def enable_sequential_cpu_offload(self):
        self.calls.append("sequential_offload")

    def enable_vae_tiling(self):
        self.calls.append("vae_tiling")

    def enable_vae_slicing(self):
        self.calls.append("vae_slicing")


def _plan(policy, *, tiling):
    return plan_diffusion_memory(
        target = _target(),
        device_memory = _discrete(80000) if policy == OFFLOAD_NONE else _discrete(4000, 8000),
        model_dense_mib = 1000 if policy == OFFLOAD_NONE else 40000,
        runtime_headroom_mib = 1000,
        requested_mode = {
            OFFLOAD_NONE: MEMORY_MODE_FAST,
            OFFLOAD_GROUP: MEMORY_MODE_BALANCED,
            OFFLOAD_MODEL: MEMORY_MODE_LOW_VRAM,
        }[policy],
    )


def _manual_plan(policy, *, tiling):
    """Build a plan for a policy the auto/explicit modes no longer emit (sequential)."""
    return MemoryPlan(
        requested_mode = "manual",
        offload_policy = policy,
        vae_tiling = tiling,
        vae_slicing = tiling,
        device_memory = _discrete(4000, 8000),
    )


def test_apply_none_places_resident():
    pipe = _RecordingPipe()
    effective, tiled = apply_memory_plan(pipe, _plan(OFFLOAD_NONE, tiling = False), device = "cuda")
    assert pipe.calls == ["to:cuda"]  # no tiling on a roomy resident run
    assert effective == OFFLOAD_NONE and tiled is False


def test_apply_model_offload_engages_offload_and_tiling():
    pipe = _RecordingPipe()
    effective, tiled = apply_memory_plan(pipe, _plan(OFFLOAD_MODEL, tiling = True), device = "cuda")
    assert "model_offload" in pipe.calls
    assert "to:cuda" not in pipe.calls  # offload owns placement; never both
    assert "vae_tiling" in pipe.calls and "vae_slicing" in pipe.calls
    assert effective == OFFLOAD_MODEL and tiled is True


def test_apply_vae_tiling_falls_back_to_vae_submodule():
    # Z-Image-style pipeline: no pipeline-level enable_vae_tiling, only pipe.vae.
    class _VaeOnly:
        def __init__(self):
            self.vae = types.SimpleNamespace(
                tiled = False,
                sliced = False,
                enable_tiling = self._tile,
                enable_slicing = self._slice,
            )

        def _tile(self):
            self.vae.tiled = True

        def _slice(self):
            self.vae.sliced = True

        def enable_model_cpu_offload(self):
            self.offloaded = True

    pipe = _VaeOnly()
    effective, tiled = apply_memory_plan(pipe, _plan(OFFLOAD_MODEL, tiling = True), device = "cuda")
    assert tiled is True and pipe.vae.tiled and pipe.vae.sliced


def test_apply_group_falls_back_to_model_without_transformer():
    # The recording pipe has no .transformer, so group offload can't engage and the
    # applier falls back to whole-module offload, reporting the real policy.
    pipe = _RecordingPipe()
    effective, _ = apply_memory_plan(pipe, _plan(OFFLOAD_GROUP, tiling = True), device = "cuda")
    assert effective == OFFLOAD_MODEL and "model_offload" in pipe.calls


def test_apply_sequential_offload():
    pipe = _RecordingPipe()
    effective, _ = apply_memory_plan(
        pipe, _manual_plan(OFFLOAD_SEQUENTIAL, tiling = True), device = "cuda"
    )
    assert "sequential_offload" in pipe.calls and "to:cuda" not in pipe.calls
    assert effective == OFFLOAD_SEQUENTIAL


def test_apply_sequential_falls_back_to_model_offload_when_unsupported():
    # Sequential offload is unreliable for GGUF on some diffusers versions; the
    # applier must fall back to whole-module offload and report what actually ran.
    class _NoSeqPipe(_RecordingPipe):
        def enable_sequential_cpu_offload(self):
            raise RuntimeError("sequential offload not supported for this transformer")

    pipe = _NoSeqPipe()
    effective, _ = apply_memory_plan(
        pipe, _manual_plan(OFFLOAD_SEQUENTIAL, tiling = True), device = "cuda"
    )
    assert effective == OFFLOAD_MODEL
    assert "model_offload" in pipe.calls


def test_apply_tolerates_pipe_without_vae_savers():
    # A pipeline missing enable_vae_* must not crash the applier.
    class _Bare:
        def __init__(self):
            self.moved = None

        def to(self, device):
            self.moved = device

    bare = _Bare()
    _, tiled = apply_memory_plan(bare, _plan(OFFLOAD_NONE, tiling = False), device = "cpu")
    assert bare.moved == "cpu" and tiled is False
