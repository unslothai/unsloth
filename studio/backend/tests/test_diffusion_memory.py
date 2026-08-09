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
    DEFAULT_IMAGE_HEIGHT,
    DEFAULT_IMAGE_WIDTH,
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
    estimate_gguf_resident_mib,
    estimate_image_runtime_mib,
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


def test_estimate_gguf_resident_mib_matches_packed_size():
    # GGUF weights stay packed (uint8) on device and diffusers dequantises per-matmul, so the resident footprint is about the
    # on-disk size at any quant level (measured on Z-Image-Turbo); a small margin covers allocator overhead.
    assert estimate_gguf_resident_mib(1000) == 1050
    assert estimate_gguf_resident_mib(7220) == 7581
    assert estimate_gguf_resident_mib(None) is None


def test_estimate_image_runtime_scales_with_pixels_and_family():
    base = estimate_image_runtime_mib(width = DEFAULT_IMAGE_WIDTH, height = DEFAULT_IMAGE_HEIGHT)
    bigger = estimate_image_runtime_mib(width = 2048, height = 2048)
    assert bigger > base
    # Distilled / turbo families get a discount.
    turbo = estimate_image_runtime_mib(
        width = DEFAULT_IMAGE_WIDTH, height = DEFAULT_IMAGE_HEIGHT, family = "z-image-turbo"
    )
    assert turbo < base


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
    # 80 GB card, ~16 GB model: fits with headroom, so stay resident (bit-identical).
    plan = plan_diffusion_memory(
        target = _target(),
        device_memory = _discrete(80000),
        model_dense_mib = 12000,
        runtime_headroom_mib = 4000,
    )
    assert plan.offload_policy == OFFLOAD_NONE
    assert plan.vae_tiling is False and plan.vae_slicing is False  # roomy -> no tiling


def test_auto_model_offload_on_tight_fit():
    # 24 GB free -> reserve 2400 -> budget 21600, 0.85*budget = 18360. required 21000 is over that but under budget, so whole-module offload.
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
    # A big transformer pushes the resident total over budget while the companions still fit, so stream the transformer.
    plan = plan_diffusion_memory(
        target = _target(),
        device_memory = _discrete(8000, 8000),
        model_dense_mib = 40000,
        companion_dense_mib = 1500,
        runtime_headroom_mib = 1000,
        base_overhead_mib = 1000,
    )
    assert plan.offload_policy == OFFLOAD_GROUP
    # Group keeps the VAE resident, so balanced uses exact slicing but NOT lossy tiling and stays bit-identical.
    assert plan.vae_slicing is True and plan.vae_tiling is False


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
    # Without a companion estimate the planner can't prove group fits, so it takes the safest cut.
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
    # A roomy GPU would stay resident under auto, but cpu_offload=True forces offload.
    plan = plan_diffusion_memory(
        target = _target(),
        device_memory = _discrete(80000),
        model_dense_mib = 4000,
        runtime_headroom_mib = 2000,
        explicit_offload = True,
    )
    assert plan.offload_policy == OFFLOAD_MODEL
    assert any("explicit cpu_offload" in r for r in plan.reasons)


def test_explicit_memory_mode_wins_over_legacy_cpu_offload():
    # memory_mode is documented to override cpu_offload, so fast + the legacy flag stays resident instead of downgrading to offload.
    plan = plan_diffusion_memory(
        target = _target(),
        device_memory = _discrete(80000),
        model_dense_mib = 4000,
        runtime_headroom_mib = 2000,
        requested_mode = MEMORY_MODE_FAST,
        explicit_offload = True,
    )
    assert plan.offload_policy == OFFLOAD_NONE
    assert not any("explicit cpu_offload" in r for r in plan.reasons)


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
        self.offload_device = None

    def to(self, device):
        self.calls.append(f"to:{device}")
        return self

    def enable_model_cpu_offload(self, device = None):
        self.calls.append("model_offload")
        self.offload_device = device

    def enable_sequential_cpu_offload(self, device = None):
        self.calls.append("sequential_offload")
        self.offload_device = device

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
        estimates = {},
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
    assert pipe.offload_device == "cuda"  # device threaded to enable_model_cpu_offload


def test_apply_model_offload_passes_target_device():
    # enable_model_cpu_offload defaults to CUDA, so a non-CUDA accelerator (e.g. Intel XPU) must have its device forwarded.
    pipe = _RecordingPipe()
    apply_memory_plan(pipe, _plan(OFFLOAD_MODEL, tiling = False), device = "xpu")
    assert pipe.offload_device == "xpu"


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

        def enable_model_cpu_offload(self, device = None):
            self.offloaded = True

    pipe = _VaeOnly()
    effective, tiled = apply_memory_plan(pipe, _plan(OFFLOAD_MODEL, tiling = True), device = "cuda")
    assert tiled is True and pipe.vae.tiled and pipe.vae.sliced


def test_apply_group_falls_back_to_model_without_transformer():
    # The recording pipe has no .transformer, so group offload cannot engage and the applier falls back to whole-module offload.
    pipe = _RecordingPipe()
    effective, _ = apply_memory_plan(pipe, _plan(OFFLOAD_GROUP, tiling = True), device = "cuda")
    assert effective == OFFLOAD_MODEL and "model_offload" in pipe.calls


def _install_fake_torch_and_hooks(monkeypatch, apply_group_offloading):
    """Fake torch.nn.Module + diffusers.hooks.apply_group_offloading for _apply_group_offload."""
    import sys

    class _Mod:  # stands in for a torch.nn.Module instance (a streamed transformer)
        pass

    fake_torch = types.ModuleType("torch")
    fake_torch.nn = types.SimpleNamespace(Module = _Mod)
    fake_torch.device = lambda d: types.SimpleNamespace(type = str(d).split(":")[0])
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    if "diffusers" not in sys.modules:
        monkeypatch.setitem(sys.modules, "diffusers", types.ModuleType("diffusers"))
    fake_hooks = types.ModuleType("diffusers.hooks")
    fake_hooks.apply_group_offloading = apply_group_offloading
    monkeypatch.setitem(sys.modules, "diffusers.hooks", fake_hooks)
    return _Mod


def test_apply_group_partial_hooks_propagates_not_crash_fallback(monkeypatch):
    # A dual-DiT pipe whose second transformer fails group offload AFTER the first installed hooks is left partial, which
    # enable_model_cpu_offload rejects, so the applier must PROPAGATE the failure instead of letting the fallback crash.
    import core.inference.diffusion_memory as mem

    calls = {"n": 0}

    def _apply(module, **kw):
        calls["n"] += 1
        if calls["n"] >= 2:
            raise RuntimeError("OOM on second DiT")

    Mod = _install_fake_torch_and_hooks(monkeypatch, _apply)

    class _DualPipe:
        transformer = Mod()
        transformer_2 = Mod()
        components: dict = {}

    with pytest.raises(RuntimeError, match = "OOM on second DiT"):
        mem._apply_group_offload(_DualPipe(), "cuda", logger = None)
    assert calls["n"] == 2  # first installed hooks, second failed -> propagated


def test_apply_group_single_transformer_failure_falls_back(monkeypatch):
    # A single-DiT pipe whose group offload fails with NO hooks installed returns False so the caller falls back cleanly.
    import core.inference.diffusion_memory as mem

    def _apply(module, **kw):
        raise RuntimeError("OOM before any hook")

    Mod = _install_fake_torch_and_hooks(monkeypatch, _apply)

    class _SinglePipe:
        transformer = Mod()
        components: dict = {}

    assert mem._apply_group_offload(_SinglePipe(), "cuda", logger = None) is False


def test_apply_group_fallback_enables_vae_tiling():
    # A balanced/group plan keeps the VAE resident (tiling off), so when group offload cannot engage and we drop to whole-module offload the applier must turn tiling ON.
    plan = _plan(OFFLOAD_GROUP, tiling = True)
    assert plan.vae_tiling is False  # group plan leaves tiling off by design
    pipe = _RecordingPipe()  # no .transformer -> group offload falls back to model
    effective, tiled = apply_memory_plan(pipe, plan, device = "cuda")
    assert effective == OFFLOAD_MODEL
    assert tiled is True and "vae_tiling" in pipe.calls


def test_apply_sequential_offload():
    pipe = _RecordingPipe()
    effective, _ = apply_memory_plan(
        pipe, _manual_plan(OFFLOAD_SEQUENTIAL, tiling = True), device = "cuda"
    )
    assert "sequential_offload" in pipe.calls and "to:cuda" not in pipe.calls
    assert effective == OFFLOAD_SEQUENTIAL
    assert pipe.offload_device == "cuda"  # device threaded to sequential offload too


def test_apply_sequential_falls_back_to_model_offload_when_unsupported():
    # Sequential offload is unreliable for GGUF on some diffusers versions, so the applier falls back to whole-module and reports what ran.
    class _NoSeqPipe(_RecordingPipe):
        def enable_sequential_cpu_offload(self, device = None):
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


# ── settled snapshot + capacity-fit retry helpers ────────────────────────────


def test_settled_snapshot_takes_max_free_over_reads(monkeypatch):
    # A transient foreign allocation can only SHRINK free, so the settled snapshot keeps the max free across reads (60 GB on an idle 183 GB card).
    from core.inference import diffusion_memory as dm

    reads = [
        DeviceMemory("cuda", "cuda", "discrete_vram", free_mib = 60_000, total_mib = 183_359),
        DeviceMemory("cuda", "cuda", "discrete_vram", free_mib = 170_000, total_mib = 183_359),
        DeviceMemory("cuda", "cuda", "discrete_vram", free_mib = 170_000, total_mib = 183_359),
    ]
    monkeypatch.setattr(dm, "snapshot_device_memory", lambda target: reads.pop(0))
    snap = dm.settled_snapshot_device_memory(_target(device = "cuda"), attempts = 3, delay_s = 0)
    assert snap.free_mib == 170_000


def test_settled_snapshot_stops_early_when_device_already_idle(monkeypatch):
    # First read already within the reserve of total: no transient to wait out, one read only.
    from core.inference import diffusion_memory as dm

    calls = []

    def fake_snapshot(target):
        calls.append(1)
        return DeviceMemory("cuda", "cuda", "discrete_vram", free_mib = 170_000, total_mib = 183_359)

    monkeypatch.setattr(dm, "snapshot_device_memory", fake_snapshot)
    snap = dm.settled_snapshot_device_memory(_target(device = "cuda"), attempts = 3, delay_s = 0)
    assert snap.free_mib == 170_000
    assert calls == [1]


def test_settled_snapshot_passthrough_off_cuda(monkeypatch):
    # Non-cuda targets keep the single-read behaviour (no settle loop).
    from core.inference import diffusion_memory as dm

    calls = []

    def fake_snapshot(target):
        calls.append(1)
        return DeviceMemory("mps", "mps", "unified_memory", free_mib = 8_000, total_mib = 16_000)

    monkeypatch.setattr(dm, "snapshot_device_memory", fake_snapshot)
    snap = dm.settled_snapshot_device_memory(_target(device = "mps"), attempts = 3, delay_s = 0)
    assert snap.memory_kind == "unified_memory"
    assert calls == [1]


def test_plan_fits_total_capacity():
    # True exactly when required fits (total - reserve) * 0.85: the decline can then only come from the instantaneous free reading, so a settled retry helps.
    from core.inference.diffusion_memory import plan_fits_total_capacity

    def plan(
        required,
        total,
        kind = "discrete_vram",
    ):
        return types.SimpleNamespace(
            estimates = {"resident_required_mib": required},
            device_memory = DeviceMemory("cuda", "cuda", kind, free_mib = 1, total_mib = total),
        )

    # FLUX.2-dev int8 incident numbers: 90,228 required on a 183,359 MiB card, so it fits.
    assert plan_fits_total_capacity(plan(90_228, 183_359)) is True
    # Larger than the capacity margin (0.85 * (183,359 - 18,335) = 140,270), so no retry.
    assert plan_fits_total_capacity(plan(150_000, 183_359)) is False
    # Unknown sizes keep today's behaviour (no retry).
    assert plan_fits_total_capacity(plan(None, 183_359)) is False
    assert plan_fits_total_capacity(plan(90_228, None)) is False
    assert plan_fits_total_capacity(types.SimpleNamespace()) is False


# ── the unified-memory oversize refusal ───────────────────────────────────────
# Apple Silicon shares one CPU/GPU pool, so the planner's OFFLOAD_NONE there is a placement
# with no fallback tier, and PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 removes the allocator's hard
# limit: an oversized load is killed by the OS with no Python exception. These cover the
# load-time refusal that replaces that SIGKILL with a message.

_MPS_TOTAL_MIB = 16 * 1024
_MPS_FREE_MIB = int(_MPS_TOTAL_MIB * 0.80)  # RAM free once macOS + a browser + Studio are up


def _unified_plan(
    *,
    model_dense_mib,
    runtime_headroom_mib = 3072,
    free_mib = _MPS_FREE_MIB,
    total_mib = _MPS_TOTAL_MIB,
    kind = "unified_memory",
    device = "mps",
):
    """A plan straight from the shipped planner, so the budget arithmetic under test is the
    real arithmetic rather than a hand-written copy of it."""
    return plan_diffusion_memory(
        target = _target(device = device, backend = device, supports_offload = False),
        device_memory = DeviceMemory(device, device, kind, free_mib, total_mib),
        model_dense_mib = model_dense_mib,
        runtime_headroom_mib = runtime_headroom_mib,
    )


def test_unified_oversize_refuses_and_names_family_and_both_numbers():
    from core.inference.diffusion_memory import unified_memory_shortfall_message

    # 16 GiB Mac, 12.8 GiB free, 20% unified reserve, so about 9.5 GiB of budget. 24 GiB cannot fit.
    plan = _unified_plan(model_dense_mib = 24 * 1024)
    assert plan.offload_policy == OFFLOAD_NONE  # the planner still has no fallback to offer
    message = unified_memory_shortfall_message(plan, family = "wan2.2-ti2v-5b")
    assert message is not None
    assert "wan2.2-ti2v-5b" in message
    # Weights + the flat base overhead, and the safe budget, both rendered in GB.
    assert "about 26 GB of memory for its weights" in message  # 24 GiB weights + 2 GiB overhead
    assert "about 10 GB is usable" in message  # 12.8 GiB free, less 20% of 16 GiB
    assert "13 GB currently free" in message
    # The most useful thing the user can change, and the escape hatch.
    assert "smaller or more quantized model" in message
    assert "UNSLOTH_DIFFUSION_ALLOW_OVERSIZED_LOAD=1" in message


def test_unified_oversize_ignores_the_soft_runtime_headroom():
    """The refusal budgets WEIGHTS only. A load whose weights fit but whose weights + activation
    headroom do not is marginal, and VAE tiling/slicing (always on for mps) cuts the decode peak
    the headroom estimate does not model, so it must NOT be refused."""
    from core.inference.diffusion_memory import unified_memory_shortfall_message

    budget = _MPS_FREE_MIB - max(2048, int(_MPS_TOTAL_MIB * 0.20))
    weights = budget - 2048  # weights + the 2048 base overhead land exactly on the budget
    plan = _unified_plan(model_dense_mib = weights, runtime_headroom_mib = 8192)
    assert plan.estimates["resident_required_mib"] > budget  # refused if headroom counted
    assert unified_memory_shortfall_message(plan) is None
    # One MiB more of weights does tip it over.
    assert unified_memory_shortfall_message(_unified_plan(model_dense_mib = weights + 1)) is not None


def test_unified_oversize_never_refuses_discrete_vram():
    """Discrete VRAM keeps its fallback ladder: an oversized model streams from host RAM under
    group / whole-module offload, so refusing it would break a load that works today."""
    from core.inference.diffusion_memory import unified_memory_shortfall_message

    plan = plan_diffusion_memory(
        target = _target(device = "cuda", backend = "cuda", supports_offload = True),
        device_memory = DeviceMemory("cuda", "cuda", "discrete_vram", 12_000, 16_384),
        model_dense_mib = 80 * 1024,
        runtime_headroom_mib = 6963,
    )
    assert plan.offload_policy == OFFLOAD_MODEL
    assert unified_memory_shortfall_message(plan, family = "ltx-2") is None


def test_unified_oversize_never_refuses_plain_cpu_system_memory():
    # A CPU target reports system_memory, has swap, and is an opt-in fringe path: unchanged.
    from core.inference.diffusion_memory import unified_memory_shortfall_message
    plan = _unified_plan(model_dense_mib = 80 * 1024, kind = "system_memory", device = "cpu")
    assert unified_memory_shortfall_message(plan) is None


@pytest.mark.parametrize(
    "kwargs",
    [
        {"free_mib": None},  # psutil unavailable: budget unknown
        {"model_dense_mib": None},  # unscannable checkpoint: size unknown
    ],
)
def test_unified_oversize_fails_open_on_unknown_inputs(kwargs):
    """Matches the planner's own "budget or model size unknown; staying resident": never turn a
    failed probe into a refusal."""
    from core.inference.diffusion_memory import unified_memory_shortfall_message

    kwargs = {"model_dense_mib": 80 * 1024, **kwargs}
    assert unified_memory_shortfall_message(_unified_plan(**kwargs)) is None


def test_unified_oversize_env_override_attempts_the_load_anyway(monkeypatch):
    from core.inference.diffusion_memory import (
        UNIFIED_OVERSIZE_ENV,
        raise_on_unified_memory_shortfall,
        unified_memory_shortfall_message,
    )

    plan = _unified_plan(model_dense_mib = 80 * 1024)
    assert unified_memory_shortfall_message(plan) is not None
    for value in ("1", "true", "YES", "on"):
        monkeypatch.setenv(UNIFIED_OVERSIZE_ENV, value)
        assert unified_memory_shortfall_message(plan) is None
        raise_on_unified_memory_shortfall(plan)  # must not raise
    monkeypatch.setenv(UNIFIED_OVERSIZE_ENV, "0")
    assert unified_memory_shortfall_message(plan) is not None


def test_unified_oversize_message_survives_a_malformed_plan():
    from core.inference.diffusion_memory import unified_memory_shortfall_message
    assert unified_memory_shortfall_message(types.SimpleNamespace()) is None


def test_raise_on_unified_memory_shortfall_raises_runtime_error_with_the_message():
    """RuntimeError matches the llama.cpp unified-memory APU refusal. Both loaders call this on
    a worker thread inside load_pipeline, where _run_load stringifies it onto load_progress, so
    the text reaches the UI toast and the 409 mapping of the synchronous route never applies."""
    from core.inference.diffusion_memory import (
        raise_on_unified_memory_shortfall,
        unified_memory_shortfall_message,
    )

    plan = _unified_plan(model_dense_mib = 80 * 1024)
    with pytest.raises(RuntimeError) as excinfo:
        raise_on_unified_memory_shortfall(plan, family = "ltx-2")
    assert str(excinfo.value) == unified_memory_shortfall_message(plan, family = "ltx-2")
    # A plan that fits is a silent no-op.
    raise_on_unified_memory_shortfall(_unified_plan(model_dense_mib = 1024))


def test_unified_oversize_decision_matrix_for_the_real_video_families():
    """The shipped video family tables against real Mac RAM sizes: the refusal must fire exactly
    where the weights genuinely cannot fit, and must stay silent where they can.

    ``video_families`` is a pure table module (no torch, no diffusers), so this stays hermetic.
    ``free`` is modelled at 80% of RAM, the share left once macOS, a browser and Studio are up.
    """
    from core.inference.video_families import _FAMILIES
    from core.inference.diffusion_memory import (
        DEFAULT_BASE_OVERHEAD_MIB,
        estimate_video_runtime_mib,
        unified_memory_shortfall_message,
    )

    mib_per_gb = 1000.0**3 / (1024.0 * 1024.0)  # the tables are DECIMAL GB
    # family: the RAM sizes (GiB) at which the load must be REFUSED.
    expected_refusals = {
        "ltx-2": {16, 24, 32, 64, 96},
        "wan2.2-ti2v-5b": {16, 24, 32},
        "wan2.2-t2v-a14b": {16, 24, 32, 64, 96},
        "hunyuanvideo-1.5": {16, 24, 32},
        "hunyuanvideo-1.5-720p": {16, 24, 32},
    }
    assert {f.name for f in _FAMILIES} == set(
        expected_refusals
    ), "a video family was added or renamed: extend the expected refusal matrix"
    for fam in _FAMILIES:
        width, height = fam.resolution_presets[0]
        dense = int(sum(fam.bf16_components_gb) * mib_per_gb)
        headroom = estimate_video_runtime_mib(
            width = width, height = height, num_frames = fam.default_num_frames
        )
        for ram_gib in (16, 24, 32, 64, 96, 128):
            total = ram_gib * 1024
            plan = _unified_plan(
                model_dense_mib = dense,
                runtime_headroom_mib = headroom,
                free_mib = int(total * 0.80),
                total_mib = total,
            )
            # The planner never has an alternative to offer on unified memory: that is the bug.
            assert plan.offload_policy == OFFLOAD_NONE
            refused = unified_memory_shortfall_message(plan, family = fam.name) is not None
            assert refused is (ram_gib in expected_refusals[fam.name]), (
                f"{fam.name} at {ram_gib} GiB: refused={refused}, "
                f"weights+overhead={dense + DEFAULT_BASE_OVERHEAD_MIB} MiB, "
                f"budget={plan.estimates['safe_device_budget_mib']} MiB"
            )
