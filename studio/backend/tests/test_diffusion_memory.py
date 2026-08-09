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


# ── the streamed-text-encoder group tier ──────────────────────────────────────
# A text encoder runs ONCE, before step 0, but group offload places every non-streamed component
# resident, so its bytes are reserved for the whole denoise. Where that is the only thing pushing
# the group floor over budget, streaming the encoders too keeps the tier instead of dropping to
# whole-module offload (measured 48m25s for a 20-step 1024x1024 image on a 16 GB card).

# The 16 GB card from the report: free 15,870 of 16,305 MiB, reserve max(2048, 10%) = 2048, so the
# safe budget is exactly the 13,822 MiB the failing plan was measured against.
_16G_FREE_MIB = 15_870
_16G_TOTAL_MIB = 16_305
_16G_BUDGET_MIB = 13_822

# Z-Image at int8, straight from the report: transformer 6451 + companions 7820 = 14,271 resident,
# of which the text encoders are 7629 (8.0 of the 8.2 GB companion table) and the VAE is the rest.
_ZIMAGE_MODEL_DENSE_MIB = 14_271
_ZIMAGE_COMPANION_MIB = 7_820
_ZIMAGE_TEXT_ENCODER_MIB = 7_629


def test_safe_budget_matches_the_reported_16g_card():
    # Anchors every number below: if the reserve rule changes, this fails first rather than
    # silently moving the floors the rest of this section is calibrated against.
    from core.inference.diffusion_memory import _safe_device_budget_mib

    assert _safe_device_budget_mib(_discrete(_16G_FREE_MIB, _16G_TOTAL_MIB)) == _16G_BUDGET_MIB


def _zimage_plan(text_encoder_dense_mib):
    return plan_diffusion_memory(
        target = _target(),
        device_memory = _discrete(_16G_FREE_MIB, _16G_TOTAL_MIB),
        model_dense_mib = _ZIMAGE_MODEL_DENSE_MIB,
        companion_dense_mib = _ZIMAGE_COMPANION_MIB,
        text_encoder_dense_mib = text_encoder_dense_mib,
        runtime_headroom_mib = 8192,
        base_overhead_mib = 2048,
    )


def test_streamed_text_encoders_rescue_the_48_minute_plan():
    # required 14,271 + 8192 + 2048 = 24,511 against a 13,822 budget: not resident either way.
    # group floor 7820 + 8192 + 2048 = 18,060 > 13,822, which is what dropped this to whole-module
    # offload. With the encoders streamed the floor is 191 + 8192 + 2048 = 10,431 and fits.
    before = _zimage_plan(None)
    assert before.offload_policy == OFFLOAD_MODEL
    assert before.stream_text_encoders is False
    assert before.estimates["group_floor_streamed_te_mib"] is None

    after = _zimage_plan(_ZIMAGE_TEXT_ENCODER_MIB)
    assert after.offload_policy == OFFLOAD_GROUP
    assert after.stream_text_encoders is True
    assert after.estimates["resident_required_mib"] == 24_511
    assert after.estimates["group_floor_mib"] == 18_060
    assert after.estimates["group_floor_streamed_te_mib"] == 10_431
    assert any("text encoders" in reason for reason in after.reasons)
    # Group keeps the VAE resident, so the decode stays bit-identical (sliced, not tiled).
    assert after.vae_slicing is True and after.vae_tiling is False
    # The flag has to reach the applier, which reads the public dict in status/logging too.
    assert after.as_public_dict()["stream_text_encoders"] is True


def test_plain_group_is_preferred_over_streaming_the_text_encoders():
    # Where the companions fit as they are, the encoders stay resident: streaming them is a small
    # loss for no gain, so the new tier must only engage as a rescue from whole-module offload.
    plan = plan_diffusion_memory(
        target = _target(),
        device_memory = _discrete(_16G_FREE_MIB, _16G_TOTAL_MIB),
        model_dense_mib = 40_000,
        companion_dense_mib = 1_500,
        text_encoder_dense_mib = 1_400,
        runtime_headroom_mib = 8192,
        base_overhead_mib = 2048,
    )
    assert plan.offload_policy == OFFLOAD_GROUP
    assert plan.stream_text_encoders is False


def test_streamed_text_encoders_still_fall_through_when_even_that_floor_is_over():
    # A VAE alone over budget has nothing left to give up, so whole-module offload still wins.
    plan = plan_diffusion_memory(
        target = _target(),
        device_memory = _discrete(_16G_FREE_MIB, _16G_TOTAL_MIB),
        model_dense_mib = 40_000,
        companion_dense_mib = 20_000,
        text_encoder_dense_mib = 1_000,
        runtime_headroom_mib = 8192,
        base_overhead_mib = 2048,
    )
    assert plan.offload_policy == OFFLOAD_MODEL
    assert plan.stream_text_encoders is False


def test_fast_mode_also_reaches_the_streamed_text_encoder_tier():
    # `fast` has its own does-not-fit branch; it must offer the same ladder as auto, or an explicit
    # fast request on a 16 GB card lands on the 48-minute tier the auto path now avoids.
    plan = plan_diffusion_memory(
        target = _target(),
        device_memory = _discrete(_16G_FREE_MIB, _16G_TOTAL_MIB),
        model_dense_mib = _ZIMAGE_MODEL_DENSE_MIB,
        companion_dense_mib = _ZIMAGE_COMPANION_MIB,
        text_encoder_dense_mib = _ZIMAGE_TEXT_ENCODER_MIB,
        runtime_headroom_mib = 8192,
        base_overhead_mib = 2048,
        requested_mode = MEMORY_MODE_FAST,
    )
    assert plan.offload_policy == OFFLOAD_GROUP
    assert plan.stream_text_encoders is True


def test_text_encoder_split_larger_than_the_companions_clamps_at_zero():
    # The two terms can come from different sources (a cache walk and a family table), so a split
    # that exceeds the total must floor at 0 rather than produce a negative resident requirement.
    plan = plan_diffusion_memory(
        target = _target(),
        device_memory = _discrete(_16G_FREE_MIB, _16G_TOTAL_MIB),
        model_dense_mib = 40_000,
        companion_dense_mib = 7_820,
        text_encoder_dense_mib = 9_999,
        runtime_headroom_mib = 8192,
        base_overhead_mib = 2048,
    )
    assert plan.estimates["group_floor_streamed_te_mib"] == 10_240  # 0 + 8192 + 2048


def _legacy_offload_policy(
    *,
    budget,
    model_dense_mib,
    companion_dense_mib,
    runtime_headroom_mib,
    base_overhead_mib,
):
    """The auto-path decision as it stood BEFORE the streamed-text-encoder tier, written out
    independently. The back-compat fence below compares the shipped planner against it, so a
    change that leaks the new tier into a caller that passed no split fails here."""
    required = model_dense_mib + runtime_headroom_mib + base_overhead_mib
    if required <= int(budget * 0.85):
        return OFFLOAD_NONE
    if companion_dense_mib is None:
        return OFFLOAD_MODEL
    group_floor = companion_dense_mib + runtime_headroom_mib + base_overhead_mib
    return OFFLOAD_GROUP if group_floor <= budget else OFFLOAD_MODEL


def test_no_text_encoder_split_reproduces_the_previous_decision():
    # The back-compat fence. Every existing caller passes no split (the keyword defaults to None),
    # so across the size matrix the planner must land exactly where it did before, and must never
    # report the new tier.
    from core.inference.diffusion_memory import _safe_device_budget_mib

    for free, total in ((6_000, 8_192), (11_000, 12_288), (15_870, 16_305), (80_000, 81_920)):
        for model_dense in (2_000, 14_271, 40_000):
            for companion in (None, 200, 7_820, 30_000):
                for headroom in (1_000, 8_192):
                    memory = _discrete(free, total)
                    plan = plan_diffusion_memory(
                        target = _target(),
                        device_memory = memory,
                        model_dense_mib = model_dense,
                        companion_dense_mib = companion,
                        runtime_headroom_mib = headroom,
                        base_overhead_mib = 2048,
                    )
                    expected = _legacy_offload_policy(
                        budget = _safe_device_budget_mib(memory),
                        model_dense_mib = model_dense,
                        companion_dense_mib = companion,
                        runtime_headroom_mib = headroom,
                        base_overhead_mib = 2048,
                    )
                    where = (free, total, model_dense, companion, headroom)
                    assert plan.offload_policy == expected, where
                    assert plan.stream_text_encoders is False, where
                    assert plan.estimates["group_floor_streamed_te_mib"] is None, where


def _stream_te_pipe(monkeypatch):
    """A pipe with two text encoders and a VAE, plus a record of which modules got group-offload
    hooks and which were placed resident."""
    applied: list[Any] = []
    Mod = _install_fake_torch_and_hooks(monkeypatch, lambda module, **kw: applied.append(module))

    class _Comp(Mod):
        def __init__(self, name):
            self.name = name
            self.placed = None

        def to(self, device):
            self.placed = device
            return self

    transformer = _Comp("transformer")
    text_encoder = _Comp("text_encoder")
    text_encoder_2 = _Comp("text_encoder_2")
    vae = _Comp("vae")

    class _Pipe:
        pass

    pipe = _Pipe()
    pipe.transformer = transformer
    pipe.components = {
        "transformer": transformer,
        "text_encoder": text_encoder,
        "text_encoder_2": text_encoder_2,
        "vae": vae,
    }
    return pipe, applied, transformer, text_encoder, text_encoder_2, vae


def test_apply_group_offload_leaves_text_encoders_resident_by_default(monkeypatch):
    # The unchanged path: only the transformer streams, every other component is placed resident.
    import core.inference.diffusion_memory as mem

    pipe, applied, transformer, te, te2, vae = _stream_te_pipe(monkeypatch)
    assert mem._apply_group_offload(pipe, "cuda", logger = None) is True
    assert applied == [transformer]
    assert te.placed is not None and te2.placed is not None and vae.placed is not None


def test_apply_group_offload_streams_text_encoders_when_asked(monkeypatch):
    # With the flag on, every text_encoder* module gets group-offload hooks and is NOT placed
    # resident. Placing them would defeat the whole point: their bytes are what did not fit.
    import core.inference.diffusion_memory as mem

    pipe, applied, transformer, te, te2, vae = _stream_te_pipe(monkeypatch)
    assert (
        mem._apply_group_offload(pipe, "cuda", logger = None, stream_text_encoders = True) is True
    )
    assert applied == [transformer, te, te2]
    assert te.placed is None and te2.placed is None
    assert vae.placed is not None  # the VAE is the companion the tier keeps resident


def test_apply_memory_plan_threads_the_stream_flag_to_the_group_applier(monkeypatch):
    # End of the wire: the planner's decision has to reach _apply_group_offload, or the plan says
    # group-with-streamed-encoders while the pipeline still places them resident and OOMs.
    import core.inference.diffusion_memory as mem

    seen = {}

    def _fake(pipe, device, logger, *, stream_text_encoders = False):
        seen["stream_text_encoders"] = stream_text_encoders
        return True

    monkeypatch.setattr(mem, "_apply_group_offload", _fake)
    apply_memory_plan(_RecordingPipe(), _zimage_plan(_ZIMAGE_TEXT_ENCODER_MIB), device = "cuda")
    assert seen["stream_text_encoders"] is True
    apply_memory_plan(_RecordingPipe(), _plan(OFFLOAD_GROUP, tiling = True), device = "cuda")
    assert seen["stream_text_encoders"] is False


# ── the generate-time activation guard ────────────────────────────────────────
# The load-time plan budgets the 1024x1024 default because load time cannot know the request, so a
# much larger frame was never compared against anything: at 1088x1920 the plan reserved half the
# working memory the pass needs. On Linux that raises OutOfMemoryError; on Windows WDDM the driver
# serves the overflow from system RAM instead, so ~27 GB lands on a 16 GB card with no exception
# and the desktop stops responding. These cover the pre-sampling refusal that replaces that.

# The Z-Image-Turbo GGUF hint from the report: the base repo carries the distilled marker, so the
# estimate here is the discounted one (0.85), which is the honest 13,872 MiB the issue measured.
_TURBO_HINT = "z-image Z-Image-Turbo-Q4_K_S.gguf unsloth/Z-Image-Turbo-GGUF Tongyi-MAI/Z-Image-Turbo"


def _shortfall(width, height, memory = None, **kw):
    from core.inference.diffusion_memory import image_activation_shortfall_message

    return image_activation_shortfall_message(
        device_memory = memory if memory is not None else _discrete(_16G_FREE_MIB, _16G_TOTAL_MIB),
        width = width,
        height = height,
        family = kw.pop("family", _TURBO_HINT),
        **kw,
    )


def test_estimate_image_runtime_scales_with_the_real_dimensions():
    # The regression fence for the estimator itself: it already scales, it was simply never called
    # with anything. 1088x1920 is 1.99x the area of 1024x1024, so the headroom must be ~2x.
    base = estimate_image_runtime_mib(width = 1024, height = 1024)
    tall = estimate_image_runtime_mib(width = 1088, height = 1920)
    assert base == 8192
    assert tall == 16_320
    assert 1.95 < tall / base < 2.05
    # And with the distilled discount that the base repo now contributes, the report's number.
    assert estimate_image_runtime_mib(width = 1088, height = 1920, family = _TURBO_HINT) == 13_872


def test_guard_refuses_the_oversized_frame_and_passes_the_default_one():
    # 13,872 MiB of working memory against a 13,822 MiB budget on the reported card: refuse.
    message = _shortfall(1088, 1920)
    assert message is not None
    # Everything the user needs to act: what they asked for, what it costs, what they have.
    assert "1088x1920" in message
    assert "13.55 GB" in message  # needed
    assert "13.50 GB" in message  # usable
    assert "15.50 GB" in message  # currently free
    assert "smaller resolution" in message
    assert "UNSLOTH_DIFFUSION_ALLOW_OVERSIZED_GENERATE" in message
    # The same card at the default resolution needs 6963 MiB and must go straight through.
    assert _shortfall(1024, 1024) is None


def test_guard_never_refuses_at_or_below_the_resolution_the_load_planned_for():
    # The load's flat headroom is a PLANNING figure: it picks an offload tier, and the tier it
    # picks runs 1024x1024 on cards whose entire budget is below that figure. Treating it as a
    # hard limit there would refuse generations that complete today, so the guard is confined to
    # requests LARGER than what was planned. An 8 GB card is the case that proves it.
    small = _discrete(int(8 * 1024 * 0.97), 8 * 1024)  # safe budget 5898 MiB, under the 6963 default
    assert _shortfall(1024, 1024, memory = small) is None
    assert _shortfall(512, 512, memory = small) is None
    # It still refuses the genuinely oversized frame on that same card.
    assert _shortfall(1088, 1920, memory = small) is not None


def test_guard_scales_with_batch_size():
    # Batch multiplies the activations exactly as area does, so the same overrun must be caught.
    assert _shortfall(1024, 1024, batch_size = 1) is None
    assert _shortfall(1024, 1024, batch_size = 4) is not None
    assert "at a batch of 4" in _shortfall(1024, 1024, batch_size = 4)


def test_guard_is_skipped_on_unified_memory():
    # Offload means something different where the CPU and GPU share one pool, "free" is a moving
    # target shared with the OS, and the load-time unified refusal already owns that device class.
    unified = DeviceMemory("cuda", "cuda", "unified_memory", _16G_FREE_MIB, _16G_TOTAL_MIB)
    assert _shortfall(1088, 1920, memory = unified) is None
    mps = DeviceMemory("mps", "mps", "unified_memory", _16G_FREE_MIB, _16G_TOTAL_MIB)
    assert _shortfall(1088, 1920, memory = mps) is None


def test_guard_is_skipped_when_free_memory_is_unknown():
    # No reading, no verdict: the planner's own rule for an unknown budget is to stay out of it.
    blind = DeviceMemory("cuda", "cuda", "discrete_vram", None, _16G_TOTAL_MIB)
    assert _shortfall(1088, 1920, memory = blind) is None


def test_guard_is_skipped_off_cuda_and_rocm():
    # ROCm reports device "cuda", so that stays covered. XPU / CPU keep today's behaviour: their
    # allocators differ and this estimate was measured against a discrete VRAM pool.
    xpu = DeviceMemory("xpu", "xpu", "discrete_vram", _16G_FREE_MIB, _16G_TOTAL_MIB)
    assert _shortfall(1088, 1920, memory = xpu) is None
    cpu = DeviceMemory("cpu", "cpu", "system_memory", _16G_FREE_MIB, _16G_TOTAL_MIB)
    assert _shortfall(1088, 1920, memory = cpu) is None


def test_guard_env_override_lets_an_oversized_generation_through(monkeypatch):
    from core.inference.diffusion_memory import OVERSIZED_GENERATE_ENV

    for value in ("1", "true", "YES", " on "):
        monkeypatch.setenv(OVERSIZED_GENERATE_ENV, value)
        assert _shortfall(1088, 1920) is None, value
    # Anything else is not an override, so the refusal stands.
    for value in ("0", "false", "", "maybe"):
        monkeypatch.setenv(OVERSIZED_GENERATE_ENV, value)
        assert _shortfall(1088, 1920) is not None, value


def test_guard_fails_open_when_the_probe_raises():
    # A broken probe must never cost a user a generation that would have worked.
    class _Exploding:
        device = "cuda"
        memory_kind = "discrete_vram"
        is_unified = False
        total_mib = _16G_TOTAL_MIB

        @property
        def free_mib(self):
            raise RuntimeError("mem_get_info exploded")

    assert _shortfall(1088, 1920, memory = _Exploding()) is None


def test_raiser_raises_valueerror_so_the_route_answers_400():
    # ValueError, not RuntimeError: /images/generate maps ValueError to a 400 carrying the reason,
    # while RuntimeError there is reserved for the not-loaded / cancelled sentinels and otherwise
    # becomes an opaque 500 with the reason stripped.
    from core.inference.diffusion_memory import raise_on_image_activation_shortfall

    with pytest.raises(ValueError, match = "1088x1920"):
        raise_on_image_activation_shortfall(
            device_memory = _discrete(_16G_FREE_MIB, _16G_TOTAL_MIB),
            width = 1088,
            height = 1920,
            family = _TURBO_HINT,
        )
    # And it is a no-op wherever the message function declines to produce a verdict.
    raise_on_image_activation_shortfall(
        device_memory = _discrete(_16G_FREE_MIB, _16G_TOTAL_MIB),
        width = 1024,
        height = 1024,
        family = _TURBO_HINT,
    )


def _zimage_plan_on(memory, text_encoder_dense_mib):
    return plan_diffusion_memory(
        target = _target(),
        device_memory = memory,
        model_dense_mib = _ZIMAGE_MODEL_DENSE_MIB,
        companion_dense_mib = _ZIMAGE_COMPANION_MIB,
        text_encoder_dense_mib = text_encoder_dense_mib,
        runtime_headroom_mib = 8192,
        base_overhead_mib = 2048,
    )


def test_default_resolution_plans_identically_across_card_sizes():
    # The cross-check between the two fixes: at the default resolution the guard is silent on every
    # card, and the plan a discrete CUDA target reaches with no text-encoder split is the plan it
    # reached before either change. Neither fix leaks into the other's territory.
    from core.inference.diffusion_memory import _safe_device_budget_mib

    for gigabytes in (8, 12, 16, 24, 32, 48, 80):
        total = gigabytes * 1024
        memory = _discrete(int(total * 0.97), total)
        assert _shortfall(1024, 1024, memory = memory) is None, gigabytes
        expected = _legacy_offload_policy(
            budget = _safe_device_budget_mib(memory),
            model_dense_mib = _ZIMAGE_MODEL_DENSE_MIB,
            companion_dense_mib = _ZIMAGE_COMPANION_MIB,
            runtime_headroom_mib = 8192,
            base_overhead_mib = 2048,
        )
        assert _zimage_plan_on(memory, None).offload_policy == expected, gigabytes
