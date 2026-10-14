# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The optimization plan: which steps a load will try, in what order, and why a step is declined.

Every decision here is the SAME predicate the primitives apply (compile_eligible, the fp16 U-Net offload rule,
family flags, the graph layer's eligibility), evaluated up front so the audit can name the reason. The primitives
still decide at apply time; the plan never overrides a registered family. For a family with no data the plan uses
the denoiser profile to decline a compile up front (no repeated blocks, graph-break suspects in a discovered block)
rather than letting it fail inside the load.

:func:`compile_cache_kwargs` is the one resolver for the compile-cache fingerprint kwargs (load-time and deferred
image compile, and the opt-in video compile cache), so the bundle keys on what the compile will actually use.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from . import targets
from .profile import ModelProfile, profile_model

# Load order of the optimizer steps (design doc, section A). Names match the audit / ``speed_optims`` keys.
STEP_ORDER: tuple[str, ...] = (
    "patches",
    "attention_backend",
    "step_cache",
    "compile_cache",
    "vae_single_frame",
    "channels_last",
    "vae_fp16_decode",
    "vae_fused",
    "cudnn_benchmark",
    "fp16_accum",
    "fused_qkv",
    "compiled",
    "compiled_dequant",
    "int8_gemm",
    "compiled_vae_decode",
    "tf32",
    "rocm_query_chunks",
    "cuda_graph",
    "block_graphs",
    "bg_compile",
)

GRAPH_BREAK_DECLINE = "graph-break suspects in a discovered block forward"

SPEED_OFF = "off"
SPEED_EAGER = "eager"
SPEED_DEFAULT = "default"
SPEED_MAX = "max"


@dataclass
class PlannedStep:
    name: str
    attempt: bool
    reason: Optional[str] = None


@dataclass
class OptimizationPlan:
    backend: str
    family: Optional[str]
    speed_mode: str
    steps: list[PlannedStep] = field(default_factory = list)
    profiles: list[ModelProfile] = field(default_factory = list)
    generic: bool = False

    def step(self, name: str) -> Optional[PlannedStep]:
        for s in self.steps:
            if s.name == name:
                return s
        return None

    def reason(self, name: str) -> Optional[str]:
        s = self.step(name)
        return s.reason if s is not None else None


def family_has_data(family: Any) -> bool:
    """Whether ``family`` comes from a registry (image or video); a bare / synthesized family takes the generic path."""
    name = getattr(family, "name", None)
    if not isinstance(name, str) or not name:
        return False
    try:
        from ..diffusion_families import _FAMILIES as image_families  # noqa: PLC0415
        from ..video_families import _FAMILIES as video_families  # noqa: PLC0415
    except Exception:  # noqa: BLE001
        return True
    return any(f is family or f.name == name for f in tuple(image_families) + tuple(video_families))


# Generic path (a family with no registry data): compile discovered repeated blocks under the runtime guard.
DISCOVER_ENV = "UNSLOTH_DIFFUSION_AUTOOPT_DISCOVER"


def discovery_enabled() -> bool:
    import os  # noqa: PLC0415
    return (os.environ.get(DISCOVER_ENV) or "").strip().lower() not in ("0", "off", "false", "no")


def compile_ineligible_reason(target: Any, family: Any) -> Optional[str]:
    """Why ``compile_eligible`` refuses, mirroring its checks in order; None when eligible."""
    from .. import diffusion_speed as ds  # noqa: PLC0415

    if not ds.torch_compile_runtime_available():
        return "torch.compile unavailable in this process (TORCHDYNAMO_DISABLE or no Triton)"
    if not bool(getattr(target, "supports_default_torch_compile", False)):
        return "device does not support the default torch.compile profile"
    if not bool(getattr(family, "supports_torch_compile", True)):
        return "family opts out of torch.compile"
    rocm_failed = "ROCm regional compile probe failed on this install"
    dtype = getattr(target, "dtype", None)
    if ds._is_bfloat16(dtype):
        return None if ds.rocm_compile_probe_passes(target) else rocm_failed
    if bool(getattr(family, "fp16_incompatible", False)):
        return "fp16-incompatible family stays eager outside bf16"
    if ds._is_float16(dtype) and ds._fp16_compile_capable(target):
        return None if ds.rocm_compile_probe_passes(target) else rocm_failed
    return f"dtype {str(dtype).replace('torch.', '')} is not compiled on this device"


def build_plan(
    pipe: Any,
    target: Any,
    *,
    family: Any,
    backend: str,
    speed_mode: str,
    is_gguf: bool = False,
    offload_active: bool = False,
) -> OptimizationPlan:
    """Predict each step's decision and reason from the same predicates the primitives use."""
    from .. import diffusion_speed as ds  # noqa: PLC0415

    mode = ds.normalize_speed_mode(speed_mode)
    generic = not family_has_data(family)
    dits = targets.denoiser_dits(pipe)
    candidates = dits or [m for m in [targets.denoiser_unet(pipe)] if m is not None]
    if generic and not candidates:
        candidates = targets.discovered_only(pipe, family)
    profiles = [
        p for p in (profile_model(m) for m in candidates if hasattr(m, "forward")) if p is not None
    ]
    plan = OptimizationPlan(
        backend = backend,
        family = getattr(family, "name", None),
        speed_mode = mode,
        profiles = profiles,
        generic = generic,
    )

    def add(
        name: str,
        attempt: bool,
        reason: Optional[str] = None,
    ) -> None:
        plan.steps.append(PlannedStep(name, attempt, reason))

    if mode == SPEED_OFF:
        for name in STEP_ORDER:
            if name not in ("patches", "attention_backend", "step_cache"):
                add(name, False, "speed off")
        return plan

    on_cuda = getattr(target, "device", None) == "cuda"
    compile_reason: Optional[str] = None
    if mode == SPEED_EAGER:
        compile_reason = "eager tier does not compile"
    elif (
        mode == SPEED_DEFAULT
        and is_gguf
        and on_cuda
        and getattr(family, "supports_torch_compile", True)
        and not ds.rocm_compile_switched_off(target)
    ):
        compile_reason = "GGUF compiles the dequant op chain instead"
    else:
        compile_reason = compile_ineligible_reason(target, family)
        if compile_reason is None and ds.fp16_unet_offloaded(
            target, pipe, offload_active = offload_active
        ):
            compile_reason = "offloaded fp16 U-Net stays eager (fused QKV costs more transfer than compile saves)"
        if compile_reason is None and not candidates:
            compile_reason = "no denoiser transformer or whole-compile U-Net"
        if compile_reason is None and dits:
            declines = [p.compile_decline() for p in profiles]
            if generic and discovery_enabled():
                # No data: discovered (unverified) blocks get a guarded trial compile instead of a decline.
                declines = [
                    None if (p.discovered_blocks and not p.repeated_blocks) else d
                    for p, d in zip(profiles, declines)
                ]
            if declines and all(declines):
                compile_reason = declines[0]
    add("compiled", compile_reason is None, compile_reason)
    if generic and compile_reason is None:
        # No measured data: a discovered block whose forward host-syncs would graph-break on every call.
        for p in profiles:
            if not p.declared_blocks and p.graph_break_suspects:
                suspects = "; ".join(
                    f"{k}: {', '.join(v)}" for k, v in p.graph_break_suspects.items()
                )
                step = plan.step("compiled")
                step.attempt = False
                step.reason = f"{GRAPH_BREAK_DECLINE} ({p.class_name}: {suspects})"
    rocm_off = ds.rocm_compile_switched_off(target)
    add(
        "compiled_dequant",
        bool(is_gguf and on_cuda and not rocm_off),
        "not a GGUF transformer" if not is_gguf else ("UNSLOTH_DIFFUSION_ROCM_COMPILE=0" if rocm_off else None),
    )
    add("tf32", mode == SPEED_MAX and on_cuda, None if mode == SPEED_MAX else "max tier only")
    if not getattr(family, "cudnn_benchmark", True):
        add(
            "cudnn_benchmark",
            False,
            "family keeps cudnn.benchmark off (per-process conv pick changes pixels)",
        )
    elif not on_cuda:
        add("cudnn_benchmark", False, "not a CUDA device")
    else:
        add("cudnn_benchmark", True)
    return plan


def compile_cache_kwargs(
    pipe: Any,
    *,
    speed_mode: str,
    fullgraph: bool,
    offload_below_hooks: bool,
    deferred: bool = False,
) -> dict[str, Any]:
    """The compile kwargs the compile cache fingerprints. ``deferred``: the image backend's 3rd-image profile, which
    always compiles the ``default`` tier with explicit ``dynamic=True``."""
    from .. import diffusion_speed as ds  # noqa: PLC0415

    mode = SPEED_DEFAULT if deferred else ds.normalize_speed_mode(speed_mode)
    transformer = getattr(pipe, "transformer", None)
    kwargs: dict[str, Any] = {
        # A step cache or planned offload graph-breaks, so the bundle keys on the fullgraph the compile uses.
        "fullgraph": bool(fullgraph),
        # max compiles DiTs with automatic dynamic (None), as does the default tier for torchao weights.
        "dynamic": ds.compile_dynamic(
            transformer, True if deferred else (None if mode == SPEED_MAX else True)
        ),
        "mode": "max-autotune-no-cudagraphs" if mode == SPEED_MAX else "default",
        "vae_decode": ds.vae_decode_compile_allowed(pipe, mode),
    }
    if offload_below_hooks:
        # An offloaded denoiser compiles its blocks below the hooks, a different graph than a traced-hook bundle.
        kwargs["below_hooks"] = True
    return kwargs
