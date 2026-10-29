# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""run_speed_layer, the one image/video speed-layer entry: runs patch stages, family hooks and
apply_speed_optims per pipe or view, recording each step in the load Audit; failures caught from
logged warnings; audit surfaces in status as resolved[speed_mode][steps]."""

from __future__ import annotations

import contextvars
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Optional

from . import patches, plan as plan_mod, targets

DISCOVER_ENV = plan_mod.DISCOVER_ENV
# Video loads do not take the image backend's persistent compile cache yet: it repoints the process-wide
# TORCHINDUCTOR_CACHE_DIR, which an image backend loaded in the same server may be compiling under, and its save /
# restore lifecycle is not wired into the video state. Recorded so the status says so.
VIDEO_COMPILE_CACHE_DECLINE = (
    "not wired for video loads yet (process-wide inductor cache dir shared with the image backend)"
)

_CURRENT_AUDIT: contextvars.ContextVar = contextvars.ContextVar(
    "unsloth_autoopt_audit", default = None
)


# ---------------------------------------------------------------------------------------------------------- audit
@dataclass
class Audit:
    """One record per load: ``{step: {"status", "reason"}}`` in first-seen order."""

    backend: str = "image"
    family: Optional[str] = None
    entries: dict[str, dict] = field(default_factory = dict)
    generic: bool = False

    def record(
        self,
        step: str,
        status: str,
        reason: Optional[str] = None,
    ) -> None:
        prev = self.entries.get(step)
        # A failure is never overwritten by a later "declined"; an engagement upgrades a decline (dual-DiT views).
        if prev is not None:
            if prev["status"] == "failed" and status != "engaged":
                return
            if prev["status"] == "engaged" and status == "declined":
                return
        self.entries[step] = {"status": status, "reason": reason}

    def status(self, step: str) -> Optional[str]:
        entry = self.entries.get(step)
        return entry["status"] if entry else None

    def as_list(self) -> list[dict]:
        return [{"step": k, **v} for k, v in self.entries.items()]

    def engaged(self) -> list[str]:
        return [k for k, v in self.entries.items() if v["status"] == "engaged"]


def current_audit() -> Optional[Audit]:
    return _CURRENT_AUDIT.get()


def new_audit(*, backend: str, family: Any) -> Audit:
    return Audit(backend = backend, family = getattr(family, "name", None))


class _AuditingLogger:
    """Forwards everything to ``inner``; records each ``"<prefix>: %s failed: %s"`` warning the primitives log."""

    def __init__(self, inner: Any, audit: Audit) -> None:
        self._inner = inner
        self._audit = audit

    def warning(self, msg: Any, *args: Any, **kwargs: Any) -> None:
        try:
            if isinstance(msg, str) and msg.endswith("%s failed: %s") and len(args) >= 2:
                self._audit.record(str(args[0]), "failed", str(args[1])[:300])
        except Exception:  # noqa: BLE001
            pass
        if self._inner is not None:
            self._inner.warning(msg, *args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        inner = self.__dict__.get("_inner")
        if inner is None:

            def _noop(*a: Any, **k: Any) -> None:
                return None

            return _noop
        return getattr(inner, name)


def auditing_logger(logger: Any, audit: Optional[Audit]) -> Any:
    if audit is None or isinstance(logger, _AuditingLogger):
        return logger
    return _AuditingLogger(logger, audit)


# ------------------------------------------------------------------------------------------------- family hooks
@dataclass(frozen = True)
class FamilyHook:
    """Per-family code that must run around the speed primitive for one view. ``applies(family)`` gates it; ``pre``
    runs before ``apply_speed_optims``; ``post(view, family, applied, logger)`` may return an updated applied dict."""

    name: str
    applies: Callable[[Any], bool]
    pre: Optional[Callable[[Any, Any], None]] = None
    post: Optional[Callable[[Any, Any, dict, Any], Optional[dict]]] = None
    backends: tuple[str, ...] = ("video",)


def _profile_flag(attr: str) -> Callable[[Any], bool]:
    """Gate a hook on a ``FamilyProfile`` flag (registry data), so a new family opts in by data."""

    def applies(family: Any) -> bool:
        try:
            from ..family_profile import profile_for  # noqa: PLC0415
        except Exception:  # noqa: BLE001
            return False
        return bool(getattr(profile_for(family), attr, False))

    return applies


def _ltx_static_pre(view: Any, family: Any) -> None:
    # Token counts depend only on (w, h, frames), so static compile: one recompile per shape, ~1.3x faster.
    dit = getattr(view, getattr(family, "denoiser_attr", "transformer"), None)
    if dit is not None:
        try:
            dit._unsloth_compile_static = True
        except Exception:  # noqa: BLE001 -- not a settable module (tests/fakes)
            pass


def _ltx_stg_post(view: Any, family: Any, applied: dict, logger: Any) -> Optional[dict]:
    if applied.get("compiled"):
        from ..video_ltx2 import install_stg_compile_adapter  # noqa: PLC0415
        install_stg_compile_adapter(
            getattr(view, getattr(family, "denoiser_attr", "transformer"), None)
        )
    return None


def _cudnn_off_post(view: Any, family: Any, applied: dict, logger: Any) -> Optional[dict]:
    if applied.get("cudnn_benchmark"):
        from ..video_ltx2 import disable_cudnn_benchmark  # noqa: PLC0415
        if disable_cudnn_benchmark():
            return {**applied, "cudnn_benchmark": False}
    return None


FAMILY_HOOKS: list[FamilyHook] = [
    FamilyHook("ltx_static_compile", _profile_flag("compile_static"), pre = _ltx_static_pre),
    FamilyHook("ltx_stg_compile_adapter", _profile_flag("stg_compile_adapter"), post = _ltx_stg_post),
    FamilyHook(
        "cudnn_benchmark_off",
        lambda fam: not getattr(fam, "cudnn_benchmark", True),
        post = _cudnn_off_post,
    ),
]


def register_family_hook(hook: FamilyHook) -> None:
    for i, existing in enumerate(FAMILY_HOOKS):
        if existing.name == hook.name:
            FAMILY_HOOKS[i] = hook
            return
    FAMILY_HOOKS.append(hook)


# ------------------------------------------------------------------------------------------------ patch stages
def install_patches(
    stage: str,
    *,
    pipe: Any,
    target: Any = None,
    family: Any = None,
    dtype: Any = None,
    device: Any = None,
    speed_mode: Optional[str] = None,
    backend: str = "image",
    audit: Optional[Audit] = None,
    logger: Any = None,
) -> dict:
    """Run one load-time patch stage (``load_eager`` / ``load``) for ``pipe``."""
    ctx = patches.PatchContext(
        pipe = pipe,
        target = target,
        family = family,
        dtype = dtype,
        device = device,
        speed_mode = speed_mode,
        backend = backend,
        logger = logger,
    )
    return patches.install_stage(stage, ctx, audit = audit)


def install_compile_patches(
    stage: str,
    transformer: Any,
    logger: Any,
    *,
    offload_active: bool,
    denoiser_offloaded: Optional[bool],
) -> dict:
    """Per-denoiser patch stage inside the regional compile (``pre_int8`` / ``pre_compile``)."""
    ctx = patches.PatchContext(
        transformer = transformer,
        offload_active = offload_active,
        denoiser_offloaded = denoiser_offloaded,
        logger = logger,
    )
    return patches.install_stage(stage, ctx, audit = current_audit())


# ------------------------------------------------------------------------------------------------ the speed layer
def _generic_family_view(
    view: Any, family: Any, plan: plan_mod.OptimizationPlan, logger: Any
) -> Any:
    """For a family with no data: trial-compile discovered blocks, or decline the compile with the plan's reason."""
    step = plan.step("compiled")
    if (
        step is not None
        and not step.attempt
        and str(step.reason or "").startswith(plan_mod.GRAPH_BREAK_DECLINE)
    ):
        # Graph-break suspects in a discovered block: decline rather than compile a graph that breaks every call.
        if logger is not None:
            logger.info("diffusion.autoopt: compile declined for %s: %s", plan.family, step.reason)
        return _family_with(family, supports_torch_compile = False)
    if step is not None and step.attempt and plan_mod.discovery_enabled():
        for dit in targets.denoiser_dits(view):
            if getattr(dit, "_repeated_blocks", None):
                continue
            prof = next((p for p in plan.profiles if p.class_name == type(dit).__name__), None)
            if prof is not None and prof.discovered_blocks and not prof.graph_break_suspects:
                try:
                    dit._repeated_blocks = list(prof.discovered_blocks)
                    if logger is not None:
                        logger.info(
                            "diffusion.autoopt: trial regional compile of discovered blocks %s on %s "
                            "(guarded: a failure falls back to eager)",
                            ", ".join(prof.discovered_blocks),
                            type(dit).__name__,
                        )
                except Exception:  # noqa: BLE001
                    pass
    return family


def _family_with(family: Any, **overrides: Any) -> Any:
    try:
        return replace(family, **overrides)
    except Exception:  # noqa: BLE001 - not a dataclass: a thin proxy

        class _Proxy:
            def __getattr__(self, name: str) -> Any:
                if name in overrides:
                    return overrides[name]
                return getattr(family, name)

        return _Proxy()


def run_speed_layer(
    view: Any,
    target: Any,
    *,
    family: Any,
    backend: str,
    speed_mode: str,
    apply: Callable[[Any, Any], dict],
    is_gguf: bool = False,
    offload_active: bool = False,
    audit: Optional[Audit] = None,
    logger: Any = None,
) -> dict[str, bool]:
    """Plan + apply the speed primitives for one pipe (image) or one denoiser view (video).

    ``apply(family, logger)`` runs the primitive (the backends' ``apply_speed_optims(...)`` call, so its arguments
    stay where the load computes them); the optimizer supplies the family it should see (a registered family is
    passed through untouched; a family with no data may get its compile declined) and the auditing logger. Family
    hooks run around it. Returns the primitive's dict after the hooks, exactly what the backends recorded before."""
    plan: Optional[plan_mod.OptimizationPlan]
    try:
        plan = plan_mod.build_plan(
            view,
            target,
            family = family,
            backend = backend,
            speed_mode = speed_mode,
            is_gguf = is_gguf,
            offload_active = offload_active,
        )
    except Exception as exc:  # noqa: BLE001 - the plan explains; it never fails a load
        if logger is not None:
            logger.warning("diffusion.autoopt: plan unavailable, applying without it: %s", exc)
        plan = None
    if audit is not None and plan is not None:
        audit.generic = audit.generic or plan.generic
    hooks = [h for h in FAMILY_HOOKS if backend in h.backends and _safe_applies(h, family)]
    for hook in hooks:
        if hook.pre is not None:
            hook.pre(view, family)
    family_for_apply = family
    if (
        plan is not None
        and plan.compile_declined_by_model
        and not is_gguf
        and getattr(target, "backend", None) == "rocm"
    ):
        # The primitive probes the card before it looks at the model: pass the decline so a model with nothing to
        # compile never pays the ROCm probe.
        family_for_apply = _family_with(family, supports_torch_compile = False)
    if plan is not None and plan.generic:
        try:
            family_for_apply = _generic_family_view(view, family_for_apply, plan, logger)
        except Exception as exc:  # noqa: BLE001 - the generic trial is optional; the stock path still runs
            if logger is not None:
                logger.warning("diffusion.autoopt: generic path unavailable: %s", exc)
    token = _CURRENT_AUDIT.set(audit)
    try:
        applied = apply(family_for_apply, auditing_logger(logger, audit))
    finally:
        _CURRENT_AUDIT.reset(token)
    for hook in hooks:
        if hook.post is not None:
            updated = hook.post(view, family, applied, logger)
            if updated is not None:
                applied = updated
    if audit is not None:
        try:
            record_applied(audit, applied, plan, view)
        except Exception:  # noqa: BLE001 - the audit never fails a load
            pass
    return applied


def _safe_applies(hook: FamilyHook, family: Any) -> bool:
    try:
        return bool(hook.applies(family))
    except Exception:  # noqa: BLE001
        return False


def record_applied(
    audit: Audit,
    applied: dict,
    plan: Optional[plan_mod.OptimizationPlan],
    view: Any = None,
) -> None:
    """Fold an ``apply_speed_optims`` result into the audit with the plan's reasons for what did not engage."""
    for name, on in applied.items():
        if on:
            audit.record(name, "engaged")
            continue
        reason = plan.reason(name) if plan is not None else None
        if name == "cuda_graph" and view is not None:
            reason = getattr(view, "_unsloth_cuda_graph_reason", None) or reason
        if reason is None and plan is not None and plan.speed_mode == plan_mod.SPEED_OFF:
            reason = "speed off"
        audit.record(name, "declined", reason or "not applicable to this model / tier")


def record_load_steps(
    audit: Optional[Audit],
    *,
    attention: Optional[str] = None,
    step_cache: Optional[str] = None,
    step_cache_reason: Optional[str] = None,
    compile_cache: Optional[bool] = None,
    compile_cache_reason: Optional[str] = None,
    speed_deferred: bool = False,
) -> None:
    """Record the load-path steps that run outside ``run_speed_layer`` (attention backend, step cache, compile
    cache)."""
    if audit is None:
        return
    audit.record(
        "attention_backend",
        "engaged" if attention else "declined",
        attention if attention else "diffusers default attention",
    )
    audit.record(
        "step_cache",
        "engaged" if step_cache else "declined",
        step_cache if step_cache else (step_cache_reason or "off"),
    )
    if compile_cache is not None:
        audit.record(
            "compile_cache",
            "engaged" if compile_cache else "declined",
            None
            if compile_cache
            else compile_cache_reason
            if compile_cache_reason
            else "deferred: engages with the compile profile on the 3rd image"
            if speed_deferred
            else "no compile on this load",
        )


def finalize(
    audit: Optional[Audit],
    speed_optims: Any,
    resolved: Optional[dict] = None,
) -> Optional[list]:
    """Reconcile the audit with the final ``speed_optims`` (post-placement arming may engage or drop a step) and attach
    it to ``resolved["speed_mode"]["steps"]``. Returns the step list; never raises."""
    if audit is None:
        return None
    try:
        return _finalize(audit, speed_optims, resolved)
    except Exception:  # noqa: BLE001 - the audit never fails a load
        return None


def _finalize(audit: Audit, speed_optims: Any, resolved: Optional[dict]) -> list:
    final = set(speed_optims or ())
    for name, entry in list(audit.entries.items()):
        if name in final and entry["status"] != "engaged":
            audit.entries[name] = {"status": "engaged", "reason": None}
        elif (
            entry["status"] == "engaged"
            and name in plan_mod.STEP_ORDER
            and name not in final
            and _is_speed_key(name)
        ):
            audit.entries[name] = {"status": "declined", "reason": "disengaged after placement"}
    for name in final:
        if name not in audit.entries:
            audit.entries[name] = {"status": "engaged", "reason": None}
    steps = audit.as_list()
    if isinstance(resolved, dict):
        entry = resolved.get("speed_mode")
        if isinstance(entry, dict):
            entry["steps"] = steps
    return steps


_SPEED_KEYS = frozenset(
    {
        "channels_last",
        "vae_fp16_decode",
        "vae_single_frame",
        "vae_fused",
        "cudnn_benchmark",
        "tf32",
        "fp16_accum",
        "fused_qkv",
        "compiled",
        "compiled_dequant",
        "rocm_query_chunks",
        "compiled_vae_decode",
        "cuda_graph",
        "int8_gemm",
    }
)


def _is_speed_key(name: str) -> bool:
    return name in _SPEED_KEYS
