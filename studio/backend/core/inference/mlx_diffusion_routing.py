# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Per-request engine routing on Apple Silicon: MLX for plain text-to-image, diffusers MPS for the rest.

When both engines do not fit the Metal working set, MLX is unloaded first and later requests stay on diffusers
(no ping-pong reloads) until the model is loaded again. ``UNSLOTH_DIFFUSION_MLX=0`` never builds this engine;
``=1`` runs the MLX engine alone.
"""

from __future__ import annotations

import inspect
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

from core.inference.generation_ticket import bind_generation_cancel
from loggers import get_logger
from utils.account_context import current_account_id

logger = get_logger(__name__)

ENGINE_MLX = "mlx"
ENGINE_DIFFUSERS = "diffusers"

COEXIST_ENV = "UNSLOTH_DIFFUSION_MLX_COEXIST"
# How long unload() / a new load waits for an in-flight fallback load to tear itself down.
_FALLBACK_DRAIN_S = 120.0
_ENABLE_TOKENS = frozenset({"1", "on", "true", "yes"})
_DISABLE_TOKENS = frozenset({"0", "off", "false", "no"})
# Only these delegate the precision choice. none/off pin the released weights, which MLX cannot honour.
_UNSET_TOKENS = frozenset({"", "auto", "default"})


@dataclass(frozen = True)
class MlxCapabilities:
    """What the MLX engine runs for a family. Everything outside it is served by diffusers."""

    workflows: tuple[str, ...] = ("txt2img",)
    loras: bool = False
    controlnet: bool = False


_DEFAULT_CAPABILITIES = MlxCapabilities()
# Per family name; a family absent here gets the default (text-to-image only).
MLX_CAPABILITIES: dict[str, MlxCapabilities] = {"z-image": _DEFAULT_CAPABILITIES}


def mlx_capabilities(family: Optional[str]) -> MlxCapabilities:
    return MLX_CAPABILITIES.get((family or "").lower(), _DEFAULT_CAPABILITIES)


def _explicit(value: Any) -> bool:
    return value is not None and str(value).strip().lower() not in _UNSET_TOKENS


def _positive(value: Any) -> bool:
    try:
        return value is not None and float(value) > 0.0
    except (TypeError, ValueError):
        return False


def load_blocker(
    *,
    loras: Any = None,
    component_files: Any = None,
    transformer_quant: Optional[str] = None,
    text_encoder_quant: Optional[str] = None,
) -> Optional[str]:
    """The first LOAD option the MLX engine cannot honour, else None. Such a load stays on diffusers."""
    if loras:
        return "LoRA adapters selected at load time need the diffusers engine"
    if component_files:
        return "separate text-encoder / VAE files need the diffusers engine"
    if _explicit(transformer_quant):
        return f"an explicit transformer precision ({transformer_quant}) needs the diffusers engine"
    if _explicit(text_encoder_quant):
        return (
            f"an explicit text-encoder precision ({text_encoder_quant}) needs the diffusers engine"
        )
    return None


# Generate kwargs a request may carry that change nothing on a text-to-image MLX run.
_IGNORED_ON_MLX = frozenset({"live_preview", "mask_blur", "inpaint_only_masked", "inpaint_padding"})
_CHECKED = frozenset(
    {
        "hires",
        "upscale_model",
        "workflow",
        "init_image",
        "mask_image",
        "reference_images",
        "reference_resolution",
        "localized_edit",
        "upscale",
        "prompts",
        "seeds",
        "loras",
        "controlnet",
        "sampler",
        "schedule_type",
        "sd_controls",
        "shift",
        "base_shift",
        "max_shift",
    }
)


def _controls_requested(controls: Any) -> bool:
    requested = getattr(controls, "requested", None)
    try:
        return bool(requested()) if callable(requested) else False
    except Exception:  # noqa: BLE001 -- unreadable controls are treated as asked for
        return True


def request_blocker(
    kwargs: dict[str, Any],
    *,
    family: Optional[str] = None,
    mlx_params: Optional[frozenset[str]] = None,
    mlx_guidance: Optional[bool] = None,
) -> Optional[str]:
    """What in a generate request MLX cannot serve, else None. Options outside ``mlx_params`` are diffusers-only.
    ``mlx_guidance``: whether the resident MLX variant applies CFG (None = unknown, not checked)."""
    caps = mlx_capabilities(family)
    get = kwargs.get
    if mlx_guidance is False and _positive(get("guidance")):
        # A guidance-distilled MLX variant (Z-Image-Turbo) drops CFG and the negative prompt; diffusers runs them.
        return "classifier-free guidance (and its negative prompt)"
    workflow = get("workflow")
    if workflow not in (None, "txt2img") and workflow not in caps.workflows:
        return f"the {workflow} workflow"
    if get("hires") is not None:
        return "Hires fix"
    if get("upscale_model"):
        return "a model upscaler"
    if get("mask_image") is not None:
        return "inpainting"
    if get("init_image") is not None:
        return "an input image (img2img)"
    if get("reference_images"):
        return "reference images"
    if get("reference_resolution") is not None:
        return "a reference resolution"
    if get("localized_edit") is not None:
        return "a localized edit"
    upscale = get("upscale")
    if upscale is not None and upscale > 1:
        return "upscaling"
    if get("prompts") is not None or get("seeds") is not None:
        return "per-image prompt or seed lists"
    if get("loras") and not caps.loras:
        return "LoRA adapters"
    controlnet = get("controlnet")
    if controlnet is not None and controlnet[3] not in (None, 0, 0.0) and not caps.controlnet:
        return "ControlNet"
    if get("sampler") not in (None, "default") or get("schedule_type") not in (None, "automatic"):
        return "a sampler choice"
    if get("sd_controls") is not None and _controls_requested(get("sd_controls")):
        return "prompt weighting, CLIP skip, CFG rescale or a variation seed"
    if any(get(name) is not None for name in ("shift", "base_shift", "max_shift")):
        return "a shift override"
    if mlx_params is not None:
        for name, value in kwargs.items():
            if name in mlx_params or name in _IGNORED_ON_MLX or name in _CHECKED:
                continue
            if value not in (None, False, [], ()):
                return f"the {name} option"
    return None


@dataclass(frozen = True)
class Route:
    engine: str
    reason: str
    load_diffusers: bool = False
    unload_mlx: bool = False


def decide_route(
    *,
    blocker: Optional[str],
    mlx_resident: bool,
    diffusers_resident: bool,
    fallback_available: bool,
    coexist: bool,
) -> Route:
    """Pure routing decision for one request. ``coexist``: both engines fit the working set together."""
    if blocker is None:
        if mlx_resident:
            return Route(ENGINE_MLX, "text-to-image runs on MLX")
        if diffusers_resident:
            return Route(
                ENGINE_DIFFUSERS,
                "MLX was unloaded to make room for diffusers; load the model again to return to MLX",
            )
        # Nothing resident (a load in flight, or none): the MLX engine answers with its own not-loaded error.
        return Route(ENGINE_MLX, "no model resident")
    if diffusers_resident:
        return Route(ENGINE_DIFFUSERS, f"{blocker} runs on diffusers")
    if not fallback_available or not mlx_resident:
        # MLX refuses with its own message, or reports that nothing is loaded.
        return Route(ENGINE_MLX, f"{blocker} has no diffusers fallback here")
    return Route(
        ENGINE_DIFFUSERS,
        f"{blocker} runs on diffusers",
        load_diffusers = True,
        unload_mlx = not coexist,
    )


def engines_coexist(mlx_bytes: int, diffusers_bytes: int, budget_bytes: int) -> bool:
    """Whether the MLX model (its plan includes the activation headroom) and a diffusers copy fit together.
    Unknown sizes never fit. ``UNSLOTH_DIFFUSION_MLX_COEXIST=1`` / ``0`` overrides."""
    raw = os.environ.get(COEXIST_ENV, "auto").strip().lower()
    if raw in _ENABLE_TOKENS:
        return True
    if raw in _DISABLE_TOKENS:
        return False
    if mlx_bytes <= 0 or diffusers_bytes <= 0 or budget_bytes <= 0:
        return False
    return mlx_bytes + diffusers_bytes <= budget_bytes


def diffusers_resident_bytes(gguf_path: Optional[str], components_dir: Optional[str]) -> int:
    """diffusers keeps the GGUF as stored plus a bf16 text encoder and VAE; 0 when unreadable."""
    if not gguf_path or not components_dir:
        return 0
    try:
        from core.inference.mlx_diffusion_backend import _safetensors_numel
        root = Path(components_dir)
        return (
            os.path.getsize(gguf_path)
            + 2 * _safetensors_numel(root / "text_encoder")
            + 2 * _safetensors_numel(root / "vae")
        )
    except Exception as exc:  # noqa: BLE001 -- unknown size: treated as not fitting
        logger.info("mlx_routing.diffusers_size_unknown: %s", exc)
        return 0


def _default_mlx() -> Any:
    from core.inference.mlx_diffusion_backend import get_mlx_diffusion_backend
    return get_mlx_diffusion_backend()


def _default_diffusers() -> Any:
    from core.inference.diffusion import get_diffusion_backend
    return get_diffusion_backend()


def _default_budget() -> int:
    from core.inference.mlx_diffusion_backend import working_set_bytes
    return working_set_bytes()


def _diffusers_capabilities(family: Any) -> dict[str, Any]:
    from core.inference import diffusion as diffusers_module
    from core.inference.diffusion_conditioning import conditioning_capabilities

    workflows = diffusers_module._family_workflows(family)
    return {
        "workflows": workflows,
        "conditioning": conditioning_capabilities(family, workflows),
        **diffusers_module._hires_status(family),
        # A request with guidance routes to diffusers when MLX drops CFG, so the Images page offers its controls.
        "supports_guidance": bool(getattr(family, "uses_guidance", False)),
        "supports_negative_prompt": bool(getattr(family, "uses_negative_prompt", False)),
        "negative_prompt_min_guidance": diffusers_module._negative_prompt_guidance_threshold(
            family
        ),
    }


# Taken from the fallback only when the MLX variant drops CFG; one that applies it keeps its own thresholds.
_GUIDANCE_KEYS = ("supports_guidance", "supports_negative_prompt", "negative_prompt_min_guidance")


def _parameters(fn: Any) -> dict[str, inspect.Parameter]:
    try:
        return dict(inspect.signature(fn).parameters)
    except (TypeError, ValueError):
        return {}


def _identity_of(engine: Any) -> Any:
    from core.inference.diffusion_families import load_identity

    try:
        status = engine.status()
    except Exception:  # noqa: BLE001
        return None
    if not status.get("loaded"):
        return None
    return load_identity(status.get("repo_id"), status.get("base_repo"), status.get("family"))


class _RoutedGenerate:
    """``generate`` with both engines' keywords, so the route's signature checks pass diffusers-only options through."""

    def __init__(self, owner: "RoutedMlxEngine") -> None:
        self.__self__ = owner

    @property
    def __signature__(self) -> inspect.Signature:
        owner = self.__self__
        params: dict[str, inspect.Parameter] = {}
        try:
            params.update(_parameters(owner._diffusers().generate))
        except Exception:  # noqa: BLE001 -- no diffusers: MLX's keywords only
            pass
        for name, param in _parameters(owner._mlx().generate).items():
            params.setdefault(name, param)
        ordered = [p for p in params.values() if p.kind is not inspect.Parameter.VAR_KEYWORD]
        return inspect.Signature([p.replace(kind = inspect.Parameter.KEYWORD_ONLY) for p in ordered])

    def __call__(self, **kwargs: Any) -> dict[str, Any]:
        return self.__self__._generate(**kwargs)


class RoutedMlxEngine:
    """The image engine behind ``ENGINE_MLX`` in auto mode. Same duck-typed surface as the other engines."""

    runs_off_torch_device = False
    parses_prompt_attention = False

    def __init__(
        self,
        mlx_factory: Optional[Callable[[], Any]] = None,
        diffusers_factory: Optional[Callable[[], Any]] = None,
        budget_fn: Optional[Callable[[], int]] = None,
        capabilities_fn: Optional[Callable[[Any], dict[str, Any]]] = None,
        poll_s: float = 0.25,
    ) -> None:
        self._mlx_factory = mlx_factory or _default_mlx
        self._diffusers_factory = diffusers_factory or _default_diffusers
        self._budget_fn = budget_fn or _default_budget
        self._capabilities_fn = capabilities_fn or _diffusers_capabilities
        self._poll_s = poll_s
        self._route_lock = threading.Lock()
        self._lock = threading.Lock()
        self._load_kwargs: Optional[dict[str, Any]] = None
        self._epoch = 0
        # Replacement loads between their first epoch bump and registering their options: no fallback registers.
        self._replacing = 0
        # diffusers holds a fallback copy of THIS load (the diffusers singleton is shared with plain diffusers loads).
        self._diffusers_ours = False
        self._mlx_dropped = False
        self._origin_identity: Any = None
        self._coexist: Optional[bool] = None
        self._serving: Optional[str] = None
        self._last: Optional[dict[str, Any]] = None
        self._fallback_cancel: Optional[threading.Event] = None
        # Set when the in-flight fallback load has finished or torn itself down; unload() waits on it.
        self._fallback_done: Optional[threading.Event] = None
        self._active_generate_account: Optional[str] = None
        self.generate = _RoutedGenerate(self)

    def _mlx(self) -> Any:
        return self._mlx_factory()

    def _diffusers(self) -> Any:
        return self._diffusers_factory()

    def _diffusers_resident(self) -> bool:
        return self._diffusers_ours and getattr(self._diffusers(), "_state", None) is not None

    def _mlx_resident(self) -> bool:
        return getattr(self._mlx(), "_state", None) is not None

    def _primary(self) -> Any:
        return self._diffusers() if self._diffusers_resident() else self._mlx()

    @property
    def _state(self) -> Any:
        return getattr(self._primary(), "_state", None)

    @property
    def is_loaded(self) -> bool:
        return self._state is not None

    def __getattr__(self, name: str) -> Any:
        if name.startswith("__") or name in (
            "_mlx_factory",
            "_diffusers_factory",
            "_lock",
            "_diffusers_ours",
        ):
            raise AttributeError(name)
        return getattr(self._primary(), name)

    def _family(self) -> Any:
        state = getattr(self._mlx(), "_state", None)
        if state is not None:
            return getattr(state, "family", None)
        state = getattr(self._diffusers(), "_state", None) if self._diffusers_ours else None
        return getattr(state, "family", None)

    def _fallback_available(self, family: Any) -> bool:
        if self._load_kwargs is None or family is None:
            return False
        try:
            from core.inference.diffusion_families import family_pipeline_available
            return bool(family_pipeline_available(family))
        except Exception:  # noqa: BLE001
            return False

    def _engines_fit_together(self) -> bool:
        state = getattr(self._mlx(), "_state", None)
        if state is None:
            return False
        plan = (getattr(state, "timings", None) or {}).get("quant_plan") or {}
        mlx_bytes = int(plan.get("planned_bytes") or 0)
        diffusers_bytes = diffusers_resident_bytes(
            getattr(state, "gguf_path", None), getattr(state, "convert_dir", None)
        )
        try:
            budget = int(self._budget_fn() or 0)
        except Exception:  # noqa: BLE001
            budget = 0
        fits = engines_coexist(mlx_bytes, diffusers_bytes, budget)
        logger.info(
            "mlx_routing.coexist: mlx=%.2f GiB diffusers=%.2f GiB budget=%.2f GiB -> %s",
            mlx_bytes / 1024**3,
            diffusers_bytes / 1024**3,
            budget / 1024**3,
            fits,
        )
        return fits

    def _drain_fallback(self, *, replacing: bool = False) -> bool:
        """Cancel an in-flight fallback load and wait until it has torn itself down; returns ``_diffusers_ours``.
        ``replacing`` raises the replacement fence in the same step as the epoch bump."""
        with self._lock:
            self._epoch += 1
            if replacing:
                self._replacing += 1
            cancel = self._fallback_cancel
            done = self._fallback_done
        if cancel is not None:
            cancel.set()
        if done is not None and not done.wait(_FALLBACK_DRAIN_S):
            logger.warning("mlx_routing.fallback_drain_timeout: %.0fs", _FALLBACK_DRAIN_S)
        with self._lock:
            return self._diffusers_ours

    def begin_load(self, repo_id: str, **kwargs: Any) -> dict[str, Any]:
        """A new load replaces everything: the old model's diffusers copy goes first, then MLX loads."""
        # Fenced until this load's options are saved: a request routed meanwhile saw the old model resident and
        # would load the old options into diffusers (or unload the new MLX load).
        ours = self._drain_fallback(replacing = True)
        try:
            if ours:
                self._diffusers().unload()
            with self._lock:
                self._diffusers_ours = False
                self._mlx_dropped = False
                self._origin_identity = None
                self._coexist = None
                self._last = None
            self._mlx().begin_load(repo_id, **kwargs)
            with self._lock:
                self._load_kwargs = {"repo_id": repo_id, **kwargs}
        finally:
            with self._lock:
                # A second bump: a route decided inside the fence holds a stale epoch once it lifts.
                self._epoch += 1
                self._replacing -= 1
        return self.status()

    def _diffusers_load_kwargs(self) -> dict[str, Any]:
        kwargs = dict(self._load_kwargs or {})
        accepted = _parameters(self._diffusers().begin_load)
        if not any(p.kind is inspect.Parameter.VAR_KEYWORD for p in accepted.values()):
            kwargs = {k: v for k, v in kwargs.items() if k in accepted}
        return kwargs

    def _load_fallback(self, *, unload_mlx: bool, epoch: int) -> Any:
        """Load the diffusers copy of this model and wait for it, unloading MLX first when both do not fit.
        Returns the diffusers copy's identity, read while ``epoch`` was still current."""
        from core.inference.diffusion_families import DIFFUSION_CANCELLED_MSG
        from core.inference.diffusion_memory import (
            UNIFIED_OVERSIZE_REFUSAL_CODE,
            UnifiedMemoryShortfallError,
        )

        mlx, diffusers = self._mlx(), self._diffusers()
        cancel, done = threading.Event(), threading.Event()
        # Epoch check and registration are one step: a new load either sees this fallback in
        # _drain_fallback (and waits for it) or this request sees the new epoch and touches nothing.
        with self._lock:
            if self._epoch != epoch or self._replacing:
                raise RuntimeError(DIFFUSION_CANCELLED_MSG)
            self._fallback_cancel = cancel
            self._fallback_done = done
            self._active_generate_account = current_account_id()
        # A stopped tool ticket must reach the load too; the engine binds its own event only once generation starts.
        bind_generation_cancel(cancel)
        dropped = False
        started = time.perf_counter()
        try:
            kwargs = self._diffusers_load_kwargs()
            repo_id = kwargs.pop("repo_id")
            if cancel.is_set() or self._epoch != epoch:
                raise RuntimeError(DIFFUSION_CANCELLED_MSG)
            if unload_mlx:
                self._origin_identity = _identity_of(mlx) or self._origin_identity
                mlx.unload()
                dropped = True
                with self._lock:
                    self._mlx_dropped = True
            if cancel.is_set() or self._epoch != epoch:
                raise RuntimeError(DIFFUSION_CANCELLED_MSG)
            diffusers.begin_load(repo_id, **kwargs)
            with self._lock:
                self._diffusers_ours = True
            while True:
                # Cancellation first: a load that turned ready after an unload or cancel is not served.
                if cancel.is_set() or self._epoch != epoch:
                    diffusers.unload()
                    raise RuntimeError(DIFFUSION_CANCELLED_MSG)
                progress = diffusers.load_progress() or {}
                phase = progress.get("phase")
                if phase == "ready" and getattr(diffusers, "_state", None) is not None:
                    break
                if phase == "error":
                    message = (
                        "Could not load the diffusers engine this request needs: "
                        f"{progress.get('error') or 'unknown error'}"
                    )
                    if progress.get("error_code") == UNIFIED_OVERSIZE_REFUSAL_CODE:
                        # Typed so the route answers with the structured refusal, not a 500.
                        raise UnifiedMemoryShortfallError(message)
                    raise RuntimeError(message)
                if phase is None:
                    diffusers.unload()
                    raise RuntimeError(DIFFUSION_CANCELLED_MSG)
                time.sleep(self._poll_s)
            identity = _identity_of(diffusers)
            if cancel.is_set() or self._epoch != epoch:
                diffusers.unload()
                raise RuntimeError(DIFFUSION_CANCELLED_MSG)
            logger.info(
                "mlx_routing.fallback_loaded: diffusers in %.1fs (mlx unloaded first: %s)",
                time.perf_counter() - started,
                dropped,
            )
            return identity
        except BaseException:
            with self._lock:
                self._diffusers_ours = False
                restore = dropped and self._epoch == epoch and self._load_kwargs is not None
                if restore:
                    self._mlx_dropped = False
            if restore:
                # The model was MLX's before this request: bring it back rather than leave nothing loaded.
                try:
                    restore_kwargs = dict(self._load_kwargs or {})
                    mlx.begin_load(restore_kwargs.pop("repo_id"), **restore_kwargs)
                except Exception as exc:  # noqa: BLE001 -- the request's own error is the one to report
                    logger.warning("mlx_routing.restore_failed: %s", exc)
            raise
        finally:
            with self._lock:
                if self._fallback_cancel is cancel:
                    self._fallback_cancel = None
                    self._fallback_done = None
                    self._active_generate_account = None
            done.set()

    def _mlx_guidance(self) -> Optional[bool]:
        try:
            value = self._mlx().status().get("supports_guidance")
        except Exception:  # noqa: BLE001 -- unknown: not checked
            return None
        return None if value is None else bool(value)

    def _identity_snapshot(self) -> dict[str, Any]:
        """Each resident engine's identity, read once: the request validates and forwards these same values."""
        out: dict[str, Any] = {}
        if self._mlx_resident():
            out[ENGINE_MLX] = _identity_of(self._mlx())
        if self._diffusers_resident():
            out[ENGINE_DIFFUSERS] = _identity_of(self._diffusers())
        return {k: v for k, v in out.items() if v is not None}

    def _accepted_identities(self, snapshot: Optional[dict[str, Any]] = None) -> list[Any]:
        ids = list((self._identity_snapshot() if snapshot is None else snapshot).values())
        if self._origin_identity is not None and ids:
            ids.append(self._origin_identity)
        return ids

    def _generate(self, **kwargs: Any) -> dict[str, Any]:
        from core.inference.diffusion_families import (
            DIFFUSION_NOT_LOADED_MSG,
            DiffusionModelReplacedError,
        )
        expected = kwargs.pop("expected_load", None)
        with self._route_lock:
            with self._lock:
                epoch = self._epoch
            mlx, diffusers = self._mlx(), self._diffusers()
            mlx_params = frozenset(_parameters(mlx.generate))
            family = self._family()
            mlx_resident = self._mlx_resident()
            blocker = request_blocker(
                kwargs,
                family = getattr(family, "name", None),
                mlx_params = mlx_params,
                mlx_guidance = self._mlx_guidance() if mlx_resident else None,
            )
            diffusers_resident = self._diffusers_resident()
            fallback_available = self._fallback_available(family)
            coexist = False
            if blocker and mlx_resident and not diffusers_resident and fallback_available:
                coexist = self._engines_fit_together()
                self._coexist = coexist
            route = decide_route(
                blocker = blocker,
                mlx_resident = mlx_resident,
                diffusers_resident = diffusers_resident,
                fallback_available = fallback_available,
                coexist = coexist,
            )
            if blocker and route.engine == ENGINE_MLX and not mlx_resident:
                # Nothing to serve it from (no model, or one still loading): the 409, not MLX's feature refusal.
                raise RuntimeError(DIFFUSION_NOT_LOADED_MSG)
            snapshot: dict[str, Any] = {}
            if expected is not None:
                snapshot = self._identity_snapshot()
                accepted = self._accepted_identities(snapshot)
                if accepted and expected not in accepted:
                    raise DiffusionModelReplacedError(expected, accepted[0])
            if route.load_diffusers and (mlx.load_progress() or {}).get("phase") != "ready":
                # A replacement load is pending (or failed) with the old model still resident: the saved load
                # options are the replacement's, so a fallback now would serve a different model.
                raise RuntimeError(
                    "A diffusion load is in progress; retry once it finishes or load the model again."
                )
            if blocker and route.engine == ENGINE_MLX and mlx_resident:
                mlx_call = {k: v for k, v in kwargs.items() if k in mlx_params}
                if request_blocker(mlx_call, family = getattr(family, "name", None)) is None:
                    # The option MLX lacks would be filtered out before MLX could refuse it: refuse here.
                    raise ValueError(
                        f"{blocker[0].upper()}{blocker[1:]} is not supported on the MLX engine, and "
                        "this install has no diffusers pipeline for the model to fall back to."
                    )
            # The served engine's identity from the validated snapshot (or the fallback copy of that load), never
            # a later read: a load replacing the model after the check is then refused by the engine itself.
            own = snapshot.get(route.engine, expected) if expected is not None else None
            if route.load_diffusers:
                logger.info("mlx_routing.fallback: %s", route.reason)
                fallback_identity = self._load_fallback(unload_mlx = route.unload_mlx, epoch = epoch)
                if expected is not None:
                    own = fallback_identity or expected
            engine = diffusers if route.engine == ENGINE_DIFFUSERS else mlx
            if route.engine == ENGINE_DIFFUSERS:
                call = kwargs
            else:
                call = {k: v for k, v in kwargs.items() if k in mlx_params}
            with self._lock:
                self._serving = route.engine
            try:
                result = engine.generate(expected_load = own, **call)
            finally:
                with self._lock:
                    self._serving = None
            out = dict(result)
            out["engine"] = route.engine
            with self._lock:
                self._last = {"served_by": route.engine, "reason": route.reason}
            return out

    def generate_progress(self) -> dict[str, Any]:
        if self._fallback_cancel is not None:
            return {
                "active": True,
                "step": 0,
                "total_steps": 0,
                "fraction": 0.0,
                "eta_seconds": None,
                "phase": None,
            }
        if self._serving == ENGINE_DIFFUSERS:
            return self._diffusers().generate_progress()
        if self._serving == ENGINE_MLX:
            return self._mlx().generate_progress()
        return self._primary().generate_progress()

    def cancel_generate(self, expected_account: Optional[str] = None) -> bool:
        with self._lock:
            cancel = self._fallback_cancel
            owner = self._active_generate_account
        if cancel is not None:
            if expected_account is not None and owner not in (None, expected_account):
                return False
            cancel.set()
            return True
        if self._serving == ENGINE_DIFFUSERS:
            return self._diffusers().cancel_generate(expected_account = expected_account)
        return self._mlx().cancel_generate(expected_account = expected_account)

    def unload(self, *, expected_account: Optional[str] = None) -> dict[str, Any]:
        # The fallback drains first and stays fenced until the teardown ends: one failing after MLX was freed would
        # otherwise see its own epoch and reload MLX, and none may register while both engines are torn down.
        ours = self._drain_fallback(replacing = True)
        try:
            if expected_account is None:
                self._mlx().unload()
            else:
                self._mlx().unload(expected_account = expected_account)
            if ours:
                if expected_account is None:
                    self._diffusers().unload()
                else:
                    self._diffusers().unload(expected_account = expected_account)
            with self._lock:
                self._diffusers_ours = False
                self._mlx_dropped = False
                self._origin_identity = None
                self._coexist = None
                self._load_kwargs = None
        finally:
            with self._lock:
                self._epoch += 1
                self._replacing -= 1
        return self.status()

    def load_progress(self) -> dict[str, Any]:
        if self._fallback_cancel is not None or self._diffusers_resident():
            return self._diffusers().load_progress()
        return self._mlx().load_progress()

    def _repo_ids(self, method: str) -> tuple[str, ...]:
        engines = [self._mlx()]
        if self._diffusers_ours:
            engines.append(self._diffusers())
        out: list[str] = []
        for engine in engines:
            fn = getattr(engine, method, None)
            if callable(fn):
                out.extend(fn())
        return tuple(dict.fromkeys(out))

    def loading_repo_ids(self) -> tuple[str, ...]:
        return self._repo_ids("loading_repo_ids")

    def draining_repo_ids(self) -> tuple[str, ...]:
        return self._repo_ids("draining_repo_ids")

    def loaded_repo_ids(self) -> tuple[str, ...]:
        return self._repo_ids("loaded_repo_ids")

    def download_plan(self, repo_id: str, **kwargs: Any) -> dict[str, Any]:
        # MLX opens the diffusers GGUF plan's files, which are also what the diffusers fallback opens.
        return self._mlx().download_plan(repo_id, **kwargs)

    def preflight_base_access(self, *args: Any, **kwargs: Any) -> Any:
        return self._mlx().preflight_base_access(*args, **kwargs)

    def _fallback_capabilities(self, family: Any) -> dict[str, Any]:
        """What the diffusers copy would add, so the Images page offers those workflows while MLX is resident."""
        if not self._fallback_available(family):
            return {}
        try:
            return dict(self._capabilities_fn(family))
        except Exception as exc:  # noqa: BLE001 -- the MLX status stands on its own
            logger.info("mlx_routing.capabilities_unavailable: %s", exc)
            return {}

    def routing_status(self) -> dict[str, Any]:
        resident = [
            name
            for name, here in (
                (ENGINE_MLX, self._mlx_resident()),
                (ENGINE_DIFFUSERS, self._diffusers_resident()),
            )
            if here
        ]
        with self._lock:
            last = dict(self._last) if self._last else None
            dropped = self._mlx_dropped
            coexist = self._coexist
        return {
            "mode": "auto",
            "default_engine": ENGINE_MLX,
            "resident": resident,
            "txt2img_engine": ENGINE_DIFFUSERS
            if dropped and not self._mlx_resident()
            else ENGINE_MLX,
            "engines_fit_together": coexist,
            "last_served_by": last["served_by"] if last else None,
            "last_reason": last["reason"] if last else None,
        }

    def status(self) -> dict[str, Any]:
        if self._diffusers_resident():
            out = dict(self._diffusers().status())
        else:
            out = dict(self._mlx().status())
            if out.get("loaded"):
                caps = self._fallback_capabilities(self._family())
                if out.get("supports_guidance"):
                    for key in _GUIDANCE_KEYS:
                        caps.pop(key, None)
                out.update(caps)
        out["engine_routing"] = self.routing_status()
        return out


_routed_mlx_engine: Optional[RoutedMlxEngine] = None
_singleton_lock = threading.Lock()


def get_routed_mlx_engine() -> RoutedMlxEngine:
    global _routed_mlx_engine
    with _singleton_lock:
        if _routed_mlx_engine is None:
            _routed_mlx_engine = RoutedMlxEngine()
        return _routed_mlx_engine
