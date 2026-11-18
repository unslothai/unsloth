# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Apple Silicon routing: MLX for plain text-to-image, diffusers MPS for the rest (MLX unloads if both do not fit)."""

from __future__ import annotations

import inspect
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Optional

from core.inference.generation_ticket import bind_generation_cancel
from loggers import get_logger
from utils.account_context import current_account_id

logger = get_logger(__name__)

ENGINE_MLX = "mlx"
ENGINE_DIFFUSERS = "diffusers"

COEXIST_ENV = "UNSLOTH_DIFFUSION_MLX_COEXIST"
_FALLBACK_DRAIN_S = 120.0
_ENABLE_TOKENS = frozenset({"1", "on", "true", "yes"})
_DISABLE_TOKENS = frozenset({"0", "off", "false", "no"})
_UNSET_TOKENS = frozenset({"", "auto", "default"})


@dataclass(frozen = True)
class MlxCapabilities:
    workflows: tuple[str, ...] = ("txt2img",)
    loras: bool = False
    controlnet: bool = False


_DEFAULT_CAPABILITIES = MlxCapabilities()
MLX_CAPABILITIES: dict[str, MlxCapabilities] = {"z-image": MlxCapabilities(loras = True)}


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
    component_files: Any = None,
    transformer_quant: Optional[str] = None,
    text_encoder_quant: Optional[str] = None,
) -> Optional[str]:
    """Load options MLX cannot honour. LoRAs are not one: ``mlx_selection`` weighs them per family."""
    if component_files:
        return "separate text-encoder / VAE files need the diffusers engine"
    if _explicit(transformer_quant):
        return f"an explicit transformer precision ({transformer_quant}) needs the diffusers engine"
    if _explicit(text_encoder_quant):
        return (
            f"an explicit text-encoder precision ({text_encoder_quant}) needs the diffusers engine"
        )
    return None


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
    except Exception:  # noqa: BLE001
        return True


def request_blocker(
    kwargs: dict[str, Any],
    *,
    family: Optional[str] = None,
    mlx_params: Optional[frozenset[str]] = None,
    mlx_guidance: Optional[bool] = None,
) -> Optional[str]:
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
    if blocker is None:
        if mlx_resident:
            return Route(ENGINE_MLX, "text-to-image runs on MLX")
        if diffusers_resident:
            return Route(
                ENGINE_DIFFUSERS,
                "MLX was unloaded to make room for diffusers; load the model again to return to MLX",
            )
        return Route(ENGINE_MLX, "no model resident")
    if diffusers_resident:
        return Route(ENGINE_DIFFUSERS, f"{blocker} runs on diffusers")
    if not fallback_available or not mlx_resident:
        return Route(ENGINE_MLX, f"{blocker} has no diffusers fallback here")
    return Route(
        ENGINE_DIFFUSERS,
        f"{blocker} runs on diffusers",
        load_diffusers = True,
        unload_mlx = not coexist,
    )


def engines_coexist(mlx_bytes: int, diffusers_bytes: int, budget_bytes: int) -> bool:
    raw = os.environ.get(COEXIST_ENV, "auto").strip().lower()
    if raw in _ENABLE_TOKENS:
        return True
    if raw in _DISABLE_TOKENS:
        return False
    if mlx_bytes <= 0 or diffusers_bytes <= 0 or budget_bytes <= 0:
        return False
    return mlx_bytes + diffusers_bytes <= budget_bytes


def diffusers_resident_bytes(gguf_path: Optional[str], components_dir: Optional[str]) -> int:
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
    except Exception as exc:  # noqa: BLE001
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


_SCHEDULER_CONFIG = "scheduler/scheduler_config.json"


def _cached_base_file(base: str, filename: str) -> Optional[Path]:
    local = Path(base).expanduser()
    if local.is_dir():
        candidate = local / filename
        return candidate if candidate.is_file() else None
    from huggingface_hub import constants, try_to_load_from_cache

    from core.inference.diffusion_families import (
        mirror_repo,
        pinned_revision_kwargs,
        prefer_ungated_mirror,
    )
    from utils.hf_cache_settings import active_hf_hub_cache

    repos = (prefer_ungated_mirror(base), base, mirror_repo(base))
    for repo in dict.fromkeys(r for r in repos if r):
        revision = pinned_revision_kwargs(repo).get("revision")
        for cache_dir in dict.fromkeys((active_hf_hub_cache(), constants.HF_HUB_CACHE)):
            hit = try_to_load_from_cache(repo, filename, cache_dir = cache_dir, revision = revision)
            if isinstance(hit, str):
                return Path(hit)
    return None


def fallback_shift_control(family: Any, status: dict[str, Any]) -> Optional[dict]:
    import json

    from core.inference.diffusion_content import content_variant_hint
    from core.inference.diffusion_families import comfy_flow_shift_for, comfy_shift_node_for
    from core.inference.diffusion_flow_shift import (
        SAMPLE_SIGMAS_KEY,
        apply_comfy_flow_shift,
        shift_control,
        valid_sample_sigmas,
    )

    repo_id, gguf_filename = status.get("repo_id"), status.get("gguf_filename")
    display_repo_id, base = status.get("display_repo_id"), status.get("base_repo")
    node = comfy_shift_node_for(family, gguf_filename, repo_id, display_repo_id, base)
    if node is None or not base:
        return None
    index_path = _cached_base_file(base, "model_index.json")
    if index_path is not None:
        index = json.loads(index_path.read_text(encoding = "utf-8"))
        if isinstance(index, dict) and valid_sample_sigmas(index.get(SAMPLE_SIGMAS_KEY)):
            return None
    config_path = _cached_base_file(base, _SCHEDULER_CONFIG)
    if config_path is None:
        return None
    config = json.loads(config_path.read_text(encoding = "utf-8"))
    import diffusers

    scheduler_cls = getattr(diffusers, str(config.get("_class_name") or ""), None)
    if scheduler_cls is None:
        return None
    pipe = SimpleNamespace(scheduler = scheduler_cls.from_config(config))
    apply_comfy_flow_shift(
        pipe,
        comfy_flow_shift_for(
            family,
            gguf_filename,
            content_variant_hint(repo_id, gguf_filename),
            repo_id,
            display_repo_id,
            base,
        ),
    )
    return shift_control(node, pipe.scheduler)


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
    def __init__(self, owner: "RoutedMlxEngine") -> None:
        self.__self__ = owner

    @property
    def __signature__(self) -> inspect.Signature:
        owner = self.__self__
        params: dict[str, inspect.Parameter] = {}
        try:
            params.update(_parameters(owner._diffusers().generate))
        except Exception:  # noqa: BLE001
            pass
        for name, param in _parameters(owner._mlx().generate).items():
            params.setdefault(name, param)
        ordered = [p for p in params.values() if p.kind is not inspect.Parameter.VAR_KEYWORD]
        return inspect.Signature([p.replace(kind = inspect.Parameter.KEYWORD_ONLY) for p in ordered])

    def __call__(self, **kwargs: Any) -> dict[str, Any]:
        return self.__self__._generate(**kwargs)


class FallbackLoadRefusedError(RuntimeError):
    """A gate a normal load passes refused the on-demand diffusers load; ``http`` is that load's refusal."""

    def __init__(self, http: Any) -> None:
        self.http = http
        detail = getattr(http, "detail", None)
        super().__init__(detail if isinstance(detail, str) else str(http))


class RoutedMlxEngine:
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
        self._fallback_gate: Optional[Callable[..., Any]] = None
        self._epoch = 0
        # The logical load's epoch (media_load_ids reads it as _load_token): moved by begin_load and
        # unload only, never by a fallback, whose engines keep counters of their own.
        self._logical_load = 0
        # Replacement loads between their first epoch bump and registering their options: no fallback registers.
        self._replacing = 0
        self._diffusers_ours = False
        self._mlx_dropped = False
        self._origin_identity: Any = None
        self._coexist: Optional[bool] = None
        self._serving: Optional[str] = None
        self._last: Optional[dict[str, Any]] = None
        self._fallback_cancel: Optional[threading.Event] = None
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
    def _load_token(self) -> int:
        return self._logical_load

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

    def begin_load(
        self,
        repo_id: str,
        *,
        fallback_gate: Optional[Callable[..., Any]] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        # Fenced until this load's options are saved, so no request routes on the old model's options.
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
                # Only once MLX accepted it: a refused load leaves the in-flight load's epoch alone.
                self._logical_load += 1
                self._load_kwargs = {"repo_id": repo_id, **kwargs}
                self._fallback_gate = fallback_gate
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
        from core.inference.diffusion_families import DIFFUSION_CANCELLED_MSG
        from core.inference.diffusion_memory import (
            UNIFIED_OVERSIZE_REFUSAL_CODE,
            UnifiedMemoryShortfallError,
        )

        mlx, diffusers = self._mlx(), self._diffusers()
        cancel, done = threading.Event(), threading.Event()
        # Epoch check and registration are atomic: a new load drains this fallback, or this request sees the new epoch.
        with self._lock:
            if self._epoch != epoch or self._replacing:
                raise RuntimeError(DIFFUSION_CANCELLED_MSG)
            self._fallback_cancel = cancel
            self._fallback_done = done
            self._active_generate_account = current_account_id()
            gate = self._fallback_gate
        bind_generation_cancel(cancel)
        dropped = False
        started = time.perf_counter()
        try:
            kwargs = self._diffusers_load_kwargs()
            repo_id = kwargs.pop("repo_id")
            if cancel.is_set() or self._epoch != epoch:
                raise RuntimeError(DIFFUSION_CANCELLED_MSG)

            def _start() -> None:
                nonlocal dropped
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

            if gate is None:
                _start()
            else:
                gate(_start, cancel = cancel)
            while True:
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
                try:
                    restore_kwargs = dict(self._load_kwargs or {})
                    mlx.begin_load(restore_kwargs.pop("repo_id"), **restore_kwargs)
                except Exception as exc:  # noqa: BLE001
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
        except Exception:  # noqa: BLE001
            return None
        return None if value is None else bool(value)

    def _identity_snapshot(self) -> dict[str, Any]:
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
        from core.inference.diffusion_memory import UnifiedMemoryShortfallError

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
                raise RuntimeError(DIFFUSION_NOT_LOADED_MSG)
            snapshot: dict[str, Any] = {}
            if expected is not None:
                snapshot = self._identity_snapshot()
                accepted = self._accepted_identities(snapshot)
                if accepted and expected not in accepted:
                    raise DiffusionModelReplacedError(expected, accepted[0])
            if route.load_diffusers and (mlx.load_progress() or {}).get("phase") != "ready":
                raise RuntimeError(
                    "A diffusion load is in progress; retry once it finishes or load the model again."
                )
            if blocker and route.engine == ENGINE_MLX and mlx_resident:
                mlx_call = {k: v for k, v in kwargs.items() if k in mlx_params}
                if request_blocker(mlx_call, family = getattr(family, "name", None)) is None:
                    raise ValueError(
                        f"{blocker[0].upper()}{blocker[1:]} is not supported on the MLX engine, and "
                        "this install has no diffusers pipeline for the model to fall back to."
                    )
            own = snapshot.get(route.engine, expected) if expected is not None else None
            if route.load_diffusers:
                logger.info("mlx_routing.fallback: %s", route.reason)
                try:
                    fallback_identity = self._load_fallback(
                        unload_mlx = route.unload_mlx, epoch = epoch
                    )
                except UnifiedMemoryShortfallError:
                    if route.unload_mlx:
                        raise
                    # Coexist estimate ignores other residents; the loader measured real free memory.
                    logger.info("mlx_routing.coexist_refused: retrying with MLX unloaded")
                    self._coexist = False
                    fallback_identity = self._load_fallback(unload_mlx = True, epoch = epoch)
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

    def unload(
        self,
        *,
        expected_account: Optional[str] = None,
        only_if: Optional[Callable[[], bool]] = None,
    ) -> Optional[dict[str, Any]]:
        # A conditional unload (a load token naming the load): refused before the drain, which would
        # end the fallback that load runs on; the MLX engine rechecks it under the lock its worker
        # settles under. None, and nothing torn down, when the named load already ended.
        if only_if is not None and not only_if():
            return None
        # Drain the fallback first and stay fenced until teardown ends, so none reloads MLX or registers meanwhile.
        if expected_account is not None:
            with self._lock:
                foreign = (
                    self._fallback_cancel is not None
                    and self._active_generate_account
                    not in (
                        None,
                        expected_account,
                    )
                )
            if foreign:
                # Refused before the drain: draining cancels the other account's in-flight fallback.
                from core.inference.gpu_arbiter import DIFFUSION, GpuBusyForAnotherAccountError
                raise GpuBusyForAnotherAccountError(DIFFUSION, 1)
        ours = self._drain_fallback(replacing = True)
        try:
            kwargs = {} if expected_account is None else {"expected_account": expected_account}
            if only_if is not None:
                kwargs["only_if"] = only_if
            if self._mlx().unload(**kwargs) is None and only_if is not None:
                return None
            if ours:
                if expected_account is None:
                    self._diffusers().unload()
                else:
                    self._diffusers().unload(expected_account = expected_account)
            with self._lock:
                self._logical_load += 1
                self._diffusers_ours = False
                self._mlx_dropped = False
                self._origin_identity = None
                self._coexist = None
                self._load_kwargs = None
                self._fallback_gate = None
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
        return self._mlx().download_plan(repo_id, **kwargs)

    def preflight_base_access(self, *args: Any, **kwargs: Any) -> Any:
        return self._mlx().preflight_base_access(*args, **kwargs)

    def _fallback_capabilities(self, family: Any) -> dict[str, Any]:
        if not self._fallback_available(family):
            return {}
        try:
            return dict(self._capabilities_fn(family))
        except Exception as exc:  # noqa: BLE001
            logger.info("mlx_routing.capabilities_unavailable: %s", exc)
            return {}

    def _fallback_shift_control(self, family: Any, status: dict[str, Any]) -> Optional[dict]:
        if not self._fallback_available(family):
            return None
        try:
            return fallback_shift_control(family, status)
        except Exception as exc:  # noqa: BLE001
            logger.info("mlx_routing.shift_control_unavailable: %s", exc)
            return None

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
                family = self._family()
                caps = self._fallback_capabilities(family)
                if out.get("supports_guidance"):
                    for key in _GUIDANCE_KEYS:
                        caps.pop(key, None)
                out.update(caps)
                if out.get("shift_control") is None:
                    shift = self._fallback_shift_control(family, out)
                    if shift is not None:
                        out["shift_control"] = shift
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
