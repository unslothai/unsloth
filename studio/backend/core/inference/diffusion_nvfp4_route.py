# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Routes an explicit NVFP4 image / video pick to INT8 (or Auto) on GPUs without FP4 tensor cores.

NVFP4 only has a native path on NVIDIA Blackwell (compute capability major >= 10). Below that it can only run
weight-only (4-bit codes widened to bf16 per layer, bf16 GEMM), which measured both slower than INT8 tensor cores and
less accurate than the INT8 ConvRot builds. So on pre-Blackwell NVIDIA an explicit NVFP4 pick loads the family's
normal INT8 instead (hosted INT8 checkpoint where registered, else on-the-fly int8), and on ROCm / XPU / MPS / CPU it
takes whatever Auto picks there. Blackwell is untouched.

This is the one decision every entry point asks (the image / video load and download-plan routes, and the backends'
``begin_load`` for direct callers), so a pre-Blackwell pick never plans or downloads an NVFP4 file. The device is read
once per process; tests replace ``_probe_device_profile``. ``UNSLOTH_NVFP4_PRE_BLACKWELL_INT8=0`` turns it off.
"""

from __future__ import annotations

import contextvars
import functools
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Optional

from .diffusion_nvfp4_flag import is_nvfp4

logger = logging.getLogger(__name__)

ROUTE_ENV = "UNSLOTH_NVFP4_PRE_BLACKWELL_INT8"
_FALSY = frozenset({"0", "false", "no", "off"})

# NVIDIA compute capability major of the first FP4 tensor-core generation (sm_100 / sm_103 / sm_110 / sm_120 / sm_121).
BLACKWELL_MAJOR = 10
# INT8 tensor cores (and Studio's int8 path) start at Turing sm_75.
INT8_MIN_CAPABILITY = (7, 5)
# The text-encoder int8 mode needs Ampere sm_80+ (``diffusion_precision.te_quant_supported``).
TE_INT8_MIN_MAJOR = 8

NVIDIA, ROCM, XPU, MPS, CPU = "nvidia", "rocm", "xpu", "mps", "cpu"

INT8 = "int8"
AUTO = "auto"

REASON_INT8 = "NVFP4 needs a Blackwell GPU; using INT8"
REASON_AUTO = "NVFP4 needs a Blackwell GPU; using Auto"

TRANSFORMER_QUANT = "transformer_quant"
TEXT_ENCODER_QUANT = "text_encoder_quant"


def route_enabled() -> bool:
    """False only for an explicit falsy ``UNSLOTH_NVFP4_PRE_BLACKWELL_INT8``; read at call time."""
    raw = os.environ.get(ROUTE_ENV)
    return raw is None or raw.strip().lower() not in _FALSY


@dataclass(frozen = True)
class DeviceProfile:
    """``kind`` is nvidia / rocm / xpu / mps / cpu; ``capabilities`` holds one (major, minor) per torch CUDA ordinal."""

    kind: str
    capabilities: tuple = ()


@functools.lru_cache(maxsize = 1)
def _probe_device_profile() -> DeviceProfile:
    """Read once per process: the compute capabilities of a box do not change while it runs."""
    try:
        import torch
    except Exception:  # noqa: BLE001 -- no torch is a CPU host for this purpose
        return DeviceProfile(CPU)
    try:
        from .rocm_bf16 import is_rocm_torch

        # The backend's own detector: AMD SDK / Radeon wheels leave ``torch.version.hip`` unset. ROCm reports gfx
        # versions through the CUDA capability call (RDNA 3 is 11.x), so it must never read as Blackwell.
        rocm = is_rocm_torch(torch)
        if not rocm:
            caps = _smi_visible_capabilities(torch)
            if caps:
                return DeviceProfile(NVIDIA, caps)
        if torch.cuda.is_available():
            kind = ROCM if rocm else NVIDIA
            caps = tuple(
                tuple(int(v) for v in torch.cuda.get_device_capability(index))
                for index in range(torch.cuda.device_count())
            )
            return DeviceProfile(kind, caps)
        xpu = getattr(torch, "xpu", None)
        if xpu is not None and xpu.is_available():
            return DeviceProfile(XPU)
        mps = getattr(getattr(torch, "backends", None), "mps", None)
        if mps is not None and mps.is_available():
            return DeviceProfile(MPS)
    except Exception:  # noqa: BLE001 -- an unreadable device is treated as no GPU
        pass
    return DeviceProfile(CPU)


def _smi_visible_capabilities(torch: Any) -> tuple:
    """The visible NVIDIA cards' capabilities from nvidia-smi, without initialising CUDA in this process (a download
    plan previewed during training must not take a context beside the trainer). Only when every visible card has the
    same capability, since nvidia-smi rows are not torch ordinals; () sends the caller to torch."""
    try:
        if torch.cuda.is_initialized():
            return ()
        from utils.hardware import nvidia
        from utils.hardware.hardware import get_parent_visible_gpu_ids

        by_row = nvidia.get_compute_capabilities()
        visible = get_parent_visible_gpu_ids()
        if not by_row or not visible or any(i not in by_row for i in visible):
            return ()
        caps = tuple(sorted(by_row[i] for i in visible))
    except Exception:  # noqa: BLE001 -- no reading: torch answers instead
        return ()
    if len(set(caps)) != 1:
        return ()
    return caps


def device_profile() -> DeviceProfile:
    return _probe_device_profile()


def _selected_ordinals(
    gpu_ordinal: Optional[int], gpu_ids: Optional[Iterable[int]]
) -> Optional[list]:
    """Torch ordinals this load is pinned to, or None when the backend picks the card itself."""
    if gpu_ordinal is not None:
        try:
            return [int(gpu_ordinal)]
        except (TypeError, ValueError):
            return None
    ids = list(gpu_ids or ())
    if not ids:
        return None
    try:
        from .diffusion_device import resolve_selected_cuda_ordinals
        return list(resolve_selected_cuda_ordinals(ids)) or None
    except Exception:  # noqa: BLE001 -- a bad pick is the route's own 400 later; decide on every card meanwhile
        return None


def load_capability(
    profile: DeviceProfile,
    gpu_ordinal: Optional[int] = None,
    gpu_ids: Optional[Iterable[int]] = None,
) -> tuple[Optional[tuple], bool]:
    """``(lowest capability that matters, routes)`` for an NVIDIA profile.

    A pinned card (or selection) routes when any selected card is pre-Blackwell; with no selection the backend may land
    on any card, so it routes only when every visible card is pre-Blackwell (a mixed box keeps today's behaviour)."""
    caps = [tuple(c) for c in profile.capabilities if c]
    if not caps:
        return None, False
    selected = _selected_ordinals(gpu_ordinal, gpu_ids)
    if selected is not None:
        picked = [caps[i] for i in selected if 0 <= i < len(caps)]
        if picked:
            low = min(picked)
            return low, low[0] < BLACKWELL_MAJOR
    low = min(caps)
    return low, max(caps)[0] < BLACKWELL_MAJOR


@dataclass(frozen = True)
class Nvfp4Route:
    """What changed: ``updates`` holds the request fields to overwrite, ``requested`` the controls' original values."""

    updates: Mapping[str, Any]
    requested: Mapping[str, Any]
    reasons: Mapping[str, str]
    device: str
    capability: Optional[tuple] = None
    replaced: Optional[str] = None
    kept: tuple = field(default_factory = tuple)

    @property
    def reason(self) -> str:
        return self.reasons.get(TRANSFORMER_QUANT) or next(iter(self.reasons.values()), "")

    def describe(self) -> dict:
        """JSON-safe summary for logs and API payloads."""
        return {
            "device": self.device,
            "capability": list(self.capability) if self.capability else None,
            "requested": dict(self.requested),
            "updates": {k: v for k, v in self.updates.items() if k != "hf_token"},
            "reason": self.reason,
            "replaced": self.replaced,
        }


def _name_says_nvfp4(name: Optional[str]) -> bool:
    stem = os.path.basename(str(name or "").replace("\\", "/").rstrip("/")).lower()
    stem = re.sub(r"\.(safetensors|pt|pth|bin)$", "", stem)
    return "nvfp4" in re.split(r"[-_.\s]+", stem)


def _local_file(model_path: Optional[str], filename: Optional[str]) -> Optional[str]:
    raw = str(model_path or "").strip()
    if not raw:
        return None
    path = os.path.expanduser(raw)
    if filename:
        path = os.path.join(path, filename)
    return path if os.path.isfile(path) else None


def checkpoint_is_nvfp4(model_path: Optional[str], filename: Optional[str] = None) -> bool:
    """Whether the picked checkpoint holds NVFP4 weights: a local file's header, else its (or its repo's) name, or a
    registered hosted ``*-NVFP4`` repo. Never downloads anything; never raises."""
    try:
        local = _local_file(model_path, filename)
        if local is not None:
            from .diffusion_prequant import local_prequant_scheme
            if is_nvfp4(local_prequant_scheme(local)):
                return True
            return _name_says_nvfp4(local)
        if filename:
            if str(filename).lower().endswith(".gguf"):
                return False
            return _name_says_nvfp4(filename)
        from .diffusion_prequant import declares_nvfp4_checkpoint

        return bool(declares_nvfp4_checkpoint(model_path))
    except Exception:  # noqa: BLE001 -- unreadable is "not known to be NVFP4"
        return False


def _switch_refuses_checkpoint(model_path: Optional[str], base_repo: Optional[str]) -> bool:
    """Whether ``refuse_disabled_nvfp4_checkpoint`` 400s this pick while the NVFP4 switch is off."""
    try:
        from .diffusion_prequant import declares_nvfp4_checkpoint
        return any(declares_nvfp4_checkpoint(p) for p in (model_path, base_repo) if p)
    except Exception:  # noqa: BLE001
        return False


def _families(media: str) -> tuple:
    try:
        if media == "video":
            from .video_families import _FAMILIES
        else:
            from .diffusion_families import _FAMILIES
        return tuple(_FAMILIES or ())
    except Exception:  # noqa: BLE001
        return ()


def hosted_nvfp4_repo_base(model_path: Optional[str], media: str) -> Optional[tuple]:
    """``(family, base repo)`` a registered hosted NVFP4 prequant repo was built from, or None."""
    key = str(model_path or "").strip().strip("/").lower()
    if not key or "/" not in key:
        return None
    for fam in _families(media):
        for row in getattr(fam, "prequant_variant_repos", ()) or ():
            if isinstance(row, tuple) and len(row) == 3 and is_nvfp4(row[1]):
                if str(row[2]).strip().lower() == key:
                    return fam, str(row[0])
        for row in getattr(fam, "prequant_repos", ()) or ():
            if isinstance(row, tuple) and len(row) == 2 and is_nvfp4(row[0]):
                if str(row[1]).strip().lower() == key:
                    return fam, getattr(fam, "base_repo", None)
    return None


def _detect_family(
    media: str, model_path: str, filename: Optional[str], override: Optional[str]
) -> Any:
    try:
        if media == "video":
            from .video_families import detect_video_family

            fam = detect_video_family(model_path, override)
            if fam is None and filename and not override:
                fam = detect_video_family(f"{model_path}/{filename}")
            return fam
        from .diffusion_families import detect_family_for_pick
        return detect_family_for_pick(model_path, filename, override)
    except Exception:  # noqa: BLE001 -- an unresolvable family keeps today's behaviour
        return None


def _fold(text: Optional[str]) -> str:
    return "".join(c for c in str(text or "").lower() if c.isalnum())


def _family_int8_base(
    media: str,
    fam: Any,
    base_repo: Optional[str],
    model_path: Optional[str] = None,
    filename: Optional[str] = None,
) -> Optional[str]:
    """The diffusers pipeline whose normal INT8 resolution replaces an NVFP4 single file of ``fam``, or None when the
    pick does not say which variant it is (a different size or distill must never load in its place)."""
    names = [n for n in (model_path, filename) if n]
    if media == "video":
        return _video_int8_base(fam, base_repo, names)
    base = str(base_repo or "").strip()
    if not base:
        try:
            from .diffusion_families import named_variant_base

            # The same name rule the loader's own base resolution uses (FLUX.2 klein 9B, Qwen-Image-2.1-Turbo).
            base = named_variant_base(fam, model_path, filename) or ""
        except Exception:  # noqa: BLE001
            base = ""
        if not base:
            plain = _fold(str(getattr(fam, "base_repo", "") or "").rsplit("/", 1)[-1])
            if not plain or not any(plain in _fold(n) for n in names):
                return None
    return base or str(getattr(fam, "base_repo", "") or "").strip() or None


def _name_tokens(names: Iterable[str]) -> set:
    return {
        t for n in names for t in re.split(r"[^a-z0-9.]+", os.path.basename(str(n)).lower()) if t
    }


def _video_int8_base(fam: Any, base_repo: Optional[str], names: list) -> Optional[str]:
    """Video: only where the family's INT8 pipeline is provably the same model as the file. LTX-2.5's pack runs only
    its distilled DiT, so a dev file is kept; the ``ltx-2`` family spans 2.0 to 2.3 with a 2.0 pipeline, so its files
    are kept; anything else must spell its pipeline's name."""
    name = str(getattr(fam, "name", "") or "")
    pack = str(getattr(fam, "base_repo", "") or "").strip()
    if not pack:
        return None
    if name == "ltx-2.5":
        return pack if "distilled" in _name_tokens(names) else None
    if name == "ltx-2":
        return None
    stem = _fold(re.sub(r"-diffusers$", "", pack.rsplit("/", 1)[-1], flags = re.IGNORECASE))
    if not stem or not any(stem in _fold(n) for n in names):
        return None
    return str(base_repo or "").strip() or pack


def _int8_denied(fam_name: Optional[str], base_repo: Optional[str]) -> bool:
    try:
        from .diffusion_transformer_quant import TQ_INT8, _family_denied
        return bool(_family_denied(fam_name, TQ_INT8, base_repo))
    except Exception:  # noqa: BLE001
        return False


def plan_nvfp4_route(
    *,
    media: str = "image",
    model_path: Optional[str] = None,
    gguf_filename: Optional[str] = None,
    model_kind: Optional[str] = None,
    base_repo: Optional[str] = None,
    family_override: Optional[str] = None,
    transformer_quant: Optional[str] = None,
    text_encoder_quant: Optional[str] = None,
    transformer_prequant_path: Optional[str] = None,
    gpu_ordinal: Optional[int] = None,
    gpu_ids: Optional[Iterable[int]] = None,
    profile: Optional[DeviceProfile] = None,
    log: bool = True,
) -> Optional[Nvfp4Route]:
    """The pre-Blackwell NVFP4 reroute for one load request, or None when the request is left exactly as it is."""
    if not route_enabled():
        return None
    from .diffusion_nvfp4_flag import nvfp4_diffusion_enabled

    # With NVFP4 off in this build an NVFP4 scheme (and a checkpoint the switch refuses) keeps its 400 on every GPU:
    # the master switch decides whether NVFP4 exists at all, this route only where it can run.
    enabled = nvfp4_diffusion_enabled()
    dit_pick = enabled and is_nvfp4(transformer_quant)
    te_pick = enabled and is_nvfp4(text_encoder_quant)
    # Picks first (names and local headers only): an ordinary request never reads the device at all.
    prequant_pick = (
        enabled
        and bool(transformer_prequant_path)
        and checkpoint_is_nvfp4(transformer_prequant_path)
    )
    ckpt_pick = bool(model_path) and checkpoint_is_nvfp4(model_path, gguf_filename)
    if ckpt_pick and not enabled and _switch_refuses_checkpoint(model_path, base_repo):
        ckpt_pick = False
    if not (dit_pick or te_pick or prequant_pick or ckpt_pick):
        return None
    profile = profile or device_profile()
    capability = None
    if profile.kind == NVIDIA:
        capability, routes = load_capability(profile, gpu_ordinal, gpu_ids)
        if not routes:
            return None
        dit_target = INT8 if capability is not None and capability >= INT8_MIN_CAPABILITY else AUTO
        te_target = INT8 if capability is not None and capability[0] >= TE_INT8_MIN_MAJOR else AUTO
    else:
        dit_target = te_target = AUTO

    updates: dict = {}
    requested: dict = {}
    reasons: dict = {}
    replaced = None
    kept: list = []

    def _dit(fam_name: Optional[str], base: Optional[str]) -> str:
        if dit_target == INT8 and _int8_denied(fam_name, base):
            return AUTO
        return dit_target

    if ckpt_pick:
        hosted = None if gguf_filename else hosted_nvfp4_repo_base(model_path, media)
        if hosted is not None:
            fam, base = hosted
        else:
            fam = _detect_family(media, str(model_path), gguf_filename, family_override)
            base = (
                _family_int8_base(media, fam, base_repo, str(model_path), gguf_filename)
                if fam is not None
                else None
            )
        if fam is not None and base:
            target = _dit(getattr(fam, "name", None), base)
            replaced = f"nvfp4:{gguf_filename or model_path}"
            updates.update(
                model_path = base,
                gguf_filename = None,
                model_kind = "pipeline",
                base_repo = None,
                display_repo_id = None,
            )
            if hosted is not None and not family_override:
                updates["family_override"] = getattr(fam, "name", None)
            # The caller's own non-NVFP4 precision still applies to the family's pipeline.
            if (
                dit_pick
                or transformer_quant is None
                or str(transformer_quant).strip().lower() in ("", AUTO)
            ):
                requested[TRANSFORMER_QUANT] = transformer_quant if dit_pick else "nvfp4"
                updates[TRANSFORMER_QUANT] = target
                reasons[TRANSFORMER_QUANT] = (
                    f"{REASON_INT8 if target == INT8 else REASON_AUTO} from {base} instead of "
                    f"{gguf_filename or model_path}"
                )
        else:
            # A user file Studio cannot tie to a family loads exactly as it did before, with the DiT precision asked.
            kept.append(gguf_filename or model_path)
            dit_pick = prequant_pick = False
            if log:
                logger.info(
                    "diffusion.nvfp4_route: kept %s as picked on %s%s: its family is not known, so there is no "
                    "INT8 build to load instead",
                    kept[0],
                    profile.kind,
                    f" sm_{capability[0]}{capability[1]}" if capability else "",
                )
    fam_name = family_override
    if fam_name is None and (dit_pick or te_pick) and model_path:
        fam = _detect_family(
            media,
            str(updates.get("model_path", model_path)),
            updates.get("gguf_filename", gguf_filename),
            updates.get("family_override", family_override),
        )
        fam_name = getattr(fam, "name", None)
    if dit_pick and TRANSFORMER_QUANT not in updates:
        target = _dit(fam_name, base_repo)
        requested[TRANSFORMER_QUANT] = transformer_quant
        updates[TRANSFORMER_QUANT] = target
        reasons[TRANSFORMER_QUANT] = REASON_INT8 if target == INT8 else REASON_AUTO
    if prequant_pick:
        # Never handed to the INT8 path: an NVFP4 artifact is not an INT8 one.
        updates["transformer_prequant_path"] = None
        requested.setdefault(TRANSFORMER_QUANT, transformer_quant or "nvfp4")
        if TRANSFORMER_QUANT not in updates:
            target = _dit(fam_name, base_repo)
            updates[TRANSFORMER_QUANT] = target
            reasons[TRANSFORMER_QUANT] = REASON_INT8 if target == INT8 else REASON_AUTO
    if te_pick:
        requested[TEXT_ENCODER_QUANT] = text_encoder_quant
        updates[TEXT_ENCODER_QUANT] = te_target
        reasons[TEXT_ENCODER_QUANT] = REASON_INT8 if te_target == INT8 else REASON_AUTO
    if not updates:
        return None
    route = Nvfp4Route(
        updates = updates,
        requested = requested,
        reasons = reasons,
        device = profile.kind,
        capability = capability,
        replaced = replaced,
        kept = tuple(kept),
    )
    if log:
        logger.info(
            "diffusion.nvfp4_route: %s on %s%s: %s",
            ", ".join(f"{k}={v!r}->{updates.get(k)!r}" for k, v in requested.items())
            or f"{replaced}->{updates.get('model_path')}",
            profile.kind,
            f" sm_{capability[0]}{capability[1]}" if capability else "",
            route.reason,
        )
    return route


def route_load_request(
    request: Any,
    *,
    media: str = "image",
    gpu_ordinal: Optional[int] = None,
) -> tuple:
    """``(request, route)``: a pydantic load request with the reroute applied (a copy), or the same object and None."""
    route = plan_nvfp4_route(
        media = media,
        model_path = getattr(request, "model_path", None),
        gguf_filename = getattr(request, "gguf_filename", None),
        model_kind = getattr(request, "model_kind", None),
        base_repo = getattr(request, "base_repo", None),
        family_override = getattr(request, "family_override", None),
        transformer_quant = getattr(request, "transformer_quant", None),
        text_encoder_quant = getattr(request, "text_encoder_quant", None),
        transformer_prequant_path = getattr(request, "transformer_prequant_path", None),
        gpu_ordinal = gpu_ordinal,
        gpu_ids = getattr(request, "gpu_ids", None),
    )
    if route is None:
        return request, None
    fields = getattr(type(request), "model_fields", None) or {}
    update = {k: v for k, v in route.updates.items() if not fields or k in fields}
    return request.model_copy(update = update), route


def apply_route_to_kwargs(route: Optional[Nvfp4Route], values: dict) -> dict:
    """``values`` with the route's updates for the keys it already holds (``repo_id`` stands for ``model_path``)."""
    if route is None:
        return values
    out = dict(values)
    for key, value in route.updates.items():
        name = "repo_id" if key == "model_path" and "repo_id" in out else key
        if name in out:
            out[name] = value
    return out


# The route of the load running on this thread, so the resolved record reports what the caller asked for.
_ACTIVE_ROUTE: contextvars.ContextVar = contextvars.ContextVar("unsloth_nvfp4_route", default = None)

ROUTE_KWARG = "_nvfp4_route"


def with_route_scope(fn: Callable) -> Callable:
    """Decorator for a backend's ``_run_load(**kwargs)``: pops ``_nvfp4_route`` and makes it active for the load."""

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        route = kwargs.pop(ROUTE_KWARG, None)
        token = _ACTIVE_ROUTE.set(route)
        try:
            return fn(*args, **kwargs)
        finally:
            _ACTIVE_ROUTE.reset(token)

    return wrapper


def active_route() -> Optional[Nvfp4Route]:
    return _ACTIVE_ROUTE.get()


def annotate_resolved(record: Optional[dict], route: Optional[Nvfp4Route] = None) -> Optional[dict]:
    """Marks the rerouted controls of a resolved record: the NVFP4 the caller asked for, ``fell_back``, and why."""
    route = route if route is not None else active_route()
    if route is None or not isinstance(record, dict):
        return record
    for control, asked in route.requested.items():
        entry = record.get(control)
        if not isinstance(entry, dict):
            continue
        reason = route.reasons.get(control) or route.reason
        prior = str(entry.get("reason") or "").strip()
        entry["requested"] = asked if asked is not None else "nvfp4"
        entry["source"] = "explicit"
        entry["status"] = "fell_back"
        entry["reason"] = (
            f"{reason}; {prior}" if prior and reason not in prior else (prior or reason)
        )
        if control == TRANSFORMER_QUANT and route.replaced and not entry.get("replaced"):
            entry["replaced"] = route.replaced
    return record
