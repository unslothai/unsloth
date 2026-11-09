# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Routes an explicit NVFP4 image / video pick to INT8 (or Auto off NVIDIA) on pre-Blackwell GPUs.

Below sm_100 NVFP4 is weight-only, measured slower than INT8 and less accurate than INT8 ConvRot. Every entry point
(load and download-plan routes, backends' ``begin_load``) asks this one decision. ``UNSLOTH_NVFP4_PRE_BLACKWELL_INT8=0``
turns it off.
"""

from __future__ import annotations

import contextvars
import functools
import logging
import os
import re
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Optional

from .diffusion_nvfp4_flag import is_nvfp4

logger = logging.getLogger(__name__)

ROUTE_ENV = "UNSLOTH_NVFP4_PRE_BLACKWELL_INT8"
_FALSY = frozenset({"0", "false", "no", "off"})

BLACKWELL_MAJOR = 10
TURING_CAPABILITY = (7, 5)
BF16_INT8_CAPABILITY = (8, 0)

NVIDIA, ROCM, XPU, MPS, CPU, UNKNOWN = "nvidia", "rocm", "xpu", "mps", "cpu", "unknown"

INT8 = "int8"
AUTO = "auto"

REASON_INT8 = "NVFP4 needs a Blackwell GPU; using INT8"
REASON_AUTO = "NVFP4 needs a Blackwell GPU; using Auto"

TRANSFORMER_QUANT = "transformer_quant"
TEXT_ENCODER_QUANT = "text_encoder_quant"

_PICKLE_SUFFIXES = (".pt", ".pth", ".bin", ".ckpt")


def route_enabled() -> bool:
    raw = os.environ.get(ROUTE_ENV)
    return raw is None or raw.strip().lower() not in _FALSY


@dataclass(frozen = True)
class DeviceProfile:
    kind: str
    capabilities: tuple = ()


_PROFILE: list = []
_PROFILE_LOCK = threading.Lock()


def _smi_visible_capabilities(torch: Any) -> tuple:
    """nvidia-smi capabilities without a CUDA context (training may run beside us); () unless all cards match,
    since nvidia-smi rows are not torch ordinals."""
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


class _CudaInitRefused(RuntimeError):
    """Only torch could read the cards and the caller forbids a CUDA context."""


def _probe_device_profile(allow_cuda_init: Any = True) -> DeviceProfile:
    """Raises when unreadable, so a transient failure is never cached."""
    try:
        import torch
    except Exception:  # noqa: BLE001 -- no torch is a CPU host for this purpose
        return DeviceProfile(CPU)
    from .rocm_bf16 import is_rocm_torch

    # Not torch.version.hip (unset on AMD SDK wheels); ROCm gfx caps (RDNA 3 = 11.x) must never read as Blackwell.
    rocm = is_rocm_torch(torch)
    if not rocm:
        caps = _smi_visible_capabilities(torch)
        if caps:
            return DeviceProfile(NVIDIA, caps)
    if torch.cuda.is_available():
        if not torch.cuda.is_initialized():
            allowed = allow_cuda_init() if callable(allow_cuda_init) else allow_cuda_init
            if not allowed:
                raise _CudaInitRefused()
        caps = tuple(
            tuple(int(v) for v in torch.cuda.get_device_capability(index))
            for index in range(torch.cuda.device_count())
        )
        return DeviceProfile(ROCM if rocm else NVIDIA, caps)
    xpu = getattr(torch, "xpu", None)
    if xpu is not None and xpu.is_available():
        return DeviceProfile(XPU)
    mps = getattr(getattr(torch, "backends", None), "mps", None)
    if mps is not None and mps.is_available():
        return DeviceProfile(MPS)
    return DeviceProfile(CPU)


def device_profile(*, allow_cuda_init: Any = True) -> DeviceProfile:
    """Cached per process; ``unknown`` when unreadable or only torch could answer and ``allow_cuda_init`` is False."""
    if _PROFILE:
        return _PROFILE[0]
    with _PROFILE_LOCK:
        if _PROFILE:
            return _PROFILE[0]
        try:
            profile = _probe_device_profile(allow_cuda_init)
        except _CudaInitRefused:
            logger.info(
                "diffusion.nvfp4_route: training is active and only CUDA could read the cards; left as is"
            )
            return DeviceProfile(UNKNOWN)
        except Exception as exc:  # noqa: BLE001 -- not cached: the next request asks again
            logger.info(
                "diffusion.nvfp4_route: device unreadable (%s), request left as is",
                type(exc).__name__,
            )
            return DeviceProfile(UNKNOWN)
        _PROFILE.append(profile)
        return profile


def reset_device_profile() -> None:
    with _PROFILE_LOCK:
        _PROFILE.clear()


def _selected_ordinals(
    gpu_ordinal: Optional[int], gpu_ids: Optional[Iterable[int]]
) -> Optional[list]:
    selected: list = []
    if gpu_ordinal is not None:
        try:
            selected.append(int(gpu_ordinal))
        except (TypeError, ValueError):
            pass
    ids = list(gpu_ids or ())
    if ids:
        try:
            from .diffusion_device import resolve_selected_cuda_ordinals
            selected.extend(int(i) for i in resolve_selected_cuda_ordinals(ids))
        except Exception:  # noqa: BLE001 -- a bad pick is the route's own 400 later
            pass
    return sorted(set(selected)) or None


def load_capability(
    profile: DeviceProfile,
    gpu_ordinal: Optional[int] = None,
    gpu_ids: Optional[Iterable[int]] = None,
) -> tuple[Optional[tuple], bool]:
    """``(lowest relevant capability, routes)``; unpinned, a mixed box routes only if every card is pre-Blackwell."""
    caps = [tuple(c) for c in profile.capabilities if c]
    if not caps:
        return None, False
    if len(set(caps)) > 1:
        selected = _selected_ordinals(gpu_ordinal, gpu_ids)
        picked = [caps[i] for i in selected or () if 0 <= i < len(caps)]
        if picked:
            low = min(picked)
            return low, low[0] < BLACKWELL_MAJOR
    low = min(caps)
    return low, max(caps)[0] < BLACKWELL_MAJOR


@dataclass(frozen = True)
class Nvfp4Route:
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


# Passed to ``begin_load`` so the backend does not re-decide on a card the route never saw.
ROUTE_UNCHANGED = Nvfp4Route(updates = {}, requested = {}, reasons = {}, device = UNKNOWN)


def _name_says_nvfp4(name: Optional[str]) -> bool:
    stem = os.path.basename(str(name or "").replace("\\", "/").rstrip("/")).lower()
    stem = re.sub(r"\.(safetensors|pt|pth|bin|ckpt)$", "", stem)
    return "nvfp4" in re.split(r"[-_.\s]+", stem)


def _looks_local(path: str) -> bool:
    return (
        os.path.isabs(path)
        or path.startswith(("~", ".", "\\"))
        or os.path.exists(os.path.expanduser(path))
    )


def checkpoint_is_nvfp4(model_path: Optional[str], filename: Optional[str] = None) -> bool:
    """Local safetensors header wins over the name; pickles are never opened, nothing is downloaded. Never raises."""
    try:
        raw = str(model_path or "").strip()
        if not raw:
            return False
        path = (
            os.path.join(os.path.expanduser(raw), filename) if filename else os.path.expanduser(raw)
        )
        if os.path.isfile(path):
            if path.lower().endswith(".gguf"):
                return False
            if not path.lower().endswith(_PICKLE_SUFFIXES):
                from .diffusion_prequant import local_prequant_scheme
                scheme = local_prequant_scheme(path)
                if scheme is not None:
                    return is_nvfp4(scheme)
            return _name_says_nvfp4(path)
        if _looks_local(raw):
            return bool(
                not filename and os.path.isdir(os.path.expanduser(raw)) and _dir_declares_nvfp4(raw)
            )
        if filename:
            if str(filename).lower().endswith(".gguf"):
                return False
            return _name_says_nvfp4(filename)
        from .diffusion_prequant import declares_nvfp4_checkpoint

        return bool(declares_nvfp4_checkpoint(raw))
    except Exception:  # noqa: BLE001 -- unreadable is "not known to be NVFP4"
        return False


def _normalized_pick(model_path: Optional[str], filename: Optional[str]) -> tuple:
    if filename or not model_path:
        return model_path, filename
    try:
        from .diffusion import split_local_checkpoint_path
        from .diffusion_local_files import resolve_local_single_file

        split = split_local_checkpoint_path(str(model_path))
        if split is not None:
            return split
        sole = resolve_local_single_file(str(model_path))
        if sole:
            return model_path, sole
    except Exception:  # noqa: BLE001 -- unreadable: decided on the pick as given
        pass
    return model_path, filename


def _dir_declares_nvfp4(path: str) -> bool:
    from .diffusion_prequant import declares_nvfp4_checkpoint
    return bool(declares_nvfp4_checkpoint(path))


def _switch_refuses_checkpoint(model_path: Optional[str], base_repo: Optional[str]) -> bool:
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


def _known_bases(fam: Any) -> dict:
    known: dict = {}
    for base in (
        getattr(fam, "base_repo", None),
        getattr(fam, "deploy_base_repo", None),
        *(getattr(fam, "train_base_repos", ()) or ()),
        *(getattr(fam, "named_variant_bases", ()) or ()),
    ):
        if isinstance(base, str) and base.strip():
            known.setdefault(base.strip().lower(), base.strip())
    return known


def _all_bases(fam: Any) -> set:
    bases = set(_known_bases(fam))
    for row in getattr(fam, "prequant_variant_repos", ()) or ():
        if isinstance(row, tuple) and row and isinstance(row[0], str):
            bases.add(row[0].strip().lower())
    for base in getattr(fam, "prequant_excluded_bases", ()) or ():
        if isinstance(base, str):
            bases.add(base.strip().lower())
    return {b for b in bases if b}


def hosted_nvfp4_repo_base(
    model_path: Optional[str],
    media: str,
    *,
    filename: Optional[str] = None,
    family_override: Optional[str] = None,
) -> Optional[tuple]:
    """``(family, canonical base)`` of a hosted NVFP4 repo; None if ambiguous across families or base spelling unknown."""
    key = str(model_path or "").strip().strip("/").lower()
    if not key or "/" not in key:
        return None
    hits: list = []
    for fam in _families(media):
        known = _known_bases(fam)
        for row in getattr(fam, "prequant_variant_repos", ()) or ():
            if isinstance(row, tuple) and len(row) == 3 and is_nvfp4(row[1]):
                if str(row[2]).strip().lower() == key:
                    hits.append((fam, known.get(str(row[0]).strip().lower())))
        for row in getattr(fam, "prequant_repos", ()) or ():
            if isinstance(row, tuple) and len(row) == 2 and is_nvfp4(row[0]):
                if str(row[1]).strip().lower() == key:
                    hits.append((fam, getattr(fam, "base_repo", None)))
    if family_override:
        wanted = str(family_override).strip().lower()
        hits = [h for h in hits if str(getattr(h[0], "name", "")).lower() == wanted]
    if len(hits) > 1 and filename:
        name = os.path.basename(str(filename)).lower()
        hits = [
            h
            for h in hits
            if any(
                isinstance(r, tuple) and r and is_nvfp4(r[0]) and str(r[-1]).lower() == name
                for r in getattr(h[0], "prequant_filenames", ()) or ()
            )
        ]
    if len(hits) != 1 or not hits[0][1]:
        return None
    return hits[0]


def _detect_family(
    media: str,
    model_path: str,
    filename: Optional[str],
    override: Optional[str],
    model_kind: Optional[str] = None,
) -> Any:
    try:
        if media == "video":
            from .video import _detect_load_family
            return _detect_load_family(model_path, filename, override, None, model_kind)
        from .diffusion_families import detect_family_for_pick
        return detect_family_for_pick(model_path, filename, override)
    except Exception:  # noqa: BLE001 -- an unresolvable family keeps today's behaviour
        return None


def _fold(text: Optional[str]) -> str:
    return "".join(c for c in str(text or "").lower() if c.isalnum())


def _stem(repo: str) -> str:
    return _fold(re.sub(r"-diffusers$", "", repo.rsplit("/", 1)[-1], flags = re.IGNORECASE))


def _named_base(fam: Any, names: list) -> Optional[str]:
    """Longest matching base wins (Qwen-Image-2512 never reads as Qwen-Image); canonical spelling only."""
    known = _known_bases(fam)
    stems = [(b, _stem(b)) for b in _all_bases(fam)]
    for name in reversed(names):
        folded = _fold(os.path.basename(str(name).replace("\\", "/").rstrip("/")) or name)
        hits = [(len(stem), b) for b, stem in stems if stem and stem in folded]
        if hits:
            return known.get(max(hits)[1])
    return None


def _family_int8_base(
    media: str,
    fam: Any,
    base_repo: Optional[str],
    model_path: Optional[str] = None,
    filename: Optional[str] = None,
) -> Optional[str]:
    """None unless the pick provably names it: a different size, version or distill must never load in its place."""
    names = [n for n in (model_path, filename) if n]
    if media == "video":
        return _video_int8_base(fam, base_repo, names)
    explicit = str(base_repo or "").strip()
    if explicit:
        return explicit
    return _named_base(fam, names)


def _name_tokens(names: Iterable[str]) -> set:
    return {
        t for n in names for t in re.split(r"[^a-z0-9.]+", os.path.basename(str(n)).lower()) if t
    }


def _video_int8_base(fam: Any, base_repo: Optional[str], names: list) -> Optional[str]:
    """LTX-2.5 pack is distilled-only; ``ltx-2`` spans 2.0-2.3 on a 2.0 pipeline, so never replaced."""
    name = str(getattr(fam, "name", "") or "")
    pack = str(getattr(fam, "base_repo", "") or "").strip()
    if not pack:
        return None
    if name == "ltx-2.5":
        return pack if "distilled" in _name_tokens(names) else None
    if name == "ltx-2":
        return None
    named = _named_base(fam, names)
    if named is None or named.lower() != pack.lower():
        return None
    return str(base_repo or "").strip() or pack


def _int8_denied(fam_name: Optional[str], base_repo: Optional[str]) -> bool:
    try:
        from .diffusion_transformer_quant import TQ_INT8, _family_denied
        return bool(_family_denied(fam_name, TQ_INT8, base_repo))
    except Exception:  # noqa: BLE001
        return False


def _turing_int8_enabled() -> bool:
    try:
        from .diffusion_transformer_quant import fp16_card_int8_enabled
        return bool(fp16_card_int8_enabled())
    except Exception:  # noqa: BLE001
        return False


def _dit_target(media: str, capability: Optional[tuple]) -> str:
    """Video INT8 needs bf16 (sm_80+); image INT8 also runs on Turing via the fp16 int8 linear."""
    if capability is None:
        return AUTO
    if capability >= BF16_INT8_CAPABILITY:
        return INT8
    if media != "video" and capability >= TURING_CAPABILITY and _turing_int8_enabled():
        return INT8
    return AUTO


def _is_auto(value: Optional[str]) -> bool:
    return value is None or str(value).strip().lower() in ("", AUTO)


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
    component_files: bool = False,
    replace_checkpoint: bool = True,
    allow_cuda_init: Any = True,
    profile: Optional[DeviceProfile] = None,
    log: bool = True,
) -> Optional[Nvfp4Route]:
    """None when left as is. ``component_files`` / ``replace_checkpoint=False`` keep the checkpoint, rerouting precision only."""
    if not route_enabled():
        return None
    from .diffusion_nvfp4_flag import nvfp4_diffusion_enabled

    # NVFP4 disabled: its 400 stands on every GPU; this route never turns it into a load.
    enabled = nvfp4_diffusion_enabled()
    dit_pick = enabled and is_nvfp4(transformer_quant)
    te_pick = enabled and is_nvfp4(text_encoder_quant)
    model_path, gguf_filename = _normalized_pick(model_path, gguf_filename)
    # Picks first: an ordinary request never reads the device.
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
    profile = profile or device_profile(allow_cuda_init = allow_cuda_init)
    capability = None
    if profile.kind == NVIDIA:
        capability, routes = load_capability(profile, gpu_ordinal, gpu_ids)
        if not routes:
            return None
        dit_target = _dit_target(media, capability)
        te_target = INT8 if capability is not None and capability >= BF16_INT8_CAPABILITY else AUTO
    elif profile.kind in (ROCM, XPU, MPS, CPU):
        dit_target = te_target = AUTO
    else:
        return None
    where = f"{profile.kind}" + (f" sm_{capability[0]}{capability[1]}" if capability else "")

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
        name = gguf_filename or model_path
        fam = base = None
        why = None
        if component_files:
            why = "it names its own text-encoder / VAE files, which the family pipeline would drop"
        elif not replace_checkpoint:
            why = "this load was not picked by hand"
        else:
            hosted = hosted_nvfp4_repo_base(
                model_path, media, filename = gguf_filename, family_override = family_override
            )
            if hosted is not None:
                fam, base = hosted
            else:
                fam = _detect_family(
                    media, str(model_path), gguf_filename, family_override, model_kind
                )
                base = (
                    _family_int8_base(media, fam, base_repo, str(model_path), gguf_filename)
                    if fam
                    else None
                )
            if fam is None or not base:
                why = (
                    "Studio cannot tell which model it is, so there is no INT8 build of it to load"
                )
        if why is None:
            target = _dit(getattr(fam, "name", None), base)
            replaced = f"nvfp4:{name}"
            updates.update(
                model_path = base,
                gguf_filename = None,
                model_kind = "pipeline",
                base_repo = None,
                display_repo_id = None,
            )
            if not family_override:
                updates["family_override"] = getattr(fam, "name", None)
            word = "INT8" if target == INT8 else "Auto"
            if dit_pick or _is_auto(transformer_quant):
                requested[TRANSFORMER_QUANT] = transformer_quant if dit_pick else "nvfp4"
                updates[TRANSFORMER_QUANT] = target
                reasons[TRANSFORMER_QUANT] = (
                    f"{REASON_INT8 if target == INT8 else REASON_AUTO} from {base} instead of {name}"
                )
            else:
                reasons[TRANSFORMER_QUANT] = (
                    f"NVFP4 needs a Blackwell GPU; loading {base} instead of {name}"
                )
                word = str(transformer_quant)
            if log:
                logger.info("diffusion.nvfp4_route: %s -> %s (%s) on %s", name, base, word, where)
        else:
            kept.append(name)
            dit_pick = prequant_pick = False
            if log:
                logger.info("diffusion.nvfp4_route: kept %s as picked on %s: %s", name, where, why)
    fam_name = updates.get("family_override") or family_override
    if fam_name is None and (dit_pick or prequant_pick) and model_path:
        fam_name = getattr(
            _detect_family(media, str(model_path), gguf_filename, None, model_kind), "name", None
        )
    if dit_pick and TRANSFORMER_QUANT not in updates:
        target = _dit(fam_name, base_repo)
        requested[TRANSFORMER_QUANT] = transformer_quant
        updates[TRANSFORMER_QUANT] = target
        reasons[TRANSFORMER_QUANT] = REASON_INT8 if target == INT8 else REASON_AUTO
    if prequant_pick:
        # An NVFP4 artifact is never handed to the INT8 path.
        updates["transformer_prequant_path"] = None
        if TRANSFORMER_QUANT not in updates and _is_auto(transformer_quant):
            target = _dit(fam_name, base_repo)
            requested[TRANSFORMER_QUANT] = "nvfp4"
            updates[TRANSFORMER_QUANT] = target
            reasons[TRANSFORMER_QUANT] = REASON_INT8 if target == INT8 else REASON_AUTO
        elif TRANSFORMER_QUANT not in reasons:
            reasons[TRANSFORMER_QUANT] = (
                "NVFP4 needs a Blackwell GPU; the NVFP4 prequant file is not used"
            )
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
    if log and requested:
        logger.info(
            "diffusion.nvfp4_route: %s on %s: %s",
            ", ".join(f"{k}={v!r}->{updates.get(k)!r}" for k, v in requested.items()),
            where,
            route.reason,
        )
    return route


def _has_component_files(request: Any) -> bool:
    for name in ("text_encoder_file", "text_encoder_files", "vae_file"):
        value = getattr(request, name, None)
        if value:
            return True
    return False


def route_load_request(
    request: Any,
    *,
    media: str = "image",
    gpu_ordinal: Optional[int] = None,
    replace_checkpoint: bool = True,
    allow_cuda_init: Any = True,
) -> tuple:
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
        component_files = _has_component_files(request),
        replace_checkpoint = replace_checkpoint,
        allow_cuda_init = allow_cuda_init,
    )
    if route is None:
        return request, None
    fields = getattr(type(request), "model_fields", None) or {}
    update = {k: v for k, v in route.updates.items() if not fields or k in fields}
    return request.model_copy(update = update), route


_ACTIVE_ROUTE: contextvars.ContextVar = contextvars.ContextVar("unsloth_nvfp4_route", default = None)

ROUTE_KWARG = "_nvfp4_route"


def with_route_scope(fn: Callable) -> Callable:
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
    route = route if route is not None else active_route()
    if route is None or not isinstance(record, dict):
        return record
    for control in dict.fromkeys((*route.requested, *route.reasons)):
        entry = record.get(control)
        if not isinstance(entry, dict):
            continue
        reason = route.reasons.get(control) or route.reason
        prior = str(entry.get("reason") or "").strip()
        if control in route.requested:
            asked = route.requested[control]
            entry["requested"] = asked if asked is not None else "nvfp4"
            entry["source"] = "explicit"
            entry["status"] = "fell_back"
        if reason:
            entry["reason"] = (
                f"{reason}; {prior}" if prior and reason not in prior else (prior or reason)
            )
        if control == TRANSFORMER_QUANT and route.replaced and not entry.get("replaced"):
            entry["replaced"] = route.replaced
    return record
