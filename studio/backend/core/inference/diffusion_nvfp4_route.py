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
only for an NVFP4 pick, once per process; tests replace ``device_profile``. ``UNSLOTH_NVFP4_PRE_BLACKWELL_INT8=0``
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

# NVIDIA compute capability major of the first FP4 tensor-core generation (sm_100 / sm_103 / sm_110 / sm_120 / sm_121).
BLACKWELL_MAJOR = 10
# Image INT8 on a Turing fp16 card is the torchao-free int8 linear (``fp16_card_int8_host``); bf16 INT8 starts at sm_80.
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
    """False only for an explicit falsy ``UNSLOTH_NVFP4_PRE_BLACKWELL_INT8``; read at call time."""
    raw = os.environ.get(ROUTE_ENV)
    return raw is None or raw.strip().lower() not in _FALSY


@dataclass(frozen = True)
class DeviceProfile:
    """``kind`` is nvidia / rocm / xpu / mps / cpu / unknown; ``capabilities`` one (major, minor) per CUDA card."""

    kind: str
    capabilities: tuple = ()


_PROFILE: list = []
_PROFILE_LOCK = threading.Lock()


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


class _CudaInitRefused(RuntimeError):
    """Reading the cards would initialise CUDA in this process, which the caller does not allow right now."""


def _probe_device_profile(allow_cuda_init: bool = True) -> DeviceProfile:
    """Raises when the device cannot be read, so a transient failure is never cached."""
    try:
        import torch
    except Exception:  # noqa: BLE001 -- no torch is a CPU host for this purpose
        return DeviceProfile(CPU)
    from .rocm_bf16 import is_rocm_torch

    # The backend's own detector: AMD SDK / Radeon wheels leave ``torch.version.hip`` unset. ROCm reports gfx versions
    # through the CUDA capability call (RDNA 3 is 11.x), so it must never read as Blackwell.
    rocm = is_rocm_torch(torch)
    if not rocm:
        caps = _smi_visible_capabilities(torch)
        if caps:
            return DeviceProfile(NVIDIA, caps)
    if torch.cuda.is_available():
        if not allow_cuda_init and not torch.cuda.is_initialized():
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


def device_profile(*, allow_cuda_init: bool = True) -> DeviceProfile:
    """Read once per process (a box's capabilities do not change while it runs); ``unknown`` when unreadable, or when
    only torch could answer and ``allow_cuda_init`` is False (training owns the GPU: no context may be taken here)."""
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
    """Every torch ordinal this load may place work on (the ranked primary plus the rest of a ``gpu_ids`` selection),
    or None when the backend picks the card itself."""
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
    """``(lowest capability that matters, routes)`` for an NVIDIA profile.

    A pinned card (or selection) routes when any selected card is pre-Blackwell; with no selection the backend may land
    on any card, so it routes only when every visible card is pre-Blackwell (a mixed box keeps today's behaviour)."""
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


# Handed to ``begin_load`` when the route decided to leave a request as it is, so the backend does not decide again on
# a card the route never saw (the plan and the load must agree).
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
    """Whether the picked checkpoint holds NVFP4 weights: a local safetensors header's recorded scheme first (it wins
    over the name), else the file's (or repo's) name, or a registered hosted ``*-NVFP4`` repo. A pickle is never opened
    to ask, a local path that does not exist is not a pick, and nothing is downloaded. Never raises."""
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
            # Missing on disk: the loader's own not-found error stays the answer.
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


def _known_bases(fam: Any) -> dict:
    """Lowercased base id -> its canonical spelling, for the bases a family names in full (never a lowercase key)."""
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
    """Every base id a family's tables mention, lowercased (variant rows and exclusions too)."""
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
    """``(family, canonical base repo)`` a registered hosted NVFP4 prequant repo was built from; None when it is not
    one, when several families share it and nothing tells them apart, or when the base's spelling is unknown."""
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
    """The family the loader itself resolves for this pick (names, headers, a local pipeline index)."""
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
    """The family base the names spell, most specific name first: the longest base name any table of the family lists
    (so a Qwen-Image-2512 file never reads as Qwen-Image), returned only in its canonical spelling."""
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
    """The diffusers pipeline whose normal INT8 resolution replaces an NVFP4 single file of ``fam``, or None when the
    pick does not provably name it (a different size, version or distill must never load in its place)."""
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
    """INT8 where the family's INT8 path runs on this card, else Auto: video INT8 needs bf16 (sm_80+); image INT8 on a
    Turing fp16 card is the torchao-free int8 linear, unless it is switched off."""
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
    allow_cuda_init: bool = True,
    profile: Optional[DeviceProfile] = None,
    log: bool = True,
) -> Optional[Nvfp4Route]:
    """The pre-Blackwell NVFP4 reroute for one load request, or None when the request is left exactly as it is.

    ``component_files``: the pick names its own text-encoder / VAE files, an assembly a family pipeline cannot keep,
    so its checkpoint is never replaced. ``replace_checkpoint=False`` (a load nobody picked by hand, whose caller tracks
    the model by its path) reroutes precision only."""
    if not route_enabled():
        return None
    from .diffusion_nvfp4_flag import nvfp4_diffusion_enabled

    # With NVFP4 off in this build an NVFP4 scheme (and a checkpoint the switch refuses) keeps its 400 on every GPU:
    # the master switch decides whether NVFP4 exists at all, this route only where it can run.
    enabled = nvfp4_diffusion_enabled()
    dit_pick = enabled and is_nvfp4(transformer_quant)
    te_pick = enabled and is_nvfp4(text_encoder_quant)
    # The file the loader will open, the way the routes normalise a local pick later: a path naming one file, or a
    # directory holding a sole checkpoint, decides the same as the (directory, file) pair it becomes.
    model_path, gguf_filename = _normalized_pick(model_path, gguf_filename)
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
        # Unreadable device: nothing is known to be wrong with the pick.
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
                # The caller's own non-NVFP4 precision still applies to the family's pipeline.
                reasons[TRANSFORMER_QUANT] = (
                    f"NVFP4 needs a Blackwell GPU; loading {base} instead of {name}"
                )
                word = str(transformer_quant)
            if log:
                logger.info("diffusion.nvfp4_route: %s -> %s (%s) on %s", name, base, word, where)
        else:
            # The pick loads exactly as it did before, with the DiT precision asked.
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
        # Never handed to the INT8 path: an NVFP4 artifact is not an INT8 one. An explicit non-NVFP4 precision stays.
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
    allow_cuda_init: bool = True,
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
        component_files = _has_component_files(request),
        replace_checkpoint = replace_checkpoint,
        allow_cuda_init = allow_cuda_init,
    )
    if route is None:
        return request, None
    fields = getattr(type(request), "model_fields", None) or {}
    update = {k: v for k, v in route.updates.items() if not fields or k in fields}
    return request.model_copy(update = update), route


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
    """Marks the rerouted controls of a resolved record: the NVFP4 the caller asked for, ``fell_back``, and why; a
    replaced checkpoint is named on ``transformer_quant`` even when the caller's own precision was kept."""
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
