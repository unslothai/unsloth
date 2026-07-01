# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Diffusion ControlNet support: family-gated discovery of ControlNet models, resolution to
a loadable diffusers repo/dir, control-image preprocessing, and a capability gate.

Mirrors ``diffusion_lora.py``. Two differences from LoRA: (1) a ControlNet is a full diffusers
repo (loaded via ``from_pretrained``), not a single-file adapter, so resolution yields a repo id
or local directory rather than a file path; (2) ControlNet needs a spatial *control image*, which
is either supplied already-preprocessed ("passthrough", as in ComfyUI where preprocessing is a
separate step) or derived here ("canny", a dependency-free edge map).

ControlNet models are architecture-specific (a FLUX ControlNet cannot drive a Qwen base), so
discovery is family-gated exactly like the LoRA picker. The request never carries a filesystem
path -- only a discovery id or a public ``owner/name`` repo id -- so a client cannot make the
backend read an arbitrary location.
"""

from __future__ import annotations

import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from utils.paths.storage_roots import studio_root

# Control map types. "passthrough": the supplied image IS the control map (a depth/pose/etc.
# map produced elsewhere). "canny": derive an edge map here (no heavy detector dependency).
CONTROL_TYPES = ("passthrough", "canny")

# Families whose diffusers pipeline supports ControlNet (declared in diffusion_families via
# controlnet_pipeline_class). Native sd.cpp ControlNet is a follow-up. Torchao fp8/int8 dense
# and GGUF-via-diffusers are gated off, same rule as LoRA.
_DIFFUSERS_BLOCKED_QUANT = ("int8", "fp8", "nvfp4", "mxfp8")


@dataclass(frozen = True)
class ControlNetCatalogEntry:
    """One discoverable ControlNet model."""

    id: str
    display_name: str
    source: str  # "local" | "hub"
    families: tuple[str, ...] = ()  # compatible family names (empty = shown, not gated)
    repo_id: Optional[str] = None  # for source == "hub"
    local_path: Optional[str] = None  # for source == "local"
    control_types: tuple[str, ...] = ("passthrough",)  # recommended control types
    is_union: bool = False  # a single model covering many control modes


@dataclass(frozen = True)
class ResolvedControlNet:
    """A ControlNet resolved to something ``from_pretrained`` can load."""

    id: str
    path: str  # repo id (hub) or local directory
    is_local: bool


# Curated, family-tagged catalog. Union models (one model, many control modes) dominate real
# usage, so they are the default picks. Extend as more are curated; local dirs + a bare public
# ``owner/name`` repo id also work.
_CURATED: tuple[ControlNetCatalogEntry, ...] = (
    ControlNetCatalogEntry(
        id = "flux-union-pro",
        display_name = "FLUX.1 ControlNet Union Pro (Shakker-Labs)",
        source = "hub",
        families = ("flux.1",),
        repo_id = "Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro",
        control_types = ("canny", "depth", "pose", "passthrough"),
        is_union = True,
    ),
    ControlNetCatalogEntry(
        id = "qwen-union",
        display_name = "Qwen-Image ControlNet Union (InstantX)",
        source = "hub",
        families = ("qwen-image",),
        repo_id = "InstantX/Qwen-Image-ControlNet-Union",
        control_types = ("canny", "depth", "pose", "passthrough"),
        is_union = True,
    ),
)


def controlnets_dir() -> Path:
    """Local directory Studio scans for user-provided ControlNet model folders."""
    d = studio_root() / "controlnets" / "diffusion"
    d.mkdir(parents = True, exist_ok = True)
    return d


def sanitize_id(raw: str) -> str:
    """Filesystem-safe id from a repo id / folder name."""
    stem = raw.rsplit("/", 1)[-1]
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._-")
    return stem or "controlnet"


def _scan_local() -> list[ControlNetCatalogEntry]:
    """A local ControlNet is a directory containing a diffusers config + weights."""
    entries: list[ControlNetCatalogEntry] = []
    root = controlnets_dir()
    try:
        children = sorted(root.iterdir())
    except OSError:
        return entries
    for p in children:
        if not p.is_dir():
            continue
        if not (p / "config.json").exists():
            continue
        entries.append(
            ControlNetCatalogEntry(
                id = p.name,
                display_name = p.name,
                source = "local",
                local_path = str(p),
                control_types = CONTROL_TYPES,
            )
        )
    return entries


def list_controlnets(*, family: Optional[str] = None) -> list[ControlNetCatalogEntry]:
    """Merged catalog (curated + local), optionally family-filtered. Cheap: one dir scan
    plus the in-memory curated list. Network is only touched on resolve()."""
    merged = list(_CURATED) + _scan_local()
    if family:
        fam = family.strip().lower()
        merged = [e for e in merged if not e.families or fam in {f.lower() for f in e.families}]
    merged.sort(key = lambda e: (e.source != "local", e.display_name.lower()))
    return merged


def _catalog_by_id() -> dict[str, ControlNetCatalogEntry]:
    return {e.id: e for e in (list(_CURATED) + _scan_local())}


def resolve_controlnet(
    spec_id: str,
    *,
    family: Optional[str] = None,
    hf_token: Optional[str] = None,
    cancel_event: Optional[threading.Event] = None,
) -> ResolvedControlNet:
    """Resolve a ControlNet id to a loadable repo id / local dir.

    Accepts a catalog/local id, or a bare public HF repo id (``owner/name``). The backend
    loads the result with ``ControlNetModelClass.from_pretrained(path)`` (download + cache
    handled there, like the base pipeline). Raises on an unknown id -> the caller maps to 400.
    """
    entry = _catalog_by_id().get(spec_id)
    if entry is not None:
        if entry.source == "local":
            path = entry.local_path or ""
            if not path or not Path(path).is_dir():
                raise FileNotFoundError(f"ControlNet '{spec_id}' is no longer present on disk")
            return ResolvedControlNet(spec_id, path, is_local = True)
        if not entry.repo_id:
            raise ValueError(f"ControlNet '{spec_id}' has no repo")
        return ResolvedControlNet(spec_id, entry.repo_id, is_local = False)

    # A bare public HF repo id (owner/name).
    if "/" in spec_id and " " not in spec_id:
        return ResolvedControlNet(spec_id, spec_id, is_local = False)

    raise FileNotFoundError(
        f"unknown ControlNet '{spec_id}': not a local model, catalog entry, or HF repo id"
    )


def preprocess_control(image: Any, control_type: str) -> Any:
    """Turn a source image into a control map.

    ``passthrough`` returns the image unchanged (it is already a depth/pose/edge map made
    elsewhere). ``canny`` derives a dependency-free gradient edge map (a rough stand-in for a
    true Canny; a cv2/kornia detector and depth/pose detectors are a follow-up). Unknown types
    pass through so a new type never hard-fails generation.
    """
    ct = (control_type or "passthrough").strip().lower()
    if ct != "canny":
        return image
    import numpy as np
    from PIL import Image

    gray = np.asarray(image.convert("L"), dtype = np.float32)
    gy, gx = np.gradient(gray)
    mag = np.hypot(gx, gy)
    peak = float(mag.max())
    if peak <= 1e-6:
        return image  # flat image -> nothing to trace; don't emit a black map
    mag = mag / peak * 255.0
    edges = (mag > 40.0).astype(np.uint8) * 255  # white edges on black, the ControlNet convention
    return Image.fromarray(edges).convert("RGB")


def supports_controlnet(
    *,
    engine: str,
    family: Optional[str],
    has_controlnet_pipeline: bool,
    model_kind: Optional[str],
    transformer_quant: Optional[str],
) -> bool:
    """Whether the loaded model can apply a ControlNet.

    diffusers only for now (native sd.cpp is a follow-up). Requires the family to declare a
    ControlNet pipeline. Blocked for the diffusers GGUF path and torchao fp8/int8 dense
    (same constraints as LoRA): those transformers cannot host the extra conditioning cleanly.
    """
    if not family or not has_controlnet_pipeline:
        return False
    if engine != "diffusers":
        return False
    if model_kind == "gguf":
        return False
    if transformer_quant and str(transformer_quant).strip().lower() in _DIFFUSERS_BLOCKED_QUANT:
        return False
    return True
