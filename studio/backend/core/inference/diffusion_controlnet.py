# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Diffusion ControlNet support: family-gated discovery, resolution to a loadable diffusers
repo/dir, control-image preprocessing, and a capability gate.

Mirrors ``diffusion_lora.py``. Two differences: (1) a ControlNet is a full diffusers repo (loaded
via ``from_pretrained``), so resolution yields a repo id / local dir, not a file; (2) it needs a
spatial *control image*, either supplied already-preprocessed ("passthrough") or derived here
("canny", a dependency-free edge map).

ControlNets are architecture-specific, so discovery is family-gated like the LoRA picker. The
request never carries a path (only a discovery id or ``owner/name`` repo id), so a client cannot
make the backend read an arbitrary location.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from utils.paths.storage_roots import studio_root

# Control map types. "passthrough": the supplied image IS the control map. "canny": derive an
# edge map here (no heavy detector dependency).
CONTROL_TYPES = ("passthrough", "canny")

# Diffusers quant schemes that cannot host ControlNet cleanly (torchao tensor-subclass weights);
# gated off like LoRA, along with GGUF-via-diffusers.
_DIFFUSERS_BLOCKED_QUANT = ("int8", "fp8", "nvfp4", "mxfp8")


@dataclass(frozen = True)
class ControlNetCatalogEntry:
    """One discoverable ControlNet model."""

    id: str
    display_name: str
    source: str  # "local" | "hub"
    families: tuple[str, ...] = ()  # compatible families (empty = shown, not gated)
    repo_id: Optional[str] = None  # source == "hub"
    local_path: Optional[str] = None  # source == "local"
    control_types: tuple[str, ...] = ("passthrough",)
    is_union: bool = False  # one model covering many control modes


@dataclass(frozen = True)
class ResolvedControlNet:
    """A ControlNet resolved to something ``from_pretrained`` can load."""

    id: str
    path: str  # repo id (hub) or local directory
    is_local: bool


# Curated, family-tagged catalog. Union models (one model, many modes) are the default picks;
# local dirs and a bare ``owner/name`` repo id also work.
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


def _has_controlnet_weights(p: Path) -> bool:
    """True when ``p`` holds a loadable diffusers ControlNet weight (or shard index).

    Guards against advertising a config-only folder (interrupted download) that then fails deep in
    ``from_pretrained``. Accepts the standard single-file weights, a shard index, or any
    ``.safetensors`` shard."""
    names = (
        "diffusion_pytorch_model.safetensors",
        "diffusion_pytorch_model.bin",
        "diffusion_pytorch_model.safetensors.index.json",
        "diffusion_pytorch_model.bin.index.json",
    )
    if any((p / n).exists() for n in names):
        return True
    try:
        return any(child.suffix == ".safetensors" for child in p.iterdir())
    except OSError:
        return False


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
        # Require BOTH config and a loadable weight/index (a config-only folder is incomplete).
        if not (p / "config.json").exists() or not _has_controlnet_weights(p):
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


def resolve_controlnet(spec_id: str, *, family: Optional[str] = None) -> ResolvedControlNet:
    """Resolve a ControlNet id to a loadable repo id / local dir.

    Accepts a catalog/local id or a bare HF repo id (``owner/name``); the backend loads it with
    ``from_pretrained``. Raises on an unknown id (caller maps to 400).

    ``family`` enforces compatibility: a ControlNet is architecture-specific, so an entry tagged
    for another family is rejected here rather than loaded through the wrong pipeline later.
    """
    entry = _catalog_by_id().get(spec_id)
    if entry is None:
        # A curated entry named by its full repo id must still hit the family gate below, not slip
        # through the bare-repo fallback and load through the wrong family's class.
        entry = next((e for e in _CURATED if e.repo_id and e.repo_id == spec_id), None)
    if entry is not None:
        # A direct API call could bypass the UI filter and send an entry for another family; reject
        # it before any download so it never reaches the wrong pipeline.
        fam = (family or "").strip().lower()
        if entry.families and fam and fam not in {f.lower() for f in entry.families}:
            raise ValueError(
                f"ControlNet '{spec_id}' is for {', '.join(entry.families)}, not the loaded "
                f"'{family}' model; pick a ControlNet built for this family."
            )
        if entry.source == "local":
            path = entry.local_path or ""
            if not path or not Path(path).is_dir():
                raise FileNotFoundError(f"ControlNet '{spec_id}' is no longer present on disk")
            return ResolvedControlNet(spec_id, path, is_local = True)
        if not entry.repo_id:
            raise ValueError(f"ControlNet '{spec_id}' has no repo")
        return ResolvedControlNet(spec_id, entry.repo_id, is_local = False)

    # A bare HF repo id (owner/name). STRICT shape (one slash, alphanumeric-leading segments) so a
    # filesystem-looking id can never reach from_pretrained and bypass the no-raw-path contract.
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*", spec_id):
        return ResolvedControlNet(spec_id, spec_id, is_local = False)

    raise FileNotFoundError(
        f"unknown ControlNet '{spec_id}': not a local model, catalog entry, or HF repo id"
    )


# Union ControlNet mode indices: a union model selects the active mode via an integer
# ``control_mode``. Standard indices for the FLUX.1 / Qwen-Image union ControlNets.
_UNION_CONTROL_MODES: dict[str, int] = {
    "canny": 0,
    "tile": 1,
    "depth": 2,
    "blur": 3,
    "pose": 4,
    "gray": 5,
    "lq": 6,
}


def union_control_mode(spec_id: str, control_type: str) -> Optional[int]:
    """The integer ``control_mode`` for a union ControlNet, or None.

    A union model requires a concrete mode. A known mode maps to its index; ``passthrough`` (or
    empty) defaults to 0 (canny head). An unknown/typo'd type raises ValueError so the route
    returns a 400 instead of running the wrong head. A non-union entry returns None."""
    entry = _catalog_by_id().get(spec_id)
    if entry is None:
        # A union model may be named by its bare repo id; the catalog is keyed by short id, so
        # match on repo_id too, else the mode is dropped and the union runs the wrong head.
        entry = next((e for e in _CURATED if e.repo_id and e.repo_id == spec_id), None)
    if entry is None or not entry.is_union:
        return None
    ct = (control_type or "").strip().lower()
    if ct in _UNION_CONTROL_MODES:
        return _UNION_CONTROL_MODES[ct]
    if ct in ("", "passthrough"):
        return 0  # preprocessed map, no intrinsic mode; canny is the default head
    raise ValueError(
        f"Unknown control type {control_type!r} for a union ControlNet. Use one of: "
        f"{', '.join(sorted(_UNION_CONTROL_MODES))}, or passthrough."
    )


def preprocess_control(image: Any, control_type: str) -> Any:
    """Turn a source image into a control map.

    ``passthrough`` returns the image unchanged. ``canny`` derives a dependency-free gradient edge
    map (a rough stand-in for true Canny). Unknown types pass through so a new type never fails.
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
        return image  # flat image -> nothing to trace
    mag = mag / peak * 255.0
    edges = (mag > 40.0).astype(np.uint8) * 255  # white edges on black (ControlNet convention)
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

    diffusers only. Requires the family to declare a ControlNet pipeline. Blocked for the GGUF
    path and torchao fp8/int8 dense (same as LoRA): they can't host the extra conditioning cleanly.
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
