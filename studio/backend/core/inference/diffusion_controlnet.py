# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Family-gated ControlNet discovery; requests carry only a discovery id or owner/name, never a path."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from utils.paths.storage_roots import account_path
from utils.paths.path_utils import is_appledouble_metadata

# Control map types. "passthrough": the supplied image IS the control map. "canny": derive an edge map here.
CONTROL_TYPES = ("passthrough", "canny")

# Diffusers quant schemes that cannot host ControlNet cleanly (torchao tensor subclasses); gated off like LoRA, along
# with GGUF-via-diffusers.
_DIFFUSERS_BLOCKED_QUANT = ("int8", "fp8", "nvfp4", "mxfp8")


@dataclass(frozen = True)
class ControlNetCatalogEntry:
    """One discoverable ControlNet model."""

    id: str
    display_name: str
    source: str
    families: tuple[str, ...] = ()
    repo_id: Optional[str] = None
    local_path: Optional[str] = None
    control_types: tuple[str, ...] = ("passthrough",)
    is_union: bool = False


@dataclass(frozen = True)
class ResolvedControlNet:
    """A ControlNet resolved to something ``from_pretrained`` can load."""

    id: str
    path: str
    is_local: bool


# Key naming mirrors the adaLN curve contract (adaln_form, curve_dim, curve_grid): a form tag plus the parameters needed to reproduce the form
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
    """Local directory Unsloth scans for user-provided ControlNet model folders; per-account, with the owner on the install root."""
    d = account_path("controlnets/diffusion")
    d.mkdir(parents = True, exist_ok = True)
    return d


def sanitize_id(raw: str) -> str:
    """Filesystem-safe id from a repo id / folder name."""
    stem = raw.rsplit("/", 1)[-1]
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._-")
    return stem or "controlnet"


def _has_controlnet_weights(p: Path) -> bool:
    """Needs real weights, not only config, so an interrupted download is rejected before
    from_pretrained."""
    names = (
        "diffusion_pytorch_model.safetensors",
        "diffusion_pytorch_model.bin",
        "diffusion_pytorch_model.safetensors.index.json",
        "diffusion_pytorch_model.bin.index.json",
    )
    if any((p / n).exists() for n in names):
        return True

    try:
        return any(
            child.suffix == ".safetensors" and not is_appledouble_metadata(child)
            for child in p.iterdir()
        )
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
    """Refuses a ControlNet tagged for another family; it is architecture-specific and would load
    wrongly."""
    entry = _catalog_by_id().get(spec_id)
    if entry is None:
        entry = next((e for e in _CURATED if e.repo_id and e.repo_id == spec_id), None)
    if entry is not None:
        # A direct API call could send an entry for another family; reject it before any download.
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

    # A bare HF repo id (owner/name). STRICT shape so a filesystem-looking id can never reach from_pretrained.
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*", spec_id):
        return ResolvedControlNet(spec_id, spec_id, is_local = False)

    raise FileNotFoundError(
        f"unknown ControlNet '{spec_id}': not a local model, catalog entry, or HF repo id"
    )


# Union ControlNet mode indices: a union model selects its head via an integer ``control_mode``.
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
    """A typo'd control type raises so the route returns 400, never running the wrong union head."""
    entry = _catalog_by_id().get(spec_id)
    if entry is None:
        # match on repo_id too (the catalog is keyed by short id), else the union runs the wrong head
        entry = next((e for e in _CURATED if e.repo_id and e.repo_id == spec_id), None)
    if entry is None or not entry.is_union:
        return None
    ct = (control_type or "").strip().lower()
    if ct in _UNION_CONTROL_MODES:
        return _UNION_CONTROL_MODES[ct]
    if ct in ("", "passthrough"):
        return 0  # canny is the default head
    raise ValueError(
        f"Unknown control type {control_type!r} for a union ControlNet. Use one of: "
        f"{', '.join(sorted(_UNION_CONTROL_MODES))}, or passthrough."
    )


def preprocess_control(image: Any, control_type: str) -> Any:
    """Unknown control types pass through unchanged, so a new type never fails the request."""
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
        # Returning the source would condition on raw luminance.
        return Image.new("RGB", image.size, (0, 0, 0))
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
    """Blocked for GGUF and torchao fp8/int8 dense, which cannot host the extra conditioning cleanly."""
    if not family or not has_controlnet_pipeline:
        return False
    if engine != "diffusers":
        return False
    if model_kind == "gguf":
        return False
    if transformer_quant and str(transformer_quant).strip().lower() in _DIFFUSERS_BLOCKED_QUANT:
        return False
    return True
