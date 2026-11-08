# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""When a diffusers load may pass ``use_safetensors = True``, and which repo files a diffusers pipeline download skips.

``use_safetensors = True`` turns diffusers' (and transformers') silent pickle fallback into an error, so it is passed
only when the folder the load reads is known to ship ``.safetensors`` weights for every weight-bearing component. A
folder that cannot be inspected (a Hub id nothing is cached for) keeps the library default, which is safetensors
first and a ``.bin`` only for a component that has no safetensors at all. No torch, never raises."""

from __future__ import annotations

import json
import os
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Optional

from .safetensors_only import is_pickle_weight

_MANIFESTS = ("model_index.json", "modular_model_index.json")
# Files a diffusers / transformers model load could read as weights.
_WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth", ".ckpt", ".pkl", ".pickle")
# dtype-variant twins (``model.fp16.safetensors``): a load without ``variant`` never reads them.
_VARIANT_MARKERS = (".fp16.", ".bf16.", ".non_ema.", ".fp32.")
# Pickle containers no diffusers / transformers component load reads: always skipped in a pipeline download.
_NEVER_READ_PICKLES = (".pt", ".pth", ".ckpt", ".pkl", ".pickle")


def _is_variant(name: str) -> bool:
    low = name.lower()
    return any(m in low for m in _VARIANT_MARKERS)


def component_weight_state(folder: Any) -> Optional[bool]:
    """True: ``folder`` holds a non-variant ``.safetensors`` weight (or its index); False: it holds weights and none
    is safetensors; None: no weight file at all (or unreadable)."""
    try:
        names = [p.name for p in Path(str(folder)).iterdir() if p.is_file()]
    except OSError:
        return None
    weights = [
        n for n in names if n.lower().endswith(_WEIGHT_SUFFIXES) or n.endswith(".index.json")
    ]
    if not weights:
        return None
    if any(
        (n.endswith(".safetensors") or n.endswith(".safetensors.index.json")) and not _is_variant(n)
        for n in weights
    ):
        return True
    return False if any(not n.endswith(".index.json") for n in weights) else None


def _manifest_components(root: Path) -> Optional[list]:
    for manifest in _MANIFESTS:
        path = root / manifest
        if not path.is_file():
            continue
        try:
            data = json.loads(path.read_text(encoding = "utf-8"))
        except Exception:  # noqa: BLE001 - unreadable manifest: unknown
            return None
        if not isinstance(data, dict):
            return None
        out = []
        for key, value in data.items():
            if key.startswith("_"):
                continue
            # classic manifests: [library, class]; modular ones: [library, class, {spec}] or nested specs
            if isinstance(value, (list, tuple)) and value and value[0] is not None:
                out.append((key, value[1] if len(value) > 1 else None))
        return out
    return None


# Manifest classes that never hold weights (a scheduler / tokenizer folder is config only).
_WEIGHTLESS_CLASS_SUFFIXES = (
    "Scheduler",
    "Tokenizer",
    "TokenizerFast",
    "Processor",
    "FeatureExtractor",
)


def _weightless(class_name: Any) -> bool:
    return isinstance(class_name, str) and class_name.endswith(_WEIGHTLESS_CLASS_SUFFIXES)


def local_pipeline_safetensors_complete(root: Any, passed: Iterable[str] = ()) -> bool:
    """True when ``root`` is a local diffusers pipeline folder whose every manifest component that holds weights
    (and is not supplied pre-built in ``passed``) has a ``.safetensors`` weight. False when unknown (a weight-bearing
    component missing or without weights, e.g. a partly cached snapshot) or any component is pickle-only."""
    try:
        base = Path(str(root)).expanduser()
        if not base.is_dir():
            return False
    except (OSError, ValueError):
        return False
    components = _manifest_components(base)
    if components is None:
        return False
    skip = set(passed or ())
    seen = False
    for name, class_name in components:
        if name in skip:
            continue
        folder = base / name
        state = component_weight_state(folder) if folder.is_dir() else None
        if state is False:
            return False
        if state is None:
            if _weightless(class_name):
                continue
            return False
        seen = True
    return seen


def _cached_snapshot_dir(repo_id: str, probe: str, cache_dir: Optional[str]) -> Optional[Path]:
    """The cached snapshot folder holding ``repo_id/probe``, or None. Offline: reads the cache only."""
    try:
        from huggingface_hub import try_to_load_from_cache
    except Exception:  # noqa: BLE001
        return None
    for root in dict.fromkeys((cache_dir, None)):
        try:
            hit = try_to_load_from_cache(repo_id, probe, cache_dir = root)
        except Exception:  # noqa: BLE001
            continue
        if isinstance(hit, str) and os.path.isfile(hit):
            return Path(hit).parents[len(probe.split("/")) - 1]
    return None


def _takes_use_safetensors(cls: Any, base_name: str) -> bool:
    """Only diffusers' own ``DiffusionPipeline`` / ``ModelMixin`` subclasses are known to accept the kwarg."""
    if cls is None:
        return True
    try:
        import diffusers
        base = getattr(diffusers, base_name, None)
        return isinstance(cls, type) and base is not None and issubclass(cls, base)
    except Exception:  # noqa: BLE001
        return False


def pipeline_safetensors_kwargs(
    source: Any,
    *,
    passed: Iterable[str] = (),
    cache_dir: Optional[str] = None,
    pipeline_cls: Any = None,
) -> dict:
    """``{"use_safetensors": True}`` for a pipeline ``from_pretrained(source)`` whose folder (``source`` itself, or the
    cached snapshot of a Hub id) ships safetensors for every weight-bearing component; ``{}`` otherwise (and for a
    ``pipeline_cls`` that is not a diffusers ``DiffusionPipeline``)."""
    try:
        if not _takes_use_safetensors(pipeline_cls, "DiffusionPipeline"):
            return {}
        text = str(source or "").strip()
        if not text:
            return {}
        if os.path.isdir(os.path.expanduser(text)):
            root: Optional[Path] = Path(text).expanduser()
        else:
            root = None
            for manifest in _MANIFESTS:
                root = _cached_snapshot_dir(text, manifest, cache_dir)
                if root is not None:
                    break
        if root is not None and local_pipeline_safetensors_complete(root, passed):
            return {"use_safetensors": True}
    except Exception:  # noqa: BLE001 - unknown keeps the library default
        pass
    return {}


def component_safetensors_kwargs(
    source: Any,
    subfolder: str,
    *,
    cache_dir: Optional[str] = None,
    model_cls: Any = None,
) -> dict:
    """``{"use_safetensors": True}`` for ``from_pretrained(source, subfolder = subfolder)`` when that component folder
    (local, or in the cached snapshot of a Hub id) holds a safetensors weight; ``{}`` otherwise (and for a
    ``model_cls`` that is not a diffusers ``ModelMixin``)."""
    try:
        if not _takes_use_safetensors(model_cls, "ModelMixin"):
            return {}
        text = str(source or "").strip()
        if not text or not subfolder:
            return {}
        if os.path.isdir(os.path.expanduser(text)):
            folder: Optional[Path] = Path(text).expanduser() / subfolder
        else:
            snap = _cached_snapshot_dir(text, f"{subfolder}/config.json", cache_dir)
            folder = snap / subfolder if snap is not None else None
        if folder is not None and component_weight_state(folder) is True:
            return {"use_safetensors": True}
    except Exception:  # noqa: BLE001
        pass
    return {}


def pipeline_pickle_files(filenames: Iterable[str]) -> list:
    """Repo files a diffusers pipeline download skips (exact names, for a repo with a pipeline manifest at its root):
    every ``.pt`` / ``.pth`` / ``.ckpt`` / ``.pkl`` (no component loader reads them), and every ``.bin`` in a folder that
    also holds a non-variant ``.safetensors`` weight. A ``.bin`` in a folder without safetensors stays (the only copy of
    that component). Empty for any other repo."""
    names = [n for n in filenames if isinstance(n, str)]
    if not any(n in _MANIFESTS for n in names):
        return []
    st_dirs = {
        n.rsplit("/", 1)[0] if "/" in n else ""
        for n in names
        if (n.endswith(".safetensors") or n.endswith(".safetensors.index.json"))
        and not _is_variant(n.rsplit("/", 1)[-1])
    }
    out = []
    for n in names:
        low = n.lower()
        folder = n.rsplit("/", 1)[0] if "/" in n else ""
        if low.endswith(_NEVER_READ_PICKLES):
            out.append(n)
        elif (low.endswith(".bin") or low.endswith(".bin.index.json")) and folder in st_dirs:
            out.append(n)
    return out


def is_never_read_pickle(name: Optional[str]) -> bool:
    """A pickle container (``.pt`` / ``.pth`` / ``.ckpt`` / ``.pkl``) no diffusers component load reads."""
    return bool(name) and is_pickle_weight(name) and str(name).lower().endswith(_NEVER_READ_PICKLES)
