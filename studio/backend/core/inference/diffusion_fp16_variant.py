# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Read a pipeline's ``.fp16`` weight twins instead of its default fp32 files when the load casts to fp16 anyway.

SDXL base / turbo ship fp32 defaults beside fp16 twins (``unet/diffusion_pytorch_model.fp16.safetensors``...). On an
fp16 card ``from_pretrained(torch_dtype=float16)`` read the 12.9 GB fp32 set and cast it down; the twins are the same
weights stored at fp16 (6.5 GB), so ``variant="fp16"`` reads half the bytes and holds bit-identical tensors. Only for
an fp16 target: fp16 -> bf16 rounds differently from fp32 -> bf16, so a bf16 or fp32 load keeps the defaults.

Declined (defaults kept) unless EVERY component the pipeline loads from the repo has a single-file safetensors twin,
and no component class pins modules to fp32 (a pinned module would upcast fp16-rounded weights). Decided from a file
listing (Hub siblings for the download plan, the snapshot on disk for the load), never assumed.
Kill switch: ``UNSLOTH_DIFFUSION_FP16_VARIANT=0``.
"""

from __future__ import annotations

import importlib
import json
import os
import re
from typing import Any, Iterable, Mapping, Optional

FP16_VARIANT = "fp16"
FP16_VARIANT_ENV = "UNSLOTH_DIFFUSION_FP16_VARIANT"

_STEMS = ("diffusion_pytorch_model", "model")
_DEFAULT_SINGLE_RE = re.compile(r"^(diffusion_pytorch_model|model)\.safetensors$")
_TWIN_SINGLE_RE = re.compile(r"^(diffusion_pytorch_model|model)\.fp16\.safetensors$")
# Any weight-looking file of a component, default or variant, single or sharded, safetensors or .bin.
_WEIGHT_RE = re.compile(
    r"^(?:diffusion_pytorch_model|pytorch_model|model)[.-].*(?:bin|safetensors)(?:\.index.*\.json)?$"
)
_WEIGHT_LIBRARIES = ("diffusers", "transformers")


def fp16_variant_enabled() -> bool:
    return (os.environ.get(FP16_VARIANT_ENV) or "").strip().lower() not in (
        "0",
        "off",
        "false",
        "no",
    )


def is_fp16_dtype(dtype: Any) -> bool:
    return str(dtype) in ("torch.float16", "float16")


def is_fp16_twin(rfilename: str) -> bool:
    """True for a single-file ``.fp16`` safetensors twin (``<component>/<stem>.fp16.safetensors``)."""
    return bool(_TWIN_SINGLE_RE.match(str(rfilename).rsplit("/", 1)[-1]))


_DEFAULT_ANY_RE = re.compile(
    r"^(?:diffusion_pytorch_model|pytorch_model|model)(?:-\d{5}-of-\d{5})?\.(?:bin|safetensors)$"
)


def shadowed_twins(paths: Iterable[str]) -> set[str]:
    """The ``.fp16`` twins in ``paths`` whose folder also lists a default weight file: the default-variant load reads
    that one, so the twin is not part of what is held (a cache listing carrying both)."""
    paths = [str(p) for p in paths]
    default_dirs = {
        p.rpartition("/")[0] for p in paths if _DEFAULT_ANY_RE.match(p.rpartition("/")[2])
    }
    return {p for p in paths if is_fp16_twin(p) and p.rpartition("/")[0] in default_dirs}


def twin_of(rfilename: str) -> Optional[str]:
    """``unet/diffusion_pytorch_model.safetensors`` -> ``unet/diffusion_pytorch_model.fp16.safetensors``; None for
    anything that is not a single-file default safetensors (shards, .bin, twins themselves)."""
    folder, _, name = str(rfilename).rpartition("/")
    match = _DEFAULT_SINGLE_RE.match(name)
    if not folder or match is None:
        return None
    return f"{folder}/{match.group(1)}.fp16.safetensors"


def _class_pins_fp32(library: Any, class_name: Any) -> bool:
    """True when the component class keeps modules in fp32 at an fp16 load, or cannot be resolved (conservative)."""
    if library not in _WEIGHT_LIBRARIES or not isinstance(class_name, str) or not class_name:
        return True
    try:
        cls = getattr(importlib.import_module(library), class_name, None)
    except Exception:  # noqa: BLE001 - an unimportable class decides nothing: keep the defaults
        return True
    if cls is None:
        return True
    return bool(
        getattr(cls, "_keep_in_fp32_modules", None)
        or getattr(cls, "_keep_in_fp32_modules_strict", None)
    )


def plan_fp16_twins(
    names: Iterable[str],
    model_index: Optional[Mapping[str, Any]],
    *,
    components: Optional[Iterable[str]] = None,
) -> Optional[dict[str, str]]:
    """``{component: twin file}`` when every weight-bearing component the load reads has a single-file fp16 twin in
    ``names`` (repo-relative paths), else None.

    ``components`` limits the check to the folders the load reads from the repo (a pre-cast encoder or a supplied
    denoiser is not read there); default: every manifest entry with a library and class. A component counts as
    weight-bearing when ``names`` holds any weight file under it; a weightless one (tokenizer, scheduler) is skipped.
    """
    if not isinstance(model_index, Mapping):
        return None
    specs: dict[str, tuple[Any, Any]] = {}
    for name, spec in model_index.items():
        if str(name).startswith("_") or not isinstance(spec, (list, tuple)) or len(spec) < 2:
            continue
        if spec[0] is None or spec[1] is None:
            continue
        specs[str(name)] = (spec[0], spec[1])
    wanted = set(specs) if components is None else {str(c) for c in components} & set(specs)
    by_folder: dict[str, set[str]] = {}
    for rel in names:
        folder, _, leaf = str(rel).rpartition("/")
        if folder in wanted and "/" not in folder:
            by_folder.setdefault(folder, set()).add(leaf)
    plan: dict[str, str] = {}
    for folder in sorted(wanted):
        leaves = by_folder.get(folder, set())
        if not any(_WEIGHT_RE.match(leaf) for leaf in leaves):
            continue  # tokenizer / scheduler / feature extractor
        twins = sorted(leaf for leaf in leaves if _TWIN_SINGLE_RE.match(leaf))
        if len(twins) != 1:
            return None  # no twin, or an ambiguous pair (diffusion_pytorch_model + model)
        library, class_name = specs[folder]
        if _class_pins_fp32(library, class_name):
            return None
        plan[folder] = f"{folder}/{twins[0]}"
    return plan or None


def read_model_index(snapshot: Optional[str]) -> Optional[dict]:
    if not snapshot:
        return None
    try:
        with open(os.path.join(snapshot, "model_index.json"), "r", encoding = "utf-8") as handle:
            payload = json.load(handle)
    except Exception:  # noqa: BLE001 - no readable manifest: no variant
        return None
    return payload if isinstance(payload, dict) else None


def _snapshot_names(snapshot: str, folders: Iterable[str]) -> list[str]:
    names: list[str] = []
    for folder in folders:
        path = os.path.join(snapshot, folder)
        try:
            entries = os.listdir(path)
        except OSError:
            continue
        for entry in entries:
            full = os.path.join(path, entry)
            try:
                if os.path.isfile(full) and os.path.getsize(full) > 0:
                    names.append(f"{folder}/{entry}")
            except OSError:
                continue
    return names


def _defaults_complete(names: Iterable[str], plan: Mapping[str, str]) -> bool:
    present = set(names)
    return all(any(f"{folder}/{stem}.safetensors" in present for stem in _STEMS) for folder in plan)


def local_fp16_variant(
    snapshot: Optional[str],
    *,
    load_dtype: Any,
    torch_dtype: Any = None,
    supplied: Iterable[str] = (),
) -> Optional[str]:
    """``"fp16"`` when ``from_pretrained(snapshot)`` should read the twins, else None.

    Twins are read when every component the load reads from ``snapshot`` (manifest entries minus ``supplied``
    objects) has its twin on disk, and either the load is a plain fp16 one (``torch_dtype`` the same fp16 dtype, not
    a per-component map), or the default files are missing (a twins-only snapshot an fp16 load staged: the only set
    on disk that loads at all).
    """
    if not fp16_variant_enabled() or not snapshot or not os.path.isdir(snapshot):
        return None
    index = read_model_index(snapshot)
    if index is None:
        return None
    skip = {str(s) for s in supplied}
    components = [
        name
        for name, spec in index.items()
        if not str(name).startswith("_")
        and name not in skip
        and isinstance(spec, (list, tuple))
        and len(spec) >= 2
        and spec[0] is not None
        and spec[1] is not None
    ]
    names = _snapshot_names(snapshot, components)
    plan = plan_fp16_twins(names, index, components = components)
    if plan is None:
        return None
    plain_fp16 = is_fp16_dtype(load_dtype) and (torch_dtype is None or is_fp16_dtype(torch_dtype))
    if plain_fp16 or not _defaults_complete(names, plan):
        return FP16_VARIANT
    return None


__all__ = [
    "FP16_VARIANT",
    "FP16_VARIANT_ENV",
    "fp16_variant_enabled",
    "is_fp16_dtype",
    "is_fp16_twin",
    "local_fp16_variant",
    "plan_fp16_twins",
    "read_model_index",
    "shadowed_twins",
    "twin_of",
]
