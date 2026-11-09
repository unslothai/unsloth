# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Reads .fp16 weight twins on fp16 loads only, and only when every component has a single-file twin
and no class pins fp32 modules; bit-identical after cast; UNSLOTH_DIFFUSION_FP16_VARIANT=0
disables."""

from __future__ import annotations

import json
import os
import re
from typing import Any, Iterable, Mapping, Optional

FP16_VARIANT = "fp16"
FP16_VARIANT_ENV = "UNSLOTH_DIFFUSION_FP16_VARIANT"

_STEMS = ("diffusion_pytorch_model", "model")
_DEFAULT_SINGLE_RE = re.compile(r"^(diffusion_pytorch_model|model)\.safetensors$")
_TWIN_SINGLE_RE = re.compile(r"^(diffusion_pytorch_model|model)\.fp16\.safetensors$")
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
    return bool(_TWIN_SINGLE_RE.match(str(rfilename).rsplit("/", 1)[-1]))


_DEFAULT_ANY_RE = re.compile(
    r"^(?:diffusion_pytorch_model|pytorch_model|model)(?:-\d{5}-of-\d{5})?\.(?:bin|safetensors)$"
)


def shadowed_twins(paths: Iterable[str]) -> set[str]:
    """Twins in ``paths`` whose folder also has a default weight file (the default load reads that one)."""
    paths = [str(p) for p in paths]
    default_dirs = {
        p.rpartition("/")[0] for p in paths if _DEFAULT_ANY_RE.match(p.rpartition("/")[2])
    }
    return {p for p in paths if is_fp16_twin(p) and p.rpartition("/")[0] in default_dirs}


def twin_of(rfilename: str) -> Optional[str]:
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
        if library == "diffusers":
            import diffusers as module
        else:
            import transformers as module
        cls = getattr(module, class_name, None)
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
    """``{component: twin}`` when every weight-bearing component has a single-file twin in ``names``, else None."""
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
            continue
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


def is_default_weight(basename: str) -> bool:
    if _DEFAULT_ANY_RE.match(basename):
        return True
    return (
        basename.endswith(".index.json")
        and ".fp16." not in basename
        and bool(_WEIGHT_RE.match(basename))
    )


def _defaults_present(names: Iterable[str], plan: Mapping[str, str]) -> bool:
    # Any default layout (single, sharded, .bin, an index) counts: only a folder with none reads the twins at bf16 / fp32.
    by_folder: dict[str, list[str]] = {}
    for name in names:
        folder, _, base = str(name).rpartition("/")
        by_folder.setdefault(folder, []).append(base)
    return all(any(is_default_weight(b) for b in by_folder.get(folder, ())) for folder in plan)


def local_fp16_variant(
    snapshot: Optional[str],
    *,
    load_dtype: Any,
    torch_dtype: Any = None,
    supplied: Iterable[str] = (),
) -> Optional[str]:
    """``"fp16"`` when every non-supplied component has a twin on disk and the load is plain fp16 or twins-only."""
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
    if plain_fp16 or not _defaults_present(names, plan):
        return FP16_VARIANT
    return None


__all__ = [
    "FP16_VARIANT",
    "FP16_VARIANT_ENV",
    "fp16_variant_enabled",
    "is_default_weight",
    "is_fp16_dtype",
    "is_fp16_twin",
    "local_fp16_variant",
    "plan_fp16_twins",
    "read_model_index",
    "shadowed_twins",
    "twin_of",
]
