# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The public Unsloth repo that Unsloth's loader substitutes for an upstream model id."""

from __future__ import annotations

import ast
import importlib.util
from functools import lru_cache
from pathlib import Path
from typing import Optional

from loggers import get_logger
from utils.paths import is_local_path

logger = get_logger(__name__)


def _unsloth_models_dir() -> Optional[Path]:
    spec = importlib.util.find_spec("unsloth")
    locations = list(spec.submodule_search_locations or ()) if spec else []
    return Path(locations[0]) / "models" if locations else None


@lru_cache(maxsize = 1)
def _mapper_tables() -> Optional[tuple[dict, dict, dict]]:
    # Load the data file directly to avoid initializing Unsloth's GPU stack.
    try:
        models_dir = _unsloth_models_dir()
        if models_dir is None:
            return None
        mapper_spec = importlib.util.spec_from_file_location(
            "_studio_unsloth_mapper", models_dir / "mapper.py"
        )
        module = importlib.util.module_from_spec(mapper_spec)
        mapper_spec.loader.exec_module(module)
        return (
            module.INT_TO_FLOAT_MAPPER,
            module.FLOAT_TO_INT_MAPPER,
            module.MAP_TO_UNSLOTH_16bit,
        )
    except Exception as error:
        logger.debug("Could not read the Unsloth model mapper: %s", error)
        return None


def _string_literal(node) -> Optional[str]:
    """``"Repo".lower()`` and ``"Repo"`` as data. BAD_MAPPINGS is written with ``.lower()``
    calls on both sides, so ast.literal_eval alone cannot read it."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if (
        isinstance(node, ast.Call)
        and not node.args
        and not node.keywords
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in ("lower", "upper", "strip")
    ):
        inner = _string_literal(node.func.value)
        return getattr(inner, node.func.attr)() if inner is not None else None
    return None


@lru_cache(maxsize = 1)
def _bad_mappings() -> Optional[dict]:
    """``loader_utils.BAD_MAPPINGS``, the corrections the loader applies AFTER a table lookup
    (a 4-bit dynamic quant that is too big, or a MoE that HF loads too slowly). Parsed out of
    the source, because importing loader_utils imports torch.

    ``None`` means the corrections could not be read, which is NOT the same as an empty table:
    an uncorrected lookup can name a repo that does not exist, so the caller redirects nothing
    rather than redirect somewhere the loader will never go."""
    try:
        models_dir = _unsloth_models_dir()
        if models_dir is None:
            return None
        tree = ast.parse((models_dir / "loader_utils.py").read_text(encoding = "utf-8"))
        for node in tree.body:
            if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Dict):
                continue
            if not any(
                isinstance(target, ast.Name) and target.id == "BAD_MAPPINGS"
                for target in node.targets
            ):
                continue
            table = {}
            for key, value in zip(node.value.keys, node.value.values):
                name, replacement = _string_literal(key), _string_literal(value)
                if name is None or replacement is None:
                    # An entry we cannot read as data. Reading the rest would silently drop
                    # whichever correction that entry was.
                    return None
                table[name.lower()] = replacement
            return table
    except Exception as error:
        logger.debug("Could not read the Unsloth loader corrections: %s", error)
    return None


def mirror_lookup_available() -> bool:
    """Whether a None from :func:`unsloth_public_mirror` means "no public copy" at all.

    It can also mean "could not tell": the tables live in the installed unsloth package, and
    find_spec can resolve to a directory that has no models/mapper.py under it. Callers that
    REFUSE on a missing mirror have to tell those apart, or an unreadable table turns every
    gated model into a refusal, including the ones Unsloth would have trained.
    """
    return _mapper_tables() is not None and _bad_mappings() is not None


def unsloth_public_mirror(model_name: Optional[str], load_in_4bit: bool = True) -> Optional[str]:
    """Return the public repo substituted by Unsloth's loader for this load mode, if any."""
    if (
        not isinstance(model_name, str)
        or model_name.strip().count("/") != 1
        or is_local_path(model_name)
    ):
        return None
    tables = _mapper_tables()
    if tables is None:
        return None
    int_to_float, float_to_int, map_to_unsloth_16bit = tables
    lower = model_name.strip().lower()
    if lower.startswith("unsloth/"):
        return None
    if load_in_4bit:
        # A 4-bit load resolves through FLOAT_TO_INT_MAPPER alone, and keeps an explicit
        # -bnb-4bit name as given (unsloth.models.loader_utils.__get_model_name).
        mirror = None if lower.endswith("-bnb-4bit") else float_to_int.get(lower)
    else:
        mirror = int_to_float.get(lower) or map_to_unsloth_16bit.get(lower)
    # get_model_name corrects the resolved name, and corrects the INPUT name when the tables
    # resolved nothing. Without this, a 4-bit Qwen/Qwen3-30B-A3B resolves to
    # unsloth/Qwen3-30B-A3B-unsloth-bnb-4bit, which does not exist: the loader never fetches it.
    corrections = _bad_mappings()
    if corrections is None:
        return None
    if isinstance(mirror, str):
        mirror = corrections.get(mirror.lower(), mirror)
    else:
        mirror = corrections.get(lower)
    if not isinstance(mirror, str) or mirror.lower() == lower:
        return None
    return mirror
