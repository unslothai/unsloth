# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The public Unsloth repo that Unsloth's loader substitutes for an upstream model id."""

from __future__ import annotations

import importlib.util
from functools import lru_cache
from pathlib import Path
from typing import Optional

from loggers import get_logger
from utils.paths import is_local_path

logger = get_logger(__name__)


@lru_cache(maxsize = 1)
def _mapper_tables() -> Optional[tuple[dict, dict]]:
    # Load the data file directly to avoid initializing Unsloth's GPU stack.
    try:
        spec = importlib.util.find_spec("unsloth")
        locations = list(spec.submodule_search_locations or ()) if spec else []
        if not locations:
            return None
        mapper_spec = importlib.util.spec_from_file_location(
            "_studio_unsloth_mapper", Path(locations[0]) / "models" / "mapper.py"
        )
        module = importlib.util.module_from_spec(mapper_spec)
        mapper_spec.loader.exec_module(module)
        return module.INT_TO_FLOAT_MAPPER, module.MAP_TO_UNSLOTH_16bit
    except Exception as error:
        logger.debug("Could not read the Unsloth model mapper: %s", error)
        return None


def unsloth_16bit_mirror(model_name: Optional[str]) -> Optional[str]:
    """Return the public repo substituted by Unsloth's 16-bit loader, if any."""
    if (
        not isinstance(model_name, str)
        or model_name.strip().count("/") != 1
        or is_local_path(model_name)
    ):
        return None
    tables = _mapper_tables()
    if tables is None:
        return None
    int_to_float, map_to_unsloth_16bit = tables
    lower = model_name.strip().lower()
    if lower.startswith("unsloth/"):
        return None
    mirror = int_to_float.get(lower) or map_to_unsloth_16bit.get(lower)
    if not isinstance(mirror, str) or mirror.lower() == lower:
        return None
    return mirror
