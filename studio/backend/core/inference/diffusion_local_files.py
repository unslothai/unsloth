# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Path-only diffusion checks, kept out of core.inference.diffusion so model listing stays light."""

from pathlib import Path
from typing import Optional

from utils.paths.path_utils import is_appledouble_metadata

from .diffusion_flux3 import FLUX3_INDEX


def resolve_local_single_file(model_path: str) -> Optional[str]:
    """The sole ``.safetensors`` basename in a local non-pipeline, non-adapter dir, else None."""
    try:
        root = Path(model_path).expanduser()
        if not root.is_dir() or any(
            (root / index).is_file() for index in ("model_index.json", FLUX3_INDEX)
        ):
            return None
        # A PEFT adapter folder is not a base checkpoint; skip it so validation 400s before eviction
        if (root / "adapter_config.json").is_file():
            return None

        checkpoints = [
            p.name
            for p in root.iterdir()
            if p.is_file()
            and p.suffix.lower() == ".safetensors"
            and p.stem.lower() != "adapter_model"
            and not is_appledouble_metadata(p)
        ]
    except OSError:
        return None
    return checkpoints[0] if len(checkpoints) == 1 else None
