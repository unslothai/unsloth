# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Path-only checks on local diffusion checkpoints. Model listing and catalog classification call
them, so they live apart from core.inference.diffusion and its pipeline stack."""

from pathlib import Path
from typing import Optional

from utils.paths.path_utils import is_appledouble_metadata

from .diffusion_flux3 import FLUX3_INDEX


def resolve_local_single_file(model_path: str) -> Optional[str]:
    """The sole single-file checkpoint basename in a local ``model_path`` directory that is NOT a
    diffusers pipeline (no ``model_index.json`` or FLUX.3 ``modular_model_index.json``) and holds exactly one ``.safetensors`` file, else
    None.

    The On-Device scanner advertises a bare single-file safetensors directory as a text-to-image
    model, but the local picker starts it as a ``pipeline`` with no filename, so a pipeline load
    400s on the missing ``model_index.json``. The images load route uses this to reinterpret such a
    pick as a ``single_file`` load of the sole checkpoint. A real pipeline dir, or an ambiguous one
    (0 or more than 1 ``.safetensors``), returns None and loads unchanged. A PEFT LoRA adapter
    folder is also skipped. Never raises.
    """
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
