# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The one list of video container extensions a training dataset may hold.

Unsloth's diffusion dataset layer grew up image-only, so the routes carry their own
``_DIFFUSION_DATASET_IMAGE_EXTS``. Clips need the same treatment, but a clip dataset is read
by the trainer and written by the upload endpoint, and those live on opposite sides of the
backend. Putting the set in either one makes the other import it across a layer it has no
business importing (the routes would pull a trainer module; the trainer would pull FastAPI).
So it lives here, in a module with no imports at all, and both sides read it from here.

Keeping it in one place is not cosmetic. Discovery, the upload allowlist and the duplicate-stem
check all have to agree on what a clip is: a container the upload accepts but discovery does
not count is a dataset that uploads fine and then trains on nothing.
"""

from __future__ import annotations

# Lowercase, with the leading dot, so a Path.suffix.lower() can be tested against the set directly.
CLIP_EXTS = frozenset({".mp4", ".mov", ".mkv", ".webm", ".m4v", ".avi"})

__all__ = ["CLIP_EXTS"]
