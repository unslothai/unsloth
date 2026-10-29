# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""``cancel``: stop Unsloth Studio work by kind."""

from __future__ import annotations

from typing import Literal, Optional

from studio_mcp import profile
from studio_mcp import jobs
from studio_mcp.caller import current_caller
from studio_mcp.outputs import CancelResult
from studio_mcp.tools import DESTRUCTIVE

CancelKind = Literal[
    "training",
    "training_start",
    "diffusion_training",
    "export",
    "recipe",
    "image",
    "video",
    "audio",
    "chat",
    "dataset_download",
    "model_download",
]


async def cancel(
    kind: CancelKind,
    id: Optional[str] = None,
    save: bool = True,
) -> CancelResult:
    """Stop Unsloth Studio work.
    "training" stops the LLM training run ``id`` (or the current one) at its next safe point, saving a checkpoint unless ``save`` is false.
    "training_start" withdraws a start request that has not begun.
    "diffusion_training" stops image LoRA training, saving the partial adapter unless ``save`` is false.
    "export" stops the running export and unloads its checkpoint; with ``id`` (an export_model job) only that job's own step, never someone else's export.
    "recipe" stops recipe job ``id``.
    "image" stops the generation in progress; "video" stops the render in progress, or with ``id`` only that video, refusing if it is not the one rendering.
    "chat" stops the reply with cancel_id ``id``.
    "audio" stops the generate_audio run whose cancel_id is ``id``; takes it already finished stay in Audio history.
    "dataset_download" stops downloading the dataset ``id`` (its repo id).
    "model_download" stops the model download ``id`` that download_model returned; a speech-to-text download can't be stopped here.
    ``cancelled`` says whether this call stopped something.
    """
    return await jobs.cancel(current_caller(), kind, id, save)


FACTS = {"cancel": {profile.STOPS_WORK}}

TOOLS = ((cancel, DESTRUCTIVE),)
