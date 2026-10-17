# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Recipe prompts for agents: each one reads what Unsloth Studio has right now and spells out the tool calls for a whole job."""

# No ``from __future__ import annotations`` here: fastmcp tells agents to send any argument whose
# annotation is not the ``str`` type itself as a "JSON string", and a postponed annotation is the text "str".

import asyncio
from typing import Any

from studio_mcp.tools import models, status

MAX_CHARS = 4000
# What the agent asked for is clipped so the recipe always fits.
MAX_ASK_CHARS = 800
CANDIDATES = 5
# GpuDevice reports GiB; sizes and fits_now use decimal GB.
GIB_TO_GB = 1.073741824
FALLBACK = (
    "Unsloth Studio could not be read from here: call studio_status and "
    'list_models(source="catalog") first.'
)


def _gb(value: float) -> str:
    return f"{value:.1f} GB"


def _gpu_line(studio: Any) -> str:
    devices = []
    for device in studio.hardware.devices:
        if device.vram_total_gb is None or device.vram_used_gb is None:
            continue
        free = max(0.0, device.vram_total_gb - device.vram_used_gb) * GIB_TO_GB
        name = f"GPU {device.index}" if device.index is not None else "GPU"
        devices.append(f"{name} {_gb(free)} free of {_gb(device.vram_total_gb * GIB_TO_GB)}")
    if not devices:
        return "Unsloth Studio reports no GPU memory right now; check studio_status before loading."
    return "Unsloth Studio free GPU memory: " + "; ".join(devices) + "."


def _loaded_line(studio: Any) -> str:
    loaded = [model.id for model in studio.chat.loaded]
    loaded += [slot.model for slot in (studio.image, studio.video) if slot.loaded and slot.model]
    if studio.stt.model:
        loaded.append(studio.stt.model)
    if not loaded:
        return "Unsloth Studio has no model loaded."
    return "Unsloth Studio has loaded: " + ", ".join(loaded) + "."


def _catalog_line(kind: str, rows: Any) -> str:
    fitting = [row for row in rows.models if getattr(row, "fits_now", None) is True]
    fitting.sort(key = lambda row: getattr(row, "size_bytes", None) or 0)
    if not fitting:
        return f"- {kind}: no Unsloth Studio catalog model fits the free GPU memory now; unload a model or try a low_vram row."
    shown = []
    for row in fitting[:CANDIDATES]:
        size = getattr(row, "size_bytes", None)
        shown.append(f"{row.id} ({_gb(size / 1e9)})" if size else row.id)
    return f"- {kind}: " + ", ".join(shown)


def _local_line(kind: str, rows: Any) -> str:
    ids = [row.id + (" (loaded)" if row.loaded else "") for row in rows.models[:CANDIDATES]]
    if not ids:
        return f"- {kind}: none downloaded in Unsloth Studio yet; start_training downloads a Hugging Face id."
    return f"- {kind}: " + ", ".join(ids)


async def _facts(kinds: tuple[str, ...], local: bool = False) -> str:
    """The GPU, loaded models and model candidates as text; never raises, so a prompt renders without HTTP."""

    async def read(kind: str) -> Any:
        if local:
            return await models.list_models(kind = kind)
        return await models.list_models(kind = kind, source = "catalog")

    results = await asyncio.gather(
        status.studio_status(), *(read(kind) for kind in kinds), return_exceptions = True
    )
    studio, listings = results[0], results[1:]
    if all(isinstance(result, BaseException) for result in results):
        return FALLBACK
    lines = (
        [FALLBACK]
        if isinstance(studio, BaseException)
        else [_gpu_line(studio), _loaded_line(studio)]
    )
    title = "Downloaded models" if local else "Catalog models that fit now, smallest first"
    lines.append(f"{title} (Unsloth Studio, at most {CANDIDATES} each):")
    for kind, rows in zip(kinds, listings):
        if isinstance(rows, BaseException):
            lines.append(f'- {kind}: unknown; call list_models(kind="{kind}") yourself.')
        else:
            lines.append(_local_line(kind, rows) if local else _catalog_line(kind, rows))
    return "\n".join(lines)


def _clip(text: str) -> str:
    text = text.strip()
    return text if len(text) <= MAX_ASK_CHARS else text[:MAX_ASK_CHARS] + "…"


def _capped(text: str) -> str:
    return text if len(text) <= MAX_CHARS else text[: MAX_CHARS - 1] + "…"


async def make_short_film(idea: str, length_seconds: str = "20") -> str:
    """Plan and render a short film with Unsloth Studio: video shots, then music or a voice-over.

    Args:
        idea: What the film is about.
        length_seconds: The total length to aim for, in seconds.
    """
    facts = await _facts(("video", "audio", "tts"))
    length = length_seconds.strip() or "20"
    return _capped(
        f"Make a short film of about {length} seconds with Unsloth Studio.\n"
        f"Idea: {_clip(idea)}\n\n"
        f"What Unsloth Studio has now:\n{facts}\n\n"
        "Steps:\n"
        "1. Call studio_status to see the GPUs and what Unsloth Studio has loaded.\n"
        '2. Call list_models(source="catalog", kind="video") and pick a row with fits_now true, '
        "or a low_vram row on a small GPU.\n"
        '3. Call load_model(model=<id>, kind="video"); Unsloth Studio downloads it first if needed.\n'
        f"4. Write a shot list: Unsloth Studio video models make clips of a few seconds, so plan several "
        f"short shots that add up to about {length} seconds.\n"
        "5. Call generate_video once per shot with a self-contained prompt, keeping characters, style "
        "and lighting the same in every shot.\n"
        '6. Poll get_job(kind="video", id=<job id>) until each shot is completed and keep its URL.\n'
        '7. Call unload_model(kind="video") before loading an audio model, so Unsloth Studio has room.\n'
        '8. For music, load a list_models(source="catalog", kind="audio") row and call '
        'generate_audio(workflow="music", mode="song", instrumental=true, duration_s=<length>); '
        'for a voice-over, load a kind="tts" row and call generate_audio(workflow="speak", text=<script>).\n'
        "9. Unload each model when you are done with it, then download the clips and audio from their "
        "Unsloth Studio URLs and stitch them yourself, for example with ffmpeg.\n"
    )


async def build_dataset_and_train(goal: str, base_model: str = "") -> str:
    """Build a training dataset with Unsloth Studio, fine-tune a model on it and export the result.

    Args:
        goal: What the fine-tuned model should do.
        base_model: A Hugging Face model id to start from; leave empty to pick one.
    """
    facts = await _facts(("llm",), local = True)
    base = base_model.strip()
    pick = (
        f'Start from {base}; list_models(model="{base}") returns Unsloth Studio\'s training defaults for it.'
        if base
        else "Pick a small base model: a downloaded one below, or a Hugging Face id such as unsloth/Qwen3-0.6B."
    )
    return _capped(
        "Build a dataset and fine-tune a model with Unsloth Studio.\n"
        f"Goal: {_clip(goal)}\n\n"
        f"What Unsloth Studio has now:\n{facts}\n\n"
        "Steps:\n"
        f"1. {pick}\n"
        '2. Find data with datasets(action="list"), or datasets(action="download", repo_id=<Hub id>), and '
        'check it with datasets(action="check_format", name=<dataset>).\n'
        "3. To make data instead, write a Data Recipe, check it with run_recipe(recipe=<recipe>, "
        'mode="validate"), try mode="preview", then run mode="full" and follow '
        'get_job(kind="recipe") until Unsloth Studio has saved the dataset.\n'
        "4. Build the training config and check it with start_training(config=<config>, "
        "validate_only=true), which starts nothing in Unsloth Studio.\n"
        "5. Call start_training(config=<config>) and poll studio_status until Unsloth Studio "
        "reports the run finished.\n"
        "6. Find the run with list_training_runs(include_checkpoints=true).\n"
        '7. Call export_model(checkpoint=<run>, format="gguf", save_directory=<folder>) and follow '
        'get_job(kind="export") until Unsloth Studio has written it.\n'
    )


PROMPTS = (
    (make_short_film, "Make a short film"),
    (build_dataset_and_train, "Build a dataset and train"),
)
