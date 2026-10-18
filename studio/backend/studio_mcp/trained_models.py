# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Load a checkpoint or an export by name. The listing routes give host paths; the load goes out with a ``ref:`` handle minted from that path, and the agent only ever sees the name it asked for."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Optional

from hub.utils.host_paths import cache_reference, redact_paths_in_text
from studio_mcp import checkpoints
from studio_mcp.caller import Caller
from studio_mcp.errors import tool_error
from studio_mcp.outputs import LoadResult
from studio_mcp.tools import as_dict, leaf_name, opt_text, route_json

NOT_LLM_KINDS = ("image", "video", "stt")
# The name must resolve before a load job can exist, so even wait_s=0 gives the listing reads this long.
RESOLVE_TIMEOUT_S = 5.0


@dataclass(frozen = True)
class TrainedModel:
    name: str
    ref: str = field(repr = False)
    is_lora: bool
    gguf_variant: Optional[str] = None
    # A GGUF export whose quantization its file name doesn't give; the load can't place it by device.
    unplaced_gguf: bool = False
    # Every listed path or ref -> its name, to name models the load evicted.
    known: dict[str, str] = field(default_factory = dict, repr = False)

    def renamed(self, result: LoadResult) -> LoadResult:
        evicted = [self.known.get(entry, entry) for entry in result.evicted]
        slots = [
            slot.model_copy(update = {"model": self.known.get(slot.model, slot.model)})
            for slot in result.evicted_slots
        ]
        return result.model_copy(
            update = {"model": self.name, "evicted": evicted, "evicted_slots": slots}
        )


def rename_job(caller: Caller, job_id: Optional[str], trained: TrainedModel) -> None:
    """Name a load that outlasted wait_s in its job result too. The callback is added before any get_job waiter, so it runs first."""
    from studio_mcp import loading

    job = loading.LOADS.jobs.get(f"{caller.account_id}:{job_id}") if job_id else None
    if job is None:
        return

    def rename(_task: object = None) -> None:
        if isinstance(job.result, LoadResult):
            job.result = trained.renamed(job.result)

    if job.task is None or job.task.done():
        rename()
    else:
        job.task.add_done_callback(rename)


def _known(rows: list[tuple[str, str]]) -> dict[str, str]:
    known = {}
    for path, name in rows:
        known[path] = name
        known[cache_reference(path)] = name
    return known


def _parent(path: str) -> str:
    trimmed = path.rstrip("\\/")
    return trimmed[: len(trimmed) - len(leaf_name(trimmed))].rstrip("\\/") or trimmed


async def _export_rows(caller: Caller) -> list[tuple[str, str, Optional[str]]]:
    listing = await route_json("GET", "/api/models/loras", caller = caller, hub_header = True)
    rows = []
    for row in as_dict(listing).get("loras") or []:
        row = as_dict(row)
        label, path = opt_text(row.get("display_name")), opt_text(row.get("adapter_path"))
        if row.get("source") == "exported" and label and path:
            # The scan names a two-level export "run / name"; export_model reports "run/name".
            rows.append((label.replace(" / ", "/"), path, row.get("export_type")))
    return rows


def _export_known(rows: list[tuple[str, str, Optional[str]]]) -> dict[str, str]:
    # A GGUF export loads as its folder, so a later eviction names the folder.
    return _known(
        [(r[1], r[0]) for r in rows] + [(_parent(r[1]), r[0]) for r in rows if r[2] == "gguf"]
    )


async def _other_known(caller: Caller, of: str) -> dict[str, str]:
    """Names for the other kind of trained model, so a load that evicts one names it too."""
    try:
        if of == "checkpoint":
            return _export_known(await _export_rows(caller))
        found, _folders = await checkpoints.list_checkpoints(caller)
        return _known([(c.path, c.name) for c in found])
    except Exception:
        return {}


async def resolve_checkpoint(caller: Caller, name: str) -> TrainedModel:
    found, _folders = await checkpoints.list_checkpoints(caller)
    match = next((c for c in found if c.name == name), None)
    if match is None:
        available = ", ".join(sorted(c.name for c in found)[:20]) or "none"
        raise tool_error(f"No checkpoint named {name}. Available: {available}")
    # The load route detects LoRA on disk; a full finetune has no peft_type.
    return TrainedModel(
        name = name,
        ref = cache_reference(match.path),
        is_lora = match.peft_type is not None,
        known = {
            **await _other_known(caller, "checkpoint"),
            **_known([(c.path, c.name) for c in found]),
        },
    )


async def resolve_export(caller: Caller, name: str, variant: Optional[str]) -> TrainedModel:
    rows = await _export_rows(caller)
    match = [r for r in rows if r[0] == name] or [
        r for r in rows if r[0].casefold() == name.casefold()
    ]
    if len(match) != 1:
        available = ", ".join(sorted({r[0] for r in rows})[:20]) or "none"
        # A hint, since the scrub would read "<run>/<name>" as a path; the names are scrubbed here.
        raise tool_error(
            f"No export named {name}.",
            hint = "A merged or LoRA export is found when its save_directory has two levels, "
            f'such as "<run>/<name>". Available: {redact_paths_in_text(available)}',
        )
    _label, path, export_type = match[0]
    gguf_variant = None
    if export_type == "gguf":
        from hub.utils.gguf import extract_quant_token

        # The scan picks one .gguf file; the folder plus a variant picks the quantization, and a
        # variant is what makes the load (and device="cpu") treat it as GGUF.
        gguf_variant = variant or extract_quant_token(leaf_name(path))
        if gguf_variant:
            path = _parent(path)
    return TrainedModel(
        name = name,
        ref = cache_reference(path),
        is_lora = export_type == "lora",
        gguf_variant = gguf_variant,
        unplaced_gguf = export_type == "gguf" and not gguf_variant,
        known = {**await _other_known(caller, "export"), **_export_known(rows)},
    )


async def resolve(
    caller: Caller,
    *,
    model: Optional[str],
    checkpoint: Optional[str],
    export: Optional[str],
    kind: str,
    variant: Optional[str],
) -> Optional[TrainedModel]:
    """None for a plain ``model`` load; refusals come before any route call."""
    if sum(value is not None for value in (model, checkpoint, export)) != 1:
        raise tool_error("load_model needs exactly one of model, checkpoint or export.")
    if model is not None:
        return None
    if kind in NOT_LLM_KINDS:
        raise tool_error(
            'checkpoint and export load chat and text-to-speech models; use kind "llm", "tts" or "audio".'
        )
    if checkpoint is not None:
        return await resolve_checkpoint(caller, checkpoint)
    return await resolve_export(caller, export, variant)


async def resolve_by(
    deadline: float, caller: Caller, **kwargs: Optional[str]
) -> Optional[TrainedModel]:
    """resolve, bounded: the listing reads run before the load job, so a stalled one must not hold the call."""
    timeout = max(deadline - time.monotonic(), RESOLVE_TIMEOUT_S)
    try:
        return await asyncio.wait_for(resolve(caller, **kwargs), timeout)
    except asyncio.TimeoutError:
        raise tool_error(
            "Unsloth Studio did not list its training runs and exports in time; call load_model again."
        ) from None
