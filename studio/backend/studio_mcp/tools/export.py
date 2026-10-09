# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""``export_model``: a trained checkpoint, by name, to GGUF, merged weights, a LoRA adapter or the base model."""

from __future__ import annotations

import os
import re
from typing import Any, Literal, Optional, Union

from fastmcp import FastMCP
from fastmcp.exceptions import ToolError

from studio_mcp import checkpoints, export_jobs
from studio_mcp.caller import Caller, current_caller
from studio_mcp.errors import tool_error
from studio_mcp.outputs import ExportJobRef
from studio_mcp.tools import WRITES, route_json

EXPORT_STATUS = "/api/export/status"
RUNNING = (
    "Another export is running. Unsloth Studio exports one at a time, so try again when it ends."
)


def checked_save_directory(value: str) -> str:
    """A folder under Unsloth Studio's exports folder; the route alone would also take any path on the host."""
    text = value.strip()
    if not text:
        raise ToolError("save_directory must name a folder.")
    if text.startswith(("/", "\\", "~")) or re.match(r"^[A-Za-z]:", text):
        raise ToolError(
            "save_directory must be relative to Unsloth Studio's exports folder, not an absolute path."
        )
    if ".." in re.split(r"[\\/]", text):
        raise ToolError("save_directory must not contain '..'.")
    return text


async def _op(caller: Caller, path: str, body: dict[str, Any]) -> Any:
    result = await route_json("POST", path, caller = caller, json_body = body)
    if isinstance(result, dict) and result.get("success") is False:
        raise tool_error(result.get("message") or "The export step failed")
    return result


def _export_body(
    format: str,
    *,
    save_directory: str,
    quantization_method: Union[str, list[str]],
    push_to_hub: bool,
    repo_id: Optional[str],
    hf_token: Optional[str],
    private: bool,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "save_directory": save_directory,
        "push_to_hub": push_to_hub,
        "private": private,
    }
    if repo_id:
        body["repo_id"] = repo_id
    # In the body, as the export routes read it; an API key never falls back to the server's token.
    if hf_token:
        body["hf_token"] = hf_token
    if format == "gguf":
        body["quantization_method"] = quantization_method
    return body


def _same_checkpoint(status: Any, path: str) -> bool:
    loaded = status.get("current_checkpoint") if isinstance(status, dict) else None
    if not isinstance(loaded, str) or not loaded:
        return False
    return os.path.realpath(loaded) == os.path.realpath(path)


async def export_model(
    checkpoint: str,
    format: Literal["gguf", "merged", "lora", "base"],
    save_directory: str,
    quantization_method: Union[str, list[str]] = "Q4_K_M",
    push_to_hub: bool = False,
    repo_id: Optional[str] = None,
    hf_token: Optional[str] = None,
    private: bool = False,
    max_seq_length: int = 2048,
    load_in_4bit: Optional[bool] = None,
) -> ExportJobRef:
    """Export a trained checkpoint and return a job id at once; follow it with get_job(kind="export"). ``checkpoint`` is a name from list_training_runs(include_checkpoints=true): "<run folder>" or "<run folder>/checkpoint-N". ``format``: "gguf" (``quantization_method`` one or a list, e.g. Q4_K_M, Q8_0), "merged" (LoRA merged into 16-bit weights), "lora" (the adapter only) or "base". ``save_directory`` is a folder name under Unsloth Studio's exports folder. ``push_to_hub`` uploads to ``repo_id`` and needs ``hf_token``. Leave ``load_in_4bit`` unset to let Unsloth Studio choose. Unsloth Studio exports one at a time, so this is refused while another export runs."""
    caller = current_caller()
    save_directory = checked_save_directory(save_directory)
    # One export worker: a job queued behind another could not be told apart from it when cancelled.
    running = await route_json("GET", EXPORT_STATUS, caller = caller)
    if (isinstance(running, dict) and running.get("is_export_active")) or export_jobs.any_running():
        raise ToolError(RUNNING)
    found = await checkpoints.resolve(caller, checkpoint)
    token = hf_token or caller.hf_token
    load: dict[str, Any] = {"checkpoint_path": found.path, "max_seq_length": max_seq_length}
    if load_in_4bit is not None:
        load["load_in_4bit"] = load_in_4bit
    if token:
        load["hf_token"] = token
    body = _export_body(
        format,
        save_directory = save_directory,
        quantization_method = quantization_method,
        push_to_hub = push_to_hub,
        repo_id = repo_id,
        hf_token = token,
        private = private,
    )

    async def run(job: export_jobs.ExportJob) -> None:
        job.phase = "loading"
        await _op(caller, "/api/export/load-checkpoint", load)
        before = await route_json("GET", EXPORT_STATUS, caller = caller)
        # The load and the export are two calls, so a load from the Export page between them would
        # be exported in this job's name.
        if not _same_checkpoint(before, found.path):
            raise ToolError(
                "Another checkpoint was loaded for export before this job's export began; nothing was exported."
            )
        job.phase = "exporting"
        # The route checks this under the export lock, which closes the gap the status check leaves.
        result = await _op(
            caller, f"/api/export/export/{format}", {**body, "expected_checkpoint": found.path}
        )
        # Its export is over; only recording the result is left, so a cancel now waits for that.
        job.phase = "finishing"
        # This call's own answer, not the export status: another export may have finished since.
        details = result.get("details") if isinstance(result, dict) else None
        if isinstance(details, dict):
            job.output = export_jobs.output_name(details.get("output_path")) or job.output

    # Checked again with no await before start, so two calls at once cannot both get past the guard.
    if export_jobs.any_running():
        raise ToolError(RUNNING)
    job = export_jobs.start(caller.account_id, format, run)
    return ExportJobRef(job_id = job.job_id)


def register_export(mcp: FastMCP) -> None:
    mcp.tool(export_model, annotations = WRITES)
