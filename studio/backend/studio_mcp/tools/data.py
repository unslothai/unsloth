# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""``datasets``: the training datasets Unsloth Studio has, and getting more."""

from __future__ import annotations

import dataclasses
from typing import Literal, Optional

from fastmcp.exceptions import ToolError

from studio_mcp.caller import current_caller
from studio_mcp.outputs import (
    CachedDataset,
    DatasetDownload,
    DatasetFormat,
    DatasetsResult,
    LocalDataset,
)
from studio_mcp.tools import WRITES, as_dict, opt_text, route_json

HUB_DATASETS = "/api/hub/datasets"


async def datasets(
    action: Literal["list", "check_format", "download", "status"],
    name: Optional[str] = None,
    repo_id: Optional[str] = None,
    is_vlm: bool = False,
    subset: Optional[str] = None,
    train_split: str = "train",
    hf_token: Optional[str] = None,
) -> DatasetsResult:
    """Training datasets. "list": datasets uploaded to Unsloth Studio and Hugging Face datasets already downloaded. "check_format" (``name``: a Hub repo id or a listed dataset): the detected format, columns and a suggested column mapping. "download" (``repo_id``): start downloading a Hub dataset; "status" (``repo_id``): how that download is going. ``hf_token`` is for gated datasets."""
    caller = current_caller()
    if hf_token:
        caller = dataclasses.replace(caller, hf_token = hf_token)
    if action == "list":
        local = await route_json("GET", f"{HUB_DATASETS}/local", caller = caller)
        cached = await route_json("GET", f"{HUB_DATASETS}/cached", caller = caller)
        return DatasetsResult(
            local = [
                LocalDataset.from_route(
                    row, label = opt_text(row.get("label")) or row["id"], source = "local"
                )
                for row in as_dict(local).get("datasets") or []
                if isinstance(row, dict) and opt_text(row.get("id"))
            ],
            cached = [
                CachedDataset.from_route(row)
                for row in as_dict(cached).get("cached") or []
                if isinstance(row, dict) and opt_text(row.get("repo_id"))
            ],
        )
    if action == "check_format":
        if not name:
            raise ToolError(
                "check_format needs name: a Hugging Face dataset id or a listed dataset."
            )
        body = {"dataset_name": name, "is_vlm": is_vlm, "train_split": train_split}
        if subset:
            body["subset"] = subset
        found = await route_json(
            "POST", f"{HUB_DATASETS}/check-format", caller = caller, json_body = body, hub_header = True
        )
        found = as_dict(found)
        mapping = found.get("suggested_mapping")
        return DatasetsResult(
            format = DatasetFormat.from_route(
                found,
                suggested_mapping = {str(k): v for k, v in mapping.items() if isinstance(v, str)}
                if isinstance(mapping, dict)
                else None,
            )
        )
    if not repo_id:
        raise ToolError(f"{action} needs repo_id: a Hugging Face dataset id.")
    if action == "download":
        started = await route_json(
            "POST",
            f"{HUB_DATASETS}/download",
            caller = caller,
            json_body = {"repo_id": repo_id},
            hub_header = True,
        )
        return DatasetsResult(
            download = DatasetDownload(
                repo_id = repo_id, state = opt_text(as_dict(started).get("state")) or "queued"
            )
        )
    state = await route_json(
        "GET", f"{HUB_DATASETS}/download-status", caller = caller, params = {"repo_id": repo_id}
    )
    state = as_dict(state)
    return DatasetsResult(
        download = DatasetDownload.from_route(
            state, repo_id = repo_id, state = opt_text(state.get("state")) or "unknown"
        )
    )


TOOLS = ((datasets, WRITES),)
