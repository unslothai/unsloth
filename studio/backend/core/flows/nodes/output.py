# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Output nodes: previews, named flow results and saving media into a Library folder."""

from __future__ import annotations

from typing import Any

from core.flows.nodes._common import media_bytes
from core.flows.registry import FlowNodeError, NodeContext, handler


def _single(values: Any) -> Any:
    if isinstance(values, list) and len(values) == 1:
        return values[0]
    return values


@handler("output.preview")
async def preview(ctx: NodeContext) -> dict:
    return {"value": _single(ctx.inputs.get("value"))}


@handler("output.result")
async def result(ctx: NodeContext) -> dict:
    return {"value": ctx.inputs.get("value")}


def _media_items(values: Any) -> list[dict]:
    found: list[dict] = []
    stack = list(values) if isinstance(values, list) else [values]
    while stack:
        item = stack.pop(0)
        if isinstance(item, list):
            stack.extend(item)
        elif (
            isinstance(item, dict)
            and item.get("kind") in ("image", "audio", "video")
            and item.get("ref")
        ):
            found.append(item)
    return found


async def _folder_id(ctx: NodeContext, name: str) -> str:
    listing = await ctx.call("GET", "/api/library")
    for folder in (listing.get("folders") if isinstance(listing, dict) else None) or []:
        if isinstance(folder, dict) and folder.get("name") == name and not folder.get("parentId"):
            return str(folder["id"])
    created = await ctx.call("POST", "/api/library/folders", json = {"name": name})
    if not isinstance(created, dict) or not created.get("id"):
        raise FlowNodeError("Could not create the Library folder")
    return str(created["id"])


@handler("output.library")
async def library(ctx: NodeContext) -> dict:
    items = _media_items(ctx.inputs.get("value"))
    if not items:
        raise FlowNodeError("Nothing to save: connect images, audio or video")
    folder = await _folder_id(ctx, str(ctx.params.get("folder") or "Flows").strip() or "Flows")
    saved: list[str] = []
    for item in items:
        kind, _, media_id = str(item["ref"]).partition(":")
        if kind in ("image", "audio", "video") and media_id:
            await ctx.call(
                "PATCH", "/api/library/items", json = {"id": f"{kind}:{media_id}", "folderId": folder}
            )
            saved.append(f"{kind}:{media_id}")
            continue
        data, mime = await media_bytes(ctx, item)
        ext = mime.split("/")[-1].replace("mpeg", "mp3").replace("jpeg", "jpg")
        answer = await ctx.call(
            "POST",
            "/api/library/uploads",
            files = {"files": (f"flow-{ctx.node_id}-{len(saved) + 1}.{ext}", data, mime)},
            data = {"folderId": folder},
        )
        saved.extend(answer.get("ids") or [] if isinstance(answer, dict) else [])
    return {"saved": saved}
