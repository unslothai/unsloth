# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Dataset nodes: Hub datasets, saved Data Recipes (declarative part only) and JSON rows."""

from __future__ import annotations

import asyncio
import json
import re
from typing import Any
from urllib.parse import quote

from core.flows.registry import FlowNodeError, NodeContext, handler

RECIPES = "/api/data-recipe"
POLL_S = 2.0
_REPO = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*(/[A-Za-z0-9][A-Za-z0-9._-]*)?$")
# Anything that can start a process, call a tool server or carry a credential is not part of a flow's recipe.
_REFUSED_KEYS = ("mcp_providers", "tool_configs", "tool_profiles", "tools", "mcp")


@handler("data.hf_dataset")
async def hf_dataset(ctx: NodeContext) -> dict:
    repo = str(ctx.params.get("repo") or "").strip()
    if not _REPO.match(repo):
        raise FlowNodeError("Name a Hugging Face dataset like owner/name")
    ref = repo
    split = str(ctx.params.get("split") or "train").strip()
    subset = str(ctx.params.get("subset") or "").strip()
    value: dict[str, Any] = {"kind": "dataset", "source": "hf", "ref": ref}
    if split:
        value["split"] = split
    if subset:
        value["subset"] = subset
    return {"dataset": value}


def declarative_recipe(payload: Any) -> dict:
    """The recipe without tool servers, tool configs or provider secrets; refuses one that still names a stdio server."""
    recipe = (
        payload.get("recipe")
        if isinstance(payload, dict) and isinstance(payload.get("recipe"), dict)
        else payload
    )
    if not isinstance(recipe, dict):
        raise FlowNodeError("That Data Recipe has no recipe")
    recipe = json.loads(json.dumps(recipe))
    for key in _REFUSED_KEYS:
        recipe.pop(key, None)
    for provider in recipe.get("model_providers") or []:
        if isinstance(provider, dict):
            for secret in ("api_key", "api_key_env", "extra_headers"):
                provider.pop(secret, None)
    for column in recipe.get("columns") or []:
        if isinstance(column, dict):
            for key in ("tool_alias", "tool_config", "mcp_tools", "tools"):
                column.pop(key, None)
    stack: list[Any] = [recipe]
    while stack:
        node = stack.pop()
        if isinstance(node, dict):
            if node.get("provider_type") == "stdio" or "command" in node and "args" in node:
                raise FlowNodeError(
                    "This recipe starts a local tool server; run it from Data Recipes instead"
                )
            stack.extend(node.values())
        elif isinstance(node, list):
            stack.extend(node)
    return recipe


@handler("data.recipe")
async def recipe(ctx: NodeContext) -> dict:
    recipe_id = str(ctx.params.get("recipe_id") or "").strip()
    if not recipe_id:
        raise FlowNodeError("Pick a saved Data Recipe")
    record = await ctx.call("GET", f"{RECIPES}/recipes/{quote(recipe_id, safe = '')}")
    body = declarative_recipe(record.get("payload") if isinstance(record, dict) else None)
    mode = "preview" if ctx.params.get("mode") == "preview" else "full"
    run: dict[str, Any] = {"execution_type": mode}
    rows = int(ctx.params.get("rows") or 0)
    if rows > 0:
        run["rows"] = rows
    started = await ctx.call("POST", f"{RECIPES}/jobs", json = {"recipe": body, "run": run})
    job_id = started.get("job_id") if isinstance(started, dict) else None
    if not job_id:
        raise FlowNodeError("Unsloth Studio did not start the recipe")

    async def cancel() -> None:
        await ctx.call("POST", f"{RECIPES}/jobs/{quote(job_id, safe = '')}/cancel")

    ctx.on_cancel(cancel)
    while True:
        await asyncio.sleep(POLL_S)
        status = await ctx.call("GET", f"{RECIPES}/jobs/{quote(job_id, safe = '')}/status")
        state = str(status.get("status") or "") if isinstance(status, dict) else ""
        progress = status.get("progress") if isinstance(status, dict) else None
        if isinstance(progress, dict) and isinstance(progress.get("percent"), (int, float)):
            ctx.progress(progress["percent"] / 100.0, phase = state)
        if state in ("completed", "done", "succeeded"):
            break
        if state in ("error", "failed"):
            raise FlowNodeError(str(status.get("error") or "The recipe failed"))
        if state in ("cancelled", "canceled"):
            ctx.check_cancelled()
            raise FlowNodeError("The recipe was cancelled")
    preview: list = []
    try:
        page = await ctx.call(
            "GET", f"{RECIPES}/jobs/{quote(job_id, safe = '')}/dataset", params = {"limit": 20}
        )
        if isinstance(page, dict):
            preview = [row for row in page.get("dataset") or [] if isinstance(row, dict)]
    except FlowNodeError:
        pass
    return {
        "dataset": {
            "kind": "dataset",
            "source": "recipe",
            "ref": job_id,
            "rows": status.get("rows") if isinstance(status, dict) else None,
        },
        "preview": preview,
    }


@handler("data.rows_to_dataset")
async def rows_to_dataset(ctx: NodeContext) -> dict:
    rows = ctx.inputs.get("rows")
    if isinstance(rows, dict):
        rows = rows.get("rows") or rows.get("data") or [rows]
    if not isinstance(rows, list) or not rows or not all(isinstance(r, dict) for r in rows):
        raise FlowNodeError("Rows to Dataset takes a JSON list of objects")
    name = (
        re.sub(r"[^A-Za-z0-9._-]+", "-", str(ctx.params.get("name") or "flow-dataset")).strip("-")
        or "flow-dataset"
    )
    data = "\n".join(json.dumps(r, ensure_ascii = False) for r in rows).encode()
    saved = await ctx.call(
        "POST",
        "/api/hub/datasets/upload",
        files = {"file": (f"{name}.jsonl", data, "application/octet-stream")},
    )
    stored = saved.get("stored_path") if isinstance(saved, dict) else None
    if not isinstance(stored, str) or not stored:
        raise FlowNodeError("Unsloth Studio did not say where it saved the dataset")
    from hub.utils.host_paths import cache_reference

    columns = sorted({k for r in rows for k in r})
    return {
        "dataset": {
            "kind": "dataset",
            "source": "local",
            "ref": cache_reference(stored),
            "rows": len(rows),
            "columns": columns,
        }
    }
