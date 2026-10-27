# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Dataset nodes: Hub datasets, Data Recipes (saved or a Recipe group, declarative only), rows."""

from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path
from typing import Any, Optional
from urllib.parse import quote

from core.flows.nodes._common import poll_job
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
    from core.training.account_jobs import managed_account

    secrets = (
        ("api_key", "extra_headers", "api_key_env")
        if managed_account()
        else ("api_key", "extra_headers")
    )
    for provider in recipe.get("model_providers") or []:
        if isinstance(provider, dict):
            for secret in secrets:
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
    return await run_recipe(ctx, body, {})


@handler("recipe.generate")
async def recipe_group(ctx: NodeContext) -> dict:
    from core.flows.compiler import RECIPE_PARAM

    compiled = ctx.params.get(RECIPE_PARAM)
    if not isinstance(compiled, dict):
        raise FlowNodeError("Put this step inside a Data Recipe group")
    body = declarative_recipe(compiled.get("recipe"))
    checked = await ctx.call("POST", f"{RECIPES}/validate", json = {"recipe": body})
    if isinstance(checked, dict) and checked.get("valid") is False:
        problems = [
            str(e.get("message") or "")
            for e in checked.get("errors") or []
            if isinstance(e, dict) and e.get("message")
        ]
        raise FlowNodeError(
            "The recipe is not ready: " + ("; ".join(problems[:3]) or "it did not validate")
        )
    saved = compiled.get("run") if isinstance(compiled.get("run"), dict) else {}
    return await run_recipe(
        ctx, body, {k: saved[k] for k in ("run_config", "merge_batches") if k in saved}
    )


async def run_recipe(ctx: NodeContext, body: dict, run: dict[str, Any]) -> dict:
    """One Data Designer job for ``body``: its dataset (parquet if saved, else rows), a preview."""
    seed = ctx.inputs.get("seed")
    if isinstance(seed, dict) and seed.get("kind") == "dataset":
        body["seed_config"] = await seed_config(ctx, seed, body.get("seed_config"))
    run = {**run, "execution_type": "preview" if ctx.params.get("mode") == "preview" else "full"}
    rows = int(ctx.params.get("rows") or 0)
    if rows > 0:
        run["rows"] = rows
    started = await ctx.call("POST", f"{RECIPES}/jobs", json = {"recipe": body, "run": run})
    job_id = started.get("job_id") if isinstance(started, dict) else None
    if not job_id:
        raise FlowNodeError("Unsloth Studio did not start the recipe")

    async def cancel() -> None:
        await ctx.call("POST", f"{RECIPES}/jobs/{quote(job_id, safe = '')}/cancel")

    def parse(status: Any) -> tuple[str, Optional[dict], Optional[str]]:
        state = str(status.get("status") or "") if isinstance(status, dict) else ""
        progress = status.get("progress") if isinstance(status, dict) else None
        shown = None
        if isinstance(progress, dict) and isinstance(progress.get("percent"), (int, float)):
            shown = {"fraction": progress["percent"] / 100.0, "phase": state}
        if state in ("completed", "done", "succeeded"):
            return "completed", shown, None
        if state in ("error", "failed"):
            return "failed", shown, str(status.get("error") or "The recipe failed")
        if state in ("cancelled", "canceled"):
            return "cancelled", shown, "The recipe was cancelled"
        return "running", shown, None

    status = await poll_job(
        ctx,
        status = lambda: ctx.call("GET", f"{RECIPES}/jobs/{quote(job_id, safe = '')}/status"),
        parse = parse,
        cancel = cancel,
        interval = POLL_S,
    )
    preview: list = []
    try:
        page = await ctx.call(
            "GET", f"{RECIPES}/jobs/{quote(job_id, safe = '')}/dataset", params = {"limit": 20}
        )
        if isinstance(page, dict):
            preview = [row for row in page.get("dataset") or [] if isinstance(row, dict)]
    except FlowNodeError:
        pass
    rows = status.get("rows") if isinstance(status, dict) else None
    artifact = status.get("artifact_path") if isinstance(status, dict) else None
    if isinstance(artifact, str) and artifact.strip():
        ref = f"recipes/{Path(artifact.strip()).name}/parquet-files"
        dataset = {
            "kind": "dataset",
            "source": "recipe",
            "ref": ref,
            "job_id": job_id,
            "rows": rows,
        }
    else:
        kept = await recipe_rows(ctx, job_id)
        if not kept:
            raise FlowNodeError("The Data Recipe produced no rows")
        dataset = await upload_rows(ctx, kept, f"recipe-{job_id}")
    return {"dataset": dataset, "preview": preview}


def recipe_artifact(ref: Any) -> bool:
    """True for a recipe dataset saved as parquet (`recipes/<run>/parquet-files`), not an older job id."""
    return isinstance(ref, str) and ref.startswith("recipes/") and ".." not in ref.split("/")


async def seed_config(ctx: NodeContext, dataset: dict, current: Any) -> dict:
    """A seed for an upstream dataset, keeping the recipe's own sampling and selection."""
    base = current if isinstance(current, dict) else {}
    keep = {k: base[k] for k in ("sampling_strategy", "selection_strategy") if k in base}
    keep.setdefault("sampling_strategy", "ordered")
    source = dataset.get("source")
    ref = dataset.get("ref")
    if source == "hf":
        from core.training.account_jobs import managed_account

        if managed_account():
            raise FlowNodeError(
                "A Hub seed needs your own Hugging Face token; set the seed in the recipe on the Data Recipes page"
            )
        if "/" not in str(ref):
            raise FlowNodeError(f"A recipe seed needs the full Hub name, like owner/{ref}")
        inspected = await ctx.call(
            "POST",
            f"{RECIPES}/seed/inspect",
            json = {
                "dataset_name": str(ref),
                "split": dataset.get("split") or "train",
                "subset": dataset.get("subset"),
                "preview_size": 1,
            },
        )
        path = inspected.get("resolved_path") if isinstance(inspected, dict) else None
        if not path:
            raise FlowNodeError(f"Unsloth Studio could not read the seed dataset {ref}")
        return {**keep, "source": {"seed_type": "hf", "path": path}}
    if source in ("local", "recipe"):
        return {**keep, "source": {"seed_type": "local", "path": local_seed_path(ref)}}
    raise FlowNodeError("A recipe seed takes a Hub dataset, a Data Recipe or Rows to Dataset")


def local_seed_path(ref: Any) -> str:
    from hub.utils.host_paths import resolve_host_path_reference
    from utils.paths import resolve_dataset_path

    text = str(ref or "")
    found = resolve_host_path_reference(text) if text.startswith("ref:") else None
    if found is None and not text.startswith("ref:"):
        try:
            found = str(resolve_dataset_path(text))
        except ValueError as exc:
            raise FlowNodeError(f"That seed dataset is not one Studio can read: {exc}") from exc
    if not found or not Path(found).exists():
        raise FlowNodeError(
            "The seed dataset is no longer available; run the step that made it again"
        )
    return f"{found}/*.parquet" if Path(found).is_dir() else found


@handler("data.rows_to_dataset")
async def rows_to_dataset(ctx: NodeContext) -> dict:
    rows = ctx.inputs.get("rows")
    if isinstance(rows, dict) and rows.get("kind") == "table":
        from core.flows.nodes.table import read_table
        rows = (await read_table(ctx, rows))["rows"]
    elif isinstance(rows, dict):
        rows = rows.get("rows") or rows.get("data") or [rows]
    if not isinstance(rows, list) or not rows or not all(isinstance(r, dict) for r in rows):
        raise FlowNodeError("Rows to Dataset takes a JSON list of objects")
    return {"dataset": await upload_rows(ctx, rows, str(ctx.params.get("name") or "flow-dataset"))}


async def upload_rows(ctx: NodeContext, rows: list[dict], name: str) -> dict:
    """Save rows as a JSONL training dataset and return its DatasetValue."""
    name = re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip("-") or "flow-dataset"
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
        "kind": "dataset",
        "source": "local",
        "ref": cache_reference(stored),
        "rows": len(rows),
        "columns": columns,
    }


async def recipe_rows(
    ctx: NodeContext,
    job_id: str,
    limit: int = 100_000,
) -> list[dict]:
    """Every row a finished Data Recipe job produced."""
    rows: list[dict] = []
    total = None
    while len(rows) < limit:
        page = await ctx.call(
            "GET",
            f"{RECIPES}/jobs/{quote(job_id, safe = '')}/dataset",
            params = {"limit": 500, "offset": len(rows)},
        )
        batch = (
            [r for r in (page.get("dataset") or []) if isinstance(r, dict)]
            if isinstance(page, dict)
            else []
        )
        rows.extend(batch)
        total = page.get("total") if isinstance(page, dict) else None
        if not batch or (isinstance(total, int) and len(rows) >= total):
            break
    if isinstance(total, int) and total > len(rows):
        ctx.log(f"The Data Recipe has {total} rows; Flows use the first {len(rows)}", "warning")
    return rows
