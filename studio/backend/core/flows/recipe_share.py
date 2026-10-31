# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""What a Data Recipe keeps when it leaves this Studio (flow export, recipe JSON export, package import).

A recipe group in a saved flow keeps everything Recipe Studio keeps, tool servers and provider keys
included. A shared copy never carries credentials, a command this host would run (stdio tool server)
or a path into this machine's uploads.
"""

from __future__ import annotations

import copy
from typing import Any

_SECRETS = ("api_key", "api_key_env", "extra_headers", "headers", "token")
_LOCAL_SEEDS = ("local", "unstructured")
_SEED_HINTS = {
    "seed_columns": [],
    "seed_drop_columns": [],
    "seed_preview_rows": [],
    "local_file_name": "",
    "unstructured_upload_uid": "",
    "unstructured_file_ids": [],
    "unstructured_file_names": [],
    "unstructured_file_sizes": [],
}


def is_stdio(provider: Any) -> bool:
    return isinstance(provider, dict) and provider.get("provider_type") == "stdio"


def stdio_names(recipe: Any) -> list[str]:
    providers = recipe.get("mcp_providers") if isinstance(recipe, dict) else None
    return [str(p.get("name")) for p in providers or [] if is_stdio(p)]


def uses_mcp(recipe: Any) -> bool:
    """True when the recipe would call MCP servers (a tool profile names one), which Flows gates like any MCP use."""
    providers = recipe.get("mcp_providers") if isinstance(recipe, dict) else None
    return any(isinstance(p, dict) and p.get("name") for p in providers or [])


def missing_seed(recipe: Any) -> str | None:
    """Why the recipe cannot run yet when its file seed came without the file (a shared recipe), else None."""
    seed = recipe.get("seed_config") if isinstance(recipe, dict) else None
    source = seed.get("source") if isinstance(seed, dict) else None
    if not isinstance(source, dict) or source.get("seed_type") not in _LOCAL_SEEDS:
        return None
    if source.get("path") or any(source.get("paths") or []):
        return None
    what = "document" if source.get("seed_type") == "unstructured" else "CSV or JSON file"
    return f"Load the seed first: this recipe has no {what} for its seed yet, so open the seed and pick one"


def share_recipe(recipe: dict, ui: dict | None = None) -> tuple[dict, dict]:
    """A copy of ``recipe`` (no columns needed) and its Recipe Studio ``ui`` hints that is safe to hand out."""
    recipe = copy.deepcopy(recipe) if isinstance(recipe, dict) else {}
    ui = copy.deepcopy(ui) if isinstance(ui, dict) else {}
    dropped = set(stdio_names(recipe))
    providers = [
        p for p in recipe.get("mcp_providers") or [] if isinstance(p, dict) and not is_stdio(p)
    ]
    for provider in providers:
        for key in (*_SECRETS, "env"):
            provider.pop(key, None)
    if "mcp_providers" in recipe:
        recipe["mcp_providers"] = providers
    for provider in recipe.get("model_providers") or []:
        if isinstance(provider, dict):
            for key in _SECRETS:
                provider.pop(key, None)
    kept_tools: list[dict] = []
    for tool in recipe.get("tool_configs") or []:
        if not isinstance(tool, dict):
            continue
        names = [p for p in tool.get("providers") or [] if p not in dropped]
        if names:
            kept_tools.append({**tool, "providers": names})
    if "tool_configs" in recipe:
        recipe["tool_configs"] = kept_tools
    aliases = {t.get("tool_alias") for t in kept_tools}
    for column in recipe.get("columns") or []:
        if (
            isinstance(column, dict)
            and column.get("tool_alias")
            and column["tool_alias"] not in aliases
        ):
            column.pop("tool_alias")
    if isinstance(ui.get("nodes"), list):
        for node in ui["nodes"]:
            tools = node.get("tools_by_provider") if isinstance(node, dict) else None
            if isinstance(tools, dict):
                node["tools_by_provider"] = {k: v for k, v in tools.items() if k not in dropped}
    seed = recipe.get("seed_config")
    source = seed.get("source") if isinstance(seed, dict) else None
    if isinstance(source, dict):
        source.pop("token", None)
        kind = source.get("seed_type")
        # An HF path stays whole: `datasets/<owner>/<name>/<config>/**/*.parquet` is what the seed reader loads.
        if kind in _LOCAL_SEEDS:
            if "path" in source:
                source["path"] = ""
            if "paths" in source:
                source["paths"] = []
            seed.pop("resolved_paths", None)
        # Preview rows hold the seed's contents (a private repo's issues, an uploaded file).
        if kind in ("hf", "github_repo", *_LOCAL_SEEDS):
            ui.update(copy.deepcopy(_SEED_HINTS))
            ui.pop("seed_total_rows", None)
    return recipe, ui


def share_group(group: dict) -> dict:
    """A Recipe group as it may appear in an exported or imported flow."""
    if not isinstance(group, dict) or group.get("kind") != "recipe":
        return group
    recipe, ui = share_recipe(group.get("config") or {}, group.get("ui"))
    shared = {**group, "config": recipe}
    if group.get("ui") is not None:
        shared["ui"] = ui
    return shared


def share_flow(doc: dict) -> dict:
    """``doc`` with every Recipe group shared, and its columns off any tool profile the share dropped."""
    if not isinstance(doc, dict) or not isinstance(doc.get("groups"), list):
        return doc
    doc["groups"] = [share_group(g) for g in doc["groups"]]
    aliases = {
        g["id"]: {t.get("tool_alias") for t in (g.get("config") or {}).get("tool_configs") or []}
        for g in doc["groups"]
        if isinstance(g, dict) and g.get("kind") == "recipe"
    }
    for node in doc.get("nodes") or []:
        column = (node.get("params") or {}).get("column") if isinstance(node, dict) else None
        if (
            node.get("group") in aliases
            and isinstance(column, dict)
            and column.get("tool_alias")
            and column["tool_alias"] not in aliases[node["group"]]
        ):
            column.pop("tool_alias")
    return doc
