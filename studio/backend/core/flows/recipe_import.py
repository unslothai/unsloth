# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Data Recipe -> a Recipe group of column nodes, the flow-side model of a recipe.

A recipe stays one batched Data Designer job. Each column becomes a `recipe.column` node inside a
group of kind "recipe"; models, providers, tools, processors and the seed stay recipe-level config
on the group. Every field is carried verbatim, so `compile_recipe(import_recipe(R))` gives back R.
Pure data: nothing is fetched, run or validated against the engine here.
"""

from __future__ import annotations

import copy
import re
from typing import Any, Optional

import jinja2
from jinja2 import meta, nodes

RECIPE_GRAPH_VERSION = 1
COLUMN_TYPE = "recipe.column"
GENERATE_TYPE = "recipe.generate"
_JINJA_BLOCK = re.compile(r"\{\{(.*?)\}\}|\{%(.*?)%\}", re.S)
_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_-]*")
_GRID = (320, 180)
_HEX = re.compile(r"^#[0-9a-fA-F]{6}$")
# Canvas layout from Recipe Studio; a Flows group lays its columns out again.
_LAYOUT_UI = ("nodes", "edges", "layout_direction", "aux_nodes", "advanced_open_by_node")
_ENV = jinja2.Environment()


def _template_names(text: str) -> list[str]:
    try:
        tree = _ENV.parse(text)
    except jinja2.TemplateSyntaxError:
        names = []
        for block in _JINJA_BLOCK.finditer(text):
            body = block.group(1) if block.group(1) is not None else block.group(2)
            for token in _NAME.findall(body):
                names.extend((token, *token.split("-")))
        return names
    free = meta.find_undeclared_variables(tree)
    names = [n.name for n in tree.find_all(nodes.Name) if n.name in free]
    for block in _JINJA_BLOCK.finditer(text):
        body = block.group(1) if block.group(1) is not None else block.group(2)
        names.extend(t for t in _NAME.findall(body) if "-" in t)
    return names


def _recipe_parts(payload: Any) -> tuple[dict, dict, dict]:
    if not isinstance(payload, dict):
        raise ValueError("A Data Recipe is a JSON object")
    if isinstance(payload.get("recipe"), dict):
        recipe, run, ui = payload["recipe"], payload.get("run"), payload.get("ui")
    else:
        recipe, run, ui = payload, None, None
    if not isinstance(recipe.get("columns"), list):
        raise ValueError("A Data Recipe needs a columns list")
    return recipe, run if isinstance(run, dict) else {}, ui if isinstance(ui, dict) else {}


def column_references(column: dict, names: set[str]) -> list[str]:
    """The other columns this column reads, in first-mention order."""
    seen: list[str] = []
    texts = [column.get(key) for key in ("prompt", "system_prompt", "expr")]
    params = column.get("params")
    conditional = column.get("conditional_params")
    nested = [params] + (list(conditional.values()) if isinstance(conditional, dict) else [])
    for rule in conditional if isinstance(conditional, dict) else ():
        if isinstance(rule, str):
            texts.append(rule if "{{" in rule or "{%" in rule else "{{ " + rule + " }}")
    for text in texts:
        if isinstance(text, str):
            for name in _template_names(text):
                if name in names and name not in seen and name != column.get("name"):
                    seen.append(name)
    for name in column.get("target_columns") or []:
        if isinstance(name, str) and name in names and name not in seen:
            seen.append(name)
    for sub in nested:
        if isinstance(sub, dict):
            for key in ("category", "reference_column_name"):
                parent = sub.get(key)
                if isinstance(parent, str) and parent in names and parent not in seen:
                    seen.append(parent)
    return seen


def _depths(columns: list[dict], names: set[str]) -> list[int]:
    """Each column's longest reference chain, so a group reads left to right (a loop stays at 0)."""
    reads = {c.get("name"): column_references(c, names) for c in columns}
    memo: dict[str, int] = {}

    def depth(name: str, path: tuple[str, ...]) -> int:
        if name in memo:
            return memo[name]
        if name in path:
            return 0
        found = max((depth(r, path + (name,)) + 1 for r in reads.get(name) or ()), default = 0)
        memo[name] = found
        return found

    return [depth(c.get("name"), ()) for c in columns]


def import_recipe(payload: Any, *, group_id: str = "recipe") -> dict:
    """A saved recipe payload (`{recipe, run, ui}` or a bare recipe) as a Recipe group fragment: the
    group, a `recipe.column` node per column, reference wires, a `recipe.generate` node, notes."""
    recipe, run, ui = _recipe_parts(payload)
    recipe = copy.deepcopy(recipe)
    columns = recipe.pop("columns")
    for i, column in enumerate(columns):
        if not isinstance(column, dict):
            raise ValueError(f"Recipe column {i + 1} is not an object")
    names = {c.get("name") for c in columns if isinstance(c.get("name"), str)}
    depths = _depths(columns, names)
    rows: dict[int, int] = {}
    nodes: list[dict] = []
    for i, (column, depth) in enumerate(zip(columns, depths)):
        name = str(column.get("name") or f"column_{i + 1}")
        row = rows.get(depth, 0)
        rows[depth] = row + 1
        nodes.append(
            {
                "id": f"{group_id}_col_{i + 1}",
                "type": COLUMN_TYPE,
                "v": 1,
                "pos": [float(depth * _GRID[0]), float(row * _GRID[1])],
                "group": group_id,
                "label": name[:200],
                "params": {"column": copy.deepcopy(column)},
            }
        )
    by_name = {n["params"]["column"].get("name"): n["id"] for n in nodes}
    edges: list[dict] = []
    for node in nodes:
        for ref in column_references(node["params"]["column"], names):
            edges.append(
                {
                    "id": f"{by_name[ref]}__{node['id']}",
                    "source": by_name[ref],
                    "source_port": "out",
                    "target": node["id"],
                    "target_port": "in",
                    "kind": "data",
                }
            )
    width = (max(depths, default = 0) + 1) * _GRID[0]
    height = max(rows.values(), default = 1) * _GRID[1]
    nodes.append(
        {
            "id": f"{group_id}_generate",
            "type": GENERATE_TYPE,
            "v": 1,
            "pos": [float(width), 0.0],
            "group": group_id,
            "params": {
                "rows": run["rows"] if isinstance(run.get("rows"), int) and run["rows"] > 0 else 5,
                "mode": "preview" if run.get("preview", True) else "full",
            },
        }
    )
    notes = [
        {
            "id": f"{group_id}_{n['id']}"[:64],
            "text": str(n.get("markdown") or "")[:20_000],
            "rect": [float(k * 340), float(height + 40), 320.0, 200.0],
            "color": n.get("note_color") if _HEX.match(str(n.get("note_color") or "")) else None,
        }
        for k, n in enumerate(
            n
            for n in ui.get("nodes") or []
            if isinstance(n, dict) and n.get("node_type") == "markdown_note" and n.get("id")
        )
    ]
    hints = {k: copy.deepcopy(v) for k, v in ui.items() if k not in _LAYOUT_UI}
    group = {
        "id": group_id,
        "kind": "recipe",
        "label": "Data Recipe",
        "color": "#3b82f6",
        "rect": [-40.0, -60.0, float(width + _GRID[0] + 40), float(height + 60)],
        "config": recipe,
        "run": copy.deepcopy(run),
    }
    if hints:
        group["ui"] = hints
    return {
        "kind": "recipe_graph",
        "version": RECIPE_GRAPH_VERSION,
        "group": group,
        "nodes": nodes,
        "edges": edges,
        "notes": notes,
    }


# Flows never carry tool servers or credentials (see `nodes/data.declarative_recipe`).
_TOOL_KEYS = ("mcp_providers", "tool_configs")
_SECRET_KEYS = ("api_key", "extra_headers")


def flow_from_recipe(
    payload: Any,
    *,
    name: str,
    source: Optional[dict] = None,
    flow_id: str = "recipe",
) -> tuple[dict, list[str]]:
    """A new pipeline flow holding the recipe as one Recipe group wired to a preview, and notices
    about anything a flow cannot keep (tool servers, provider secrets)."""
    recipe, run, ui = _recipe_parts(payload)
    recipe = copy.deepcopy(recipe)
    notices: list[str] = []
    if any(recipe.get(key) for key in _TOOL_KEYS) or any(
        isinstance(c, dict) and c.get("tool_alias") for c in recipe["columns"]
    ):
        notices.append(
            "Tool servers stay on the Data Recipes page; columns that used them run without tools."
        )
    for key in _TOOL_KEYS:
        recipe.pop(key, None)
    for column in recipe["columns"]:
        if isinstance(column, dict):
            column.pop("tool_alias", None)
    for provider in recipe.get("model_providers") or []:
        if isinstance(provider, dict) and any(provider.get(k) for k in _SECRET_KEYS):
            notices.append(
                f"Provider {provider.get('name')}: add its API key again, flows do not keep keys."
            )
        for key in _SECRET_KEYS:
            if isinstance(provider, dict):
                provider.pop(key, None)
    graph = import_recipe({"recipe": recipe, "run": run, "ui": ui}, group_id = "recipe")
    graph["group"]["label"] = name[:200] or "Data Recipe"
    generate = graph["nodes"][-1]
    x, y = generate["pos"]
    preview = {
        "id": "preview",
        "type": "output.preview",
        "v": 1,
        "pos": [x + _GRID[0] + 40, y],
        "label": "Rows",
        "params": {},
    }
    edge = {
        "id": "recipe_rows",
        "source": generate["id"],
        "source_port": "preview",
        "target": "preview",
        "target_port": "value",
        "kind": "data",
    }
    meta: dict[str, Any] = {"tags": ["dataset", "data recipe"]}
    if source:
        meta["from_recipe"] = source
    doc = {
        "format": "unsloth-flow",
        "version": 1,
        "id": flow_id,
        "name": name[:200] or "Data Recipe",
        "description": "",
        "kind": "pipeline",
        "seed": 0,
        "params": [],
        "requirements": {"models": []},
        "nodes": [*graph["nodes"], preview],
        "edges": [*graph["edges"], edge],
        "groups": [graph["group"]],
        "notes": graph["notes"],
        "assets": [],
        "baked": {},
        "editor": {},
        "meta": meta,
    }
    return doc, notices
